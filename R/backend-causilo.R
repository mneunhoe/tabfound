# Causilo 1.5 backend.
#
# A port of the Causilo 1.5 runtime in Nums AI's `causilo` 1.5.0
# (Apache-2.0); weights from `nums-ai/causilo-1.5` (Nums AI Model License:
# non-commercial research and evaluation). One architecture, two
# checkpoints (classifier, regressor), published as `model.safetensors` +
# `config.json`, so nothing is converted:
#
#   cells [B, C, R, 1] -> Fourier + raw value + context mid-rank waves
#                         -> Linear(41, D) (+ learned missing vector)
#                         + target embedding on every column's context rows
#   8 x { column layer: 256 latents gather each column's context rows,
#                       every row reads the latents back;
#         row layer:    4 latent tokens per row gather the row's C cells
#                       and are broadcast back to them }
#   4 latents -> RMSNorm -> [B, R, 4D]
#   prediction: + target embedding on the context rows; 24 layers in
#               which every row attends to the context rows (the last
#               advances the query rows only)
#   head: RMSNorm -> Linear(4D, 8D) -> GELU -> 10 logits or 999 quantiles
#
# The layers live in `R/backend-causilo-layers.R`, the preprocessing and
# ensemble recipe in `R/prep-causilo.R`. Module names equal the
# checkpoint keys.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Constants of the architecture (`causilo/v1_5/nn/model_config.py`). The
# published `config.json` carries only width, the two depths and the class
# count; everything else is fixed.
CAUSILO_ROW_LATENTS      <- 4L
CAUSILO_NATIVE_CLASSES   <- 10L
CAUSILO_NUM_QUANTILES    <- 999L
CAUSILO_FOURIER_FREQS    <- 16L
CAUSILO_ECDF_FREQS       <- 4L
CAUSILO_EXPANSION        <- 2L
CAUSILO_COLUMN_HEADS     <- 4L
CAUSILO_COLUMN_LATENTS   <- 256L
CAUSILO_ROW_HEADS        <- 8L
CAUSILO_PREDICTION_HEADS <- 8L
CAUSILO_ARCHITECTURE     <- "causilo-v1.5"

# Read and check the published config, as `validate_checkpoint()` does:
# the three version fields, exactly the four model fields, a width that
# splits into the row heads, and a task that agrees with `max_classes`.
# Returns the model dimensions as integers plus the task.
# @keywords internal
.causilo_model_config <- function(config) {
  if (!identical(config$architecture, CAUSILO_ARCHITECTURE) ||
      !identical(as.integer(config$format_version %||% NA), 1L) ||
      !identical(as.integer(config$config_version %||% NA), 1L)) {
    cli::cli_abort("Unsupported Causilo 1.5 configuration.")
  }
  m <- config$model
  fields <- c("max_classes", "width", "column_depth", "prediction_depth")
  if (!is.list(m) || !setequal(names(m), fields)) {
    cli::cli_abort(c(
      "A Causilo 1.5 config must hold exactly the model fields \\
       {.field {fields}}.",
      i = "Got {.field {names(m)}}."
    ))
  }
  m <- lapply(m[fields], as.integer)
  if (!m$max_classes %in% c(0L, CAUSILO_NATIVE_CLASSES)) {
    cli::cli_abort("{.field max_classes} must be 0 (regression) or 10 \\
                    (classification), not {m$max_classes}.")
  }
  for (f in fields[-1]) {
    if (is.na(m[[f]]) || m[[f]] < 1L) {
      cli::cli_abort("{.field {f}} must be a positive integer.")
    }
  }
  if (m$width %% CAUSILO_ROW_HEADS != 0L) {
    cli::cli_abort("{.field width} must be divisible by {CAUSILO_ROW_HEADS}.")
  }
  task <- if (m$max_classes > 0L) "classification" else "regression"
  if (!identical(config$task, task)) {
    cli::cli_abort(c(
      "This Causilo config's {.field task} does not match its model.",
      i = "{.field task} is {.val {config$task}}, but \\
           {.field max_classes} = {m$max_classes} means {.val {task}}."
    ))
  }
  c(m, list(task = task))
}

#' @keywords internal
causilo_detect <- function(config) {
  identical(config$architecture, CAUSILO_ARCHITECTURE)
}

#' @keywords internal
causilo_task_of <- function(config) {
  tryCatch(.causilo_model_config(config)$task, error = function(e) NULL)
}


# ---------------------------------------------------------------------------
# Artifact integrity
# ---------------------------------------------------------------------------

# The `__metadata__` block of a safetensors header, read without loading
# a tensor: an 8-byte little-endian length, then that many bytes of JSON.
# @keywords internal
.safetensors_metadata <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)
  lo <- readBin(con, "integer", n = 2L, size = 4L, endian = "little")
  if (length(lo) != 2L) cli::cli_abort("{.path {path}} is not a safetensors file.")
  # The length is a uint64; headers are far below 2^31.
  n <- (lo[[1]] %% 2^32) + lo[[2]] * 2^32
  if (!is.finite(n) || n <= 0 || n > 1e8) {
    cli::cli_abort("{.path {path}} has an unreadable safetensors header.")
  }
  hdr <- rawToChar(readBin(con, "raw", n = n))
  jsonlite::fromJSON(hdr, simplifyVector = TRUE)[["__metadata__"]]
}

# Nums AI writes the SHA-256 of the exact `config.json` bytes, the
# architecture, the task and the format version into each checkpoint's
# header. Checking them catches a classifier config beside regressor
# weights, or a config that was rewritten -- which is also why the store
# keeps `config.json` byte for byte as published.
# @keywords internal
.causilo_verify_artifacts <- function(dir) {
  weights <- file.path(dir, "model.safetensors")
  config  <- file.path(dir, "config.json")
  meta <- .safetensors_metadata(weights)
  cfg <- read_model_config(config)
  sha <- .file_sha256(config)
  problems <- c(
    if (!identical(meta$architecture, CAUSILO_ARCHITECTURE))
      sprintf("the weights' architecture is %s", meta$architecture %||% "missing"),
    if (!identical(meta$format_version, "1"))
      "the weights' format version is not 1",
    if (!identical(meta$task, cfg$task))
      sprintf("the weights are for %s, the config for %s",
              meta$task %||% "?", cfg$task %||% "?"),
    if (!identical(meta$config_sha256, sha))
      "config.json is not the file these weights were published with"
  )
  if (length(problems)) {
    cli::cli_abort(c(
      "The Causilo artifacts in {.path {dir}} do not belong together.",
      set_names(problems, rep("x", length(problems)))
    ))
  }
  invisible(TRUE)
}

# SHA-256 of a file's bytes.
# @keywords internal
.file_sha256 <- function(path) {
  require_suggested("digest")
  digest::digest(file = path, algo = "sha256")
}

