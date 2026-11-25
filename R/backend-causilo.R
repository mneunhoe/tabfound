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


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

#' Causilo column stage: one column layer per row layer
#' @param m Model dimensions.
#' @keywords internal
causilo_column_block <- torch::nn_module(
  "CausiloColumnBlock",
  initialize = function(m) {
    self$layers <- torch::nn_module_list(lapply(seq_len(m$column_depth), function(i) {
      causilo_column_layer(m$width)
    }))
  }
)

#' Causilo row stage: latent tokens carried through the row layers
#' @param m Model dimensions.
#' @keywords internal
causilo_row_block <- torch::nn_module(
  "CausiloRowBlock",
  initialize = function(m) {
    L <- m$column_depth
    self$layers <- torch::nn_module_list(lapply(seq_len(L), function(i) {
      causilo_row_layer(m$width, final = i == L)
    }))
    self$latents <- torch::nn_parameter(torch::torch_zeros(CAUSILO_ROW_LATENTS, m$width))
    self$output_norm <- kumo_rms_norm(m$width, eps = 1e-5)
  },
  # `[..., K, W]` -> `[..., K * W]`.
  pool = function(latents) {
    self$output_norm(latents)$flatten(start_dim = -2L)
  }
)

#' Causilo prediction head: RMSNorm, Linear(P, 2P), GELU, Linear
#' @param width Prediction width `P`.
#' @param outputs Output width.
#' @keywords internal
causilo_head <- torch::nn_module(
  "CausiloHead",
  initialize = function(width, outputs) {
    self$normalize <- kumo_rms_norm(width, eps = 1e-5)
    self$hidden <- torch::nn_linear(width, 2L * width)
    self$output <- torch::nn_linear(2L * width, outputs)
  },
  forward = function(x) {
    self$output(torch::nnf_gelu(self$hidden(self$normalize(x))))
  }
)

#' Causilo prediction stage: every row attends to the context rows
#' @param m Model dimensions.
#' @keywords internal
causilo_prediction_block <- torch::nn_module(
  "CausiloPredictionBlock",
  initialize = function(m) {
    P <- CAUSILO_ROW_LATENTS * m$width
    self$width <- P
    self$layers <- torch::nn_module_list(lapply(seq_len(m$prediction_depth), function(i) {
      causilo_attention_layer(P, CAUSILO_PREDICTION_HEADS, scale_query = TRUE)
    }))
    self$target <- causilo_target_embedding(P, m$max_classes)
    self$head <- causilo_head(P, if (m$max_classes > 0L) m$max_classes
                                 else CAUSILO_NUM_QUANTILES)
  },
  # A copy of `rows` `[B, R, P]` with the targets `[B, n]` added to the
  # context prefix.
  add_context_targets = function(rows, targets) {
    out <- rows$clone()
    n <- targets$size(2L)
    if (n > 0L) out$narrow(2L, 1L, n)$add_(self$target(targets))
    out
  }
)


#' The Causilo 1.5 network
#'
#' Module names equal the checkpoint keys. `forward(table, targets)` takes
#' the preprocessed, signed member tables `[B, R, C]` (float32, NaN for
#' missing; the first `ncol(targets)` rows are context) and the member
#' targets `[B, n]` (0-based class codes or the signed, scaled regression
#' target), and returns the head output for the query rows
#' `[B, R - n, 10 | 999]`.
#'
#' When `TABFOUND_DUMP_DIR` is set, stage outputs are written there under
#' the names `inst/parity/causilo_reference.py` uses.
#'
#' @param config The published `config.json`, read.
#' @keywords internal
causilo_model <- torch::nn_module(
  "CausiloModel",

  initialize = function(config) {
    m <- .causilo_model_config(config)
    self$dims <- m
    self$feature_embedding <- causilo_feature_embedding(m)
    self$columns <- causilo_column_block(m)
    self$row <- causilo_row_block(m)
    self$prediction <- causilo_prediction_block(m)
  },

  # Cells, columns leading: `[B, R, C]` -> `[B, C, R, 1]`.
  column_features = function(table) {
    table$to(dtype = torch::torch_float32())$transpose(2L, 3L)$unsqueeze(-1L)
  },

  # Feature embedding, the paired column/row layers and the pooled latents:
  # `[B, R, C]` -> `[B, R, 4W]`.
  #
  # @param targets `[B, n]`, or `NULL` when every row is a query row and
  #   `cache` holds the context.
  # @param cache A `causilo_kv_cache` to read (query rows only) or, with
  #   `record = TRUE`, a list to fill with the column layers' K/V.
  encode_rows = function(table, targets, ecdf = NULL, column_kv = NULL,
                         record = FALSE) {
    feats <- self$column_features(table)
    C <- feats$size(2L)
    n <- if (!is.null(targets)) targets$size(2L) else NULL
    if (is.null(ecdf)) ecdf <- causilo_ecdf_context(feats, n)
    col_targets <- if (!is.null(targets)) {
      targets$to(dtype = torch::torch_float32())$unsqueeze(2L)$expand(c(-1L, C, -1L))
    }
    emb <- self$feature_embedding(feats, ecdf, col_targets)
    dump_if_enabled("causilo_embed", emb)
    features <- emb$transpose(2L, 3L)
    shp <- features$shape
    latents <- self$row$latents$expand(c(shp[1:2], self$row$latents$shape))
    kv <- if (isTRUE(record)) vector("list", length(self$columns$layers))
    for (i in seq_along(self$columns$layers)) {
      rows <- features$transpose(2L, 3L)
      layer <- self$columns$layers[[i]]
      rows <- if (!is.null(column_kv)) {
        layer$query(rows, column_kv[[i]])
      } else if (isTRUE(record)) {
        r <- layer(rows, n, return_kv = TRUE)
        kv[[i]] <- r$kv
        r$rows
      } else {
        layer(rows, n)
      }
      dump_if_enabled(sprintf("causilo_col%d", i - 1L), rows)
      features <- rows$transpose(2L, 3L)
      r <- self$row$layers[[i]](latents, features)
      latents <- r$latents$contiguous()
      features <- r$features
      dump_if_enabled(sprintf("causilo_row%d", i - 1L),
                      if (self$row$layers[[i]]$final) latents
                      else torch::torch_cat(list(latents, features), dim = -2L))
    }
    pooled <- self$row$pool(latents)
    dump_if_enabled("causilo_reps", pooled)
    if (isTRUE(record)) list(rows = pooled, column_kv = kv, ecdf = ecdf) else pooled
  },

  # Prediction layers over the target-conditioned rows: `[B, R, P]` ->
  # head output for the query rows.
  predict_rows = function(rows, targets) {
    n <- targets$size(2L)
    R <- rows$size(2L)
    x <- self$prediction$add_context_targets(rows, targets)
    layers <- self$prediction$layers
    L <- length(layers)
    for (j in seq_len(L - 1L)) {
      x <- layers[[j]](x, context_rows = n)
      dump_if_enabled(sprintf("causilo_pred%d", j - 1L), x)
    }
    q <- layers[[L]](x$narrow(2L, n + 1L, R - n), x$narrow(2L, 1L, n))
    dump_if_enabled(sprintf("causilo_pred%d", L - 1L), q)
    out <- self$prediction$head(q)
    dump_if_enabled("causilo_head", out)
    out
  },

  forward = function(table, targets) {
    self$predict_rows(self$encode_rows(table, targets), targets)
  }
)


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

# The children of one pre-normalised attention layer, in the order it runs
# them: the shared norm, attention, then the SwiGLU feedforward behind its
# own norm, each inside a residual.
# @keywords internal
causilo_layer_children <- function(pfx, id, label, axis, attn, ffn) {
  p <- function(x) paste0(pfx, ".", x)
  list(
    arch_residual(
      arch_stage(paste0(id, "_n1"), "RMSNorm, queries and keys/values",
                 kind = "norm", prefix = p("attention_norm")),
      arch_stage(id, label, kind = "attention", axis = axis, attn = attn,
                 prefix = p("attention"))
    ),
    arch_residual(
      arch_stage(paste0(id, "_n2"), "RMSNorm", kind = "norm",
                 prefix = p("feedforward_norm")),
      arch_stage(paste0(id, "_ff"), "SwiGLU feed-forward", kind = "ffn",
                 ffn = ffn, prefix = p("feedforward"))
    )
  )
}

# @keywords internal
causilo_row_children <- function(ffn, pfx = "row.layers.0") {
  p <- function(x) paste0(pfx, ".", x)
  attn <- function(keys) arch_attn(keys, qk_norm = TRUE,
                                   scaling = "softmax-scaling MLP")
  list(
    arch_residual(
      arch_stage("rg_n", "RMSNorm, latents and cells", kind = "norm",
                 prefix = p("gather_norm")),
      arch_stage("rg", "Latents attend to themselves and the row's cells",
                 kind = "attention", axis = "columns", prefix = p("gather"),
                 attn = attn("all"))
    ),
    arch_residual(
      arch_stage("rb_n", "RMSNorm", kind = "norm", prefix = p("broadcast_norm")),
      arch_stage("rb", "Latents and cells attend to the latents",
                 kind = "attention", axis = "columns", prefix = p("broadcast"),
                 attn = attn("all"))
    ),
    arch_residual(
      arch_stage("rf_n", "RMSNorm, latents", kind = "norm",
                 prefix = p("feedforward_norm")),
      arch_stage("rf", "SwiGLU feed-forward, latents only", kind = "ffn",
                 ffn = ffn, prefix = p("feedforward"))
    ),
    arch_stage("onorm", "RMSNorm, pooled latents", kind = "norm",
               role = "final", prefix = "row.output_norm")
  )
}

#' @keywords internal
causilo_describe <- function(config, task) {
  m <- .causilo_model_config(config)
  W <- m$width
  P <- CAUSILO_ROW_LATENTS * W
  L <- m$column_depth
  Lp <- m$prediction_depth
  K <- CAUSILO_ROW_LATENTS
  clf <- m$max_classes > 0L
  n_out <- if (clf) m$max_classes else CAUSILO_NUM_QUANTILES
  scaled <- function(keys, queries = NULL) {
    arch_attn(keys, queries = queries, qk_norm = TRUE,
              scaling = "softmax-scaling MLP")
  }

  stages <- list(
    arch_input_stage(detail = "per member: the preprocessed, sign-flipped feature view, NaN kept"),
    arch_stage(
      "cells", "Cell embedding", kind = "embed", group = "Row encoding",
      detail = sprintf("each cell alone: sin/cos of %d learned frequencies of the value, the value, sin/cos of %d frequencies of its mid-rank among the column's context values; a learned vector for missing cells",
                       CAUSILO_FOURIER_FREQS, CAUSILO_ECDF_FREQS),
      shape = sprintf("(C, R, %d)", W),
      prefix = paste0("feature_embedding.",
                      c("frequencies", "rank_frequencies", "projection", "missing"))),
    arch_stage(
      "cell_y", "Target embedding (cells)", kind = "embed", group = "Row encoding",
      detail = if (clf) "one-hot class, added to every cell of the context rows"
               else "scaled target, added to every cell of the context rows",
      join = "add", join_rows = "context rows",
      prefix = "feature_embedding.target"),
    arch_stage(
      "latents", "Row latents", kind = "embed", group = "Row encoding",
      detail = sprintf("%d learned tokens per row, carried through the row layers", K),
      join = "prepend", prefix = "row.latents"),
    arch_stage(
      "col", "Column layer", kind = "attention", group = "Row encoding",
      repeats = L, axis = "rows", interleave = "embedding",
      detail = sprintf("%d learned latents read a column's context rows (learned query gain), every row reads them back; %d heads",
                       CAUSILO_COLUMN_LATENTS, CAUSILO_COLUMN_HEADS),
      shape = sprintf("(C, R, %d)", W), prefix = "columns.layers",
      children = c(
        list(arch_stage("iv", "Learned column latents", kind = "embed",
                        role = "input", feeds = "cg",
                        prefix = "columns.layers.0.latents")),
        causilo_layer_children("columns.layers.0.gather", "cg",
                               "Latents attend to the context rows", "rows",
                               scaled("context rows"),
                               arch_ffn(W, 2L * W, type = "swiglu")),
        causilo_layer_children("columns.layers.0.broadcast", "cb",
                               "Rows attend to the latents", "inducing",
                               arch_attn("inducing points", qk_norm = TRUE),
                               arch_ffn(W, 2L * W, type = "swiglu"))
      )),
    arch_stage(
      "row", "Row layer", kind = "attention", group = "Row encoding",
      repeats = L, axis = "columns", interleave = "embedding",
      detail = sprintf("the %d latents gather a row's cells, then cells and latents read the latents back; the last layer only updates the latents; %d heads",
                       K, CAUSILO_ROW_HEADS),
      shape = sprintf("(R, %d + C, %d)", K, W), collapses = TRUE,
      prefix = c("row.layers", "row.output_norm"),
      children = causilo_row_children(arch_ffn(W, 2L * W, type = "swiglu"))),
    arch_stage(
      "icl_y", "Target embedding (rows)", kind = "embed", group = "Prediction",
      detail = "added to the context rows", join = "add",
      join_rows = "context rows", prefix = "prediction.target"),
    arch_stage(
      "icl", "Prediction layer", kind = "attention", group = "Prediction",
      repeats = Lp, axis = "rows",
      detail = sprintf("every row attends to the context rows (learned query gain); the last layer advances the query rows only; %d heads of %d",
                       CAUSILO_PREDICTION_HEADS, P %/% CAUSILO_PREDICTION_HEADS),
      shape = sprintf("(R, %d)", P), prefix = "prediction.layers",
      children = causilo_layer_children("prediction.layers.0", "pa",
                                        "Rows attend to the context rows", "rows",
                                        scaled("context rows"),
                                        arch_ffn(P, 2L * P, type = "swiglu"))),
    arch_stage(
      "head", "Decoder MLP", kind = "decode", group = "Decode",
      detail = sprintf("RMSNorm, %d -> %d -> %d, GELU", P, 2L * P, n_out),
      shape = sprintf("(n_test, %d)", n_out), prefix = "prediction.head"),
    if (clf)
      arch_output_stage("Class logits",
                        detail = sprintf("%d slots; error-correcting codes above %d classes", n_out, n_out))
    else
      arch_output_stage("Quantiles",
                        detail = sprintf("%d quantiles of the scaled target", n_out))
  )

  list(
    title = sprintf("Causilo 1.5 - %s", if (clf) "classifier" else "regressor"),
    subtitle = "Nums  -  cell embedding -> column/row layers -> prediction stack",
    facts = c(
      "Column-stage blocks" = L,
      "Row-stage blocks"  = L,
      "ICL layers"        = Lp,
      "Embedding width"   = W,
      "ICL width"         = sprintf("%d (%d latents x %d)", P, K, W),
      "ICL heads"         = sprintf("%d x %d", CAUSILO_PREDICTION_HEADS,
                                    P %/% CAUSILO_PREDICTION_HEADS),
      "Embedding heads"   = sprintf("%d (columns), %d (rows)",
                                    CAUSILO_COLUMN_HEADS, CAUSILO_ROW_HEADS),
      "Inducing points"   = sprintf("%d latents per column layer", CAUSILO_COLUMN_LATENTS),
      "Fourier frequencies" = sprintf("%d (values), %d (ranks)",
                                      CAUSILO_FOURIER_FREQS, CAUSILO_ECDF_FREQS),
      "Attention scaling" = "queries and keys scaled to unit root-mean-square; a learned gain from log(keys) and the query",
      "Output width"      = n_out,
      "Missing values"    = "handled by the network"
    ),
    symbols = c(arch_default_symbols(), R = "rows", C = "columns"),
    stages = Filter(Negate(is.null), stages)
  )
}


# ---------------------------------------------------------------------------
# Memory scaling
# ---------------------------------------------------------------------------

#' Causilo 1.5's peak-memory shapes
#'
#' The three stages of the column/row/in-context family under the shared
#' stage names, so [.peak_from_terms()] and the calibration harness read
#' them unchanged:
#'
#' * column embedding: every cell at width `W`, plus each column's 256
#'   latents (`C x 256 x W`);
#' * row interaction: a row's `4 + C` tokens at width `W`;
#' * in-context learning: every row at the prediction width `4W`.
#'
#' One forward pass holds `opts$member_batch_size` members (default one).
#' A KV cache holds, per member, every prediction layer's keys and values
#' of the context rows and every column layer's broadcast keys and values
#' of the 256 latents.
#'
#' @inheritParams tabicl_peak_terms
#' @keywords internal
causilo_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  m <- .causilo_model_config(config)
  W <- as.numeric(m$width)
  P <- CAUSILO_ROW_LATENTS * W
  C <- as.numeric(n_features)
  n_est <- max(1, as.numeric(opts$n_estimators %||% 8))
  G <- max(1, min(n_est, as.numeric(opts$member_batch_size %||% 1)))
  n <- as.numeric(n_context) + as.numeric(n_query)
  stages <- list(
    list(name = "column summaries", act = 0, res = 0, att = 0, prepass = 0),
    list(name = "column embedding",
         act = G * (n * C * W + C * CAUSILO_COLUMN_LATENTS * W), res = 0, att = 0),
    list(name = "row interaction",
         act = G * n * (CAUSILO_ROW_LATENTS + C) * W, res = 0, att = 0),
    list(name = "in-context learning", act = G * n * P, res = 0,
         copies = "icl", att = 0)
  )
  cache <- 2 * (m$prediction_depth * as.numeric(n_context) * P +
                  m$column_depth * C * CAUSILO_COLUMN_LATENTS * W)
  persistent <- if (isTRUE(opts$kv_cache_kept)) n_est * cache
                else if (isTRUE(opts$kv_cache)) cache else 0
  list(stages = stages, persistent = persistent, n_estimators = n_est,
       n_passes = ceiling(n_est / G))
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#' @keywords internal
causilo_build <- function(config, task) {
  m <- .causilo_model_config(config)
  if (!is.null(task) && !identical(task, m$task)) {
    cli::cli_abort("This Causilo checkpoint is a {m$task} model, not a {task} one.")
  }
  cli::cli_alert_info(
    "Building Causilo 1.5 ({.val {m$task}}, W={m$width}, \\
     {m$column_depth} column/row layers, {m$prediction_depth} prediction layers)..."
  )
  causilo_model(config)
}

register_causilo_backend <- function() {
  register_backend(
    name          = "causilo",
    build         = causilo_build,
    detect        = causilo_detect,
    task_of       = causilo_task_of,
    classifier    = NULL,
    regressor     = NULL,
    describe      = causilo_describe,
    peak_terms    = causilo_peak_terms,
    # The network embeds missing cells itself: a learned vector, with the
    # value and rank waves zeroed.
    kv_cache_capable   = FALSE,
    handles_missing    = TRUE,
    imputes_internally = FALSE,
    description   = "Causilo 1.5 (Nums AI) column/row latent transformer with a prediction stack",
    parity        = "causilo 1.5.0 (PyPI)"
  )
}
