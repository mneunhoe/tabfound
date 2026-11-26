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
    self$chunks <- list()
  },

  # Cells, columns leading: `[B, R, C]` -> `[B, C, R, 1]`.
  column_features = function(table) {
    table$to(dtype = torch::torch_float32())$transpose(2L, 3L)$unsqueeze(-1L)
  },

  # Chunk sizes for the stages whose work is independent along an axis:
  # columns in the column layers, rows in the row layers and in the
  # prediction layers (whose keys are the context rows, fixed per layer).
  # `NULL` runs a stage in one pass. `self$chunks`, set by the predictors.

  # Feature embedding, the paired column/row layers and the pooled latents:
  # `[B, R, C]` -> `[B, R, 4W]`.
  #
  # @param targets `[B, n]`, or `NULL` when every row is a query row and
  #   `ecdf` / `column_kv` hold the context (a [causilo_kv_cache()]).
  # @param record Also return the column layers' broadcast K/V and the
  #   sorted context: the column half of a cache.
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
    rm(feats, col_targets)
    dump_if_enabled("causilo_embed", emb)
    features <- emb$transpose(2L, 3L)
    rm(emb)
    shp <- features$shape
    latents <- self$row$latents$expand(c(shp[1:2], self$row$latents$shape))
    kv <- if (isTRUE(record)) vector("list", length(self$columns$layers))
    for (i in seq_along(self$columns$layers)) {
      rows <- features$transpose(2L, 3L)
      layer <- self$columns$layers[[i]]
      res <- .causilo_split_apply(rows, 2L, self$chunks$col, function(part, idx) {
        if (!is.null(column_kv)) {
          list(rows = layer$query(part, .causilo_kv_slice(column_kv[[i]], 2L, idx)))
        } else if (isTRUE(record)) {
          layer(part, n, return_kv = TRUE)
        } else {
          list(rows = layer(part, n))
        }
      })
      rows <- res$rows
      if (isTRUE(record)) kv[[i]] <- res$kv
      rm(res)
      dump_if_enabled(sprintf("causilo_col%d", i - 1L), rows)
      features <- rows$transpose(2L, 3L)
      rm(rows)
      # R frees a layer's intermediates only when it collects; without this
      # they pile up across layers (see [.causilo_collect()]).
      .causilo_collect(features)
      rl <- self$row$layers[[i]]
      r <- .causilo_split_apply2(latents, features, 2L, self$chunks$row,
                                 function(l, f) rl(l, f))
      latents <- r$latents$contiguous()
      features <- r$features
      rm(r)
      .causilo_collect(features)
      dump_if_enabled(sprintf("causilo_row%d", i - 1L),
                      if (rl$final) latents
                      else torch::torch_cat(list(latents, features), dim = -2L))
    }
    pooled <- self$row$pool(latents)
    dump_if_enabled("causilo_reps", pooled)
    if (isTRUE(record)) list(rows = pooled, column_kv = kv, ecdf = ecdf) else pooled
  },

  # One prediction layer over `x` against `kv` (the context's K/V), the
  # rows a chunk at a time.
  .advance = function(layer, x, kv) {
    .causilo_split_apply(x, 2L, self$chunks$row, function(part, idx) {
      list(rows = layer$query(part, kv))
    })$rows
  },

  # Prediction layers over the target-conditioned rows: `[B, R, P]` ->
  # head output for the query rows. With `record`, also the context's
  # K/V at every layer (the prediction half of a cache).
  predict_rows = function(rows, targets, record = FALSE) {
    n <- targets$size(2L)
    R <- rows$size(2L)
    x <- self$prediction$add_context_targets(rows, targets)
    layers <- self$prediction$layers
    L <- length(layers)
    kv <- if (isTRUE(record)) vector("list", L)
    # A chunk as long as the rows splits nothing: run the plain path.
    plain <- !isTRUE(record) && (is.null(self$chunks$row) || self$chunks$row >= R)
    for (j in seq_len(L - 1L)) {
      layer <- layers[[j]]
      if (plain) {
        x <- layer(x, context_rows = n)
      } else {
        # The keys of every chunk are the context rows of this layer's
        # input, normalized: fix them first, then advance the rows.
        k <- layer$prepare(x$narrow(2L, 1L, n))
        if (isTRUE(record)) kv[[j]] <- k
        x <- self$.advance(layer, x, k)
      }
      .causilo_collect(x)
      dump_if_enabled(sprintf("causilo_pred%d", j - 1L), x)
    }
    if (isTRUE(record)) kv[[L]] <- layers[[L]]$prepare(x$narrow(2L, 1L, n))
    q <- if (plain) {
      layers[[L]](x$narrow(2L, n + 1L, R - n), x$narrow(2L, 1L, n))
    } else {
      self$.advance(layers[[L]], x$narrow(2L, n + 1L, R - n),
                    kv[[L]] %||% layers[[L]]$prepare(x$narrow(2L, 1L, n)))
    }
    dump_if_enabled(sprintf("causilo_pred%d", L - 1L), q)
    out <- self$prediction$head(q)
    dump_if_enabled("causilo_head", out)
    if (isTRUE(record)) list(out = out, kv = kv) else out
  },

  forward = function(table, targets) {
    self$predict_rows(self$encode_rows(table, targets), targets)
  },

  # Encode the context rows `[B, n, C]` once: the sorted context values,
  # every column layer's broadcast K/V and every prediction layer's
  # context K/V. See [causilo_kv_cache()].
  build_kv_cache = function(table, targets) {
    enc <- self$encode_rows(table, targets, record = TRUE)
    pr <- self$predict_rows(enc$rows, targets, record = TRUE)
    causilo_kv_cache(enc$ecdf, enc$column_kv, pr$kv,
                     n_train = targets$size(2L), n_features = table$size(3L))
  },

  # Query rows `[B, q, C]` against a cache built for the same members.
  forward_cached = function(table, cache) {
    if (table$size(3L) != cache$n_features) {
      cli::cli_abort("This Causilo cache was built for {cache$n_features} \\
                      feature{?s}, not {table$size(3L)}.")
    }
    x <- self$encode_rows(table, NULL, ecdf = cache$ecdf, column_kv = cache$column)
    layers <- self$prediction$layers
    for (j in seq_along(layers)) {
      x <- self$.advance(layers[[j]], x, cache$prediction[[j]])
      .causilo_collect(x)
    }
    self$prediction$head(x)
  }
)

# [collect_between_layers()] with Causilo's ratio. That helper sizes the
# collection by the tensor a loop carries, and here the carried tensor is
# small beside what one layer allocates and drops: a column layer's
# SwiGLU and projections hold about sixteen times the cell grid it passes
# on (one layer on 6,900 x 8 cells: a 56 MB grid, 1.5 GB transient), and
# a level-0 collection frees none of it once it has been promoted. Measured
# at 6,400 + 500 rows x 8 columns, one member: 9.9 GB peak without full
# collections, 6.6 GB with. Honours `tabfound.collect_between_layers`.
# @keywords internal
.causilo_collect <- function(x, ratio = 16) {
  mode <- getOption("tabfound.collect_between_layers", "auto")
  if (isFALSE(mode)) return(invisible(FALSE))
  bytes <- prod(as.numeric(x$size())) * as.numeric(x$element_size())
  if (!isTRUE(mode) && bytes * ratio < getOption("tabfound.collect_min_bytes",
                                                 COLLECT_MIN_BYTES) * 16) {
    return(invisible(FALSE))
  }
  gc(full = TRUE)
  invisible(TRUE)
}

# Apply `f(part, idx)` to slices of `x` along `axis` (`size` at a time, or
# all at once when `size` is NULL or not smaller than the axis), where `f`
# returns a list of tensors (`rows`, and optionally `kv = list(key,
# value)` with the same axis); concatenate each piece.
# @keywords internal
.causilo_split_apply <- function(x, axis, size, f) {
  len <- x$size(axis)
  if (is.null(size) || size >= len || len == 0L) return(f(x, NULL))
  starts <- seq.int(1L, len, by = size)
  parts <- lapply(starts, function(s) {
    m <- min(size, len - s + 1L)
    f(x$narrow(axis, s, m), c(s, m))
  })
  out <- list(rows = torch::torch_cat(lapply(parts, `[[`, "rows"), dim = axis))
  if (!is.null(parts[[1]]$kv)) {
    out$kv <- list(
      key = torch::torch_cat(lapply(parts, function(p) p$kv$key), dim = axis),
      value = torch::torch_cat(lapply(parts, function(p) p$kv$value), dim = axis))
  }
  out
}

# Two tensors sliced together along `axis` (latents and features of the
# row layers).
# @keywords internal
.causilo_split_apply2 <- function(a, b, axis, size, f) {
  len <- a$size(axis)
  if (is.null(size) || size >= len || len == 0L) return(f(a, b))
  parts <- lapply(seq.int(1L, len, by = size), function(s) {
    m <- min(size, len - s + 1L)
    f(a$narrow(axis, s, m), b$narrow(axis, s, m))
  })
  list(latents = torch::torch_cat(lapply(parts, `[[`, "latents"), dim = axis),
       features = torch::torch_cat(lapply(parts, `[[`, "features"), dim = axis))
}

# @keywords internal
.causilo_kv_slice <- function(kv, axis, idx) {
  if (is.null(idx)) return(kv)
  list(key = kv$key$narrow(axis, idx[[1]], idx[[2]]),
       value = kv$value$narrow(axis, idx[[1]], idx[[2]]))
}


#' A Causilo 1.5 context cache
#'
#' What the query rows read from the context, per member (the leading
#' axis): the context values of every column, sorted (`ecdf`, `[B, C, n,
#' 1]`, for the cell mid-ranks); each column layer's broadcast keys and
#' values of its 256 gathered latents (`column`, `[B, C, 4, 256, W/4]`
#' each); and each prediction layer's keys and values of the context rows
#' (`prediction`, `[B, 8, n, 4W/8]` each). The row layers mix only within
#' a row and cache nothing. Query rows read only this, so a cached
#' prediction is the plain one up to float32 rounding.
#'
#' At the published size one member's prediction half is about 197 KB per
#' context row (1.97 GB at 10,000 rows), and the column half 4.2 MB per
#' column.
#'
#' @param ecdf,column,prediction The three parts.
#' @param n_train,n_features Context shape.
#' @keywords internal
causilo_kv_cache <- function(ecdf, column, prediction, n_train, n_features) {
  structure(list(ecdf = ecdf, column = column, prediction = prediction,
                 n_train = as.integer(n_train), n_features = as.integer(n_features)),
            class = "causilo_kv_cache")
}

#' @export
#' @keywords internal
print.causilo_kv_cache <- function(x, ...) {
  bytes <- function(l) sum(vapply(l, function(p) {
    if (inherits(p, "torch_tensor")) return(p$numel() * p$element_size())
    sum(vapply(p, function(t) t$numel() * t$element_size(), numeric(1)))
  }, numeric(1)))
  b <- bytes(list(x$ecdf)) + bytes(x$column) + bytes(x$prediction)
  cat(sprintf("<causilo_kv_cache> %d member(s), %d context rows x %d columns, %.1f MB\n",
              as.integer(x$ecdf$size(1L)), x$n_train, x$n_features, b / 1e6))
  invisible(x)
}


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

# `causilo/v1_5/execution/quantiles.py::interpolate_quantiles`: read the
# sorted native quantiles `[..., Q]` (at levels `k / (Q + 1)`) at
# `levels`; linear between native levels, log-linear in `p` / `1 - p`
# beyond them. In float32, as the reference does it.
# @keywords internal
.causilo_interp_quantiles <- function(q, levels) {
  tol <- 1e-6
  count <- q$size(-1L)
  f32 <- torch::torch_float32()
  grid <- torch::torch_linspace(0, 1, count + 2L, dtype = f32)$narrow(1L, 2L, count)
  req <- torch::torch_tensor(levels, dtype = f32)
  interior <- all(as.numeric(grid[1]) <= as.numeric(req) &
                    as.numeric(req) <= as.numeric(grid[count]))
  lower <- (torch::torch_searchsorted(grid$narrow(1L, 1L, count - 1L), req,
                                      right = TRUE) - 1L)$clamp(0L, count - 2L)
  upper <- lower + 1L
  g_lo <- grid$index_select(1L, lower + 1L)
  g_hi <- grid$index_select(1L, upper + 1L)
  fraction <- (req - g_lo) / (g_hi - g_lo)$clamp(min = tol)
  low <- q$index_select(-1L, lower + 1L)
  high <- q$index_select(-1L, upper + 1L)
  middle <- low + fraction$clamp(0, 1) * (high - low)
  left <- q$narrow(-1L, 1L, 1L)
  right <- q$narrow(-1L, count, 1L)
  middle <- torch::torch_where(req >= grid[count], right, middle)
  if (interior) return(middle)
  log_left <- grid$narrow(1L, 1L, 2L)$clamp(min = tol)$log()
  log_right <- (1 - grid$narrow(1L, count - 1L, 2L))$clamp(min = tol)$log()
  left_scale <- (q$narrow(-1L, 2L, 1L) - left) / (log_left[2] - log_left[1])$clamp(min = tol)
  right_scale <- (right - q$narrow(-1L, count - 1L, 1L)) /
    (log_right[1] - log_right[2])$clamp(min = tol)
  left_gap <- req$clamp(min = tol)$log() - log_left[1]
  right_gap <- (1 - req)$clamp(min = tol)$log() - log_right[2]
  left_tail <- left + left_scale * left_gap
  right_tail <- right - right_scale * right_gap
  torch::torch_where(req < grid[1], left_tail,
                     torch::torch_where(req > grid[count], right_tail, middle))
}

# Members to fit. `light = TRUE` is "Causilo 1.5 (Light)", one member;
# giving it with `n_estimators` is refused, as the reference refuses
# `mode` with `n_estimators`. (The reference's argument is `mode =
# "light"`; here `mode` would be taken for `model` by R's partial
# matching in `tabular_classifier()`, and `tabfound()` already has a
# `mode`, the task.)
# @keywords internal
.causilo_n_estimators <- function(n_estimators, light) {
  if (!is.logical(light) || length(light) != 1L || is.na(light)) {
    cli::cli_abort("{.arg light} must be {.code TRUE} or {.code FALSE}.")
  }
  if (light) {
    if (!is.null(n_estimators)) {
      cli::cli_abort("Give either {.code light = TRUE} or {.arg n_estimators}, not both.")
    }
    return(1L)
  }
  if (is.null(n_estimators)) return(8L)
  if (!is.numeric(n_estimators) || length(n_estimators) != 1L ||
      is.na(n_estimators) || n_estimators < 1 || n_estimators != round(n_estimators)) {
    cli::cli_abort("{.arg n_estimators} must be a positive whole number.")
  }
  as.integer(n_estimators)
}

# @keywords internal
.causilo_check_seed <- function(seed) {
  if (is.null(seed)) return(NULL)
  if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) || seed < 0 ||
      seed != round(seed)) {
    cli::cli_abort("{.arg seed} must be a non-negative whole number or {.code NULL}.")
  }
  seed
}

# The members, `batch` at a time in balanced groups (as the reference's
# `EnsembleRunner` splits them): a list of member index vectors.
# @keywords internal
.causilo_member_groups <- function(M, batch) {
  groups <- ceiling(M / batch)
  sizes <- M %/% groups + (seq_len(groups) <= M %% groups)
  ends <- cumsum(sizes)
  lapply(seq_len(groups), function(g) (ends[g] - sizes[g] + 1L):ends[g])
}

# One group's context cache, built from its members' context rows.
# @keywords internal
.causilo_group_cache <- function(ctx, rec, members) {
  inp <- causilo_member_inputs(rec, NULL, members)
  torch::with_no_grad(ctx$net$build_kv_cache(inp$x$to(device = ctx$device),
                                             inp$y$to(device = ctx$device)))
}

# Every member's caches, group by group: what `tabfound_cache()` keeps.
# @keywords internal
.causilo_build_caches <- function(ctx, rec, batch, chunks) {
  .causilo_with_chunks(ctx$net, chunks, {
    groups <- .causilo_member_groups(rec$plan$count, batch)
    caches <- lapply(groups, function(g) .causilo_group_cache(ctx, rec, g))
  })
  structure(caches, groups = groups, class = "causilo_member_caches")
}

# @keywords internal
.causilo_with_chunks <- function(net, chunks, expr) {
  old <- net$chunks
  net$chunks <- chunks
  on.exit(net$chunks <- old, add = TRUE)
  force(expr)
}

# Every member's network output for the query rows `[M, q, out]` (CPU,
# float32). Members run in groups, the query rows `chunk` at a time.
# Uncached, each chunk carries the whole context. With `caches` (one per
# group, from `tabfound_cache()`), or with `kv_cache = TRUE` (a group's
# cache built for this call and dropped after its chunks), each chunk
# reads the context from the cache instead.
# @keywords internal
.causilo_member_outputs <- function(ctx, rec, qviews, batch, chunk, n_out,
                                    caches = NULL, kv_cache = FALSE,
                                    chunks = list()) {
  q <- nrow(qviews[[1]])
  groups <- attr(caches, "groups") %||% .causilo_member_groups(rec$plan$count, batch)
  chunk <- if (is.null(chunk)) max(q, 1L) else as.integer(chunk)
  row_sets <- lapply(seq.int(1L, max(q, 1L), by = chunk), function(r0) {
    if (q) r0:min(q, r0 + chunk - 1L) else integer(0)
  })
  cached <- !is.null(caches) || isTRUE(kv_cache)
  per_group <- .causilo_with_chunks(ctx$net, chunks, lapply(seq_along(groups), function(g) {
    members <- groups[[g]]
    cache <- if (!is.null(caches)) caches[[g]]
             else if (cached) .causilo_group_cache(ctx, rec, members)
    outs <- lapply(row_sets, function(rows) {
      qv <- lapply(qviews, function(v) v[rows, , drop = FALSE])
      inp <- causilo_member_inputs(rec, qv, members, context = !cached)
      x <- inp$x$to(device = ctx$device)
      out <- torch::with_no_grad(
        if (cached) ctx$net$forward_cached(x, cache)
        else ctx$net(x, inp$y$to(device = ctx$device)))
      out$narrow(-1L, 1L, n_out)$cpu()
    })
    rm(cache)
    if (length(outs) == 1L) outs[[1]] else torch::torch_cat(outs, dim = 2L)
  }))
  torch::torch_cat(per_group, dim = 1L)
}

# Chunking options as the network reads them.
# @keywords internal
.causilo_chunks <- function(row_chunk_size, col_chunk_size) {
  chk <- function(v, nm) {
    if (is.null(v)) return(NULL)
    if (!is.numeric(v) || length(v) != 1L || is.na(v) || v < 1) {
      cli::cli_abort("{.arg {nm}} must be a positive whole number or {.code NULL}.")
    }
    as.integer(v)
  }
  Filter(Negate(is.null), list(row = chk(row_chunk_size, "row_chunk_size"),
                               col = chk(col_chunk_size, "col_chunk_size")))
}

# Classification: undo each member's class shift and average the logits
# (float32), as `classification_logits()`.
# @keywords internal
.causilo_mean_logits <- function(rec, out) {
  K <- length(rec$target$classes)
  aligned <- lapply(seq_len(rec$plan$count), function(m) {
    idx <- (seq_len(K) - 1L - rec$plan$shifts[[m]]) %% K + 1L
    out[m, , ]$index_select(-1L, torch::torch_tensor(idx, dtype = torch::torch_long()))
  })
  torch::torch_stack(aligned)$mean(dim = 1L)
}

# Regression, `regression_predictions()` / `trimmed_regression_predictions()`.
# @param out `[M, q, 999]` raw head outputs.
# @return Numeric vector (mean, median) or matrix (quantiles, grid), in
#   target units.
# @keywords internal
.causilo_regression <- function(rec, out, type, levels = NULL) {
  signs <- rec$plan$target_signs
  trim <- rec$plan$trim
  sc <- rec$target
  M <- length(signs)
  q <- out$size(2L)
  sorted <- out$sort(dim = -1L)[[1]]
  neg <- which(signs < 0)
  member <- function(m) sorted[m, , ]
  inverse <- function(x) x * sc$scale + sc$mean
  if (trim == 0L) {
    per <- lapply(seq_len(M), function(m) {
      s <- member(m)
      v <- switch(type,
        mean = s$mean(dim = -1L)$unsqueeze(-1L),
        median = .causilo_interp_quantiles(s, 0.5),
        quantiles = {
          lv <- c(levels, if (length(neg)) 1 - levels)
          r <- .causilo_interp_quantiles(s, lv)
          k <- length(levels)
          if (signs[[m]] < 0) -r$narrow(-1L, k + 1L, k) else r$narrow(-1L, 1L, k)
        },
        grid = if (signs[[m]] < 0) -s$flip(-1L) else s)
      if (type %in% c("mean", "median") && signs[[m]] < 0) v <- -v
      as.matrix(v$to(dtype = torch::torch_float64()))
    })
    combined <- Reduce(`+`, per) / M
    res <- inverse(combined)
    return(if (type %in% c("mean", "median")) as.numeric(res) else res)
  }
  # Four or more members: per row and level, sort the members' values,
  # drop `trim` from each end and average in float64.
  if (type == "grid") {
    fwd <- seq_len(sorted$size(-1L)); rev_ <- rev(fwd)
    vals <- sorted
  } else {
    lv <- switch(type, mean = CAUSILO_MEAN_LEVELS, median = 0.5, quantiles = levels)
    pos <- as.numeric(torch::torch_tensor(lv, dtype = torch::torch_float32()))
    negl <- as.numeric(torch::torch_tensor(1 - lv, dtype = torch::torch_float32()))
    if (identical(pos, rev(negl))) {
      member_levels <- pos; fwd <- seq_along(pos); rev_ <- rev(fwd)
    } else {
      member_levels <- sort(unique(c(pos, negl)))
      fwd <- match(pos, member_levels); rev_ <- match(negl, member_levels)
    }
    vals <- .causilo_interp_quantiles(sorted, member_levels)
  }
  aligned <- lapply(seq_len(M), function(m) {
    idx <- torch::torch_tensor(if (signs[[m]] < 0) rev_ else fwd, dtype = torch::torch_long())
    v <- vals[m, , ]$index_select(-1L, idx)
    if (signs[[m]] < 0) -v else v
  })
  a <- torch::torch_stack(aligned)$sort(dim = 1L)[[1]]$narrow(1L, trim + 1L, M - 2L * trim)
  combined <- as.array(a$to(dtype = torch::torch_float64()))
  d <- dim(combined)
  combined <- matrix(colMeans(matrix(combined, nrow = d[[1]])), nrow = q)
  quant <- inverse(combined)
  switch(type,
    median = quant[, 1L],
    mean = {
      lower <- pmin(quant[, 1L], sc$limits[[1]])
      upper <- pmax(quant[, ncol(quant)], sc$limits[[2]])
      as.numeric(quant %*% CAUSILO_MEAN_WEIGHTS) +
        lower * (CAUSILO_MEAN_MASS[[1]] * 0.5) +
        upper * (CAUSILO_MEAN_MASS[[length(CAUSILO_MEAN_MASS)]] * 0.5)
    },
    quant)
}

# The trapezoid rule the trimmed mean integrates over
# (`execution/outputs.py`).
CAUSILO_MEAN_LEVELS <- c(0.0001, seq_len(999) / 1000, 0.9999)
CAUSILO_MEAN_MASS <- local({
  m <- diff(c(0, CAUSILO_MEAN_LEVELS, 1)); m / sum(m)
})
CAUSILO_MEAN_WEIGHTS <- (CAUSILO_MEAN_MASS[-length(CAUSILO_MEAN_MASS)] +
                           CAUSILO_MEAN_MASS[-1]) * 0.5

#' Build the Causilo 1.5 classifier predictor
#'
#' Reproduces `CausiloClassifier` (`causilo` 1.5.0) on the CPU path: the
#' recipe of [causilo_recipe_fit()] fits up to `n_estimators` members
#' (fewer when the table has too few distinct columns to tell them apart);
#' each sees its own feature view, column signs and cyclic class shift;
#' the shifts are undone, the logits averaged and passed through a softmax.
#' Light ("Causilo 1.5 (Light)" on the leaderboards) is the same weights
#' with one member: `n_estimators = 1` or `light = TRUE`.
#'
#' Classes are coded in label order (numeric, else code point), as the
#' reference's `LabelEncoder` codes them; classes absent from the training
#' rows cannot be predicted and come back with probability zero.
#' Categorical columns are those declared; their observed values, sorted,
#' become the category codes. [tabfound()] declares a frame's factor
#' columns and sorts their levels by code point first, so the codes are
#' the ones the reference gives the same labels.
#'
#' The model is large (275M parameters, a 24-layer prediction stack over
#' every row): eight members cost about eight times one, so on a CPU and
#' a larger table `n_estimators = 1` is the practical setting.
#'
#' @param ctx Loaded-model context from [load_backend_model()].
#' @param n_estimators Ensemble members; `NULL` is 8 (the reference's
#'   default).
#' @param light `TRUE` for Light, one member (the reference's `mode =
#'   "light"`). Give this or `n_estimators`, not both.
#' @param seed Seed of the ensemble plan (CPython's `random.Random`, so the
#'   reference's plan for the same seed); `NULL` for an unseeded plan.
#' @param predict_chunk_size Query rows per forward pass. Query rows read
#'   only context-derived state, so chunking changes nothing but float32
#'   rounding; the context is re-encoded with every chunk.
#' @param member_batch_size Members per forward pass (the reference runs
#'   at most 8 together).
#' @param kv_cache Encode each group of members' context once per
#'   `predict()` call and run every query chunk against it, instead of
#'   re-encoding the context with every chunk (see [causilo_kv_cache()]).
#'   [tabfound_cache()] keeps the caches across calls and sessions. The
#'   answer moves only by float32 rounding. A cache is large -- about
#'   197 KB per context row per member -- so with many context rows keep
#'   `n_estimators` (or `member_batch_size`) small when caching.
#' @param row_chunk_size Rows per pass through the row layers and the
#'   prediction layers (whose keys, the context rows, are fixed per layer
#'   first). Bounds the activations those stages hold; float32 rounding
#'   only.
#' @param col_chunk_size Columns per pass through the column layers.
#' @param categorical_features 1-based indices of categorical columns.
#' @keywords internal
causilo_classifier <- function(ctx, n_estimators = NULL, light = FALSE, seed = 42L,
                               predict_chunk_size = 1024L, member_batch_size = 8L,
                               kv_cache = FALSE, row_chunk_size = NULL,
                               col_chunk_size = NULL, categorical_features = NULL) {
  chunks <- .causilo_chunks(row_chunk_size, col_chunk_size)
  n_est <- .causilo_n_estimators(n_estimators, light)
  seed <- .causilo_check_seed(seed)
  batch <- max(1L, as.integer(member_batch_size))

  fit_fn <- function(X, y) {
    X <- .causilo_check_matrix(X)
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    if (anyNA(y)) cli::cli_abort("{.arg y} has missing values.")
    levels_ <- if (is.factor(y)) levels(y) else .causilo_class_order(unique(as.character(y)))
    categorical <- .kumo_categorical(categorical_features, ncol(X))
    rec <- causilo_recipe_fit(X, if (is.factor(y)) as.character(y) else y, TRUE,
                              categorical, n_est, seed)
    list(recipe = rec, class_levels = levels_, n_train = nrow(X),
         categorical = categorical)
  }

  .logits <- function(state, X) {
    rec <- state$recipe
    K <- length(rec$target$classes)
    qv <- causilo_recipe_apply(rec, X)
    out <- .causilo_member_outputs(ctx, rec, qv, batch, predict_chunk_size, K,
                                   caches = state$kv_caches, kv_cache = kv_cache,
                                   chunks = chunks)
    logits <- as.matrix(.causilo_mean_logits(rec, out))
    full <- matrix(-Inf, nrow(X), length(state$class_levels))
    full[, match(rec$target$classes, as.character(state$class_levels))] <- logits
    colnames(full) <- as.character(state$class_levels)
    full
  }

  predict_fn <- function(state, newdata, type = "class", ...) {
    X <- .causilo_check_matrix(newdata, state$recipe$encoder$width)
    lg <- .logits(state, X)
    if (type == "logits") return(lg)
    # Softmax in float32, as the reference.
    probs <- as.matrix(torch::nnf_softmax(torch::torch_tensor(lg, dtype = torch::torch_float32()),
                                          dim = -1L))
    colnames(probs) <- colnames(lg)
    if (type == "prob") return(probs)
    state$class_levels[max.col(probs, ties.method = "first")]
  }

  list(fit = fit_fn, predict = predict_fn,
       build_cache = function(state) .causilo_build_caches(ctx, state$recipe, batch, chunks))
}

#' Build the Causilo 1.5 regressor predictor
#'
#' Reproduces `CausiloRegressor` on the CPU path. Each member predicts 999
#' quantiles of the scaled target (levels `k / 1000`); members whose target
#' sign is negative are read at `1 - p` and negated. With fewer than four
#' members the members are averaged and the mean is each member's mean of
#' its 999 quantiles; with four or more, every requested level is a
#' trimmed mean across members (`count %/% 4` dropped at each end) and the
#' mean integrates the combined quantile function, its end pieces reaching
#' the training range. Light and default therefore define the mean
#' differently, as the reference does.
#'
#' `type = "grid"` returns the combined 999 quantiles, which is what
#' [tabfound_impute()] draws from.
#'
#' @inheritParams causilo_classifier
#' @keywords internal
causilo_regressor <- function(ctx, n_estimators = NULL, light = FALSE, seed = 42L,
                              predict_chunk_size = 1024L, member_batch_size = 8L,
                              kv_cache = FALSE, row_chunk_size = NULL,
                              col_chunk_size = NULL, categorical_features = NULL) {
  chunks <- .causilo_chunks(row_chunk_size, col_chunk_size)
  n_est <- .causilo_n_estimators(n_estimators, light)
  seed <- .causilo_check_seed(seed)
  batch <- max(1L, as.integer(member_batch_size))
  levels_ <- seq_len(CAUSILO_NUM_QUANTILES) / (CAUSILO_NUM_QUANTILES + 1)

  fit_fn <- function(X, y) {
    X <- .causilo_check_matrix(X)
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    y <- as.numeric(y)
    if (anyNA(y) || any(!is.finite(y))) cli::cli_abort("{.arg y} must be finite.")
    categorical <- .kumo_categorical(categorical_features, ncol(X))
    rec <- causilo_recipe_fit(X, y, FALSE, categorical, n_est, seed)
    list(recipe = rec, n_train = nrow(X), categorical = categorical)
  }

  predict_fn <- function(state, newdata, type = "mean",
                         quantiles = c(0.1, 0.5, 0.9), ...) {
    if (identical(type, "raw_quantiles")) type <- "grid"
    rec <- state$recipe
    X <- .causilo_check_matrix(newdata, rec$encoder$width)
    qv <- causilo_recipe_apply(rec, X)
    out <- .causilo_member_outputs(ctx, rec, qv, batch, predict_chunk_size,
                                   CAUSILO_NUM_QUANTILES, caches = state$kv_caches,
                                   kv_cache = kv_cache, chunks = chunks)
    res <- .causilo_regression(rec, out, type, if (type == "quantiles") quantiles)
    if (type == "quantiles") colnames(res) <- .quantile_colnames(quantiles)
    res
  }

  list(fit = fit_fn, predict = predict_fn,
       build_cache = function(state) .causilo_build_caches(ctx, state$recipe, batch, chunks),
       types = c("mean", "median", "quantiles", "grid"),
       quantile_levels = levels_)
}


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
#'   latents (`C x 256 x W`); `col_chunk_size` bounds the columns a pass
#'   holds;
#' * row interaction: a row's `4 + C` tokens at width `W`;
#' * in-context learning: every row at the prediction width `4W`.
#'
#' `row_chunk_size` bounds the rows of the last two; chunked, the cell grid,
#' the row tokens and the pooled rows of the whole pass stay resident
#' (`res`). One pass holds `member_batch_size` members (default 8, capped
#' at the members run) and the context plus one chunk of query rows; with
#' a cache, the larger of the two. A cache holds, per member, every
#' prediction layer's keys and values of the context rows, every column
#' layer's broadcast keys and values of the 256 latents and the sorted
#' context values: one group's for `kv_cache = TRUE`, every member's when
#' kept by [tabfound_cache()].
#'
#' @inheritParams tabicl_peak_terms
#' @keywords internal
causilo_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  m <- .causilo_model_config(config)
  W <- as.numeric(m$width)
  P <- CAUSILO_ROW_LATENTS * W
  K <- CAUSILO_ROW_LATENTS
  C <- as.numeric(n_features)
  n_ctx <- as.numeric(n_context)
  n_est <- max(1, as.numeric(opts$n_estimators %||% 8))
  if (isTRUE(opts$light)) n_est <- 1
  G <- max(1, min(n_est, as.numeric(opts$member_batch_size %||% 8)))
  # One pass holds the context and one chunk of query rows; a cached pass
  # holds whichever is larger, the context (while the cache is built) or
  # one chunk.
  q <- as.numeric(n_query)
  cached <- isTRUE(opts$kv_cache) || isTRUE(opts$kv_cache_kept)
  n <- if (cached) max(n_ctx, q) else n_ctx + q
  rc <- as.numeric(opts$row_chunk_size %||% Inf)
  cc <- as.numeric(opts$col_chunk_size %||% Inf)
  r_on <- is.finite(rc) && rc < n
  c_on <- is.finite(cc) && cc < C
  rows <- if (r_on) rc else n
  cols <- if (c_on) cc else C
  # What outlives a chunked stage: the cell grid `[n, C, W]` (both
  # chunkings), the row tokens `[n, K, W]` and the pooled rows `[n, P]`.
  res_col <- if (c_on || r_on) G * n * (C + K) * W else 0
  stages <- list(
    list(name = "column summaries", act = 0, res = 0, att = 0, prepass = 0),
    list(name = "column embedding",
         act = G * (n * cols * W + cols * CAUSILO_COLUMN_LATENTS * W),
         res = res_col, att = 0),
    list(name = "row interaction",
         act = G * rows * (K + C) * W, res = res_col, att = 0),
    list(name = "in-context learning", act = G * rows * P,
         res = if (r_on) G * n * P else 0, copies = "icl", att = 0)
  )
  cache <- 2 * (m$prediction_depth * n_ctx * P +
                  m$column_depth * C * CAUSILO_COLUMN_LATENTS * W) + n_ctx * C
  persistent <- if (isTRUE(opts$kv_cache_kept)) n_est * cache
                else if (isTRUE(opts$kv_cache)) G * cache else 0
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
    classifier    = causilo_classifier,
    regressor     = causilo_regressor,
    describe      = causilo_describe,
    peak_terms    = causilo_peak_terms,
    # The network embeds missing cells itself: a learned vector, with the
    # value and rank waves zeroed.
    kv_cache_capable   = TRUE,
    handles_missing    = TRUE,
    imputes_internally = FALSE,
    description   = "Causilo 1.5 (Nums AI) column/row latent transformer with a prediction stack",
    parity        = "causilo 1.5.0 (PyPI)"
  )
}
