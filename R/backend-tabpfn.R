# TabPFN v2 backend (Prior-Labs).
#
# Registers the `per_feature_transformer` architecture with the core:
# how to build it from a config, how checkpoint keys map onto R module
# paths, and how the classifier / regressor predictors work.
#
# Checkpoints are published as PyTorch Lightning pickles. Convert them
# once, offline, with `inst/python/tabpfn_convert_ckpt.py`; nothing in
# this file touches Python.

# ---------------------------------------------------------------------------
# Checkpoint key translation
# ---------------------------------------------------------------------------

# R torch's `nn_module_list` adds a `steps` component where PyTorch's
# `nn.Sequential` has none, so three prefixes need rewriting. Everything
# else -- `transformer_encoder.layers.<i>.*`, the attention `_w_qkv` /
# `_w_out`, `add_thinking_tokens.row_token_values`,
# `feature_positional_embedding_embeddings.*`, the regressor's
# `criterion.*` buffers -- is an identity mapping.
#
#   encoder.<i>.<rest>                -> encoder.steps.<i>.<rest>
#   y_encoder.<i>.<rest>              -> y_encoder.steps.<i>.<rest>
#   decoder_dict.standard.<i>.<rest>  -> decoder_dict.standard.steps.<i>.<rest>

#' Translate a TabPFN checkpoint key to its R module path
#' @keywords internal
tabpfn_translate_key <- function(key) {
  if (grepl("^encoder\\.", key)) {
    return(sub("^encoder\\.", "encoder.steps.", key))
  }
  if (grepl("^y_encoder\\.", key)) {
    return(sub("^y_encoder\\.", "y_encoder.steps.", key))
  }
  if (grepl("^decoder_dict\\.standard\\.", key)) {
    return(sub("^decoder_dict\\.standard\\.",
               "decoder_dict.standard.steps.", key))
  }
  key
}


# ---------------------------------------------------------------------------
# Pre-generated column embeddings
# ---------------------------------------------------------------------------

#' Load the 2000 x 48 pre-generated column embeddings bundled with the package
#'
#' TabPFN's "subspace" feature positional embedding draws a per-feature
#' 48-dim vector from a seeded generator. The reference implementation
#' overwrites the first 2000 rows with a fixed buffer for determinism;
#' that buffer ships in `inst/extdata/`.
#' @keywords internal
load_column_embeddings <- function(device = "cpu") {
  require_suggested("safetensors")
  path <- tabfound_file("extdata", "tabpfn_col_embedding.safetensors")
  if (!nzchar(path)) {
    cli::cli_abort(
      "Cannot locate {.file tabpfn_col_embedding.safetensors}; \\
       install the package or run from the repo root."
    )
  }
  obj <- safetensors::safe_load_file(path, framework = "torch")
  obj$column_embeddings$to(device = device)
}


# ---------------------------------------------------------------------------
# Backend hooks
# ---------------------------------------------------------------------------

#' @keywords internal
tabpfn_build <- function(config, task) {
  cli::cli_alert_info(
    "Building per_feature_transformer ({.val {config$head}}, \\
     {config$n_layers} layers, emb={config$embedding_dim})..."
  )
  per_feature_transformer(config)
}

#' @keywords internal
tabpfn_detect <- function(config) {
  identical(config$arch, "per_feature_transformer")
}

#' @keywords internal
tabpfn_task_of <- function(config) {
  switch(config$head %||% "",
         classifier = "classification",
         regressor  = "regression",
         NULL)
}


# ---------------------------------------------------------------------------
# Capability guard
# ---------------------------------------------------------------------------

# The predictors here serve all three TabPFN generations, and each one
# added a memory path the ones before it do not have. Asking an older
# network for a newer path has to fail rather than be silently dropped --
# a caller who set `kv_cache = TRUE` for the speed, or a chunk size to
# survive a large table, would otherwise never learn they did not get it.
#
# `NA` is not "asked for": it is the default, meaning "whatever the
# checkpoint says", and on a backend without stage chunking that is
# nothing.
# @keywords internal
.require_kv_cache_support <- function(ctx, kv_cache, save_peak_memory_factor,
                                      row_chunk_size = NA_integer_,
                                      col_chunk_size = NA_integer_) {
  asked <- function(x) !is.null(x) && !(length(x) == 1L && is.na(x))
  missing <- c(
    if (isTRUE(kv_cache) && !isTRUE(ctx$net$supports_kv_cache)) "kv_cache",
    if (!is.null(save_peak_memory_factor) &&
        !isTRUE(ctx$net$supports_chunked_eval)) "save_peak_memory_factor",
    if ((asked(row_chunk_size) || asked(col_chunk_size)) &&
        !isTRUE(ctx$net$supports_stage_chunking))
      c(if (asked(row_chunk_size)) "row_chunk_size",
        if (asked(col_chunk_size)) "col_chunk_size")
  )
  if (!length(missing)) return(invisible(TRUE))
  cli::cli_abort(c(
    "The {.val {ctx$backend$name}} backend does not support {.arg {missing}}.",
    i = "{.arg kv_cache} and {.arg save_peak_memory_factor} need \\
         TabPFN v2.5 or newer; {.arg row_chunk_size} and \\
         {.arg col_chunk_size} need v3."
  ))
  invisible(TRUE)
}


# The pre-generated column embedding table, or NULL for a network that
# does not use one. v3 replaced it with RoPE over the feature axis, and
# loading a 2000 x 48 buffer it will never read would be both wasteful and
# misleading about what the model does.
# @keywords internal
.column_embeddings_for <- function(net, device) {
  if (!isTRUE(net$needs_column_embeddings)) return(NULL)
  load_column_embeddings(device = device)
}

# The bar-distribution borders, wherever this generation keeps them. v2
# and v2.6 hang them off a `criterion` submodule; v3 registers them
# directly as `regression_borders`, on both heads.
# @keywords internal
.regression_borders_of <- function(net) {
  b <- net$criterion$borders %||% net$regression_borders
  if (is.null(b)) {
    cli::cli_abort("This network carries no bar-distribution borders.")
  }
  b
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

#' Build the TabPFN classifier predictor
#'
#' @param ctx Loaded-model context from [load_backend_model()].
#' @param ensemble_configs_dir Optional path to a directory produced by
#'   `inst/python/tabpfn_dump_ensemble.py --head classifier`. When
#'   supplied, `$predict_proba()` averages over all members; otherwise a
#'   single forward pass is used.
#' @param predict_chunk_size Integer. Max test rows per forward pass.
#' @param softmax_temperature Numeric. Decoder logits are divided by this
#'   before the softmax. Defaults to `0.9`, matching the reference
#'   `TabPFNClassifier` / `TabPFNRegressor` default; set to `1` for the
#'   untempered distribution.
#' @param trace_dir Optional directory. When set, each ensemble member's
#'   post-preprocessing inputs and raw logits are written there in the
#'   same layout the Python parity harness produces, so the two can be
#'   diffed file-for-file. See `inst/parity/`.
#' @param categorical_features Integer vector of 1-based column indices to
#'   treat as categorical, or `NULL` to infer. Only the ensemble path uses
#'   it: a single forward pass hands the matrix to the network unchanged.
#'   See [detect_categorical_features()] for what inference will and will
#'   not pick up on its own.
#' @param kv_cache Condition on the training rows once per `predict()`
#'   call and reuse that across every prediction chunk and ensemble
#'   member, instead of rebuilding the whole context per chunk. Cuts the
#'   cost of predicting many rows from `n_members * n_chunks` passes over
#'   the training set to `n_members`. Off by default. On `tabpfn` and
#'   `tabpfn3` it is exactly equivalent to an uncached pass, because every
#'   statistic those architectures fit comes from the training rows alone.
#'   On `tabpfn26` it is not: that generation fits its constant-column and
#'   informative-feature masks over train and test together, so caching
#'   them can change the prediction -- see [tabpfn26_kv_cache()].
#'   Supported by the `tabpfn` (v2.5), `tabpfn26` and `tabpfn3` backends.
#' @param save_peak_memory_factor Integer, or `NULL` (default) to disable.
#'   Splits each sublayer's work into this many chunks, shrinking the
#'   attention score matrix each one materialises. Bit-identical output,
#'   lower peak memory, slightly more overhead. Supported by the
#'   `tabpfn26` and `tabpfn3` backends.
#' @param row_chunk_size,col_chunk_size Stage-0-2 chunking, `tabpfn3`
#'   only. The first drives the cell embedding, distribution embedder and
#'   column aggregator a chunk of rows at a time, so the
#'   `(rows, columns, embedding)` tensor is never resident whole; the
#'   second bounds the column-wise pre-pass that builds the distribution
#'   embedder's inducing summaries. Left alone, both take the checkpoint's
#'   own values -- 2048 and 4, which is what the Python reference does by
#'   default on this architecture. `NULL` runs every row in one pass,
#'   which is what this package did before the chunking landed.
#'
#'   Unlike `save_peak_memory_factor` this is not bit-identical: it
#'   changes the batch shapes the attention kernel sees, and on the
#'   package's large fixture it moves the logits by 1.4e-5 of their own
#'   scale -- less than the reference's own chunked pass moves them.
#' @keywords internal
tabpfn_classifier <- function(ctx,
                              ensemble_configs_dir = NULL,
                              predict_chunk_size = 1024L,
                              softmax_temperature = 0.9,
                              trace_dir = NULL,
                              categorical_features = NULL,
                              kv_cache = FALSE,
                              save_peak_memory_factor = NULL,
                              row_chunk_size = NA_integer_,
                              col_chunk_size = NA_integer_) {
  .require_kv_cache_support(ctx, kv_cache, save_peak_memory_factor,
                            row_chunk_size, col_chunk_size)
  net     <- ctx$net
  dev     <- ctx$device
  col_emb <- .column_embeddings_for(net, dev)

  ensemble_configs <- if (!is.null(ensemble_configs_dir))
    load_ensemble_configs_from_dump(ensemble_configs_dir) else NULL

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) {
      cli::cli_abort("length(y) must equal nrow(X).")
    }
    X_train <- as.matrix(X); storage.mode(X_train) <- "double"
    if (is.factor(y)) {
      levels_ <- levels(y)
      y_int   <- as.integer(y) - 1L
    } else {
      levels_ <- sort(unique(y))
      y_int   <- match(y, levels_) - 1L
    }
    # Plain data only: this is what gets copied on fit and written by
    # `tabfound_save()`. The categorical schema is decided here, on the
    # training data, and reused at predict time -- inferring it again from
    # a test batch could disagree with itself.
    list(X_train = X_train, y_train_int = y_int, class_levels = levels_,
         n_train = nrow(X_train),
         categorical_features = detect_categorical_features(
           X_train, categorical_features))
  }

  .single_pass_chunk <- function(state, X_test_chunk, cache, pipelines = NULL) {
    x_te <- as_float_tensor(as.matrix(X_test_chunk), device = dev)$unsqueeze(1L)
    out <- if (is.null(cache)) {
      x_tr <- as_float_tensor(state$X_train, device = dev)$unsqueeze(1L)
      y_tr <- as_float_tensor(
        matrix(as.numeric(state$y_train_int), ncol = 1L), device = dev
      )$squeeze(-1L)$unsqueeze(1L)
      torch::with_no_grad(tabpfn_forward(
        net, x_tr, y_tr, x_te, col_emb,
        save_peak_memory_factor = save_peak_memory_factor,
        row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size))
    } else {
      torch::with_no_grad(tabpfn_forward(
        net, NULL, NULL, x_te, col_emb, kv_cache = cache[[1L]],
        save_peak_memory_factor = save_peak_memory_factor,
        row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size))
    }
    logits <- out$logits[1, , 1:length(state$class_levels)]
    if (softmax_temperature != 1) logits <- logits / softmax_temperature
    as.matrix(torch::nnf_softmax(logits, dim = -1L)$cpu())
  }

  .ensemble_chunk <- function(state, X_test_chunk, cache, pipelines) {
    apply_ensemble_predict_classifier(
      net = net, col_emb = col_emb,
      X_train = state$X_train, X_test = as.matrix(X_test_chunk),
      y_train = state$y_train_int, configs = ensemble_configs,
      n_classes = length(state$class_levels), device = dev,
      softmax_temperature = softmax_temperature, trace_dir = trace_dir,
      categorical_features = state$categorical_features %||% integer(),
      kv_caches = cache, save_peak_memory_factor = save_peak_memory_factor,
      row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size,
      pipelines = pipelines
    )
  }

  # Each member's preprocessing is fitted on the training rows, so it is
  # built once per `predict()` and shared by every chunk -- and by the
  # cache builder, which fits the same pipelines.
  .build_pipelines <- function(state) {
    if (is.null(ensemble_configs)) return(NULL)
    member_pipeline_store(state$X_train, ensemble_configs,
                          state$categorical_features %||% integer())
  }

  # One cache per member (or one, for a single pass), built once per
  # `predict()` call and shared by every chunk. Deliberately not stored on
  # the model: it is derived from the fitted state, `tabfound_save()`
  # writes plain data only, and a torch object outliving its session comes
  # back as a dangling pointer.
  .build_caches <- function(state, pipelines, force = FALSE) {
    # A cache carried on the fitted state was built once and, via
    # `tabfound_save()`, possibly in another session. It is what
    # `kv_cache = TRUE` buys, kept.
    if (!is.null(state$kv_caches)) return(state$kv_caches)
    if (!isTRUE(force) && !isTRUE(kv_cache)) return(NULL)
    if (is.null(ensemble_configs)) {
      x_tr <- as_float_tensor(state$X_train, device = dev)$unsqueeze(1L)
      y_tr <- as_float_tensor(
        matrix(as.numeric(state$y_train_int), ncol = 1L), device = dev
      )$squeeze(-1L)$unsqueeze(1L)
      no_rows <- torch::torch_zeros(c(1L, 0L, x_tr$size(3)), device = dev)
      return(list(torch::with_no_grad(tabpfn_forward(
        net, x_tr, y_tr, no_rows, col_emb, return_kv_cache = TRUE,
        save_peak_memory_factor = save_peak_memory_factor,
        row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size
      ))$kv_cache))
    }
    build_member_kv_caches(
      net, col_emb, state$X_train, state$y_train_int, ensemble_configs,
      member_y = function(cfg, y) cfg$class_perm[as.integer(y) + 1L],
      device = dev,
      categorical_features = state$categorical_features %||% integer(),
      save_peak_memory_factor = save_peak_memory_factor,
      row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size,
      pipelines = pipelines
    )
  }

  predict_fn <- function(state, newdata, type = "class", ...) {
    X_te <- as.matrix(newdata)
    pipelines <- .build_pipelines(state)
    caches <- .build_caches(state, pipelines)
    chunk_fn <- if (is.null(ensemble_configs)) .single_pass_chunk else .ensemble_chunk
    probs <- chunk_apply(X_te, predict_chunk_size,
                         function(chunk) chunk_fn(state, chunk, caches, pipelines))
    colnames(probs) <- as.character(state$class_levels)
    if (type == "prob") return(probs)
    state$class_levels[max.col(probs, ties.method = "first")]
  }

  list(fit = fit_fn, predict = predict_fn,
       build_cache = function(state) {
         .build_caches(state, .build_pipelines(state), force = TRUE)
       },
       ensemble_configs = ensemble_configs)
}


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------

#' Build the TabPFN regressor predictor
#'
#' @inheritParams tabpfn_classifier
#' @keywords internal
tabpfn_regressor <- function(ctx,
                             ensemble_configs_dir = NULL,
                             predict_chunk_size = 1024L,
                             softmax_temperature = 0.9,
                             trace_dir = NULL,
                             categorical_features = NULL,
                             kv_cache = FALSE,
                             save_peak_memory_factor = NULL,
                             row_chunk_size = NA_integer_,
                             col_chunk_size = NA_integer_) {
  .require_kv_cache_support(ctx, kv_cache, save_peak_memory_factor,
                            row_chunk_size, col_chunk_size)
  net     <- ctx$net
  dev     <- ctx$device
  col_emb <- .column_embeddings_for(net, dev)
  borders <- .regression_borders_of(net)

  ensemble_configs <- if (!is.null(ensemble_configs_dir))
    load_ensemble_configs_from_dump(ensemble_configs_dir) else NULL

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) {
      cli::cli_abort("length(y) must equal nrow(X).")
    }
    X_train <- as.matrix(X); storage.mode(X_train) <- "double"
    y <- as.numeric(y)
    # The bar-distribution borders live in z-normalized target space:
    # the reference standardizes y before the forward pass and
    # un-standardizes predictions. Population std (ddof = 0) + eps.
    y_mean <- mean(y)
    y_std  <- sqrt(mean((y - y_mean) ^ 2)) + 1e-20
    list(X_train = X_train, y_train = y, y_mean = y_mean, y_std = y_std,
         n_train = nrow(X_train),
         categorical_features = detect_categorical_features(
           X_train, categorical_features))
  }

  .single_pass_chunk <- function(state, X_test_chunk, quantiles, cache,
                                 pipelines = NULL) {
    x_te <- as_float_tensor(as.matrix(X_test_chunk), device = dev)$unsqueeze(1L)
    out <- if (is.null(cache)) {
      x_tr <- as_float_tensor(state$X_train, device = dev)$unsqueeze(1L)
      y_z  <- (state$y_train - state$y_mean) / state$y_std
      y_tr <- as_float_tensor(matrix(y_z, ncol = 1L),
                              device = dev)$squeeze(-1L)$unsqueeze(1L)
      torch::with_no_grad(tabpfn_forward(
        net, x_tr, y_tr, x_te, col_emb,
        save_peak_memory_factor = save_peak_memory_factor,
        row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size))
    } else {
      torch::with_no_grad(tabpfn_forward(
        net, NULL, NULL, x_te, col_emb, kv_cache = cache[[1L]],
        save_peak_memory_factor = save_peak_memory_factor,
        row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size))
    }
    logits <- out$logits[1, , ]
    if (softmax_temperature != 1) logits <- logits / softmax_temperature
    list(
      mean      = as.numeric(bar_logits_to_mean(logits, borders)$cpu()) *
                    state$y_std + state$y_mean,
      quantiles = as.matrix(bar_logits_to_quantiles(logits, borders, quantiles)$cpu()) *
                    state$y_std + state$y_mean
    )
  }

  .ensemble_chunk <- function(state, X_test_chunk, quantiles, cache,
                              pipelines) {
    apply_ensemble_predict_regressor(
      net, col_emb, state$X_train, X_test_chunk, state$y_train,
      configs = ensemble_configs, borders = borders,
      quantiles = quantiles, device = dev,
      softmax_temperature = softmax_temperature, trace_dir = trace_dir,
      categorical_features = state$categorical_features %||% integer(),
      kv_caches = cache, save_peak_memory_factor = save_peak_memory_factor,
      row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size,
      pipelines = pipelines
    )
  }

  # One fit per member per `predict()`; see the classifier.
  .build_pipelines <- function(state) {
    if (is.null(ensemble_configs)) return(NULL)
    member_pipeline_store(state$X_train, ensemble_configs,
                          state$categorical_features %||% integer())
  }

  # See the classifier's `.build_caches()` for why this lives in the call
  # rather than on the model. The per-member target is the z-standardised
  # one, put through that member's target transform where it has one.
  .build_caches <- function(state, pipelines, force = FALSE) {
    # A cache carried on the fitted state was built once and, via
    # `tabfound_save()`, possibly in another session. It is what
    # `kv_cache = TRUE` buys, kept.
    if (!is.null(state$kv_caches)) return(state$kv_caches)
    if (!isTRUE(force) && !isTRUE(kv_cache)) return(NULL)
    y_z <- (state$y_train - state$y_mean) / state$y_std
    if (is.null(ensemble_configs)) {
      x_tr <- as_float_tensor(state$X_train, device = dev)$unsqueeze(1L)
      y_tr <- as_float_tensor(matrix(y_z, ncol = 1L),
                              device = dev)$squeeze(-1L)$unsqueeze(1L)
      no_rows <- torch::torch_zeros(c(1L, 0L, x_tr$size(3)), device = dev)
      return(list(torch::with_no_grad(tabpfn_forward(
        net, x_tr, y_tr, no_rows, col_emb, return_kv_cache = TRUE,
        save_peak_memory_factor = save_peak_memory_factor,
        row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size
      ))$kv_cache))
    }
    build_member_kv_caches(
      net, col_emb, state$X_train, y_z, ensemble_configs,
      member_y = function(cfg, y) {
        if (is.null(cfg$target_transform_lambda)) return(y)
        apply_target_transform(
          y, fit_target_transform_state(y, cfg$target_transform_lambda[1]))
      },
      device = dev,
      categorical_features = state$categorical_features %||% integer(),
      save_peak_memory_factor = save_peak_memory_factor,
      row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size,
      pipelines = pipelines
    )
  }

  .common <- function(state, newdata, quantiles) {
    X_te <- as.matrix(newdata)
    pipelines <- .build_pipelines(state)
    caches <- .build_caches(state, pipelines)
    chunk_fn <- if (is.null(ensemble_configs)) .single_pass_chunk else .ensemble_chunk
    n <- nrow(X_te)
    if (n <= predict_chunk_size) {
      return(chunk_fn(state, X_te, quantiles, caches, pipelines))
    }
    starts <- seq(1L, n, by = as.integer(predict_chunk_size))
    chunks <- lapply(seq_along(starts), function(k) {
      # Between chunks: the previous one's tensors are dead, and the next
      # one allocates before R would otherwise collect them.
      if (k > 1L) collect_between_chunks()
      s <- starts[[k]]
      e <- min(s + as.integer(predict_chunk_size) - 1L, n)
      chunk_fn(state, X_te[s:e, , drop = FALSE], quantiles, caches, pipelines)
    })
    list(mean      = unlist(lapply(chunks, `[[`, "mean")),
         quantiles = do.call(rbind, lapply(chunks, `[[`, "quantiles")),
         avg_probs = do.call(rbind, lapply(chunks, `[[`, "avg_probs")))
  }

  predict_fn <- function(state, newdata, type = "mean",
                         quantiles = c(0.1, 0.5, 0.9),
                         n_samples = 100L, seed = NULL, ...) {
    if (type == "mean") {
      return(.common(state, newdata, quantiles = 0.5)$mean)
    }
    if (type == "quantiles") {
      q <- .common(state, newdata, quantiles = quantiles)$quantiles
      colnames(q) <- paste0("q", format(quantiles, trim = TRUE, drop0trailing = TRUE))
      return(q)
    }
    # type == "sample".
    #
    # With an ensemble, sampling goes through the same object the mean
    # and the quantiles do: the members' bucket probabilities, translated
    # onto shared borders and averaged, are the ensemble's predictive
    # distribution. The reference already takes its log and hands that to
    # the head as pseudo-logits for `mean()` and `icdf()`; drawing from
    # the same pseudo-logits is the ensemble's sampler, and it is what
    # makes a draw agree with the quantiles reported beside it. (This
    # used to warn and quietly use one unensembled forward pass.)
    if (!is.null(ensemble_configs)) {
      probs <- .common(state, newdata, quantiles = 0.5)$avg_probs
      raw_borders <- borders * state$y_std + state$y_mean
      pseudo <- torch::torch_log(torch::torch_tensor(
        probs, dtype = torch::torch_float(), device = dev))
      return(as.matrix(bar_logits_to_samples(pseudo, raw_borders,
                                             n_samples = as.integer(n_samples),
                                             seed = seed)$cpu()))
    }
    x_tr <- as_float_tensor(state$X_train, device = dev)$unsqueeze(1L)
    y_z  <- (state$y_train - state$y_mean) / state$y_std
    y_tr <- as_float_tensor(matrix(y_z, ncol = 1L),
                            device = dev)$squeeze(-1L)$unsqueeze(1L)
    x_te <- as_float_tensor(as.matrix(newdata), device = dev)$unsqueeze(1L)
    out <- torch::with_no_grad({
      net(x_tr, y_tr, x_te, column_embeddings = col_emb)
    })
    as.matrix(bar_logits_to_samples(out$logits[1, , ], borders,
                                    n_samples = as.integer(n_samples),
                                    seed = seed)$cpu()) *
      state$y_std + state$y_mean
  }

  list(fit = fit_fn, predict = predict_fn,
       build_cache = function(state) {
         .build_caches(state, .build_pipelines(state), force = TRUE)
       },
       types = c("mean", "quantiles", "sample"),
       borders = as.numeric(borders$cpu()),
       ensemble_configs = ensemble_configs)
}


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

#' Stage list for the TabPFN v2.5 diagram
#'
#' One embedding pass, then 24 identical layers that each attend twice --
#' across a row's feature groups, then down each feature's column -- and
#' a two-layer decoder. The two attentions are the architecture: nothing
#' else in the stack mixes information between rows or between columns.
#'
#' @param config The checkpoint's parsed `config.json`.
#' @param task `"classification"` or `"regression"`.
#' @keywords internal
tabpfn_describe <- function(config, task) {
  E   <- as.integer(config$embedding_dim)
  H   <- as.integer(config$n_heads %||% 3L)
  L   <- as.integer(config$n_layers)
  ff  <- as.integer(config$mlp_hidden_dim %||% (2L * E))
  grp <- as.integer(config$features_per_group %||% 3L)
  clf <- identical(config$head, "classifier")
  n_out <- if (clf) as.integer(config$n_out_classes %||% 10L)
           else as.integer(config$n_bar_bins %||% 5000L)
  ver <- config$tabpfn_version %||% "2"
  lyr <- "transformer_encoder.layers.0"

  stages <- list(
    arch_input_stage(),
    arch_stage(
      "encoder", "Feature-group encoder", kind = "embed", group = "Embed",
      detail = sprintf("Linear(%d -> %d), no bias: %d values + %d NaN flags",
                       2L * grp, E, grp, grp),
      shape = "(B, n, F, E)", prefix = "encoder"),
    arch_stage(
      "y_encoder", "Target encoder", kind = "embed", group = "Embed",
      detail = "Linear(2 -> E); added on the training rows only",
      prefix = "y_encoder"),
    arch_stage(
      "pos", "Feature positional embedding", kind = "embed", group = "Embed",
      detail = sprintf("subspace: fixed 48-d per column, Linear(48 -> %d)", E),
      prefix = "feature_positional_embedding_embeddings"),
    arch_stage(
      "think", "Thinking rows", kind = "embed", group = "Embed",
      detail = "64 learned rows prepended to the context",
      shape = "(B, 64 + n, F + 1, E)",
      prefix = "add_thinking_tokens"),
    arch_stage(
      "layers", "Per-feature encoder layer", kind = "attention",
      group = "Transformer", repeats = L,
      detail = sprintf("post-norm; %d heads of %d; MLP %d -> %d", H,
                       as.integer(E / H), E, ff),
      shape = "(B, 64 + n, F + 1, E)", prefix = "transformer_encoder",
      children = list(
        arch_stage("a_feat", "Attention between features", kind = "attention",
                   axis = "features",
                   prefix = paste0(lyr, ".self_attn_between_features")),
        arch_stage("ln1", "LayerNorm", kind = "norm"),
        arch_stage("a_item", "Attention between items", kind = "attention",
                   axis = "rows",
                   prefix = paste0(lyr, ".self_attn_between_items")),
        arch_stage("ln2", "LayerNorm", kind = "norm"),
        arch_stage("mlp", "MLP, GELU", kind = "ffn",
                   prefix = paste0(lyr, ".mlp")),
        arch_stage("ln3", "LayerNorm", kind = "norm")
      )),
    arch_stage(
      "decoder", "Decoder MLP", kind = "decode", group = "Decode",
      detail = sprintf("target column only: %d -> 384 -> %d, GELU", E, n_out),
      shape = sprintf("(n_test, %d)", n_out),
      prefix = "decoder_dict"),
    if (clf)
      arch_output_stage("Class logits",
                        detail = sprintf("%d slots; softmax over the classes seen", n_out))
    else
      arch_output_stage("Bar distribution",
                        detail = sprintf("%d bins over the scaled target", n_out))
  )

  list(
    title = sprintf("TabPFN v%s - %s", ver,
                    if (clf) "classifier" else "regressor"),
    subtitle = "Prior-Labs  -  per-feature transformer, two-way attention",
    facts = c(
      "Layers"              = L,
      "Embedding width"     = E,
      "Attention heads"     = sprintf("%d x %d", H, as.integer(E / H)),
      "MLP width"           = ff,
      "Features per group"  = grp,
      "Thinking rows"       = 64L,
      "Attention per layer" = "2 (features, rows)",
      "Output width"        = n_out,
      "Normalisation"       = "LayerNorm, post, affine-free",
      "KV cache"            = "exact"
    ),
    stages = Filter(Negate(is.null), stages)
  )
}


# ---------------------------------------------------------------------------
# Memory scaling
# ---------------------------------------------------------------------------

# Pull a dimension out of the config's recorded parameter shapes, which
# the converted checkpoints carry precisely so questions like "how wide
# is a head?" can be answered without the weights.
# @keywords internal
.shape_dim <- function(config, key, index, default) {
  s <- config$state_dict_shapes[[key]]
  if (is.null(s) || length(s) < index) return(default)
  as.numeric(unlist(s)[index])
}

#' TabPFN v2's peak-memory shapes
#'
#' A per-feature transformer: the state is `rows x feature-groups x
#' embedding_dim`, with three columns to a group and the target riding
#' along as one more group. Every layer attends twice -- across rows for
#' each feature group, then across feature groups for each row -- so the
#' two sublayers are separate stages with quite different score shapes.
#'
#' Both v2.5 and v2.6 use this; `save_peak_memory_factor`, which v2.6
#' honours, is applied by the core rather than here, since it divides
#' whatever the stages turn out to be.
#'
#' @inheritParams mitra_peak_terms
#' @keywords internal
tabpfn_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  e <- as.numeric(config$embedding_dim)
  h <- as.numeric(config$n_heads)
  l <- as.numeric(config$n_layers)
  gs <- as.numeric(config$features_per_group %||% 3)
  cap <- as.numeric(config$max_num_features %||% Inf)
  p <- min(n_features, if (is.finite(cap)) cap else n_features)
  # Feature groups, plus one for the target's own token.
  fg <- max(1, ceiling(p / gs)) + 1
  think <- .shape_dim(config, "add_thinking_tokens.row_token_values", 1L, 0)
  hd <- .shape_dim(config,
                   "transformer_encoder.layers.0.self_attn_between_items._w_qkv",
                   3L, e / max(h, 1))

  nq <- .resident_query(n_query, opts)
  n  <- n_context + nq + think
  # The ensemble is one pass per stored config; without a dump loaded
  # there is exactly one member.
  n_est <- max(1, as.numeric(opts$n_estimators %||% 1))

  list(
    stages = list(
      list(name = "attention between items",
           act = n * fg * e,
           # No mask, so torch's fused SDPA takes a memory-efficient
           # kernel and the `(n, n)` scores are never materialised. See
           # the note in `.icl_family_terms()` for the measurement.
           att = 0),
      list(name = "attention between features",
           act = n * fg * e,
           # Unmasked as well, and over the feature axis rather than the
           # row axis, so its scores would be `fg x fg` even if they were
           # materialised. They are not.
           att = 0)
    ),
    # The cache keeps head 0's keys and values only -- the test-row branch
    # broadcasts one head across all of them -- which is why this is
    # `head_dim` and not `embedding_dim` wide.
    persistent = if (isTRUE(opts$kv_cache))
                   n_est * l * 2 * fg * n_context * hd else 0,
    n_estimators = n_est
  )
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#' @keywords internal
register_tabpfn_backend <- function() {
  register_backend(
    name          = "tabpfn",
    build         = tabpfn_build,
    translate_key = tabpfn_translate_key,
    detect        = tabpfn_detect,
    task_of       = tabpfn_task_of,
    classifier    = tabpfn_classifier,
    regressor     = tabpfn_regressor,
    describe      = tabpfn_describe,
    peak_terms    = tabpfn_peak_terms,
    # NaN is a first-class input: the encoder appends an is-missing
    # indicator channel alongside the zero-filled value.
    kv_cache_capable = TRUE,
    handles_missing = TRUE,
    description   = "TabPFN v2 per-feature transformer (Prior-Labs)",
    parity        = "tabpfn (PyPI)"
  )
}
