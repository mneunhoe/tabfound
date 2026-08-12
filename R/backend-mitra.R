# Mitra backend (AutoGluon / Amazon Science).
#
# A 12-layer transformer with 2-D attention: each layer attends once
# across observations (rows) and once across features (columns), with an
# MLP after each. The target is carried as an extra feature column, the
# same trick TabPFN uses.
#
# Structurally the simplest of the four backends -- no set transformer,
# no inducing points, no RoPE, no scalable softmax. The one distinctive
# piece is the input embedding: instead of scaling feature values, Mitra
# replaces each value by its *rank* among 999 quantiles of the support
# set, which makes the representation invariant to any monotone
# transformation of a column.
#
# Ships `model.safetensors` + `config.json` on the Hub under Apache-2.0,
# so nothing here needs a conversion step or a license caveat.
#
# Reference: https://github.com/autogluon/autogluon
#            tabular/src/autogluon/tabular/models/mitra/_internal/models/

# ---------------------------------------------------------------------------
# Input embedding
# ---------------------------------------------------------------------------

#' Quantile-rank embedding of the feature matrix
#'
#' Replaces every value by its bucket index among the 999 quantiles of
#' the support column, divides by the number of support rows to land in
#' `[0, 1]`, then centres and scales by the support's own mean and
#' population variance. Query values are bucketed against the *support's*
#' quantiles, so nothing about the test set leaks into the mapping.
#'
#' Carries no parameters — it is a fixed transformation, which is why the
#' checkpoint has nothing for it.
#'
#' **Missing values silently delete a column.** `torch.quantile` returns
#' all-`NaN` quantiles for a column containing one, `searchsorted`
#' against all-`NaN` boundaries returns 0 for every value, the column's
#' variance is then zero, and the zero-variance guard below flattens it
#' to zeros. No `NaN` reaches the output and no error is raised — the
#' feature is simply gone. This is the reference's behaviour, reproduced
#' faithfully — and it is why [mitra_preprocessor_fit()] mean-imputes
#' before anything reaches here, exactly as AutoGluon's own preprocessor
#' does. Nothing in the normal flow can hit this path; the note stands
#' because a caller invoking this function directly still can.
#'
#' @param x_support `(B, S, F)`; @param x_query `(B, Q, F)`.
#' @return A list with the transformed `support` and `query`.
#' @keywords internal
mitra_quantile_embedding <- function(x_support, x_query) {
  fit <- mitra_quantile_fit(x_support)
  list(support = fit$support,
       query   = mitra_quantile_apply(x_query, fit$state))
}

#' Fit the quantile embedding on the support set
#'
#' Everything the mapping depends on -- the 999 boundaries, the mean, the
#' standard deviation, and which columns were constant -- comes from the
#' support rows alone, which is what lets a [mitra_kv_cache()] carry it
#' forward to a later batch of query rows.
#'
#' @param x_support `(B, S, F)`.
#' @return `list(support, state)`: the embedded support rows, and the
#'   state [mitra_quantile_apply()] needs.
#' @keywords internal
mitra_quantile_fit <- function(x_support) {
  b <- x_support$size(1); s <- x_support$size(2); f <- x_support$size(3)
  dev <- x_support$device

  q <- torch::torch_arange(1L, 999L, dtype = torch::torch_float(),
                           device = dev)$div(1000)
  # (999, B, F) -> (B*F, 999)
  quantiles <- torch::torch_quantile(x_support, q, dim = 2L)
  quantiles <- quantiles$permute(c(2L, 3L, 1L))$contiguous()$reshape(c(b * f, 999L))

  sup <- .mitra_bucketize(x_support, quantiles, b, f) / s
  mu <- sup$sum(dim = 2L, keepdim = TRUE) / s
  sup <- sup - mu

  # Population variance of the already-centred support.
  v <- (sup^2)$sum(dim = 2L, keepdim = TRUE) / s
  sd <- v$sqrt()
  sup <- sup / sd

  # A constant column has zero variance; the division above leaves NaN
  # there, so replace it rather than letting it propagate.
  zero <- v == 0
  state <- list(quantiles = quantiles, n_support = s, mu = mu, sd = sd,
                zero = zero)
  list(support = torch::torch_where(zero, torch::torch_zeros_like(sup), sup),
       state = state)
}

#' Apply a fitted quantile embedding to fresh rows
#' @param x `(B, N, F)`; @param state From [mitra_quantile_fit()].
#' @keywords internal
mitra_quantile_apply <- function(x, state) {
  b <- x$size(1); f <- x$size(3)
  out <- .mitra_bucketize(x, state$quantiles, b, f) / state$n_support
  out <- (out - state$mu) / state$sd
  torch::torch_where(state$zero, torch::torch_zeros_like(out), out)
}

# Bucket every value against its own (batch, feature) column's
# boundaries, and hand back `(B, N, F)`.
#
# `torch.bucketize(v, boundaries)` is `searchsorted(boundaries, v)`; the
# batched form lets every column use its own boundaries, which is what
# the reference's `vmap` achieves.
# @keywords internal
.mitra_bucketize <- function(x, quantiles, b, f) {
  n <- x$size(2)
  flat <- x$permute(c(1L, 3L, 2L))$contiguous()$reshape(c(b * f, n))
  out <- torch::torch_searchsorted(quantiles, flat$contiguous())$
    to(dtype = torch::torch_float())
  out$reshape(c(b, f, n))$permute(c(1L, 3L, 2L))$contiguous()
}


#' Per-cell linear embedding
#' @keywords internal
mitra_x_embedding <- torch::nn_module(
  "MitraXEmbedding",
  initialize = function(dim) {
    self$x_embedding <- torch::nn_linear(1L, dim)
  },
  # (B, S, F) -> (B, S, F, D)
  forward = function(x) self$x_embedding(x$unsqueeze(-1L))
)


#' Target embedding for classification
#'
#' The support rows look up a learned per-class vector; the query rows
#' all get the same learned "unknown" vector, so the model can tell which
#' rows it is being asked about purely from the target column.
#' @keywords internal
mitra_y_embedding_classes <- torch::nn_module(
  "MitraYEmbeddingClasses",
  initialize = function(dim, n_classes) {
    self$y_embedding <- torch::nn_embedding(n_classes, dim)
    # Masking is modelled as one extra learned class.
    self$y_mask <- torch::nn_embedding(1L, dim)
  },
  #' The "unknown label" vector every query row gets, `(B, n_query, 1, D)`.
  #' @keywords internal
  mask_embedding = function(n_query, b, device) {
    mitra_mask_embedding(self$y_mask, n_query, b, device)
  },
  #' @keywords internal
  support_embedding = function(y_support) {
    idx <- y_support$to(dtype = torch::torch_long())$unsqueeze(-1L)
    # R torch's nn_embedding indexes from 1.
    self$y_embedding(idx + 1L)                                 # (B, S, 1, D)
  },
  forward = function(y_support, n_query) {
    list(support = self$support_embedding(y_support),
         query = self$mask_embedding(n_query, y_support$size(1),
                                     y_support$device))
  }
)

# The query-side target embedding, which is the same vector for every
# query row and does not depend on the support set at all -- so a cached
# pass rebuilds it rather than storing it.
# @keywords internal
mitra_mask_embedding <- function(y_mask, n_query, b, device) {
  y_mask(torch::torch_ones(c(b, n_query, 1L), dtype = torch::torch_long(),
                           device = device))
}

#' Target embedding for scalar regression
#' @keywords internal
mitra_y_embedding_regression <- torch::nn_module(
  "MitraYEmbeddingRegression",
  initialize = function(dim) {
    self$y_embedding <- torch::nn_linear(1L, dim)
    self$y_mask <- torch::nn_embedding(1L, dim)
  },
  #' @keywords internal
  mask_embedding = function(n_query, b, device) {
    mitra_mask_embedding(self$y_mask, n_query, b, device)
  },
  #' @keywords internal
  support_embedding = function(y_support) {
    # Linear(1, D) on (B, S, 1) gives (B, S, D); the feature axis is
    # added here so both heads hand back the same shape.
    self$y_embedding(y_support$unsqueeze(-1L))$unsqueeze(3L)
  },
  forward = function(y_support, n_query) {
    list(support = self$support_embedding(y_support),
         query = self$mask_embedding(n_query, y_support$size(1),
                                     y_support$device))
  }
)


# ---------------------------------------------------------------------------
# Attention and layers
# ---------------------------------------------------------------------------

#' Multi-head attention with four separate projections
#'
#' Plain `q`/`k`/`v`/`o` linears with biases and the default SDPA scale —
#' no fused packing, no per-head norms, no learned scale.
#' @keywords internal
mitra_attention <- torch::nn_module(
  "MitraAttention",
  initialize = function(dim, n_heads) {
    self$dim <- as.integer(dim)
    self$n_heads <- as.integer(n_heads)
    self$head_dim <- as.integer(dim / n_heads)
    self$q <- torch::nn_linear(dim, dim, bias = TRUE)
    self$k <- torch::nn_linear(dim, dim, bias = TRUE)
    self$v <- torch::nn_linear(dim, dim, bias = TRUE)
    self$o <- torch::nn_linear(dim, dim, bias = TRUE)
  },
  heads = function(x) {
    x$view(c(x$size(1), x$size(2), self$n_heads, self$head_dim))$
      permute(c(1L, 3L, 2L, 4L))
  },

  # (B, T, D) inputs; key/value may be a different sequence.
  # @param cached_kv Optional `list(key, value)` from [cache_kv()],
  #   replacing `key`/`value` and their projections.
  forward = function(query, key = NULL, value = NULL, cached_kv = NULL) {
    q <- self$heads(self$q(query))
    if (is.null(cached_kv)) {
      if (is.null(key) || is.null(value)) {
        cli::cli_abort("Supply {.arg key} and {.arg value}, or {.arg cached_kv}.")
      }
      k <- self$heads(self$k(key)); v <- self$heads(self$v(value))
    } else {
      k <- cached_kv$key; v <- cached_kv$value
    }
    ctx <- sdpa(
      query = q, key = k, value = v, dropout_p = 0
    )
    ctx <- ctx$permute(c(1L, 3L, 2L, 4L))$contiguous()$
      reshape(c(query$size(1), query$size(2), self$dim))
    self$o(ctx)
  },

  #' Key/value projections of `key`, for a KV cache.
  #' @keywords internal
  cache_kv = function(key) {
    list(key   = self$heads(self$k(key))$detach()$contiguous(),
         value = self$heads(self$v(key))$detach()$contiguous())
  }
)


#' One Mitra layer: attend across rows, MLP, attend across features, MLP
#'
#' Pre-norm throughout. The row attention is the only place the query
#' rows see the support rows — they attend *to* the support and never to
#' each other, which is what keeps predictions independent across test
#' rows. The feature attention is self-attention on both sides.
#' @keywords internal
mitra_layer <- torch::nn_module(
  "MitraLayer",
  initialize = function(dim, n_heads) {
    self$layer_norm1 <- affine_layer_norm(dim)
    self$attention1  <- mitra_attention(dim, n_heads)
    self$layer_norm2 <- affine_layer_norm(dim)
    self$linear1 <- torch::nn_linear(dim, dim * 4L, bias = TRUE)
    self$linear2 <- torch::nn_linear(dim * 4L, dim, bias = TRUE)

    self$layer_norm3 <- affine_layer_norm(dim)
    self$attention2  <- mitra_attention(dim, n_heads)
    self$layer_norm4 <- affine_layer_norm(dim)
    self$linear3 <- torch::nn_linear(dim, dim * 4L, bias = TRUE)
    self$linear4 <- torch::nn_linear(dim * 4L, dim, bias = TRUE)
  },

  # (B, S, F, D) -> (B*F, S, D): every feature column becomes its own
  # sequence over rows.
  to_rows = function(x) {
    x$permute(c(1L, 3L, 2L, 4L))$contiguous()$
      reshape(c(x$size(1) * x$size(3), x$size(2), x$size(4)))
  },
  from_rows = function(x, b, s, f, d) {
    x$reshape(c(b, f, s, d))$permute(c(1L, 3L, 2L, 4L))$contiguous()
  },

  #' Key/value projections of the support rows' observation attention.
  #'
  #' This is the layer's entire dependence on the support set: nothing
  #' downstream of it -- the two MLPs and the feature attention -- looks
  #' beyond a single row.
  #' @keywords internal
  row_kv = function(support) {
    self$attention1$cache_kv(self$to_rows(self$layer_norm1(support)))
  },

  #' Push one stream of rows through the layer against a support cache.
  #'
  #' The support rows and the query rows run through *the same* four
  #' sublayers against *the same* keys and values -- the support set
  #' attends to itself, and that is what "attends to the support" means
  #' for it too. So this one function is the whole layer, called twice.
  #' @keywords internal
  #' @param save_peak_memory_factor Split each of the four sublayers'
  #'   work into this many chunks of rows. Every one of them is
  #'   independent across rows -- a query row attends to the cached
  #'   support and never to another query row, the feature attention is a
  #'   sequence per row, and both MLPs are elementwise -- so this
  #'   reorganises work that was already separate and changes nothing
  #'   about the arithmetic.
  #'
  #'   The split is on the row axis rather than through
  #'   [chunked_evaluate()]'s leading-dimension fold, because
  #'   `to_rows()` folds `(batch, features)` into the attention batch:
  #'   flattening `(batch, rows)` first would hand a chunk spanning two
  #'   datasets to a key/value cache indexed by one.
  forward_rows = function(x, kv, save_peak_memory_factor = NULL) {
    b <- x$size(1); f <- x$size(3); d <- x$size(4)
    spmf <- save_peak_memory_factor
    ln1 <- self$layer_norm1; a1 <- self$attention1
    ln2 <- self$layer_norm2; l1 <- self$linear1; l2 <- self$linear2
    ln3 <- self$layer_norm3; a2 <- self$attention2
    ln4 <- self$layer_norm4; l3 <- self$linear3; l4 <- self$linear4

    # --- attention across observations ---
    x <- chunked_evaluate_axis(function(z) {
      self$from_rows(a1(self$to_rows(ln1(z)), cached_kv = kv),
                     b, z$size(2), f, d)
    }, x, spmf, axis = 2L)

    # --- MLP ---
    x <- chunked_evaluate_axis(function(z) {
      l2(torch::nnf_gelu(l1(ln2(z))))
    }, x, spmf, axis = 2L)

    # --- attention across features (each row is its own sequence) ---
    x <- chunked_evaluate_axis(function(z) {
      zn <- ln3(z)$reshape(c(b * z$size(2), f, d))
      a2(zn, zn, zn)$reshape(c(b, z$size(2), f, d))
    }, x, spmf, axis = 2L)

    # --- MLP ---
    chunked_evaluate_axis(function(z) {
      l4(torch::nnf_gelu(l3(ln4(z))))
    }, x, spmf, axis = 2L)
  },

  forward = function(support, query, save_peak_memory_factor = NULL) {
    # Built before either stream runs, which is what makes it safe for
    # `forward_rows()` to write its result back into the tensor it was
    # handed.
    kv <- self$row_kv(support)
    list(support = self$forward_rows(support, kv, save_peak_memory_factor),
         query   = self$forward_rows(query, kv, save_peak_memory_factor))
  }
)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

#' Mitra (top level)
#' @keywords internal
mitra_model <- torch::nn_module(
  "Tab2D",

  initialize = function(config) {
    dim <- as.integer(config$dim)
    self$dim <- dim
    self$dim_output <- as.integer(config$dim_output)
    self$task <- toupper(config$task %||% "CLASSIFICATION")
    self$is_classifier <- identical(self$task, "CLASSIFICATION")

    self$x_embedding <- mitra_x_embedding(dim)
    self$y_embedding <- if (self$is_classifier || self$dim_output > 1L) {
      mitra_y_embedding_classes(dim, self$dim_output)
    } else {
      mitra_y_embedding_regression(dim)
    }

    self$layers <- torch::nn_module_list(
      lapply(seq_len(as.integer(config$n_layers)), function(i)
        mitra_layer(dim, as.integer(config$n_heads)))
    )
    self$final_layer_norm <- affine_layer_norm(dim)
    self$final_layer <- torch::nn_linear(dim, self$dim_output, bias = TRUE)

    # Read by the predictors; see `mitra_kv_cache()`.
    self$supports_kv_cache <- TRUE
    self$supports_chunked_eval <- TRUE
    self$kv_cache_is_exact <- TRUE
  },

  # Everything the support set contributes to a prediction.
  #
  # @param x_support `(B, S, F)`; @param y_support `(B, S)`.
  build_kv_cache = function(x_support, y_support) {
    qf <- mitra_quantile_fit(x_support)
    support <- torch::torch_cat(
      list(self$y_embedding$support_embedding(y_support),
           self$x_embedding(qf$support)),
      dim = 3L
    )
    kv <- vector("list", length(self$layers))
    for (i in seq_along(self$layers)) {
      layer <- self$layers[[i]]
      kv[[i]] <- layer$row_kv(support)
      support <- layer$forward_rows(support, kv[[i]])
      collect_between_layers(support)
    }
    mitra_kv_cache(kv = kv, quantile_state = qf$state,
                   n_support = x_support$size(2),
                   n_features = x_support$size(3))
  },

  # @param x_support `(B, S, F)`; @param y_support `(B, S)`;
  # @param x_query `(B, Q, F)`.
  # @param kv_cache A [mitra_kv_cache()] to predict against. When given,
  #   `x_support` and `y_support` are ignored.
  # @return `(B, Q, dim_output)`.
  forward = function(x_support, y_support, x_query, kv_cache = NULL,
                     save_peak_memory_factor = NULL) {
    if (!is.null(kv_cache)) {
      return(self$forward_cached(x_query, kv_cache, save_peak_memory_factor))
    }

    qe <- mitra_quantile_embedding(x_support, x_query)
    dump_if_enabled("mitra_quantile", qe$support)

    xs <- self$x_embedding(qe$support)     # (B, S, F, D)
    xq <- self$x_embedding(qe$query)
    ye <- self$y_embedding(y_support, x_query$size(2))

    # The target rides along as feature column 1.
    support <- torch::torch_cat(list(ye$support, xs), dim = 3L)
    query   <- torch::torch_cat(list(ye$query,   xq), dim = 3L)
    dump_if_enabled("mitra_embedded", query)

    for (i in seq_along(self$layers)) {
      out <- self$layers[[i]](support, query, save_peak_memory_factor)
      support <- out$support; query <- out$query
      collect_between_layers(support)
    }
    dump_if_enabled("mitra_encoded", query)

    self$decode(query)
  },

  #' @keywords internal
  forward_cached = function(x_query, kv_cache,
                            save_peak_memory_factor = NULL) {
    if (x_query$size(3) != kv_cache$n_features) {
      cli::cli_abort(
        "This cache was built for {kv_cache$n_features} feature{?s}; \\
         got {x_query$size(3)}."
      )
    }
    xq <- self$x_embedding(
      mitra_quantile_apply(x_query, kv_cache$quantile_state)
    )
    ye <- self$y_embedding$mask_embedding(x_query$size(2), x_query$size(1),
                                          x_query$device)
    query <- torch::torch_cat(list(ye, xq), dim = 3L)
    dump_if_enabled("mitra_embedded", query)

    for (i in seq_along(self$layers)) {
      query <- self$layers[[i]]$forward_rows(query, kv_cache$kv[[i]],
                                             save_peak_memory_factor)
      collect_between_layers(query)
    }
    dump_if_enabled("mitra_encoded", query)

    self$decode(query)
  },

  #' @keywords internal
  decode = function(query) {
    query <- self$final_layer(self$final_layer_norm(query))
    out <- query[, , 1, ]                  # the target column only
    dump_if_enabled("mitra_logits", out)
    out
  }
)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

#' Everything the support set contributes to a Mitra prediction
#'
#' Mitra is the cleanest of the four architectures to cache, because the
#' support set is never told the query rows exist. Its row attention is
#' one-directional -- support attends to support, query attends to support
#' -- and the feature attention and both MLPs act on a single row at a
#' time. So the support half of the forward pass can be run once, on its
#' own, and every later query batch reads the same answer.
#'
#' Stored per layer: the key and value projections of the observation
#' attention over the support rows. Also stored is the quantile
#' embedding's fitted state, which is likewise support-only -- the
#' boundaries, mean and standard deviation query values are bucketed and
#' scaled against.
#'
#' **This one is not small.** Unlike TabPFN's cache, which keeps a single
#' attention head, Mitra has no multi-query path and every head has to be
#' kept: `2 * n_layers * F * S * dim` floats, where `F` counts the target
#' column. For the released 12-layer, 512-wide checkpoint that is about a
#' megabyte per support row per twenty features. `print()` on the cache
#' reports the actual figure. It buys back a full support-set pass per
#' prediction chunk and per ensemble member, so it is worth it when
#' predicting many rows and wasteful when predicting a handful.
#'
#' @param kv Per-layer `list(key, value)` over the support rows.
#' @param quantile_state State from [mitra_quantile_fit()].
#' @param n_support Number of support rows it was built from.
#' @param n_features Feature count it was built for.
#' @keywords internal
mitra_kv_cache <- function(kv, quantile_state, n_support, n_features) {
  structure(
    list(kv = kv, quantile_state = quantile_state,
         n_support = as.integer(n_support),
         n_features = as.integer(n_features)),
    class = "mitra_kv_cache"
  )
}

#' @export
print.mitra_kv_cache <- function(x, ...) {
  cli::cli_text("{.strong Mitra KV cache}")
  cli::cli_bullets(c(
    "*" = "built from {.val {x$n_support}} support row{?s}, \\
           {.val {x$n_features}} feature{?s}",
    "*" = "{length(x$kv)} layer{?s}, \\
           {round(.tensor_bytes(x$kv) / 1e6, 1)} MB of key/value projections"
  ))
  invisible(x)
}


# ---------------------------------------------------------------------------
# Backend hooks
# ---------------------------------------------------------------------------

#' @keywords internal
mitra_build <- function(config, task) {
  cli::cli_alert_info(
    "Building Mitra ({.val {tolower(config$task)}}, dim={config$dim}, \\
     {config$n_layers} layers)..."
  )
  mitra_model(config)
}

#' @keywords internal
mitra_detect <- function(config) {
  # The published config is five fields with no architecture marker; the
  # upper-case `task` plus `dim_output` is what no other backend has.
  !is.null(config$task) && !is.null(config$dim) &&
    !is.null(config$dim_output) && !is.null(config$n_layers) &&
    is.character(config$task) &&
    toupper(config$task) %in% c("CLASSIFICATION", "REGRESSION")
}

#' @keywords internal
mitra_task_of <- function(config) {
  switch(toupper(config$task %||% ""),
         CLASSIFICATION = "classification",
         REGRESSION     = "regression",
         NULL)
}


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

# @keywords internal
.mitra_batch <- function(X_train, y_train, X_test, device) {
  xs <- as.matrix(X_train); storage.mode(xs) <- "double"
  xq <- as.matrix(X_test);  storage.mode(xq) <- "double"
  list(
    x_support = as_float_tensor(xs, device = device)$unsqueeze(1L),
    y_support = as_float_tensor(matrix(as.numeric(y_train), nrow = 1L),
                                device = device),
    x_query   = as_float_tensor(xq, device = device)$unsqueeze(1L)
  )
}

# Run one ensemble member over a test chunk, cached or not, and return its
# `(1, n_test, dim_output)` output.
#
# The support side of each member is fixed once the preprocessor is fitted
# -- the same training rows through the same sign flips, chunk after chunk
# -- so with caching on it is transformed and encoded exactly once.
# @keywords internal
.mitra_member_out <- function(net, dev, state, pp, y_train, X_test_chunk,
                              cache_store, i,
                              save_peak_memory_factor = NULL) {
  xq <- mitra_preprocessor_transform_X(as.matrix(X_test_chunk), pp)
  if (is.null(cache_store)) {
    xs <- mitra_preprocessor_transform_X(state$X_train, pp)
    ys <- mitra_preprocessor_transform_y(y_train, pp)
    b <- .mitra_batch(xs, ys, xq, dev)
    return(torch::with_no_grad({
      net(b$x_support, b$y_support, b$x_query,
          save_peak_memory_factor = save_peak_memory_factor)
    }))
  }
  cache <- member_cache(cache_store, i, function() {
    xs <- mitra_preprocessor_transform_X(state$X_train, pp)
    ys <- mitra_preprocessor_transform_y(y_train, pp)
    x_support <- as_float_tensor(as.matrix(xs), device = dev)$unsqueeze(1L)
    y_support <- as_float_tensor(matrix(as.numeric(ys), nrow = 1L), device = dev)
    torch::with_no_grad({ net$build_kv_cache(x_support, y_support) })
  })
  x_query <- as_float_tensor(as.matrix(xq), device = dev)$unsqueeze(1L)
  torch::with_no_grad({
    net(NULL, NULL, x_query, kv_cache = cache,
        save_peak_memory_factor = save_peak_memory_factor)
  })
}

# Fit one preprocessor per ensemble member.
#
# Mitra's ensemble is unlike the other two. It has no designed set of
# views: every member is a fresh `Preprocessor` whose only source of
# variation is the random per-column sign flip (and, for regression, a
# possible mirroring of the target). AutoGluon also defaults to
# `n_estimators = 1`, so out of the box there is exactly one member --
# the ensembling exists, but the default does not use it.
# @keywords internal
.mitra_prepare <- function(X, y, task, n_estimators, random_mirror_x,
                           random_mirror_regression, random_state) {
  lapply(seq_len(n_estimators), function(i) {
    mitra_preprocessor_fit(
      X, y, task = task,
      random_mirror_x = random_mirror_x,
      random_mirror_regression = random_mirror_regression,
      # One seed per member, or the members would be identical copies.
      seed = if (is.null(random_state)) NULL else random_state + i - 1L
    )
  })
}

#' Build the Mitra classifier predictor
#'
#' Reproduces `MitraClassifier` minus fine-tuning: each member's
#' preprocessor imputes missing values with the training column means,
#' drops constant columns and applies its own random sign flips; each
#' member's logits are softmaxed and the probabilities averaged.
#'
#' The imputation is the substantive part. Mitra does not propagate
#' `NaN`, it absorbs it — one missing value makes a column's quantiles
#' all-`NaN`, every value then buckets to zero, and the zero-variance
#' guard flattens the column. The output is finite and the feature has
#' silently vanished. The reference never sees that because it imputes
#' first, and now neither does this backend.
#'
#' @param ctx Loaded-model context from [load_backend_model()].
#' @param n_estimators Number of ensemble members. AutoGluon's default is
#'   1; more members only help if `random_mirror_x` is on, since that is
#'   the only thing distinguishing them.
#' @param random_mirror_x Flip a random sign per column, per member.
#' @param random_state Seed for the sign flips, or `NULL`. The reference
#'   draws these from NumPy's *global* generator and never seeds it, so
#'   its own flips differ run to run; a seed is offered here because an R
#'   user can reasonably expect reproducibility.
#' @param predict_chunk_size Max test rows per forward pass.
#' @param kv_cache Condition on the support set once per `predict()` call
#'   and reuse that across every chunk and ensemble member, instead of
#'   re-encoding the whole support set for each. Off by default, and the
#'   memory it costs is the reason -- see [mitra_kv_cache()]. It answers
#'   the same question an uncached pass does: the support half of this
#'   architecture never looks at a query row.
#' @param save_peak_memory_factor Integer, or `NULL` (default) to
#'   disable. Splits each of a layer's four sublayers into this many
#'   chunks of rows. Every one of them is already independent across
#'   rows, so the output is bit-identical and the only cost is loop
#'   overhead. This is the knob Mitra most needs: it is the one backend
#'   that attends across rows *and* columns, so its activation carries
#'   every row at full embedding width on both axes and is the steepest
#'   in the package.
#' @keywords internal
mitra_classifier <- function(ctx, n_estimators = 1L, random_mirror_x = TRUE,
                             random_state = 42L, predict_chunk_size = 1024L,
                             kv_cache = FALSE,
                             save_peak_memory_factor = NULL) {
  net <- ctx$net; dev <- ctx$device

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    X_train <- as.matrix(X); storage.mode(X_train) <- "double"
    if (is.factor(y)) {
      levels_ <- levels(y); y_int <- as.integer(y) - 1L
    } else {
      levels_ <- sort(unique(y)); y_int <- match(y, levels_) - 1L
    }
    if (length(levels_) > net$dim_output) {
      cli::cli_abort(
        "This Mitra checkpoint supports at most {net$dim_output} classes; \\
         got {length(levels_)}."
      )
    }
    preps <- .mitra_prepare(X_train, y_int, "classification",
                            as.integer(n_estimators), random_mirror_x,
                            FALSE, random_state)
    list(X_train = X_train, y_train_int = y_int, class_levels = levels_,
         preps = preps, n_train = nrow(X_train))
  }

  .chunk <- function(state, X_test_chunk, caches) {
    n_cls <- length(state$class_levels)
    acc <- NULL
    for (i in seq_along(state$preps)) {
      # Between members: a whole forward pass' worth of torch
      # allocations, which R's collector cannot see.
      if (i > 1L) collect_between_chunks()
      out <- .mitra_member_out(net, dev, state, state$preps[[i]],
                               state$y_train_int, X_test_chunk, caches, i,
                               save_peak_memory_factor)
      # The head always emits `dim_output` logits; the reference slices to
      # the classes actually present before the softmax.
      probs <- as.matrix(torch::nnf_softmax(out[1, , 1:n_cls],
                                            dim = -1L)$cpu())
      acc <- if (is.null(acc)) probs else acc + probs
    }
    acc / length(state$preps)
  }

  predict_fn <- function(state, newdata, type = "class", ...) {
    # A cache carried on the fitted state was built once, possibly in
    # another session; otherwise build lazily as before.
    caches <- member_cache_store_from(state$kv_caches) %||%
      (if (isTRUE(kv_cache)) member_cache_store() else NULL)
    probs <- chunk_apply(as.matrix(newdata), predict_chunk_size,
                         function(chunk) .chunk(state, chunk, caches))
    colnames(probs) <- as.character(state$class_levels)
    if (type == "prob") return(probs)
    state$class_levels[max.col(probs, ties.method = "first")]
  }

  # Building every member's cache is exactly what one forward pass over a
  # single query row does, so ask for that rather than duplicating the
  # builders: the store comes back full.
  .build_all_caches <- function(state) {
    store <- member_cache_store()
    one <- state$X_train[1L, , drop = FALSE]
    .chunk(state, one, store)
    member_cache_list(store)
  }

  list(fit = fit_fn, predict = predict_fn,
       build_cache = .build_all_caches)
}

#' Build the Mitra regressor predictor
#'
#' Mitra's regressor emits a single value per row, so there is no
#' predictive distribution to draw quantiles from. Its target is mapped
#' to `[0, 1]` by min-max — `normalize_y`, not the standardization TabFM
#' and TabICL use — and optionally mirrored to `1 - y` per member.
#'
#' @inheritParams mitra_classifier
#' @param random_mirror_regression Possibly mirror the scaled target.
#' @keywords internal
mitra_regressor <- function(ctx, n_estimators = 1L, random_mirror_x = TRUE,
                            random_mirror_regression = TRUE,
                            random_state = 42L, predict_chunk_size = 1024L,
                            kv_cache = FALSE,
                            save_peak_memory_factor = NULL) {
  net <- ctx$net; dev <- ctx$device

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    X_train <- as.matrix(X); storage.mode(X_train) <- "double"
    preps <- .mitra_prepare(X_train, y, "regression", as.integer(n_estimators),
                            random_mirror_x, random_mirror_regression,
                            random_state)
    list(X_train = X_train, y_train = as.numeric(y), preps = preps,
         n_train = nrow(X_train))
  }

  .chunk <- function(state, X_test_chunk, caches) {
    acc <- NULL
    for (i in seq_along(state$preps)) {
      # Between members: a whole forward pass' worth of torch
      # allocations, which R's collector cannot see.
      if (i > 1L) collect_between_chunks()
      pp <- state$preps[[i]]
      out <- .mitra_member_out(net, dev, state, pp, state$y_train,
                               X_test_chunk, caches, i,
                               save_peak_memory_factor)
      preds <- mitra_preprocessor_invert_y(as.numeric(out[1, , 1]$cpu()), pp)
      acc <- if (is.null(acc)) preds else acc + preds
    }
    matrix(acc / length(state$preps), ncol = 1L)
  }

  predict_fn <- function(state, newdata, type = "mean", ...) {
    # A cache carried on the fitted state was built once, possibly in
    # another session; otherwise build lazily as before.
    caches <- member_cache_store_from(state$kv_caches) %||%
      (if (isTRUE(kv_cache)) member_cache_store() else NULL)
    as.numeric(chunk_apply(as.matrix(newdata), predict_chunk_size,
                           function(chunk) .chunk(state, chunk, caches)))
  }

  # Building every member's cache is exactly what one forward pass over a
  # single query row does, so ask for that rather than duplicating the
  # builders: the store comes back full.
  .build_all_caches <- function(state) {
    store <- member_cache_store()
    one <- state$X_train[1L, , drop = FALSE]
    .chunk(state, one, store)
    member_cache_list(store)
  }

  list(fit = fit_fn, predict = predict_fn, types = "mean",
       build_cache = .build_all_caches)
}


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

#' Stage list for the Mitra diagram
#'
#' The flattest of the six: no separate column, row and in-context
#' stages, just one stack of identical layers that attends twice --
#' across rows, then across features -- with an MLP after each. The
#' target is not a side input but feature column 1, and the whole
#' preprocessing step is a fixed quantile-rank transform with no weights
#' at all.
#'
#' @param config The checkpoint's parsed `config.json`.
#' @param task `"classification"` or `"regression"`.
#' @keywords internal
mitra_describe <- function(config, task) {
  D     <- as.integer(config$dim)
  H     <- as.integer(config$n_heads)
  L     <- as.integer(config$n_layers)
  n_out <- as.integer(config$dim_output)
  clf   <- identical(toupper(config$task %||% "CLASSIFICATION"), "CLASSIFICATION")

  stages <- list(
    arch_input_stage(detail = "numeric matrix; the preprocessor mean-imputes first"),
    arch_stage(
      "quant", "Quantile-rank embedding", kind = "embed", group = "Embed",
      detail = "999 quantiles of the support column; query values bucketed against them; no weights",
      shape = "(B, S + Q, F)"),
    arch_stage(
      "x_embed", "Cell embedding", kind = "embed", group = "Embed",
      detail = sprintf("Linear(1 -> %d) per cell", D),
      shape = "(B, S + Q, F, D)", prefix = "x_embedding"),
    arch_stage(
      "y_embed", "Target embedding", kind = "embed", group = "Embed",
      detail = if (clf)
        sprintf("a learned vector per class, plus one \"unknown\" for the query rows; prepended as feature column 1")
      else "Linear(1 -> D) on the support rows, a learned vector on the query rows",
      shape = "(B, S + Q, F + 1, D)", prefix = "y_embedding"),
    arch_stage(
      "layers", "Layer", kind = "attention", group = "Transformer",
      repeats = L,
      detail = sprintf("pre-norm; %d heads of %d; two attentions, two MLPs of %d -> %d",
                       H, as.integer(D / H), D, 4L * D),
      shape = "(B, S + Q, F + 1, D)", prefix = "layers",
      children = list(
        arch_stage("ln1", "LayerNorm", kind = "norm", prefix = "layers.0.layer_norm1"),
        arch_stage("a_row", "Attention across rows - query rows read the support",
                   kind = "attention", axis = "rows",
                   prefix = "layers.0.attention1"),
        arch_stage("ln2", "LayerNorm", kind = "norm", prefix = "layers.0.layer_norm2"),
        arch_stage("mlp1", "MLP, GELU", kind = "ffn",
                   prefix = c("layers.0.linear1", "layers.0.linear2")),
        arch_stage("ln3", "LayerNorm", kind = "norm", prefix = "layers.0.layer_norm3"),
        arch_stage("a_feat", "Attention across features - self-attention",
                   kind = "attention", axis = "features",
                   prefix = "layers.0.attention2"),
        arch_stage("ln4", "LayerNorm", kind = "norm", prefix = "layers.0.layer_norm4"),
        arch_stage("mlp2", "MLP, GELU", kind = "ffn",
                   prefix = c("layers.0.linear3", "layers.0.linear4"))
      )),
    arch_stage(
      "final_ln", "Final LayerNorm", kind = "norm", group = "Decode",
      prefix = "final_layer_norm"),
    arch_stage(
      "final", "Output projection", kind = "decode", group = "Decode",
      detail = sprintf("target column only: Linear(%d -> %d)", D, n_out),
      shape = sprintf("(Q, %d)", n_out), prefix = "final_layer"),
    if (clf)
      arch_output_stage("Class logits",
                        detail = sprintf("%d slots; softmax over the classes seen", n_out))
    else
      arch_output_stage("Scalar prediction",
                        detail = "one number per query row, on the scaled target")
  )

  list(
    title = sprintf("Mitra - %s", if (clf) "classifier" else "regressor"),
    subtitle = "AutoGluon  -  one stack, attending across rows and across features",
    facts = c(
      "Layers"              = L,
      "Width"               = D,
      "Attention heads"     = sprintf("%d x %d", H, as.integer(D / H)),
      "MLP width"           = 4L * D,
      "Attention per layer" = "2 (rows, features)",
      "Output width"        = n_out,
      "Preprocessing"       = "quantile ranks, no weights",
      "Normalisation"       = "LayerNorm, pre",
      "KV cache"            = "exact"
    ),
    symbols = c(
      B = "ensemble members (batch)", S = "support rows", Q = "query rows",
      F = "features", D = sprintf("model width (%d)", D)
    ),
    stages = Filter(Negate(is.null), stages)
  )
}


# ---------------------------------------------------------------------------
# Memory scaling
# ---------------------------------------------------------------------------

#' Mitra's peak-memory shapes
#'
#' The steepest curve in the package, and the reason the preflight
#' exists. Every other backend narrows the table down to something small
#' before attending across rows: TabICL's in-context stage carries one
#' `embed_dim * n_cls` vector per row, TabPFN v2 one vector per feature
#' *group*. Mitra never narrows. Its state stays `rows x features x dim`
#' from the first layer to the last, so a 90-column table costs 91 times
#' what a single column would -- which is what turns 6,426 rows on CPU
#' from slow into impossible.
#'
#' Features count one higher than the table's: the target is not a side
#' input here but column 1 of the same tensor.
#'
#' @inheritParams .icl_family_terms
#' @param opts Resolved predictor arguments.
#' @param config The model config.
#' @keywords internal
mitra_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  d <- as.numeric(config$dim)
  h <- as.numeric(config$n_heads %||% 4)
  l <- as.numeric(config$n_layers)
  f <- n_features + 1
  nq <- .resident_query(n_query, opts)
  n  <- n_context + nq
  n_est <- max(1, as.numeric(opts$n_estimators %||% 1))

  list(
    stages = list(
      # Support and query rows run through the same four sublayers; the
      # keys are always the support set, so the largest score block is
      # the support attending to itself.
      # Both axes attend unmasked, so neither materialises its scores;
      # what makes Mitra the steepest in the package is the activation
      # itself, which carries every row *and* every column at full
      # embedding width.
      list(name = "attention across rows",
           act = n * f * d,
           att = 0),
      list(name = "attention across features",
           act = n * f * d,
           att = 0)
    ),
    # `mitra_kv_cache()` holds one key and one value per layer, each the
    # full `features x support x dim` -- which is why the cache is off by
    # default and why its cost belongs in the estimate rather than in a
    # footnote.
    persistent = if (isTRUE(opts$kv_cache)) n_est * l * 2 * f * n_context * d
                 else 0,
    n_estimators = n_est
  )
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#' @keywords internal
register_mitra_backend <- function() {
  register_backend(
    name          = "mitra",
    build         = mitra_build,
    describe      = mitra_describe,
    translate_key = identity,
    detect        = mitra_detect,
    task_of       = mitra_task_of,
    classifier    = mitra_classifier,
    regressor     = mitra_regressor,
    peak_terms    = mitra_peak_terms,
    aliases       = c("mitra-classifier" = "autogluon/mitra-classifier",
                      "mitra-regressor"  = "autogluon/mitra-regressor"),
    # A single NaN in a column makes its quantiles all-NaN, which buckets
    # every value to 0 and collapses the column to zeros -- silently. See
    # `mitra_quantile_embedding()`. The predictors never let that happen:
    # each member's preprocessor mean-imputes first, as AutoGluon's does.
    # That is the sharpest edge in the package and the clearest case for
    # the two flags being separate.
    kv_cache_capable   = TRUE,
    handles_missing    = FALSE,
    imputes_internally = TRUE,
    description   = "Mitra 2-D attention transformer (AutoGluon)",
    parity        = "autogluon mitra"
  )
}
