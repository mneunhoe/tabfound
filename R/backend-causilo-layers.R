# Causilo 1.5 building blocks.
#
# A port of `causilo/v1_5/nn/` from Nums AI's `causilo` 1.5.0
# (Apache-2.0). Module and parameter names follow the checkpoint keys, so
# loading needs no key translation. Only the FP32 path is ported: the
# CUDA autocast, FP16/BF16 and Triton branches of the reference compute
# the same function in other dtypes.
#
# Conventions that differ from the other backends' blocks:
#
#   * one packed `projection: Linear(W, 3W)` per attention, rows
#     `[0, W)` = Q, `[W, 2W)` = K, `[2W, 3W)` = V. Queries and keys always
#     come from different tensors here, so -- as `F._in_projection_packed`
#     does for `q is not k` -- Q is one `linear()` over the first third
#     and K/V one over the other two;
#   * Q and K heads are scaled to unit RMS over the head width (no
#     weight, `mean(x^2)` clamped at float32 eps); V is left alone;
#   * an optional learned query gain (`scaling`) multiplies the
#     normalised queries by a bounded function of `log(n_keys)` and of the
#     query itself, then plain scaled-dot-product attention;
#   * heads sit on axis -3: `[..., H, S, hd]`.
#
#   AttentionLayer:  x <- x + attn(norm_a(x), norm_a(context))
#                    x <- x + ff(norm_f(x)),   ff = SwiGLU, expansion 2


# torch.finfo(torch.float32).eps, the clamp in `unit_rms()`.
.causilo_f32_eps <- 1.1920928955078125e-07

# Gain limits of the two query scales (`nn/layers/scaling.py`).
.causilo_length_gain_limit  <- 8
.causilo_content_gain_limit <- 2


#' Scale heads to unit root-mean-square
#'
#' `x / sqrt(max(mean(x^2), eps))` over the last axis, in float32.
#'
#' @param x A tensor whose last axis is the head width.
#' @keywords internal
causilo_unit_rms <- function(x) {
  xf <- x$to(dtype = torch::torch_float32())
  rms <- xf$square()$mean(dim = -1L, keepdim = TRUE)$clamp(min = .causilo_f32_eps)$sqrt()
  (xf / rms)$to(dtype = x$dtype)
}

#' Map values into `(1 / limit, limit)` in log space
#'
#' `exp(log(limit) * tanh(x / log(limit)))`.
#'
#' @param x A tensor.
#' @param limit The bound.
#' @keywords internal
causilo_bounded_multiplier <- function(x, limit) {
  radius <- log(limit)
  (torch::torch_tanh(x$to(dtype = torch::torch_float32()) / radius) * radius)$exp()$to(dtype = x$dtype)
}


#' Learned query gain from the key count and the query's content
#'
#' Two small GELU MLPs: `length` maps `log(max(n_keys, 1))` to one gain per
#' head and channel, `content` maps each normalised query head to one gain
#' per channel. Each is bounded by [causilo_bounded_multiplier()] (8 and 2)
#' and the query is multiplied by their product.
#'
#' @param heads Number of heads.
#' @param head_width Channels per head.
#' @keywords internal
causilo_query_scale <- torch::nn_module(
  "CausiloQueryScale",

  initialize = function(heads, head_width) {
    self$heads <- as.integer(heads)
    self$head_width <- as.integer(head_width)
    self$length <- torch::nn_sequential(
      torch::nn_linear(1L, 64L), torch::nn_gelu(),
      torch::nn_linear(64L, self$heads * self$head_width)
    )
    self$content <- torch::nn_sequential(
      torch::nn_linear(self$head_width, 64L), torch::nn_gelu(),
      torch::nn_linear(64L, self$head_width)
    )
  },

  # `[1, H, 1, hd]`: one gain per head and channel for `n_keys` keys.
  length_gain = function(query, n_keys) {
    logn <- torch::torch_tensor(log(max(as.numeric(n_keys), 1)),
                                dtype = query$dtype, device = query$device)
    g <- self$length(logn$reshape(c(-1L, 1L)))
    causilo_bounded_multiplier(g, .causilo_length_gain_limit)$view(
      c(-1L, self$heads, 1L, self$head_width))
  },

  # @param query `[N, H, S, hd]`, already unit-RMS.
  forward = function(query, n_keys) {
    length_gain <- self$length_gain(query, n_keys)
    content <- self$content(query)
    query * (length_gain * causilo_bounded_multiplier(content, .causilo_content_gain_limit))
  }
)


#' SwiGLU feedforward
#'
#' `output(silu(gate(x)) * value(x))`.
#'
#' @param width Input and output width.
#' @param expansion Hidden width multiple.
#' @keywords internal
causilo_feedforward <- torch::nn_module(
  "CausiloFeedForward",

  initialize = function(width, expansion = CAUSILO_EXPANSION) {
    self$value <- torch::nn_linear(width, width * expansion)
    self$gate <- torch::nn_linear(width, width * expansion)
    self$output <- torch::nn_linear(width * expansion, width)
  },

  forward = function(x) {
    value <- self$value(x)
    gate <- self$gate(x)
    self$output(torch::nnf_silu(gate) * value)
  }
)


#' Packed attention with unit-RMS heads
#'
#' Inputs are `[..., S, W]`; projected queries and keys/values are
#' `[..., H, S, hd]` with the same leading axes.
#'
#' @param width Model width `W`.
#' @param heads Number of heads.
#' @param scale_query Add a [causilo_query_scale()] as `scaling`.
#' @keywords internal
causilo_attention <- torch::nn_module(
  "CausiloAttention",

  initialize = function(width, heads, scale_query) {
    self$width <- as.integer(width)
    self$heads <- as.integer(heads)
    self$head_width <- self$width %/% self$heads
    self$projection <- torch::nn_linear(self$width, 3L * self$width)
    self$output <- torch::nn_linear(self$width, self$width)
    if (isTRUE(scale_query)) {
      self$scaling <- causilo_query_scale(self$heads, self$head_width)
    }
    self$scale_query <- isTRUE(scale_query)
  },

  # `[..., S, W]` -> `[..., H, S, hd]`.
  split_heads = function(x) {
    shp <- x$shape
    nd <- length(shp)
    x$view(c(shp[-nd], self$heads, self$head_width))$transpose(-3L, -2L)
  },

  # Q from the first third of `projection`.
  prepare_query = function(rows) {
    W <- self$width
    q <- torch::nnf_linear(rows, self$projection$weight[1:W, ],
                           self$projection$bias[1:W])
    causilo_unit_rms(self$split_heads(q))
  },

  # K/V from the other two thirds, in one call: `list(key, value)`.
  prepare_context = function(rows) {
    W <- self$width
    kv <- torch::nnf_linear(rows, self$projection$weight[(W + 1L):(3L * W), ],
                            self$projection$bias[(W + 1L):(3L * W)])
    list(key = causilo_unit_rms(self$split_heads(kv[.., 1:W])),
         value = self$split_heads(kv[.., (W + 1L):(2L * W)]))
  },

  # @param query `[..., H, Sq, hd]`, from `prepare_query()`.
  # @param kv `list(key, value)`, `[..., H, Sk, hd]`.
  # @return `[..., Sq, W]`.
  attend = function(query, kv) {
    shp <- query$shape
    nd <- length(shp)
    lead <- shp[seq_len(nd - 3L)]
    n <- if (length(lead)) prod(lead) else 1L
    q <- query$reshape(c(n, shp[(nd - 2L):nd]))
    k <- kv$key$reshape(c(n, kv$key$shape[(nd - 2L):nd]))
    v <- kv$value$reshape(c(n, kv$value$shape[(nd - 2L):nd]))
    if (self$scale_query) q <- self$scaling(q, k$size(3L))
    res <- if (shp[[nd - 1L]] == 0L) q else sdpa(q, k, v)
    res <- res$reshape(shp)$transpose(-3L, -2L)$contiguous()
    self$output(res$view(c(lead, shp[[nd - 1L]], self$width)))
  },

  forward = function(query, context) {
    self$attend(self$prepare_query(query), self$prepare_context(context))
  }
)


#' Pre-normalised attention and feedforward residual layer
#'
#' @param width Model width.
#' @param heads Number of heads.
#' @param scale_query Learned query gain in the attention.
#' @keywords internal
causilo_attention_layer <- torch::nn_module(
  "CausiloAttentionLayer",

  initialize = function(width, heads, scale_query) {
    self$attention <- causilo_attention(width, heads, scale_query)
    self$feedforward <- causilo_feedforward(width)
    self$attention_norm <- kumo_rms_norm(width, eps = 1e-5)
    self$feedforward_norm <- kumo_rms_norm(width, eps = 1e-5)
  },

  finish = function(rows, update) {
    combined <- rows + update
    combined + self$feedforward(self$feedforward_norm(combined))
  },

  # K/V of a context, for replay (`prepare()` in the reference).
  prepare = function(context) {
    self$attention$prepare_context(self$attention_norm(context))
  },

  # Advance `rows` against prepared K/V (`query()` in the reference).
  query = function(rows, kv) {
    q <- self$attention$prepare_query(self$attention_norm(rows))
    self$finish(rows, self$attention$attend(q, kv))
  },

  # @param rows `[..., S, W]`.
  # @param context `[..., Sk, W]`, normalised here; or `NULL` with
  #   `context_rows`, which takes the first `context_rows` of the
  #   normalised `rows` as the context.
  # @param return_kv Also return the K/V the rows attended to.
  forward = function(rows, context = NULL, context_rows = NULL,
                     return_kv = FALSE) {
    normalized <- self$attention_norm(rows)
    ctx <- if (!is.null(context_rows)) {
      normalized$narrow(-2L, 1L, context_rows)
    } else {
      self$attention_norm(context)
    }
    q <- self$attention$prepare_query(normalized)
    kv <- self$attention$prepare_context(ctx)
    out <- self$finish(rows, self$attention$attend(q, kv))
    if (isTRUE(return_kv)) list(rows = out, kv = kv) else out
  }
)


#' Column layer: latents gather each column's context, rows read them back
#'
#' Operates on `[..., C, R, W]` with rows as the sequence; columns are a
#' batch axis and never mix.
#'
#' @param width Model width.
#' @param heads Attention heads.
#' @param latents Number of learned latents.
#' @keywords internal
causilo_column_layer <- torch::nn_module(
  "CausiloColumnLayer",

  initialize = function(width, heads = CAUSILO_COLUMN_HEADS,
                        latents = CAUSILO_COLUMN_LATENTS) {
    self$gather <- causilo_attention_layer(width, heads, scale_query = TRUE)
    self$broadcast <- causilo_attention_layer(width, heads, scale_query = FALSE)
    self$latents <- torch::nn_parameter(torch::torch_zeros(latents, width))
  },

  # The learned latents after attending to the context rows `[..., n, W]`.
  gather_context = function(context) {
    shp <- context$shape
    nd <- length(shp)
    lat <- self$latents$expand(c(shp[seq_len(nd - 2L)], self$latents$shape))
    self$gather(lat, context)
  },

  # Broadcast K/V of the gathered latents, for the KV cache.
  prepare = function(context) {
    self$broadcast$prepare(self$gather_context(context))
  },

  # Advance query rows against prepared broadcast K/V.
  query = function(rows, kv) self$broadcast$query(rows, kv),

  # @param rows `[..., C, R, W]`; the first `context_rows` are context.
  # @param return_kv Also return the broadcast K/V (the cache entry).
  forward = function(rows, context_rows, return_kv = FALSE) {
    latents <- self$gather_context(rows$narrow(-2L, 1L, context_rows))
    self$broadcast(rows, latents, return_kv = return_kv)
  }
)


#' Row layer: latent tokens gather a row's cells and broadcast back
#'
#' Operates on `[..., R, C, W]` with the cells of one row as the sequence,
#' plus `ROW_LATENTS` latent tokens per row carried from layer to layer.
#' The final layer updates only the latents, gathering first with the
#' broadcast weights, then with the gather weights.
#'
#' @param width Model width.
#' @param heads Attention heads.
#' @param final Whether this is the last row layer.
#' @keywords internal
causilo_row_layer <- torch::nn_module(
  "CausiloRowLayer",

  initialize = function(width, heads = CAUSILO_ROW_HEADS, final = FALSE) {
    self$final <- isTRUE(final)
    self$gather <- causilo_attention(width, heads, scale_query = TRUE)
    self$broadcast <- causilo_attention(width, heads, scale_query = TRUE)
    self$feedforward <- causilo_feedforward(width)
    self$gather_norm <- kumo_rms_norm(width, eps = 1e-5)
    self$broadcast_norm <- kumo_rms_norm(width, eps = 1e-5)
    self$feedforward_norm <- kumo_rms_norm(width, eps = 1e-5)
  },

  .gather = function(latents, features, attention, norm) {
    q <- norm(latents)
    ctx <- torch::torch_cat(list(q, norm(features)), dim = -2L)
    latents + attention(q, ctx)
  },

  # @param latents `[..., R, K, W]`; @param features `[..., R, C, W]`.
  # @return `list(latents, features)`; the final layer returns the
  #   features unchanged.
  forward = function(latents, features) {
    K <- latents$size(-2L)
    if (!self$final) {
      bl <- self$.gather(latents, features, self$gather, self$gather_norm)
      combined <- torch::torch_cat(list(bl, features), dim = -2L)
      q <- self$broadcast_norm(combined)
      ctx <- q$narrow(-2L, 1L, K)$contiguous()
      combined <- combined + self$broadcast(q, ctx)
      lat <- combined$narrow(-2L, 1L, K)
      lat <- lat + self$feedforward(self$feedforward_norm(lat))
      return(list(latents = lat,
                  features = combined$narrow(-2L, K + 1L, combined$size(-2L) - K)))
    }
    lat <- self$.gather(latents, features, self$broadcast, self$broadcast_norm)
    lat <- self$.gather(lat, features, self$gather, self$gather_norm)
    list(latents = lat + self$feedforward(self$feedforward_norm(lat)),
         features = features)
  }
)


#' Context-fitted mid-ranks of every cell
#'
#' Per column, the finite context values sorted (non-finite ones pushed to
#' `+Inf`), then for each cell `(#less + #less-or-equal) / 2 / #finite`.
#' Missing or non-finite cells, and every cell of a column with no finite
#' context value, get `NaN`.
#'
#' @param features `[..., R, 1]` (one scalar per cell, columns leading).
#' @param context Sorted context `[..., n, 1]` from
#'   [causilo_ecdf_context()].
#' @return `[..., R, 1]` float32.
#' @keywords internal
causilo_ecdf_ranks <- function(features, context) {
  boundaries <- context$to(dtype = torch::torch_float32())$transpose(-2L, -1L)$contiguous()
  queries <- features$to(dtype = torch::torch_float32())$transpose(-2L, -1L)$contiguous()
  finite <- torch::torch_isfinite(queries)
  queries <- queries$masked_fill(finite$logical_not(), 0)
  left <- torch::torch_searchsorted(boundaries, queries, right = FALSE)
  right <- torch::torch_searchsorted(boundaries, queries, right = TRUE)
  shp <- boundaries$shape
  padding <- torch::torch_full(c(shp[-length(shp)], 1L), Inf,
                               dtype = boundaries$dtype, device = boundaries$device)
  counts <- torch::torch_searchsorted(boundaries, padding)
  observed <- counts$ne(0L)
  ranks <- (left$to(dtype = torch::torch_float32()) +
              right$to(dtype = torch::torch_float32())) * 0.5 / counts$clamp(min = 1L)
  ranks <- ranks$masked_fill(finite$logical_not() | observed$logical_not(), NaN)
  ranks$transpose(-2L, -1L)$contiguous()
}

#' Sort each column's context values for [causilo_ecdf_ranks()]
#'
#' @param features `[..., R, 1]`.
#' @param context_rows Number of leading context rows.
#' @return `[..., n, 1]`, non-finite values replaced by `+Inf`.
#' @keywords internal
causilo_ecdf_context <- function(features, context_rows) {
  ctx <- features$narrow(-2L, 1L, context_rows)$to(dtype = torch::torch_float32())
  ctx <- ctx$masked_fill(torch::torch_isfinite(ctx)$logical_not(), Inf)
  ctx <- ctx$transpose(-2L, -1L)$contiguous()
  ctx$sort(dim = -1L)[[1]]$transpose(-2L, -1L)$contiguous()
}


#' Target embedding: one-hot classes or the scaled scalar target
#'
#' @param width Output width.
#' @param classes 10 for classification, 0 for regression.
#' @keywords internal
causilo_target_embedding <- torch::nn_module(
  "CausiloTargetEmbedding",

  initialize = function(width, classes) {
    self$classes <- as.integer(classes)
    self$projection <- torch::nn_linear(max(self$classes, 1L), width)
  },

  # @param targets 0-based class codes (any numeric dtype) or the scaled
  #   regression target, `[...]`.
  forward = function(targets) {
    values <- if (self$classes > 0L) {
      # R's one_hot is 1-based.
      torch::nnf_one_hot(targets$to(dtype = torch::torch_long()) + 1L, self$classes)$to(
        dtype = torch::torch_float32())
    } else {
      targets$unsqueeze(-1L)
    }
    self$projection(values)
  }
)


#' Cell embedding: value waves, the raw value and context mid-rank waves
#'
#' Each cell becomes 41 numbers -- `sin`/`cos` of `2 pi v f` over 16
#' learned frequencies, the value `v` itself, and `sin`/`cos` of
#' `2 pi rank g` over 4 learned frequencies -- projected to the model
#' width, plus a learned `missing` vector on missing cells. Missing and
#' infinite values enter as 0; the value waves are zeroed where the cell
#' is `NaN`, the rank waves where the rank is not finite.
#'
#' @param config The model dimensions from `.causilo_model_config()`.
#' @keywords internal
causilo_feature_embedding <- torch::nn_module(
  "CausiloFeatureEmbedding",

  initialize = function(config) {
    W <- config$width
    self$width <- W
    self$frequencies <- torch::nn_parameter(torch::torch_zeros(1L, CAUSILO_FOURIER_FREQS))
    self$rank_frequencies <- torch::nn_parameter(torch::torch_zeros(1L, CAUSILO_ECDF_FREQS))
    self$projection <- torch::nn_linear(
      2L * (CAUSILO_FOURIER_FREQS + CAUSILO_ECDF_FREQS) + 1L, W)
    self$missing <- torch::nn_parameter(torch::torch_zeros(1L, W))
    self$target <- causilo_target_embedding(W, config$max_classes)
  },

  .waves = function(values, frequencies) {
    phases <- 2 * pi * values$unsqueeze(-1L) * frequencies
    torch::torch_cat(list(torch::torch_sin(phases), torch::torch_cos(phases)), dim = -1L)
  },

  # @param features `[..., R, 1]`; @param ranks `[..., R, 1]`.
  project_features = function(features, ranks) {
    missing <- torch::torch_isnan(features)
    values <- features$nan_to_num(nan = 0, posinf = 0, neginf = 0)
    waves <- self$.waves(values, self$frequencies)$masked_fill(missing$unsqueeze(-1L), 0)
    rank_waves <- self$.waves(ranks$nan_to_num(), self$rank_frequencies)
    rank_waves <- rank_waves$masked_fill(torch::torch_isfinite(ranks)$logical_not()$unsqueeze(-1L), 0)
    encoded <- torch::torch_cat(list(waves, values$unsqueeze(-1L), rank_waves), dim = -1L)
    emb <- self$projection(encoded)$sum(dim = -2L)
    emb + torch::torch_matmul(missing$to(dtype = emb$dtype), self$missing)
  },

  # @param features `[B, C, R, 1]`.
  # @param context Sorted context values `[B, C, n, 1]`.
  # @param targets `[B, C, n]` or `NULL` (query rows only).
  forward = function(features, context, targets = NULL) {
    emb <- self$project_features(features, causilo_ecdf_ranks(features, context))
    if (!is.null(targets)) {
      n <- targets$size(-1L)
      if (n > 0L) emb$narrow(-2L, 1L, n)$add_(self$target(targets))
    }
    emb
  }
)
