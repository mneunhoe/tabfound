# TabICL transformer blocks.
#
# TabICL's blocks derive from `nn.TransformerEncoderLayer`: pre-norm with
# affine LayerNorm, a plain GELU feed-forward, and the residual added
# outside. That is a different convention from TabFM's RMSNorm sandwich
# with SwiGLU, so these live here rather than sharing `R/nn-blocks.R`.
# What they do share is the primitives in `R/nn-*.R`.
#
#   x <- q + attn(norm1(q), norm1(k), norm1(v))
#   x <- x + linear2(gelu(linear1(norm2(x))))

#' Pre-norm attention block with an optional scalable softmax
#'
#' @param embedding_dim Model width.
#' @param n_heads Number of attention heads.
#' @param dim_ff Feed-forward width.
#' @param ssmax Passed to [mha_fused_inproj()].
#' @param bias_free_ln When `TRUE` the LayerNorms have no bias. The
#'   released classifier sets this `FALSE` and the regressor `TRUE`, so
#'   it is not a detail that can be hard-coded.
#' @param activation Feed-forward activation.
#' @keywords internal
tabicl_mab <- torch::nn_module(
  "TabiclAttentionBlock",

  initialize = function(embedding_dim, n_heads, dim_ff, ssmax = FALSE,
                        bias_free_ln = FALSE, activation = "gelu") {
    self$attn <- mha_fused_inproj(embedding_dim, n_heads, ssmax = ssmax)
    self$norm1 <- affine_layer_norm(embedding_dim, eps = 1e-5, bias = !bias_free_ln)
    self$norm2 <- affine_layer_norm(embedding_dim, eps = 1e-5, bias = !bias_free_ln)
    self$linear1 <- torch::nn_linear(embedding_dim, dim_ff, bias = TRUE)
    self$linear2 <- torch::nn_linear(dim_ff, embedding_dim, bias = TRUE)
    self$activation <- activation
  },

  feed_forward = function(x) {
    self$linear2(apply_activation(self$linear1(x), self$activation))
  },

  # @param q Query input `(..., Tq, E)`.
  # @param k,v Key/value inputs. Both `NULL` means self-attention, in
  #   which case the *same* normalized tensor is reused for all three —
  #   which is also what selects the fused in-projection path.
  # @param train_size When given (self-attention only), keys and values
  #   are the first `train_size` positions of the normalized query. This
  #   is how the ICL stage restricts context to labelled rows: by
  #   slicing, not by masking.
  # @param cached_kv Optional `list(key, value)` from [cache_kv()],
  #   covering exactly the positions `train_size` would have sliced to.
  forward = function(q, k = NULL, v = NULL, attn_mask = NULL,
                     train_size = NULL, rope = NULL, cached_kv = NULL) {
    q_n <- self$norm1(q)

    if (!is.null(cached_kv)) {
      if (!is.null(k) || !is.null(v) || !is.null(train_size)) {
        cli::cli_abort(
          "{.arg k}/{.arg v}/{.arg train_size} must be NULL when \\
           {.arg cached_kv} is given."
        )
      }
      attn <- self$attn(q_n, attn_mask = attn_mask, rope = rope,
                        cached_kv = cached_kv)
    } else if (!is.null(train_size)) {
      if (!is.null(k) || !is.null(v)) {
        cli::cli_abort("{.arg k}/{.arg v} must be NULL when {.arg train_size} is given.")
      }
      k_n <- q_n[.., 1:train_size, ]
      attn <- self$attn(q_n, k_n, k_n, attn_mask = attn_mask, rope = rope)
    } else if (is.null(k) && is.null(v)) {
      attn <- self$attn(q_n, attn_mask = attn_mask, rope = rope)
    } else {
      k_n <- self$norm1(k)
      # The reference reuses the normalized key for the value whenever
      # they are the same object, which every call site here satisfies.
      v_n <- if (identical(v, k)) k_n else self$norm1(v)
      attn <- self$attn(q_n, k_n, v_n, attn_mask = attn_mask, rope = rope)
    }

    x <- q + attn
    x + self$feed_forward(self$norm2(x))
  },

  #' Key/value projections of `k`, for a KV cache.
  #'
  #' `k` is the block's key source *before* `norm1` -- for the sliced
  #' `train_size` path that is the training rows of the block's own input,
  #' which is why a cache built from a training-only pass lines up with
  #' what the full pass would have sliced.
  #' @keywords internal
  cache_kv = function(k, rope = NULL) {
    self$attn$cache_kv(self$norm1(k), rope = rope)
  }
)


#' Induced self-attention block
#'
#' Inducing points attend over the source, then the source attends back.
#' Unlike TabFM's variant, the restriction to training rows is done by
#' **slicing** the source before the first attention, not by masking.
#' @keywords internal
tabicl_isab <- torch::nn_module(
  "TabiclInducedSelfAttentionBlock",

  initialize = function(embedding_dim, n_heads, dim_ff, num_inds,
                        ssmax = FALSE, bias_free_ln = FALSE,
                        activation = "gelu") {
    self$ind_vectors <- torch::nn_parameter(
      torch::torch_zeros(num_inds, embedding_dim)
    )
    self$num_inds <- as.integer(num_inds)
    self$multihead_attn1 <- tabicl_mab(embedding_dim, n_heads, dim_ff,
                                       ssmax = ssmax,
                                       bias_free_ln = bias_free_ln,
                                       activation = activation)
    # Only the first attention carries a scalable softmax: it is the one
    # whose source length varies with the number of rows.
    self$multihead_attn2 <- tabicl_mab(embedding_dim, n_heads, dim_ff,
                                       ssmax = FALSE,
                                       bias_free_ln = bias_free_ln,
                                       activation = activation)
    self$skip_value <- -100.0
  },

  #' Summarise the labelled rows of `src` into the inducing points.
  #'
  #' The whole block's output depends on its input only through this
  #' `(num_inds, E)` summary, and the slice keeps just the labelled rows,
  #' so the summary is a function of them alone -- which is what a cache
  #' can stand in for.
  #' @keywords internal
  induce = function(src, train_size = NULL) {
    sz <- as.integer(src$size())
    lead <- sz[-c(length(sz) - 1L, length(sz))]
    d_model <- sz[length(sz)]
    ind <- self$ind_vectors$expand(c(lead, self$num_inds, d_model))

    kv <- if (is.null(train_size)) src else src[.., 1:train_size, ]
    self$multihead_attn1(ind, kv, kv)
  },

  # @param src `(..., T, E)`.
  # @param train_size Restrict the inducing points to the first
  #   `train_size` positions.
  # @param hidden Precomputed [induce()] output, or NULL.
  induced_attention = function(src, train_size = NULL, hidden = NULL) {
    if (is.null(hidden)) hidden <- self$induce(src, train_size)
    self$multihead_attn2(src, hidden, hidden)
  },

  forward = function(src, train_size = NULL, hidden = NULL) {
    if (!is.null(hidden)) {
      # The cached path never reaches the sentinel branch below: a cache
      # is only built when no column is uniformly `-100` (see
      # `build_hidden()`), so `is_skip` is FALSE throughout and this is
      # the branch the uncached pass would have taken anyway.
      return(self$induced_attention(src, hidden = hidden))
    }
    # Column slots reserved for the row stage's CLS tokens arrive filled
    # with the -100 sentinel. The reference detects those and passes them
    # through untouched rather than letting them enter attention; the row
    # stage overwrites them a moment later anyway.
    #
    # In the released (target-aware) configuration this never fires: the
    # target embedding is added to the training positions *before* the
    # set transformer runs, so a reserved column is no longer uniformly
    # -100 by the time it gets here. The branch is kept because the
    # reference has it and a non-target-aware checkpoint would use it.
    is_skip <- (src == self$skip_value)$all(dim = -1L)$all(dim = -1L)
    if (!as.logical(is_skip$any()$item())) {
      return(self$induced_attention(src, train_size))
    }
    if (as.logical(is_skip$all()$item())) {
      return(torch::torch_full_like(src, self$skip_value))
    }

    # `src[~skip_mask]` in the reference flattens every leading axis;
    # do the same rather than assuming a single batch axis.
    sz <- as.integer(src$size())
    n <- length(sz)
    flat <- src$reshape(c(-1L, sz[n - 1L], sz[n]))
    keep <- torch::torch_nonzero(!is_skip$reshape(-1L))$squeeze(-1L)
    kept <- torch::torch_index_select(flat, dim = 1L, index = keep)
    out_kept <- self$induced_attention(kept, train_size)
    out <- torch::torch_full_like(flat, self$skip_value)
    out <- out$index_copy(1L, keep, out_kept)
    out$reshape(sz)
  }
)


#' Stack of induced self-attention blocks
#' @keywords internal
tabicl_set_transformer <- torch::nn_module(
  "TabiclSetTransformer",

  initialize = function(num_blocks, embedding_dim, n_heads, dim_ff, num_inds,
                        ssmax = FALSE, bias_free_ln = FALSE,
                        activation = "gelu") {
    self$blocks <- torch::nn_module_list(
      lapply(seq_len(num_blocks), function(i)
        tabicl_isab(embedding_dim, n_heads, dim_ff, num_inds,
                    ssmax = ssmax, bias_free_ln = bias_free_ln,
                    activation = activation))
    )
  },

  # @param hidden List of per-block summaries from [build_hidden()], or
  #   NULL to compute them from `src`.
  forward = function(src, train_size = NULL, hidden = NULL) {
    for (i in seq_along(self$blocks)) {
      src <- self$blocks[[i]](
        src, train_size = if (is.null(hidden)) train_size else NULL,
        hidden = if (is.null(hidden)) NULL else hidden[[i]]
      )
    }
    src
  },

  # Run the stack over labelled rows only, keeping each block's summary.
  #
  # @return `list(src, hidden)` -- the stack's usual output, plus the
  #   per-block summaries a later test-only pass needs.
  #
  # Refuses when a column is uniformly `-100`, because that is the one
  # input the blocks route around rather than attend over, and a cache
  # cannot represent "this column was skipped". In the released
  # target-aware checkpoints it never happens: the target embedding is
  # added to the labelled rows before the set transformer runs, so no
  # column arrives uniformly sentinel.
  build_hidden = function(src) {
    out <- vector("list", length(self$blocks))
    for (i in seq_along(self$blocks)) {
      blk <- self$blocks[[i]]
      is_skip <- (src == blk$skip_value)$all(dim = -1L)$all(dim = -1L)
      if (as.logical(is_skip$any()$item())) {
        cli::cli_abort(c(
          "Cannot build a column-stage cache for this checkpoint.",
          i = "A column of the labelled rows is uniformly \\
               {blk$skip_value}, which the set transformer passes through \\
               rather than attending over. Predict without {.arg kv_cache}."
        ))
      }
      h <- blk$induce(src)
      out[[i]] <- h$detach()
      src <- blk(src, hidden = h)
    }
    list(src = src, hidden = out)
  }
)


#' Stack of pre-norm attention blocks, optionally with one shared RoPE
#' @keywords internal
tabicl_encoder <- torch::nn_module(
  "TabiclEncoder",

  initialize = function(num_blocks, embedding_dim, n_heads, dim_ff,
                        ssmax = FALSE, bias_free_ln = FALSE,
                        activation = "gelu", rope_base = NULL,
                        rope_interleaved = FALSE) {
    self$use_rope <- !is.null(rope_base)
    if (self$use_rope) {
      # A learnable table: TabICL trained these, so they are parameters
      # in the checkpoint rather than a closed form.
      self$rope <- rope(as.integer(embedding_dim / n_heads), rope_base,
                        interleaved = rope_interleaved, learnable = TRUE)
    }
    self$blocks <- torch::nn_module_list(
      lapply(seq_len(num_blocks), function(i)
        tabicl_mab(embedding_dim, n_heads, dim_ff, ssmax = ssmax,
                   bias_free_ln = bias_free_ln, activation = activation))
    )
  },

  # @param cached_kv List of per-block `list(key, value)` from
  #   [build_kv()], or NULL.
  forward = function(x, attn_mask = NULL, train_size = NULL,
                     cached_kv = NULL) {
    rp <- if (self$use_rope) self$rope else NULL
    for (i in seq_along(self$blocks)) {
      x <- self$blocks[[i]](
        x, attn_mask = attn_mask,
        train_size = if (is.null(cached_kv)) train_size else NULL,
        rope = rp,
        cached_kv = if (is.null(cached_kv)) NULL else cached_kv[[i]]
      )
    }
    x
  },

  #' Run the stack over labelled rows only, keeping each block's key/value
  #' pair.
  #'
  #' The `train_size` path slices its keys out of the block's own
  #' normalized input, so a pass carrying nothing but the labelled rows
  #' produces exactly the tensor that slice would have selected.
  #' @keywords internal
  build_kv = function(x) {
    rp <- if (self$use_rope) self$rope else NULL
    kv <- vector("list", length(self$blocks))
    # Every row here is a labelled row, so the slice covers all of them --
    # but it still has to be *taken*. Passing `train_size = NULL` instead
    # would drop into the fused single-matmul self-attention branch, and
    # the fused and split projections disagree in the last bit, which is
    # exactly the seed a stack of blocks amplifies.
    full <- x$size(x$dim() - 1L)
    for (i in seq_along(self$blocks)) {
      kv[[i]] <- self$blocks[[i]]$cache_kv(x, rope = rp)
      x <- self$blocks[[i]](x, train_size = full, rope = rp)
    }
    kv
  }
)
