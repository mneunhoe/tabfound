# Transformer blocks shared by the set-transformer family (TabFM, and
# TabICL when it lands).
#
# The block convention here is "sandwich norm": each sublayer is
# normalized on the way in *and* on the way out, with the residual added
# outside both. That differs from TabPFN's post-norm layer, which is why
# these live separately rather than being parameterised into one module.

#' Multi-head attention block with sandwich norms and a SwiGLU feed-forward
#'
#'   x <- q + post_attn_ln(attn(pre_attn_ln(q), pre_attn_ln(k), pre_attn_ln(v)))
#'   x <- x + post_ff_ln(linear2(silu(linear1_gate(pre_ff_ln(x))) * linear1(pre_ff_ln(x))))
#'
#' Note the same `pre_attn_ln` normalizes the query, key and value
#' inputs — it is one module applied three times, not three modules.
#'
#' @param embedding_dim Model width.
#' @param n_heads Number of attention heads.
#' @param dim_ff Feed-forward width.
#' @param use_rope Passed through to [mha_qk_norm()].
#' @keywords internal
mab <- torch::nn_module(
  "MultiheadAttentionBlock",

  initialize = function(embedding_dim, n_heads, dim_ff, use_rope = FALSE) {
    self$attn <- mha_qk_norm(embedding_dim, n_heads, use_rope = use_rope)
    self$pre_attn_ln  <- rms_norm(embedding_dim)
    self$post_attn_ln <- rms_norm(embedding_dim)
    self$pre_ff_ln    <- rms_norm(embedding_dim)
    self$post_ff_ln   <- rms_norm(embedding_dim)
    self$linear1      <- torch::nn_linear(embedding_dim, dim_ff, bias = TRUE)
    self$linear1_gate <- torch::nn_linear(embedding_dim, dim_ff, bias = TRUE)
    self$linear2      <- torch::nn_linear(dim_ff, embedding_dim, bias = TRUE)
  },

  feed_forward = function(x) {
    xn <- self$pre_ff_ln(x)
    h <- torch::nnf_silu(self$linear1_gate(xn)) * self$linear1(xn)
    self$post_ff_ln(self$linear2(h))
  },

  # @param q Query input `(B, Tq, E)`.
  # @param k,v Key/value inputs. Default to `q` (self-attention).
  # @param cached_kv Optional `list(key, value)` from [cache_kv()]. When
  #   given it replaces `k`/`v` and `attn_mask` must be `NULL` -- the
  #   cache already holds exactly the positions the mask would have kept.
  forward = function(q, k = NULL, v = NULL, attn_mask = NULL, rope = NULL,
                     cached_kv = NULL) {
    q_n <- self$pre_attn_ln(q)

    a <- if (is.null(cached_kv)) {
      k_n <- self$pre_attn_ln(if (is.null(k)) q else k)
      v_n <- self$pre_attn_ln(if (is.null(v)) q else v)
      self$post_attn_ln(
        self$attn(q_n, k_n, v_n, attn_mask = attn_mask, rope = rope)
      )
    } else {
      self$post_attn_ln(
        self$attn(q_n, cached_kv = cached_kv, rope = rope)
      )
    }
    x <- q + a
    x + self$feed_forward(x)
  },

  #' Key/value projections of this block's key input, for a KV cache.
  #' @keywords internal
  cache_kv = function(k, v = NULL, rope = NULL) {
    k_n <- self$pre_attn_ln(k)
    v_n <- if (is.null(v)) k_n else self$pre_attn_ln(v)
    self$attn$cache_kv(k_n, v_n, rope = rope)
  }
)


#' Induced self-attention block (set transformer)
#'
#' Learned inducing points attend over the input (`mab1`), then the input
#' attends back over those inducing points (`mab2`). This is what turns
#' the cost of attending over `n` rows from `O(n^2)` into `O(n * k)` for
#' `k` inducing points, and it is why the column stage can take an entire
#' training fold as one sequence.
#'
#' @param num_inds Number of inducing points.
#' @keywords internal
isab <- torch::nn_module(
  "InducedSelfAttentionBlock",

  initialize = function(embedding_dim, n_heads, dim_ff, num_inds) {
    self$ind_vectors <- torch::nn_parameter(
      torch::torch_zeros(num_inds, embedding_dim)
    )
    self$mab1 <- mab(embedding_dim, n_heads, dim_ff)
    self$mab2 <- mab(embedding_dim, n_heads, dim_ff)
  },

  #' Summarise `src` into the inducing points.
  #'
  #' Everything the block's input contributes to its output passes through
  #' this `(num_inds, E)` bottleneck, which is what makes the block
  #' cacheable: `attn_mask` keeps only the training rows, so `hidden`
  #' depends on them alone and a later call can supply it instead of the
  #' rows it was computed from.
  #' @keywords internal
  induce = function(src, attn_mask = NULL) {
    ind <- self$ind_vectors$unsqueeze(1L)$expand(
      c(src$size(1), self$ind_vectors$size(1), self$ind_vectors$size(2))
    )
    self$mab1(ind, src, src, attn_mask = attn_mask)
  },

  # @param src `(B, T, E)`.
  # @param attn_mask Mask over `src`, applied in `mab1` only.
  # @param hidden Precomputed [induce()] output. When given, `src` is not
  #   summarised at all -- only attended back over the supplied points.
  forward = function(src, attn_mask = NULL, hidden = NULL) {
    if (is.null(hidden)) hidden <- self$induce(src, attn_mask)
    self$mab2(src, hidden, hidden)
  }
)


#' Stack of induced self-attention blocks
#' @keywords internal
set_transformer <- torch::nn_module(
  "SetTransformer",

  initialize = function(num_blocks, embedding_dim, n_heads, dim_ff, num_inds) {
    self$blocks <- torch::nn_module_list(
      lapply(seq_len(num_blocks), function(i)
        isab(embedding_dim, n_heads, dim_ff, num_inds))
    )
  },

  # @param hidden List of per-block inducing summaries from
  #   [build_hidden()], or NULL to compute them from `src`.
  forward = function(src, attn_mask = NULL, hidden = NULL) {
    for (i in seq_along(self$blocks)) {
      src <- self$blocks[[i]](src, attn_mask = attn_mask,
                              hidden = if (is.null(hidden)) NULL else hidden[[i]])
    }
    src
  },

  # Run the stack over training rows, keeping each block's summary.
  #
  # @return `list(src, hidden)` -- the stack's usual output, plus
  #   everything a later pass needs to push test rows through without
  #   the training rows being present at all.
  build_hidden = function(src, attn_mask = NULL) {
    out <- vector("list", length(self$blocks))
    for (i in seq_along(self$blocks)) {
      h <- self$blocks[[i]]$induce(src, attn_mask)
      out[[i]] <- h$detach()
      src <- self$blocks[[i]](src, hidden = h)
    }
    list(src = src, hidden = out)
  }
)


#' Stack of attention blocks with one shared RoPE
#'
#' `rope_base = NULL` builds a stack without any positional encoding,
#' which is what the in-context-learning stage uses — the training rows
#' are a set, not a sequence.
#' @keywords internal
encoder_stack <- torch::nn_module(
  "EncoderStack",

  initialize = function(num_blocks, embedding_dim, n_heads, dim_ff,
                        rope_base = 100000.0) {
    use_rope <- !is.null(rope_base)
    # One RoPE per stack, shared by every block, so the checkpoint has a
    # single `rope.freqs` buffer rather than one per block.
    if (use_rope) {
      # TabFM rotates (B, T, N, Dh) along axis 2, with adjacent-channel
      # pairing.
      self$rope <- rope(as.integer(embedding_dim / n_heads), rope_base,
                        interleaved = TRUE, seq_axis = 2L)
    }
    self$use_rope <- use_rope
    self$blocks <- torch::nn_module_list(
      lapply(seq_len(num_blocks), function(i)
        mab(embedding_dim, n_heads, dim_ff, use_rope = use_rope))
    )
  },

  # @param cached_kv List of per-block `list(key, value)` from
  #   [build_kv()], or NULL for the ordinary self-attention pass.
  forward = function(x, attn_mask = NULL, cached_kv = NULL) {
    rp <- if (self$use_rope) self$rope else NULL
    for (i in seq_along(self$blocks)) {
      x <- self$blocks[[i]](
        x, attn_mask = if (is.null(cached_kv)) attn_mask else NULL, rope = rp,
        cached_kv = if (is.null(cached_kv)) NULL else cached_kv[[i]]
      )
    }
    x
  },

  #' Run the stack over training rows, keeping each block's key/value pair.
  #'
  #' Each block's cache is taken from that block's *input*, before the
  #' block runs -- which is the tensor the uncached pass would have
  #' projected into keys and values at that point.
  #' @keywords internal
  build_kv = function(x) {
    rp <- if (self$use_rope) self$rope else NULL
    kv <- vector("list", length(self$blocks))
    for (i in seq_along(self$blocks)) {
      kv[[i]] <- self$blocks[[i]]$cache_kv(x, rope = rp)
      x <- self$blocks[[i]](x, rope = rp)
    }
    kv
  }
)
