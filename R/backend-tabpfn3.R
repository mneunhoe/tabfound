# TabPFN v3 backend (Prior-Labs).
#
# v3 is a different model from v2.x, not a wider one. Every generation up
# to v2.6 was a single stack of blocks that alternated attention between a
# row's features and between a column's cells, with the target riding
# along as one extra column. v3 splits that into four stages, each with
# its own module tree and its own idea of what a "sequence" is:
#
#   0. Cell embedding. Preprocessing (NaN/Inf indicators, mean imputation,
#      standard scaling) happens in `forward()` as it does in v2.6, but
#      the grouping is different: instead of packing consecutive features
#      into one token, every column becomes a token carrying the values of
#      the columns 1, 2 and 4 places to its right, circularly. The column
#      count is therefore preserved rather than divided by the group size.
#
#   1. Feature distribution embedder. Per column, a stack of
#      SetTransformer-style induced self-attention blocks: `n_ind` learned
#      inducing vectors attend to the training rows, then every row
#      attends back to them. That is what makes the cost linear in rows
#      instead of quadratic, and it is also why the training rows' entire
#      contribution to this stage fits in `n_ind` vectors per column --
#      see [tabpfn3_kv_cache()].
#
#   2. Column aggregator. Per row, a small transformer over the columns
#      with `n_cls` learned CLS tokens prepended and RoPE over the feature
#      axis. The last block is a CLS-only readout. Concatenating the CLS
#      outputs gives the row embedding, `n_cls * embed_dim` wide.
#
#   3. ICL transformer. 24 blocks of attention across rows, keys and
#      values restricted to the training rows -- which is both the causal
#      mask and the reason the whole training set can be cached. Test rows
#      use a single KV head (`icl_num_kv_heads_test = 1`), so the cache is
#      one head wide.
#
# The decoder differs by head. The regressor projects the row embedding to
# 5000 bar-distribution logits, as v2 did. The classifier does not project
# at all: it runs one more attention, with the test rows as queries, the
# training rows as keys, and their *one-hot labels* as values, so the
# output is a weighted average of training labels -- retrieval, not
# classification -- and the logits are its log. That is what lets one
# checkpoint serve up to `max_num_classes = 160` classes.
#
# There is no feature positional embedding: v2's pre-generated 2000 x 48
# buffer has no counterpart here, because RoPE in stage 2 does that job.
#
# Reference: `tabpfn/architectures/tabpfn_v3.py` in the PyPI `tabpfn`
# package (8.2.0). Checkpoints are PyTorch pickles; convert them once with
# `inst/python/tabpfn_convert_ckpt.py`, which writes `arch: "tabpfn_v3"`
# into the config so `detect_backend()` can tell the generations apart.
#
# Module names below are the checkpoint's own, so no key translation is
# needed: R torch's `nn_sequential` and `nn_module_list` produce the same
# dotted integer paths PyTorch does.

# `torch.finfo(torch.float32).eps`. Used in the two places the reference
# leaves implicit: `TorchStandardScaler.transform`'s denominator, and
# `nn.RMSNorm`'s epsilon, which defaults to the input dtype's eps.
TABPFN3_F32_EPS <- 1.1920928955078125e-07

# Sentinels marking NaN / +Inf / -Inf in the indicator channel. Same
# values as v2.5 and v2.6, repeated here because they belong to the
# architecture rather than to a shared utility.
TABPFN3_NAN_INDICATOR     <- -2.0
TABPFN3_POS_INF_INDICATOR <-  2.0
TABPFN3_NEG_INF_INDICATOR <-  4.0


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------

#' Scaled dot-product attention on `(B, S, H, D)` tensors
#'
#' The reference keeps queries, keys and values head-last and permutes
#' inside its own `scaled_dot_product_attention` wrapper; this does the
#' same so the shapes in the callers line up with the reference's suffix
#' notation one for one.
#'
#' When there are fewer key/value heads than query heads the keys and
#' values are repeated. Torch only has a fused grouped-query kernel for
#' half precision, so the reference falls back to `repeat_interleave` in
#' float32 -- which is what this reproduces, and is exact either way.
#'
#' @param q `(B, S, H, D)`; @param k,v `(B, J, Hkv, D)`.
#' @param scaling Optional [tabpfn3_softmax_scaling_mlp()], applied to the
#'   queries before the attention with `J` as its length argument.
#' @keywords internal
tabpfn3_sdpa <- function(q, k, v, scaling = NULL) {
  if (!is.null(scaling)) q <- scaling(q, k$size(2))
  qh <- q$permute(c(1L, 3L, 2L, 4L))
  kh <- k$permute(c(1L, 3L, 2L, 4L))
  vh <- v$permute(c(1L, 3L, 2L, 4L))
  n_q <- qh$size(2); n_kv <- kh$size(2)
  if (n_q != n_kv) {
    reps <- as.integer(n_q / n_kv)
    kh <- torch::torch_repeat_interleave(kh, repeats = reps, dim = 2L)
    vh <- torch::torch_repeat_interleave(vh, repeats = reps, dim = 2L)
  }
  ctx <- torch:::torch_scaled_dot_product_attention(
    query = qh, key = kh, value = vh, dropout_p = 0
  )
  ctx$permute(c(1L, 3L, 2L, 4L))
}


#' Query-aware attention temperature
#'
#' Softmax attention sharpens as the key sequence grows; v3 learns the
#' correction rather than fixing it at `1/sqrt(d)`. Queries are scaled by
#'
#'   `base_mlp(log n) * (1 + tanh(query_mlp(q)))`
#'
#' where `n` is the number of keys, so the first factor is a per-head,
#' per-channel function of context length and the second is a
#' per-query modulation of it. The usual `1/sqrt(d)` is still applied
#' afterwards by the attention itself.
#'
#' @param num_heads,head_dim Shape of the queries this scales.
#' @param n_hidden Width of both MLPs' hidden layer.
#' @keywords internal
tabpfn3_softmax_scaling_mlp <- torch::nn_module(
  "TabPFN3SoftmaxScalingMLP",

  initialize = function(num_heads, head_dim, n_hidden = 64L) {
    self$num_heads <- as.integer(num_heads)
    self$head_dim  <- as.integer(head_dim)
    n_hidden <- as.integer(n_hidden)
    self$base_mlp <- torch::nn_sequential(
      torch::nn_linear(1L, n_hidden),
      torch::nn_gelu(),
      torch::nn_linear(n_hidden, self$num_heads * self$head_dim)
    )
    self$query_mlp <- torch::nn_sequential(
      torch::nn_linear(self$head_dim, n_hidden),
      torch::nn_gelu(),
      torch::nn_linear(n_hidden, self$head_dim)
    )
  },

  # @param q `(B, S, H, D)`; @param n Number of keys attended to.
  forward = function(q, n) {
    logn <- torch::torch_tensor(
      log(max(as.numeric(n), 1)), dtype = q$dtype, device = q$device
    )$reshape(c(1L, 1L))
    base <- self$base_mlp(logn)$view(c(1L, 1L, self$num_heads, self$head_dim))
    q * (base * (1 + torch::torch_tanh(self$query_mlp(q))))
  }
)


#' Multi-head self-attention, optionally rotary
#'
#' Four bias-free linears named exactly as the checkpoint stores them.
#' Used by the column aggregator, where the sequence axis is the feature
#' axis and RoPE supplies the column ordering.
#' @keywords internal
tabpfn3_attention <- torch::nn_module(
  "TabPFN3Attention",

  initialize = function(embedding_size, num_heads, head_dim) {
    self$num_heads <- as.integer(num_heads)
    self$head_dim  <- as.integer(head_dim)
    inner <- self$num_heads * self$head_dim
    self$q_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$k_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$v_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$out_projection <- torch::nn_linear(inner, embedding_size, bias = FALSE)
  },

  # @param x `(B, S, E)`; @param rope A [rope()] module, or NULL.
  forward = function(x, rope = NULL) {
    B <- x$size(1); S <- x$size(2)
    q <- self$q_projection(x)$view(c(B, S, -1L, self$head_dim))
    k <- self$k_projection(x)$view(c(B, S, -1L, self$head_dim))
    v <- self$v_projection(x)$view(c(B, S, -1L, self$head_dim))
    if (!is.null(rope)) {
      # `rope()` rotates along the second-to-last axis, so the head axis
      # has to be moved out of the way and back.
      q <- rope(q$transpose(2L, 3L))$transpose(2L, 3L)
      k <- rope(k$transpose(2L, 3L))$transpose(2L, 3L)
    }
    out <- tabpfn3_sdpa(q, k, v)
    self$out_projection(out$reshape(c(B, S, self$num_heads * self$head_dim)))
  }
)


#' Multi-head cross-attention
#'
#' Queries come from one sequence and keys/values from another. Used in
#' both halves of an induced self-attention block: inducing vectors
#' attending to rows, then rows attending to inducing vectors.
#' @keywords internal
tabpfn3_cross_attention <- torch::nn_module(
  "TabPFN3CrossAttention",

  initialize = function(embedding_size, num_heads, head_dim,
                        softmax_scaling_layer = NULL) {
    self$num_heads <- as.integer(num_heads)
    self$head_dim  <- as.integer(head_dim)
    inner <- self$num_heads * self$head_dim
    self$q_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$k_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$v_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$out_projection <- torch::nn_linear(inner, embedding_size, bias = FALSE)
    if (!is.null(softmax_scaling_layer)) {
      self$softmax_scaling_layer <- softmax_scaling_layer
    }
    self$has_scaling <- !is.null(softmax_scaling_layer)
  },

  # @param x_q `(B, Q, E)`; @param x_kv `(B, V, E)`.
  forward = function(x_q, x_kv) {
    B <- x_q$size(1); Q <- x_q$size(2); V <- x_kv$size(2)
    q <- self$q_projection(x_q)$view(c(B, Q, -1L, self$head_dim))
    k <- self$k_projection(x_kv)$view(c(B, V, -1L, self$head_dim))
    v <- self$v_projection(x_kv)$view(c(B, V, -1L, self$head_dim))
    out <- tabpfn3_sdpa(
      q, k, v,
      scaling = if (self$has_scaling) self$softmax_scaling_layer else NULL
    )
    self$out_projection(out$reshape(c(B, Q, self$num_heads * self$head_dim)))
  }
)


#' In-context attention: every row attends to the training rows only
#'
#' The keys and values are projected from `x[1:single_eval_pos]`, so a
#' test row can see the training rows but neither itself nor the other
#' test rows. No mask is needed and none is built; the restriction is the
#' slice.
#'
#' Test rows additionally use only the first `num_kv_heads_test` key/value
#' heads, broadcast across all query heads. That is what keeps
#' [tabpfn3_kv_cache()] one head wide per block instead of eight.
#'
#' @keywords internal
tabpfn3_icl_attention <- torch::nn_module(
  "TabPFN3ICLAttention",

  initialize = function(embedding_size, num_heads, head_dim,
                        softmax_scaling_layer = NULL,
                        num_kv_heads = NULL, num_kv_heads_test = NULL) {
    self$num_heads <- as.integer(num_heads)
    self$head_dim  <- as.integer(head_dim)
    self$num_kv_heads <- if (is.null(num_kv_heads)) self$num_heads
                         else as.integer(num_kv_heads)
    self$num_kv_heads_test <- if (is.null(num_kv_heads_test)) NULL
                              else as.integer(num_kv_heads_test)
    inner    <- self$num_heads * self$head_dim
    inner_kv <- self$num_kv_heads * self$head_dim
    self$q_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$k_projection   <- torch::nn_linear(embedding_size, inner_kv, bias = FALSE)
    self$v_projection   <- torch::nn_linear(embedding_size, inner_kv, bias = FALSE)
    self$out_projection <- torch::nn_linear(inner, embedding_size, bias = FALSE)
    if (!is.null(softmax_scaling_layer)) {
      self$softmax_scaling_layer <- softmax_scaling_layer
    }
    self$has_scaling <- !is.null(softmax_scaling_layer)
  },

  # @param x `(B, R, E)`. In the cached path `R` counts test rows only.
  # @param single_eval_pos Rows from this index on are test rows.
  # @param cached_kv `list(key, value)` from a previous pass, holding the
  #   training rows' projections for the test KV heads. Supplied, the K/V
  #   projections are skipped entirely -- that is the whole saving.
  # @param return_kv Also return that entry, for building a cache.
  # @return `list(out, kv)`.
  forward = function(x, single_eval_pos, cached_kv = NULL, return_kv = FALSE) {
    B <- x$size(1); R <- x$size(2)
    scaling <- if (self$has_scaling) self$softmax_scaling_layer else NULL
    q <- self$q_projection(x)$view(c(B, R, self$num_heads, self$head_dim))

    if (!is.null(cached_kv)) {
      out <- tabpfn3_sdpa(q, cached_kv$key, cached_kv$value, scaling = scaling)
      return(list(
        out = self$out_projection(
          out$reshape(c(B, R, self$num_heads * self$head_dim))
        ),
        kv = NULL
      ))
    }

    # Not named `N`: inside `[`, R torch reads that symbol as "to the end"
    # rather than as the local, and `x[, 1:N, ]` would silently take every
    # row -- test rows included, which is the one thing this must not do.
    n_ctx <- if (is.null(single_eval_pos)) R else as.integer(single_eval_pos)
    x_train <- if (n_ctx == R) x else x[, 1:n_ctx, ]
    k <- self$k_projection(x_train)$view(c(B, n_ctx, self$num_kv_heads, self$head_dim))
    v <- self$v_projection(x_train)$view(c(B, n_ctx, self$num_kv_heads, self$head_dim))

    n_test_heads <- self$num_kv_heads_test
    if (!is.null(n_test_heads) && n_ctx < R) {
      out_train <- tabpfn3_sdpa(q[, 1:n_ctx, , ], k, v, scaling = scaling)
      out_test  <- tabpfn3_sdpa(
        q[, (n_ctx + 1L):R, , ], k[, , 1:n_test_heads, ], v[, , 1:n_test_heads, ],
        scaling = scaling
      )
      out <- torch::torch_cat(list(out_train, out_test), dim = 2L)
    } else {
      out <- tabpfn3_sdpa(q, k, v, scaling = scaling)
    }

    kv <- NULL
    if (isTRUE(return_kv)) {
      # Only the heads a test row can reach are worth keeping. `$contiguous()`
      # so the slice owns its storage and the full projection can be freed.
      keep <- if (is.null(n_test_heads)) self$num_kv_heads else n_test_heads
      kv <- list(key   = k[, , 1:keep, ]$detach()$contiguous(),
                 value = v[, , 1:keep, ]$detach()$contiguous())
    }
    list(
      out = self$out_projection(
        out$reshape(c(B, R, self$num_heads * self$head_dim))
      ),
      kv = kv
    )
  }
)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

# The feedforward network every block uses: two bias-free linears with a
# GELU between them. `nn_sequential` gives it the `.0` / `.2` key names the
# checkpoint has.
# @keywords internal
tabpfn3_mlp <- function(emsize, dim_feedforward) {
  torch::nn_sequential(
    torch::nn_linear(emsize, dim_feedforward, bias = FALSE),
    torch::nn_gelu(),
    torch::nn_linear(dim_feedforward, emsize, bias = FALSE)
  )
}


#' Pre-norm cross-attention block
#'
#' Query and key/value streams are normalised separately -- they are
#' different sequences and, in the induced-attention block, different
#' kinds of thing -- and each of the two sublayers adds its own residual.
#' @keywords internal
tabpfn3_cross_attention_block <- torch::nn_module(
  "TabPFN3CrossAttentionBlock",

  initialize = function(emsize, nhead, dim_feedforward,
                        softmax_scaling_layer = NULL, eps = NULL) {
    if (is.null(eps)) eps <- TABPFN3_F32_EPS
    self$attn <- tabpfn3_cross_attention(
      embedding_size = emsize, num_heads = nhead,
      head_dim = as.integer(emsize / nhead),
      softmax_scaling_layer = softmax_scaling_layer
    )
    self$mlp <- tabpfn3_mlp(emsize, dim_feedforward)
    self$layernorm_q  <- rms_norm(emsize, eps = eps)
    self$layernorm_kv <- rms_norm(emsize, eps = eps)
    self$layernorm2   <- rms_norm(emsize, eps = eps)
  },

  forward = function(x_q, context) {
    x_q <- x_q + self$attn(self$layernorm_q(x_q), self$layernorm_kv(context))
    x_q + self$mlp(self$layernorm2(x_q))
  }
)


#' Pre-norm self-attention block, used by the column aggregator
#'
#' `forward()` is ordinary self-attention over the feature axis.
#' `forward_cross()` is the CLS-only readout the aggregator's last block
#' runs instead: the same weights, but with the CLS tokens alone as
#' queries and the whole sequence as keys and values.
#' @keywords internal
tabpfn3_transformer_block <- torch::nn_module(
  "TabPFN3TransformerBlock",

  initialize = function(emsize, nhead, dim_feedforward, eps = NULL) {
    if (is.null(eps)) eps <- TABPFN3_F32_EPS
    self$attention <- tabpfn3_attention(
      embedding_size = emsize, num_heads = nhead,
      head_dim = as.integer(emsize / nhead)
    )
    self$layernorm     <- rms_norm(emsize, eps = eps)
    self$layernorm_mlp <- rms_norm(emsize, eps = eps)
    self$mlp <- tabpfn3_mlp(emsize, dim_feedforward)
  },

  # @param x `(B, R, C, E)` -- rows by columns. Attention is over `C`, so
  #   the rows fold into the batch and the work chunks along `B * R`.
  forward = function(x, rope = NULL, save_peak_memory_factor = NULL) {
    attn <- self$attention
    ln   <- self$layernorm
    x <- chunked_evaluate(function(z) attn(ln(z), rope = rope), x,
                          save_peak_memory_factor, residual = TRUE,
                          batch_dims = 2L)
    mlp <- self$mlp
    lnm <- self$layernorm_mlp
    chunked_evaluate(function(z) mlp(lnm(z)), x,
                     save_peak_memory_factor, residual = TRUE, batch_dims = 3L)
  },

  # @param query `(B, R, Q, E)`; @param context `(B, R, V, E)`.
  forward_cross = function(query, context, rope = NULL) {
    B <- query$size(1); R <- query$size(2); Q <- query$size(3)
    V <- context$size(3); E <- context$size(4)
    a <- self$attention
    q_flat <- self$layernorm(query)$reshape(c(B * R, Q, E))
    c_flat <- self$layernorm(context)$reshape(c(B * R, V, E))
    q <- a$q_projection(q_flat)$view(c(B * R, Q, -1L, a$head_dim))
    k <- a$k_projection(c_flat)$view(c(B * R, V, -1L, a$head_dim))
    v <- a$v_projection(c_flat)$view(c(B * R, V, -1L, a$head_dim))
    if (!is.null(rope)) {
      q <- rope(q$transpose(2L, 3L))$transpose(2L, 3L)
      k <- rope(k$transpose(2L, 3L))$transpose(2L, 3L)
    }
    out <- tabpfn3_sdpa(q, k, v)$reshape(c(B * R, Q, a$num_heads * a$head_dim))
    x <- query + a$out_projection(out)$view(c(B, R, Q, E))
    x + self$mlp(self$layernorm_mlp(x))
  }
)


#' One block of the in-context-learning transformer
#'
#' Pre-norm attention across rows with training-only keys, then a
#' per-row MLP. This is where 24 of the model's 33 attention layers live.
#' @keywords internal
tabpfn3_icl_block <- torch::nn_module(
  "TabPFN3ICLBlock",

  initialize = function(emsize, nhead, dim_feedforward,
                        softmax_scaling_layer = NULL,
                        num_kv_heads = NULL, num_kv_heads_test = NULL,
                        eps = NULL) {
    if (is.null(eps)) eps <- TABPFN3_F32_EPS
    self$icl_attention <- tabpfn3_icl_attention(
      embedding_size = emsize, num_heads = nhead,
      head_dim = as.integer(emsize / nhead),
      softmax_scaling_layer = softmax_scaling_layer,
      num_kv_heads = num_kv_heads, num_kv_heads_test = num_kv_heads_test
    )
    self$layernorm     <- rms_norm(emsize, eps = eps)
    self$layernorm_mlp <- rms_norm(emsize, eps = eps)
    self$mlp <- tabpfn3_mlp(emsize, dim_feedforward)
  },

  # @param x `(B, R, E)`. @return `list(state, kv)`.
  forward = function(x, single_eval_pos, cached_kv = NULL, return_kv = FALSE,
                     save_peak_memory_factor = NULL) {
    att <- self$icl_attention
    ln  <- self$layernorm
    kv  <- NULL
    if (isTRUE(return_kv)) {
      # Building a cache bypasses chunking *here*: the key/value tensors
      # have to be produced whole, not a slice at a time. The reference
      # does the same, and in both the MLP below is chunked regardless of
      # which branch produced `x`, so a factor still reaches most of the
      # block on a cache build.
      res <- att(ln(x), single_eval_pos = single_eval_pos, return_kv = TRUE)
      x <- x + res$out
      kv <- res$kv
    } else {
      x <- chunked_evaluate(
        function(z) att(ln(z), single_eval_pos = single_eval_pos,
                        cached_kv = cached_kv)$out,
        x, save_peak_memory_factor, residual = TRUE, batch_dims = 1L
      )
    }
    mlp <- self$mlp
    lnm <- self$layernorm_mlp
    list(
      state = chunked_evaluate(function(z) mlp(lnm(z)), x,
                               save_peak_memory_factor, residual = TRUE,
                               batch_dims = 2L),
      kv = kv
    )
  }
)


#' Induced self-attention over a column's cells
#'
#' Attention between every pair of rows would cost `O(n^2)`. Instead a
#' fixed set of learned inducing vectors attends to the training rows,
#' and every row -- training and test alike -- attends back to that
#' summary. Two cross-attentions, both linear in the row count.
#'
#' The summary is a function of the training rows alone, which is what
#' makes it cacheable: `cached_hidden` skips the first half entirely.
#' @keywords internal
tabpfn3_induced_self_attention_block <- torch::nn_module(
  "TabPFN3InducedSelfAttentionBlock",

  initialize = function(emsize, nhead, num_inducing_points, dim_feedforward,
                        softmax_scaling_layer = NULL) {
    self$cross_attn_block1 <- tabpfn3_cross_attention_block(
      emsize = emsize, nhead = nhead, dim_feedforward = dim_feedforward,
      softmax_scaling_layer = softmax_scaling_layer
    )
    self$cross_attn_block2 <- tabpfn3_cross_attention_block(
      emsize = emsize, nhead = nhead, dim_feedforward = dim_feedforward
    )
    self$num_inducing_points <- as.integer(num_inducing_points)
    self$inducing_vectors <- torch::nn_parameter(
      torch::torch_empty(self$num_inducing_points, emsize)
    )
    torch::nn_init_trunc_normal_(self$inducing_vectors, std = 0.02)
  },

  # The inducing summary for one batch of columns. Split out because the
  # cache builder wants it without the second half.
  # @param x `(B * C, R, E)`; keys and values are the first `n_train` rows.
  inducing_hidden = function(x, n_train) {
    n <- if (is.null(n_train)) x$size(2) else as.integer(n_train)
    ind <- self$inducing_vectors$unsqueeze(1L)$expand(
      c(x$size(1), self$num_inducing_points, x$size(3))
    )
    self$cross_attn_block1(ind, if (n == x$size(2)) x else x[, 1:n, ])
  },

  # @param x `(B, R, C, E)`; @return `list(state, hidden)`.
  forward = function(x, n_train = NULL, cached_hidden = NULL,
                     return_hidden = FALSE, save_peak_memory_factor = NULL) {
    B <- x$size(1); R <- x$size(2); C <- x$size(3); E <- x$size(4)
    # Fold the columns into the batch: every column is embedded on its own.
    x_flat <- x$transpose(2L, 3L)$contiguous()$reshape(c(B * C, R, E))

    if (isTRUE(return_hidden) || !is.null(cached_hidden)) {
      # Both run whole, which is what the reference does too, though it
      # takes a longer route to the same place: its `return_hidden` branch
      # calls the unchunked path outright, and its `cached_hidden` branch
      # goes through `chunked_evaluate_maybe_inplace` but is only ever
      # reached with `save_peak_memory_factor = None` -- `_process_row_chunk`
      # withholds it on exactly this path. Which is as well: chunking
      # splits along `B * C`, and the reference hands each chunk the
      # *whole* hidden state, so a caller that did pass a factor would get
      # a shape mismatch rather than a smaller peak.
      #
      # Nothing is lost by it. What bounds this stage on a large table is
      # `stages_0_to_2()`'s row chunking, which caps `R` before the fold
      # into `B * C` ever happens; measured on a 8,000-row cache build, it
      # takes the peak from 16.9 GB to 8.7 GB where a factor of 8 here
      # reaches 12.4 GB, and the two together add nothing to the first.
      hidden <- cached_hidden %||% self$inducing_hidden(x_flat, n_train)
      out <- self$cross_attn_block2(x_flat, hidden)
      hidden_out <- if (isTRUE(return_hidden)) hidden$detach() else NULL
    } else {
      out <- chunked_evaluate(
        function(chunk) {
          self$cross_attn_block2(chunk, self$inducing_hidden(chunk, n_train))
        },
        x_flat, save_peak_memory_factor, residual = FALSE, batch_dims = 1L
      )
      hidden_out <- NULL
    }

    list(state = out$reshape(c(B, C, R, E))$transpose(2L, 3L)$contiguous(),
         hidden = hidden_out)
  }
)


#' The per-column distribution embedder (stage 1)
#'
#' A stack of [tabpfn3_induced_self_attention_block()]s. Each column's
#' cells are a sequence; what the stack learns is that column's empirical
#' distribution, which is why a value only becomes meaningful relative to
#' the others in its column.
#' @keywords internal
tabpfn3_feature_distribution_embedder <- torch::nn_module(
  "TabPFN3FeatureDistributionEmbedder",

  initialize = function(emsize, nhead, num_inducing_points, dim_feedforward,
                        num_layers, softmax_scaling_layer_factory = NULL) {
    self$layers <- torch::nn_module_list(
      lapply(seq_len(as.integer(num_layers)), function(i) {
        tabpfn3_induced_self_attention_block(
          emsize = emsize, nhead = nhead,
          num_inducing_points = num_inducing_points,
          dim_feedforward = dim_feedforward,
          softmax_scaling_layer = if (is.null(softmax_scaling_layer_factory)) NULL
                                  else softmax_scaling_layer_factory()
        )
      })
    )
  },

  # @return `list(state, hidden)` where `hidden` is one tensor per block,
  #   or NULL.
  forward = function(x, n_train = NULL, cached_hidden = NULL,
                     return_hidden = FALSE, save_peak_memory_factor = NULL) {
    n <- length(self$layers)
    hidden <- if (isTRUE(return_hidden)) vector("list", n) else NULL
    for (i in seq_len(n)) {
      res <- self$layers[[i]](
        x, n_train = n_train,
        cached_hidden = if (is.null(cached_hidden)) NULL else cached_hidden[[i]],
        return_hidden = return_hidden,
        save_peak_memory_factor = save_peak_memory_factor
      )
      x <- res$state
      if (isTRUE(return_hidden)) hidden[[i]] <- res$hidden
      collect_between_layers(x)
    }
    list(state = x, hidden = hidden)
  }
)


#' Cross-feature aggregation onto CLS tokens (stage 2)
#'
#' Per row, a transformer over the columns with `num_cls_tokens` learned
#' tokens prepended. All but the last block are ordinary self-attention;
#' the last reads out the CLS tokens alone. Concatenating them gives the
#' row embedding the ICL stage works on.
#'
#' RoPE over the feature axis is the only thing that tells the model which
#' column is which -- v2's pre-generated positional embedding table has no
#' counterpart here.
#' @keywords internal
tabpfn3_column_aggregator <- torch::nn_module(
  "TabPFN3ColumnAggregator",

  initialize = function(emsize, nhead, num_layers, dim_feedforward,
                        num_cls_tokens, use_rope = TRUE, rope_base = 100000,
                        eps = NULL) {
    if (is.null(eps)) eps <- TABPFN3_F32_EPS
    self$embed_dim      <- as.integer(emsize)
    self$num_cls_tokens <- as.integer(num_cls_tokens)
    self$blocks <- torch::nn_module_list(
      lapply(seq_len(as.integer(num_layers)), function(i) {
        tabpfn3_transformer_block(
          emsize = emsize, nhead = nhead, dim_feedforward = dim_feedforward
        )
      })
    )
    self$use_rope <- isTRUE(use_rope)
    if (self$use_rope) {
      # `freqs` is a non-trainable parameter in the reference, not a
      # buffer, so it is a parameter here too and the strict loader finds
      # it where the checkpoint puts it.
      self$rope <- rope(dim = as.integer(emsize / nhead), base = rope_base,
                        interleaved = FALSE, learnable = TRUE)
    }
    self$cls_tokens <- torch::nn_parameter(
      torch::torch_empty(self$num_cls_tokens, emsize)
    )
    torch::nn_init_trunc_normal_(self$cls_tokens, std = 0.02)
    self$out_ln <- rms_norm(emsize, eps = eps)
  },

  # @param x `(B, Ri, C, E)`; @return `(B, Ri, num_cls_tokens, E)`.
  forward = function(x, save_peak_memory_factor = NULL) {
    B <- x$size(1); Ri <- x$size(2); E <- x$size(4)
    r <- if (self$use_rope) self$rope else NULL
    cls <- self$cls_tokens$unsqueeze(1L)$unsqueeze(1L)$
      expand(c(B, Ri, self$num_cls_tokens, E))
    x <- torch::torch_cat(list(cls, x), dim = 3L)

    n <- length(self$blocks)
    if (n > 1L) {
      for (i in seq_len(n - 1L)) {
        x <- self$blocks[[i]](x, rope = r,
                              save_peak_memory_factor = save_peak_memory_factor)
        collect_between_layers(x)
      }
    }
    cls_out <- self$blocks[[n]]$forward_cross(
      x[, , 1:self$num_cls_tokens, ], x, rope = r
    )
    self$out_ln(cls_out)
  }
)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

#' Retrieval decoder for many-class classification
#'
#' Not a classifier head: there is no output projection sized to the class
#' count. Test rows query the training rows, and the *values* are the
#' training labels one-hot encoded, so the attention output is a weighted
#' average of training labels -- a probability vector by construction.
#' The logits are its log, floored so a class no training row carries
#' cannot produce `-Inf`.
#'
#' Because the class count only enters as the width of the one-hot values,
#' one checkpoint serves every class count up to `max_num_classes`.
#' @keywords internal
tabpfn3_many_class_decoder <- torch::nn_module(
  "TabPFN3ManyClassDecoder",

  initialize = function(max_num_classes, input_size, head_dim = 64L,
                        num_heads = 6L, softmax_scaling_layer = NULL) {
    self$max_num_classes <- as.integer(max_num_classes)
    self$head_dim  <- as.integer(head_dim)
    self$num_heads <- as.integer(num_heads)
    attention_size <- self$head_dim * self$num_heads
    self$q_projection <- torch::nn_linear(input_size, attention_size)
    self$k_projection <- torch::nn_linear(input_size, attention_size)
    if (!is.null(softmax_scaling_layer)) {
      self$softmax_scaling_layer <- softmax_scaling_layer
    }
    self$has_scaling <- !is.null(softmax_scaling_layer)
  },

  # @param train_emb `(B, N, E)`; @param test_emb `(B, M, E)`;
  # @param targets `(B, N)` ordinal class indices, 0-based.
  # @return `(B, M, max_num_classes)` log-probabilities.
  forward = function(train_emb, test_emb, targets) {
    B <- test_emb$size(1); M <- test_emb$size(2); n_ctx <- train_emb$size(2)
    D <- self$head_dim; H <- self$num_heads
    q <- self$q_projection(test_emb)$view(c(B, M, H, D))
    k <- self$k_projection(train_emb)$view(c(B, n_ctx, H, D))
    if (self$has_scaling) q <- self$softmax_scaling_layer(q, n_ctx)

    # The reference splits the one-hot values into head-sized chunks so a
    # flash-attention kernel can take them; that is a memory layout, not a
    # different computation, so the scores are formed directly here.
    qh <- q$permute(c(1L, 3L, 2L, 4L))                       # (B, H, M, D)
    kh <- k$permute(c(1L, 3L, 2L, 4L))                       # (B, H, N, D)  
    scores <- torch::torch_matmul(qh, kh$transpose(3L, 4L)) / sqrt(D)
    w <- torch::nnf_softmax(scores, dim = -1L)               # (B, H, M, N)

    classes <- torch::torch_arange(
      0L, self$max_num_classes - 1L, dtype = targets$dtype, device = targets$device
    )$view(c(1L, 1L, self$max_num_classes))
    one_hot <- (targets$unsqueeze(-1L) == classes)$to(dtype = w$dtype)  # (B, N, T)

    probs <- torch::torch_matmul(
      w, one_hot$unsqueeze(2L)$expand(c(B, H, n_ctx, self$max_num_classes))
    )$mean(dim = 2L)                                         # (B, M, T)
    torch::torch_log(torch::torch_clamp(probs, min = 1e-5) + 3e-5)
  }
)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

# The signed indicator channel: -2 for NaN, +2 for +Inf, +4 for -Inf.
# @keywords internal
tabpfn3_nan_inf_indicator <- function(x) {
  is_inf <- torch::torch_isinf(x)
  pos <- is_inf$logical_and(torch::torch_sign(x) == 1)
  neg <- is_inf$logical_and(torch::torch_sign(x) == -1)
  torch::torch_isnan(x)$to(dtype = x$dtype) * TABPFN3_NAN_INDICATOR +
    pos$to(dtype = x$dtype) * TABPFN3_POS_INF_INDICATOR +
    neg$to(dtype = x$dtype) * TABPFN3_NEG_INF_INDICATOR
}

# Everything the training rows fix about the input scale: the per-feature
# mean (which doubles as the imputation value) and standard deviation.
# Split out because a cached prediction has to reuse them rather than
# refit on test rows.
#
# `x` is `(Ri, B, C)`. Reuses v2.6's NaN-aware reductions, which have the
# same semantics as `tabpfn.preprocessing.torch.ops`.
# @keywords internal
tabpfn3_fit_scaler <- function(x, n_train) {
  fit_rows <- if (n_train > 0L && n_train < x$size(1)) x[1:n_train, , ] else x
  mean <- tabpfn26_nanmean(fit_rows)
  std  <- tabpfn26_nanstd(fit_rows)
  std  <- torch::torch_where(std == 0, torch::torch_ones_like(std), std)
  if (fit_rows$size(1) == 1L) std <- torch::torch_ones_like(std)
  list(mean = mean, std = std)
}


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

#' Everything the training rows contribute to a prediction
#'
#' TabPFN has no weights to fit, so "training" is conditioning: the
#' training rows sit in the context of every forward pass. That makes
#' prediction cost grow with the training set, and grow again for every
#' batch of test rows, since the context is rebuilt each time.
#'
#' In v3 it does not have to. The training rows reach a test row through
#' exactly four channels, and all four are functions of the training rows
#' alone:
#'
#' * the scaler statistics fitted on them, which are also the values used
#'   to impute NaN and Inf;
#' * the inducing summary of each distribution-embedder block, `n_ind`
#'   vectors per column;
#' * the key/value projections of each ICL block, one head wide because a
#'   test row attends to one head;
#' * the post-norm training row embeddings, which the classifier's
#'   retrieval decoder reads.
#'
#' Unlike v2.6's cache, this one changes nothing: v2.6 fits its
#' constant-column and informative-feature masks over training *and* test
#' rows in an ordinary pass, so caching them shifts the prediction. v3
#' fits nothing on test rows, so a cached prediction and an uncached one
#' are the same computation.
#'
#' @param kv List of `list(key, value)`, one per ICL block.
#' @param scaler Fitted `list(mean, std)`.
#' @param inducing_hidden List of inducing summaries, one per
#'   distribution-embedder block.
#' @param train_embeddings `(B, n_train, D)` post-norm row embeddings.
#' @param y_train `(B, n_train)` ordinal targets, for the classifier's
#'   decoder.
#' @param n_train Number of training rows it was built from.
#' @keywords internal
tabpfn3_kv_cache <- function(kv, scaler, inducing_hidden, train_embeddings,
                             y_train, n_train) {
  structure(
    list(kv = kv, scaler = scaler, inducing_hidden = inducing_hidden,
         train_embeddings = train_embeddings, y_train = y_train,
         n_train = as.integer(n_train)),
    class = "tabpfn3_kv_cache"
  )
}

#' @export
print.tabpfn3_kv_cache <- function(x, ...) {
  bytes <- function(t) prod(as.numeric(t$size())) * 4
  n_kv <- sum(vapply(x$kv, function(e) sum(vapply(e, bytes, numeric(1))),
                     numeric(1)))
  n_ind <- sum(vapply(x$inducing_hidden, bytes, numeric(1)))
  n_emb <- bytes(x$train_embeddings)
  cli::cli_text("{.strong TabPFN v3 KV cache}")
  cli::cli_bullets(c(
    "*" = "built from {.val {x$n_train}} training row{?s}",
    "*" = "{length(x$kv)} ICL block{?s}, {round(n_kv / 1e6, 1)} MB of key/value projections",
    "*" = "{length(x$inducing_hidden)} inducing summar{?y/ies}, {round(n_ind / 1e6, 1)} MB",
    "*" = "{round(n_emb / 1e6, 1)} MB of training row embeddings"
  ))
  invisible(x)
}


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------

#' TabPFN v3 network
#'
#' Mirrors `TabPFNV3.forward` in the Python reference. The row and column
#' chunking of `_stages_0_to_2` is not reproduced: it exists to bound peak
#' memory and, because a distribution-embedder block only ever attends to
#' training rows, computes exactly what the unchunked path does. The
#' within-sublayer chunking that *is* reproduced is
#' `save_peak_memory_factor`, which is likewise bit-identical.
#'
#' @param config Parsed `config.json` with `arch == "tabpfn_v3"`.
#' @keywords internal
tabpfn_v3_transformer <- torch::nn_module(
  "TabPFNv3",

  initialize = function(config) {
    E <- as.integer(config$embed_dim %||% 128L)
    n_cls <- as.integer(config$feat_agg_num_cls_tokens %||% 4L)
    ff <- as.integer(config$ff_factor %||% 2L)
    self$embed_dim  <- E
    self$icl_emsize <- E * n_cls
    self$head <- config$head
    self$task_type <- if (identical(config$head, "classifier"))
      "multiclass" else "regression"
    self$feature_group_size <- as.integer(config$feature_group_size %||% 3L)
    self$use_nan_indicators <- isTRUE(config$use_nan_indicators %||% TRUE)
    self$max_num_classes <- as.integer(config$max_num_classes %||% 0L)
    self$n_out <- as.integer(
      config$n_out %||% (if (identical(self$head, "classifier"))
        self$max_num_classes else (config$num_buckets %||% 5000L))
    )
    ss_hidden <- as.integer(config$softmax_scaling_mlp_hidden_dim %||% 64L)

    # --- Stage 0: one linear over a column's group of values, plus their
    # NaN/Inf indicators when the checkpoint was trained with them.
    in_features <- self$feature_group_size
    if (self$use_nan_indicators) in_features <- in_features * 2L
    self$x_embed <- torch::nn_linear(in_features, E)

    # --- The target enters twice: once per column in stage 1, once per
    # row in stage 3. Classification embeds the ordinal label directly;
    # regression projects the scalar.
    if (identical(self$task_type, "multiclass")) {
      self$col_y_encoder <- tabpfn3_class_embedding(self$max_num_classes, E)
      self$icl_y_encoder <- tabpfn3_class_embedding(self$max_num_classes,
                                                    self$icl_emsize)
    } else {
      self$col_y_encoder <- torch::nn_linear(1L, E)
      self$icl_y_encoder <- torch::nn_linear(1L, self$icl_emsize)
    }

    # --- Stage 1.
    dist_heads <- as.integer(config$dist_embed_num_heads %||% 8L)
    self$feature_distribution_embedder <- tabpfn3_feature_distribution_embedder(
      emsize = E, nhead = dist_heads,
      num_inducing_points = as.integer(config$dist_embed_num_inducing_points %||% 128L),
      dim_feedforward = E * ff,
      num_layers = as.integer(config$dist_embed_num_blocks %||% 3L),
      softmax_scaling_layer_factory = function() {
        tabpfn3_softmax_scaling_mlp(
          num_heads = dist_heads, head_dim = as.integer(E / dist_heads),
          n_hidden = ss_hidden
        )
      }
    )

    # --- Stage 2.
    self$column_aggregator <- tabpfn3_column_aggregator(
      emsize = E, nhead = as.integer(config$feat_agg_num_heads %||% 8L),
      num_layers = as.integer(config$feat_agg_num_blocks %||% 3L),
      dim_feedforward = E * ff, num_cls_tokens = n_cls,
      use_rope = isTRUE(config$use_rope %||% TRUE),
      rope_base = as.numeric(config$feat_agg_rope_base %||% 100000)
    )

    # --- Stage 3.
    icl_heads <- as.integer(config$icl_num_heads %||% 8L)
    n_kv      <- config$icl_num_kv_heads
    n_kv_test <- config$icl_num_kv_heads_test
    self$icl_blocks <- torch::nn_module_list(
      lapply(seq_len(as.integer(config$nlayers %||% 24L)), function(i) {
        tabpfn3_icl_block(
          emsize = self$icl_emsize, nhead = icl_heads,
          dim_feedforward = self$icl_emsize * ff,
          num_kv_heads = if (is.null(n_kv)) NULL else as.integer(n_kv),
          num_kv_heads_test = if (is.null(n_kv_test)) NULL else as.integer(n_kv_test),
          softmax_scaling_layer = tabpfn3_softmax_scaling_mlp(
            num_heads = icl_heads,
            head_dim = as.integer(self$icl_emsize / icl_heads),
            n_hidden = ss_hidden
          )
        )
      })
    )
    self$output_norm <- rms_norm(self$icl_emsize, eps = TABPFN3_F32_EPS)

    # --- Decoder.
    if (identical(self$task_type, "multiclass")) {
      self$many_class_decoder <- tabpfn3_many_class_decoder(
        max_num_classes = self$max_num_classes, input_size = self$icl_emsize,
        head_dim  = as.integer(config$decoder_head_dim %||% 64L),
        num_heads = as.integer(config$decoder_num_heads %||% 6L),
        softmax_scaling_layer = if (isTRUE(config$decoder_use_softmax_scaling)) {
          tabpfn3_softmax_scaling_mlp(
            num_heads = as.integer(config$decoder_num_heads %||% 6L),
            head_dim  = as.integer(config$decoder_head_dim %||% 64L),
            n_hidden  = ss_hidden
          )
        } else NULL
      )
    } else {
      self$output_projection <- torch::nn_sequential(
        torch::nn_linear(self$icl_emsize, self$icl_emsize * ff),
        torch::nn_gelu(),
        torch::nn_linear(self$icl_emsize * ff, self$n_out)
      )
    }

    # The bar-distribution borders are registered on both heads in the
    # reference, so a classifier checkpoint carries them too and the
    # strict loader needs somewhere to put them. Only the regressor reads
    # them; see `tabpfn3_regressor()`.
    n_bins <- as.integer(config$n_bar_bins %||% config$num_buckets %||% 5000L)
    self$register_buffer("regression_borders", torch::torch_zeros(n_bins + 1L))

    # The reference's own stage-0-2 chunk sizes, which it applies by
    # default on v3 and nowhere else. Carried on the module so a caller
    # that says nothing gets what the Python estimator would do, and an
    # older `config.json` written before the converter emitted them still
    # lands on the same numbers.
    self$inference_row_chunk_size <-
      as.integer(config$inference_row_chunk_size %||% 2048L)
    self$inference_col_chunk_size <-
      as.integer(config$inference_col_chunk_size %||% 4L)

    # Read by the shared predictor helpers. There is no feature positional
    # embedding table to hand in: stage 2's RoPE does that job.
    self$needs_column_embeddings <- FALSE
    self$supports_kv_cache <- TRUE
    self$supports_chunked_eval <- TRUE
    self$supports_stage_chunking <- TRUE
  },

  # @param x_train `(B, n_train, F)`; @param y_train `(B, n_train)`;
  #   @param x_test `(B, n_test, F)`. With `kv_cache` supplied, `x_train`
  #   and `y_train` are ignored -- everything the training rows contribute
  #   is already in the cache -- and `x_test` holds the rows to predict.
  # @param return_kv_cache Build and return one. `x_test` may be empty,
  #   which is the usual way to build from training rows alone.
  # @param save_peak_memory_factor Chunk count for [chunked_evaluate()].
  # @param row_chunk_size,col_chunk_size Stage-0-2 chunking; see
  #   `stages_0_to_2()`. Left alone they take the checkpoint's own values,
  #   which is what the reference does by default on this architecture;
  #   `NULL` runs every row in one pass.
  forward = function(x_train, y_train, x_test, kv_cache = NULL,
                     return_kv_cache = FALSE, save_peak_memory_factor = NULL,
                     row_chunk_size = NA_integer_, col_chunk_size = NA_integer_) {
    row_chunk_size <- .tabpfn3_chunk_arg(row_chunk_size,
                                         self$inference_row_chunk_size)
    col_chunk_size <- .tabpfn3_chunk_arg(col_chunk_size,
                                         self$inference_col_chunk_size)
    if (!is.null(kv_cache)) {
      return(self$forward_cached(
        x_test, kv_cache, save_peak_memory_factor = save_peak_memory_factor,
        row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size
      ))
    }
    B <- x_train$size(1)
    n_train <- x_train$size(2)
    n_test  <- x_test$size(2)
    if (B != 1L) {
      cli::cli_abort(c(
        "The TabPFN v3 backend runs one dataset at a time.",
        i = "Got a batch of {B}; call it once per dataset."
      ))
    }
    if (n_train == 0L) {
      cli::cli_abort("TabPFN v3 needs at least one training row to condition on.")
    }
    if (identical(self$task_type, "multiclass") &&
        as.logical((y_train > (self$n_out - 1L))$any()$cpu())) {
      cli::cli_abort(c(
        "Target out of range for a {self$n_out}-class head.",
        i = "Labels must be ordinal-encoded in {.val {0}}..{.val {self$n_out - 1L}}."
      ))
    }

    # The reference works in (rows, batch, columns) until stage 0 ends.
    x_RiBC <- torch::torch_cat(list(x_train, x_test), dim = 2L)$
      transpose(1L, 2L)$contiguous()
    dump_if_enabled("input_x_raw", x_RiBC)

    y_BN <- self$prepare_targets(y_train, n_train)
    s02 <- self$stages_0_to_2(
      x_RiBC, y_BN, n_train,
      return_hidden = isTRUE(return_kv_cache),
      row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size,
      save_peak_memory_factor = save_peak_memory_factor
    )

    # (B, Ri, n_cls, E) -> (B, Ri, n_cls * E): the CLS tokens concatenated
    # are the row embedding.
    x <- s02$state$flatten(start_dim = 3L)
    x <- .tabpfn3_add_to_train_rows(x, self$embed_icl_targets(y_BN), n_train)
    dump_if_enabled("icl_input", x)

    icl <- self$run_icl(x, single_eval_pos = n_train,
                        return_kv = isTRUE(return_kv_cache),
                        save_peak_memory_factor = save_peak_memory_factor)
    x <- self$output_norm(icl$state)
    dump_if_enabled("icl_out", x)

    train_emb <- x[, 1:n_train, ]
    test_emb  <- if (n_test > 0L) x[, (n_train + 1L):(n_train + n_test), ] else NULL

    res <- list(
      logits = if (n_test > 0L) self$decode(train_emb, test_emb, y_BN) else NULL,
      test_hidden = test_emb, train_hidden = train_emb
    )
    if (isTRUE(return_kv_cache)) {
      res$kv_cache <- tabpfn3_kv_cache(
        kv = icl$kv, scaler = s02$scaler, inducing_hidden = s02$hidden,
        train_embeddings = train_emb$detach(), y_train = y_BN$detach(),
        n_train = n_train
      )
    }
    res
  },

  #' Predict test rows against a prebuilt cache.
  #' @keywords internal
  forward_cached = function(x_test, cache, save_peak_memory_factor = NULL,
                            row_chunk_size = NA_integer_,
                            col_chunk_size = NA_integer_) {
    row_chunk_size <- .tabpfn3_chunk_arg(row_chunk_size,
                                         self$inference_row_chunk_size)
    col_chunk_size <- .tabpfn3_chunk_arg(col_chunk_size,
                                         self$inference_col_chunk_size)
    B <- x_test$size(1); n_test <- x_test$size(2)
    if (B != 1L) {
      cli::cli_abort("The TabPFN v3 backend runs one dataset at a time.")
    }
    x_RiBC <- x_test$transpose(1L, 2L)$contiguous()

    # `n_train = 0` here is not "no training rows": the scaler carries
    # their statistics, and it is also what imputation draws on. Every
    # row is a test row, so there is no target to embed and the inducing
    # summaries come from the cache rather than from this pass.
    s02 <- self$stages_0_to_2(
      x_RiBC, y_BN = NULL, n_train = 0L, scaler = cache$scaler,
      cached_hidden = cache$inducing_hidden,
      row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size,
      save_peak_memory_factor = save_peak_memory_factor
    )
    x <- s02$state$flatten(start_dim = 3L)
    # No ICL target embedding: every row here is a test row.
    icl <- self$run_icl(x, single_eval_pos = 0L, cached_kv = cache$kv,
                        save_peak_memory_factor = save_peak_memory_factor)
    test_emb <- self$output_norm(icl$state)

    train_emb <- cache$train_embeddings$to(device = test_emb$device)
    list(
      logits = if (n_test > 0L)
        self$decode(train_emb, test_emb,
                    cache$y_train$to(device = test_emb$device)) else NULL,
      test_hidden = test_emb, train_hidden = train_emb
    )
  },

  #' Stages 0 to 2, optionally a row chunk at a time
  #'
  #' Mirrors the reference's `_stages_0_to_2`, and carries the same three
  #' paths in one function: consuming a cache, chunked, and whole.
  #'
  #' The point of the chunking is what it *avoids*: `(B, Ri, C, E)`, which
  #' at 128 embedding channels is 51 KB per row per 100 columns and is the
  #' term that decides how large a table fits. Between preprocessing and
  #' the cell embedding the state is only `(B, Ri, C, G)`, `G` = 6, so the
  #' loop starts there, and what survives it is `(B, Ri, n_cls, E)` -- a
  #' quarter of a column's width rather than every column's. Neither
  #' bound involves the row count, which is why this changes the shape of
  #' the curve rather than its constant.
  #'
  #' `save_peak_memory_factor` is the within-sublayer counterpart and
  #' composes with this; it cannot substitute for it, because the tensor
  #' it chunks is the temporaries and not the state they are made from.
  #'
  #' @param x_RiBC `(Ri, B, C)` raw input, train rows first.
  #' @param y_BN `(B, N)` cleaned training targets, or `NULL` when there
  #'   are none (the cache-consuming path).
  #' @param n_train Leading rows of `x_RiBC` that are training rows.
  #' @param scaler Fitted scaler to reuse, or `NULL` to fit one.
  #' @param cached_hidden Per-block inducing summaries from a cache.
  #' @param return_hidden Return them, for building one.
  #' @param row_chunk_size Rows per pass through stages 0-2, or `NULL` for
  #'   all at once. Ignored when it is not smaller than the row count.
  #' @param col_chunk_size Columns per chunk of the inducing-summary
  #'   pre-pass, or `NULL` for all at once.
  #' @return `list(state, hidden, scaler)`, `state` being
  #'   `(B, Ri, n_cls, E)`.
  #' @keywords internal
  stages_0_to_2 = function(x_RiBC, y_BN, n_train, scaler = NULL,
                           cached_hidden = NULL, return_hidden = FALSE,
                           row_chunk_size = NULL, col_chunk_size = NULL,
                           save_peak_memory_factor = NULL) {
    pre <- self$preprocess(x_RiBC, n_train = n_train, scaler = scaler)
    grouped <- self$group_features(pre$x_BRiC, pre$indicators)
    n_rows <- grouped$size(2)
    y_col <- if (!is.null(y_BN) && n_train > 0L)
      self$embed_col_targets(y_BN) else NULL

    size <- suppressWarnings(as.integer(row_chunk_size %||% NA_integer_))
    use_chunks <- !is.na(size) && size >= 1L && size < n_rows
    if (!use_chunks) size <- n_rows

    # The summaries have to exist before the first row chunk can run.
    # Consuming a cache means they already do.
    hidden <- cached_hidden
    if (use_chunks && is.null(hidden)) {
      hidden <- self$all_inducing_hidden(grouped, y_col, n_train,
                                         col_chunk_size)
    }
    # "Full" is the path that owns its own summaries: one pass over every
    # row, computing them as it goes. Only there can a block be asked to
    # hand them back, and only there does chunking a sublayer of it apply
    # -- the other two paths take the branch that runs whole.
    full <- !use_chunks && is.null(hidden)
    dump <- !use_chunks

    parts <- vector("list", length(seq(1L, n_rows, by = size)))
    own_hidden <- NULL
    j <- 0L
    for (s in seq(1L, n_rows, by = size)) {
      j <- j + 1L
      len <- min(s + size - 1L, n_rows) - s + 1L
      x_emb <- self$embed_cells(grouped$narrow(2L, s, len))
      if (dump) dump_if_enabled("embedded_x", x_emb)

      # How many rows of *this* chunk are training rows. The train/test
      # boundary falls inside a chunk in general, and a chunk past it has
      # none at all.
      n_tr_chunk <- max(0L, min(n_train - (s - 1L), len))
      if (!is.null(y_col) && n_tr_chunk > 0L) {
        x_emb <- .tabpfn3_add_to_train_rows(
          x_emb, y_col$narrow(2L, s, n_tr_chunk)$unsqueeze(3L), n_tr_chunk)
      }

      dist <- self$feature_distribution_embedder(
        x_emb, n_train = n_tr_chunk, cached_hidden = hidden,
        return_hidden = isTRUE(return_hidden) && full,
        save_peak_memory_factor = if (full) save_peak_memory_factor else NULL
      )
      if (dump) dump_if_enabled("dist_embedder_out", dist$state)
      if (!is.null(dist$hidden)) own_hidden <- dist$hidden

      parts[[j]] <- self$column_aggregator(
        dist$state, save_peak_memory_factor = save_peak_memory_factor
      )
      if (dump) dump_if_enabled("column_aggregator_out", parts[[j]])
      # A finished chunk leaves everything but `parts[[j]]` dead, and the
      # whole point of the loop is not to be holding it when the next one
      # allocates. Gated on the chunk's own embedded width rather than on
      # the part kept, which is 25x smaller.
      if (use_chunks) collect_between_layers(x_emb)
    }

    list(
      state = if (j == 1L) parts[[1L]] else torch::torch_cat(parts, dim = 2L),
      hidden = if (use_chunks) hidden else own_hidden,
      scaler = pre$scaler
    )
  },

  #' Run the ICL stack, optionally building or consuming a cache.
  #' @keywords internal
  run_icl = function(x, single_eval_pos, cached_kv = NULL, return_kv = FALSE,
                     save_peak_memory_factor = NULL) {
    n <- length(self$icl_blocks)
    kv <- if (isTRUE(return_kv)) vector("list", n) else NULL
    for (i in seq_len(n)) {
      res <- self$icl_blocks[[i]](
        x, single_eval_pos = single_eval_pos,
        cached_kv = if (is.null(cached_kv)) NULL else cached_kv[[i]],
        return_kv = return_kv,
        save_peak_memory_factor = save_peak_memory_factor
      )
      x <- res$state
      if (isTRUE(return_kv)) kv[[i]] <- res$kv
      # 24 blocks deep, this is where the package's largest single
      # accumulation is; see [collect_between_layers()].
      collect_between_layers(x)
    }
    list(state = x, kv = kv)
  },

  #' Turn training and test row embeddings into per-row outputs.
  #' @keywords internal
  decode = function(train_emb, test_emb, y_BN) {
    out <- if (identical(self$task_type, "multiclass")) {
      self$many_class_decoder(train_emb, test_emb, y_BN)
    } else {
      # The reference projects in (rows, batch, embedding) order; keep the
      # same layout so the matmul sees the same shapes.
      self$output_projection(test_emb$transpose(1L, 2L))$transpose(1L, 2L)
    }
    # `_nan_safe_output`: an all-NaN column, or a class no training row
    # carries, can leave a NaN here. The reference zeroes them rather than
    # letting one propagate into the softmax and take the row with it.
    out <- torch::torch_where(torch::torch_isnan(out),
                              torch::torch_zeros_like(out), out)
    dump_if_enabled("logits", out)
    out
  },

  #' Indicator capture, mean imputation, standard scaling.
  #'
  #' The order matters and is not the obvious one: indicators are read off
  #' the raw input, imputation replaces every non-finite cell with the
  #' training mean, and only then is the scaler fitted -- on the imputed
  #' rows, so its statistics stay finite even when the raw input carried
  #' infinities.
  #' @keywords internal
  preprocess = function(x_RiBC, n_train, scaler = NULL) {
    indicators <- if (self$use_nan_indicators) {
      tabpfn3_nan_inf_indicator(x_RiBC)$transpose(1L, 2L)
    } else NULL

    is_finite <- torch::torch_isfinite(x_RiBC)
    means <- if (!is.null(scaler)) scaler$mean else {
      fit <- if (n_train > 0L && n_train < x_RiBC$size(1))
        x_RiBC[1:n_train, , ] else x_RiBC
      # Ignoring Inf as well as NaN: an infinite cell is about to be
      # replaced, so it must not set the value it is replaced with.
      tabpfn26_nanmean(fit, include_inf = TRUE)
    }
    x <- torch::torch_where(is_finite, x_RiBC,
                            means$unsqueeze(1L)$expand_as(x_RiBC))
    if (is.null(scaler)) scaler <- tabpfn3_fit_scaler(x, n_train)

    x <- (x - scaler$mean$unsqueeze(1L)) /
      (scaler$std$unsqueeze(1L) + TABPFN3_F32_EPS)
    x <- torch::torch_clamp(x, min = -100, max = 100)
    dump_if_enabled("preproc_x", x)
    list(x_BRiC = x$transpose(1L, 2L), indicators = indicators, scaler = scaler)
  },

  #' Group the columns, without embedding them.
  #'
  #' Every column becomes a token carrying the values of the columns 1, 2
  #' and 4 places to its right, wrapping around. Not its own value: the
  #' shifts are `2^i` for `i = 0..group_size-1`, and none of them is zero.
  #' The column count is preserved, unlike v2's packing, which divided it
  #' by the group size.
  #'
  #' Kept separate from `embed_cells()` because this is where the pipeline
  #' can be cut: the grouped tensor is `(B, Ri, C, G)` with `G` 6, and the
  #' embedded one is `(B, Ri, C, E)` with `E` 128, so everything that runs
  #' a chunk of rows at a time has to start on this side of the boundary.
  #' The reference splits it the same way (`_group_features` vs the
  #' `x_embed` call inside `_process_row_chunk`).
  #' @keywords internal
  group_features = function(x_BRiC, indicators) {
    g <- self$feature_group_size
    roll_stack <- function(z) {
      torch::torch_stack(
        lapply(0:(g - 1L), function(i) {
          torch::torch_roll(z, shifts = -(2L^i), dims = 3L)
        }),
        dim = 4L
      )
    }
    grouped <- roll_stack(x_BRiC)
    if (!is.null(indicators)) {
      grouped <- torch::torch_cat(list(grouped, roll_stack(indicators)), dim = 4L)
    }
    dump_if_enabled("x_grouped", grouped)
    grouped
  },

  #' Embed each column group into one token.
  #'
  #' @param grouped `(B, Ri, C, G)` from `group_features()`, or a slice of
  #'   one along the row axis.
  #' @keywords internal
  embed_cells = function(grouped) {
    self$x_embed(grouped)
  },

  #' Every distribution-embedder block's inducing summary, in column chunks
  #'
  #' Mirrors the reference's `_compute_all_inducing_hidden` /
  #' `_process_col_chunk`. This is the half of stage 1 that a row-chunked
  #' pass cannot do for itself: block `l`'s inducing summary is a function
  #' of the *training* rows' state after blocks `1..l-1`, so it has to
  #' exist before any row chunk starts. Chunking it along columns instead
  #' is free -- every column is embedded on its own -- and holds
  #' `(B * Cj, n_train, E)` rather than `(B * C, n_train, E)`.
  #'
  #' What comes back is small whatever the table: `num_inducing_points`
  #' rows per column per block, so 3 x 100 x 128 x 128 floats is 20 MB at
  #' 100 features. It is the same object [tabpfn3_kv_cache()] stores.
  #'
  #' @param grouped `(B, Ri, C, G)` for train and test rows together; only
  #'   the leading `n_train` are read.
  #' @param y_col `(B, N, E)` target embedding, added to every column of a
  #'   training row, or `NULL`.
  #' @param col_chunk_size Columns per chunk, or `NULL` for all at once.
  #' @return One `(B * C, num_inducing_points, E)` tensor per block, in the
  #'   column-major order [tabpfn3_induced_self_attention_block()] folds
  #'   its columns into the batch with.
  #' @keywords internal
  all_inducing_hidden = function(grouped, y_col, n_train,
                                 col_chunk_size = NULL) {
    if (n_train <= 0L) {
      cli::cli_abort("The inducing summaries are built from training rows.")
    }
    layers <- self$feature_distribution_embedder$layers
    n_blocks <- length(layers)
    B <- grouped$size(1); C <- grouped$size(3)
    train <- if (n_train >= grouped$size(2)) grouped
             else grouped$narrow(2L, 1L, n_train)
    cc <- if (is.null(col_chunk_size)) C
          else max(1L, min(as.integer(col_chunk_size), C))

    parts <- replicate(n_blocks, list(), simplify = FALSE)
    for (s in seq(1L, C, by = cc)) {
      cj <- min(s + cc - 1L, C) - s + 1L
      x_emb <- self$embed_cells(train$narrow(3L, s, cj))
      if (!is.null(y_col)) x_emb <- x_emb + y_col$unsqueeze(3L)
      E <- x_emb$size(4)
      # Same fold as the block's own forward: (B, N, Cj, E) -> (B, Cj, N,
      # E) -> (B * Cj, N, E), so `b` varies slowest and the chunks
      # concatenate back into the whole in the right order.
      x_flat <- x_emb$transpose(2L, 3L)$contiguous()$
        reshape(c(B * cj, n_train, E))
      for (i in seq_len(n_blocks)) {
        blk <- layers[[i]]
        hidden <- blk$inducing_hidden(x_flat, n_train)
        parts[[i]][[length(parts[[i]]) + 1L]] <-
          hidden$reshape(c(B, cj, -1L, E))
        # The next block's summary is taken from this one's output on the
        # training rows, which is exactly what the unchunked path feeds
        # forward. The last block's is never needed.
        if (i < n_blocks) x_flat <- blk$cross_attn_block2(x_flat, hidden)
      }
    }
    lapply(parts, function(p) {
      h <- if (length(p) == 1L) p[[1L]] else torch::torch_cat(p, dim = 2L)
      h$flatten(start_dim = 1L, end_dim = 2L)$detach()
    })
  },

  #' Clean the training targets and put them in `(B, N)`.
  #' @keywords internal
  prepare_targets = function(y_train, n_train) {
    B <- y_train$size(1)
    y <- y_train$reshape(c(B, n_train, 1L))$transpose(1L, 2L)$contiguous()
    is_finite <- torch::torch_isfinite(y)
    means <- tabpfn26_nanmean(y, include_inf = TRUE)
    y <- torch::torch_where(is_finite, y, means$unsqueeze(1L)$expand_as(y))
    if (identical(self$task_type, "multiclass")) {
      # An imputed class label is rounded up, and only the imputed
      # positions are -- the reference keeps this for backwards
      # compatibility rather than because a mean label means anything.
      y <- torch::torch_where(is_finite, y, y$ceil())
    }
    y$squeeze(-1L)$transpose(1L, 2L)
  },

  #' @keywords internal
  embed_col_targets = function(y_BN) {
    if (identical(self$task_type, "multiclass")) return(self$col_y_encoder(y_BN))
    self$col_y_encoder(y_BN$unsqueeze(-1L))
  },

  #' @keywords internal
  embed_icl_targets = function(y_BN) {
    if (identical(self$task_type, "multiclass")) return(self$icl_y_encoder(y_BN))
    self$icl_y_encoder(y_BN$unsqueeze(-1L))
  }
)


#' Learned per-class embedding
#'
#' Wraps `nn_embedding` so the parameter lands at `<name>.embedding.weight`,
#' where the checkpoint puts it. The reference initialises the rows
#' orthogonally; that only matters for training, so it is not reproduced.
#'
#' Labels arrive 0-based, as the reference's ordinal encoding produces
#' them, and R torch's `nn_embedding` indexes from one.
#' @keywords internal
tabpfn3_class_embedding <- torch::nn_module(
  "TabPFN3ClassEmbedding",

  initialize = function(num_classes, embed_dim) {
    self$embedding <- torch::nn_embedding(as.integer(num_classes),
                                          as.integer(embed_dim))
  },

  forward = function(y) {
    self$embedding(y$to(dtype = torch::torch_long()) + 1L)
  }
)


# Add a per-row term to the leading `n_train` rows of `x`, leaving the
# test rows alone. Built by concatenation rather than in-place assignment:
# a slice assignment on a tensor that may be a view is one of the few
# places R torch and PyTorch disagree about aliasing.
# @keywords internal
.tabpfn3_add_to_train_rows <- function(x, term, n_train) {
  Ri <- x$size(2)
  if (n_train >= Ri) return(x + term)
  if (n_train == 0L) return(x)
  torch::torch_cat(
    list(.tabpfn3_row_slice(x, 1L, n_train) + term,
         .tabpfn3_row_slice(x, n_train + 1L, Ri)),
    dim = 2L
  )
}

# Resolve a stage-chunking argument against the checkpoint's own value.
#
# Three states, because there are three things a caller can mean. `NA`
# (the default) is "whatever the checkpoint says", which for every
# released v3 is the reference's 2048/4 and is what the Python estimator
# would do. `NULL` is "off", which is how the parity harness asks for the
# unchunked pass. An integer is an integer.
# @keywords internal
.tabpfn3_chunk_arg <- function(x, default) {
  if (is.null(x)) return(NULL)
  if (length(x) == 1L && is.na(x)) return(as.integer(default))
  as.integer(x)
}

# Slice rows `from:to` out of a 3-D or 4-D tensor whose second axis is the
# row axis.
# @keywords internal
.tabpfn3_row_slice <- function(x, from, to) {
  if (x$dim() == 4L) x[, from:to, , ] else x[, from:to, ]
}


# ---------------------------------------------------------------------------
# Backend hooks
# ---------------------------------------------------------------------------

#' @keywords internal
tabpfn3_build <- function(config, task) {
  cli::cli_alert_info(
    "Building tabpfn_v3 ({.val {config$head}}, \\
     {config$nlayers} ICL layers, emb={config$embed_dim})..."
  )
  tabpfn_v3_transformer(config)
}

#' @keywords internal
tabpfn3_detect <- function(config) {
  identical(config$arch, "tabpfn_v3")
}


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

#' Stage list for the TabPFN v3 diagram
#'
#' v3 abandons the single stack that attends both ways and splits the
#' work into three: summarise each column's distribution over rows,
#' aggregate a row's columns into a fixed-width vector, then run a deep
#' in-context stack over rows at that wider width. That third stage is
#' where nearly all the parameters are, and the reason its width is
#' `4 x embed_dim` is the four CLS tokens the aggregator reads out.
#'
#' @param config The checkpoint's parsed `config.json`.
#' @param task `"classification"` or `"regression"`.
#' @keywords internal
tabpfn3_describe <- function(config, task) {
  E     <- as.integer(config$embed_dim %||% 128L)
  n_cls <- as.integer(config$feat_agg_num_cls_tokens %||% 4L)
  D     <- E * n_cls
  ff    <- as.integer(config$ff_factor %||% 2L)
  grp   <- as.integer(config$feature_group_size %||% 3L)
  nan   <- isTRUE(config$use_nan_indicators %||% TRUE)
  L     <- as.integer(config$nlayers %||% 24L)
  clf   <- identical(config$head, "classifier")
  n_ind <- as.integer(config$dist_embed_num_inducing_points %||% 128L)
  n_dst <- as.integer(config$dist_embed_num_blocks %||% 3L)
  n_agg <- as.integer(config$feat_agg_num_blocks %||% 3L)
  h_icl <- as.integer(config$icl_num_heads %||% 8L)
  kv_t  <- as.integer(config$icl_num_kv_heads_test %||% h_icl)
  n_out <- as.integer(config$n_out %||% (if (clf)
    (config$max_num_classes %||% 160L) else (config$num_buckets %||% 5000L)))

  stages <- list(
    arch_input_stage(),
    arch_stage(
      "x_embed", "Cell embedder", kind = "embed", group = "Embed",
      detail = sprintf("Linear(%d -> %d): %d values%s", grp * (1L + nan), E, grp,
                       if (nan) " + their NaN/Inf flags" else ""),
      shape = "(B, n, F, E)", prefix = "x_embed"),
    arch_stage(
      "col_y", "Target encoder (per column)", kind = "embed", group = "Embed",
      detail = if (clf) sprintf("class embedding -> %d", E)
               else sprintf("Linear(1 -> %d)", E),
      prefix = "col_y_encoder"),
    arch_stage(
      "dist", "Feature-distribution embedder", kind = "attention",
      group = "Stage 1 - per column", repeats = n_dst, axis = "rows",
      detail = sprintf("induced self-attention, %d inducing vectors; keys are training rows only",
                       n_ind),
      shape = "(B, n, F, E)", prefix = "feature_distribution_embedder",
      children = list(
        arch_stage("ind", "Inducing vectors attend to the rows",
                   kind = "attention", axis = "rows",
                   prefix = "feature_distribution_embedder.layers.0.cross_attn_block1"),
        arch_stage("back", "Rows attend back to the inducing vectors",
                   kind = "attention", axis = "inducing",
                   prefix = "feature_distribution_embedder.layers.0.cross_attn_block2"),
        arch_stage("iv", "Learned inducing vectors", kind = "embed",
                   prefix = "feature_distribution_embedder.layers.0.inducing_vectors")
      )),
    arch_stage(
      "agg", "Column aggregator", kind = "attention",
      group = "Stage 2 - per row", repeats = n_agg, axis = "columns",
      detail = sprintf("pre-norm self-attention with RoPE; last block reads out %d CLS tokens",
                       n_cls),
      shape = sprintf("(B, n, %d)", D), prefix = "column_aggregator",
      children = list(
        arch_stage("attn", "Attention over a row's columns", kind = "attention",
                   axis = "columns",
                   prefix = "column_aggregator.blocks.0"),
        arch_stage("rope", "RoPE frequencies (learned)", kind = "embed",
                   prefix = "column_aggregator.rope"),
        arch_stage("cls", "CLS tokens", kind = "embed",
                   prefix = "column_aggregator.cls_tokens"),
        arch_stage("oln", "RMSNorm", kind = "norm",
                   prefix = "column_aggregator.out_ln")
      )),
    arch_stage(
      "icl_y", "Target encoder (per row)", kind = "embed",
      group = "Stage 3 - in-context",
      detail = if (clf) sprintf("class embedding -> %d", D)
               else sprintf("Linear(1 -> %d)", D),
      prefix = "icl_y_encoder"),
    arch_stage(
      "icl", "In-context learning block", kind = "attention",
      group = "Stage 3 - in-context", repeats = L, axis = "rows",
      detail = sprintf("pre-norm; %d heads of %d; test rows read %d kv head%s; MLP %d -> %d",
                       h_icl, as.integer(D / h_icl), kv_t,
                       if (kv_t == 1L) "" else "s", D, D * ff),
      shape = sprintf("(B, n, %d)", D), prefix = "icl_blocks",
      children = list(
        arch_stage("iattn", "Attention over rows, training-only keys",
                   kind = "attention", axis = "rows",
                   prefix = "icl_blocks.0.icl_attention"),
        arch_stage("iln", "RMSNorm", kind = "norm",
                   prefix = c("icl_blocks.0.layernorm",
                              "icl_blocks.0.layernorm_mlp")),
        arch_stage("imlp", "MLP, GELU", kind = "ffn",
                   prefix = "icl_blocks.0.mlp")
      )),
    arch_stage(
      "onorm", "Output RMSNorm", kind = "norm", group = "Decode",
      prefix = "output_norm"),
    if (clf)
      arch_stage(
        "dec", "Many-class decoder", kind = "decode", group = "Decode",
        detail = sprintf("attention against %d class queries, %d heads of %d",
                         n_out, as.integer(config$decoder_num_heads %||% 6L),
                         as.integer(config$decoder_head_dim %||% 64L)),
        shape = sprintf("(n_test, %d)", n_out), prefix = "many_class_decoder")
    else
      arch_stage(
        "dec", "Output projection", kind = "decode", group = "Decode",
        detail = sprintf("%d -> %d -> %d, GELU", D, D * ff, n_out),
        shape = sprintf("(n_test, %d)", n_out), prefix = "output_projection"),
    if (clf)
      arch_output_stage("Class logits",
                        detail = sprintf("up to %d classes", n_out))
    else
      arch_output_stage("Bar distribution",
                        detail = sprintf("%d bins over the scaled target", n_out))
  )

  list(
    title = sprintf("TabPFN v3 - %s", if (clf) "classifier" else "regressor"),
    subtitle = "Prior-Labs  -  three stages: per column, per row, in context",
    facts = c(
      "ICL layers"          = L,
      "Column-stage blocks" = n_dst,
      "Aggregator blocks"   = n_agg,
      "Embedding width"     = E,
      "ICL width"           = sprintf("%d (%d CLS x %d)", D, n_cls, E),
      "ICL heads"           = sprintf("%d x %d", h_icl, as.integer(D / h_icl)),
      "Inducing points"     = n_ind,
      "Features per group"  = grp,
      "Output width"        = n_out,
      "Normalisation"       = "RMSNorm, pre",
      "Attention scaling"   = "learned (softmax-scaling MLP)",
      "KV cache"            = "exact"
    ),
    symbols = c(arch_default_symbols(), D = sprintf("ICL width (%d)", D)),
    stages = Filter(Negate(is.null), stages)
  )
}


# ---------------------------------------------------------------------------
# Memory scaling
# ---------------------------------------------------------------------------

#' TabPFN v3's peak-memory shapes
#'
#' v3 left v2's per-feature transformer behind for the same three-stage
#' shape TabICL and TabFM use -- embed each feature's distribution
#' against inducing points, aggregate a row's features, then attend
#' across rows in context -- so it is estimated the same way, and it is
#' much the cheapest of the six at fold-sized tables.
#'
#' One caveat belongs in the estimate rather than in a footnote: the
#' reference's `_stages_0_to_2` row/column chunking is deliberately not
#' ported (see the README's known gaps), so this port's peak is *higher*
#' than the Python reference's on large tables. The preflight is the
#' compensating control, and says so in its suggestions.
#'
#' @inheritParams mitra_peak_terms
#' @keywords internal
tabpfn3_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  e <- as.numeric(config$embed_dim)
  n_cls <- as.numeric(config$feat_agg_num_cls_tokens %||% 4)
  l <- as.numeric(config$nlayers)
  icl_heads <- as.numeric(config$icl_num_heads %||% 8)
  n_est <- max(1, as.numeric(opts$n_estimators %||% 1))
  nq <- .resident_query(n_query, opts)

  # One token per *column*, not per group of them: v3 groups by rolling
  # and stacking, which preserves the column count where v2's packing
  # divided it. Getting this wrong understates stages 0-2 by the group
  # size and lets the fitted `act_copies` absorb the difference, which
  # hides it at one shape and is wrong at every other.
  g <- as.numeric(config$feature_group_size %||% 3)

  terms <- .icl_family_terms(
    n_context    = n_context,
    n_query      = nq,
    n_features   = n_features,
    embed_dim    = e,
    group_size   = 1,
    n_cls        = n_cls,
    col_heads    = as.numeric(config$dist_embed_num_heads %||% 8),
    row_heads    = as.numeric(config$feat_agg_num_heads %||% 8),
    icl_heads    = icl_heads,
    col_inducing = as.numeric(config$dist_embed_num_inducing_points %||% 128),
    icl_blocks   = l,
    kv_cache     = FALSE,
    n_estimators = n_est,
    # v3 is the only backend that bounds the row axis of stages 0-2.
    row_chunk    = .stage_row_chunk(opts, config),
    col_chunk    = .stage_col_chunk(opts, config),
    # The grouped input the row loop slices from: `feature_group_size`
    # values per column, doubled when the checkpoint carries NaN/Inf
    # indicators.
    group_channels = g * (if (isTRUE(config$use_nan_indicators %||% TRUE)) 2 else 1)
  )

  # v3's cached path keeps `icl_num_kv_heads_test` heads rather than all
  # of them (one, in the published checkpoints), so the cache is a
  # fraction of the width the other backends store -- plus the training
  # rows' embeddings, which it also holds on to.
  if (isTRUE(opts$kv_cache)) {
    d_icl <- e * n_cls
    kv_heads <- as.numeric(config$icl_num_kv_heads_test %||% icl_heads)
    head_dim <- d_icl / max(icl_heads, 1)
    terms$persistent <- n_est * (l * 2 * n_context * kv_heads * head_dim +
                                 n_context * d_icl)
  }
  terms
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#' @keywords internal
register_tabpfn3_backend <- function() {
  register_backend(
    name          = "tabpfn3",
    build         = tabpfn3_build,
    describe      = tabpfn3_describe,
    translate_key = identity,
    detect        = tabpfn3_detect,
    task_of       = tabpfn_task_of,
    # The v2 predictors serve v3 unchanged: the preprocessing pipeline,
    # the ensembling and the bar-distribution head are all shared, and
    # what does differ -- no column embedding table, borders on the module
    # rather than on a `criterion` -- is asked of the network rather than
    # branched on here.
    classifier    = tabpfn_classifier,
    regressor     = tabpfn_regressor,
    peak_terms    = tabpfn3_peak_terms,
    # NaN and Inf are first-class inputs: imputed with the training mean
    # and flagged in a dedicated indicator channel.
    handles_missing = TRUE,
    description   = "TabPFN v3 (Prior-Labs)",
    parity        = "tabpfn 8.2.0 (PyPI)"
  )
}
