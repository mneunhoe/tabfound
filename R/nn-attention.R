# Attention variants shared across backends.
#
# Tabular foundation models differ in how they store attention weights,
# and the storage layout dictates the matmul order, which in turn
# dictates float32 rounding. Each variant here reproduces one reference
# implementation's ordering exactly rather than normalising them to a
# common form -- that is what makes bit-exact parity achievable.
#
#   mha_fused_qkv  -- TabPFN: `_w_qkv` (3, H, D, E) and `_w_out` (H, D, E),
#                     no biases, optional multi-query for test rows.
#
# Further variants (standard q/k/v projections with per-head RMSNorm and
# learned softplus scaling for TabFM; scalable-softmax for TabICL) get
# added here as those backends land.

# ---------------------------------------------------------------------------
# The one place that calls SDPA
# ---------------------------------------------------------------------------

#' Scaled dot-product attention
#'
#' R torch registers `torch_scaled_dot_product_attention` but does not
#' export it, and every backend here wants it: the fused kernel is what
#' makes this package's float32 rounding match the reference's, since a
#' hand-written `softmax(QK'/sqrt(d)) V` accumulates differently.
#'
#' Reaching into another package's namespace is also a CRAN blocker, and
#' a hard dependency on an unexported function is a real fragility --
#' torch is free to rename it. So there is one call site instead of ten,
#' the lookup is resolved once and cached, and the package still works
#' when it is gone: `.sdpa_fallback()` is the same operation written in
#' public API, differing only in float rounding (~1e-7 relative on the
#' package's fixtures, checked in the tests).
#'
#' @param query,key,value `(B, H, L, D)` tensors.
#' @param attn_mask Optional additive or boolean mask.
#' @param dropout_p Kept for signature parity; always 0 here (inference).
#' @param is_causal Passed through.
#' @param scale Optional override for `1/sqrt(D)`.
#' @section Options:
#' `tabfound.sdpa` is `"auto"` (default: the fused kernel when torch has
#' it), `"torch"` (require it, error if absent) or `"r"` (always the
#' fallback -- what the agreement test runs).
#' @keywords internal
sdpa <- function(query, key, value, attn_mask = NULL, dropout_p = 0,
                 is_causal = FALSE, scale = NULL) {
  mode <- getOption("tabfound.sdpa", "auto")
  fn <- if (identical(mode, "r")) NULL else .torch_sdpa()
  if (is.null(fn)) {
    if (identical(mode, "torch")) {
      cli::cli_abort(c(
        "This build of {.pkg torch} has no \\
         {.fn torch_scaled_dot_product_attention}.",
        i = "Use {.code options(tabfound.sdpa = \"auto\")} for the pure-R \\
             fallback."
      ))
    }
    return(.sdpa_fallback(query, key, value, attn_mask, is_causal, scale))
  }
  args <- list(query = query, key = key, value = value,
               attn_mask = attn_mask, dropout_p = dropout_p,
               is_causal = is_causal)
  if (!is.null(scale)) args$scale <- scale
  do.call(fn, args)
}

# Resolved once per session: `getFromNamespace()` on every attention call
# of every layer of every member would be a measurable tax.
.sdpa_cache <- new.env(parent = emptyenv())

# @keywords internal
.torch_sdpa <- function() {
  if (!is.null(.sdpa_cache$fn)) return(.sdpa_cache$fn)
  if (isTRUE(.sdpa_cache$missing)) return(NULL)
  fn <- tryCatch(
    utils::getFromNamespace("torch_scaled_dot_product_attention", "torch"),
    error = function(e) NULL
  )
  if (is.null(fn)) .sdpa_cache$missing <- TRUE else .sdpa_cache$fn <- fn
  fn
}

# The same operation in public API. Not the default: it materialises the
# `(B, H, Lq, Lk)` score matrix that the fused kernel avoids, and its
# rounding differs in the last couple of float32 digits.
# @keywords internal
.sdpa_fallback <- function(query, key, value, attn_mask = NULL,
                           is_causal = FALSE, scale = NULL) {
  d <- as.numeric(query$size(query$dim()))
  s <- scale %||% (1 / sqrt(d))
  scores <- torch::torch_matmul(query, key$transpose(-2L, -1L)) * s
  if (isTRUE(is_causal)) {
    Lq <- query$size(query$dim() - 1L)
    Lk <- key$size(key$dim() - 1L)
    causal <- torch::torch_ones(c(Lq, Lk), dtype = torch::torch_bool(),
                                device = query$device)$tril()
    scores <- scores$masked_fill(causal$logical_not(), -Inf)
  }
  if (!is.null(attn_mask)) {
    scores <- if (attn_mask$dtype == torch::torch_bool()) {
      scores$masked_fill(attn_mask$logical_not(), -Inf)
    } else {
      scores + attn_mask
    }
  }
  torch::torch_matmul(torch::nnf_softmax(scores, dim = -1L), value)
}


#' Scaled-dot-product multi-head attention with fused per-head weights
#' @keywords internal
mha_fused_qkv <- torch::nn_module(
  "MhaFusedQkv",

  initialize = function(embedding_dim, n_heads, head_dim = NULL) {
    self$embedding_dim <- as.integer(embedding_dim)
    self$n_heads       <- as.integer(n_heads)
    if (is.null(head_dim)) {
      if (embedding_dim %% n_heads != 0L) {
        cli::cli_abort(
          "embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})."
        )
      }
      head_dim <- embedding_dim / n_heads
    }
    self$head_dim <- as.integer(head_dim)

    self$`_w_qkv` <- torch::nn_parameter(
      torch::torch_empty(3L, self$n_heads, self$head_dim, self$embedding_dim)
    )
    self$`_w_out` <- torch::nn_parameter(
      torch::torch_empty(self$n_heads, self$head_dim, self$embedding_dim)
    )
    self$reset_parameters()
  },

  reset_parameters = function() {
    # Xavier-style init; actual values are overwritten at weight load.
    torch::nn_init_xavier_uniform_(self$`_w_qkv`)
    torch::nn_init_xavier_uniform_(self$`_w_out`)
  },

  # @param x        `(B, L_q, emb)` — query input.
  # @param x_kv     Optional `(B, L_kv, emb)` — K/V source for cross-attention.
  #'                  When NULL, defaults to self-attention with `x_kv = x`.
  # @param add_input If TRUE, return `x + attn(...)`; the residual is added to
  #'   the QUERY input (i.e. shape matches `x`).
  # @param reuse_first_head_kv  If TRUE, K and V are projected using only
  #'   `_w_qkv[2:3, 1, , ]` (first head's K/V weights) and broadcast across
  #'   all heads — Multi-Query Attention. Used by Python's
  #'   `multiquery_item_attention_for_test_set` path for test-row queries.
  # @param cached_kv Optional `list(key, value)`, each `(B, 1, Lk, D)`, from
  #'   a previous pass over the training rows. Supplying it skips the K/V
  #'   projection entirely and broadcasts head 0 across all heads — the
  #'   same arithmetic `reuse_first_head_kv` performs, minus the work.
  #'
  #' @details
  #' The matmul ordering follows Python's reference
  #' (`tabpfn/architectures/base/attention/full_attention.py::compute_qkv`)
  #' to match its float32 rounding:
  #' - Self-attention path (`x_kv` is `x` AND `!reuse_first_head_kv`): fuse
  #'   all three projections into one `x @ W_flat.T` then reshape/unbind.
  #' - Cross-attention path: separate Q einsum, fused KV einsum (2 projections).
  forward = function(x, x_kv = NULL, add_input = FALSE,
                     reuse_first_head_kv = FALSE, cached_kv = NULL) {
    dims_q <- x$size()
    B <- dims_q[1]; Lq <- dims_q[2]; E <- dims_q[3]
    H <- self$n_heads; D <- self$head_dim
    w_qkv <- self$`_w_qkv`   # (3, H, D, E)

    if (!is.null(cached_kv)) {
      # Cached path: queries only. The stored head is broadcast across all
      # of them, which is what the uncached test-row branch does too.
      Lk <- cached_kv$key$size(3)
      q <- torch::torch_einsum("ble,hde->bhld", list(x, w_qkv[1, , , ]))
      k <- cached_kv$key$expand(c(B, H, Lk, D))
      v <- cached_kv$value$expand(c(B, H, Lk, D))
      ctx <- sdpa(
        query = q, key = k, value = v, dropout_p = 0
      )
      out <- torch::torch_einsum("bhld,hde->ble", list(ctx, self$`_w_out`))
      if (isTRUE(add_input)) out <- out + x
      return(out)
    }

    is_self_attn <- is.null(x_kv) || identical(x_kv, x)
    if (is.null(x_kv)) x_kv <- x
    dims_kv <- x_kv$size()
    Lk <- dims_kv[2]

    if (is_self_attn && !isTRUE(reuse_first_head_kv)) {
      # FUSED path — mirrors Python's `qkv = x @ w_flat.T; reshape; unbind`
      w_flat <- w_qkv$reshape(c(3L * H * D, E))        # (3*H*D, E)
      qkv_flat <- torch::torch_matmul(x, w_flat$t())   # (B, Lq, 3*H*D)
      qkv <- qkv_flat$reshape(c(B, Lq, 3L, H, D))      # (B, Lq, 3, H, D)
      q <- qkv$select(dim = 3L, index = 1L)            # (B, Lq, H, D)
      k <- qkv$select(dim = 3L, index = 2L)
      v <- qkv$select(dim = 3L, index = 3L)
      # Move head axis in front of seq for matmul.
      q <- q$permute(c(1L, 3L, 2L, 4L))                # (B, H, Lq, D)
      k <- k$permute(c(1L, 3L, 2L, 4L))
      v <- v$permute(c(1L, 3L, 2L, 4L))
    } else {
      # CROSS-ATTN path — separate Q, fused KV.
      w_q  <- w_qkv[1, , , ]                           # (H, D, E)
      w_kv <- w_qkv[2:3, , , ]                         # (2, H, D, E)
      if (isTRUE(reuse_first_head_kv)) {
        w_kv <- w_kv[, 1, , ]$unsqueeze(2L)            # (2, 1, D, E)
      }
      q <- torch::torch_einsum("ble,hde->bhld", list(x, w_q))            # (B, H, Lq, D)
      kv <- torch::torch_einsum("ble,jhde->bjhld", list(x_kv, w_kv))     # (B, 2, H', Lk, D)
      k <- kv$select(dim = 2L, index = 1L)             # (B, H', Lk, D)
      v <- kv$select(dim = 2L, index = 2L)
      if (isTRUE(reuse_first_head_kv)) {
        k <- k$expand(c(B, H, Lk, D))
        v <- v$expand(c(B, H, Lk, D))
      }
    }

    # The fused SDPA kernel, to match Python's
    # `torch.nn.functional.scaled_dot_product_attention` numerics exactly.
    # See `sdpa()` for why it is reached through a wrapper.
    ctx <- sdpa(
      query = q, key = k, value = v, dropout_p = 0
    )   # (B, H, Lq, D)

    out <- torch::torch_einsum("bhld,hde->ble", list(ctx, self$`_w_out`))

    if (isTRUE(add_input)) out <- out + x
    out
  },

  #' Head-0 key/value projections of `x_kv`, for a KV cache.
  #'
  #' Computed through the *same* einsum the `reuse_first_head_kv` branch
  #' above uses, not by slicing head 0 out of the fused self-attention
  #' projection. The two are algebraically identical and disagree in the
  #' last bit, because the fused form contracts over `3 * H * D` outputs at
  #' once and this one over `2 * 1 * D`. Feeding a cache built the other
  #' way into that branch leaves a 1-ULP seed that twenty-four layers turn
  #' into ~5e-5; matching the arrangement makes a cached prediction
  #' bit-identical to an uncached one instead.
  #' @keywords internal
  cache_kv = function(x_kv) {
    w_kv <- self$`_w_qkv`[2:3, , , ][, 1, , ]$unsqueeze(2L)   # (2, 1, D, E)
    kv <- torch::torch_einsum("ble,jhde->bjhld", list(x_kv, w_kv))
    list(key   = kv$select(dim = 2L, index = 1L)$detach()$contiguous(),
         value = kv$select(dim = 2L, index = 2L)$detach()$contiguous())
  }
)


# ---------------------------------------------------------------------------
# Separate q/k/v projections with per-head norm and a learned scale (TabFM)
# ---------------------------------------------------------------------------

#' Multi-head attention with per-head query/key RMSNorm and a learned scale
#'
#' Four independent `nn_linear` projections with biases, RMSNorm applied
#' to each head's queries and keys, and a learned per-dimension scale
#' folded into the query before an unscaled dot product.
#'
#' The scale is `1.442695041 / sqrt(head_dim) * softplus(per_dim_scale)`,
#' computed in float32, and SDPA is then called with `scale = 1`. The
#' magic constant is `1 / ln(2)`: the reference's attention was written
#' against a base-2 softmax, and the factor survives in the released
#' weights, so it has to be reproduced rather than folded away.
#'
#' @param embedding_dim Model width.
#' @param n_heads Number of heads; must divide `embedding_dim`.
#' @param use_rope Logical. When `TRUE`, the caller must supply a `rope`
#'   module to `forward()`; queries and keys are rotated before the
#'   query/key norms.
#' @keywords internal
mha_qk_norm <- torch::nn_module(
  "MhaQkNorm",

  initialize = function(embedding_dim, n_heads, use_rope = FALSE) {
    if (embedding_dim %% n_heads != 0L) {
      cli::cli_abort(
        "embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})."
      )
    }
    self$n_heads <- as.integer(n_heads)
    self$head_dim <- as.integer(embedding_dim / n_heads)
    self$use_rope <- isTRUE(use_rope)

    self$q_proj <- torch::nn_linear(embedding_dim, embedding_dim, bias = TRUE)
    self$k_proj <- torch::nn_linear(embedding_dim, embedding_dim, bias = TRUE)
    self$v_proj <- torch::nn_linear(embedding_dim, embedding_dim, bias = TRUE)
    self$out_proj <- torch::nn_linear(embedding_dim, embedding_dim, bias = TRUE)
    self$query_ln <- rms_norm(self$head_dim)
    self$key_ln   <- rms_norm(self$head_dim)
    self$per_dim_scale <- torch::nn_parameter(torch::torch_zeros(self$head_dim))
  },

  # @param query `(B, Tq, E)`; @param key,value `(B, Tk, E)`. Both may be
  #   NULL when `cached_kv` is supplied.
  # @param attn_mask Optional boolean tensor broadcastable to
  #   `(B, n_heads, Tq, Tk)`, `TRUE` where attention is allowed.
  # @param rope Optional [rope()] module, required when `use_rope`.
  # @param cached_kv Optional `list(key, value)` from [cache_kv()], each
  #   `(B, H, Tk, D)` and already normalized and rotated. Supplying it
  #   replaces `key`/`value` and skips their projections.
  forward = function(query, key = NULL, value = NULL, attn_mask = NULL,
                     rope = NULL, cached_kv = NULL) {
    b  <- query$size(1)
    tq <- query$size(2)
    e  <- query$size(3)
    H  <- self$n_heads
    D  <- self$head_dim

    q <- self$q_proj(query)$view(c(b, tq, H, D))

    if (is.null(cached_kv)) {
      if (is.null(key) || is.null(value)) {
        cli::cli_abort("Supply {.arg key} and {.arg value}, or {.arg cached_kv}.")
      }
      k <- self$k_proj(key)$view(c(b, key$size(2), H, D))
      v <- self$v_proj(value)$view(c(b, value$size(2), H, D))
    }

    if (self$use_rope) {
      if (is.null(rope)) {
        cli::cli_abort("This attention layer needs a {.cls rope} module.")
      }
      q <- rope(q)
      if (is.null(cached_kv)) k <- rope(k)
    }

    q <- self$query_ln(q)
    if (is.null(cached_kv)) k <- self$key_ln(k)

    scale <- 1.442695041 / sqrt(D) *
      torch::nnf_softplus(self$per_dim_scale$to(dtype = torch::torch_float32()))
    q <- q * scale$to(dtype = q$dtype)

    # (B, T, H, D) -> (B, H, T, D)
    q <- q$permute(c(1L, 3L, 2L, 4L))
    if (is.null(cached_kv)) {
      k <- k$permute(c(1L, 3L, 2L, 4L))
      v <- v$permute(c(1L, 3L, 2L, 4L))
    } else {
      k <- cached_kv$key
      v <- cached_kv$value
    }

    ctx <- sdpa(
      query = q, key = k, value = v, attn_mask = attn_mask,
      dropout_p = 0, scale = 1.0
    )

    self$out_proj(
      ctx$permute(c(1L, 3L, 2L, 4L))$contiguous()$reshape(c(b, tq, e))
    )
  },

  #' Key/value projections of `key`/`value`, for a KV cache.
  #'
  #' Runs exactly the steps `forward()` runs on its key and value inputs
  #' -- projection, optional rotation, the key norm, and the move of the
  #' head axis in front of the sequence -- and stops there. Feeding the
  #' result back through `cached_kv` therefore reproduces the uncached
  #' arithmetic bit for bit, not merely closely.
  #' @keywords internal
  cache_kv = function(key, value = NULL, rope = NULL) {
    if (is.null(value)) value <- key
    b <- key$size(1); H <- self$n_heads; D <- self$head_dim
    k <- self$k_proj(key)$view(c(b, key$size(2), H, D))
    v <- self$v_proj(value)$view(c(b, value$size(2), H, D))
    if (self$use_rope) {
      if (is.null(rope)) {
        cli::cli_abort("This attention layer needs a {.cls rope} module.")
      }
      k <- rope(k)
    }
    k <- self$key_ln(k)
    list(key   = k$permute(c(1L, 3L, 2L, 4L))$detach()$contiguous(),
         value = v$permute(c(1L, 3L, 2L, 4L))$detach()$contiguous())
  }
)


# ---------------------------------------------------------------------------
# Fused in-projection, PyTorch nn.MultiheadAttention layout (TabICL)
# ---------------------------------------------------------------------------

#' Multi-head attention with a packed in-projection
#'
#' Stores queries, keys and values in one `in_proj_weight` of shape
#' `(3E, E)` plus an `in_proj_bias`, which is what `nn.MultiheadAttention`
#' does and therefore what TabICL's checkpoint contains.
#'
#' The projection is computed the way `F._in_projection_packed` computes
#' it, because the arrangement determines the float32 summation order:
#'
#' * self-attention (`k` and `v` both `NULL`): one fused `(3E, E)` matmul,
#'   then split;
#' * cross-attention (`k` and `v` the same tensor): a separate query
#'   matmul and one fused `(2E, E)` key/value matmul.
#'
#' @param embedding_dim Model width.
#' @param n_heads Number of heads; must divide `embedding_dim`.
#' @param ssmax When `TRUE`, attach a [ssmax_qa_mlp()] that rescales
#'   queries by context length before the dot product.
#' @keywords internal
mha_fused_inproj <- torch::nn_module(
  "MhaFusedInProj",

  initialize = function(embedding_dim, n_heads, ssmax = FALSE) {
    if (embedding_dim %% n_heads != 0L) {
      cli::cli_abort(
        "embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})."
      )
    }
    self$embedding_dim <- as.integer(embedding_dim)
    self$n_heads <- as.integer(n_heads)
    self$head_dim <- as.integer(embedding_dim / n_heads)

    self$in_proj_weight <- torch::nn_parameter(
      torch::torch_empty(3L * self$embedding_dim, self$embedding_dim)
    )
    self$in_proj_bias <- torch::nn_parameter(
      torch::torch_zeros(3L * self$embedding_dim)
    )
    torch::nn_init_xavier_uniform_(self$in_proj_weight)
    self$out_proj <- torch::nn_linear(embedding_dim, embedding_dim, bias = TRUE)

    self$use_ssmax <- isTRUE(ssmax)
    if (self$use_ssmax) {
      self$ssmax_layer <- ssmax_qa_mlp(self$n_heads, self$head_dim)
    }
  },

  # Split a (..., T, 3E)-style projection result into heads:
  # (..., T, E) -> (..., H, T, D).
  to_heads = function(x, t) {
    sz <- as.integer(x$size())
    lead <- sz[-c(length(sz) - 1L, length(sz))]
    x <- x$view(c(lead, t, self$n_heads, self$head_dim))
    x$transpose(-3L, -2L)
  },

  #' Key/value projections of `key`, for a KV cache.
  #'
  #' Computed through the fused `(2E, E)` matmul the cross-attention
  #' branch of `forward()` uses, not by slicing the `(3E, E)` one, so a
  #' cached prediction matches an uncached one bit for bit rather than to
  #' within a rounding step. The two arrangements are algebraically the
  #' same and disagree in the last bit.
  #' @keywords internal
  cache_kv = function(key, rope = NULL) {
    E <- self$embedding_dim
    tk <- key$size(key$dim() - 1L)
    w_kv <- self$in_proj_weight[(E + 1L):(3L * E), ]
    b_kv <- self$in_proj_bias[(E + 1L):(3L * E)]
    kv <- torch::nnf_linear(key, w_kv, b_kv)
    kv_chunks <- kv$chunk(2L, dim = -1L)
    k <- self$to_heads(kv_chunks[[1]], tk)
    v <- self$to_heads(kv_chunks[[2]], tk)
    if (!is.null(rope)) k <- rope(k)
    list(key = k$detach()$contiguous(), value = v$detach()$contiguous())
  },

  # @param query `(..., Tq, E)`.
  # @param key,value `(..., Tk, E)`. Both `NULL` means self-attention.
  # @param attn_mask Additive float mask broadcastable to
  #   `(..., H, Tq, Tk)`, or `NULL`.
  # @param rope Optional [rope()] applied to queries and keys.
  # @param cached_kv Optional `list(key, value)` from [cache_kv()], each
  #   already in `(..., H, Tk, D)` head layout. Supplying it replaces
  #   `key`/`value` and skips their projection.
  forward = function(query, key = NULL, value = NULL, attn_mask = NULL,
                     rope = NULL, cached_kv = NULL) {
    E <- self$embedding_dim
    nd <- query$dim()
    tq <- query$size(nd - 1L)
    is_self <- is.null(key) && is.null(value)

    if (!is.null(cached_kv)) {
      q <- self$to_heads(
        torch::nnf_linear(query, self$in_proj_weight[1:E, ],
                          self$in_proj_bias[1:E]),
        tq
      )
      k <- cached_kv$key
      v <- cached_kv$value
    } else if (is_self) {
      proj <- torch::nnf_linear(query, self$in_proj_weight, self$in_proj_bias)
      chunks <- proj$chunk(3L, dim = -1L)
      q <- self$to_heads(chunks[[1]], tq)
      k <- self$to_heads(chunks[[2]], tq)
      v <- self$to_heads(chunks[[3]], tq)
    } else {
      if (is.null(key) || is.null(value)) {
        cli::cli_abort("Supply both {.arg key} and {.arg value}, or neither.")
      }
      tk <- key$size(key$dim() - 1L)
      w_q  <- self$in_proj_weight[1:E, ]
      w_kv <- self$in_proj_weight[(E + 1L):(3L * E), ]
      b_q  <- self$in_proj_bias[1:E]
      b_kv <- self$in_proj_bias[(E + 1L):(3L * E)]

      q <- self$to_heads(torch::nnf_linear(query, w_q, b_q), tq)
      kv <- torch::nnf_linear(key, w_kv, b_kv)
      kv_chunks <- kv$chunk(2L, dim = -1L)
      k <- self$to_heads(kv_chunks[[1]], tk)
      v <- self$to_heads(kv_chunks[[2]], tk)
    }

    if (!is.null(rope)) {
      q <- rope(q)
      # A cached key was rotated when it was built; rotating it again
      # would advance it a second time.
      if (is.null(cached_kv)) k <- rope(k)
    }

    if (self$use_ssmax) {
      q <- self$ssmax_layer(q, k$size(k$dim() - 1L))
    }

    # SDPA takes exactly four dimensions, so collapse any extra leading
    # axes first -- TabICL runs blocks over (B, T, HC, E), which reaches
    # here as five. Mirrors the reference's `sdpa_with_flattened_batch`.
    qs <- as.integer(q$size())
    q4 <- q$reshape(c(-1L, qs[length(qs) - 2L], qs[length(qs) - 1L], qs[length(qs)]))
    ks <- as.integer(k$size())
    k4 <- k$reshape(c(-1L, ks[length(ks) - 2L], ks[length(ks) - 1L], ks[length(ks)]))
    v4 <- v$reshape(c(-1L, ks[length(ks) - 2L], ks[length(ks) - 1L], ks[length(ks)]))
    m4 <- attn_mask
    if (!is.null(m4) && m4$dim() > 4L) {
      ms <- as.integer(m4$size())
      m4 <- m4$reshape(c(-1L, ms[length(ms) - 2L], ms[length(ms) - 1L], ms[length(ms)]))
    }

    # Default scale (1 / sqrt(head_dim)); the ssmax factor is folded into
    # the query rather than replacing the scale.
    ctx <- sdpa(
      query = q4, key = k4, value = v4, attn_mask = m4, dropout_p = 0
    )
    ctx <- ctx$reshape(qs)

    ctx <- ctx$transpose(-3L, -2L)$contiguous()
    sz <- as.integer(ctx$size())
    lead <- sz[-c(length(sz) - 2L, length(sz) - 1L, length(sz))]
    self$out_proj(ctx$view(c(lead, tq, E)))
  }
)
