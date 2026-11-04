# Kumo Tabular building blocks.
#
# A port of the attention stack in NVIDIA's `structured-data-models`
# (`sdm` 0.2.1: `sdm/nn/` and `sdm/models/kumo/tabular/`), Apache-2.0.
# Module and parameter names follow the checkpoint keys, so loading needs
# no key translation.
#
# The conventions differ enough from the TabICL and TabFM blocks that
# nothing in `R/nn-*.R` is reused beyond `sdpa()`:
#
#   * heads live on the second-to-last axis, `[..., S, H, hd]`, and RoPE
#     rotates along `S`, the axis before them;
#   * Q and K/V are always projected by separate `linear()` calls on row
#     slices of `qkv_lin` -- the reference never runs the fused
#     projection, because every block normalizes query and key/value with
#     different weights -- and the rounding follows the call shapes;
#   * the per-head transform is RoPE *then* a weightless RMSNorm;
#   * query scaling multiplies the query by `head_scale * log(n)` before
#     SDPA, with `n` the number of keys.
#
#   h   <- q + attn(query_norm(q), key_value_norm(kv))
#   out <- h + mlp(h),   mlp = RMSNorm -> Linear(c, 2c) -> GELU -> Linear(2c, c)


# torch.finfo(torch.float32).eps, which is what `nn.RMSNorm(eps = None)`
# resolves to for float32 input.
.kumo_f32_eps <- 1.1920928955078125e-07


#' RMS normalization in the reference's composite form
#'
#' `x * rsqrt(mean(x^2) + eps)`, then times `weight` when affine. The
#' package's [rms_norm()] multiplies in a different order, which is
#' enough to change the last bit, so this one follows
#' `torch.nn.functional.rms_norm`'s CPU path literally.
#'
#' @param dim Size of the normalized (last) axis.
#' @param eps `NULL` for the float32 machine epsilon, as `nn.RMSNorm`
#'   resolves `eps = None`.
#' @param affine Learnable per-channel `weight`.
#' @keywords internal
kumo_rms_norm <- torch::nn_module(
  "KumoRMSNorm",

  initialize = function(dim, eps = NULL, affine = TRUE) {
    self$eps <- eps %||% .kumo_f32_eps
    self$affine <- isTRUE(affine)
    if (self$affine) self$weight <- torch::nn_parameter(torch::torch_ones(dim))
  },

  forward = function(x) {
    r <- torch::torch_rsqrt(x$pow(2)$mean(dim = -1L, keepdim = TRUE) + self$eps)
    out <- x * r
    if (self$affine) out <- out * self$weight
    out
  }
)


#' Split-half rotary position embedding over `[..., S, H, hd]`
#'
#' Positions `0..S-1` along the axis before the heads. `inv_freq` is a
#' parameter so the checkpoint's table loads into it; the reference
#' initialises it as `1 / theta^(2i / hd)` and freezes it.
#'
#' @param head_dim Channels per head.
#' @param theta Base for the initial (overwritten) table.
#' @keywords internal
kumo_rope <- torch::nn_module(
  "KumoRoPE",

  initialize = function(head_dim, theta = 1e5) {
    ar <- torch::torch_arange(0L, head_dim - 1L, 2L, dtype = torch::torch_float())
    self$inv_freq <- torch::nn_parameter(1 / theta^(ar / head_dim),
                                         requires_grad = FALSE)
  },

  forward = function(x) {
    nd <- x$dim()
    s <- x$size(nd - 2L)
    seq <- torch::torch_arange(0L, s - 1L, dtype = torch::torch_float(),
                               device = x$device)
    freq <- seq$view(c(-1L, 1L)) * self$inv_freq$view(c(1L, -1L))
    sin <- freq$sin()$unsqueeze(2L)$to(dtype = x$dtype)
    cos <- freq$cos()$unsqueeze(2L)$to(dtype = x$dtype)
    half <- self$inv_freq$size(1)
    x1 <- x$narrow(nd, 1L, half)
    x2 <- x$narrow(nd, half + 1L, half)
    torch::torch_cat(list(x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim = -1L)
  }
)


#' Logarithmic query scaling
#'
#' `q * log(max(n, 1)) * head_scale[h]`, with `n` the number of keys.
#'
#' @param n_heads Number of query heads.
#' @keywords internal
kumo_log_scale <- torch::nn_module(
  "KumoLogScale",

  initialize = function(n_heads) {
    self$head_scale <- torch::nn_parameter(torch::torch_ones(n_heads))
  },

  .scale = function(query, n_keys) {
    logn <- torch::torch_tensor(as.numeric(n_keys), dtype = torch::torch_float(),
                                device = query$device)$clamp(min = 1)$log()
    hs <- self$head_scale$to(dtype = query$dtype)$view(c(-1L, 1L))
    query * logn$to(dtype = query$dtype) * hs
  },

  forward = function(query, n_keys) self$.scale(query, n_keys)
)

#' Gated logarithmic query scaling
#'
#' [kumo_log_scale()] times `1 + tanh(gate(q))`, the gate a two-layer
#' GELU MLP over each head's (transformed, unscaled) query vector.
#'
#' @param head_dim Channels per head.
#' @param n_heads Number of query heads.
#' @param hidden Gate width.
#' @keywords internal
kumo_gated_log_scale <- torch::nn_module(
  "KumoGatedLogScale",
  inherit = kumo_log_scale,

  initialize = function(head_dim, n_heads, hidden = 64L) {
    super$initialize(n_heads)
    self$gate <- torch::nn_sequential(
      torch::nn_linear(head_dim, hidden),
      torch::nn_gelu(),
      torch::nn_linear(hidden, head_dim)
    )
  },

  forward = function(query, n_keys) {
    self$.scale(query, n_keys) * (1 + self$gate(query)$tanh())
  }
)


# The module that holds the query scaling, so its parameters sit at
# `attn.sdpa.query_scaling.*` as in the checkpoint.
kumo_sdpa <- torch::nn_module(
  "KumoSDPA",

  initialize = function(query_scaling = NULL) {
    if (!is.null(query_scaling)) self$query_scaling <- query_scaling
  },

  # @param query `[..., Q, Hq, hd]`; key, value `[..., KV, Hkv, hd]`.
  # Leading dimensions broadcast. Returns `[..., Q, Hq, hd]`.
  forward = function(query, key, value) {
    if (query$numel() == 0L) return(query)
    if (!is.null(self$query_scaling)) {
      query <- self$query_scaling(query, key$size(key$dim() - 2L))
    }
    qs <- query$shape; ks <- key$shape; vs <- value$shape
    nq <- length(qs); nk <- length(ks)
    batch <- .kumo_broadcast(qs[seq_len(nq - 3L)], ks[seq_len(nk - 3L)],
                             vs[seq_len(length(vs) - 3L)])
    q_tail <- qs[(nq - 2L):nq]
    k_tail <- ks[(nk - 2L):nk]
    v_tail <- vs[(length(vs) - 2L):length(vs)]

    # The reference's FlashAttention guard: no keys, or an empty batch,
    # is zeros rather than a kernel call.
    if (k_tail[1] == 0L || any(batch == 0L)) {
      return(torch::torch_zeros(c(batch, q_tail[1:2], v_tail[3]),
                                dtype = query$dtype, device = query$device))
    }
    flat <- function(t, tail) t$expand(c(batch, tail))$reshape(c(-1L, tail))
    out <- sdpa(
      flat(query, q_tail)$transpose(2L, 3L),
      flat(key, k_tail)$transpose(2L, 3L),
      flat(value, v_tail)$transpose(2L, 3L),
      enable_gqa = q_tail[2] != k_tail[2]
    )$transpose(2L, 3L)
    out$view(c(batch, out$shape[2:4]))
  }
)

# Broadcast leading shapes the way torch.broadcast_shapes does.
.kumo_broadcast <- function(...) {
  shapes <- list(...)
  n <- max(vapply(shapes, length, integer(1)), 0L)
  if (n == 0L) return(integer())
  pad <- lapply(shapes, function(s) c(rep(1L, n - length(s)), s))
  m <- do.call(rbind, pad)
  apply(m, 2L, function(col) {
    big <- col[col != 1L]
    if (!length(big)) 1L else big[1]
  })
}


# Weightless per-head RMSNorm (eps 1e-6), the last step of the query and
# key transforms.
kumo_head_norm <- torch::nn_module(
  "KumoHeadNorm",
  initialize = function(eps = 1e-6) self$eps <- eps,
  forward = function(x) {
    x * torch::torch_rsqrt(x$pow(2)$mean(dim = -1L, keepdim = TRUE) + self$eps)
  }
)


#' Multi-head attention with separate Q and K/V projections
#'
#' Parameters: `qkv_lin` (`Linear(c, 3c)`, rows Q | K | V), `out_lin`,
#' `query_transform` / `key_transform` (`[rope?, head norm]`), and `sdpa`
#' holding the query scaling.
#'
#' `forward(query, key_value)` takes the key/value source tensor, or a
#' cached `list(key, value)` of already-transformed per-head projections
#' as [kumo_attention]'s `return_kv` hands back -- possibly with fewer
#' heads than the query, which is grouped-query attention.
#'
#' @param channels Model width.
#' @param n_heads Number of heads.
#' @param query_scaling A [kumo_log_scale()] / [kumo_gated_log_scale()],
#'   or `NULL`.
#' @param rope `TRUE` to rotate query and key before the head norm.
#' @keywords internal
kumo_attention <- torch::nn_module(
  "KumoAttention",

  initialize = function(channels, n_heads, query_scaling = NULL, rope = FALSE) {
    self$channels <- as.integer(channels)
    self$n_heads <- as.integer(n_heads)
    self$head_dim <- self$channels %/% self$n_heads
    self$qkv_lin <- torch::nn_linear(channels, 3L * channels)
    self$query_transform <- if (isTRUE(rope)) {
      torch::nn_sequential(kumo_rope(self$head_dim), kumo_head_norm())
    } else {
      torch::nn_sequential(kumo_head_norm())
    }
    self$key_transform <- if (isTRUE(rope)) {
      torch::nn_sequential(kumo_rope(self$head_dim), kumo_head_norm())
    } else {
      torch::nn_sequential(kumo_head_norm())
    }
    self$sdpa <- kumo_sdpa(query_scaling)
    self$out_lin <- torch::nn_linear(channels, channels)
  },

  # `[..., S, n*hd]` -> `[..., S, n, hd]`.
  .heads = function(t) {
    t$unflatten(t$dim(), c(t$size(t$dim()) %/% self$head_dim, self$head_dim))
  },

  # Transformed K and raw V of `kv`, per head: what a cache stores.
  project_kv = function(kv) {
    c_ <- self$channels
    w <- self$qkv_lin$weight[(c_ + 1L):(3L * c_), ]
    b <- self$qkv_lin$bias[(c_ + 1L):(3L * c_)]
    kv_p <- torch::torch_chunk(torch::nnf_linear(kv, w, b), 2L, dim = -1L)
    list(key = self$key_transform(self$.heads(kv_p[[1]])),
         value = self$.heads(kv_p[[2]]))
  },

  forward = function(query, key_value, return_kv = FALSE) {
    c_ <- self$channels
    q <- torch::nnf_linear(query, self$qkv_lin$weight[1:c_, ],
                           self$qkv_lin$bias[1:c_])
    kv <- if (is.list(key_value)) key_value else self$project_kv(key_value)
    q <- self$query_transform(self$.heads(q))
    out <- self$sdpa(q, kv$key, kv$value)
    out <- self$out_lin(out$flatten(start_dim = -2L))
    if (isTRUE(return_kv)) list(out = out, kv = kv) else out
  }
)


#' Pre-norm transformer block
#'
#' `h = q + attn(query_norm(q), key_value_norm(kv)); h + mlp(h)`. The two
#' norms are separate weights even when `kv` is `q`.
#'
#' @inheritParams kumo_attention
#' @keywords internal
kumo_block <- torch::nn_module(
  "KumoTransformerBlock",

  initialize = function(channels, n_heads, query_scaling = NULL, rope = FALSE) {
    self$query_norm <- kumo_rms_norm(channels)
    self$key_value_norm <- kumo_rms_norm(channels)
    self$attn <- kumo_attention(channels, n_heads, query_scaling, rope)
    self$mlp <- torch::nn_sequential(
      kumo_rms_norm(channels),
      torch::nn_linear(channels, 2L * channels),
      torch::nn_gelu(),
      torch::nn_linear(2L * channels, channels)
    )
  },

  # @param key_value Source tensor, or a cached `list(key, value)`, which
  #   skips `key_value_norm` (the cache holds post-projection tensors).
  # @param return_kv Also return the key/value projections.
  forward = function(query, key_value, return_kv = FALSE) {
    kv <- if (is.list(key_value)) key_value else self$key_value_norm(key_value)
    a <- self$attn(self$query_norm(query), kv, return_kv = return_kv)
    attn_out <- if (isTRUE(return_kv)) a$out else a
    h <- attn_out + query
    out <- h + self$mlp(h)
    if (isTRUE(return_kv)) list(out = out, kv = a$kv) else out
  }
)


#' Induced-set attention over the rows of each column
#'
#' `M` learned inducing points attend to the context rows
#' (`inducing_block`, query-scaled by `log n_context`), then every row
#' attends to the `M` results (`output_block`, unscaled). Columns are
#' the batch axis.
#'
#' @param channels Model width.
#' @param n_heads Heads in both blocks.
#' @param n_inducing Number of inducing points `M`.
#' @keywords internal
kumo_induced_block <- torch::nn_module(
  "KumoInducedBlock",

  initialize = function(channels, n_heads, n_inducing) {
    self$inducing_block <- kumo_block(channels, n_heads,
                                      query_scaling = kumo_log_scale(n_heads))
    self$output_block <- kumo_block(channels, n_heads)
    self$inducing_points <- torch::nn_parameter(
      torch::torch_zeros(n_inducing, channels)
    )
  },

  # The inducing summary of the context rows, `[..., M, D]`.
  induce = function(context) {
    self$inducing_block(self$inducing_points, context)
  },

  # @param x All rows `[..., R, D]`.
  # @param context The context rows `[..., Rt, D]`, or a cached
  #   `list(key, value)` of the output block's projection of the summary.
  forward = function(x, context, return_kv = FALSE) {
    hidden <- if (is.list(context)) context else self$induce(context)
    self$output_block(x, hidden, return_kv = return_kv)
  }
)


#' Per-cell Fourier embedding with a learned missingness projection
#'
#' Each cell is embedded together with two neighbours: slot `g` of column
#' `c` reads column `(c + {0, 1, 3}[g]) mod C`, wrapping around with no
#' padding. A slot's value `v` becomes `[sin(v f), cos(v f)]` with the
#' frequency row for that slot and that column's type (numerical or
#' categorical), the three slots are projected together by `num_lin` /
#' `cat_lin` (bias added once per slot), and `nan_lin` adds a projection
#' of the slots' missingness. Missing cells are filled with the context
#' column mean first (zero for an all-missing column).
#'
#' @param channels Output width `D`.
#' @param group_size Slots per cell (3).
#' @param n_freq Frequencies per slot (32).
#' @keywords internal
kumo_cell_embedding <- torch::nn_module(
  "KumoCellEmbedding",

  initialize = function(channels, group_size = 3L, n_freq = 32L) {
    self$channels <- as.integer(channels)
    self$group_size <- as.integer(group_size)
    self$num_freq <- torch::nn_parameter(torch::torch_zeros(group_size, n_freq))
    self$cat_freq <- torch::nn_parameter(torch::torch_zeros(group_size, n_freq))
    self$num_lin <- torch::nn_linear(2L * n_freq, channels)
    self$cat_lin <- torch::nn_linear(2L * n_freq, channels)
    self$nan_lin <- torch::nn_linear(group_size, channels, bias = FALSE)
  },

  # The context column means used to fill missing cells, `[..., 1, C]`.
  impute_mean = function(x, train_size) {
    m <- .kumo_nanmean(x$narrow(-2L, 1L, train_size), dim = -2L)
    m$masked_fill(m$isnan(), 0)
  },

  # @param x `[..., R, C]` float32, NaN for missing.
  # @param categorical_mask `[..., C]` logical tensor.
  # @param train_size Number of leading context rows, or `impute_mean`
  #   given directly (a cache replay).
  forward = function(x, categorical_mask, train_size = NULL, impute_mean = NULL) {
    missing <- x$isnan()
    mu <- impute_mean %||% self$impute_mean(x, train_size)
    x <- torch::torch_where(missing, mu, x)

    nd <- x$dim()
    C <- x$size(nd)
    G <- self$group_size
    # 1-based column index of slot g of column c: ((c - 1) + 2^g - 1) mod C + 1.
    shift <- 2L^(seq_len(G) - 1L) - 1L
    index <- outer(seq_len(C) - 1L, shift, `+`) %% C + 1L
    idx <- torch::torch_tensor(as.vector(t(index)), dtype = torch::torch_long(),
                               device = x$device)
    gather <- function(t) {
      d <- t$dim()
      t$index_select(d, idx)$unflatten(d, c(C, G))
    }
    xg <- gather(x)                                   # [..., R, C, G]
    mg <- gather(missing)
    cm <- gather(categorical_mask)$unsqueeze(-3L)    # [..., 1, C, G]

    freq <- torch::torch_where(cm$unsqueeze(-1L), self$cat_freq, self$num_freq)
    lead <- rep(1L, cm$dim())
    weight <- torch::torch_where(
      cm$unsqueeze(-1L)$unsqueeze(-1L),
      self$cat_lin$weight$view(c(lead, self$cat_lin$weight$shape)),
      self$num_lin$weight$view(c(lead, self$num_lin$weight$shape))
    )                                                 # [..., 1, C, G, D, 2F]
    bias <- torch::torch_where(
      cm$unsqueeze(-1L),
      self$cat_lin$bias$view(c(lead, -1L)),
      self$num_lin$bias$view(c(lead, -1L))
    )$sum(dim = -2L)                                  # [..., 1, C, D]

    z <- xg$unsqueeze(-1L) * freq                     # [..., R, C, G, F]
    fourier <- torch::torch_cat(list(z$sin(), z$cos()), dim = -1L)
    w <- weight$transpose(-3L, -2L)$flatten(start_dim = -2L)$squeeze(-4L)$transpose(-2L, -1L)
    f <- fourier$transpose(-4L, -3L)$flatten(start_dim = -2L)  # [..., C, R, G*2F]
    out <- torch::torch_matmul(f, w)$transpose(-3L, -2L)       # [..., R, C, D]
    out <- out + bias
    out + self$nan_lin(mg$to(dtype = out$dtype))
  }
)

# torch.nanmean without relying on R torch exporting it: sum of the
# finite-or-not-NaN entries over their count. Matches `nanmean`, which is
# itself `nansum / count`.
.kumo_nanmean <- function(x, dim) {
  ok <- x$isnan()$logical_not()
  s <- torch::torch_nansum(x, dim = dim, keepdim = TRUE)
  n <- ok$sum(dim = dim, keepdim = TRUE)$to(dtype = x$dtype)
  s / n
}
