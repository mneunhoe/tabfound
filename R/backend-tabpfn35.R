# TabPFN v3.5 backend (Prior-Labs).
#
# v3.5 keeps v3's four-stage shape -- embed each cell, summarise each
# column's distribution against inducing points, aggregate a row's columns
# onto CLS tokens, then attend across rows in context -- and changes four
# things inside it. Each is small on its own; together they are enough
# that the two cannot share a module tree.
#
#   1. The cell embedder is no longer a linear. Every cell now arrives as
#      three channels rather than two -- its standard-scaled value, its
#      NaN/Inf indicator, and its *in-context ECDF rank*, the fraction of
#      training rows in that column at or below it. The value goes through
#      a learned Fourier bank, the rank through a fixed sin/cos expansion,
#      the two sums are added and layer-normed. A column's values are
#      therefore read relative to that column's own distribution before
#      stage 1 ever sees them, which is what the release notes mean by
#      better handling of high-cardinality categoricals.
#
#   2. Every attention normalises its queries and keys per head (`q_norm`,
#      `k_norm`, RMSNorm over `head_dim`) before the scores are formed.
#      In the ICL stack the keys are normed *before* they are cached, so a
#      cached pass must not norm them again.
#
#   3. One checkpoint serves both tasks. The target encoders are pairs --
#      a class embedding and a scalar projection, each followed by a
#      LayerNorm -- and the output heads are bundled in a `heads` module
#      holding both. Which branch runs is decided per forward pass, not by
#      the weights, so `build()` reads the `task` it is handed rather than
#      a `head` field in the config. Everything in the tree exists in
#      every checkpoint, so the strict loader is satisfied either way.
#
#   4. Each head gets a residual pre-norm MLP of its own before its final
#      projection, and the regressor's projection is now a single linear
#      onto the 5000 bars rather than v3's three-layer stack.
#
# The ICL stack is also wider -- 1024 instead of 512, 16 heads, 8 CLS
# tokens instead of 4 -- but that is config, not code.
#
# Reference: `tabpfn/architectures/tabpfn_v3_5.py` in the PyPI `tabpfn`
# package (9.0.0). Unlike every earlier generation the checkpoints are
# already safetensors, with the config in the file's own `__metadata__`
# header; `inst/python/tabpfn35_convert_ckpt.py` lifts it out and writes
# `arch: "tabpfn_v3_5"` so `detect_backend()` can tell the generations
# apart.
#
# Module names below are the checkpoint's own, so no key translation is
# needed.

# `torch.finfo(torch.float32).eps`. Used where the reference leaves it
# implicit: `TorchStandardScaler.transform`'s denominator, and
# `nn.RMSNorm`'s epsilon, which defaults to the input dtype's eps.
TABPFN35_F32_EPS <- 1.1920928955078125e-07

# `nn.LayerNorm`'s default eps. The three LayerNorms in this architecture
# -- the cell embedder's and the two target encoders' -- take it, unlike
# the RMSNorms everywhere else.
TABPFN35_LN_EPS <- 1e-5

# Sentinels marking NaN / +Inf / -Inf in the indicator channel. Same
# values as v2.5, v2.6 and v3, repeated here because they belong to the
# architecture rather than to a shared utility.
TABPFN35_NAN_INDICATOR     <- -2.0
TABPFN35_POS_INF_INDICATOR <-  2.0
TABPFN35_NEG_INF_INDICATOR <-  4.0

# Cells the ECDF context build and query work on per pass. Both carry
# several intermediates the size of the cells they are given, so a
# million-row table done in one go would cost a multiple of the table
# itself. Neither result depends on how the cells are split -- the context
# is built per column, the ranks per cell -- so this bounds the transients
# and nothing else. `1 << 23`, as the reference has it.
TABPFN35_ECDF_CELL_BUDGET <- 8388608


# ---------------------------------------------------------------------------
# In-context ECDF
# ---------------------------------------------------------------------------
#
# The new metadata channel. For each cell, what fraction of the training
# rows in its column lie at or below it -- a midrank, so ties get the
# average of the ranks they span.
#
# Computing that exactly would mean keeping every training value per
# column, which grows the inference cache with the table. Instead each
# column is summarised once into at most `cell_ecdf_num_buckets` edges
# with the exact counts below and at each, and a query interpolates
# between the two edges that bracket it. A column with no more distinct
# values than there are buckets gets one edge per distinct value, so
# nothing is left between consecutive edges to interpolate over and the
# rank is exact.
#
# A caution that runs through all three functions: `torch_searchsorted()`
# returns *0-based* positions, as PyTorch does, while `torch_gather()`
# indexes from 1. Every searchsorted result used as a gather index is
# therefore `+ 1L`, and every one used as a count is not.

# Low-frequency sin/cos features of ECDF values in [0, 1]: `(...) -> (..., 2K)`.
#
# Half-period phases (`pi * k * u`, `k = 1..K`) so that `u = 0` and
# `u = 1` stay distinguishable at every frequency parity -- with full
# periods the first frequency would wrap them onto each other.
# @keywords internal
.tabpfn35_ecdf_fourier_features <- function(u, num_frequencies) {
  k <- torch::torch_arange(1L, as.integer(num_frequencies),
                           dtype = u$dtype, device = u$device)
  phase <- u$unsqueeze(-1L) * (pi * k)
  torch::torch_cat(list(torch::torch_sin(phase), torch::torch_cos(phase)),
                   dim = -1L)
}


# Summarise the training rows per (batch, column) into ECDF bucket edges.
#
# Returns `(3, B, C, K)` with `K = min(num_buckets, n_train)`: for each
# edge, its value, and the counts of training values strictly below it and
# at most equal to it. The counts are exact, so a query that *is* an edge
# gets the true midrank.
#
# Two rulers, because they answer different questions. Evenly spaced
# distinct indices hit every value a column has, but only while it has at
# most `K` of them. Above that the error is paid in mass rather than in
# distinct values: a column whose rows pile onto a few values inside a
# wide distinct range would starve exactly those values of edges. Row
# positions are mass-uniform by construction, so they take over there.
#
# @param x_BRiC `(B, Ri, C)` imputed values, training rows first.
# @keywords internal
.tabpfn35_build_ecdf_context <- function(x_BRiC, n_train, num_buckets) {
  rows_BNC <- if (n_train > 0L && n_train < x_BRiC$size(2))
    x_BRiC$narrow(2L, 1L, n_train) else x_BRiC
  columns <- rows_BNC$size(3)
  per_pass <- max(1L, as.integer(TABPFN35_ECDF_CELL_BUDGET / rows_BNC$size(2)))
  if (per_pass < columns) {
    parts <- lapply(seq(1L, columns, by = per_pass), function(s) {
      cj <- min(s + per_pass - 1L, columns) - s + 1L
      .tabpfn35_build_ecdf_context(x_BRiC$narrow(3L, s, cj), n_train, num_buckets)
    })
    return(torch::torch_cat(parts, dim = 3L))
  }

  sorted_BCN <- torch::torch_sort(
    rows_BNC$transpose(2L, 3L)$to(dtype = torch::torch_float32())$contiguous(),
    dim = -1L
  )[[1L]]
  n <- sorted_BCN$size(3)
  k <- min(as.integer(num_buckets), n)

  # Where each sorted position sits among the column's *distinct* values,
  # so a searchsorted over it maps a distinct index back to a row position.
  head_ <- sorted_BCN$narrow(3L, 1L, n - 1L)
  tail_ <- sorted_BCN$narrow(3L, 2L, n - 1L)
  is_new <- torch::torch_cat(
    list(torch::torch_ones_like(sorted_BCN$narrow(3L, 1L, 1L)),
         (tail_ != head_)$to(dtype = sorted_BCN$dtype)),
    dim = 3L
  )
  distinct_idx_BCN <- torch::torch_cumsum(is_new, dim = 3L) - 1
  num_distinct_BC1 <- distinct_idx_BCN$narrow(3L, n, 1L) + 1

  steps <- torch::torch_arange(0L, k - 1L, dtype = sorted_BCN$dtype,
                               device = sorted_BCN$device)
  if (k > 1L) {
    distinct_BCK <- steps * ((num_distinct_BC1 - 1) / (k - 1))
    # 0-based row positions; `+ 1L` only where they become a gather index.
    rows_K <- torch::torch_round(steps * ((n - 1) / (k - 1)))$
      to(dtype = torch::torch_long())
    by_rows <- torch::torch_gather(
      distinct_idx_BCN, -1L,
      (rows_K + 1L)$expand(c(sorted_BCN$size(1), sorted_BCN$size(2), k))
    )
    targets_BCK <- torch::torch_where(
      num_distinct_BC1 <= k, torch::torch_round(distinct_BCK), by_rows
    )$contiguous()
  } else {
    targets_BCK <- steps$expand(
      c(sorted_BCN$size(1), sorted_BCN$size(2), k))$contiguous()
  }

  below   <- torch::torch_searchsorted(distinct_idx_BCN, targets_BCK,
                                       right = FALSE)
  at_most <- torch::torch_searchsorted(distinct_idx_BCN, targets_BCK,
                                       right = TRUE)
  edges_BCK <- torch::torch_gather(sorted_BCN, -1L, below + 1L)
  torch::torch_stack(
    list(edges_BCK,
         below$to(dtype = sorted_BCN$dtype),
         at_most$to(dtype = sorted_BCN$dtype)),
    dim = 1L
  )
}


# Midrank of each value against the buckets, as a training-row count.
#
# @param values_BCRi `(B, C, R)` queries; @param ctx the `(3, B, C, K)`
#   context from [.tabpfn35_build_ecdf_context()].
# @keywords internal
.tabpfn35_ecdf_midrank_counts <- function(values_BCRi, ctx) {
  edges_BCK   <- ctx[1, , , ]
  below_BCK   <- ctx[2, , , ]
  at_most_BCK <- ctx[3, , , ]
  k <- edges_BCK$size(3)

  left  <- torch::torch_searchsorted(edges_BCK, values_BCRi, right = FALSE)
  right <- torch::torch_searchsorted(edges_BCK, values_BCRi, right = TRUE)

  # 0-based, as searchsorted returns them; `+ 1L` at every gather.
  lo_idx <- torch::torch_clamp(left - 1L, min = 0L)
  hi_idx <- torch::torch_clamp(left, max = k - 1L)
  edge_lo    <- torch::torch_gather(edges_BCK,   -1L, lo_idx + 1L)
  edge_hi    <- torch::torch_gather(edges_BCK,   -1L, hi_idx + 1L)
  at_most_lo <- torch::torch_gather(at_most_BCK, -1L, lo_idx + 1L)
  below_hi   <- torch::torch_gather(below_BCK,   -1L, hi_idx + 1L)
  at_most_hi <- torch::torch_gather(at_most_BCK, -1L, hi_idx + 1L)

  # The two ends coincide only outside the edge range, where the clamp has
  # already left the right count: `n` above the last edge, and 0 below the
  # first once the override at the end applies.
  width  <- edge_hi - edge_lo
  inside <- width > 0
  # Divide by 1 outside a bucket rather than masking the quotient
  # afterwards: `0 * Inf` is NaN, and prompt tuning optimises the cells, so
  # that NaN would reach them.
  weight <- torch::torch_where(
    inside, (values_BCRi - edge_lo) /
      torch::torch_where(inside, width, torch::torch_ones_like(width)),
    torch::torch_zeros_like(width)
  )
  counts  <- at_most_lo + weight * (below_hi - at_most_lo)
  is_edge <- right > left
  exact   <- 0.5 * (below_hi + at_most_hi)
  counts  <- torch::torch_where(is_edge, exact, counts)
  torch::torch_where((left == 0L)$logical_and(is_edge$logical_not()),
                     torch::torch_zeros_like(counts), counts)
}


# Midrank ECDF of every cell against the training rows, via the buckets.
#
# A value that is itself an edge gets that edge's exact midrank, which is
# how ties are handled. A value inside a bucket is interpolated between
# the counts its ends bracket -- an interval that is empty when the
# buckets hold every distinct training value, so the estimate is then
# exact too. Inputs must be finite: torch sorts NaN last, so a NaN query
# would come out at rank 1.
# @keywords internal
.tabpfn35_in_context_ecdf <- function(x_BRiC, ctx) {
  num_rows <- x_BRiC$size(2); columns <- x_BRiC$size(3)
  # The last edge is the column maximum, so its at-most count is the row count.
  n <- ctx[3, , , ]$narrow(3L, ctx$size(4), 1L)

  rank_rows <- function(rows_BRjC) {
    values_BCRj <- rows_BRjC$transpose(2L, 3L)$
      to(dtype = torch::torch_float32())$contiguous()
    counts <- .tabpfn35_ecdf_midrank_counts(values_BCRj, ctx)
    (counts / n)$transpose(2L, 3L)$to(dtype = x_BRiC$dtype)
  }

  per_pass <- max(1L, as.integer(TABPFN35_ECDF_CELL_BUDGET / columns))
  if (per_pass >= num_rows) return(rank_rows(x_BRiC))
  parts <- lapply(seq(1L, num_rows, by = per_pass), function(s) {
    len <- min(s + per_pass - 1L, num_rows) - s + 1L
    rank_rows(x_BRiC$narrow(2L, s, len))
  })
  torch::torch_cat(parts, dim = 2L)
}


# ---------------------------------------------------------------------------
# Cell embedding
# ---------------------------------------------------------------------------

#' Fourier-feature embedding of a grouped cell block
#'
#' Maps `(..., G) -> (..., E)`: each grouped value is expanded into
#' `[sin, cos]` against one learnable frequency bank shared by every cell,
#' the features are summed over the group, and a shared bias-free linear
#' projects the sum. Summing before the linear is exact -- `sum_g W f_g =
#' W sum_g f_g` -- and avoids materialising the per-group `(..., G, E)`
#' projection.
#'
#' No bias: the summed embedding is layer-normed, with a learnable bias,
#' by [tabpfn35_cell_embedder()].
#' @keywords internal
tabpfn35_fourier_group_embedder <- torch::nn_module(
  "TabPFN35FourierGroupEmbedder",

  initialize = function(group_size, embed_dim, num_freq) {
    self$frequencies <- torch::nn_parameter(
      torch::torch_randn(as.integer(group_size), as.integer(num_freq)) * 2
    )
    self$in_linear <- torch::nn_linear(as.integer(num_freq) * 2L,
                                       as.integer(embed_dim), bias = FALSE)
  },

  # @param x_G `(..., G)` grouped cell values.
  forward = function(x_G) {
    proj <- x_G$unsqueeze(-1L) * self$frequencies
    feats <- torch::torch_cat(
      list(torch::torch_sin(proj), torch::torch_cos(proj)), dim = -1L)
    self$in_linear(feats$sum(dim = -2L))
  }
)


#' The v3.5 cell embedder: Fourier over values, linear over metadata
#'
#' A drop-in replacement for v3's `nn_linear(G * 2, E)`, and the one place
#' the architectures genuinely diverge. The grouped tensor arrives as
#' three blocks of `group_size` channels each -- standard-scaled values,
#' NaN/Inf indicators, raw ECDF ranks, in that order -- and two `E`-wide
#' embeddings are summed:
#'
#' * the values through a [tabpfn35_fourier_group_embedder()];
#' * the whole metadata block through one bias-free linear. The ranks are
#'   lifted to `2K` sin/cos features each on the way in, which is why that
#'   linear is wider than the tensor the caller passes: `G * 2` value and
#'   indicator channels plus `G * 2K` rank features.
#'
#' A final LayerNorm puts the sum on the same scale as the target
#' embedding it is about to be added to.
#'
#' Rows are chunked because the two Fourier expansions transiently
#' materialise `(..., G, 2F)` and `(..., G, 2K)` per cell across the whole
#' row axis at once. Rows are independent here, so chunk-and-concatenate
#' is exact.
#' @keywords internal
tabpfn35_cell_embedder <- torch::nn_module(
  "TabPFN35CellEmbedder",

  initialize = function(group_size, embed_dim, num_freq, ecdf_num_frequencies,
                        row_chunk_size = 2048L) {
    g <- as.integer(group_size)
    self$group_size <- g
    self$ecdf_num_frequencies <- as.integer(ecdf_num_frequencies)
    self$row_chunk_size <- if (is.null(row_chunk_size)) NULL
                           else as.integer(row_chunk_size)
    self$fourier <- tabpfn35_fourier_group_embedder(
      group_size = g, embed_dim = embed_dim, num_freq = num_freq)
    # Each grouped value arrives with its NaN/Inf indicator, and one raw
    # ECDF rank per group position becomes `2K` features inside `embed()`.
    self$input_width    <- g * 3L
    self$metadata_width <- g * 2L + g * 2L * self$ecdf_num_frequencies
    self$metadata_linear <- torch::nn_linear(self$metadata_width,
                                             as.integer(embed_dim), bias = FALSE)
    self$layernorm <- torch::nn_layer_norm(as.integer(embed_dim),
                                           eps = TABPFN35_LN_EPS)
  },

  # @param x_grouped `(B, Ri, C, 3G)`; @return `(B, Ri, C, E)`.
  forward = function(x_grouped) {
    rc <- self$row_chunk_size
    if (is.null(rc) || x_grouped$size(2) <= rc) return(self$embed(x_grouped))
    n_rows <- x_grouped$size(2)
    parts <- lapply(seq(1L, n_rows, by = rc), function(s) {
      len <- min(s + rc - 1L, n_rows) - s + 1L
      self$embed(x_grouped$narrow(2L, s, len))
    })
    torch::torch_cat(parts, dim = 2L)
  },

  #' @keywords internal
  embed = function(x_grouped) {
    g <- self$group_size
    w <- x_grouped$size(-1L)
    fourier_out <- self$fourier(x_grouped$narrow(-1L, 1L, g))
    # The trailing `G` channels are the raw ranks; everything before them
    # is the values and their indicators. Lifting the ranks here rather
    # than before grouping bounds the `(..., G, 2K)` expansion by the row
    # chunk and keeps grouping to one channel per group position.
    ranks_G <- x_grouped$narrow(-1L, w - g + 1L, g)
    metadata <- torch::torch_cat(
      list(x_grouped$narrow(-1L, 1L, w - g),
           .tabpfn35_ecdf_fourier_features(
             ranks_G, self$ecdf_num_frequencies)$flatten(start_dim = -2L)),
      dim = -1L
    )
    self$layernorm(fourier_out + self$metadata_linear(metadata))
  }
)


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------

#' Scaled dot-product attention on `(B, S, H, D)` tensors
#'
#' As in v3: queries, keys and values are head-last and are permuted
#' inside, so the shapes in the callers line up with the reference's
#' suffix notation one for one. Grouped-query attention falls back to
#' `repeat_interleave`, which is what the reference does in float32 and is
#' exact either way.
#'
#' @param scaling Optional [tabpfn35_softmax_scaling_mlp()], applied to the
#'   queries with the key count as its length argument.
#' @keywords internal
tabpfn35_sdpa <- function(q, k, v, scaling = NULL) {
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
  sdpa(query = qh, key = kh, value = vh, dropout_p = 0)$
    permute(c(1L, 3L, 2L, 4L))
}


#' Query-aware attention temperature
#'
#' Unchanged from v3. Softmax attention sharpens as the key sequence
#' grows; the correction is learned rather than fixed at `1/sqrt(d)`.
#' Queries are scaled by `base_mlp(log n) * (1 + tanh(query_mlp(q)))`, and
#' the usual `1/sqrt(d)` is still applied afterwards by the attention.
#' @keywords internal
tabpfn35_softmax_scaling_mlp <- torch::nn_module(
  "TabPFN35SoftmaxScalingMLP",

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

  forward = function(q, n) {
    logn <- torch::torch_tensor(
      log(max(as.numeric(n), 1)), dtype = q$dtype, device = q$device
    )$reshape(c(1L, 1L))
    base <- self$base_mlp(logn)$view(c(1L, 1L, self$num_heads, self$head_dim))
    q * (base * (1 + torch::torch_tanh(self$query_mlp(q))))
  }
)


#' Multi-head self-attention with RoPE and QK-norm
#'
#' Used by the column aggregator, where the sequence axis is the feature
#' axis. The per-head RMSNorms are new in v3.5 and go *after* the rotation,
#' so they see the rotated queries and keys.
#' @keywords internal
tabpfn35_attention <- torch::nn_module(
  "TabPFN35Attention",

  initialize = function(embedding_size, num_heads, head_dim) {
    self$num_heads <- as.integer(num_heads)
    self$head_dim  <- as.integer(head_dim)
    inner <- self$num_heads * self$head_dim
    self$q_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$k_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$v_projection   <- torch::nn_linear(embedding_size, inner, bias = FALSE)
    self$out_projection <- torch::nn_linear(inner, embedding_size, bias = FALSE)
    self$q_norm <- rms_norm(self$head_dim, eps = TABPFN35_F32_EPS)
    self$k_norm <- rms_norm(self$head_dim, eps = TABPFN35_F32_EPS)
  },

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
    q <- self$q_norm(q); k <- self$k_norm(k)
    out <- tabpfn35_sdpa(q, k, v)
    self$out_projection(out$reshape(c(B, S, self$num_heads * self$head_dim)))
  }
)


#' Multi-head cross-attention with QK-norm
#'
#' Queries from one sequence, keys and values from another. Used in both
#' halves of an induced self-attention block.
#' @keywords internal
tabpfn35_cross_attention <- torch::nn_module(
  "TabPFN35CrossAttention",

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
    self$q_norm <- rms_norm(self$head_dim, eps = TABPFN35_F32_EPS)
    self$k_norm <- rms_norm(self$head_dim, eps = TABPFN35_F32_EPS)
  },

  forward = function(x_q, x_kv) {
    B <- x_q$size(1); Q <- x_q$size(2); V <- x_kv$size(2)
    q <- self$q_projection(x_q)$view(c(B, Q, -1L, self$head_dim))
    k <- self$k_projection(x_kv)$view(c(B, V, -1L, self$head_dim))
    v <- self$v_projection(x_kv)$view(c(B, V, -1L, self$head_dim))
    q <- self$q_norm(q); k <- self$k_norm(k)
    out <- tabpfn35_sdpa(
      q, k, v,
      scaling = if (self$has_scaling) self$softmax_scaling_layer else NULL
    )
    self$out_projection(out$reshape(c(B, Q, self$num_heads * self$head_dim)))
  }
)


#' In-context attention: every row attends to the training rows only
#'
#' The keys and values come from `x[1:single_eval_pos]`, so a test row can
#' see the training rows but neither itself nor the other test rows. No
#' mask is needed; the restriction is the slice.
#'
#' Test rows additionally use only the first `num_kv_heads_test` key/value
#' heads, which is what keeps [tabpfn35_kv_cache()] one head wide per
#' block.
#'
#' One ordering detail decides whether a cached prediction matches an
#' uncached one: `k_norm` is applied to the freshly projected keys
#' *before* they are cached, so the cached path must use them as they come
#' and must not norm them again.
#' @keywords internal
tabpfn35_icl_attention <- torch::nn_module(
  "TabPFN35ICLAttention",

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
    self$q_norm <- rms_norm(self$head_dim, eps = TABPFN35_F32_EPS)
    self$k_norm <- rms_norm(self$head_dim, eps = TABPFN35_F32_EPS)
  },

  # @param x `(B, R, E)`. In the cached path `R` counts test rows only.
  # @return `list(out, kv)`.
  forward = function(x, single_eval_pos, cached_kv = NULL, return_kv = FALSE) {
    B <- x$size(1); R <- x$size(2)
    scaling <- if (self$has_scaling) self$softmax_scaling_layer else NULL
    q <- self$q_norm(
      self$q_projection(x)$view(c(B, R, self$num_heads, self$head_dim))
    )

    if (!is.null(cached_kv)) {
      # The cached keys are already normed; norming them twice would be a
      # silently different model.
      out <- tabpfn35_sdpa(q, cached_kv$key, cached_kv$value, scaling = scaling)
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
    k <- self$k_norm(k)

    n_test_heads <- self$num_kv_heads_test
    if (!is.null(n_test_heads) && n_ctx < R) {
      out_train <- tabpfn35_sdpa(q[, 1:n_ctx, , ], k, v, scaling = scaling)
      out_test  <- tabpfn35_sdpa(
        q[, (n_ctx + 1L):R, , ], k[, , 1:n_test_heads, ], v[, , 1:n_test_heads, ],
        scaling = scaling
      )
      out <- torch::torch_cat(list(out_train, out_test), dim = 2L)
    } else {
      out <- tabpfn35_sdpa(q, k, v, scaling = scaling)
    }

    kv <- NULL
    if (isTRUE(return_kv)) {
      # Only the heads a test row can reach are worth keeping.
      # `$contiguous()` so the slice owns its storage and the full
      # projection can be freed.
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
tabpfn35_mlp <- function(emsize, dim_feedforward) {
  torch::nn_sequential(
    torch::nn_linear(emsize, dim_feedforward, bias = FALSE),
    torch::nn_gelu(),
    torch::nn_linear(dim_feedforward, emsize, bias = FALSE)
  )
}


#' Pre-norm cross-attention block
#'
#' Query and key/value streams are normalised separately, and each of the
#' two sublayers adds its own residual.
#' @keywords internal
tabpfn35_cross_attention_block <- torch::nn_module(
  "TabPFN35CrossAttentionBlock",

  initialize = function(emsize, nhead, dim_feedforward,
                        softmax_scaling_layer = NULL, eps = NULL) {
    if (is.null(eps)) eps <- TABPFN35_F32_EPS
    self$attn <- tabpfn35_cross_attention(
      embedding_size = emsize, num_heads = nhead,
      head_dim = as.integer(emsize / nhead),
      softmax_scaling_layer = softmax_scaling_layer
    )
    self$mlp <- tabpfn35_mlp(emsize, dim_feedforward)
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
tabpfn35_transformer_block <- torch::nn_module(
  "TabPFN35TransformerBlock",

  initialize = function(emsize, nhead, dim_feedforward, eps = NULL) {
    if (is.null(eps)) eps <- TABPFN35_F32_EPS
    self$attention <- tabpfn35_attention(
      embedding_size = emsize, num_heads = nhead,
      head_dim = as.integer(emsize / nhead)
    )
    self$layernorm     <- rms_norm(emsize, eps = eps)
    self$layernorm_mlp <- rms_norm(emsize, eps = eps)
    self$mlp <- tabpfn35_mlp(emsize, dim_feedforward)
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
    # QK-norm after the rotation here too, as in `tabpfn35_attention()`.
    q <- a$q_norm(q); k <- a$k_norm(k)
    out <- tabpfn35_sdpa(q, k, v)$reshape(c(B * R, Q, a$num_heads * a$head_dim))
    x <- query + a$out_projection(out)$view(c(B, R, Q, E))
    x + self$mlp(self$layernorm_mlp(x))
  }
)


#' One block of the in-context-learning transformer
#'
#' Pre-norm attention across rows with training-only keys, then a per-row
#' MLP. This is where most of the model's parameters live.
#' @keywords internal
tabpfn35_icl_block <- torch::nn_module(
  "TabPFN35ICLBlock",

  initialize = function(emsize, nhead, dim_feedforward,
                        softmax_scaling_layer = NULL,
                        num_kv_heads = NULL, num_kv_heads_test = NULL,
                        eps = NULL) {
    if (is.null(eps)) eps <- TABPFN35_F32_EPS
    self$icl_attention <- tabpfn35_icl_attention(
      embedding_size = emsize, num_heads = nhead,
      head_dim = as.integer(emsize / nhead),
      softmax_scaling_layer = softmax_scaling_layer,
      num_kv_heads = num_kv_heads, num_kv_heads_test = num_kv_heads_test
    )
    self$layernorm     <- rms_norm(emsize, eps = eps)
    self$layernorm_mlp <- rms_norm(emsize, eps = eps)
    self$mlp <- tabpfn35_mlp(emsize, dim_feedforward)
  },

  # @param x `(B, R, E)`. @return `list(state, kv)`.
  forward = function(x, single_eval_pos, cached_kv = NULL, return_kv = FALSE,
                     save_peak_memory_factor = NULL) {
    att <- self$icl_attention
    ln  <- self$layernorm
    kv  <- NULL
    if (isTRUE(return_kv)) {
      # Building a cache bypasses chunking here: the key/value tensors have
      # to be produced whole, not a slice at a time. The MLP below is
      # chunked regardless, so a factor still reaches most of the block.
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
#' fixed set of learned inducing vectors attends to the training rows, and
#' every row attends back to that summary. Two cross-attentions, both
#' linear in the row count.
#'
#' The summary is a function of the training rows alone, which is what
#' makes it cacheable.
#' @keywords internal
tabpfn35_induced_self_attention_block <- torch::nn_module(
  "TabPFN35InducedSelfAttentionBlock",

  initialize = function(emsize, nhead, num_inducing_points, dim_feedforward,
                        softmax_scaling_layer = NULL) {
    self$cross_attn_block1 <- tabpfn35_cross_attention_block(
      emsize = emsize, nhead = nhead, dim_feedforward = dim_feedforward,
      softmax_scaling_layer = softmax_scaling_layer
    )
    self$cross_attn_block2 <- tabpfn35_cross_attention_block(
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
      # Both run whole. Chunking splits along `B * C` and each chunk would
      # be handed the *whole* hidden state, so a factor here would be a
      # shape mismatch rather than a smaller peak. What bounds this stage
      # on a large table is `stages_0_to_2()`'s row chunking, which caps
      # `R` before the fold into `B * C` ever happens.
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
#' A stack of [tabpfn35_induced_self_attention_block()]s. Each column's
#' cells are a sequence; what the stack learns is that column's empirical
#' distribution.
#' @keywords internal
tabpfn35_feature_distribution_embedder <- torch::nn_module(
  "TabPFN35FeatureDistributionEmbedder",

  initialize = function(emsize, nhead, num_inducing_points, dim_feedforward,
                        num_layers, softmax_scaling_layer_factory = NULL) {
    self$layers <- torch::nn_module_list(
      lapply(seq_len(as.integer(num_layers)), function(i) {
        tabpfn35_induced_self_attention_block(
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
#' row embedding the ICL stage works on -- eight tokens of 128 in v3.5,
#' which is where the ICL width of 1024 comes from.
#'
#' RoPE over the feature axis is the only thing that tells the model which
#' column is which, and in v3.5 it is no longer optional.
#' @keywords internal
tabpfn35_column_aggregator <- torch::nn_module(
  "TabPFN35ColumnAggregator",

  initialize = function(emsize, nhead, num_layers, dim_feedforward,
                        num_cls_tokens, rope_base = 100000, eps = NULL) {
    if (is.null(eps)) eps <- TABPFN35_F32_EPS
    self$embed_dim      <- as.integer(emsize)
    self$num_cls_tokens <- as.integer(num_cls_tokens)
    self$blocks <- torch::nn_module_list(
      lapply(seq_len(as.integer(num_layers)), function(i) {
        tabpfn35_transformer_block(
          emsize = emsize, nhead = nhead, dim_feedforward = dim_feedforward
        )
      })
    )
    # `freqs` is a non-trainable parameter in the reference, not a buffer,
    # so it is a parameter here too and the strict loader finds it where
    # the checkpoint puts it.
    self$rope <- rope(dim = as.integer(emsize / nhead), base = rope_base,
                      interleaved = FALSE, learnable = TRUE)
    self$cls_tokens <- torch::nn_parameter(
      torch::torch_empty(self$num_cls_tokens, emsize)
    )
    torch::nn_init_trunc_normal_(self$cls_tokens, std = 0.02)
    self$out_ln <- rms_norm(emsize, eps = eps)
  },

  # @param x `(B, Ri, C, E)`; @return `(B, Ri, num_cls_tokens, E)`.
  forward = function(x, save_peak_memory_factor = NULL) {
    B <- x$size(1); Ri <- x$size(2); E <- x$size(4)
    r <- self$rope
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
# Target encoders
# ---------------------------------------------------------------------------

#' Learned per-class embedding
#'
#' Wraps `nn_embedding` so the parameter lands at `<name>.embedding.weight`,
#' where the checkpoint puts it. Labels arrive 0-based, as the reference's
#' ordinal encoding produces them, and R torch's `nn_embedding` indexes
#' from one.
#' @keywords internal
tabpfn35_class_embedding <- torch::nn_module(
  "TabPFN35ClassEmbedding",

  initialize = function(num_classes, embed_dim) {
    self$embedding <- torch::nn_embedding(as.integer(num_classes),
                                          as.integer(embed_dim))
  },

  forward = function(y) {
    self$embedding(y$to(dtype = torch::torch_long()) + 1L)
  }
)


#' Both target encoders, side by side
#'
#' The multitask change in miniature. A v3 checkpoint carries one target
#' encoder, chosen when the weights were trained; a v3.5 checkpoint
#' carries both, and the forward pass picks. Both are built whatever the
#' task, because both are in every checkpoint and the loader is strict.
#'
#' Named to match the reference's `nn.ModuleDict`, so the parameters land
#' at `<name>.multiclass.embedding.weight` and `<name>.regression.weight`.
#' @keywords internal
tabpfn35_y_encoder <- torch::nn_module(
  "TabPFN35YEncoder",

  initialize = function(max_num_classes, embed_dim) {
    self$multiclass <- tabpfn35_class_embedding(max_num_classes, embed_dim)
    self$regression <- torch::nn_linear(1L, as.integer(embed_dim), bias = TRUE)
  },

  # @param y_BN `(B, N)` targets; @param task_type the branch to take.
  forward = function(y_BN, task_type) {
    if (identical(task_type, "multiclass")) return(self$multiclass(y_BN))
    self$regression(y_BN$unsqueeze(-1L))
  }
)


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

#' Residual pre-norm MLP in front of an output head
#'
#' New in v3.5: each head gets one of these before its final projection.
#' The reference zero-initialises the inner MLP's last projection so the
#' block starts as the identity; that matters only for training, so it is
#' not reproduced.
#' @keywords internal
tabpfn35_pre_head_mlp <- torch::nn_module(
  "TabPFN35PreHeadMLP",

  initialize = function(emsize, dim_feedforward, eps = NULL) {
    if (is.null(eps)) eps <- TABPFN35_F32_EPS
    self$norm <- rms_norm(emsize, eps = eps)
    self$mlp  <- tabpfn35_mlp(emsize, dim_feedforward)
  },

  forward = function(x) x + self$mlp(self$norm(x))
)


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
tabpfn35_many_class_decoder <- torch::nn_module(
  "TabPFN35ManyClassDecoder",

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

  # @param train_emb `(B, N, E)`, already through the classification
  #   pre-head MLP; @param test_emb `(B, M, E)` likewise.
  # @param targets `(B, N)` ordinal class indices, 0-based, *raw* -- a
  #   non-finite one contributes nothing rather than being imputed.
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

    # The reference narrows the one-hot to the classes actually present and
    # pads the rest back with zeros. That is the same number: an absent
    # class holds an all-zero value column, and attention over zeros
    # returns exactly the zero the padding would have written. The full
    # width is used here because it needs no label scan.
    is_finite <- torch::torch_isfinite(targets)
    safe <- torch::torch_where(is_finite, targets,
                               torch::torch_zeros_like(targets))
    classes <- torch::torch_arange(
      0L, self$max_num_classes - 1L, dtype = safe$dtype, device = safe$device
    )$view(c(1L, 1L, self$max_num_classes))
    one_hot <- (safe$unsqueeze(-1L) == classes)$
      logical_and(is_finite$unsqueeze(-1L))$to(dtype = w$dtype)  # (B, N, T)

    probs <- torch::torch_matmul(
      w, one_hot$unsqueeze(2L)$expand(c(B, H, n_ctx, self$max_num_classes))
    )$mean(dim = 2L)                                         # (B, M, T)
    torch::torch_log(torch::torch_clamp(probs, min = 1e-5) + 3e-5)
  }
)


#' Both output heads, bundled
#'
#' Mirrors the reference's `MultiTaskHeads`, and is why the v3.5
#' checkpoint's decoder keys are under `heads.` rather than at the top
#' level. Classification runs its pre-head MLP and then the retrieval
#' decoder; regression runs its own pre-head MLP and then a single linear
#' onto the bar-distribution logits -- v3 had a three-layer stack there,
#' and the pre-head MLP has taken over that work.
#' @keywords internal
tabpfn35_heads <- torch::nn_module(
  "TabPFN35Heads",

  initialize = function(input_size, max_num_classes, num_buckets,
                        decoder_head_dim, decoder_num_heads,
                        decoder_softmax_scaling_layer = NULL,
                        mlp_dim_feedforward) {
    self$many_class_decoder <- tabpfn35_many_class_decoder(
      max_num_classes = max_num_classes, input_size = input_size,
      head_dim = decoder_head_dim, num_heads = decoder_num_heads,
      softmax_scaling_layer = decoder_softmax_scaling_layer
    )
    self$output_projection <- torch::nn_linear(input_size,
                                               as.integer(num_buckets))
    self$mlp_classification <- tabpfn35_pre_head_mlp(input_size,
                                                     mlp_dim_feedforward)
    self$mlp_regression     <- tabpfn35_pre_head_mlp(input_size,
                                                     mlp_dim_feedforward)
    self$register_buffer("regression_borders",
                         torch::torch_zeros(as.integer(num_buckets) + 1L))
  },

  # @param train_emb `(B, N, D)` post-norm training row embeddings, or
  #   NULL for regression, which has no decoder.
  # @param test_emb `(B, M, D)`; @param y_BN `(B, N)` raw targets.
  # @return `(B, M, n_out)` for the branch `task_type` selects.
  forward = function(train_emb, test_emb, y_BN, task_type) {
    if (identical(task_type, "regression")) {
      out <- self$mlp_regression(test_emb)
      # The reference projects in (rows, batch, embedding) order; keep the
      # same layout so the matmul sees the same shapes.
      return(self$output_projection(out$transpose(1L, 2L))$transpose(1L, 2L))
    }
    # Both streams go through the *same* pre-head MLP before the decoder
    # projects them -- the reference builds its cached decoder keys from
    # `mlp_classification(train_emb)`, not from the raw embeddings.
    self$many_class_decoder(self$mlp_classification(train_emb),
                            self$mlp_classification(test_emb), y_BN)
  }
)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

# The signed indicator channel: -2 for NaN, +2 for +Inf, +4 for -Inf.
# @keywords internal
tabpfn35_nan_inf_indicator <- function(x) {
  is_inf <- torch::torch_isinf(x)
  pos <- is_inf$logical_and(torch::torch_sign(x) == 1)
  neg <- is_inf$logical_and(torch::torch_sign(x) == -1)
  torch::torch_isnan(x)$to(dtype = x$dtype) * TABPFN35_NAN_INDICATOR +
    pos$to(dtype = x$dtype) * TABPFN35_POS_INF_INDICATOR +
    neg$to(dtype = x$dtype) * TABPFN35_NEG_INF_INDICATOR
}

# Everything the training rows fix about the input scale: the per-feature
# mean and standard deviation. `x` is `(Ri, B, C)`. Reuses v2.6's NaN-aware
# reductions, which have the same semantics as
# `tabpfn.preprocessing.torch.ops`.
# @keywords internal
tabpfn35_fit_scaler <- function(x, n_train) {
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
#' It does not have to. The training rows reach a test row through exactly
#' five channels in v3.5, and all five are functions of the training rows
#' alone:
#'
#' * the scaler statistics fitted on them;
#' * the ECDF bucket context each column is ranked against -- new in
#'   v3.5, and the reason a cached prediction still sees the same ranks
#'   the uncached one would have produced;
#' * the inducing summary of each distribution-embedder block, `n_ind`
#'   vectors per column;
#' * the key/value projections of each ICL block, one head wide because a
#'   test row attends to one head;
#' * the post-norm training row embeddings, which the classifier's
#'   retrieval decoder reads.
#'
#' As in v3, and unlike v2.6's cache, this one changes nothing: nothing is
#' fitted on test rows, so a cached prediction and an uncached one are the
#' same computation.
#'
#' @param kv List of `list(key, value)`, one per ICL block. The keys are
#'   post-`k_norm`.
#' @param scaler Fitted `list(mean, std)`.
#' @param ecdf_context The `(3, B, C, K)` bucket context.
#' @param inducing_hidden List of inducing summaries, one per
#'   distribution-embedder block.
#' @param train_embeddings `(B, n_train, D)` post-norm row embeddings.
#' @param y_train `(B, n_train)` targets, for the classifier's decoder.
#' @param n_train Number of training rows it was built from.
#' @keywords internal
tabpfn35_kv_cache <- function(kv, scaler, ecdf_context, inducing_hidden,
                              train_embeddings, y_train, n_train) {
  structure(
    list(kv = kv, scaler = scaler, ecdf_context = ecdf_context,
         inducing_hidden = inducing_hidden,
         train_embeddings = train_embeddings, y_train = y_train,
         n_train = as.integer(n_train)),
    class = "tabpfn35_kv_cache"
  )
}

#' @export
print.tabpfn35_kv_cache <- function(x, ...) {
  bytes <- function(t) prod(as.numeric(t$size())) * 4
  n_kv <- sum(vapply(x$kv, function(e) sum(vapply(e, bytes, numeric(1))),
                     numeric(1)))
  n_ind <- sum(vapply(x$inducing_hidden, bytes, numeric(1)))
  n_emb <- bytes(x$train_embeddings)
  n_ecdf <- bytes(x$ecdf_context)
  cli::cli_text("{.strong TabPFN v3.5 KV cache}")
  cli::cli_bullets(c(
    "*" = "built from {.val {x$n_train}} training row{?s}",
    "*" = "{length(x$kv)} ICL block{?s}, {round(n_kv / 1e6, 1)} MB of key/value projections",
    "*" = "{length(x$inducing_hidden)} inducing summar{?y/ies}, {round(n_ind / 1e6, 1)} MB",
    "*" = "{round(n_ecdf / 1e6, 1)} MB of ECDF bucket context",
    "*" = "{round(n_emb / 1e6, 1)} MB of training row embeddings"
  ))
  invisible(x)
}


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------

#' TabPFN v3.5 network
#'
#' Mirrors `TabPFNV3p5.forward` in the Python reference.
#'
#' @param config Parsed `config.json` with `arch == "tabpfn_v3_5"`.
#' @param task `"classification"` or `"regression"`. Unlike every earlier
#'   generation this is not in the checkpoint -- one set of weights serves
#'   both -- so it is chosen here and fixes which target encoder and which
#'   head the forward pass runs. The module tree is the same either way,
#'   because every checkpoint carries both.
#' @keywords internal
tabpfn_v35_transformer <- torch::nn_module(
  "TabPFNv35",

  initialize = function(config, task = "classification") {
    E <- as.integer(config$embed_dim %||% 128L)
    n_cls <- as.integer(config$feat_agg_num_cls_tokens %||% 8L)
    ff <- as.integer(config$ff_factor %||% 2L)
    self$embed_dim  <- E
    self$icl_emsize <- E * n_cls
    self$task <- task
    self$task_type <- if (identical(task, "regression"))
      "regression" else "multiclass"
    self$feature_group_size <- as.integer(config$feature_group_size %||% 3L)
    self$max_num_classes <- as.integer(config$max_num_classes %||% 160L)
    self$num_buckets <- as.integer(config$num_buckets %||% 5000L)
    self$n_out <- if (identical(self$task_type, "multiclass"))
      self$max_num_classes else self$num_buckets
    self$ecdf_num_buckets <- as.integer(config$cell_ecdf_num_buckets %||% 8192L)
    ss_hidden <- as.integer(config$softmax_scaling_mlp_hidden_dim %||% 64L)

    # --- Stage 0: the Fourier + metadata cell embedder. Three channels per
    # cell go in -- value, NaN/Inf indicator, ECDF rank -- where v3 had two
    # and a plain linear.
    self$x_embed <- tabpfn35_cell_embedder(
      group_size = self$feature_group_size, embed_dim = E,
      num_freq = as.integer(config$fourier_encoding_num_frequencies %||% 32L),
      ecdf_num_frequencies = as.integer(config$cell_ecdf_num_frequencies %||% 4L),
      row_chunk_size = config$cell_embed_row_chunk_size %||% 2048L
    )

    # --- The target enters twice: once per column in stage 1, once per row
    # in stage 3. Both encoders carry a classification and a regression
    # branch, and both are followed by a LayerNorm -- new in v3.5, and what
    # puts the target embedding on the same scale as the cell embedding it
    # is added to.
    self$col_y_encoder <- tabpfn35_y_encoder(self$max_num_classes, E)
    self$col_y_layernorm <- torch::nn_layer_norm(E, eps = TABPFN35_LN_EPS)
    self$icl_y_encoder <- tabpfn35_y_encoder(self$max_num_classes,
                                             self$icl_emsize)
    self$icl_y_layernorm <- torch::nn_layer_norm(self$icl_emsize,
                                                 eps = TABPFN35_LN_EPS)

    # --- Stage 1.
    dist_heads <- as.integer(config$dist_embed_num_heads %||% 8L)
    self$feature_distribution_embedder <- tabpfn35_feature_distribution_embedder(
      emsize = E, nhead = dist_heads,
      num_inducing_points = as.integer(config$dist_embed_num_inducing_points %||% 128L),
      dim_feedforward = E * ff,
      num_layers = as.integer(config$dist_embed_num_blocks %||% 3L),
      softmax_scaling_layer_factory = function() {
        tabpfn35_softmax_scaling_mlp(
          num_heads = dist_heads, head_dim = as.integer(E / dist_heads),
          n_hidden = ss_hidden
        )
      }
    )

    # --- Stage 2.
    self$column_aggregator <- tabpfn35_column_aggregator(
      emsize = E, nhead = as.integer(config$feat_agg_num_heads %||% 8L),
      num_layers = as.integer(config$feat_agg_num_blocks %||% 3L),
      dim_feedforward = E * ff, num_cls_tokens = n_cls,
      rope_base = as.numeric(config$feat_agg_rope_base %||% 100000)
    )

    # --- Stage 3.
    icl_heads <- as.integer(config$icl_num_heads %||% 16L)
    # `[[` and not `$`: v3.5's checkpoints omit `icl_num_kv_heads`
    # entirely, and R's `$` on a list partial-matches, so `config$` would
    # silently return `icl_num_kv_heads_test`'s 1 and build every ICL
    # block with a single KV head -- a checkpoint that then fails to load,
    # or worse, a different model that does. v3 was never exposed to this
    # because its converter writes the key explicitly, as `null`.
    n_kv      <- config[["icl_num_kv_heads"]]
    n_kv_test <- config[["icl_num_kv_heads_test"]]
    self$icl_blocks <- torch::nn_module_list(
      lapply(seq_len(as.integer(config$nlayers %||% 24L)), function(i) {
        tabpfn35_icl_block(
          emsize = self$icl_emsize, nhead = icl_heads,
          dim_feedforward = self$icl_emsize * ff,
          num_kv_heads = if (is.null(n_kv)) NULL else as.integer(n_kv),
          num_kv_heads_test = if (is.null(n_kv_test)) NULL else as.integer(n_kv_test),
          softmax_scaling_layer = tabpfn35_softmax_scaling_mlp(
            num_heads = icl_heads,
            head_dim = as.integer(self$icl_emsize / icl_heads),
            n_hidden = ss_hidden
          )
        )
      })
    )
    self$output_norm <- rms_norm(self$icl_emsize, eps = TABPFN35_F32_EPS)

    # --- Both heads, always. The forward pass picks; the loader needs all
    # of it either way.
    dec_heads <- as.integer(config$decoder_num_heads %||% 6L)
    dec_dim   <- as.integer(config$decoder_head_dim %||% 64L)
    self$heads <- tabpfn35_heads(
      input_size = self$icl_emsize,
      max_num_classes = self$max_num_classes,
      num_buckets = self$num_buckets,
      decoder_head_dim = dec_dim, decoder_num_heads = dec_heads,
      decoder_softmax_scaling_layer =
        if (isTRUE(config$decoder_use_softmax_scaling %||% TRUE)) {
          tabpfn35_softmax_scaling_mlp(num_heads = dec_heads, head_dim = dec_dim,
                                       n_hidden = ss_hidden)
        } else NULL,
      mlp_dim_feedforward = self$icl_emsize * ff
    )
    # The shared predictor helpers look for the bar-distribution borders on
    # the module; in v3.5 they live one level down, under `heads`. Mirror
    # the reference's `regression_borders` property so
    # `.regression_borders_of()` finds them without knowing that.
    self$regression_borders <- self$heads$regression_borders

    # The reference's own stage-0-2 chunk sizes, which it applies by default
    # on this architecture. Carried on the module so a caller that says
    # nothing gets what the Python estimator would do.
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
  #   and `y_train` are ignored and `x_test` holds the rows to predict.
  # @param return_kv_cache Build and return one. `x_test` may be empty.
  # @param save_peak_memory_factor Chunk count for [chunked_evaluate()].
  # @param row_chunk_size,col_chunk_size Stage-0-2 chunking. Left alone
  #   they take the checkpoint's own values; `NULL` runs every row at once.
  forward = function(x_train, y_train, x_test, kv_cache = NULL,
                     return_kv_cache = FALSE, save_peak_memory_factor = NULL,
                     row_chunk_size = NA_integer_, col_chunk_size = NA_integer_) {
    row_chunk_size <- .tabpfn35_chunk_arg(row_chunk_size,
                                          self$inference_row_chunk_size)
    col_chunk_size <- .tabpfn35_chunk_arg(col_chunk_size,
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
        "The TabPFN v3.5 backend runs one dataset at a time.",
        i = "Got a batch of {B}; call it once per dataset."
      ))
    }
    if (n_train == 0L) {
      cli::cli_abort("TabPFN v3.5 needs at least one training row to condition on.")
    }
    if (identical(self$task_type, "multiclass")) {
      # NaN-to-zero before the max, as the reference does, so an imputed
      # label cannot make the range check pass or fail on a NaN.
      finite_y <- torch::torch_where(torch::torch_isnan(y_train),
                                     torch::torch_zeros_like(y_train), y_train)
      hi <- as.numeric(finite_y$max()$cpu())
      lo <- as.logical((y_train < 0)$any()$cpu())
      if (hi + 1 > self$max_num_classes || lo) {
        cli::cli_abort(c(
          "Target out of range for a {self$max_num_classes}-class head.",
          i = "Labels must be ordinal-encoded in \\
               {.val {0}}..{.val {self$max_num_classes - 1L}}."
        ))
      }
    }

    # The reference works in (rows, batch, columns) until stage 0 ends.
    x_RiBC <- torch::torch_cat(list(x_train, x_test), dim = 2L)$
      transpose(1L, 2L)$contiguous()
    dump_if_enabled("input_x_raw", x_RiBC)

    # Two versions of the target. The decoder reads the raw one and masks
    # the non-finite entries itself; the embedding path reads the imputed
    # one, because an embedding table has no way to represent a gap.
    y_raw_BN <- y_train$reshape(c(B, n_train))
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
    x <- .tabpfn35_add_to_train_rows(x, self$embed_icl_targets(y_BN), n_train)
    dump_if_enabled("icl_input", x)

    icl <- self$run_icl(x, single_eval_pos = n_train,
                        return_kv = isTRUE(return_kv_cache),
                        save_peak_memory_factor = save_peak_memory_factor)
    x <- self$output_norm(icl$state)
    dump_if_enabled("icl_out", x)

    train_emb <- x[, 1:n_train, ]
    test_emb  <- if (n_test > 0L) x[, (n_train + 1L):(n_train + n_test), ] else NULL

    res <- list(
      logits = if (n_test > 0L) self$decode(train_emb, test_emb, y_raw_BN) else NULL,
      test_hidden = test_emb, train_hidden = train_emb
    )
    if (isTRUE(return_kv_cache)) {
      res$kv_cache <- tabpfn35_kv_cache(
        kv = icl$kv, scaler = s02$scaler, ecdf_context = s02$ecdf_context,
        inducing_hidden = s02$hidden,
        train_embeddings = train_emb$detach(), y_train = y_raw_BN$detach(),
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
    row_chunk_size <- .tabpfn35_chunk_arg(row_chunk_size,
                                          self$inference_row_chunk_size)
    col_chunk_size <- .tabpfn35_chunk_arg(col_chunk_size,
                                          self$inference_col_chunk_size)
    B <- x_test$size(1); n_test <- x_test$size(2)
    if (B != 1L) {
      cli::cli_abort("The TabPFN v3.5 backend runs one dataset at a time.")
    }
    x_RiBC <- x_test$transpose(1L, 2L)$contiguous()

    # `n_train = 0` here is not "no training rows": the scaler carries
    # their statistics, the ECDF context carries their distribution, and
    # both are what a test cell is read against. Every row is a test row,
    # so there is no target to embed and the inducing summaries come from
    # the cache rather than from this pass.
    s02 <- self$stages_0_to_2(
      x_RiBC, y_BN = NULL, n_train = 0L, scaler = cache$scaler,
      ecdf_context = cache$ecdf_context,
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
  #' the cell embedding the state is only `(B, Ri, C, 3G)`, `3G` = 9, so
  #' the loop starts there, and what survives it is `(B, Ri, n_cls, E)`.
  #'
  #' @param x_RiBC `(Ri, B, C)` raw input, train rows first.
  #' @param y_BN `(B, N)` cleaned training targets, or `NULL`.
  #' @param scaler,ecdf_context Fitted statistics to reuse, or `NULL` to
  #'   fit them. They travel together: the ECDF context is fitted on the
  #'   same imputed training rows the scaler is.
  #' @return `list(state, hidden, scaler, ecdf_context)`.
  #' @keywords internal
  stages_0_to_2 = function(x_RiBC, y_BN, n_train, scaler = NULL,
                           ecdf_context = NULL,
                           cached_hidden = NULL, return_hidden = FALSE,
                           row_chunk_size = NULL, col_chunk_size = NULL,
                           save_peak_memory_factor = NULL) {
    pre <- self$preprocess(x_RiBC, n_train = n_train, scaler = scaler,
                           ecdf_context = ecdf_context)
    grouped <- self$group_features(pre$x_BRiC, pre$indicators, pre$ecdf)
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
      hidden <- self$all_inducing_hidden(grouped, y_col, n_train, col_chunk_size)
    }
    # "Full" is the path that owns its own summaries: one pass over every
    # row, computing them as it goes. Only there can a block be asked to
    # hand them back.
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
        x_emb <- .tabpfn35_add_to_train_rows(
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
      if (use_chunks) collect_between_layers(x_emb)
    }

    list(
      state = if (j == 1L) parts[[1L]] else torch::torch_cat(parts, dim = 2L),
      hidden = if (use_chunks) hidden else own_hidden,
      scaler = pre$scaler,
      ecdf_context = pre$ecdf_context
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
      collect_between_layers(x)
    }
    list(state = x, kv = kv)
  },

  #' Turn training and test row embeddings into per-row outputs.
  #' @keywords internal
  decode = function(train_emb, test_emb, y_BN) {
    out <- self$heads(train_emb, test_emb, y_BN, self$task_type)
    # `_nan_safe_output`: an all-NaN column, or a class no training row
    # carries, can leave a NaN here. The reference zeroes them rather than
    # letting one propagate into the softmax and take the row with it.
    out <- torch::torch_where(torch::torch_isnan(out),
                              torch::torch_zeros_like(out), out)
    dump_if_enabled("logits", out)
    out
  },

  #' Indicator capture, mean imputation, ECDF ranking, standard scaling.
  #'
  #' The order matters and is not the obvious one:
  #'
  #' 1. indicators are read off the raw input;
  #' 2. every non-finite cell is replaced with the training nanmean;
  #' 3. the scaler is fitted on *that*, so its statistics stay finite even
  #'    when the raw input carried infinities;
  #' 4. the non-finite cells are filled again, now from the scaler's mean.
  #'    Mathematically the same number as step 2 -- the mean of a column
  #'    filled with its own mean is that mean -- but not the same float,
  #'    and the difference is enough to move a filled test cell out of the
  #'    tie block it belongs in. The cached path fills from the scaler's
  #'    mean, so the uncached one has to as well;
  #' 5. the ECDF is ranked against the *imputed* values, not the scaled
  #'    ones: the scaler's +/-100 clip would collapse extreme outliers into
  #'    a tie;
  #' 6. only then the standardisation.
  #' @keywords internal
  preprocess = function(x_RiBC, n_train, scaler = NULL, ecdf_context = NULL) {
    indicators <- tabpfn35_nan_inf_indicator(x_RiBC)$transpose(1L, 2L)

    is_finite <- torch::torch_isfinite(x_RiBC)
    fitting <- is.null(scaler)
    means <- if (!fitting) scaler$mean else {
      fit <- if (n_train > 0L && n_train < x_RiBC$size(1))
        x_RiBC[1:n_train, , ] else x_RiBC
      # Ignoring Inf as well as NaN: an infinite cell is about to be
      # replaced, so it must not set the value it is replaced with.
      tabpfn26_nanmean(fit, include_inf = TRUE)
    }
    x <- torch::torch_where(is_finite, x_RiBC,
                            means$unsqueeze(1L)$expand_as(x_RiBC))
    if (fitting) {
      scaler <- tabpfn35_fit_scaler(x, n_train)
      # Step 4: re-fill from the fitted mean, so train and cached test rows
      # agree to the last bit.
      x <- torch::torch_where(is_finite, x,
                              scaler$mean$unsqueeze(1L)$expand_as(x))
    }

    x_imputed_BRiC <- x$transpose(1L, 2L)
    if (is.null(ecdf_context)) {
      ecdf_context <- .tabpfn35_build_ecdf_context(x_imputed_BRiC, n_train,
                                                   self$ecdf_num_buckets)
    }
    ecdf <- .tabpfn35_in_context_ecdf(x_imputed_BRiC, ecdf_context)
    dump_if_enabled("ecdf", ecdf)

    x <- (x - scaler$mean$unsqueeze(1L)) /
      (scaler$std$unsqueeze(1L) + TABPFN35_F32_EPS)
    x <- torch::torch_clamp(x, min = -100, max = 100)
    dump_if_enabled("preproc_x", x)
    list(x_BRiC = x$transpose(1L, 2L), indicators = indicators, ecdf = ecdf,
         scaler = scaler, ecdf_context = ecdf_context)
  },

  #' Group the columns, without embedding them.
  #'
  #' Every column becomes a token carrying the values of the columns 1, 2
  #' and 4 places to its right, wrapping around -- not its own value: the
  #' shifts are `2^i` for `i = 0..group_size-1`, and none is zero. The
  #' column count is preserved.
  #'
  #' Three blocks come out, in this order: the standard-scaled values, the
  #' NaN/Inf indicators, the raw ECDF ranks. [tabpfn35_cell_embedder()]
  #' slices the values off the front and the ranks off the back, so those
  #' two must stay at their ends.
  #'
  #' Kept separate from `embed_cells()` because this is where the pipeline
  #' can be cut: the grouped tensor is `(B, Ri, C, 3G)` with `3G` 9 and the
  #' embedded one is `(B, Ri, C, E)` with `E` 128, so everything that runs
  #' a chunk of rows at a time has to start on this side of the boundary.
  #' @keywords internal
  group_features = function(x_BRiC, indicators, ecdf) {
    g <- self$feature_group_size
    roll_stack <- function(z) {
      torch::torch_stack(
        lapply(0:(g - 1L), function(i) {
          torch::torch_roll(z, shifts = -(2L^i), dims = 3L)
        }),
        dim = 4L
      )
    }
    grouped <- torch::torch_cat(
      list(roll_stack(x_BRiC), roll_stack(indicators), roll_stack(ecdf)),
      dim = 4L
    )
    dump_if_enabled("x_grouped", grouped)
    grouped
  },

  #' Embed each column group into one token.
  #' @keywords internal
  embed_cells = function(grouped) {
    self$x_embed(grouped)
  },

  #' Every distribution-embedder block's inducing summary, in column chunks
  #'
  #' This is the half of stage 1 that a row-chunked pass cannot do for
  #' itself: block `l`'s inducing summary is a function of the *training*
  #' rows' state after blocks `1..l-1`, so it has to exist before any row
  #' chunk starts. Chunking it along columns instead is free -- every
  #' column is embedded on its own.
  #' @keywords internal
  all_inducing_hidden = function(grouped, y_col, n_train, col_chunk_size = NULL) {
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
      # Same fold as the block's own forward, so `b` varies slowest and the
      # chunks concatenate back into the whole in the right order.
      x_flat <- x_emb$transpose(2L, 3L)$contiguous()$
        reshape(c(B * cj, n_train, E))
      for (i in seq_len(n_blocks)) {
        blk <- layers[[i]]
        hidden <- blk$inducing_hidden(x_flat, n_train)
        parts[[i]][[length(parts[[i]]) + 1L]] <- hidden$reshape(c(B, cj, -1L, E))
        # The next block's summary is taken from this one's output on the
        # training rows, which is what the unchunked path feeds forward.
        if (i < n_blocks) x_flat <- blk$cross_attn_block2(x_flat, hidden)
      }
    }
    lapply(parts, function(p) {
      h <- if (length(p) == 1L) p[[1L]] else torch::torch_cat(p, dim = 2L)
      h$flatten(start_dim = 1L, end_dim = 2L)$detach()
    })
  },

  #' Clean the training targets and put them in `(B, N)`.
  #'
  #' Only the embedding path uses this; the decoder reads the raw targets
  #' and masks the non-finite ones itself.
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
    self$col_y_layernorm(self$col_y_encoder(y_BN, self$task_type))
  },

  #' @keywords internal
  embed_icl_targets = function(y_BN) {
    self$icl_y_layernorm(self$icl_y_encoder(y_BN, self$task_type))
  }
)


# Add a per-row term to the leading `n_train` rows of `x`, leaving the test
# rows alone. Built by concatenation rather than in-place assignment: a
# slice assignment on a tensor that may be a view is one of the few places
# R torch and PyTorch disagree about aliasing.
# @keywords internal
.tabpfn35_add_to_train_rows <- function(x, term, n_train) {
  Ri <- x$size(2)
  if (n_train >= Ri) return(x + term)
  if (n_train == 0L) return(x)
  torch::torch_cat(
    list(.tabpfn35_row_slice(x, 1L, n_train) + term,
         .tabpfn35_row_slice(x, n_train + 1L, Ri)),
    dim = 2L
  )
}

# Resolve a stage-chunking argument against the checkpoint's own value.
#
# Three states, because there are three things a caller can mean. `NA`
# (the default) is "whatever the checkpoint says". `NULL` is "off", which
# is how the parity harness asks for the unchunked pass. An integer is an
# integer.
# @keywords internal
.tabpfn35_chunk_arg <- function(x, default) {
  if (is.null(x)) return(NULL)
  if (length(x) == 1L && is.na(x)) return(as.integer(default))
  as.integer(x)
}

# Slice rows `from:to` out of a 3-D or 4-D tensor whose second axis is the
# row axis.
# @keywords internal
.tabpfn35_row_slice <- function(x, from, to) {
  if (x$dim() == 4L) x[, from:to, , ] else x[, from:to, ]
}


# ---------------------------------------------------------------------------
# Backend hooks
# ---------------------------------------------------------------------------

#' @keywords internal
tabpfn35_build <- function(config, task) {
  # Unlike every earlier TabPFN backend this reads `task`: the checkpoint
  # has no head of its own.
  cli::cli_alert_info(
    "Building tabpfn_v3_5 (multitask, running as {.val {task}}, \\
     {config$nlayers} ICL layers, emb={config$embed_dim})..."
  )
  tabpfn_v35_transformer(config, task = task)
}

#' @keywords internal
tabpfn35_detect <- function(config) {
  identical(config$arch, "tabpfn_v3_5")
}


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

#' Stage list for the TabPFN v3.5 diagram
#'
#' The same three-stage skeleton v3 introduced, drawn with the two things
#' that make v3.5 a different picture: a cell embedder that reads a value
#' against its own column's distribution before anything else runs, and a
#' decoder that carries both heads and picks one per pass.
#'
#' @param config The checkpoint's parsed `config.json`.
#' @param task `"classification"` or `"regression"`. Here this decides
#'   which head is drawn, because the checkpoint no longer does.
#' @keywords internal
tabpfn35_describe <- function(config, task) {
  E     <- as.integer(config$embed_dim %||% 128L)
  n_cls <- as.integer(config$feat_agg_num_cls_tokens %||% 8L)
  D     <- E * n_cls
  ff    <- as.integer(config$ff_factor %||% 2L)
  grp   <- as.integer(config$feature_group_size %||% 3L)
  L     <- as.integer(config$nlayers %||% 24L)
  clf   <- !identical(task, "regression")
  n_ind <- as.integer(config$dist_embed_num_inducing_points %||% 128L)
  n_dst <- as.integer(config$dist_embed_num_blocks %||% 3L)
  n_agg <- as.integer(config$feat_agg_num_blocks %||% 3L)
  h_icl <- as.integer(config$icl_num_heads %||% 16L)
  kv_t  <- as.integer(config$icl_num_kv_heads_test %||% h_icl)
  n_cls_max <- as.integer(config$max_num_classes %||% 160L)
  n_bkt <- as.integer(config$num_buckets %||% 5000L)
  n_out <- if (clf) n_cls_max else n_bkt
  nfreq <- as.integer(config$fourier_encoding_num_frequencies %||% 32L)
  kfreq <- as.integer(config$cell_ecdf_num_frequencies %||% 4L)
  nbuck <- as.integer(config$cell_ecdf_num_buckets %||% 8192L)

  stages <- list(
    arch_input_stage(),
    arch_stage(
      "ecdf", "In-context ECDF", kind = "embed", group = "Embed",
      detail = sprintf(
        "per column, each cell's midrank against the training rows, via %d bucket edges",
        nbuck),
      shape = "(B, n, F)"),
    arch_stage(
      "x_embed", "Cell embedder", kind = "embed", group = "Embed",
      detail = sprintf(
        "Fourier bank (%d frequencies) over %d values + linear over %d metadata channels, LayerNorm",
        nfreq, grp, grp * 2L + grp * 2L * kfreq),
      shape = sprintf("(B, n, F, %d)", E), prefix = "x_embed",
      children = list(
        arch_stage("four", "Learned Fourier frequencies", kind = "embed",
                   prefix = "x_embed.fourier"),
        arch_stage("meta", "Metadata projection", kind = "embed",
                   prefix = "x_embed.metadata_linear"),
        arch_stage("xln", "LayerNorm", kind = "norm",
                   prefix = "x_embed.layernorm")
      )),
    arch_stage(
      "col_y", "Target encoder (per column)", kind = "embed", group = "Embed",
      detail = sprintf("%s, then LayerNorm",
                       if (clf) sprintf("class embedding -> %d", E)
                       else sprintf("Linear(1 -> %d)", E)),
      prefix = c("col_y_encoder", "col_y_layernorm")),
    arch_stage(
      "dist", "Feature-distribution embedder", kind = "attention",
      group = "Stage 1 - per column", repeats = n_dst, axis = "rows",
      detail = sprintf(
        "induced self-attention, %d inducing vectors; keys are training rows only",
        n_ind),
      shape = sprintf("(B, n, F, %d)", E),
      prefix = "feature_distribution_embedder",
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
      detail = sprintf(
        "pre-norm self-attention with RoPE and QK-norm; last block reads out %d CLS tokens",
        n_cls),
      shape = sprintf("(B, n, %d)", D), prefix = "column_aggregator",
      children = list(
        arch_stage("attn", "Attention over a row's columns", kind = "attention",
                   axis = "columns", prefix = "column_aggregator.blocks.0"),
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
      detail = sprintf("%s, then LayerNorm",
                       if (clf) sprintf("class embedding -> %d", D)
                       else sprintf("Linear(1 -> %d)", D)),
      prefix = c("icl_y_encoder", "icl_y_layernorm")),
    arch_stage(
      "icl", "In-context learning block", kind = "attention",
      group = "Stage 3 - in-context", repeats = L, axis = "rows",
      detail = sprintf(
        "pre-norm, QK-norm; %d heads of %d; test rows read %d kv head%s; MLP %d -> %d",
        h_icl, as.integer(D / h_icl), kv_t, if (kv_t == 1L) "" else "s",
        D, D * ff),
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
    arch_stage(
      "phead", if (clf) "Pre-head MLP (classification)"
               else "Pre-head MLP (regression)",
      kind = "ffn", group = "Decode",
      detail = sprintf("residual pre-norm, %d -> %d -> %d", D, D * ff, D),
      prefix = if (clf) "heads.mlp_classification" else "heads.mlp_regression"),
    if (clf)
      arch_stage(
        "dec", "Many-class decoder", kind = "decode", group = "Decode",
        detail = sprintf(
          "retrieval over the training labels, %d heads of %d; up to %d classes",
          as.integer(config$decoder_num_heads %||% 6L),
          as.integer(config$decoder_head_dim %||% 64L), n_cls_max),
        shape = sprintf("(n_test, %d)", n_out),
        prefix = "heads.many_class_decoder")
    else
      arch_stage(
        "dec", "Output projection", kind = "decode", group = "Decode",
        detail = sprintf("Linear(%d -> %d)", D, n_bkt),
        shape = sprintf("(n_test, %d)", n_out),
        prefix = "heads.output_projection"),
    # The head this task does not use is still in the checkpoint -- that
    # is what "multitask" means -- so the diagram has to account for its
    # parameters rather than leave them off the tally. Drawn last, and
    # named for what it is: carried, not run.
    arch_stage(
      "unused", if (clf) "Regression head (not on this path)"
                else "Classification head (not on this path)",
      kind = "decode", group = "Decode",
      detail = paste("present in the checkpoint and loaded, but the task",
                     "selects the other branch"),
      prefix = if (clf) c("heads.mlp_regression", "heads.output_projection")
               else c("heads.mlp_classification", "heads.many_class_decoder")),
    if (clf)
      arch_output_stage("Class logits",
                        detail = sprintf("up to %d classes", n_out))
    else
      arch_output_stage("Bar distribution",
                        detail = sprintf("%d bins over the scaled target", n_out))
  )

  list(
    title = sprintf("TabPFN v3.5 - %s", if (clf) "classifier" else "regressor"),
    subtitle = "Prior-Labs  -  one multitask checkpoint; three stages: per column, per row, in context",
    facts = c(
      "ICL layers"          = L,
      "Column-stage blocks" = n_dst,
      "Aggregator blocks"   = n_agg,
      "Embedding width"     = E,
      "ICL width"           = sprintf("%d (%d CLS x %d)", D, n_cls, E),
      "ICL heads"           = sprintf("%d x %d", h_icl, as.integer(D / h_icl)),
      "Inducing points"     = n_ind,
      "Features per group"  = grp,
      "Cell channels"       = "value, NaN/Inf flag, ECDF rank",
      "Output width"        = n_out,
      "Heads in checkpoint" = sprintf("both (%d classes / %d bars)", n_cls_max, n_bkt),
      "Normalisation"       = "RMSNorm, pre; QK-norm per head",
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

#' TabPFN v3.5's peak-memory shapes
#'
#' The same three-stage family as TabICL, TabFM and v3, so it is estimated
#' the same way, with one term of its own.
#'
#' The ECDF context is resident for the whole pass -- three floats per
#' bucket edge per column -- and, when a cache is built, for the whole
#' `predict()` and once per ensemble member. It is bounded by
#' `cell_ecdf_num_buckets` rather than by the row count, which is the
#' point of bucketing it, so it flattens out on a tall table instead of
#' growing with it.
#'
#' Its *transients* -- the sort in the context build, the searchsorted in
#' the query -- are deliberately not modelled: both are chunked to a fixed
#' cell budget, so they are a constant independent of the table and belong
#' in `intercept_bytes` rather than in a shape.
#'
#' @inheritParams mitra_peak_terms
#' @keywords internal
tabpfn35_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  e <- as.numeric(config$embed_dim %||% 128)
  n_cls <- as.numeric(config$feat_agg_num_cls_tokens %||% 8)
  l <- as.numeric(config$nlayers %||% 24)
  icl_heads <- as.numeric(config$icl_num_heads %||% 16)
  n_est <- max(1, as.numeric(opts$n_estimators %||% 1))
  nq <- .resident_query(n_query, opts)

  # One token per *column*, not per group of them: v3.5 groups by rolling
  # and stacking, as v3 does, which preserves the column count where v2's
  # packing divided it.
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
    row_chunk    = .stage_row_chunk(opts, config),
    col_chunk    = .stage_col_chunk(opts, config),
    # The grouped input the row loop slices from. Three channels per group
    # position in v3.5 -- value, indicator, ECDF rank -- where v3 had two.
    group_channels = g * 3
  )

  # Edges per column, capped by the bucket count: this is what stops the
  # ranking context growing with a tall table.
  k_edges <- min(as.numeric(config$cell_ecdf_num_buckets %||% 8192), n_context)
  ecdf_ctx <- 3 * n_features * k_edges

  if (isTRUE(opts$kv_cache)) {
    # The cached path keeps `icl_num_kv_heads_test` heads rather than all
    # of them (one, in the published checkpoints), plus the training rows'
    # embeddings and the ECDF context -- all of it once per member.
    d_icl <- e * n_cls
    kv_heads <- as.numeric(config$icl_num_kv_heads_test %||% icl_heads)
    head_dim <- d_icl / max(icl_heads, 1)
    terms$persistent <- n_est * (l * 2 * n_context * kv_heads * head_dim +
                                 n_context * d_icl + ecdf_ctx)
  } else {
    terms$persistent <- terms$persistent + ecdf_ctx
  }
  terms
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#' @keywords internal
register_tabpfn35_backend <- function() {
  register_backend(
    name          = "tabpfn35",
    build         = tabpfn35_build,
    describe      = tabpfn35_describe,
    translate_key = identity,
    detect        = tabpfn35_detect,
    # No `task_of`: one checkpoint serves both tasks, so the artifacts
    # have no opinion about which one is being loaded. The converter
    # writes `head: "multitask"`, which the shared `tabpfn_task_of()`
    # falls through to NULL on -- exactly the "no opinion" answer
    # `load_backend_model()` needs -- but saying so here is clearer than
    # relying on that.
    task_of       = function(config) NULL,
    # The v2 predictors serve v3.5 unchanged, as they do v3: the
    # preprocessing pipeline, the ensembling and the bar-distribution head
    # are all shared, and what differs is asked of the network rather than
    # branched on here.
    classifier    = tabpfn_classifier,
    regressor     = tabpfn_regressor,
    peak_terms    = tabpfn35_peak_terms,
    # NaN and Inf are first-class inputs: imputed with the training mean
    # and flagged in a dedicated indicator channel.
    kv_cache_capable = TRUE,
    handles_missing = TRUE,
    description   = "TabPFN v3.5, multitask (Prior-Labs)",
    parity        = "tabpfn 9.0.0 (PyPI)"
  )
}
