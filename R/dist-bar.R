# Bar-distribution output head.
#
# TabPFN's regressor emits logits over `n_bar_bins` buckets whose edges
# live in the `criterion.borders` buffer. This file holds the
# distribution math -- posterior mean, inverse CDF, sampling -- plus the
# border-translation helpers needed when an ensemble member applied a
# target transform and its buckets must be mapped back onto the common
# grid before averaging.
#
# Mirrors `FullSupportBarDistribution` in
# `tabpfn/architectures/base/bar_distribution.py` and
# `translate_probs_across_borders` in `tabpfn/utils.py`.
#
# Other backends use different heads (TabICL: quantile distribution,
# TabFM: scalar decoder); those live in their own `dist-*.R` files.

#' @keywords internal
bar_distribution_decoder <- NULL

# ---------------------------------------------------------------------------
# Bar-distribution post-processing (regressor head)
#
# Mirrors `FullSupportBarDistribution` in
# `tabpfn/architectures/shared/bar_distribution.py`:
#
#   mean(logits)      = softmax(logits) @ bucket_means, where the FIRST
#                       and LAST bucket means come from half-normal tails
#                       rather than from the bucket midpoint
#   icdf(logits, q)   = linear interpolation inside the bucket whose CDF
#                       crosses q, using (borders[idx], borders[idx+1])
#                       and the bucket probability mass
#
# All helpers accept tensors of shape `(..., n_bar_bins)` and return
# tensors of shape `(..., [len(quantiles)])`. `borders` is a 1-D tensor
# of length `n_bar_bins + 1` (matching the `criterion.borders` buffer).

# The outer two buckets of a FullSupportBarDistribution are half-normal
# tails, not uniform bars, so the distribution has support on all of R.
# The tail's scale is set so half its mass falls within the bucket:
#
#   s = width / HalfNormal(1).icdf(0.5)      HalfNormal(1).icdf(0.5) = qnorm(0.75)
#   E[HalfNormal(s)] = s * sqrt(2 / pi)
#
# @keywords internal
.halfnormal_tail_mean <- function(width) {
  s <- width / stats::qnorm(0.75)
  s * sqrt(2 / pi)
}

#' Convert bar-distribution logits to the posterior mean
#'
#' @param logits  `torch_tensor` of shape `(..., n_bar_bins)`.
#' @param borders `torch_tensor` of length `n_bar_bins + 1`.
#' @param full_support Logical. When `TRUE` (the default, and what TabPFN
#'   uses) the outermost buckets are treated as half-normal tails. Setting
#'   it to `FALSE` gives the plain piecewise-uniform bar distribution.
#' @return `torch_tensor` of shape `logits$size()[-last]`.
#' @export
bar_logits_to_mean <- function(logits, borders, full_support = TRUE) {
  # Match Python's `borders[:-1] + bucket_widths / 2` ordering exactly —
  # `(borders[i] + borders[i+1]) / 2` is algebraically equivalent but
  # has slightly different float32 rounding when borders span wide ranges.
  n_bins <- as.integer(borders$size(1)) - 1L
  left_edges   <- borders[1:n_bins]
  right_edges  <- borders[2:(n_bins + 1L)]
  bucket_widths <- right_edges - left_edges
  bucket_means  <- left_edges + bucket_widths / 2

  if (isTRUE(full_support)) {
    # bucket_means[0]  = -HalfNormal(w_first).mean + borders[1]
    # bucket_means[-1] =  HalfNormal(w_last).mean  + borders[-2]
    w_first <- as.numeric(bucket_widths[1]$cpu())
    w_last  <- as.numeric(bucket_widths[n_bins]$cpu())
    bucket_means <- bucket_means$clone()
    bucket_means[1]      <- borders[2] - .halfnormal_tail_mean(w_first)
    bucket_means[n_bins] <- borders[n_bins] + .halfnormal_tail_mean(w_last)
  }

  probs <- torch::nnf_softmax(logits, dim = -1L)
  torch::torch_matmul(probs, bucket_means)
}


#' Compute one inverse-CDF evaluation (a single quantile)
#'
#' The one-level-at-a-time form of `.bar_icdf()`, kept as the reference
#' the batched decode is checked against: stacking this over the levels
#' gives the same tensor, bit for bit.
#'
#' @param logits     `torch_tensor` of shape `(..., n_bar_bins)`.
#' @param borders     `torch_tensor` of length `n_bar_bins + 1`.
#' @param left_prob   Scalar numeric in `(0, 1)`, OR a `torch_tensor`
#'   broadcastable to `logits$size()[-last]`.
#' @return `torch_tensor` shaped like `logits` without the last dim.
#' @keywords internal
bar_logits_to_quantile_single <- function(logits, borders, left_prob) {
  n_bins <- as.integer(borders$size(1)) - 1L
  probs <- torch::nnf_softmax(logits, dim = -1L)
  cumprobs <- torch::torch_cumsum(probs, dim = -1L)   # (..., n_bins)

  lead_shape <- as.integer(cumprobs$size())
  lead_shape[length(lead_shape)] <- 1L

  if (is.numeric(left_prob) && length(left_prob) == 1L) {
    needle <- torch::torch_full(
      lead_shape, left_prob,
      dtype = probs$dtype, device = probs$device
    )
    lp_scalar <- as.numeric(left_prob)
  } else {
    needle <- left_prob$to(dtype = probs$dtype, device = probs$device)$
      view(lead_shape)
    lp_scalar <- NULL
  }

  # R torch's torch_searchsorted returns 0-indexed positions.
  # `needle` must be contiguous to avoid a runtime copy warning.
  idx0 <- torch::torch_searchsorted(cumprobs, needle$contiguous())$
    clamp(min = 0L, max = n_bins - 1L)$squeeze(-1L)   # (...,)

  # cumprobs_padded = cat([0, cumprobs], -1) -> (..., n_bins + 1)
  zero_pad <- torch::torch_zeros(
    lead_shape, dtype = probs$dtype, device = probs$device
  )
  cumprobs_padded <- torch::torch_cat(list(zero_pad, cumprobs), dim = -1L)

  # Gather uses 1-indexed indices in R torch. idx0 is 0-indexed,
  # cumprobs_padded's position idx0 (0-indexed) == R position (idx0 + 1).
  idx_r <- (idx0 + 1L)$unsqueeze(-1L)
  cp_at_idx <- torch::torch_gather(cumprobs_padded, dim = -1L, index = idx_r)$
    squeeze(-1L)
  probs_at_idx <- torch::torch_gather(probs, dim = -1L, index = idx_r)$
    squeeze(-1L)

  # borders[idx0 + 1]  and  borders[idx0 + 2]  in R 1-indexed.
  idx_flat <- idx0$contiguous()$view(-1L) + 1L
  left_border <- torch::torch_index_select(borders, dim = 1L, index = idx_flat)$
    view(idx0$size())
  right_border <- torch::torch_index_select(borders, dim = 1L,
                                             index = idx_flat + 1L)$
    view(idx0$size())

  if (!is.null(lp_scalar)) {
    rest_prob <- lp_scalar - cp_at_idx
  } else {
    rest_prob <- left_prob - cp_at_idx
  }
  left_border + (right_border - left_border) * rest_prob / probs_at_idx
}


#' Convert bar-distribution logits to requested quantiles
#'
#' @param logits    `torch_tensor` of shape `(..., n_bar_bins)`.
#' @param borders   `torch_tensor` of length `n_bar_bins + 1`.
#' @param quantiles Numeric vector of quantiles in `(0, 1)`.
#' @return `torch_tensor` with the quantile axis appended last.
#' @export
bar_logits_to_quantiles <- function(logits, borders, quantiles) {
  lead <- as.integer(logits$size())
  lead <- lead[-length(lead)]
  needle <- torch::torch_tensor(as.numeric(quantiles), dtype = logits$dtype,
                                device = logits$device)
  .bar_icdf(logits, borders, needle$expand(c(lead, length(quantiles))))
}


# The inverse CDF at K levels per row, in one pass: `needle` is
# `(..., K)`, row i's levels, and the result has the same shape.
#
# The softmax, the cumulative sum and its zero-padded copy are each built
# once, and one batched `searchsorted` and `gather` place every level.
# Going a level at a time rebuilt all three `(n, n_bins)` tensors per
# level -- K times over, with K = 199 quantiles or 100 draws -- and R's GC
# cannot see their size, so generations of them piled up before anything
# was freed. Each level gets the arithmetic
# `bar_logits_to_quantile_single()` gives it, in the same order, so the
# result is that function's, stacked, bit for bit.
# @keywords internal
.bar_icdf <- function(logits, borders, needle) {
  n_bins <- as.integer(borders$size(1)) - 1L
  probs <- torch::nnf_softmax(logits, dim = -1L)
  cumprobs <- torch::torch_cumsum(probs, dim = -1L)   # (..., n_bins)
  # `searchsorted` copies a non-contiguous needle, with a warning.
  needle <- needle$to(dtype = probs$dtype, device = probs$device)$contiguous()

  # R torch's torch_searchsorted returns 0-indexed positions.
  idx0 <- torch::torch_searchsorted(cumprobs, needle)$
    clamp(min = 0L, max = n_bins - 1L)                 # (..., K)

  lead_shape <- as.integer(cumprobs$size())
  lead_shape[length(lead_shape)] <- 1L
  zero_pad <- torch::torch_zeros(
    lead_shape, dtype = probs$dtype, device = probs$device
  )
  cumprobs_padded <- torch::torch_cat(list(zero_pad, cumprobs), dim = -1L)

  # Gather is 1-indexed: padded position idx0 (0-indexed) is R's idx0 + 1,
  # and so are borders[idx0] and borders[idx0 + 1].
  idx_r <- idx0 + 1L
  cp_at_idx <- torch::torch_gather(cumprobs_padded, dim = -1L, index = idx_r)
  probs_at_idx <- torch::torch_gather(probs, dim = -1L, index = idx_r)
  idx_flat <- idx_r$contiguous()$view(-1L)
  left_border <- torch::torch_index_select(borders, dim = 1L, index = idx_flat)$
    view(idx0$size())
  right_border <- torch::torch_index_select(borders, dim = 1L,
                                             index = idx_flat + 1L)$
    view(idx0$size())

  left_border + (right_border - left_border) * (needle - cp_at_idx) /
    probs_at_idx
}


#' Sample from a bar distribution
#'
#' Draws `n_samples` uniform variates per row and inverts the CDF.
#' Deterministic when `seed` is supplied.
#'
#' @param logits    `torch_tensor` of shape `(..., n_bar_bins)`.
#' @param borders   `torch_tensor` of length `n_bar_bins + 1`.
#' @param n_samples Integer number of samples per logits row.
#' @param seed      Optional integer seed.
#' @return `torch_tensor` with a sample axis of length `n_samples` appended.
#' @export
bar_logits_to_samples <- function(logits, borders, n_samples, seed = NULL) {
  if (!is.null(seed)) torch::torch_manual_seed(as.integer(seed))
  dims <- as.integer(logits$size())
  # Draw one batch of uniforms of shape (..., n_samples).
  sample_shape <- c(dims[-length(dims)], as.integer(n_samples))
  u <- torch::torch_rand(sample_shape,
                          dtype = logits$dtype, device = logits$device)
  .bar_icdf(logits, borders, u)
}


# Draws for `n` rows whose logits arrive a chunk at a time: `logits_of(s, e)`
# returns rows `s:e` as `(e - s + 1, n_bins)`. Returns the `(n, n_samples)`
# R matrix of draws on the borders' scale.
#
# The uniforms are drawn once, for all `n` rows, right after seeding, and
# each chunk decodes its own rows of them. That is the stream one
# `bar_logits_to_samples()` call over every row draws, so a seeded result
# does not depend on the chunk size. Drawing per chunk would not be: on
# `mps` two consecutive `torch_rand()` calls are not one call split in two.
# The uniforms are `n * n_samples` floats; the logits they replace holding
# are `n * n_bins`.
# @keywords internal
.bar_samples_over_chunks <- function(n, chunk_size, logits_of, borders,
                                     n_samples, seed = NULL) {
  chunk_size <- as.integer(chunk_size)
  starts <- seq(1L, n, by = chunk_size)
  u <- NULL
  draws <- lapply(seq_along(starts), function(k) {
    if (k > 1L) collect_between_chunks()
    s <- starts[[k]]
    e <- min(s + chunk_size - 1L, n)
    logits <- logits_of(s, e)
    if (is.null(u)) {
      # Drawn once the first chunk says which dtype and device. Seeding
      # right before means no forward pass, here or later, moves the stream.
      if (!is.null(seed)) torch::torch_manual_seed(as.integer(seed))
      u <<- torch::torch_rand(c(n, as.integer(n_samples)),
                              dtype = logits$dtype, device = logits$device)
    }
    as.matrix(.bar_icdf(logits, borders, u[s:e, , drop = FALSE])$cpu())
  })
  do.call(rbind, draws)
}

# ---------------------------------------------------------------------------
# Border transform + prob translation helpers (target_transform members)
# ---------------------------------------------------------------------------

REGRESSION_NAN_BORDER_LIMIT_UPPER <- 1e3
REGRESSION_NAN_BORDER_LIMIT_LOWER <- -1e3

# Internal: repair NaN / extreme borders in-place (returns modified vector).
.repair_borders <- function(borders) {
  b <- as.numeric(borders)
  if (is.nan(b[length(b)]) || is.na(b[length(b)])) {
    nans <- is.na(b) | is.nan(b)
    largest <- max(b[!nans], na.rm = TRUE)
    b[nans] <- largest
    b[length(b)] <- b[length(b)] * 2
  }
  if (b[length(b)] - b[length(b) - 1L] < 1e-6) {
    b[length(b)] <- b[length(b)] * 1.1
  }
  if (b[1] == b[2]) {
    b[1] <- b[1] - abs(b[1] * 0.1)
  }
  b
}

# Internal: cancel-nan-border logic mirroring Python's helper.
.cancel_nan_borders <- function(borders, broken_mask) {
  b <- borders
  mask <- broken_mask

  # Left (broken leading chunk)
  if (isTRUE(mask[1])) {
    # find first index k where mask transitions TRUE -> FALSE
    k <- which(mask[-length(mask)] > mask[-1L])[1]
    if (!is.na(k)) {
      # Python: borders[: k + 1] = borders[k + 1]
      b[1:(k + 1L)] <- b[k + 1L]
      mask[1:(k + 1L)] <- FALSE
      # The outermost left border gets pushed a bit: matches Python's
      # setting borders[0] -= abs(borders[0]*0.1) at repair time. Leave to
      # .repair_borders().
    }
  }
  # Right (broken trailing chunk)
  if (isTRUE(mask[length(mask)])) {
    # find first index k where mask transitions FALSE -> TRUE
    k <- which(mask[-length(mask)] < mask[-1L])[1]
    if (!is.na(k)) {
      b[(k + 1L):length(b)] <- b[k]
      mask[(k + 1L):length(b)] <- FALSE
    }
  }
  # Python: `broken_mask[1:] | broken_mask[:-1]` — collapse borders (n+1)
  # to bars (n). A bar is cancelled if either of its two borders was broken.
  nb <- length(broken_mask)
  bar_mask <- broken_mask[2:nb] | broken_mask[1:(nb - 1L)]
  list(borders = b, logit_cancel_mask = bar_mask)
}

#' Transform znorm-space borders back through the target_transform's
#' inverse, mirroring Python's `transform_borders_one`.
#' @keywords internal
transform_borders_inverse_yeojohnson <- function(znorm_borders, lambda) {
  borders_t <- yeojohnson_inverse(as.numeric(znorm_borders), lambda)

  broken_mask <- !is.finite(borders_t) |
    borders_t > REGRESSION_NAN_BORDER_LIMIT_UPPER |
    borders_t < REGRESSION_NAN_BORDER_LIMIT_LOWER
  logit_cancel_mask <- NULL
  if (any(broken_mask)) {
    cnc <- .cancel_nan_borders(borders_t, broken_mask)
    borders_t <- cnc$borders
    logit_cancel_mask <- cnc$logit_cancel_mask
  }
  borders_t <- .repair_borders(borders_t)

  # Descending check
  descending <- all(rev(order(borders_t)) == seq_along(borders_t))
  if (descending) {
    borders_t <- rev(borders_t)
    if (!is.null(logit_cancel_mask)) logit_cancel_mask <- rev(logit_cancel_mask)
  }
  list(
    borders_t          = borders_t,
    logit_cancel_mask  = logit_cancel_mask,
    descending         = descending
  )
}

# The share of an outer bucket's half-normal tail that lies more than
# `distance` past its inner border: 1 at the border, 1/2 one bucket width
# out. `erfc` is taken as an upper normal tail, which keeps its precision
# far out where `1 - cdf` would cancel to 0. A degenerate outer bucket is
# floored to the smallest positive width, which makes it a point mass.
# @keywords internal
.halfnormal_tail_survival <- function(distance, width) {
  s <- max(width, .Machine$double.xmin) / stats::qnorm(0.75)
  2 * stats::pnorm(distance / s, lower.tail = FALSE)
}

# Weights that move bucket masses from the `frm` grid to the `to` grid.
#
# Destination bucket `j` receives `weight[k] * probs[source[k]]` for every
# pair `k` with `destination[k] == j`, plus `lower_tail[j] * probs[1]` and
# `upper_tail[j] * probs[n]`. An interior source bucket is a uniform bar,
# so its weight on a destination bucket is the share of it the two
# overlap; the two outer source buckets are half-normal tails, as the
# full-support bar distribution reads them, and the first and last
# destination buckets take whatever mass lies beyond `to`. A zero-width
# interior source bucket is a point mass and goes whole to the destination
# bucket holding its point. Every weight is non-negative, so no bucket's
# mass comes out of a cancellation.
#
# A port of `_remap_weights()` from `tabpfn` 9.1+ (PriorLabs/TabPFN#1288),
# in double precision like the reference's, since the weights depend on
# the two grids alone. Indices are 1-based.
# @keywords internal
.remap_weights <- function(frm, to) {
  frm <- as.numeric(frm); to <- as.numeric(to)
  n_frm <- length(frm) - 1L
  n_to  <- length(to) - 1L
  if (n_frm < 3L) {
    cli::cli_abort("{.arg frm} needs an interior bucket between its two tails.")
  }

  # Cut the axis at every border of both grids. Each piece lies in one
  # source bucket and one destination bucket; `findInterval(left.open =
  # TRUE)` counts the borders strictly below, which is `searchsorted()`.
  edges <- sort(unique(c(frm[2:n_frm], to)))
  mids  <- (edges[-length(edges)] + edges[-1L]) / 2
  source <- findInterval(mids, frm, left.open = TRUE)
  destination <- pmin(pmax(findInterval(mids, to, left.open = TRUE), 1L), n_to)
  interior <- source >= 2L & source <= n_frm - 1L
  source <- source[interior]
  destination <- destination[interior]
  weight <- diff(edges)[interior] / (frm[source + 1L] - frm[source])

  point <- which(frm[2:(n_frm - 1L)] == frm[3:n_frm]) + 1L
  if (length(point)) {
    source <- c(source, point)
    destination <- c(destination,
                     pmin(pmax(findInterval(frm[point], to), 1L), n_to))
    weight <- c(weight, rep(1, length(point)))
  }
  # Grouped by destination, pieces in axis order within each, which is the
  # order the reference sums them in.
  ord <- order(destination)

  lower <- c(-Inf, to[2:n_to])
  upper <- c(to[2:n_to], Inf)
  inner_low  <- frm[2L];     width_low  <- frm[2L] - frm[1L]
  inner_high <- frm[n_frm];  width_high <- frm[n_frm + 1L] - frm[n_frm]
  lower_survival <- function(y) {
    .halfnormal_tail_survival(inner_low - pmin(y, inner_low), width_low)
  }
  upper_survival <- function(y) {
    .halfnormal_tail_survival(pmax(y, inner_high) - inner_high, width_high)
  }

  list(
    source      = source[ord],
    destination = destination[ord],
    weight      = weight[ord],
    lower_tail  = pmax(lower_survival(upper) - lower_survival(lower), 0),
    upper_tail  = pmax(upper_survival(lower) - upper_survival(upper), 0)
  )
}

#' Translate member-logits from `frm` borders onto `to` borders (matches
#' Python `translate_probs_across_borders`, `tabpfn` 9.1 and later).
#'
#' Each destination bucket's mass is a sum of non-negative terms -- its
#' overlap with each source bucket times that bucket's probability, plus a
#' slice of each outer half-normal tail -- so no bucket is lost to
#' rounding however narrow it is. Until `tabpfn` 9.1 the reference took
#' differences of float32 CDF values and clamped them at zero, which gave
#' neighbouring CDF values that rounded together a probability of exactly
#' zero (PriorLabs/TabPFN#1288).
#'
#' A member whose grid is the destination grid comes back as its softmax:
#' every destination bucket is then a source bucket, and reading the outer
#' ones as tails only moves mass around inside them.
#'
#' Returns bucket probabilities (NOT logits) of shape `(..., len(to)-1)`,
#' in the dtype of `logits`.
#' @keywords internal
translate_probs_across_borders_r <- function(logits, frm, to) {
  if (identical(as.integer(frm$size()), as.integer(to$size())) &&
      frm$dtype == to$dtype && torch::torch_equal(frm$cpu(), to$cpu())) {
    return(torch::nnf_softmax(logits, dim = -1L))
  }
  w <- .remap_weights(as.numeric(frm$cpu()), as.numeric(to$cpu()))
  dev <- logits$device
  dtype <- logits$dtype
  as_t <- function(x) torch::torch_tensor(x, dtype = dtype, device = dev)
  source      <- torch::torch_tensor(w$source, dtype = torch::torch_long(), device = dev)
  destination <- torch::torch_tensor(w$destination, dtype = torch::torch_long(), device = dev)
  weight      <- as_t(w$weight)
  lower_tail  <- as_t(w$lower_tail)
  upper_tail  <- as_t(w$upper_tail)

  shape <- as.integer(logits$size())
  n_from <- shape[length(shape)]
  n_to <- length(w$lower_tail)
  flat <- logits$reshape(c(-1L, n_from))
  apply_chunk <- function(x) {
    probs <- torch::nnf_softmax(x, dim = -1L)
    out <- probs[, 1, drop = FALSE] * lower_tail +
      probs[, n_from, drop = FALSE] * upper_tail
    out$index_add_(2L, destination, probs$index_select(2L, source) * weight)
  }
  # The `(rows, pairs)` product is the big intermediate; the reference caps
  # it at 1e7 elements a chunk, about 40 MB in float32. Chunking does not
  # change the result.
  n_rows <- as.integer(flat$size(1))
  chunk <- max(1L, 10000000L %/% max(length(w$source), 1L))
  out <- if (n_rows <= chunk) {
    apply_chunk(flat)
  } else {
    starts <- seq(1L, n_rows, by = chunk)
    torch::torch_cat(lapply(starts, function(i) {
      apply_chunk(flat[i:min(i + chunk - 1L, n_rows), , drop = FALSE])
    }), dim = 1L)
  }
  out$reshape(c(shape[-length(shape)], n_to))
}
