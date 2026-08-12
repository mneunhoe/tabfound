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
#' @param logits      `torch_tensor` of shape `(..., n_bar_bins)`.
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
  outs <- lapply(quantiles, function(q) {
    bar_logits_to_quantile_single(logits, borders, q)
  })
  torch::torch_stack(outs, dim = -1L)
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
  # For each sample k, compute icdf(logits, u[..., k]) and stack.
  outs <- lapply(seq_len(n_samples), function(k) {
    bar_logits_to_quantile_single(
      logits, borders,
      u$select(dim = -1L, index = as.integer(k))
    )
  })
  torch::torch_stack(outs, dim = -1L)
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

# Compute CDF of the bar distribution (logits + borders) at query points ys.
# Matches Python's `_cdf` in tabpfn/utils.py.
.cdf_bar_dist <- function(logits, borders_from, ys) {
  # logits: (..., n_bars)
  # borders_from: (n_bars + 1,)  (1-D torch tensor on same device)
  # ys:    (n_query,)             (1-D torch tensor)
  n_bars <- as.integer(borders_from$size(1)) - 1L
  # searchsorted returns 0-indexed; Python subtracts 1.
  y_buckets_0 <- torch::torch_searchsorted(borders_from, ys$contiguous())$to(dtype = torch::torch_long()) - 1L
  y_buckets_0 <- y_buckets_0$clamp(min = 0L, max = n_bars - 1L)

  # Broadcast y_buckets_0 to (..., n_query)
  lead_shape <- as.integer(logits$size()); lead_shape[length(lead_shape)] <- ys$size(1)
  y_buckets_b <- y_buckets_0$unsqueeze(1L)$expand(lead_shape)   # (1, n_query) -> broadcast
  # Actually for simplicity, since logits may be (n_test, n_bars) we want
  # (n_test, n_query). Tile y_buckets to the leading dims of logits.
  # We'll rebuild via repeat.
  lead_shape_wo_last <- as.integer(logits$size())[-length(logits$size())]
  y_buckets_full <- y_buckets_0$view(c(rep(1L, length(lead_shape_wo_last)), ys$size(1)))$
    expand(c(lead_shape_wo_last, ys$size(1)))   # broadcast to logits shape w/o last dim

  # Gather indices need to be 1-indexed for torch_gather in R.
  idx_gather <- (y_buckets_full + 1L)$to(dtype = torch::torch_long())

  probs <- torch::nnf_softmax(logits, dim = -1L)
  prob_so_far <- torch::torch_cumsum(probs, dim = -1L) - probs
  prob_left_of_bucket <- torch::torch_gather(prob_so_far, dim = -1L, index = idx_gather)

  widths <- borders_from[2:(n_bars + 1L)] - borders_from[1:n_bars]
  # Gather borders and widths by 0-index
  border_at_bucket <- torch::torch_index_select(
    borders_from, dim = 1L,
    index = y_buckets_full$contiguous()$view(-1L) + 1L   # 0-idx -> R 1-idx
  )$view(as.integer(y_buckets_full$size()))
  width_at_bucket  <- torch::torch_index_select(
    widths, dim = 1L,
    index = y_buckets_full$contiguous()$view(-1L) + 1L
  )$view(as.integer(y_buckets_full$size()))

  # ys broadcast to match
  ys_b <- ys$view(c(rep(1L, length(lead_shape_wo_last)), ys$size(1)))$
    expand(c(lead_shape_wo_last, ys$size(1)))
  share <- ((ys_b - border_at_bucket) / width_at_bucket)$clamp(min = 0, max = 1)
  prob_in_bucket <- torch::torch_gather(probs, dim = -1L, index = idx_gather) * share
  prob_left_of_ys <- prob_left_of_bucket + prob_in_bucket

  # Boundary conditions
  below <- (ys_b <= borders_from[1])$to(dtype = prob_left_of_ys$dtype)
  above <- (ys_b >= borders_from[n_bars + 1L])$to(dtype = prob_left_of_ys$dtype)
  prob_left_of_ys <- torch::torch_where(below$to(dtype = torch::torch_bool()),
                                         torch::torch_zeros_like(prob_left_of_ys),
                                         prob_left_of_ys)
  prob_left_of_ys <- torch::torch_where(above$to(dtype = torch::torch_bool()),
                                         torch::torch_ones_like(prob_left_of_ys),
                                         prob_left_of_ys)
  prob_left_of_ys$clamp(min = 0, max = 1)
}

#' Translate member-logits from `frm` borders onto `to` borders (matches
#' Python `translate_probs_across_borders`).
#'
#' Returns bucket probabilities (NOT logits) of shape `(..., len(to)-1)`.
#' @keywords internal
translate_probs_across_borders_r <- function(logits, frm, to) {
  prob_left <- .cdf_bar_dist(logits, frm, to)
  # Set endpoints exactly (Python does prob_left[..., 0] = 0, [..., -1] = 1)
  n_last <- as.integer(prob_left$size(prob_left$dim()))
  dev_   <- prob_left$device
  # Instead of in-place, just clamp endpoints via explicit where
  zeros_idx <- torch::torch_zeros_like(prob_left)
  ones_idx  <- torch::torch_ones_like(prob_left)
  # Positional mask: last dim == 0 -> replace with 0; last dim == n-1 -> 1
  mask_shape <- as.integer(prob_left$size())
  # Build a 1-D position mask of length n_last. It is compared against
  # tensors that live wherever `prob_left` does, so it has to be built
  # there too -- `torch_where` will not mix devices.
  positions <- torch::torch_arange(1L, n_last, dtype = torch::torch_long(),
                                   device = dev_)
  # Broadcast to prob_left shape for comparison
  pos_expanded <- positions$view(c(rep(1L, length(mask_shape) - 1L), n_last))
  is_left  <- (pos_expanded == 1L)
  is_right <- (pos_expanded == n_last)
  prob_left <- torch::torch_where(is_left,  zeros_idx, prob_left)
  prob_left <- torch::torch_where(is_right, ones_idx,  prob_left)

  # Diff along last dim, clamp non-negative
  left_idx  <- torch::torch_arange(1L, n_last - 1L, dtype = torch::torch_long(),
                                    device = dev_)
  right_idx <- torch::torch_arange(2L, n_last, dtype = torch::torch_long(),
                                    device = dev_)
  right <- torch::torch_index_select(prob_left, dim = -1L, index = right_idx)
  left  <- torch::torch_index_select(prob_left, dim = -1L, index = left_idx)
  (right - left)$clamp(min = 0)
}
