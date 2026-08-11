# TabICL's `QuantileDistribution`.
#
# TabICL's regression head emits a grid of predicted quantiles -- 999
# levels for the released checkpoint -- and the obvious way to read a
# summary off it is to take the middle level for the median and the
# nearest level for a requested quantile. That is not what the reference
# does, and the difference is not cosmetic:
#
# * The predicted levels are not guaranteed monotone. A network that
#   predicts Q(0.4) > Q(0.6) has produced a crossing, and every summary
#   read off the raw grid inherits it. The reference sorts first.
# * `predict(output_type = "mean")` is the mean of the *whole* sorted
#   grid, an average of 999 numbers -- not the median. On a skewed
#   predictive distribution those are visibly different numbers.
# * A requested quantile is interpolated between the two adjacent levels,
#   and one outside `[1/(n+1), n/(n+1)]` is extrapolated along an
#   exponential tail fitted to the outermost 20 levels. Snapping to the
#   nearest level quantises the answer and cannot leave the grid at all.
#
# Only the pieces the R predictors need are ported: monotonicity, the
# spline, and the exponential tails. The reference also offers
# generalised-Pareto tails, a CDF, a PDF and an analytic CRPS; none of
# them is on any path this package exposes, and an unexercised port of
# them would be a liability rather than an asset.
#
# Reference: tabicl/_model/quantile_dist.py

# Matching `QuantileDistributionConfig`.
.QD_TOL <- 1e-6
.QD_MIN_BETA <- 0.01
.QD_MAX_BETA <- 100.0
.QD_TAIL_QUANTILES <- 20L

#' Fit exponential tail scales to a quantile grid
#'
#' The left tail is modelled as `Q(a) = beta_l * log(a) + c`, the right as
#' `Q(a) = -beta_r * log(1 - a) + c`, and `beta` is estimated by
#' regressing the outermost `k` levels on the corresponding log-alphas.
#'
#' @param q Numeric matrix `(n_rows, n_levels)`, already sorted along
#'   levels.
#' @param alpha Numeric vector of level probabilities.
#' @param num_tail_quantiles Levels per tail used for the fit; the
#'   reference caps this at a quarter of the grid.
#' @return A list with per-row `beta_l` and `beta_r`.
#' @keywords internal
estimate_exp_tail_params <- function(q, alpha, num_tail_quantiles = .QD_TAIL_QUANTILES) {
  n <- ncol(q)
  k <- min(as.integer(num_tail_quantiles), n %/% 4L)
  k <- max(k, 1L)

  fit_beta <- function(qs, ln_a) {
    ln_c <- ln_a - mean(ln_a)
    var_ln <- max(mean(ln_c^2), .QD_TOL)
    qc <- qs - rowMeans(qs)
    cov <- as.numeric(qc %*% ln_c) / length(ln_c)
    cov / var_ln
  }

  ln_alpha_l <- log(pmax(alpha[seq_len(k)], .QD_TOL))
  beta_l <- fit_beta(q[, seq_len(k), drop = FALSE], ln_alpha_l)
  beta_l <- pmin(pmax(abs(beta_l), .QD_MIN_BETA), .QD_MAX_BETA)

  idx_r <- (n - k + 1L):n
  ln_one_minus <- log(pmax(1 - alpha[idx_r], .QD_TOL))
  # `Q = -beta * log(1 - a) + c`, so the regression slope is `-beta`.
  beta_r <- -fit_beta(q[, idx_r, drop = FALSE], ln_one_minus)
  beta_r <- pmin(pmax(abs(beta_r), .QD_MIN_BETA), .QD_MAX_BETA)

  list(beta_l = beta_l, beta_r = beta_r)
}

#' Build a quantile distribution from a predicted grid
#'
#' @param grid Numeric matrix `(n_rows, n_levels)` of predicted quantiles.
#' @param alpha Numeric vector of level probabilities, ascending.
#' @keywords internal
quantile_dist <- function(grid, alpha) {
  grid <- as.matrix(grid)
  # `fix_crossing = TRUE, crossing_method = "sort"`.
  q <- t(apply(grid, 1L, sort))
  if (!is.matrix(q) || ncol(q) != ncol(grid)) {
    q <- matrix(q, nrow = nrow(grid), ncol = ncol(grid))
  }
  n <- ncol(q)
  tails <- estimate_exp_tail_params(q, alpha)
  q_l <- q[, 1L]; q_r <- q[, n]
  alpha_l <- alpha[1L]; alpha_r <- alpha[n]

  list(
    quantiles = q, alpha = alpha, n = n,
    alpha_l = alpha_l, alpha_r = alpha_r, q_l = q_l, q_r = q_r,
    tail_a_l = tails$beta_l,
    tail_b_l = q_l - tails$beta_l * log(max(alpha_l, .QD_TOL)),
    tail_a_r = -tails$beta_r,
    tail_b_r = q_r + tails$beta_r * log(1 - min(alpha_r, 1 - .QD_TOL))
  )
}

#' Evaluate the quantile function of a fitted quantile distribution
#'
#' @param dist Result of [quantile_dist()].
#' @param probs Numeric vector of probabilities.
#' @return A matrix `(n_rows, length(probs))`.
#' @keywords internal
quantile_dist_icdf <- function(dist, probs) {
  q <- dist$quantiles
  n_rows <- nrow(q); n <- dist$n
  alpha_lo <- dist$alpha[seq_len(n - 1L)]
  alpha_hi <- dist$alpha[2L:n]

  out <- matrix(NA_real_, n_rows, length(probs))
  for (i in seq_along(probs)) {
    a <- probs[[i]]
    col <- if (a < dist$alpha_l) {
      dist$tail_a_l * log(max(a, .QD_TOL)) + dist$tail_b_l
    } else if (a > dist$alpha_r) {
      dist$tail_a_r * log(max(1 - a, .QD_TOL)) + dist$tail_b_r
    } else {
      # Piecewise-linear between the two bracketing levels. `right = TRUE`
      # in the reference's `searchsorted`, i.e. a probability landing
      # exactly on a knot belongs to the segment starting there.
      seg <- findInterval(a, alpha_lo, left.open = TRUE)
      seg <- min(max(seg, 1L), n - 1L)
      t <- (a - alpha_lo[seg]) / max(alpha_hi[seg] - alpha_lo[seg], .QD_TOL)
      t <- min(max(t, 0), 1)
      v <- q[, seg] + t * (q[, seg + 1L] - q[, seg])
      if (a >= dist$alpha_r) dist$q_r else v
    }
    out[, i] <- col
  }
  out
}

#' Summary statistics from a predicted quantile grid
#'
#' The single entry point the TabICL regressor uses. `"mean"` is the mean
#' of the sorted grid, `"median"` and `"quantiles"` go through the
#' distribution's inverse CDF, and `"raw_quantiles"` returns the sorted
#' grid itself.
#'
#' @param grid Numeric matrix `(n_rows, n_levels)`.
#' @param alpha Level probabilities.
#' @param type One of `"mean"`, `"median"`, `"quantiles"`,
#'   `"raw_quantiles"`.
#' @param quantiles Probabilities for `type = "quantiles"`.
#' @keywords internal
quantile_dist_stat <- function(grid, alpha, type = "mean",
                               quantiles = c(0.1, 0.5, 0.9)) {
  dist <- quantile_dist(grid, alpha)
  switch(
    type,
    "mean" = rowMeans(dist$quantiles),
    "median" = drop(quantile_dist_icdf(dist, 0.5)),
    "quantiles" = quantile_dist_icdf(dist, quantiles),
    "raw_quantiles" = dist$quantiles,
    cli::cli_abort("Unknown output type: {.val {type}}.")
  )
}
