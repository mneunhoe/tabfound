# The scikit-learn transformers the TabFM / TabICL / Mitra wrappers use.
#
# `prep-transforms.R` holds TabPFN's preprocessing, which is a bespoke
# pipeline of the reference's own steps. This file is a different thing:
# the other three models reach for stock sklearn estimators, and the
# ensemble wrappers chain them in a fixed order. Porting them
# individually -- rather than folding them into one "normalise X"
# function -- is what lets the parity harness pin each one on its own,
# without any checkpoint.
#
# Everything here works in float64 and takes an `NA`-aware view of the
# data where the reference does (`np.nanmean` and friends).
#
# References:
#   sklearn/preprocessing/_data.py  (QuantileTransformer, PowerTransformer,
#                                    RobustScaler, StandardScaler)
#   scipy/stats/_morestats.py       (yeojohnson_normmax, yeojohnson_llf)
#   scipy/optimize/_optimize.py     (_minimize_scalar_bounded)
#   tabicl/_sklearn/preprocessing.py, tabfm/src/classifier_and_regressor.py

# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

.SK_EPS <- .Machine$double.eps

#' `np.interp`, including its behaviour on repeated breakpoints
#'
#' R's [stats::approx()] and NumPy's `interp` agree everywhere except on
#' a value that lands exactly on a *repeated* breakpoint, and there they
#' disagree by a whole interval: NumPy's binary search returns the **last**
#' matching index, R's bisection may return either.
#'
#' That looks like a corner case and is not. `QuantileTransformer`
#' interpolates a column against its own quantiles, so every training
#' value *is* a breakpoint, and any column with ties has repeated ones.
#' The transformer then interpolates forwards and backwards and averages
#' the two, precisely to land in the middle of a tie run — which only
#' works if each direction picks the end of the run NumPy picks. Getting
#' it wrong shifts tied values by half a rank step.
#'
#' @param x Values to interpolate at.
#' @param xp Ascending breakpoints.
#' @param fp Values at the breakpoints.
#' @keywords internal
.np_interp <- function(x, xp, fp) {
  n <- length(xp)
  out <- rep(NA_real_, length(x))
  ok <- !is.na(x)
  if (!any(ok)) return(out)
  xv <- x[ok]
  # `findInterval` counts breakpoints <= x, so a tie run yields its last
  # index -- the same index NumPy's search settles on.
  j <- findInterval(xv, xp)
  res <- numeric(length(xv))
  below <- j == 0L
  atop  <- j >= n
  mid   <- !below & !atop
  res[below] <- fp[1L]
  res[atop]  <- fp[n]
  if (any(mid)) {
    jm <- j[mid]
    dx <- xp[jm + 1L] - xp[jm]
    slope <- (fp[jm + 1L] - fp[jm]) / dx
    v <- slope * (xv[mid] - xp[jm]) + fp[jm]
    # NumPy's own fallbacks when the first form is not finite.
    bad <- is.na(v)
    if (any(bad)) {
      v[bad] <- (slope * (xv[mid] - xp[jm + 1L]) + fp[jm + 1L])[bad]
      still <- is.na(v) & (xp[jm] == xp[jm + 1L])
      if (any(still)) v[still] <- fp[jm][still]
    }
    res[mid] <- v
  }
  out[ok] <- res
  out
}

#' `np.nanpercentile(x, p)` with NumPy's exact arithmetic
#'
#' Nominally this is R's `quantile(type = 7)`, and on most data the two
#' agree to the last bit. They are not the same expression, though:
#' NumPy's `lerp` interpolates up from the lower point while the weight
#' is below 0.5 and *down from the upper point* once it is above, where R
#' always writes `(1-h)*lo + h*hi`.
#'
#' A 1-ULP disagreement would normally be beneath notice. It is not here.
#' `QuantileTransformer` fitted with `n_quantiles = n_samples` produces
#' knots that are supposed to *be* the sorted data, and the transform then
#' tests `q <= x` against them. One knot a single ULP above its own data
#' value flips that test, and the value's rank — and hence its transformed
#' output — moves by a whole step.
#'
#' `p` is a **percentage**, as `np.nanpercentile` takes it, and the
#' division by 100 happens here — deliberately. sklearn passes
#' `references_ * 100` and NumPy divides it straight back, and that round
#' trip is not the identity in floating point: `10/59 * 100 / 100` misses
#' `10/59` by an ULP, which is enough to turn an exact knot index into an
#' interpolation between two neighbours. Doing the multiply outside and
#' the divide inside is what reproduces it.
#' @keywords internal
.nanquantile <- function(x, p) {
  x <- sort(x[!is.na(x)])
  n <- length(x)
  if (!n) return(rep(NA_real_, length(p)))
  if (n == 1L) return(rep(x, length(p)))
  p <- p / 100
  vi <- (n - 1) * p
  prev0 <- floor(vi)
  p_idx <- prev0
  n_idx <- prev0 + 1
  above <- vi >= n - 1
  below <- vi < 0
  p_idx[above] <- n - 1; n_idx[above] <- n - 1
  p_idx[below] <- 0;     n_idx[below] <- 0
  a <- x[p_idx + 1L]
  b <- x[n_idx + 1L]
  gamma <- vi - p_idx
  res <- a + (b - a) * gamma
  hi <- gamma >= 0.5
  if (any(hi)) res[hi] <- (b - (b - a) * (1 - gamma))[hi]
  res
}

# `_handle_zeros_in_scale`: a scale below `10 * eps` is a constant
# feature in disguise, and dividing by it amplifies noise rather than
# normalising anything.
# @keywords internal
.handle_zeros_in_scale <- function(scale) {
  scale[!is.finite(scale) | scale < 10 * .SK_EPS] <- 1
  scale
}

# `_is_constant_feature`: the two-pass variance algorithm's own error
# bound. A column whose computed variance is below it carries no
# information the arithmetic can be trusted about.
# @keywords internal
.is_constant_feature <- function(var, mean_, n_samples) {
  upper <- n_samples * .SK_EPS * var + (n_samples * mean_ * .SK_EPS)^2
  var <= upper
}

# Column means/sds as `np.mean` / `np.std` see them (population, ddof=0).
# @keywords internal
.col_mean <- function(X) colMeans(X)

# @keywords internal
.col_sd <- function(X, ddof = 0L, na.rm = FALSE) {
  vapply(seq_len(ncol(X)), function(j) {
    v <- X[, j]
    if (na.rm) v <- v[!is.na(v)]
    n <- length(v)
    if (n == 0L) return(NA_real_)
    if (n - ddof <= 0L) return(0)
    m <- mean(v)
    sqrt(sum((v - m)^2) / (n - ddof))
  }, numeric(1))
}

# @keywords internal
.col_nanmean <- function(X) {
  vapply(seq_len(ncol(X)), function(j) {
    v <- X[, j]; v <- v[!is.na(v)]
    if (!length(v)) NA_real_ else mean(v)
  }, numeric(1))
}


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

#' Fit sklearn's `StandardScaler` column-wise
#'
#' Population standard deviation, with near-constant columns given a
#' scale of 1. Used inside [fit_power_transformer()] and the
#' `quantile_rtdl` pipeline, where the reference chains one on the end.
#'
#' @param X Numeric matrix.
#' @return A list with `mean` and `scale`.
#' @keywords internal
fit_standard_scaler <- function(X) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  m <- .col_mean(X)
  s <- .col_sd(X, ddof = 0L)
  list(mean = m, scale = .handle_zeros_in_scale(s))
}

#' @rdname fit_standard_scaler
#' @param fit Result of [fit_standard_scaler()].
#' @keywords internal
transform_standard_scaler <- function(X, fit) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  sweep(sweep(X, 2L, fit$mean, `-`), 2L, fit$scale, `/`)
}


# ---------------------------------------------------------------------------
# CustomStandardScaler  (TabICL + TabFM)
# ---------------------------------------------------------------------------

#' Fit the wrappers' `CustomStandardScaler`
#'
#' Not sklearn's: both wrappers ship their own, and it differs in two
#' ways that matter. `epsilon` is *added* to the standard deviation
#' rather than triggering a constant-column branch, so a constant column
#' comes out as `(x - mean) / 1e-6` — a huge number, which the clip then
#' catches. And the output is clipped to `[clip_min, clip_max]`, which is
#' the only thing standing between a degenerate column and an infinity
#' downstream.
#'
#' @param X Numeric matrix.
#' @param clip_min,clip_max Output clipping bounds.
#' @param epsilon Added to every column's standard deviation.
#' @keywords internal
fit_custom_standard_scaler <- function(X, clip_min = -100, clip_max = 100,
                                       epsilon = 1e-6) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  list(mean = .col_mean(X), scale = .col_sd(X, ddof = 0L) + epsilon,
       clip_min = clip_min, clip_max = clip_max)
}

#' @rdname fit_custom_standard_scaler
#' @param fit Result of [fit_custom_standard_scaler()].
#' @keywords internal
transform_custom_standard_scaler <- function(X, fit) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  z <- sweep(sweep(X, 2L, fit$mean, `-`), 2L, fit$scale, `/`)
  pmin(pmax(z, fit$clip_min), fit$clip_max)
}


# ---------------------------------------------------------------------------
# OutlierRemover  (TabICL + TabFM, identical source)
# ---------------------------------------------------------------------------

#' Fit the two-stage z-score outlier bounds
#'
#' Stage one flags anything beyond `threshold` sample standard deviations
#' and blanks it; stage two recomputes the mean and sd without those
#' points, so a single wild value cannot inflate the bounds that are
#' supposed to catch it.
#'
#' Note the `ddof`: the reference uses the *sample* standard deviation
#' here (`ddof = 1`), unlike every other scaler in this file, and drops
#' to `ddof = 0` only for a single row.
#'
#' @param X Numeric matrix.
#' @param threshold Z-score beyond which a value is an outlier.
#' @keywords internal
fit_outlier_remover <- function(X, threshold = 4.0) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  n <- nrow(X)
  ddof <- if (n > 1L) 1L else 0L

  means <- .col_nanmean(X)
  sds   <- pmax(.col_sd(X, ddof = ddof, na.rm = TRUE), 1e-6)

  lower <- means - threshold * sds
  upper <- means + threshold * sds
  X_clean <- X
  out <- (X < matrix(lower, n, ncol(X), byrow = TRUE)) |
         (X > matrix(upper, n, ncol(X), byrow = TRUE))
  out[is.na(out)] <- FALSE
  X_clean[out] <- NA_real_

  means <- .col_nanmean(X_clean)
  sds   <- pmax(.col_sd(X_clean, ddof = ddof, na.rm = TRUE), 1e-6)

  list(means = means, stds = sds,
       lower_bounds = means - threshold * sds,
       upper_bounds = means + threshold * sds)
}

#' Apply the fitted outlier bounds
#'
#' Not a hard clip: the bound is softened by `log1p(|x|)`, so a value far
#' outside stays outside, just compressed. The ordering is not
#' commutative — the reference applies the lower bound first and then the
#' upper, and both use the *already-updated* `x` in their `log1p`.
#'
#' @param X Numeric matrix.
#' @param fit Result of [fit_outlier_remover()].
#' @keywords internal
transform_outlier_remover <- function(X, fit) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  n <- nrow(X); p <- ncol(X)
  lo <- matrix(fit$lower_bounds, n, p, byrow = TRUE)
  hi <- matrix(fit$upper_bounds, n, p, byrow = TRUE)
  X <- pmax(-log1p(abs(X)) + lo, X)
  pmin(log1p(abs(X)) + hi, X)
}


# ---------------------------------------------------------------------------
# UniqueFeatureFilter  (TabICL + TabFM, identical source)
# ---------------------------------------------------------------------------

#' Which columns survive the unique-value filter
#'
#' A column with `threshold` or fewer distinct values is dropped. Two
#' details are load-bearing:
#'
#' * `np.unique` collapses all `NaN`s into a single entry (numpy >= 1.21),
#'   so a column of one value plus missings has *two* distinct values and
#'   survives. Counting `NA` as one category here matches that.
#' * With `n_samples <= threshold` nothing is dropped at all. With that
#'   few rows a column can look constant by accident, and the reference
#'   would rather keep a useless column than lose a useful one.
#'
#' @param X_train Numeric matrix.
#' @param threshold Columns with at most this many distinct values go.
#' @return A list with the logical `keep` mask and `n_features_out`.
#' @keywords internal
fit_unique_feature_filter <- function(X_train, threshold = 1L) {
  X <- as.matrix(X_train); storage.mode(X) <- "double"
  p <- ncol(X)
  keep <- if (nrow(X) <= threshold) {
    rep(TRUE, p)
  } else {
    vapply(seq_len(p), function(j) {
      col <- X[, j]
      n_unique <- length(unique(col[!is.na(col)])) + as.integer(anyNA(col))
      n_unique > threshold
    }, logical(1))
  }
  list(keep = keep, n_features_in = p, n_features_out = sum(keep))
}

#' @rdname fit_unique_feature_filter
#' @param X Numeric matrix.
#' @param fit Result of [fit_unique_feature_filter()].
#' @keywords internal
transform_unique_feature_filter <- function(X, fit) {
  as.matrix(X)[, fit$keep, drop = FALSE]
}


# ---------------------------------------------------------------------------
# QuantileTransformer  (stock sklearn, both output distributions)
# ---------------------------------------------------------------------------
#
# `prep-transforms.R` already has a quantile transformer, but it is
# TabPFN's: it caps `n_quantiles` at 20000 and offers a linear
# extrapolation mode TabPFN v3 asks for. This is the plain sklearn
# estimator, which the other wrappers construct directly and which needs
# the `normal` output distribution they use.

#' Fit sklearn's `QuantileTransformer`
#'
#' @param X_train Numeric matrix.
#' @param n_quantiles Requested number of knots. The fitted count is
#'   `max(1, min(n_quantiles, n_samples))`.
#' @param output_distribution `"uniform"` or `"normal"`.
#' @param subsample Row cap above which sklearn subsamples before
#'   computing percentiles. Reproducing *which* rows it picks would mean
#'   reproducing NumPy's `RandomState`, so this errors rather than
#'   quietly fitting on a different sample; pass `NULL` to fit on all
#'   rows.
#' @keywords internal
fit_sk_quantile_transformer <- function(X_train, n_quantiles = 1000L,
                                        output_distribution = "uniform",
                                        subsample = 10000L) {
  if (!output_distribution %in% c("uniform", "normal")) {
    cli::cli_abort("{.arg output_distribution} must be {.val uniform} or {.val normal}.")
  }
  X <- as.matrix(X_train); storage.mode(X) <- "double"
  n <- nrow(X)
  if (!is.null(subsample) && subsample < n) {
    cli::cli_abort(c(
      "This dataset ({n} rows) is above the transformer's subsample cap \\
       ({subsample}).",
      x = "sklearn would fit the quantiles on a NumPy-drawn subsample, which \\
           this port cannot reproduce row for row.",
      i = "Pass {.code subsample = NULL} to fit on every row instead."
    ))
  }
  n_quantiles_ <- max(1L, min(as.integer(n_quantiles), n))
  refs <- seq(0, 1, length.out = n_quantiles_)
  # `references_ * 100`, exactly as sklearn hands them to NumPy.
  quantiles_ <- vapply(seq_len(ncol(X)),
                       function(j) .nanquantile(X[, j], refs * 100),
                       numeric(n_quantiles_))
  if (!is.matrix(quantiles_)) quantiles_ <- matrix(quantiles_, nrow = n_quantiles_)
  list(quantiles_ = quantiles_, references_ = refs,
       output_distribution = output_distribution, n_features = ncol(X))
}

# sklearn clips the normal output so the inverse stays finite; the bounds
# come from the normal quantile at `BOUNDS_THRESHOLD`.
.SK_BOUNDS_THRESHOLD <- 1e-7

#' @rdname fit_sk_quantile_transformer
#' @param X Numeric matrix.
#' @param fit Result of [fit_sk_quantile_transformer()].
#' @keywords internal
transform_sk_quantile_transformer <- function(X, fit) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  normal <- identical(fit$output_distribution, "normal")
  refs <- fit$references_
  out <- X
  for (j in seq_len(ncol(X))) {
    q <- fit$quantiles_[, j]
    col <- X[, j]
    lower_x <- q[1L]; upper_x <- q[length(q)]

    # Which entries get pinned to the endpoints. For `normal` the test is
    # a tolerance rather than equality, because the ppf blows up there.
    if (normal) {
      lower_idx <- !is.na(col) & (col - .SK_BOUNDS_THRESHOLD < lower_x)
      upper_idx <- !is.na(col) & (col + .SK_BOUNDS_THRESHOLD > upper_x)
    } else {
      lower_idx <- !is.na(col) & col == lower_x
      upper_idx <- !is.na(col) & col == upper_x
    }

    # Interpolate forwards and backwards and average, so a run of tied
    # training values maps to the middle of its rank range instead of one
    # of its ends.
    f_fwd <- .np_interp(col, q, refs)
    f_bwd <- .np_interp(-col, -rev(q), -rev(refs))
    v <- 0.5 * (f_fwd - f_bwd)
    v[upper_idx] <- 1
    v[lower_idx] <- 0

    if (normal) {
      v <- stats::qnorm(v)
      clip_min <- stats::qnorm(.SK_BOUNDS_THRESHOLD - .SK_EPS)
      clip_max <- stats::qnorm(1 - (.SK_BOUNDS_THRESHOLD - .SK_EPS))
      v <- pmin(pmax(v, clip_min), clip_max)
    }
    out[, j] <- v
  }
  out
}


# ---------------------------------------------------------------------------
# RTDLQuantileTransformer
# ---------------------------------------------------------------------------

#' Fit the RTDL variant of the quantile transformer
#'
#' Two differences from the stock estimator: the knot count is tied to
#' the row count (`n / 30`, floored at 10) rather than fixed, and
#' Gaussian noise is added to the training data before fitting, which
#' breaks up ties so the empirical CDF has no flat spots.
#'
#' **The noise is the one thing here that is not bit-comparable.** The
#' reference draws it from NumPy's PCG64; this uses R's RNG, so a run
#' with `noise > 0` is reproducible under `set.seed()` but is not the
#' reference's draw. `noise = 0` is exact. This only bites the
#' `quantile_rtdl` normalisation method, which is not in either
#' wrapper's default list.
#'
#' @param X_train Numeric matrix.
#' @param noise Relative noise magnitude; `0` disables it.
#' @param n_quantiles Upper bound on the knot count.
#' @param output_distribution `"uniform"` or `"normal"`.
#' @param subsample Passed to [fit_sk_quantile_transformer()].
#' @keywords internal
fit_rtdl_quantile_transformer <- function(X_train, noise = 1e-3,
                                          n_quantiles = 1000L,
                                          output_distribution = "normal",
                                          subsample = NULL) {
  X <- as.matrix(X_train); storage.mode(X) <- "double"
  n_q <- max(min(nrow(X) %/% 30L, as.integer(n_quantiles)), 10L)
  X_fit <- X
  if (noise > 0) {
    stds <- .col_sd(X, ddof = 0L)
    noise_std <- noise / pmax(stds, noise)
    draw <- matrix(stats::rnorm(length(X)), nrow(X), ncol(X))
    X_fit <- X + sweep(draw, 2L, noise_std, `*`)
  }
  fit_sk_quantile_transformer(X_fit, n_quantiles = n_q,
                              output_distribution = output_distribution,
                              subsample = subsample)
}


# ---------------------------------------------------------------------------
# RobustScaler
# ---------------------------------------------------------------------------

#' Fit sklearn's `RobustScaler`
#'
#' Centres on the median and scales by the interquartile range. With
#' `unit_variance`, the IQR is further divided by the IQR of a standard
#' normal, so normally-distributed data comes out with unit variance
#' rather than unit IQR.
#'
#' @param X_train Numeric matrix.
#' @param quantile_range Length-2 vector of percentiles.
#' @param unit_variance Rescale so a normal sample has unit variance.
#' @keywords internal
fit_robust_scaler <- function(X_train, quantile_range = c(25, 75),
                              unit_variance = TRUE) {
  X <- as.matrix(X_train); storage.mode(X) <- "double"
  q_min <- quantile_range[1L]; q_max <- quantile_range[2L]
  center <- vapply(seq_len(ncol(X)), function(j) {
    v <- X[, j]; v <- v[!is.na(v)]
    if (!length(v)) NA_real_ else stats::median(v)
  }, numeric(1))
  scale_ <- vapply(seq_len(ncol(X)), function(j) {
    qs <- .nanquantile(X[, j], c(q_min, q_max))
    qs[2L] - qs[1L]
  }, numeric(1))
  scale_ <- .handle_zeros_in_scale(scale_)
  if (unit_variance) {
    adjust <- stats::qnorm(q_max / 100) - stats::qnorm(q_min / 100)
    scale_ <- scale_ / adjust
  }
  list(center = center, scale = scale_)
}

#' @rdname fit_robust_scaler
#' @param X Numeric matrix.
#' @param fit Result of [fit_robust_scaler()].
#' @keywords internal
transform_robust_scaler <- function(X, fit) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  sweep(sweep(X, 2L, fit$center, `-`), 2L, fit$scale, `/`)
}


# ---------------------------------------------------------------------------
# PowerTransformer (Yeo-Johnson)
# ---------------------------------------------------------------------------
#
# The forward and inverse transforms already live in `prep-transforms.R`
# (`yeojohnson_forward` / `yeojohnson_inverse`) because TabPFN's target
# transform uses them with a *given* lambda. What is new here is finding
# lambda by maximum likelihood, which is what `PowerTransformer` does and
# what makes it a fitted estimator rather than a function.
#
# sklearn hands that off to `scipy.stats.yeojohnson`, which maximises the
# profile log-likelihood with `optimize.fminbound` over data-derived
# bounds. Both are ported rather than approximated: R's `optimize()` is
# also Brent, but it brackets differently, and the lambda it lands on
# differs in the sixth digit — enough to show up as a diffuse
# preprocessing mismatch that would be very hard to attribute.

# `special.logsumexp` with signed weights: log|sum b_i exp(a_i)|, plus
# the sign. Only the two-term case the variance computation needs.
# @keywords internal
.logsumexp2_signed <- function(a1, a2) {
  m <- pmax(a1, a2)
  # b = (+1, -1): exp(a1 - m) - exp(a2 - m)
  s <- exp(a1 - m) - exp(a2 - m)
  list(log = m + log(abs(s)), sign = sign(s))
}

# `_log_var`: variance of `x` computed from `log(x)`, in log space.
# @keywords internal
.log_var_from_log <- function(logx) {
  n <- length(logx)
  logmean <- .logsumexp(logx) - log(n)
  lse <- .logsumexp2_signed(logx, rep(logmean, n))
  .logsumexp(2 * lse$log) - log(n)
}

# @keywords internal
.logsumexp <- function(x) {
  m <- max(x)
  if (!is.finite(m)) return(m)
  m + log(sum(exp(x - m)))
}

#' Yeo-Johnson profile log-likelihood
#'
#' `l = -n/2 log(var(y)) + (lambda - 1) sum(sign(x) log1p(|x|))`, where
#' `y` is the transformed data.
#'
#' All-positive and all-negative columns take a log-space route to the
#' variance. That is not a micro-optimisation: for a large `|lambda|` the
#' transformed values overflow float64 outright, and the direct
#' computation returns `inf` where the log-space one returns a finite
#' number. The optimiser walks through exactly that region.
#'
#' @param lambda Transform parameter.
#' @param x Numeric vector, no missing values.
#' @keywords internal
yeojohnson_llf <- function(lambda, x) {
  n <- length(x)
  if (n == 0L) return(NA_real_)
  pos <- x >= 0

  logvar <- if (all(pos)) {
    if (abs(lambda) < .SK_EPS) {
      log(stats::var(log1p(x)) * (n - 1) / n)
    } else {
      .log_var_from_log(lambda * log1p(x)) - 2 * log(abs(lambda))
    }
  } else if (all(!pos)) {
    if (abs(lambda - 2) < .SK_EPS) {
      log(stats::var(log1p(-x)) * (n - 1) / n)
    } else {
      .log_var_from_log((2 - lambda) * log1p(-x)) - 2 * log(abs(2 - lambda))
    }
  } else {
    y <- yeojohnson_forward(x, lambda)
    sigma <- sum((y - mean(y))^2) / n
    # `np.log` of a subnormal variance is where the reference gives up.
    if (sigma >= 2.2250738585072014e-308) log(sigma) else -Inf
  }

  -n / 2 * logvar + (lambda - 1) * sum(sign(x) * log1p(abs(x)))
}

#' scipy's `optimize.fminbound`, ported
#'
#' Brent's bounded minimiser: golden-section steps with parabolic
#' interpolation when the last step justifies it. Ported step for step
#' because the iterate sequence, not just the limit, decides the answer
#' to the last few digits.
#'
#' @param f Function of one variable.
#' @param x1,x2 Bounds.
#' @param xatol Absolute tolerance on the minimiser.
#' @param maxfun Function-evaluation budget.
#' @keywords internal
scipy_fminbound <- function(f, x1, x2, xatol = 1e-5, maxfun = 500L) {
  if (x1 > x2) cli::cli_abort("The lower bound exceeds the upper bound.")
  sqrt_eps <- sqrt(2.2e-16)
  golden_mean <- 0.5 * (3 - sqrt(5))
  a <- x1; b <- x2
  fulc <- a + golden_mean * (b - a)
  nfc <- fulc; xf <- fulc
  rat <- 0; e <- 0
  x <- xf
  fx <- f(x)
  num <- 1L
  ffulc <- fx; fnfc <- fx
  xm <- 0.5 * (a + b)
  tol1 <- sqrt_eps * abs(xf) + xatol / 3
  tol2 <- 2 * tol1

  # `sign(0)` is 0 in both languages, and the reference adds the
  # `== 0` indicator to turn that into +1.
  sgn <- function(v) sign(v) + (v == 0)

  while (abs(xf - xm) > (tol2 - 0.5 * (b - a))) {
    golden <- TRUE
    if (abs(e) > tol1) {
      golden <- FALSE
      r <- (xf - nfc) * (fx - ffulc)
      q <- (xf - fulc) * (fx - fnfc)
      p <- (xf - fulc) * q - (xf - nfc) * r
      q <- 2 * (q - r)
      if (q > 0) p <- -p
      q <- abs(q)
      r <- e
      e <- rat
      if (abs(p) < abs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf)) {
        rat <- p / q
        x <- xf + rat
        if ((x - a) < tol2 || (b - x) < tol2) rat <- tol1 * sgn(xm - xf)
      } else {
        golden <- TRUE
      }
    }
    if (golden) {
      e <- if (xf >= xm) a - xf else b - xf
      rat <- golden_mean * e
    }
    x <- xf + sgn(rat) * max(abs(rat), tol1)
    fu <- f(x)
    num <- num + 1L

    if (fu <= fx) {
      if (x >= xf) a <- xf else b <- xf
      fulc <- nfc; ffulc <- fnfc
      nfc <- xf;   fnfc <- fx
      xf <- x;     fx <- fu
    } else {
      if (x < xf) a <- x else b <- x
      if (fu <= fnfc || nfc == xf) {
        fulc <- nfc; ffulc <- fnfc
        nfc <- x;    fnfc <- fu
      } else if (fu <= ffulc || fulc == xf || fulc == nfc) {
        fulc <- x; ffulc <- fu
      }
    }
    xm <- 0.5 * (a + b)
    tol1 <- sqrt_eps * abs(xf) + xatol / 3
    tol2 <- 2 * tol1
    if (num >= maxfun) break
  }
  xf
}

#' Maximum-likelihood Yeo-Johnson lambda
#'
#' `scipy.stats.yeojohnson_normmax(x)` with `brack = NULL`, which is what
#' `PowerTransformer` calls. The search bounds are derived from the data
#' rather than fixed: they are the lambdas at which the transform of the
#' largest observed value would over- or underflow, which is the widest
#' interval where the objective is even computable.
#'
#' This is the one transform in this file that is not bit-exact against
#' the reference, and the reason is worth stating precisely. The
#' likelihood agrees to about one ULP — the log-space variance goes
#' through a `logsumexp` whose summation order R cannot be made to share.
#' Brent's accept/reject test is a strict comparison, so a last-bit
#' difference occasionally sends the two searches down different final
#' steps. They then stop within `xatol` of each other, which is
#' `1.48e-8`: measured, the lambdas differ by up to ~2e-8 and the
#' transformed values by ~8e-8. That is an order of magnitude inside the
#' `preprocess:*` parity tolerance, and it is a property of the objective
#' being flat at its optimum rather than of either answer being wrong.
#'
#' @param x Numeric vector, missing values already dropped.
#' @keywords internal
yeojohnson_normmax <- function(x) {
  x <- as.numeric(x)
  x <- x[!is.na(x)]
  if (!length(x)) return(1)
  if (all(x == 0)) return(1)
  if (!all(is.finite(x))) cli::cli_abort("Yeo-Johnson input must be finite.")

  log1p_max_x <- log1p(20 * max(abs(x)))
  log_eps <- log(.SK_EPS)
  log_tiny <- (log(2.2250738585072014e-308) - log_eps) / 2
  log_max  <- (log(.Machine$double.xmax) + log_eps) / 2
  lb <- log_tiny / log1p_max_x
  ub <- log_max / log1p_max_x
  if (all(x < 0)) {
    tmp <- lb; lb <- 2 - ub; ub <- 2 - tmp
  } else if (any(x < 0)) {
    lb2 <- max(2 - ub, lb); ub2 <- min(2 - lb, ub)
    lb <- lb2; ub <- ub2
  }

  neg_llf <- function(lmb) {
    v <- yeojohnson_llf(lmb, x)
    if (is.infinite(v) && v > 0) v <- -Inf   # reject +inf likelihoods
    -v
  }
  scipy_fminbound(neg_llf, lb, ub, xatol = 1.48e-08)
}

#' Fit sklearn's `PowerTransformer(method = "yeo-johnson")`
#'
#' One lambda per column by maximum likelihood, then — with
#' `standardize` — a `StandardScaler` on the transformed training data.
#' Constant columns keep `lambda = 1`, the identity, rather than being
#' optimised over a likelihood that has no maximum.
#'
#' @param X_train Numeric matrix.
#' @param standardize Chain a `StandardScaler` on the output.
#' @keywords internal
fit_power_transformer <- function(X_train, standardize = TRUE) {
  X <- as.matrix(X_train); storage.mode(X) <- "double"
  n <- nrow(X); p <- ncol(X)
  means <- .col_mean(X)
  vars_ <- .col_sd(X, ddof = 0L)^2

  lambdas <- numeric(p)
  Xt <- X
  for (j in seq_len(p)) {
    if (.is_constant_feature(vars_[j], means[j], n)) {
      lambdas[j] <- 1
    } else {
      lambdas[j] <- yeojohnson_normmax(X[, j])
    }
    Xt[, j] <- yeojohnson_forward(X[, j], lambdas[j])
  }

  scaler <- if (standardize) fit_standard_scaler(Xt) else NULL
  list(lambdas = lambdas, scaler = scaler, n_features = p)
}

#' @rdname fit_power_transformer
#' @param X Numeric matrix.
#' @param fit Result of [fit_power_transformer()].
#' @keywords internal
transform_power_transformer <- function(X, fit) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  for (j in seq_len(ncol(X))) X[, j] <- yeojohnson_forward(X[, j], fit$lambdas[j])
  if (!is.null(fit$scaler)) X <- transform_standard_scaler(X, fit$scaler)
  X
}


# ---------------------------------------------------------------------------
# PreprocessingPipeline
# ---------------------------------------------------------------------------

#' The wrappers' shared preprocessing pipeline
#'
#' Scale, normalise, clip — in that order, and the order matters: the
#' normaliser is fitted on already-standardised data, and the outlier
#' bounds on already-normalised data.
#'
#' TabICL and TabFM ship byte-identical copies of this class, so one
#' implementation serves both. What differs between them is only which
#' `normalization_method` each ensemble member is handed.
#'
#' The fitted training output is cached in `X_transformed`, exactly as
#' the reference caches `X_transformed_` — every ensemble member sharing
#' a normalisation method reuses it rather than re-running the pipeline.
#'
#' @param X_train Numeric matrix.
#' @param normalization_method One of `"none"`, `"power"`, `"quantile"`,
#'   `"quantile_rtdl"`, `"robust"`.
#' @param outlier_threshold Passed to [fit_outlier_remover()].
#' @param quantile_subsample Row cap for the quantile normalisers; see
#'   [fit_sk_quantile_transformer()].
#' @keywords internal
fit_preprocessing_pipeline <- function(X_train, normalization_method = "power",
                                       outlier_threshold = 4.0,
                                       quantile_subsample = NULL) {
  X <- as.matrix(X_train); storage.mode(X) <- "double"
  if (ncol(X) == 0L) {
    return(list(normalization_method = normalization_method, n_features = 0L,
                scaler = NULL, normalizer = NULL, outlier = NULL,
                X_transformed = X))
  }

  scaler <- fit_custom_standard_scaler(X)
  X_scaled <- transform_custom_standard_scaler(X, scaler)

  normalizer <- NULL
  X_norm <- X_scaled
  X_min <- NULL; X_max <- NULL
  if (!identical(normalization_method, "none")) {
    X_min <- apply(X_scaled, 2L, min)
    X_max <- apply(X_scaled, 2L, max)
    normalizer <- switch(
      normalization_method,
      "power"   = list(kind = "power",
                       fit = fit_power_transformer(X_scaled, standardize = TRUE)),
      "quantile" = list(kind = "quantile",
                        fit = fit_sk_quantile_transformer(
                          X_scaled, n_quantiles = 1000L,
                          output_distribution = "normal",
                          subsample = quantile_subsample %||% 10000L)),
      "quantile_rtdl" = {
        qfit <- fit_rtdl_quantile_transformer(
          X_scaled, output_distribution = "normal",
          subsample = quantile_subsample)
        tmp <- transform_sk_quantile_transformer(X_scaled, qfit)
        list(kind = "quantile_rtdl", fit = qfit,
             std = fit_standard_scaler(tmp))
      },
      "robust" = list(kind = "robust",
                      fit = fit_robust_scaler(X_scaled, unit_variance = TRUE)),
      cli::cli_abort("Unknown normalization method: {.val {normalization_method}}.")
    )
    X_norm <- .apply_normalizer(X_scaled, normalizer)
  }

  outlier <- fit_outlier_remover(X_norm, threshold = outlier_threshold)
  list(
    normalization_method = normalization_method,
    n_features = ncol(X),
    scaler = scaler, normalizer = normalizer, outlier = outlier,
    X_min = X_min, X_max = X_max,
    X_transformed = transform_outlier_remover(X_norm, outlier)
  )
}

# @keywords internal
.apply_normalizer <- function(X, normalizer) {
  switch(
    normalizer$kind,
    "power"    = transform_power_transformer(X, normalizer$fit),
    "quantile" = transform_sk_quantile_transformer(X, normalizer$fit),
    "quantile_rtdl" = transform_standard_scaler(
      transform_sk_quantile_transformer(X, normalizer$fit), normalizer$std),
    "robust"   = transform_robust_scaler(X, normalizer$fit)
  )
}

#' @rdname fit_preprocessing_pipeline
#' @param X Numeric matrix.
#' @param fit Result of [fit_preprocessing_pipeline()].
#' @keywords internal
transform_preprocessing_pipeline <- function(X, fit) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  if (fit$n_features == 0L) return(X)
  X <- transform_custom_standard_scaler(X, fit$scaler)
  if (!is.null(fit$normalizer)) {
    # The reference wraps this in a try/except: a test value outside the
    # training range can make a normaliser raise, and the fallback is to
    # clip to the training range and retry. Nothing here raises on such a
    # value, so the clip is unreachable and deliberately not emulated.
    X <- .apply_normalizer(X, fit$normalizer)
  }
  transform_outlier_remover(X, fit$outlier)
}
