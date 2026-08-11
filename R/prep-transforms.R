# Model-agnostic column transforms shared by every backend.
#
# All three reference implementations (TabPFN, TabICL, TabFM) build
# ensemble members by applying the same small family of preprocessors to
# the design matrix: drop constant columns, rescale robustly, quantile-
# or power-transform, append SVD components, append a row fingerprint,
# then permute columns and (for classifiers) rotate the class labels.
#
# The implementations here are ports of the *Python* semantics, matched
# against sklearn / the reference packages rather than rewritten to the
# nearest R idiom -- `quantile(type = 7)` for numpy's linear
# interpolation, population variance for sklearn's `StandardScaler`,
# and so on. Keep them that way: parity depends on it.

# Interpret 16 hex chars as an unsigned 64-bit integer, then divide by
# (2^64 - 1) to produce a float in [0, 1]. Matches Python's
#   `(int(sha256(row).hexdigest(), 16) & (2**64 - 1)) / (2**64 - 1)`
# to within float64 rounding (<=1 ULP).
# @keywords internal
.hex64_to_unit_float <- function(hex_digest) {
  low16 <- substr(hex_digest,
                  nchar(hex_digest) - 15L,
                  nchar(hex_digest))                        # 16 hex chars = 64 bits
  # Combine from 4 x 16-bit chunks. Each strtoi fits safely in a double.
  p1 <- strtoi(substr(low16,  1L,  4L), base = 16L)         # bits 48..63
  p2 <- strtoi(substr(low16,  5L,  8L), base = 16L)         # bits 32..47
  p3 <- strtoi(substr(low16,  9L, 12L), base = 16L)         # bits 16..31
  p4 <- strtoi(substr(low16, 13L, 16L), base = 16L)         # bits  0..15
  value <- p1 * 2^48 + p2 * 2^32 + p3 * 2^16 + p4
  value / (2^64 - 1)
}

# sha256(row_bytes || salt_bytes) -> unit float
# @keywords internal
.hash_row_to_unit_float <- function(row_bytes, salt_uint64) {
  require_suggested("digest")
  salt_le <- .uint64_to_le_raw(salt_uint64)
  combined <- c(as.raw(row_bytes), salt_le)
  hex <- digest::digest(combined, algo = "sha256", serialize = FALSE)
  .hex64_to_unit_float(hex)
}

# Encode an integer salt as 8-byte little-endian raw.
# @keywords internal
.uint64_to_le_raw <- function(n) {
  bytes <- integer(8L)
  x <- n
  for (i in 1:8) {
    bytes[i] <- x %% 256
    x <- x %/% 256
  }
  as.raw(bytes)
}

# NumPy's `np.around(a, decimals)` for floats: multiply by 10^decimals,
# round half to even, divide back. R's own `round()` deliberately does
# something better-conditioned, so it cannot be used where the *bytes*
# have to match.
# @keywords internal
.np_around <- function(x, decimals) {
  s <- 10^decimals
  r <- x * s
  # `round()` at digits = 0 is round-half-to-even, matching C `rint`
  # under the default rounding mode, which is what NumPy uses.
  finite <- is.finite(r)
  r[finite] <- round(r[finite])
  r / s
}

#' Compute per-row fingerprint values for X, matching Python's
#' `AddFingerprintFeaturesStep`.
#'
#' Salt = `n_train * n_features` where both are taken from the TRAIN
#' shape (not combined). For `is_test = FALSE`, duplicate rows are
#' re-hashed with incrementing offsets until all hashes are unique.
#' For `is_test = TRUE`, duplicates share the same hash.
#'
#' @param X Numeric matrix with shape `(n, n_features)`, dtype matching
#'   what Python saw at fingerprint time (float32 if post-primary).
#' @param salt_uint64 The `n_cells = n_train_rows * n_features` salt
#'   fitted on training data. Pass the TRAIN salt when transforming
#'   both train and test.
#' @param is_test Logical. TRUE for test rows (no collision resolution).
#' @return Numeric vector of length `nrow(X)` with values in `[0, 1]`.
#' @keywords internal
apply_fingerprint <- function(X, salt_uint64, is_test = FALSE) {
  X_arr <- as.matrix(X)
  storage.mode(X_arr) <- "double"
  n <- nrow(X_arr); f <- ncol(X_arr)
  # The reference hashes `np.around(X, 12).tobytes()` -- float64, 8 bytes
  # per value, little-endian.
  #
  # The rounding must be bit-identical, not merely close: a 1-ULP
  # difference changes the bytes and therefore the entire SHA-256, which
  # turns one row's fingerprint into an unrelated number. R's `round()`
  # uses a long-double algorithm that is *more* accurate than NumPy's,
  # and the two disagree on a small fraction of values. Reproduce
  # NumPy's `around` exactly instead: scale, round-half-to-even, unscale.
  X_rounded <- .np_around(X_arr, 12L)
  row_bytes_list <- lapply(seq_len(n), function(i) {
    writeBin(as.vector(X_rounded[i, ], mode = "numeric"),
             raw(), size = 8L, endian = "little")
  })

  out <- numeric(n)
  if (is_test) {
    for (i in seq_len(n)) {
      out[i] <- .hash_row_to_unit_float(row_bytes_list[[i]], salt_uint64)
    }
  } else {
    seen <- new.env(hash = TRUE, parent = emptyenv())
    counter <- new.env(hash = TRUE, parent = emptyenv())
    for (i in seq_len(n)) {
      rb <- row_bytes_list[[i]]
      h_base <- .hash_row_to_unit_float(rb, salt_uint64)
      key_base <- format(h_base, digits = 17L)
      add_offset <- counter[[key_base]]
      if (is.null(add_offset)) add_offset <- 0L
      h <- if (add_offset == 0L) h_base else
        .hash_row_to_unit_float(rb, salt_uint64 + add_offset)
      # Resolve further collisions
      retries <- 0L
      while (!is.null(seen[[format(h, digits = 17L)]])) {
        add_offset <- add_offset + 1L
        retries <- retries + 1L
        if (retries > 100L) {
          stop("Fingerprint hash collision not resolved after 100 retries on row ", i)
        }
        h <- .hash_row_to_unit_float(rb, salt_uint64 + add_offset)
      }
      out[i] <- h
      seen[[format(h, digits = 17L)]] <- TRUE
      counter[[key_base]] <- add_offset + 1L
    }
  }
  out
}


# ---------------------------------------------------------------------------
# Feature shift (column permutation)
# ---------------------------------------------------------------------------

#' Apply a column permutation to X.
#'
#' @param X     Numeric matrix `(n, n_features)`.
#' @param perm  Integer vector of 1-indexed column indices (length =
#'   ncol(X)). Produced by the ensemble-config generator or dumped
#'   from Python.
#' @keywords internal
apply_feature_shift <- function(X, perm) {
  stopifnot(length(perm) == ncol(X))
  as.matrix(X)[, perm, drop = FALSE]
}


# ---------------------------------------------------------------------------
# Class permutation (classifier only)
# ---------------------------------------------------------------------------

#' Permute integer class labels for a classifier ensemble member.
#'
#' @param y    Integer vector of class labels in `0..(n_classes-1)`.
#' @param perm Integer vector -- `perm[i]` is the new label for old label
#'   `i-1` (R 1-indexed: `perm[1]` = new label for class 0). Length =
#'   n_classes.
#' @return Integer vector of permuted labels.
#' @keywords internal
apply_class_permutation <- function(y, perm) {
  y_int <- as.integer(y)
  # y_int holds 0..(n_classes-1); perm is 1-indexed mapping.
  perm[y_int + 1L]
}

#' Invert a class permutation applied to predicted probabilities.
#'
#' If `class_perm[k] = p` mapped original class `k` to permuted label `p`
#' for training, then the model's column `p` equals `P(orig = k)`. So
#'   `probs_orig[, k] = probs[, class_perm[k]]`
#' directly. No `order()` / inverse computation needed.
#'
#' @param probs Matrix `(n_test, n_classes)` -- outputs from a single
#'   ensemble member, where column p was the permuted class p-1.
#' @param perm  `class_perm + 1L` (1-indexed). `perm[k]` is the column
#'   of `probs` that corresponds to original class `k-1`.
#' @keywords internal
apply_class_permutation_inverse <- function(probs, perm) {
  probs[, perm, drop = FALSE]
}


# ---------------------------------------------------------------------------
# Remove-constant-features (train-fit mask applied to train + test)
# ---------------------------------------------------------------------------

#' Drop columns that are constant across all training rows.
#' @keywords internal
remove_constant_features_fit <- function(X_train) {
  X <- as.matrix(X_train)
  first_row <- matrix(X[1L, ], nrow = 1L)

  # Reference:
  #   sel_ = ((X[0:1, :] == X).mean(axis=0) < 1.0) & ~np.all(np.isnan(X), axis=0)
  #
  # NumPy evaluates `NaN == NaN` as False, so any column holding a
  # missing value is *not* constant and survives. R's `==` yields NA
  # instead, which would propagate through `colMeans()` and silently
  # drop every column that has a single NaN in it -- so force NA to
  # FALSE first. The second term keeps all-NaN columns out.
  eq <- sweep(X, 2, first_row, `==`)
  eq[is.na(eq)] <- FALSE
  all_nan <- apply(X, 2L, function(col) all(is.na(col)))
  keep <- (colMeans(eq) < 1.0) & !all_nan

  if (!any(keep)) {
    stop("All features are constant -- the model cannot predict.")
  }
  which(keep)
}


# ---------------------------------------------------------------------------
# Which columns are categorical
# ---------------------------------------------------------------------------

#' Decide which predictors are categorical
#'
#' A port of `detect_feature_modalities`, restricted to the numeric
#' matrices this package takes. The reference's rule is deliberately
#' asymmetric, and the asymmetry is the whole point:
#'
#' * A column the caller **declares** categorical is taken at its word,
#'   unless its cardinality is so high that treating it as categorical
#'   would be useless — more than `max_unique` distinct values.
#' * A column nobody declared is only *inferred* categorical when it is
#'   extremely low-cardinality (fewer than `min_unique_numerical` distinct
#'   values) and there are enough rows to trust the count at all. A
#'   continuous column with a handful of repeats must not be swept up.
#'
#' Distinct values are counted with `NA` as a category, matching pandas'
#' `nunique(dropna = FALSE)`.
#'
#' Constant columns (one distinct value or fewer) are not categorical
#' unless declared; the reference calls them a modality of their own, and
#' they are dropped by [remove_constant_features_fit()] before any of this
#' matters.
#'
#' @param X Numeric matrix.
#' @param categorical_features Integer vector of 1-based column indices
#'   the caller declares categorical, or `NULL`.
#' @param max_unique Cardinality ceiling for a declared column to stay
#'   categorical. The reference's `MAX_UNIQUE_FOR_CATEGORICAL_FEATURES`.
#' @param min_unique_numerical Below this many distinct values, an
#'   undeclared column is inferred categorical.
#'   `MIN_UNIQUE_FOR_NUMERICAL_FEATURES`.
#' @param min_samples_inference Inference is only attempted on more than
#'   this many rows. `MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE`.
#' @return Sorted integer vector of 1-based categorical column indices.
#' @keywords internal
detect_categorical_features <- function(X,
                                        categorical_features = NULL,
                                        max_unique = 30L,
                                        min_unique_numerical = 4L,
                                        min_samples_inference = 100L) {
  X <- as.matrix(X)
  declared <- as.integer(categorical_features %||% integer())
  if (length(declared)) {
    bad <- declared[declared < 1L | declared > ncol(X)]
    if (length(bad)) {
      n_col <- ncol(X)
      cli::cli_abort(c(
        "{.arg categorical_features} names {length(bad)} column{?s} \\
         outside the predictor matrix.",
        x = "Out of range: {.val {bad}}; there are {n_col} predictors.",
        i = "Indices are 1-based positions in the predictor matrix."
      ))
    }
  }
  big_enough <- nrow(X) > min_samples_inference

  keep <- vapply(seq_len(ncol(X)), function(j) {
    col <- X[, j]
    # NA counts as one category, as pandas' `nunique(dropna = FALSE)` does.
    n_unique <- length(unique(col[!is.na(col)])) + as.integer(anyNA(col))
    reported <- j %in% declared
    if (n_unique <= 1L && !reported) return(FALSE)
    if (reported) return(n_unique <= max_unique)
    big_enough && n_unique < min_unique_numerical
  }, logical(1))

  which(keep)
}


# ---------------------------------------------------------------------------
# OrdinalEncoder
# ---------------------------------------------------------------------------

#' Fit sklearn's `OrdinalEncoder` on selected columns
#'
#' Categories are the column's sorted distinct values; codes are their
#' 0-based positions. Two details are load-bearing and easy to lose:
#'
#' * `NA` is a *category* for counting purposes (it sorts last, as NumPy
#'   puts it), so it occupies a code slot — but it is never emitted as a
#'   code. An `NA` in, an `NA` out.
#' * A value not seen during fitting also becomes `NA`, which is what
#'   `handle_unknown = "use_encoded_value", unknown_value = nan` does.
#'   The network's own missing-value channel then absorbs it, so an unseen
#'   level at predict time degrades rather than erroring.
#'
#' @param X_train Numeric matrix to fit on.
#' @param cols Integer vector of 1-based columns to encode.
#' @return Fit state for [transform_ordinal_encoder()], including
#'   `n_categories` per column (counting the `NA` slot), which is the
#'   length the shuffled-code permutations have to match.
#' @keywords internal
fit_ordinal_encoder <- function(X_train, cols) {
  X <- as.matrix(X_train)
  cols <- as.integer(cols)
  cats <- lapply(cols, function(j) sort(unique(X[, j][!is.na(X[, j])])))
  n_cats <- vapply(seq_along(cols), function(k) {
    length(cats[[k]]) + as.integer(anyNA(X[, cols[k]]))
  }, integer(1))
  list(cols = cols, categories = cats, n_categories = n_cats)
}

#' @rdname fit_ordinal_encoder
#' @param X Matrix to transform.
#' @param fit State from [fit_ordinal_encoder()].
#' @param mappings Optional list of integer permutations, one per encoded
#'   column, 0-based as the reference draws them. Applied to the codes
#'   (`ordinal_shuffled` and friends); `NULL` leaves codes in order.
#' @return Matrix with one column per entry of `fit$cols`, in that order.
#' @keywords internal
transform_ordinal_encoder <- function(X, fit, mappings = NULL) {
  X <- as.matrix(X)
  out <- matrix(NA_real_, nrow = nrow(X), ncol = length(fit$cols))
  for (k in seq_along(fit$cols)) {
    code <- match(X[, fit$cols[k]], fit$categories[[k]]) - 1
    if (!is.null(mappings)) {
      perm <- as.integer(mappings[[k]])
      ok <- !is.na(code)
      code[ok] <- perm[code[ok] + 1L]
    }
    out[, k] <- as.numeric(code)
  }
  out
}

#' Least-common category count, as the reference's filters use it
#'
#' `NA` counts as a category here too, so a column with a single missing
#' value has a least-common count of 1 and is filtered out by the
#' `common_categories` variants.
#' @keywords internal
least_common_category_count <- function(col) {
  if (!length(col)) return(0L)
  counts <- table(col, useNA = "ifany")
  as.integer(min(counts))
}


# ---------------------------------------------------------------------------
# StandardScaler(with_mean = FALSE) -- the NaN-aware scale sklearn fits
# ---------------------------------------------------------------------------

# sklearn computes `var_` with NaN-aware accumulators: the mean and the
# mean squared deviation are both taken over the *non-missing* entries of
# each column, with no (n-1) correction. A column that is constant (to
# within accumulation noise) gets a scale of 1 rather than 0, decided by
# `_is_constant_feature`, which compares the variance against a bound
# derived from the column's own magnitude -- not against a fixed epsilon,
# because a column of values around 1e8 has float noise around 1e-8.
# @keywords internal
fit_standard_scale_no_mean <- function(X) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  n_obs <- colSums(!is.na(X))
  mean_ <- colSums(X, na.rm = TRUE) / pmax(n_obs, 1)
  dev <- sweep(X, 2, mean_, `-`)
  var_ <- colSums(dev * dev, na.rm = TRUE) / pmax(n_obs, 1)

  # sklearn's `_is_constant_feature`, which always uses float64 eps.
  eps <- .Machine$double.eps
  upper <- n_obs * eps * var_ + (n_obs * eps * mean_) ^ 2
  scale_ <- sqrt(var_)
  scale_[var_ <= upper] <- 1
  scale_[!is.finite(scale_) | scale_ == 0] <- 1
  scale_
}


# ---------------------------------------------------------------------------
# NanHandlingPolynomialFeaturesStep -- pairwise products of scaled columns
# ---------------------------------------------------------------------------

#' Append pairwise products of the (rescaled) predictors
#'
#' A port of `NanHandlingPolynomialFeaturesStep`. Two things about it are
#' easy to miss:
#'
#' * It **replaces** the base columns with their
#'   `StandardScaler(with_mean = FALSE)`-scaled versions, rather than
#'   appending to the originals. The scaling is what keeps the products
#'   from spanning wildly different magnitudes.
#' * Which pairs are multiplied is drawn from the reference's NumPy
#'   generator, so it is part of the ensemble member's configuration, not
#'   something to recompute. Pass the dumped indices; the pure-R generator
#'   draws its own (see [generate_ensemble_configs_native()]).
#'
#' Missing values propagate: a NaN in either factor gives a NaN product,
#' which the network's own NaN handling then absorbs.
#'
#' @param X_train Numeric matrix, fitted on.
#' @param factor_1,factor_2 Integer vectors of equal length, 1-indexed
#'   column positions. Column `k` of the appended block is
#'   `X[, factor_1[k]] * X[, factor_2[k]]`.
#' @return Fit state for [transform_polynomial_features()].
#' @keywords internal
fit_polynomial_features <- function(X_train, factor_1, factor_2) {
  X <- as.matrix(X_train)
  if (length(factor_1) != length(factor_2)) {
    cli::cli_abort("{.arg factor_1} and {.arg factor_2} must be the same length.")
  }
  idx <- c(factor_1, factor_2)
  if (length(idx) && (min(idx) < 1L || max(idx) > ncol(X))) {
    cli::cli_abort(c(
      "Polynomial factor indices fall outside the {ncol(X)} predictors.",
      i = "Range seen: {min(idx)}..{max(idx)}."
    ))
  }
  list(scale = fit_standard_scale_no_mean(X),
       i1 = as.integer(factor_1), i2 = as.integer(factor_2))
}

#' @rdname fit_polynomial_features
#' @param X Matrix to transform.
#' @param fit State from [fit_polynomial_features()].
#' @return Matrix of `ncol(X) + length(factor_1)` columns: the scaled
#'   originals followed by the products.
#' @keywords internal
transform_polynomial_features <- function(X, fit) {
  Xs <- sweep(as.matrix(X), 2, fit$scale, `/`)
  if (!length(fit$i1)) return(Xs)
  cbind(Xs, Xs[, fit$i1, drop = FALSE] * Xs[, fit$i2, drop = FALSE])
}

#' How many polynomial columns a member appends
#'
#' `p * (p - 1) / 2 + p` distinct unordered pairs (including squares),
#' capped at `max_features`.
#' @param n_features Predictor count the step sees.
#' @param max_features Cap from the member's `polynomial_features` setting.
#' @keywords internal
n_polynomial_features <- function(n_features, max_features = NULL) {
  n <- (n_features * (n_features - 1L)) %/% 2L + n_features
  if (!is.null(max_features)) n <- min(as.integer(max_features), n)
  as.integer(n)
}


# ---------------------------------------------------------------------------
# SquashingScaler -- port of `tabpfn.preprocessing.steps.squashing_scaler_transformer.SquashingScaler`
# ---------------------------------------------------------------------------

#' Fit a SquashingScaler on training X.
#'
#' @param X Numeric matrix `(n, n_features)`; NaN/Inf handled column-wise.
#' @param max_absolute_value Default 3.0 (matches `"squashing_scaler_default"`).
#' @param quantile_range Length-2 numeric of lower/upper percentiles.
#'   Default `c(25, 75)`.
#' @return A list capturing the fitted per-column centers/scales/masks.
#' @keywords internal
fit_squashing_scaler <- function(X, max_absolute_value = 3.0,
                                  quantile_range = c(25, 75)) {
  X_arr <- as.matrix(X); storage.mode(X_arr) <- "double"
  # Map +/-Inf -> NaN for quantile computation (sign retained separately at transform).
  X_for_fit <- ifelse(is.infinite(X_arr), NaN, X_arr)

  col_max <- apply(X_for_fit, 2L, function(col) {
    finite <- col[is.finite(col)]
    if (length(finite) == 0L) NaN else max(finite)
  })
  col_min <- apply(X_for_fit, 2L, function(col) {
    finite <- col[is.finite(col)]
    if (length(finite) == 0L) NaN else min(finite)
  })
  zero_cols <- (col_max == col_min) & !is.na(col_max)

  medians <- apply(X_for_fit, 2L, function(col) {
    median(col[is.finite(col)])
  })
  q_low <- apply(X_for_fit, 2L, function(col) {
    quantile(col[is.finite(col)], probs = quantile_range[1] / 100,
             names = FALSE, type = 7, na.rm = TRUE)
  })
  q_hi <- apply(X_for_fit, 2L, function(col) {
    quantile(col[is.finite(col)], probs = quantile_range[2] / 100,
             names = FALSE, type = 7, na.rm = TRUE)
  })
  minmax_cols <- (q_low == q_hi) & !zero_cols & !is.na(q_low)
  robust_cols <- !(zero_cols | minmax_cols)

  scale <- rep(NA_real_, ncol(X_arr))
  scale[robust_cols] <- 1 / (q_hi[robust_cols] - q_low[robust_cols])
  # MinMax columns: scale = 2 / (max - min + eps_float32_tiny)
  eps32 <- 1.1754943508222875e-38
  scale[minmax_cols] <- 2 / (col_max[minmax_cols] - col_min[minmax_cols] + eps32)
  scale[zero_cols] <- 0

  list(
    max_absolute_value = max_absolute_value,
    quantile_range     = quantile_range,
    medians            = medians,
    scale              = scale,
    zero_cols          = zero_cols,
    minmax_cols        = minmax_cols,
    robust_cols        = robust_cols
  )
}

#' Apply a fitted SquashingScaler to data.
#' @keywords internal
transform_squashing_scaler <- function(X, fit) {
  X_arr <- as.matrix(X); storage.mode(X_arr) <- "double"
  inf_mask <- matrix(0, nrow(X_arr), ncol(X_arr))
  if (any(is.infinite(X_arr))) {
    sign_mat <- sign(X_arr)
    inf_mask <- ifelse(is.infinite(X_arr), sign_mat, 0)
    X_arr[is.infinite(X_arr)] <- NaN
  }

  X_tr <- X_arr
  for (j in seq_len(ncol(X_arr))) {
    if (fit$zero_cols[j]) {
      fin <- is.finite(X_tr[, j])
      X_tr[fin, j] <- 0
    } else if (fit$minmax_cols[j] || fit$robust_cols[j]) {
      X_tr[, j] <- fit$scale[j] * (X_arr[, j] - fit$medians[j])
    }
  }

  # Soft clip: X / sqrt(1 + (X / B)^2); Inf positions -> +/-B
  B <- fit$max_absolute_value
  X_clipped <- X_tr / sqrt(1 + (X_tr / B) ^ 2)
  X_clipped[inf_mask ==  1] <-  B
  X_clipped[inf_mask == -1] <- -B
  X_clipped
}


# ---------------------------------------------------------------------------
# AdaptiveQuantileTransformer -- port of sklearn.QuantileTransformer with
# adaptive n_quantiles. Used by regressor ensemble member
# `quantile_uni_coarse` (n_quantiles = max(N // 10, 2), uniform output).
# ---------------------------------------------------------------------------

#' Quantile count for a named `quantile_*` preset
#'
#' The presets differ only in how finely they grid the ECDF:
#' `_coarse` uses `n/10`, the plain form `n/5`, `_fine` every sample.
#' `_extrapolate` grids like the plain form and differs at transform time
#' instead — see [quantile_preset_extrapolate_ratio()].
#' @param preset Preset name, e.g. `"quantile_uni"`.
#' @param n_samples Training row count.
#' @keywords internal
quantile_preset_n_quantiles <- function(preset, n_samples) {
  n <- as.integer(n_samples)
  switch(
    preset,
    "quantile_uni"        = ,
    "quantile_uni_extrapolate" = ,
    "quantile_norm"       = max(n %/% 5L, 2L),
    "quantile_uni_coarse" = ,
    "quantile_norm_coarse" = max(n %/% 10L, 2L),
    "quantile_uni_fine"   = ,
    "quantile_norm_fine"  = n,
    cli::cli_abort("Unknown quantile preset: {.val {preset}}.")
  )
}

#' Extrapolation ratio for a named `quantile_*` preset
#'
#' `NULL` for every preset but `quantile_uni_extrapolate`, which asks for
#' `1`. See [transform_quantile_transformer()] for what it does.
#' @param preset Preset name.
#' @keywords internal
quantile_preset_extrapolate_ratio <- function(preset) {
  if (identical(preset, "quantile_uni_extrapolate")) 1 else NULL
}

#' Fit a per-column QuantileTransformer matching sklearn's uniform output.
#'
#' @param extrapolate_ratio `NULL` for the ordinary clipping transform, or
#'   a non-negative number to extrapolate past `[0, 1]` at transform time.
#'   Nothing extra is fitted for it: the training range it extrapolates
#'   from is `quantiles_[1, ]` and `quantiles_[n, ]`, which are the 0th and
#'   100th percentiles and so already the per-column min and max. The
#'   reference's two implementations read it both ways -- sklearn stores
#'   `nanmin`/`nanmax`, the torch one takes the outer breakpoints -- and
#'   they are the same numbers.
#' @keywords internal
fit_quantile_transformer <- function(X_train, n_quantiles,
                                      output_distribution = "uniform",
                                      extrapolate_ratio = NULL) {
  if (output_distribution != "uniform") {
    cli::cli_abort("Only 'uniform' output is implemented.")
  }
  if (!is.null(extrapolate_ratio)) {
    if (!is.numeric(extrapolate_ratio) || length(extrapolate_ratio) != 1L ||
        is.na(extrapolate_ratio) || extrapolate_ratio < 0) {
      cli::cli_abort("{.arg extrapolate_ratio} must be a single non-negative number.")
    }
  }
  X_arr <- as.matrix(X_train); storage.mode(X_arr) <- "double"
  n <- nrow(X_arr); p <- ncol(X_arr)
  # `compute_effective_n_quantiles`: never more than the rows available,
  # and never more than a fifth of the subsample cap -- past that,
  # `np.nanpercentile` inside sklearn dominates the fit time.
  effective_q <- max(1L, min(as.integer(n_quantiles), n, 20000L))
  refs <- seq(0, 1, length.out = effective_q)

  # quantiles_[:, j] = np.nanpercentile(X[:, j], refs*100)
  # sklearn uses numpy's default linear interpolation == R quantile type 7
  quantiles_ <- apply(X_arr, 2L, function(col) {
    col_clean <- col[!is.na(col)]
    quantile(col_clean, probs = refs, type = 7L, names = FALSE)
  })
  if (!is.matrix(quantiles_)) quantiles_ <- matrix(quantiles_, nrow = 1L)

  list(
    quantiles_           = quantiles_,                       # (effective_q, p)
    references_          = refs,                             # length effective_q
    output_distribution  = output_distribution,
    extrapolate_ratio    = extrapolate_ratio,
    n_features           = p
  )
}

#' Apply a fitted quantile transformer.
#'
#' Without `extrapolate_ratio` this is sklearn's `QuantileTransformer`:
#' every value maps to its rank in the training ECDF, and anything outside
#' the training range is clipped to 0 or 1.
#'
#' That clip is the problem `quantile_uni_extrapolate` exists to solve. A
#' test value ten times beyond the largest training value and one a hair
#' beyond it both come out as exactly 1, so the model cannot tell "at the
#' edge" from "far outside" -- precisely the distinction an
#' out-of-distribution prediction turns on. With a ratio set, values below
#' the training minimum or above its maximum are instead placed on a
#' linear continuation of the training range and clipped only at
#' `-ratio` / `1 + ratio`, so how far outside they are survives.
#'
#' Values *at* the boundary keep the ordinary 0 / 1, and constant columns
#' are left alone: with no training range there is nothing to extrapolate
#' along, and dividing by it would be a division by zero.
#' @keywords internal
transform_quantile_transformer <- function(X, fit) {
  X_arr <- as.matrix(X); storage.mode(X_arr) <- "double"
  out <- X_arr
  refs <- fit$references_
  for (j in seq_len(ncol(X_arr))) {
    q <- fit$quantiles_[, j]
    col <- X_arr[, j]
    # sklearn's symmetric-fold interp (handles ties continuously)
    f_fwd <- approx(q, refs, xout = col, rule = 2, ties = "ordered")$y
    f_bwd <- approx(-rev(q), -rev(refs), xout = -col, rule = 2, ties = "ordered")$y
    v <- 0.5 * (f_fwd - f_bwd)

    # sklearn then pins the fitted range's endpoints exactly, upper first
    # and lower second. On a well-spread column the fold already lands on
    # 0 and 1 there, so this changes nothing -- but on a *constant* column
    # every quantile is the same value, so it is simultaneously the lower
    # and the upper bound, the fold returns 0.5, and the order of these two
    # assignments is what decides the answer: 0, not 0.5.
    v[!is.na(col) & col == q[length(q)]] <- 1
    v[!is.na(col) & col == q[1L]] <- 0

    # Linear continuation past the training range, for the presets that
    # ask for it. `(x - x_max) / range + 1` simplifies to
    # `(x - x_min) / range`, so one normalisation serves both directions --
    # the reference makes the same simplification and says so.
    if (!is.null(fit$extrapolate_ratio)) {
      x_min <- q[1L]; x_max <- q[length(q)]
      rng <- x_max - x_min
      if (rng > 0) {
        ratio <- fit$extrapolate_ratio
        norm  <- (col - x_min) / rng
        below <- !is.na(col) & col < x_min
        above <- !is.na(col) & col > x_max
        v[below] <- pmin(pmax(norm[below], -ratio), 0)
        v[above] <- pmin(pmax(norm[above], 1), 1 + ratio)
      }
    }
    out[, j] <- v
  }
  out
}


# ---------------------------------------------------------------------------
# AddSVDFeaturesStep -- port of `tabpfn.preprocessing.steps.add_svd_features_step`
# ---------------------------------------------------------------------------

#' Compute n_components for the svd transformer preset.
#' @keywords internal
.svd_n_components <- function(global_name, n_samples, n_features) {
  divisor <- switch(
    global_name,
    "svd"                    = 2L,
    "svd_quarter_components" = 4L,
    stop("unknown global transformer name: ", global_name)
  )
  max(1L, min(n_samples %/% 10L + 1L, n_features %/% divisor))
}

# Column means for a mean-imputer, ignoring NaN. sklearn's
# `SimpleImputer(strategy="mean", keep_empty_features=True)` imputes an
# all-missing column with 0 rather than dropping it.
# @keywords internal
.impute_fit <- function(X) {
  m <- apply(X, 2L, function(col) {
    ok <- col[is.finite(col)]
    if (!length(ok)) 0 else mean(ok)
  })
  m
}

# Apply inf -> NaN followed by mean imputation, the `_make_finite_steps`
# pair the reference wraps around every scaler.
# @keywords internal
.impute_apply <- function(X, means) {
  X[!is.finite(X)] <- NA_real_
  for (j in seq_len(ncol(X))) {
    na <- is.na(X[, j])
    if (any(na)) X[na, j] <- means[[j]]
  }
  X
}

# sklearn's `svd_flip(u, v, u_based_decision = FALSE)`: for each
# component (row of VT, i.e. column of R's `v`), find the entry of
# largest magnitude and flip the sign so that entry is positive. ARPACK
# and LAPACK agree on the subspace but not on sign, so without this the
# appended SVD features can come out negated relative to the reference.
# @keywords internal
.svd_flip_v <- function(v) {
  for (k in seq_len(ncol(v))) {
    idx <- which.max(abs(v[, k]))
    s <- sign(v[idx, k])
    if (s != 0) v[, k] <- v[, k] * s
  }
  v
}

#' Fit the SVD global transformer
#'
#' Reproduces `get_svd_features_transformer()`:
#'
#'   inf->nan -> mean impute -> StandardScaler(with_mean=FALSE)
#'            -> inf->nan -> mean impute -> TruncatedSVD(arpack)
#'
#' The imputation matters: the untransformed `X` keeps its NaNs and is
#' handed to the model as-is, but the *appended* SVD columns are always
#' finite because the SVD only ever sees imputed data.
#'
#' Returns fit state usable by [transform_svd_features()].
#'
#' @keywords internal
fit_transform_svd_features <- function(X_train, global_name = "svd_quarter_components") {
  X_train <- as.matrix(X_train)
  n_samples <- nrow(X_train); n_features <- ncol(X_train)
  if (n_features < 2L) {
    return(list(is_no_op = TRUE))
  }
  n_components <- .svd_n_components(global_name, n_samples, n_features)

  pre_means <- .impute_fit(X_train)
  X_imp <- .impute_apply(X_train, pre_means)

  # sklearn's StandardScaler(with_mean=False).scale_ = sqrt(var), where var
  # is the POPULATION variance around the column mean (ddof=0). The
  # with_mean=False flag only suppresses subtracting the mean during
  # transform -- it does NOT change how scale_ is computed.
  col_mean <- colMeans(X_imp)
  col_var  <- colSums((X_imp - matrix(col_mean, n_samples, n_features,
                                      byrow = TRUE))^2) / n_samples
  col_std  <- sqrt(col_var)
  col_std[col_std == 0] <- 1

  X_scaled <- sweep(X_imp, 2, col_std, `/`)
  post_means <- .impute_fit(X_scaled)
  X_scaled <- .impute_apply(X_scaled, post_means)

  # Truncated SVD on scaled X: keep top n_components right singular
  # vectors. `transform()` in the reference is `X @ components_.T`, so
  # only V is needed downstream -- but the sign flip is decided from V
  # itself (u_based_decision = FALSE), which keeps this reproducible.
  svd_out <- svd(X_scaled, nu = 0, nv = n_components)
  v <- .svd_flip_v(svd_out$v[, seq_len(n_components), drop = FALSE])

  list(
    is_no_op     = FALSE,
    pre_means    = pre_means,
    col_scale    = col_std,
    post_means   = post_means,
    v            = v,
    n_components = n_components
  )
}

#' Apply a fitted SVD transformer.
#' @keywords internal
transform_svd_features <- function(X, fit) {
  if (isTRUE(fit$is_no_op)) return(NULL)
  X <- as.matrix(X)
  X_imp    <- .impute_apply(X, fit$pre_means)
  X_scaled <- sweep(X_imp, 2, fit$col_scale, `/`)
  X_scaled <- .impute_apply(X_scaled, fit$post_means)
  X_scaled %*% fit$v
}

# ---------------------------------------------------------------------------
# Yeo-Johnson power transform (sklearn PowerTransformer semantics)
# ---------------------------------------------------------------------------

#' Forward Yeo-Johnson power transform, given `lambda`.
#' @keywords internal
yeojohnson_forward <- function(x, lambda) {
  x <- as.numeric(x)
  out <- numeric(length(x))
  pos <- x >= 0
  eps <- .Machine$double.eps   # equivalent to np.spacing(1.0)

  if (abs(lambda) < eps) {
    out[pos] <- log1p(x[pos])
  } else {
    out[pos] <- expm1(lambda * log1p(x[pos])) / lambda
  }
  if (abs(lambda - 2) > eps) {
    out[!pos] <- -expm1((2 - lambda) * log1p(-x[!pos])) / (2 - lambda)
  } else {
    out[!pos] <- -log1p(-x[!pos])
  }
  out
}

#' Inverse Yeo-Johnson, given `lambda`.
#'
#' NaN output is expected, not exceptional: the transform is only
#' invertible where `1 + y * lambda > 0`, and the regressor deliberately
#' pushes the outermost bar-distribution borders far outside that range.
#' Those NaNs are the signal that a border is unusable, and
#' `.cancel_nan_borders()` / `.repair_borders()` consume them downstream.
#' The warnings `log1p()` would raise are therefore suppressed here.
#' @keywords internal
yeojohnson_inverse <- function(y, lambda) {
  y <- as.numeric(y)
  out <- numeric(length(y))
  pos <- y >= 0
  eps <- .Machine$double.eps

  suppressWarnings({
    if (abs(lambda) < eps) {
      out[pos] <- expm1(y[pos])
    } else {
      out[pos] <- expm1(log1p(y[pos] * lambda) / lambda)
    }
    if (abs(lambda - 2) > eps) {
      out[!pos] <- -expm1(log1p(-y[!pos] * (2 - lambda)) / (2 - lambda))
    } else {
      out[!pos] <- -expm1(-y[!pos])
    }
  })
  out
}


# ---------------------------------------------------------------------------
# Target standardization
# ---------------------------------------------------------------------------

#' Fit sklearn's `StandardScaler` on a target vector
#'
#' Both the TabFM and TabICL regressors are trained and evaluated on
#' standardized targets: their sklearn wrappers fit a `StandardScaler` on
#' `y`, feed the scaled values to the network, and inverse-transform
#' whatever comes back. Skipping this does not error — it produces
#' predictions on the wrong scale entirely.
#'
#' Population standard deviation (`ddof = 0`), with a zero variance
#' mapped to a scale of 1, matching `_handle_zeros_in_scale`.
#'
#' @param y Numeric vector.
#' @return A list with `mean` and `scale`.
#' @keywords internal
fit_target_scaler <- function(y) {
  y <- as.numeric(y)
  m <- mean(y)
  s <- sqrt(mean((y - m)^2))
  if (!is.finite(s) || s == 0) s <- 1
  list(mean = m, scale = s)
}

#' @rdname fit_target_scaler
#' @param scaler Result of [fit_target_scaler()].
#' @keywords internal
apply_target_scaler <- function(y, scaler) (as.numeric(y) - scaler$mean) / scaler$scale

#' @rdname fit_target_scaler
#' @keywords internal
invert_target_scaler <- function(y, scaler) y * scaler$scale + scaler$mean


#' Fit a min-max target scaler
#'
#' Mitra's regressor is trained on targets mapped to `[0, 1]` by
#' `(y - min) / (max - min)`, not standardized — its preprocessor calls
#' this `normalize_y` and inverts it on the way out. Different convention
#' from TabFM and TabICL, hence a second scaler rather than a flag.
#'
#' @param y Numeric vector.
#' @return A list with `min` and `range`.
#' @keywords internal
fit_minmax_scaler <- function(y) {
  y <- as.numeric(y)
  lo <- min(y); hi <- max(y)
  rng <- hi - lo
  if (!is.finite(rng) || rng == 0) rng <- 1
  list(min = lo, range = rng)
}

#' @rdname fit_minmax_scaler
#' @param scaler Result of [fit_minmax_scaler()].
#' @keywords internal
apply_minmax_scaler <- function(y, scaler) (as.numeric(y) - scaler$min) / scaler$range

#' @rdname fit_minmax_scaler
#' @keywords internal
invert_minmax_scaler <- function(y, scaler) y * scaler$range + scaler$min
