# The stock sklearn estimators the TabFM / TabICL wrappers chain, each
# checked against the library it was ported from.
#
# These are what stand between raw columns and the distribution these
# networks were trained on, so a silent drift here is a silent drop in
# accuracy with every stage-level network check still green. Regenerate
# the reference with `inst/parity/ensemble_reference.py`.

test_that("CustomStandardScaler matches, including its clip", {
  r <- ensemble_ref()
  X <- ref_mat(r$t$X); X_test <- ref_mat(r$t$X_test)

  fit <- fit_custom_standard_scaler(X)
  expect_equal(fit$mean, ref_num(r$s$css_mean), tolerance = 1e-13)
  expect_equal(fit$scale, ref_num(r$s$css_scale), tolerance = 1e-13)
  expect_equal(transform_custom_standard_scaler(X, fit),
               ref_mat(r$t$css_train), tolerance = 1e-13, ignore_attr = TRUE)
  expect_equal(transform_custom_standard_scaler(X_test, fit),
               ref_mat(r$t$css_test), tolerance = 1e-13, ignore_attr = TRUE)

  # The constant column is the point of the epsilon: without it the
  # scale is zero. With it, the column divides by 1e-6 and the clip is
  # what keeps the result finite.
  const_col <- ncol(X)
  expect_equal(fit$scale[const_col], 1e-6)
  expect_true(all(abs(transform_custom_standard_scaler(X, fit)) <= 100))
})


test_that("OutlierRemover matches, and softens rather than clips", {
  r <- ensemble_ref()
  X <- ref_mat(r$t$X); X_test <- ref_mat(r$t$X_test)
  Xs <- transform_custom_standard_scaler(X, fit_custom_standard_scaler(X))
  Xs_test <- transform_custom_standard_scaler(
    X_test, fit_custom_standard_scaler(X))

  fit <- fit_outlier_remover(Xs, threshold = 4.0)
  expect_equal(fit$lower_bounds, ref_num(r$s$orm_lower), tolerance = 1e-12)
  expect_equal(fit$upper_bounds, ref_num(r$s$orm_upper), tolerance = 1e-12)
  expect_equal(transform_outlier_remover(Xs, fit), ref_mat(r$t$orm_train),
               tolerance = 1e-12, ignore_attr = TRUE)
  expect_equal(transform_outlier_remover(Xs_test, fit), ref_mat(r$t$orm_test),
               tolerance = 1e-12, ignore_attr = TRUE)

  # The bound is soft: a value far past it stays past it, compressed by
  # log1p rather than pinned. A hard clip would map every outlier to the
  # same number and lose the ordering among them.
  z <- matrix(c(0, 0, 0, 1000), ncol = 1L)
  f2 <- fit_outlier_remover(matrix(c(0, 0.5, 1, 1.5), ncol = 1L))
  out <- transform_outlier_remover(z, f2)
  expect_gt(out[4, 1], f2$upper_bounds[1])
  expect_lt(out[4, 1], 1000)
})


test_that("UniqueFeatureFilter drops only genuinely single-valued columns", {
  r <- ensemble_ref()
  fit <- fit_unique_feature_filter(ref_mat(r$t$X_uff))
  expect_identical(fit$keep, ref_lgl(r$s$uff_keep))

  # A two-valued column survives; a constant one does not.
  Z <- cbind(rep(1, 6), c(0, 0, 0, 1, 1, 1), 1:6)
  expect_identical(fit_unique_feature_filter(Z)$keep, c(FALSE, TRUE, TRUE))

  # NumPy collapses every NaN into one entry, so a constant column with a
  # missing value has two distinct values and stays.
  expect_identical(
    fit_unique_feature_filter(cbind(c(1, 1, 1, NA), c(2, 2, 2, 2)))$keep,
    c(TRUE, FALSE)
  )

  # Fewer rows than the threshold and nothing is dropped at all: a column
  # that looks constant in one row probably is not.
  expect_true(all(fit_unique_feature_filter(matrix(1, 1, 3))$keep))
})


test_that("UniqueFeatureFilter keeps one column of an all-constant table on request", {
  # tabicl >= 2.2.0 keeps the first column rather than returning nothing;
  # TabFM's copy of the filter does not, so this is opt-in.
  r <- ensemble_ref()
  X <- ref_mat(r$t$X_const)
  expect_identical(fit_unique_feature_filter(X, keep_one = TRUE)$keep,
                   ref_lgl(r$s$uff_keep_const))
  expect_false(any(fit_unique_feature_filter(X)$keep))
  # Only the all-dropped case is touched.
  Z <- cbind(rep(1, 6), c(0, 0, 0, 1, 1, 1), 1:6)
  expect_identical(fit_unique_feature_filter(Z, keep_one = TRUE)$keep,
                   c(FALSE, TRUE, TRUE))
})


test_that("Yeo-Johnson's log-likelihood matches scipy on all three branches", {
  r <- ensemble_ref()
  grid <- ref_num(r$s$yj_grid)
  for (nm in names(r$s$yj_cols)) {
    col <- ref_num(r$s$yj_cols[[nm]])
    expect_equal(vapply(grid, function(l) yeojohnson_llf(l, col), 0),
                 ref_num(r$s$yj_llf[[nm]]), tolerance = 1e-12, info = nm)
  }
})


test_that("the Yeo-Johnson lambda search matches scipy's fminbound", {
  r <- ensemble_ref()
  for (nm in names(r$s$yj_cols)) {
    col <- ref_num(r$s$yj_cols[[nm]])
    # Not bitwise: the likelihood agrees to about an ULP, and Brent's
    # accept/reject test is a strict comparison, so the two searches can
    # take different final steps and stop `xatol` (1.48e-8) apart. See
    # `yeojohnson_normmax()`.
    expect_equal(yeojohnson_normmax(col), r$s$yj_lambdas[[nm]],
                 tolerance = 1e-7, info = nm)
  }
})


test_that("PowerTransformer matches sklearn", {
  r <- ensemble_ref()
  X <- ref_mat(r$t$X); X_test <- ref_mat(r$t$X_test)
  css <- fit_custom_standard_scaler(X)
  Xs <- transform_custom_standard_scaler(X, css)
  Xs_test <- transform_custom_standard_scaler(X_test, css)

  fit <- fit_power_transformer(Xs, standardize = TRUE)
  expect_equal(fit$lambdas, ref_num(r$s$pt_lambdas), tolerance = 1e-7)
  expect_equal(transform_power_transformer(Xs, fit), ref_mat(r$t$pt_train),
               tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(transform_power_transformer(Xs_test, fit), ref_mat(r$t$pt_test),
               tolerance = 1e-6, ignore_attr = TRUE)

  # A constant column keeps lambda = 1, the identity, rather than being
  # optimised over a likelihood with no maximum.
  const <- fit_power_transformer(matrix(rep(3, 20), ncol = 1L))
  expect_equal(const$lambdas, 1)
})


test_that("QuantileTransformer matches sklearn in both output distributions", {
  r <- ensemble_ref()
  X <- ref_mat(r$t$X); X_test <- ref_mat(r$t$X_test)
  css <- fit_custom_standard_scaler(X)
  Xs <- transform_custom_standard_scaler(X, css)
  Xs_test <- transform_custom_standard_scaler(X_test, css)

  fit <- fit_sk_quantile_transformer(Xs, 1000L, "normal", subsample = NULL)
  # The knots inherit the few-ULP gap the scaler upstream already has
  # (`np.std` sums pairwise, R's does not). It does not propagate: both
  # sides compare each value against their *own* knots, so the ranks come
  # out identical and only the last bits move.
  expect_equal(fit$quantiles_, ref_mat(r$t$qt_quantiles), tolerance = 1e-13,
               ignore_attr = TRUE)
  expect_equal(transform_sk_quantile_transformer(Xs, fit),
               ref_mat(r$t$qt_train), tolerance = 1e-13, ignore_attr = TRUE)
  expect_equal(transform_sk_quantile_transformer(Xs_test, fit),
               ref_mat(r$t$qt_test), tolerance = 1e-13, ignore_attr = TRUE)

  fitu <- fit_sk_quantile_transformer(Xs, 19L, "uniform", subsample = NULL)
  expect_equal(transform_sk_quantile_transformer(Xs, fitu),
               ref_mat(r$t$qtu_train), tolerance = 1e-13, ignore_attr = TRUE)
  expect_equal(transform_sk_quantile_transformer(Xs_test, fitu),
               ref_mat(r$t$qtu_test), tolerance = 1e-13, ignore_attr = TRUE)
})


test_that("a tied column maps to the middle of its rank run", {
  # This is what the forward/backward interpolation is for, and it only
  # comes out right if both directions pick the end of the tie run NumPy
  # picks. Getting it wrong shifts tied values by half a rank step --
  # a wrong answer that still looks entirely plausible.
  x <- matrix(c(0, 1, 1, 1, 2), ncol = 1L)
  fit <- fit_sk_quantile_transformer(x, 5L, "uniform", subsample = NULL)
  out <- transform_sk_quantile_transformer(x, fit)
  expect_equal(out[1, 1], 0)
  expect_equal(out[5, 1], 1)
  # The three tied values sit at ranks 1..3 of 0..4, so the midpoint is 2/4.
  expect_equal(unique(out[2:4, 1]), 0.5)
})


test_that("RobustScaler matches sklearn with unit_variance", {
  r <- ensemble_ref()
  X <- ref_mat(r$t$X)
  Xs <- transform_custom_standard_scaler(X, fit_custom_standard_scaler(X))
  fit <- fit_robust_scaler(Xs, unit_variance = TRUE)
  expect_equal(fit$center, ref_num(r$s$rs_center), tolerance = 1e-13)
  expect_equal(fit$scale, ref_num(r$s$rs_scale), tolerance = 1e-13)
  expect_equal(transform_robust_scaler(Xs, fit), ref_mat(r$t$rs_train),
               tolerance = 1e-13, ignore_attr = TRUE)
})


test_that("the whole PreprocessingPipeline matches, per normalisation method", {
  r <- ensemble_ref()
  X <- ref_mat(r$t$X); X_test <- ref_mat(r$t$X_test)
  for (m in unlist(r$s$pipe_methods)) {
    fit <- fit_preprocessing_pipeline(X, m, quantile_subsample = NULL)
    expect_equal(fit$X_transformed, ref_mat(r$t[[paste0("pipe_", m, "_train")]]),
                 tolerance = 1e-6, ignore_attr = TRUE, info = m)
    expect_equal(transform_preprocessing_pipeline(X_test, fit),
                 ref_mat(r$t[[paste0("pipe_", m, "_test")]]),
                 tolerance = 1e-6, ignore_attr = TRUE, info = m)
  }
  expect_error(fit_preprocessing_pipeline(X, "nope"), "Unknown normalization")
})


test_that("the quantile normalisers refuse to guess at a NumPy subsample", {
  # Above sklearn's row cap the reference fits on a NumPy-drawn subsample.
  # Reproducing *which* rows would mean reproducing NumPy's RandomState,
  # so this says so rather than quietly fitting on a different sample.
  X <- matrix(seq_len(60), ncol = 2L)
  expect_error(fit_sk_quantile_transformer(X, 10L, "normal", subsample = 10L),
               "subsample cap")
  expect_silent(fit_sk_quantile_transformer(X, 10L, "normal", subsample = NULL))
})


test_that("np.interp's tie convention is reproduced", {
  # A value landing on a repeated breakpoint takes the *last* matching
  # index, which is what NumPy's binary search settles on.
  expect_equal(.np_interp(2, c(1, 2, 2, 2, 3), c(10, 20, 30, 40, 50)), 40)
  expect_equal(.np_interp(0, c(1, 2, 3), c(10, 20, 30)), 10)   # below range
  expect_equal(.np_interp(9, c(1, 2, 3), c(10, 20, 30)), 30)   # above range
  expect_equal(.np_interp(1.5, c(1, 2), c(10, 20)), 15)
  expect_true(is.na(.np_interp(NA_real_, c(1, 2), c(10, 20))))
})
