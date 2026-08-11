# Each shared column transform, checked against values produced by the
# library it was ported from (sklearn / NumPy / TabPFN's own step
# classes). Regenerate with:
#
#   .venvs/ref/bin/python inst/parity/transforms_reference.py \
#       --out inst/parity/transforms/transforms.safetensors
#
# These need no model weights, so they run anywhere the package does.

skip_if_not_installed("jsonlite")
skip_if_not_installed("digest")
skip_if_not_installed("safetensors")

ref_path <- tabfound_file("parity", "transforms", "transforms.safetensors.gz")
skip_if(!nzchar(ref_path), "transform reference not found")

# Arrays live in safetensors because the fixtures deliberately contain
# NaN and Inf, which JSON cannot round-trip; scalars live alongside in
# a .json sibling.
ref <- read_reference_tensors(ref_path)
scl <- jsonlite::fromJSON(sub("\\.safetensors\\.gz$", ".json", ref_path))

as_mat <- function(x) {
  m <- as.matrix(x)
  storage.mode(m) <- "double"
  m
}
X      <- as_mat(ref$X)
X_test <- as_mat(ref$X_test)
keep   <- as.integer(as.array(ref$remove_constant_keep))


test_that("remove_constant_features_fit keeps NaN columns and drops constants", {
  expect_identical(remove_constant_features_fit(X), keep)

  # A column that is constant apart from a NaN is still non-constant,
  # because NaN != NaN -- the behaviour R's `==` would get wrong.
  Z <- cbind(c(1, 1, 1, 1), c(1, 1, NA, 1), c(1, 2, 3, 4))
  expect_identical(remove_constant_features_fit(Z), c(2L, 3L))

  expect_error(remove_constant_features_fit(matrix(1, 4, 2)), "constant")
})


test_that("squashing scaler matches SquashingScaler", {
  fit <- fit_squashing_scaler(X[, keep, drop = FALSE], max_absolute_value = 3.0)
  tr  <- transform_squashing_scaler(X[, keep, drop = FALSE], fit)
  te  <- transform_squashing_scaler(X_test[, keep, drop = FALSE], fit)
  expect_equal(tr, as_mat(ref$squashing_train), tolerance = 1e-12,
               ignore_attr = TRUE)
  expect_equal(te, as_mat(ref$squashing_test), tolerance = 1e-12,
               ignore_attr = TRUE)
})


test_that("quantile transformer matches sklearn's uniform output", {
  Xq  <- as_mat(ref$quantile_X)
  Xqt <- as_mat(ref$quantile_X_test)
  fit <- fit_quantile_transformer(Xq, n_quantiles = scl$quantile_n_quantiles)
  expect_equal(transform_quantile_transformer(Xq, fit),
               as_mat(ref$quantile_train), tolerance = 1e-10,
               ignore_attr = TRUE)
  expect_equal(transform_quantile_transformer(Xqt, fit),
               as_mat(ref$quantile_test), tolerance = 1e-10,
               ignore_attr = TRUE)
})


test_that("a constant column comes out 0, not 0.5, from the quantile transform", {
  # Every quantile of a constant column is the same value, so it is at once
  # the fitted minimum and the fitted maximum. sklearn pins the maximum to
  # 1 and then the minimum to 0, so the lower bound wins; the symmetric
  # fold on its own would say 0.5. TabPFN v2.6 makes this reachable --
  # `quantile_uni` is its only primary transform, so any constant column
  # that survives RemoveConstant lands here.
  Xc  <- as_mat(ref$quantile_const_X)
  fit <- fit_quantile_transformer(Xc, n_quantiles = scl$quantile_const_n_quantiles)
  got <- transform_quantile_transformer(Xc, fit)
  expect_equal(got, as_mat(ref$quantile_const_train), tolerance = 1e-10,
               ignore_attr = TRUE)
  expect_true(all(got[, 2] == 0))
})


test_that("quantile_uni_extrapolate continues past the training range", {
  Xe  <- as_mat(ref$quantile_extrap_X)
  Xet <- as_mat(ref$quantile_extrap_X_test)
  ratio <- scl$quantile_extrap_ratio
  expect_identical(ratio, 1)
  expect_identical(quantile_preset_extrapolate_ratio("quantile_uni_extrapolate"), 1)
  expect_null(quantile_preset_extrapolate_ratio("quantile_uni"))
  # It grids the ECDF exactly like the plain preset; only transform differs.
  expect_identical(quantile_preset_n_quantiles("quantile_uni_extrapolate", 40L),
                   quantile_preset_n_quantiles("quantile_uni", 40L))

  fit <- fit_quantile_transformer(
    Xe, n_quantiles = scl$quantile_extrap_n_quantiles, extrapolate_ratio = ratio)
  expect_equal(transform_quantile_transformer(Xe, fit),
               as_mat(ref$quantile_extrap_train), tolerance = 1e-10,
               ignore_attr = TRUE)
  got <- transform_quantile_transformer(Xet, fit)
  expect_equal(got, as_mat(ref$quantile_extrap_test), tolerance = 1e-10,
               ignore_attr = TRUE)

  # The fixture is built so the two presets genuinely disagree -- without
  # out-of-range test values they are the same function, and matching the
  # reference would prove nothing.
  plain_fit <- fit_quantile_transformer(
    Xe, n_quantiles = scl$quantile_extrap_n_quantiles)
  plain <- transform_quantile_transformer(Xet, plain_fit)
  expect_equal(plain, as_mat(ref$quantile_extrap_plain_test), tolerance = 1e-10,
               ignore_attr = TRUE)
  expect_false(isTRUE(all.equal(got, plain)))

  # What the preset exists for: the clipping form cannot tell "just past
  # the edge" from "far outside", and this one can. Rows 1 and 2 of column
  # 1 are 5 and 0.01 below the training minimum.
  expect_identical(plain[1, 1], plain[2, 1])
  expect_true(got[1, 1] < got[2, 1])
  expect_true(got[2, 1] < 0)
  # Clipped at -ratio, not left unbounded.
  expect_gte(got[1, 1], -ratio)
  # And symmetrically above: rows 3 and 4 of column 2.
  expect_identical(plain[3, 2], plain[4, 2])
  expect_true(got[3, 2] > got[4, 2])
  expect_true(got[4, 2] > 1)
  expect_lte(got[3, 2], 1 + ratio)

  # Values exactly at the boundary keep the ordinary 0 / 1 -- the masks are
  # strict, so the endpoint pinning still owns them.
  expect_identical(got[5, 3], 0)
  expect_identical(got[6, 3], 1)
  # A constant column has no range to extrapolate along; it is left alone
  # even though every test value is out of its range.
  expect_true(all(got[, 4] == plain[, 4]))

  expect_error(fit_quantile_transformer(Xe, 5L, extrapolate_ratio = -1),
               "non-negative")
})


test_that("polynomial features match NanHandlingPolynomialFeaturesStep", {
  Xq  <- as_mat(ref$quantile_X)
  Xqt <- as_mat(ref$quantile_X_test)
  f1 <- as.integer(as.array(ref$poly_factor_1)) + 1L
  f2 <- as.integer(as.array(ref$poly_factor_2)) + 1L

  fit <- fit_polynomial_features(Xq, f1, f2)
  expect_equal(transform_polynomial_features(Xq, fit),
               as_mat(ref$poly_train), tolerance = 1e-10, ignore_attr = TRUE)
  # Test rows reuse the train-fitted scale.
  expect_equal(transform_polynomial_features(Xqt, fit),
               as_mat(ref$poly_test), tolerance = 1e-10, ignore_attr = TRUE)

  # The step *replaces* the base columns with their scaled versions rather
  # than appending to the originals -- easy to get backwards, and silent
  # if you do.
  expect_false(isTRUE(all.equal(transform_polynomial_features(Xq, fit)[, 1],
                                Xq[, 1])))

  # Pair count, and a guard against indices that do not fit the input.
  expect_identical(n_polynomial_features(4L), 10L)
  expect_identical(n_polynomial_features(4L, max_features = 6L), 6L)
  expect_error(fit_polynomial_features(Xq, c(1L, 99L), c(1L, 2L)), "outside")
  expect_error(fit_polynomial_features(Xq, 1L, c(1L, 2L)), "same length")
})


test_that("categorical detection matches detect_feature_modalities", {
  cat_X <- as_mat(ref$cat_X)
  ints <- function(x) as.integer(as.array(x))

  # Nothing declared: only the column with fewer than four distinct values
  # is inferred. A six- or fifteen-level column stays numeric, which is
  # exactly why declaring matters.
  expect_identical(detect_categorical_features(cat_X), ints(ref$cat_detected_auto))
  # Everything declared: the ceiling is 30 levels, so the continuous
  # column is refused however loudly it is declared.
  expect_identical(detect_categorical_features(cat_X, seq_len(ncol(cat_X))),
                   ints(ref$cat_detected_declared))

  expect_identical(
    vapply(seq_len(ncol(cat_X)), function(j) least_common_category_count(cat_X[, j]),
           integer(1)),
    ints(ref$cat_least_common)
  )

  # Too few rows to trust a cardinality count: inference declines.
  expect_identical(detect_categorical_features(cat_X[1:40, , drop = FALSE]), integer())
  # ...but a declaration still lands, because it is not an inference.
  expect_identical(detect_categorical_features(cat_X[1:40, , drop = FALSE], 2L), 2L)

  expect_error(detect_categorical_features(cat_X, 99L), "outside the predictor matrix")
})


test_that("ordinal encoding matches sklearn, NaN and unseen levels included", {
  cat_X <- as_mat(ref$cat_X)
  cat_X_test <- as_mat(ref$cat_X_test)
  cols <- 1:3
  enc <- fit_ordinal_encoder(cat_X, cols)

  # Category counts include the NA slot -- that is the length the shuffle
  # permutations have to match.
  expect_identical(enc$n_categories,
                   as.integer(as.array(ref$cat_ordinal_n_categories)))
  expect_equal(transform_ordinal_encoder(cat_X, enc),
               as_mat(ref$cat_ordinal_train), tolerance = 1e-12,
               ignore_attr = TRUE)

  # An unseen level and an NA both come back NA rather than erroring or
  # colliding with a real code.
  got_test <- transform_ordinal_encoder(cat_X_test, enc)
  expect_equal(got_test, as_mat(ref$cat_ordinal_test), tolerance = 1e-12,
               ignore_attr = TRUE)
  expect_true(is.na(got_test[1, 1]))
  expect_true(is.na(got_test[2, 2]))

  perms <- lapply(0:2, function(k) as.integer(as.array(ref[[paste0("cat_ordinal_perm_", k)]])))
  expect_equal(transform_ordinal_encoder(cat_X, enc, perms),
               as_mat(ref$cat_ordinal_shuffled_train), tolerance = 1e-12,
               ignore_attr = TRUE)
})


test_that("SVD features match TruncatedSVD, including the sign convention", {
  sq  <- fit_squashing_scaler(X[, keep, drop = FALSE], max_absolute_value = 3.0)
  Xsq <- transform_squashing_scaler(X[, keep, drop = FALSE], sq)
  Xte <- transform_squashing_scaler(X_test[, keep, drop = FALSE], sq)

  fit <- fit_transform_svd_features(Xsq, "svd_quarter_components")
  expect_equal(as.numeric(fit$n_components), as.numeric(scl$svd_n_components))

  # Sign matters: ARPACK and LAPACK agree on the subspace but not on the
  # sign of each component, so the port reproduces sklearn's `svd_flip`.
  # Without it these come out negated and the model sees mirrored features.
  expect_equal(transform_svd_features(Xsq, fit), as_mat(ref$svd_train),
               tolerance = 1e-9, ignore_attr = TRUE)
  expect_equal(transform_svd_features(Xte, fit), as_mat(ref$svd_test),
               tolerance = 1e-9, ignore_attr = TRUE)
})


test_that("fingerprint reproduces the reference hash", {
  sq  <- fit_squashing_scaler(X[, keep, drop = FALSE], max_absolute_value = 3.0)
  Xsq <- transform_squashing_scaler(X[, keep, drop = FALSE], sq)
  Xte <- transform_squashing_scaler(X_test[, keep, drop = FALSE], sq)
  salt <- nrow(Xsq) * ncol(Xsq)
  expect_identical(as.numeric(salt), as.numeric(scl$fingerprint_salt))

  # Exact, not approximate: the value is a hash, so a near-miss means a
  # different row was hashed, not a rounding difference.
  expect_equal(apply_fingerprint(Xsq, salt, is_test = FALSE),
               as.numeric(as.array(ref$fingerprint_train)), tolerance = 1e-12)
  expect_equal(apply_fingerprint(Xte, salt, is_test = TRUE),
               as.numeric(as.array(ref$fingerprint_test)), tolerance = 1e-12)
})


test_that("Yeo-Johnson round-trips and matches SafePowerTransformer", {
  y   <- as.numeric(as.array(ref$yeojohnson_y))
  lam <- scl$yeojohnson_lambda
  fwd <- yeojohnson_forward(y, lam)
  expect_equal(fwd, as.numeric(as.array(ref$yeojohnson_forward)), tolerance = 1e-10)
  expect_equal(yeojohnson_inverse(fwd, lam),
               as.numeric(as.array(ref$yeojohnson_inverse)), tolerance = 1e-10)
  expect_equal(yeojohnson_inverse(yeojohnson_forward(y, lam), lam), y,
               tolerance = 1e-9)
})


test_that("class permutation inverts itself", {
  y <- c(0L, 1L, 2L, 2L, 1L, 0L)
  perm <- c(2L, 0L, 1L)                       # class k -> perm[k + 1]
  y_perm <- apply_class_permutation(y, perm)
  expect_identical(y_perm, c(2L, 0L, 1L, 1L, 0L, 2L))

  probs <- matrix(c(0.7, 0.2, 0.1,
                    0.1, 0.8, 0.1), nrow = 2, byrow = TRUE)
  back <- apply_class_permutation_inverse(probs, perm + 1L)
  expect_equal(back[1, ], c(0.1, 0.7, 0.2))
})


test_that("np_around matches NumPy's scale-round-unscale", {
  # R's own round() is better conditioned, which is exactly why it cannot
  # be used where the resulting bytes get hashed.
  x <- c(0.5, 1.5, 2.5, -0.5, 1 / 3, 2 / 3)
  expect_equal(.np_around(x, 0L), c(0, 2, 2, 0, 0, 1))
  expect_equal(.np_around(c(NA, Inf, -Inf, 1.23456789012345), 12L),
               c(NA, Inf, -Inf, 1.234567890123), tolerance = 1e-14)
})
