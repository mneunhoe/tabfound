# The ensemble generators, member for member, against the reference
# wrappers that produce them.
#
# The comparison is deliberately of *both* the configuration and the
# data. A matching feature order with a mismatched member matrix means
# the preprocessing drifted; a matching matrix with a mismatched order
# means the shuffler did. Only dumping both tells those apart.

test_that("TabICL's Shuffler matches, in all four methods", {
  r <- ensemble_ref()
  for (key in names(r$s$shuffler)) {
    parts <- strsplit(key, "_", fixed = TRUE)[[1]]
    got <- py_shuffler(as.integer(parts[1]), parts[2], 8L,
                       random_state = as.integer(parts[3]))
    want <- lapply(r$s$shuffler[[key]], ref_int)
    expect_identical(lapply(got, as.integer), want, info = key)
  }
})


test_that("the Shuffler's structural promises hold", {
  # `shift` is the n circular rotations, in order, starting at identity.
  s <- py_shuffler(4L, "shift", 8L, random_state = 1L)
  expect_length(s, 4L)
  expect_identical(s[[1L]], 0:3)
  expect_identical(s[[2L]], c(3L, 0L, 1L, 2L))

  # `latin` returns n permutations in which every element visits every
  # position exactly once across the set -- that is what makes it a
  # Latin square rather than n arbitrary permutations.
  lat <- py_shuffler(6L, "latin", 6L, random_state = 3L)
  expect_length(lat, 6L)
  m <- do.call(rbind, lat)
  for (j in seq_len(ncol(m))) expect_identical(sort(m[, j]), 0:5)
  for (i in seq_len(nrow(m))) expect_identical(sort(m[i, ]), 0:5)

  # `none`, and a single estimator, both mean "leave it alone".
  expect_identical(py_shuffler(5L, "none", 8L, random_state = 1L), list(0:4))
  expect_identical(py_shuffler(5L, "latin", 1L, random_state = 1L), list(0:4))

  # Above the Latin cap it silently becomes `random`, as the reference does.
  big <- py_shuffler(9L, "latin", 3L, max_elements_for_latin = 5L,
                     random_state = 1L)
  expect_identical(big, py_shuffler(9L, "random", 3L, random_state = 1L))

  expect_error(py_shuffler(4L, "spiral", 3L), "Unknown shuffle method")
})


# Line the two sides up by normalisation method rather than by position.
# The reference groups members by iterating a `set` of method names,
# whose order is a Python implementation detail (it moves with the hash
# seed). Members are averaged, so the order cannot change a prediction --
# but the comparison has to know that.
align_by_norm <- function(got, want) {
  wn <- vapply(want, function(m) m$norm, "")
  gn <- vapply(got, function(m) m$norm, "")
  used <- integer(0)
  for (i in seq_along(got)) {
    cand <- setdiff(which(wn == gn[i]), used)
    expect_gt(length(cand), 0)
    used <- c(used, cand[1L])
  }
  want[used]
}

expect_members_match <- function(tag, got, ref, extra = function(w, g) invisible()) {
  meta <- ref$s$ensembles[[tag]]
  expect_length(got, length(meta$members))
  want <- align_by_norm(got, meta$members)
  # Position within the aligned list is also the position in the dumped
  # tensor names, so recover the original index to look them up.
  idx <- vapply(want, function(w) {
    which(vapply(meta$members, identical, logical(1), w))[1L]
  }, integer(1)) - 1L
  for (i in seq_along(got)) {
    g <- got[[i]]; w <- want[[i]]
    expect_identical(as.integer(g$feat), ref_int(w$feat),
                     info = paste(tag, "member", i, "feature order"))
    expect_equal(g$X, ref_mat(ref$t[[sprintf("%s_%02d_X", tag, idx[i])]]),
                 tolerance = 1e-6, ignore_attr = TRUE,
                 info = paste(tag, "member", i, "X"))
    expect_equal(as.numeric(g$y),
                 ref_vec(ref$t[[sprintf("%s_%02d_y", tag, idx[i])]]),
                 tolerance = 1e-9, info = paste(tag, "member", i, "y"))
    extra(w, g)
  }
}


test_that("TabICL's ensemble reproduces every member", {
  r <- ensemble_ref()
  X <- ref_mat(r$t$X); X_test <- ref_mat(r$t$X_test)
  y_cls <- as.integer(ref_num(r$s$y_cls))
  y_reg <- ref_num(r$s$y_reg)

  specs <- list(
    list(tag = "icl_clf", y = y_cls, classification = TRUE, n = 8L,
         norms = c("none", "power"), feat = "latin", cls = "shift", seed = 42L),
    list(tag = "icl_clf_rand", y = y_cls, classification = TRUE, n = 6L,
         norms = c("none", "quantile"), feat = "random", cls = "random",
         seed = 7L),
    list(tag = "icl_reg", y = y_reg, classification = FALSE, n = 8L,
         norms = c("none", "power"), feat = "latin", cls = "shift", seed = 42L)
  )

  for (sp in specs) {
    gen <- tabicl_ensemble_fit(
      X, sp$y, classification = sp$classification, n_estimators = sp$n,
      norm_methods = sp$norms, feat_shuffle_method = sp$feat,
      class_shuffle_method = sp$cls, random_state = sp$seed,
      quantile_subsample = NULL
    )
    expect_identical(gen$filter$keep, ref_lgl(r$s$ensembles[[sp$tag]]$keep),
                     info = sp$tag)
    members <- tabicl_ensemble_transform(gen, X_test)
    expect_members_match(sp$tag, members, r, function(w, g) {
      if (!is.null(w$class_shuffle)) {
        expect_identical(as.integer(g$class_shuffle), ref_int(w$class_shuffle))
      }
    })
  }
})


test_that("TabFM's ensemble reproduces every member, including cat masks", {
  r <- ensemble_ref()
  X <- ref_mat(r$t$X); X_test <- ref_mat(r$t$X_test)
  y_cls <- as.integer(ref_num(r$s$y_cls))
  y_reg <- ref_num(r$s$y_reg)

  specs <- list(
    list(tag = "fm_clf", y = y_cls, task = "classification", n = 8L,
         norms = c("none", "power"), shift = TRUE, cat = c(1L, 4L), seed = 42L),
    list(tag = "fm_clf_noshift", y = y_cls, task = "classification", n = 5L,
         norms = "none", shift = FALSE, cat = NULL, seed = 13L),
    list(tag = "fm_reg", y = y_reg, task = "regression", n = 8L,
         norms = c("none", "power"), shift = FALSE, cat = NULL, seed = 42L)
  )

  for (sp in specs) {
    gen <- tabfm_ensemble_fit(
      X, sp$y, task = sp$task, n_estimators = sp$n, norm_methods = sp$norms,
      class_shift = sp$shift, cat_features = sp$cat, random_state = sp$seed,
      quantile_subsample = NULL
    )
    expect_identical(gen$filter$keep, ref_lgl(r$s$ensembles[[sp$tag]]$keep),
                     info = sp$tag)
    members <- tabfm_ensemble_transform(gen, X_test)
    expect_members_match(sp$tag, members, r, function(w, g) {
      expect_identical(as.integer(g$shift), as.integer(w$shift))
      expect_identical(as.logical(g$cat_mask), ref_lgl(w$cat_mask))
    })
  }
})


test_that("the two generators are genuinely different constructions", {
  # Same seed, same data, same nominal settings -- and different
  # ensembles, because TabICL draws Latin-square permutations and
  # permutes class labels while TabFM samples permutations and rotates
  # them. Collapsing the two into one shared implementation would be the
  # easy mistake here, so it is pinned as a fact.
  X <- matrix(as.numeric(1:120), ncol = 6L)
  X[, 2] <- X[, 2] * 3 + 1
  y <- as.integer(rep(0:2, length.out = 20))
  icl <- tabicl_ensemble_fit(X, y, TRUE, n_estimators = 6L, random_state = 42L,
                             quantile_subsample = NULL)
  fm <- tabfm_ensemble_fit(X, y, "classification", n_estimators = 6L,
                           random_state = 42L, quantile_subsample = NULL)
  icl_feats <- lapply(tabicl_ensemble_transform(icl, X[1:3, ]), function(m) m$feat)
  fm_feats <- lapply(tabfm_ensemble_transform(fm, X[1:3, ]), function(m) m$feat)
  expect_false(identical(icl_feats, fm_feats))
})


test_that("the ensemble refuses degenerate input rather than guessing", {
  X <- matrix(rep(1, 20), ncol = 2L)
  y <- as.integer(rep(0:1, 5))
  expect_error(tabicl_ensemble_fit(X, y, TRUE, quantile_subsample = NULL),
               "constant")
  expect_error(tabfm_ensemble_fit(X, y, "classification",
                                  quantile_subsample = NULL), "constant")
})


test_that("Mitra's preprocessor matches AutoGluon's", {
  r <- ensemble_ref()
  skip_if(is.null(r$s$mitra), "Mitra reference not generated (needs --mitra-src)")
  X <- ref_mat(r$t$X_na); X_test <- ref_mat(r$t$X_test_na)

  # `random_mirror_x` is off on both sides: the reference draws its sign
  # flips from NumPy's *global* generator and never seeds it, so its
  # mirrors differ between two of its own runs. Everything either side of
  # them is deterministic and is what gets compared.
  clf <- mitra_preprocessor_fit(X, as.integer(ref_num(r$s$y_cls)),
                                task = "classification",
                                random_mirror_x = FALSE)
  m <- r$s$mitra$mitra_clf
  expect_identical(!clf$keep, ref_lgl(m$singular))
  expect_equal(clf$pre_nan_mean, ref_num(m$pre_nan_mean), tolerance = 1e-13)
  # The reference casts to float32 on the way out; this port narrows one
  # step later, in `as_float_tensor()`, so compare after that rounding.
  as_f32 <- function(x) {
    matrix(as.numeric(torch::torch_tensor(as.numeric(x),
                                          dtype = torch::torch_float())$to(
             dtype = torch::torch_double())), nrow = nrow(x))
  }
  expect_equal(as_f32(mitra_preprocessor_transform_X(X, clf)),
               ref_mat(r$t$mitra_clf_train), tolerance = 0, ignore_attr = TRUE)
  expect_equal(as_f32(mitra_preprocessor_transform_X(X_test, clf)),
               ref_mat(r$t$mitra_clf_test), tolerance = 0, ignore_attr = TRUE)

  y_reg <- ref_num(r$s$y_reg)
  reg <- mitra_preprocessor_fit(X, y_reg, task = "regression",
                                random_mirror_x = FALSE,
                                random_mirror_regression = FALSE)
  mr <- r$s$mitra$mitra_reg
  expect_equal(reg$scaler$min, mr$y_min, tolerance = 1e-13)
  expect_equal(reg$scaler$min + reg$scaler$range, mr$y_max, tolerance = 1e-13)
  expect_equal(mitra_preprocessor_invert_y(ref_num(mr$preds), reg),
               ref_num(mr$inverse), tolerance = 1e-12)
})


test_that("Mitra's preprocessor imputes rather than letting a column vanish", {
  # Left alone, one missing value makes a column's quantiles all-NaN
  # inside the network, every value buckets to zero, and the feature
  # disappears without an error. The imputation is what stops that.
  X <- cbind(c(1, 2, NA, 4, 5), c(10, 20, 30, 40, 50))
  fit <- mitra_preprocessor_fit(X, c(0L, 1L, 0L, 1L, 0L),
                                random_mirror_x = FALSE)
  out <- mitra_preprocessor_transform_X(X, fit)
  expect_true(all(is.finite(out)))
  expect_equal(out[3, 1], 3)              # mean of 1, 2, 4, 5

  # Constant columns are dropped, and a target with no spread is refused.
  Xc <- cbind(c(1, 2, 3, 4), rep(7, 4))
  fit2 <- mitra_preprocessor_fit(Xc, c(0L, 1L, 0L, 1L), random_mirror_x = FALSE)
  expect_identical(fit2$keep, c(TRUE, FALSE))
  expect_error(
    mitra_preprocessor_fit(Xc, rep(2, 4), task = "regression",
                           random_mirror_x = FALSE),
    "constant"
  )
})


test_that("Mitra's sign mirrors are reproducible under a seed", {
  X <- matrix(as.numeric(1:40), ncol = 4L)
  y <- as.integer(rep(0:1, 5))
  a <- mitra_preprocessor_fit(X, y, seed = 5L)
  b <- mitra_preprocessor_fit(X, y, seed = 5L)
  d <- mitra_preprocessor_fit(X, y, seed = 6L)
  expect_identical(a$mirror, b$mirror)
  expect_true(all(a$mirror %in% c(1, -1)))
  # Different seeds give different members -- which is the only thing
  # making a Mitra ensemble of more than one member worth anything.
  expect_false(identical(a$mirror, d$mirror))
})


test_that("the quantile distribution matches TabICL's", {
  r <- ensemble_ref()
  for (tag in names(r$s$quantile_dist)) {
    meta <- r$s$quantile_dist[[tag]]
    grid <- ref_mat(r$t[[paste0("qd_", tag, "_grid")]])
    n_q <- as.integer(meta$n_q)
    alpha <- seq_len(n_q) / (n_q + 1)
    d <- quantile_dist(grid, alpha)

    expect_equal(d$quantiles, ref_mat(r$t[[paste0("qd_", tag, "_sorted")]]),
                 tolerance = 0, ignore_attr = TRUE, info = tag)
    expect_equal(d$tail_a_l, ref_num(meta$beta_l), tolerance = 1e-13, info = tag)
    expect_equal(-d$tail_a_r, ref_num(meta$beta_r), tolerance = 1e-13, info = tag)
    expect_equal(quantile_dist_stat(grid, alpha, "mean"),
                 ref_vec(r$t[[paste0("qd_", tag, "_mean")]]),
                 tolerance = 1e-13, info = tag)
    expect_equal(quantile_dist_stat(grid, alpha, "median"),
                 ref_num(meta$median), tolerance = 1e-13, info = tag)
    # The requested levels straddle the grid: 0.001 and 0.9995 fall
    # outside the 40-level grid's range, so the exponential tails are
    # exercised rather than just the spline.
    expect_equal(quantile_dist_icdf(d, ref_num(meta$alphas)),
                 ref_mat(r$t[[paste0("qd_", tag, "_icdf")]]),
                 tolerance = 1e-13, ignore_attr = TRUE, info = tag)
  }
})


test_that("the quantile distribution's mean is not its median", {
  # Reading `type = "mean"` off the middle grid level is the natural
  # mistake, and it is wrong on any skewed predictive distribution.
  grid <- matrix(c(0, 0.1, 0.2, 0.3, 10), nrow = 1L)
  alpha <- seq_len(5L) / 6
  expect_equal(quantile_dist_stat(grid, alpha, "mean"), 2.12)
  expect_equal(quantile_dist_stat(grid, alpha, "median"), 0.2)
})


test_that("SimpleImputer fills with training means and drops empty columns", {
  X <- cbind(c(1, NA, 3), c(NA, NA, NA), c(5, 6, 7))
  fit <- fit_simple_imputer(X)
  expect_identical(fit$keep, c(TRUE, FALSE, TRUE))
  out <- transform_simple_imputer(X, fit)
  expect_identical(dim(out), c(3L, 2L))
  expect_equal(out[2, 1], 2)                       # mean of 1 and 3
  # Test rows are filled with the *training* mean, not their own.
  te <- transform_simple_imputer(cbind(NA_real_, NA_real_, 9), fit)
  expect_equal(te[1, 1], 2)
})


test_that("infinities are rejected rather than imputed", {
  # `Inf` is not a missing value and sklearn does not treat it as one:
  # the reference raises `ValueError: Input X contains infinity` before
  # any of this runs. Imputing it would be worse than erroring -- it is
  # not recoverable, and left alone it poisons the column mean and every
  # scaler downstream, silently.
  X <- matrix(as.numeric(1:30), ncol = 3)
  X[2, 2] <- Inf
  expect_error(fit_simple_imputer(X), "infinite values.*column 2")

  # ...at predict time too, where sklearn re-validates.
  fit <- fit_simple_imputer(matrix(as.numeric(1:30), ncol = 3))
  Xte <- matrix(as.numeric(1:6), ncol = 3)
  Xte[1, 1] <- -Inf
  expect_error(transform_simple_imputer(Xte, fit), "infinite values")

  # The index is not mistaken for a count when only one column is bad.
  Y <- matrix(as.numeric(1:30), ncol = 3); Y[1, 3] <- Inf
  expect_error(fit_simple_imputer(Y), "column 3")
  Y[1, 1] <- Inf
  expect_error(fit_simple_imputer(Y), "columns 1 and 3")

  # Mitra is the exception: its preprocessor zeroes non-finite values
  # itself, as AutoGluon's does, so it never reaches this check.
  Z <- cbind(c(1, 2, 3, 4), c(1, Inf, 3, 4))
  pp <- mitra_preprocessor_fit(Z, c(0L, 1L, 0L, 1L), random_mirror_x = FALSE)
  out <- mitra_preprocessor_transform_X(Z, pp)
  expect_true(all(is.finite(out)))
  expect_equal(out[2, 2], 0)
})
