# The ensemble member pipeline, without model weights.
#
# One composer serves every TabPFN member type, driven by the member's own
# config. What that config says has to translate into the right steps in
# the right order, and the width it produces has to be exactly what the
# shuffle permutation expects -- a mismatch there is silent until the
# network gets a transposed-looking input.

skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

set.seed(4)
X_tr <- matrix(rnorm(60 * 5), ncol = 5)
X_tr[, 4] <- 3.5                       # constant, dropped by RemoveConstant
X_tr[2, 1] <- NA
X_te <- matrix(rnorm(20 * 5), ncol = 5)
X_te[, 4] <- 3.5

# Width before the shuffle, which is what the permutation has to match.
member_width <- function(cfg) {
  cfg$shuffle_perm <- NULL
  ncol(apply_member_pipeline(X_tr, X_te, cfg)$X_train)
}


test_that("append_original = auto follows the reference's width rule", {
  expect_true(resolve_append_original("auto", 4L, 500L))
  expect_true(resolve_append_original("auto", 250L, 500L))
  # More than half the per-estimator budget: appending would blow it.
  expect_false(resolve_append_original("auto", 251L, 500L))
  # Under the budget but past the absolute ceiling.
  expect_false(resolve_append_original("auto", 600L, 2000L))
  # An explicit setting is taken as given, whatever the widths say.
  expect_false(resolve_append_original(FALSE, 4L, 500L))
  expect_true(resolve_append_original(TRUE, 600L, 500L))
})


test_that("each config field lands as the step it names", {
  base <- list(preset = "none", append_original = FALSE,
               max_features_per_estimator = 500L,
               global_transformer_name = NULL, add_fingerprint = TRUE)

  # 5 columns, one constant -> 4 survive, + 1 fingerprint.
  expect_identical(member_width(base), 5L)

  # No fingerprint column when the member switches it off.
  expect_identical(member_width(modifyList(base, list(add_fingerprint = FALSE))), 4L)

  # append_original doubles the transformed block.
  q <- modifyList(base, list(preset = "quantile_uni", append_original = TRUE))
  expect_identical(member_width(q), 4L + 4L + 1L)

  # svd_quarter_components on 4 columns: max(1, min(n/10 + 1, 4/4)) = 1.
  s <- modifyList(base, list(global_transformer_name = "svd_quarter_components"))
  expect_identical(member_width(s), 4L + 1L + 1L)

  # Polynomial features run first, so they widen what everything else sees:
  # 5 raw columns + 6 products = 11, of which the constant one drops out.
  p <- modifyList(base, list(polynomial_features = 6L,
                             poly_factor_1 = c(1L, 1L, 2L, 2L, 3L, 5L),
                             poly_factor_2 = c(1L, 2L, 2L, 3L, 5L, 5L)))
  expect_identical(member_width(p), 10L + 1L)
})


test_that("the composer refuses what it cannot reproduce", {
  base <- list(preset = "none", append_original = FALSE,
               max_features_per_estimator = 500L, shuffle_perm = NULL)

  expect_error(apply_member_pipeline(X_tr, X_te,
                                     modifyList(base, list(preset = "kdi"))),
               "not implemented")
  # Polynomial features without the drawn pairs: the pairs come from the
  # reference's NumPy generator and cannot be guessed.
  expect_error(apply_member_pipeline(X_tr, X_te,
                                     modifyList(base, list(polynomial_features = 4L))),
               "factor indices")
  # Past the per-estimator budget the reference subsamples features.
  expect_error(apply_member_pipeline(X_tr, X_te,
                                     modifyList(base,
                                       list(max_features_per_estimator = 2L))),
               "exceeds this member's budget")
})


test_that("the shuffle is the last step and permutes the whole width", {
  cfg <- list(preset = "quantile_uni", append_original = FALSE,
              max_features_per_estimator = 500L,
              global_transformer_name = "svd_quarter_components",
              add_fingerprint = TRUE)
  w <- member_width(cfg)
  cfg$shuffle_perm <- rev(seq_len(w))
  out <- apply_member_pipeline(X_tr, X_te, cfg)
  expect_identical(ncol(out$X_train), w)
  expect_identical(ncol(out$X_test), w)
  expect_identical(nrow(out$X_train), nrow(X_tr))

  cfg$shuffle_perm <- NULL
  unshuffled <- apply_member_pipeline(X_tr, X_te, cfg)
  expect_equal(out$X_train, unshuffled$X_train[, rev(seq_len(w))],
               ignore_attr = TRUE)
})


test_that("categorical columns move to the front when the transform skips them", {
  # 5 columns: 1 and 3 categorical, one constant (4) that RemoveConstant
  # drops before any of this. `numeric` means "these are just numbers", so
  # everything goes through the primary transform in place; anything else
  # passes the categorical columns through in front of the transformed
  # block, which reorders the matrix.
  set.seed(8)
  n <- 150
  Xc <- cbind(sample.int(3L, n, TRUE), rnorm(n),
              sample.int(6L, n, TRUE), 2.5, rnorm(n))
  Xc_te <- cbind(sample.int(3L, 20L, TRUE), rnorm(20),
                 sample.int(6L, 20L, TRUE), 2.5, rnorm(20))
  cats <- c(1L, 3L)

  base <- list(preset = "quantile_uni", append_original = FALSE,
               max_features_per_estimator = 500L,
               global_transformer_name = NULL, add_fingerprint = FALSE,
               shuffle_perm = NULL)

  # `numeric`: every surviving column transformed, order preserved.
  num <- apply_member_pipeline(Xc, Xc_te, base, cats)
  expect_identical(ncol(num$X_train), 4L)
  # A quantile-transformed column lands in [0, 1]; a passed-through
  # categorical column keeps its raw codes.
  expect_true(all(num$X_train >= 0 & num$X_train <= 1))

  # `ordinal`: categoricals bypass the primary transform, get encoded, and
  # end up first -- so column 1 holds codes, not quantiles.
  ord <- apply_member_pipeline(
    Xc, Xc_te, modifyList(base, list(categorical_name = "ordinal")), cats)
  expect_identical(ncol(ord$X_train), 4L)
  expect_setequal(unique(ord$X_train[, 1]), c(0, 1, 2))
  expect_setequal(unique(ord$X_train[, 2]), 0:5)
  expect_true(all(ord$X_train[, 3:4] >= 0 & ord$X_train[, 3:4] <= 1))

  # `append_original` keeps the originals in place and appends the
  # transformed numerics after them.
  app <- apply_member_pipeline(
    Xc, Xc_te,
    modifyList(base, list(categorical_name = "none", append_original = TRUE)),
    cats)
  expect_identical(ncol(app$X_train), 4L + 2L)
  expect_equal(app$X_train[, 1], Xc[, 1], ignore_attr = TRUE)
})


test_that("the encoders' selection filters match their names", {
  set.seed(12)
  n <- 300
  # The two filters can only be told apart by a perfectly balanced column:
  # "every level seen at least ten times" and "fewer than n/10 levels" are
  # in tension, and 30 levels of exactly 10 rows each sits on the boundary
  # of the second while clearing the first.
  #
  # Column 1: 3 common levels, kept by both. Column 2: one level appears
  # twice, dropped by both. Column 3: the balanced 30-level one, kept by
  # `_common_categories` and dropped by `_very_common_categories`.
  X <- cbind(sample.int(3L, n, TRUE),
             c(1, 1, sample(2:4, n - 2L, TRUE)),
             rep(1:30, each = 10L))
  cats <- 1:3

  expect_identical(select_encoded_categoricals(X, cats, "ordinal"), 1:3)
  expect_identical(select_encoded_categoricals(X, cats, "ordinal_shuffled"), 1:3)
  expect_identical(
    select_encoded_categoricals(X, cats, "ordinal_common_categories_shuffled"),
    c(1L, 3L))
  # Both conditions are needed: the balanced column clears the count but
  # not the level ceiling.
  expect_identical(least_common_category_count(X[, 3]), 10L)
  expect_identical(
    select_encoded_categoricals(X, cats, "ordinal_very_common_categories_shuffled"),
    1L)
  expect_identical(select_encoded_categoricals(X, integer(), "ordinal"), integer())
})


test_that("shuffled category codes need mappings that fit", {
  set.seed(3)
  n <- 120
  X <- cbind(sample.int(4L, n, TRUE), rnorm(n))
  cfg <- list(preset = "none", append_original = FALSE,
              max_features_per_estimator = 500L, shuffle_perm = NULL,
              categorical_name = "ordinal_shuffled")

  expect_error(apply_member_pipeline(X, X, cfg, 1L), "no mappings")
  expect_error(
    apply_member_pipeline(X, X, modifyList(cfg, list(cat_mappings = list(0:3, 0:3))), 1L),
    "2 code mappings for 1 encoded column")
  expect_error(
    apply_member_pipeline(X, X, modifyList(cfg, list(cat_mappings = list(0:1))), 1L),
    "4 categories but its mapping is 2 long")

  # Drawing its own is the generator's path, not the predictor's.
  drawn <- apply_member_pipeline(X, X, cfg, 1L, draw_missing = TRUE)
  expect_length(drawn$cat_mappings, 1L)
  expect_setequal(drawn$cat_mappings[[1]], 0:3)

  expect_error(
    apply_member_pipeline(X, X, modifyList(cfg, list(categorical_name = "onehot")), 1L),
    "not implemented")
})


test_that("the native generator writes configs its own pipeline can read", {
  for (variant in c("v2.5", "v2.6")) {
    for (head in c("classifier", "regressor")) {
      d <- withr::local_tempdir()
      y <- if (head == "classifier") rep(0:2, length.out = nrow(X_tr))
           else rnorm(nrow(X_tr))
      generate_ensemble_configs_native(X_tr, y, n_estimators = 4L, head = head,
                                       variant = variant, output_dir = d)
      cfgs <- load_ensemble_configs_from_dump(d)
      expect_length(cfgs, 4L)

      for (cfg in cfgs) {
        mem <- apply_member_pipeline(X_tr, X_te, cfg)
        # The permutation was drawn for this member's own width, so it has
        # to still fit after a round trip through JSON and safetensors.
        expect_identical(ncol(mem$X_train), length(cfg$shuffle_perm),
                         info = paste(variant, head))
        expect_identical(ncol(mem$X_test), length(cfg$shuffle_perm))
        if (head == "classifier") expect_length(cfg$class_perm, 3L)
      }
      # Only v2.6's regressor uses polynomial features; only v2.5's
      # carries a target transform.
      has_poly <- vapply(cfgs, function(c) !is.null(c$poly_factor_1), logical(1))
      has_tt <- vapply(cfgs, function(c) !is.null(c$target_transform_lambda),
                       logical(1))
      expect_identical(all(has_poly), variant == "v2.6" && head == "regressor")
      expect_identical(any(has_tt), variant == "v2.5" && head == "regressor")
    }
  }
})


test_that("the forward helper only names arguments the network has", {
  # The predictors serve both TabPFN generations from one code path, and
  # v2.5's `forward` has no `kv_cache` or `save_peak_memory_factor`.
  # Naming them anyway is an R error, not an ignored argument -- so the
  # fakes here are real `nn_module`s with the real signatures.
  v25 <- torch::nn_module(
    "FakeV25",
    initialize = function() {},
    forward = function(x_train, y_train, x_test, column_embeddings = NULL) "plain"
  )()
  # v2.5 declares the cache but not the chunked forward.
  v25c <- torch::nn_module(
    "FakeV25Cache",
    initialize = function() { self$supports_kv_cache <- TRUE },
    forward = function(x_train, y_train, x_test, column_embeddings = NULL,
                       kv_cache = NULL, return_kv_cache = FALSE) {
      paste0("kv=", !is.null(kv_cache))
    }
  )()
  v26 <- torch::nn_module(
    "FakeV26",
    initialize = function() {
      self$supports_kv_cache <- TRUE
      self$supports_chunked_eval <- TRUE
    },
    forward = function(x_train, y_train, x_test, column_embeddings = NULL,
                       kv_cache = NULL, return_kv_cache = FALSE,
                       save_peak_memory_factor = NULL) {
      paste0("kv=", !is.null(kv_cache), ",spmf=", !is.null(save_peak_memory_factor))
    }
  )()

  # v2 gets the four positional arguments and nothing else, even though
  # the caller named two more.
  expect_identical(
    tabpfn_forward(v25, 1, 2, 3, NULL, kv_cache = "x",
                   save_peak_memory_factor = 8L),
    "plain")
  # v2.5 gets the cache but not the chunk factor -- naming that one would
  # be an error, not an ignored argument.
  expect_identical(
    tabpfn_forward(v25c, 1, 2, 3, NULL, kv_cache = "x",
                   save_peak_memory_factor = 8L),
    "kv=TRUE")
  # v2.6 gets both.
  expect_identical(
    tabpfn_forward(v26, 1, 2, 3, NULL, kv_cache = "x",
                   save_peak_memory_factor = 8L),
    "kv=TRUE,spmf=TRUE")
  expect_identical(tabpfn_forward(v26, 1, 2, 3, NULL), "kv=FALSE,spmf=FALSE")
})


test_that("sampling uses the ensemble rather than one unensembled pass", {
  d <- Sys.getenv("TABFOUND_TABPFN_REG_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN_REG_DIR not configured")

  set.seed(6)
  n_tr <- 200L; n_te <- 60L; p <- 4L
  X <- matrix(rnorm((n_tr + n_te) * p), ncol = p)
  y <- as.numeric(X %*% rnorm(p)) + rnorm(n_tr + n_te, sd = 0.4)
  tr <- seq_len(n_tr); te <- n_tr + seq_len(n_te)

  cfg <- withr::local_tempdir()
  generate_ensemble_configs_native(X[tr, ], y[tr], n_estimators = 4L,
                                   head = "regressor", variant = "v2.5",
                                   random_state = 0L, output_dir = cfg)
  reg <- fit(tabular_regressor(d, ensemble_configs_dir = cfg), X[tr, ], y[tr])

  # It used to warn and quietly drop the ensemble.
  expect_no_warning(
    s <- predict(reg, X[te, ], type = "sample", n_samples = 500L, seed = 1L)
  )
  expect_identical(dim(s), c(length(te), 500L))

  # The draws and the analytic summaries now describe the same
  # distribution, which is the point: a sample that disagreed with the
  # quantiles printed beside it would be worse than no sample.
  mu <- predict(reg, X[te, ], type = "mean")
  expect_gt(cor(rowMeans(s), mu), 0.99)
  q <- predict(reg, X[te, ], type = "quantiles", quantiles = c(0.1, 0.9))
  emp <- t(apply(s, 1L, stats::quantile, probs = c(0.1, 0.9)))
  expect_lt(mean(abs(emp[, 1] - q[, 1])), 0.1)
  expect_lt(mean(abs(emp[, 2] - q[, 2])), 0.1)

  # A seed still reproduces the draw.
  expect_identical(predict(reg, X[te, ], type = "sample", n_samples = 5L, seed = 3L),
                   predict(reg, X[te, ], type = "sample", n_samples = 5L, seed = 3L))
})
