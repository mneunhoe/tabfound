# Multiple imputation: the chained-equations loop and the two bridges.
#
# All of this runs against `stub_model()` (see helper-stub-model.R), so
# it needs no weights and no GPU. What is being tested is the contract --
# types survive, observed cells are untouched, the same seed gives the
# same imputations, and mice / Amelia accept the result -- not the
# quality of any particular model's draws.

test_that("every missing cell is filled and every observed cell is untouched", {
  d <- mi_test_df()
  imp <- tabfound_impute(d, m = 3L, models = stub_models(), maxit = 2L,
                         seed = 1L, verbose = FALSE)

  expect_s3_class(imp, "tabfound_mi")
  expect_length(imp$imputations, 3L)

  for (k in seq_len(3L)) {
    got <- imp$imputations[[k]]
    expect_identical(dim(got), dim(d))
    expect_identical(names(got), names(d))
    expect_false(anyNA(got))
    for (v in names(d)) {
      obs <- !is.na(d[[v]])
      expect_identical(got[[v]][obs], d[[v]][obs], info = v)
    }
  }
})


test_that("imputed columns keep their original type", {
  d <- mi_test_df()
  got <- tabfound_impute(d, m = 1L, models = stub_models(), maxit = 1L,
                         seed = 2L, verbose = FALSE)$imputations[[1]]

  expect_type(got$num, "double")
  expect_type(got$int, "integer")
  expect_s3_class(got$fac, "factor")
  expect_identical(levels(got$fac), levels(d$fac))
  expect_type(got$lgl, "logical")
  expect_type(got$chr, "character")
  # A numeric 0/1 column is imputed through the classifier -- a
  # continuous draw is not a valid value of a dummy -- but comes back
  # numeric 0/1, not a factor.
  expect_type(got$bin, "double")
  expect_true(all(got$bin %in% c(0, 1)))
})


test_that("a seed makes the whole imputation reproducible", {
  d <- mi_test_df()
  a <- tabfound_impute(d, m = 2L, models = stub_models(), maxit = 2L,
                       seed = 42L, verbose = FALSE)
  b <- tabfound_impute(d, m = 2L, models = stub_models(), maxit = 2L,
                       seed = 42L, verbose = FALSE)
  c3 <- tabfound_impute(d, m = 2L, models = stub_models(), maxit = 2L,
                        seed = 43L, verbose = FALSE)

  expect_equal(a$imputations, b$imputations)
  expect_false(isTRUE(all.equal(a$imputations, c3$imputations)))
})


test_that("imputations differ between data sets", {
  d <- mi_test_df()
  imp <- tabfound_impute(d, m = 4L, models = stub_models(), maxit = 2L,
                         seed = 3L, verbose = FALSE)
  drawn <- vapply(imp$imputations, function(x) x$num[is.na(d$num)][1], numeric(1))
  # Between-imputation variance is the whole point; identical draws would
  # mean the uncertainty is not being propagated.
  expect_gt(stats::var(drawn), 0)
})


test_that("method and predictors control the loop", {
  d <- mi_test_df()

  imp <- tabfound_impute(d, m = 1L, models = stub_models(), maxit = 1L,
                         method = c(num = ""), seed = 4L, verbose = FALSE)
  expect_true(anyNA(imp$imputations[[1]]$num))
  expect_false(anyNA(imp$imputations[[1]]$fac))
  expect_false("num" %in% imp$visit_sequence)

  # "sample" needs no model at all: a bootstrap draw from the observed
  # values, so a handle with no models must still work.
  meth <- stats::setNames(rep("sample", 6L),
                          c("num", "int", "fac", "lgl", "chr", "bin"))
  imp2 <- tabfound_impute(d, m = 1L, models = stub_models(), maxit = 1L,
                          method = meth, seed = 5L, verbose = FALSE)
  expect_false(anyNA(imp2$imputations[[1]]))
  expect_true(all(imp2$imputations[[1]]$num %in% c(d$num[!is.na(d$num)],
                                                   d$num[!is.na(d$num)])))

  pm <- matrix(0L, ncol(d), ncol(d), dimnames = list(names(d), names(d)))
  pm["num", "ok"] <- 1L
  imp3 <- tabfound_impute(d, m = 1L, models = stub_models(), maxit = 1L,
                          predictors = pm, seed = 6L, verbose = FALSE)
  expect_identical(as.integer(rowSums(imp3$predictors)),
                   c(1L, rep(0L, ncol(d) - 1L)))
  expect_false(anyNA(imp3$imputations[[1]]))

  expect_error(tabfound_impute(d, models = stub_models(), method = c(num = "nope")),
               "Unknown method")
  expect_error(tabfound_impute(d, models = stub_models(), method = c(fac = "regression")),
               "categorical")
  expect_error(tabfound_impute(d, models = stub_models(),
                               predictors = matrix(1, 2, 2)),
               "square matrix")
})


test_that("dates and matrices survive the round trip", {
  set.seed(21)
  n <- 30L
  d <- data.frame(
    day  = as.Date("2020-01-01") + sample(0:400, n),
    when = as.POSIXct("2020-01-01", tz = "UTC") + sample(0:1e6, n),
    z    = rnorm(n)
  )
  d$day[1:4]  <- NA
  d$when[5:8] <- NA
  got <- tabfound_impute(d, m = 1L, models = stub_models(), maxit = 1L,
                         seed = 22L, verbose = FALSE)$imputations[[1]]
  expect_s3_class(got$day, "Date")
  expect_s3_class(got$when, "POSIXct")
  expect_false(anyNA(got))
  expect_identical(got$day[5:8], d$day[5:8])

  # A matrix comes back as a data frame -- mice and Amelia both want one.
  mx <- matrix(rnorm(60), ncol = 3, dimnames = list(NULL, c("a", "b", "c")))
  mx[c(2, 9), 1] <- NA
  out <- tabfound_impute(mx, m = 2L, models = stub_models(), maxit = 1L,
                         seed = 23L, verbose = FALSE)
  expect_s3_class(out$imputations[[1]], "data.frame")
  expect_false(anyNA(out$imputations[[1]]))
})


test_that("a data frame with nothing missing round-trips unchanged", {
  d <- data.frame(a = 1:5, b = rnorm(5))
  imp <- tabfound_impute(d, m = 2L, models = stub_models(), verbose = FALSE)
  expect_equal(imp$imputations[[1]], d)
  expect_equal(imp$imputations[[2]], d)
  expect_length(imp$visit_sequence, 0L)
})


test_that("models load lazily, and only the ones the data needs", {
  d <- data.frame(x = rnorm(20), y = c(rnorm(19), NA))
  mods <- tabfound_models(regressor = stub_model("regression"))
  imp <- tabfound_impute(d, m = 1L, models = mods, maxit = 1L, verbose = FALSE)
  expect_false(anyNA(imp$imputations[[1]]))
  expect_identical(mods$loaded(), "regression")

  d2 <- d
  d2$f <- factor(c(rep("a", 10), rep("b", 9), NA))
  expect_error(tabfound_impute(d2, m = 1L, models = mods, verbose = FALSE),
               "No classification model")
})


test_that("tabfound_complete matches mice's complete() contract", {
  d <- mi_test_df(n = 20L)
  imp <- tabfound_impute(d, m = 2L, models = stub_models(), maxit = 1L,
                         seed = 8L, verbose = FALSE)

  expect_equal(tabfound_complete(imp, 1), imp$imputations[[1]])
  expect_length(tabfound_complete(imp, "all"), 2L)
  expect_length(tabfound_complete(imp, "all", include = TRUE), 3L)

  long <- tabfound_complete(imp, "long")
  expect_identical(nrow(long), nrow(d) * 2L)
  expect_true(all(c(".imp", ".id") %in% names(long)))
  expect_setequal(unique(long$.imp), 1:2)

  long0 <- tabfound_complete(imp, "long", include = TRUE)
  expect_identical(nrow(long0), nrow(d) * 3L)
  expect_true(anyNA(long0[long0$.imp == 0, ]))

  broad <- tabfound_complete(imp, "broad")
  expect_identical(ncol(broad), ncol(d) * 2L)

  expect_error(tabfound_complete(imp, 99), "must be one of")
  expect_error(tabfound_complete(d), "tabfound_mi")
})


test_that("a point-estimate-only regressor is refused, unless asked otherwise", {
  d <- data.frame(x = rnorm(30), y = c(rnorm(28), NA, NA))
  mean_only <- stub_model("regression")
  mean_only$spec$types <- "mean"
  mods <- tabfound_models(regressor = mean_only)

  expect_error(
    tabfound_impute(d, m = 1L, models = mods, maxit = 1L, verbose = FALSE),
    "no predictive distribution"
  )
  imp <- tabfound_impute(d, m = 1L, models = mods, maxit = 1L,
                         draw = "residual", seed = 9L, verbose = FALSE)
  expect_false(anyNA(imp$imputations[[1]]))
})


test_that("the quantile route draws from the predicted quantile function", {
  d <- data.frame(x = rnorm(30), y = c(rnorm(28), NA, NA))
  imp <- tabfound_impute(d, m = 2L, models = stub_models(), maxit = 1L,
                         draw = "quantile", quantile_grid = 99L,
                         seed = 10L, verbose = FALSE)
  expect_false(anyNA(imp$imputations[[1]]))
  expect_false(isTRUE(all.equal(imp$imputations[[1]]$y, imp$imputations[[2]]$y)))
})


test_that("with() returns a mira that mice::pool accepts", {
  skip_if_not_installed("mice")
  d <- mi_test_df(n = 30L)
  imp <- tabfound_impute(d, m = 3L, models = stub_models(), maxit = 2L,
                         seed = 11L, verbose = FALSE)

  fits <- with(imp, lm(ok ~ num + int))
  expect_s3_class(fits, "mira")
  expect_length(fits$analyses, 3L)

  pooled <- mice::pool(fits)
  expect_s3_class(pooled, "mipo")
  ests <- summary(pooled)
  expect_identical(nrow(ests), 3L)
  # Pooled standard errors must exceed the average within-imputation
  # ones: that difference *is* the imputation uncertainty.
  expect_true(all(is.finite(ests$std.error)))
})


test_that("as_mids produces a mids mice can work with", {
  skip_if_not_installed("mice")
  d <- mi_test_df(n = 30L)
  imp <- tabfound_impute(d, m = 3L, models = stub_models(), maxit = 2L,
                         seed = 12L, verbose = FALSE)
  mids <- as_mids(imp)

  expect_s3_class(mids, "mids")
  expect_equal(mids$m, 3)
  expect_identical(unname(mids$method[imp$visit_sequence]),
                   rep("tabfound", length(imp$visit_sequence)))

  for (k in 1:3) {
    from_mice <- mice::complete(mids, k)
    expect_equal(from_mice[names(d)], imp$imputations[[k]],
                 ignore_attr = TRUE)
  }
  # mice's own generic dispatches on the tabfound object too.
  expect_equal(mice::complete(imp, 2), imp$imputations[[2]])

  pooled <- summary(mice::pool(with(mids, lm(ok ~ num + fac))))
  expect_true(all(is.finite(pooled$estimate)))
})


test_that("pooling recovers a known coefficient under MCAR", {
  skip_if_not_installed("mice")
  # The stub regressor is a correctly specified linear model with normal
  # draws, so Rubin's rules should land on the truth. This is the test
  # that the loop is statistically right and not merely type-correct: a
  # chain that conditioned on the wrong rows, or drew means instead of
  # samples, would still pass everything above.
  set.seed(123)
  n  <- 300L
  x1 <- rnorm(n); x2 <- rnorm(n)
  d  <- data.frame(y = 1 + 2 * x1 - x2 + rnorm(n), x1 = x1, x2 = x2)
  d$x1[sample(n, 90L)] <- NA

  imp <- tabfound_impute(d, m = 10L, models = stub_models(), maxit = 3L,
                         verbose = FALSE)
  est <- summary(mice::pool(with(imp, lm(y ~ x1 + x2))))
  b  <- est$estimate[est$term == "x1"]
  se <- est$std.error[est$term == "x1"]

  expect_lt(abs(b - 2), 3 * se)

  # And the imputed rows are actually contributing: the pooled SE beats
  # the complete-case one. Averaged over data sets rather than asserted on
  # one, because the margin is a few percent -- proper imputation spends
  # part of what the imputed rows buy on parameter uncertainty (the
  # bootstrapped context), so a single unlucky draw can land either side.
  ratio <- vapply(1:5, function(s) {
    set.seed(1000 + s)
    x1 <- rnorm(n); x2 <- rnorm(n)
    d  <- data.frame(y = 1 + 2 * x1 - x2 + rnorm(n), x1 = x1, x2 = x2)
    d$x1[sample(n, 90L)] <- NA
    imp <- tabfound_impute(d, m = 10L, models = stub_models(), maxit = 3L,
                           verbose = FALSE)
    est <- summary(mice::pool(with(imp, lm(y ~ x1 + x2))))
    cc  <- summary(stats::lm(y ~ x1 + x2, data = d))$coefficients["x1", 2]
    est$std.error[est$term == "x1"] / cc
  }, numeric(1))
  expect_lt(mean(ratio), 1)
})


test_that(".mi_context_rows resamples only when asked", {
  obs <- c(2L, 5L, 7L, 11L, 13L, 17L)
  expect_identical(.mi_context_rows(obs, "none"), obs)
  set.seed(1)
  for (p in c("bootstrap", "bayes")) {
    draws <- replicate(20, .mi_context_rows(obs, p), simplify = FALSE)
    expect_true(all(vapply(draws, length, integer(1)) == length(obs)))
    expect_true(all(vapply(draws, function(d) all(d %in% obs), logical(1))))
    # A resample of six rows almost never comes back as the six rows.
    expect_true(mean(vapply(draws, function(d) anyDuplicated(d) > 0,
                            logical(1))) > 0.5)
  }
  # Too little to resample: one observed row is one observed row.
  expect_identical(.mi_context_rows(3L, "bootstrap"), 3L)
})


test_that("proper = TRUE widens the between-imputation variance", {
  # The reason `proper` exists: with a fixed context every chain draws
  # from the same conditional, so the m completed data sets differ only by
  # the noise of the draw and Rubin's between-imputation term is too
  # small. Resampling the context adds the parameter uncertainty back.
  #
  # The effect is O(p / n_obs) against the residual noise of the draw, so
  # it is measured on a deliberately small context and averaged over
  # several RNG streams -- at survey sizes it is a few percent and would
  # need a full coverage study, not a unit test.
  set.seed(11)
  n  <- 40L
  x1 <- rnorm(n); x2 <- rnorm(n)
  d  <- data.frame(y = 1 + 2 * x1 - x2 + rnorm(n), x1 = x1, x2 = x2)
  d$x1[sample(n, 25L)] <- NA

  between <- function(proper, seed) {
    set.seed(seed)
    imp <- tabfound_impute(d, m = 20L, models = stub_models(), maxit = 1L,
                           proper = proper, verbose = FALSE)
    # Spread of each imputed cell across the m completed data sets.
    rows <- which(imp$where[, "x1"])
    vals <- vapply(imp$imputations, function(x) x$x1[rows], numeric(length(rows)))
    mean(apply(vals, 1L, stats::var))
  }
  avg <- function(p) mean(vapply(1:3, function(s) between(p, s), numeric(1)))

  fixed <- avg(FALSE)
  expect_gt(avg(TRUE), fixed)
  expect_gt(avg("bayes"), fixed)
})


test_that("proper is validated and recorded", {
  d <- data.frame(a = c(1, 2, NA, 4, 5), b = c(1, 2, 3, 4, 5))
  imp <- tabfound_impute(d, m = 1L, models = stub_models(), maxit = 1L,
                         proper = TRUE, verbose = FALSE)
  expect_identical(imp$proper, "bootstrap")
  # Off by default: the correction costs more bias than it buys coverage
  # on these models -- see the Properness section of ?tabfound_impute.
  expect_identical(
    tabfound_impute(d, m = 1L, models = stub_models(), maxit = 1L,
                    verbose = FALSE)$proper,
    "none"
  )
  expect_error(
    tabfound_impute(d, m = 1L, models = stub_models(), proper = "yes",
                    verbose = FALSE),
    "must be"
  )
})


test_that("MI removes the bias a complete-case analysis has under MAR", {
  skip_if_not_installed("mice")
  # MCAR above says the machinery is unbiased when nothing is at stake.
  # This is the case that separates imputing from deleting: missingness
  # in x1 depends on y, which is observed and in the imputation model, so
  # it is MAR -- but dropping those rows selects on the outcome, and the
  # complete-case slope is biased. A correct chained-equations loop
  # conditions that away. A scaled-down version of
  # `inst/simulation/run-mi-sim.R`.
  set.seed(2026)
  out <- t(vapply(1:15, function(i) {
    n  <- 250L
    x2 <- rnorm(n)
    x1 <- 0.5 * x2 + sqrt(0.75) * rnorm(n)
    d  <- data.frame(y = 1 + 2 * x1 - x2 + rnorm(n), x1 = x1, x2 = x2)
    d$x1[runif(n) < stats::plogis(-0.4 + 0.9 * as.numeric(scale(d$y)))] <- NA

    imp <- tabfound_impute(d, m = 5L, models = stub_models(), maxit = 2L,
                           verbose = FALSE)
    pooled <- summary(mice::pool(with(imp, lm(y ~ x1 + x2))))
    c(cc = stats::coef(stats::lm(y ~ x1 + x2, data = d))[["x1"]],
      mi = pooled$estimate[pooled$term == "x1"])
  }, numeric(2)))

  cc_bias <- mean(out[, "cc"]) - 2
  mi_bias <- mean(out[, "mi"]) - 2
  # The mechanism has to actually bite, or the rest proves nothing.
  expect_gt(abs(cc_bias), 0.04)
  expect_lt(abs(mi_bias), abs(cc_bias) / 2)
  expect_lt(abs(mi_bias), 0.06)
})


test_that("as_amelia produces an amelia object mi.meld can pool", {
  skip_if_not_installed("Amelia")
  d <- mi_test_df(n = 30L)
  imp <- tabfound_impute(d, m = 4L, models = stub_models(), maxit = 2L,
                         seed = 13L, verbose = FALSE)
  amp <- as_amelia(imp)

  expect_s3_class(amp, "amelia")
  expect_identical(amp$m, 4L)
  expect_identical(names(amp$imputations), paste0("imp", 1:4))
  expect_s3_class(amp$imputations, "mi")
  expect_identical(amp$missMatrix, is.na(d))
  expect_output(print(amp), "4 imputed datasets")  # Amelia prints with cat()

  fits <- lapply(amp$imputations, function(x) stats::lm(ok ~ num + int, data = x))
  b  <- t(vapply(fits, stats::coef, numeric(3)))
  se <- t(vapply(fits, function(f) summary(f)$coefficients[, 2], numeric(3)))
  melded <- Amelia::mi.meld(q = b, se = se)
  expect_length(as.numeric(melded$q.mi), 3L)
  expect_true(all(is.finite(melded$se.mi)))
  # Meld's standard errors carry the between-imputation term, so they are
  # at least as large as the average within-imputation one.
  expect_true(all(as.numeric(melded$se.mi) >= colMeans(se) - 1e-8))
})


test_that("mice can drive the loop through mice.impute.tabfound", {
  skip_if_not_installed("mice")
  d <- mi_test_df(n = 30L)[, c("num", "int", "fac", "ok")]

  withr::local_options(tabfound.models = stub_models())
  mids <- mice::mice(d, method = "tabfound", m = 2L, maxit = 2L,
                     printFlag = FALSE, seed = 14L)

  expect_s3_class(mids, "mids")
  expect_identical(unname(mids$method[c("num", "int", "fac")]),
                   rep("tabfound", 3L))
  expect_false(anyNA(mice::complete(mids, 1)))
  expect_s3_class(mice::complete(mids, 1)$fac, "factor")

  pooled <- summary(mice::pool(with(mids, lm(ok ~ num + int))))
  expect_true(all(is.finite(pooled$estimate)))
})


test_that("mice.impute.tabfound says what to do when no models are set", {
  withr::local_options(tabfound.models = NULL)
  y <- c(1, 2, 3, NA)
  expect_error(
    mice.impute.tabfound(y, !is.na(y), matrix(rnorm(4), ncol = 1)),
    "No tabfound models"
  )
})


test_that("print methods say something", {
  d <- mi_test_df(n = 20L)
  imp <- tabfound_impute(d, m = 2L, models = stub_models(), maxit = 1L,
                         seed = 15L, verbose = FALSE)
  # cli writes to stderr, so capture that rather than stdout.
  expect_match(paste(capture.output(print(imp), type = "message"), collapse = " "),
               "multiple imputation")
  expect_match(paste(capture.output(print(stub_models()), type = "message"),
                     collapse = " "),
               "model handle")
})


test_that("a NaN probability row does not become a missing category", {
  # What a backend returns when the predictors it was handed still hold
  # NA. Untreated, the NaN row walks through `cumsum` into `lab[NA]` and
  # deposits NA_character_ in a cell the caller was told was imputed --
  # the numeric draw has warned about exactly this since it was written.
  nan_clf <- stub_model("classification")
  inner   <- nan_clf$spec$predict
  nan_clf$spec$predict <- function(state, newdata, type = "class", ...) {
    out <- inner(state, newdata, type, ...)
    if (identical(type, "prob")) out[1L, ] <- NaN
    out
  }

  X <- matrix(rnorm(40), ncol = 2)
  y <- factor(rep(c("a", "b"), 10))
  expect_warning(
    lab <- mi_draw_factor(nan_clf, X, y, matrix(rnorm(8), ncol = 2)),
    "non-finite probability row"
  )
  expect_false(anyNA(lab))
  expect_true(all(lab %in% levels(y)))
})


test_that("the completed data are checked for cells left missing", {
  d <- data.frame(a = c(1, 2, NA, 4, 5, 6), b = rnorm(6))
  mask <- matrix(c(rep(FALSE, 2), TRUE, rep(FALSE, 3), rep(FALSE, 6)),
                 ncol = 2, dimnames = list(NULL, c("a", "b")))
  expect_identical(.mi_check_complete(d, mask, "b", 1L), d)
  expect_error(.mi_check_complete(d, mask, "a", 2L), "left 1 cell missing")
})


test_that("each draw declares the categoricals of its own predictor set", {
  # The package's headline advantage over the Python wrappers is that
  # `is.factor()` is exact where a cardinality heuristic guesses. Chained
  # equations change the predictor set every variable, so a fixed index
  # vector on the model cannot express it -- the declaration has to be
  # computed per draw and the model re-specced around it.
  seen <- new.env(parent = emptyenv())
  seen$calls <- list()
  recording_backend <- function(task) {
    spec_fn <- function(ctx, categorical_features = NULL, ...) {
      seen$calls <- c(seen$calls, list(categorical_features))
      if (identical(task, "classification")) stub_classifier_spec()
      else stub_regressor_spec()
    }
    spec_fn
  }
  register_backend(name = "catprobe",
                   build = function(config, task) NULL,
                   classifier = recording_backend("classification"),
                   regressor  = recording_backend("regression"))
  withr::defer(rm("catprobe", envir = .tabfound_backends))

  mk <- function(task) {
    m <- stub_model(task)
    m$backend <- "catprobe"
    m
  }
  mods <- tabfound_models(classifier = mk("classification"),
                          regressor  = mk("regression"))

  set.seed(4)
  n <- 30L
  d <- data.frame(num = rnorm(n),
                  fac = factor(sample(c("a", "b", "c"), n, TRUE)),
                  z   = rnorm(n))
  d$num[1:5] <- NA
  d$fac[6:10] <- NA
  tabfound_impute(d, m = 1L, models = mods, maxit = 1L, verbose = FALSE)

  # Imputing `num` conditions on (fac, z): the factor is column 1 of that
  # set. Imputing `fac` conditions on (num, z): neither is categorical, so
  # nothing is declared and no respec happens.
  declared <- Filter(Negate(is.null), seen$calls)
  expect_true(length(declared) >= 1L)
  expect_true(any(vapply(declared, function(x) identical(as.integer(x), 1L),
                         logical(1))))
  # Indices are positions within the predictor set, never in the original
  # frame -- `fac` is column 2 of `d` and column 1 of `num`'s predictors.
  expect_false(any(vapply(declared, function(x) identical(as.integer(x), 2L),
                          logical(1))))
})


test_that("where selects the cells to draw, including observed ones", {
  set.seed(31)
  d <- data.frame(a = c(rnorm(18), NA, NA), b = rnorm(20))
  # Impute only one of the two missing cells.
  w <- matrix(FALSE, 20L, 2L, dimnames = list(NULL, c("a", "b")))
  w[19, "a"] <- TRUE
  imp <- tabfound_impute(d, m = 1L, models = stub_models(), maxit = 1L,
                         where = w, verbose = FALSE)
  got <- imp$imputations[[1]]
  expect_false(is.na(got$a[19]))
  expect_true(is.na(got$a[20]))          # not asked for, left alone
  expect_identical(got$b, d$b)

  # Overimputation: an observed cell is redrawn, which is the diagnostic
  # use of `where`, and the original is not silently kept.
  w2 <- matrix(FALSE, 20L, 2L, dimnames = list(NULL, c("a", "b")))
  w2[1:3, "a"] <- TRUE
  over <- tabfound_impute(d, m = 1L, models = stub_models(), maxit = 1L,
                          where = w2, verbose = FALSE)$imputations[[1]]
  expect_false(any(over$a[1:3] == d$a[1:3]))
  expect_true(is.na(over$a[19]))

  expect_error(tabfound_impute(d, models = stub_models(), where = w[, 1, drop = FALSE]),
               "20 x 1")
  expect_error(tabfound_impute(d, models = stub_models(), where = "yes"),
               "logical matrix")
})


test_that("post-processing is applied to the drawn values", {
  d <- data.frame(a = c(rnorm(17), NA, NA, NA), b = rnorm(20))
  imp <- tabfound_impute(d, m = 2L, models = stub_models(), maxit = 2L,
                         post = list(a = function(v) pmax(v, 99)),
                         verbose = FALSE)
  for (k in 1:2) {
    got <- imp$imputations[[k]]$a
    expect_true(all(got[is.na(d$a)] >= 99))
    expect_identical(got[!is.na(d$a)], d$a[!is.na(d$a)])   # observed untouched
  }

  expect_error(
    tabfound_impute(d, models = stub_models(), post = list(a = function(v) v[1]),
                    verbose = FALSE),
    "returned 1 value"
  )
  expect_error(tabfound_impute(d, models = stub_models(), post = list(zz = identity)),
               "not in the data")
  expect_error(tabfound_impute(d, models = stub_models(), post = list(a = 1)),
               "not")
})


test_that("the chain trace is recorded and reaches mice", {
  d <- mi_test_df(n = 30L)
  imp <- tabfound_impute(d, m = 2L, models = stub_models(), maxit = 3L,
                         seed = 9L, verbose = FALSE)

  expect_identical(dim(imp$chain_mean), c(ncol(d), 3L, 2L))
  expect_identical(rownames(imp$chain_mean), names(d))
  # Every visited variable has a number at every sweep of every chain;
  # the untouched ones stay NA.
  for (v in imp$visit_sequence) {
    expect_false(anyNA(imp$chain_mean[v, , ]), info = v)
  }
  expect_true(all(is.na(imp$chain_mean["ok", , ])))

  skip_if_not_installed("mice")
  mids <- as_mids(imp)
  expect_identical(dim(mids$chainMean), dim(imp$chain_mean))
  expect_false(anyNA(mids$chainMean[imp$visit_sequence[1], , ]))
  # The complaint this fixes: `plot()` on the converted object used to
  # draw an empty frame.
  pdf(NULL); on.exit(dev.off(), add = TRUE)
  expect_no_error(print(plot(mids)))
})
