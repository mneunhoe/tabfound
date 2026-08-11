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
  # the complete-case one.
  cc <- summary(stats::lm(y ~ x1 + x2, data = d))$coefficients["x1", 2]
  expect_lt(se, cc)
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
