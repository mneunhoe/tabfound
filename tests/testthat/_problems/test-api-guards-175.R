# Extracted from test-api-guards.R:175

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "tabfound", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
dir <- local_fake_backend()
bk <- get("faketest", envir = tabfound:::.tabfound_backends)
bk$regressor <- function(ctx, categorical_features = NULL, ...) {
    spec <- stub_regressor_spec()
    fit0 <- spec$fit
    spec$fit <- function(X, y) {
      state <- fit0(X, y)
      state$declared <- categorical_features
      state
    }
    spec
  }
assign("faketest", bk, envir = tabfound:::.tabfound_backends)
X <- data.frame(a = rnorm(20), b = rnorm(20), c = sample(0:4, 20, TRUE))
y <- rnorm(20)
m <- expect_silent(fit(tabular_regressor(dir, categorical_features = "c"),
                         X, y))
