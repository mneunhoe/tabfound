# What `fit()` and `predict()` accept, and what they refuse.
#
# Each of these guards a failure mode that used to be silent: wrong
# numbers rather than an error.

test_that("a data frame with a factor column is encoded, not mangled", {
  # `as.matrix()` on a mixed frame goes through `format()`: the factor
  # column comes back as character (and then NA under `storage.mode<-`),
  # and every numeric column is rounded to 7 significant digits. Both
  # happen without an error.
  d <- data.frame(big = c(1234567.89, 2345678.91, 3456789.12, 4567890.13),
                  fac = factor(c("a", "b", "a", "c")),
                  chr = c("x", "y", "x", "y"),
                  lgl = c(TRUE, FALSE, TRUE, FALSE),
                  stringsAsFactors = FALSE)

  # The behaviour being replaced, stated so the test explains itself.
  old <- suppressWarnings({
    mm <- as.matrix(d); storage.mode(mm) <- "double"; mm
  })
  expect_true(all(is.na(old[, "fac"])))
  expect_false(isTRUE(all.equal(old[, "big"], d$big)))

  x <- .as_model_matrix(d)
  expect_true(is.numeric(x))
  expect_false(anyNA(x))
  expect_identical(x[, "big"], d$big)                  # full precision
  expect_identical(x[, "fac"], c(0, 1, 0, 2))          # 0-based codes
  expect_identical(x[, "chr"], c(0, 1, 0, 1))
  expect_identical(x[, "lgl"], c(1, 0, 1, 0))
})


test_that("fit() routes data frames through the encoder and says so", {
  d <- data.frame(a = c(1, 2, 3, 4), f = factor(c("p", "q", "p", "q")))
  y <- c(1.5, 2.5, 3.5, 4.5)
  m <- stub_model("regression")

  expect_warning(m2 <- fit(m, d, y), "categorical column")
  expect_identical(m2$state$n_features, 2L)
  expect_identical(m2$state$feature_names, c("a", "f"))

  # Declaring them is the way to silence it, and it is what tabfound() does.
  m$model_ref$args$categorical_features <- 2L
  expect_silent(fit(m, d, y))

  # An all-numeric frame has nothing to declare.
  expect_silent(fit(stub_model("regression"), d["a"], y))
})


test_that("a non-numeric matrix is refused rather than coerced", {
  m <- stub_model("regression")
  expect_error(fit(m, matrix(letters[1:4], ncol = 2), c(1, 2)),
               "matrix; these models take numbers")
  expect_error(fit(m, list(1, 2), c(1, 2)), "must be a matrix or a data frame")
  # A logical matrix is numbers in disguise and is fine.
  expect_no_error(fit(m, matrix(c(TRUE, FALSE, TRUE, FALSE), ncol = 2), c(1, 2)))
})


test_that("predict checks new data against the fitted schema", {
  X <- matrix(rnorm(30), ncol = 3, dimnames = list(NULL, c("a", "b", "c")))
  m <- fit(stub_model("regression"), X, rnorm(10))

  expect_no_error(predict(m, X))
  # Right columns, no names: nothing to check, and matrices often have none.
  expect_no_error(predict(m, unname(X)))

  expect_error(predict(m, X[, 1:2]), "2 columns; the model was fitted on 3")
  expect_error(predict(m, cbind(X, d = 1)), "4 columns")
  # The one that used to be silent: same columns, different order.
  expect_error(predict(m, X[, c("b", "a", "c")]), "different order")
  expect_error(predict(m, `colnames<-`(X, c("a", "b", "z"))), "Missing")
})


test_that("factor codes at predict mean what they meant at fit", {
  # A factor's code is its position in its own level set, so a test frame
  # holding only two of the three training levels would encode "c" as 1
  # where training encoded it as 2 -- the same class of silent error as
  # reordered columns. hardhat's forge() covers the formula path.
  train <- data.frame(f = factor(c("a", "b", "c", "a", "b", "c")),
                      n = c(1, 2, 3, 4, 5, 6))
  m <- suppressWarnings(fit(stub_model("regression"), train, rnorm(6)))
  expect_identical(m$state$feature_levels, list(f = c("a", "b", "c")))

  fresh <- data.frame(f = factor(c("c", "b")), n = c(1, 2))
  expect_identical(.encode_predictors(fresh)[, "f"], c(1, 0))   # wrong on its own
  expect_identical(
    .as_model_matrix(fresh, levels = m$state$feature_levels)[, "f"],
    c(2, 1)                                                     # right against train
  )

  # A level nobody trained on cannot be encoded; it becomes NA, loudly.
  novel <- data.frame(f = factor(c("a", "z")), n = c(1, 2))
  expect_warning(x <- .as_model_matrix(novel, levels = m$state$feature_levels),
                 "not fitted on")
  expect_identical(x[, "f"], c(0, NA_real_))
})


test_that("out-of-range quantiles are refused", {
  m <- fit(stub_model("regression"), matrix(rnorm(20), ncol = 2), rnorm(10))
  nd <- matrix(rnorm(6), ncol = 2)

  expect_error(predict(m, nd, type = "quantiles", quantiles = c(-0.5, 1.7)),
               "strictly inside")
  expect_error(predict(m, nd, type = "quantiles", quantiles = c(0.1, 1)),
               "strictly inside")
  expect_error(predict_quantiles(m, nd, quantiles = c(0, 0.5)), "strictly inside")
  expect_no_error(predict(m, nd, type = "quantiles", quantiles = c(0.1, 0.9)))
})


test_that("predict-time arguments the backend does not take are reported", {
  reg <- fit(stub_model("regression"), matrix(rnorm(20), ncol = 2), rnorm(10))
  clf <- fit(stub_model("classification"), matrix(rnorm(20), ncol = 2),
             factor(rep(c("a", "b"), 5)))
  nd <- matrix(rnorm(6), ncol = 2)

  # The real knobs are constructor-time, so a misspelt one at predict
  # time does nothing at all.
  expect_warning(predict(reg, nd, softmax_temperatur = 0.1), "softmax_temperatur")
  expect_warning(predict(clf, nd, type = "prob", n_samples = 5L), "n_samples")
  # What the backend does declare passes without comment.
  expect_silent(predict(reg, nd, type = "sample", n_samples = 2L, seed = 1L))
  expect_silent(predict(clf, nd, type = "prob"))
})


test_that("a fitted tabfound_fit survives a save/load round trip", {
  skip_if_not_installed("hardhat")
  dir <- local_fake_backend()
  tr <- c(1:20, 51:70, 101:120)
  f  <- tabfound(Species ~ ., data = iris[tr, ], model = dir)

  expect_true(is_fitted(f))
  file <- withr::local_tempfile()
  expect_identical(tabfound_save(f, file), file)

  blob <- readRDS(file)
  expect_identical(blob$format, "tabfound-fit")
  expect_lt(file.size(file), 200000)      # a reference, not the weights

  f2 <- tabfound_load(file)
  expect_s3_class(f2, "tabfound_fit")
  expect_true(is_fitted(f2))
  expect_identical(predict(f2, iris[c(1, 60, 130), ]),
                   predict(f, iris[c(1, 60, 130), ]))
  # The blueprint travelled with it, so new data is still re-encoded and
  # re-ordered rather than taken on trust.
  shuffled <- iris[c(1, 60, 130), c(3, 1, 4, 2, 5)]
  expect_identical(predict(f2, shuffled), predict(f, iris[c(1, 60, 130), ]))
  expect_equal(predict_proba(f2, iris[1:3, ]), predict_proba(f, iris[1:3, ]))
  expect_identical(f2$mode, f$mode)
  expect_identical(f2$na_action, f$na_action)
  expect_no_error(print(f2))
})


test_that("save/load refuses objects and formats it cannot handle", {
  file <- withr::local_tempfile()
  expect_error(tabfound_save(list(a = 1), file), "must be a")

  m <- fit(stub_model("classification"), matrix(rnorm(20), ncol = 2),
           factor(rep(c("a", "b"), 5)))
  tabfound_save(m, file)
  blob <- readRDS(file)
  # A payload from a future version is refused, not half-read.
  blob$version <- 99L
  saveRDS(blob, file)
  expect_error(tabfound_load(file), "format version")
})


test_that("a fit-and-predict cycle runs on the accelerator", {
  # The memory guard is a no-op off CPU and nothing else in the suite
  # leaves it, so this is the one test that says the device path works at
  # all. It runs wherever there is an accelerator and skips where there
  # is not.
  skip_if_not_installed("torch")
  dev <- if (torch::cuda_is_available()) "cuda"
         else if (torch::backends_mps_is_available()) "mps"
         else NULL
  skip_if(is.null(dev), "no accelerator available")
  d <- Sys.getenv("TABFOUND_TABPFN_CLF_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN_CLF_DIR not configured")

  set.seed(2)
  X <- matrix(rnorm(120 * 4), ncol = 4)
  y <- factor(ifelse(X[, 1] > 0, "b", "a"))
  tr <- 1:80; te <- 81:120

  on_dev <- fit(tabular_classifier(d, device = dev), X[tr, ], y[tr])
  p_dev  <- predict(on_dev, X[te, ], type = "prob")
  expect_true(all(is.finite(p_dev)))
  expect_equal(rowSums(p_dev), rep(1, length(te)), tolerance = 1e-5)

  on_cpu <- fit(tabular_classifier(d, device = "cpu"), X[tr, ], y[tr])
  p_cpu  <- predict(on_cpu, X[te, ], type = "prob")
  # Different kernels, same arithmetic to within float32 noise.
  expect_equal(p_dev, p_cpu, tolerance = 1e-4)
})
