# The formula / data-frame interface.
#
# Most of this needs no model weights: the encoding and the hardhat
# blueprint are exercised directly. The few tests that do need a model
# skip unless TABFOUND_TABICL_CLF_DIR / TABFOUND_TABPFN_CLF_DIR are set.

skip_if_not_installed("hardhat")

mixed_df <- function(n = 12L) {
  set.seed(11)
  data.frame(
    num  = rnorm(n),
    int  = seq_len(n),
    fac  = factor(rep(c("a", "b", "c"), length.out = n)),
    ord  = factor(rep(c("lo", "hi"), length.out = n),
                  levels = c("lo", "hi"), ordered = TRUE),
    lgl  = rep(c(TRUE, FALSE), length.out = n),
    chr  = rep(c("x", "y"), length.out = n),
    date = as.Date("2020-01-01") + seq_len(n),
    y    = factor(rep(c("p", "q"), length.out = n)),
    stringsAsFactors = FALSE
  )
}


test_that("coercion makes formula and xy paths agree on column count", {
  d <- mixed_df()
  # A bare logical would be dummy-expanded into two columns by the
  # formula path's model.matrix step while surviving intact through the
  # xy path. Coercing to a factor first keeps both at one column each.
  bp <- hardhat::default_formula_blueprint(indicators = "none", intercept = FALSE)
  m_f <- hardhat::mold(y ~ ., .coerce_for_mold(d), blueprint = bp)
  m_x <- hardhat::mold(.coerce_for_mold(d[setdiff(names(d), "y")]), d$y)

  expect_identical(ncol(m_f$predictors), ncol(m_x$predictors))
  expect_setequal(names(m_f$predictors), names(m_x$predictors))
  expect_false("lglTRUE" %in% names(m_f$predictors))
})


test_that("every R column type encodes to one numeric column", {
  d <- .coerce_for_mold(mixed_df())
  m <- hardhat::mold(d[setdiff(names(d), "y")], d$y)
  x <- .encode_predictors(m$predictors)

  expect_true(is.matrix(x))
  expect_true(is.numeric(x))
  expect_identical(ncol(x), ncol(m$predictors))
  expect_identical(nrow(x), nrow(d))
  expect_false(anyNA(x))

  # Factors become 0-based ordinal codes, matching what the reference
  # implementations feed their networks -- not one-hot indicators.
  expect_identical(sort(unique(x[, "fac"])), c(0, 1, 2))
  expect_identical(sort(unique(x[, "lgl"])), c(0, 1))
  expect_equal(x[, "date"], as.numeric(d$date), ignore_attr = TRUE)

  expect_error(.encode_predictors(data.frame(z = complex(real = 1:3))),
               "cannot be used as a predictor")
})


test_that("factor columns are declared categorical, by position", {
  d <- .coerce_for_mold(mixed_df(40L))
  bp <- hardhat::default_formula_blueprint(indicators = "none", intercept = FALSE)
  processed <- hardhat::mold(y ~ ., d, blueprint = bp)

  cats <- tabfound:::.categorical_predictor_indices(processed$predictors)
  x <- tabfound:::.encode_predictors(processed$predictors)

  # Positions, not names: the reference takes column indices, and hardhat
  # is free to reorder the molded frame. Both functions read that same
  # frame, so they cannot disagree -- which is the point of checking the
  # names line up rather than hard-coding the indices.
  expect_identical(colnames(x)[cats], names(processed$predictors)[cats])
  expect_true(all(vapply(processed$predictors[cats], is.factor, logical(1))))
  # Characters and logicals were coerced to factors upstream, so they are
  # declared too; numerics, dates and integers are not.
  expect_setequal(colnames(x)[cats], c("fac", "ord", "lgl", "chr"))

  # A declaration only sticks if the column is low-cardinality enough for
  # the model to treat it as categorical at all.
  expect_setequal(detect_categorical_features(x, cats), cats)
})


test_that("a backend without categorical handling says so", {
  # Silently ignoring a declaration would be the worst outcome: the user
  # believes the column is being encoded and it is not.
  fake <- list(name = "nocat")
  fn <- function(ctx, predict_chunk_size = 1L) NULL
  expect_warning(
    tabfound:::.predictor_args(fn, list(backend = fake), c(1L, 2L), list()),
    "no categorical handling"
  )
  # Nothing declared: nothing to warn about.
  expect_silent(tabfound:::.predictor_args(fn, list(backend = fake),
                                           integer(), list()))
  # A backend that does take it gets it passed through.
  fn2 <- function(ctx, categorical_features = NULL) NULL
  args <- tabfound:::.predictor_args(fn2, list(backend = fake), c(1L, 2L), list())
  expect_identical(args$categorical_features, c(1L, 2L))
})


test_that("the blueprint validates new data at predict time", {
  d <- .coerce_for_mold(mixed_df())
  bp <- hardhat::default_formula_blueprint(indicators = "none", intercept = FALSE)
  m <- hardhat::mold(y ~ ., d, blueprint = bp)

  # This is the job the Python wrappers do by re-running fitted encoders;
  # here it is the blueprint's, and it fails loudly rather than silently
  # producing a differently-shaped matrix.
  nd <- d[1:3, setdiff(names(d), "y")]
  expect_silent(hardhat::forge(nd, m$blueprint))

  short <- nd[, c("num", "fac")]
  expect_error(hardhat::forge(short, m$blueprint), "missing")

  novel <- nd
  novel$fac <- factor(c("z", "a", "b"))
  expect_warning(hardhat::forge(novel, m$blueprint), "Novel level")
})


test_that("mode is inferred from the outcome type", {
  d <- mixed_df()
  expect_identical(.infer_mode(factor(c("a", "b"))), "classification")
  expect_identical(.infer_mode(c("a", "b")), "classification")
  expect_identical(.infer_mode(c(TRUE, FALSE)), "classification")
  expect_identical(.infer_mode(c(1.5, 2.5)), "regression")
  expect_identical(.infer_mode(1:5), "regression")
})


test_that("na_action resolves from the backend's own capability", {
  # Every shipped backend now absorbs NaN somewhere -- TabPFN encodes it,
  # TabFM maps it to a sentinel, and TabICL and Mitra impute it in their
  # predictors the way their reference wrappers do -- so "auto" passes it
  # through and lets the backend's own (parity-checked) handling run,
  # rather than layering this package's imputer in front of it.
  x <- matrix(c(1, NA, 3, 4), ncol = 2)
  for (backend in c("tabpfn", "tabfm", "tabicl", "mitra")) {
    expect_identical(.resolve_na_action("auto", backend, x), "pass")
    expect_silent(.resolve_na_action("pass", backend, x))
  }
  # The warning path is still live for any backend that declares it
  # cannot cope; it is the declaration, not the backend name, that
  # decides.
  withr::with_options(list(), {
    register_backend(name = ".na_probe", build = function(config, task) NULL,
                     handles_missing = FALSE)
    on.exit(rm(".na_probe", envir = tabfound:::.tabfound_backends), add = TRUE)
    expect_warning(.resolve_na_action("pass", ".na_probe", x), "no missing-value")
    expect_silent(.resolve_na_action("pass", ".na_probe", matrix(1:4, ncol = 2)))
    expect_identical(.resolve_na_action("auto", ".na_probe", x), "impute")
  })
})


test_that("the imputer fills only non-finite cells, with column means", {
  x <- matrix(c(1, NA, 3, 10, 20, NaN), ncol = 2)
  means <- .fit_imputer(x)
  expect_equal(means, c(2, 15))
  filled <- .apply_imputer(x, means)
  expect_false(anyNA(filled))
  expect_equal(filled[, 1], c(1, 2, 3))
  expect_equal(filled[, 2], c(10, 20, 15))
  # An all-missing column has no mean to take; 0 is the documented
  # fallback rather than NaN leaking onward.
  expect_equal(.fit_imputer(matrix(c(NA_real_, NA_real_), ncol = 1)), 0)
})


# --- tests that need weights ----------------------------------------------

model_dir <- function(var) {
  d <- Sys.getenv(var, unset = "")
  if (!nzchar(d) || !dir.exists(d)) NULL else d
}

test_that("formula and xy interfaces agree, and predictions are tidy", {
  md <- model_dir("TABFOUND_TABICL_CLF_DIR")
  skip_if(is.null(md), "TABFOUND_TABICL_CLF_DIR not configured")

  set.seed(1)
  tr <- sort(sample.int(150, 100)); te <- setdiff(seq_len(150), tr)

  f1 <- tabfound(Species ~ ., data = iris[tr, ], model = md)
  f2 <- tabfound(iris[tr, 1:4], iris[tr, "Species"], model = md)

  p1 <- predict(f1, iris[te, ])
  p2 <- predict(f2, iris[te, 1:4])
  expect_identical(p1$.pred_class, p2$.pred_class)

  expect_identical(names(p1), ".pred_class")
  expect_true(is.factor(p1$.pred_class))
  expect_identical(levels(p1$.pred_class), levels(iris$Species))
  expect_gt(mean(p1$.pred_class == iris[te, "Species"]), 0.9)

  pp <- predict(f1, iris[te, ], type = "prob")
  expect_identical(names(pp),
                   paste0(".pred_", levels(iris$Species)))
  expect_true(all(abs(rowSums(pp) - 1) < 1e-5))

  # Prediction must not need the outcome column present.
  expect_no_error(predict(f1, iris[te, 1:4]))
})


test_that("regression goes through the same path and reports .pred", {
  md <- model_dir("TABFOUND_TABICL_REG_DIR")
  skip_if(is.null(md), "TABFOUND_TABICL_REG_DIR not configured")

  set.seed(1)
  tr <- sort(sample.int(150, 100)); te <- setdiff(seq_len(150), tr)
  f <- tabfound(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width,
                data = iris[tr, ], model = md)
  expect_identical(f$mode, "regression")

  p <- predict(f, iris[te, ])
  expect_identical(names(p), ".pred")
  expect_lt(sqrt(mean((p$.pred - iris[te, "Sepal.Length"])^2)), 0.6)

  q <- predict(f, iris[te, ], type = "quantiles",
               quantiles = c(0.1, 0.5, 0.9))
  expect_identical(names(q), c(".pred_q0.1", ".pred_q0.5", ".pred_q0.9"))
  expect_true(all(q$.pred_q0.1 <= q$.pred_q0.9))
})


test_that("missing predictors are imputed for tabicl and passed for tabpfn", {
  icl <- model_dir("TABFOUND_TABICL_CLF_DIR")
  pfn <- model_dir("TABFOUND_TABPFN_CLF_DIR")
  skip_if(is.null(icl) || is.null(pfn), "model dirs not configured")

  set.seed(1)
  tr <- sort(sample.int(150, 100)); te <- setdiff(seq_len(150), tr)
  d <- iris; d[tr[1:5], "Sepal.Width"] <- NA

  f_icl <- tabfound(Species ~ ., data = d[tr, ], model = icl)
  expect_identical(f_icl$na_action, "impute")
  p <- predict(f_icl, d[te, ], type = "prob")
  expect_false(anyNA(p))

  f_pfn <- tabfound(Species ~ ., data = d[tr, ], model = pfn)
  expect_identical(f_pfn$na_action, "pass")
  expect_null(f_pfn$imputer)
  # TabPFN encodes missingness explicitly, so NA in *new* data is fine too.
  nd <- d[te, ]; nd[1, "Petal.Length"] <- NA
  expect_false(anyNA(predict(f_pfn, nd, type = "prob")))
})
