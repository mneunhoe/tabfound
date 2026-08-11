# Sequential synthesis: the single-pass loop and the synthpop bridges.
#
# Everything here runs against `stub_model()` (see helper-stub-model.R),
# so it needs no weights and no GPU. What is under test is the contract --
# types and support survive, the level coding is consistent between the
# context and the query, the same seed reproduces the same syntheses, and
# synthpop accepts the result -- plus one statistical check that the pass
# actually transmits a conditional relationship rather than just filling
# cells with plausible-looking numbers.

syn_test_df <- function(n = 60L, seed = 7L) {
  set.seed(seed)
  data.frame(
    num = rnorm(n),
    int = sample(1:20, n, replace = TRUE),
    fac = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    lgl = sample(c(TRUE, FALSE), n, replace = TRUE),
    chr = sample(c("x", "y"), n, replace = TRUE),
    bin = as.numeric(sample(0:1, n, replace = TRUE)),
    ibin = sample(0:1, n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}


test_that("every column is synthesised, with its own type and shape", {
  d <- syn_test_df()
  s <- tabfound_syn(d, m = 3L, models = stub_models(), seed = 1L,
                    verbose = FALSE)

  expect_s3_class(s, "tabfound_syn")
  expect_length(s$syn, 3L)
  for (k in seq_len(3L)) {
    got <- s$syn[[k]]
    expect_identical(dim(got), dim(d))
    expect_identical(names(got), names(d))
    expect_type(got$num, "double")
    expect_type(got$int, "integer")
    expect_s3_class(got$fac, "factor")
    expect_identical(levels(got$fac), levels(d$fac))
    expect_type(got$lgl, "logical")
    expect_type(got$chr, "character")
    # A numeric 0/1 column takes the classifier path whether or not it has
    # missing values: in synthesis every cell is drawn, and a bar
    # distribution cannot produce a dummy.
    expect_type(got$bin, "double")
    expect_true(all(got$bin %in% c(0, 1)))
    expect_true(all(got$chr %in% c("x", "y")))
    # ... and an integer dummy stays an integer.
    expect_type(got$ibin, "integer")
    expect_true(all(got$ibin %in% 0:1))
  }
  expect_identical(unname(s$method[["num"]]), "sample")
  expect_identical(unname(s$method[["int"]]), "regression")
  expect_identical(unname(s$method[["fac"]]), "classification")
})


test_that("a seed makes the whole synthesis reproducible", {
  d <- syn_test_df()
  a <- tabfound_syn(d, m = 2L, models = stub_models(), seed = 42L,
                    verbose = FALSE)
  b <- tabfound_syn(d, m = 2L, models = stub_models(), seed = 42L,
                    verbose = FALSE)
  c3 <- tabfound_syn(d, m = 2L, models = stub_models(), seed = 43L,
                     verbose = FALSE)

  expect_equal(a$syn, b$syn)
  expect_false(isTRUE(all.equal(a$syn, c3$syn)))
})


test_that("the syntheses differ from each other and from the real data", {
  d <- syn_test_df(n = 80L)
  s <- tabfound_syn(d, m = 4L, models = stub_models(), seed = 2L,
                    verbose = FALSE)
  means <- vapply(s$syn, function(x) mean(x$int), numeric(1))
  expect_gt(stats::var(means), 0)
  for (k in seq_len(4L)) expect_false(isTRUE(all.equal(s$syn[[k]], d)))
})


test_that("k != n generates a data set of the requested size", {
  d <- syn_test_df()
  s <- tabfound_syn(d, m = 2L, k = 25L, models = stub_models(), seed = 3L,
                    verbose = FALSE)
  expect_identical(nrow(s$syn[[1]]), 25L)
  expect_identical(s$k, 25L)
  expect_identical(s$n, 60L)
  expect_identical(names(s$syn[[2]]), names(d))
})


test_that("the visit sequence is respected and only earlier variables predict", {
  d <- syn_test_df()
  s <- tabfound_syn(d, m = 1L, models = stub_models(),
                    visit_sequence = c("fac", "num", "int", "lgl", "chr",
                                       "bin", "ibin"),
                    seed = 4L, verbose = FALSE)
  expect_identical(s$visit_sequence[1:2], c("fac", "num"))
  # The first variable has nothing to condition on, so it is a bootstrap
  # of its own marginal whatever its type.
  expect_identical(unname(s$method[["fac"]]), "sample")
  expect_identical(as.integer(s$predictors["fac", ]), rep(0L, ncol(d)))
  expect_identical(unname(s$predictors["num", "fac"]), 1L)
  expect_identical(unname(s$predictors["fac", "num"]), 0L)
  # Upper-triangular: nothing conditions on a variable generated later.
  ord <- s$visit_sequence
  expect_true(all(s$predictors[ord, ord][upper.tri(diag(length(ord)),
                                                   diag = TRUE)] == 0L))
})


test_that("predictors pointing forward are dropped with a warning", {
  d <- syn_test_df(n = 40L)
  pm <- matrix(1L, ncol(d), ncol(d), dimnames = list(names(d), names(d)))
  expect_warning(
    s <- tabfound_syn(d, m = 1L, models = stub_models(), predictors = pm,
                      seed = 5L, verbose = FALSE),
    "later in the visit sequence"
  )
  expect_identical(unname(s$predictors["num", "int"]), 0L)
  expect_identical(unname(s$predictors["int", "num"]), 1L)

  pm2 <- matrix(0L, ncol(d), ncol(d), dimnames = list(names(d), names(d)))
  pm2["int", "num"] <- 1L
  s2 <- tabfound_syn(d, m = 1L, models = stub_models(), predictors = pm2,
                     seed = 6L, verbose = FALSE)
  expect_identical(as.integer(rowSums(s2$predictors)),
                   c(0L, 1L, rep(0L, ncol(d) - 2L)))
  # Everything with no predictor left falls back to its marginal.
  expect_identical(unname(s2$method[["fac"]]), "sample")

  expect_error(tabfound_syn(d, models = stub_models(),
                            predictors = matrix(1, 2, 2)),
               "square matrix")
})


test_that("method controls the loop and '' carries the real column through", {
  d <- syn_test_df(n = 40L)

  meth <- stats::setNames(rep("sample", ncol(d)), names(d))
  s <- tabfound_syn(d, m = 1L, models = stub_models(), method = meth,
                    seed = 7L, verbose = FALSE)
  expect_true(all(s$syn[[1]]$num %in% d$num))
  expect_true(all(s$syn[[1]]$fac %in% d$fac))

  s2 <- tabfound_syn(d, m = 1L, models = stub_models(),
                     method = c(num = ""), seed = 8L, verbose = FALSE)
  expect_identical(s2$syn[[1]]$num, d$num)
  expect_false("num" %in% s2$visit_sequence)
  # Carried-through columns are still available to condition on.
  expect_identical(unname(s2$predictors["int", "num"]), 1L)

  expect_error(tabfound_syn(d, k = 10L, method = c(num = ""),
                            models = stub_models(), verbose = FALSE),
               "k == nrow\\(data\\)")
  expect_error(tabfound_syn(d, models = stub_models(), method = c(num = "nope")),
               "Unknown method")
  expect_error(tabfound_syn(d, models = stub_models(),
                            method = c(fac = "regression")),
               "categorical")
  expect_error(tabfound_syn(d, models = stub_models(),
                            method = c(num = "classification")),
               "continuous")
})


test_that("the draw modes have the support behaviour they claim", {
  set.seed(31)
  d <- data.frame(x = rnorm(120), y = round(rnorm(120) * 10))

  pred <- tabfound_syn(d, m = 1L, models = stub_models(), draw_mode = "predictive",
                       seed = 9L, verbose = FALSE)$syn[[1]]
  # Integrality is a type fact, so it is restored whatever the mode; the
  # observed range is restored only under `clamp`.
  expect_true(all(pred$y == round(pred$y)))
  expect_gte(min(pred$y), min(d$y))
  expect_lte(max(pred$y), max(d$y))

  loose <- tabfound_syn(d, m = 1L, models = stub_models(), clamp = FALSE,
                        seed = 9L, verbose = FALSE)$syn[[1]]
  expect_true(all(loose$y == round(loose$y)))

  pmm <- tabfound_syn(d, m = 1L, models = stub_models(), draw_mode = "pmm",
                      donors = 3L, seed = 10L, verbose = FALSE)$syn[[1]]
  # PMM returns real donors, so every value is one the data contains.
  expect_true(all(pmm$y %in% d$y))

  # Rank mapping reproduces the marginal exactly -- that is the whole
  # point of it, and also why its marginal fidelity is not evidence. It
  # reproduces the *context's* marginal, so exactness needs an
  # unbootstrapped context.
  rk <- tabfound_syn(d, m = 1L, models = stub_models(), draw_mode = "rank",
                     proper = FALSE, seed = 11L, verbose = FALSE)$syn[[1]]
  expect_identical(sort(rk$y), sort(d$y))
  rkp <- tabfound_syn(d, m = 1L, models = stub_models(), draw_mode = "rank",
                      proper = TRUE, seed = 11L, verbose = FALSE)$syn[[1]]
  expect_true(all(rkp$y %in% d$y))

  sm <- tabfound_syn(d, m = 1L, models = stub_models(), draw_mode = "pmm",
                     smoothing = TRUE, seed = 12L, verbose = FALSE)$syn[[1]]
  expect_false(all(sm$y %in% d$y))
  expect_gte(min(sm$y), min(d$y))
  expect_lte(max(sm$y), max(d$y))
})


test_that("NA is reproduced rather than filled", {
  set.seed(41)
  n <- 100L
  d <- data.frame(x = rnorm(n), f = factor(sample(c("a", "b"), n, TRUE)),
                  y = rnorm(n))
  d$f[1:20] <- NA
  d$y[1:25] <- NA

  s <- tabfound_syn(d, m = 3L, models = stub_models(), seed = 13L,
                    verbose = FALSE)
  for (k in 1:3) {
    got <- s$syn[[k]]
    expect_true(anyNA(got$f))
    expect_true(anyNA(got$y))
    # The NA level must not leak into the factor's levels.
    expect_identical(levels(got$f), levels(d$f))
    expect_true(all(stats::na.omit(as.character(got$f)) %in% c("a", "b")))
    expect_true(all(is.finite(stats::na.omit(got$y))))
  }
  rate <- mean(vapply(s$syn, function(x) mean(is.na(x$y)), numeric(1)))
  expect_lt(abs(rate - 0.25), 0.15)
})


test_that("cont_na models a point mass instead of smearing it", {
  set.seed(42)
  n <- 150L
  d <- data.frame(x = rnorm(n),
                  inc = c(rep(0, 50), round(rlnorm(100), 3)))

  s <- tabfound_syn(d, m = 2L, models = stub_models(), cont_na = list(inc = 0),
                    seed = 14L, verbose = FALSE)
  zeros <- vapply(s$syn, function(x) mean(x$inc == 0), numeric(1))
  expect_true(all(zeros > 0.15))
  expect_true(all(zeros < 0.55))

  # Without it the spike is just a region of a continuous distribution and
  # nothing lands exactly on it. (`clamp` has to be off to see that: the
  # spike sits at the observed minimum, and clamping piles every draw
  # below it onto exactly that value, which is a boundary artefact rather
  # than a modelled point mass.)
  s2 <- tabfound_syn(d, m = 1L, models = stub_models(), clamp = FALSE,
                     seed = 14L, verbose = FALSE)
  expect_lt(mean(s2$syn[[1]]$inc == 0), 0.05)

  expect_error(tabfound_syn(d, models = stub_models(), cont_na = list(nope = 0)),
               "not in the data")
  expect_error(tabfound_syn(d, models = stub_models(), cont_na = 0),
               "named list")
})


test_that("factor levels mean the same thing in the context and the query", {
  # The highest-probability silent bug in a synthesis pass: coding the
  # factor independently in the real predictors and the synthetic ones
  # makes the synthetic covariates address different categories than the
  # model was fitted on. A mismatch here would flip or flatten the gap.
  set.seed(51)
  n <- 400L
  g <- factor(sample(c("lo", "hi"), n, replace = TRUE), levels = c("lo", "hi"))
  d <- data.frame(g = g, y = ifelse(g == "hi", 5, 0) + rnorm(n))

  s <- tabfound_syn(d, m = 1L, models = stub_models(), seed = 15L,
                    verbose = FALSE)$syn[[1]]
  real_gap <- mean(d$y[d$g == "hi"]) - mean(d$y[d$g == "lo"])
  syn_gap  <- mean(s$y[s$g == "hi"]) - mean(s$y[s$g == "lo"])
  expect_gt(syn_gap, 4)
  expect_lt(abs(syn_gap - real_gap), 0.6)
})


test_that("degenerate columns do not derail the pass", {
  set.seed(61)
  n <- 50L
  d <- data.frame(
    x     = rnorm(n),
    const = rep(3.5, n),
    cfac  = factor(rep("only", n), levels = c("only", "unused")),
    one   = rnorm(n)
  )
  s <- tabfound_syn(d, m = 2L, models = stub_models(), seed = 16L,
                    verbose = FALSE)
  expect_true(all(s$syn[[1]]$const == 3.5))
  expect_true(all(as.character(s$syn[[1]]$cfac) == "only"))
  # An unused level is part of the column's type and must survive.
  expect_identical(levels(s$syn[[1]]$cfac), c("only", "unused"))
  expect_false(anyNA(s$syn[[1]]))

  # A single predictor, and a single column overall.
  s2 <- tabfound_syn(d[, c("x", "one")], m = 1L, models = stub_models(),
                     seed = 17L, verbose = FALSE)
  expect_identical(ncol(s2$syn[[1]]), 2L)
  s3 <- tabfound_syn(d[, "x", drop = FALSE], m = 1L, models = stub_models(),
                     seed = 18L, verbose = FALSE)
  expect_true(all(s3$syn[[1]]$x %in% d$x))

  expect_error(tabfound_syn(d[0, ], models = stub_models()), "empty")
  expect_error(tabfound_syn(d, m = 0L, models = stub_models()),
               "positive integer")
})


test_that("proper and improper both run, and improper shares one fit", {
  d <- syn_test_df(n = 60L)
  imp <- tabfound_syn(d, m = 3L, models = stub_models(), proper = FALSE,
                      seed = 19L, verbose = FALSE)
  pro <- tabfound_syn(d, m = 3L, models = stub_models(), proper = TRUE,
                      seed = 19L, verbose = FALSE)
  expect_false(imp$proper)
  expect_true(pro$proper)
  expect_length(imp$syn, 3L)
  for (k in 1:3) expect_identical(dim(imp$syn[[k]]), dim(d))
  # Batching the m queries behind one fit must not make them identical:
  # the draws are still independent.
  expect_false(isTRUE(all.equal(imp$syn[[1]], imp$syn[[2]])))
})


test_that("a matrix comes back as a data frame", {
  set.seed(71)
  mx <- matrix(rnorm(150), ncol = 3, dimnames = list(NULL, c("a", "b", "c")))
  s <- tabfound_syn(mx, m = 2L, models = stub_models(), seed = 20L,
                    verbose = FALSE)
  expect_s3_class(s$syn[[1]], "data.frame")
  expect_identical(names(s$syn[[1]]), c("a", "b", "c"))
})


test_that("models load lazily, and only the ones the data needs", {
  set.seed(81)
  d <- data.frame(x = rnorm(30), y = rnorm(30))
  mods <- tabfound_models(regressor = stub_model("regression"))
  s <- tabfound_syn(d, m = 1L, models = mods, seed = 21L, verbose = FALSE)
  expect_false(anyNA(s$syn[[1]]))
  expect_identical(mods$loaded(), "regression")

  d2 <- d
  d2$f <- factor(sample(c("a", "b"), 30, replace = TRUE))
  expect_error(tabfound_syn(d2, m = 1L, models = mods, verbose = FALSE),
               "No classification model")
})


test_that("tabfound_synthetic pulls the frames out", {
  d <- syn_test_df(n = 30L)
  s <- tabfound_syn(d, m = 2L, models = stub_models(), seed = 22L,
                    verbose = FALSE)

  expect_equal(tabfound_synthetic(s, 1), s$syn[[1]])
  expect_length(tabfound_synthetic(s, "all"), 2L)
  expect_identical(names(tabfound_synthetic(s, "all")), c("syn1", "syn2"))

  long <- tabfound_synthetic(s, "long")
  expect_identical(nrow(long), nrow(d) * 2L)
  expect_true(".syn" %in% names(long))
  expect_setequal(unique(long$.syn), 1:2)

  stacked <- tabfound_synthetic(s, "stacked")
  expect_identical(names(stacked), names(d))

  expect_error(tabfound_synthetic(s, 99), "must be one of")
  expect_error(tabfound_synthetic(d), "tabfound_syn")
})


test_that("the pass transmits a conditional relationship, not just a marginal", {
  # The statistical counterpart of the type checks above. The stub
  # regressor is a correctly specified linear model with normal draws, so
  # synthesising a linear DGP one variable at a time must reproduce its
  # regression coefficients. A pass that conditioned on the wrong rows,
  # coded factors inconsistently, or drew means instead of samples would
  # still pass every structural test and fail this one.
  set.seed(91)
  n  <- 500L
  x1 <- rnorm(n)
  x2 <- 0.5 * x1 + rnorm(n)
  d  <- data.frame(x1 = x1, x2 = x2, y = 1 + 2 * x1 - x2 + rnorm(n))

  s <- tabfound_syn(d, m = 5L, models = stub_models(), seed = 23L,
                    verbose = FALSE)
  b <- rowMeans(vapply(s$syn, function(x) stats::coef(stats::lm(y ~ x1 + x2, x)),
                       numeric(3)))
  expect_lt(abs(b[["x1"]] - 2), 0.2)
  expect_lt(abs(b[["x2"]] + 1), 0.2)
  # The x1/x2 dependence has to survive too, or the joint is wrong even
  # where every conditional looks right.
  r <- mean(vapply(s$syn, function(x) stats::cor(x$x1, x$x2), numeric(1)))
  expect_lt(abs(r - stats::cor(d$x1, d$x2)), 0.12)
})


# ---------------------------------------------------------------------------
# synthpop
# ---------------------------------------------------------------------------

test_that("as_synds produces a synds synthpop can work with", {
  skip_if_not_installed("synthpop")
  set.seed(101)
  n <- 120L
  d <- data.frame(g = factor(sample(c("a", "b"), n, replace = TRUE)),
                  x = rnorm(n), y = rnorm(n))
  s <- tabfound_syn(d, m = 3L, models = stub_models(), seed = 24L,
                    verbose = FALSE)
  sds <- as_synds(s)

  expect_s3_class(sds, "synds")
  expect_identical(sds$m, 3L)
  expect_identical(sds$k, s$k)
  expect_identical(sds$n, s$n)
  expect_true(sds$proper)
  expect_identical(unname(sds$method[["g"]]), "sample")
  expect_identical(unname(sds$method[c("x", "y")]), rep("tabfound", 2L))
  expect_equal(sds$syn[[2]], s$syn[[2]])
  expect_identical(unname(sds$visit.sequence), 1:3)

  # m = 1 puts a bare data frame in `syn`, which is the shape synthpop's
  # own functions branch on.
  s1 <- tabfound_syn(d, m = 1L, models = stub_models(), seed = 25L,
                     verbose = FALSE)
  expect_s3_class(as_synds(s1)$syn, "data.frame")

  fit <- synthpop::lm.synds(y ~ x + g, as_synds(s))
  expect_s3_class(fit, "fit.synds")
  est <- summary(fit)
  expect_true(all(is.finite(est$coefficients[, 1])))
})


test_that("synthpop's generics take a tabfound_syn object directly", {
  skip_if_not_installed("synthpop")
  set.seed(111)
  n <- 120L
  d <- data.frame(g = factor(sample(c("a", "b"), n, replace = TRUE)),
                  x = rnorm(n), y = rnorm(n))
  s <- tabfound_syn(d, m = 2L, models = stub_models(), seed = 26L,
                    verbose = FALSE)

  cmp <- synthpop::compare(s, d, plot = FALSE, print.flag = FALSE)
  expect_s3_class(cmp, "compare.synds")

  u <- synthpop::utility.gen(s, d, print.flag = FALSE)
  expect_true(all(is.finite(u$pMSE)))

  ut <- synthpop::utility.tab(s, d, vars = "g", print.flag = FALSE)
  expect_true(all(is.finite(ut$pMSE)))
})


test_that("synthpop can drive the pass through syn.tabfound", {
  skip_if_not_installed("synthpop")
  set.seed(121)
  n <- 120L
  d <- data.frame(a = rnorm(n),
                  b = factor(sample(c("p", "q", "r"), n, replace = TRUE)),
                  c = rnorm(n),
                  e = sample(1:9, n, replace = TRUE))

  withr::local_options(tabfound.models = stub_models())
  sds <- synthpop::syn(d, method = "tabfound", m = 2L, print.flag = FALSE,
                       seed = 27L)

  expect_s3_class(sds, "synds")
  expect_identical(unname(sds$method[c("b", "c", "e")]), rep("tabfound", 3L))
  got <- sds$syn[[1]]
  expect_identical(names(got), names(d))
  expect_s3_class(got$b, "factor")
  expect_identical(levels(got$b), levels(d$b))
  expect_type(got$e, "integer")
  expect_false(anyNA(got))
  expect_s3_class(synthpop::compare(sds, d, plot = FALSE, print.flag = FALSE),
                  "compare.synds")
})


test_that("syn.tabfound is tuned through the option synthpop leaves open", {
  skip_if_not_installed("synthpop")
  set.seed(131)
  n <- 120L
  d <- data.frame(a = rnorm(n), b = round(rnorm(n), 2))
  withr::local_options(tabfound.models = stub_models())

  withr::local_options(tabfound.syn = list(draw_mode = "pmm", donors = 3L))
  sds <- synthpop::syn(d, method = "tabfound", m = 1L, proper = TRUE,
                       print.flag = FALSE, seed = 28L)
  expect_true(all(sds$syn$b %in% d$b))

  withr::local_options(tabfound.syn = list(draw_mode = "rank"))
  sds2 <- synthpop::syn(d, method = "tabfound", m = 1L, print.flag = FALSE,
                        seed = 29L)
  expect_identical(sort(sds2$syn$b), sort(d$b))

  withr::local_options(tabfound.syn = list(nope = 1))
  expect_error(synthpop::syn(d, method = "tabfound", m = 1L,
                             print.flag = FALSE),
               "Unknown .*option")
})


test_that("syn.tabfound says what to do when no models are set", {
  withr::local_options(tabfound.models = NULL, tabfound.syn = NULL)
  y <- rnorm(20)
  x <- data.frame(z = rnorm(20))
  expect_error(syn.tabfound(y, x, x), "No tabfound models")
})


test_that("syn.tabfound handles the degenerate cases synthpop hands it", {
  withr::local_options(tabfound.models = stub_models(), tabfound.syn = NULL)
  y <- rnorm(20)

  # No predictors at all: the marginal is the only honest draw.
  out <- syn.tabfound(y, NULL, matrix(0, 7L, 0L))
  expect_length(out$res, 7L)
  expect_true(all(out$res %in% y))
  expect_identical(out$fit, "sample")

  out2 <- syn.tabfound(rep(2, 20), data.frame(z = rnorm(20)),
                       data.frame(z = rnorm(5)))
  expect_identical(out2$res, rep(2, 5))
  expect_identical(out2$fit, "constant")

  # A factor level the query has never seen must not shift the coding.
  x  <- data.frame(g = factor(sample(c("a", "b", "c"), 20, replace = TRUE),
                              levels = c("a", "b", "c")))
  xp <- data.frame(g = factor(rep("c", 6), levels = c("c", "a", "b")))
  out3 <- syn.tabfound(y, x, xp)
  expect_length(out3$res, 6L)
  expect_true(all(is.finite(out3$res)))
})


test_that("print says something", {
  d <- syn_test_df(n = 20L)
  s <- tabfound_syn(d, m = 2L, models = stub_models(), seed = 30L,
                    verbose = FALSE)
  # cli writes to stderr, so capture that rather than stdout.
  expect_match(paste(capture.output(print(s), type = "message"), collapse = " "),
               "synthetic data")
})
