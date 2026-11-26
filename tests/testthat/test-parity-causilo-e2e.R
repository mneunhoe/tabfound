# End-to-end parity against `CausiloClassifier` / `CausiloRegressor`
# (`causilo` 1.5.0, CPU, float32, `random_state = 42`).
#
# The whole predictor: recipe, ensemble plan, network, member combination
# -- for 1, 2, 4, 5 and 8 members (views and trimming change with the
# count), on numeric, missing-value and categorical fixtures. Regenerate
# with `inst/parity/causilo_e2e_reference.py`. Skips unless
# TABFOUND_CAUSILO_CLF_DIR / TABFOUND_CAUSILO_REG_DIR point at the
# published artifacts.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

ref_dir <- tabfound_file("parity", "reference", "causilo_e2e")
skip_if(!nzchar(ref_dir), "no stored Causilo end-to-end reference")
clf_dir <- Sys.getenv("TABFOUND_CAUSILO_CLF_DIR", unset = "")
reg_dir <- Sys.getenv("TABFOUND_CAUSILO_REG_DIR", unset = "")
skip_if(!dir.exists(clf_dir) || !dir.exists(reg_dir),
        "Causilo 1.5 artifacts not configured")

ref  <- read_reference_tensors(file.path(ref_dir, "e2e.safetensors"))
meta <- jsonlite::fromJSON(file.path(ref_dir, "reference.json"),
                           simplifyVector = FALSE)
as_mat <- function(t) {
  m <- as.array(t$to(dtype = torch::torch_float64()))
  if (!is.matrix(m)) m <- matrix(m, ncol = 1L)
  m
}

ctx_clf <- suppressMessages(load_backend_model(clf_dir, "classification", "causilo"))
ctx_reg <- suppressMessages(load_backend_model(reg_dir, "regression", "causilo"))

# The network agrees with the reference to ~2e-6 in probability and
# ~7e-6 of scale in quantiles (`test-parity-causilo.R`); the members'
# inputs to a float32 ulp (`test-prep-causilo.R`).
TOL_PROBA  <- 1e-5
TOL_LOGITS <- 1e-4
TOL_REG    <- 1e-4   # relative to the target's standard deviation

for (nm in names(meta$cases)) {
  local({
    name <- nm
    cs <- meta$cases[[name]]
    test_that(paste0("tabfound matches causilo end to end on ", name), {
      p <- paste0(name, ".")
      g <- function(k) as_mat(ref[[paste0(p, k)]])
      X_train <- g("x_train"); X_test <- g("x_test")
      y <- as.numeric(g("y_train"))
      cats <- unlist(cs$categorical)
      if (identical(cs$task, "classification")) {
        spec <- causilo_classifier(ctx_clf, n_estimators = cs$n_estimators,
                                   categorical_features = cats)
        st <- spec$fit(X_train, factor(y))
        book <- ref[[paste0(p, "codebook")]]
        if (!is.null(book)) {
          # Above 10 classes the reference draws its codebook with NumPy's
          # PCG64, which tabfound does not replay: grade everything else on
          # the reference's own codebook.
          expect_identical(dim(st$recipe$codebook), as.integer(book$shape))
          st$recipe$codebook <- matrix(as.integer(as.array(book)), nrow = book$size(1L))
        }
        got <- spec$predict(st, X_test, type = "prob")
        want <- g("plain.proba")
        expect_lt(max(abs(got - want)), TOL_PROBA)
        if (is.null(book)) {
          lg <- spec$predict(st, X_test, type = "logits")
          expect_lt(max(abs(lg - g("plain.logits"))), TOL_LOGITS)
        } else {
          expect_error(spec$predict(st, X_test, type = "logits"), "no logits")
        }
      } else {
        spec <- causilo_regressor(ctx_reg, n_estimators = cs$n_estimators,
                                  categorical_features = cats)
        st <- spec$fit(X_train, y)
        s <- sd(y)
        chk <- function(got, want, what) {
          expect_identical(length(got), length(want), info = what)
          expect_lt(max(abs(got - want)) / s, TOL_REG, label = paste(name, what))
        }
        chk(spec$predict(st, X_test, type = "mean"), as.numeric(g("plain.mean")), "mean")
        chk(spec$predict(st, X_test, type = "median"), as.numeric(g("plain.median")), "median")
        chk(unname(spec$predict(st, X_test, type = "quantiles",
                                quantiles = unlist(meta$q_sym))), g("plain.q_sym"), "q_sym")
        chk(unname(spec$predict(st, X_test, type = "quantiles",
                                quantiles = unlist(meta$q_asym))), g("plain.q_asym"), "q_asym")
        chk(spec$predict(st, X_test, type = "grid"), g("plain.raw"), "grid")
      }
    })
  })
}

test_that("the public constructors reach the backend by family name", {
  withr::local_options(tabfound.download = "never")
  skip_if(!tabfound:::.model_is_downloaded("causilo-1.5-classifier") ||
            !tabfound:::.model_is_downloaded("causilo-1.5-regressor"),
          "Causilo 1.5 not in the store")
  clf <- suppressMessages(tabular_classifier("causilo", n_estimators = 2L))
  clf <- fit(clf, as.matrix(iris[, 1:4]), iris$Species)
  p <- predict(clf, as.matrix(iris[c(1, 51, 101), 1:4]), type = "prob")
  expect_identical(colnames(p), levels(iris$Species))
  expect_identical(as.character(predict(clf, as.matrix(iris[c(1, 51, 101), 1:4]))),
                   c("setosa", "versicolor", "virginica"))
  reg <- suppressMessages(tabular_regressor("causilo", light = TRUE))
  reg <- fit(reg, as.matrix(mtcars[, -1]), mtcars$mpg)
  mu <- predict(reg, as.matrix(mtcars[1:5, -1]))
  expect_lt(mean(abs(mu - mtcars$mpg[1:5])), 4)
  q <- predict(reg, as.matrix(mtcars[1:5, -1]), type = "quantiles", quantiles = c(0.1, 0.9))
  expect_true(all(q[, 1] <= q[, 2]))
  expect_error(suppressMessages(tabular_regressor("causilo", light = TRUE, n_estimators = 2L)),
               "not both")
})

test_that("tabfound() sorts factor levels by code point for Causilo", {
  # The reference's categories are strings; it codes them in sorted
  # order. Hand tabfound() the same labels as factors whose levels are in
  # some other order: the probabilities must be the reference's.
  name <- "clf_categorical_e8"
  skip_if(is.null(meta$cases[[name]]), "no categorical case")
  g <- function(k) as_mat(ref[[paste0(name, ".", k)]])
  X_train <- g("x_train"); X_test <- g("x_test"); y <- as.numeric(g("y_train"))
  cats <- unlist(meta$cases[[name]]$categorical)
  to_frame <- function(X) {
    df <- as.data.frame(X)
    for (j in cats) {
      lab <- ifelse(is.na(X[, j]), NA, sprintf("c%03d", as.integer(X[, j])))
      all <- sort(unique(stats::na.omit(c(sprintf("c%03d", as.integer(X_train[, j])),
                                          sprintf("c%03d", as.integer(X_test[, j]))))))
      df[[j]] <- factor(lab, levels = rev(all))
    }
    df
  }
  train <- to_frame(X_train); train$y <- factor(y)
  test <- to_frame(X_test)
  withr::local_options(tabfound.download = "never")
  fit <- suppressMessages(tabfound(y ~ ., data = train, model = clf_dir,
                                   backend = "causilo", n_estimators = 8L))
  got <- as.matrix(predict(fit, test, type = "prob"))
  expect_lt(max(abs(got - g("plain.proba"))), TOL_PROBA)
})

# The cache and the chunk sizes change batch shapes, never which rows see
# which: graded against the reference's plain predictions, they clear the
# same bounds. (The reference's own cached-vs-plain gap is recorded in
# `reference.json`: up to 1.3e-6 in probability, 7.4e-6 in quantiles.)
VARIANTS <- list(
  cached  = list(kv_cache = TRUE, predict_chunk_size = 5L),
  chunked = list(row_chunk_size = 16L, col_chunk_size = 2L, predict_chunk_size = 7L,
                 member_batch_size = 3L)
)
for (nm in c("clf_binary_missing_e8", "clf_categorical_e8", "reg_tiny_e5",
             "reg_categorical_e8", "clf_many_classes_e8")) {
  for (vn in names(VARIANTS)) {
    local({
      name <- nm; variant <- VARIANTS[[vn]]; label <- vn
      cs <- meta$cases[[name]]
      test_that(sprintf("the %s Causilo path matches causilo on %s", label, name), {
        p <- paste0(name, ".")
        g <- function(k) as_mat(ref[[paste0(p, k)]])
        X_train <- g("x_train"); X_test <- g("x_test")
        y <- as.numeric(g("y_train"))
        args <- c(list(n_estimators = cs$n_estimators,
                       categorical_features = unlist(cs$categorical)), variant)
        if (identical(cs$task, "classification")) {
          spec <- do.call(causilo_classifier, c(list(ctx_clf), args))
          st <- spec$fit(X_train, factor(y))
          book <- ref[[paste0(p, "codebook")]]
          if (!is.null(book)) {
            st$recipe$codebook <- matrix(as.integer(as.array(book)), nrow = book$size(1L))
          }
          got <- spec$predict(st, X_test, type = "prob")
          expect_lt(max(abs(got - g("plain.proba"))), TOL_PROBA)
        } else {
          spec <- do.call(causilo_regressor, c(list(ctx_reg), args))
          st <- spec$fit(X_train, y)
          expect_lt(max(abs(spec$predict(st, X_test, type = "grid") - g("plain.raw"))) / sd(y),
                    TOL_REG)
          expect_lt(max(abs(spec$predict(st, X_test) - as.numeric(g("plain.mean")))) / sd(y),
                    TOL_REG)
        }
      })
    })
  }
}

test_that("a Causilo cache from tabfound_cache() survives a bundle", {
  name <- "clf_binary_missing_e2"
  g <- function(k) as_mat(ref[[paste0(name, ".", k)]])
  m <- suppressMessages(tabular_classifier(clf_dir, backend = "causilo", n_estimators = 2L))
  m <- fit(m, g("x_train"), factor(as.numeric(g("y_train"))))
  plain <- predict(m, g("x_test"), type = "prob")
  cached <- tabfound_cache(m)
  expect_true(has_cache(cached))
  p1 <- predict(cached, g("x_test"), type = "prob")
  expect_lt(max(abs(p1 - g("plain.proba"))), TOL_PROBA)
  bundle <- file.path(withr::local_tempdir(), "causilo-bundle")
  tabfound_save(cached, bundle)
  back <- suppressMessages(tabfound_load(bundle))
  expect_true(has_cache(back))
  expect_identical(predict(back, g("x_test"), type = "prob"), p1)
})

test_that("Causilo's own codebook gives a sound many-class model", {
  name <- "clf_many_classes_e8"
  skip_if(is.null(meta$cases[[name]]), "no many-class case")
  g <- function(k) as_mat(ref[[paste0(name, ".", k)]])
  y <- factor(as.numeric(g("y_train")), levels = 0:15)
  spec <- causilo_classifier(ctx_clf)
  st <- spec$fit(g("x_train"), y)
  expect_identical(dim(st$recipe$codebook), c(8L, 15L))
  p <- spec$predict(st, g("x_test"), type = "prob")
  expect_identical(ncol(p), 16L)
  expect_equal(unname(rowSums(p)), rep(1, nrow(p)), tolerance = 1e-9)
  # The level absent from the training rows comes back at zero.
  absent <- setdiff(levels(y), as.character(unique(y)))
  expect_true(all(p[, absent] == 0))
  # Not the reference's codebook, so not its numbers; but the same model
  # class, so the same picture: the two agree on most argmaxes.
  want <- g("plain.proba")
  agree <- mean(max.col(p[, colnames(p) %in% unlist(meta$cases[[name]]$classes)]) ==
                  max.col(want))
  expect_gt(agree, 0.6)
})
