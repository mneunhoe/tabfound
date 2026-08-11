# A model object with the same shape as a real backend's, built out of
# base R instead of 600 MB of weights.
#
# `tabfound_models()` accepts an already-constructed model object, and
# nothing in the MI code touches the network -- it goes through `fit()`
# and `predict()` only. So the chained-equations loop, the draws, and the
# mice / Amelia bridges can all be tested end-to-end, on any machine,
# with no checkpoint present.

stub_regressor_spec <- function() {
  fit_fn <- function(X, y) {
    X <- cbind(1, as.matrix(X)); storage.mode(X) <- "double"
    X[!is.finite(X)] <- 0
    y <- as.numeric(y)
    b <- qr.coef(qr(X), y)
    b[is.na(b)] <- 0
    resid <- as.numeric(y - X %*% b)
    list(b = b, sigma = max(stats::sd(resid), 1e-6), n_train = nrow(X))
  }
  predict_fn <- function(state, newdata, type = "mean",
                         quantiles = c(0.1, 0.5, 0.9),
                         n_samples = 1L, seed = NULL, ...) {
    X <- cbind(1, as.matrix(newdata)); storage.mode(X) <- "double"
    X[!is.finite(X)] <- 0
    mu <- as.numeric(X %*% state$b)
    if (type == "mean") return(mu)
    if (type == "quantiles") {
      out <- outer(mu, stats::qnorm(quantiles) * state$sigma, `+`)
      colnames(out) <- paste0("q", quantiles)
      return(out)
    }
    if (!is.null(seed)) set.seed(seed)
    matrix(stats::rnorm(length(mu) * n_samples, rep(mu, n_samples), state$sigma),
           ncol = as.integer(n_samples))
  }
  list(fit = fit_fn, predict = predict_fn,
       types = c("mean", "quantiles", "sample"))
}

# Predicts the observed class frequencies, ignoring the predictors. Dull
# on purpose: it makes the draws' distribution something a test can state
# exactly, and the point here is the plumbing, not the model.
stub_classifier_spec <- function() {
  fit_fn <- function(X, y) {
    y <- droplevels(as.factor(y))
    list(y = y, class_levels = levels(y), n_train = nrow(as.matrix(X)))
  }
  predict_fn <- function(state, newdata, type = "class", ...) {
    n <- nrow(as.matrix(newdata))
    p <- as.numeric(table(state$y)) / length(state$y)
    probs <- matrix(p, nrow = n, ncol = length(p), byrow = TRUE,
                    dimnames = list(NULL, as.character(state$class_levels)))
    if (type == "prob") return(probs)
    state$class_levels[max.col(probs, ties.method = "first")]
  }
  list(fit = fit_fn, predict = predict_fn)
}

stub_model <- function(task = c("classification", "regression")) {
  task <- match.arg(task)
  spec <- if (task == "classification") stub_classifier_spec()
          else stub_regressor_spec()
  structure(
    list(spec = spec, state = NULL, model = NULL, config = NULL,
         device = "cpu", backend = "stub", task = task,
         model_ref = list(model = "stub", backend = "stub", device = "cpu",
                          args = list())),
    class = c(if (task == "classification") "tabfound_classifier"
              else "tabfound_regressor",
              "tabfound_model")
  )
}

stub_models <- function() {
  tabfound_models(classifier = stub_model("classification"),
                  regressor  = stub_model("regression"))
}

# A small frame with one column of every type the imputer has to handle.
mi_test_df <- function(n = 40L, seed = 7L) {
  set.seed(seed)
  d <- data.frame(
    num = rnorm(n),
    int = sample(1:20, n, replace = TRUE),
    fac = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    lgl = sample(c(TRUE, FALSE), n, replace = TRUE),
    chr = sample(c("x", "y"), n, replace = TRUE),
    bin = as.numeric(sample(0:1, n, replace = TRUE)),
    ok  = rnorm(n),
    stringsAsFactors = FALSE
  )
  for (v in c("num", "int", "fac", "lgl", "chr", "bin")) {
    d[[v]][sample(n, 6L)] <- NA
  }
  d
}
