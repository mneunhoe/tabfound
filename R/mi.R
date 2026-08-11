# Multiple imputation by chained equations.
#
# The models in this package predict a *distribution*, not a point: the
# TabPFN regressor carries a bar-distribution head you can sample from,
# TabICL's regressor emits a full quantile grid, and every classifier
# emits a categorical distribution. That is exactly what proper multiple
# imputation needs, so this file wires it into the usual MICE loop:
#
#   1. fill every missing cell with a random draw from that column's
#      observed values;
#   2. sweep over the incomplete variables, refitting the model on the
#      rows where the variable is observed (other columns at their
#      current completed values) and drawing new values for the rows
#      where it is missing;
#   3. repeat `maxit` times, and the whole thing `m` times.
#
# Between-imputation variability comes from the posterior predictive: the
# in-context prior carries the epistemic uncertainty that a parametric
# imputer would get from drawing new coefficients. There is no bootstrap
# of the context rows -- that is a robustness variant, not the default.
#
# Nothing here is a port of a reference implementation, and none of it is
# covered by the parity harness. It is a use of the fitted models, not a
# claim about them.
#
# The result carries its own class and converts to the two objects the
# rest of the ecosystem pools from: `as_mids()` for mice and
# `as_amelia()` for Amelia. See R/mi-interop.R.


# ---------------------------------------------------------------------------
# Model handles
# ---------------------------------------------------------------------------

#' Model handles for multiple imputation
#'
#' Chained equations need a classifier for the categorical variables and
#' a regressor for the continuous ones. Most families ship those as two
#' separate checkpoints, and loading either one costs anywhere from a
#' second to a minute, so [tabfound_impute()] takes a *handle* that loads
#' each at most once, on first use, and reuses it for every variable,
#' sweep and imputation.
#'
#' A model that is never needed is never loaded: a data frame with no
#' incomplete factors will not touch the classifier.
#'
#' @param model Artifacts to use for both tasks: a local directory, a
#'   HuggingFace repo id, or a registered alias (see [list_backends()]).
#'   For families that ship one checkpoint per task -- TabPFN does --
#'   give `classifier` and `regressor` instead.
#' @param classifier,regressor Task-specific overrides. Either a model
#'   reference as above, or an already-constructed (and possibly already
#'   configured) [tabular_classifier()] / [tabular_regressor()] object.
#' @param device One of `"cpu"`, `"cuda"`, `"mps"`.
#' @param ... Passed to [tabular_classifier()] / [tabular_regressor()],
#'   e.g. `softmax_temperature`.
#' @return An object of class `tabfound_models`.
#' @examples
#' \dontrun{
#' mods <- tabfound_models(classifier = "path/to/tabpfn-v2.5-clf",
#'                         regressor  = "path/to/tabpfn-v2.5-reg")
#' imp <- tabfound_impute(airquality, m = 5, models = mods)
#' }
#' @export
tabfound_models <- function(model = NULL, classifier = NULL, regressor = NULL,
                            device = "cpu", ...) {
  classifier <- classifier %||% model
  regressor  <- regressor  %||% model
  if (is.null(classifier) && is.null(regressor)) {
    cli::cli_abort(c(
      "No model artifacts supplied.",
      i = "Pass {.arg model}, or {.arg classifier} / {.arg regressor} separately."
    ))
  }
  dots  <- list(...)
  cache <- new.env(parent = emptyenv())

  get_one <- function(task) {
    if (exists(task, envir = cache, inherits = FALSE)) {
      return(get(task, envir = cache, inherits = FALSE))
    }
    is_clf <- identical(task, "classification")
    spec <- if (is_clf) classifier else regressor
    if (is.null(spec)) {
      cli::cli_abort(c(
        "No {task} model was supplied.",
        x = "The data has {if (is_clf) 'categorical' else 'continuous'} \\
             variables to impute.",
        i = "Pass {.arg {if (is_clf) 'classifier' else 'regressor'}} to \\
             {.fn tabfound_models}."
      ))
    }
    want <- if (is_clf) "tabfound_classifier" else "tabfound_regressor"
    obj <- if (inherits(spec, "tabfound_model")) {
      if (!inherits(spec, want)) {
        cli::cli_abort("The {task} model must be a {.cls {want}}, \\
                        not a {.cls {class(spec)[1]}}.")
      }
      spec
    } else {
      ctor <- if (is_clf) tabular_classifier else tabular_regressor
      do.call(ctor, c(list(spec, device = device), dots))
    }
    assign(task, obj, envir = cache)
    obj
  }

  describe <- function(spec) {
    if (is.null(spec)) NA_character_
    else if (inherits(spec, "tabfound_model")) paste0("<", spec$backend, " object>")
    else as.character(spec)[1]
  }

  structure(
    list(get    = get_one,
         loaded = function() ls(cache),
         device = device,
         source = c(classification = describe(classifier),
                    regression     = describe(regressor))),
    class = "tabfound_models"
  )
}

#' @export
print.tabfound_models <- function(x, ...) {
  cli::cli_text("{.strong tabfound} model handle")
  loaded <- x$loaded()
  cli::cli_bullets(c(
    "*" = "classifier: {.val {x$source[['classification']]}}\\
           {if ('classification' %in% loaded) ' (loaded)' else ''}",
    "*" = "regressor: {.val {x$source[['regression']]}}\\
           {if ('regression' %in% loaded) ' (loaded)' else ''}",
    "*" = "device: {.val {x$device}}"
  ))
  invisible(x)
}

# Accept the several things a user might reasonably pass as `models`.
# @keywords internal
.as_models <- function(models, ...) {
  if (inherits(models, "tabfound_models")) return(models)
  if (is.null(models)) return(tabfound_models(...))
  if (inherits(models, "tabfound_classifier")) {
    return(tabfound_models(classifier = models, ...))
  }
  if (inherits(models, "tabfound_regressor")) {
    return(tabfound_models(regressor = models, ...))
  }
  tabfound_models(model = models, ...)
}


# ---------------------------------------------------------------------------
# Column plans
# ---------------------------------------------------------------------------

# Every column has to make two round trips: into the representation the
# models take (a factor for the classifier path, a double for the
# regressor path) and back into its original type. `to_model` and
# `restore` are those two halves; `kind` picks the head.
#
# Numeric 0/1 columns with missing values take the classifier path even
# though they are numeric. A continuous draw for a dummy is not a valid
# value of that variable, and it breaks any downstream binomial model;
# Amelia treats such columns as continuous, which is the behaviour this
# deliberately does not copy. They are mapped back to numeric 0/1 so the
# analysis model sees the type it started with.
# @keywords internal
.mi_column_plan <- function(col, name = "x") {
  if (is.factor(col)) {
    lv <- levels(col); ord <- is.ordered(col)
    return(list(kind = "categorical",
                to_model = function(x) x,
                restore  = function(x) factor(as.character(x), levels = lv,
                                              ordered = ord)))
  }
  if (is.character(col)) {
    lv <- sort(unique(col[!is.na(col)]))
    return(list(kind = "categorical",
                to_model = function(x) factor(x, levels = lv),
                restore  = function(x) as.character(x)))
  }
  if (is.logical(col)) {
    return(list(kind = "categorical",
                to_model = function(x) factor(x, levels = c("FALSE", "TRUE")),
                restore  = function(x) as.character(x) == "TRUE"))
  }
  if (inherits(col, "Date")) {
    return(list(kind = "continuous",
                to_model = as.numeric,
                restore  = function(x) structure(round(x), class = "Date")))
  }
  if (inherits(col, "POSIXct")) {
    tz <- attr(col, "tzone") %||% ""
    return(list(kind = "continuous",
                to_model = as.numeric,
                restore  = function(x) as.POSIXct(x, origin = "1970-01-01", tz = tz)))
  }
  if (is.numeric(col)) {
    obs <- col[!is.na(col)]
    if (anyNA(col) && length(obs) && all(obs %in% c(0, 1))) {
      return(list(kind = "categorical",
                  to_model = function(x) factor(x, levels = c("0", "1")),
                  restore  = function(x) as.numeric(as.character(x))))
    }
    if (is.integer(col)) {
      return(list(kind = "continuous",
                  to_model = as.numeric,
                  restore  = function(x) as.integer(round(x))))
    }
    return(list(kind = "continuous", to_model = as.numeric, restore = as.numeric))
  }
  cli::cli_abort("Column {.field {name}} of class {.cls {class(col)[1]}} \\
                  cannot be imputed.")
}


# ---------------------------------------------------------------------------
# The two univariate draws
# ---------------------------------------------------------------------------

# Backend sampling runs in torch's RNG, which R's `set.seed()` does not
# reach. Drawing the seed from R's stream instead makes one `set.seed()`
# reproduce the whole imputation.
# @keywords internal
.mi_seed <- function() sample.int(.Machine$integer.max, 1L)

# Inverse-CDF draw off a matrix of predicted quantiles: one uniform per
# row, interpolated against that row's quantile function.
# @keywords internal
.mi_icdf <- function(q, levels_) {
  u <- stats::runif(nrow(q))
  vapply(seq_len(nrow(q)), function(i) {
    stats::approx(levels_, q[i, ], xout = u[i], rule = 2)$y
  }, numeric(1))
}

#' Draw from a regressor's predictive distribution
#'
#' @param reg An unfitted [tabular_regressor()].
#' @param X_obs,y_obs Rows where the target is observed.
#' @param X_mis Rows to draw for.
#' @param draw How to turn the prediction into a draw; see
#'   [tabfound_impute()].
#' @param quantile_grid Number of levels for the `"quantile"` route.
#' @return A numeric vector, one draw per row of `X_mis`.
#' @keywords internal
mi_draw_numeric <- function(reg, X_obs, y_obs, X_mis, draw = "auto",
                            quantile_grid = 199L) {
  n <- nrow(X_mis)
  if (!n) return(numeric(0))
  if (length(unique(y_obs)) < 2L) return(rep(y_obs[1L], n))

  reg   <- fit(reg, X_obs, y_obs)
  types <- reg$spec$types %||% c("mean", "quantiles")
  mode  <- draw
  if (identical(mode, "auto")) {
    mode <- if ("sample" %in% types) "sample"
            else if ("grid" %in% types) "grid"
            else if ("quantiles" %in% types) "quantile"
            else "none"
  }
  if (identical(mode, "none")) {
    cli::cli_abort(c(
      "The {.val {reg$backend}} regressor has no predictive distribution \\
       to draw from.",
      x = "It supports only {.val {types}}.",
      i = "Use a backend that samples ({.val tabpfn}) or predicts a quantile \\
           grid ({.val tabicl}), or pass {.code draw = \"residual\"} to draw \\
           around the point prediction instead."
    ))
  }

  if (identical(mode, "sample")) {
    s <- predict(reg, X_mis, type = "sample", n_samples = 1L, seed = .mi_seed())
    return(as.numeric(s[, 1L]))
  }
  if (identical(mode, "residual")) {
    # Approximate: a normal draw around the point prediction, with the
    # spread taken from the in-sample residuals. It ignores everything
    # the predictive distribution knows about heteroskedasticity and
    # skew, which is why it is opt-in rather than a silent fallback.
    mu_obs <- as.numeric(predict(reg, X_obs, type = "mean"))
    sigma  <- stats::sd(y_obs - mu_obs)
    if (!is.finite(sigma)) sigma <- 0
    mu <- as.numeric(predict(reg, X_mis, type = "mean"))
    return(mu + stats::rnorm(n, 0, sigma))
  }
  if (identical(mode, "grid")) {
    q  <- predict(reg, X_mis, type = "grid")
    lv <- reg$spec$quantile_levels %||% (seq_len(ncol(q)) / (ncol(q) + 1))
  } else {
    lv <- seq_len(quantile_grid) / (quantile_grid + 1)
    q  <- predict(reg, X_mis, type = "quantiles", quantiles = lv)
  }
  .mi_icdf(q, lv)
}

#' Draw from a classifier's predicted category distribution
#'
#' One category per row, drawn with the predicted probabilities -- not
#' the modal class, which would understate between-imputation variance.
#'
#' @param clf An unfitted [tabular_classifier()].
#' @param X_obs,y_obs Rows where the target is observed; `y_obs` a factor.
#' @param X_mis Rows to draw for.
#' @return A character vector of level labels, one per row of `X_mis`.
#' @keywords internal
mi_draw_factor <- function(clf, X_obs, y_obs, X_mis) {
  n <- nrow(X_mis)
  if (!n) return(character(0))
  y_obs <- droplevels(as.factor(y_obs))
  lv    <- levels(y_obs)
  if (length(lv) < 2L) return(rep(lv[1L], n))

  clf <- fit(clf, X_obs, y_obs)
  p   <- predict(clf, X_mis, type = "prob")
  p   <- matrix(as.numeric(p), nrow = n, dimnames = list(NULL, colnames(p)))
  lab <- colnames(p) %||% lv
  # Inverse-CDF on the categorical: one uniform per row against that
  # row's cumulative probabilities.
  cs  <- t(apply(p, 1L, cumsum))
  u   <- stats::runif(n) * cs[, ncol(cs)]
  lab[pmin(1L + rowSums(cs < u), ncol(p))]
}


# ---------------------------------------------------------------------------
# The chained-equations loop
# ---------------------------------------------------------------------------

# One variable, one sweep. `comp` holds the current completed data in
# model representation, `mdf` the original (with its NAs) so the fit
# always uses genuinely observed rows.
# @keywords internal
.mi_impute_one <- function(v, comp, mdf, idx, pm, meth, models, draw,
                           quantile_grid, pools) {
  n_mis <- sum(idx)
  marginal <- function() sample(pools[[v]], n_mis, replace = TRUE)
  if (identical(meth[[v]], "sample")) return(marginal())

  obs <- !is.na(mdf[[v]])
  preds <- setdiff(names(which(pm[v, ] == 1L)), v)
  # Nothing to condition on, or nothing to learn from: fall back to the
  # observed marginal, which is still a valid (if uninformative) draw.
  if (!length(preds) || sum(obs) < 2L) return(marginal())

  X <- .encode_predictors(comp[, preds, drop = FALSE])
  X_obs <- X[obs, , drop = FALSE]
  X_mis <- X[idx, , drop = FALSE]

  if (identical(meth[[v]], "classification")) {
    return(mi_draw_factor(models$get("classification"), X_obs,
                          mdf[[v]][obs], X_mis))
  }
  out <- mi_draw_numeric(models$get("regression"), X_obs,
                         as.numeric(mdf[[v]][obs]), X_mis,
                         draw = draw, quantile_grid = quantile_grid)
  # A backend that cannot handle the missingness still in `comp` can
  # return NaN. Keep the chain alive and say so rather than propagating
  # it into every later sweep.
  bad <- !is.finite(out)
  if (any(bad)) {
    cli::cli_warn(c(
      "{sum(bad)} non-finite draw{?s} for {.field {v}}; \\
       filled from the observed values instead.",
      i = "This usually means the backend saw {.val NA} in the predictors."
    ))
    out[bad] <- sample(as.numeric(pools[[v]]), sum(bad), replace = TRUE)
  }
  out
}

# Model representation -> original types, with observed cells restored
# bit-for-bit from the input rather than round-tripped.
# @keywords internal
.mi_restore <- function(comp, orig, plans, na_mask) {
  out <- orig
  for (v in names(orig)) {
    if (!any(na_mask[, v])) next
    idx <- na_mask[, v]
    out[[v]][idx] <- plans[[v]]$restore(comp[[v]])[idx]
  }
  out
}

#' Multiple imputation with a tabular foundation model
#'
#' Chained equations, drawing each missing cell from the model's
#' predictive distribution: the regressor's sampled bar distribution (or
#' quantile grid) for continuous variables, the classifier's predicted
#' category probabilities for categorical ones. `m` completed data sets
#' come back in an object that converts to whatever your pooling
#' machinery expects -- [as_mids()] for mice, [as_amelia()] for Amelia --
#' or that you can pool from directly with [with.tabfound_mi()] and
#' `mice::pool()`.
#'
#' Column types decide the head, the way they do everywhere else in this
#' package: factors, characters and logicals go to the classifier,
#' numerics to the regressor. Numeric 0/1 columns with missing values are
#' the one exception — they take the classifier path and are mapped back
#' to 0/1, because a continuous draw is not a valid value of a dummy.
#'
#' Reproducibility: pass `seed`, or call `set.seed()` first. The backends
#' sample in torch's RNG, which `set.seed()` does not reach, so each draw
#' is handed a seed taken from R's stream.
#'
#' This is a *use* of the fitted models, not a port of anything. Unlike
#' the rest of the package it has no reference implementation to be
#' verified against, and it is not covered by the parity harness.
#'
#' @param data A data frame (or matrix) with missing values.
#' @param m Number of imputed data sets.
#' @param models A [tabfound_models()] handle. A model reference or a
#'   single fitted model object is accepted too and wrapped for you; when
#'   `NULL`, `...` is passed to [tabfound_models()], so
#'   `tabfound_impute(data, model = "...")` works.
#' @param maxit Number of chained-equations sweeps per imputation.
#' @param method Optional per-variable override of the automatic choice.
#'   A single string, or a named vector indexed by column name. One of
#'   `"regression"`, `"classification"`, `"sample"` (a bootstrap draw
#'   from the observed values, no model) or `""` (leave missing).
#' @param predictors Optional 0/1 predictor matrix, rows and columns both
#'   named by the columns of `data`, in the same layout as mice's
#'   `predictorMatrix`: `predictors[v, w] == 1` means `w` is used when
#'   imputing `v`. Defaults to every other column.
#' @param draw How to turn a regressor's prediction into a draw.
#'   `"auto"` picks the best the backend offers: `"sample"` (TabPFN's bar
#'   distribution), else `"grid"` (TabICL's quantile grid), else
#'   `"quantile"` (inverse-CDF off a fine grid of predicted quantiles).
#'   `"residual"` is an approximation for point-estimate-only backends —
#'   a normal draw around the prediction, sized by the in-sample
#'   residuals.
#' @param quantile_grid Number of quantile levels for `draw = "quantile"`.
#' @param seed Optional integer seed.
#' @param verbose Show a progress bar.
#' @param ... Passed to [tabfound_models()] when `models` is `NULL`.
#' @return An object of class `tabfound_mi`:
#'   * `imputations` — list of `m` completed data frames
#'   * `data` — the input, unchanged
#'   * `where` — logical matrix of imputed cells
#'   * `method`, `m`, `maxit`, `predictors`, `seed`, `call`
#' @seealso [as_mids()], [as_amelia()], [tabfound_complete()],
#'   [with.tabfound_mi()], [mice.impute.tabfound()]
#' @examples
#' \dontrun{
#' mods <- tabfound_models(classifier = "path/to/tabpfn-v2.5-clf",
#'                         regressor  = "path/to/tabpfn-v2.5-reg")
#' imp <- tabfound_impute(airquality, m = 5, models = mods, seed = 1)
#'
#' # Pool with mice.
#' fit <- with(imp, lm(Ozone ~ Wind + Temp))
#' summary(mice::pool(fit))
#'
#' # Or hand the whole thing to mice / Amelia.
#' mids <- as_mids(imp)
#' amp  <- as_amelia(imp)
#' }
#' @export
tabfound_impute <- function(data, m = 5L, models = NULL, maxit = 5L,
                            method = NULL, predictors = NULL,
                            draw = c("auto", "sample", "grid", "quantile",
                                     "residual"),
                            quantile_grid = 199L, seed = NULL,
                            verbose = TRUE, ...) {
  cl   <- match.call()
  draw <- match.arg(draw)
  if (!is.null(seed)) set.seed(seed)

  if (is.matrix(data)) data <- as.data.frame(data, stringsAsFactors = FALSE)
  if (!is.data.frame(data)) {
    cli::cli_abort("{.arg data} must be a data frame or a matrix, \\
                    not a {.cls {class(data)[1]}}.")
  }
  if (!nrow(data) || !ncol(data)) cli::cli_abort("{.arg data} is empty.")
  m     <- as.integer(m)
  maxit <- as.integer(maxit)
  if (is.na(m) || m < 1L) cli::cli_abort("{.arg m} must be a positive integer.")
  if (is.na(maxit) || maxit < 1L) {
    cli::cli_abort("{.arg maxit} must be a positive integer.")
  }
  models <- .as_models(models, ...)

  vars    <- names(data)
  na_mask <- is.na(data)
  # `is.na()` on a one-column data frame still gives a matrix, but be
  # explicit -- the code below indexes it as one throughout.
  if (!is.matrix(na_mask)) {
    na_mask <- matrix(na_mask, ncol = length(vars), dimnames = list(NULL, vars))
  }
  nmis <- colSums(na_mask)

  plans <- lapply(vars, function(v) .mi_column_plan(data[[v]], v))
  names(plans) <- vars
  kind  <- vapply(plans, `[[`, character(1), "kind")

  meth <- ifelse(nmis > 0L,
                 ifelse(kind == "categorical", "classification", "regression"),
                 "")
  names(meth) <- vars
  if (!is.null(method)) meth <- .mi_resolve_method(method, meth, kind, nmis)

  pm <- .mi_resolve_predictors(predictors, vars)

  visit <- vars[nmis > 0L & nzchar(meth)]
  if (!length(visit)) {
    cli::cli_alert_info("Nothing to impute: no missing values selected.")
  }

  mdf <- data
  for (v in vars) mdf[[v]] <- plans[[v]]$to_model(data[[v]])
  pools <- lapply(mdf, function(col) col[!is.na(col)])
  for (v in visit) {
    if (!length(pools[[v]])) {
      cli::cli_abort("{.field {v}} is missing in every row; nothing to impute from.")
    }
  }

  imps <- vector("list", m)
  if (verbose && length(visit)) {
    cli::cli_progress_bar("Imputing", total = m * maxit * length(visit),
                          .envir = environment())
  }
  for (j in seq_len(m)) {
    comp <- mdf
    # Start from the observed marginal, so the first sweep conditions on
    # something plausible rather than on NA.
    for (v in visit) {
      idx <- na_mask[, v]
      comp[[v]][idx] <- sample(pools[[v]], sum(idx), replace = TRUE)
    }
    for (s in seq_len(maxit)) {
      for (v in visit) {
        idx <- na_mask[, v]
        comp[[v]][idx] <- .mi_impute_one(v, comp, mdf, idx, pm, meth, models,
                                         draw, quantile_grid, pools)
        if (verbose) cli::cli_progress_update(.envir = environment())
      }
    }
    imps[[j]] <- .mi_restore(comp, data, plans, na_mask)
  }
  if (verbose && length(visit)) cli::cli_progress_done(.envir = environment())

  structure(
    list(data = data, imputations = imps, m = m, maxit = maxit,
         method = meth, where = na_mask, nmis = nmis, visit_sequence = visit,
         predictors = pm, draw = draw, seed = seed,
         backend = c(classification = models$source[["classification"]],
                     regression     = models$source[["regression"]]),
         call = cl),
    class = "tabfound_mi"
  )
}

# @keywords internal
.mi_resolve_method <- function(method, meth, kind, nmis) {
  ok <- c("regression", "classification", "sample", "")
  if (is.null(names(method)) && length(method) == 1L) {
    method <- stats::setNames(rep(method, sum(nmis > 0L)),
                              names(meth)[nmis > 0L])
  }
  if (is.null(names(method))) {
    cli::cli_abort("{.arg method} must be a single string or a named vector.")
  }
  unknown <- setdiff(names(method), names(meth))
  if (length(unknown)) {
    cli::cli_abort("{.arg method} names column{?s} not in the data: {.val {unknown}}.")
  }
  bad <- setdiff(unique(method), ok)
  if (length(bad)) {
    cli::cli_abort(c("Unknown method{?s} {.val {bad}}.",
                     i = "Use one of {.val {ok}}."))
  }
  for (v in names(method)) {
    want <- method[[v]]
    if (identical(want, "classification") && kind[[v]] != "categorical") {
      cli::cli_abort("{.field {v}} is continuous; it cannot be imputed by \\
                      classification.")
    }
    if (identical(want, "regression") && kind[[v]] == "categorical") {
      cli::cli_abort("{.field {v}} is categorical; it cannot be imputed by \\
                      regression.")
    }
    meth[[v]] <- want
  }
  meth
}

# @keywords internal
.mi_resolve_predictors <- function(predictors, vars) {
  pm <- matrix(1L, length(vars), length(vars), dimnames = list(vars, vars))
  diag(pm) <- 0L
  if (is.null(predictors)) return(pm)
  if (!is.matrix(predictors)) {
    cli::cli_abort("{.arg predictors} must be a matrix.")
  }
  rn <- rownames(predictors); cn <- colnames(predictors)
  if (is.null(rn) || is.null(cn) ||
      !setequal(rn, vars) || !setequal(cn, vars)) {
    cli::cli_abort(c(
      "{.arg predictors} must be a square matrix named by the columns of {.arg data}.",
      i = "Same layout as mice's {.field predictorMatrix}."
    ))
  }
  out <- pm
  out[vars, vars] <- as.integer(predictors[vars, vars] != 0)
  diag(out) <- 0L
  out
}


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

#' @export
print.tabfound_mi <- function(x, ...) {
  cli::cli_text("{.strong tabfound} multiple imputation")
  imputed <- x$visit_sequence
  cli::cli_bullets(c(
    "*" = "{.val {x$m}} imputation{?s}, {.val {x$maxit}} sweep{?s} each",
    "*" = "{length(imputed)} variable{?s} imputed: {.field {utils::head(imputed, 6)}}\\
           {if (length(imputed) > 6) ' ...' else ''}",
    "*" = "{sum(x$nmis)} missing cell{?s} of {nrow(x$data) * ncol(x$data)}",
    "*" = "draw: {.val {x$draw}}",
    i = "{.fn as_mids} for mice, {.fn as_amelia} for Amelia."
  ))
  invisible(x)
}

#' Extract completed data from a `tabfound_mi` object
#'
#' Same contract as `mice::complete()`, which this is also registered as
#' a method for -- so `mice::complete(imp, "long")` works on the object
#' directly, and `tabfound_complete()` is the route that needs no mice.
#'
#' @param x A `tabfound_mi` object.
#' @param action `1..m` for a single completed data set; `"all"` for a
#'   list of all of them; `"long"` for them stacked with `.imp` and `.id`
#'   columns; `"broad"` for them side by side.
#' @param include Include the original, incomplete data as `.imp == 0`.
#'   Only meaningful for `"long"`, `"broad"` and `"all"`.
#' @param ... Unused.
#' @return A data frame, or a list of them for `action = "all"`.
#' @examples
#' \dontrun{
#' tabfound_complete(imp, 1)        # first completed data set
#' tabfound_complete(imp, "long")   # stacked, with .imp / .id
#' }
#' @export
tabfound_complete <- function(x, action = 1L, include = FALSE, ...) {
  if (!inherits(x, "tabfound_mi")) {
    cli::cli_abort("{.arg x} must be a {.cls tabfound_mi} object.")
  }
  sets <- x$imputations
  if (isTRUE(include)) sets <- c(list(x$data), sets)
  imp_id <- if (isTRUE(include)) 0:(x$m) else seq_len(x$m)

  if (is.numeric(action)) {
    k <- as.integer(action)
    if (length(k) != 1L || !k %in% imp_id) {
      cli::cli_abort("{.arg action} must be one of {.val {imp_id}}.")
    }
    return(sets[[match(k, imp_id)]])
  }
  action <- match.arg(as.character(action),
                      c("all", "long", "broad", "stacked", "repeated"))
  if (action == "all") {
    names(sets) <- paste0("imp", imp_id)
    return(sets)
  }
  if (action %in% c("long", "stacked")) {
    ids <- rownames(x$data) %||% as.character(seq_len(nrow(x$data)))
    long <- do.call(rbind, lapply(seq_along(sets), function(i) {
      d <- sets[[i]]
      cbind(.imp = imp_id[[i]], .id = ids, d, stringsAsFactors = FALSE)
    }))
    rownames(long) <- NULL
    if (action == "stacked") long <- long[, setdiff(names(long), c(".imp", ".id"))]
    return(long)
  }
  # broad / repeated: the data sets side by side, suffixed by imputation.
  broad <- do.call(cbind, lapply(seq_along(sets), function(i) {
    d <- sets[[i]]
    names(d) <- paste0(names(d), ".", imp_id[[i]])
    d
  }))
  rownames(broad) <- rownames(x$data)
  broad
}

#' Run an analysis on every completed data set
#'
#' Evaluates `expr` once per imputation and returns the fits in a `mira`
#' object, mice's container for repeated analyses -- so `mice::pool()`
#' takes the result directly, without going through [as_mids()].
#'
#' @param data A `tabfound_mi` object.
#' @param expr An expression, evaluated with the completed data frame's
#'   columns in scope.
#' @param ... Unused.
#' @return An object of class `mira`.
#' @examples
#' \dontrun{
#' fit <- with(imp, lm(Ozone ~ Wind + Temp))
#' summary(mice::pool(fit))
#' }
#' @export
with.tabfound_mi <- function(data, expr, ...) {
  call <- match.call()
  expr <- substitute(expr)
  pf   <- parent.frame()
  analyses <- lapply(seq_len(data$m), function(j) {
    eval(expr, data$imputations[[j]], pf)
  })
  structure(list(call = call, call1 = data$call, nmis = data$nmis,
                 analyses = analyses),
            class = c("mira", "matrix"))
}
