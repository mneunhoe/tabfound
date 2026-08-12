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
# Between-imputation variability ought to carry *parameter* uncertainty,
# not just predictive noise. A PFN draws each query row independently
# given a fixed context, so `m` chains conditioned on the same observed
# rows differ only in the sampling noise of the draw -- which is the
# defect that makes an imputer improper under Rubin's rules, and the one
# `?tabfound_syn` corrects by bootstrapping the context.
#
# `proper =` offers the same correction here, and the simulation in
# `inst/simulation/` says not to take it by default. For a parametric
# imputer the bootstrap does what the textbook says: coverage moves to
# nominal, intervals widen ~11%, bias is unchanged. For a PFN it costs
# an order of magnitude in bias (-0.008 -> -0.062 on the coefficient of
# interest) and nearly doubles the intervals, because a bootstrap context
# is not the same model with different parameters -- it is a *worse
# model*. 63% of its rows are distinct, and for a learner whose entire
# fit is its context, that is 37% of the training data thrown away, with
# ties the network never saw in training.
#
# Hence `proper = FALSE` by default, against the theory and with the
# numbers: coverage bought that way is bought with width, not accuracy.
# `proper = TRUE` remains available and is the right choice if nominal
# coverage matters more than the point estimate. See ?tabfound_impute.
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

  # Keyed on the task *and* the categorical declaration: chained equations
  # change predictor sets from one variable to the next, so which columns
  # are categorical changes with them. The weights are read once per task
  # regardless -- the per-declaration entries are predictor closures around
  # the same network. See `.respec_categoricals()`.
  get_one <- function(task, categorical_features = NULL) {
    key <- if (length(categorical_features))
             paste0(task, "|", paste(sort(as.integer(categorical_features)),
                                     collapse = ","))
           else task
    if (exists(key, envir = cache, inherits = FALSE)) {
      return(get(key, envir = cache, inherits = FALSE))
    }
    if (!identical(key, task)) {
      obj <- .respec_categoricals(get_one(task), categorical_features)
      assign(key, obj, envir = cache)
      return(obj)
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
         # One entry per task, whatever categorical declarations
         # were derived from it.
         loaded = function() unique(sub("\\|.*$", "", ls(cache))),
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
#
# `rule = 2` clamps: a uniform below the lowest predicted level returns
# that level's value rather than extrapolating. With the default grid of
# 199 levels that puts an atom of mass 1/200 at each end of every row's
# support -- the tails are truncated, not modelled. `draw = "sample"`
# (TabPFN's bar distribution) has no such grid and is the route to prefer
# when the backend offers it.
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

  # A backend handed NA in the predictors returns a NaN probability row,
  # and a NaN row walks through `cumsum` into `lab[NA]` and deposits
  # NA_character_ in a cell the caller was told had been imputed. The
  # numeric sibling has this guard; without it here the categorical path
  # silently un-completes the data.
  bad <- !stats::complete.cases(p) | rowSums(p) <= 0
  if (any(bad)) {
    cli::cli_warn(c(
      "{sum(bad)} non-finite probability row{?s}; drawn from the observed \\
       category frequencies instead.",
      i = "This usually means the backend saw {.val NA} in the predictors."
    ))
    freq <- as.numeric(table(y_obs)[lab])
    freq[!is.finite(freq)] <- 0
    p[bad, ] <- rep(freq, each = sum(bad))
  }

  # Inverse-CDF on the categorical: one uniform per row against that
  # row's cumulative probabilities.
  cs  <- t(apply(p, 1L, cumsum))
  if (!is.matrix(cs) || ncol(p) == 1L) cs <- matrix(cs, nrow = n)
  u   <- stats::runif(n) * cs[, ncol(cs)]
  lab[pmin(1L + rowSums(cs < u), ncol(p))]
}


# ---------------------------------------------------------------------------
# The chained-equations loop
# ---------------------------------------------------------------------------

# Which observed rows this univariate fit conditions on.
#
# The properness machinery, in one place. Conditioning on the observed
# rows as they are ("none") gives every one of the `m` chains the same
# context, so they differ only by the noise of the draw: between-
# imputation variance carries no parameter uncertainty and Rubin's rules
# undercover. Resampling the context is the nonparametric stand-in for
# drawing new parameters -- the ordinary bootstrap, or the Bayesian
# bootstrap's Dirichlet(1, ..., 1) weights, which vary the context more
# smoothly (no row is ever dropped outright, and none is duplicated as
# hard).
# @keywords internal
.mi_context_rows <- function(obs_idx, proper) {
  n <- length(obs_idx)
  if (identical(proper, "none") || n < 2L) return(obs_idx)
  w <- if (identical(proper, "bayes")) {
    g <- stats::rexp(n)
    if (sum(g) <= 0) NULL else g / sum(g)
  } else NULL
  obs_idx[sample.int(n, n, replace = TRUE, prob = w)]
}

# One variable, one sweep. `comp` holds the current completed data in
# model representation, `mdf` the original (with its NAs) so the fit
# always uses genuinely observed rows.
# @keywords internal
.mi_impute_one <- function(v, comp, mdf, idx, pm, meth, models, draw,
                           quantile_grid, pools, ctx_rows = NULL) {
  n_mis <- sum(idx)
  marginal <- function() sample(pools[[v]], n_mis, replace = TRUE)
  if (identical(meth[[v]], "sample")) return(marginal())

  obs <- !is.na(mdf[[v]])
  preds <- setdiff(names(which(pm[v, ] == 1L)), v)
  # Nothing to condition on, or nothing to learn from: fall back to the
  # observed marginal, which is still a valid (if uninformative) draw.
  if (!length(preds) || sum(obs) < 2L) return(marginal())

  pred_frame <- comp[, preds, drop = FALSE]
  X <- .encode_predictors(pred_frame)
  # R knows which of *these* predictors are factors, and the set changes
  # from one variable to the next -- which is why the declaration cannot
  # be a fixed vector on the model and is computed per draw instead.
  cat_ix <- unname(.categorical_predictor_indices(pred_frame))
  # Drawn once per imputation, not once per sweep: the whole chain then
  # runs against one context, which is the textbook proper structure
  # (draw the parameters, then sample) and what `syn.tabfound()` does.
  ctx   <- ctx_rows[[v]] %||% which(obs)
  X_obs <- X[ctx, , drop = FALSE]
  X_mis <- X[idx, , drop = FALSE]

  if (identical(meth[[v]], "classification")) {
    return(mi_draw_factor(models$get("classification", cat_ix), X_obs,
                          mdf[[v]][ctx], X_mis))
  }
  out <- mi_draw_numeric(models$get("regression", cat_ix), X_obs,
                         as.numeric(mdf[[v]][ctx]), X_mis,
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
#' @section Properness:
#' A PFN draws each query row independently given a fixed context, so `m`
#' chains that condition on the same observed rows differ only in the
#' noise of the draw. That is an *improper* imputer in Rubin's sense: the
#' between-imputation variance carries no parameter uncertainty, and
#' pooled standard errors run small. `proper = TRUE` applies the usual
#' correction -- resample the observed rows once per imputation, the
#' nonparametric stand-in for drawing new parameters, which is what
#' [tabfound_syn()] does by default and what mice's `norm.boot` does.
#'
#' It is **off** by default here, which is not what the theory says, and
#' the reason is measured rather than argued. On the MAR simulation in
#' `inst/simulation/` (200 replications, `n = 400`, ~42% missing), the
#' correction behaves for a correctly specified parametric imputer
#' exactly as advertised: coverage on the fully observed covariate moves
#' from 0.910 to 0.950, intervals widen 11%, bias does not move. Applied
#' to TabPFN it moves coverage the same way and costs an order of
#' magnitude in bias -- −0.008 to −0.062 on the coefficient of interest,
#' against a complete-case bias of −0.111 -- while widening intervals by
#' up to 92%.
#'
#' The reason is that a bootstrap of the context is not the same model
#' with different parameters. It is a *worse model*: only 63% of its rows
#' are distinct (50% under `proper = "bayes"`), and for a learner whose
#' entire fit is its context that is a third of the training data thrown
#' away, with ties the network never saw during pre-training. The draws
#' come back no wider but less informative -- correlation with the values
#' that were deleted drops -- which attenuates every coefficient. Coverage
#' improves because the intervals grow faster than the error does.
#'
#' So: leave it off to estimate, turn it on if you need nominal coverage
#' more than you need the point estimate, and read
#' `inst/simulation/README.md` before deciding. Whether properness can be
#' had for an in-context learner without degrading the context is an open
#' question -- every resampling scheme duplicates rows.
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
#'
#'   `"quantile"` and `"grid"` invert a finite grid of predicted
#'   quantiles and clamp outside it, so the extreme levels carry an atom
#'   of mass and the tails are truncated. `"sample"` has no grid.
#' @param quantile_grid Number of quantile levels for `draw = "quantile"`.
#' @param proper Resample the context once per imputation rather than
#'   conditioning on the observed rows as they are. `FALSE` (default) is
#'   no resampling, `TRUE` a bootstrap, `"bayes"` a Bayesian bootstrap.
#'   The default is against the theory and with the measurements; read
#'   *Properness* before changing it.
#' @param where Optional logical matrix, same shape as `data`, marking
#'   the cells to impute. Defaults to the missing ones -- mice's `where`.
#'   Marking an *observed* cell overimputes it, which is how you check a
#'   model against values you already have.
#' @param post Optional named list of functions, one per column, applied
#'   to each variable's drawn values before they go back into the
#'   completed data -- the place to squeeze a draw into a plausible range
#'   or enforce a constraint. mice's `post` is a string evaluated inside
#'   its sampler; a function is the same idea without the reach-in.
#' @param seed Optional integer seed.
#' @param verbose Show a progress bar.
#' @param ... Passed to [tabfound_models()] when `models` is `NULL`.
#' @return An object of class `tabfound_mi`:
#'   * `imputations` — list of `m` completed data frames
#'   * `data` — the input, unchanged
#'   * `where` — logical matrix of imputed cells
#'   * `chain_mean`, `chain_var` — per-sweep summaries of the imputed
#'     cells, `variable x iteration x imputation`, which [as_mids()]
#'     hands to mice so `plot()` has a convergence trace to draw
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
                            quantile_grid = 199L, proper = FALSE,
                            where = NULL, post = NULL, seed = NULL,
                            verbose = TRUE, ...) {
  cl     <- match.call()
  draw   <- match.arg(draw)
  proper <- .mi_resolve_proper(proper)
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
  where <- .mi_resolve_where(where, na_mask, vars)
  post  <- .mi_resolve_post(post, vars)
  nmis <- colSums(na_mask)

  plans <- lapply(vars, function(v) .mi_column_plan(data[[v]], v))
  names(plans) <- vars
  kind  <- vapply(plans, `[[`, character(1), "kind")

  meth <- ifelse(colSums(where) > 0L,
                 ifelse(kind == "categorical", "classification", "regression"),
                 "")
  names(meth) <- vars
  if (!is.null(method)) meth <- .mi_resolve_method(method, meth, kind, nmis)

  pm <- .mi_resolve_predictors(predictors, vars)

  visit <- vars[colSums(where) > 0L & nzchar(meth)]
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
  # Per-sweep summaries of the imputed cells, in mice's own layout
  # (variable x iteration x imputation). Without them `plot()` on the
  # converted `mids` draws an empty frame, which is a poor look for a
  # method whose whole loop is worth watching converge.
  chain_mean <- array(NA_real_, c(length(vars), maxit, m),
                      dimnames = list(vars, NULL, NULL))
  chain_var <- chain_mean
  if (verbose && length(visit)) {
    cli::cli_progress_bar("Imputing", total = m * maxit * length(visit),
                          .envir = environment())
  }
  for (j in seq_len(m)) {
    comp <- mdf
    # One context per imputation, held across every sweep: this is the
    # "draw the parameters, then sample" structure that makes an imputer
    # proper. With `proper = "none"` it is just the observed rows.
    ctx_rows <- lapply(stats::setNames(visit, visit), function(v) {
      .mi_context_rows(which(!is.na(mdf[[v]])), proper)
    })
    # Start from the observed marginal, so the first sweep conditions on
    # something plausible rather than on NA.
    for (v in visit) {
      idx <- where[, v]
      comp[[v]][idx] <- sample(pools[[v]], sum(idx), replace = TRUE)
    }
    for (s in seq_len(maxit)) {
      for (v in visit) {
        idx <- where[, v]
        drawn <- .mi_impute_one(v, comp, mdf, idx, pm, meth, models,
                                draw, quantile_grid, pools, ctx_rows)
        if (!is.null(post[[v]])) drawn <- .mi_apply_post(post[[v]], drawn, v)
        comp[[v]][idx] <- drawn
        num <- .mi_as_numeric(comp[[v]][idx])
        chain_mean[v, s, j] <- mean(num)
        chain_var[v, s, j]  <- stats::var(num)
        if (verbose) cli::cli_progress_update(.envir = environment())
      }
    }
    imps[[j]] <- .mi_check_complete(.mi_restore(comp, data, plans, where),
                                    where, visit, j)
  }
  if (verbose && length(visit)) cli::cli_progress_done(.envir = environment())

  structure(
    list(data = data, imputations = imps, m = m, maxit = maxit,
         method = meth, where = where, nmis = nmis, visit_sequence = visit,
         chain_mean = chain_mean, chain_var = chain_var,
         predictors = pm, draw = draw, proper = proper, seed = seed,
         backend = c(classification = models$source[["classification"]],
                     regression     = models$source[["regression"]]),
         call = cl),
    class = "tabfound_mi"
  )
}

# Which cells to draw. mice's `where`: a logical matrix over the whole
# frame, defaulting to the missing cells. Asking for a cell that is
# *observed* is legitimate and useful -- it is how you overimpute to
# check a model against values you already have -- but the fit still
# conditions on genuinely observed rows, so an all-TRUE column would
# leave nothing to learn from and is refused.
# @keywords internal
.mi_resolve_where <- function(where, na_mask, vars) {
  if (is.null(where)) return(na_mask)
  if (is.data.frame(where)) where <- as.matrix(where)
  if (!is.matrix(where) || !is.logical(where)) {
    cli::cli_abort("{.arg where} must be a logical matrix, \\
                    the same shape as {.arg data}.")
  }
  if (!identical(dim(where), dim(na_mask))) {
    cli::cli_abort(c(
      "{.arg where} is {nrow(where)} x {ncol(where)}; {.arg data} is \\
       {nrow(na_mask)} x {ncol(na_mask)}.",
      i = "Same layout as mice's {.field where}: one cell per cell."
    ))
  }
  colnames(where) <- vars
  where[is.na(where)] <- FALSE
  full <- vars[colSums(where) == nrow(where) & colSums(na_mask) < nrow(na_mask)]
  if (length(full)) {
    cli::cli_warn(c(
      "{.arg where} asks for every row of {.field {full}}.",
      i = "The model still fits on the rows where the variable is \\
           observed, so those cells are drawn from a model that has seen \\
           them -- an overimputation diagnostic, not an imputation."
    ))
  }
  where
}

# mice's `post` is a string evaluated in the sampler's own frame, which
# only makes sense inside mice. Ours is a function of the drawn vector:
# same purpose (squeeze into a range, round, enforce a constraint),
# testable on its own, and it cannot reach into the loop's internals.
# @keywords internal
.mi_resolve_post <- function(post, vars) {
  if (is.null(post)) return(stats::setNames(vector("list", length(vars)), vars))
  if (!is.list(post) || is.null(names(post))) {
    cli::cli_abort("{.arg post} must be a named list of functions, \\
                    one per column to post-process.")
  }
  unknown <- setdiff(names(post), vars)
  if (length(unknown)) {
    cli::cli_abort("{.arg post} names column{?s} not in the data: \\
                    {.val {unknown}}.")
  }
  bad <- names(post)[!vapply(post, is.function, logical(1))]
  if (length(bad)) {
    cli::cli_abort("{.arg post} entr{?y/ies} {.field {bad}} {?is/are} not \\
                    {?a function/functions}.")
  }
  out <- stats::setNames(vector("list", length(vars)), vars)
  out[names(post)] <- post
  out
}

# @keywords internal
.mi_apply_post <- function(fn, drawn, v) {
  out <- fn(drawn)
  if (length(out) != length(drawn)) {
    cli::cli_abort(c(
      "The {.arg post} function for {.field {v}} returned \\
       {length(out)} value{?s} for {length(drawn)} cell{?s}.",
      i = "It must return one value per drawn cell, in order."
    ))
  }
  out
}

# Chain diagnostics are numeric summaries, and a categorical draw is a
# factor or a label. mice records the codes; so does this.
# @keywords internal
.mi_as_numeric <- function(x) {
  if (is.factor(x)) return(as.numeric(x))
  if (is.character(x)) return(as.numeric(as.factor(x)))
  if (is.logical(x)) return(as.numeric(x))
  as.numeric(x)
}

# `proper` is one knob with three settings, so it takes the logical a
# user reaches for first and the string for the variant.
# @keywords internal
.mi_resolve_proper <- function(proper) {
  if (isTRUE(proper))  return("bootstrap")
  if (isFALSE(proper)) return("none")
  if (is.character(proper) && length(proper) == 1L &&
      proper %in% c("bootstrap", "bayes", "none")) {
    return(proper)
  }
  cli::cli_abort(c(
    "{.arg proper} must be {.val {TRUE}}, {.val {FALSE}}, or one of \\
     {.val {c('bootstrap', 'bayes', 'none')}}.",
    x = "Got {.val {proper}}."
  ))
}

# The promise this function makes is that every cell it reports in
# `where` comes back filled. A guard that fires here is a bug in a draw,
# not a user error -- but a data set with NA where the caller was told
# there is none is worse than a loud stop.
# @keywords internal
.mi_check_complete <- function(completed, na_mask, visit, j) {
  left <- vapply(visit, function(v) sum(is.na(completed[[v]][na_mask[, v]])),
                 integer(1))
  if (any(left > 0L)) {
    v <- names(left)[left > 0L]
    cli::cli_abort(c(
      "Imputation {j} left {sum(left)} cell{?s} missing in \\
       {.field {v}} after every sweep.",
      i = "This is a bug in {.pkg tabfound}; please report it with the \\
           backend and a description of the data."
    ))
  }
  completed
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
    "*" = "draw: {.val {x$draw}}, context: {.val {x$proper %||% 'none'}}",
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
