# User-facing API.
#
# A model object is a plain list with two halves:
#
#   spec   immutable -- the loaded network, its config, and the pure
#          functions that fit and predict with it
#   state  the fitted context: the training rows, class levels, target
#          scaler. Plain R data, or NULL before fitting.
#
# Nothing here holds mutable state. `fit()` returns a *new* object, so
# `m2 <- fit(m1, X, y)` leaves `m1` untouched, which is what every other
# R modelling function does. The network itself is shared rather than
# copied -- it is read-only and can be several gigabytes.

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

.new_tabfound_model <- function(ctx, spec, task, args) {
  structure(
    list(
      spec      = spec,
      state     = NULL,
      model     = ctx$net,
      config    = ctx$config,
      device    = ctx$device,
      backend   = ctx$backend$name,
      task      = task,
      # Enough to rebuild this object from disk; see `tabfound_save()`.
      model_ref = list(model = ctx$model_ref, backend = ctx$backend$name,
                       device = ctx$device, args = args)
    ),
    class = c(if (task == "classification") "tabfound_classifier"
              else "tabfound_regressor",
              "tabfound_model")
  )
}

# Assemble the argument list for a backend's predictor constructor.
#
# `categorical_features` is offered to every backend but declared by only
# some. Dropping it silently for the others would be wrong when the caller
# meant it, so a non-empty declaration that nobody can honour says so.
# @keywords internal
.predictor_args <- function(fn, ctx, categorical_features, args) {
  if (!is.null(categorical_features) &&
      !"categorical_features" %in% names(formals(fn))) {
    if (length(categorical_features)) {
      cli::cli_warn(c(
        "The {.val {ctx$backend$name}} backend has no categorical handling.",
        i = "{length(categorical_features)} declared categorical \
             column{?s} will be treated as plain numbers."
      ))
    }
    categorical_features <- NULL
  }
  c(list(ctx),
    if (!is.null(categorical_features))
      list(categorical_features = categorical_features),
    args)
}

#' Load a tabular foundation model for classification
#'
#' @param model Local directory of model artifacts, a HuggingFace repo
#'   id, or a registered alias (see [list_backends()]).
#' @param backend Optional backend name. Inferred from `config.json`
#'   when `NULL`.
#' @param device One of `"cpu"`, `"cuda"`, `"mps"`.
#' @param categorical_features Integer vector of 1-based column indices to
#'   treat as categorical, or `NULL` to let the backend infer. Declaring
#'   is much stronger than inferring: inference only catches columns with
#'   fewer than four distinct values, so an ordinal-coded factor with five
#'   levels is invisible to it. [tabfound()] fills this in from the frame's
#'   factor columns automatically.
#' @param ... Backend-specific arguments, e.g. `ensemble_configs_dir` or
#'   `softmax_temperature`.
#' @return An unfitted object of class `tabfound_classifier`. Fit it with
#'   [fit()], which returns a fitted copy.
#' @examples
#' \dontrun{
#' clf <- tabular_classifier("path/to/tabpfn-v2.5-clf")
#' clf <- fit(clf, iris[1:100, 1:4], iris[1:100, 5])
#' predict(clf, iris[101:150, 1:4], type = "prob")
#' }
#' @export
tabular_classifier <- function(model, backend = NULL, device = "cpu",
                               categorical_features = NULL, ...) {
  ctx <- load_backend_model(model, task = "classification",
                            backend = backend, device = device)
  if (is.null(ctx$backend$classifier)) {
    cli::cli_abort("Backend {.val {ctx$backend$name}} does not provide a classifier.")
  }
  args <- .predictor_args(ctx$backend$classifier, ctx, categorical_features,
                          list(...))
  .new_tabfound_model(ctx, do.call(ctx$backend$classifier, args),
                      "classification", args[-1L])
}

#' Load a tabular foundation model for regression
#'
#' @inheritParams tabular_classifier
#' @return An unfitted object of class `tabfound_regressor`.
#' @export
tabular_regressor <- function(model, backend = NULL, device = "cpu",
                              categorical_features = NULL, ...) {
  ctx <- load_backend_model(model, task = "regression",
                            backend = backend, device = device)
  if (is.null(ctx$backend$regressor)) {
    cli::cli_abort("Backend {.val {ctx$backend$name}} does not provide a regressor.")
  }
  args <- .predictor_args(ctx$backend$regressor, ctx, categorical_features,
                          list(...))
  .new_tabfound_model(ctx, do.call(ctx$backend$regressor, args),
                      "regression", args[-1L])
}


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------

# A torch module survives `saveRDS()` structurally but its tensors become
# dangling external pointers, so the failure only shows up in use, several
# frames deep, as "external pointer is not valid". Probe cheaply and say
# something actionable instead.
# @keywords internal
.check_weights_alive <- function(object) {
  if (is.null(object$model)) return(invisible(TRUE))
  p <- tryCatch(object$model$parameters[[1]], error = function(e) NULL)
  if (is.null(p)) return(invisible(TRUE))
  ok <- tryCatch({ p$size(); TRUE }, error = function(e) FALSE)
  if (!ok) {
    cli::cli_abort(c(
      "This model's weights are no longer available.",
      x = "Its tensors are dangling pointers, which is what {.fn saveRDS} \\
           leaves behind: it writes the object's shape but not the weights.",
      i = "Use {.fn tabfound_save} and {.fn tabfound_load} instead."
    ))
  }
  invisible(TRUE)
}

#' Has this model been fitted?
#' @param object A `tabfound_model`.
#' @export
is_fitted <- function(object) UseMethod("is_fitted")

#' @export
is_fitted.tabfound_model <- function(object) !is.null(object$state)

# @keywords internal
.require_fitted <- function(object) {
  if (!is_fitted(object)) {
    cli::cli_abort(c(
      "This model has not been fitted.",
      i = "Call {.code object <- fit(object, X, y)} first \\
           -- {.fn fit} returns a fitted copy rather than modifying in place."
    ))
  }
  .check_weights_alive(object)
  invisible(TRUE)
}


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------

#' Fit a tabular foundation model to a training context
#'
#' These models do no gradient training: fitting stores the training
#' rows, which are supplied as in-context examples at predict time.
#'
#' Returns a **new** object; the input is unchanged.
#'
#' @section Memory preflight:
#' Before anything is allocated, `fit()` and [predict()][predict.tabfound_classifier]
#' estimate the peak memory the fit-and-predict cycle will need and compare
#' it with what the machine has free. Running out inside libtorch kills the
#' R process outright -- no condition, no traceback, no output -- so there
#' is nothing to catch after the fact and the check has to come first.
#'
#' `options(tabfound.memory_guard =)` controls what happens when the
#' estimate does not fit:
#'
#' * `"warn"` (default) -- a warning naming the estimate, the available
#'   memory and the knobs that would help; the run proceeds.
#' * `"error"` -- refuse, with the same message.
#' * `"off"` -- silence.
#'
#' The guard is never the reason a run fails: a device it has no constants
#' for, a backend without a scaling formula, or a machine it cannot probe
#' all mean silence rather than a guess. It also says a given thing once,
#' so a chained-equations loop does not repeat itself. Call
#' [estimate_peak_memory()] for the full breakdown, or to ask the question
#' about dimensions you have not run yet.
#'
#' @param object A `tabfound_model`.
#' @param X Training features (matrix or data.frame).
#' @param y Training targets.
#' @param ... Unused.
#' @return A fitted copy of `object`.
#' @seealso [estimate_peak_memory()]
#' @export
fit <- function(object, X, y, ...) UseMethod("fit")

#' @export
fit.tabfound_model <- function(object, X, y, ...) {
  .check_weights_alive(object)
  # Checked here rather than per backend: the dimensions and the options
  # are the same question for all six, and a backend contributes only its
  # `peak_terms()` formula. `fit()` allocates almost nothing itself -- the
  # peak arrives at `predict()` -- so the check looks one call ahead, at
  # one chunk of query rows against this context.
  .memory_guard(object, n_context = NROW(X), n_query = .predict_chunk_of(object),
                n_features = NCOL(X), stage = "fit")
  # Assigning into a list copies it, so the caller's object is untouched.
  # The network is shared, not copied -- it is read-only and large.
  object$state <- object$spec$fit(X, y)
  object
}

#' Predict from a tabular foundation model
#'
#' @param object A fitted `tabfound_model`.
#' @param newdata Test features.
#' @param type For classifiers, `"class"` (default) or `"prob"`. For
#'   regressors, `"mean"` (default), `"median"`, `"quantiles"`, `"grid"`
#'   (the full predicted quantile grid, where the backend has one) or
#'   `"sample"`. Not every backend offers every one — a request for a
#'   type it lacks errors rather than being approximated, and the message
#'   lists what it does have. `"mean"` and `"median"` differ on any
#'   backend with a real predictive distribution: on TabICL the mean
#'   averages the whole 999-level grid.
#' @param ... Passed to the backend, e.g. `quantiles`, `n_samples`,
#'   `seed`.
#' @inheritSection fit Memory preflight
#' @seealso [estimate_peak_memory()]
#' @export
predict.tabfound_classifier <- function(object, newdata,
                                        type = c("class", "prob"), ...) {
  type <- match.arg(type)
  .require_fitted(object)
  .guard_predict(object, newdata)
  object$spec$predict(object$state, newdata, type, ...)
}

#' @rdname predict.tabfound_classifier
#' @export
predict.tabfound_regressor <- function(object, newdata,
                                       type = c("mean", "median", "quantiles",
                                                "grid", "sample"), ...) {
  type <- match.arg(type)
  .require_fitted(object)
  supported <- object$spec$types %||% c("mean", "quantiles")
  if (!type %in% supported) {
    cli::cli_abort(c(
      "The {.val {object$backend}} backend has no {.val {type}} prediction.",
      i = "Available: {.val {supported}}."
    ))
  }
  .guard_predict(object, newdata)
  object$spec$predict(object$state, newdata, type, ...)
}

# The fitted context is the other half of a prediction's dimensions, and
# it lives on the object rather than in the call.
# @keywords internal
.guard_predict <- function(object, newdata) {
  n_train <- object$state$n_train %||% NROW(object$state$X_train)
  .memory_guard(object, n_context = n_train %||% 0,
                n_query = NROW(newdata), n_features = NCOL(newdata),
                stage = "predict")
}

#' Predicted class probabilities
#'
#' Shorthand for `predict(object, newdata, type = "prob")`.
#' @param object A fitted `tabfound_classifier`.
#' @param newdata Test features.
#' @param ... Passed on.
#' @export
predict_proba <- function(object, newdata, ...) UseMethod("predict_proba")

#' @export
predict_proba.tabfound_classifier <- function(object, newdata, ...) {
  predict(object, newdata, type = "prob", ...)
}

#' Predicted quantiles
#'
#' Shorthand for `predict(object, newdata, type = "quantiles")`.
#' @param object A fitted `tabfound_regressor`.
#' @param newdata Test features.
#' @param quantiles Numeric vector in `(0, 1)`.
#' @param ... Passed on.
#' @export
predict_quantiles <- function(object, newdata, quantiles = c(0.1, 0.5, 0.9),
                              ...) UseMethod("predict_quantiles")

#' @export
predict_quantiles.tabfound_regressor <- function(object, newdata,
                                                 quantiles = c(0.1, 0.5, 0.9),
                                                 ...) {
  predict(object, newdata, type = "quantiles", quantiles = quantiles, ...)
}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

#' Save and reload a tabular foundation model
#'
#' `saveRDS()` does not work on these objects: a torch module survives
#' the round trip structurally but its tensors come back as dangling
#' pointers, and the failure only surfaces later, in use. These functions
#' are the supported path.
#'
#' What gets written is small — the fitted context (training rows, class
#' levels, target scaler) plus a *reference* to the model artifacts, not
#' the weights themselves. Fitting one of these models stores context
#' rows rather than learning parameters, so the weights on disk are
#' already the same file the object was built from; duplicating them
#' would mean writing 6.5 GB per saved TabFM model. [tabfound_load()]
#' re-resolves them from that reference, so the artifacts (or the
#' HuggingFace cache) must still be reachable.
#'
#' @param object A `tabfound_model`, fitted or not.
#' @param file Path to write to.
#' @return `file`, invisibly.
#' @examples
#' \dontrun{
#' clf <- fit(tabular_classifier("path/to/model"), X, y)
#' tabfound_save(clf, "clf.tabfound")
#' clf2 <- tabfound_load("clf.tabfound")
#' }
#' @export
tabfound_save <- function(object, file) {
  if (!inherits(object, "tabfound_model")) {
    cli::cli_abort("{.arg object} must be a {.cls tabfound_model}.")
  }
  saveRDS(
    list(
      format    = "tabfound-model",
      version   = 1L,
      task      = object$task,
      model_ref = object$model_ref,
      state     = object$state,
      pkg_version = as.character(utils::packageVersion("tabfound"))
    ),
    file
  )
  invisible(file)
}

#' @rdname tabfound_save
#' @param device Optional device override for the reloaded model.
#' @return For `tabfound_load()`, a `tabfound_model`.
#' @export
tabfound_load <- function(file, device = NULL) {
  blob <- readRDS(file)
  if (!identical(blob$format, "tabfound-model")) {
    cli::cli_abort(c(
      "{.path {file}} is not a saved tabfound model.",
      i = "It was probably written with {.fn saveRDS}, which cannot \\
           capture torch weights. Re-save with {.fn tabfound_save}."
    ))
  }
  ref <- blob$model_ref
  ctor <- if (identical(blob$task, "classification")) tabular_classifier
          else tabular_regressor
  obj <- do.call(ctor, c(
    list(model = ref$model, backend = ref$backend,
         device = device %||% ref$device),
    ref$args
  ))
  obj$state <- blob$state
  obj
}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

#' @export
print.tabfound_model <- function(x, ...) {
  kind <- if (inherits(x, "tabfound_classifier")) "classifier" else "regressor"
  cli::cli_text("{.strong tabfound} {kind} <{x$backend}>")

  n_par <- tryCatch(
    sum(vapply(x$model$parameters,
               function(p) prod(as.integer(p$size())), numeric(1))),
    error = function(e) NA_real_
  )
  bullets <- c(
    "*" = "device: {.val {x$device}}",
    "*" = if (is.na(n_par)) "weights: {.emph unavailable (see ?tabfound_save)}"
          else "params: {.val {n_par}}"
  )
  if (is_fitted(x)) {
    n <- x$state$n_train %||% NROW(x$state$X_train)
    bullets <- c(bullets, "v" = "fitted on {.val {n}} rows")
  } else {
    bullets <- c(bullets, "i" = "not fitted -- {.code object <- fit(object, X, y)}")
  }
  cli::cli_bullets(bullets)
  invisible(x)
}
