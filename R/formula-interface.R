# Formula / data-frame interface.
#
# `tabular_classifier()` and `tabular_regressor()` are the engine: they
# take a numeric matrix and hand it to a network. This file is the layer
# an R user actually wants on top --
#
#     fit <- tabfound(Species ~ ., data = iris, model = "...")
#     predict(fit, newdata, type = "prob")
#
# -- and it delegates the hard part of that (remember what the training
# data looked like, re-apply it to new data, complain precisely when new
# data does not match) to hardhat.
#
# The Python wrappers solve the same problem with a `TransformToNumerical`
# step that sniffs pandas dtypes to guess which columns are categorical.
# R does not need to guess: `is.factor()` and `inherits(x, "Date")` are
# exact. The encoder below dispatches on real types, and a hardhat
# blueprint carries the levels forward to predict time.

# ---------------------------------------------------------------------------
# Column coercion and encoding
# ---------------------------------------------------------------------------

# Applied to the data frame *before* mold() and to new data before
# forge(), so both paths see the same column types.
#
# Logicals and characters become factors. For logicals this is not
# cosmetic: the formula path runs a model.matrix expansion, which turns a
# bare logical into two dummy columns while leaving a factor alone under
# `indicators = "none"`. Coercing first keeps the formula and xy
# interfaces producing the same one-column-per-predictor result.
# @keywords internal
.coerce_for_mold <- function(data) {
  if (!is.data.frame(data)) return(data)
  for (j in seq_along(data)) {
    col <- data[[j]]
    if (is.logical(col)) {
      data[[j]] <- factor(col, levels = c(FALSE, TRUE))
    } else if (is.character(col)) {
      data[[j]] <- factor(col)
    }
  }
  data
}

# Turn molded predictors into the numeric matrix the networks take.
#
# Factors become 0-based ordinal codes, matching what all three reference
# implementations do (none of them one-hot encode features; the models
# learn column semantics from the column's own distribution). Dates and
# datetimes become their numeric representation. Everything else must
# already be numeric.
# @keywords internal
.encode_predictors <- function(predictors) {
  out <- vapply(predictors, function(col) {
    if (is.factor(col)) {
      as.numeric(as.integer(col) - 1L)
    } else if (inherits(col, c("Date", "POSIXct", "POSIXt", "difftime"))) {
      as.numeric(col)
    } else if (is.numeric(col) || is.logical(col)) {
      as.numeric(col)
    } else {
      cli::cli_abort(
        "Column of class {.cls {class(col)[1]}} cannot be used as a predictor."
      )
    }
  }, numeric(nrow(predictors)))
  if (!is.matrix(out)) out <- matrix(out, nrow = nrow(predictors))
  colnames(out) <- names(predictors)
  out
}

# Which encoded columns came from a factor.
#
# This is the one thing the Python wrappers have to guess at: they sniff
# pandas dtypes and fall back on a cardinality heuristic that only fires
# below four distinct values. R knows exactly, because `is.factor()` is
# exact -- so a two-level factor and a five-level one are both declared,
# not inferred. Logicals and characters have already been coerced to
# factors by `.coerce_predictor_types()` before this runs.
# @keywords internal
.categorical_predictor_indices <- function(predictors) {
  which(vapply(predictors, is.factor, logical(1)))
}


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

# Fit a simple imputer on the encoded training matrix: column means, as
# TabICL's preprocessing uses for numerics. This is *not* a port of any
# reference implementation's imputation and is not covered by the parity
# harness -- it exists so that a backend which cannot see NaN at all is
# still usable on real data frames.
#
# A factor's ordinal codes are not a scale to average over: the mean of a
# three-level factor is 0.83, which is not a level of anything. Those
# columns get the modal code instead, which is at least a value the
# variable can take.
# @keywords internal
.fit_imputer <- function(x, categorical = integer()) {
  vapply(seq_len(ncol(x)), function(j) {
    v <- x[, j]
    v <- v[is.finite(v)]
    if (!length(v)) return(0)
    if (j %in% categorical) {
      tab <- table(v)
      return(as.numeric(names(tab)[which.max(tab)]))
    }
    mean(v)
  }, numeric(1))
}

# @keywords internal
.apply_imputer <- function(x, means) {
  for (j in seq_len(ncol(x))) {
    bad <- !is.finite(x[, j])
    if (any(bad)) x[bad, j] <- means[[j]]
  }
  x
}

# "Handled" is two different guarantees, and `auto` reads their
# disjunction: a network that conditions on missingness and a predictor
# that mean-fills before the network both mean this layer need not
# impute. What they do not mean is the same thing to the caller -- see
# `?register_backend` -- which is why the flags are separate and
# `list_backends()` names which one applies.
# @keywords internal
.resolve_na_action <- function(na_action, backend_name, x) {
  spec <- get_backend(backend_name)
  covered <- .backend_covers_missing(spec)
  if (identical(na_action, "auto")) {
    na_action <- if (covered) "pass" else "impute"
  }
  if (identical(na_action, "pass") && !covered && anyNA(x)) {
    cli::cli_warn(c(
      "The {.val {backend_name}} backend has no missing-value handling.",
      x = "Predictions for rows with {.val NA} will be {.val NaN}.",
      i = "Use {.code na_action = \"impute\"} or impute before fitting."
    ))
  }
  na_action
}


# ---------------------------------------------------------------------------
# The user-facing fit function
# ---------------------------------------------------------------------------

#' Fit a tabular foundation model with a formula or data frame
#'
#' The R-facing interface: takes a formula and a data frame (or `x`/`y`
#' directly), works out whether the task is classification or regression
#' from the outcome's type, encodes predictors to the numeric matrix the
#' networks expect, and remembers enough to re-apply that encoding to new
#' data.
#'
#' @param x A formula, a data frame of predictors, or a matrix.
#' @param data When `x` is a formula, the data frame to evaluate it in.
#' @param y When `x` is a data frame or matrix, the outcome vector.
#' @param model Model artifacts: a local directory, a HuggingFace repo
#'   id, or a registered alias. See [list_backends()].
#' @param mode `"auto"` (default) infers classification from a factor,
#'   character or logical outcome and regression from a numeric one.
#' @param backend Optional backend name; inferred from the config when
#'   `NULL`.
#' @param device One of `"cpu"`, `"cuda"`, `"mps"`.
#' @param na_action What to do about missing predictors. `"auto"`
#'   (default) passes them through whenever the backend deals with them
#'   itself -- which every backend in the package currently does, by one
#'   of two different routes. TabPFN encodes missingness in the network,
#'   as an is-missing channel beside the value, so the model conditions
#'   on it; TabICL, Mitra and TabFM run their reference wrapper's mean
#'   imputer first, so the model never learns a value was missing.
#'   `list_backends()` names which. `"impute"` puts this package's own
#'   column-mean imputer in front instead (modal code for factors),
#'   `"fail"` refuses, and `"pass"` forces the data through as it is --
#'   which warns, and produces `NaN` predictions, on a backend that
#'   handles neither.
#' @param ... Passed to [tabular_classifier()] / [tabular_regressor()].
#' @return An object of class `tabfound_fit`.
#' @examples
#' \dontrun{
#' fit <- tabfound(Species ~ ., data = iris[1:100, ], model = "path/to/model")
#' predict(fit, iris[101:150, ], type = "prob")
#' }
#' @export
tabfound <- function(x, ...) {
  UseMethod("tabfound")
}

#' @rdname tabfound
#' @export
tabfound.formula <- function(x, data, model, mode = "auto", backend = NULL,
                             device = "cpu", na_action = "auto", ...) {
  require_suggested("hardhat")
  data <- .coerce_for_mold(data)
  # `indicators = "none"` keeps factors as single columns rather than
  # dummy-expanding them: these models take ordinal codes, and one-hot
  # would both widen the table and hide the column's identity.
  bp <- hardhat::default_formula_blueprint(indicators = "none",
                                           intercept = FALSE)
  processed <- hardhat::mold(x, data, blueprint = bp)
  .tabfound_bridge(processed, model, mode, backend, device, na_action, ...)
}

#' @rdname tabfound
#' @export
tabfound.data.frame <- function(x, y, model, mode = "auto", backend = NULL,
                                device = "cpu", na_action = "auto", ...) {
  require_suggested("hardhat")
  processed <- hardhat::mold(.coerce_for_mold(x), y)
  .tabfound_bridge(processed, model, mode, backend, device, na_action, ...)
}

#' @rdname tabfound
#' @export
tabfound.matrix <- function(x, y, model, mode = "auto", backend = NULL,
                            device = "cpu", na_action = "auto", ...) {
  require_suggested("hardhat")
  processed <- hardhat::mold(x, y)
  .tabfound_bridge(processed, model, mode, backend, device, na_action, ...)
}

#' @rdname tabfound
#' @export
tabfound.default <- function(x, ...) {
  cli::cli_abort(
    "{.fn tabfound} has no method for {.cls {class(x)[1]}}; \\
     supply a formula, data frame or matrix."
  )
}


# R's type system already says what kind of task this is, so there is
# nothing to configure by default: a factor, character or logical outcome
# is classification, a numeric one is regression.
# @keywords internal
.infer_mode <- function(outcome) {
  if (is.factor(outcome) || is.character(outcome) || is.logical(outcome)) {
    "classification"
  } else {
    "regression"
  }
}

# Everything the three entry points share, once hardhat has produced a
# `processed` list.
# @keywords internal
.tabfound_bridge <- function(processed, model, mode, backend, device,
                             na_action, ...) {
  hardhat::validate_outcomes_are_univariate(processed$outcomes)
  outcome <- processed$outcomes[[1]]

  mode <- match.arg(mode, c("auto", "classification", "regression"))
  if (identical(mode, "auto")) mode <- .infer_mode(outcome)
  na_action <- match.arg(na_action, c("auto", "pass", "impute", "fail"))

  x <- .encode_predictors(processed$predictors)

  # Hand the backend the columns we know are categorical, unless the
  # caller has already said. Nothing downstream can recover this: by the
  # time the matrix exists, a factor is just small integers, and the
  # reference's own cardinality heuristic would only catch the columns
  # with fewer than four levels.
  dots <- list(...)
  cat_idx <- unname(.categorical_predictor_indices(processed$predictors))
  if (!"categorical_features" %in% names(dots)) {
    dots$categorical_features <- cat_idx
  }
  ctor_args <- c(list(model, backend = backend, device = device), dots)

  if (identical(mode, "classification")) {
    y <- if (is.factor(outcome)) outcome else factor(outcome)
    obj <- do.call(tabular_classifier, ctor_args)
  } else {
    if (!is.numeric(outcome)) {
      cli::cli_abort("A regression outcome must be numeric, not {.cls {class(outcome)[1]}}.")
    }
    y <- as.numeric(outcome)
    obj <- do.call(tabular_regressor, ctor_args)
  }

  na_action <- .resolve_na_action(na_action, obj$backend, x)
  if (identical(na_action, "fail") && anyNA(x)) {
    cli::cli_abort("Missing values in predictors and {.code na_action = \"fail\"}.")
  }
  imputer <- NULL
  if (identical(na_action, "impute")) {
    imputer <- .fit_imputer(x, cat_idx)
    x <- .apply_imputer(x, imputer)
  }

  hardhat::new_model(
    inner     = fit(obj, x, y),
    mode      = mode,
    na_action = na_action,
    imputer   = imputer,
    blueprint = processed$blueprint,
    class     = "tabfound_fit"
  )
}


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

#' Predict from a formula-fitted tabular foundation model
#'
#' @param object A `tabfound_fit`.
#' @param new_data A data frame or matrix with the columns the model was
#'   fitted on. Factor levels and column types are validated against the
#'   training data and a clear error is raised if they do not match.
#' @param type For classification, `"class"` (default) or `"prob"`. For
#'   regression, `"mean"` (default), `"quantiles"`, `"grid"` or
#'   `"sample"`, subject to what the backend supports.
#' @param ... Passed to the backend, e.g. `quantiles`.
#' @return A tibble when tibble is available, otherwise a data frame:
#'   `.pred_class` and `.pred_<level>` for classification, `.pred` (plus
#'   `.pred_q<level>`) for regression.
#' @export
predict.tabfound_fit <- function(object, new_data, type = NULL, ...) {
  require_suggested("hardhat")
  type <- type %||% if (identical(object$mode, "classification")) "class" else "mean"

  forged <- hardhat::forge(.coerce_for_mold(new_data), object$blueprint)
  x <- .encode_predictors(forged$predictors)
  if (identical(object$na_action, "impute") && !is.null(object$imputer)) {
    x <- .apply_imputer(x, object$imputer)
  }

  out <- predict(object$inner, x, type = type, ...)
  .as_prediction_frame(out, type, object$mode, object$inner)
}


# Shape a backend's raw output into the column-naming convention R users
# expect from a `predict()` method.
# @keywords internal
.as_prediction_frame <- function(out, type, mode, inner) {
  as_frame <- function(l) {
    if (requireNamespace("tibble", quietly = TRUE)) tibble::as_tibble(l)
    else as.data.frame(l, stringsAsFactors = FALSE)
  }

  if (identical(mode, "classification")) {
    if (identical(type, "prob")) {
      cols <- as.list(as.data.frame(out))
      names(cols) <- paste0(".pred_", colnames(out))
      return(as_frame(cols))
    }
    levs <- inner$state$class_levels
    return(as_frame(list(.pred_class = factor(out, levels = as.character(levs)))))
  }

  if (identical(type, "mean")) {
    return(as_frame(list(.pred = as.numeric(out))))
  }
  # quantiles / grid / sample all come back as a matrix of one column per
  # level or draw.
  cols <- as.list(as.data.frame(out))
  nm <- colnames(out)
  names(cols) <- if (is.null(nm)) paste0(".pred_", seq_along(cols))
                 else paste0(".pred_", nm)
  as_frame(cols)
}


#' @export
print.tabfound_fit <- function(x, ...) {
  cli::cli_text("{.strong tabfound} fit <{x$inner$backend}>, mode {.val {x$mode}}")
  preds <- names(x$blueprint$ptypes$predictors)
  cli::cli_bullets(c(
    "*" = "{length(preds)} predictor{?s}: {.field {utils::head(preds, 6)}}\\
           {if (length(preds) > 6) ' ...' else ''}",
    "*" = "fitted on {.val {x$inner$state$n_train}} rows",
    "*" = "missing values: {.val {x$na_action}}"
  ))
  invisible(x)
}
