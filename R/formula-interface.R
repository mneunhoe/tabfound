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
#
# `keep_character` names string columns to leave as strings: text
# candidates, which the frame expansion may turn into LSA features after
# mold. Made factors here, their level set would be baked into the
# blueprint, and every unseen sentence at predict time would be reported
# as a novel level and blanked.
.coerce_for_mold <- function(data, keep_character = character()) {
  if (!is.data.frame(data)) return(data)
  for (j in seq_along(data)) {
    col <- data[[j]]
    if (is.logical(col)) {
      data[[j]] <- factor(col, levels = c(FALSE, TRUE))
    } else if (is.character(col) && !names(data)[[j]] %in% keep_character) {
      data[[j]] <- factor(col, levels = .code_point_levels(col))
    }
  }
  data
}

# The level set sklearn's `OrdinalEncoder` would give a string column:
# its distinct values in code-point order. Plain `factor()` sorts by the
# session's collation instead, which puts "a" before "B" under
# `en_US.UTF-8` but after it in the C locale -- so the codes, and the
# category shuffles the ensemble draws from them, would depend on how R
# was started. `method = "radix"` always compares in the C locale, which
# on UTF-8 strings is code-point order.
# @keywords internal
.code_point_levels <- function(x) {
  sort(unique(x[!is.na(x)]), method = "radix")
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
    } else if (inherits(col, "difftime")) {
      # Seconds, whatever unit the object carries. Plain `as.numeric()`
      # returned the magnitude in its own units, so the same duration was
      # 1.5 or 90 depending on how it had been built.
      as.numeric(col, units = "secs")
    } else if (inherits(col, c("Date", "POSIXct", "POSIXt"))) {
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
#' @param transform_text,transform_dates,min_cardinality_for_text,text_n_components
#'   How text and date columns are expanded before they reach the model;
#'   see [tabular_classifier()]. `"auto"` follows the checkpoint's own
#'   recipe, which turns both on for TabPFN v3.5 and leaves them off
#'   elsewhere.
#' @param categorical_features Predictors to declare categorical, as names
#'   or as positions. Positions count the predictors in the order the
#'   formula lists them (for `y ~ .`, the order of `data`'s columns), or
#'   the columns of `x`. By default every factor, character and logical
#'   predictor is declared; a declaration replaces that, so list those
#'   columns too if they should stay declared.
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
                             device = "cpu", na_action = "auto",
                             transform_text = "auto", transform_dates = "auto",
                             min_cardinality_for_text = 30L,
                             text_n_components = 30L,
                             categorical_features = NULL, ...) {
  require_suggested("hardhat")
  opts <- .preprocess_options(transform_text, transform_dates,
                              min_cardinality_for_text, text_n_components)
  raw <- data
  keep <- .text_columns(raw, opts$min_cardinality_for_text)
  data <- .coerce_for_mold(data, keep_character = keep)
  # `indicators = "none"` keeps factors as single columns rather than
  # dummy-expanding them: these models take ordinal codes, and one-hot
  # would both widen the table and hide the column's identity.
  bp <- hardhat::default_formula_blueprint(indicators = "none",
                                           intercept = FALSE)
  processed <- hardhat::mold(x, data, blueprint = bp)
  .check_inline_factors(processed)
  # The molded frame puts numerics first and factors last, so its order is
  # not the formula's; a position a caller gives means the formula's.
  terms_order <- attr(stats::terms(x, data = data), "term.labels")
  .tabfound_bridge(processed, model, mode, backend, device, na_action,
                   opts = opts, raw = raw, keep_character = keep,
                   categorical_features = categorical_features,
                   predictor_order = terms_order, ...)
}

#' @rdname tabfound
#' @export
tabfound.data.frame <- function(x, y, model, mode = "auto", backend = NULL,
                                device = "cpu", na_action = "auto",
                                transform_text = "auto", transform_dates = "auto",
                                min_cardinality_for_text = 30L,
                                text_n_components = 30L,
                                categorical_features = NULL, ...) {
  require_suggested("hardhat")
  opts <- .preprocess_options(transform_text, transform_dates,
                              min_cardinality_for_text, text_n_components)
  keep <- .text_columns(x, opts$min_cardinality_for_text)
  processed <- hardhat::mold(.coerce_for_mold(x, keep_character = keep), y)
  .tabfound_bridge(processed, model, mode, backend, device, na_action,
                   opts = opts, raw = x, keep_character = keep,
                   categorical_features = categorical_features, ...)
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


# `indicators = "none"` keeps only factor *columns* whole. A term that makes
# a factor inside the formula -- `factor(k)`, `cut(x, 3)`, `I(x > 0)` -- is
# one dummy column per level to `model.matrix()`, which is the one-hot
# layout these models do not take, and at predict `forge()` builds dummies
# from the levels the new rows happen to have, so it fails on any batch
# without every one. Hardhat's predictor terms keep such a term and record
# what it evaluated to; factor columns of the data have been taken out of
# them, so whatever is left that is not numeric is the problem.
# @keywords internal
.check_inline_factors <- function(processed) {
  classes <- attr(processed$blueprint$terms$predictors, "dataClasses")
  bad <- names(classes)[classes %in% c("factor", "ordered", "logical", "character")]
  bad <- setdiff(bad, names(processed$predictors))
  if (!length(bad)) return(invisible(TRUE))
  cli::cli_abort(c(
    "Formula term{?s} {.code {bad}} make{?s/} a factor inside the formula, \\
     which would be one-hot encoded.",
    x = "These models take a factor as one column, and the dummies fail at \\
         {.fn predict} on new data that lacks one of the levels.",
    i = "Add the factor to {.arg data} as a column, and name the column in \\
         the formula."
  ))
}

# A `categorical_features` declaration as positions in the molded
# predictors, which is what the matrix the backend sees is laid out by.
# `order` is the caller's view of the predictors: the formula's terms, or
# the columns of `x`.
# @keywords internal
.declared_positions <- function(declared, order, molded) {
  # An explicit none stays none: it is what turns the backend's own
  # inference off.
  if (!length(declared)) return(integer())
  if (is.character(declared)) {
    # A non-syntactic name is backquoted in the terms, and in the molded
    # names after them; the caller may write it either way.
    hit <- match(declared, order)
    bare <- match(declared, gsub("^`|`$", "", order))
    hit[is.na(hit)] <- bare[is.na(hit)]
    if (anyNA(hit)) {
      unknown <- declared[is.na(hit)]
      cli::cli_abort(c(
        "{.arg categorical_features} names {?a column that is not a \\
         predictor/columns that are not predictors}: {.field {unknown}}.",
        i = "Predictors, in order: {.field {order}}."
      ))
    }
  } else if (is.numeric(declared) && !anyNA(declared) &&
             all(declared == round(declared))) {
    hit <- as.integer(declared)
    if (any(hit < 1L | hit > length(order))) {
      bad <- as.character(hit[hit < 1L | hit > length(order)])
      cli::cli_abort(c(
        "{.arg categorical_features} has position{?s} {bad} outside the \\
         predictors.",
        i = "Predictors, in order: {.field {order}}."
      ))
    }
  } else {
    cli::cli_abort(
      "{.arg categorical_features} must be predictor names or whole-number \\
       positions, not {.obj_type_friendly {declared}}."
    )
  }
  pos <- match(order[hit], molded)
  if (anyNA(pos)) {
    # A matrix-valued term such as `poly(x, 2)`: several columns, none of
    # them the term.
    wide <- order[hit][is.na(pos)]
    cli::cli_abort(
      "Term{?s} {.code {wide}} span{?s/} several columns, so cannot be \\
       declared categorical."
    )
  }
  unique(pos)
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
#
# `predictor_order` is the order the caller sees the predictors in, which
# positions in `categorical_features` count; only a formula's differs from
# the molded one.
.tabfound_bridge <- function(processed, model, mode, backend, device,
                             na_action, opts = .preprocess_options(),
                             raw = NULL, keep_character = character(),
                             categorical_features = NULL,
                             predictor_order = NULL, ...) {
  hardhat::validate_outcomes_are_univariate(processed$outcomes)
  outcome <- processed$outcomes[[1]]

  mode <- match.arg(mode, c("auto", "classification", "regression"))
  if (identical(mode, "auto")) mode <- .infer_mode(outcome)
  na_action <- match.arg(na_action, c("auto", "pass", "impute", "fail"))
  dots <- list(...)
  molded_names <- names(processed$predictors)
  predictor_order <- predictor_order %||% molded_names
  if (!is.null(categorical_features)) {
    dots$categorical_features <- .declared_positions(
      categorical_features, predictor_order, molded_names)
  }

  # Text and date expansion, on the molded predictors with their typed
  # columns put back (see `.restore_typed_columns()`). Done here rather
  # than inside `fit()` because which columns are categorical has to be
  # known before the model is constructed, and expansion decides where
  # they end up.
  predictors <- .restore_typed_columns(processed$predictors, raw,
                                       keep_character = keep_character)
  restored <- attr(predictors, "restored")
  attr(predictors, "restored") <- NULL
  expansion <- NULL
  if (.frame_needs_expansion(predictors, opts)) {
    # "auto" is decided by the checkpoint's recipe, read without loading
    # weights -- and only when there is something it would decide.
    cfg <- if (identical(opts$transform_text, "auto") ||
               identical(opts$transform_dates, "auto")) {
      .resolve_backend_config(model, mode, backend)$config
    } else NULL
    declared <- if (length(dots$categorical_features))
      names(predictors)[dots$categorical_features] else character()
    fitted <- fit_frame_expansion(
      predictors,
      transform_text  = .resolve_transform_flag(opts$transform_text, cfg,
                                                "TRANSFORM_TEXT", "transform_text"),
      transform_dates = .resolve_transform_flag(opts$transform_dates, cfg,
                                                "TRANSFORM_DATES", "transform_dates"),
      min_cardinality_for_text = opts$min_cardinality_for_text,
      text_n_components = opts$text_n_components,
      declared = declared
    )
    predictors <- fitted$data
    expansion <- fitted$state
    expansion$keep_character <- keep_character
    expansion$restored <- restored
    if (length(declared)) {
      dots$categorical_features <- .expanded_indices_of(declared, expansion)
    }
  }

  x <- .encode_predictors(predictors)

  # Hand the backend the columns we know are categorical, unless the
  # caller has already said. Nothing downstream can recover this: by the
  # time the matrix exists, a factor is just small integers, and the
  # reference's own cardinality heuristic would only catch the columns
  # with fewer than four levels. A frame with no factors has said nothing
  # about its numeric columns, so it declares nothing: an empty declaration
  # would turn the backend's inference off.
  cat_idx <- unname(.categorical_predictor_indices(predictors))
  if (!"categorical_features" %in% names(dots) && length(cat_idx)) {
    dots$categorical_features <- cat_idx
  }
  # The inner model gets the same options: a refit on the engine API should
  # decide what this did. It is handed an encoded matrix here, so it has
  # nothing to expand itself.
  ctor_args <- c(list(model, backend = backend, device = device), opts, dots)

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
    expansion = expansion,
    # What the model reads, in the caller's order. The blueprint's ptype
    # holds every column the formula referenced, `y ~ . - k`'s `k` included.
    predictor_names = predictor_order,
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
#'   regression, `"mean"` (default), `"median"`, `"quantiles"`, `"grid"`
#'   or `"sample"`, subject to what the backend supports.
#' @param ... Passed to the backend, e.g. `quantiles`.
#' @return A tibble when tibble is available, otherwise a data frame:
#'   `.pred_class` and `.pred_<level>` for classification. For regression,
#'   `.pred` for `"mean"` and `"median"`, `.pred_q<level>` for
#'   `"quantiles"` and `"grid"` (a repeated level gets a `_1`, `_2`, ...
#'   suffix), and `.pred_1`, `.pred_2`, ... for `"sample"`.
#' @export
predict.tabfound_fit <- function(object, new_data, type = NULL, ...) {
  require_suggested("hardhat")
  type <- type %||% if (identical(object$mode, "classification")) "class" else "mean"

  exp_state <- object$expansion
  keep <- exp_state$keep_character %||% character()
  forged <- hardhat::forge(.coerce_for_mold(new_data, keep_character = keep),
                           object$blueprint)
  predictors <- forged$predictors
  if (!is.null(exp_state)) {
    predictors <- .restore_typed_columns(predictors, new_data,
                                         exp_state$restored %||% character())
    attr(predictors, "restored") <- NULL
    predictors <- transform_frame_expansion(predictors, exp_state)
  }
  x <- .encode_predictors(predictors)
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

  if (type %in% c("mean", "median")) {
    return(as_frame(list(.pred = as.numeric(out))))
  }
  # quantiles / grid / sample all come back as a matrix of one column per
  # level or draw. The backend leaves a grid's columns unnamed, but its
  # levels are on the spec, so they get the same `q<level>` names a
  # quantile request does.
  cols <- as.list(as.data.frame(out))
  nm <- colnames(out)
  levels_ <- inner$spec$quantile_levels
  if (is.null(nm) && identical(type, "grid") &&
      length(levels_) == length(cols)) {
    nm <- .quantile_colnames(levels_)
  }
  nm <- if (is.null(nm)) paste0(".pred_", seq_along(cols))
        else paste0(".pred_", nm)
  # A repeated level (`quantiles = c(0.5, 0.5)`) is a legitimate request,
  # but a data frame cannot hold two columns of one name: tibble aborts
  # and `data.frame()` renames silently. Repair it the same way in both.
  names(cols) <- make.unique(nm, sep = "_")
  as_frame(cols)
}


#' @export
print.tabfound_fit <- function(x, ...) {
  cli::cli_text("{.strong tabfound} fit <{x$inner$backend}>, mode {.val {x$mode}}")
  preds <- x$predictor_names %||% names(x$blueprint$ptypes$predictors)
  cli::cli_bullets(c(
    "*" = "{length(preds)} predictor{?s}: {.field {utils::head(preds, 6)}}\\
           {if (length(preds) > 6) ' ...' else ''}",
    "*" = "fitted on {.val {x$inner$state$n_train}} rows",
    "*" = "missing values: {.val {x$na_action}}"
  ))
  invisible(x)
}
