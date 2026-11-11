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

.new_tabfound_model <- function(ctx, spec, task, args,
                                preprocess = .preprocess_options()) {
  structure(
    list(
      spec      = spec,
      state     = NULL,
      model     = ctx$net,
      config    = ctx$config,
      device    = ctx$device,
      backend   = ctx$backend$name,
      task      = task,
      # How text and date columns are handled at `fit()`; see
      # `R/prep-frame.R`. Options only -- what a fit decided lives in
      # `state$expansion`.
      preprocess = preprocess,
      # Enough to rebuild this object from disk; see `tabfound_save()`.
      model_ref = list(model = ctx$model_ref, backend = ctx$backend$name,
                       device = ctx$device, args = args,
                       preprocess = preprocess)
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
  categorical_features <- .check_categorical_spec(categorical_features)
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

# `categorical_features` as the caller wrote it: positions, or column
# names. Anything else -- a logical mask, a list -- is refused here, where
# the mistake was made, instead of reaching a backend's `as.integer()` and
# coming back as "NAs introduced by coercion" followed by an index error.
# A factor of names is read as its labels, not its codes.
# @keywords internal
.check_categorical_spec <- function(categorical_features) {
  if (is.null(categorical_features)) return(NULL)
  if (is.factor(categorical_features)) {
    categorical_features <- as.character(categorical_features)
  }
  if (is.character(categorical_features)) {
    bad <- is.na(categorical_features) | !nzchar(categorical_features)
    if (any(bad)) {
      cli::cli_abort("{.arg categorical_features} has {sum(bad)} missing or \\
                      empty column name{?s}.")
    }
    return(categorical_features)
  }
  if (!is.numeric(categorical_features)) {
    cli::cli_abort(c(
      "{.arg categorical_features} takes column positions or column names.",
      x = "Got a {.cls {class(categorical_features)[1]}} vector."
    ))
  }
  categorical_features
}

# Turn a `categorical_features` declaration into column positions in a
# table with columns `cols`. Positions pass through untouched -- their
# range is the backend's to check, against the matrix it is given. Names
# are matched exactly; any that match nothing abort, naming them, since a
# misspelt name silently dropped would leave that column to inference.
# @keywords internal
.categorical_positions <- function(categorical_features, cols) {
  if (!is.character(categorical_features)) return(categorical_features)
  if (is.null(cols)) {
    cli::cli_abort(c(
      "{.arg categorical_features} names columns, but {.arg X} has no \\
       column names to match them against.",
      i = "Give {.arg X} column names, or declare the columns by position."
    ))
  }
  idx <- match(categorical_features, cols)
  if (anyNA(idx)) {
    missing <- categorical_features[is.na(idx)]
    cli::cli_abort(c(
      "{.arg categorical_features} names {length(missing)} column{?s} \\
       not in {.arg X}: {.field {missing}}.",
      i = "{.arg X} has {.field {cols}}."
    ))
  }
  idx
}

# A declaration by name can only become positions once `fit()` sees the
# columns, so the predictor is built without one and rebuilt here. The
# names stay on `model_ref`: they are what the caller wrote, they survive
# a refit on a reordered frame, and `tabfound_load()` rebuilds from them.
# @keywords internal
.resolve_categorical_names <- function(object, X) {
  declared <- object$model_ref$args$categorical_features
  if (!is.character(declared)) return(object)
  idx <- .categorical_positions(declared, colnames(X))
  keep_ref <- object$model_ref
  object <- .rebuild_predictor(object, idx)
  object$model_ref <- keep_ref
  object
}

# The arguments a predictor constructor is actually called with: names in
# `categorical_features` wait for `fit()`, which has the columns.
# @keywords internal
.ctor_args <- function(args) {
  if (is.character(args$categorical_features)) args$categorical_features <- NULL
  args
}

# Read an `ensemble_configs_dir` into `ensemble_configs`, once, here.
#
# The argument list is what `tabfound_save()` records to rebuild the
# object. A directory in it would be re-read at `tabfound_load()`, so a
# model reloaded after the directory was regenerated -- or deleted, as a
# `tempfile()` is with its session -- would run other members than the
# ones it was fitted with, or none. The configs are plain R data and
# small, so they travel with the object instead.
# @keywords internal
.resolve_configs_dir <- function(fn, args) {
  # `[[`, not `$`: `args$ensemble_configs` would partially match the
  # directory argument itself.
  dir <- args[["ensemble_configs_dir"]]
  # Both given is the backend's error to raise, in its own words.
  if (is.null(dir) || !is.null(args[["ensemble_configs"]]) ||
      !"ensemble_configs" %in% names(formals(fn))) {
    return(args)
  }
  args$ensemble_configs_dir <- NULL
  args$ensemble_configs <- load_ensemble_configs_from_dump(dir)
  args
}

#' Load a tabular foundation model for classification
#'
#' @param model Local directory of model artifacts, a HuggingFace repo
#'   id, or a registered alias (see [list_backends()]).
#' @param backend Optional backend name. Inferred from `config.json`
#'   when `NULL`.
#' @param device One of `"cpu"`, `"cuda"`, `"mps"`.
#' @param categorical_features Integer vector of 1-based column indices to
#'   treat as categorical, a character vector of column names (matched
#'   against `colnames(X)` at [fit()]), `integer()` to declare that none
#'   are, or `NULL` to let the backend infer. Declaring is much stronger than inferring:
#'   on TabPFN, inference takes a column nobody declared as categorical
#'   only when it has fewer than four distinct values and the training
#'   set has more than 100 rows, so an ordinal-coded factor with five
#'   levels is invisible to it -- and a count or time step with three
#'   values is swept up by it. Declared columns are still checked, and the
#'   other columns still inferred; only `integer()` turns inference off.
#'   [tabfound()] fills this in from the frame's factor columns
#'   automatically.
#' @param transform_text Expand text columns into numeric features at
#'   [fit()]: `"auto"` (default) does so when the checkpoint's own recipe
#'   asks for it -- TabPFN v3.5's does -- and not otherwise; `TRUE` or
#'   `FALSE` decides for every backend. A text column is a character column
#'   with more than `min_cardinality_for_text` distinct values that are not
#'   all numbers; it becomes `text_n_components` features, by latent
#'   semantic analysis over its character n-grams, exactly as TabPFN's
#'   Python estimator computes them. Off, such a column is coded as a
#'   categorical, as it always has been.
#' @param transform_dates Expand `Date` and `POSIXct` columns into calendar
#'   features: year, minute, second, seconds since the epoch, day of year,
#'   and circular month, day, hour and weekday. `"auto"` as for
#'   `transform_text`. Off, a date becomes a single number, as it always
#'   has. A `difftime` becomes its length in seconds either way.
#' @param min_cardinality_for_text Distinct-value count above which a
#'   character column counts as text rather than categorical.
#' @param text_n_components Features each text column becomes, at most.
#' @param ... Backend-specific arguments, e.g. `ensemble_configs` (or
#'   `ensemble_configs_dir`), `member_weights` or `softmax_temperature`.
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
                               categorical_features = NULL,
                               transform_text = "auto", transform_dates = "auto",
                               min_cardinality_for_text = 30L,
                               text_n_components = 30L, ...) {
  .tabular_model("classification", model, backend, device,
                 categorical_features,
                 .preprocess_options(transform_text, transform_dates,
                                     min_cardinality_for_text,
                                     text_n_components),
                 list(...))
}

# The body of `tabular_classifier()` / `tabular_regressor()`.
#
# `defaults` are arguments applied only where the backend's constructor
# declares them and the caller did not give them in `args`. That is how
# `tabfound_models()` asks for `softmax_temperature = 1` without handing
# an argument Kumo or Mitra have never heard of to their constructors --
# which backend a reference resolves to is only known once it is loaded.
# @keywords internal
.tabular_model <- function(task, model, backend, device, categorical_features,
                           preprocess, args, defaults = NULL) {
  ctx <- load_backend_model(model, task = task, backend = backend,
                            device = device)
  is_clf <- identical(task, "classification")
  fn <- if (is_clf) ctx$backend$classifier else ctx$backend$regressor
  if (is.null(fn)) {
    cli::cli_abort("Backend {.val {ctx$backend$name}} does not provide a \
                    {if (is_clf) 'classifier' else 'regressor'}.")
  }
  for (nm in names(defaults)) {
    if (is.null(args[[nm]]) && nm %in% names(formals(fn))) {
      args[[nm]] <- defaults[[nm]]
    }
  }
  args <- .predictor_args(fn, ctx, categorical_features,
                          .resolve_configs_dir(fn, args))
  .new_tabfound_model(ctx, do.call(fn, .ctor_args(args)), task, args[-1L],
                      preprocess)
}

# `tabular_classifier()` / `tabular_regressor()` with `defaults` applied as
# `.tabular_model()` describes: same arguments, same everything else.
# @keywords internal
.tabular_ctor <- function(task, defaults = NULL) {
  f <- if (identical(task, "classification")) tabular_classifier
       else tabular_regressor
  body(f) <- bquote(
    .tabular_model(.(task), model, backend, device, categorical_features,
                   .preprocess_options(transform_text, transform_dates,
                                       min_cardinality_for_text,
                                       text_n_components),
                   list(...), defaults = .(defaults))
  )
  f
}

#' Load a tabular foundation model for regression
#'
#' @inheritParams tabular_classifier
#' @return An unfitted object of class `tabfound_regressor`.
#' @export
tabular_regressor <- function(model, backend = NULL, device = "cpu",
                              categorical_features = NULL,
                              transform_text = "auto", transform_dates = "auto",
                              min_cardinality_for_text = 30L,
                              text_n_components = 30L, ...) {
  .tabular_model("regression", model, backend, device,
                 categorical_features,
                 .preprocess_options(transform_text, transform_dates,
                                     min_cardinality_for_text,
                                     text_n_components),
                 list(...))
}


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------

# Every backend's `fit_fn` opens with `as.matrix(X)` and then forces the
# storage mode to double. That is safe for a numeric matrix and silently
# destructive for a data frame with one non-numeric column: `as.matrix()`
# on a mixed frame goes through `format()`, so the result is a *character*
# matrix -- the factor column becomes all-`NA` and every numeric column is
# rounded to 7 significant digits. The fit succeeds, with one coercion
# warning, on wrong data.
#
# So no data frame ever reaches a backend. It is encoded here first, by
# the same function the formula interface uses, and anything that is
# neither numeric nor encodable is refused rather than mangled.
#
# `levels` is the training data's factor levels, recorded at `fit()` and
# replayed here: a factor's code is its position in its own level set, so
# a test frame whose column happens to hold only two of the three
# training levels would otherwise encode "c" as 1 where training encoded
# it as 2. This is `forge()`'s job on the formula path, and it has to be
# done on this one too.
# @keywords internal
.as_model_matrix <- function(x, arg = "X", levels = NULL) {
  if (is.data.frame(x)) {
    x <- .coerce_for_mold(x)
    if (length(levels)) x <- .apply_training_levels(x, levels, arg)
    return(.encode_predictors(x))
  }
  if (is.matrix(x)) {
    if (is.numeric(x) || is.logical(x)) {
      storage.mode(x) <- "double"
      return(x)
    }
    cli::cli_abort(c(
      "{.arg {arg}} is a {.cls {typeof(x)}} matrix; these models take numbers.",
      i = "Pass a data frame instead -- factors and dates are encoded for \\
           you -- or convert the columns yourself."
    ))
  }
  # A bare vector is one predictor; its names are row labels, not a
  # column name, so the matrix gets none.
  if (is.numeric(x) || is.logical(x)) return(matrix(as.double(x), ncol = 1L))
  cli::cli_abort(
    "{.arg {arg}} must be a matrix or a data frame, not a {.cls {class(x)[1]}}."
  )
}

# The target, checked before any backend sees it. None of them can use a
# missing or infinite target, and none says so: a regressor standardises
# `y`, so one `NA` turns its mean, its scale and then every border into
# `NA`, and every prediction with them; a classifier codes an `NA` label as
# an `NA` class index and fits on it. Aborting rather than dropping the
# rows matches how `X` is treated -- which rows to drop is the caller's
# call, not something to do behind their back.
# @keywords internal
.check_target <- function(y, call = sys.call(-1)) {
  if (!is.atomic(y) && !is.factor(y)) return(invisible(y))
  bad <- if (is.numeric(y)) !is.finite(y) else is.na(y)
  if (!any(bad)) return(invisible(y))
  rows <- which(bad)
  n_bad <- length(rows)
  # As text, so cli counts the rows rather than reading one as a quantity.
  shown <- as.character(utils::head(rows, 5L))
  where <- if (n_bad > length(shown)) {
    paste0("At rows ", paste(shown, collapse = ", "), ", and ",
           n_bad - length(shown), " more.")
  } else {
    "At row{?s} {shown}."
  }
  what <- if (is.numeric(y)) "missing or non-finite" else "missing"
  # One quantity per line for cli to pluralise by; the rest is pasted in.
  cli::cli_abort(c(
    paste0("{.arg y} has {n_bad} ", what, " value{?s} out of ", length(y), "."),
    x = where,
    i = "Drop those rows, or impute the target, before {.fn fit}."
  ), call = call)
}

# The factor levels of a coerced frame, for the columns that have any.
# @keywords internal
.training_levels <- function(d) {
  if (!is.data.frame(d)) return(NULL)
  lv <- lapply(d, function(col) if (is.factor(col)) levels(col) else NULL)
  lv <- lv[!vapply(lv, is.null, logical(1))]
  if (!length(lv)) NULL else lv
}

# @keywords internal
.apply_training_levels <- function(d, levels, arg = "newdata") {
  novel <- character()
  for (nm in intersect(names(levels), names(d))) {
    col <- d[[nm]]
    if (!is.factor(col)) next
    if (identical(levels(col), levels[[nm]])) next
    chr <- as.character(col)
    seen <- setdiff(unique(chr[!is.na(chr)]), levels[[nm]])
    if (length(seen)) novel <- c(novel, sprintf("%s: %s", nm,
                                                paste(seen, collapse = ", ")))
    d[[nm]] <- factor(chr, levels = levels[[nm]])
  }
  if (length(novel)) {
    cli::cli_warn(c(
      "{.arg {arg}} has level{?s} the model was not fitted on; \\
       they become {.val NA}.",
      set_names(novel, rep("*", length(novel)))
    ))
  }
  d
}

# A data frame's factor columns are encoded as ordinal codes, which is
# what the reference implementations do -- but only `tabfound()` also
# tells the backend *which* columns those are. On the matrix path the
# information exists (it is in the frame) and is being thrown away, and
# the backends' own fallback heuristic only catches columns with fewer
# than four distinct values.
# @keywords internal
.warn_undeclared_categoricals <- function(object, X) {
  if (!is.data.frame(X)) return(invisible(FALSE))
  if (!is.null(object$model_ref$args$categorical_features)) return(invisible(FALSE))
  cat_cols <- names(X)[vapply(X, function(col) {
    is.factor(col) || is.character(col) || is.logical(col)
  }, logical(1))]
  if (!length(cat_cols)) return(invisible(FALSE))
  cli::cli_warn(c(
    "{length(cat_cols)} categorical column{?s} encoded as ordinal codes: \\
     {.field {cat_cols}}.",
    i = "The backend was not told they are categorical. Use {.fn tabfound}, \\
         which declares them from the frame, or pass \\
         {.arg categorical_features} when constructing the model."
  ))
  invisible(TRUE)
}


# Rebuild a model's predictor with a different categorical declaration,
# reusing the loaded network.
#
# Chained equations and sequential synthesis fit a *different* predictor
# set for every variable, so which columns are categorical changes from
# one draw to the next and no fixed index vector can be right. Building a
# fresh `tabular_classifier()` per variable would re-read the weights
# from disk, which for TabFM is 6.5 GB; this rebuilds only the predictor
# closure around the network that is already in memory.
#
# A declaration the caller made explicitly wins: they know something we
# are inferring.
# @keywords internal
.respec_categoricals <- function(object, categorical_features) {
  if (is.null(categorical_features) || !length(categorical_features)) return(object)
  if (!is.null(object$model_ref$args$categorical_features)) return(object)
  .rebuild_predictor(object, categorical_features)
}

# The rebuild itself, with no opinion about whether it should happen. Also
# used when text or date expansion moves the columns an explicit
# declaration pointed at: the declaration is the caller's, but the indices
# it was written in no longer name the same columns.
# @keywords internal
.rebuild_predictor <- function(object, categorical_features) {
  bk  <- tryCatch(get_backend(object$backend), error = function(e) NULL)
  if (is.null(bk)) return(object)
  ctor <- if (inherits(object, "tabfound_classifier")) bk$classifier else bk$regressor
  if (is.null(ctor) || !"categorical_features" %in% names(formals(ctor))) {
    return(object)
  }
  ctx <- list(net = object$model, config = object$config, device = object$device,
              backend = bk, task = object$task, model_ref = object$model_ref$model)
  # The new declaration replaces any old one rather than sitting beside it:
  # `do.call()` refuses an argument named twice, and which of the two it
  # would have meant is exactly the question.
  rest <- object$model_ref$args
  rest$categorical_features <- NULL
  args <- c(list(ctx), list(categorical_features = as.integer(categorical_features)),
            rest)
  out <- .new_tabfound_model(ctx, do.call(ctor, args), object$task, args[-1L],
                             object$preprocess %||% .preprocess_options())
  out$model_ref <- object$model_ref
  out$model_ref$args <- args[-1L]
  out
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

#' @export
is_fitted.tabfound_fit <- function(object) is_fitted(object$inner)

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
#' @return A fitted copy of `object`. For TabICL it also carries
#'   `$n_members`, the number of ensemble members the fit built: fewer
#'   than `n_estimators` on a table with few features, and the number of
#'   `member_weights` that apply.
#' @seealso [estimate_peak_memory()]
#' @export
fit <- function(object, X, y, ...) UseMethod("fit")

#' @export
fit.tabfound_model <- function(object, X, y, ...) {
  .check_weights_alive(object)
  .check_target(y)
  # Before expansion, so names are matched against the caller's columns.
  object <- .resolve_categorical_names(object, X)
  expanded <- .fit_expansion_for(object, X)
  X <- expanded$X
  .warn_undeclared_categoricals(object, X)
  train_levels <- .training_levels(.coerce_for_mold(X))
  X <- .as_model_matrix(X, "X")
  # Checked here rather than per backend: the dimensions and the options
  # are the same question for all six, and a backend contributes only its
  # `peak_terms()` formula. `fit()` allocates almost nothing itself -- the
  # peak arrives at `predict()` -- so the check looks one call ahead, at
  # one chunk of query rows against this context.
  .memory_guard(object, n_context = NROW(X), n_query = .predict_chunk_of(object),
                n_features = NCOL(X), stage = "fit",
                n_classes = if (identical(object$task, "classification"))
                  length(unique(y[!is.na(y)])))
  # Assigning into a list copies it, so the caller's object is untouched.
  # The network is shared, not copied -- it is read-only and large.
  state <- expanded$fit_spec$fit(X, y)
  # The training schema, kept next to the training rows: what `predict()`
  # checks new data against. Backends record their own view of `X_train`,
  # which by then is a bare matrix with no promise about column names.
  state$n_features    <- ncol(X)
  state$feature_names <- colnames(X)
  state$feature_levels <- train_levels
  state$expansion     <- expanded$state
  object$state <- state
  # How many ensemble members the fit built, where a backend can build
  # fewer than it was asked for (TabICL on a table with few features).
  object$n_members <- state$n_members
  object
}

# Text and date expansion for the engine API, before anything else sees
# the frame. Returns the spec to fit with alongside, because an explicit
# `categorical_features` was written in the caller's column positions,
# and expansion moves every column that follows an expanded one.
#
# That spec is for this one fit only. The object `fit()` returns keeps its
# own: the moved positions belong to this frame's layout, and a refit on a
# frame that expands differently -- or not at all -- has to start from the
# declaration as written. `predict()` needs nothing from it, since the
# categoricals a fit decided live in `state`.
# @keywords internal
.fit_expansion_for <- function(object, X) {
  opts <- object$preprocess %||% .preprocess_options()
  if (!.frame_needs_expansion(X, opts)) {
    return(list(fit_spec = object$spec, X = X, state = NULL))
  }
  declared_idx <- .categorical_positions(
    object$model_ref$args$categorical_features, names(X))
  declared <- if (length(declared_idx)) names(X)[declared_idx] else character()
  fitted <- fit_frame_expansion(
    X,
    transform_text  = .resolve_transform_flag(opts$transform_text, object$config,
                                              "TRANSFORM_TEXT", "transform_text"),
    transform_dates = .resolve_transform_flag(opts$transform_dates, object$config,
                                              "TRANSFORM_DATES", "transform_dates"),
    min_cardinality_for_text = opts$min_cardinality_for_text,
    text_n_components = opts$text_n_components,
    declared = declared
  )
  spec <- object$spec
  if (length(declared_idx)) {
    moved <- .expanded_indices_of(declared, fitted$state)
    if (!identical(as.integer(moved), as.integer(declared_idx))) {
      spec <- .rebuild_predictor(object, moved)$spec
    }
  }
  list(fit_spec = spec, X = fitted$data, state = fitted$state)
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
  .check_predict_dots(object, ...)
  newdata <- .prepare_newdata(object, newdata)
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
  .check_predict_dots(object, ...)
  .check_quantiles(object$spec$predict, ...)
  newdata <- .prepare_newdata(object, newdata)
  object$spec$predict(object$state, newdata, type, ...)
}

# Coerce new data the same way the training data was coerced, check it
# against the training schema, and run the memory preflight.
#
# The fitted context is the other half of a prediction's dimensions, and
# it lives on the object rather than in the call.
# @keywords internal
.prepare_newdata <- function(object, newdata) {
  .kv_guard_check(object$state, object$model_ref$args)
  newdata <- .replay_expansion(object$state$expansion, newdata)
  newdata <- .as_model_matrix(newdata, "newdata", object$state$feature_levels)
  .check_newdata_schema(object, newdata)
  n_train <- object$state$n_train %||% NROW(object$state$X_train)
  .memory_guard(object, n_context = n_train %||% 0,
                n_query = NROW(newdata), n_features = NCOL(newdata),
                stage = "predict", n_classes = .n_classes_fitted(object))
  newdata
}

# Classes the fitted context holds, for the memory guard: the observed
# ones where the backend kept the codes, else every level.
# @keywords internal
.n_classes_fitted <- function(object) {
  if (!identical(object$task, "classification")) return(NULL)
  st <- object$state
  if (!is.null(st$y_train_int)) return(length(unique(st$y_train_int)))
  if (!is.null(st$class_levels)) length(st$class_levels) else NULL
}

# Lay new rows out as `fit()` laid out the training rows. Only a data frame
# can carry text and date columns, so once a fit expanded any, a matrix is
# refused rather than guessed into the expanded layout -- its raw width can
# even match, which is what would let it through the schema check below.
# @keywords internal
.replay_expansion <- function(state, newdata) {
  if (is.null(state)) return(newdata)
  if (!is.data.frame(newdata)) {
    if (length(state$dates) || length(state$text)) {
      cli::cli_abort(c(
        "The model expanded text or date columns at fit, so {.arg newdata} \
         has to be a data frame carrying them.",
        i = "Got a {.cls {class(newdata)[1]}}."
      ))
    }
    return(newdata)
  }
  transform_frame_expansion(newdata, state)
}

# Without this, a wrong column count surfaces as a libtorch C++ stack
# trace tens of frames down, and *reordered* columns do not surface at
# all: the network happily conditions column 3 of the context on column 3
# of the query whatever they mean. hardhat's `forge()` covers the formula
# path; this covers the matrix one.
# @keywords internal
.check_newdata_schema <- function(object, newdata) {
  p_train <- object$state$n_features
  if (!is.null(p_train) && ncol(newdata) != p_train) {
    cli::cli_abort(c(
      "{.arg newdata} has {ncol(newdata)} column{?s}; the model was fitted \\
       on {p_train}.",
      i = "Predictors must be the same columns, in the same order, as at \\
           {.fn fit} time."
    ))
  }
  .check_column_names(object$state$feature_names, colnames(newdata))
}

# The name half of the schema check, on its own so the frame expansion can
# run it on the raw columns: once a date or text column has been expanded,
# the expanded frame is renamed by position, and a check after that would
# pass whatever the columns had been called.
# @keywords internal
.check_column_names <- function(train_nm, new_nm) {
  if (is.null(train_nm) || is.null(new_nm) || identical(train_nm, new_nm)) {
    return(invisible(TRUE))
  }
  if (setequal(train_nm, new_nm)) {
    # Quoted and comma-joined by hand: `{.val}` would collapse the vector
    # with an "and", which is not code anyone can paste.
    cols <- paste0('"', train_nm, '"', collapse = ", ")
    cli::cli_abort(c(
      "{.arg newdata} has the training columns in a different order.",
      x = "Column {which(train_nm != new_nm)[1]} is \\
           {.field {new_nm[which(train_nm != new_nm)[1]]}}, but was \\
           {.field {train_nm[which(train_nm != new_nm)[1]]}} at fit time.",
      i = "Reorder with {.code newdata[, c({cols})]}."
    ))
  }
  # An expanded fit reaches here with only extra columns, which the count
  # check would otherwise have caught first; say only what applies.
  missing <- setdiff(train_nm, new_nm)
  extra <- setdiff(new_nm, train_nm)
  cli::cli_abort(c(
    "{.arg newdata} does not have the columns the model was fitted on.",
    x = if (length(missing)) "Missing: {.field {missing}}.",
    x = if (length(extra)) "Unexpected: {.field {extra}}."
  ))
}

# The real knobs on these backends are constructor-time
# (`tabular_regressor(model, softmax_temperature = )`), so a name misspelt
# at predict time -- or one that only exists on another backend -- is
# swallowed by the `...` every `predict_fn` ends with and silently does
# nothing. Say so.
# @keywords internal
.check_predict_dots <- function(object, ...) {
  given <- names(list(...))
  given <- given[nzchar(given %||% "")]
  if (!length(given)) return(invisible(TRUE))
  known <- setdiff(names(formals(object$spec$predict)),
                   c("state", "newdata", "type", "..."))
  unknown <- setdiff(given, known)
  if (!length(unknown)) return(invisible(TRUE))
  cli::cli_warn(c(
    "{.fn predict} ignore{?s/} the argument{?s} {.arg {unknown}}.",
    i = if (length(known)) "The {.val {object$backend}} backend takes \\
                            {.arg {known}} here."
        else "The {.val {object$backend}} backend takes no extra arguments here.",
    i = "Backend options are set when the model is constructed, not at \\
         predict time."
  ))
  invisible(FALSE)
}

# Quantile levels outside (0, 1) are extrapolated off the end of the
# predicted distribution and come back as plausible-looking numbers.
#
# The dots reach the backend's predict function after `state`, `newdata`
# and `type`, so an unnamed one binds to its `quantiles` by position, and
# `quant =` by partial matching. Matching them against that function's
# formals checks the value the backend will actually use, not only a dot
# that happens to be spelled `quantiles`.
# @keywords internal
.check_quantiles <- function(predict_fn, ...) {
  dots <- list(...)
  if ("quantiles" %in% names(formals(predict_fn))) {
    call <- match.call(predict_fn,
                       as.call(c(quote(predict_fn), list(NULL, NULL, NULL),
                                 dots)))
    if (!"quantiles" %in% names(call)) return(invisible(TRUE))
    quantiles <- call[["quantiles"]]
  } else {
    if (!"quantiles" %in% names(dots)) return(invisible(TRUE))
    quantiles <- dots[["quantiles"]]
  }
  if (!is.numeric(quantiles) || !length(quantiles)) {
    cli::cli_abort("{.arg quantiles} must be a numeric vector.")
  }
  bad <- !is.finite(quantiles) | quantiles <= 0 | quantiles >= 1
  if (any(bad)) {
    cli::cli_abort(c(
      "{.arg quantiles} must lie strictly inside {.val {c(0, 1)}}.",
      x = "Out of range: {.val {quantiles[bad]}}."
    ))
  }
  invisible(TRUE)
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

#' @export
predict_proba.tabfound_fit <- function(object, newdata, ...) {
  if (!identical(object$mode, "classification")) {
    cli::cli_abort("{.fn predict_proba} needs a classification fit, \\
                    not a {.val {object$mode}} one.")
  }
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

#' @export
predict_quantiles.tabfound_fit <- function(object, newdata,
                                           quantiles = c(0.1, 0.5, 0.9), ...) {
  if (!identical(object$mode, "regression")) {
    cli::cli_abort("{.fn predict_quantiles} needs a regression fit, \\
                    not a {.val {object$mode}} one.")
  }
  predict(object, newdata, type = "quantiles", quantiles = quantiles, ...)
}

#' Per-member ensemble predictions
#'
#' The pieces an ensemble prediction is assembled from, before they are
#' combined: one entry per member, in the order the model's ensemble
#' runs them. This is the hook for anything that reweights, selects or
#' recalibrates an ensemble after the fact -- each member is run once,
#' and every combination of them is then arithmetic on what this
#' returns. Every backend supports it.
#'
#' What a member is depends on the backend:
#'
#' * **TabPFN** (v2.5 to v3.5): one entry of `ensemble_configs` (or
#'   `ensemble_configs_dir`). A model built without either is a single
#'   forward pass and has no members.
#' * **TabICL, TabFM, Mitra, Kumo**: one of the `n_estimators` members the
#'   backend generates from its own settings and seed.
#'
#' The output names how the backend combines its members, in `$combine`:
#'
#' | `$combine` | the model's prediction is |
#' |---|---|
#' | `"mean_of_softmax"` | the (weighted) mean of `softmax(logits / softmax_temperature)` |
#' | `"softmax_of_mean_logits"` | `softmax(mean(logits) / softmax_temperature)`, with a weighted mean under weights |
#' | `"bar_mixture"` | the log of the (weighted) mean of `$probs`, as logits for [bar_logits_to_mean()] and [bar_logits_to_quantiles()] on `$borders` |
#' | `"mean_of_grid_stats"` | each member's grid summarised by [quantile_grid_stat()], then the (weighted) mean of the summaries |
#' | `"trimmed_mean_grid"` | the grids combined level by level by a mean that drops `floor(0.2 * n)` members from each tail, then summarised |
#' | `"mean_grid"` | the grids combined level by level by a weighted mean (Kumo with weights), then summarised |
#' | `"mean"` | the (weighted) mean of `$mean` |
#'
#' Classifier logits are raw: *before* any temperature, with the
#' member's class permutation or shift undone, one column per class
#' level. A class Kumo could not predict (absent from the training rows)
#' has logit `-Inf`.
#'
#' @param object A fitted `tabfound_model`.
#' @param newdata Test features, as for `predict()`.
#' @param ... Unused.
#' @return A list with one of:
#'   * `$logits`: classifiers; a list of `(n_new, n_classes)` matrices.
#'   * `$probs` and `$borders`: TabPFN regressors; bucket probabilities on
#'     the shared grid (after the temperature) and its raw-scale borders.
#'   * `$grids` and `$quantile_levels`: TabICL and Kumo regressors;
#'     `(n_new, n_levels)` quantile grids on the raw y scale.
#'   * `$mean`: TabFM and Mitra regressors; point predictions on the raw
#'     y scale.
#'
#'   Plus:
#'   * `$member_index`: which members these are, as positions in the
#'     ensemble. Members weighted zero are not run, so not returned.
#'   * `$member_weights`: the weights the model combines with, normalised
#'     to sum to one, or `NULL` for uniform.
#'   * `$combine`: as in the table above.
#'   * `$softmax_temperature`: where the backend has one.
#' @examples
#' \dontrun{
#' cfg <- generate_ensemble_configs_native(X, y, n_estimators = 8,
#'                                         variant = "v2.6")
#' clf <- fit(tabular_classifier("tabpfn-v2.6-classifier",
#'                               ensemble_configs = cfg), X, y)
#' m <- predict_members(clf, X_new)
#' p <- Reduce(`+`, lapply(m$logits, function(l) {
#'   e <- exp(l / m$softmax_temperature); e / rowSums(e)
#' })) / length(m$logits)                 # == predict(clf, X_new, type = "prob")
#'
#' # TabICL: pick members, then refit with weights
#' icl <- fit(tabular_classifier("tabicl-v2-classifier", n_estimators = 16), X, y)
#' m <- predict_members(icl, X_val)       # score each member on held-out rows
#' w <- c(rep(1, 4), rep(0, 12))          # keep the first four, say
#' icl_w <- fit(tabular_classifier("tabicl-v2-classifier", n_estimators = 16,
#'                                 member_weights = w), X, y)
#' }
#' @export
predict_members <- function(object, newdata, ...) UseMethod("predict_members")

#' @export
predict_members.tabfound_model <- function(object, newdata, ...) {
  .require_fitted(object)
  if (is.null(object$spec$predict_members)) {
    cli::cli_abort("The {.val {object$backend}} backend has no \\
                    per-member predictions.")
  }
  newdata <- .prepare_newdata(object, newdata)
  out <- object$spec$predict_members(object$state, newdata)
  if (!is.null(out$member_weights)) {
    out$member_weights <- out$member_weights / sum(out$member_weights)
  }
  out
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
#' HuggingFace cache) must still be reachable. A local model directory is
#' recorded as an absolute path, so it is found again from any working
#' directory. TabPFN ensemble configs are saved themselves, not the
#' `ensemble_configs_dir` they were read from, so the reloaded model runs
#' the same members even once that directory is gone.
#'
#' Both object types are supported: the engine-level `tabfound_model`
#' from [tabular_classifier()] / [tabular_regressor()], and the
#' `tabfound_fit` that [tabfound()] returns. For the latter the hardhat
#' blueprint travels with it, so the reloaded object re-applies the same
#' encoding and the same factor levels to new data.
#'
#' @param object A `tabfound_model` or a `tabfound_fit`, fitted or not.
#' @param file Path to write to: a file, or a directory bundle when the
#'   model carries a cache from [tabfound_cache()]. An earlier save at the
#'   same path is replaced, whichever of the two forms it took.
#' @return `file`, invisibly.
#' @examples
#' \dontrun{
#' clf <- fit(tabular_classifier("path/to/model"), X, y)
#' tabfound_save(clf, "clf.tabfound")
#' clf2 <- tabfound_load("clf.tabfound")
#' }
#' @export
tabfound_save <- function(object, file) {
  base <- list(
    version     = .tabfound_save_version,
    pkg_version = as.character(utils::packageVersion("tabfound"))
  )
  blob <- if (inherits(object, "tabfound_fit")) {
    c(list(format = "tabfound-fit",
           task   = object$inner$task,
           model_ref = object$inner$model_ref,
           state     = object$inner$state,
           mode      = object$mode,
           na_action = object$na_action,
           imputer   = object$imputer,
           expansion = object$expansion,
           blueprint = object$blueprint),
      base)
  } else if (inherits(object, "tabfound_model")) {
    c(list(format = "tabfound-model",
           task   = object$task,
           model_ref = object$model_ref,
           state     = object$state),
      base)
  } else {
    cli::cli_abort("{.arg object} must be a {.cls tabfound_model} or a \\
                    {.cls tabfound_fit}.")
  }

  # A precomputed cache is torch tensors, which is the one thing an RDS
  # cannot carry -- so a cached model is written as a *bundle* directory:
  # `state.rds` beside `cache.safetensors`. Without a cache nothing has
  # changed and the single file stays a single file.
  caches <- blob$state$kv_caches
  .clear_save_target(file, bundle = !is.null(caches))
  if (is.null(caches)) {
    saveRDS(blob, file)
    return(invisible(file))
  }
  dir.create(file, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(file)) {
    cli::cli_abort("Could not create the bundle directory {.path {file}}.")
  }
  written <- .kv_write(caches, file)
  blob$state$kv_caches <- NULL          # the tensors live in the other file
  blob$cache_skeleton  <- written$skeleton
  blob$cache_file      <- written$file
  saveRDS(blob, file.path(file, "state.rds"))
  invisible(file)
}

# Saving over an earlier save is an overwrite, whichever form either one
# took. Left alone, a file where a bundle should go cannot become a
# directory, and a bundle where a file should go makes `saveRDS()` fail
# on "it is a directory" -- while `tabfound_load()`, which goes by what is
# on disk, keeps returning the old model. A file is replaced as
# `saveRDS()` would replace it anyway. A directory is removed only when it
# holds nothing but what `tabfound_save()` writes: anything else in it is
# not ours to delete.
# @keywords internal
.clear_save_target <- function(file, bundle) {
  if (bundle) {
    if (file.exists(file) && !dir.exists(file)) unlink(file)
    return(invisible())
  }
  if (!dir.exists(file)) return(invisible())
  ours <- c("state.rds", "cache.safetensors")
  found <- list.files(file, all.files = TRUE, no.. = TRUE)
  if (!"state.rds" %in% found || !all(found %in% ours)) {
    cli::cli_abort(c(
      "Cannot save to {.path {file}}: it is a directory.",
      x = "It is not a {.fn tabfound_save} bundle, so it is not replaced.",
      i = "Choose another path, or remove the directory first."
    ))
  }
  unlink(file, recursive = TRUE)
  invisible()
}

# Bumped when the payload's layout changes in a way an older reader could
# not make sense of. Readers refuse anything newer than they know.
.tabfound_save_version <- 1L

#' @rdname tabfound_save
#' @param device Optional device override for the reloaded model.
#' @return For `tabfound_load()`, whichever of the two object types was
#'   saved.
#' @export
tabfound_load <- function(file, device = NULL) {
  # A bundle directory (see `tabfound_save()`) or a plain file.
  bundle <- dir.exists(file)
  blob <- readRDS(if (bundle) file.path(file, "state.rds") else file)
  if (!identical(blob$format, "tabfound-model") &&
      !identical(blob$format, "tabfound-fit")) {
    cli::cli_abort(c(
      "{.path {file}} is not a saved tabfound model.",
      i = "It was probably written with {.fn saveRDS}, which cannot \\
           capture torch weights. Re-save with {.fn tabfound_save}."
    ))
  }
  ver     <- blob$version %||% 0L
  max_ver <- .tabfound_save_version
  if (!is.numeric(ver) || ver > max_ver) {
    cli::cli_abort(c(
      "{.path {file}} is in format version {.val {ver}}; this version of \\
       tabfound reads up to {.val {max_ver}}.",
      i = "It was written by tabfound {.val {blob$pkg_version %||% 'unknown'}}. \\
           Upgrade the package to read it."
    ))
  }

  ref  <- blob$model_ref
  .check_saved_configs_dir(file, ref$args$ensemble_configs_dir,
                           cached = !is.null(blob$cache_skeleton))
  ctor <- if (identical(blob$task, "classification")) tabular_classifier
          else tabular_regressor
  obj <- do.call(ctor, c(
    list(model = ref$model, backend = ref$backend,
         device = device %||% ref$device),
    ref$preprocess,
    ref$args
  ))
  obj$state <- blob$state
  # The cache travels as tensors in a sibling file, on whichever device
  # this model was just loaded onto.
  if (!is.null(blob$cache_skeleton)) {
    obj$state$kv_caches <- .kv_read(blob$cache_skeleton,
                                    file.path(file, blob$cache_file),
                                    device = obj$device)
  }
  if (identical(blob$format, "tabfound-model")) return(obj)

  require_suggested("hardhat")
  hardhat::new_model(
    inner     = obj,
    mode      = blob$mode,
    na_action = blob$na_action,
    imputer   = blob$imputer,
    expansion = blob$expansion,
    blueprint = blob$blueprint,
    class     = "tabfound_fit"
  )
}

# Saves written before the configs were read in at construction record
# the directory instead, and loading re-reads it. That still works while
# the directory is there and unchanged; nothing on disk can say whether it
# is unchanged, so a cache from such a save, which the guard cannot check
# against the members, is flagged rather than trusted silently.
# @keywords internal
.check_saved_configs_dir <- function(file, dir, cached) {
  if (is.null(dir)) return(invisible())
  if (!file.exists(file.path(dir, "ensemble_configs.json"))) {
    cli::cli_abort(c(
      "Cannot load {.path {file}}: its ensemble configs are gone.",
      x = "It was saved by an older tabfound, which recorded \\
           {.path {dir}} rather than the configs themselves.",
      i = "Restore that directory, or refit the model and save it again."
    ))
  }
  if (cached) {
    cli::cli_warn(c(
      "{.path {file}} was saved by an older tabfound and re-reads its \\
       ensemble configs from {.path {dir}}.",
      i = "If that directory changed since the save, the saved cache no \\
           longer matches the members. Rebuild it with \\
           {.fn tabfound_cache} and save again, which records the \\
           configs and checks the cache against them."
    ))
  }
  invisible()
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
