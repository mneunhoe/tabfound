# Interoperability for multiple imputation.
#
# Nobody wants a fourth MI object class. What people want is to run the
# imputations here and then pool with the machinery they already use, so
# this file is entirely about handing the result over:
#
#   as_mids()               -> mice's `mids`, for with() / pool() / plot()
#   as_amelia()             -> Amelia's `amelia`, for mi.meld() and friends
#   complete()              -> registered as a method on mice's generic
#   mice.impute.tabfound()  -> the other direction: mice drives the loop
#                              and calls into these models per variable
#
# The `mids` is built from a real `mice(maxit = 0)` skeleton rather than
# assembled by hand: mice owns the invariants of its own object, and a
# hand-built one goes stale the moment mice adds a slot.


# ---------------------------------------------------------------------------
# mice
# ---------------------------------------------------------------------------

#' Convert to a mice `mids` object
#'
#' Repackages imputations as mice's multiply-imputed data set, so the
#' whole mice toolchain -- `with()`, `pool()`, `complete()`,
#' `densityplot()`, `stripplot()` -- applies unchanged.
#'
#' What is *not* carried over is the diagnostic history mice's own
#' sampler records: `chainMean` and `chainVar` are empty, so
#' `plot()` on the result has nothing to draw. Everything that depends on
#' the imputed values themselves works normally.
#'
#' @param x A `tabfound_mi` object from [tabfound_impute()].
#' @param ... Unused.
#' @return An object of class `mids`.
#' @examples
#' \dontrun{
#' mids <- as_mids(imp)
#' summary(mice::pool(with(mids, lm(Ozone ~ Wind + Temp))))
#' mice::densityplot(mids, ~ Ozone)
#' }
#' @export
as_mids <- function(x, ...) UseMethod("as_mids")

#' @rdname as_mids
#' @export
as_mids.tabfound_mi <- function(x, ...) {
  require_suggested("mice")
  # maxit = 0 runs no sampler: it returns the fully-formed object with
  # empty `imp` slots of exactly the right shape, which is the part that
  # is tedious and version-sensitive to build by hand.
  skel <- mice::mice(x$data, m = x$m, maxit = 0L, printFlag = FALSE,
                     remove.collinear = FALSE, remove.constant = FALSE)

  for (v in names(x$data)) {
    if (x$nmis[[v]] == 0L || is.null(skel$imp[[v]])) next
    rows <- which(x$where[, v])
    im <- skel$imp[[v]]
    for (j in seq_len(x$m)) im[[j]] <- x$imputations[[j]][[v]][rows]
    skel$imp[[v]] <- im
  }

  skel$method[] <- ""
  skel$method[x$visit_sequence] <- "tabfound"
  if (identical(dim(skel$predictorMatrix), dim(x$predictors))) {
    skel$predictorMatrix[] <- x$predictors[rownames(skel$predictorMatrix),
                                           colnames(skel$predictorMatrix)]
  }
  skel$visitSequence <- x$visit_sequence
  skel$iteration <- x$maxit
  skel$seed <- x$seed %||% NA
  skel$call <- x$call
  skel
}

# Registered on mice's own generic, so `mice::complete(imp, "long")`
# dispatches here. mice's generic names its first argument `data`.
#' @rdname tabfound_complete
#' @param data A `tabfound_mi` object.
#' @exportS3Method mice::complete
complete.tabfound_mi <- function(data, action = 1L, include = FALSE, ...) {
  tabfound_complete(data, action = action, include = include, ...)
}


# ---------------------------------------------------------------------------
# Amelia
# ---------------------------------------------------------------------------

#' Convert to an Amelia `amelia` object
#'
#' Repackages imputations the way `Amelia::amelia()` returns them, so
#' anything that consumes one -- most usefully `Amelia::mi.meld()` for
#' pooling estimates and standard errors -- takes this directly.
#'
#' The EM-specific slots (`theta`, `mu`, `covMatrices`) are `NULL`: they
#' describe a multivariate-normal model that was never fitted here, and
#' filling them with plausible-looking numbers would be worse than
#' leaving them empty. Amelia's diagnostics that read them
#' (`overimpute()`, `disperse()`, `compare.density()` in part) will not
#' work; `mi.meld()`, `missmap()` and the imputations themselves will.
#'
#' @param x A `tabfound_mi` object from [tabfound_impute()].
#' @param ... Unused.
#' @return An object of class `amelia`.
#' @examples
#' \dontrun{
#' amp <- as_amelia(imp)
#' fits <- lapply(amp$imputations, function(d) lm(Ozone ~ Wind + Temp, data = d))
#' b  <- t(sapply(fits, coef))
#' se <- t(sapply(fits, function(f) summary(f)$coefficients[, 2]))
#' Amelia::mi.meld(q = b, se = se)
#' }
#' @export
as_amelia <- function(x, ...) UseMethod("as_amelia")

#' @rdname as_amelia
#' @export
as_amelia.tabfound_mi <- function(x, ...) {
  imps <- x$imputations
  names(imps) <- paste0("imp", seq_len(x$m))
  class(imps) <- c("mi", "list")

  is_fac <- vapply(x$data, function(col) is.factor(col) || is.character(col),
                   logical(1))
  is_ord <- vapply(x$data, is.ordered, logical(1))
  args <- list(
    idvars = NULL, logs = NULL, ts = NULL, cs = NULL, empri = NULL,
    tolerance = NULL, polytime = NULL, splinetime = NULL, lags = NULL,
    leads = NULL, intercs = FALSE, sqrts = NULL, lgstc = NULL,
    noms = if (any(is_fac & !is_ord)) which(is_fac & !is_ord) else NULL,
    ords = if (any(is_ord)) which(is_ord) else NULL,
    priors = NULL, autopri = NULL, bounds = NULL, max.resample = NULL,
    startvals = NULL, overimp = NULL, emburn = NULL, boot.type = "none",
    m = x$m
  )
  class(args) <- c("ameliaArgs", "list")

  # `print.amelia` reports a chain length per imputation off `iterHist`.
  # Ours is the sweep count, which is the honest analogue.
  iter_hist <- replicate(x$m,
                         matrix(NA_real_, nrow = x$maxit, ncol = 1L,
                                dimnames = list(NULL, "sweep")),
                         simplify = FALSE)

  structure(
    list(imputations = imps, m = x$m, missMatrix = x$where,
         overvalues = NULL, theta = NULL, mu = NULL, covMatrices = NULL,
         code = 1L,
         message = "Imputations drawn from a tabfound predictive distribution.",
         iterHist = iter_hist, arguments = args, orig.vars = names(x$data)),
    class = "amelia"
  )
}


# ---------------------------------------------------------------------------
# The other direction: a mice imputation method
# ---------------------------------------------------------------------------

#' Use a tabular foundation model as a mice imputation method
#'
#' The mirror image of [tabfound_impute()]: instead of this package
#' running the chained-equations loop, mice runs it and calls in here for
#' each variable. That buys mice's full apparatus -- `where`, `blocks`,
#' `post`, `ignore`, convergence plots -- with these models doing the
#' univariate draws.
#'
#' `mice()` looks the method up by name, so `tabfound` must be attached
#' (`library(tabfound)`), and the models have to reach this function
#' somehow. Either pass `models` through `mice()`'s `...`, or set
#' `options(tabfound.models = tabfound_models(...))` once.
#'
#' Continuous targets are drawn from the regressor's predictive
#' distribution, categorical ones from the classifier's predicted
#' probabilities. Note that mice hands over an already-dummy-coded
#' numeric design matrix, so factor predictors reach the model one column
#' per contrast rather than as ordinal codes -- unlike
#' [tabfound_impute()], which keeps them whole.
#'
#' @param y The incomplete target, as mice supplies it.
#' @param ry Logical: `TRUE` where `y` is observed.
#' @param x Numeric design matrix of predictors, complete.
#' @param wy Logical: rows to impute. Defaults to `!ry`.
#' @param models A [tabfound_models()] handle. Defaults to
#'   `getOption("tabfound.models")`.
#' @param draw Passed to the regressor draw; see [tabfound_impute()].
#' @param ... Unused; absorbs the rest of what mice passes down.
#' @return A vector of length `sum(wy)`, in `y`'s own type.
#' @examples
#' \dontrun{
#' library(tabfound)
#' options(tabfound.models = tabfound_models(classifier = "...", regressor = "..."))
#' imp <- mice::mice(airquality, method = "tabfound", m = 5)
#' summary(mice::pool(with(imp, lm(Ozone ~ Wind + Temp))))
#' }
#' @export
mice.impute.tabfound <- function(y, ry, x, wy = NULL, models = NULL,
                                 draw = "auto", ...) {
  if (is.null(wy)) wy <- !ry
  models <- models %||% getOption("tabfound.models")
  if (is.null(models)) {
    cli::cli_abort(c(
      "No tabfound models available to impute with.",
      i = "Set {.code options(tabfound.models = tabfound_models(...))}, or \\
           pass {.arg models} through {.fn mice::mice}."
    ))
  }
  models <- .as_models(models)

  x <- as.matrix(x)
  storage.mode(x) <- "double"
  X_obs <- x[ry, , drop = FALSE]
  X_mis <- x[wy, , drop = FALSE]

  if (is.numeric(y) && !is.factor(y)) {
    out <- mi_draw_numeric(models$get("regression"), X_obs, as.numeric(y[ry]),
                           X_mis, draw = draw)
    return(if (is.integer(y)) as.integer(round(out)) else out)
  }
  # Factor, character or logical: draw a category and hand it back in the
  # type mice put in, which is what mice writes into the data.
  lab <- mi_draw_factor(models$get("classification"), X_obs,
                        as.factor(y[ry]), X_mis)
  if (is.logical(y)) return(lab == "TRUE")
  if (is.character(y)) return(lab)
  factor(lab, levels = levels(y))
}
