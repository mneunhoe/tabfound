# Interoperability for synthetic data.
#
# The same argument as R/mi-interop.R, for the other half of the problem.
# Nobody wants a second synthetic-data object class; what people want is
# to generate here and then evaluate, disclose-check and do inference
# with synthpop, which already owns all of that:
#
#   as_synds()          -> synthpop's `synds`, for compare() / lm.synds() /
#                          utility.gen() / disclosure() / replicated.uniques()
#   compare(), utility.gen(), utility.tab(), disclosure()
#                       -> registered on synthpop's own generics, so they
#                          take a `tabfound_syn` object directly
#   syn.tabpfn-style    -> the other direction: `synthpop::syn()` drives the
#   syn.tabfound()         visit sequence and calls in here per variable
#
# Writing `syn.tabfound()` is a small function with a large payoff: it
# buys the visit sequence and predictor-matrix logic, `rules` / `rvalues`
# constraints, missingness as a factor level, `k != n` generation,
# `compare()`, `utility.gen()` / `utility.tab()`, `replicated.uniques()`,
# `sdc()` and -- most importantly -- `lm.synds()` / `glm.synds()`, which
# implement the synthetic-data variance estimators. Reimplementing any of
# it would also make a benchmark against `syn.cart` less credible.
#
# As with `as_mids()`, the `synds` is built from a real skeleton
# (`syn(m = 0)`) rather than assembled by hand: synthpop owns the
# invariants of its own object, and a hand-built one goes stale the
# moment synthpop adds a slot.


# ---------------------------------------------------------------------------
# tabfound -> synthpop
# ---------------------------------------------------------------------------

#' Convert to a synthpop `synds` object
#'
#' Repackages the syntheses as synthpop's synthetic-data object, so the
#' whole synthpop toolchain applies unchanged: `compare()` for
#' distributional fidelity, `utility.gen()` / `utility.tab()` for general
#' utility, `lm.synds()` / `glm.synds()` for inference under the
#' synthetic-data variance estimators, and `replicated.uniques()`,
#' `disclosure()` and `sdc()` for the disclosure side.
#'
#' The skeleton comes from `synthpop::syn(data, m = 0)`, which builds the
#' object without synthesising anything. What is filled in afterwards is
#' the part that is ours: the syntheses, the visit sequence, the predictor
#' matrix, `k`, and `proper`.
#'
#' Methods are reported as `"tabfound"` (or `"sample"` for variables drawn
#' from their marginal, and `""` for variables carried through
#' unsynthesised) — which is what `lm.synds()` reads to decide whether a
#' model is being fitted to partially synthetic data.
#'
#' @param x A `tabfound_syn` object from [tabfound_syn()].
#' @param ... Unused.
#' @return An object of class `synds`.
#' @examples
#' \dontrun{
#' sds <- as_synds(syn)
#' synthpop::compare(sds, iris)
#' summary(synthpop::lm.synds(Sepal.Length ~ Petal.Length + Species, sds))
#' synthpop::replicated.uniques(sds, iris)$no.uniques
#' }
#' @export
as_synds <- function(x, ...) UseMethod("as_synds")

#' @rdname as_synds
#' @export
as_synds.tabfound_syn <- function(x, ...) {
  require_suggested("synthpop")
  # m = 0 synthesises nothing: it returns the fully-formed object with an
  # empty `syn` slot, which is the part that is tedious and
  # version-sensitive to build by hand. synthpop reports on the data with
  # `cat()` regardless of `print.flag`, hence the capture.
  invisible(utils::capture.output(
    skel <- suppressWarnings(
      synthpop::syn(x$data, m = 0L, print.flag = FALSE)
    )
  ))

  vars <- names(x$data)
  if (!setequal(names(skel$method), vars)) {
    cli::cli_abort(c(
      "{.fn synthpop::syn} restructured the columns, so the syntheses \\
       cannot be matched onto its object.",
      i = "This happens when synthpop splits or groups columns; synthesise \\
           through {.fn syn.tabfound} instead."
    ))
  }

  skel$m   <- x$m
  skel$syn <- if (x$m == 1L) x$syn[[1L]] else x$syn
  skel$method[] <- ""
  skel$method[x$visit_sequence] <-
    ifelse(x$method[x$visit_sequence] == "sample", "sample", "tabfound")
  skel$visit.sequence <- stats::setNames(match(x$visit_sequence, vars),
                                         x$visit_sequence)
  skel$predictor.matrix[vars, vars] <- x$predictors[vars, vars]
  skel$n <- x$n
  skel$k <- x$k
  skel$proper <- x$proper
  skel$seed <- x$seed %||% NA
  skel$call <- x$call
  skel
}

# Registered on synthpop's own generics, so a `tabfound_syn` object works
# wherever a `synds` does without the user converting first. synthpop
# names the first two arguments `object` and `data`.

#' Evaluate synthetic data with synthpop
#'
#' [tabfound_syn()] results are accepted directly by synthpop's utility
#' and disclosure generics — `compare()`, `utility.gen()`,
#' `utility.tab()` and `disclosure()` — which are registered here and
#' forward to [as_synds()].
#'
#' Functions that are not generic (`lm.synds()`, `glm.synds()`,
#' `replicated.uniques()`, `sdc()`) need the conversion spelled out:
#' `synthpop::lm.synds(y ~ x, as_synds(sds))`.
#'
#' @param object A `tabfound_syn` object.
#' @param data The original data frame.
#' @param ... Passed to synthpop.
#' @return Whatever the synthpop method returns.
#' @name synthpop-methods
#' @examples
#' \dontrun{
#' compare(sds, iris)
#' utility.gen(sds, iris)
#' }
NULL

#' @rdname synthpop-methods
#' @exportS3Method synthpop::compare
compare.tabfound_syn <- function(object, data, ...) {
  synthpop::compare(as_synds(object), data, ...)
}

#' @rdname synthpop-methods
#' @exportS3Method synthpop::utility.gen
utility.gen.tabfound_syn <- function(object, data, ...) {
  synthpop::utility.gen(as_synds(object), data, ...)
}

#' @rdname synthpop-methods
#' @exportS3Method synthpop::utility.tab
utility.tab.tabfound_syn <- function(object, data, ...) {
  synthpop::utility.tab(as_synds(object), data, ...)
}

#' @rdname synthpop-methods
#' @exportS3Method synthpop::disclosure
disclosure.tabfound_syn <- function(object, data, ...) {
  synthpop::disclosure(as_synds(object), data, ...)
}


# ---------------------------------------------------------------------------
# The other direction: a synthpop synthesis method
# ---------------------------------------------------------------------------

# synthpop rejects any `...` argument to `syn()` whose name is not
# `<one of its own methods>.<arg>`, so `syn(..., tabfound.draw_mode =)`
# cannot work. An option is the remaining channel, and it matches the one
# `mice.impute.tabfound()` already uses for its models.
# @keywords internal
.syn_method_options <- function(supplied) {
  defaults <- list(models = NULL, draw = "auto", draw_mode = "predictive",
                   donors = 5L, smoothing = FALSE, clamp = TRUE,
                   quantile_grid = 199L)
  opt <- getOption("tabfound.syn", list())
  if (!is.list(opt)) {
    cli::cli_abort("{.code options(tabfound.syn = )} must be a named list.")
  }
  unknown <- setdiff(names(opt), names(defaults))
  if (length(unknown)) {
    cli::cli_abort(c("Unknown {.code tabfound.syn} option{?s}: {.val {unknown}}.",
                     i = "Available: {.val {names(defaults)}}."))
  }
  out <- utils::modifyList(defaults, opt)
  out <- utils::modifyList(out, supplied[!vapply(supplied, is.null, logical(1))])
  out$models <- out$models %||% getOption("tabfound.models")
  if (is.null(out$models)) {
    cli::cli_abort(c(
      "No tabfound models available to synthesise with.",
      i = "Set {.code options(tabfound.models = tabfound_models(...))} \\
           before calling {.fn synthpop::syn}."
    ))
  }
  out$models    <- .as_models(out$models)
  out$draw_mode <- match.arg(out$draw_mode, c("predictive", "pmm", "rank"))
  out$draw      <- match.arg(out$draw, c("auto", "sample", "grid", "quantile",
                                         "residual"))
  out
}

#' Use a tabular foundation model as a synthpop synthesis method
#'
#' The mirror image of [tabfound_syn()]: instead of this package running
#' the synthesis pass, `synthpop::syn()` runs it and calls in here for
#' each variable in the visit sequence. That buys synthpop's full
#' apparatus — visit sequence and predictor matrix, `rules` / `rvalues`
#' constraints, `cont.na` and missingness handling, `k != n` generation,
#' and the whole evaluation and inference toolchain on the result.
#'
#' `syn()` looks the method up by name, so `tabfound` must be attached
#' (`library(tabfound)`) and the models have to reach this function:
#'
#' ```r
#' options(tabfound.models = tabfound_models(classifier = "...", regressor = "..."))
#' sds <- synthpop::syn(mydata, method = "tabfound", m = 5, proper = TRUE)
#' ```
#'
#' `syn()` refuses `...` arguments that are not named for one of its own
#' methods, so `tabfound.draw_mode = "pmm"` is not available. Tuning goes
#' through an option instead:
#'
#' ```r
#' options(tabfound.syn = list(draw_mode = "pmm", donors = 3))
#' ```
#'
#' Recognised entries are `models`, `draw`, `draw_mode`, `donors`,
#' `smoothing`, `clamp` and `quantile_grid`, all documented in
#' [tabfound_syn()].
#'
#' Two differences from [tabfound_syn()] are worth knowing. synthpop hands
#' over its own data frames, so factor predictors reach the model as
#' ordinal codes taken from the *context's* levels — the query is recoded
#' against them here, because coding `x` and `xp` independently would make
#' the synthetic covariates address different categories than the model
#' was fitted on. And `proper` follows synthpop's default of `FALSE`,
#' whereas [tabfound_syn()] defaults it to `TRUE`; see there for why that
#' matters.
#'
#' @param y The real values of the variable being synthesised.
#' @param x The real values of its predictors.
#' @param xp The already-synthesised values of those same predictors.
#' @param smoothing synthpop's smoothing request; anything other than
#'   `""` turns kernel smoothing on for donor-based draw modes.
#' @param proper Bootstrap the context before conditioning.
#' @param models,draw,draw_mode,donors,clamp,quantile_grid Overrides for
#'   the `tabfound.syn` option; see [tabfound_syn()].
#' @param ... Absorbs the rest of what synthpop passes down.
#' @return A list with `res` (the synthesised values) and `fit`, as
#'   synthpop's method contract requires.
#' @seealso [tabfound_syn()], [as_synds()]
#' @examples
#' \dontrun{
#' library(tabfound)
#' options(tabfound.models = tabfound_models(classifier = "...", regressor = "..."))
#' sds <- synthpop::syn(iris, method = "tabfound", m = 5)
#' synthpop::compare(sds, iris)
#' }
#' @export
syn.tabfound <- function(y, x, xp, smoothing = "", proper = FALSE,
                         models = NULL, draw = NULL, draw_mode = NULL,
                         donors = NULL, clamp = NULL, quantile_grid = NULL,
                         ...) {
  k <- if (is.null(dim(xp))) length(xp) else nrow(xp)
  y_obs <- y[!is.na(y)]

  if (is.null(x) || NCOL(x) == 0L || !length(y_obs)) {
    return(list(res = y[sample.int(length(y), k, replace = TRUE)],
                fit = "sample"))
  }
  if (length(unique(y_obs)) < 2L) {
    return(list(res = rep(y_obs[[1L]], k), fit = "constant"))
  }

  opts <- .syn_method_options(list(
    models = models, draw = draw, draw_mode = draw_mode, donors = donors,
    clamp = clamp, quantile_grid = quantile_grid,
    smoothing = if (isTRUE(nzchar(as.character(smoothing)[1L]))) TRUE else NULL
  ))

  x  <- as.data.frame(x)
  xp <- as.data.frame(xp)[, names(x), drop = FALSE]
  # Level coding must be identical in context and query, or the synthetic
  # covariates address different categories than the model was given.
  # This is the highest-probability silent bug in the integration.
  for (v in names(x)) {
    if (is.factor(x[[v]])) {
      xp[[v]] <- factor(as.character(xp[[v]]), levels = levels(x[[v]]))
    }
  }
  X_ctx <- .encode_predictors(x)
  X_q   <- .encode_predictors(xp)

  if (isTRUE(proper)) {
    b <- sample.int(nrow(X_ctx), nrow(X_ctx), replace = TRUE)
    X_ctx <- X_ctx[b, , drop = FALSE]
    y <- y[b]
  }

  if (!is.numeric(y) || is.factor(y)) {
    lab <- .syn_draw_categorical(as.factor(y), X_ctx, X_q, opts$models)
    res <- if (is.factor(y)) factor(lab, levels = levels(y))
           else if (is.logical(y)) lab == "TRUE"
           else lab
    return(list(res = res, fit = "tabfound"))
  }

  # Never `predict()`: a posterior mean would make every synthetic record
  # a fitted value and collapse the within-conditional variance.
  res <- .syn_draw_continuous(y, X_ctx, X_q, opts$models, opts,
                              cont_na = numeric(0), block = rep(1L, k))
  if (is.integer(y)) res <- as.integer(round(res))
  list(res = res, fit = paste0("tabfound-", opts$draw_mode))
}
