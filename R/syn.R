# Fully conditional synthesis: sequential synthetic data generation.
#
# Synthesis and multiple imputation (R/mi.R) are the same idea pointed at
# different cells, and they diverge on four things:
#
#                 imputation                    synthesis
#   targets       the missing cells             every row of every variable
#   conditioning  all other variables           only variables already visited
#   iteration     `maxit` sweeps                one ordered pass
#   context       the current completed data,   the *original* data,
#                 refit every sweep             fit once per variable
#
# The last row is the substance. Synthesis factorises the joint as
# p(x1) * p(x2 | x1) * p(x3 | x1, x2) * ..., fits each conditional on the
# real data, and draws at the *synthetic* covariates generated earlier in
# the same pass. The first variable is a bootstrap of the real marginal.
#
# Two consequences follow, and both are load-bearing:
#
#   * It is cheaper than the imputation path -- p fits rather than
#     m * maxit * p. And with `proper = FALSE` the context for a variable
#     is identical across all m syntheses, so one fit and one forward pass
#     over an (m * k)-row query batch produce all of them. A tree has to
#     be regrown per synthesis; conditioning on a context does not.
#
#   * It is statistically riskier. A PFN draws each query row
#     independently given a fixed context, so between-synthesis variance
#     carries predictive noise and *no parameter uncertainty*. That is the
#     same defect that makes an imputer improper under Rubin's rules, and
#     the synthetic-data variance estimators in `synthpop` assume it away.
#     Hence `proper = TRUE` by default here: bootstrap the context before
#     conditioning, which is exactly synthpop's definition of properness.
#     This is the one place where this function's default deliberately
#     differs from `synthpop::syn()`.
#
# The context *is* the real data at generation time. No differential
# privacy claim is available and none is made; see `?tabfound_syn`.
#
# Like R/mi.R this is a use of the fitted models, not a port of anything,
# and the parity harness has nothing to say about it. The bridge to
# synthpop -- `as_synds()` and the `syn.tabfound()` method -- is in
# R/syn-interop.R.


# ---------------------------------------------------------------------------
# Column plans
# ---------------------------------------------------------------------------

# The imputation plans (`.mi_column_plan()`) already map every column type
# onto the representation a head takes and back again. Synthesis needs two
# changes on top:
#
#   * Numeric 0/1 columns take the classifier path whether or not they
#     have missing values. In imputation only the missing cells are drawn,
#     so a fully observed dummy never reaches a head at all; in synthesis
#     every cell is drawn, and a bar distribution cannot produce a dummy.
#
#   * NA is a value to be reproduced, not a hole to be filled. For a
#     categorical column it becomes an explicit level the classifier can
#     draw, mapped back to NA on the way out. (Continuous columns get the
#     two-part treatment instead; see `.syn_draw_continuous()`.)
# @keywords internal
.syn_column_plan <- function(col, name = "x") {
  obs <- col[!is.na(col)]
  base <- if (is.numeric(col) && !is.factor(col) && length(obs) &&
              all(obs %in% c(0, 1)) && length(unique(obs)) == 2L) {
    # Back to the storage mode it arrived in, not to whatever the
    # round trip through a factor label happens to produce.
    as_orig <- if (is.integer(col)) as.integer else as.numeric
    list(kind     = "categorical",
         to_model = function(x) factor(x, levels = c("0", "1")),
         restore  = function(x) as_orig(as.character(x)))
  } else {
    .mi_column_plan(col, name)
  }

  if (!identical(base$kind, "categorical") || !anyNA(col)) {
    return(c(base, list(na_label = NA_character_)))
  }

  lv <- levels(base$to_model(obs))
  na_label <- ".NA"
  while (na_label %in% lv) na_label <- paste0(na_label, ".")
  to_model0 <- base$to_model
  restore0  <- base$restore

  list(
    kind     = "categorical",
    na_label = na_label,
    to_model = function(x) {
      f <- factor(as.character(to_model0(x)), levels = c(lv, na_label))
      f[is.na(f)] <- na_label
      f
    },
    restore = function(x) {
      chr <- as.character(x)
      chr[!is.na(chr) & chr == na_label] <- NA
      restore0(factor(chr, levels = lv))
    }
  )
}


# ---------------------------------------------------------------------------
# Turning a draw into a value
# ---------------------------------------------------------------------------

# A CART leaf is a set of observed values, so `syn.cart` cannot produce a
# value the data does not contain, cannot leave the observed range, and
# cannot break integrality. A bar distribution is a continuum and does all
# three. `draw_mode` is how far from CART's support behaviour you want to
# be, and it is the utility/disclosure axis -- an experimental factor to
# report, not a default to hide.

# "predictive": keep the draw, repair only what is unambiguously a type
# error -- integrality and, optionally, the observed range.
# @keywords internal
.syn_restore_support <- function(v, y, clamp = TRUE) {
  if (length(y) && all(abs(y - round(y)) < .Machine$double.eps^0.5)) {
    v <- round(v)
  }
  if (isTRUE(clamp) && length(y)) v <- pmin(pmax(v, min(y)), max(y))
  v
}

# "pmm": return a real donor from the `d` observed values nearest the
# draw. Exact CART support semantics -- observed value set, bounds and
# integrality all for free -- at the highest replication-disclosure risk.
# @keywords internal
.syn_pmm <- function(raw, y, d = 5L) {
  ys <- sort(y)
  n  <- length(ys)
  if (!n) return(raw)
  if (n < 2L) return(rep(ys[1L], length(raw)))
  d <- max(1L, min(as.integer(d), n))
  j <- findInterval(raw, ys, all.inside = TRUE)
  lo <- pmin(pmax(1L, j - d %/% 2L), n - d + 1L)
  # `sample.int(d, ...)`, not `sample(a:b)`, which misbehaves when the
  # donor window has length 1.
  ys[lo + sample.int(d, length(raw), replace = TRUE) - 1L]
}

# "rank": map the draws onto the observed order statistics by rank, the
# way `syn.normrank` does. Reproduces the real marginal exactly while
# keeping the conditional shape the model drew -- which also means the
# marginal fidelity is partly by construction rather than by merit.
# @keywords internal
.syn_rankmap <- function(raw, y) {
  if (!length(y)) return(raw)
  p <- (rank(raw, ties.method = "random") - 0.5) / length(raw)
  stats::quantile(y, probs = p, type = 1L, names = FALSE)
}

# synthpop's `smoothing = "density"`: Gaussian noise at `bw.nrd0`, with
# boundary values left alone so the observed min and max are not pushed
# outside the support.
# @keywords internal
.syn_smooth <- function(v, y) {
  if (length(y) < 2L) return(v)
  bw <- stats::bw.nrd0(y)
  if (!is.finite(bw) || bw <= 0) return(v)
  keep <- v <= min(y) | v >= max(y)
  out  <- v + stats::rnorm(length(v), 0, bw)
  out[keep] <- v[keep]
  pmin(pmax(out, min(y)), max(y))
}

# @keywords internal
.syn_post <- function(raw, y, draw_mode, donors, smoothing, clamp) {
  out <- switch(draw_mode,
                predictive = .syn_restore_support(raw, y, clamp = clamp),
                pmm        = .syn_pmm(raw, y, donors),
                rank       = .syn_rankmap(raw, y))
  # A predictive draw is already smooth; smoothing it again would only
  # blur a distribution the model meant.
  if (isTRUE(smoothing) && !identical(draw_mode, "predictive")) {
    out <- .syn_smooth(out, y)
  }
  out
}


# ---------------------------------------------------------------------------
# One variable
# ---------------------------------------------------------------------------

# Continuous variables that carry NA, or a point mass at some value, need
# two parts: a categorical model for "which special value, or a real
# number?" and a continuous model for the real numbers. A bar distribution
# cannot represent a point mass, so a one-part model would smear the spike
# and never produce NA at all.
# @keywords internal
.syn_draw_continuous <- function(y, X_ctx, X_query, models, opts, cont_na,
                                 block) {
  n_q <- nrow(X_query)
  special <- is.na(y) | (!is.na(y) & y %in% cont_na)
  label <- rep(".cont", length(y))
  label[is.na(y)] <- ".NA"
  keep <- !is.na(y) & y %in% cont_na
  label[keep] <- as.character(y[keep])

  out <- rep(NA_real_, n_q)
  # When every value is special the column is a categorical variable
  # wearing a numeric type, and the second part has nothing left to do.
  part <- if (any(special)) {
    .syn_draw_categorical(factor(label), X_ctx, X_query, models)
  } else {
    rep(".cont", n_q)
  }
  is_num <- part == ".cont"
  out[!is_num & part != ".NA"] <- as.numeric(part[!is_num & part != ".NA"])

  if (any(is_num)) {
    obs   <- !special
    y_obs <- as.numeric(y[obs])
    raw <- mi_draw_numeric(models$get("regression"), X_ctx[obs, , drop = FALSE],
                           y_obs, X_query[is_num, , drop = FALSE],
                           draw = opts$draw, quantile_grid = opts$quantile_grid)
    bad <- !is.finite(raw)
    if (any(bad)) {
      cli::cli_warn(c(
        "{sum(bad)} non-finite draw{?s}; replaced by a bootstrap of the \\
         observed values.",
        i = "This usually means the backend saw {.val NA} in the predictors."
      ))
      raw[bad] <- sample(y_obs, sum(bad), replace = TRUE)
    }
    # Post-processing runs per synthesis, not over the stacked batch:
    # `rank` maps onto the observed order statistics, and doing that
    # across all m at once would give each synthesis a subsample of the
    # marginal rather than the marginal.
    b <- block[is_num]
    for (j in unique(b)) {
      sel <- b == j
      raw[sel] <- .syn_post(raw[sel], y_obs, opts$draw_mode, opts$donors,
                            opts$smoothing, opts$clamp)
    }
    out[is_num] <- raw
  }
  out
}

# @keywords internal
.syn_draw_categorical <- function(y, X_ctx, X_query, models) {
  y <- droplevels(as.factor(y))
  lv <- levels(y)
  if (length(lv) < 2L) return(rep(lv[1L], nrow(X_query)))
  mi_draw_factor(models$get("classification"), X_ctx, y, X_query)
}


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

# @keywords internal
.syn_resolve_visit <- function(visit_sequence, vars) {
  if (is.null(visit_sequence)) return(vars)
  if (is.numeric(visit_sequence)) {
    if (any(is.na(visit_sequence)) ||
        !all(visit_sequence %in% seq_along(vars))) {
      cli::cli_abort("{.arg visit_sequence} indexes columns not in {.arg data}.")
    }
    visit_sequence <- vars[visit_sequence]
  }
  unknown <- setdiff(visit_sequence, vars)
  if (length(unknown)) {
    cli::cli_abort("{.arg visit_sequence} names column{?s} not in \\
                    {.arg data}: {.val {unknown}}.")
  }
  if (anyDuplicated(visit_sequence)) {
    cli::cli_abort("{.arg visit_sequence} must not repeat a column.")
  }
  as.character(visit_sequence)
}

# Which variables are already in the synthetic frame when `v` is
# synthesised: everything earlier in the visit sequence, plus everything
# carried over from the real data unsynthesised.
# @keywords internal
.syn_available <- function(vars, visit, meth) {
  carried <- vars[!nzchar(meth)]
  out <- stats::setNames(vector("list", length(visit)), visit)
  for (i in seq_along(visit)) {
    out[[i]] <- setdiff(union(visit[seq_len(i - 1L)], carried), visit[[i]])
  }
  out
}

# @keywords internal
.syn_resolve_predictors <- function(predictors, vars, avail) {
  pm <- matrix(0L, length(vars), length(vars), dimnames = list(vars, vars))
  for (v in names(avail)) pm[v, avail[[v]]] <- 1L

  if (is.null(predictors)) return(pm)
  if (!is.matrix(predictors)) {
    cli::cli_abort("{.arg predictors} must be a matrix.")
  }
  rn <- rownames(predictors); cn <- colnames(predictors)
  if (is.null(rn) || is.null(cn) ||
      !setequal(rn, vars) || !setequal(cn, vars)) {
    cli::cli_abort(c(
      "{.arg predictors} must be a square matrix named by the columns of \\
       {.arg data}.",
      i = "Same layout as mice's {.field predictorMatrix}: \\
           {.code predictors[v, w] == 1} means {.field w} is used to \\
           synthesise {.field v}."
    ))
  }
  want <- pm
  want[vars, vars] <- as.integer(predictors[vars, vars] != 0)
  # A single ordered pass has nothing later to condition on. Silently
  # honouring a request for a not-yet-generated variable would be worse
  # than saying it was dropped.
  dropped <- want & !pm
  if (any(dropped)) {
    who <- rownames(which(dropped, arr.ind = TRUE))
    cli::cli_warn(c(
      "{sum(dropped)} entr{?y/ies} in {.arg predictors} name a variable that \\
       is not generated until later in the visit sequence.",
      i = "Dropped for {.field {unique(who)}}; a single ordered pass can \\
           condition only on what already exists."
    ))
  }
  (want & pm) * 1L
}

# @keywords internal
.syn_resolve_method <- function(method, meth, kind) {
  ok <- c("regression", "classification", "sample", "")
  if (is.null(names(method)) && length(method) == 1L) {
    method <- stats::setNames(rep(method, length(meth)), names(meth))
  }
  if (is.null(names(method))) {
    cli::cli_abort("{.arg method} must be a single string or a named vector.")
  }
  unknown <- setdiff(names(method), names(meth))
  if (length(unknown)) {
    cli::cli_abort("{.arg method} names column{?s} not in the data: \\
                    {.val {unknown}}.")
  }
  bad <- setdiff(unique(method), ok)
  if (length(bad)) {
    cli::cli_abort(c("Unknown method{?s} {.val {bad}}.",
                     i = "Use one of {.val {ok}}."))
  }
  for (v in names(method)) {
    want <- method[[v]]
    if (identical(want, "classification") && kind[[v]] != "categorical") {
      cli::cli_abort("{.field {v}} is continuous; it cannot be synthesised \\
                      by classification.")
    }
    if (identical(want, "regression") && kind[[v]] == "categorical") {
      cli::cli_abort("{.field {v}} is categorical; it cannot be synthesised \\
                      by regression.")
    }
    meth[[v]] <- want
  }
  meth
}

#' Synthetic data with a tabular foundation model
#'
#' Sequential (fully conditional) synthesis, the way `synthpop::syn()`
#' does it, with these models supplying the conditionals. Each variable in
#' the visit sequence is modelled on the **real** data given the variables
#' before it, then drawn at the **synthetic** values of those variables
#' generated earlier in the same pass. The first variable is a bootstrap
#' of its real marginal. One pass, no iteration.
#'
#' This is the generative sibling of [tabfound_impute()]: same models,
#' same heads, a different contract. Imputation fills the missing cells
#' conditioning on everything else and sweeps to a stationary
#' distribution; synthesis replaces every cell conditioning only on what
#' has already been generated, in one ordered pass.
#'
#' # Sampling modes
#'
#' A CART leaf is a set of observed values, so `synthpop::syn.cart()`
#' cannot produce an unobserved value, leave the observed range, or break
#' integrality. A bar distribution is a continuum and can do all three.
#' `draw_mode` is that tradeoff, and it belongs in the write-up rather
#' than in a default nobody looks at:
#'
#' * `"predictive"` — return the draw, restoring integrality and (with
#'   `clamp`) the observed range. Smooth and genuinely novel values, low
#'   exact-replication risk; point masses get smeared.
#' * `"pmm"` — draw, then return a real donor from the `donors` observed
#'   values nearest the draw. CART's support semantics exactly, at the
#'   highest replication-disclosure risk.
#' * `"rank"` — draw for every row, then map the draws onto the observed
#'   order statistics by rank. Reproduces the marginal exactly, which also
#'   means that part of the fidelity is by construction rather than by
#'   merit. "Exactly" means the *context's* marginal, so under
#'   `proper = TRUE` it is the bootstrapped one.
#'
#' `clamp` interacts with all of this: it holds `"predictive"` draws
#' inside the observed range by flooring and capping them, which piles the
#' excess mass onto the observed minimum and maximum. That is a boundary
#' artefact, not a modelled point mass — if a variable has a genuine spike
#' at its minimum, say it with `cont_na` rather than letting `clamp`
#' manufacture one.
#'
#' Categorical variables need no such choice: a softmax over levels and
#' one categorical draw per row already *is* the analogue of drawing from
#' a leaf's class proportions.
#'
#' # Properness
#'
#' These models draw each query row independently given a fixed context,
#' so between-synthesis variance carries predictive noise and no parameter
#' uncertainty — the same defect that makes an imputer improper under
#' Rubin's rules. The synthetic-data variance estimators
#' (`synthpop::lm.synds()` and friends) assume the mechanism propagates
#' it. `proper = TRUE` bootstraps the context rows before conditioning,
#' once per synthesis, which is exactly synthpop's definition of
#' properness, and is the default here. It costs one fit per synthesis
#' instead of one fit shared by all of them.
#'
#' # Missing and special values
#'
#' `NA` is a value to reproduce, not a hole to fill. In a categorical
#' variable it becomes a level the classifier draws like any other. In a
#' continuous variable it is modelled in two parts — a categorical model
#' for *which* special value (or "a real number"), then the regressor for
#' the real numbers — because a bar distribution cannot put mass on a
#' point. `cont_na` extends that to any other spike, such as a zero-
#' inflated income variable: `cont_na = list(income = 0)`.
#'
#' # Scope
#'
#' The context is the real data at generation time. This is a fully
#' conditional synthesiser, not a disclosure-control method, and it
#' supports no differential-privacy claim. Assess replication and
#' attribute disclosure before releasing anything —
#' `synthpop::replicated.uniques()` and `synthpop::disclosure()` take the
#' result via [as_synds()].
#'
#' @param data A data frame (or matrix) to synthesise from.
#' @param m Number of synthetic data sets.
#' @param models A [tabfound_models()] handle. A model reference or a
#'   single fitted model object is accepted too and wrapped for you; when
#'   `NULL`, `...` is passed to [tabfound_models()], so
#'   `tabfound_syn(data, model = "...")` works.
#' @param k Rows per synthetic data set. Defaults to `nrow(data)`.
#'   Note that `k != nrow(data)` changes which variance estimator applies
#'   downstream.
#' @param visit_sequence Column names (or indices) in the order they are
#'   synthesised. Defaults to the column order. With no iteration to wash
#'   it out, the order matters — quantify its effect rather than assuming
#'   it away.
#' @param predictors Optional 0/1 matrix, rows and columns both named by
#'   the columns of `data`, in mice's `predictorMatrix` layout:
#'   `predictors[v, w] == 1` means `w` is used to synthesise `v`. Entries
#'   pointing at variables not yet generated are dropped with a warning.
#'   Defaults to everything already available.
#' @param method Optional per-variable override. A single string, or a
#'   named vector indexed by column name. One of `"regression"`,
#'   `"classification"`, `"sample"` (a bootstrap of the observed values,
#'   no model) or `""` (not synthesised — the real column is carried
#'   through, which needs `k == nrow(data)`).
#' @param draw How to turn a regressor's prediction into a draw; see
#'   [tabfound_impute()]. Never a point prediction: a posterior mean would
#'   make every synthetic record a fitted value and collapse the
#'   within-conditional variance.
#' @param draw_mode Continuous sampling mode: `"predictive"`, `"pmm"` or
#'   `"rank"`. See Details.
#' @param donors Donor pool size for `draw_mode = "pmm"`.
#' @param smoothing Apply kernel smoothing to donor draws, as synthpop's
#'   `smoothing = "density"` does. Ignored for `"predictive"`, which is
#'   already smooth.
#' @param clamp Clamp `"predictive"` draws to the observed range.
#' @param proper Bootstrap the context rows once per synthesis. See
#'   Details; `TRUE` by default, unlike `synthpop::syn()`.
#' @param cont_na Named list of extra values per column to treat as point
#'   masses rather than as draws from the continuous part, e.g.
#'   `list(income = 0)`. `NA` is always treated this way.
#' @param quantile_grid Number of quantile levels for `draw = "quantile"`.
#' @param seed Optional integer seed.
#' @param verbose Show a progress bar.
#' @param ... Passed to [tabfound_models()] when `models` is `NULL`.
#' @return An object of class `tabfound_syn`:
#'   * `syn` — list of `m` synthetic data frames
#'   * `data` — the input, unchanged
#'   * `method`, `m`, `k`, `n`, `visit_sequence`, `predictors`,
#'     `draw_mode`, `proper`, `seed`, `call`
#' @seealso [as_synds()] for the synthpop toolchain, [syn.tabfound()] to
#'   let synthpop drive instead, [tabfound_synthetic()] to pull the frames
#'   out, and [tabfound_impute()] for the imputation sibling.
#' @examples
#' \dontrun{
#' mods <- tabfound_models(classifier = "path/to/tabpfn-v2.5-clf",
#'                         regressor  = "path/to/tabpfn-v2.5-reg")
#' sds <- tabfound_syn(iris, m = 5, models = mods, seed = 1)
#'
#' # Hand it to synthpop for utility, disclosure and inference.
#' synthpop::compare(as_synds(sds), iris)
#' summary(synthpop::lm.synds(Sepal.Length ~ Petal.Length, as_synds(sds)))
#' }
#' @export
tabfound_syn <- function(data, m = 1L, models = NULL, k = NULL,
                         visit_sequence = NULL, predictors = NULL,
                         method = NULL,
                         draw = c("auto", "sample", "grid", "quantile",
                                  "residual"),
                         draw_mode = c("predictive", "pmm", "rank"),
                         donors = 5L, smoothing = FALSE, clamp = TRUE,
                         proper = TRUE, cont_na = NULL, quantile_grid = 199L,
                         seed = NULL, verbose = TRUE, ...) {
  cl        <- match.call()
  draw      <- match.arg(draw)
  draw_mode <- match.arg(draw_mode)
  if (!is.null(seed)) set.seed(seed)

  if (is.matrix(data)) data <- as.data.frame(data, stringsAsFactors = FALSE)
  if (!is.data.frame(data)) {
    cli::cli_abort("{.arg data} must be a data frame or a matrix, \\
                    not a {.cls {class(data)[1]}}.")
  }
  if (!nrow(data) || !ncol(data)) cli::cli_abort("{.arg data} is empty.")
  n <- nrow(data)
  k <- as.integer(k %||% n)
  m <- as.integer(m)
  if (is.na(m) || m < 1L) cli::cli_abort("{.arg m} must be a positive integer.")
  if (is.na(k) || k < 1L) cli::cli_abort("{.arg k} must be a positive integer.")
  models <- .as_models(models, ...)

  vars  <- names(data)
  plans <- lapply(vars, function(v) .syn_column_plan(data[[v]], v))
  names(plans) <- vars
  kind <- vapply(plans, `[[`, character(1), "kind")

  visit <- .syn_resolve_visit(visit_sequence, vars)
  meth  <- stats::setNames(rep("", length(vars)), vars)
  meth[visit] <- ifelse(kind[visit] == "categorical", "classification",
                        "regression")
  if (!is.null(method)) meth <- .syn_resolve_method(method, meth, kind)
  visit <- visit[nzchar(meth[visit])]

  carried <- vars[!nzchar(meth)]
  if (length(carried) && k != n) {
    cli::cli_abort(c(
      "{.field {carried}} {?is/are} not synthesised, so the real column{?s} \\
       would have to be carried through.",
      x = "That needs {.code k == nrow(data)}, but {.arg k} is {.val {k}}."
    ))
  }

  avail <- .syn_available(vars, visit, meth)
  pm    <- .syn_resolve_predictors(predictors, vars, avail)
  # No predictors available means no conditional to fit: the honest draw
  # is the marginal, and that is what the first variable in the sequence
  # always gets.
  for (v in visit) {
    if (!sum(pm[v, ])) meth[[v]] <- "sample"
  }

  cont_na <- .syn_resolve_cont_na(cont_na, vars, kind)

  mdf <- data
  for (v in vars) mdf[[v]] <- plans[[v]]$to_model(data[[v]])

  # Improper synthesis shares one context across all m, so all m can share
  # one fit and one forward pass over a stacked query. Properness
  # bootstraps per synthesis, which forces a fit each.
  groups <- if (isTRUE(proper)) as.list(seq_len(m)) else list(seq_len(m))
  ctx_rows <- if (isTRUE(proper)) {
    lapply(seq_len(m), function(j) sample.int(n, n, replace = TRUE))
  } else {
    rep(list(seq_len(n)), m)
  }

  # An all-NA slice of the model frame: right types, right factor levels,
  # `k` rows. Unsynthesised columns hold the real values from the start,
  # so they can be conditioned on like anything else.
  blank <- mdf[rep(NA_integer_, k), , drop = FALSE]
  rownames(blank) <- NULL
  for (v in carried) blank[[v]] <- mdf[[v]]
  syn <- rep(list(blank), m)

  opts <- list(draw = draw, quantile_grid = quantile_grid,
               draw_mode = draw_mode, donors = as.integer(donors),
               smoothing = isTRUE(smoothing), clamp = isTRUE(clamp))

  if (verbose && length(visit)) {
    cli::cli_progress_bar("Synthesising",
                          total = length(groups) * length(visit),
                          .envir = environment())
  }
  for (g in groups) {
    rows <- ctx_rows[[g[[1L]]]]
    for (v in visit) {
      preds <- names(which(pm[v, ] == 1L))
      y <- mdf[[v]][rows]

      if (identical(meth[[v]], "sample") || !length(preds)) {
        # `pool[sample.int(...)]`, not `sample(pool, ...)`, which samples
        # from `1:pool` when the pool happens to be a single number.
        for (j in g) {
          syn[[j]][[v]] <- y[sample.int(length(y), k, replace = TRUE)]
        }
        if (verbose) cli::cli_progress_update(.envir = environment())
        next
      }

      X_ctx <- .encode_predictors(mdf[rows, preds, drop = FALSE])
      X_q   <- .encode_predictors(
        do.call(rbind, lapply(g, function(j) syn[[j]][, preds, drop = FALSE]))
      )
      block <- rep(g, each = k)

      out <- if (identical(kind[[v]], "categorical")) {
        .syn_draw_categorical(y, X_ctx, X_q, models)
      } else {
        .syn_draw_continuous(y, X_ctx, X_q, models, opts, cont_na[[v]], block)
      }
      for (i in seq_along(g)) {
        sel <- seq_len(k) + (i - 1L) * k
        syn[[g[[i]]]][[v]] <- .syn_refactor(out[sel], mdf[[v]])
      }
      if (verbose) cli::cli_progress_update(.envir = environment())
    }
  }
  if (verbose && length(visit)) cli::cli_progress_done(.envir = environment())

  out <- lapply(syn, function(d) {
    for (v in vars) d[[v]] <- plans[[v]]$restore(d[[v]])
    rownames(d) <- NULL
    d
  })

  structure(
    list(data = data, syn = out, m = m, k = k, n = n,
         method = meth, visit_sequence = visit, predictors = pm,
         draw = draw, draw_mode = draw_mode, donors = as.integer(donors),
         smoothing = isTRUE(smoothing), clamp = isTRUE(clamp),
         proper = isTRUE(proper), cont_na = cont_na, seed = seed,
         backend = c(classification = models$source[["classification"]],
                     regression     = models$source[["regression"]]),
         call = cl),
    class = "tabfound_syn"
  )
}

# Draws come back as bare labels or numbers; the synthetic frame has to
# hold the same factor with the same levels as the context, or the next
# variable's predictors address different categories than the model was
# fitted on. This is the highest-probability silent bug in the whole
# design, so the coding happens in exactly one place.
# @keywords internal
.syn_refactor <- function(v, template) {
  if (!is.factor(template)) return(as.numeric(v))
  factor(as.character(v), levels = levels(template))
}

# @keywords internal
.syn_resolve_cont_na <- function(cont_na, vars, kind) {
  out <- stats::setNames(vector("list", length(vars)), vars)
  if (is.null(cont_na)) return(out)
  if (!is.list(cont_na) || is.null(names(cont_na))) {
    cli::cli_abort("{.arg cont_na} must be a named list, e.g. \\
                    {.code list(income = 0)}.")
  }
  unknown <- setdiff(names(cont_na), vars)
  if (length(unknown)) {
    cli::cli_abort("{.arg cont_na} names column{?s} not in the data: \\
                    {.val {unknown}}.")
  }
  for (v in names(cont_na)) {
    if (identical(kind[[v]], "categorical")) {
      cli::cli_abort("{.field {v}} is categorical; {.arg cont_na} applies to \\
                      continuous columns only.")
    }
    out[[v]] <- stats::na.omit(as.numeric(cont_na[[v]]))
  }
  out
}


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

#' @export
print.tabfound_syn <- function(x, ...) {
  cli::cli_text("{.strong tabfound} synthetic data")
  cli::cli_bullets(c(
    "*" = "{.val {x$m}} synthetic data set{?s} of {.val {x$k}} row{?s} \\
           from {.val {x$n}}",
    "*" = "{length(x$visit_sequence)} variable{?s} synthesised: \\
           {.field {utils::head(x$visit_sequence, 6)}}\\
           {if (length(x$visit_sequence) > 6) ' ...' else ''}",
    "*" = "draw mode: {.val {x$draw_mode}}{if (x$smoothing) ' (smoothed)' else ''}",
    "*" = "proper: {.val {x$proper}}",
    i = "{.fn as_synds} for the synthpop toolchain \\
         ({.fn compare}, {.fn utility.gen}, {.fn lm.synds})."
  ))
  invisible(x)
}

#' Extract synthetic data from a `tabfound_syn` object
#'
#' The same contract [tabfound_complete()] has for imputations.
#'
#' @param x A `tabfound_syn` object.
#' @param action `1..m` for a single synthetic data set, `"all"` for a
#'   list of all of them, `"long"` for them stacked with a `.syn` column,
#'   `"stacked"` for them stacked without it.
#' @param ... Unused.
#' @return A data frame, or a list of them for `action = "all"`.
#' @examples
#' \dontrun{
#' tabfound_synthetic(sds, 1)       # first synthetic data set
#' tabfound_synthetic(sds, "long")  # stacked, with .syn
#' }
#' @export
tabfound_synthetic <- function(x, action = 1L, ...) {
  if (!inherits(x, "tabfound_syn")) {
    cli::cli_abort("{.arg x} must be a {.cls tabfound_syn} object.")
  }
  sets <- x$syn
  if (is.numeric(action)) {
    j <- as.integer(action)
    if (length(j) != 1L || is.na(j) || !j %in% seq_len(x$m)) {
      cli::cli_abort("{.arg action} must be one of {.val {seq_len(x$m)}}.")
    }
    return(sets[[j]])
  }
  action <- match.arg(as.character(action), c("all", "long", "stacked"))
  if (identical(action, "all")) {
    names(sets) <- paste0("syn", seq_len(x$m))
    return(sets)
  }
  long <- do.call(rbind, lapply(seq_along(sets), function(i) {
    cbind(.syn = i, sets[[i]], stringsAsFactors = FALSE)
  }))
  rownames(long) <- NULL
  if (identical(action, "stacked")) long <- long[, setdiff(names(long), ".syn")]
  long
}
