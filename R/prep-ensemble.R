# Ensemble generation for TabFM, TabICL and Mitra.
#
# None of these three models is meant to be run once. Each ships an
# sklearn wrapper that builds several *views* of the same table -- a
# different feature order, a different class labelling, a different
# normalisation -- runs the network on each, and combines the results.
# The network is the part this package already had; the views are the
# part this file adds.
#
# The views are not interchangeable. Which permutation member 3 gets is
# decided by `random.Random(random_state)`, so the whole construction is
# reproduced draw for draw on top of `py-random.R`. A "statistically
# equivalent" ensemble would be a different set of predictions.
#
# TabICL and TabFM ship separate generators that look similar and are
# not: TabICL permutes class *labels* while TabFM *shifts* them, TabICL's
# feature permutations come from a Latin square while TabFM's are plain
# random samples, and they assign normalisation methods to members in
# different ways. Both are implemented, separately, on purpose.
#
# References:
#   tabicl/_sklearn/preprocessing.py :: Shuffler, EnsembleGenerator
#   tabfm/src/classifier_and_regressor.py :: FeatureShuffler, EnsembleGenerator
#   autogluon .../mitra/_internal/data/preprocessor.py :: Preprocessor

# ---------------------------------------------------------------------------
# SimpleImputer
# ---------------------------------------------------------------------------

#' Fit sklearn's `SimpleImputer(strategy = "mean")`
#'
#' TabICL's `TransformToNumerical` runs one over the numeric columns
#' before anything else, which is why its network never sees the `NaN` it
#' cannot handle. A column that is *entirely* missing has no mean to
#' impute, and sklearn's default is to drop it rather than invent one —
#' so this returns a `keep` mask alongside the means.
#'
#' @param X_train Numeric matrix.
#' @keywords internal
fit_simple_imputer <- function(X_train) {
  X <- as.matrix(X_train); storage.mode(X) <- "double"
  .reject_infinite(X)
  means <- vapply(seq_len(ncol(X)), function(j) {
    v <- X[, j]; v <- v[!is.na(v)]
    if (!length(v)) NA_real_ else mean(v)
  }, numeric(1))
  list(means = means, keep = !is.na(means))
}

# `Inf` is not a missing value and is not treated as one.
#
# sklearn draws the line here too: its estimators take `NaN` where they
# advertise it and reject `Inf` outright, so the reference raises
# `ValueError: Input X contains infinity` before any of this runs. The
# reason is that an infinity is not recoverable by imputation -- it
# poisons a column mean, and from there the standard scaler, and the
# damage is silent. Better to say so at the point the user can act on it.
#
# Mitra is the exception and handles its own: its preprocessor maps
# non-finite values to zero, so it never comes through here.
# @keywords internal
.reject_infinite <- function(X) {
  bad <- is.infinite(X)
  if (!any(bad)) return(invisible(TRUE))
  cols <- unname(unique(which(bad, arr.ind = TRUE)[, "col"]))
  cli::cli_abort(c(
    # `qty()` because a lone index like `2` would otherwise be read as a
    # count and pluralise the noun after it.
    "Predictors contain infinite values, in {cli::qty(length(cols))} \\
     column{?s} {cols}.",
    x = "The reference implementation rejects these rather than imputing \\
         them, and so does this backend: an infinity poisons the column \\
         mean and every scaler downstream of it, silently.",
    i = "Replace them first -- with {.code NA} to have them imputed, or \\
         with a finite bound if they encode one."
  ))
  invisible(FALSE)
}

#' @rdname fit_simple_imputer
#' @param X Numeric matrix.
#' @param fit Result of [fit_simple_imputer()].
#' @keywords internal
transform_simple_imputer <- function(X, fit) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  # Checked at predict time too: sklearn re-validates on `transform`, and
  # an infinity arriving only in the test set is just as damaging.
  .reject_infinite(X)
  for (j in seq_len(ncol(X))) {
    na <- is.na(X[, j])
    if (any(na)) X[na, j] <- fit$means[[j]]
  }
  X[, fit$keep, drop = FALSE]
}


# ---------------------------------------------------------------------------
# TabICL's Shuffler
# ---------------------------------------------------------------------------

# `_rls`, unrolled.
#
# The reference builds a random Latin square by recursion: pick a symbol,
# recurse on the rest, then splice the symbol back in along the diagonal.
# The recursion depth is the number of elements, which is why the
# reference wraps it in a `setrecursionlimit(100000)`. Unrolling it costs
# nothing and avoids inheriting that problem -- but the order of the
# `choice` calls has to be preserved exactly, because they come off the
# same stream everything else does. The recursion draws from the largest
# pool first, so the loop below does too.
# @keywords internal
.latin_rls <- function(n, rng) {
  pool <- as.list(seq_len(n) - 1L)
  chosen <- vector("list", n - 1L)
  for (level in seq.int(n, 2L)) {
    sym <- rng$choice(pool)
    # `list.remove` semantics: drop the first occurrence.
    pool[[which(vapply(pool, identical, logical(1), sym))[1L]]] <- NULL
    chosen[[level - 1L]] <- sym
  }
  square <- list(unlist(pool))          # the single symbol left over
  for (level in seq.int(2L, n)) {
    sym <- chosen[[level - 1L]]
    square[[level]] <- square[[1L]]     # append a copy of the first row
    for (i in seq_len(level)) {
      # Python inserts at 0-based position `i - 1`.
      square[[i]] <- append(square[[i]], sym, after = i - 1L)
    }
  }
  square
}

# `_shuffle_transpose_shuffle`: shuffle rows, transpose, shuffle again.
# @keywords internal
.latin_squares <- function(n, rng) {
  square <- .latin_rls(n, rng)
  square <- rng$shuffle(square)
  m <- do.call(rbind, square)
  trans <- lapply(seq_len(ncol(m)), function(j) m[, j])
  rng$shuffle(trans)
}

#' Generate permutation patterns for ensemble diversity
#'
#' A port of TabICL's `Shuffler`. Returns a list of 0-based index
#' vectors.
#'
#' The four methods differ in more than flavour. `latin` gives a set of
#' permutations in which every element visits every position exactly once
#' across the set, which spreads a transformer's position bias evenly;
#' `shift` gives the `n` circular rotations; `random` draws independent
#' permutations, exhaustively when there are five or fewer elements.
#'
#' @param n_elements Number of things to permute.
#' @param method One of `"none"`, `"shift"`, `"random"`, `"latin"`.
#' @param n_estimators How many patterns are wanted. Some methods return
#'   a different number — `shift` always returns `n_elements`, `latin`
#'   likewise — and the caller is expected to cope.
#' @param max_elements_for_latin Above this, `latin` silently becomes
#'   `random`; building the square is quadratic.
#' @param random_state Seed for the generator, which is created fresh
#'   here exactly as the reference creates it inside `shuffle()`.
#' @keywords internal
py_shuffler <- function(n_elements, method = "latin", n_estimators = 8L,
                        max_elements_for_latin = 4000L, random_state = NULL) {
  rng <- py_random(random_state)
  indices <- seq_len(n_elements) - 1L

  if (n_elements > max_elements_for_latin && identical(method, "latin")) {
    method <- "random"
  }
  if (identical(method, "none") || n_estimators == 1L) return(list(indices))

  switch(
    method,
    "shift" = lapply(seq_len(n_elements) - 1L, function(i) {
      # Python's `indices[-i:] + indices[:-i]`, where `-0` slices the
      # whole list and `[:-0]` slices nothing, so i = 0 is the identity.
      if (i == 0L) indices else c(indices[(n_elements - i + 1L):n_elements],
                                  indices[seq_len(n_elements - i)])
    }),
    "random" = {
      if (n_elements <= 5L) {
        all_perms <- .all_permutations(indices)
        rng$sample(all_perms, min(n_estimators, length(all_perms)))
      } else {
        lapply(seq_len(n_estimators), function(i) rng$sample(indices, n_elements))
      }
    },
    "latin" = .latin_squares(n_elements, rng),
    cli::cli_abort("Unknown shuffle method: {.val {method}}.")
  )
}

# `itertools.permutations` order: lexicographic on positions.
# @keywords internal
.all_permutations <- function(x) {
  n <- length(x)
  if (n <= 1L) return(list(x))
  out <- list()
  for (i in seq_len(n)) {
    rest <- .all_permutations(x[-i])
    for (r in rest) out[[length(out) + 1L]] <- c(x[i], r)
  }
  out
}

# `itertools.product(a, b)`, a-major.
# @keywords internal
.product2 <- function(a, b) {
  out <- vector("list", length(a) * length(b))
  k <- 1L
  for (i in seq_along(a)) for (j in seq_along(b)) {
    out[[k]] <- list(a[[i]], b[[j]]); k <- k + 1L
  }
  out
}


# ---------------------------------------------------------------------------
# TabICL's EnsembleGenerator
# ---------------------------------------------------------------------------

#' Build TabICL's ensemble of dataset views
#'
#' Drops single-valued columns, builds `n_estimators` (feature
#' permutation, class permutation, normalisation) triples, and fits one
#' preprocessing pipeline per normalisation method in use.
#'
#' Members are grouped by normalisation method, because the pipeline is
#' the expensive part and every member sharing a method shares its fit.
#' That grouping is also the order predictions come back in, so the class
#' permutations are stored the same way and stay aligned with them.
#'
#' Fewer than `n_estimators` members can come out. With two features and
#' two classes there are only so many distinct views, and the reference
#' takes however many the product yields rather than repeating itself.
#'
#' @param X Numeric matrix of training predictors, already imputed.
#' @param y Integer vector of 0-based class ids (classification) or
#'   numeric targets (regression).
#' @param classification Logical.
#' @param n_estimators Number of ensemble members requested.
#' @param norm_methods Character vector of normalisation methods.
#' @param feat_shuffle_method,class_shuffle_method Passed to [py_shuffler()].
#' @param outlier_threshold Passed to [fit_preprocessing_pipeline()].
#' @param random_state Integer seed.
#' @param quantile_subsample See [fit_sk_quantile_transformer()].
#' @keywords internal
tabicl_ensemble_fit <- function(X, y, classification,
                                n_estimators = 8L,
                                norm_methods = c("none", "power"),
                                feat_shuffle_method = "latin",
                                class_shuffle_method = "shift",
                                outlier_threshold = 4.0,
                                random_state = 42L,
                                quantile_subsample = NULL) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  n_estimators <- as.integer(n_estimators)

  filter_ <- fit_unique_feature_filter(X)
  X <- transform_unique_feature_filter(X, filter_)
  if (ncol(X) == 0L) {
    cli::cli_abort("Every predictor is constant; there is nothing to learn from.")
  }
  n_features <- ncol(X)
  n_classes <- if (classification) length(unique(y)) else 0L

  rng <- py_random(random_state)

  feat_shuffles <- py_shuffler(n_features, feat_shuffle_method, n_estimators,
                               random_state = random_state)
  y_patterns <- if (classification) {
    py_shuffler(n_classes, class_shuffle_method, n_estimators,
                random_state = random_state)
  } else {
    list(NULL)
  }

  configs <- .product2(feat_shuffles, y_patterns)
  configs <- rng$shuffle(configs)
  norm_configs <- .product2(configs, as.list(norm_methods))
  norm_configs <- norm_configs[seq_len(min(n_estimators, length(norm_configs)))]

  # The reference groups by `set(...)`, whose iteration order is a
  # Python implementation detail. Order of *methods* cannot change any
  # prediction -- members are averaged -- so first-appearance order is
  # used here, which at least is stable.
  used <- unique(vapply(norm_configs, function(c) c[[2L]], character(1)))

  ensemble <- list(); preprocessors <- list()
  for (m in used) {
    keep <- vapply(norm_configs, function(c) identical(c[[2L]], m), logical(1))
    ensemble[[m]] <- lapply(norm_configs[keep], function(c) c[[1L]])
    preprocessors[[m]] <- fit_preprocessing_pipeline(
      X, normalization_method = m, outlier_threshold = outlier_threshold,
      quantile_subsample = quantile_subsample
    )
  }

  list(
    filter = filter_, X = X, y = y, classification = classification,
    n_features = n_features, n_classes = n_classes,
    norm_methods = used, ensemble = ensemble, preprocessors = preprocessors,
    n_members = length(norm_configs)
  )
}

#' TabICL's ensemble views for a test set, one at a time
#'
#' Each member is a `list(X, y)` with `X` the train-then-test rows in
#' that member's feature order and `y` its relabelled training targets,
#' in the same flattened method-then-member order the class permutations
#' are stored in.
#'
#' The iterator exists because the eager version holds every member's
#' full `(n_train + n_test) x p` matrix in one list -- eight for TabICL,
#' thirty-two for TabFM, hundreds of megabytes on a table of any size --
#' when the consumer only ever looks at one. What is genuinely shared
#' between members of a normalisation method, the transformed matrix
#' itself, is still computed once and held; only the per-member column
#' slice is deferred. Members are cheapest to visit in order, which is
#' what every caller does.
#'
#' @param gen Result of [tabicl_ensemble_fit()].
#' @param X_test Numeric matrix of test predictors, already imputed.
#' @return `list(n, get)`: the member count, and a function of the member
#'   index.
#' @keywords internal
tabicl_ensemble_iter <- function(gen, X_test) {
  X_test <- transform_unique_feature_filter(as.matrix(X_test), gen$filter)
  index <- .ensemble_flat_index(gen)
  cur_m <- NULL; cur_X <- NULL

  get <- function(i) {
    m <- index$method[[i]]
    if (!identical(cur_m, m)) {
      pp <- gen$preprocessors[[m]]
      # Drop the previous method's matrix before building the next.
      cur_X <<- NULL
      cur_X <<- rbind(pp$X_transformed,
                      transform_preprocessing_pipeline(X_test, pp))
      cur_m <<- m
    }
    cfg <- gen$ensemble[[m]][[index$j[[i]]]]
    feat <- cfg[[1L]]; y_pattern <- cfg[[2L]]
    list(
      X = cur_X[, feat + 1L, drop = FALSE],
      y = if (gen$classification) y_pattern[as.integer(gen$y) + 1L] else gen$y,
      class_shuffle = y_pattern,
      feat = feat,
      norm = m
    )
  }
  list(n = nrow(index), get = get)
}

# The (method, member) pairs in flattened order, which is the order the
# class permutations were stored in.
# @keywords internal
.ensemble_flat_index <- function(gen) {
  rows <- lapply(gen$norm_methods, function(m) {
    k <- length(gen$ensemble[[m]])
    if (!k) return(NULL)
    data.frame(method = rep(m, k), j = seq_len(k), stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}

#' @rdname tabicl_ensemble_iter
#' @keywords internal
tabicl_ensemble_transform <- function(gen, X_test) {
  it <- tabicl_ensemble_iter(gen, X_test)
  lapply(seq_len(it$n), it$get)
}


# ---------------------------------------------------------------------------
# TabFM's EnsembleGenerator
# ---------------------------------------------------------------------------

#' Build TabFM's ensemble of dataset views
#'
#' Same idea as TabICL's, different construction throughout.
#'
#' Feature orders come from plain random samples rather than a Latin
#' square. Class relabelling is a *rotation* by an offset drawn without
#' replacement from `0:(n_classes - 1)` and then cycled, not an arbitrary
#' permutation. Normalisation methods are handed out round-robin over the
#' shuffled member list, and grouping is by the caller's method order
#' rather than by set iteration.
#'
#' Categorical columns are tracked through the constant-column filter so
#' the network can be told which they are — see the `cat_mask` argument
#' of [tabfm_model()].
#'
#' @param X Numeric matrix of training predictors.
#' @param y Targets: 0-based class ids, or numerics for regression.
#' @param task `"classification"` or `"regression"`.
#' @param n_estimators Number of members.
#' @param norm_methods Character vector of normalisation methods.
#' @param feat_shuffle_method `"random"` or `"none"`.
#' @param class_shift Apply the class rotation (classification only).
#' @param cat_features 1-based indices of categorical columns, or `NULL`.
#' @param outlier_threshold Passed to [fit_preprocessing_pipeline()].
#' @param max_num_features Cap on columns sampled per member.
#' @param max_num_rows Cap on training rows sampled per member.
#' @param random_state Integer seed.
#' @param quantile_subsample See [fit_sk_quantile_transformer()].
#' @keywords internal
tabfm_ensemble_fit <- function(X, y, task = "classification",
                               n_estimators = 32L,
                               norm_methods = c("none", "power"),
                               feat_shuffle_method = "random",
                               class_shift = TRUE,
                               cat_features = NULL,
                               outlier_threshold = 4.0,
                               max_num_features = 500L,
                               max_num_rows = NULL,
                               random_state = 42L,
                               quantile_subsample = NULL) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  n_estimators <- as.integer(n_estimators)
  classification <- identical(task, "classification")

  filter_ <- fit_unique_feature_filter(X)
  X <- transform_unique_feature_filter(X, filter_)
  if (ncol(X) == 0L) {
    cli::cli_abort("Every predictor is constant; there is nothing to learn from.")
  }

  # Categorical indices survive the filter by riding a mask through it.
  cat_idx <- integer(0)
  if (!is.null(cat_features) && length(cat_features)) {
    mask <- rep(FALSE, filter_$n_features_in)
    mask[cat_features] <- TRUE
    cat_idx <- which(mask[filter_$keep]) - 1L        # 0-based
  }

  rng <- py_random(random_state)
  n_original <- ncol(X)
  n_cols <- if (is.null(max_num_features)) n_original
            else min(n_original, as.integer(max_num_features))
  is_subsampling <- n_cols < n_original

  shuffle_patterns <- if (is_subsampling) {
    lapply(seq_len(n_estimators), function(i) {
      cols <- rng$sample(seq_len(n_original) - 1L, n_cols)
      rng$sample(cols, length(cols))
    })
  } else {
    p <- .tabfm_feature_shuffles(n_original, feat_shuffle_method, n_estimators,
                                 random_state)
    if (length(p) < n_estimators) {
      cycles <- ceiling(n_estimators / length(p))
      p <- rep(p, cycles)[seq_len(n_estimators)]
    }
    p
  }

  n_classes <- if (classification) length(unique(y)) else 0L
  shift_offsets <- if (classification && class_shift && n_estimators > 1L &&
                       n_classes > 1L) {
    base <- rng$sample(seq_len(n_classes) - 1L, n_classes)
    cycles <- ceiling(n_estimators / length(base))
    rep(base, cycles)[seq_len(n_estimators)]
  } else {
    rep(0L, n_estimators)
  }

  n_rows <- if (is.null(max_num_rows)) nrow(X)
            else min(nrow(X), as.integer(max_num_rows))
  row_patterns <- if (n_rows < nrow(X)) {
    lapply(seq_len(n_estimators),
           function(i) rng$sample(seq_len(nrow(X)) - 1L, n_rows))
  } else {
    vector("list", n_estimators)   # all NULL
  }

  configs <- lapply(seq_len(n_estimators), function(i) {
    list(feat = shuffle_patterns[[i]], shift = shift_offsets[[i]],
         rows = row_patterns[[i]])
  })
  configs <- rng$shuffle(configs)

  # Round-robin over the requested methods, then grouped in that order.
  cycles <- ceiling(n_estimators / length(norm_methods))
  assigned <- rep(norm_methods, cycles)[seq_len(n_estimators)]

  ensemble <- list(); preprocessors <- list(); used <- character(0)
  for (m in norm_methods) {
    idx <- which(assigned == m)
    if (!length(idx)) next
    used <- c(used, m)
    ensemble[[m]] <- configs[idx]
    preprocessors[[m]] <- fit_preprocessing_pipeline(
      X, normalization_method = m, outlier_threshold = outlier_threshold,
      quantile_subsample = quantile_subsample
    )
  }

  list(
    filter = filter_, X = X, y = y, classification = classification,
    n_features = ncol(X), n_classes = n_classes, cat_features = cat_idx,
    norm_methods = used, ensemble = ensemble, preprocessors = preprocessors,
    n_members = n_estimators
  )
}

# TabFM's `FeatureShuffler`, which is not TabICL's `Shuffler`: only
# `random` and `none`, and no Latin squares.
# @keywords internal
.tabfm_feature_shuffles <- function(n_features, method, n_estimators,
                                    random_state) {
  rng <- py_random(random_state)
  indices <- seq_len(n_features) - 1L
  if (identical(method, "none") || n_estimators == 1L) return(list(indices))
  if (!identical(method, "random")) {
    cli::cli_abort("Unknown method: {.val {method}}. Use {.val random} or {.val none}.")
  }
  if (n_features <= 5L) {
    all_perms <- .all_permutations(indices)
    rng$sample(all_perms, min(n_estimators, length(all_perms)))
  } else {
    lapply(seq_len(n_estimators), function(i) rng$sample(indices, n_features))
  }
}

#' TabFM's ensemble views for a test set, one at a time
#'
#' The 32-member counterpart of [tabicl_ensemble_iter()]; see there for
#' why the members are produced on demand rather than as a list.
#'
#' @param gen Result of [tabfm_ensemble_fit()].
#' @param X_test Numeric matrix of test predictors.
#' @return `list(n, get)`. Each member has `X`, `y`, `shift` and the
#'   `cat_mask` for its own column order.
#' @keywords internal
tabfm_ensemble_iter <- function(gen, X_test) {
  X_test <- transform_unique_feature_filter(as.matrix(X_test), gen$filter)
  cat_flag <- rep(FALSE, gen$n_features)
  if (length(gen$cat_features)) cat_flag[gen$cat_features + 1L] <- TRUE
  index <- .ensemble_flat_index(gen)
  cur_m <- NULL; cur_test <- NULL

  get <- function(i) {
    m <- index$method[[i]]
    if (!identical(cur_m, m)) {
      cur_test <<- NULL
      cur_test <<- transform_preprocessing_pipeline(X_test,
                                                    gen$preprocessors[[m]])
      cur_m <<- m
    }
    cfg <- gen$ensemble[[m]][[index$j[[i]]]]
    # Each member may subsample rows, so unlike TabICL the train block is
    # per-member too; what is shared is the transformed *test* block.
    rows <- if (is.null(cfg$rows)) seq_len(nrow(gen$X)) else cfg$rows + 1L
    X_variant <- rbind(gen$preprocessors[[m]]$X_transformed[rows, , drop = FALSE],
                       cur_test)
    y_train <- gen$y[rows]
    list(
      X = X_variant[, cfg$feat + 1L, drop = FALSE],
      y = if (gen$classification) {
        (as.integer(y_train) + cfg$shift) %% gen$n_classes
      } else {
        y_train
      },
      shift = cfg$shift,
      cat_mask = cat_flag[cfg$feat + 1L],
      feat = cfg$feat,
      norm = m
    )
  }
  list(n = nrow(index), get = get)
}

#' @rdname tabfm_ensemble_iter
#' @keywords internal
tabfm_ensemble_transform <- function(gen, X_test) {
  it <- tabfm_ensemble_iter(gen, X_test)
  lapply(seq_len(it$n), it$get)
}


# ---------------------------------------------------------------------------
# Mitra's Preprocessor
# ---------------------------------------------------------------------------

#' Fit Mitra's preprocessor
#'
#' Much the smallest of the three, and the only one that is not an
#' ensemble generator: `MitraClassifier` defaults to `n_estimators = 1`,
#' and what diversity it has comes from each member drawing its own
#' random sign flips rather than from a designed set of views.
#'
#' With AutoGluon's own defaults (`use_quantile_transformer = False`,
#' `use_feature_count_scaling = False`, `use_random_transforms = False`,
#' `shuffle_classes = False`, `shuffle_features = False`) the pipeline is:
#' impute missing values with the training column means, drop constant
#' columns, flip a random per-column sign, and map a regression target to
#' `[0, 1]` — possibly mirrored.
#'
#' The imputation is why this matters more than its size suggests. Mitra
#' does not propagate `NaN`, it *absorbs* it: one missing value makes a
#' column's quantiles all-`NaN`, every value buckets to zero, and the
#' column silently vanishes. The reference never hits that because it
#' imputes first. Until now this port did not.
#'
#' @param X Numeric matrix of training predictors.
#' @param y Targets.
#' @param task `"classification"` or `"regression"`.
#' @param random_mirror_x Flip a random sign per column.
#' @param random_mirror_regression Possibly replace `y` with `1 - y`.
#' @param seed Integer seed for the mirrors, or `NULL`. The reference
#'   draws these from NumPy's *global* generator and never seeds it, so
#'   its mirrors are not reproducible even run to run; a seed is offered
#'   here because an R user can reasonably expect `set.seed()` to work.
#' @keywords internal
mitra_preprocessor_fit <- function(X, y, task = "classification",
                                   random_mirror_x = TRUE,
                                   random_mirror_regression = TRUE,
                                   seed = NULL) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  classification <- identical(task, "classification")

  pre_nan_mean <- .col_nanmean(X)
  X_imp <- X
  for (j in seq_len(ncol(X))) {
    na <- is.na(X_imp[, j])
    if (any(na)) X_imp[na, j] <- pre_nan_mean[[j]]
  }

  # `len(np.unique(col)) == 1`, on the imputed data.
  singular <- vapply(seq_len(ncol(X_imp)),
                     function(j) length(unique(X_imp[, j])) == 1L, logical(1))
  keep <- !singular
  if (!any(keep)) {
    cli::cli_abort("Every predictor is constant; there is nothing to learn from.")
  }

  if (!is.null(seed)) set.seed(seed)
  n_keep <- sum(keep)
  mirror <- if (random_mirror_x) sample(c(1, -1), n_keep, replace = TRUE)
            else rep(1, n_keep)

  scaler <- NULL; mirror_y <- FALSE
  if (!classification) {
    scaler <- fit_minmax_scaler(y)
    if (identical(scaler$range, 1) && diff(range(y)) == 0) {
      cli::cli_abort("The target is constant; regression makes no sense.")
    }
    if (random_mirror_regression) mirror_y <- sample(c(TRUE, FALSE), 1L)
  }

  list(pre_nan_mean = pre_nan_mean, keep = keep, mirror = mirror,
       classification = classification, scaler = scaler, mirror_y = mirror_y,
       random_mirror_regression = random_mirror_regression)
}

#' @rdname mitra_preprocessor_fit
#' @param fit Result of [mitra_preprocessor_fit()].
#' @keywords internal
mitra_preprocessor_transform_X <- function(X, fit) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  for (j in seq_len(ncol(X))) {
    na <- is.na(X[, j])
    if (any(na)) X[na, j] <- fit$pre_nan_mean[[j]]
  }
  X <- X[, fit$keep, drop = FALSE]
  X <- sweep(X, 2L, fit$mirror, `*`)
  X[!is.finite(X)] <- 0
  X
}

#' @rdname mitra_preprocessor_fit
#' @keywords internal
mitra_preprocessor_transform_y <- function(y, fit) {
  if (fit$classification) return(as.numeric(y))
  y <- apply_minmax_scaler(y, fit$scaler)
  if (fit$random_mirror_regression && fit$mirror_y) y <- 1 - y
  y
}

#' @rdname mitra_preprocessor_fit
#' @param pred Numeric vector of predictions in the model's target space.
#' @keywords internal
mitra_preprocessor_invert_y <- function(pred, fit) {
  if (fit$classification) return(pred)
  if (fit$random_mirror_regression && fit$mirror_y) pred <- 1 - pred
  invert_minmax_scaler(pred, fit$scaler)
}
