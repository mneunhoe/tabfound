# Causilo 1.5 preprocessing and ensemble recipe.
#
# A port of `causilo/v1_5/data/` (and the v1.0 rank-Gaussian normaliser it
# reuses) from Nums AI's `causilo` 1.5.0, Apache-2.0. Everything here is
# fitted on the context rows, in float64, and is plain R data; the views
# the network reads are float32.
#
#   1. target: classes -> 0..K-1 (sorted labels), or (y - mean) / sd_pop
#   2. features -> one float64 table [categorical codes | log counts | numeric]
#   3. canonical columns: drop the non-varying ones, order the rest by the
#      bytes of their values, group byte-identical columns
#   4. ensemble plan (CPython `random.Random(seed)`): per member a sign per
#      column group, a class shift (or a target sign) and a feature view
#   5. the feature views in use: 0 Base, 1 Robust, 2 Rank-Gaussian
#
# There is no feature permutation: column order is canonical and shared.


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

# Exactly rounded sum (Shewchuk), as Python's `math.fsum`.
# @keywords internal
.causilo_fsum <- function(x) {
  partials <- numeric(0)
  for (v in x) {
    i <- 0L
    for (y in partials) {
      if (abs(v) < abs(y)) { t <- v; v <- y; y <- t }
      hi <- v + y
      lo <- y - (hi - v)
      if (lo != 0) { i <- i + 1L; partials[i] <- lo }
      v <- hi
    }
    partials <- c(partials[seq_len(i)], v)
  }
  if (!length(partials)) return(0)
  # Sum the non-overlapping partials from the top, with the round-half-even
  # correction CPython applies.
  n <- length(partials)
  hi <- partials[n]
  lo <- 0
  if (n > 1L) {
    k <- n - 1L
    while (k >= 1L) {
      x0 <- hi; y0 <- partials[k]; k <- k - 1L
      hi <- x0 + y0
      yr <- hi - x0
      lo <- y0 - yr
      if (lo != 0) break
    }
    if (k >= 1L && ((lo < 0 && partials[k] < 0) || (lo > 0 && partials[k] > 0))) {
      y0 <- lo * 2
      x0 <- hi + y0
      yr <- x0 - hi
      if (y0 == yr) hi <- x0
    }
  }
  hi
}

# numpy's `np.interp(x, xp, fp)` for increasing `xp`: clamped at the ends.
# @keywords internal
.causilo_np_interp <- function(x, xp, fp) {
  if (length(xp) == 1L) return(rep(fp[[1]], length(x)))
  stats::approx(xp, fp, xout = x, rule = 2, ties = "ordered")$y
}

# scipy's `ndtri`.
.causilo_ndtri <- function(p) stats::qnorm(p)

# sklearn `StandardScaler` statistics of each column (no missing cells):
# the mean, and the scale with near-constant columns set to 1.
# @keywords internal
.sk_standard_scaler <- function(x) {
  n <- nrow(x)
  mean <- colSums(x) / n
  centred <- sweep(x, 2L, mean)
  correction <- colSums(centred)
  var <- (colSums(centred^2) - correction^2 / n) / n
  eps <- .Machine$double.eps
  constant <- var <= n * eps * var + (n * mean * eps)^2
  scale <- sqrt(var)
  scale[constant] <- 1
  list(mean = mean, scale = scale)
}

# `clip(v) + asinh(v - clip(v))` where outside `[lower, upper]`.
# @keywords internal
.causilo_compress <- function(v, lower, upper) {
  lo <- rep(lower, each = nrow(v)); hi <- rep(upper, each = nrow(v))
  res <- pmin(pmax(v, lo), hi)
  excess <- v - res
  out <- !is.na(excess) & excess != 0
  res[out] <- res[out] + asinh(excess[out])
  res[is.nan(v)] <- NaN
  res
}

# v1.5's two-pass four-sigma bounds (`normalization.outlier_bounds`).
# Missing cells count for nothing; `ddof = 1` when the table has more
# than one row; spreads floored at 1e-6.
# @keywords internal
.causilo_outlier_bounds <- function(values) {
  correction <- as.integer(nrow(values) > 1L)
  moments <- function(missing) {
    centred <- values
    centred[missing] <- 0
    counts <- nrow(values) - colSums(missing)
    mean <- colSums(centred) / pmax(counts, 1)
    centred <- sweep(centred, 2L, mean)
    centred[missing] <- 0
    var <- colSums(centred^2) / pmax(counts - correction, 1)
    list(mean = mean, sd = pmax(sqrt(var), 1e-6))
  }
  missing <- is.na(values)
  m0 <- moments(missing)
  lo <- rep(m0$mean - 4 * m0$sd, each = nrow(values))
  hi <- rep(m0$mean + 4 * m0$sd, each = nrow(values))
  missing <- missing | (!is.na(values) & (values < lo | values > hi))
  m1 <- moments(missing)
  list(lower = m1$mean - 4 * m1$sd, upper = m1$mean + 4 * m1$sd)
}

# v1.0's bounds (`v1_0/data/normalization.outlier_bounds`), for a table
# with no missing cells: cells further than four (ddof) standard
# deviations from the mean are dropped, then the mean and spread of the
# rest.
# @keywords internal
.causilo_outlier_bounds_v10 <- function(values) {
  n <- nrow(values)
  correction <- as.integer(n > 1L)
  lower <- upper <- numeric(ncol(values))
  for (j in seq_len(ncol(values))) {
    v <- values[, j]
    sdv <- function(x) {
      m <- mean(x)
      sqrt(sum((x - m)^2) / max(length(x) - correction, 0))
    }
    prelim <- max(sdv(v), 1e-6)
    keep <- v[!(abs(v - mean(v)) > 4 * prelim)]
    centre <- if (length(keep)) mean(keep) else 0
    spread <- if (length(keep) > correction) sdv(keep) else NaN
    if (!is.finite(spread)) spread <- 1
    spread <- max(spread, 1e-6)
    lower[j] <- centre - 4 * spread
    upper[j] <- centre + 4 * spread
  }
  list(lower = lower, upper = upper)
}


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

# The class labels in the order `LabelEncoder` sorts them: numerically
# when every label reads as a number, else by code point.
# @keywords internal
.causilo_class_order <- function(labels) {
  num <- suppressWarnings(as.numeric(labels))
  if (!anyNA(num)) labels[order(num)] else sort(labels, method = "radix")
}

# @keywords internal
.causilo_target_fit <- function(y, classification) {
  if (classification) {
    lab <- as.character(y)
    classes <- .causilo_class_order(unique(lab))
    return(list(classes = classes, codes = match(lab, classes) - 1L))
  }
  y <- as.numeric(y)
  sc <- .sk_standard_scaler(matrix(y, ncol = 1L))
  list(mean = sc$mean, scale = sc$scale,
       scaled = (y - sc$mean) / sc$scale,
       limits = range(y))
}


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

# Categories with more than this many observed values also get a
# `log1p(count)` column.
CAUSILO_COUNT_THRESHOLD <- 50L

# Fit the encoding on the context rows: each categorical column's observed
# values, sorted, become codes 0..K-1 (missing and unseen ones NaN); the
# numeric columns pass through. Laid out `[codes | log counts | numeric]`.
# @keywords internal
.causilo_encoder_fit <- function(X, categorical) {
  cat_idx <- which(categorical)
  num_idx <- which(!categorical)
  cats <- lapply(cat_idx, function(j) sort(unique(X[!is.na(X[, j]), j])))
  counts <- list()
  for (k in seq_along(cat_idx)) {
    if (length(cats[[k]]) > CAUSILO_COUNT_THRESHOLD) {
      codes <- match(X[, cat_idx[k]], cats[[k]])
      tab <- tabulate(codes[!is.na(codes)], nbins = length(cats[[k]]))
      if (sum(tab > 0L) > CAUSILO_COUNT_THRESHOLD) {
        counts[[as.character(k)]] <- log1p(tab)
      }
    }
  }
  list(categorical = cat_idx, numeric = num_idx, categories = cats,
       log_counts = counts, width = ncol(X))
}

# Encode rows with a fitted encoder.
# @keywords internal
.causilo_encode <- function(enc, X) {
  codes <- vapply(seq_along(enc$categorical), function(k) {
    as.numeric(match(X[, enc$categorical[k]], enc$categories[[k]]) - 1L)
  }, numeric(nrow(X)))
  codes <- matrix(codes, nrow = nrow(X))
  count_cols <- vapply(names(enc$log_counts), function(k) {
    k <- as.integer(k)
    lc <- enc$log_counts[[as.character(k)]]
    code <- codes[, k]
    out <- lc[code + 1L]
    # Unseen categories have zero frequency; missing ones stay missing.
    miss <- is.na(code)
    out[miss] <- ifelse(is.na(X[miss, enc$categorical[k]]), NaN, 0)
    out
  }, numeric(nrow(X)))
  count_cols <- matrix(count_cols, nrow = nrow(X))
  num <- X[, enc$numeric, drop = FALSE]
  out <- cbind(codes, count_cols, num)
  storage.mode(out) <- "double"
  out[is.na(out)] <- NaN
  unname(out)
}


# ---------------------------------------------------------------------------
# Canonical columns
# ---------------------------------------------------------------------------

# The little-endian float64 bytes of a column, with every NaN written as
# numpy's `np.nan` (0x7FF8000000000000) and -0 as 0, as hex.
# @keywords internal
.causilo_column_key <- function(x) {
  x[!is.na(x) & x == 0] <- 0
  b <- writeBin(as.double(x), raw(), size = 8L, endian = "little")
  nan <- which(is.na(x))
  if (length(nan)) {
    pos <- rep((nan - 1L) * 8L, each = 8L) + 1:8
    b[pos] <- rep(as.raw(c(0, 0, 0, 0, 0, 0, 0xf8, 0x7f)), length(nan))
  }
  paste(as.character(b), collapse = "")
}

# Retain the columns that vary (all of them when there is one row), and
# order them by their bytes, lexicographically, as `CanonicalColumns.fit`
# does; byte-identical columns form one group, kept in input order.
# @keywords internal
.causilo_columns_fit <- function(encoded) {
  varies <- function(col) {
    first <- col[[1]]
    if (is.na(first)) any(!is.na(col)) else any(is.na(col) | col != first)
  }
  retained <- if (nrow(encoded) == 1L) seq_len(ncol(encoded))
              else which(vapply(seq_len(ncol(encoded)),
                                function(j) varies(encoded[, j]), logical(1)))
  if (!length(retained)) {
    cli::cli_abort("At least one feature that varies across the training rows is required.")
  }
  keys <- vapply(retained, function(j) .causilo_column_key(encoded[, j]), character(1))
  o <- order(keys, retained, method = "radix")
  order_ <- retained[o]
  k <- keys[o]
  groups <- cumsum(c(TRUE, k[-1] != k[-length(k)]))
  ties <- lapply(unique(groups[duplicated(groups)]), function(g) range(which(groups == g)))
  list(order = order_, groups = as.integer(groups), ties = ties,
       n_groups = max(groups))
}

# Apply the order; within each group of identical training columns, sort
# every row's values (NaN last), as the reference resolves query ties.
# @keywords internal
.causilo_columns_apply <- function(cols, encoded) {
  out <- encoded[, cols$order, drop = FALSE]
  for (t in cols$ties) {
    j <- t[1]:t[2]
    out[, j] <- t(apply(out[, j, drop = FALSE], 1L, function(r) {
      c(sort(r), rep(NaN, sum(is.na(r))))
    }))
  }
  out
}


# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------

# numpy's linear-interpolation quartiles of each column's finite values,
# as `feature_views._quartiles`.
# @keywords internal
.causilo_quartiles <- function(v) {
  v <- sort(v[is.finite(v)])
  n <- length(v)
  if (!n) return(c(NaN, NaN, NaN))
  ranks <- (n - 1) * c(0.25, 0.5, 0.75)
  lr <- floor(ranks)
  ur <- pmin(lr + 1, n - 1)
  lower <- v[lr + 1]; upper <- v[ur + 1]
  f <- ranks - lr
  d <- upper - lower
  ifelse(f < 0.5, lower + f * d, upper - (1 - f) * d)
}

# @keywords internal
.causilo_robust <- function(z, median, scale) {
  v <- sweep(sweep(z, 2L, median), 2L, scale, "/")
  res <- v / sqrt(1 + (v / 3)^2)
  inf <- !is.na(v) & is.infinite(v)
  res[inf] <- 3 * sign(v[inf])
  res
}

# Fit the views `0..(view_count - 1)` on the canonical training table.
# Returns the fitted statistics and the training views (float64 values,
# rounded to float32 on their way to the network).
# @keywords internal
.causilo_views_fit <- function(x, view_count) {
  st <- list(view_count = view_count)
  missing <- is.na(x)
  counts <- colSums(!missing)
  x0 <- x; x0[missing] <- 0
  center <- ifelse(counts > 0, colSums(x0) / pmax(counts, 1), 0)
  centred <- sweep(x, 2L, center)
  sq <- centred^2; sq[missing] <- 0
  var <- ifelse(counts > 0, colSums(sq) / pmax(counts, 1), 0)
  st$center <- center
  st$scale <- sqrt(var) + 1e-6
  z <- sweep(centred, 2L, st$scale, "/")
  b <- .causilo_outlier_bounds(z)
  st$bounds0 <- b
  views <- list(.causilo_compress(z, b$lower, b$upper))
  if (view_count > 1L) {
    q <- apply(z, 2L, .causilo_quartiles)
    st$robust_median <- q[2, ]
    sc <- q[3, ] - q[1, ]
    sc[is.na(sc) | sc == 0] <- 1
    st$robust_scale <- sc
    r <- .causilo_robust(z, st$robust_median, st$robust_scale)
    b1 <- .causilo_outlier_bounds(r)
    st$bounds1 <- b1
    views[[2]] <- .causilo_compress(r, b1$lower, b1$upper)
  }
  if (view_count > 2L) {
    rk <- .causilo_rank_fit(x)
    st$rank <- rk$state
    views[[3]] <- rk$view
  }
  list(state = st, views = views)
}

# Transform query rows with fitted view statistics.
# @keywords internal
.causilo_views_apply <- function(st, x) {
  z <- sweep(sweep(x, 2L, st$center), 2L, st$scale, "/")
  views <- list(.causilo_compress(z, st$bounds0$lower, st$bounds0$upper))
  if (st$view_count > 1L) {
    r <- .causilo_robust(z, st$robust_median, st$robust_scale)
    views[[2]] <- .causilo_compress(r, st$bounds1$lower, st$bounds1$upper)
  }
  if (st$view_count > 2L) views[[3]] <- .causilo_rank_apply(st$rank, x)
  views
}

# The Rank-Gaussian view (Causilo 1.0's `Normalizer(method =
# "rank2gaussian")`): missing cells mean-filled, every value mapped to the
# normal quantile of its mid-rank among the training values, standardized,
# tails compressed, missing cells restored.
# @keywords internal
.causilo_rank_fit <- function(x) {
  missing <- is.na(x)
  fill <- numeric(ncol(x))
  for (j in seq_len(ncol(x))) {
    obs <- x[!missing[, j], j]
    if (!any(missing[, j])) {
      fill[j] <- if (length(obs)) sum(obs) / length(obs) else 0
      next
    }
    # Columns with missing cells: the grouped `math.fsum` mean, so a
    # one-ulp difference cannot split a tie.
    if (!length(obs)) { fill[j] <- 0; next }
    u <- sort(unique(obs))
    freq <- tabulate(match(obs, u), nbins = length(u))
    scale <- max(abs(u))
    if (scale) {
      total <- sum(freq)
      m <- .causilo_fsum((u / scale) * (freq / total))
      fill[j] <- min(max(m, -1), 1) * scale
    }
  }
  filled <- x
  for (j in seq_len(ncol(x))) filled[missing[, j], j] <- fill[j]
  knots <- lapply(seq_len(ncol(x)), function(j) {
    col <- filled[, j]
    u <- sort(unique(col))
    cnt <- tabulate(match(col, u), nbins = length(u))
    list(values = u, prob = (cumsum(cnt) - cnt / 2) / length(col))
  })
  g <- vapply(seq_len(ncol(x)), function(j) {
    .causilo_ndtri(.causilo_np_interp(filled[, j], knots[[j]]$values, knots[[j]]$prob))
  }, numeric(nrow(x)))
  g <- matrix(g, nrow = nrow(x))
  sc <- .sk_standard_scaler(g)
  g <- sweep(sweep(g, 2L, sc$mean), 2L, sc$scale, "/")
  b <- .causilo_outlier_bounds_v10(g)
  state <- list(fill = fill, knots = knots, mean = sc$mean, scale = sc$scale,
                lower = b$lower, upper = b$upper)
  list(state = state, view = .causilo_rank_finish(state, g, missing))
}

# @keywords internal
.causilo_rank_finish <- function(state, g, missing) {
  lo <- rep(state$lower, each = nrow(g)); hi <- rep(state$upper, each = nrow(g))
  g <- ifelse(g < lo, lo - asinh(lo - g), g)
  g <- ifelse(g > hi, hi + asinh(g - hi), g)
  g <- matrix(g, nrow = length(missing) / length(state$lower))
  g[missing] <- NaN
  g
}

# @keywords internal
.causilo_rank_apply <- function(state, x) {
  missing <- is.na(x)
  for (j in seq_len(ncol(x))) x[missing[, j], j] <- state$fill[j]
  g <- vapply(seq_len(ncol(x)), function(j) {
    .causilo_ndtri(.causilo_np_interp(x[, j], state$knots[[j]]$values, state$knots[[j]]$prob))
  }, numeric(nrow(x)))
  g <- matrix(g, nrow = nrow(x))
  g <- sweep(sweep(g, 2L, state$mean), 2L, state$scale, "/")
  .causilo_rank_finish(state, g, missing)
}


# ---------------------------------------------------------------------------
# Ensemble plan
# ---------------------------------------------------------------------------

#' Feature view of each fitted member
#'
#' Balanced over Base (0), Robust (1) and Rank-Gaussian (2), using both
#' non-Base slots for Robust when `n_estimators` is 4 or 5
#' (`data/ensemble.py::feature_view_order`).
#' @param n_estimators Requested members.
#' @param count Fitted members.
#' @keywords internal
.causilo_view_order <- function(n_estimators, count) {
  other <- count %/% 2L
  base <- other + count %% 2L
  robust <- (other + 1L) %/% 2L
  if (n_estimators %in% c(4L, 5L)) robust <- other
  c(rep(0L, base), rep(1L, robust), rep(2L, other - robust))
}

# Members (0-based) whose sign is -1 for one column group.
# @keywords internal
.causilo_negative_members <- function(groups, capacity, count, rng) {
  quota <- count %/% 2L
  if (max(lengths(groups)) <= capacity) {
    return(as.integer(unlist(rng$sample(seq_len(count - 1L), quota))))
  }
  selected <- integer(0)
  room <- integer(length(groups))
  cand_m <- integer(0); cand_g <- integer(0)
  for (gi in seq_along(groups)) {
    g <- groups[[gi]]
    eligible <- g[g != 0L]
    lower <- max(0L, length(g) - capacity)
    selected <- c(selected, as.integer(unlist(rng$sample(eligible, lower))))
    room[gi] <- min(length(eligible), capacity) - lower
    keep <- eligible[!eligible %in% selected]
    cand_m <- c(cand_m, keep); cand_g <- c(cand_g, rep(gi, length(keep)))
  }
  for (i in rng$shuffle(seq_along(cand_m))) {
    if (length(selected) == quota) break
    if (room[cand_g[i]] > 0L) {
      selected <- c(selected, cand_m[i])
      room[cand_g[i]] <- room[cand_g[i]] - 1L
    }
  }
  selected
}

#' The Causilo ensemble plan
#'
#' A port of `data/ensemble.py::make_ensemble_plan`, replayed bit for bit
#' over CPython's generator ([py_random()]).
#'
#' @param n_groups Canonical column groups.
#' @param n_classes Classes, or `NULL` for regression.
#' @param n_estimators Requested members.
#' @param seed Non-negative integer, or `NULL` for an unseeded plan.
#' @return `list(signs [count, n_groups], shifts, target_signs, views,
#'   count, trim)`.
#' @keywords internal
causilo_ensemble_plan <- function(n_groups, n_classes, n_estimators, seed) {
  shift_count <- if (is.null(n_classes)) 2L else as.integer(n_classes)
  count <- as.integer(min(n_estimators, 2^n_groups * shift_count))
  views <- .causilo_view_order(as.integer(n_estimators), count)
  rng <- py_random(seed)
  target_order <- c(0L, as.integer(unlist(rng$sample(seq_len(shift_count - 1L),
                                                     shift_count - 1L))))
  shifts <- target_order[(seq_len(count) - 1L) %% shift_count + 1L]
  groups <- lapply(seq_len(min(count, shift_count)) - 1L,
                   function(t) seq.int(t, count - 1L, by = shift_count))
  signs <- matrix(1, count, n_groups)
  for (column in seq_len(n_groups)) {
    neg <- .causilo_negative_members(groups, 2^(n_groups - column), count, rng)
    signs[neg + 1L, column] <- -1
    if (length(groups) < count) {
      groups <- unlist(lapply(groups, function(g) {
        Filter(length, list(g[!g %in% neg], g[g %in% neg]))
      }), recursive = FALSE)
    }
  }
  list(signs = signs, shifts = shifts,
       target_signs = if (is.null(n_classes)) 1 - 2 * shifts,
       views = views, count = count,
       trim = if (is.null(n_classes)) count %/% 4L else 0L)
}


# ---------------------------------------------------------------------------
# The whole recipe
# ---------------------------------------------------------------------------

#' Fit the Causilo 1.5 preprocessing on the context rows
#'
#' @param X Numeric matrix of predictors (NA for missing); categorical
#'   columns hold codes or any numbers naming a category.
#' @param y Class labels or the numeric target.
#' @param classification Task.
#' @param categorical Logical per column.
#' @param n_estimators Requested members.
#' @param seed Plan seed.
#' @return A recipe: encoder, canonical columns, plan, view statistics,
#'   target transform and the training views.
#' @keywords internal
causilo_recipe_fit <- function(X, y, classification, categorical,
                               n_estimators, seed) {
  X <- .causilo_check_matrix(X)
  tgt <- .causilo_target_fit(y, classification)
  enc <- .causilo_encoder_fit(X, categorical)
  encoded <- .causilo_encode(enc, X)
  cols <- .causilo_columns_fit(encoded)
  train <- .causilo_columns_apply(cols, encoded)
  K <- if (classification) length(tgt$classes) else NULL
  codebook <- NULL
  if (!is.null(K) && K > CAUSILO_NATIVE_CLASSES) {
    # Error-correcting output codes: one Base-view member per code row,
    # whatever `n_estimators` says, as the reference does.
    codebook <- causilo_ecoc_codebook(K, CAUSILO_NATIVE_CLASSES, seed)
    K <- CAUSILO_NATIVE_CLASSES
    n_estimators <- 1L
  }
  plan <- causilo_ensemble_plan(cols$n_groups, K, n_estimators, seed)
  plan$signs <- plan$signs[, cols$groups, drop = FALSE]
  fv <- .causilo_views_fit(train, max(plan$views) + 1L)
  list(classification = classification, target = tgt, encoder = enc,
       columns = cols, plan = plan, views = fv$state, train_views = fv$views,
       n_train = nrow(X), codebook = codebook)
}

# The fitted float64 views of query rows.
# @keywords internal
causilo_recipe_apply <- function(rec, X) {
  X <- .causilo_check_matrix(X, rec$encoder$width)
  enc <- .causilo_encode(rec$encoder, X)
  .causilo_views_apply(rec$views, .causilo_columns_apply(rec$columns, enc))
}

# Member inputs: `[B, n + q, C]` float32 (view times sign, context rows
# first) and the member targets `[B, n]`.
# @keywords internal
# `context = FALSE` leaves the context rows out (for a cached pass).
# With an ECOC codebook, `members` are code rows: each the plan's single
# member, with that row's symbols as its targets.
causilo_member_inputs <- function(rec, query_views, members, context = TRUE) {
  n <- rec$n_train
  ecoc <- !is.null(rec$codebook)
  views <- lapply(members, function(m) {
    v <- if (ecoc) rec$plan$views[[1]] + 1L else rec$plan$views[[m]] + 1L
    rbind(if (context) rec$train_views[[v]],
          if (!is.null(query_views)) query_views[[v]])
  })
  # The reference multiplies the float32 views by float32 signs.
  d <- dim(views[[1]])
  x <- torch::torch_tensor(array(unlist(views), c(d, length(views))),
                           dtype = torch::torch_float32())$permute(c(3L, 1L, 2L))
  sign_rows <- if (ecoc) rep(1L, length(members)) else members
  s <- torch::torch_tensor(rec$plan$signs[sign_rows, , drop = FALSE],
                           dtype = torch::torch_float32())$unsqueeze(2L)
  x <- x * s
  y <- if (ecoc) {
    t(vapply(members, function(r) as.numeric(rec$codebook[r, rec$target$codes + 1L]),
             numeric(n)))
  } else if (rec$classification) {
    K <- length(rec$target$classes)
    t(vapply(members, function(m) {
      as.numeric((rec$target$codes - rec$plan$shifts[[m]]) %% K)
    }, numeric(n)))
  } else {
    yt <- as.numeric(torch::torch_tensor(rec$target$scaled, dtype = torch::torch_float32()))
    t(vapply(members, function(m) yt * rec$plan$target_signs[[m]], numeric(n)))
  }
  list(x = x, y = torch::torch_tensor(matrix(y, nrow = length(members)),
                                      dtype = torch::torch_float32()))
}

# @keywords internal
.causilo_check_matrix <- function(X, width = NULL) {
  if (is.data.frame(X)) X <- as.matrix(X)
  if (!is.numeric(X) && !is.logical(X)) {
    cli::cli_abort("Causilo predictors must be numeric (encode factors as codes).")
  }
  X <- unname(X)
  storage.mode(X) <- "double"
  if (any(is.infinite(X))) {
    cli::cli_abort("Causilo does not accept infinite feature values.")
  }
  if (!is.null(width) && ncol(X) != width) {
    cli::cli_abort("Expected {width} feature columns, got {ncol(X)}.")
  }
  if (!nrow(X) && is.null(width)) cli::cli_abort("No training rows.")
  if (!ncol(X)) cli::cli_abort("Causilo needs at least one feature column.")
  X
}

# ---------------------------------------------------------------------------
# More than 10 classes: error-correcting output codes
# ---------------------------------------------------------------------------
#
# Causilo 1.0's codec (`v1_0/ecoc.py`), which 1.5 reuses. Each code row
# splits the classes into `symbols - 1` active ones (each with its own
# output symbol) and the rest (the last symbol); the network runs once per
# code row, as a single Base-view member with no sign flips, and the
# classes are scored by their mean log-probability over the rows that
# make them active.
#
# The reference draws the codebook with NumPy's PCG64 `default_rng`, which
# tabfound does not have. This draws it with the same construction over
# CPython's generator (`py_random()`, seeded with `seed`): a valid codebook
# of the same shape and quality, not the reference's. The decoder is the
# reference's, and is graded against it on the reference's own codebook.

# Code rows of the search path (`_row_budget`).
# @keywords internal
.causilo_ecoc_rows <- function(classes, symbols) {
  coverage <- (classes + symbols - 2L) %/% (symbols - 1L)
  digits <- 0L; representable <- 1
  while (representable < classes) {
    digits <- digits + 1L
    representable <- representable * symbols
  }
  minimum <- max(digits, coverage)
  min(4L * minimum, max(minimum, 4L * digits))
}

# Minimum and mean pairwise Hamming distance of an active-set schedule
# (`_separation`); `active` is logical `[rows, classes]`.
# @keywords internal
.causilo_ecoc_separation <- function(active) {
  a <- active * 1
  app <- colSums(a)
  both <- crossprod(a)
  d <- outer(app, app, `+`) - both
  pairs <- d[upper.tri(d)]
  c(min(pairs), mean(pairs))
}

# The search path (`_search_codes`): 50 balanced schedules, the
# best-separated one kept, its active slots labelled at random.
# @keywords internal
.causilo_ecoc_search <- function(classes, symbols, rng) {
  rows <- .causilo_ecoc_rows(classes, symbols)
  width <- symbols - 1L
  best <- NULL; best_score <- c(-Inf, -Inf)
  for (trial in seq_len(50L)) {
    active <- matrix(FALSE, rows, classes)
    uses <- integer(classes)
    for (r in seq_len(rows)) {
      shuffled <- rng$shuffle(seq_len(classes))
      # A stable sort by usage keeps the shuffled order within ties.
      pick <- shuffled[order(uses[shuffled], method = "radix")[seq_len(width)]]
      active[r, pick] <- TRUE
      uses[pick] <- uses[pick] + 1L
    }
    sc <- .causilo_ecoc_separation(active)
    if (sc[1] > best_score[1] || (sc[1] == best_score[1] && sc[2] > best_score[2])) {
      best <- active; best_score <- sc
    }
  }
  codes <- matrix(width, rows, classes)
  for (r in seq_len(rows)) {
    codes[r, which(best[r, ])] <- rng$shuffle(seq_len(width) - 1L)
  }
  storage.mode(codes) <- "integer"
  codes
}

# The partition path (`_partition_codes`, above 200 classes): one shuffled
# class order cut into groups of `symbols - 1`, the last group padded with
# classes from earlier groups, each given a symbol it did not have before.
# @keywords internal
.causilo_ecoc_partition <- function(classes, symbols, rng) {
  width <- symbols - 1L
  full <- classes %/% width; remainder <- classes %% width
  codes <- matrix(width, full + (remainder > 0L), classes)
  order_ <- rng$shuffle(seq_len(classes))
  blocks <- matrix(order_[seq_len(full * width)], full, width, byrow = TRUE)
  for (r in seq_len(full)) codes[r, blocks[r, ]] <- rng$shuffle(seq_len(width) - 1L)
  if (remainder) {
    padding <- width - remainder
    group_order <- rng$shuffle(seq_len(full))
    slots <- seq_len(padding) - 1L
    g <- group_order[slots %% full + 1L]
    donors <- blocks[cbind(g, slots %/% full + 1L)]
    members <- c(order_[(full * width + 1L):classes], donors)
    assigned <- rng$shuffle(seq_len(width) - 1L)
    previous <- codes[cbind(g, donors)]
    for (shift in rng$shuffle(seq_len(width) - 1L)) {
      shifted <- (assigned + shift) %% width
      if (all(shifted[(remainder + 1L):width] != previous)) {
        codes[full + 1L, members] <- shifted
        break
      }
    }
  }
  storage.mode(codes) <- "integer"
  codes
}

#' Draw a Causilo ECOC codebook
#'
#' @param classes Number of classes (more than `symbols`).
#' @param symbols Output symbols of the network (10).
#' @param seed Seed for [py_random()].
#' @return Integer matrix `[code rows, classes]` of 0-based symbols; the
#'   last symbol (`symbols - 1`) marks the classes a row leaves at rest.
#' @keywords internal
causilo_ecoc_codebook <- function(classes, symbols = CAUSILO_NATIVE_CLASSES, seed = 42L) {
  rng <- py_random(seed)
  if (classes > 200L) .causilo_ecoc_partition(classes, symbols, rng)
  else .causilo_ecoc_search(classes, symbols, rng)
}

#' Decode ECOC row probabilities into class probabilities
#'
#' `ECOCAccumulator` of `v1_0/ecoc.py`: per code row, the active classes
#' gain `log(max(p, 1e-12))` of their symbol's probability; each class's
#' total is divided by its number of active rows, then softmaxed.
#'
#' @param probs List over code rows of `[q, symbols]` probabilities.
#' @param codebook From [causilo_ecoc_codebook()].
#' @return `[q, classes]`.
#' @keywords internal
causilo_ecoc_decode <- function(probs, codebook) {
  rest <- ncol(probs[[1]]) - 1L
  K <- ncol(codebook)
  scores <- matrix(0, nrow(probs[[1]]), K)
  appearances <- integer(K)
  for (r in seq_along(probs)) {
    p <- probs[[r]]
    p <- p / apply(p, 1L, max)
    p <- p / rowSums(p)
    cls <- which(codebook[r, ] != rest)
    sym <- codebook[r, cls] + 1L
    scores[, cls] <- scores[, cls] + log(pmax(p[, sym, drop = FALSE], 1e-12))
    appearances[cls] <- appearances[cls] + 1L
  }
  scores <- sweep(scores, 2L, appearances, "/")
  scores <- scores - apply(scores, 1L, max)
  e <- exp(scores)
  e / rowSums(e)
}
