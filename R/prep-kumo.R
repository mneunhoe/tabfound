# The Kumo Tabular preprocessing recipe and ensemble.
#
# A port of `sdm.models.kumo.tabular.recipe.default_recipe()` and the
# processors it uses (`sdm/processing/`, structured-data-models 0.2.1,
# Apache-2.0). Unlike the sklearn-derived wrappers of the other backends,
# `sdm`'s recipe is torch code driven by one `torch.Generator`, so it is
# ported as torch code too: every statistic goes through the same libtorch
# reductions, in float64 where the reference casts up, and every random
# draw is the same generator call in the same order. On the same platform
# that makes the members bit-identical to the reference's (checked by
# `test-prep-kumo.R` against `kumo_ensemble_reference.py`).
#
# Per member, fitted on the context rows and replayed on every row:
#
#   target    classification: classes observed in the context, sorted by
#             value, then a per-member cyclic shift (offsets from
#             `randperm(K)` pools); regression: float32 standardization
#             and a random sign per member
#   features  numerical columns:   numerical_processor(), random signs
#             categorical columns: codes by sorted value (-1 unseen or
#                                  missing), log-count columns for the
#                                  high-cardinality ones, codes as
#                                  numbers, numerical_processor()
#             a latin-square column permutation, the first 500 columns,
#             float32
#
#   numerical_processor(): float64, drop constant columns, standardize
#   (+1e-6), clip to +-100, then per member one of {identity, Yeo-Johnson,
#   robust scale + soft clip at 3, rank-Gaussian} in round robin, then a
#   4-sigma log clip.
#
# Fitted state is kept as plain R vectors so a fitted model can be saved;
# tensors are rebuilt on use.


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

# R torch's `torch_randperm()` and `torch_rand()` take no generator, and
# the generic wrapper refuses one, but the generated bindings underneath
# accept it -- and with it reproduce CPython torch's stream exactly. This
# is the one place that reaches for them.
# @keywords internal
.kumo_generator <- function(seed) {
  g <- torch::torch_generator()
  g$set_current_seed(as.integer(seed))
  g
}

# `torch.randperm(n, generator = g)`, 0-based.
# @keywords internal
.kumo_randperm <- function(n, g) {
  fn <- .kumo_torch_binding("cpp_torch_namespace_randperm_n_int64_t_generator_Generator")
  as.integer(fn(as.integer(n), g, list(dtype = torch::torch_long())))
}

# `torch.empty(shape, dtype).bernoulli_(p, generator = g)`.
# @keywords internal
.kumo_bernoulli <- function(shape, dtype, g, p = 0.5) {
  torch::torch_empty(shape, dtype = dtype)$bernoulli_(p, generator = g)
}

# `torch.rand(shape, generator = g)`, float32.
# @keywords internal
.kumo_rand <- function(shape, g) {
  fn <- .kumo_torch_binding("cpp_torch_namespace_rand_size_IntArrayRef_generator_Generator")
  fn(as.integer(shape), g, list(dtype = torch::torch_float32()))
}

# `torch.special.ndtri` / `ndtr`: registered by R torch, not exported.
.kumo_ndtri <- function(x) .kumo_torch_binding("torch_special_ndtri")(x)
.kumo_ndtr <- function(x) .kumo_torch_binding("torch_special_ndtr")(x)

.kumo_torch_binding <- function(name) {
  fn <- tryCatch(utils::getFromNamespace(name, "torch"), error = function(e) NULL)
  if (is.null(fn)) {
    cli::cli_abort(c(
      "This build of {.pkg torch} has no {.fn {name}}.",
      i = "Kumo's recipe needs it; it is present in torch 0.17."
    ))
  }
  fn
}


# ---------------------------------------------------------------------------
# Numerical processors, on float64 tensors [R, C]
# ---------------------------------------------------------------------------

.kumo_f64 <- function(x, ncol = length(x)) {
  torch::torch_tensor(matrix(as.numeric(x), nrow = 1L, ncol = ncol),
                      dtype = torch::torch_float64())
}

# Columns of a 2-d tensor by 1-based index; R torch's `x[, integer(0)]`
# is an error, and a recipe that has dropped every column needs it.
.kumo_cols <- function(x, idx) {
  x$index_select(2L, torch::torch_tensor(as.integer(idx), dtype = torch::torch_long()))
}

.kumo_isfinite <- function(x) (x > -Inf)$logical_and(x < Inf)

# sklearn's constant-feature rule, written as `sdm` writes it: `n * eps`
# is a long tensor times a Python float, which torch promotes to float32
# before it meets `var`. The same expression here takes the same path.
.kumo_constant_mask <- function(var, mean, n) {
  eps <- .Machine$double.eps
  if (var$dtype == torch::torch_float32()) eps <- 2^-23
  upper <- n$mul(eps) * var + (n * mean * eps)$square()
  var <= upper
}

# DropConstantColumns(threshold = 1): keep a column iff some value differs
# from the first row's (NaN differs from everything) and it is not all
# NaN. One row or none keeps everything.
.kumo_keep_columns <- function(x) {
  if (x$size(1) <= 1L) return(rep(TRUE, x$size(2)))
  diff <- (x != x[1:1, ])$any(dim = 1L)
  allnan <- x$isnan()$all(dim = 1L)
  as.logical(diff$logical_and(allnan$logical_not()))
}

# Standardize(eps): finite-only population moments.
.kumo_standardize_fit <- function(x, eps = 0) {
  finite <- .kumo_isfinite(x)
  count <- finite$sum(dim = 1L, keepdim = TRUE)
  fon <- x$masked_fill(finite$logical_not(), NaN)
  mean <- fon$nansum(dim = 1L, keepdim = TRUE)$div(count)
  mean <- mean$masked_fill(mean$isnan(), 0)
  var <- (fon - mean)$square()$nansum(dim = 1L, keepdim = TRUE)$div(count)
  var <- var$masked_fill(var$isnan(), 0)
  scale <- var$sqrt()
  if (eps == 0) {
    scale <- scale$masked_fill(.kumo_constant_mask(var, mean, count), 1)
  } else {
    scale <- scale + eps
  }
  list(mean = as.numeric(mean), scale = as.numeric(scale))
}

.kumo_standardize <- function(x, st) {
  (x - .kumo_f64(st$mean))$div(.kumo_f64(st$scale))
}

# --- Yeo-Johnson ---------------------------------------------------------

.kumo_yj <- function(inp, lambdas, ml = NULL, pos = NULL) {
  eps <- .Machine$double.eps
  tml <- 2 - lambdas
  ml <- ml %||% inp$abs()$log1p()
  pos <- pos %||% (inp >= 0)
  ex <- torch::torch_where(pos, lambdas, tml)
  out <- (ex * ml)$expm1()$div(ex)
  zero <- torch::torch_where(pos, lambdas$abs() < eps, tml$abs() < eps)
  out <- torch::torch_where(zero, ml, out)
  out$copysign(inp)
}

.kumo_yj_bounds <- function(inp) {
  missing <- inp$isnan()
  max_abs <- inp$abs()$nan_to_num(nan = 0)$amax(dim = 1L, keepdim = TRUE)
  l1p <- (20 * max_abs)$log1p()
  l1p <- torch::torch_where(max_abs == 0, torch::torch_ones_like(l1p), l1p)
  log_eps <- log(.Machine$double.eps)
  log_tiny <- (log(.Machine$double.xmin) - log_eps) / 2
  log_max <- (log(.Machine$double.xmax) + log_eps) / 2
  pl <- log_tiny / l1p
  pu <- log_max / l1p
  negative <- inp < 0
  all_neg <- negative$logical_or(missing)$all(dim = 1L, keepdim = TRUE)
  any_neg <- negative$any(dim = 1L, keepdim = TRUE)
  ml <- torch::torch_maximum(2 - pu, pl)
  mu <- torch::torch_minimum(2 - ml, pu)
  lo <- torch::torch_where(any_neg, ml, pl)
  hi <- torch::torch_where(any_neg, mu, pu)
  lo <- torch::torch_where(all_neg, 2 - pu, lo)
  hi <- torch::torch_where(all_neg, 2 - pl, hi)
  zero <- max_abs == 0
  list(lo = torch::torch_where(zero, torch::torch_ones_like(lo), lo),
       hi = torch::torch_where(zero, torch::torch_ones_like(hi), hi))
}

.kumo_yj_loglik <- function(inp, lambdas, ml, pos, logjac, count) {
  t <- .kumo_yj(inp, lambdas, ml, pos)
  mean <- t$nansum(dim = 1L, keepdim = TRUE)$div(count)
  v <- (t - mean)$square()$nansum(dim = 1L, keepdim = TRUE)$div(count)
  valid <- v$isfinite()$logical_and(v >= .Machine$double.xmin)
  ll <- v$log() * count$to(dtype = torch::torch_float64())$neg()$div(2)
  ll <- ll + (lambdas - 1) * logjac
  ll$masked_fill(valid$logical_not(), -Inf)
}

# Golden-section search with a fixed 44 steps, literally as `sdm` does it:
# the branch taken on each step decides the final bracket, so the order
# of the updates is part of the answer.
.kumo_yj_lambdas <- function(inp, constant, count) {
  b <- .kumo_yj_bounds(inp)
  left <- b$lo$masked_fill(constant, 1)
  right <- b$hi$masked_fill(constant, 1)
  ml <- inp$abs()$log1p()
  pos <- inp >= 0
  logjac <- ml$copysign(inp)$nansum(dim = 1L, keepdim = TRUE)
  invphi <- (sqrt(5) - 1) / 2
  span <- (right - left) * invphi
  c_ <- right - span
  d_ <- left + span
  f <- function(l) .kumo_yj_loglik(inp, l, ml, pos, logjac, count)
  fc <- f(c_); fd <- f(d_)
  for (i in seq_len(44L)) {
    cr <- fc < fd
    left <- torch::torch_where(cr, c_, left)
    right <- torch::torch_where(cr, right, d_)
    span <- (right - left) * invphi
    c_next <- torch::torch_where(cr, d_, right - span)
    d_next <- torch::torch_where(cr, left + span, c_)
    new_point <- torch::torch_where(cr, d_next, c_next)
    ns <- f(new_point)
    fc_new <- torch::torch_where(cr, ns, fc)
    fd_new <- torch::torch_where(cr, fd, ns)
    fc <- fd_new; fd <- fc_new
    c_old <- c_; c_ <- c_next; c_next <- c_old
    d_old <- d_; d_ <- d_next; d_next <- d_old
  }
  ((left + right) / 2)$masked_fill(constant, 1)
}

.kumo_power_fit <- function(x) {
  finite <- .kumo_isfinite(x)
  fon <- x$masked_fill(finite$logical_not(), NaN)
  count <- finite$sum(dim = 1L, keepdim = TRUE)$clamp(min = 1L)
  mean <- fon$nansum(dim = 1L, keepdim = TRUE)$div(count)
  var <- (fon - mean)$square()$nansum(dim = 1L, keepdim = TRUE)$div(count)
  constant <- .kumo_constant_mask(var, mean, count)
  lambdas <- .kumo_yj_lambdas(fon, constant, count)
  t <- .kumo_yj(fon, lambdas)
  mean <- t$nansum(dim = 1L, keepdim = TRUE)$div(count)
  var <- (t - mean)$square()$nansum(dim = 1L, keepdim = TRUE)$div(count)
  scale <- var$sqrt()$masked_fill(.kumo_constant_mask(var, mean, count), 1)
  list(lambdas = as.numeric(lambdas), mean = as.numeric(mean),
       scale = as.numeric(scale))
}

.kumo_power <- function(x, st) {
  out <- (.kumo_yj(x, .kumo_f64(st$lambdas)) - .kumo_f64(st$mean))$div(.kumo_f64(st$scale))
  out$clamp(min = -.Machine$double.xmax, max = .Machine$double.xmax)
}

# --- Robust scale + soft clip ----------------------------------------------

.kumo_robust_fit <- function(x) {
  fon <- x$masked_fill(.kumo_isfinite(x)$logical_not(), NaN)
  q <- torch::torch_tensor(c(0.25, 0.5, 0.75), dtype = torch::torch_float64())
  qs <- torch::torch_nanquantile(fon, q, dim = 1L, keepdim = TRUE)  # [3, 1, C]
  lower <- qs[1, , ]; median <- qs[2, , ]; upper <- qs[3, , ]
  scale <- torch::torch_where(lower == upper, torch::torch_ones_like(lower),
                              upper - lower)
  list(median = as.numeric(median), scale = as.numeric(scale))
}

.kumo_robust <- function(x, st) {
  (x - .kumo_f64(st$median))$div(.kumo_f64(st$scale))
}

# ClipSoft(3): x / sqrt(1 + (x/3)^2), +-Inf to +-3, NaN kept.
.kumo_clip_soft <- function(x, bound = 3) {
  unit <- x$div(bound)$abs()
  root <- unit$square()$add(1)$sqrt()
  unit <- unit$div(root)$masked_fill(.kumo_isfinite(root)$logical_not(), 1)
  clipped <- unit$mul(bound)$mul(x$sign())
  torch::torch_where(x$isnan(), x, clipped)
}

# --- Rank-Gaussian -----------------------------------------------------------

.kumo_rank_gaussian_fit <- function(x, max_knots = 8192L) {
  n_rows <- x$size(1)
  cols <- x$t()$contiguous()                       # [C, R]
  finite <- .kumo_isfinite(cols)
  count <- finite$sum(dim = 2L, keepdim = TRUE)    # [C, 1]
  values <- cols$masked_fill(finite$logical_not(), Inf)
  values <- torch::torch_sort(values, dim = 2L)[[1]]$contiguous()
  left <- torch::torch_searchsorted(values, values, right = FALSE)
  right <- torch::torch_searchsorted(values, values, right = TRUE)
  p <- (left + right)$to(dtype = torch::torch_float64())$div(2 * count$clamp(min = 1L))
  p <- p$clamp(max = 1 - .Machine$double.eps)
  last <- (count - 1L)$clamp(min = 0L)              # 0-based
  rows <- torch::torch_arange(0L, n_rows - 1L, dtype = torch::torch_long())
  if (n_rows <= max_knots) {
    positions <- torch::torch_minimum(rows$unsqueeze(1L), last)
  } else {
    distinct <- (left == rows)$cumsum(dim = 2L) - 1L
    num_distinct <- distinct$gather(2L, last + 1L) + 1L
    steps <- torch::torch_arange(0L, max_knots - 1L, dtype = torch::torch_long())
    positions <- torch::torch_searchsorted(
      distinct, torch::torch_minimum(steps$unsqueeze(1L), num_distinct - 1L)
    )
    lower <- .kumo_ndtri(p[, 1:1])
    upper <- .kumo_ndtri(p$gather(2L, last + 1L))
    w <- steps$to(dtype = p$dtype)$div(max_knots - 1L)
    quantiles <- .kumo_ndtr(torch::torch_lerp(lower, upper, w))
    above <- torch::torch_minimum(torch::torch_searchsorted(p, quantiles), last)
    below <- (above - 1L)$clamp(min = 0L)
    closer <- (quantiles - p$gather(2L, below + 1L)) <
      (p$gather(2L, above + 1L) - quantiles)
    spaced <- torch::torch_where(closer, below, above)
    spaced[, 1] <- 0L
    spaced[, spaced$size(2)] <- last$squeeze(2L)
    positions <- torch::torch_where(num_distinct <= max_knots, positions, spaced)
  }
  missing <- count == 0L
  kv <- values$gather(2L, positions + 1L)$masked_fill(missing, NaN)
  kp <- p$gather(2L, positions + 1L)$masked_fill(missing, NaN)
  list(values = as.matrix(kv), probs = as.matrix(kp))
}

# `sdm`'s batched linear interpolation, with torch's own `lerp`.
.kumo_interp <- function(v, b, r) {
  n <- b$size(2)
  if (n == 1L) return(r[, 1:1]$expand_as(v))
  idx <- torch::torch_searchsorted(b, v, right = TRUE)$clamp(1L, n - 1L)
  x0 <- b$gather(2L, idx); x1 <- b$gather(2L, idx + 1L)
  y0 <- r$gather(2L, idx); y1 <- r$gather(2L, idx + 1L)
  denom <- x1 - x0
  w <- torch::torch_where(denom != 0, (v - x0)$div(denom),
                          torch::torch_zeros_like(denom))
  out <- torch::torch_lerp(y0, y1, w)
  out <- torch::torch_where(v <= b[, 1:1], r[, 1:1], out)
  torch::torch_where(v >= b[, n:n], r[, n:n], out)
}

.kumo_rank_gaussian <- function(x, st) {
  cols <- x$t()$contiguous()
  b <- torch::torch_tensor(st$values, dtype = torch::torch_float64())
  r <- torch::torch_tensor(st$probs, dtype = torch::torch_float64())
  q <- .kumo_interp(cols, b, r)
  .kumo_ndtri(q)$masked_fill(cols$isnan(), NaN)$t()
}

# --- 4-sigma log clip --------------------------------------------------------

.kumo_clip_sigma_fit <- function(x, threshold = 4) {
  finite <- .kumo_isfinite(x)
  nf <- finite$sum(dim = 1L, keepdim = TRUE)
  fon <- x$masked_fill(finite$logical_not(), NaN)
  mean <- fon$nansum(dim = 1L, keepdim = TRUE)$div(nf)
  mean <- mean$masked_fill(mean$isnan(), 0)
  var <- (fon - mean)$square()$nansum(dim = 1L, keepdim = TRUE)$div((nf - 1L)$clamp(min = 1L))
  std <- var$sqrt()$clamp(min = 1e-6)
  lower <- mean - threshold * std
  upper <- mean + threshold * std
  keep <- (x >= lower)$logical_and(x <= upper)$logical_and(finite)
  count <- keep$sum(dim = 1L, keepdim = TRUE)
  kmean <- torch::torch_where(keep, x, torch::torch_zeros_like(x))$sum(dim = 1L, keepdim = TRUE)
  kmean <- kmean$div(count$clamp(min = 1L))
  centered <- (x - kmean)$masked_fill(keep$logical_not(), 0)
  kvar <- centered$square()$sum(dim = 1L, keepdim = TRUE)$div((count - 1L)$clamp(min = 1L))
  kstd <- kvar$sqrt()$clamp(min = 1e-6)
  has <- count > 0L
  mean <- torch::torch_where(has, kmean, mean)
  std <- torch::torch_where(has, kstd, std)
  list(lower = as.numeric(mean - threshold * std),
       upper = as.numeric(mean + threshold * std))
}

.kumo_clip_sigma <- function(x, st) {
  la <- x$abs()$log1p()
  out <- torch::torch_maximum(.kumo_f64(st$lower) - la, x)
  torch::torch_minimum(la + .kumo_f64(st$upper), out)
}

# --- numerical_processor() ---------------------------------------------------

# Fitted once on the context; the per-member option only picks the branch,
# so each option is fitted at most once whatever the ensemble size.
# @keywords internal
.kumo_numproc_fit <- function(x, options) {
  keep <- .kumo_keep_columns(x)
  st <- list(keep = keep, options = list())
  if (!any(keep)) return(st)
  x <- .kumo_cols(x, which(keep))
  st$standardize <- .kumo_standardize_fit(x, eps = 1e-6)
  x <- .kumo_standardize(x, st$standardize)$clamp(-100, 100)
  for (o in sort(unique(options))) {
    opt <- switch(as.character(o),
      "0" = list(),
      "1" = .kumo_power_fit(x),
      "2" = .kumo_robust_fit(x),
      "3" = .kumo_rank_gaussian_fit(x))
    xo <- .kumo_numproc_option(x, o, opt)
    opt$sigma <- .kumo_clip_sigma_fit(xo)
    st$options[[as.character(o)]] <- opt
  }
  st
}

.kumo_numproc_option <- function(x, o, opt) {
  switch(as.character(o),
    "0" = x,
    "1" = .kumo_power(x, opt),
    "2" = .kumo_clip_soft(.kumo_robust(x, opt)),
    "3" = .kumo_rank_gaussian(x, opt))
}

# @keywords internal
.kumo_numproc <- function(st, x, o) {
  x <- .kumo_cols(x, which(st$keep))
  if (!any(st$keep)) return(x)
  x <- .kumo_standardize(x, st$standardize)$clamp(-100, 100)
  opt <- st$options[[as.character(o)]]
  .kumo_clip_sigma(.kumo_numproc_option(x, o, opt), opt$sigma)
}


# ---------------------------------------------------------------------------
# Categorical columns
# ---------------------------------------------------------------------------

# AlignCategories(sort_by = "value"): the vocabulary is the values
# observed in the context, sorted; everything else is -1.
.kumo_vocab <- function(v) {
  u <- unique(v[!is.na(v)])
  if (is.character(u)) sort(u, method = "radix") else sort(u)
}

.kumo_codes <- function(v, vocab) {
  code <- match(v, vocab) - 1L
  code[is.na(code)] <- -1L
  code
}


# ---------------------------------------------------------------------------
# Fit and transform
# ---------------------------------------------------------------------------

#' Fit the Kumo Tabular recipe on the context rows
#'
#' Draws everything random up front -- target shifts or signs, feature
#' signs, the column permutation -- from one generator seeded with
#' `seed`, in the order the reference draws them.
#'
#' @param X Numeric matrix of context rows; categorical columns hold codes
#'   or values, `NA` for missing.
#' @param y Integer class codes (0-based, any subset of `0..K-1`) for
#'   classification, numeric for regression.
#' @param classification Task.
#' @param categorical Logical vector, one per column of `X`.
#' @param n_members Ensemble size.
#' @param seed Generator seed.
#' @param max_columns Columns kept per member after the shuffle.
#' @param max_classes Classes the network separates in one pass. With
#'   more, each member also draws an error-correcting codebook (see
#'   [.kumo_ecoc_codebook()]) -- after every other draw, in member order,
#'   which is where the reference's network pass makes them.
#' @return A fitted recipe (plain R data).
#' @keywords internal
kumo_recipe_fit <- function(X, y, classification, categorical = NULL,
                            n_members = 8L, seed = 42L, max_columns = 500L,
                            max_classes = 10L) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  E <- as.integer(n_members)
  categorical <- categorical %||% rep(FALSE, ncol(X))
  g <- .kumo_generator(seed)
  options <- (seq_len(E) - 1L) %% 4L

  # Target first: that is where the generator is consumed first.
  target <- if (classification) {
    classes <- sort(unique(as.integer(y)))
    K <- length(classes)
    pool <- integer(); shifts <- integer(E)
    for (m in seq_len(E)) {
      if (K <= 1L) { shifts[m] <- 0L; next }
      if (!length(pool)) pool <- .kumo_randperm(K, g)
      shifts[m] <- pool[1]; pool <- pool[-1]
    }
    list(classes = classes, shifts = shifts)
  } else {
    y32 <- torch::torch_tensor(matrix(as.numeric(y), ncol = 1L),
                               dtype = torch::torch_float32())
    st <- .kumo_standardize_fit(y32, eps = 0)
    signs <- vapply(seq_len(E), function(m) {
      1 - 2 * as.numeric(.kumo_bernoulli(c(1L, 1L), torch::torch_float32(), g))
    }, numeric(1))
    list(mean = st$mean, scale = st$scale, signs = signs)
  }

  num <- which(!categorical); cat_ <- which(categorical)
  rec <- list(classification = classification, n_members = E, options = options,
              ncol = ncol(X), num = num, cat = cat_, target = target)

  # Numerical route. The reference reads the frame into float32 before
  # its own cast to float64, so the values are rounded there first.
  if (length(num)) {
    x <- .kumo_round_f64(X[, num, drop = FALSE])
    rec$num_proc <- .kumo_numproc_fit(x, options)
    cn <- sum(rec$num_proc$keep)
    rec$num_signs <- lapply(seq_len(E), function(m) {
      as.numeric(1 - 2 * .kumo_bernoulli(c(1L, cn), torch::torch_float64(), g))
    })
  }

  # Categorical route: no draws.
  if (length(cat_)) {
    vocab <- lapply(cat_, function(j) .kumo_vocab(X[, j]))
    codes <- vapply(seq_along(cat_), function(i) .kumo_codes(X[, cat_[i]], vocab[[i]]),
                    integer(nrow(X)))
    if (!is.matrix(codes)) codes <- matrix(codes, nrow = nrow(X))
    big <- which(lengths(vocab) > 50L)
    counts <- lapply(big, function(i) {
      k <- length(vocab[[i]])
      slot <- ifelse(codes[, i] >= 0L, codes[, i], k)
      tab <- tabulate(slot + 1L, nbins = k + 1L)
      as.numeric(torch::torch_tensor(tab, dtype = torch::torch_float32())$log1p())
    })
    rec$cat_vocab <- vocab
    rec$cat_counts <- counts
    rec$cat_big <- big
    rec$cat_proc <- .kumo_numproc_fit(.kumo_cat_block(rec, X), options)
  }

  # Names, the categorical flag and the column permutation.
  cols <- .kumo_columns(rec)
  W <- length(cols$names)
  base <- .kumo_randperm(W, g)
  rws <- .kumo_randperm(W, g)
  rec$perms <- lapply(seq_len(E) - 1L, function(m) {
    if (W <= 1L) return(seq_len(W) - 1L)
    base[((m %% W) - rws) %% W + 1L]
  })
  rec$max_columns <- as.integer(max_columns)

  K <- length(target$classes)
  if (classification && K > max_classes) {
    rec$target$codebooks <- lapply(seq_len(E), function(m) {
      .kumo_ecoc_codebook(K, g, max_classes)
    })
  }
  rec
}

#' Draw one member's error-correcting output codebook
#'
#' A port of `sdm.models.ECOC._draw_codebook()` (itself from TabPFN
#' Extensions). Each of `T = max(ceil(K / 9), 4 ceil(log_10 K))` sub-tasks
#' gives 9 classes their own symbol `0..8` and merges the rest into
#' symbol 9; classes are picked least-covered first, with a random
#' tie-break. With `K <= 200`, 50 candidate codebooks are drawn and the
#' one with the largest minimum pairwise Hamming distance kept (ties: the
#' largest total distance, then the first). Same generator calls, in the
#' same order, as the reference.
#'
#' @param K Number of classes.
#' @param g A torch generator.
#' @param max_classes Network classes per pass.
#' @return Integer matrix `[T, K]` of 0-based symbols.
#' @keywords internal
.kumo_ecoc_codebook <- function(K, g, max_classes = 10L) {
  rest <- as.integer(max_classes) - 1L
  T_ <- as.integer(max(ceiling(K / rest),
                       4 * ceiling(log(K) / log(max_classes))))
  nd <- if (K <= 200L) 50L else 1L
  book <- array(rest, c(nd, T_, K))
  coverage <- torch::torch_zeros(nd, K)   # float32, as the reference's
  for (t in seq_len(T_)) {
    priority <- coverage + 0.1 * .kumo_rand(c(nd, K), g)
    chosen <- torch::torch_argsort(priority, dim = -1L)[, 1:rest]
    symbols <- as.matrix(torch::torch_argsort(.kumo_rand(c(nd, rest), g),
                                              dim = -1L)) - 1L
    ch <- as.matrix(chosen)
    for (d in seq_len(nd)) book[d, t, ch[d, ]] <- symbols[d, ]
    coverage$scatter_add_(-1L, chosen, torch::torch_ones_like(chosen,
                                                               dtype = coverage$dtype))
  }
  if (nd == 1L) return(matrix(book[1, , ], T_, K))

  pairs <- which(upper.tri(diag(K)), arr.ind = TRUE)
  score <- vapply(seq_len(nd), function(d) {
    b <- matrix(book[d, , ], T_, K)
    dist <- colSums(b[, pairs[, 1], drop = FALSE] != b[, pairs[, 2], drop = FALSE])
    min(dist) * (nrow(pairs) * T_ + 1) + sum(dist)
  }, numeric(1))
  matrix(book[which.max(score), , ], T_, K)
}

# Float32 rounding, then float64 -- what `TableTensor.from_pandas` and
# `Cast(float64)` do to a numerical column.
.kumo_round_f64 <- function(x) {
  torch::torch_tensor(x, dtype = torch::torch_float32())$to(dtype = torch::torch_float64())
}

# The categorical route's numerical block before numerical_processor():
# log-count columns, then the codes, all float32 (the sub-table's dtype)
# and then float64.
.kumo_cat_block <- function(rec, X) {
  codes <- vapply(seq_along(rec$cat),
                  function(i) .kumo_codes(X[, rec$cat[i]], rec$cat_vocab[[i]]),
                  integer(nrow(X)))
  if (!is.matrix(codes)) codes <- matrix(codes, nrow = nrow(X))
  cnt <- lapply(seq_along(rec$cat_big), function(b) {
    i <- rec$cat_big[b]; k <- length(rec$cat_vocab[[i]])
    slot <- ifelse(codes[, i] >= 0L, codes[, i], k)
    rec$cat_counts[[b]][slot + 1L]
  })
  blk <- cbind(do.call(cbind, c(cnt, list(matrix(numeric(0), nrow(X), 0)))),
               codes)
  storage.mode(blk) <- "double"
  .kumo_round_f64(blk)
}

# Column names and categorical flags after both routes, before the
# shuffle: numericals that survived, then count columns, then codes.
.kumo_columns <- function(rec) {
  nm <- character(); is_cat <- logical()
  if (length(rec$num)) {
    k <- rec$num[rec$num_proc$keep]
    nm <- c(nm, sprintf("x%d", k)); is_cat <- c(is_cat, rep(FALSE, length(k)))
  }
  if (length(rec$cat)) {
    # `sprintf()`, not `paste0()`: the latter turns no count columns into
    # one called "x__count".
    raw <- c(sprintf("x%d__count", rec$cat[rec$cat_big]), sprintf("x%d", rec$cat))
    flag <- c(rep(FALSE, length(rec$cat_big)), rep(TRUE, length(rec$cat)))
    keep <- rec$cat_proc$keep
    nm <- c(nm, raw[keep]); is_cat <- c(is_cat, flag[keep])
  }
  list(names = nm, categorical = is_cat)
}

#' Transform rows into one member's network input
#'
#' @param rec A fitted recipe from [kumo_recipe_fit()].
#' @param X Rows to transform (context, query, or both stacked).
#' @param m Member index, 1-based.
#' @return `list(x, categorical_mask, columns)`: `x` a float32 tensor
#'   `[nrow(X), C]`.
#' @keywords internal
kumo_recipe_member <- function(rec, X, m) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  o <- rec$options[m]
  parts <- list()
  if (length(rec$num)) {
    x <- .kumo_numproc(rec$num_proc, .kumo_round_f64(X[, rec$num, drop = FALSE]), o)
    if (x$size(2) > 0L) x <- x * .kumo_f64(rec$num_signs[[m]])
    parts <- c(parts, list(x))
  }
  if (length(rec$cat)) {
    parts <- c(parts, list(.kumo_numproc(rec$cat_proc, .kumo_cat_block(rec, X), o)))
  }
  cols <- .kumo_columns(rec)
  x <- if (length(parts)) torch::torch_cat(parts, dim = 2L)
       else torch::torch_zeros(nrow(X), 0L, dtype = torch::torch_float64())
  perm <- rec$perms[[m]]
  x <- .kumo_cols(x, perm + 1L)
  keep <- seq_len(min(length(perm), rec$max_columns))
  x <- .kumo_cols(x, keep)
  list(x = x$to(dtype = torch::torch_float32()),
       categorical_mask = cols$categorical[perm + 1L][keep],
       columns = cols$names[perm + 1L][keep])
}

#' One member's target as the network sees it
#'
#' @inheritParams kumo_recipe_member
#' @param y Context targets: 0-based class codes, or numbers.
#' @return A long tensor of shifted codes, or a float32 tensor of signed
#'   standardized values, `[length(y)]`.
#' @keywords internal
kumo_recipe_target <- function(rec, y, m) {
  t <- rec$target
  if (rec$classification) {
    K <- length(t$classes)
    code <- match(as.integer(y), t$classes) - 1L
    if (K > 1L) code <- (code - t$shifts[m]) %% K
    return(torch::torch_tensor(code, dtype = torch::torch_long()))
  }
  y32 <- torch::torch_tensor(as.numeric(y), dtype = torch::torch_float32())
  mu <- torch::torch_tensor(t$mean, dtype = torch::torch_float32())
  sc <- torch::torch_tensor(t$scale, dtype = torch::torch_float32())
  ((y32 - mu)$div(sc)) * t$signs[m]
}

#' Undo one member's target transform on its quantile grid
#'
#' FlipSign's inverse then Standardize's, in float32, then sorted.
#'
#' @param q Float32 tensor `[n, Q]`.
#' @keywords internal
kumo_recipe_inverse <- function(rec, q, m) {
  t <- rec$target
  mu <- torch::torch_tensor(t$mean, dtype = torch::torch_float32())
  sc <- torch::torch_tensor(t$scale, dtype = torch::torch_float32())
  out <- (q * t$signs[m]) * sc + mu
  torch::torch_sort(out, dim = -1L)[[1]]
}

#' Member logits back in the value-sorted class order
#'
#' Member `m`'s column `j` is class `(j + shift) mod K`; this returns
#' column `k` as class `k`.
#'
#' @param logits Tensor `[n, K]` (already cut to the class count).
#' @keywords internal
kumo_recipe_unshift <- function(rec, logits, m) {
  K <- length(rec$target$classes)
  if (K <= 1L) return(logits)
  idx <- ((seq_len(K) - 1L) - rec$target$shifts[m]) %% K + 1L
  logits$index_select(-1L, torch::torch_tensor(idx, dtype = torch::torch_long()))
}

#' Combine member outputs as the reference's output recipe does
#'
#' Classification: mean of the logits, then a softmax. Regression: a
#' per-coordinate trimmed mean, dropping `floor(trim * E)` members from
#' each tail.
#'
#' @param outs List of float32 tensors `[n, K]` or `[n, Q]`.
#' @keywords internal
kumo_recipe_reduce <- function(outs, classification, trim = 0.2) {
  s <- torch::torch_stack(outs, dim = 1L)
  if (classification) {
    return(torch::nnf_softmax(s$mean(dim = 1L), dim = -1L))
  }
  E <- length(outs)
  if (E == 1L) return(s[1, , ])
  cut <- as.integer(floor(trim * E))
  if (cut > 0L) s <- torch::torch_sort(s, dim = 1L)[[1]][(cut + 1L):(E - cut), , ]
  s$mean(dim = 1L)
}
