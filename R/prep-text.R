# Text columns, as TabPFN v3.5's estimator expands them.
#
# Port of `tabpfn.preprocessing.text.TextTransformer`'s per-column step,
# which is skrub's `StringEncoder` with `n_components = 30` and
# `random_state = 0`: latent semantic analysis over character n-grams.
#
#   strings -> char_wb 3- and 4-grams -> tf-idf -> float32
#           -> randomized truncated SVD (seeded) -> block-normalised
#
# A string is encoded by the n-grams it shares with the training column, so
# an unseen sentence in the same language lands near its neighbours, and a
# missing value -- encoded as the empty string, which shares none -- becomes
# an all-zero row.
#
# The expensive-looking part is the only part that needed care. scikit-learn's
# `TruncatedSVD` defaults to `algorithm = "randomized"`, which draws a
# Gaussian test matrix from `np.random.RandomState(0)`. On a well-conditioned
# column that barely matters; on a flat spectrum -- few rows, many distinct
# n-grams, which is ordinary for short text -- the trailing components are
# decided by the draw rather than by the data, and two seeds disagree by
# nearly one unit. So the draw is reproduced exactly ([np_random_state()]),
# and the SVD around it follows `sklearn.utils.extmath._randomized_svd` step
# for step.
#
# What is *not* reproduced is float32 arithmetic. The reference casts the
# tf-idf matrix to float32 and the SVD then runs in float32; R has no
# single-precision LAPACK. The port rounds the matrix to float32 where the
# reference does ([.round_float32()]) and computes in float64 from there. It
# is therefore scikit-learn's algorithm run at higher precision, and the gap
# to the reference is scikit-learn's own float32-versus-float64 gap --
# measured at 2e-5 on a well-conditioned column and 2e-4 on a very flat one,
# far below the 0.8 that a different seed costs.
#
# Reference: scikit-learn `feature_extraction/text.py` (`_char_wb_ngrams`,
# `TfidfTransformer`), `utils/extmath.py` (`_randomized_svd`,
# `_randomized_range_finder`, `svd_flip`), `decomposition/_truncated_svd.py`;
# skrub `_string_encoder.py` and `_scaling_factor.py`.

# The characters Python's `str.split()` treats as whitespace: Unicode
# White_Space plus the four ASCII separators \x1c-\x1f, which PCRE's `\s`
# does not include. Splitting on anything else would give the same words
# for ordinary text and different ones for a pasted spreadsheet cell.
#
# `(*UTF)` forces PCRE into UTF mode. Without it R compiles the pattern in
# byte mode whenever the input happens to be pure ASCII -- no string marked
# UTF-8, nothing to switch it on -- and the escapes above 0xFF then fail to
# compile at all.
.PY_WHITESPACE <- paste0(
  "(*UTF)",
  "[\\x{0009}-\\x{000D}\\x{001C}-\\x{001F}\\x{0020}\\x{0085}\\x{00A0}",
  "\\x{1680}\\x{2000}-\\x{200A}\\x{2028}\\x{2029}\\x{202F}\\x{205F}",
  "\\x{3000}]+"
)

#' Character n-grams within word boundaries
#'
#' Port of scikit-learn's `CountVectorizer._char_wb_ngrams`. Each word is
#' padded with one space either side, and every `n`-gram of the padded word
#' is emitted for `n` in `min_n..max_n`. A word too short for `n` is
#' emitted whole, once, and then contributes nothing longer -- so `"a"`
#' yields `" a "` and no 4-gram, where a naive sliding window would yield
#' nothing at all.
#'
#' Lowercasing happens first, as `lowercase = TRUE` does; accents are left
#' alone, as `strip_accents = None` does.
#'
#' @param docs Character vector; `NA` should already be `""`.
#' @param min_n,max_n The n-gram range.
#' @return A list with `doc` (the 1-based document of each n-gram) and
#'   `gram` (the n-gram), in the reference's emission order.
#' @keywords internal
.char_wb_ngrams <- function(docs, min_n = 3L, max_n = 4L) {
  docs <- tolower(enc2utf8(as.character(docs)))
  words <- strsplit(docs, .PY_WHITESPACE, perl = TRUE)
  # `str.split()` drops empty strings at the ends; `strsplit` keeps a
  # leading one.
  words <- lapply(words, function(w) w[nzchar(w)])
  n_words <- lengths(words)
  if (!sum(n_words)) return(list(doc = integer(0), gram = character(0)))

  doc_of_word <- rep.int(seq_along(docs), n_words)
  w <- paste0(" ", unlist(words, use.names = FALSE), " ")
  len <- nchar(w, type = "chars")

  owners <- list(); ns <- list(); starts <- list()
  active <- rep(TRUE, length(w))
  for (n in seq.int(min_n, max_n)) {
    idx <- which(active)
    if (!length(idx)) break
    # A word longer than `n` yields every window; one no longer than `n`
    # yields itself, once.
    count <- pmax(1L, len[idx] - n + 1L)
    owners[[length(owners) + 1L]] <- rep.int(idx, count)
    ns[[length(ns) + 1L]] <- rep.int(n, sum(count))
    starts[[length(starts) + 1L]] <- sequence(count)
    # `if offset == 0: break` -- a word that fit in one window is done.
    active[idx[len[idx] <= n]] <- FALSE
  }
  owner <- unlist(owners, use.names = FALSE)
  n <- unlist(ns, use.names = FALSE)
  start <- unlist(starts, use.names = FALSE)
  # Built a length at a time for speed, then put back in the reference's
  # order: word by word, and within a word every 3-gram before any 4-gram.
  # The counts would not care, but a function that says it is a port of
  # the analyzer should emit what the analyzer emits.
  o <- order(owner, n, start)
  owner <- owner[o]; n <- n[o]; start <- start[o]
  list(doc = doc_of_word[owner],
       gram = substring(w[owner], start, start + n - 1L))
}

# Document-term counts over a fixed vocabulary, as a sparse matrix. An
# n-gram outside the vocabulary is dropped, which is what `transform` does
# with terms it did not see at fit.
# @keywords internal
.ngram_counts <- function(docs, vocabulary) {
  ng <- .char_wb_ngrams(docs)
  j <- match(ng$gram, vocabulary)
  keep <- !is.na(j)
  Matrix::sparseMatrix(
    i = ng$doc[keep], j = j[keep], x = 1,
    dims = c(length(docs), length(vocabulary))
  )
}

#' Fit a tf-idf vectorizer over character n-grams
#'
#' `TfidfVectorizer(analyzer = "char_wb", ngram_range = c(3, 4))` with
#' scikit-learn's defaults: the vocabulary is every n-gram seen, sorted by
#' code point; `smooth_idf` gives `idf = log((1 + n) / (1 + df)) + 1`.
#'
#' Code-point order is what `sort(method = "radix")` gives on UTF-8
#' strings, because UTF-8 byte order *is* code-point order. The default
#' sort would follow the session's collation instead, and put `"B"` next to
#' `"b"` -- a different vocabulary order and so a different matrix.
#'
#' @param docs Character vector, `NA` already replaced by `""`.
#' @return A fit for [transform_tfidf()].
#' @keywords internal
fit_tfidf <- function(docs) {
  ng <- .char_wb_ngrams(docs)
  vocabulary <- sort(unique(ng$gram), method = "radix")
  counts <- .ngram_counts(docs, vocabulary)
  n <- nrow(counts)
  # Documents containing each term: nonzeros per column.
  df <- diff(counts@p)
  idf <- log((n + 1) / (df + 1)) + 1
  list(vocabulary = vocabulary, idf = idf)
}

#' Apply a fitted tf-idf vectorizer
#'
#' Raw counts times idf, then each row scaled to unit l2 norm, then rounded
#' to float32 -- skrub's `.astype("float32")`, which is what the SVD sees. An
#' all-zero row (a missing string, or one sharing no n-gram with the
#' vocabulary) stays all zero.
#'
#' @return A `dgCMatrix`, documents by vocabulary.
#' @keywords internal
transform_tfidf <- function(docs, fit) {
  x <- .ngram_counts(docs, fit$vocabulary)
  # On the nonzeros directly, in the reference's operations: one multiply
  # by the column's idf, then one *division* by the row's norm. Scaling by
  # a reciprocal instead differs in the last bit, which is enough to land
  # on the other side of a float32 rounding boundary now and then.
  col <- rep.int(seq_len(ncol(x)), diff(x@p))
  x@x <- x@x * fit$idf[col]
  norms <- sqrt(Matrix::rowSums(x^2))
  row <- x@i + 1L
  nz <- norms[row] != 0
  x@x[nz] <- x@x[nz] / norms[row][nz]
  x@x <- .round_float32(x@x)
  x
}

# LU factorisation with partial pivoting, returning `P %*% L` -- what
# `scipy.linalg.lu(A, permute_l = TRUE)` hands back and what the power
# iterations keep as their next basis. Works for tall and wide `A`: for an
# `m x n` matrix the result is `m x min(m, n)`, which is how the basis
# shrinks when the matrix has fewer rows than the 40 columns asked for.
#
# The pivot is the first entry of largest magnitude in the column, as
# LAPACK's `idamax` chooses it. Written out rather than taken from
# `Matrix::lu()`, which only factorises square matrices.
# @keywords internal
.lu_permute_l <- function(A) {
  m <- nrow(A); n <- ncol(A); p <- min(m, n)
  perm <- seq_len(m)
  for (k in seq_len(p)) {
    piv <- k - 1L + which.max(abs(A[k:m, k]))
    if (piv != k) {
      A[c(k, piv), ] <- A[c(piv, k), ]
      perm[c(k, piv)] <- perm[c(piv, k)]
    }
    if (k < m && A[k, k] != 0) {
      rows <- (k + 1L):m
      A[rows, k] <- A[rows, k] / A[k, k]
      if (k < n) {
        cols <- (k + 1L):n
        A[rows, cols] <- A[rows, cols] -
          A[rows, k, drop = FALSE] %*% A[k, cols, drop = FALSE]
      }
    }
  }
  L <- A[, seq_len(p), drop = FALSE]
  L[upper.tri(L)] <- 0
  diag(L) <- 1
  # `A[perm, ] = L U`, so the unpermuted rows of `L` are `P %*% L`.
  PL <- matrix(0, m, p)
  PL[perm, ] <- L
  PL
}

# `svd_flip(u, v, u_based_decision = FALSE)`: make the largest-magnitude
# entry of each row of `v` positive, and flip the matching column of `u`.
# `which.max` takes the first maximum, as `argmax` does.
# @keywords internal
.svd_flip_rows <- function(u, v) {
  signs <- vapply(seq_len(nrow(v)), function(r) {
    s <- sign(v[r, which.max(abs(v[r, ]))])
    if (s == 0) 1 else s
  }, numeric(1))
  list(u = if (is.null(u)) NULL else sweep(u, 2L, signs, `*`),
       v = sweep(v, 1L, signs, `*`))
}

#' Randomized truncated SVD, as scikit-learn computes it
#'
#' Port of `sklearn.utils.extmath._randomized_svd` with the arguments
#' `TruncatedSVD` passes: `n_oversamples = 10`, `n_iter = 5`,
#' `power_iteration_normalizer = "auto"` -- which resolves to LU, since
#' `n_iter > 2` -- `transpose = "auto"` and `flip_sign = FALSE`, followed by
#' `TruncatedSVD`'s own `svd_flip(u_based_decision = FALSE)`.
#'
#' `transpose = "auto"` means that when there are fewer rows than columns the
#' decomposition runs on `t(M)`, so the Gaussian test matrix is drawn with
#' as many rows as `M` has *rows*. Getting that wrong draws a different
#' shape from the same stream, which is every value different.
#'
#' The final QR is taken with base R's `qr()`. Its sign and pivoting
#' conventions differ from LAPACK's `geqrf`, and it does not matter: any
#' orthonormal basis of the same range gives the same `B` up to an
#' orthogonal factor, which the SVD absorbs and `svd_flip` then removes.
#'
#' @param M A matrix (sparse or dense), rows by columns.
#' @param n_components Components to keep.
#' @param seed The `RandomState` seed; `TruncatedSVD(random_state = 0)`.
#' @return `list(v, d)`: `v` is `n_components x ncol(M)` -- scikit-learn's
#'   `components_` -- and `d` the singular values.
#' @keywords internal
randomized_svd <- function(M, n_components, seed = 0L, n_oversamples = 10L,
                           n_iter = 5L) {
  n_random <- n_components + n_oversamples
  transpose <- nrow(M) < ncol(M)
  if (transpose) M <- Matrix::t(M)
  Mt <- Matrix::t(M)

  # `random_state.normal(size = (M.shape[1], n_random))`: C order, so the
  # stream fills the matrix a row at a time.
  rs <- np_random_state(seed)
  Q <- matrix(rs$normal(ncol(M) * n_random), nrow = ncol(M), byrow = TRUE)

  dense <- function(x) as.matrix(x)
  for (i in seq_len(n_iter)) {
    Q <- .lu_permute_l(dense(M %*% Q))
    Q <- .lu_permute_l(dense(Mt %*% Q))
  }
  Q <- qr.Q(qr(dense(M %*% Q)))

  B <- dense(Matrix::t(Q) %*% M)
  s <- svd(B)
  U <- Q %*% s$u
  k <- seq_len(min(n_components, length(s$d)))
  if (transpose) {
    # `Vt[:k].T, U[:, :k].T` -- the roles swap back.
    u <- s$v[, k, drop = FALSE]
    vt <- t(U[, k, drop = FALSE])
  } else {
    u <- U[, k, drop = FALSE]
    vt <- t(s$v)[k, , drop = FALSE]
  }
  flipped <- .svd_flip_rows(u, vt)
  list(v = flipped$v, d = s$d[k])
}

# skrub's block normaliser: the square root of the summed population
# variances of the columns, so that the whole embedding has unit total
# variance. Clipped to 1 below ten float32 epsilons, as the reference does
# -- its matrix is float32, so that is the epsilon it clips against.
# @keywords internal
.block_scaling_factor <- function(X) {
  v <- apply(X, 2L, function(col) {
    col <- col[!is.na(col)]
    if (!length(col)) return(NA_real_)
    mean((col - mean(col))^2)
  })
  f <- sqrt(sum(v, na.rm = TRUE))
  if (f < 10 * 1.1920928955078125e-07) 1 else f
}

#' Fit skrub's `StringEncoder` on one column
#'
#' Tf-idf over character n-grams, then a truncated SVD down to
#' `n_components`, then block normalisation. Three branches, as skrub has
#' them: when both dimensions exceed `n_components` the SVD runs; when the
#' vocabulary is exactly `n_components` wide the matrix passes through; and
#' otherwise its leading `n_components` columns are kept. The last two are
#' why the output can be narrower than asked, and why the width is recorded
#' at fit.
#'
#' @param x Character vector (a factor is read as its labels).
#' @param name Column name, for the output names.
#' @param n_components Target width.
#' @param seed `random_state` for the SVD. tabpfn fixes it at 0 so that the
#'   features a column becomes are a property of the data, not of the
#'   estimator's seed.
#' @return A fit for [transform_string_encoder()].
#' @keywords internal
fit_string_encoder <- function(x, name, n_components = 30L, seed = 0L) {
  docs <- .text_as_docs(x)
  tfidf <- fit_tfidf(docs)
  X <- transform_tfidf(docs, tfidf)
  k <- as.integer(n_components)

  if (min(dim(X)) > k) {
    svd_fit <- randomized_svd(X, k, seed = seed)
    projection <- t(svd_fit$v)
    result <- as.matrix(X %*% projection)
    mode <- "svd"
  } else {
    projection <- NULL
    width <- min(k, ncol(X))
    result <- as.matrix(X[, seq_len(width), drop = FALSE])
    mode <- if (ncol(X) == k) "passthrough" else "truncate"
  }
  scaling <- .block_scaling_factor(result)
  n_out <- ncol(result)
  digits <- nchar(as.character(n_out - 1L))
  structure(
    list(name = name, tfidf = tfidf, projection = projection, mode = mode,
         n_out = n_out, scaling = scaling,
         outputs = sprintf("%s_%0*d", name, digits, seq_len(n_out) - 1L)),
    class = "tabfound_string_fit"
  )
}

#' Apply a fitted `StringEncoder`
#'
#' The same projection and the same scaling as at fit, so the width holds
#' and an unseen string is placed by the n-grams it shares with the
#' training column.
#'
#' @return A numeric matrix, one column per component, named as at fit.
#' @keywords internal
transform_string_encoder <- function(x, fit) {
  docs <- .text_as_docs(x)
  X <- transform_tfidf(docs, fit$tfidf)
  result <- if (identical(fit$mode, "svd")) {
    as.matrix(X %*% fit$projection)
  } else {
    as.matrix(X[, seq_len(fit$n_out), drop = FALSE])
  }
  result <- result / fit$scaling
  dimnames(result) <- list(NULL, fit$outputs)
  result
}

# `ToStr(convert_category = TRUE)` then `fill_nulls("")`: labels for a
# factor, strings for anything string-like, and the empty string for a
# missing value -- which shares no n-gram with anything and so encodes as a
# row of zeros.
# @keywords internal
.text_as_docs <- function(x) {
  docs <- as.character(x)
  docs[is.na(docs)] <- ""
  docs
}
