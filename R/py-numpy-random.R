# NumPy's legacy `RandomState`, ported.
#
# `R/py-random.R` ports CPython's `random.Random`, because the TabICL and
# TabFM wrappers draw their ensembles from it. scikit-learn draws from
# NumPy instead: `check_random_state(0)` is `np.random.RandomState(0)`, and
# that is what seeds the randomized SVD at the end of skrub's
# `StringEncoder` -- the text preprocessor TabPFN v3.5 turns on.
#
# The two generators share their core. Both are MT19937, and CPython's
# `random()` and NumPy's `random_sample()` assemble a double from two
# 32-bit draws identically. They differ in exactly two places:
#
# * **Seeding.** CPython seeds an integer through `init_by_array`; NumPy's
#   `RandomState(int)` calls `mt19937_seed`, which is plain
#   `init_genrand`. The streams disagree from the first draw.
# * **Gaussians.** NumPy's legacy `standard_normal` is the Marsaglia polar
#   method with a cached second deviate (`legacy_gauss`), which has no
#   counterpart in the CPython port because nothing here needed one.
#
# So this file reuses the MT19937 core in `R/py-random.R` unchanged and adds
# only those two things. It is deliberately a separate generator rather
# than a seeding option on `py_random()`: a caller that means NumPy and
# gets CPython does not get an error, it gets a different ensemble.
#
# Draws are made a whole MT19937 block (624 words) at a time. The
# randomized SVD needs one Gaussian per n-gram per component -- 92,000 for
# a 2,300-gram vocabulary -- and a closure per 32-bit word would spend
# seconds on bookkeeping. The block is exactly what the scalar generator
# would produce, just tempered in one vectorised pass.
#
# Reference: NumPy `numpy/random/_mt19937.pyx` (`_legacy_seeding`) and
# `numpy/random/src/legacy/legacy-distributions.c` (`legacy_gauss`).

#' A NumPy-compatible legacy `RandomState`
#'
#' Mutable by design, like the reference: every draw advances the stream.
#' Hold one in a variable and call its methods in the order the Python code
#' does.
#'
#' @param seed Integer seed in `[0, 2^32)`, as `RandomState(seed)` accepts.
#' @return An environment with `random_sample(n)`, `standard_normal(n)` and
#'   `normal(n, loc, scale)`.
#' @keywords internal
np_random_state <- function(seed) {
  seed <- as.numeric(seed)
  if (length(seed) != 1L || !is.finite(seed) || seed != floor(seed) ||
      seed < 0 || seed >= .U32) {
    cli::cli_abort(
      "{.arg seed} must be a whole number in [0, 2^32), as {.code RandomState} requires."
    )
  }

  self <- new.env(parent = emptyenv())
  self$mt <- .mt_init_genrand(seed)
  # Tempered output words not yet consumed. Empty, so the first draw
  # regenerates -- `init_genrand` leaves `mti = N` in the reference too.
  self$buf <- numeric(0)
  # A second polar-method deviate waiting to be returned, or NULL.
  self$gauss_cache <- NULL

  # One MT19937 block, tempered. The same four steps as `py_random()`'s
  # `genrand`, applied to all 624 words at once.
  refill <- function() {
    self$mt <- .mt_regenerate(self$mt)
    y <- self$mt
    y <- .xor32(y, .shr32(y, 11))
    y <- .xor32(y, .and32(.shl32(y, 7), .MT_TEMPER_B))
    y <- .xor32(y, .and32(.shl32(y, 15), .MT_TEMPER_C))
    .xor32(y, .shr32(y, 18))
  }

  # The next `k` words, without consuming them. Needed because the polar
  # method rejects a data-dependent number of pairs, and the stream must
  # end exactly where the reference's does -- so candidates are inspected
  # first and only the ones actually used are taken.
  peek <- function(k) {
    short <- k - length(self$buf)
    if (short > 0) {
      # All the blocks needed, then one concatenation. Appending a block at
      # a time copies the growing buffer on every append -- quadratic, and
      # at the ~2 million words a large vocabulary needs, seven seconds of
      # nothing but copying.
      blocks <- vector("list", ceiling(short / .MT_N))
      for (b in seq_along(blocks)) blocks[[b]] <- refill()
      self$buf <- c(self$buf, unlist(blocks, use.names = FALSE))
    }
    self$buf[seq_len(k)]
  }
  consume <- function(k) {
    if (k > 0L) self$buf <- self$buf[-seq_len(k)]
    invisible(NULL)
  }

  # 53-bit doubles from pairs of words: `(a >> 5) * 2^26 + (b >> 6)`,
  # over 2^53. Identical to CPython's `random()`.
  words_to_doubles <- function(w) {
    a <- .shr32(w[c(TRUE, FALSE)], 5)
    b <- .shr32(w[c(FALSE, TRUE)], 6)
    (a * 67108864 + b) / 9007199254740992
  }

  self$random_sample <- function(n = 1L) {
    n <- as.integer(n)
    if (n <= 0L) return(numeric(0))
    w <- peek(2L * n)
    consume(2L * n)
    words_to_doubles(w)
  }

  # `legacy_gauss`: draw `x1, x2` uniform on (-1, 1) until
  # `0 < x1^2 + x2^2 < 1`, return `f * x2` and cache `f * x1` for the next
  # call. The order -- x2 first -- is the reference's, and getting it
  # backwards swaps every pair.
  self$standard_normal <- function(n = 1L) {
    n <- as.integer(n)
    out <- numeric(0)
    if (n <= 0L) return(out)
    if (!is.null(self$gauss_cache)) {
      out <- self$gauss_cache
      self$gauss_cache <- NULL
    }
    need <- n - length(out)
    while (need > 0L) {
      pairs <- ceiling(need / 2)
      # Each attempt costs four words. About 21% of attempts are rejected
      # (1 - pi/4), so over-peek and then take exactly what was used.
      attempts <- as.integer(ceiling(pairs * 1.35)) + 8L
      d <- words_to_doubles(peek(4L * attempts))
      x1 <- 2 * d[c(TRUE, FALSE)] - 1
      x2 <- 2 * d[c(FALSE, TRUE)] - 1
      r2 <- x1 * x1 + x2 * x2
      ok <- which(r2 < 1 & r2 != 0)
      if (length(ok) >= pairs) {
        ok <- ok[seq_len(pairs)]
        consume(4L * ok[pairs])
      } else {
        consume(4L * attempts)
      }
      f <- sqrt(-2 * log(r2[ok]) / r2[ok])
      z <- as.vector(rbind(f * x2[ok], f * x1[ok]))
      take <- min(length(z), need)
      out <- c(out, z[seq_len(take)])
      # An odd request leaves the pair's second deviate for next time.
      if (take < length(z)) self$gauss_cache <- z[take + 1L]
      need <- need - take
    }
    out
  }

  # `RandomState.normal(loc, scale, size)`: `loc + scale * gauss`, one
  # Gaussian per element, drawn in C order. A caller wanting a matrix
  # fills it by row.
  self$normal <- function(n = 1L, loc = 0, scale = 1) {
    loc + scale * self$standard_normal(n)
  }

  self
}
