# CPython's `random.Random`, ported.
#
# The TabICL and TabFM sklearn wrappers do not merely *use* randomness --
# they use `random.Random(random_state)` specifically, and every ensemble
# member's identity comes out of that one stream: which feature
# permutation it gets, which class shift, which normalisation method.
# Two implementations that both "shuffle with seed 42" produce different
# ensembles, and no amount of tolerance in a parity check papers over
# that: member 3 is comparing a different view of the data.
#
# So the stream itself has to match, bit for bit. That means MT19937 with
# CPython's `init_by_array` seeding, CPython's `getrandbits` and its
# rejection-sampling `_randbelow`, and CPython's `shuffle` / `sample` /
# `choice` on top -- R's `sample()` is a different algorithm on a
# different generator and agrees with none of it.
#
# Reference: CPython `Modules/_randommodule.c` and `Lib/random.py`.

# ---------------------------------------------------------------------------
# uint32 arithmetic on doubles
# ---------------------------------------------------------------------------
#
# R has no unsigned 32-bit type, and its `bitw*` helpers work on *signed*
# 32-bit ints, so anything with the top bit set is out of reach. Values
# are therefore carried as doubles in `[0, 2^32)` -- exact, since 2^32
# fits comfortably inside a double's 53-bit mantissa -- and the bitwise
# operations are done a half-word at a time, where `bitwAnd` and friends
# are safe.

.U32 <- 4294967296     # 2^32
.U16 <- 65536          # 2^16

# @keywords internal
.xor32 <- function(a, b) {
  ah <- a %/% .U16; al <- a %% .U16
  bh <- b %/% .U16; bl <- b %% .U16
  bitwXor(ah, bh) * .U16 + bitwXor(al, bl)
}

# @keywords internal
.and32 <- function(a, b) {
  ah <- a %/% .U16; al <- a %% .U16
  bh <- b %/% .U16; bl <- b %% .U16
  bitwAnd(ah, bh) * .U16 + bitwAnd(al, bl)
}

# @keywords internal
.or32 <- function(a, b) {
  ah <- a %/% .U16; al <- a %% .U16
  bh <- b %/% .U16; bl <- b %% .U16
  bitwOr(ah, bh) * .U16 + bitwOr(al, bl)
}

# @keywords internal
.shr32 <- function(x, k) x %/% (2^k)

# @keywords internal
.shl32 <- function(x, k) (x * 2^k) %% .U32

# `(a * b) mod 2^32` without ever forming `a * b`, which for two uint32s
# reaches 1.8e19 and would round. Splitting `a` at 16 bits keeps every
# intermediate under 2^48.
# @keywords internal
.mul32 <- function(a, b) {
  lo <- (a %% .U16) * b
  hi <- ((a %/% .U16) * b) %% .U16
  (hi * .U16 + lo) %% .U32
}


# ---------------------------------------------------------------------------
# MT19937
# ---------------------------------------------------------------------------

.MT_N <- 624L
.MT_M <- 397L
.MT_MATRIX_A  <- 2567483615   # 0x9908b0df
.MT_UPPER     <- 2147483648   # 0x80000000
.MT_LOWER     <- 2147483647   # 0x7fffffff
.MT_TEMPER_B  <- 2636928640   # 0x9d2c5680
.MT_TEMPER_C  <- 4022730752   # 0xefc60000

# @keywords internal
.mt_init_genrand <- function(s) {
  mt <- numeric(.MT_N)
  mt[1L] <- s %% .U32
  for (i in 2L:.MT_N) {
    prev <- mt[i - 1L]
    mt[i] <- (.mul32(1812433253, .xor32(prev, .shr32(prev, 30))) + (i - 1L)) %% .U32
  }
  mt
}

# CPython seeds from an integer via `init_by_array`, never `init_genrand`.
# The two disagree from the first draw, so this is not an implementation
# detail that can be simplified away.
# @keywords internal
.mt_init_by_array <- function(key) {
  mt <- .mt_init_genrand(19650218)
  i <- 1L; j <- 0L
  k <- max(.MT_N, length(key))
  while (k > 0L) {
    prev <- mt[i]                              # mt[i-1] in 0-based terms
    mt[i + 1L] <- (.xor32(mt[i + 1L],
                          .mul32(.xor32(prev, .shr32(prev, 30)), 1664525)) +
                     key[j + 1L] + j) %% .U32
    i <- i + 1L; j <- j + 1L
    if (i >= .MT_N) { mt[1L] <- mt[.MT_N]; i <- 1L }
    if (j >= length(key)) j <- 0L
    k <- k - 1L
  }
  k <- .MT_N - 1L
  while (k > 0L) {
    prev <- mt[i]
    mt[i + 1L] <- (.xor32(mt[i + 1L],
                          .mul32(.xor32(prev, .shr32(prev, 30)), 1566083941)) -
                     i) %% .U32
    i <- i + 1L
    if (i >= .MT_N) { mt[1L] <- mt[.MT_N]; i <- 1L }
    k <- k - 1L
  }
  mt[1L] <- .MT_UPPER
  mt
}

# Refill the whole state. Written as three vectorised slices rather than
# one 624-iteration loop: the recurrence reads `mt[kk + M]` (mod N), and
# the read either lands entirely in already-updated territory or entirely
# in not-yet-updated territory within each slice, so each slice can be
# done at once. Boundaries: slice 1 writes 0..N-M-1, slice 2 writes
# N-M..2(N-M)-1 (reading only slice 1's output), slice 3 the rest.
# @keywords internal
.mt_regenerate <- function(mt) {
  n <- .MT_N; m <- .MT_M
  # Slice 1: kk = 0 .. n-m-1 (0-based), reads mt[kk+m], untouched.
  kk <- 0:(n - m - 1L)
  y <- .or32(.and32(mt[kk + 1L], .MT_UPPER), .and32(mt[kk + 2L], .MT_LOWER))
  mt[kk + 1L] <- .xor32(.xor32(mt[kk + m + 1L], .shr32(y, 1)),
                        ifelse(y %% 2 == 1, .MT_MATRIX_A, 0))
  # Slice 2: kk = n-m .. 2(n-m)-1, reads mt[kk+m-n] = 0 .. n-m-1 (slice 1).
  lo <- n - m; hi <- min(2L * (n - m) - 1L, n - 2L)
  kk <- lo:hi
  y <- .or32(.and32(mt[kk + 1L], .MT_UPPER), .and32(mt[kk + 2L], .MT_LOWER))
  mt[kk + 1L] <- .xor32(.xor32(mt[kk + m - n + 1L], .shr32(y, 1)),
                        ifelse(y %% 2 == 1, .MT_MATRIX_A, 0))
  # Slice 3: the remainder, reads mt[kk+m-n] written by slice 2.
  if (hi + 1L <= n - 2L) {
    kk <- (hi + 1L):(n - 2L)
    y <- .or32(.and32(mt[kk + 1L], .MT_UPPER), .and32(mt[kk + 2L], .MT_LOWER))
    mt[kk + 1L] <- .xor32(.xor32(mt[kk + m - n + 1L], .shr32(y, 1)),
                          ifelse(y %% 2 == 1, .MT_MATRIX_A, 0))
  }
  # Wrap-around element.
  y <- .or32(.and32(mt[n], .MT_UPPER), .and32(mt[1L], .MT_LOWER))
  mt[n] <- .xor32(.xor32(mt[m], .shr32(y, 1)),
                  if (y %% 2 == 1) .MT_MATRIX_A else 0)
  mt
}


# ---------------------------------------------------------------------------
# The generator object
# ---------------------------------------------------------------------------

#' A CPython-compatible `random.Random`
#'
#' Mutable by design: every draw advances the stream, exactly as the
#' reference's generator object does. Hold one in a variable and call the
#' methods on it in the same order the Python code does.
#'
#' @param seed Non-negative integer seed, or `NULL` to draw one from R's
#'   own RNG (so `set.seed()` still makes a run reproducible). Matches
#'   `random.Random(seed)`; CPython takes `abs(seed)`, so a negative seed
#'   is folded the same way here.
#' @return An environment with `random()`, `getrandbits(k)`,
#'   `randbelow(n)`, `shuffle(x)`, `sample(population, k)` and
#'   `choice(seq)`.
#' @keywords internal
py_random <- function(seed = NULL) {
  if (is.null(seed)) seed <- floor(stats::runif(1L, 0, 2^31))
  seed <- abs(as.numeric(seed))
  if (!is.finite(seed) || seed != floor(seed)) {
    cli::cli_abort("{.arg seed} must be a whole number.")
  }
  # Little-endian 32-bit words of the seed, as `_PyLong_AsByteArray` lays
  # them out. Anything under 2^32 is a single word, which is every seed
  # this package actually passes.
  key <- if (seed < .U32) seed else c(seed %% .U32, floor(seed / .U32))

  self <- new.env(parent = emptyenv())
  self$mt <- .mt_init_by_array(key)
  self$mti <- .MT_N       # forces a refill on the first draw

  self$genrand <- function() {
    if (self$mti >= .MT_N) {
      self$mt <- .mt_regenerate(self$mt)
      self$mti <- 0L
    }
    y <- self$mt[self$mti + 1L]
    self$mti <- self$mti + 1L
    y <- .xor32(y, .shr32(y, 11))
    y <- .xor32(y, .and32(.shl32(y, 7), .MT_TEMPER_B))
    y <- .xor32(y, .and32(.shl32(y, 15), .MT_TEMPER_C))
    .xor32(y, .shr32(y, 18))
  }

  # `random.random()`: 53 bits assembled from two 32-bit draws.
  self$random <- function() {
    a <- .shr32(self$genrand(), 5)
    b <- .shr32(self$genrand(), 6)
    (a * 67108864 + b) / 9007199254740992
  }

  self$getrandbits <- function(k) {
    k <- as.integer(k)
    if (k == 0L) return(0)
    if (k > 32L) {
      # Reachable only for populations above 2^32; nothing in this
      # package gets near it, and guessing would be worse than saying so.
      cli::cli_abort("{.fn getrandbits} above 32 bits is not implemented.")
    }
    .shr32(self$genrand(), 32L - k)
  }

  # `_randbelow_with_getrandbits`: draw `bit_length(n)` bits and reject
  # anything >= n. The rejections consume stream, which is why an
  # equivalent-in-distribution shortcut would still desynchronise.
  self$randbelow <- function(n) {
    n <- as.numeric(n)
    if (n <= 0) return(0)
    k <- floor(log2(n)) + 1L
    # log2 of an exact power of two can land a hair low in floating point.
    if (2^(k - 1L) > n) k <- k - 1L
    if (2^k < n) k <- k + 1L
    repeat {
      r <- self$getrandbits(k)
      if (r < n) return(r)
    }
  }

  # `random.shuffle`, Fisher-Yates downward. Takes and returns a vector;
  # Python mutates in place.
  self$shuffle <- function(x) {
    n <- length(x)
    if (n < 2L) return(x)
    for (i in seq.int(n, 2L)) {
      j <- self$randbelow(i) + 1L     # Python: randbelow(i+1) on 0-based i
      tmp <- x[[i]]; x[[i]] <- x[[j]]; x[[j]] <- tmp
    }
    x
  }

  # `random.sample`. Both branches are implemented because they consume
  # the stream differently, and which one runs depends on `n` and `k`.
  self$sample <- function(population, k) {
    n <- length(population)
    k <- as.integer(k)
    if (k < 0L || k > n) {
      cli::cli_abort("Sample larger than population.")
    }
    setsize <- 21
    if (k > 5L) setsize <- setsize + 4^ceiling(log(k * 3) / log(4))
    result <- vector("list", k)
    if (n <= setsize) {
      pool <- as.list(population)
      for (i in seq_len(k)) {
        j <- self$randbelow(n - (i - 1L)) + 1L
        result[[i]] <- pool[[j]]
        pool[[j]] <- pool[[n - i + 1L]]
      }
    } else {
      selected <- new.env(parent = emptyenv())
      for (i in seq_len(k)) {
        j <- self$randbelow(n)
        while (!is.null(selected[[as.character(j)]])) j <- self$randbelow(n)
        assign(as.character(j), TRUE, envir = selected)
        result[[i]] <- population[[j + 1L]]
      }
    }
    if (is.list(population)) result else unlist(result, use.names = FALSE)
  }

  self$choice <- function(seq_) seq_[[self$randbelow(length(seq_)) + 1L]]

  self
}
