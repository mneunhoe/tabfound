# The text encoder, against scikit-learn and skrub.
#
# Stored reference: `inst/parity/textdate/`, regenerated with
#   .venvs/ref35/bin/python inst/parity/textdate_reference.py \
#       --out inst/parity/textdate/textdate.safetensors
#
# The port is graded two ways. Against a float64 run of scikit-learn's own
# algorithm ("the twin") it must agree to rounding: that proves the n-grams,
# the tf-idf, the NumPy draw, the transpose and the LU pivoting are all the
# same. Against skrub's real output, which is computed in float32, it may be
# no further off than the twin itself is -- the remaining gap is
# scikit-learn's own float32 rounding, which R cannot reproduce and does not
# need to.

skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

ref_path <- tabfound_file("parity", "textdate", "textdate.safetensors.gz")
skip_if(!nzchar(ref_path), "text/date reference not installed")
ref <- read_reference_tensors(ref_path)
scl <- jsonlite::fromJSON(sub("\\.safetensors\\.gz$", ".json", ref_path),
                          simplifyVector = FALSE)
as_mat <- function(t) as.matrix(as.array(t$cpu()))
docs_of <- function(x) vapply(x, function(d) if (is.null(d)) NA_character_ else d, "")


test_that("character n-grams are scikit-learn's char_wb, in its order", {
  # Values from `CountVectorizer(analyzer = "char_wb", ngram_range = (3, 4))`.
  expect_identical(.char_wb_ngrams("Hello World")$gram,
                   c(" he", "hel", "ell", "llo", "lo ", " hel", "hell", "ello", "llo ",
                     " wo", "wor", "orl", "rld", "ld ", " wor", "worl", "orld", "rld "))
  # A one-letter word is emitted whole, once, and yields no 4-gram.
  expect_identical(.char_wb_ngrams("a")$gram, " a ")
  expect_identical(.char_wb_ngrams("I")$gram, " i ")
  # Runs of whitespace, a tab, and the \x1c separator Python splits on.
  expect_identical(.char_wb_ngrams("ab  cd\tef")$gram,
                   c(" ab", "ab ", " ab ", " cd", "cd ", " cd ", " ef", "ef ", " ef "))
  expect_identical(.char_wb_ngrams(paste0("x", intToUtf8(28), "y"))$gram,
                   c(" x ", " y "))
  # Pure-ASCII input, which is what exposed the byte-mode regex, and empty.
  expect_length(.char_wb_ngrams(c("plain", ""))$gram, 9L)   # 5 three- + 4 four-grams
  expect_identical(.char_wb_ngrams("")$gram, character(0))
  # Lowercasing reaches non-ASCII letters.
  expect_true(" ün" %in% .char_wb_ngrams("Ün")$gram)
})

for (name in c("wellcond", "flat", "veryflat")) {
  local({
    fx <- name
    test_that(paste0("the encoder reproduces skrub on the ", fx, " corpus"), {
      docs <- docs_of(scl[[paste0("text_", fx, "_docs")]])
      test <- docs_of(scl[[paste0("text_", fx, "_test_docs")]])
      real <- as_mat(ref[[paste0("text_", fx, "_real")]])
      twin <- as_mat(ref[[paste0("text_", fx, "_twin")]])

      fit <- fit_string_encoder(docs, "t", n_components = 30L, seed = 0L)
      expect_identical(fit$outputs, unlist(scl[[paste0("text_", fx, "_names")]]))
      got <- transform_string_encoder(docs, fit)

      # Tight against the twin: the algorithm is the same.
      expect_lt(max(abs(got - twin)), 1e-9)
      expect_lt(max(abs(t(fit$projection) -
                          as_mat(ref[[paste0("text_", fx, "_components_twin")]]))), 1e-10)
      # Against the real float32 output, no further off than the twin is.
      own_gap <- max(abs(twin - real))
      expect_lte(max(abs(got - real)), own_gap * 1.01 + 1e-9)

      # Unseen strings are placed by the n-grams they share; a missing one,
      # sharing none, is a row of zeros.
      got_test <- transform_string_encoder(test, fit)
      expect_lt(max(abs(got_test - as_mat(ref[[paste0("text_", fx, "_test_real")]]))), 5e-4)
      expect_true(all(got_test[3, ] == 0))
    })
  })
}

test_that("on a flat spectrum the seed decides the answer, which is why it is reproduced", {
  # The claim the NumPy port exists for: with any other seed the trailing
  # components are different numbers, not slightly different ones.
  docs <- docs_of(scl$text_veryflat_docs)
  twin <- as_mat(ref$text_veryflat_twin)
  other <- transform_string_encoder(docs, fit_string_encoder(docs, "t", seed = 1L))
  expect_gt(max(abs(other - twin)), 0.1)
})

test_that("a vocabulary too small for an SVD keeps the leading tf-idf columns", {
  docs <- docs_of(scl$text_small_docs)
  fit <- fit_string_encoder(docs, "s")
  expect_false(isTRUE(scl$text_small_has_svd))
  expect_identical(fit$mode, "truncate")
  # Fewer than 30 components, so single-digit names, as skrub pads them.
  expect_identical(fit$outputs, unlist(scl$text_small_names))
  expect_lt(max(abs(transform_string_encoder(docs, fit) - as_mat(ref$text_small_real))), 1e-6)
})

test_that("the tf-idf matrix is scikit-learn's, to the float32 bit", {
  docs <- docs_of(scl$text_flat_docs)
  tf <- fit_tfidf(.text_as_docs(docs))
  # Sorted by code point, not by the session's collation.
  expect_identical(tf$vocabulary, sort(tf$vocabulary, method = "radix"))
  x <- transform_tfidf(.text_as_docs(docs), tf)
  # float32 values carried as doubles: rounding them again changes nothing.
  expect_identical(x@x, .round_float32(x@x))
  # Unit l2 rows, except the missing document's, which is all zero.
  norms <- sqrt(Matrix::rowSums(x^2))
  expect_equal(norms[-4], rep(1, length(norms) - 1L), tolerance = 1e-6)
  expect_identical(unname(norms[4]), 0)
})

test_that("the LU basis is P %*% L, for tall and wide matrices alike", {
  set.seed(3)
  for (dims in list(c(9, 4), c(4, 9), c(5, 5))) {
    A <- matrix(rnorm(prod(dims)), dims[1], dims[2])
    PL <- .lu_permute_l(A)
    expect_identical(dim(PL), as.integer(c(dims[1], min(dims))))
    # Some row of P %*% L is the unit row of each column: L is unit lower.
    expect_true(all(apply(PL, 2, function(col) any(abs(col - 1) < 1e-12))))
    # And `PL %*% U` rebuilds A, with U read back by least squares.
    U <- qr.solve(PL, A)
    expect_lt(max(abs(PL %*% U - A)), 1e-10)
  }
})
