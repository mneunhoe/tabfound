# Test rows go through the network in chunks; what comes back must be one
# result per row whatever the chunking. A regressor's mean is a vector per
# chunk, and `rbind()` on vectors lays each out as a row and recycles the
# short last chunk: 3,000 rows in chunks of 1,024 once came back as 3,072
# numbers.

test_that("vector results are concatenated, not recycled", {
  X <- matrix(seq_len(250 * 2), 250)
  got <- chunk_apply(X, 100L, function(x) x[, 1] * 2)
  expect_identical(got, X[, 1] * 2)
  # A last chunk that does not divide the chunk size, as in the report.
  X <- matrix(as.numeric(seq_len(3000)), 3000)
  expect_identical(chunk_apply(X, 1024L, function(x) x[, 1]), X[, 1])
})

test_that("matrix results are stacked by row", {
  X <- matrix(rnorm(250 * 3), 250)
  got <- chunk_apply(X, 100L, function(x) x[, 2:3, drop = FALSE])
  expect_identical(got, X[, 2:3])
  # One column is still a matrix.
  got <- chunk_apply(X, 100L, function(x) x[, 1, drop = FALSE])
  expect_identical(dim(got), c(250L, 1L))
})

test_that("a chunk that returns the wrong number of rows is an error", {
  X <- matrix(rnorm(250), 250)
  expect_error(chunk_apply(X, 100L, function(x) x[-1, 1]),
               "returned 247 results for 250 rows")
})
