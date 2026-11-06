# Summarising a predicted quantile grid, and naming quantile columns.
#
# Nothing here needs model weights.

test_that("quantile column names format each level on its own", {
  qn <- tabfound:::.quantile_colnames

  # A joint format() put both in scientific notation.
  expect_identical(qn(c(1e-4, 0.5)), c("q0.0001", "q0.5"))
  expect_identical(qn(c(0.1, 0.5, 0.9)), c("q0.1", "q0.5", "q0.9"))
  # ...and gave these two the same name at seven significant digits.
  expect_identical(qn(c(0.123456789, 0.1234567891)),
                   c("q0.123456789", "q0.1234567891"))
  # The backends' 999-level grid prints as typed, not as its binary
  # expansion.
  lv <- qn(seq_len(999) / 1000)
  expect_identical(lv[c(1, 3, 999)], c("q0.001", "q0.003", "q0.999"))
  expect_false(anyDuplicated(lv) > 0)
})


test_that("quantile_grid_stat() reads a vector as one row, and checks width", {
  levels <- seq_len(9) / 10
  grid <- rbind(qnorm(levels), qnorm(levels, mean = 1))

  one <- quantile_grid_stat(grid[1, ], levels, "quantiles")
  expect_identical(dim(one), c(1L, 3L))
  expect_equal(one, quantile_grid_stat(grid[1, , drop = FALSE], levels,
                                       "quantiles"))
  expect_length(quantile_grid_stat(grid[1, ], levels, "mean"), 1L)

  expect_error(quantile_grid_stat(grid[, 1:4], levels, "quantiles"),
               "one column per level")
  expect_error(quantile_grid_stat(grid[1, 1:4], levels, "median"),
               "one column per level")
})


test_that("a grid sorted on the device summarises exactly as one sorted in R", {
  # The TabICL regressor sorts its grid with `torch_sort()` and passes
  # `sorted = TRUE`, so every summary must come out as the R row sort's
  # would, to the bit -- on a grid with crossings and ties.
  set.seed(7)
  levels <- seq_len(99) / 100
  grid <- matrix(rnorm(20 * 99), 20, 99)
  grid[, 10] <- grid[, 40]
  qs <- c(1e-4, 0.005, 0.1, 0.5, 0.9, 0.995, 0.9999)
  dev <- torch::torch_sort(torch::torch_tensor(grid, dtype = torch::torch_double()),
                           dim = -1L)[[1]]
  dev <- as.matrix(dev$cpu())
  stat <- tabfound:::quantile_dist_stat
  for (type in c("mean", "median", "quantiles", "raw_quantiles")) {
    expect_identical(stat(dev, levels, type, qs, sorted = TRUE),
                     stat(grid, levels, type, qs), info = type)
  }

  # The regressor's mean skips the sort altogether, which moves only the
  # summation order.
  expect_lt(max(abs(rowMeans(grid) - stat(grid, levels, "mean"))), 1e-14)
})
