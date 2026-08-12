# Bar-distribution head. These are closed-form checks that need no
# weights: with a known distribution over known buckets, the mean and
# quantiles are computable by hand.

skip_if_not_installed("torch")

test_that("posterior mean is the probability-weighted bucket mean", {
  # Interior buckets only: put all mass in the middle so the half-normal
  # tails contribute nothing and the answer is the plain midpoint.
  borders <- torch::torch_tensor(c(0, 1, 2, 3, 4), dtype = torch::torch_float())
  logits  <- torch::torch_tensor(matrix(c(-1e6, 0, 0, -1e6), nrow = 1),
                                 dtype = torch::torch_float())
  m <- as.numeric(bar_logits_to_mean(logits, borders)$cpu())
  expect_equal(m, 2.0, tolerance = 1e-5)   # half at 1.5, half at 2.5
})


test_that("full support replaces the outer bucket means with half-normal tails", {
  borders <- torch::torch_tensor(c(0, 1, 2, 3, 4), dtype = torch::torch_float())
  # All mass in the first bucket.
  logits <- torch::torch_tensor(matrix(c(0, -1e6, -1e6, -1e6), nrow = 1),
                                dtype = torch::torch_float())

  plain <- as.numeric(bar_logits_to_mean(logits, borders, full_support = FALSE)$cpu())
  expect_equal(plain, 0.5, tolerance = 1e-5)             # bucket midpoint

  # borders[1] - E[HalfNormal(width / qnorm(0.75))], width = 1
  expected <- 1 - (1 / stats::qnorm(0.75)) * sqrt(2 / pi)
  full <- as.numeric(bar_logits_to_mean(logits, borders, full_support = TRUE)$cpu())
  expect_equal(full, expected, tolerance = 1e-5)
  expect_lt(full, plain)   # the tail pulls the mean outward, below the bar
})


test_that("quantiles interpolate inside the crossing bucket", {
  borders <- torch::torch_tensor(c(0, 1, 2, 3, 4), dtype = torch::torch_float())
  # Uniform over 4 buckets spanning [0, 4] -> the distribution is
  # Uniform(0, 4), so the q-th quantile is exactly 4q.
  logits <- torch::torch_zeros(c(1, 4))
  q <- as.numeric(bar_logits_to_quantiles(logits, borders,
                                          c(0.1, 0.25, 0.5, 0.9))$cpu())
  expect_equal(q, c(0.4, 1.0, 2.0, 3.6), tolerance = 1e-5)
})


test_that("quantiles are monotone and bracket the median", {
  set.seed(3)
  borders <- torch::torch_tensor(seq(-3, 3, length.out = 51),
                                 dtype = torch::torch_float())
  logits  <- torch::torch_tensor(matrix(rnorm(50 * 6), nrow = 6),
                                 dtype = torch::torch_float())
  qs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  q  <- as.matrix(bar_logits_to_quantiles(logits, borders, qs)$cpu())
  expect_equal(dim(q), c(6L, length(qs)))
  expect_true(all(apply(q, 1L, function(r) all(diff(r) >= -1e-6))))
})


test_that("sampling is reproducible under a seed and spans the support", {
  borders <- torch::torch_tensor(seq(0, 10, length.out = 21),
                                 dtype = torch::torch_float())
  logits  <- torch::torch_zeros(c(3, 20))
  s1 <- as.matrix(bar_logits_to_samples(logits, borders, 50L, seed = 42)$cpu())
  s2 <- as.matrix(bar_logits_to_samples(logits, borders, 50L, seed = 42)$cpu())
  expect_identical(s1, s2)
  expect_equal(dim(s1), c(3L, 50L))
  expect_true(all(s1 >= 0 & s1 <= 10))
})


test_that("border translation preserves total mass and endpoint conventions", {
  frm <- torch::torch_tensor(seq(0, 10, length.out = 11),
                             dtype = torch::torch_float())
  to  <- torch::torch_tensor(seq(0, 10, length.out = 6),
                             dtype = torch::torch_float())
  set.seed(5)
  logits <- torch::torch_tensor(matrix(rnorm(10 * 4), nrow = 4),
                                dtype = torch::torch_float())
  p <- as.matrix(translate_probs_across_borders_r(logits, frm, to)$cpu())
  expect_equal(dim(p), c(4L, 5L))
  expect_true(all(p >= 0))
  expect_equal(rowSums(p), rep(1, 4), tolerance = 1e-5)
})


test_that("border translation runs on an accelerator", {
  # `torch_where` refuses to mix devices, so every tensor this builds --
  # including the positional mask -- has to be created where the
  # probabilities live. Every ensembled TabPFN regression goes through
  # here, so a CPU-only index vector breaks GPU/MPS regression outright.
  dev <- if (torch::cuda_is_available()) "cuda"
         else if (torch::backends_mps_is_available()) "mps"
         else NULL
  skip_if(is.null(dev), "no accelerator available")

  frm <- torch::torch_tensor(seq(0, 10, length.out = 11),
                             dtype = torch::torch_float(), device = dev)
  to  <- torch::torch_tensor(seq(0, 10, length.out = 6),
                             dtype = torch::torch_float(), device = dev)
  set.seed(5)
  logits <- torch::torch_tensor(matrix(rnorm(10 * 4), nrow = 4),
                                dtype = torch::torch_float(), device = dev)
  p_dev <- translate_probs_across_borders_r(logits, frm, to)
  expect_identical(p_dev$device$type, dev)

  p_cpu <- translate_probs_across_borders_r(logits$cpu(), frm$cpu(), to$cpu())
  expect_equal(as.matrix(p_dev$cpu()), as.matrix(p_cpu), tolerance = 1e-5)
})
