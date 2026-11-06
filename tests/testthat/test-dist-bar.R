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


test_that("the batched decode is the one-level-at-a-time decode, bit for bit", {
  # Every level is decoded in one batched pass now; stacking the kept
  # single-level form is what it replaced, and the two have to agree
  # exactly for the regressor's quantiles and draws not to move.
  set.seed(11)
  borders <- torch::torch_tensor(sort(rnorm(201, sd = 3)),
                                 dtype = torch::torch_float())
  logits <- torch::torch_tensor(matrix(rnorm(37 * 200, sd = 4), nrow = 37),
                                dtype = torch::torch_float())
  for (qs in list(0.5, c(0.1, 0.5, 0.9), 1:199 / 200)) {
    looped <- torch::torch_stack(lapply(qs, function(q) {
      bar_logits_to_quantile_single(logits, borders, q)
    }), dim = -1L)
    expect_true(torch::torch_equal(bar_logits_to_quantiles(logits, borders, qs),
                                   looped), label = length(qs))
  }

  torch::torch_manual_seed(9L)
  u <- torch::torch_rand(c(37L, 25L))
  looped <- torch::torch_stack(lapply(seq_len(25L), function(k) {
    bar_logits_to_quantile_single(logits, borders, u$select(dim = -1L, index = k))
  }), dim = -1L)
  expect_true(torch::torch_equal(
    bar_logits_to_samples(logits, borders, 25L, seed = 9L), looped))

  # A single row, without the row axis.
  expect_true(torch::torch_equal(
    bar_logits_to_quantiles(logits[1, ], borders, c(0.2, 0.7)),
    bar_logits_to_quantiles(logits, borders, c(0.2, 0.7))[1, ]))
})


test_that("drawing a chunk at a time draws what one call over every row does", {
  set.seed(12)
  borders <- torch::torch_tensor(seq(-4, 4, length.out = 31),
                                 dtype = torch::torch_float())
  logits <- torch::torch_tensor(matrix(rnorm(23 * 30), nrow = 23),
                                dtype = torch::torch_float())
  whole <- as.matrix(bar_logits_to_samples(logits, borders, 40L, seed = 5L)$cpu())
  for (size in c(1L, 7L, 23L, 100L)) {
    got <- .bar_samples_over_chunks(
      23L, size, function(s, e) logits[s:e, , drop = FALSE], borders,
      n_samples = 40L, seed = 5L)
    expect_identical(got, whole, label = paste("chunk size", size))
  }
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


test_that("border translation matches the reference's, tabpfn 9.1 and later", {
  # Values from `tabpfn.utils.translate_probs_across_borders` itself, on
  # grids built to stress it: a transformed grid, buckets narrow enough for
  # float32 CDF values to round together, a zero-width bucket, and a grid
  # that the destination reaches past on both sides. Regenerate with
  # `inst/parity/remap_reference.py`.
  skip_if_not_installed("safetensors")
  ref_path <- tabfound_file("parity", "transforms", "remap.safetensors.gz")
  skip_if(!nzchar(ref_path), "remap reference not found")
  ref <- read_reference_tensors(ref_path)
  for (g in c("transformed", "narrow", "point_mass", "inside")) {
    p <- translate_probs_across_borders_r(ref[[paste0(g, "_logits")]],
                                          ref[[paste0(g, "_frm")]], ref$to)
    expect_equal(as.matrix(p$cpu()), as.matrix(ref[[paste0(g, "_probs")]]),
                 tolerance = 1e-6, label = g)
    expect_equal(rowSums(as.matrix(p$cpu())), rep(1, 6), tolerance = 1e-5,
                 label = g)
  }
})

test_that("no destination bucket inside the source's support loses its mass", {
  # The old translation took differences of float32 CDF values and clamped
  # them at zero, so a destination bucket whose two CDF values rounded
  # together came out with probability exactly 0 (PriorLabs/TabPFN#1288).
  # Peaked rows put the CDF within float32 rounding of 1 over most of the
  # grid while every source bucket still has mass; on these four rows the
  # old translation zeroes 949 of the 5,240 destination buckets inside the
  # source grid's interior. Every one of them must keep its mass.
  f <- c(seq(-4, 4, length.out = 120), 0.5 + seq(-0.002, 0.002, length.out = 41))
  f <- as.numeric(torch::torch_tensor(sort(unique(f)), dtype = torch::torch_float()))
  frm <- torch::torch_tensor(f, dtype = torch::torch_float())
  to  <- torch::torch_tensor(seq(-6, 6, length.out = 2001), dtype = torch::torch_float())
  mid <- (f[-length(f)] + f[-1]) / 2
  peaks <- list(c(-2, 0.6), c(-1, 0.8), c(1.5, 0.5), c(-2.5, 1))
  logits <- torch::torch_tensor(
    t(vapply(peaks, function(cs) -0.5 * ((mid - cs[1]) / cs[2])^2, numeric(length(mid)))),
    dtype = torch::torch_float())
  expect_true(all(as.matrix(torch::nnf_softmax(logits, dim = -1L)$cpu()) > 0))
  p <- as.matrix(translate_probs_across_borders_r(logits, frm, to)$cpu())
  b <- as.numeric(to)
  inside <- b[-length(b)] >= f[2] & b[-1] <= f[length(f) - 1]
  expect_true(all(p[, inside] > 0))
  expect_equal(rowSums(p), rep(1, 4), tolerance = 1e-5)
})

test_that("a member on the destination grid translates to its softmax", {
  to <- torch::torch_tensor(seq(-3, 3, length.out = 41), dtype = torch::torch_float())
  set.seed(2)
  logits <- torch::torch_tensor(matrix(rnorm(3 * 40), nrow = 3),
                                dtype = torch::torch_float())
  expect_identical(as.matrix(translate_probs_across_borders_r(logits, to, to)$cpu()),
                   as.matrix(torch::nnf_softmax(logits, dim = -1L)$cpu()))
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
