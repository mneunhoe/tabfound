# CPython's `random.Random`, checked against CPython.
#
# This is the foundation the whole wrapper layer stands on: TabICL's and
# TabFM's ensemble generators decide every member's feature order, class
# relabelling and normalisation method by drawing from this one stream.
# If it drifts, the members drift with it, and the symptom is a diffuse
# disagreement in the predictions with nothing to point at. So it is
# pinned here, on its own, first.

test_that("the MT19937 stream matches CPython", {
  ref <- ensemble_ref()$s$py_random
  seeds <- c(0, 1, 42, 12345, 2^31 - 1)

  for (seed in seeds) {
    key <- function(what) paste0(what, "_", format(seed, scientific = FALSE))

    r <- py_random(seed)
    expect_equal(vapply(1:5, function(i) r$random(), 0),
                 ref_num(ref[[key("random")]]),
                 tolerance = 0, info = key("random"))

    r <- py_random(seed)
    expect_identical(vapply(c(1, 3, 8, 17, 32, 32, 5), r$getrandbits, 0),
                     ref_num(ref[[key("getrandbits")]]), info = key("getrandbits"))

    # The rejection loop matters as much as the values: a rejected draw
    # still consumes the stream, so a shortcut that is equivalent in
    # distribution would desynchronise everything after it.
    r <- py_random(seed)
    expect_identical(vapply(c(2, 3, 7, 10, 64, 1000, 999999), r$randbelow, 0),
                     ref_num(ref[[key("randbelow")]]), info = key("randbelow"))

    r <- py_random(seed)
    expect_identical(as.numeric(r$shuffle(0:11)),
                     ref_num(ref[[key("shuffle")]]), info = key("shuffle"))

    r <- py_random(seed)
    expect_identical(vapply(1:6, function(i) r$choice(0:16), 0),
                     ref_num(ref[[key("choice")]]), info = key("choice"))
  }
})


test_that("both of sample()'s branches match CPython", {
  # `random.sample` switches between a pool-swap algorithm and a
  # rejection-with-a-seen-set algorithm depending on how `k` compares
  # with `n`. They consume the stream differently, so both are exercised:
  # 10-of-10 takes the pool branch, 30-of-5000 the set branch.
  ref <- ensemble_ref()$s$py_random
  for (seed in c(0, 1, 42, 12345, 2^31 - 1)) {
    tag <- format(seed, scientific = FALSE)
    r <- py_random(seed)
    expect_identical(as.numeric(r$sample(0:9, 10)),
                     ref_num(ref[[paste0("sample_pool_", tag)]]))
    r <- py_random(seed)
    expect_identical(as.numeric(r$sample(0:4999, 30)),
                     ref_num(ref[[paste0("sample_set_", tag)]]))
  }
})


test_that("py_random is reproducible and seed-sensitive", {
  a <- py_random(1L); b <- py_random(1L); c <- py_random(2L)
  draw <- function(r) vapply(1:8, function(i) r$random(), 0)
  da <- draw(a)
  expect_identical(da, draw(b))
  expect_false(identical(da, draw(c)))

  # An unseeded generator follows R's RNG, so `set.seed()` still governs
  # reproducibility from the user's point of view.
  withr::with_seed(99, x <- draw(py_random()))
  withr::with_seed(99, y <- draw(py_random()))
  expect_identical(x, y)
})


test_that("getrandbits refuses widths it has not implemented", {
  r <- py_random(0L)
  expect_identical(r$getrandbits(0), 0)
  expect_error(r$getrandbits(33), "not implemented")
})


test_that("sample rejects k larger than the population", {
  expect_error(py_random(0L)$sample(1:3, 4), "larger than population")
})
