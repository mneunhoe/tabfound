# NumPy's legacy `RandomState`, against values NumPy itself produced.
#
# Everything the text preprocessor does downstream -- the randomized SVD,
# and so which features a text column becomes -- rests on this stream
# being NumPy's to the draw. The values below were generated with
# `numpy.random.RandomState` (numpy 2.x, legacy generator) and are pinned
# here so the check needs no Python.

test_that("the uniform stream is NumPy's, bit for bit", {
  # `RandomState(0).random_sample(4)`.
  expect_identical(
    signif(np_random_state(0)$random_sample(4), 12),
    signif(c(0.548813503927325, 0.715189366372419, 0.602763376071644,
             0.544883182996897), 12)
  )
  # The end of the seed range, which `init_genrand` masks to 32 bits.
  expect_length(np_random_state(2^32 - 1)$random_sample(3), 3L)
})

test_that("Gaussians follow NumPy's polar method, cache included", {
  g <- np_random_state(0)
  # An odd count leaves the second deviate of the last pair cached ...
  expect_equal(g$standard_normal(3),
               c(1.764052345967664, 0.4001572083672233, 0.9787379841057392),
               tolerance = 1e-14)
  # ... uniforms in between draw from the stream but leave it there ...
  expect_equal(g$random_sample(2),
               c(0.4236547993389047, 0.6458941130666561),
               tolerance = 1e-14)
  # ... so the next Gaussian is the cached one, not a fresh draw. Only
  # then does the stream resume, from after the uniforms.
  expect_equal(g$standard_normal(4),
               c(2.240893199201458, 0.9500884175255894,
                 -0.1513572082976979, -0.10321885179355784),
               tolerance = 1e-14)

  expect_equal(np_random_state(42)$standard_normal(4),
               c(0.4967141530112327, -0.13826430117118466,
                 0.6476885381006925, 1.5230298564080254),
               tolerance = 1e-14)
})

test_that("a large draw stays in step across many MT19937 blocks", {
  # The randomized SVD's range finder: `normal(size = (2306, 40))`, which
  # is 92,240 Gaussians over roughly 190 blocks and 40,000 rejections.
  # One rejection decided differently would shift every value after it.
  big <- np_random_state(0)$normal(2306 * 40)
  expect_equal(sum(big), -9.657505869612521, tolerance = 1e-10)
  expect_equal(utils::tail(big, 3),
               c(-0.503025569989645, -1.2392387576769957, -0.1868620265140323),
               tolerance = 1e-13)
})

test_that("seeding is NumPy's, not CPython's", {
  # The same MT19937 core seeded two ways: `init_genrand` here,
  # `init_by_array` in `py_random()`. They must disagree, or one of them
  # is not what it claims to be.
  expect_false(isTRUE(all.equal(np_random_state(0)$random_sample(3),
                                c(py_random(0)$random(), py_random(0)$random(),
                                  py_random(0)$random()))))
  expect_error(np_random_state(-1), "whole number")
  expect_error(np_random_state(2^32), "whole number")
  expect_error(np_random_state(1.5), "whole number")
})
