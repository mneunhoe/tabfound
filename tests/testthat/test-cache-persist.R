# Persisting a KV cache (C2).
#
# The flattener and the guard are pure R + torch and are tested directly.
# The end-to-end claim -- a cache built in one session reproduces its
# predictions in the next -- needs weights, so it runs against the fake
# backend where it can and against a real checkpoint where one is
# configured.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")

nested_cache <- function() {
  structure(
    list(
      kv = list(list(key = torch::torch_randn(c(1, 3, 4)),
                     value = torch::torch_randn(c(1, 3, 4))),
                list(key = torch::torch_randn(c(1, 3, 4)),
                     value = torch::torch_randn(c(1, 3, 4)))),
      mask = torch::torch_tensor(matrix(c(TRUE, FALSE, TRUE, TRUE), 2)),
      n_train = 17L,
      label = "member-1"
    ),
    class = "probe_kv_cache"
  )
}


test_that("a cache flattens to paths and grafts back to the same object", {
  x <- list(nested_cache(), nested_cache())
  flat <- .kv_flatten(x)

  expect_length(flat$tensors, 10L)      # 2 members x (4 kv + 1 mask)
  expect_true(all(grepl("^kv\\.", names(flat$tensors))))
  expect_identical(names(flat$tensors)[1:2], c("kv.1.kv.1.key", "kv.1.kv.1.value"))
  # The skeleton is plain R data -- which is the point, since that half
  # goes into the RDS.
  expect_false(any(vapply(unlist(flat$skeleton), function(v)
    inherits(v, "torch_tensor"), logical(1))))

  back <- .kv_graft(flat$skeleton, flat$tensors)
  expect_s3_class(back[[1]], "probe_kv_cache")
  expect_identical(names(back[[1]]), names(x[[1]]))
  expect_identical(back[[1]]$n_train, 17L)
  expect_identical(back[[1]]$label, "member-1")
  expect_true(as.logical(torch::torch_equal(back[[1]]$kv[[2]]$value,
                                            x[[1]]$kv[[2]]$value)))
  expect_true(as.logical(torch::torch_equal(back[[1]]$mask, x[[1]]$mask)))
})


test_that("a tensor that is a slice of a bigger one round-trips intact", {
  # A row slice is *contiguous* -- it just starts partway into the
  # storage -- and safetensors writes from the start of the storage. Such
  # a tensor comes back the right shape holding its neighbour's numbers,
  # with no error anywhere. One tensor of a real TabPFN cache is exactly
  # this, so the round trip is silently wrong without the clone.
  big <- torch::torch_randn(c(8, 5))
  slice <- big[4:5, ]
  expect_true(slice$is_contiguous())

  d <- withr::local_tempdir()
  w <- .kv_write(list(list(a = slice)), d)
  back <- .kv_read(w$skeleton, file.path(d, w$file))
  expect_true(as.logical(torch::torch_equal(back[[1]]$a, slice)))
})


test_that("a written cache reads back off disk", {
  d <- withr::local_tempdir()
  x <- list(nested_cache())
  w <- .kv_write(x, d)
  expect_identical(w$file, "cache.safetensors")
  expect_true(file.exists(file.path(d, w$file)))

  back <- .kv_read(w$skeleton, file.path(d, w$file))
  expect_true(as.logical(torch::torch_equal(back[[1]]$kv[[1]]$key,
                                            x[[1]]$kv[[1]]$key)))
  # A bundle whose tensor file went missing says so rather than failing
  # somewhere inside the network.
  expect_error(.kv_read(w$skeleton, file.path(d, "gone.safetensors")),
               "missing")
})


test_that("the guard notices a context the cache did not come from", {
  state <- list(X_train = matrix(rnorm(20), ncol = 2), n_train = 10L,
                y_train_int = rep(0:1, 5))
  state$kv_caches <- list(1)
  state$kv_guard <- .kv_guard_of(state)
  expect_silent(.kv_guard_check(state))

  # Same shape, different numbers: only the digest can catch this.
  moved <- state
  moved$X_train[1, 1] <- moved$X_train[1, 1] + 1
  skip_if_not_installed("digest")
  expect_error(.kv_guard_check(moved), "does not match its training rows")

  # Different shape.
  smaller <- state
  smaller$X_train <- smaller$X_train[1:5, , drop = FALSE]
  smaller$n_train <- 5L
  expect_error(.kv_guard_check(smaller), "does not match")

  # No cache, no guard: nothing to check.
  expect_silent(.kv_guard_check(list(X_train = matrix(1))))
})


test_that("tabfound_cache refuses what it cannot do", {
  m <- stub_model("classification")
  expect_error(tabfound_cache(m), "has not been fitted")
  m <- fit(m, matrix(rnorm(20), ncol = 2), factor(rep(c("a", "b"), 5)))
  # The stub has no `build_cache` hook, which is what a backend without a
  # conditioned state to hand over looks like.
  expect_error(tabfound_cache(m), "cannot precompute")
  expect_false(has_cache(m))
})


test_that("a cache survives save and load, and predictions do not move", {
  dir <- local_fake_backend()
  X <- matrix(rnorm(40), ncol = 2)
  y <- factor(rep(c("a", "b"), 10))
  m <- fit(tabular_classifier(dir), X, y)

  # The fake backend has no cache hook either, so this is the file-format
  # half: a model with a cache attached by hand still round-trips.
  m$state$kv_caches <- list(nested_cache())
  m$state$kv_guard  <- .kv_guard_of(m$state)

  bundle <- file.path(withr::local_tempdir(), "bundle")
  tabfound_save(m, bundle)
  expect_true(dir.exists(bundle))
  expect_setequal(list.files(bundle), c("state.rds", "cache.safetensors"))

  m2 <- tabfound_load(bundle)
  expect_true(has_cache(m2))
  expect_true(as.logical(torch::torch_equal(
    m2$state$kv_caches[[1]]$kv[[1]]$key, m$state$kv_caches[[1]]$kv[[1]]$key)))
  expect_identical(predict(m2, X, type = "prob"), predict(m, X, type = "prob"))

  # And a model without a cache is still one file, as before.
  plain <- file.path(withr::local_tempdir(), "plain.tabfound")
  tabfound_save(tabfound_cache(m, build = FALSE), plain)
  expect_true(file.exists(plain) && !dir.exists(plain))
  expect_false(has_cache(tabfound_load(plain)))
})


test_that("a real checkpoint's cache reproduces its predictions off disk", {
  d <- Sys.getenv("TABFOUND_TABPFN_CLF_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN_CLF_DIR not configured")

  set.seed(5)
  X <- matrix(rnorm(300 * 5), ncol = 5)
  y <- factor(ifelse(X[, 1] + X[, 2] > 0, "b", "a"))
  tr <- 1:200; te <- 201:300

  m <- fit(tabular_classifier(d, kv_cache = TRUE), X[tr, ], y[tr])
  p_plain <- predict(m, X[te, ], type = "prob")

  cached <- tabfound_cache(m)
  expect_true(has_cache(cached))
  # Conditioning on the training rows is the whole cost; a cache that
  # changed the answer would be worse than no cache.
  expect_identical(predict(cached, X[te, ], type = "prob"), p_plain)

  bundle <- file.path(withr::local_tempdir(), "clf-bundle")
  tabfound_save(cached, bundle)
  reloaded <- tabfound_load(bundle)
  expect_true(has_cache(reloaded))
  expect_identical(predict(reloaded, X[te, ], type = "prob"), p_plain)

  # The guard is the reason a persisted cache is safe to keep: refit on
  # different rows and the stale cache is refused, not used.
  refit <- fit(reloaded, X[tr[1:100], ], y[tr[1:100]])
  refit$state$kv_caches <- reloaded$state$kv_caches
  refit$state$kv_guard  <- reloaded$state$kv_guard
  expect_error(predict(refit, X[te, ]), "does not match its training rows")
})
