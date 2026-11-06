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


test_that("a Kumo cache survives a bundle and does not move predictions", {
  d <- Sys.getenv("TABFOUND_KUMO_CLF_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_KUMO_CLF_DIR not configured")

  set.seed(6)
  X <- matrix(rnorm(160 * 4), ncol = 4)
  y <- factor(ifelse(X[, 1] - X[, 3] > 0, "b", "a"))
  tr <- 1:120; te <- 121:160

  m <- fit(tabular_classifier(d, kv_cache = TRUE, n_estimators = 2L), X[tr, ], y[tr])
  p_plain <- predict(m, X[te, ], type = "prob")

  cached <- tabfound_cache(m)
  expect_true(has_cache(cached))
  # Built once by `tabfound_cache()` and once lazily by `predict()` with
  # `kv_cache = TRUE`: the same computation, so the same bits.
  expect_identical(predict(cached, X[te, ], type = "prob"), p_plain)

  bundle <- file.path(withr::local_tempdir(), "kumo-bundle")
  tabfound_save(cached, bundle)
  reloaded <- tabfound_load(bundle)
  expect_true(has_cache(reloaded))
  expect_identical(predict(reloaded, X[te, ], type = "prob"), p_plain)

  refit <- fit(reloaded, X[tr[1:60], ], y[tr[1:60]])
  refit$state$kv_caches <- reloaded$state$kv_caches
  refit$state$kv_guard  <- reloaded$state$kv_guard
  expect_error(predict(refit, X[te, ]), "does not match its training rows")
})


test_that("the guard notices a predictor the cache was not built by", {
  state <- list(X_train = matrix(rnorm(20), ncol = 2), n_train = 10L,
                y_train_int = rep(0:1, 5))
  args <- list(ensemble_configs = list(list(shuffle_perm = 1:2)), kv_cache = TRUE)
  state$kv_caches <- list(1)
  state$kv_guard <- .kv_guard_of(state, args)
  expect_silent(.kv_guard_check(state, args))
  # Order is not meaning, and an RDS round trip changes nothing.
  expect_silent(.kv_guard_check(state, rev(args)))
  f <- withr::local_tempfile(fileext = ".rds")
  saveRDS(args, f)
  expect_silent(.kv_guard_check(state, readRDS(f)))

  # Same rows, other members: the rows alone cannot tell.
  other <- args
  other$ensemble_configs[[1]]$shuffle_perm <- 2:1
  expect_error(.kv_guard_check(state, other), "different predictor")

  # A guard from before the arguments were fingerprinted is checked on
  # the rows alone, as it was.
  state$kv_guard$args <- NULL
  expect_silent(.kv_guard_check(state, other))
})


test_that("saving over an earlier save replaces it, whichever form it took", {
  dir <- local_fake_backend()
  X <- matrix(rnorm(40), ncol = 2)
  y <- factor(rep(c("a", "b"), 10))
  m <- fit(tabular_classifier(dir), X, y)
  cached <- m
  cached$state$kv_caches <- list(nested_cache())
  cached$state$kv_guard  <- .kv_guard_of(cached$state, cached$model_ref$args)

  # An uncached model over a bundle: one file, and it is what loads.
  target <- file.path(withr::local_tempdir(), "model")
  tabfound_save(cached, target)
  expect_true(dir.exists(target))
  tabfound_save(m, target)
  expect_true(file.exists(target) && !dir.exists(target))
  expect_false(has_cache(tabfound_load(target)))

  # And back: a cached model over that file becomes a bundle.
  tabfound_save(cached, target)
  expect_true(dir.exists(target))
  expect_true(has_cache(tabfound_load(target)))

  # A directory that is not a bundle is not ours to delete.
  other <- withr::local_tempdir()
  writeLines("keep me", file.path(other, "notes.txt"))
  expect_error(tabfound_save(m, other), "not a .*bundle")
  expect_true(file.exists(file.path(other, "notes.txt")))
})


test_that("a save from before the configs travelled loads, or says why not", {
  dir <- local_fake_backend()
  X <- matrix(rnorm(40), ncol = 2)
  y <- factor(rep(c("a", "b"), 10))
  m <- fit(tabular_classifier(dir), X, y)
  # What an older tabfound wrote: the directory, not the configs. The
  # fake backend ignores the argument, which is all this needs.
  cfg_dir <- withr::local_tempdir()
  m$model_ref$args$ensemble_configs_dir <- cfg_dir
  f <- withr::local_tempfile(fileext = ".tabfound")
  tabfound_save(m, f)

  # Gone, as a `tempfile()` is with its session: a message, not a JSON
  # parser's complaint about the path.
  expect_error(tabfound_load(f), "ensemble configs are gone")

  # Still there: loads, and a cache it cannot check is flagged.
  writeLines("[]", file.path(cfg_dir, "ensemble_configs.json"))
  expect_no_warning(tabfound_load(f))
  m$state$kv_caches <- list(nested_cache())
  m$state$kv_guard  <- .kv_guard_of(m$state)
  bundle <- file.path(withr::local_tempdir(), "bundle")
  tabfound_save(m, bundle)
  expect_warning(back <- tabfound_load(bundle), "older tabfound")
  expect_true(has_cache(back))
})


test_that("a reloaded TabPFN runs the members it was saved with", {
  d <- Sys.getenv("TABFOUND_TABPFN26_CLF_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN26_CLF_DIR not configured")

  set.seed(5)
  X  <- matrix(rnorm(180), 60); y <- factor(ifelse(X[, 1] > 0, "a", "b"))
  Xn <- matrix(rnorm(45), 15)
  root <- withr::local_tempdir()
  cfg_dir <- file.path(root, "cfg")
  gen <- function(seed, out = NULL) {
    generate_ensemble_configs_native(X, y, n_estimators = 4, head = "classifier",
                                     variant = "v2.6", output_dir = out,
                                     random_state = seed)
  }
  gen(0L, cfg_dir)
  m  <- fit(tabular_classifier(d, ensemble_configs_dir = cfg_dir), X, y)
  p0 <- predict(m, Xn, type = "prob")
  expect_null(m$model_ref$args[["ensemble_configs_dir"]])
  bundle <- file.path(root, "bundle")
  plain  <- file.path(root, "plain.tabfound")
  tabfound_save(tabfound_cache(m), bundle)
  tabfound_save(m, plain)

  # The directory is regenerated with another seed. Before, the reload
  # re-read it: an uncached model moved by 1e-3, and a cached one mixed
  # the old members' cache with the new configs and moved by 0.74.
  unlink(cfg_dir, recursive = TRUE)
  gen(1L, cfg_dir)
  expect_identical(predict(tabfound_load(bundle), Xn, type = "prob"), p0)
  expect_identical(predict(tabfound_load(plain), Xn, type = "prob"), p0)
  unlink(cfg_dir, recursive = TRUE)
  expect_identical(predict(tabfound_load(plain), Xn, type = "prob"), p0)

  # A cache moved onto a predictor with other members is refused, though
  # the training rows are the same.
  other <- fit(tabular_classifier(d, ensemble_configs = gen(1L)), X, y)
  cached <- tabfound_cache(m)
  other$state$kv_caches <- cached$state$kv_caches
  other$state$kv_guard  <- cached$state$kv_guard
  expect_error(predict(other, Xn), "different predictor")
})
