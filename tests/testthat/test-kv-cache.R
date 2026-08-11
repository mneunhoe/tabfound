# KV caches for the Mitra, TabICL and TabFM backends.
#
# One question, asked three times: does conditioning on the training rows
# once and reusing that give the same answer as re-encoding them for every
# batch of test rows? Randomly-initialised networks at a fraction of the
# real width answer it without a checkpoint, because the property being
# tested is structural -- which rows the architecture lets see which --
# and not a property of the weights.
#
# How exact "the same answer" can be differs by backend, and the
# tolerances below are not arbitrary:
#
#   mitra   bit-identical. The query rows are their own tensor either
#           way, so nothing about their arithmetic changes.
#   tabicl  float32 rounding. Dropping the training rows changes the
#           sequence length every matmul sees.
#   tabfm   float32 rounding, plus the regrouping that comes of reducing
#           over `n_train` keys instead of `T` keys of which most are
#           masked to zero.
#
# Agreement with the Python references is a separate question, answered in
# `test-parity-*.R`.

skip_if_not_installed("torch")


# ---------------------------------------------------------------------------
# Mitra
# ---------------------------------------------------------------------------

tiny_mitra <- function(task = "CLASSIFICATION", n_layers = 3L) {
  torch::torch_manual_seed(3L)
  net <- mitra_model(list(dim = 32L, dim_output = if (task == "CLASSIFICATION") 4L else 1L,
                          n_layers = n_layers, n_heads = 4L, task = task))
  net$eval()
  net
}

mitra_inputs <- function(s = 30L, q = 9L, f = 6L, seed = 21L) {
  torch::torch_manual_seed(seed)
  list(
    x_support = torch::torch_randn(c(1L, s, f)),
    y_support = torch::torch_randint(0L, 3L, c(1L, s))$
      to(dtype = torch::torch_float()),
    x_query   = torch::torch_randn(c(1L, q, f))
  )
}


test_that("the Mitra network advertises an exact cache", {
  net <- tiny_mitra()
  expect_true(isTRUE(net$supports_kv_cache))
  expect_true(isTRUE(net$kv_cache_is_exact))
})


test_that("a Mitra cache reproduces the forward it was built from, bit for bit", {
  net <- tiny_mitra()
  d <- mitra_inputs()

  plain <- torch::with_no_grad(net(d$x_support, d$y_support, d$x_query))
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_support, d$y_support))
  expect_s3_class(cache, "mitra_kv_cache")
  expect_length(cache$kv, 3L)
  expect_identical(cache$n_support, 30L)

  cached <- torch::with_no_grad(net(NULL, NULL, d$x_query, kv_cache = cache))
  # Not `tolerance =`: the query rows travel through the same modules in
  # the same shapes either way, so anything but exact equality would mean
  # the cached path is doing different arithmetic, not rounding.
  expect_identical(as.array(cached), as.array(plain))
})


test_that("one Mitra cache serves any split of the query rows", {
  net <- tiny_mitra()
  d <- mitra_inputs(q = 9L)
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_support, d$y_support))

  full <- as.array(torch::with_no_grad(
    net(NULL, NULL, d$x_query, kv_cache = cache)
  ))[1, , ]
  head <- as.array(torch::with_no_grad(
    net(NULL, NULL, d$x_query[, 1:4, ], kv_cache = cache)
  ))[1, , ]
  expect_identical(head, full[1:4, , drop = FALSE])
})


test_that("a Mitra cache is not reusable across a different feature count", {
  net <- tiny_mitra()
  d <- mitra_inputs(f = 6L)
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_support, d$y_support))
  expect_error(
    net(NULL, NULL, torch::torch_randn(c(1L, 3L, 4L)), kv_cache = cache),
    "6 features"
  )
})


test_that("Mitra's quantile embedding is fitted on the support set alone", {
  # This is what makes the cache equivalent rather than merely cheaper:
  # the boundaries, mean and standard deviation query values are mapped
  # against come from the support rows, so wildly different query rows
  # cannot move them.
  d <- mitra_inputs()
  a <- mitra_quantile_fit(d$x_support)
  b <- mitra_quantile_fit(d$x_support)
  expect_identical(as.array(a$state$mu), as.array(b$state$mu))

  wild <- d$x_query * 1000
  tame <- mitra_quantile_apply(d$x_query, a$state)
  # A different query batch is transformed by the same fitted numbers,
  # and the support embedding is untouched by either.
  expect_identical(as.array(mitra_quantile_apply(wild, a$state)$size()),
                   as.array(tame$size()))
  expect_identical(as.array(a$support),
                   as.array(mitra_quantile_embedding(d$x_support, d$x_query)$support))
})


test_that("the Mitra regressor caches too", {
  net <- tiny_mitra("REGRESSION", n_layers = 2L)
  d <- mitra_inputs(s = 20L, q = 5L, f = 4L)
  y <- torch::torch_randn(c(1L, 20L))

  plain <- torch::with_no_grad(net(d$x_support, y, d$x_query))
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_support, y))
  cached <- torch::with_no_grad(net(NULL, NULL, d$x_query, kv_cache = cache))
  expect_identical(as.array(cached), as.array(plain))
})


# ---------------------------------------------------------------------------
# TabICL
# ---------------------------------------------------------------------------

tiny_tabicl_config <- function(head = "classifier") {
  list(
    arch = "tabicl", head = head,
    max_classes = if (head == "classifier") 4L else 0L,
    num_quantiles = if (head == "regressor") 9L else 0L,
    embed_dim = 16L, col_num_blocks = 2L, col_nhead = 2L, col_num_inds = 5L,
    col_affine = FALSE, col_feature_group = "same", col_feature_group_size = 3L,
    col_target_aware = TRUE, col_ssmax = "qassmax-mlp-elementwise",
    row_num_blocks = 2L, row_nhead = 2L, row_num_cls = 2L,
    row_rope_base = 100000, row_rope_interleaved = FALSE,
    icl_num_blocks = 3L, icl_nhead = 2L, icl_ssmax = "qassmax-mlp-elementwise",
    ff_factor = 2L, activation = "gelu", bias_free_ln = FALSE
  )
}

tiny_tabicl <- function(head = "classifier") {
  torch::torch_manual_seed(11L)
  net <- tabicl_model(tiny_tabicl_config(head))
  net$eval()
  net
}


test_that("the TabICL network advertises a cache", {
  net <- tiny_tabicl()
  expect_true(isTRUE(net$supports_kv_cache))
  expect_true(isTRUE(net$kv_cache_is_exact))
})


test_that("a TabICL cache reproduces the forward it was built from", {
  net <- tiny_tabicl()
  n_tr <- 20L; n_te <- 7L; H <- 5L
  torch::torch_manual_seed(2L)
  x_tr <- torch::torch_randn(c(1L, n_tr, H))
  x_te <- torch::torch_randn(c(1L, n_te, H))
  y <- torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())

  plain <- torch::with_no_grad(
    net(torch::torch_cat(list(x_tr, x_te), dim = 2L), y)
  )
  plain_te <- as.array(plain[1, (n_tr + 1L):(n_tr + n_te), ])

  cache <- torch::with_no_grad(net$build_kv_cache(x_tr, y))
  expect_s3_class(cache, "tabicl_kv_cache")
  expect_length(cache$col_hidden, 2L)   # col_num_blocks
  expect_length(cache$icl_kv, 3L)       # icl_num_blocks
  expect_identical(cache$n_train, 20L)

  cached <- torch::with_no_grad(net(x_te, NULL, kv_cache = cache))
  expect_equal(as.array(cached[1, , ]), plain_te, tolerance = 1e-5)
})


test_that("the TabICL cache holds only what the labelled rows produced", {
  # The column stage's summary is the inducing bottleneck over the
  # labelled rows: fixed size however many rows there were, and identical
  # whichever test rows are later predicted against it.
  net <- tiny_tabicl()
  torch::torch_manual_seed(3L)
  x_tr <- torch::torch_randn(c(1L, 24L, 5L))
  y <- torch::torch_randint(0L, 3L, c(1L, 24L))$to(dtype = torch::torch_float())

  a <- torch::with_no_grad(net$build_kv_cache(x_tr, y))
  b <- torch::with_no_grad(net$build_kv_cache(x_tr, y))
  expect_identical(as.array(a$col_hidden[[1]]), as.array(b$col_hidden[[1]]))
  # num_inds = 5, not 24: the summary does not grow with the training set.
  expect_identical(as.integer(a$col_hidden[[1]]$size(3)), 5L)
  # The ICL cache does, one key per labelled row.
  expect_identical(as.integer(a$icl_kv[[1]]$key$size(3)), 24L)
})


test_that("one TabICL cache serves any split of the test rows", {
  net <- tiny_tabicl()
  torch::torch_manual_seed(5L)
  x_tr <- torch::torch_randn(c(1L, 18L, 5L))
  x_te <- torch::torch_randn(c(1L, 8L, 5L))
  y <- torch::torch_randint(0L, 3L, c(1L, 18L))$to(dtype = torch::torch_float())
  cache <- torch::with_no_grad(net$build_kv_cache(x_tr, y))

  full <- as.array(torch::with_no_grad(net(x_te, NULL, kv_cache = cache))[1, , ])
  head <- as.array(
    torch::with_no_grad(net(x_te[, 1:3, ], NULL, kv_cache = cache))[1, , ]
  )
  expect_equal(head, full[1:3, , drop = FALSE], tolerance = 1e-5)
})


test_that("a TabICL cache is not reusable across a different feature count", {
  net <- tiny_tabicl()
  torch::torch_manual_seed(6L)
  x_tr <- torch::torch_randn(c(1L, 12L, 5L))
  y <- torch::torch_randint(0L, 3L, c(1L, 12L))$to(dtype = torch::torch_float())
  cache <- torch::with_no_grad(net$build_kv_cache(x_tr, y))
  expect_error(
    net(torch::torch_randn(c(1L, 4L, 9L)), NULL, kv_cache = cache),
    "5 features"
  )
})


test_that("TabICL refuses to cache a column the set transformer skips", {
  # A uniformly-sentinel column is passed through rather than attended
  # over, and no summary can stand in for "this column was skipped". The
  # released target-aware checkpoints never produce one, so this is a
  # guard rather than a path anything normally takes.
  st <- tabicl_set_transformer(num_blocks = 1L, embedding_dim = 8L, n_heads = 2L,
                               dim_ff = 16L, num_inds = 3L)
  src <- torch::torch_randn(c(1L, 3L, 6L, 8L))
  src[, 2, , ] <- -100.0
  expect_error(torch::with_no_grad(st$build_hidden(src)), "uniformly")
})


test_that("the TabICL regressor caches too", {
  net <- tiny_tabicl("regressor")
  torch::torch_manual_seed(8L)
  x_tr <- torch::torch_randn(c(1L, 16L, 4L))
  x_te <- torch::torch_randn(c(1L, 5L, 4L))
  y <- torch::torch_randn(c(1L, 16L))

  plain <- torch::with_no_grad(
    net(torch::torch_cat(list(x_tr, x_te), dim = 2L), y)
  )
  cache <- torch::with_no_grad(net$build_kv_cache(x_tr, y))
  cached <- torch::with_no_grad(net(x_te, NULL, kv_cache = cache))
  expect_equal(as.array(cached[1, , ]),
               as.array(plain[1, 17:21, ]), tolerance = 1e-5)
})


# ---------------------------------------------------------------------------
# TabFM
# ---------------------------------------------------------------------------

tiny_tabfm <- function(is_classifier = TRUE) {
  torch::torch_manual_seed(7L)
  net <- tabfm_model(list(
    embed_dim = 16L, ff_factor = 2L, row_num_cls = 2L,
    is_classifier = is_classifier, max_classes = 4L,
    feature_group_size = 3L, num_freq = 8L,
    col_num_blocks = 2L, col_nhead = 2L, col_num_inds = 4L,
    row_num_blocks = 2L, row_nhead = 2L,
    icl_num_blocks = 2L, icl_nhead = 2L, decoder_hidden = NULL
  ))
  net$eval()
  net
}

tabfm_inputs <- function(n_tr = 18L, n_te = 6L, H = 5L, seed = 13L,
                         classifier = TRUE) {
  torch::torch_manual_seed(seed)
  y_tr <- if (classifier) {
    torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())
  } else {
    torch::torch_randn(c(1L, n_tr))
  }
  list(
    x_tr = torch::torch_randn(c(1L, n_tr, H)),
    x_te = torch::torch_randn(c(1L, n_te, H)),
    y_tr = y_tr,
    # Unlabelled rows carry the -100 sentinel, as `.tabfm_batch()` builds
    # them.
    y_all = torch::torch_cat(
      list(y_tr, torch::torch_full(c(1L, n_te), -100.0)), dim = 2L
    ),
    train_size = torch::torch_tensor(n_tr, dtype = torch::torch_long()),
    n_tr = n_tr, n_te = n_te
  )
}


test_that("the TabFM network advertises a cache", {
  net <- tiny_tabfm()
  expect_true(isTRUE(net$supports_kv_cache))
  expect_true(isTRUE(net$kv_cache_is_exact))
})


test_that("a TabFM cache reproduces the forward it was built from", {
  net <- tiny_tabfm()
  d <- tabfm_inputs()

  plain <- torch::with_no_grad(net(
    torch::torch_cat(list(d$x_tr, d$x_te), dim = 2L), d$y_all, d$train_size
  ))
  plain_te <- as.array(plain[1, (d$n_tr + 1L):(d$n_tr + d$n_te), ])

  cache <- torch::with_no_grad(net$build_kv_cache(d$x_tr, d$y_tr, d$train_size))
  expect_s3_class(cache, "tabfm_kv_cache")
  # One entry per column stage, each with one summary per block.
  expect_length(cache$col_hidden, 2L)
  expect_length(cache$col_hidden[[1]], 2L)
  expect_length(cache$icl_kv, 2L)
  expect_identical(cache$n_train, 18L)
  expect_false(cache$has_cat_mask)

  cached <- torch::with_no_grad(net(d$x_te, NULL, NULL, kv_cache = cache))
  expect_equal(as.array(cached[1, , ]), plain_te, tolerance = 1e-5)
})


test_that("TabFM's column summaries do not grow with the training set", {
  net <- tiny_tabfm()
  small <- torch::with_no_grad(net$build_kv_cache(
    tabfm_inputs(n_tr = 10L)$x_tr,
    tabfm_inputs(n_tr = 10L)$y_tr,
    torch::torch_tensor(10L, dtype = torch::torch_long())
  ))
  big <- torch::with_no_grad(net$build_kv_cache(
    tabfm_inputs(n_tr = 40L)$x_tr,
    tabfm_inputs(n_tr = 40L)$y_tr,
    torch::torch_tensor(40L, dtype = torch::torch_long())
  ))
  # num_inds = 4 either way -- that is the point of the set transformer.
  expect_identical(as.integer(small$col_hidden[[1]][[1]]$size(2)), 4L)
  expect_identical(as.integer(big$col_hidden[[1]][[1]]$size(2)), 4L)
  # The ICL half is the part that scales.
  expect_identical(as.integer(small$icl_kv[[1]]$key$size(3)), 10L)
  expect_identical(as.integer(big$icl_kv[[1]]$key$size(3)), 40L)
})


test_that("one TabFM cache serves any split of the test rows", {
  net <- tiny_tabfm()
  d <- tabfm_inputs(n_te = 8L)
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_tr, d$y_tr, d$train_size))

  full <- as.array(
    torch::with_no_grad(net(d$x_te, NULL, NULL, kv_cache = cache))[1, , ]
  )
  head <- as.array(
    torch::with_no_grad(net(d$x_te[, 1:3, ], NULL, NULL, kv_cache = cache))[1, , ]
  )
  expect_equal(head, full[1:3, , drop = FALSE], tolerance = 1e-5)
})


test_that("a TabFM cache refuses a different width or a different cat_mask", {
  net <- tiny_tabfm()
  d <- tabfm_inputs(H = 5L)
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_tr, d$y_tr, d$train_size))

  expect_error(
    net(torch::torch_randn(c(1L, 3L, 8L)), NULL, NULL, kv_cache = cache),
    "5 features"
  )
  # The mask routes cells through a different Fourier basis, so a cache
  # built without one describes a different embedding of the same rows.
  cm <- torch::torch_tensor(matrix(c(TRUE, rep(FALSE, 4L)), nrow = 1L),
                            dtype = torch::torch_bool())
  expect_error(
    net(d$x_te, NULL, NULL, cm, kv_cache = cache),
    "categorical mask"
  )
})


test_that("the TabFM regressor caches too", {
  net <- tiny_tabfm(is_classifier = FALSE)
  d <- tabfm_inputs(n_tr = 14L, n_te = 4L, classifier = FALSE)

  plain <- torch::with_no_grad(net(
    torch::torch_cat(list(d$x_tr, d$x_te), dim = 2L), d$y_all, d$train_size
  ))
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_tr, d$y_tr, d$train_size))
  cached <- torch::with_no_grad(net(d$x_te, NULL, NULL, kv_cache = cache))
  expect_equal(as.array(cached[1, , ]),
               as.array(plain[1, 15:18, ]), tolerance = 1e-5)
})


# ---------------------------------------------------------------------------
# The per-member store the predictors use
# ---------------------------------------------------------------------------

# A loaded-model context is two fields as far as these predictors are
# concerned, so a random network stands in for a checkpoint.
fake_ctx <- function(net, config = NULL) {
  list(net = net, device = "cpu", config = config)
}

# Deliberately smaller than the row count, so `predict()` runs several
# chunks against one cache -- which is the thing worth testing here.
CHUNK <- 7L

cache_test_matrix <- function(n = 30L, p = 4L, seed = 41L) {
  set.seed(seed)
  matrix(stats::rnorm(n * p), ncol = p)
}


test_that("the Mitra predictor gives the same answer with the cache on", {
  net <- tiny_mitra()
  X <- cache_test_matrix(); y <- factor(rep(c("a", "b", "c"), length.out = 30L))
  X_new <- cache_test_matrix(n = 22L, seed = 42L)

  args <- list(fake_ctx(net), n_estimators = 2L, predict_chunk_size = CHUNK)
  plain <- do.call(mitra_classifier, args)
  cached <- do.call(mitra_classifier, c(args, list(kv_cache = TRUE)))

  st <- plain$fit(X, y)
  a <- plain$predict(st, X_new, type = "prob")
  b <- cached$predict(cached$fit(X, y), X_new, type = "prob")
  expect_identical(a, b)
})


test_that("the TabICL predictor gives the same answer with the cache on", {
  net <- tiny_tabicl()
  X <- cache_test_matrix(); y <- factor(rep(c("a", "b", "c"), length.out = 30L))
  X_new <- cache_test_matrix(n = 22L, seed = 43L)

  args <- list(fake_ctx(net), n_estimators = 2L, predict_chunk_size = CHUNK)
  plain <- do.call(tabicl_classifier, args)
  cached <- do.call(tabicl_classifier, c(args, list(kv_cache = TRUE)))

  a <- plain$predict(plain$fit(X, y), X_new, type = "prob")
  b <- cached$predict(cached$fit(X, y), X_new, type = "prob")
  expect_equal(a, b, tolerance = 1e-5)
})


test_that("the TabFM predictor gives the same answer with the cache on", {
  net <- tiny_tabfm()
  X <- cache_test_matrix(); y <- factor(rep(c("a", "b", "c"), length.out = 30L))
  X_new <- cache_test_matrix(n = 22L, seed = 44L)

  args <- list(fake_ctx(net), n_estimators = 2L, predict_chunk_size = CHUNK)
  plain <- do.call(tabfm_classifier, args)
  cached <- do.call(tabfm_classifier, c(args, list(kv_cache = TRUE)))

  a <- plain$predict(plain$fit(X, y), X_new, type = "prob")
  b <- cached$predict(cached$fit(X, y), X_new, type = "prob")
  expect_equal(a, b, tolerance = 1e-5)
})


test_that("a cached regressor prediction matches an uncached one", {
  X <- cache_test_matrix(); y <- as.numeric(X[, 1] * 2 + stats::rnorm(30L, sd = 0.1))
  X_new <- cache_test_matrix(n = 15L, seed = 45L)

  mit <- function(...) do.call(mitra_regressor, list(
    fake_ctx(tiny_mitra("REGRESSION", n_layers = 2L)),
    n_estimators = 2L, predict_chunk_size = CHUNK, ...
  ))
  expect_identical(
    mit()$predict(mit()$fit(X, y), X_new),
    mit(kv_cache = TRUE)$predict(mit(kv_cache = TRUE)$fit(X, y), X_new)
  )

  icl <- function(...) do.call(tabicl_regressor, list(
    fake_ctx(tiny_tabicl("regressor"),
             config = list(num_quantiles = 9L)),
    n_estimators = 2L, predict_chunk_size = CHUNK, ...
  ))
  expect_equal(
    icl()$predict(icl()$fit(X, y), X_new),
    icl(kv_cache = TRUE)$predict(icl(kv_cache = TRUE)$fit(X, y), X_new),
    tolerance = 1e-5
  )
})


test_that("the member cache store builds each member once and reuses it", {
  store <- member_cache_store()
  built <- 0L
  make <- function(tag) function() { built <<- built + 1L; tag }

  expect_identical(member_cache(store, 1L, make("a")), "a")
  expect_identical(member_cache(store, 2L, make("b")), "b")
  expect_identical(member_cache(store, 1L, make("ignored")), "a")
  expect_identical(built, 2L)

  # A NULL store is how the predictors express "caching is off": nothing
  # is built and nothing is kept.
  expect_null(member_cache(NULL, 1L, make("c")))
  expect_identical(built, 2L)
})
