# KV caches for the Mitra, TabICL, TabFM and Kumo backends.
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
#   kumo    float32 rounding, as for TabICL: the query rows go through
#           without the context rows beside them.
#
# Agreement with the Python references is a separate question, answered in
# `test-parity-*.R`.

skip_if_not_installed("torch")


# ---------------------------------------------------------------------------
# Mitra
# ---------------------------------------------------------------------------


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


test_that("Mitra's last layer skips the support stream without changing a bit", {
  # The support and query rows are separate tensors, and only the query
  # rows are decoded, so the last layer's support stream is dead work.
  # Reference: every layer in full, as the forward used to run.
  net <- tiny_mitra()
  d <- mitra_inputs()
  full <- torch::with_no_grad({
    qe <- mitra_quantile_embedding(d$x_support, d$x_query)
    ye <- net$y_embedding(d$y_support, d$x_query$size(2))
    s <- torch::torch_cat(list(ye$support, net$x_embedding(qe$support)), dim = 3L)
    q <- torch::torch_cat(list(ye$query, net$x_embedding(qe$query)), dim = 3L)
    for (i in seq_along(net$layers)) {
      out <- net$layers[[i]](s, q)
      s <- out$support; q <- out$query
    }
    net$decode(q)
  })
  plain <- torch::with_no_grad(net(d$x_support, d$y_support, d$x_query))
  expect_identical(as.array(plain), as.array(full))
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
  # The uncached pass returns the test rows alone, as the cached one does.
  expect_identical(as.integer(plain$size(2)), n_te)
  plain_te <- as.array(plain[1, , ])

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
               as.array(plain[1, , ]), tolerance = 1e-5)
})


# ---------------------------------------------------------------------------
# TabFM
# ---------------------------------------------------------------------------


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
# Kumo Tabular
# ---------------------------------------------------------------------------



kumo_inputs <- function(n_tr = 20L, n_te = 9L, p = 5L, seed = 4L, clf = TRUE) {
  torch::torch_manual_seed(seed)
  x <- torch::torch_randn(c(1L, n_tr + n_te, p))
  x[1, 3, 2] <- NaN
  list(x = x, x_tr = x[, 1:n_tr, ], x_te = x[, (n_tr + 1L):(n_tr + n_te), ],
       y = if (clf) torch::torch_randint(0L, 3L, c(1L, n_tr), dtype = torch::torch_long())
           else torch::torch_randn(c(1L, n_tr)),
       mask = c(FALSE, TRUE, rep(FALSE, p - 2L)))
}


test_that("the Kumo network advertises a cache", {
  net <- tiny_kumo()
  expect_true(isTRUE(net$supports_kv_cache))
  expect_true(isTRUE(net$kv_cache_is_exact))
  expect_true(isTRUE(get_backend("kumo")$kv_cache_capable))
})


test_that("a Kumo cache reproduces the forward it was built from", {
  for (kv in list(NULL, 2L)) {
    net <- tiny_kumo(kv = kv)
    d <- kumo_inputs()
    plain <- torch::with_no_grad(net(d$x, d$y, d$mask))
    cache <- torch::with_no_grad(net$build_kv_cache(d$x_tr, d$y, d$mask))
    expect_s3_class(cache, "kumo_kv_cache")
    expect_length(cache$col_kv, 2L)
    expect_length(cache$icl_kv, 2L)
    expect_identical(cache$n_train, 20L)
    cached <- torch::with_no_grad(net(d$x_te, NULL, d$mask, kv_cache = cache))
    expect_equal(as.array(cached), as.array(plain), tolerance = 1e-5)
  }
})


test_that("the Kumo cache holds only what the context rows produced", {
  net <- tiny_kumo()
  d <- kumo_inputs(n_tr = 24L)
  a <- torch::with_no_grad(net$build_kv_cache(d$x_tr, d$y, d$mask))
  # The column summaries are M inducing vectors per column, whatever the
  # context size; the ICL keys are one per context row, on the 2 heads
  # the query rows use.
  expect_identical(as.integer(a$col_kv[[1]]$key$size(3)), 3L)
  expect_identical(as.integer(a$icl_kv[[1]]$key$size(2)), 24L)
  expect_identical(as.integer(a$icl_kv[[1]]$key$size(3)), 2L)
  # Changing a query row cannot touch it.
  x2 <- d$x$clone(); x2[1, 30, ] <- 3
  b <- torch::with_no_grad(net$build_kv_cache(x2[, 1:24, ], d$y, d$mask))
  expect_true(as.logical(torch::torch_equal(a$icl_kv[[2]]$key, b$icl_kv[[2]]$key)))
})


test_that("one Kumo cache serves any split of the query rows", {
  net <- tiny_kumo()
  d <- kumo_inputs()
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_tr, d$y, d$mask))
  full <- as.array(torch::with_no_grad(net(d$x_te, NULL, d$mask, kv_cache = cache)))
  head <- as.array(torch::with_no_grad(net(d$x_te[, 1:3, ], NULL, d$mask,
                                           kv_cache = cache)))
  expect_equal(head, full[, 1:3, , drop = FALSE], tolerance = 1e-5)
})


test_that("a Kumo cache refuses a different width", {
  net <- tiny_kumo()
  d <- kumo_inputs()
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_tr, d$y, d$mask))
  expect_error(net(torch::torch_randn(c(1L, 4L, 7L)), NULL, rep(FALSE, 7L),
                   kv_cache = cache), "5 features")
})


test_that("the Kumo regressor caches too", {
  net <- tiny_kumo("regressor", kv = NULL)
  d <- kumo_inputs(clf = FALSE)
  plain <- torch::with_no_grad(net(d$x, d$y, d$mask))
  cache <- torch::with_no_grad(net$build_kv_cache(d$x_tr, d$y, d$mask))
  cached <- torch::with_no_grad(net(d$x_te, NULL, d$mask, kv_cache = cache))
  expect_equal(as.array(cached), as.array(plain), tolerance = 1e-5)
})


# ---------------------------------------------------------------------------
# The per-member store the predictors use
# ---------------------------------------------------------------------------


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


test_that("a row-chunked cache build gives the uncached answer", {
  # `row_chunk_size` now reaches the cache build as well as the forward,
  # with a chunk that splits the 30 training rows. A chunk that dropped
  # the target embedding of the labelled rows it carries would still run.
  X <- cache_test_matrix(); y <- factor(rep(c("a", "b", "c"), length.out = 30L))
  X_new <- cache_test_matrix(n = 22L, seed = 47L)
  for (make in list(
    function(...) tabicl_classifier(fake_ctx(tiny_tabicl()), n_estimators = 2L, ...),
    function(...) tabfm_classifier(fake_ctx(tiny_tabfm()), n_estimators = 2L, ...)
  )) {
    plain <- make()
    cached <- make(kv_cache = TRUE, row_chunk_size = 8L)
    a <- plain$predict(plain$fit(X, y), X_new, type = "prob")
    b <- cached$predict(cached$fit(X, y), X_new, type = "prob")
    expect_equal(a, b, tolerance = 1e-5)
  }
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


test_that("the Kumo predictors give the same answer with the cache on", {
  X <- cache_test_matrix(); y <- factor(rep(c("a", "b", "c"), length.out = 30L))
  X_new <- cache_test_matrix(n = 22L, seed = 46L)
  clf <- function(...) do.call(kumo_classifier, list(
    fake_ctx(tiny_kumo(), config = list(num_classes = 10L)),
    n_estimators = 2L, predict_chunk_size = CHUNK, ...))
  # The baseline re-encodes the context per chunk: without
  # `chunk_cache = FALSE` this 4-chunk call would build a cache of its own.
  plain <- clf(chunk_cache = FALSE)
  a <- plain$predict(plain$fit(X, y), X_new, type = "prob")
  b <- clf(kv_cache = TRUE)$predict(clf(kv_cache = TRUE)$fit(X, y), X_new,
                                    type = "prob")
  expect_equal(a, b, tolerance = 1e-5)

  yr <- as.numeric(X[, 1] * 2 + stats::rnorm(30L, sd = 0.1))
  reg <- function(...) do.call(kumo_regressor, list(
    fake_ctx(tiny_kumo("regressor"), config = list(num_quantiles = 9L)),
    n_estimators = 2L, predict_chunk_size = CHUNK, ...))
  plain <- reg(chunk_cache = FALSE)
  expect_equal(plain$predict(plain$fit(X, yr), X_new),
               reg(kv_cache = TRUE)$predict(reg(kv_cache = TRUE)$fit(X, yr), X_new),
               tolerance = 1e-5)
})


test_that("a Kumo cache built by tabfound_cache() survives save and load", {
  # The whole persistence path, with a random network standing in for a
  # checkpoint: build, predict, write the bundle, read it back.
  net <- tiny_kumo()
  spec <- kumo_classifier(fake_ctx(net, config = list(num_classes = 10L)),
                          n_estimators = 2L, predict_chunk_size = CHUNK)
  X <- cache_test_matrix(); y <- factor(rep(c("a", "b", "c"), length.out = 30L))
  X_new <- cache_test_matrix(n = 12L, seed = 47L)
  state <- spec$fit(X, y)
  caches <- spec$build_cache(state)
  expect_length(caches, 2L)
  expect_s3_class(caches[[1]], "kumo_kv_cache")
  plain <- spec$predict(state, X_new, type = "prob")
  state$kv_caches <- caches
  expect_equal(spec$predict(state, X_new, type = "prob"), plain, tolerance = 1e-5)

  dir <- withr::local_tempdir()
  w <- .kv_write(caches, dir)
  back <- .kv_read(w$skeleton, file.path(dir, w$file))
  expect_s3_class(back[[1]], "kumo_kv_cache")
  state$kv_caches <- back
  expect_equal(spec$predict(state, X_new, type = "prob"), plain, tolerance = 1e-5)
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


test_that("ECOC scores are mean log-probabilities over a class's own tasks", {
  torch::torch_manual_seed(9L)
  logits <- torch::torch_randn(3L, 4L, 10L)
  book <- rbind(c(0L, 1L, 9L, 9L, 2L, 9L, 3L, 4L, 5L, 6L, 7L, 8L),
                c(9L, 0L, 1L, 2L, 9L, 3L, 4L, 5L, 6L, 7L, 8L, 9L),
                c(8L, 9L, 0L, 1L, 2L, 3L, 9L, 4L, 5L, 6L, 7L, 9L))
  got <- as.matrix(.kumo_ecoc_scores(logits, book))
  lp <- as.array(torch::nnf_log_softmax(logits, dim = -1L))
  want <- sapply(seq_len(ncol(book)), function(k) {
    own <- which(book[, k] != 9L)
    rowMeans(sapply(own, function(t) lp[t, , book[t, k] + 1L]), dims = 1L)
  })
  if (!is.matrix(want)) want <- matrix(want, nrow = 4L)
  expect_equal(got, want, tolerance = 1e-6)
})

test_that("a Kumo classifier past 10 classes runs ECOC, cached and chunked", {
  X <- cache_test_matrix(n = 60L)
  lv <- letters[1:14]
  y <- factor(rep(lv[-5], length.out = 60L), levels = lv)   # "e" never seen
  X_new <- cache_test_matrix(n = 9L, seed = 48L)
  clf <- function(...) do.call(kumo_classifier, list(
    fake_ctx(tiny_kumo(), config = list(num_classes = 10L)),
    n_estimators = 2L, predict_chunk_size = CHUNK, ...))
  st <- clf()$fit(X, y)
  expect_length(st$recipe$target$codebooks, 2L)
  a <- clf(chunk_cache = FALSE)$predict(st, X_new, type = "prob")
  expect_identical(dim(a), c(9L, 14L))
  expect_equal(unname(rowSums(a)), rep(1, 9L), tolerance = 1e-6)
  expect_true(all(a[, "e"] == 0))
  b <- clf(kv_cache = TRUE)$predict(clf(kv_cache = TRUE)$fit(X, y), X_new,
                                    type = "prob")
  expect_equal(a, b, tolerance = 1e-5)
  ch <- clf(row_chunk_size = 16L, col_chunk_size = 2L, save_peak_memory_factor = 2L)
  expect_equal(ch$predict(ch$fit(X, y), X_new, type = "prob"), a, tolerance = 1e-5)
})


# ---------------------------------------------------------------------------
# Kumo: a cache built for one predict() call (chunk_cache)
# ---------------------------------------------------------------------------

# Counts the member caches built, and how many were alive in the store
# when each one was.
kumo_count_builds <- function(env = parent.frame()) {
  seen <- list(n = 0L, alive = integer())
  orig <- member_cache
  local_mocked_bindings(member_cache = function(store, i, build) {
    orig(store, i, function() {
      seen$n <<- seen$n + 1L
      seen$alive <<- c(seen$alive, length(ls(store)))
      build()
    })
  }, .env = env)
  function() seen
}

kumo_spec <- function(head = "classifier", ...) {
  ctor <- if (head == "classifier") kumo_classifier else kumo_regressor
  cfg <- if (head == "classifier") list(num_classes = 10L) else list(num_quantiles = 9L)
  do.call(ctor, list(fake_ctx(tiny_kumo(head), config = cfg),
                     n_estimators = 3L, predict_chunk_size = CHUNK, ...))
}

test_that("Kumo's chunk_cache agrees with the uncached pass, cache by cache", {
  X <- cache_test_matrix(n = 60L)
  X_new <- cache_test_matrix(n = 20L, seed = 50L)        # 3 chunks of 7
  cases <- list(
    regressor = list(head = "regressor", y = as.numeric(X[, 1] * 2 + X[, 2]),
                     type = "mean"),
    binary = list(head = "classifier",
                  y = factor(rep(c("a", "b"), length.out = 60L)), type = "prob"),
    ecoc = list(head = "classifier",
                y = factor(rep(letters[1:14], length.out = 60L)), type = "prob")
  )
  for (nm in names(cases)) {
    cs <- cases[[nm]]
    builds <- kumo_count_builds()
    plain <- kumo_spec(cs$head, chunk_cache = FALSE)
    auto <- kumo_spec(cs$head)
    state <- auto$fit(X, cs$y)
    ref <- plain$predict(state, X_new, type = cs$type)
    expect_identical(builds()$n, 0L, info = nm)

    got <- auto$predict(state, X_new, type = cs$type)
    # One cache per member, each built once for all three chunks, and
    # each built with the previous member's already dropped.
    expect_identical(builds()$n, 3L, info = nm)
    expect_identical(builds()$alive, c(0L, 0L, 0L), info = nm)
    expect_equal(got, ref, tolerance = 1e-5, info = nm)
    if (nm == "ecoc") expect_identical(dim(got), c(20L, 14L))

    # The members' own outputs take the same path.
    m_ref <- plain$predict_members(state, X_new)
    m_got <- auto$predict_members(state, X_new)
    expect_identical(builds()$n, 6L, info = nm)
    expect_equal(m_got, m_ref, tolerance = 1e-5, info = nm)

    # A single chunk has nothing to share a cache across.
    one <- auto$predict(state, X_new[1:CHUNK, , drop = FALSE], type = cs$type)
    expect_identical(builds()$n, 6L, info = nm)
    expect_identical(one, plain$predict(state, X_new[1:CHUNK, , drop = FALSE],
                                        type = cs$type), info = nm)
    # And nothing is kept.
    expect_null(state$kv_caches)
  }
})

test_that("Kumo's per-call caches live one member at a time and are dropped", {
  X <- cache_test_matrix(n = 40L); y <- as.numeric(X[, 1])
  X_new <- cache_test_matrix(n = 15L, seed = 51L)
  stores <- list()
  orig <- .kumo_store
  local_mocked_bindings(.kumo_store = function(...) {
    src <- orig(...)
    stores[[length(stores) + 1L]] <<- src$store
    src
  })
  builds <- kumo_count_builds()
  for (kv in c(FALSE, TRUE)) {
    spec <- kumo_spec("regressor", kv_cache = kv)
    spec$predict(spec$fit(X, y), X_new)
  }
  # `kv_cache = TRUE` builds per call too, now a member at a time.
  expect_identical(builds()$alive, rep(0L, 6L))
  expect_length(stores, 2L)
  for (s in stores) expect_length(ls(s), 0L)
})

test_that("Kumo's chunk_cache steps aside where the cache would not fit", {
  X <- cache_test_matrix(); y <- as.numeric(X[, 1])
  X_new <- cache_test_matrix(n = 15L, seed = 52L)
  builds <- kumo_count_builds()
  local_mocked_bindings(.chunk_cache_fits = function(...) FALSE)
  spec <- kumo_spec("regressor")
  state <- spec$fit(X, y)
  expect_identical(spec$predict(state, X_new),
                   kumo_spec("regressor", chunk_cache = FALSE)$predict(state, X_new))
  expect_identical(builds()$n, 0L)
})

test_that("Kumo's gate asks the memory estimate with the cache on", {
  seen <- NULL
  local_mocked_bindings(.chunk_cache_fits = function(ctx, task, n_context, n_query,
                                                     n_features, opts,
                                                     or_lower = FALSE) {
    seen <<- list(task = task, n_context = n_context, n_query = n_query,
                  n_features = n_features, opts = opts, or_lower = or_lower)
    TRUE
  })
  X <- cache_test_matrix(n = 60L)
  y <- factor(rep(letters[1:14], length.out = 60L))
  spec <- kumo_spec("classifier", row_chunk_size = 16L)
  spec$predict(spec$fit(X, y), cache_test_matrix(n = 15L, seed = 53L))
  expect_identical(seen$task, "classification")
  expect_equal(c(seen$n_context, seen$n_query, seen$n_features), c(60, 15, 4))
  # The sub-tasks ECOC multiplies every term by, and the predictor's knobs.
  expect_identical(seen$opts$n_classes, 14L)
  expect_identical(seen$opts$n_estimators, 3L)
  expect_identical(seen$opts$row_chunk_size, 16L)
  expect_identical(as.integer(seen$opts$predict_chunk_size), CHUNK)
  # A Kumo cache lowers the peak, so it may build past "ok" when it does.
  expect_true(seen$or_lower)
})

test_that("the gate builds a cache that lowers the peak, on any verdict", {
  ctx <- list(config = mem_config_kumo("small"), backend = get_backend("kumo"),
              device = "cpu")
  opts <- list(n_estimators = 8L, predict_chunk_size = 1024L)
  # Tight with or without the cache: the plain rule refuses, the Kumo
  # rule builds, since the cached estimate is the lower one.
  local_mocked_bindings(.system_memory = function() {
    list(total = 64e9, available = 20e9, source = "test")
  })
  on <- estimate_peak_memory(ctx$config, 8000, 4096, 10, backend = "kumo",
                             task = "regression", kv_cache = TRUE,
                             n_estimators = 8L, predict_chunk_size = 1024L)
  off <- estimate_peak_memory(ctx$config, 8000, 4096, 10, backend = "kumo",
                              task = "regression", kv_cache = FALSE,
                              n_estimators = 8L, predict_chunk_size = 1024L)
  expect_identical(on$verdict, "tight")
  expect_lt(on$total_peak_bytes, off$total_peak_bytes)
  expect_false(.chunk_cache_fits(ctx, "regression", 8000, 4096, 10, opts))
  expect_true(.chunk_cache_fits(ctx, "regression", 8000, 4096, 10, opts,
                                or_lower = TRUE))
  # Where the cache would raise it, "ok" is still the only way in.
  local_mocked_bindings(estimate_peak_memory = function(model, ..., kv_cache) {
    list(verdict = "tight", total_peak_bytes = if (kv_cache) 2 else 1)
  })
  expect_false(.chunk_cache_fits(ctx, "regression", 8000, 4096, 10, opts,
                                 or_lower = TRUE))
})

test_that("a Kumo chunk cache that cannot be built leaves the member uncached", {
  X <- cache_test_matrix(); y <- factor(rep(c("a", "b", "c"), length.out = 30L))
  X_new <- cache_test_matrix(n = 15L, seed = 54L)
  spec <- kumo_spec()
  state <- spec$fit(X, y)
  plain <- kumo_spec(chunk_cache = FALSE)$predict(state, X_new, type = "prob")
  # The build is the one call that transforms the training rows alone.
  orig <- kumo_recipe_member
  local_mocked_bindings(kumo_recipe_member = function(rec, X, m) {
    if (nrow(X) == 30L) stop("no room")
    orig(rec, X, m)
  })
  expect_identical(spec$predict(state, X_new, type = "prob"), plain)
  # Asked for explicitly, the same failure is the caller's to see.
  expect_error(kumo_spec(kv_cache = TRUE)$predict(state, X_new, type = "prob"),
               "no room")
})

test_that("Kumo's chunk_cache is TRUE or FALSE", {
  expect_error(kumo_spec(chunk_cache = NA), "TRUE.*FALSE")
  expect_error(kumo_spec("regressor", chunk_cache = "auto"), "TRUE.*FALSE")
})


test_that("chunked regression returns one value per row on every backend path", {
  # The point estimate is a vector per chunk; 25 rows in chunks of 7 used
  # to come back as 28 (the last chunk recycled to the chunk width).
  X <- cache_test_matrix(); yr <- as.numeric(X[, 1] * 2)
  X_new <- cache_test_matrix(n = 25L, seed = 49L)
  reg <- kumo_regressor(fake_ctx(tiny_kumo("regressor", kv = NULL),
                                 config = list(num_quantiles = 9L)),
                        n_estimators = 2L, predict_chunk_size = CHUNK)
  st <- reg$fit(X, yr)
  for (type in c("mean", "median")) {
    p <- reg$predict(st, X_new, type = type)
    expect_length(p, 25L)
    # Each row's answer is the one it gets when predicted on its own.
    expect_equal(p[25], reg$predict(st, X_new[25, , drop = FALSE], type = type),
                 tolerance = 1e-5, label = type)
  }
  expect_identical(dim(reg$predict(st, X_new, type = "grid")), c(25L, 9L))
})
