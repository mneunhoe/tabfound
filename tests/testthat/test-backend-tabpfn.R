# TabPFN v2 / v2.5 architecture, without the 41 MB of weights.
#
# A randomly-initialised model at 1/8 the width answers what does not need
# real weights: does the KV cache reproduce the forward it was built from,
# and does it stay independent of how the test rows are batched. Agreement
# with the reference is a separate question, answered in
# `test-parity-tabpfn.R`.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")

tiny_v25_config <- function(head = "classifier") {
  list(
    arch = "per_feature_transformer", head = head,
    n_layers = 2L, embedding_dim = 24L, n_heads = 3L, mlp_hidden_dim = 48L,
    features_per_group = 3L, layer_norm_eps = 1e-5, activation = "gelu",
    encoder_mlp_hidden_dim = 16L,
    n_out_classes = if (head == "classifier") 10L else NULL,
    n_bar_bins = if (head == "regressor") 7L else NULL
  )
}

# The bundled buffer is 2000 x 48, sized for the released 192-wide models.
tiny_v25_col_emb <- function(n = 64L) {
  torch::torch_manual_seed(5L)
  torch::torch_randn(c(as.integer(n), 48L))
}


test_that("the v2.5 network advertises both the cache and chunked evaluation", {
  net <- per_feature_transformer(tiny_v25_config())
  expect_true(isTRUE(net$supports_kv_cache))
  # `save_peak_memory_factor` used to be a v2.6-and-newer path. It is
  # not any more: v2.5's layer has the same alternating shape and the
  # same independent folds, so the same wrapping applies.
  expect_true(isTRUE(net$supports_chunked_eval))
  # The stage chunking, though, really is v3 only.
  expect_false(isTRUE(net$supports_stage_chunking))
})


test_that("more classes than the head has are refused at fit()", {
  # v2.5 used to accept them and fail at predict on a dimnames mismatch;
  # v2.6 only once its forward pass saw the labels. The fit itself needs
  # no network, only the head's width from the config.
  X <- matrix(as.numeric(seq_len(2 * 24)), ncol = 2L)
  y <- factor(paste0("c", rep(0:11, 2)))
  for (arch in c("per_feature_transformer", "tabpfn_v2_6")) {
    ctx <- list(net = list(), device = "cpu",
                config = list(arch = arch, head = "classifier",
                              n_out_classes = 10L))
    spec <- tabpfn_classifier(ctx)
    expect_error(spec$fit(X, y), "10-class head", info = arch)
    expect_error(spec$fit(X, y), "has 12 classes", info = arch)
    # Ten still fit.
    keep <- y %in% levels(y)[1:10]
    expect_no_error(spec$fit(X[keep, ], droplevels(y[keep])))
  }
})


test_that("the v2.5 KV cache reproduces the forward it was built from", {
  net <- per_feature_transformer(tiny_v25_config())
  net$eval()
  col <- tiny_v25_col_emb()
  set.seed(31)
  x_tr <- torch::torch_randn(c(1L, 40L, 5L))
  x_te <- torch::torch_randn(c(1L, 11L, 5L))
  y_tr <- torch::torch_randint(0L, 3L, c(1L, 40L))$to(dtype = torch::torch_float())

  plain <- torch::with_no_grad(net(x_tr, y_tr, x_te, column_embeddings = col))$logits

  no_rows <- torch::torch_zeros(c(1L, 0L, 5L))
  built <- torch::with_no_grad(net(x_tr, y_tr, no_rows, column_embeddings = col,
                                   return_kv_cache = TRUE))
  cache <- built$kv_cache
  expect_s3_class(cache, "tabpfn_kv_cache")
  expect_length(cache$kv, 2L)
  expect_identical(cache$n_train, 40L)
  # Only head 0 of the key/value projections is kept: test rows attend to
  # that head alone, so nothing else could ever be read back.
  expect_identical(as.integer(cache$kv[[1]]$key$size(2)), 1L)
  # A build pass with no test rows has nothing to decode.
  expect_null(built$logits)

  cached <- torch::with_no_grad(net(NULL, NULL, x_te, column_embeddings = col,
                                    kv_cache = cache))$logits
  expect_equal(as.array(cached), as.array(plain), tolerance = 1e-5)
})


test_that("a v2.5 cache is not reusable across a different feature count", {
  net <- per_feature_transformer(tiny_v25_config())
  net$eval()
  col <- tiny_v25_col_emb()
  set.seed(32)
  x_tr <- torch::torch_randn(c(1L, 20L, 5L))
  y_tr <- torch::torch_randint(0L, 2L, c(1L, 20L))$to(dtype = torch::torch_float())
  cache <- torch::with_no_grad(net(x_tr, y_tr, torch::torch_zeros(c(1L, 0L, 5L)),
                                   column_embeddings = col,
                                   return_kv_cache = TRUE))$kv_cache

  # The cached key/value projections are shaped by the training matrix, so
  # feeding rows of a different width has to be refused rather than
  # silently broadcast into the wrong columns.
  expect_error(
    net(NULL, NULL, torch::torch_randn(c(1L, 4L, 8L)),
        column_embeddings = col, kv_cache = cache),
    "feature group"
  )
})


test_that("the v2.5 cache's fitted statistics come only from the training rows", {
  # This is what makes the cache exactly equivalent rather than merely
  # close: `preprocess_x_for_encoder` never looks at a test row, so
  # reusing its state cannot answer a different question. Feeding wildly
  # different test rows must not move the fitted statistics.
  set.seed(33)
  x <- torch::torch_randn(c(30L, 4L, 3L))
  a <- preprocess_x_for_encoder(x, single_eval_pos = 20L)
  x2 <- x$clone()
  x2[21:30, , ] <- x2[21:30, , ] * 1000 + 500
  b <- preprocess_x_for_encoder(x2, single_eval_pos = 20L)

  for (nm in c("feature_means", "mean", "std")) {
    expect_true(as.logical((a$state[[nm]] == b$state[[nm]])$all()), info = nm)
  }
  expect_true(as.logical((a$state$non_const == b$state$non_const)$all()))
  # And the training rows come out the same however the test rows change.
  expect_true(as.logical((a$main[1:20, , ] == b$main[1:20, , ])$all()))
})


# ---------------------------------------------------------------------------
# Within-sublayer chunking
# ---------------------------------------------------------------------------

test_that("v2.5's chunking agrees with the plain layer at every factor", {
  # This splits folds of *leading* dimensions only -- rows for the
  # feature attention, columns for the item attention, every cell for the
  # MLP and the norms -- so no attention's own sequence length changes.
  # What does change is its batch size, and the kernel blocks by batch,
  # so the result is exact when the fold divides evenly and float32's
  # last bits otherwise: ~1.6e-7 relative here, against 5.4e-7 for
  # Mitra's chunking and 1.4e-5 for v3's stage chunking.
  net <- per_feature_encoder_layer(embedding_dim = 16L, n_heads = 2L,
                                   mlp_hidden_dim = 32L)
  net$eval()
  set.seed(81)
  x <- torch::torch_randn(c(1L, 24L, 5L, 16L))
  base <- as.array(torch::with_no_grad(
    net(x$clone(), single_eval_pos = 15L)$state))
  scale <- max(abs(base))
  for (k in c(2L, 3L, 8L, 100L)) {
    got <- as.array(torch::with_no_grad(
      net(x$clone(), single_eval_pos = 15L,
          save_peak_memory_factor = k)$state))
    expect_equal(dim(got), dim(base))
    expect_lt(max(abs(got - base)), 1e-5 * scale)
  }
  # 24 rows into 2 is an even fold, and there it is exact.
  expect_identical(
    as.array(torch::with_no_grad(
      net(x$clone(), single_eval_pos = 15L,
          save_peak_memory_factor = 2L)$state)),
    base
  )
})

test_that("v2.5's chunking leaves the cache paths alone", {
  # Building or reading a cache needs the key/value tensors whole, so
  # those branches bypass the chunking -- exactly as v2.6 does. What
  # matters is that asking for a factor there is still correct, not that
  # it saves anything.
  net <- per_feature_encoder_layer(embedding_dim = 16L, n_heads = 2L,
                                   mlp_hidden_dim = 32L)
  net$eval()
  set.seed(82)
  x <- torch::torch_randn(c(1L, 24L, 5L, 16L))

  built <- torch::with_no_grad(
    net(x$clone(), single_eval_pos = 15L, return_kv = TRUE))
  expect_false(is.null(built$kv))
  chunked_build <- torch::with_no_grad(
    net(x$clone(), single_eval_pos = 15L, return_kv = TRUE,
        save_peak_memory_factor = 4L))
  expect_identical(as.array(chunked_build$state), as.array(built$state))
  expect_identical(as.array(chunked_build$kv$key), as.array(built$kv$key))

  xt <- torch::torch_randn(c(1L, 7L, 5L, 16L))
  a <- as.array(torch::with_no_grad(
    net(xt$clone(), single_eval_pos = 0L, cached_kv = built$kv)$state))
  b <- as.array(torch::with_no_grad(
    net(xt$clone(), single_eval_pos = 0L, cached_kv = built$kv,
        save_peak_memory_factor = 4L)$state))
  # The item attention ran whole on both -- the cache branch bypasses
  # chunking -- but the *feature* attention did not, and 7 rows into 4
  # chunks is ragged, so its batch size differs and the last bits move.
  expect_lt(max(abs(a - b)), 1e-5 * max(abs(a)))
})


test_that("a single-pass sample is chunked and cached like the mean", {
  d <- Sys.getenv("TABFOUND_TABPFN_REG_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN_REG_DIR not configured")

  # The sample branch used to run one uncached forward over all of
  # `newdata`. It now goes a `predict_chunk_size` at a time, through the
  # cache when there is one, with the uniforms drawn once for every row:
  # a seed gives the same draws whatever the chunking, up to the last
  # bits the forward's batch size moves.
  set.seed(13)
  X <- matrix(rnorm(150 * 4), ncol = 4)
  y <- as.numeric(X %*% c(1, -1, 0.5, 0)) + rnorm(150, sd = 0.3)
  tr <- 1:110; te <- 111:150
  draw <- function(...) {
    m <- fit(tabular_regressor(d, ...), X[tr, ], y[tr])
    predict(m, X[te, ], type = "sample", n_samples = 30L, seed = 4L)
  }
  whole <- draw(predict_chunk_size = 1024L)
  expect_identical(dim(whole), c(40L, 30L))
  expect_equal(draw(predict_chunk_size = 9L), whole, tolerance = 1e-5)
  expect_equal(draw(predict_chunk_size = 9L, kv_cache = TRUE), whole,
               tolerance = 1e-5)
})

# ---------------------------------------------------------------------------
# Member-at-a-time ensembles
# ---------------------------------------------------------------------------
#
# A cached ensemble runs members outer and chunks inner, so only one
# member's cache is alive. The question is whether that is still the
# answer the old order gave -- chunks outer, every member in turn inside
# each -- and it has to be the same answer bit for bit, weighted sums
# included: each row still adds its members in the same order.

tiny_v25_predictor <- function(head = "classifier", ...) {
  torch::torch_manual_seed(7L)
  net <- per_feature_transformer(tiny_v25_config(head))
  net$eval()
  ctx <- list(net = net, device = "cpu", config = tiny_v25_config(head))
  if (head == "classifier") tabpfn_classifier(ctx, ...) else
    tabpfn_regressor(ctx, ...)
}

tiny_v25_data <- function(task, n_tr = 48L, n_te = 23L, p = 5L) {
  set.seed(61)
  X <- matrix(rnorm((n_tr + n_te) * p), ncol = p)
  eta <- X[, 1] - X[, 2]
  y <- if (task == "classification") {
    factor(cut(eta, c(-Inf, -0.5, 0.5, Inf), labels = c("a", "b", "c")))
  } else eta + rnorm(length(eta), sd = 0.2)
  list(X_tr = X[seq_len(n_tr), ], y_tr = y[seq_len(n_tr)],
       X_te = X[n_tr + seq_len(n_te), ])
}

spec_net <- function(spec) environment(spec$fit)$net

# The old order, kept as the reference: chunks outer, and each chunk
# running every member against caches that are all alive at once.
chunks_outer_classifier <- function(spec, state, X_te, chunk, cfg, caches,
                                    w = NULL, return_members = FALSE) {
  net <- spec_net(spec)
  col <- load_column_embeddings()
  fn <- function(ch) {
    apply_ensemble_predict_classifier(
      net, col, state$X_train, ch, state$y_train_int, cfg,
      n_classes = length(state$class_levels), softmax_temperature = 0.9,
      categorical_features = state$categorical_features,
      kv_caches = caches, member_weights = w,
      return_members = return_members)
  }
  if (return_members) {
    .members_over_chunks(X_te, chunk, function(ch) fn(ch)$members)
  } else chunk_apply(X_te, chunk, fn)
}

test_that("a cached classifier ensemble is bit-identical member-outer", {
  dat <- tiny_v25_data("classification")
  cfg <- generate_ensemble_configs_native(dat$X_tr, dat$y_tr,
                                          n_estimators = 4L, random_state = 2L)
  for (w in list(NULL, c(3, 1, 2, 0.5))) {
    spec <- tiny_v25_predictor(ensemble_configs = cfg, kv_cache = TRUE,
                               predict_chunk_size = 10L, member_weights = w)
    state <- spec$fit(dat$X_tr, dat$y_tr)
    caches <- spec$build_cache(state)
    ref <- chunks_outer_classifier(spec, state, dat$X_te, 10L, cfg, caches,
                                   w = w)
    got <- spec$predict(state, dat$X_te, type = "prob")
    expect_identical(unname(got), unname(ref))

    # Uncached, the reordering is the same arithmetic again.
    plain <- tiny_v25_predictor(ensemble_configs = cfg, chunk_cache = FALSE,
                                predict_chunk_size = 10L, member_weights = w)
    expect_identical(
      unname(plain$predict(state, dat$X_te, type = "prob")),
      unname(chunks_outer_classifier(plain, state, dat$X_te, 10L, cfg, NULL,
                                     w = w)))

    # A kept cache (every member, by design) gives the same numbers as one
    # built a member at a time for the call.
    kept <- state; kept$kv_caches <- caches
    expect_identical(spec$predict(kept, dat$X_te, type = "prob"), got)

    # And so do the members, chunk by chunk.
    m <- spec$predict_members(state, dat$X_te)$logits
    ref_m <- chunks_outer_classifier(spec, state, dat$X_te, 10L, cfg, caches,
                                     w = w, return_members = TRUE)
    expect_identical(lapply(m, unname), lapply(ref_m, unname))
  }
})

test_that("a cached regressor ensemble is bit-identical member-outer", {
  dat <- tiny_v25_data("regression")
  cfg <- generate_ensemble_configs_native(dat$X_tr, dat$y_tr, head = "regressor",
                                          n_estimators = 4L, random_state = 4L)
  q <- c(0.1, 0.5, 0.9)
  for (w in list(NULL, c(1, 2, 0.5, 3))) {
    spec <- tiny_v25_predictor("regressor", ensemble_configs = cfg,
                               kv_cache = TRUE, predict_chunk_size = 10L,
                               member_weights = w)
    state <- spec$fit(dat$X_tr, dat$y_tr)
    caches <- spec$build_cache(state)
    net <- spec_net(spec)
    rows <- split(seq_len(nrow(dat$X_te)), ceiling(seq_len(nrow(dat$X_te)) / 10))
    ref <- lapply(rows, function(r) {
      apply_ensemble_predict_regressor(
        net, load_column_embeddings(), state$X_train,
        dat$X_te[r, , drop = FALSE], state$y_train, cfg,
        borders = .regression_borders_of(net), quantiles = q,
        softmax_temperature = 0.9,
        categorical_features = state$categorical_features,
        kv_caches = caches, member_weights = w, return_members = TRUE)
    })
    expect_identical(spec$predict(state, dat$X_te),
                     unlist(lapply(ref, `[[`, "mean"), use.names = FALSE))
    expect_identical(
      unname(spec$predict(state, dat$X_te, type = "quantiles", quantiles = q)),
      unname(do.call(rbind, lapply(ref, `[[`, "quantiles"))))
    m <- spec$predict_members(state, dat$X_te)$probs
    for (i in seq_along(m)) {
      expect_identical(
        m[[i]], unname(do.call(rbind, lapply(ref, function(r) r$members[[i]]))))
    }
  }
})

test_that("each member's cache is built, used on every chunk, then dropped", {
  log <- character()
  steps <- list(
    start  = function(X) list(seen = character()),
    add    = function(st, i, cache) {
      log <<- c(log, sprintf("add %d/%s", i, cache))
      st$seen <- c(st$seen, cache)
      st
    },
    finish = function(st) st$seen
  )
  out <- run_members_outer(matrix(0, 25L, 2L), 10L, 3L, steps,
                           member_cache = function(i) {
                             log <<- c(log, sprintf("build %d", i))
                             paste0("c", i)
                           })
  # The next member's cache is only built once the last one has served
  # every chunk.
  expect_identical(log, c(
    "build 1", "add 1/c1", "add 1/c1", "add 1/c1",
    "build 2", "add 2/c2", "add 2/c2", "add 2/c2",
    "build 3", "add 3/c3", "add 3/c3", "add 3/c3"))
  # Three chunks, each having seen the members in order.
  expect_identical(out, rep(list(c("c1", "c2", "c3")), 3L))
})


# ---------------------------------------------------------------------------
# A cache built for one predict() call: chunk_cache
# ---------------------------------------------------------------------------

# Counts the member caches built, and builds them.
count_builds <- function(env = parent.frame()) {
  n <- 0L
  orig <- build_member_kv_cache
  local_mocked_bindings(build_member_kv_cache = function(...) {
    n <<- n + 1L
    orig(...)
  }, .env = env)
  function() n
}

test_that("chunk_cache builds a cache only for a call of two or more chunks", {
  dat <- tiny_v25_data("classification")
  cfg <- generate_ensemble_configs_native(dat$X_tr, dat$y_tr,
                                          n_estimators = 3L, random_state = 2L)
  builds <- count_builds()
  pred <- function(X, ...) {
    spec <- tiny_v25_predictor(ensemble_configs = cfg,
                               predict_chunk_size = 10L, ...)
    spec$predict(spec$fit(dat$X_tr, dat$y_tr), X, type = "prob")
  }

  plain <- pred(dat$X_te, chunk_cache = FALSE)
  expect_identical(builds(), 0L)
  auto <- pred(dat$X_te)
  # One per member, each built once for all three chunks.
  expect_identical(builds(), 3L)
  # Within the float32 rounding of the narrower batch: this random tiny
  # network is not bit-exact the way the released v2.5 weights are.
  expect_equal(auto, plain, tolerance = 1e-5)

  # A single chunk has nothing to share a cache across.
  pred(dat$X_te[1:10, ])
  expect_identical(builds(), 3L)
})

test_that("chunk_cache steps aside where the cache would not fit", {
  dat <- tiny_v25_data("regression")
  builds <- count_builds()
  spec <- tiny_v25_predictor("regressor", predict_chunk_size = 10L)
  state <- spec$fit(dat$X_tr, dat$y_tr)
  local_mocked_bindings(.chunk_cache_fits = function(...) FALSE)
  plain <- tiny_v25_predictor("regressor", predict_chunk_size = 10L,
                              chunk_cache = FALSE)$predict(state, dat$X_te)
  expect_identical(spec$predict(state, dat$X_te), plain)
  # The single pass builds through the predictor rather than the member
  # builder, so ask the source directly.
  src <- .cache_source(state, 23L, function(i) stop("built"),
                       list(ctx = list(net = list(supports_kv_cache = TRUE)),
                            kv_cache = FALSE, chunk_cache = TRUE,
                            opts = list(predict_chunk_size = 10L)))
  expect_null(src$get(1L))
})

test_that("the gate asks the memory estimate with the cache on", {
  ctx <- list(config = tiny_v25_config(), backend = get_backend("tabpfn"),
              device = "cpu")
  opts <- list(n_estimators = 8L, predict_chunk_size = 1024L)
  local_mocked_bindings(.system_memory = function() {
    list(total = 64e9, available = 32e9, source = "test")
  })
  expect_true(.chunk_cache_fits(ctx, "classification", 2000, 5000, 30, opts))
  local_mocked_bindings(.system_memory = function() {
    list(total = 64e9, available = 1e6, source = "test")
  })
  expect_false(.chunk_cache_fits(ctx, "classification", 2000, 5000, 30, opts))
  # A question the estimate cannot answer is not a reason to refuse.
  expect_true(.chunk_cache_fits(list(config = list()), "classification",
                                2000, 5000, 30, opts))
})

test_that("a chunk cache that cannot be built leaves the member uncached", {
  dat <- tiny_v25_data("classification")
  cfg <- generate_ensemble_configs_native(dat$X_tr, dat$y_tr,
                                          n_estimators = 2L, random_state = 2L)
  spec <- tiny_v25_predictor(ensemble_configs = cfg, predict_chunk_size = 10L)
  state <- spec$fit(dat$X_tr, dat$y_tr)
  plain <- tiny_v25_predictor(ensemble_configs = cfg, predict_chunk_size = 10L,
                              chunk_cache = FALSE)$predict(state, dat$X_te,
                                                           type = "prob")
  local_mocked_bindings(build_member_kv_cache = function(...) stop("no room"))
  expect_identical(spec$predict(state, dat$X_te, type = "prob"), plain)
  # Asked for explicitly, the same failure is the caller's to see.
  asked <- tiny_v25_predictor(ensemble_configs = cfg, predict_chunk_size = 10L,
                              kv_cache = TRUE)
  expect_error(asked$predict(state, dat$X_te, type = "prob"), "no room")
})

test_that("chunk_cache is TRUE or FALSE", {
  expect_error(tiny_v25_predictor(chunk_cache = NA), "TRUE.*FALSE")
  expect_error(tiny_v25_predictor(chunk_cache = "auto"), "TRUE.*FALSE")
})
