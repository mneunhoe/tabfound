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
