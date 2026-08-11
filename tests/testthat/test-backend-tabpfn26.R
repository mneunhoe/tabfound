# TabPFN v2.6 architecture, without the 43 MB of weights.
#
# A randomly-initialised model at 1/8 the width answers the questions that
# do not need real weights: does the module tree carry exactly the names
# the checkpoint does, does the forward pass hold its shapes, and does the
# in-architecture preprocessing survive the inputs it exists to handle
# (constant columns, NaN, Inf, a feature count that is not a multiple of
# the group size). Numerical agreement with the reference is a separate
# question, answered in `test-parity-tabpfn26.R`.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")

tiny_config <- function(head = "classifier", encoder_type = "linear") {
  list(
    arch = "tabpfn_v2_6", head = head,
    n_layers = 2L, embedding_dim = 24L, n_heads = 3L, mlp_hidden_dim = 48L,
    features_per_group = 3L, num_thinking_rows = 4L,
    encoder_type = encoder_type, encoder_mlp_hidden_dim = 16L,
    n_out_classes = if (head == "classifier") 10L else NULL,
    n_bar_bins = if (head == "regressor") 7L else NULL
  )
}

# The real buffer is 2000 x 48, sized for the released 192-wide models.
# At test width the shape has to follow, so stand in a deterministic one.
tiny_col_emb <- function(net, n = 64L) {
  torch::torch_manual_seed(7L)
  torch::torch_randn(c(as.integer(n), net$pos_emb_dim))
}


test_that("the module tree carries the checkpoint's own parameter names", {
  net <- tabpfn_v2_6_transformer(tiny_config())
  nms <- names(net$state_dict())

  expect_true(all(c(
    "feature_group_embedder.weight",
    "target_embedder.weight", "target_embedder.bias",
    "add_thinking_rows.row_token_values_TE",
    "blocks.0.per_sample_attention_between_features.q_projection.weight",
    "blocks.0.per_sample_attention_between_features.out_projection.weight",
    "blocks.0.per_column_attention_between_cells.k_projection.weight",
    "blocks.0.per_column_attention_between_cells.v_projection.weight",
    "blocks.1.layernorm_mha1.weight", "blocks.1.layernorm_mha2.weight",
    "blocks.1.layernorm_mlp.weight",
    "blocks.1.mlp.0.weight", "blocks.1.mlp.2.weight",
    "output_projection.0.weight", "output_projection.0.bias",
    "output_projection.2.weight", "output_projection.2.bias",
    "feature_positional_embedding_embeddings.weight",
    "feature_positional_embedding_embeddings.bias"
  ) %in% nms))

  # Biases the reference explicitly turns off must not reappear: an extra
  # parameter would leave the strict loader with nothing to fill it from.
  expect_false(any(grepl("projection\\.bias$", nms)))
  expect_false("blocks.0.mlp.0.bias" %in% nms)
  expect_false("feature_group_embedder.bias" %in% nms)
  # A classifier has no bar-distribution buffer.
  expect_false(any(grepl("^criterion\\.", nms)))

  # The regressor adds the buffers and swaps the linear embedder for an MLP.
  reg <- tabpfn_v2_6_transformer(tiny_config("regressor", "mlp"))
  rnms <- names(reg$state_dict())
  expect_true(all(c("criterion.borders", "criterion.losses_per_bucket",
                    "feature_group_embedder.0.weight",
                    "feature_group_embedder.2.weight") %in% rnms))
  expect_false("feature_group_embedder.weight" %in% rnms)
  expect_equal(as.integer(reg$state_dict()[["criterion.borders"]]$size()), 8L)
})


test_that("the forward pass holds its shapes", {
  net <- tabpfn_v2_6_transformer(tiny_config())
  net$eval()
  col <- tiny_col_emb(net)

  x_tr <- torch::torch_randn(c(1L, 20L, 5L))
  x_te <- torch::torch_randn(c(1L, 7L, 5L))
  y_tr <- torch::torch_randint(0L, 3L, c(1L, 20L))$to(dtype = torch::torch_float())

  out <- torch::with_no_grad(net(x_tr, y_tr, x_te, column_embeddings = col))
  expect_equal(as.integer(out$logits$size()), c(1L, 7L, 10L))
  # 5 features -> 2 groups of 3 (one zero-padded), plus the target column.
  expect_equal(as.integer(out$encoder_out$size())[3], 3L)
  # 4 thinking rows sit in front of the 20 training rows.
  expect_identical(out$single_eval_pos, 24L)
  expect_true(as.logical(torch::torch_isfinite(out$logits)$all()))
})


test_that("NaN, Inf and constant columns are absorbed, not propagated", {
  net <- tabpfn_v2_6_transformer(tiny_config())
  net$eval()
  col <- tiny_col_emb(net)

  x_tr <- torch::torch_randn(c(1L, 24L, 4L))
  x_te <- torch::torch_randn(c(1L, 6L, 4L))
  x_tr[1, 1, 1] <- NaN
  x_tr[1, 2, 2] <- Inf
  x_te[1, 1, 3] <- -Inf
  x_tr[1, , 4] <- 1.5          # constant column, dropped by the architecture
  x_te[1, , 4] <- 1.5
  y_tr <- torch::torch_randint(0L, 2L, c(1L, 24L))$to(dtype = torch::torch_float())

  out <- torch::with_no_grad(net(x_tr, y_tr, x_te, column_embeddings = col))
  expect_true(as.logical(torch::torch_isfinite(out$logits)$all()))
  # The constant column is gone, so 3 informative features make one group.
  expect_equal(as.integer(out$encoder_out$size())[3], 2L)
})


test_that("out-of-range labels are refused rather than silently truncated", {
  net <- tabpfn_v2_6_transformer(tiny_config())
  net$eval()
  col <- tiny_col_emb(net)
  x_tr <- torch::torch_randn(c(1L, 12L, 3L))
  x_te <- torch::torch_randn(c(1L, 3L, 3L))
  # The head is 10 wide; a label of 11 has no slot to be decoded into.
  y_bad <- torch::torch_full(c(1L, 12L), 11)$to(dtype = torch::torch_float())
  expect_error(net(x_tr, y_bad, x_te, column_embeddings = col),
               "out of range")
})


test_that("more feature groups than pre-generated embeddings is an error", {
  net <- tabpfn_v2_6_transformer(tiny_config())
  # The reference falls back to a device-dependent random draw past the
  # buffer's length, which cannot be reproduced here; guessing silently
  # would give wrong predictions that look fine.
  expect_error(
    net$column_positional_embedding(
      2001L, torch::torch_zeros(c(2000L, 6L)), device = "cpu",
      dtype = torch::torch_float()
    ),
    "exceeds"
  )
})


test_that("chunked evaluation splits work without changing it", {
  # `chunked_evaluate` is where a subtle bug would be invisible: the
  # shapes stay right whether or not the in-place write-back lands, so
  # a silent no-op looks exactly like success.
  set.seed(21)
  x <- torch::torch_randn(c(2L, 5L, 3L, 4L))
  f <- function(t) t * 2 + 1

  base <- chunked_evaluate(f, x$clone(), NULL, residual = FALSE, batch_dims = 3L)
  for (k in c(1L, 2L, 7L, 100L)) {
    got <- chunked_evaluate(f, x$clone(), k, residual = FALSE, batch_dims = 3L)
    expect_true(as.logical((got == base)$all()), info = paste("factor", k))
  }
  # The residual form, and a fold that leaves a non-trivial inner shape.
  res <- chunked_evaluate(f, x$clone(), 3L, residual = TRUE, batch_dims = 2L)
  expect_true(as.logical((res == x + base)$all()))

  # A non-contiguous input still has to come back with the result in it:
  # `flatten` would silently copy, and the in-place writes would land in
  # the copy.
  xt <- torch::torch_randn(c(3L, 4L, 2L))$transpose(1L, 2L)
  expect_false(xt$is_contiguous())
  got <- chunked_evaluate(f, xt, 2L, residual = FALSE, batch_dims = 2L)
  expect_true(as.logical((got == f(xt$contiguous()))$all()))
})


test_that("the KV cache reproduces the forward it was built from", {
  net <- tabpfn_v2_6_transformer(tiny_config())
  net$eval()
  col <- tiny_col_emb(net)
  set.seed(22)
  x_tr <- torch::torch_randn(c(1L, 30L, 5L))
  x_te <- torch::torch_randn(c(1L, 9L, 5L))
  y_tr <- torch::torch_randint(0L, 3L, c(1L, 30L))$to(dtype = torch::torch_float())

  plain <- torch::with_no_grad(net(x_tr, y_tr, x_te, column_embeddings = col))$logits

  no_rows <- torch::torch_zeros(c(1L, 0L, 5L))
  built <- torch::with_no_grad(net(x_tr, y_tr, no_rows, column_embeddings = col,
                                   return_kv_cache = TRUE))
  cache <- built$kv_cache
  expect_s3_class(cache, "tabpfn26_kv_cache")
  expect_length(cache$kv, length(net$blocks))
  expect_identical(cache$n_train, 30L)
  # Only one key/value head is kept, whatever the model's head count.
  expect_identical(as.integer(cache$kv[[1]]$key$size(2)), 1L)
  # A build pass with no test rows decodes nothing.
  expect_null(built$logits)

  cached <- torch::with_no_grad(net(NULL, NULL, x_te, column_embeddings = col,
                                    kv_cache = cache))$logits
  expect_equal(as.array(cached), as.array(plain), tolerance = 1e-6)

  # And the whole point: the answer must not depend on how the test rows
  # are batched, because that is what lets one cache serve every chunk.
  halves <- lapply(list(1:4, 5:9), function(ix) {
    torch::with_no_grad(net(NULL, NULL, x_te[, ix, , drop = FALSE],
                            column_embeddings = col, kv_cache = cache))$logits
  })
  expect_equal(as.array(torch::torch_cat(halves, dim = 2L)), as.array(cached),
               tolerance = 1e-6)
})


test_that("chunking and the cache compose", {
  net <- tabpfn_v2_6_transformer(tiny_config("regressor", "mlp"))
  net$eval()
  col <- tiny_col_emb(net)
  set.seed(23)
  x_tr <- torch::torch_randn(c(1L, 24L, 4L))
  x_te <- torch::torch_randn(c(1L, 6L, 4L))
  y_tr <- torch::torch_randn(c(1L, 24L))

  plain <- torch::with_no_grad(net(x_tr, y_tr, x_te, column_embeddings = col))$logits
  both <- torch::with_no_grad({
    cache <- net(x_tr, y_tr, torch::torch_zeros(c(1L, 0L, 4L)),
                 column_embeddings = col, return_kv_cache = TRUE,
                 save_peak_memory_factor = 4L)$kv_cache
    net(NULL, NULL, x_te, column_embeddings = col, kv_cache = cache,
        save_peak_memory_factor = 4L)$logits
  })
  expect_equal(as.array(both), as.array(plain), tolerance = 1e-6)
})
