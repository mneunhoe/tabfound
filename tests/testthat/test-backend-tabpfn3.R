# TabPFN v3 architecture, without the 211 MB of weights.
#
# A randomly-initialised model at a fraction of the width answers the
# questions that do not need real weights: does the module tree carry
# exactly the names the checkpoint does, do the four stages hold their
# shapes, and does the in-architecture preprocessing survive the inputs it
# exists to handle (constant columns, NaN, Inf, fewer columns than the
# feature grouping shifts by). Numerical agreement with the reference is a
# separate question, answered in `test-parity-tabpfn3.R`.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")

tiny_config <- function(head = "classifier") {
  list(
    arch = "tabpfn_v3", head = head,
    embed_dim = 16L, feature_group_size = 3L, use_nan_indicators = TRUE,
    dist_embed_num_blocks = 2L, dist_embed_num_heads = 2L,
    dist_embed_num_inducing_points = 4L,
    feat_agg_num_blocks = 2L, feat_agg_num_heads = 2L,
    feat_agg_num_cls_tokens = 2L, feat_agg_rope_base = 100000, use_rope = TRUE,
    nlayers = 2L, icl_num_heads = 2L,
    icl_num_kv_heads = NULL, icl_num_kv_heads_test = 1L,
    decoder_head_dim = 4L, decoder_num_heads = 2L,
    decoder_use_softmax_scaling = identical(head, "classifier"),
    ff_factor = 2L, softmax_scaling_mlp_hidden_dim = 8L,
    layernorm_elementwise_affine = TRUE,
    max_num_classes = if (head == "classifier") 6L else 0L,
    num_buckets = 7L, n_bar_bins = 7L,
    n_out = if (head == "classifier") 6L else 7L
  )
}

tiny_net <- function(head = "classifier") {
  net <- tabpfn_v3_transformer(tiny_config(head))
  net$eval()
  net
}


test_that("the module tree carries the checkpoint's own parameter names", {
  nms <- names(tiny_net("classifier")$state_dict())

  expect_true(all(c(
    "x_embed.weight", "x_embed.bias",
    "col_y_encoder.embedding.weight", "icl_y_encoder.embedding.weight",
    "feature_distribution_embedder.layers.0.inducing_vectors",
    "feature_distribution_embedder.layers.0.cross_attn_block1.attn.q_projection.weight",
    "feature_distribution_embedder.layers.0.cross_attn_block1.attn.softmax_scaling_layer.base_mlp.0.weight",
    "feature_distribution_embedder.layers.0.cross_attn_block1.attn.softmax_scaling_layer.query_mlp.2.bias",
    "feature_distribution_embedder.layers.0.cross_attn_block1.layernorm_q.weight",
    "feature_distribution_embedder.layers.0.cross_attn_block1.layernorm_kv.weight",
    "feature_distribution_embedder.layers.0.cross_attn_block1.layernorm2.weight",
    "feature_distribution_embedder.layers.1.cross_attn_block2.mlp.0.weight",
    "column_aggregator.cls_tokens", "column_aggregator.rope.freqs",
    "column_aggregator.out_ln.weight",
    "column_aggregator.blocks.0.attention.out_projection.weight",
    "column_aggregator.blocks.1.layernorm_mlp.weight",
    "icl_blocks.0.icl_attention.q_projection.weight",
    "icl_blocks.0.icl_attention.k_projection.weight",
    "icl_blocks.0.icl_attention.softmax_scaling_layer.base_mlp.2.weight",
    "icl_blocks.1.mlp.2.weight",
    "output_norm.weight",
    "many_class_decoder.q_projection.weight", "many_class_decoder.k_projection.bias",
    "many_class_decoder.softmax_scaling_layer.base_mlp.0.bias",
    "regression_borders"
  ) %in% nms))

  # Biases the reference explicitly turns off must not reappear: an extra
  # parameter would leave the strict loader with nothing to fill it from.
  expect_false(any(grepl("^(icl_blocks|column_aggregator|feature_distribution_embedder).*projection\\.bias$",
                         nms)))
  # The block feedforward is bias-free; the softmax-scaling MLPs beside it
  # are not, so the pattern has to distinguish `.mlp.` from `_mlp.`.
  expect_false(any(grepl("^(icl_blocks|column_aggregator).*\\.mlp\\.[02]\\.bias$", nms)))
  expect_true(any(grepl("softmax_scaling_layer\\.base_mlp\\.0\\.bias$", nms)))
  # A classifier has no output projection, and its decoder is the
  # retrieval head instead.
  expect_false(any(grepl("^output_projection\\.", nms)))

  # The regressor swaps the decoder and embeds a scalar target.
  rnms <- names(tiny_net("regressor")$state_dict())
  expect_true(all(c("output_projection.0.weight", "output_projection.0.bias",
                    "output_projection.2.weight", "output_projection.2.bias",
                    "col_y_encoder.weight", "icl_y_encoder.bias",
                    # Registered on both heads by the reference.
                    "regression_borders") %in% rnms))
  expect_false(any(grepl("^many_class_decoder\\.", rnms)))
  expect_false(any(grepl("embedding\\.weight$", rnms)))
})


test_that("the forward pass holds its shapes through all four stages", {
  net <- tiny_net("classifier")
  set.seed(3)
  x_tr <- torch::torch_randn(c(1L, 20L, 5L))
  x_te <- torch::torch_randn(c(1L, 7L, 5L))
  y_tr <- torch::torch_randint(0L, 3L, c(1L, 20L))$to(dtype = torch::torch_float())

  out <- torch::with_no_grad(net(x_tr, y_tr, x_te))
  # The decoder is `max_num_classes` wide whatever the data holds.
  expect_equal(as.integer(out$logits$size()), c(1L, 7L, 6L))
  # The row embedding is the CLS tokens concatenated: 2 x 16.
  expect_equal(as.integer(out$test_hidden$size()), c(1L, 7L, 32L))
  expect_equal(as.integer(out$train_hidden$size()), c(1L, 20L, 32L))
  expect_true(as.logical(torch::torch_isfinite(out$logits)$all()))
  # Retrieval logits are logs of probabilities, so they are never positive.
  expect_true(as.logical((out$logits <= 0)$all()))

  reg <- tiny_net("regressor")
  y_num <- torch::torch_randn(c(1L, 20L))
  rout <- torch::with_no_grad(reg(x_tr, y_num, x_te))
  expect_equal(as.integer(rout$logits$size()), c(1L, 7L, 7L))
})


test_that("feature grouping keeps the column count and reads its neighbours", {
  net <- tiny_net("classifier")
  # Columns 1..6 as distinguishable constants, one row.
  x <- torch::torch_tensor(matrix(1:6, nrow = 1))$to(dtype = torch::torch_float())$
    unsqueeze(1L)                                       # (B = 1, Ri = 1, C = 6)
  rolled <- torch::torch_stack(
    lapply(c(1L, 2L, 4L), function(s) torch::torch_roll(x, shifts = -s, dims = 3L)),
    dim = 4L
  )
  expect_equal(as.integer(rolled$size()), c(1L, 1L, 6L, 3L))
  # Column 1 carries columns 2, 3 and 5 -- its own value is never in its
  # own group, because the shifts are 2^i and none of them is zero.
  expect_equal(as.numeric(as.array(rolled[1, 1, 1, ])), c(2, 3, 5))
  # And the last column wraps around.
  expect_equal(as.numeric(as.array(rolled[1, 1, 6, ])), c(1, 2, 4))

  # The embedded tensor keeps one token per column, unlike v2's packing,
  # which divided the count by the group size.
  emb <- net$embed_cells(x, torch::torch_zeros_like(x))
  expect_equal(as.integer(emb$size()), c(1L, 1L, 6L, 16L))
})


test_that("fewer columns than the grouping shifts still runs", {
  # With 2 columns the shifts 1, 2 and 4 all wrap onto the same two
  # columns. That is degenerate but legal, and it must not crash.
  net <- tiny_net("classifier")
  set.seed(4)
  x_tr <- torch::torch_randn(c(1L, 10L, 2L))
  x_te <- torch::torch_randn(c(1L, 3L, 2L))
  y_tr <- torch::torch_randint(0L, 2L, c(1L, 10L))$to(dtype = torch::torch_float())
  out <- torch::with_no_grad(net(x_tr, y_tr, x_te))
  expect_true(as.logical(torch::torch_isfinite(out$logits)$all()))
})


test_that("NaN, Inf and constant columns are absorbed, not propagated", {
  net <- tiny_net("classifier")
  set.seed(5)
  x_tr <- torch::torch_randn(c(1L, 24L, 4L))
  x_te <- torch::torch_randn(c(1L, 6L, 4L))
  x_tr[1, 1, 1] <- NaN
  x_tr[1, 2, 2] <- Inf
  x_te[1, 1, 3] <- -Inf
  x_tr[1, , 4] <- 1.5          # constant column: the scaler's std is zero
  x_te[1, , 4] <- 1.5
  y_tr <- torch::torch_randint(0L, 2L, c(1L, 24L))$to(dtype = torch::torch_float())

  out <- torch::with_no_grad(net(x_tr, y_tr, x_te))
  expect_true(as.logical(torch::torch_isfinite(out$logits)$all()))

  # Unlike v2.6, the constant column is not dropped -- v3 has no
  # constant-column removal, it just leaves the scaled column at zero.
  pre <- net$preprocess(
    torch::torch_cat(list(x_tr, x_te), dim = 2L)$transpose(1L, 2L)$contiguous(),
    n_train = 24L
  )
  expect_true(as.logical((pre$x_BRiC[1, , 4] == 0)$all()))
  # The indicators flag what imputation then hid.
  expect_equal(as.numeric(as.array(pre$indicators[1, 1, 1])), TABPFN3_NAN_INDICATOR)
  expect_equal(as.numeric(as.array(pre$indicators[1, 2, 2])), TABPFN3_POS_INF_INDICATOR)
  expect_equal(as.numeric(as.array(pre$indicators[1, 25, 3])), TABPFN3_NEG_INF_INDICATOR)
})


test_that("out-of-range labels are refused rather than silently truncated", {
  net <- tiny_net("classifier")
  x_tr <- torch::torch_randn(c(1L, 12L, 3L))
  x_te <- torch::torch_randn(c(1L, 3L, 3L))
  # The head is 6 wide; a label of 7 has no slot to be decoded into.
  y_bad <- torch::torch_full(c(1L, 12L), 7)$to(dtype = torch::torch_float())
  expect_error(net(x_tr, y_bad, x_te), "out of range")
})


test_that("test rows cannot see each other", {
  # The ICL stage restricts keys and values to the training rows, so a
  # test row's prediction must not move when its neighbours change. This
  # is the property a mis-set `single_eval_pos` would quietly break.
  net <- tiny_net("classifier")
  set.seed(6)
  x_tr <- torch::torch_randn(c(1L, 16L, 4L))
  x_te <- torch::torch_randn(c(1L, 5L, 4L))
  y_tr <- torch::torch_randint(0L, 3L, c(1L, 16L))$to(dtype = torch::torch_float())

  base <- torch::with_no_grad(net(x_tr, y_tr, x_te))$logits
  alone <- torch::with_no_grad(net(x_tr, y_tr, x_te[, 1, , drop = FALSE]))$logits
  expect_equal(as.array(alone[1, 1, ]), as.array(base[1, 1, ]), tolerance = 1e-5)
})


test_that("the KV cache reproduces the forward it was built from", {
  net <- tiny_net("classifier")
  set.seed(7)
  x_tr <- torch::torch_randn(c(1L, 30L, 5L))
  x_te <- torch::torch_randn(c(1L, 9L, 5L))
  y_tr <- torch::torch_randint(0L, 3L, c(1L, 30L))$to(dtype = torch::torch_float())

  plain <- torch::with_no_grad(net(x_tr, y_tr, x_te))$logits

  no_rows <- torch::torch_zeros(c(1L, 0L, 5L))
  built <- torch::with_no_grad(net(x_tr, y_tr, no_rows, return_kv_cache = TRUE))
  cache <- built$kv_cache
  expect_s3_class(cache, "tabpfn3_kv_cache")
  expect_length(cache$kv, length(net$icl_blocks))
  expect_length(cache$inducing_hidden, length(net$feature_distribution_embedder$layers))
  expect_identical(cache$n_train, 30L)
  # Only the head a test row attends to is kept, whatever the head count.
  expect_identical(as.integer(cache$kv[[1]]$key$size(3)), 1L)
  # A build pass with no test rows decodes nothing.
  expect_null(built$logits)

  cached <- torch::with_no_grad(net(NULL, NULL, x_te, kv_cache = cache))$logits
  expect_equal(as.array(cached), as.array(plain), tolerance = 1e-5)

  # And the whole point: the answer must not depend on how the test rows
  # are batched, because that is what lets one cache serve every chunk.
  halves <- lapply(list(1:4, 5:9), function(ix) {
    torch::with_no_grad(net(NULL, NULL, x_te[, ix, , drop = FALSE],
                            kv_cache = cache))$logits
  })
  expect_equal(as.array(torch::torch_cat(halves, dim = 2L)),
               as.array(cached), tolerance = 1e-5)
})


test_that("save_peak_memory_factor does not change the answer", {
  net <- tiny_net("regressor")
  set.seed(8)
  x_tr <- torch::torch_randn(c(1L, 18L, 6L))
  x_te <- torch::torch_randn(c(1L, 5L, 6L))
  y_tr <- torch::torch_randn(c(1L, 18L))

  base <- as.array(torch::with_no_grad(net(x_tr, y_tr, x_te))$logits)
  for (k in c(2L, 3L, 8L)) {
    got <- as.array(torch::with_no_grad(
      net(x_tr, y_tr, x_te, save_peak_memory_factor = k))$logits)
    expect_equal(got, base, tolerance = 1e-5, info = paste("factor", k))
  }
})


test_that("the backend is registered and detects its own config", {
  bk <- get_backend("tabpfn3")
  expect_identical(bk$name, "tabpfn3")
  expect_true(bk$handles_missing)
  expect_true(bk$detect(list(arch = "tabpfn_v3")))
  expect_false(bk$detect(list(arch = "tabpfn_v2_6")))
  expect_identical(bk$task_of(list(head = "regressor")), "regression")
  # Exactly one backend may claim a v3 config, or `detect_backend()` has
  # nothing to dispatch on.
  expect_identical(detect_backend(list(arch = "tabpfn_v3"))$name, "tabpfn3")
})
