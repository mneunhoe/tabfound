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

  # `group_features()` produces exactly that, and with the indicator
  # channel concatenated it is (B, Ri, C, 2 * group_size) -- six floats
  # per column against the embedding's sixteen, which is why this is
  # where a row-chunked pass has to start.
  grouped <- net$group_features(x, torch::torch_zeros_like(x))
  expect_equal(as.integer(grouped$size()), c(1L, 1L, 6L, 6L))
  expect_equal(as.numeric(as.array(grouped[1, 1, 1, 1:3])), c(2, 3, 5))

  # The embedded tensor keeps one token per column, unlike v2's packing,
  # which divided the count by the group size.
  emb <- net$embed_cells(grouped)
  expect_equal(as.integer(emb$size()), c(1L, 1L, 6L, 16L))

  # Embedding a row slice of the grouped tensor is the same as slicing
  # the embedded one: the property the whole row-chunked path rests on.
  expect_equal(
    as.array(net$embed_cells(grouped[, 1, , , drop = FALSE])),
    as.array(emb[, 1, , , drop = FALSE])
  )
})


test_that("the inducing summaries survive being computed in column chunks", {
  # Stage 1's inducing summaries are the one thing a row-chunked pass
  # cannot produce for itself, so they are precomputed over the training
  # rows -- in column chunks, because every column is embedded on its own.
  # This checks that chunking the columns does not change what comes back,
  # and in particular that the chunks concatenate in the order the block
  # folds its columns into the batch: a permutation here would silently
  # give every column another column's distribution.
  net <- tiny_net("classifier")
  set.seed(11)
  n_tr <- 24L; n_te <- 5L; p <- 7L
  x_tr <- torch::torch_randn(c(1L, n_tr, p))
  x_te <- torch::torch_randn(c(1L, n_te, p))
  y_tr <- torch::torch_randint(0L, 4L, c(1L, n_tr))$to(dtype = torch::torch_float())

  # What the unchunked forward stores in its cache is the reference.
  ref <- torch::with_no_grad(
    net(x_tr, y_tr, x_te, return_kv_cache = TRUE))$kv_cache$inducing_hidden

  x_RiBC <- torch::torch_cat(list(x_tr, x_te), dim = 2L)$
    transpose(1L, 2L)$contiguous()
  y_BN    <- net$prepare_targets(y_tr, n_tr)
  pre     <- net$preprocess(x_RiBC, n_train = n_tr)
  grouped <- net$group_features(pre$x_BRiC, pre$indicators)
  y_col   <- net$embed_col_targets(y_BN)

  # 1 and 3 are ragged against 7 columns, 4 is the reference's default,
  # 7 and NULL are the no-chunking cases.
  for (cc in list(1L, 3L, 4L, 7L, NULL)) {
    got <- torch::with_no_grad(
      net$all_inducing_hidden(grouped, y_col, n_tr, cc))
    expect_length(got, length(ref))
    for (i in seq_along(ref)) {
      expect_equal(as.integer(got[[i]]$size()), as.integer(ref[[i]]$size()))
      # Not bit-exact even unchunked: this path carries the training rows
      # alone, where the block's own forward carries train and test
      # together, so the attention kernel sees a different batch size.
      # That is float32 reduction order, not different arithmetic.
      scale <- max(abs(as.array(ref[[i]])))
      expect_lt(max(abs(as.array(got[[i]]) - as.array(ref[[i]]))),
                1e-4 * scale)
    }
  }
})


test_that("stage-0-2 row chunking is one iteration away from the plain pass", {
  # The property the whole driver rests on: a chunk size that cannot bite
  # must produce the plain forward *bit for bit*, not merely close to it.
  # This is what catches a loop that copies, reorders or re-fits
  # something even when it has nothing to divide.
  net <- tiny_net("classifier")
  set.seed(21)
  n_tr <- 40L; n_te <- 9L; p <- 6L
  x_tr <- torch::torch_randn(c(1L, n_tr, p))
  x_te <- torch::torch_randn(c(1L, n_te, p))
  y_tr <- torch::torch_randint(0L, 4L, c(1L, n_tr))$to(dtype = torch::torch_float())

  plain <- as.array(torch::with_no_grad(net(x_tr, y_tr, x_te))$logits)
  for (rc in c(n_tr + n_te, n_tr + n_te + 1L, 10000L)) {
    once <- as.array(torch::with_no_grad(
      net(x_tr, y_tr, x_te, row_chunk_size = rc, col_chunk_size = 4L))$logits)
    expect_identical(once, plain)
  }
})


test_that("stage-0-2 row chunking agrees with the plain pass at every size", {
  net <- tiny_net("classifier")
  set.seed(22)
  n_tr <- 40L; n_te <- 9L; p <- 6L
  x_tr <- torch::torch_randn(c(1L, n_tr, p))
  x_te <- torch::torch_randn(c(1L, n_te, p))
  y_tr <- torch::torch_randint(0L, 4L, c(1L, n_tr))$to(dtype = torch::torch_float())

  plain <- as.array(torch::with_no_grad(net(x_tr, y_tr, x_te))$logits)
  scale <- max(abs(plain))

  # 16 and 32 put the train/test boundary at row 40 inside a chunk, which
  # is the only place a chunk-relative training-row count can go wrong:
  # get it wrong and test rows are handed a target embedding, or training
  # rows are not.
  for (rc in c(1L, 7L, 16L, 32L, 40L, 48L)) {
    for (cc in c(1L, 4L, 6L)) {
      got <- as.array(torch::with_no_grad(net(
        x_tr, y_tr, x_te, row_chunk_size = rc, col_chunk_size = cc))$logits)
      expect_equal(dim(got), dim(plain))
      expect_lt(max(abs(got - plain)), 1e-4 * scale)
    }
  }
})


test_that("row chunking a cached prediction is exact, and caches build chunked", {
  net <- tiny_net("classifier")
  set.seed(23)
  n_tr <- 40L; n_te <- 12L; p <- 6L
  x_tr <- torch::torch_randn(c(1L, n_tr, p))
  x_te <- torch::torch_randn(c(1L, n_te, p))
  y_tr <- torch::torch_randint(0L, 4L, c(1L, n_tr))$to(dtype = torch::torch_float())
  no_rows <- torch::torch_zeros(c(1L, 0L, p))

  cache <- torch::with_no_grad(
    net(x_tr, y_tr, no_rows, return_kv_cache = TRUE))$kv_cache
  base <- as.array(torch::with_no_grad(
    net(NULL, NULL, x_te, kv_cache = cache))$logits)

  # Against a cache, a test row's path through stages 0-2 depends on
  # nothing but itself and the cached summaries -- so chunking the rows
  # computes the same function of each row, but not bit for bit: both
  # stages fold the rows into the attention batch, and a batch of five
  # sums its reductions in a different order than a batch of twelve.
  # Float32's last bits, not a different answer.
  for (rc in c(1L, 5L, 12L)) {
    got <- as.array(torch::with_no_grad(
      net(NULL, NULL, x_te, kv_cache = cache, row_chunk_size = rc))$logits)
    expect_lt(max(abs(got - base)), 1e-5 * max(abs(base)))
  }

  # A cache built on the chunked path carries the precomputed summaries
  # rather than the ones a block would have produced itself. Same shapes,
  # same answer to float32 noise.
  chunked_cache <- torch::with_no_grad(net(
    x_tr, y_tr, no_rows, return_kv_cache = TRUE,
    row_chunk_size = 16L, col_chunk_size = 4L))$kv_cache
  expect_length(chunked_cache$inducing_hidden, length(cache$inducing_hidden))
  from_chunked <- as.array(torch::with_no_grad(
    net(NULL, NULL, x_te, kv_cache = chunked_cache))$logits)
  expect_lt(max(abs(from_chunked - base)), 1e-4 * max(abs(base)))
})


test_that("stage chunking defaults to the checkpoint's own sizes", {
  # The reference turns this on by default on v3 and nowhere else, so a
  # caller who says nothing has to get 2048/4 rather than the unchunked
  # pass -- and a `config.json` written before the converter emitted the
  # keys has to land on the same numbers rather than on nothing.
  net <- tiny_net("classifier")
  expect_identical(net$inference_row_chunk_size, 2048L)
  expect_identical(net$inference_col_chunk_size, 4L)
  expect_true(net$supports_stage_chunking)

  cfg <- tiny_config("classifier")
  cfg$inference_row_chunk_size <- 512L
  cfg$inference_col_chunk_size <- 2L
  from_cfg <- tabpfn_v3_transformer(cfg)
  expect_identical(from_cfg$inference_row_chunk_size, 512L)
  expect_identical(from_cfg$inference_col_chunk_size, 2L)

  # The three states of the argument, as `forward()` resolves them.
  expect_identical(.tabpfn3_chunk_arg(NA_integer_, 2048L), 2048L)
  expect_null(.tabpfn3_chunk_arg(NULL, 2048L))
  expect_identical(.tabpfn3_chunk_arg(64, 2048L), 64L)

  # And end to end: the default must be the same path as saying 512/2 out
  # loud, and a different one from saying NULL.
  set.seed(41)
  x_tr <- torch::torch_randn(c(1L, 60L, 5L))
  x_te <- torch::torch_randn(c(1L, 8L, 5L))
  y_tr <- torch::torch_randint(0L, 3L, c(1L, 60L))$to(dtype = torch::torch_float())
  cfg$inference_row_chunk_size <- 16L
  net16 <- tabpfn_v3_transformer(cfg)
  net16$eval()
  default  <- as.array(torch::with_no_grad(net16(x_tr, y_tr, x_te))$logits)
  explicit <- as.array(torch::with_no_grad(
    net16(x_tr, y_tr, x_te, row_chunk_size = 16L, col_chunk_size = 2L))$logits)
  off <- as.array(torch::with_no_grad(
    net16(x_tr, y_tr, x_te, row_chunk_size = NULL))$logits)
  expect_identical(default, explicit)
  expect_false(isTRUE(all.equal(default, off, tolerance = 0)))
})


test_that("the between-layer collect is gated on size and changes nothing", {
  small <- torch::torch_zeros(c(1L, 8L, 8L))          # 256 B
  large <- torch::torch_zeros(c(1L, 4096L, 512L))     # 8 MB

  withr::with_options(list(tabfound.collect_between_layers = "auto"), {
    expect_false(collect_between_layers(small))
    expect_true(collect_between_layers(large))
    # Nothing to measure means no, rather than a collection on every call
    # from a caller that cannot say how big its state is.
    expect_false(collect_between_layers(NULL))
  })
  withr::with_options(list(tabfound.collect_between_layers = FALSE), {
    expect_false(collect_between_layers(large))
  })
  withr::with_options(list(tabfound.collect_between_layers = TRUE), {
    expect_true(collect_between_layers(small))
    expect_true(collect_between_layers(NULL))
  })
  withr::with_options(
    list(tabfound.collect_between_layers = "auto",
         tabfound.collect_min_bytes = 128), {
    expect_true(collect_between_layers(small))
  })

  # It reclaims memory; it must not touch the arithmetic. Forced on and
  # forced off, on a table far below the threshold so "auto" would not
  # fire either way.
  net <- tiny_net("classifier")
  set.seed(31)
  x_tr <- torch::torch_randn(c(1L, 30L, 5L))
  x_te <- torch::torch_randn(c(1L, 8L, 5L))
  y_tr <- torch::torch_randint(0L, 3L, c(1L, 30L))$to(dtype = torch::torch_float())
  on <- withr::with_options(list(tabfound.collect_between_layers = TRUE),
    as.array(torch::with_no_grad(net(x_tr, y_tr, x_te))$logits))
  off <- withr::with_options(list(tabfound.collect_between_layers = FALSE),
    as.array(torch::with_no_grad(net(x_tr, y_tr, x_te))$logits))
  expect_identical(on, off)
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


test_that("backends without stage chunking refuse the knobs by name", {
  # A silently-dropped memory knob is the failure mode this guard exists
  # for: the caller set it to survive a table that will now kill them.
  ctx26 <- list(backend = list(name = "tabpfn26"),
                net = list(supports_kv_cache = TRUE,
                           supports_chunked_eval = TRUE,
                           supports_stage_chunking = FALSE))
  ctx3 <- list(backend = list(name = "tabpfn3"),
               net = list(supports_kv_cache = TRUE,
                          supports_chunked_eval = TRUE,
                          supports_stage_chunking = TRUE))

  expect_error(
    .require_kv_cache_support(ctx26, FALSE, NULL, row_chunk_size = 2048L),
    "row_chunk_size"
  )
  expect_error(
    .require_kv_cache_support(ctx26, FALSE, NULL, col_chunk_size = 4L),
    "col_chunk_size"
  )
  # The headline names only what was actually asked for. (The hint below
  # it mentions both, on purpose -- it is explaining which knob needs
  # which generation.)
  headline <- function(expr) {
    strsplit(tryCatch(expr, error = conditionMessage), "\n")[[1]][1]
  }
  expect_match(
    headline(.require_kv_cache_support(ctx26, FALSE, NULL, row_chunk_size = 2048L)),
    "row_chunk_size"
  )
  expect_false(grepl(
    "col_chunk_size",
    headline(.require_kv_cache_support(ctx26, FALSE, NULL, row_chunk_size = 2048L))
  ))
  # `NA` is the default, not a request, so it must pass on a backend that
  # has no stage chunking at all -- otherwise nothing but v3 would build.
  expect_true(.require_kv_cache_support(ctx26, FALSE, NULL))
  expect_true(.require_kv_cache_support(ctx26, FALSE, NULL,
                                        NA_integer_, NA_integer_))
  # And NULL means "off", which every backend can honour.
  expect_true(.require_kv_cache_support(ctx26, FALSE, NULL, NULL, NULL))
  expect_true(.require_kv_cache_support(ctx3, FALSE, NULL, 2048L, 4L))
})
