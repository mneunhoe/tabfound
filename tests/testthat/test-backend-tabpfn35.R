# TabPFN v3.5 architecture, without the 334 MB of weights.
#
# A randomly-initialised model at a fraction of the width answers the
# questions that do not need real weights: does the module tree carry
# exactly the names the checkpoint does, does *one* tree serve both tasks,
# do the four stages hold their shapes, and is the in-context ECDF -- the
# one genuinely new computation in this generation -- right. Numerical
# agreement with the reference is a separate question, answered in
# `test-parity-tabpfn35.R`.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")

tiny35_config <- function() {
  # No `head`: that is the point of v3.5. Both heads are in every
  # checkpoint and the task is chosen when the network is built.
  list(
    arch = "tabpfn_v3_5", head = "multitask", tabpfn_version = "3.5",
    embed_dim = 16L, feature_group_size = 3L, use_nan_indicators = TRUE,
    fourier_encoding_num_frequencies = 5L,
    cell_ecdf_num_frequencies = 2L, cell_ecdf_num_buckets = 64L,
    cell_embed_row_chunk_size = 2048L,
    dist_embed_num_blocks = 2L, dist_embed_num_heads = 2L,
    dist_embed_num_inducing_points = 4L,
    feat_agg_num_blocks = 2L, feat_agg_num_heads = 2L,
    feat_agg_num_cls_tokens = 2L, feat_agg_rope_base = 100000,
    nlayers = 2L, icl_num_heads = 2L, icl_num_kv_heads_test = 1L,
    decoder_head_dim = 4L, decoder_num_heads = 2L,
    decoder_use_softmax_scaling = TRUE,
    ff_factor = 2L, softmax_scaling_mlp_hidden_dim = 8L,
    layernorm_elementwise_affine = TRUE,
    max_num_classes = 6L, num_buckets = 7L, n_bar_bins = 7L
  )
}

tiny35_net <- function(task = "classification") {
  net <- tabpfn_v35_transformer(tiny35_config(), task = task)
  net$eval()
  net
}

# A small train/test problem with the awkward inputs built in.
tiny35_data <- function(n_train = 24L, n_test = 5L, n_feat = 4L, seed = 11L) {
  torch::torch_manual_seed(seed)
  x <- torch::torch_randn(1L, n_train + n_test, n_feat)
  y <- torch::torch_tensor(
    matrix(sample(0:2, n_train, replace = TRUE), nrow = 1L)
  )$to(dtype = torch::torch_float())
  list(x_train = x[, 1:n_train, ], y_train = y,
       x_test = x[, (n_train + 1L):(n_train + n_test), ],
       n_train = n_train, n_test = n_test, n_feat = n_feat)
}


test_that("the module tree carries the checkpoint's own parameter names", {
  nms <- names(tiny35_net("classification")$state_dict())

  expect_true(all(c(
    # The Fourier cell embedder, where v3 had a plain linear.
    "x_embed.fourier.frequencies", "x_embed.fourier.in_linear.weight",
    "x_embed.metadata_linear.weight",
    "x_embed.layernorm.weight", "x_embed.layernorm.bias",
    # Both target encoders, plus their new LayerNorms.
    "col_y_encoder.multiclass.embedding.weight",
    "col_y_encoder.regression.weight", "col_y_encoder.regression.bias",
    "col_y_layernorm.weight", "col_y_layernorm.bias",
    "icl_y_encoder.multiclass.embedding.weight",
    "icl_y_encoder.regression.weight", "icl_y_encoder.regression.bias",
    "icl_y_layernorm.weight", "icl_y_layernorm.bias",
    # Stage 1 and 2, as in v3 but with QK-norm.
    "feature_distribution_embedder.layers.0.inducing_vectors",
    "feature_distribution_embedder.layers.0.cross_attn_block1.attn.q_norm.weight",
    "feature_distribution_embedder.layers.0.cross_attn_block1.attn.k_norm.weight",
    "feature_distribution_embedder.layers.0.cross_attn_block2.attn.q_norm.weight",
    "feature_distribution_embedder.layers.0.cross_attn_block1.attn.softmax_scaling_layer.base_mlp.0.weight",
    "column_aggregator.cls_tokens", "column_aggregator.rope.freqs",
    "column_aggregator.out_ln.weight",
    "column_aggregator.blocks.0.attention.q_norm.weight",
    "column_aggregator.blocks.0.attention.k_norm.weight",
    # Stage 3.
    "icl_blocks.0.icl_attention.q_norm.weight",
    "icl_blocks.0.icl_attention.k_norm.weight",
    "icl_blocks.1.layernorm_mlp.weight", "output_norm.weight",
    # The bundled heads: both of them, in every checkpoint.
    "heads.mlp_classification.norm.weight",
    "heads.mlp_classification.mlp.0.weight",
    "heads.mlp_regression.norm.weight",
    "heads.mlp_regression.mlp.2.weight",
    "heads.many_class_decoder.q_projection.weight",
    "heads.many_class_decoder.k_projection.bias",
    "heads.many_class_decoder.softmax_scaling_layer.base_mlp.0.weight",
    "heads.output_projection.weight", "heads.output_projection.bias",
    "heads.regression_borders"
  ) %in% nms))

  # v3's names are gone, not merely joined: the borders and the decoder
  # moved under `heads`, and the y encoders grew a branch each.
  expect_false(any(c(
    "x_embed.weight", "regression_borders", "output_projection.weight",
    "many_class_decoder.q_projection.weight",
    "col_y_encoder.embedding.weight"
  ) %in% nms))

  # Bias-free projections stay bias-free.
  expect_false(any(grepl("(q|k|v|out)_projection\\.bias$",
                         grep("^(icl_blocks|column_aggregator|feature_distribution)",
                              nms, value = TRUE))))
})


test_that("one checkpoint serves both tasks from the same tree", {
  clf <- tiny35_net("classification")
  reg <- tiny35_net("regression")

  # The module tree is identical -- this is what lets a single artifact
  # load under `tabular_classifier()` and `tabular_regressor()` alike.
  expect_identical(sort(names(clf$state_dict())), sort(names(reg$state_dict())))
  expect_identical(clf$task_type, "multiclass")
  expect_identical(reg$task_type, "regression")

  # What differs is which branch runs, and therefore the output width.
  d <- tiny35_data()
  torch::with_no_grad({
    lc <- clf(d$x_train, d$y_train, d$x_test,
              row_chunk_size = NULL, col_chunk_size = NULL)$logits
    yr <- torch::torch_randn(1L, d$n_train)
    lr <- reg(d$x_train, yr, d$x_test,
              row_chunk_size = NULL, col_chunk_size = NULL)$logits
  })
  expect_equal(dim(lc), c(1L, d$n_test, 6L))   # max_num_classes
  expect_equal(dim(lr), c(1L, d$n_test, 7L))   # num_buckets
  # The classifier's head is retrieval, so its logits are log-probabilities.
  expect_true(as.logical((lc <= 0)$all()$cpu()))
})


test_that("the forward pass holds its shapes through all four stages", {
  net <- tiny35_net("classification")
  d <- tiny35_data()
  torch::with_no_grad({
    out <- net(d$x_train, d$y_train, d$x_test,
               row_chunk_size = NULL, col_chunk_size = NULL)
  })
  expect_equal(dim(out$logits), c(1L, d$n_test, 6L))
  # 2 CLS tokens x 16 embedding = the ICL width.
  expect_equal(dim(out$test_hidden), c(1L, d$n_test, 32L))
  expect_equal(dim(out$train_hidden), c(1L, d$n_train, 32L))
  expect_true(as.logical(torch::torch_isfinite(out$logits)$all()$cpu()))
})


test_that("the cell embedder's widths follow from the group size", {
  cfg <- tiny35_config()
  emb <- tabpfn35_cell_embedder(
    group_size = cfg$feature_group_size, embed_dim = cfg$embed_dim,
    num_freq = cfg$fourier_encoding_num_frequencies,
    ecdf_num_frequencies = cfg$cell_ecdf_num_frequencies
  )
  g <- cfg$feature_group_size; k <- cfg$cell_ecdf_num_frequencies

  # Three channels per group position go in: value, indicator, rank.
  expect_identical(emb$input_width, g * 3L)
  # The ranks are lifted to 2K features each inside, so the metadata
  # linear is wider than the tensor the caller passes.
  expect_identical(emb$metadata_width, g * 2L + g * 2L * k)
  expect_equal(dim(emb$metadata_linear$weight), c(cfg$embed_dim, g * 2L + g * 2L * k))
  expect_equal(dim(emb$fourier$frequencies),
               c(g, cfg$fourier_encoding_num_frequencies))
  expect_equal(dim(emb$fourier$in_linear$weight),
               c(cfg$embed_dim, 2L * cfg$fourier_encoding_num_frequencies))

  x <- torch::torch_randn(1L, 6L, 4L, g * 3L)
  torch::with_no_grad(out <- emb(x))
  expect_equal(dim(out), c(1L, 6L, 4L, cfg$embed_dim))

  # Row chunking is exact: rows are independent here.
  emb$eval()
  torch::with_no_grad({
    whole <- emb(x)
    emb$row_chunk_size <- 2L
    chunked <- emb(x)
  })
  expect_lt(as.numeric((whole - chunked)$abs()$max()$cpu()), 1e-6)
})


test_that("the in-context ECDF reproduces the midrank it approximates", {
  # The bucketed context is only an approximation when a column has more
  # distinct values than there are buckets. Below that it must be exact,
  # ties included -- that is the whole claim the bucketing rests on.
  brute <- function(m, n_train) {
    apply(m, 2, function(col) {
      tr <- col[seq_len(n_train)]
      vapply(col, function(v) (sum(tr < v) + sum(tr <= v)) / 2 / length(tr),
             numeric(1))
    })
  }
  got <- function(m, n_train, buckets) {
    storage.mode(m) <- "double"
    x <- torch::torch_tensor(m)$unsqueeze(1L)
    ctx <- .tabpfn35_build_ecdf_context(x, n_train, buckets)
    as.matrix(.tabpfn35_in_context_ecdf(x, ctx)$squeeze(1L))
  }
  err <- function(m, n_train, buckets) {
    storage.mode(m) <- "double"
    max(abs(got(m, n_train, buckets) - brute(m, n_train)))
  }
  set.seed(3)

  # Exact: at most as many distinct values as buckets.
  expect_lt(err(matrix(sample(1:7, 60 * 4, TRUE), nrow = 60), 50L, 8192L), 1e-6)
  expect_lt(err(matrix(sample(c(0, 1), 60 * 4, TRUE), nrow = 60), 50L, 8192L), 1e-6)
  expect_lt(err(matrix(rnorm(60 * 4), nrow = 60), 50L, 8192L), 1e-6)
  # A constant column: every row ties with every other, so the midrank is
  # 0.5 everywhere and nothing is left to interpolate.
  expect_equal(unname(got(matrix(rep(2.5, 60 * 2), nrow = 60), 50L, 8192L)),
               matrix(0.5, 60, 2), tolerance = 1e-6)
  # Test rows outside the training range still rank correctly: 0 below the
  # smallest training value, 1 above the largest.
  m <- rbind(matrix(rnorm(50 * 3), nrow = 50), c(99, -99, 0))
  expect_lt(err(m, 50L, 8192L), 1e-6)
  # A single training row.
  expect_lt(err(matrix(rnorm(60 * 4), nrow = 60), 1L, 8192L), 1e-6)

  # Approximate, and bounded, once the buckets bite. `K = 2` keeps only
  # the two extremes, so this is the worst the interpolation can do.
  expect_gt(err(matrix(rnorm(60 * 4), nrow = 60), 50L, 2L), 1e-3)
  expect_lt(err(matrix(rnorm(60 * 4), nrow = 60), 50L, 2L), 0.5)
  # Every rank stays a probability whatever the bucket count.
  g2 <- got(matrix(rnorm(60 * 4), nrow = 60), 50L, 2L)
  expect_true(all(g2 >= 0 & g2 <= 1))
})


test_that("the ECDF's sin/cos lift keeps the ends apart", {
  # Half-period phases, so u = 0 and u = 1 stay distinguishable at every
  # frequency -- with full periods the first one would wrap them together.
  u <- torch::torch_tensor(c(0, 0.5, 1))
  f <- .tabpfn35_ecdf_fourier_features(u, 4L)
  expect_equal(dim(f), c(3L, 8L))
  expect_gt(as.numeric((f[1, ] - f[3, ])$abs()$max()$cpu()), 0.5)
})


test_that("feature grouping preserves the column count and the block order", {
  net <- tiny35_net("classification")
  B <- 1L; Ri <- 5L; C <- 4L; g <- net$feature_group_size
  x   <- torch::torch_randn(B, Ri, C)
  ind <- torch::torch_zeros(B, Ri, C)
  ecdf <- torch::torch_rand(B, Ri, C)
  torch::with_no_grad(grouped <- net$group_features(x, ind, ecdf))

  # One token per column, not per group of them -- v2's packing divided
  # the column count, v3 and v3.5 preserve it.
  expect_equal(dim(grouped), c(B, Ri, C, g * 3L))

  # The three blocks must stay in this order: the embedder slices the
  # values off the front and the ranks off the back.
  expect_lt(as.numeric(
    (grouped[, , , 1:g] - torch::torch_stack(
      lapply(0:(g - 1L), function(i) torch::torch_roll(x, -(2L^i), dims = 3L)),
      dim = 4L))$abs()$max()$cpu()), 1e-6)
  expect_lt(as.numeric(
    (grouped[, , , (2L * g + 1L):(3L * g)] - torch::torch_stack(
      lapply(0:(g - 1L), function(i) torch::torch_roll(ecdf, -(2L^i), dims = 3L)),
      dim = 4L))$abs()$max()$cpu()), 1e-6)

  # None of the shifts is zero, so a column does not carry its own value
  # -- except by wraparound, when a shift is a multiple of the column
  # count. At C = 4 the third shift is 4, so it wraps exactly onto itself;
  # that is the architecture's behaviour on a narrow table, not a bug, and
  # it is why the grouping is still well defined with fewer columns than
  # shifts.
  expect_false(any(2L^(0:(g - 1L)) == 0L))
  expect_true(4L %% C == 0L)
  expect_lt(as.numeric((grouped[, , , 3] - x)$abs()$max()$cpu()), 1e-6)
})


test_that("stage chunking agrees with the unchunked pass", {
  net <- tiny35_net("classification")
  d <- tiny35_data(n_train = 24L, n_test = 5L)
  torch::with_no_grad({
    base <- net(d$x_train, d$y_train, d$x_test,
                row_chunk_size = NULL, col_chunk_size = NULL)$logits
    # 8 and 16 put the train/test boundary inside a chunk; 29 and 32 are
    # one chunk, where the loop is skipped entirely.
    for (rc in c(1L, 7L, 8L, 16L, 29L, 32L)) {
      for (cc in c(1L, 2L, 4L)) {
        got <- net(d$x_train, d$y_train, d$x_test,
                   row_chunk_size = rc, col_chunk_size = cc)$logits
        expect_lt(as.numeric((got - base)$abs()$max()$cpu()), 1e-4)
      }
    }
  })
})


test_that("stage chunking defaults to the checkpoint's own sizes", {
  net <- tiny35_net("classification")
  expect_identical(net$inference_row_chunk_size, 2048L)
  expect_identical(net$inference_col_chunk_size, 4L)
  # Three states, because there are three things a caller can mean.
  expect_identical(.tabpfn35_chunk_arg(NA_integer_, 2048L), 2048L)
  expect_null(.tabpfn35_chunk_arg(NULL, 2048L))
  expect_identical(.tabpfn35_chunk_arg(64L, 2048L), 64L)
})


test_that("save_peak_memory_factor does not change the answer", {
  net <- tiny35_net("classification")
  d <- tiny35_data()
  torch::with_no_grad({
    base <- net(d$x_train, d$y_train, d$x_test,
                row_chunk_size = NULL, col_chunk_size = NULL)$logits
    for (k in c(2L, 3L, 8L)) {
      got <- net(d$x_train, d$y_train, d$x_test, save_peak_memory_factor = k,
                 row_chunk_size = NULL, col_chunk_size = NULL)$logits
      expect_lt(as.numeric((got - base)$abs()$max()$cpu()), 1e-4)
    }
  })
})


test_that("the KV cache reproduces the forward it was built from", {
  net <- tiny35_net("classification")
  d <- tiny35_data()
  empty <- torch::torch_zeros(c(1L, 0L, d$n_feat))
  torch::with_no_grad({
    base <- net(d$x_train, d$y_train, d$x_test,
                row_chunk_size = NULL, col_chunk_size = NULL)$logits
    built <- net(d$x_train, d$y_train, empty, return_kv_cache = TRUE,
                 row_chunk_size = NULL, col_chunk_size = NULL)
    cache <- built$kv_cache
    got <- net(d$x_train, d$y_train, d$x_test, kv_cache = cache,
               row_chunk_size = NULL, col_chunk_size = NULL)$logits
  })
  expect_s3_class(cache, "tabpfn35_kv_cache")
  expect_lt(as.numeric((got - base)$abs()$max()$cpu()), 1e-4)

  # The cache carries the ECDF context, which v3's did not have to: a test
  # cell is ranked against the training distribution, so without it a
  # cached prediction would rank against itself.
  expect_false(is.null(cache$ecdf_context))
  expect_equal(dim(cache$ecdf_context)[1], 3L)
  # Only the heads a test row can reach are kept.
  expect_equal(dim(cache$kv[[1]]$key)[3], 1L)
  expect_identical(cache$n_train, d$n_train)

  # A cache built under chunking is the same cache.
  torch::with_no_grad({
    built2 <- net(d$x_train, d$y_train, empty, return_kv_cache = TRUE,
                  row_chunk_size = 8L, col_chunk_size = 2L)
    got2 <- net(d$x_train, d$y_train, d$x_test, kv_cache = built2$kv_cache,
                row_chunk_size = NULL, col_chunk_size = NULL)$logits
  })
  expect_lt(as.numeric((got2 - base)$abs()$max()$cpu()), 1e-4)
})


test_that("test rows cannot see each other", {
  # The ICL stack restricts keys to the training rows by slicing, not by
  # masking, so this is the check that the slice is where it should be.
  net <- tiny35_net("classification")
  d <- tiny35_data(n_test = 6L)
  torch::with_no_grad({
    all_rows <- net(d$x_train, d$y_train, d$x_test,
                    row_chunk_size = NULL, col_chunk_size = NULL)$logits
    first_two <- net(d$x_train, d$y_train, d$x_test[, 1:2, ],
                     row_chunk_size = NULL, col_chunk_size = NULL)$logits
  })
  expect_lt(as.numeric((first_two - all_rows[, 1:2, ])$abs()$max()$cpu()), 1e-5)
})


test_that("NaN, Inf and constant columns are absorbed", {
  net <- tiny35_net("classification")
  d <- tiny35_data(n_feat = 4L)
  x_train <- d$x_train$clone()
  x_train[1, 2, 1] <- NaN
  x_train[1, 3, 2] <- Inf
  x_train[1, 4, 2] <- -Inf
  x_train[1, , 3] <- 5                      # constant column
  x_test <- d$x_test$clone()
  x_test[1, 1, 1] <- NaN
  torch::with_no_grad({
    out <- net(x_train, d$y_train, x_test,
               row_chunk_size = NULL, col_chunk_size = NULL)
  })
  expect_true(as.logical(torch::torch_isfinite(out$logits)$all()$cpu()))

  # The indicator channel carries the three sentinels the architecture
  # uses, and it is read off the *raw* input, before imputation.
  ind <- tabpfn35_nan_inf_indicator(x_train$transpose(1L, 2L))
  expect_equal(as.numeric(ind[2, 1, 1]$cpu()), TABPFN35_NAN_INDICATOR)
  expect_equal(as.numeric(ind[3, 1, 2]$cpu()), TABPFN35_POS_INF_INDICATOR)
  expect_equal(as.numeric(ind[4, 1, 2]$cpu()), TABPFN35_NEG_INF_INDICATOR)
})


test_that("out-of-range labels are refused", {
  net <- tiny35_net("classification")
  d <- tiny35_data()
  bad_hi <- d$y_train$clone(); bad_hi[1, 1] <- 99
  bad_lo <- d$y_train$clone(); bad_lo[1, 1] <- -1
  expect_error(net(d$x_train, bad_hi, d$x_test), "out of range")
  expect_error(net(d$x_train, bad_lo, d$x_test), "out of range")
})


test_that("the backend is registered and detects only its own config", {
  bk <- get_backend("tabpfn35")
  expect_identical(bk$name, "tabpfn35")
  expect_true(bk$handles_missing)
  expect_true(bk$kv_cache_capable)
  expect_true(bk$detect(list(arch = "tabpfn_v3_5")))
  expect_false(bk$detect(list(arch = "tabpfn_v3")))

  # The artifacts have no opinion about the task; that is what lets one
  # download serve `tabular_classifier()` and `tabular_regressor()` both.
  expect_null(bk$task_of(list(head = "multitask")))

  # Exactly one backend may claim a config, or `detect_backend()` has
  # nothing to dispatch on. v3.5 must not answer to a v3 config, and the
  # v3 backend must not answer to this one.
  expect_identical(detect_backend(list(arch = "tabpfn_v3_5"))$name, "tabpfn35")
  expect_identical(detect_backend(list(arch = "tabpfn_v3"))$name, "tabpfn3")
})


test_that("the checkpoint's nullable keys are read exactly, not by prefix", {
  # R's `$` on a list partial-matches. v3.5's checkpoints omit
  # `icl_num_kv_heads` entirely, and `icl_num_kv_heads_test` is a longer
  # key with the same prefix, so a `$` read would silently return 1 and
  # build every ICL block with a single KV head.
  cfg <- tiny35_config()
  expect_false("icl_num_kv_heads" %in% names(cfg))
  expect_identical(cfg$icl_num_kv_heads, 1L)          # the trap
  expect_null(cfg[["icl_num_kv_heads"]])              # what the backend reads

  net <- tiny35_net("classification")
  att <- net$icl_blocks[[1]]$icl_attention
  # Absent means "same as the query heads", not "one".
  expect_identical(att$num_kv_heads, 2L)
  expect_identical(att$num_kv_heads_test, 1L)
  expect_equal(dim(att$k_projection$weight)[1],
               att$num_kv_heads * att$head_dim)
})
