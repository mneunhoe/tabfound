# The Kumo Tabular network, with no checkpoint.
#
# Two halves. The first checks each block, and two tiny whole networks,
# against `sdm`'s own classes with the same random weights and inputs:
#
#   .venvs/sdm/bin/python inst/parity/kumo_layers_reference.py \
#       --out inst/parity/layers/kumo_layers.safetensors
#
# The tiny classifier is shaped like the medium model -- a `row_project`
# and grouped-query attention for the query rows -- which is the only
# parity that path gets without medium's weights. The second half checks
# structure: that the module tree is the checkpoint's key list for each
# size, and the properties the forward pass must have whatever the
# weights are.

skip_if_not_installed("torch")

ref_path <- tabfound_file("parity", "layers", "kumo_layers.safetensors.gz")

kumo_ref <- function() {
  skip_if(!nzchar(ref_path), "Kumo layer reference not found")
  skip_if_not_installed("safetensors")
  list(t = read_reference_tensors(ref_path),
       meta = jsonlite::fromJSON(sub("\\.safetensors\\.gz$", ".json", ref_path),
                                 simplifyVector = FALSE))
}

# Copy `<prefix>.<path>` tensors into the module by name; every parameter
# must be filled, and every tensor under the prefix that names one used.
load_kumo_ref <- function(module, ref, prefix) {
  keys <- grep(paste0("^", prefix, "\\."), names(ref), value = TRUE)
  w <- stats::setNames(ref[keys], sub(paste0("^", prefix, "\\."), "", keys))
  w <- w[names(w) %in% names(module$parameters)]
  expect_setequal(names(w), names(module$parameters))
  for (k in names(w)) module$parameters[[k]]$set_data(w[[k]])
  invisible(module)
}

# Float32 through a few matmuls and transcendental functions, against a
# different libtorch build: agreement to a few ulps of the output scale.
expect_kumo_close <- function(r, py, tol = 2e-6, label = "") {
  rv <- as.numeric(as.array(r$cpu())); pv <- as.numeric(as.array(py$cpu()))
  expect_identical(dim(as.array(r$cpu())), dim(as.array(py$cpu())), label = label)
  scale <- max(1, max(abs(pv)))
  expect_lt(max(abs(rv - pv)) / scale, tol, label = label)
}


# --- Blocks against sdm ----------------------------------------------------

test_that("kumo_rms_norm matches sdm's RMSNorm (eps = None)", {
  R <- kumo_ref(); ref <- R$t
  m <- load_kumo_ref(kumo_rms_norm(R$meta$kumo_rmsnorm_dim), ref, "kumo_rmsnorm")
  expect_kumo_close(m(ref$kumo_rmsnorm.x), ref$kumo_rmsnorm.y)
})

test_that("kumo_rope rotates split-half along the axis before the heads", {
  R <- kumo_ref(); ref <- R$t
  m <- load_kumo_ref(kumo_rope(R$meta$kumo_rope_dim), ref, "kumo_rope")
  expect_kumo_close(m(ref$kumo_rope.x), ref$kumo_rope.y)
})

test_that("log scaling, plain and gated, matches sdm", {
  R <- kumo_ref(); ref <- R$t; mt <- R$meta
  ls <- load_kumo_ref(kumo_log_scale(mt$kumo_scale_heads), ref, "kumo_logscale")
  expect_kumo_close(ls(ref$kumo_logscale.x, mt$kumo_scale_keys),
                    ref$kumo_logscale.y)
  gls <- load_kumo_ref(
    kumo_gated_log_scale(R$meta$kumo_rope_dim, mt$kumo_scale_heads,
                         mt$kumo_scale_hidden),
    ref, "kumo_gatedlogscale")
  expect_kumo_close(gls(ref$kumo_gatedlogscale.x, mt$kumo_scale_keys),
                    ref$kumo_gatedlogscale.y)
})

test_that("the transformer block matches sdm, with and without GQA", {
  R <- kumo_ref(); ref <- R$t; mt <- R$meta
  C <- mt$kumo_block_dim; H <- mt$kumo_block_heads
  b <- load_kumo_ref(kumo_block(C, H, query_scaling = kumo_log_scale(H)),
                     ref, "kumo_block")
  torch::with_no_grad({
    expect_kumo_close(b(ref$kumo_block.q, ref$kumo_block.kv), ref$kumo_block.y)
    kv <- b$attn$project_kv(b$key_value_norm(ref$kumo_block.kv))
    kv2 <- lapply(kv, function(t) t$narrow(t$dim() - 1L, 1L, 2L)$contiguous())
    expect_kumo_close(b(ref$kumo_block.q, kv2), ref$kumo_block.y_gqa,
                      label = "query rows on 2 of 4 KV heads")
  })

  rb <- load_kumo_ref(
    kumo_block(C, H, query_scaling = kumo_gated_log_scale(C %/% H, H, 16L),
               rope = TRUE),
    ref, "kumo_rowblock")
  x <- ref$kumo_rowblock.x
  torch::with_no_grad({
    expect_kumo_close(rb(x, x), ref$kumo_rowblock.y)
    expect_kumo_close(rb(x$narrow(2L, 1L, 2L), x), ref$kumo_rowblock.y_last,
                      label = "readout-only last layer")
  })
})

test_that("the induced block matches sdm", {
  R <- kumo_ref(); ref <- R$t; mt <- R$meta
  m <- load_kumo_ref(
    kumo_induced_block(mt$kumo_block_dim, mt$kumo_block_heads,
                       mt$kumo_induced_inducing),
    ref, "kumo_induced")
  x <- ref$kumo_induced.x
  torch::with_no_grad(
    expect_kumo_close(m(x, x$narrow(2L, 1L, mt$kumo_induced_train)),
                      ref$kumo_induced.y)
  )
})

test_that("the cell embedding matches sdm, NaN and categoricals included", {
  R <- kumo_ref(); ref <- R$t; mt <- R$meta
  m <- load_kumo_ref(
    kumo_cell_embedding(mt$kumo_cells_dim, mt$kumo_cells_group,
                        mt$kumo_cells_freq),
    ref, "kumo_cells")
  # Five columns with an all-missing one, and two columns, where the
  # circular grouping wraps onto the cell itself.
  for (tag in c("kumo_cells", "kumo_cells_narrow")) {
    mask <- ref[[paste0(tag, ".mask")]]$to(dtype = torch::torch_bool())
    got <- torch::with_no_grad(
      m(ref[[paste0(tag, ".x")]], mask, train_size = mt$kumo_cells_train)
    )
    expect_kumo_close(got, ref[[paste0(tag, ".y")]], label = tag)
  }
})

test_that("two tiny whole networks match sdm (GQA + row_project; all heads)", {
  R <- kumo_ref(); ref <- R$t
  for (tag in c("clf", "reg")) {
    cfg <- R$meta[[paste0("kumo_net_", tag)]]
    net <- kumo_model(cfg)
    load_kumo_ref(net, ref, paste0("kumo_net_", tag))
    net$eval()
    p <- paste0("kumo_net_", tag, ".")
    y <- ref[[paste0(p, "y")]]
    if (tag == "clf") y <- y$to(dtype = torch::torch_long())
    mask <- ref[[paste0(p, "mask")]]$to(dtype = torch::torch_bool())
    got <- torch::with_no_grad(net(ref[[paste0(p, "x")]], y, mask))
    expect_kumo_close(got, ref[[paste0(p, "out")]], tol = 1e-5, label = tag)
  }
  expect_false(is.null(kumo_model(R$meta$kumo_net_clf)$icl_block$kv_heads))
  expect_null(kumo_model(R$meta$kumo_net_reg)$icl_block$kv_heads)
})


# --- Structure ---------------------------------------------------------------

# The sizes, as `inst/python/kumo_convert_ckpt.py` writes them.
kumo_size_config <- function(size, head = "classifier") {
  base <- list(arch = "kumo_tabular", head = head, size = size,
               group_size = 3L, num_frequencies = 32L, num_readout_tokens = 4L,
               gate_hidden = 64L,
               num_classes = if (head == "classifier") 10L else 0L,
               num_quantiles = if (head == "classifier") 0L else 999L)
  dims <- switch(size,
    small  = list(cell_channels = 128L, num_embedding_layers = 4L,
                  num_embedding_heads = 4L, num_inducing_points = 128L,
                  icl_channels = 512L, num_icl_layers = 12L, num_icl_heads = 8L,
                  num_icl_key_value_heads_for_query = NULL),
    medium = list(cell_channels = 256L, num_embedding_layers = 6L,
                  num_embedding_heads = 4L, num_inducing_points = 256L,
                  icl_channels = 512L, num_icl_layers = 24L, num_icl_heads = 8L,
                  num_icl_key_value_heads_for_query = 2L),
    large  = list(cell_channels = 256L, num_embedding_layers = 6L,
                  num_embedding_heads = 4L, num_inducing_points = 256L,
                  icl_channels = 1024L, num_icl_layers = 24L, num_icl_heads = 16L,
                  num_icl_key_value_heads_for_query = 2L))
  c(base, dims)
}

# The checkpoint's key list for a config, derived independently of the R
# module tree: the same enumeration the converter verifies files against.
kumo_expected_keys <- function(cfg) {
  blk <- function(p, scaled = TRUE, gated = FALSE, rope = FALSE) {
    c(paste0(p, c("query_norm.weight", "key_value_norm.weight",
                  "attn.qkv_lin.weight", "attn.qkv_lin.bias",
                  "attn.out_lin.weight", "attn.out_lin.bias")),
      if (scaled) paste0(p, "attn.sdpa.query_scaling.head_scale"),
      if (gated) paste0(p, "attn.sdpa.query_scaling.gate.",
                        c("0.weight", "0.bias", "2.weight", "2.bias")),
      if (rope) paste0(p, c("attn.query_transform.0.inv_freq",
                            "attn.key_transform.0.inv_freq")),
      paste0(p, c("mlp.0.weight", "mlp.1.weight", "mlp.1.bias",
                  "mlp.3.weight", "mlp.3.bias")))
  }
  clf <- cfg$num_classes > 0
  ce <- "row_embedding.cell_embedding."
  keys <- c(paste0(ce, c("num_freq", "cat_freq", "num_lin.weight", "num_lin.bias",
                         "cat_lin.weight", "cat_lin.bias", "nan_lin.weight")),
            if (clf) c("row_embedding.y_emb.weight", "icl_block.y_emb.weight")
            else c("row_embedding.y_lin.weight", "icl_block.y_lin.weight"),
            "row_embedding.readout_token", "row_embedding.norm.weight")
  for (i in seq_len(cfg$num_embedding_layers) - 1L) {
    cb <- sprintf("row_embedding.col_blocks.%d.", i)
    keys <- c(keys, paste0(cb, "inducing_points"),
              blk(paste0(cb, "inducing_block.")),
              blk(paste0(cb, "output_block."), scaled = FALSE),
              blk(sprintf("row_embedding.row_blocks.%d.", i), gated = TRUE,
                  rope = TRUE))
  }
  if (cfg$num_readout_tokens * cfg$cell_channels != cfg$icl_channels) {
    keys <- c(keys, "row_project.weight", "row_project.bias")
  }
  for (j in seq_len(cfg$num_icl_layers) - 1L) {
    keys <- c(keys, blk(sprintf("icl_block.layers.%d.", j)))
  }
  c(keys, "icl_block.norm.weight",
    paste0("icl_block.head.", c("0.weight", "0.bias", "2.weight", "2.bias")))
}

test_that("the module tree is the checkpoint key list for every size", {
  # 328 / 558 / 556 keys: the counts the real small checkpoints have and
  # the untrained medium/large models give.
  counts <- c(small = 328L, medium = 558L, large = 556L)
  for (size in names(counts)) for (head in c("classifier", "regressor")) {
    cfg <- kumo_size_config(size, head)
    keys <- kumo_expected_keys(cfg)
    expect_length(keys, counts[[size]])
    # Building large in full allocates ~0.9 GB, so check the tree on a
    # meta device-free shortcut: the names alone, from a model whose
    # widths are shrunk but whose depths and switches are the real ones.
    small_widths <- utils::modifyList(cfg, list(
      cell_channels = cfg$cell_channels %/% 32L,
      num_inducing_points = 2L,
      icl_channels = cfg$icl_channels %/% 32L,
      num_frequencies = 4L, gate_hidden = 4L
    ))
    net <- kumo_model(small_widths)
    expect_setequal(names(net$parameters), keys)
  }
})

tiny_cfg <- function(head = "classifier", kv = NULL, E = 32L) {
  list(arch = "kumo_tabular", head = head, cell_channels = 8L,
       num_embedding_layers = 2L, num_embedding_heads = 2L,
       num_inducing_points = 3L, group_size = 3L, num_frequencies = 4L,
       num_readout_tokens = 4L, icl_channels = E, num_icl_layers = 2L,
       num_icl_heads = 4L, num_icl_key_value_heads_for_query = kv,
       gate_hidden = 8L,
       num_classes = if (head == "classifier") 10L else 0L,
       num_quantiles = if (head == "classifier") 0L else 7L)
}

randomized <- function(net, seed = 1L) {
  torch::torch_manual_seed(seed)
  ps <- net$parameters
  torch::with_no_grad(for (nm in names(ps)) {
    if (!grepl("inv_freq$", nm)) ps[[nm]]$normal_(std = 0.3)
  })
  net$eval()
  net
}

tiny_input <- function(R = 11L, C = 4L, Rt = 7L, clf = TRUE, B = 2L) {
  torch::torch_manual_seed(3L)
  x <- torch::torch_randn(B, R, C)
  y <- if (clf) torch::torch_randint(0L, 3L, c(B, Rt), dtype = torch::torch_long())
       else torch::torch_randn(B, Rt)
  list(x = x, y = y, mask = c(TRUE, rep(FALSE, C - 1L)))
}

test_that("forward shapes: 10 logits or the quantile grid, per member", {
  for (head in c("classifier", "regressor")) {
    net <- randomized(kumo_model(tiny_cfg(head)))
    i <- tiny_input(clf = head == "classifier")
    out <- torch::with_no_grad(net(i$x, i$y, i$mask))
    expect_identical(dim(out), c(2L, 4L, if (head == "classifier") 10L else 7L))
    expect_false(anyNA(as.array(out)))
  }
})

test_that("query rows do not see each other, and members are independent", {
  # Every query row attends only to context rows, so a query row's output
  # depends on that row alone; and the leading axis is a batch.
  for (kv in list(NULL, 2L)) {
    net <- randomized(kumo_model(tiny_cfg(kv = kv)))
    i <- tiny_input()
    full <- torch::with_no_grad(net(i$x, i$y, i$mask))
    x2 <- i$x$clone()
    x2[, 11, ] <- torch::torch_randn(2, 4)     # change the last query row
    alt <- torch::with_no_grad(net(x2, i$y, i$mask))
    expect_equal(as.array(alt[, 1:3, ]), as.array(full[, 1:3, ]), tolerance = 1e-6)
    expect_gt(max(abs(as.array(alt[, 4, ] - full[, 4, ]))), 1e-4)

    one <- torch::with_no_grad(net(i$x[1:1, , ], i$y[1:1, ], i$mask))
    expect_equal(as.array(one[1, , ]), as.array(full[1, , ]), tolerance = 1e-6)
  }
})

test_that("grouped-query attention reads the first KV heads, not all", {
  # With kv_heads = NULL every head is used; with 2 of 4 the query rows
  # must come out different, and the context rows' path is unchanged.
  a <- randomized(kumo_model(tiny_cfg(kv = NULL)))
  b <- kumo_model(tiny_cfg(kv = 2L))
  b$load_state_dict(a$state_dict())
  b$eval()
  i <- tiny_input()
  oa <- torch::with_no_grad(a(i$x, i$y, i$mask))
  ob <- torch::with_no_grad(b(i$x, i$y, i$mask))
  expect_gt(max(abs(as.array(oa - ob))), 1e-4)
  # kv_heads equal to all heads is plain attention.
  c4 <- kumo_model(tiny_cfg(kv = 4L)); c4$load_state_dict(a$state_dict()); c4$eval()
  expect_equal(as.array(torch::with_no_grad(c4(i$x, i$y, i$mask))), as.array(oa),
               tolerance = 1e-6)
})

test_that("missing values are handled inside the network", {
  net <- randomized(kumo_model(tiny_cfg()))
  i <- tiny_input()
  x <- i$x$clone()
  x[1, 2, 2] <- NaN          # context
  x[2, 9, 3] <- NaN          # query
  x[, , 4] <- NaN            # an all-missing column
  out <- torch::with_no_grad(net(x, i$y, i$mask))
  expect_false(anyNA(as.array(out)))
  # Infinite values are the caller's problem (the recipe clips them);
  # a constant column and a single column still run.
  x1 <- i$x[, , 1:1]
  expect_identical(dim(torch::with_no_grad(net(x1, i$y, TRUE))), c(2L, 4L, 10L))
  xc <- i$x$clone(); xc[, , 2] <- 5
  expect_false(anyNA(as.array(torch::with_no_grad(net(xc, i$y, i$mask)))))
})

test_that("an empty query returns no rows", {
  for (kv in list(NULL, 2L)) {
    net <- randomized(kumo_model(tiny_cfg(kv = kv)))
    i <- tiny_input(R = 7L)
    out <- torch::with_no_grad(net(i$x, i$y, i$mask))
    expect_identical(dim(out), c(2L, 0L, 10L))
  }
})

test_that("the cell embedding's grouping and sums are what the docs say", {
  # Closed form for one cell: slot g of column c reads column
  # (c + {0,1,3}[g]) mod C; [sin, cos] of value * slot frequency; one
  # projection over all three slots; the bias three times; plus the
  # missingness projection.
  m <- kumo_cell_embedding(4L, 3L, 2L)
  torch::torch_manual_seed(5L)
  torch::with_no_grad(for (p in m$parameters) p$normal_())
  x <- matrix(c(0.5, -1, 2, 0.25, NaN, 1.5, 3, -2), nrow = 2, byrow = TRUE)
  mask <- c(FALSE, TRUE, FALSE, FALSE)
  got <- as.array(torch::with_no_grad(
    m(torch::torch_tensor(x, dtype = torch::torch_float()),
      torch::torch_tensor(mask), train_size = 1L)))
  mu <- x[1, ]                                 # one context row
  xf <- ifelse(is.nan(x), matrix(mu, 2, 4, byrow = TRUE), x)
  nf <- as.matrix(m$num_freq); cf <- as.matrix(m$cat_freq)
  Wn <- as.matrix(m$num_lin$weight); bn <- as.numeric(m$num_lin$bias)
  Wc <- as.matrix(m$cat_lin$weight); bc <- as.numeric(m$cat_lin$bias)
  Wm <- as.matrix(m$nan_lin$weight)
  for (r in 1:2) for (c in 1:4) {
    acc <- numeric(4)
    for (g in 1:3) {
      cc <- (c - 1 + c(0, 1, 3)[g]) %% 4 + 1
      f <- if (mask[cc]) cf[g, ] else nf[g, ]
      W <- if (mask[cc]) Wc else Wn; b <- if (mask[cc]) bc else bn
      z <- xf[r, cc] * f
      acc <- acc + W %*% c(sin(z), cos(z)) + b + Wm[, g] * is.nan(x[r, cc])
    }
    expect_equal(got[r, c, ], as.numeric(acc), tolerance = 1e-5)
  }
})

test_that("Kumo registers, detects only its own config, and maps the task", {
  expect_true("kumo" %in% list_backends()$name)
  cfg <- kumo_size_config("small")
  expect_identical(detect_backend(cfg)$name, "kumo")
  expect_identical(kumo_task_of(cfg), "classification")
  expect_identical(kumo_task_of(kumo_size_config("small", "regressor")), "regression")
  # Without its marker a Kumo config is not TabFM's or Mitra's either.
  bare <- cfg; bare$arch <- NULL
  expect_false(tabfm_detect(bare))
  expect_false(mitra_detect(bare))
  spec <- get_backend("kumo")
  expect_true(spec$handles_missing)
  expect_false(spec$imputes_internally)
})


test_that("chunking that splits nothing is the plain forward, bit for bit", {
  for (kv in list(NULL, 2L)) {
    net <- randomized(kumo_model(tiny_cfg(kv = kv)))
    i <- tiny_input(R = 30L, Rt = 20L, C = 5L)
    run <- function(...) as.array(torch::with_no_grad(net(i$x, i$y, i$mask, ...)))
    plain <- run()
    expect_identical(run(save_peak_memory_factor = 1L), plain)
    expect_identical(run(row_chunk_size = 30L), plain)
    expect_identical(run(row_chunk_size = 1000L), plain)
    expect_identical(run(col_chunk_size = 5L), plain)
  }
})

test_that("chunking that splits moves the answer only by float32 rounding", {
  # Columns are the column blocks' batch axis, rows are independent in
  # every chunked step, and the inducing step and the log-n scaling always
  # see the whole context -- so a different answer here would be a bug,
  # not noise. What is left is the rounding of different batch shapes.
  for (kv in list(NULL, 2L)) {
    net <- randomized(kumo_model(tiny_cfg(kv = kv)))
    i <- tiny_input(R = 30L, Rt = 20L, C = 5L)
    run <- function(...) as.array(torch::with_no_grad(net(i$x, i$y, i$mask, ...)))
    plain <- run()
    for (args in list(list(save_peak_memory_factor = 3L),
                      list(row_chunk_size = 7L),
                      list(row_chunk_size = 1L),
                      list(col_chunk_size = 2L),
                      list(row_chunk_size = 4L, save_peak_memory_factor = 2L,
                           col_chunk_size = 2L))) {
      expect_equal(do.call(run, args), plain, tolerance = 1e-5,
                   label = paste(names(args), unlist(args), collapse = ", "))
    }
  }
})

test_that("the chunk-length rule takes the smaller of size and factor", {
  expect_identical(.kumo_chunk_len(10L, 4L, NULL), 4L)
  expect_identical(.kumo_chunk_len(10L, NULL, 3L), 4L)
  expect_identical(.kumo_chunk_len(10L, 6L, 5L), 2L)
  expect_true(is.na(.kumo_chunk_len(10L)))
  expect_true(is.na(.kumo_chunk_len(0L, 4L)))
})


test_that("every Kumo layer loop collects between layers", {
  # Kumo used to be the one backend whose layer loops never collected, so
  # a forward's intermediates stayed alive until torch's own threshold
  # fired mid-block (#99). Each loop is counted -- the row embedding's
  # column/row blocks, the ICL stack's plain and GQA branches, the
  # cache-building `record` loop and the cached loop -- so dropping any
  # one of the calls fails.
  calls <- 0L
  local_mocked_bindings(collect_between_layers = function(x = NULL,
                                                          only_full = FALSE) {
    calls <<- calls + 1L
    invisible(FALSE)
  })
  count <- function(net, i, ...) {
    calls <<- 0L
    torch::with_no_grad(net(i$x, i$y, i$mask, ...))
    calls
  }
  i <- tiny_input(R = 30L, Rt = 20L, C = 5L)
  for (kv in list(NULL, 2L)) {
    net <- randomized(kumo_model(tiny_cfg(kv = kv)))
    # Two row-embedding layers and two ICL layers.
    expect_identical(count(net, i), 4L)
    # The cache over the context (2 + 2 in `record`), then the ten query
    # rows against it (2 + 2).
    expect_identical(count(net, i, row_chunk_size = 10L), 8L)
  }
})

test_that("collecting between Kumo's layers does not move the answer", {
  i <- tiny_input(R = 30L, Rt = 20L, C = 5L)
  off  <- list(tabfound.collect_between_layers = FALSE)
  full <- list(tabfound.collect_between_layers = TRUE,
               tabfound.collect_full_min_bytes = 0)
  for (kv in list(NULL, 2L)) {
    net <- randomized(kumo_model(tiny_cfg(kv = kv)))
    run <- function(opts, ...) withr::with_options(opts,
      as.array(torch::with_no_grad(net(i$x, i$y, i$mask, ...))))
    expect_identical(run(full), run(off))
    expect_identical(run(full, row_chunk_size = 10L),
                     run(off, row_chunk_size = 10L))
  }
})
