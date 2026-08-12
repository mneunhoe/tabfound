# The registry is what makes a second backend a drop-in addition rather
# than a fork of the API, so its contract gets its own tests.

test_that("the tabpfn backend registers itself on load", {
  bks <- list_backends()
  expect_true("tabpfn" %in% bks$name)
  expect_true(nzchar(bks$description[bks$name == "tabpfn"]))
})


test_that("get_backend errors helpfully on an unknown name", {
  expect_error(get_backend("nope"), "Unknown backend")
})


test_that("detection routes a config to exactly one backend", {
  clf_cfg <- list(arch = "per_feature_transformer", head = "classifier")
  reg_cfg <- list(arch = "per_feature_transformer", head = "regressor")
  expect_identical(detect_backend(clf_cfg)$name, "tabpfn")
  expect_identical(tabpfn_task_of(clf_cfg), "classification")
  expect_identical(tabpfn_task_of(reg_cfg), "regression")

  # A TabFM-style config must not be claimed by the TabPFN backend.
  expect_error(detect_backend(list(embed_dim = 256, is_classifier = TRUE)),
               "No registered backend")
})


test_that("a third-party backend can be registered and detected", {
  on.exit(rm("dummy", envir = .tabfound_backends), add = TRUE)
  register_backend(
    name    = "dummy",
    build   = function(config, task) NULL,
    detect  = function(config) identical(config$arch, "dummy_arch"),
    task_of = function(config) "classification",
    aliases = c("dummy-small" = "org/dummy-small"),
    description = "test backend"
  )
  expect_true("dummy" %in% list_backends()$name)
  expect_identical(detect_backend(list(arch = "dummy_arch"))$name, "dummy")
  expect_identical(expand_model_alias("dummy-small"), "org/dummy-small")
  # Anything that is not an alias passes through untouched, so local
  # paths and Hub ids keep working.
  expect_identical(expand_model_alias("/some/local/dir"), "/some/local/dir")
})


test_that("checkpoint keys map onto R module paths", {
  # nn_module_list inserts a `steps` component that nn.Sequential lacks.
  expect_identical(tabpfn_translate_key("encoder.5.layer.weight"),
                   "encoder.steps.5.layer.weight")
  expect_identical(tabpfn_translate_key("y_encoder.1.layer.bias"),
                   "y_encoder.steps.1.layer.bias")
  expect_identical(tabpfn_translate_key("decoder_dict.standard.2.weight"),
                   "decoder_dict.standard.steps.2.weight")
  # Everything else is identity, including the fused attention weights.
  expect_identical(
    tabpfn_translate_key("transformer_encoder.layers.0.self_attn_between_items._w_qkv"),
    "transformer_encoder.layers.0.self_attn_between_items._w_qkv"
  )
  expect_identical(tabpfn_translate_key("criterion.borders"), "criterion.borders")
})


test_that("the tabpfn26 backend registers and stays disjoint from tabpfn", {
  bks <- list_backends()
  expect_true("tabpfn26" %in% bks$name)

  v26 <- list(arch = "tabpfn_v2_6", head = "classifier")
  v25 <- list(arch = "per_feature_transformer", head = "classifier")

  # The two architectures share a family name and nothing else, so each
  # must claim exactly its own config -- a cross-claim would load one
  # checkpoint into the other's module tree.
  expect_identical(detect_backend(v26)$name, "tabpfn26")
  expect_identical(detect_backend(v25)$name, "tabpfn")
  expect_false(tabpfn_detect(v26))
  expect_false(tabpfn26_detect(v25))

  # v2.6 module names are the checkpoint's own, so nothing is rewritten.
  bk <- get_backend("tabpfn26")
  for (k in c("feature_group_embedder.weight",
              "blocks.0.per_sample_attention_between_features.q_projection.weight",
              "blocks.23.layernorm_mlp.weight",
              "add_thinking_rows.row_token_values_TE",
              "output_projection.2.bias",
              "criterion.borders")) {
    expect_identical(bk$translate_key(k), k)
  }
})


test_that("load_state_dict refuses to leave parameters unfilled", {
  skip_if_not_installed("torch")
  net <- torch::nn_module(
    "Toy",
    initialize = function() {
      self$a <- torch::nn_linear(2, 3, bias = FALSE)
      self$b <- torch::nn_linear(3, 1, bias = FALSE)
    },
    forward = function(x) self$b(self$a(x))
  )()

  w <- list(
    "a.weight" = torch::torch_ones(c(3, 2)),
    "b.weight" = torch::torch_ones(c(1, 3))
  )
  # The loader reports what it filled, so check for absence of error
  # rather than absence of output.
  expect_no_error(load_state_dict(net, w, names(w)))
  expect_true(as.logical((net$parameters[["a.weight"]] == 1)$all()$item()))

  # Filling only half the model is the failure mode that produces
  # plausible-but-wrong predictions, so it must abort, not warn.
  expect_error(load_state_dict(net, w["a.weight"], "a.weight"),
               "Weight loading incomplete")
  # A key with no matching module is equally fatal.
  expect_error(
    load_state_dict(net, c(w, list("c.weight" = torch::torch_ones(2))),
                    c(names(w), "c.weight")),
    "no matching R module"
  )
  # As is a shape disagreement.
  bad <- w; bad[["a.weight"]] <- torch::torch_ones(c(4, 2))
  expect_error(load_state_dict(net, bad, names(bad)), "shape mismatch")
})


test_that("the tabfm backend registers and claims its own config", {
  bks <- list_backends()
  expect_true("tabfm" %in% bks$name)

  # `num_freq` is what distinguishes TabFM from its close cousin TabICL,
  # which carries the same stage-width fields.
  tabfm_cfg <- list(embed_dim = 256L, icl_num_blocks = 24L,
                    col_num_inds = 256L, row_num_cls = 8L,
                    num_freq = 32L, is_classifier = TRUE)
  expect_identical(detect_backend(tabfm_cfg)$name, "tabfm")
  expect_identical(tabfm_task_of(tabfm_cfg), "classification")
  tabfm_cfg$is_classifier <- FALSE
  expect_identical(tabfm_task_of(tabfm_cfg), "regression")

  # TabPFN's config must not be claimed by TabFM, and vice versa —
  # detection is what makes `tabular_classifier(path)` work without the
  # caller naming a backend.
  expect_false(tabfm_detect(list(arch = "per_feature_transformer",
                                 head = "classifier")))
  expect_false(tabpfn_detect(tabfm_cfg))

  # TabFM's weights live in task subfolders on the Hub.
  bk <- get_backend("tabfm")
  expect_identical(bk$subfolder_for("classification"), "classification")
  expect_identical(bk$subfolder_for("regression"), "regression")
  expect_identical(expand_model_alias("tabfm-1.0.0"), "google/tabfm-1.0.0-pytorch")

  # Its state dict maps straight onto the R module tree.
  expect_identical(bk$translate_key("col_embedder.tf_col.blocks.0.mab1.attn.q_proj.weight"),
                   "col_embedder.tf_col.blocks.0.mab1.attn.q_proj.weight")
})


test_that("feature grouping wraps cyclically at offsets 2^i - 1", {
  skip_if_not_installed("torch")
  # (B, T, H) = (1, 1, 4), values 1..4 so the wrap is readable.
  x <- torch::torch_tensor(array(c(1, 2, 3, 4), dim = c(1, 1, 4)))
  g <- as.array(tabfm_group_features(x, 3L))
  expect_equal(dim(g), c(1L, 1L, 4L, 3L))
  # offsets 0, 1, 3 -> column j pairs with j, j+1, j+3 (mod 4)
  expect_equal(as.numeric(g[1, 1, 1, ]), c(1, 2, 4))
  expect_equal(as.numeric(g[1, 1, 3, ]), c(3, 4, 2))
})


test_that("a mistyped local path is reported as a path, not a Hub repo", {
  # Falling through to hfhub here produces "missing commit header",
  # which tells the user nothing about the actual mistake.
  expect_error(resolve_artifacts("/no/such/dir"), "is not a directory")
  expect_error(resolve_artifacts("./nope"), "is not a directory")

  d <- withr::local_tempdir()
  expect_error(resolve_artifacts(d), "missing model artifacts")
})


test_that("tabfm and tabicl configs do not both claim a model", {
  # These two architectures share most of their config field names --
  # embed_dim, icl_num_blocks, col_num_inds, row_num_cls -- so detection
  # has to key off what is genuinely distinctive. The registry catches an
  # ambiguity rather than silently picking one, and this pins the case.
  tabfm_cfg <- list(embed_dim = 256L, icl_num_blocks = 24L, col_num_inds = 256L,
                    row_num_cls = 8L, num_freq = 32L, is_classifier = TRUE)
  tabicl_cfg <- list(arch = "tabicl", head = "classifier", embed_dim = 128L,
                     icl_num_blocks = 12L, col_num_inds = 128L,
                     row_num_cls = 4L, max_classes = 10L)

  expect_identical(detect_backend(tabfm_cfg)$name, "tabfm")
  expect_identical(detect_backend(tabicl_cfg)$name, "tabicl")
  expect_false(tabfm_detect(tabicl_cfg))
  expect_false(tabicl_detect(tabfm_cfg))
})


test_that("feature grouping offsets differ between tabfm and tabicl", {
  skip_if_not_installed("torch")
  x <- torch::torch_tensor(array(c(1, 2, 3, 4), dim = c(1, 1, 4)))
  # TabFM: offsets 2^i - 1 -> 0, 1, 3 (the column itself comes first).
  fm <- as.array(tabfm_group_features(x, 3L))
  expect_equal(as.numeric(fm[1, 1, 1, ]), c(1, 2, 4))
  # TabICL: offsets 2^i -> 1, 2, 4 (the column itself is not in its group).
  icl <- as.array(tabicl_group_features(x, 3L))
  expect_equal(as.numeric(icl[1, 1, 1, ]), c(2, 3, 1))
})


test_that("fit() returns a new object and leaves the original unfitted", {
  # A model object must behave like every other R modelling object:
  # `m2 <- fit(m1, ...)` fits m2 and leaves m1 alone. The state used to
  # live in an R6 object captured by closure, so a "copy" shared it.
  spec <- list(
    fit = function(X, y) list(n_train = nrow(X), levels = sort(unique(y))),
    predict = function(state, newdata, type, ...) rep(state$levels[1], nrow(newdata))
  )
  ctx <- list(net = NULL, config = list(), device = "cpu",
              backend = list(name = "fake"), model_ref = "fake")
  m1 <- .new_tabfound_model(ctx, spec, "classification", list())

  expect_false(is_fitted(m1))
  m2 <- fit(m1, matrix(1:6, ncol = 2), c("a", "b", "a"))
  expect_true(is_fitted(m2))
  expect_false(is_fitted(m1))          # the point of the test
  expect_identical(m2$state$n_train, 3L)

  # Refitting produces yet another independent object.
  m3 <- fit(m2, matrix(1:4, ncol = 2), c("a", "b"))
  expect_identical(m2$state$n_train, 3L)
  expect_identical(m3$state$n_train, 2L)
})


test_that("predicting before fitting says what to do", {
  spec <- list(fit = function(X, y) list(n_train = nrow(X)),
               predict = function(state, newdata, type, ...) NULL)
  ctx <- list(net = NULL, config = list(), device = "cpu",
              backend = list(name = "fake"), model_ref = "fake")
  m <- .new_tabfound_model(ctx, spec, "classification", list())
  expect_error(predict(m, matrix(1:4, ncol = 2)), "has not been fitted")
})


test_that("an unsupported prediction type is refused by name", {
  spec <- list(fit = function(X, y) list(n_train = nrow(X)),
               predict = function(state, newdata, type, ...) type,
               types = "mean")
  ctx <- list(net = NULL, config = list(), device = "cpu",
              backend = list(name = "fake"), model_ref = "fake")
  m <- fit(.new_tabfound_model(ctx, spec, "regression", list()),
           matrix(1:4, ncol = 2), c(1, 2))
  expect_identical(predict(m, matrix(1:4, ncol = 2)), "mean")
  # TabFM's regression head is a point estimate; asking for quantiles
  # should say so rather than invent them.
  expect_error(predict(m, matrix(1:4, ncol = 2), type = "quantiles"),
               "no .* prediction")
})


test_that("tabfound_save round-trips the fitted state, saveRDS does not", {
  skip_if_not_installed("torch")
  # Uses a real (tiny) torch module so the dangling-pointer check is
  # exercised rather than mocked.
  net <- torch::nn_linear(2, 2)
  spec <- list(fit = function(X, y) list(n_train = nrow(X), y = y),
               predict = function(state, newdata, type, ...) state$y)
  ctx <- list(net = net, config = list(), device = "cpu",
              backend = list(name = "fake"), model_ref = "fake")
  m <- fit(.new_tabfound_model(ctx, spec, "classification", list()),
           matrix(1:4, ncol = 2), c("a", "b"))

  f <- withr::local_tempfile()
  tabfound_save(m, f)
  blob <- readRDS(f)
  expect_identical(blob$format, "tabfound-model")
  expect_identical(blob$state$y, c("a", "b"))
  # The weights are referenced, not copied -- a saved model must not be
  # a multi-gigabyte duplicate of its checkpoint.
  expect_lt(file.size(f), 5000)

  # saveRDS leaves dangling pointers; using the result must say so
  # instead of failing deep inside torch.
  g <- withr::local_tempfile()
  saveRDS(m, g)
  expect_error(predict(readRDS(g), matrix(1:4, ncol = 2)),
               "no longer available")
  expect_error(fit(readRDS(g), matrix(1:4, ncol = 2), c("a", "b")),
               "no longer available")

  # And loading a plain RDS through tabfound_load is caught by format.
  expect_error(tabfound_load(g), "not a saved tabfound model")
})


test_that("all four backends detect only their own config", {
  cfgs <- list(
    tabpfn = list(arch = "per_feature_transformer", head = "classifier"),
    tabfm  = list(embed_dim = 256L, icl_num_blocks = 24L, col_num_inds = 256L,
                  row_num_cls = 8L, num_freq = 32L, is_classifier = TRUE),
    tabicl = list(arch = "tabicl", head = "classifier", embed_dim = 128L,
                  icl_num_blocks = 12L, col_num_inds = 128L,
                  row_num_cls = 4L, max_classes = 10L),
    mitra  = list(dim = 512L, dim_output = 10L, n_layers = 12L,
                  n_heads = 4L, task = "CLASSIFICATION")
  )
  # Every config must be claimed by exactly one backend. Mitra's is the
  # thinnest of the four -- five fields, no architecture marker -- so it
  # is the one most at risk of being swallowed by a looser `detect()`.
  for (nm in names(cfgs)) {
    expect_identical(detect_backend(cfgs[[nm]])$name, nm,
                     label = paste("detect", nm))
  }
  expect_identical(mitra_task_of(cfgs$mitra), "classification")
  expect_identical(mitra_task_of(list(task = "REGRESSION")), "regression")
  expect_identical(get_backend("mitra")$translate_key("layers.0.attention1.q.weight"),
                   "layers.0.attention1.q.weight")
  expect_identical(expand_model_alias("mitra-classifier"),
                   "autogluon/mitra-classifier")
})


test_that("asking a backend for a path it lacks is refused", {
  # `kv_cache` and `save_peak_memory_factor` are TabPFN v2.6 paths. A
  # caller who set them for the speed has to be told they did not get it,
  # rather than silently paying the old cost.
  # The two capabilities are separate: v2.5 gained the cache, v2.6 has
  # both, and a network with neither must refuse both.
  v2 <- list(backend = list(name = "tabpfn"), net = list())
  v25 <- list(backend = list(name = "tabpfn"),
              net = list(supports_kv_cache = TRUE))
  v26 <- list(backend = list(name = "tabpfn26"),
              net = list(supports_kv_cache = TRUE, supports_chunked_eval = TRUE))

  expect_error(tabfound:::.require_kv_cache_support(v2, TRUE, NULL), "kv_cache")
  expect_error(tabfound:::.require_kv_cache_support(v2, FALSE, 8L),
               "save_peak_memory_factor")
  # v2.5: the cache is fine, the chunk factor is not.
  expect_true(tabfound:::.require_kv_cache_support(v25, TRUE, NULL))
  expect_error(tabfound:::.require_kv_cache_support(v25, TRUE, 8L),
               "save_peak_memory_factor")
  # Not asked for: nothing to refuse.
  expect_true(tabfound:::.require_kv_cache_support(v2, FALSE, NULL))
  # v2.6: both allowed.
  expect_true(tabfound:::.require_kv_cache_support(v26, TRUE, 8L))
})


test_that("a checkpoint in another dtype is cast, loudly", {
  skip_if_not_installed("torch")
  net <- torch::nn_linear(2, 2)
  keys <- c("weight", "bias")
  w32 <- list(weight = torch::torch_randn(c(2, 2)), bias = torch::torch_randn(2))

  # Matching dtypes say nothing: two dtype objects for the same dtype are
  # different R objects, so a naive comparison would report every tensor.
  expect_silent(suppressMessages(load_state_dict(net, w32, keys)))

  # A half-precision checkpoint would otherwise be `set_data`'d into
  # float32 slots and run at a precision nobody chose.
  w16 <- lapply(w32, function(t) t$to(dtype = torch::torch_half()))
  expect_warning(suppressMessages(load_state_dict(net, w16, keys)), "cast to")
  expect_identical(as.character(net$parameters$weight$dtype), "Float")
  expect_equal(as.array(net$parameters$weight),
               as.array(w16$weight$to(dtype = torch::torch_float())),
               tolerance = 1e-3)
})
