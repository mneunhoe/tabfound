# Cut-down architecture configs, shared by `test-arch.R` and
# `test-arch-styles.R`. Building one needs torch; the callers skip
# without it.

# A model object of the shape `tabfound_architecture()` expects, without
# the 600 MB of weights a real one carries.
arch_test_model <- function(config, backend, task = "classification") {
  net <- suppressMessages(get_backend(backend)$build(config, task))
  structure(
    list(spec = NULL, state = NULL, model = net, config = config,
         device = "cpu", backend = backend, task = task,
         model_ref = list(model = "test", backend = backend, device = "cpu",
                          args = list())),
    class = c(if (task == "classification") "tabfound_classifier"
              else "tabfound_regressor", "tabfound_model")
  )
}

arch_test_configs <- function() {
  list(
    tabpfn = list(
      config = list(arch = "per_feature_transformer", head = "classifier",
                    n_layers = 2L, embedding_dim = 12L, n_heads = 3L,
                    mlp_hidden_dim = 24L, features_per_group = 3L,
                    n_out_classes = 10L),
      task = "classification"),
    tabpfn26 = list(
      config = list(arch = "tabpfn_v2_6", head = "regressor", n_layers = 2L,
                    embedding_dim = 12L, n_heads = 3L, mlp_hidden_dim = 24L,
                    features_per_group = 3L, num_thinking_rows = 4L,
                    encoder_type = "linear", n_bar_bins = 16L),
      task = "regression"),
    tabpfn3 = list(
      config = list(arch = "tabpfn_v3", head = "classifier", embed_dim = 8L,
                    feat_agg_num_cls_tokens = 2L, ff_factor = 2L,
                    feature_group_size = 3L, use_nan_indicators = TRUE,
                    dist_embed_num_blocks = 1L, dist_embed_num_heads = 2L,
                    dist_embed_num_inducing_points = 4L,
                    feat_agg_num_blocks = 1L, feat_agg_num_heads = 2L,
                    nlayers = 2L, icl_num_heads = 2L, icl_num_kv_heads_test = 1L,
                    decoder_head_dim = 4L, decoder_num_heads = 2L,
                    decoder_use_softmax_scaling = TRUE,
                    softmax_scaling_mlp_hidden_dim = 4L, max_num_classes = 6L,
                    n_out = 6L, n_bar_bins = 16L),
      task = "classification"),
    # v3.5 carries no `head`: one checkpoint serves both tasks, so `task`
    # is what picks which decoder the diagram draws. Run as a regressor
    # here so the pair covers both branches of `tabpfn35_describe()`.
    tabpfn35 = list(
      config = list(arch = "tabpfn_v3_5", head = "multitask", embed_dim = 8L,
                    feat_agg_num_cls_tokens = 2L, ff_factor = 2L,
                    feature_group_size = 3L, use_nan_indicators = TRUE,
                    fourier_encoding_num_frequencies = 4L,
                    cell_ecdf_num_frequencies = 2L, cell_ecdf_num_buckets = 64L,
                    dist_embed_num_blocks = 1L, dist_embed_num_heads = 2L,
                    dist_embed_num_inducing_points = 4L,
                    feat_agg_num_blocks = 1L, feat_agg_num_heads = 2L,
                    nlayers = 2L, icl_num_heads = 2L, icl_num_kv_heads_test = 1L,
                    decoder_head_dim = 4L, decoder_num_heads = 2L,
                    decoder_use_softmax_scaling = TRUE,
                    softmax_scaling_mlp_hidden_dim = 4L, max_num_classes = 6L,
                    num_buckets = 16L, n_bar_bins = 16L),
      task = "regression"),
    tabicl = list(
      config = list(arch = "tabicl", head = "classifier", max_classes = 6L,
                    num_quantiles = 9L, embed_dim = 8L, col_num_blocks = 1L,
                    col_nhead = 2L, col_num_inds = 4L, col_feature_group = "same",
                    col_feature_group_size = 3L, col_target_aware = TRUE,
                    col_ssmax = "qassmax-mlp-elementwise", row_num_blocks = 2L,
                    row_nhead = 2L, row_num_cls = 2L, row_rope_base = 100000,
                    row_rope_interleaved = FALSE, icl_num_blocks = 2L,
                    icl_nhead = 2L, icl_ssmax = "qassmax-mlp-elementwise",
                    ff_factor = 2L, activation = "gelu", bias_free_ln = FALSE),
      task = "classification"),
    tabfm = list(
      config = list(arch = "tabfm", embed_dim = 8L, ff_factor = 2L,
                    row_num_cls = 2L, col_num_blocks = 1L, col_nhead = 2L,
                    col_num_inds = 4L, row_num_blocks = 1L, row_nhead = 2L,
                    icl_num_blocks = 2L, icl_nhead = 2L, max_classes = 6L,
                    feature_group_size = 3L, num_freq = 4L,
                    is_classifier = TRUE, decoder_hidden = NULL),
      task = "classification"),
    # Medium's shape (a row projection, grouped-query attention) so the
    # optional `row_project` stage is the one exercised.
    kumo = list(
      config = list(arch = "kumo_tabular", head = "classifier", size = "medium",
                    cell_channels = 8L, num_embedding_layers = 2L,
                    num_embedding_heads = 2L, num_inducing_points = 4L,
                    group_size = 3L, num_frequencies = 4L,
                    num_readout_tokens = 2L, icl_channels = 8L,
                    num_icl_layers = 2L, num_icl_heads = 2L,
                    num_icl_key_value_heads_for_query = 1L, gate_hidden = 4L,
                    num_classes = 10L, num_quantiles = 0L),
      task = "classification"),
    mitra = list(
      config = list(dim = 8L, dim_output = 6L, n_heads = 2L, n_layers = 2L,
                    task = "CLASSIFICATION"),
      task = "classification")
  )
}

# Every config as both a classifier and a regressor, plus Kumo small,
# which has no row projection. The key and vocabulary tests run on all
# of them: a regressor draws different decoders and outputs, and so
# different symbols and abbreviations.
arch_test_variants <- function() {
  cs <- arch_test_configs()
  flip <- list(
    tabpfn   = list(head = "regressor", n_bar_bins = 16L),
    tabpfn26 = list(head = "classifier", n_out_classes = 6L),
    tabpfn3  = list(head = "regressor", n_out = 16L, num_buckets = 16L),
    tabpfn35 = list(),
    tabicl   = list(head = "regressor", max_classes = 0L),
    tabfm    = list(is_classifier = FALSE),
    kumo     = list(head = "regressor", num_classes = 0L, num_quantiles = 9L),
    mitra    = list(task = "REGRESSION", dim_output = 1L)
  )
  other <- c(classification = "regression", regression = "classification")
  out <- list()
  for (nm in names(cs)) {
    out[[paste(nm, cs[[nm]]$task)]] <- list(
      backend = nm, config = cs[[nm]]$config, task = cs[[nm]]$task)
    task2 <- other[[cs[[nm]]$task]]
    out[[paste(nm, task2)]] <- list(
      backend = nm, config = utils::modifyList(cs[[nm]]$config, flip[[nm]]),
      task = task2)
  }
  # Small: the readout tokens flatten straight to the in-context width.
  out[["kumo small"]] <- list(
    backend = "kumo", task = "classification",
    config = utils::modifyList(cs$kumo$config,
                               list(size = "small", icl_channels = 16L,
                                    num_icl_key_value_heads_for_query = NULL)))
  out
}

arch_test_variant_arch <- function(v) {
  tabfound_architecture(arch_test_model(v$config, v$backend, v$task))
}
