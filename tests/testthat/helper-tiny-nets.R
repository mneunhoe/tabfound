# Randomly initialised networks at a fraction of the real width, for the
# tests whose question is structural -- what the predictors do with the
# network's outputs -- rather than a property of the weights. Shared by
# `test-kv-cache.R` and `test-ensemble-members.R`.

tiny_mitra <- function(task = "CLASSIFICATION", n_layers = 3L) {
  torch::torch_manual_seed(3L)
  net <- mitra_model(list(dim = 32L, dim_output = if (task == "CLASSIFICATION") 4L else 1L,
                          n_layers = n_layers, n_heads = 4L, task = task))
  net$eval()
  net
}

tiny_tabicl_config <- function(head = "classifier") {
  list(
    arch = "tabicl", head = head,
    max_classes = if (head == "classifier") 4L else 0L,
    num_quantiles = if (head == "regressor") 9L else 0L,
    embed_dim = 16L, col_num_blocks = 2L, col_nhead = 2L, col_num_inds = 5L,
    col_affine = FALSE, col_feature_group = "same", col_feature_group_size = 3L,
    col_target_aware = TRUE, col_ssmax = "qassmax-mlp-elementwise",
    row_num_blocks = 2L, row_nhead = 2L, row_num_cls = 2L,
    row_rope_base = 100000, row_rope_interleaved = FALSE,
    icl_num_blocks = 3L, icl_nhead = 2L, icl_ssmax = "qassmax-mlp-elementwise",
    ff_factor = 2L, activation = "gelu", bias_free_ln = FALSE
  )
}

tiny_tabicl <- function(head = "classifier") {
  torch::torch_manual_seed(11L)
  net <- tabicl_model(tiny_tabicl_config(head))
  net$eval()
  net
}

tiny_tabfm <- function(is_classifier = TRUE) {
  torch::torch_manual_seed(7L)
  net <- tabfm_model(list(
    embed_dim = 16L, ff_factor = 2L, row_num_cls = 2L,
    is_classifier = is_classifier, max_classes = 4L,
    feature_group_size = 3L, num_freq = 8L,
    col_num_blocks = 2L, col_nhead = 2L, col_num_inds = 4L,
    row_num_blocks = 2L, row_nhead = 2L,
    icl_num_blocks = 2L, icl_nhead = 2L, decoder_hidden = NULL
  ))
  net$eval()
  net
}

tiny_kumo_config <- function(head = "classifier", kv = 2L) {
  list(arch = "kumo_tabular", head = head, cell_channels = 8L,
       num_embedding_layers = 2L, num_embedding_heads = 2L,
       num_inducing_points = 3L, group_size = 3L, num_frequencies = 4L,
       num_readout_tokens = 4L, icl_channels = 32L, num_icl_layers = 2L,
       num_icl_heads = 4L, num_icl_key_value_heads_for_query = kv,
       gate_hidden = 8L,
       num_classes = if (head == "classifier") 10L else 0L,
       num_quantiles = if (head == "classifier") 0L else 9L)
}

# `sdm` zero-initialises the output projections, which would make every
# block an identity; redraw everything so the cache has something to get
# wrong.
tiny_kumo <- function(head = "classifier", kv = 2L) {
  torch::torch_manual_seed(13L)
  net <- kumo_model(tiny_kumo_config(head, kv))
  ps <- net$parameters
  torch::with_no_grad(for (nm in names(ps)) {
    if (!grepl("inv_freq$", nm)) ps[[nm]]$normal_(std = 0.3)
  })
  net$eval()
  net
}

# A loaded-model context is two fields as far as these predictors are
# concerned, so a random network stands in for a checkpoint.
fake_ctx <- function(net, config = NULL) {
  list(net = net, device = "cpu", config = config)
}
