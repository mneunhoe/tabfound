# Architecture dimensions for the memory preflight tests.
#
# These are configs, not models: `estimate_peak_memory()` never touches
# the weights, so the whole estimator is testable on any machine with no
# checkpoint present -- the same trick `helper-stub-model.R` plays for
# the MI code.
#
# The TabPFN and TabICL entries are the published checkpoints' real
# dimensions, trimmed to the fields the estimator reads. `state_dict_shapes`
# is kept only where a test asserts something about it, since the real
# tables run to hundreds of entries.

mem_config_tabpfn25 <- function() {
  list(
    arch = "per_feature_transformer", head = "classifier",
    n_layers = 24, embedding_dim = 192, n_heads = 3,
    features_per_group = 3, max_num_features = 500,
    state_dict_shapes = list(
      "add_thinking_tokens.row_token_values" = c(64, 192),
      "transformer_encoder.layers.0.self_attn_between_items._w_qkv" =
        c(3, 3, 64, 192)
    )
  )
}

mem_config_tabpfn3 <- function() {
  list(
    arch = "tabpfn_v3", head = "classifier",
    embed_dim = 128, feature_group_size = 3, nlayers = 24,
    dist_embed_num_heads = 8, dist_embed_num_inducing_points = 128,
    feat_agg_num_heads = 8, feat_agg_num_cls_tokens = 4,
    icl_num_heads = 8, icl_num_kv_heads_test = 1, ff_factor = 2
  )
}

mem_config_tabicl <- function() {
  list(
    arch = "tabicl", head = "classifier", max_classes = 10,
    embed_dim = 128, col_num_blocks = 3, col_nhead = 8, col_num_inds = 128,
    col_feature_group_size = 3, row_num_blocks = 3, row_nhead = 8,
    row_num_cls = 4, icl_num_blocks = 12, icl_nhead = 8, ff_factor = 2
  )
}

mem_config_tabfm <- function() {
  list(
    num_freq = 32, is_classifier = TRUE, embed_dim = 256, max_classes = 10,
    feature_group_size = 3, col_num_blocks = 3, col_nhead = 4,
    col_num_inds = 256, row_num_blocks = 3, row_nhead = 8, row_num_cls = 8,
    icl_num_blocks = 24, icl_nhead = 8, ff_factor = 4, decoder_hidden = NULL
  )
}

mem_config_mitra <- function() {
  list(task = "CLASSIFICATION", dim = 512, dim_output = 10,
       n_layers = 12, n_heads = 4)
}

# The Muchlinski civil-war fold the hand-off is written around:
# 7,140 rows split 6,426 / 714, 90 features.
MEM_FOLD <- list(n_context = 6426, n_query = 714, n_features = 90)

# An idle 48 GB machine, and the same machine with a 7 GB neighbour.
MEM_IDLE <- 40e9
MEM_BUSY <- 6e9


# A `tabfound_model` with a registered backend and a real config, but no
# weights: enough for the guard, which asks the object for its backend,
# its config and the arguments it was built with, and never touches a
# tensor. Same trick as `helper-stub-model.R`, one level up.
mem_stub_model <- function(backend, config, task = "classification",
                           args = list(), n_train = NULL) {
  structure(
    list(
      spec = list(
        fit = function(X, y) list(n_train = NROW(X)),
        predict = function(state, newdata, type, ...) rep(NA, NROW(newdata))
      ),
      state = if (is.null(n_train)) NULL else list(n_train = n_train),
      model = NULL, config = config, device = "cpu",
      backend = backend, task = task,
      model_ref = list(model = backend, backend = backend, device = "cpu",
                       args = args)
    ),
    class = c(if (task == "classification") "tabfound_classifier"
              else "tabfound_regressor",
              "tabfound_model")
  )
}
