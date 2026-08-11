# TabPFN encoder layer and layer stack.
#
# Post-norm is the convention used at inference: every sublayer runs
# `state <- layer_norm(sublayer(state, add_input = TRUE))`, with the
# sublayer adding its own residual. The LayerNorms have
# `elementwise_affine = FALSE`, so there is nothing to load for them.
# Linear layers in both attention and MLP have no biases.

#' Per-feature encoder layer
#'
#' Applies three sublayers in sequence, each followed by a stateless
#' LayerNorm:
#'   1. attention across features (over the feature-group axis)
#'   2. attention across items    (over the sample axis)
#'   3. MLP
#'
#' The attention/MLP forwards add the residual internally
#' (`add_input = TRUE`); we then apply the LayerNorm to the sum.
#'
#' @keywords internal
per_feature_encoder_layer <- torch::nn_module(
  "PerFeatureEncoderLayer",

  initialize = function(embedding_dim, n_heads, mlp_hidden_dim,
                        layer_norm_eps = 1e-5,
                        activation = "gelu") {

    self$self_attn_between_features <- mha_fused_qkv(embedding_dim, n_heads)
    self$self_attn_between_items    <- mha_fused_qkv(embedding_dim, n_heads)
    self$mlp <- ff_mlp(embedding_dim, mlp_hidden_dim, activation = activation)

    self$layer_norms <- torch::nn_module_list(lapply(
      seq_len(3L),
      function(i) stateless_layer_norm(embedding_dim, eps = layer_norm_eps)
    ))
  },

  # @param x Tensor of shape `(B, N, F_groups, emb)` where N includes
  #'   thinking tokens, train and test rows.
  # @param cached_kv This layer's entry from a [tabpfn_kv_cache()], or
  #'   NULL. When supplied every row of `x` is a test row and the training
  #'   rows are not present at all.
  # @param return_kv Collect this layer's train-row key/value projections.
  # @return `list(state, kv)`.
  forward = function(x, single_eval_pos, dump_prefix = NULL,
                     cached_kv = NULL, return_kv = FALSE) {
    dims <- x$size()
    # Not named `N`: R torch's `[` reads that symbol as "to the end of
    # this dimension" rather than as the local, so `h_flat[, k:N, ]` below
    # would mean something else entirely. It happens to mean the same
    # thing here -- `h_flat`'s item axis is exactly this long -- but only
    # by coincidence, and a coincidence is not what the test-row slice
    # should rest on. (`Inf` is the same sentinel; nothing else is.)
    B <- dims[1]; n_rows <- dims[2]; F_ <- dims[3]; E <- dims[4]

    # --- Sublayer 1: attention across features (full self-attention) ---
    h <- x$reshape(c(B * n_rows, F_, E))
    h <- self$self_attn_between_features(h, add_input = TRUE)
    h <- h$reshape(c(B, n_rows, F_, E))
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_attn_features"), h)
    x <- self$layer_norms[[1]](h)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_norm_1"), x)

    # --- Sublayer 2: attention across items, with multi-query split ---
    # Python's `multiquery_item_attention_for_test_set` runs items attention
    # twice: train queries attend to train K/V (full multi-head), then test
    # queries attend to train K/V using only head 0's K/V (multi-query),
    # broadcast across all heads. Concat along the sample axis.
    n_test <- n_rows - single_eval_pos
    # Reshape to (B*F, N, E) so the second axis is the item axis.
    h_flat <- x$permute(c(1L, 3L, 2L, 4L))$contiguous()$reshape(c(B * F_, n_rows, E))
    kv <- NULL
    if (!is.null(cached_kv)) {
      # Every row is a test row attending to the cached training K/V.
      h_flat <- self$self_attn_between_items(
        h_flat, add_input = TRUE, cached_kv = cached_kv
      )
    } else {
      train_h <- h_flat[, 1:single_eval_pos, ]
      if (isTRUE(return_kv)) kv <- self$self_attn_between_items$cache_kv(train_h)
      train_out <- self$self_attn_between_items(
        train_h, x_kv = train_h, add_input = TRUE
      )
      if (n_test > 0L) {
        test_h <- h_flat[, (single_eval_pos + 1L):n_rows, ]
        test_out <- self$self_attn_between_items(
          test_h, x_kv = train_h, add_input = TRUE,
          reuse_first_head_kv = TRUE
        )
        h_flat <- torch::torch_cat(list(train_out, test_out), dim = 2L)
      } else {
        h_flat <- train_out
      }
    }
    h <- h_flat$reshape(c(B, F_, n_rows, E))$permute(c(1L, 3L, 2L, 4L))$contiguous()
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_attn_items"), h)
    x <- self$layer_norms[[2]](h)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_norm_2"), x)

    # --- Sublayer 3: MLP (per-token) ---
    h <- self$mlp(x, add_input = TRUE)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_mlp"), h)
    list(state = self$layer_norms[[3]](h), kv = kv)
  }
)

# Stack of `per_feature_encoder_layer`s.
#
# The submodule is named `layers` to match the ckpt key prefix
# `transformer_encoder.layers.<i>.*`.

#' Layer stack
#' @keywords internal
tabpfn_layer_stack <- torch::nn_module(
  "TabpfnLayerStack",

  initialize = function(n_layers, embedding_dim, n_heads, mlp_hidden_dim,
                        layer_norm_eps = 1e-5,
                        activation = "gelu") {
    self$layers <- torch::nn_module_list(
      lapply(seq_len(n_layers), function(i) {
        per_feature_encoder_layer(
          embedding_dim  = embedding_dim,
          n_heads        = n_heads,
          mlp_hidden_dim = mlp_hidden_dim,
          layer_norm_eps = layer_norm_eps,
          activation     = activation
        )
      })
    )
  },

  forward = function(x, single_eval_pos) {
    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](x, single_eval_pos = single_eval_pos)$state
    }
    x
  }
)
