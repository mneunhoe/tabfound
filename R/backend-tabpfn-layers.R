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
  # @param save_peak_memory_factor Chunk count for [chunked_evaluate()].
  #   Each sublayer is independent across some leading fold -- the
  #   feature attention across rows, the item attention across columns,
  #   the MLP and the norms across every cell -- so this reorganises work
  #   that was already separate. Same mechanism v2.6 has, and the same
  #   `add_input = FALSE` + `residual = TRUE` arrangement, so the helper
  #   owns the residual and can write it back in place.
  # @return `list(state, kv)`.
  forward = function(x, single_eval_pos, dump_prefix = NULL,
                     cached_kv = NULL, return_kv = FALSE,
                     save_peak_memory_factor = NULL) {
    dims <- x$size()
    # Not named `N`: R torch's `[` reads that symbol as "to the end of
    # this dimension" rather than as the local, so `h_flat[, k:N, ]` below
    # would mean something else entirely. It happens to mean the same
    # thing here -- `h_flat`'s item axis is exactly this long -- but only
    # by coincidence, and a coincidence is not what the test-row slice
    # should rest on. (`Inf` is the same sentinel; nothing else is.)
    B <- dims[1]; n_rows <- dims[2]; F_ <- dims[3]; E <- dims[4]
    spmf <- save_peak_memory_factor

    # --- Sublayer 1: attention across features (full self-attention) ---
    # The rows fold into the batch, so this is independent per row.
    attn_f <- self$self_attn_between_features
    x <- chunked_evaluate(function(z) attn_f(z), x, spmf,
                          residual = TRUE, batch_dims = 2L)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_attn_features"), x)
    # The norms treat every cell on its own, so they fold all three.
    ln1 <- self$layer_norms[[1]]
    x <- chunked_evaluate(function(z) ln1(z), x, spmf,
                          residual = FALSE, batch_dims = 3L)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_norm_1"), x)

    # --- Sublayer 2: attention across items, with multi-query split ---
    # Python's `multiquery_item_attention_for_test_set` runs items attention
    # twice: train queries attend to train K/V (full multi-head), then test
    # queries attend to train K/V using only head 0's K/V (multi-query),
    # broadcast across all heads. Concat along the sample axis.
    n_test <- n_rows - single_eval_pos
    attn_i <- self$self_attn_between_items
    # Columns fold into the batch here, so the chunking is along `B * F`
    # -- never along the item axis, which is the one the attention
    # actually reads across.
    xf <- x$permute(c(1L, 3L, 2L, 4L))$contiguous()             # (B, F, N, E)
    kv <- NULL

    # One column's worth of the multi-query split: train queries attend
    # to train K/V with every head, test queries to the same K/V with
    # head 0 alone, broadcast.
    attend_items <- function(h_flat) {
      train_h <- h_flat[, 1:single_eval_pos, ]
      train_out <- attn_i(train_h, x_kv = train_h)
      if (n_test > 0L) {
        test_h <- h_flat[, (single_eval_pos + 1L):n_rows, ]
        test_out <- attn_i(test_h, x_kv = train_h,
                           reuse_first_head_kv = TRUE)
        torch::torch_cat(list(train_out, test_out), dim = 2L)
      } else {
        train_out
      }
    }

    if (isTRUE(return_kv) || !is.null(cached_kv)) {
      # Building or using a cache bypasses chunking, as it does on v2.6:
      # the key/value tensors have to be produced -- or read -- whole,
      # not a slice of the batch at a time.
      h_flat <- xf$reshape(c(B * F_, n_rows, E))
      if (!is.null(cached_kv)) {
        out <- attn_i(h_flat, add_input = FALSE, cached_kv = cached_kv)
      } else {
        kv <- attn_i$cache_kv(h_flat[, 1:single_eval_pos, ])
        out <- attend_items(h_flat)
      }
      xf <- xf + out$reshape(c(B, F_, n_rows, E))
    } else {
      xf <- chunked_evaluate(attend_items, xf, spmf,
                             residual = TRUE, batch_dims = 2L)
    }
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_attn_items"), xf$permute(c(1L, 3L, 2L, 4L))$contiguous())
    ln2 <- self$layer_norms[[2]]
    xf <- chunked_evaluate(function(z) ln2(z), xf, spmf,
                           residual = FALSE, batch_dims = 3L)
    x <- xf$permute(c(1L, 3L, 2L, 4L))$contiguous()
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_norm_2"), x)

    # --- Sublayer 3: MLP (per-token) ---
    mlp <- self$mlp
    x <- chunked_evaluate(function(z) mlp(z), x, spmf,
                          residual = TRUE, batch_dims = 3L)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_mlp"), x)
    ln3 <- self$layer_norms[[3]]
    list(state = chunked_evaluate(function(z) ln3(z), x, spmf,
                                  residual = FALSE, batch_dims = 3L),
         kv = kv)
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

  forward = function(x, single_eval_pos, save_peak_memory_factor = NULL) {
    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](
        x, single_eval_pos = single_eval_pos,
        save_peak_memory_factor = save_peak_memory_factor
      )$state
      collect_between_layers(x)
    }
    x
  }
)
