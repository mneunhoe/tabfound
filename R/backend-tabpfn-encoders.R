# TabPFN input encoders.
#
# The real encoder in TabPFN is an nn_sequential with a stateless
# NaN-handling step at the front and a parameterized step at the end.
# The classifier and regressor use the same sequential layout but the
# regressor's final step is a 2-layer MLP (hidden_dim=1024) instead of
# a single linear.
#
# State-dict layout (from Prior-Labs/tabpfn_2_5 default ckpts):
#   classifier X encoder :  encoder.5.layer.weight         shape [192, 6]
#   regressor  X encoder :  encoder.5.mlp.0.weight         shape [1024, 6]
#                          encoder.5.mlp.2.weight         shape [192, 1024]
#   classifier y encoder :  y_encoder.2.layer.weight/bias  shape [192, 2]/[192]
#   regressor  y encoder :  y_encoder.1.layer.weight/bias  shape [192, 2]/[192]
#
# Shape contracts:
#   Input X arrives as (B, N, F_groups, features_per_group) where
#   features_per_group = 3 and missing features are zero-padded by the
#   caller. The encoder appends an is-nan indicator (→ 2 * features_per_group
#   channels = 6) before the final linear/MLP.
#   Output: (B, N, F_groups, emb_dim).
#
#   Input y arrives as (B, N, 1). The y encoder appends an is-nan channel
#   (→ 2) and projects to emb_dim. Output: (B, N, emb_dim).

#' Replace NaNs with 0 and append an is-nan indicator channel
#' @keywords internal
nan_handling_encoder_step <- torch::nn_module(
  "NanHandlingEncoderStep",

  initialize = function() {},

  forward = function(x) {
    is_nan <- torch::torch_isnan(x)
    x_clean <- torch::torch_where(
      is_nan, torch::torch_zeros_like(x), x
    )
    torch::torch_cat(list(x_clean, is_nan$to(dtype = x$dtype)), dim = -1L)
  }
)


#' Final linear step of the X encoder (classifier)
#'
#' Holds the inner `layer` submodule so the state_dict key is
#' `encoder.5.layer.weight`.
#' @keywords internal
linear_encoder_step <- torch::nn_module(
  "LinearEncoderStep",
  initialize = function(in_channels, embedding_dim, bias = FALSE) {
    self$layer <- torch::nn_linear(in_channels, embedding_dim, bias = bias)
  },
  forward = function(x) self$layer(x)
)


#' Final MLP step of the X encoder (regressor)
#'
#' `encoder.5.mlp.0` and `encoder.5.mlp.2` are the two linears with a
#' GELU (no params) between them.
#' @keywords internal
mlp_encoder_step <- torch::nn_module(
  "MlpEncoderStep",
  initialize = function(in_channels, hidden_dim, embedding_dim,
                        activation = "gelu", bias = FALSE) {
    self$mlp <- torch::nn_sequential(
      torch::nn_linear(in_channels, hidden_dim, bias = bias),
      if (activation == "gelu") torch::nn_gelu() else torch::nn_relu(),
      torch::nn_linear(hidden_dim, embedding_dim, bias = bias)
    )
  },
  forward = function(x) self$mlp(x)
)


#' y encoder's final linear step: `y_encoder.*.layer.*`
#' @keywords internal
y_linear_step <- torch::nn_module(
  "YLinearStep",
  initialize = function(in_channels, embedding_dim, bias = TRUE) {
    self$layer <- torch::nn_linear(in_channels, embedding_dim, bias = bias)
  },
  forward = function(x) self$layer(x)
)


#' X encoder: NaN-handle then linear (classifier) or MLP (regressor)
#'
#' Builds a 6-step sequential so the final parameterized step lands at
#' `encoder.5.*`, matching the ckpt keys. Steps 0..4 are pass-through
#' `nn_identity`s; step 5 is the linear or MLP.
#'
#' @param head `"classifier"` or `"regressor"`.
#' @keywords internal
input_encoder <- torch::nn_module(
  "InputEncoder",

  initialize = function(embedding_dim,
                        features_per_group,
                        head = c("classifier", "regressor"),
                        mlp_hidden_dim = 1024L) {
    head <- match.arg(head)
    self$features_per_group <- as.integer(features_per_group)
    self$head <- head

    in_channels <- 2L * self$features_per_group

    # Steps 0..4 are stateless in the reference ckpts. We store them as
    # nn_identity so the 5-step structure is preserved and the final
    # linear/MLP lives at attribute index `5`.
    final_step <- if (head == "classifier") {
      linear_encoder_step(in_channels, embedding_dim, bias = FALSE)
    } else {
      mlp_encoder_step(in_channels, mlp_hidden_dim, embedding_dim, bias = FALSE)
    }

    self$steps <- torch::nn_module_list(list(
      nan_handling_encoder_step(),     # step 0: NaN -> append indicator
      torch::nn_identity(),             # step 1
      torch::nn_identity(),             # step 2
      torch::nn_identity(),             # step 3
      torch::nn_identity(),             # step 4
      final_step                        # step 5
    ))
  },

  # @param x `(B, N, F_groups, features_per_group)` — numeric cells with NaNs.
  forward = function(x) {
    for (i in seq_along(self$steps)) {
      x <- self$steps[[i]](x)
    }
    x
  }
)


#' y encoder
#'
#' Classifier puts its final linear at step index 2; regressor at 1.
#' The difference is an extra identity step in the classifier. A 3-step
#' sequential with the final linear at `.2` covers the classifier case;
#' for regressor we use a 2-step variant with the linear at `.1`.
#'
#' @param head `"classifier"` or `"regressor"`.
#' @keywords internal
y_encoder <- torch::nn_module(
  "YEncoder",

  initialize = function(embedding_dim, head = c("classifier", "regressor")) {
    head <- match.arg(head)
    self$head <- head

    lin <- y_linear_step(in_channels = 2L, embedding_dim = embedding_dim, bias = TRUE)

    if (head == "classifier") {
      # y_encoder.0 = nan_step ; .1 = identity ; .2 = linear
      self$steps <- torch::nn_module_list(list(
        nan_handling_encoder_step(),
        torch::nn_identity(),
        lin
      ))
    } else {
      # y_encoder.0 = nan_step ; .1 = linear
      self$steps <- torch::nn_module_list(list(
        nan_handling_encoder_step(),
        lin
      ))
    }
  },

  # @param y `(B, N, 1)` — test positions should be NaN.
  forward = function(y) {
    for (i in seq_along(self$steps)) {
      y <- self$steps[[i]](y)
    }
    y
  }
)
