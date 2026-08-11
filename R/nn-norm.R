# Normalization layers.

# Build a stateless LayerNorm matching Python's
# `LayerNorm(emb, eps, elementwise_affine=FALSE)`.
# @keywords internal
stateless_layer_norm <- function(embedding_dim, eps = 1e-5) {
  torch::nn_layer_norm(
    normalized_shape   = embedding_dim,
    eps                = eps,
    elementwise_affine = FALSE
  )
}


#' Root-mean-square layer normalization
#'
#' `x * rsqrt(mean(x^2) + eps) * weight`, with a learnable per-dimension
#' `weight` and no mean subtraction or bias.
#'
#' The whole computation runs in float32 and is cast back at the end,
#' even when the input is float32 already — that is what the reference
#' does, and TabFM applies ~36 of these per forward pass, so the
#' convention has to be reproduced rather than approximated.
#'
#' @param dim Size of the normalized (last) axis.
#' @param eps Added to the mean square before the reciprocal square root.
#' @keywords internal
rms_norm <- torch::nn_module(
  "RmsNorm",

  initialize = function(dim, eps = 1e-6) {
    self$weight <- torch::nn_parameter(torch::torch_ones(dim))
    self$eps <- eps
  },

  forward = function(x) {
    dt <- x$dtype
    xf <- x$to(dtype = torch::torch_float32())
    v  <- xf$pow(2)$mean(dim = -1L, keepdim = TRUE)
    ((xf * torch::torch_rsqrt(v + self$eps)) *
       self$weight$to(dtype = torch::torch_float32()))$to(dtype = dt)
  }
)


#' Affine layer normalization with an optional bias
#'
#' `torch::nn_layer_norm()` offers `elementwise_affine`, which is
#' all-or-nothing: weight *and* bias, or neither. TabICL needs the third
#' combination — weight but no bias — because its `bias_free_ln` flag is
#' `FALSE` in the released classifier and `TRUE` in the regressor, so a
#' hard-coded choice would load one checkpoint and fail the other.
#'
#' Parameters are named `weight` and `bias`, matching the checkpoint.
#'
#' @param dim Size of the normalized (last) axis.
#' @param eps Added to the variance.
#' @param bias Whether to include a learnable bias.
#' @keywords internal
affine_layer_norm <- torch::nn_module(
  "AffineLayerNorm",

  initialize = function(dim, eps = 1e-5, bias = TRUE) {
    self$normalized_shape <- as.integer(dim)
    self$eps <- eps
    self$weight <- torch::nn_parameter(torch::torch_ones(self$normalized_shape))
    if (isTRUE(bias)) {
      self$bias <- torch::nn_parameter(torch::torch_zeros(self$normalized_shape))
    }
    self$use_bias <- isTRUE(bias)
  },

  forward = function(x) {
    torch::nnf_layer_norm(
      x, self$normalized_shape, self$weight,
      if (self$use_bias) self$bias else NULL, self$eps
    )
  }
)
