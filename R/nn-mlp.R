# Feed-forward blocks.

#' Two-layer feed-forward MLP with no biases
#'
#' `linear1 -> activation -> linear2`. `add_input = TRUE` folds the
#' residual into the sublayer, which is how TabPFN's encoder layer calls
#' it.
#' @keywords internal
ff_mlp <- torch::nn_module(
  "FeedForwardMlp",

  initialize = function(embedding_dim, hidden_dim, activation = "gelu") {
    self$linear1 <- torch::nn_linear(embedding_dim, hidden_dim, bias = FALSE)
    self$linear2 <- torch::nn_linear(hidden_dim, embedding_dim, bias = FALSE)
    self$activation <- activation
  },

  # @param add_input If TRUE, return `x + mlp(x)`.
  forward = function(x, add_input = FALSE) {
    h <- self$linear1(x)
    h <- switch(
      self$activation,
      gelu = torch::nnf_gelu(h),
      relu = torch::nnf_relu(h),
      cli::cli_abort("Unsupported activation: {.val {self$activation}}")
    )
    out <- self$linear2(h)
    if (isTRUE(add_input)) out <- out + x
    out
  }
)


#' Multi-layer perceptron with activations between layers only
#'
#' Mirrors the reference's `MLP`: an `nn_module_list` named `layers`
#' holding `length(hidden_dims) + 1` linears, with the activation applied
#' after every layer except the last.
#'
#' @param in_dim,out_dim Input and output widths.
#' @param hidden_dims Integer vector of hidden widths.
#' @param activation `"gelu"` (exact, torch's default), `"gelu_tanh"`
#'   (the approximation JAX defaults to), `"relu"` or `"silu"`.
#' @keywords internal
mlp_stack <- torch::nn_module(
  "MlpStack",

  initialize = function(in_dim, hidden_dims, out_dim, activation = "gelu") {
    dims <- c(as.integer(in_dim), as.integer(hidden_dims))
    lins <- lapply(seq_along(hidden_dims), function(i)
      torch::nn_linear(dims[i], dims[i + 1L], bias = TRUE))
    lins[[length(lins) + 1L]] <- torch::nn_linear(
      dims[length(dims)], as.integer(out_dim), bias = TRUE
    )
    self$layers <- torch::nn_module_list(lins)
    self$activation <- activation
  },

  forward = function(x) {
    n <- length(self$layers)
    for (i in seq_len(n)) {
      x <- self$layers[[i]](x)
      if (i < n) x <- apply_activation(x, self$activation)
    }
    x
  }
)


#' Apply a named activation
#'
#' There are two GELUs in circulation and they are not interchangeable:
#' `jax.nn.gelu` defaults to a tanh approximation, while torch's
#' `F.gelu` defaults to the exact erf form. They differ by ~1e-3 at
#' moderate inputs, which compounds across a stack.
#'
#' `"gelu"` here is the **exact** form, matching torch — that is what a
#' config field saying `"gelu"` means for a PyTorch-native model like
#' TabICL. Models ported from JAX (TabFM) must ask for `"gelu_tanh"`
#' explicitly.
#' @keywords internal
apply_activation <- function(x, name) {
  switch(
    name,
    gelu      = torch::nnf_gelu(x),
    gelu_tanh = torch::nnf_gelu(x, approximate = "tanh"),
    relu      = torch::nnf_relu(x),
    silu      = torch::nnf_silu(x),
    cli::cli_abort("Unsupported activation: {.val {name}}")
  )
}
