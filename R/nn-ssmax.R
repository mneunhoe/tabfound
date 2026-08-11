# Scalable softmax.
#
# Ordinary attention divides logits by sqrt(head_dim), a constant. That
# makes the softmax sharpen as the sequence grows, because the number of
# competing keys grows while the logit scale does not. Scalable softmax
# instead makes the query scale depend on the context length, so a model
# trained on short in-context sequences keeps behaving sensibly on long
# ones -- which for a tabular in-context learner means training folds of
# wildly different sizes.

#' Query-aware scalable softmax (elementwise)
#'
#' Scales queries by a factor that depends on the source length **and**
#' on the query itself:
#'
#'   scale = base_mlp(log n) * (1 + tanh(query_mlp(q)))
#'
#' `base_mlp` maps the scalar `log n` to a per-head, per-dimension base
#' scale; `query_mlp` maps each query vector to a modulation in
#' `(0, 2)`. Both are two-layer MLPs with an exact-erf GELU between them
#' — TabICL uses `nn.GELU()`, not the tanh approximation TabFM inherits
#' from JAX.
#'
#' Held under `attn.ssmax_layer.*` in the checkpoint, with the linears at
#' `base_mlp.0` / `base_mlp.2` (index 1 is the parameterless activation).
#'
#' @param n_heads Number of attention heads.
#' @param head_dim Per-head dimension.
#' @param n_hidden Hidden width of both MLPs.
#' @keywords internal
ssmax_qa_mlp <- torch::nn_module(
  "SSMaxQueryAware",

  initialize = function(n_heads, head_dim, n_hidden = 64L) {
    self$n_heads <- as.integer(n_heads)
    self$head_dim <- as.integer(head_dim)
    # `nn_sequential` numbers its children 0, 1, 2 like nn.Sequential, so
    # the checkpoint keys line up without translation.
    self$base_mlp <- torch::nn_sequential(
      torch::nn_linear(1L, n_hidden),
      torch::nn_gelu(),
      torch::nn_linear(n_hidden, self$n_heads * self$head_dim)
    )
    self$query_mlp <- torch::nn_sequential(
      torch::nn_linear(self$head_dim, n_hidden),
      torch::nn_gelu(),
      torch::nn_linear(n_hidden, self$head_dim)
    )
  },

  # @param q `(B, H, Tq, D)` queries after projection.
  # @param n Source sequence length.
  forward = function(q, n) {
    logn <- torch::torch_tensor(log(max(n, 1)), dtype = q$dtype,
                                device = q$device)$reshape(c(1L, 1L))
    base <- self$base_mlp(logn)$view(c(1L, self$n_heads, 1L, self$head_dim))
    modulation <- 1 + torch::torch_tanh(self$query_mlp(q))
    q * (base * modulation)
  }
)
