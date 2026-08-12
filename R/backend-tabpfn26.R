# TabPFN v2.6 backend (Prior-Labs).
#
# v2.6 shares TabPFN v2's idea -- alternating attention between a row's
# features and between a column's cells, with the target carried as an
# extra column -- but it is a different module tree, not a differently
# configured one, so it gets its own backend rather than a flag on the
# v2/v2.5 one:
#
#   * RMSNorm with a learnable weight, where v2.5 has a stateless,
#     affine-free LayerNorm. The checkpoint therefore carries three norm
#     weights per block that v2.5 does not have at all.
#   * Separate `q/k/v/out_projection` linears instead of the fused
#     `_w_qkv` / `_w_out` per-head tensors.
#   * Preprocessing moved into the architecture: constant columns are
#     dropped, NaN/Inf are imputed with the train mean and flagged in a
#     signed indicator channel, and a standard scaler is fitted on the
#     train rows -- all inside `forward()`, not in encoder "steps".
#   * No multiclass rank encoder. v2.5 re-ranked `y` against the sorted
#     unique training labels before embedding it; v2.6 embeds the ordinal
#     label directly and raises if a label exceeds the head width.
#
# Reference: `tabpfn/architectures/tabpfn_v2_6.py` in the PyPI `tabpfn`
# package (8.2.0). Checkpoints are PyTorch pickles; convert them once with
# `inst/python/tabpfn_convert_ckpt.py`, which writes `arch: "tabpfn_v2_6"`
# into the config so `detect_backend()` can tell the two apart.
#
# Module names below are the checkpoint's names, so no key translation is
# needed: R torch's `nn_sequential` and `nn_module_list` produce the same
# dotted integer paths PyTorch does.

# Sentinels marking NaN / +Inf / -Inf in the indicator channel. Same
# values as v2.5 (see `backend-tabpfn-inputprep.R`), repeated here
# because they belong to the architecture, not to a shared utility.
TABPFN26_NAN_INDICATOR     <- -2.0
TABPFN26_POS_INF_INDICATOR <-  2.0
TABPFN26_NEG_INF_INDICATOR <-  4.0

# `torch.finfo(torch.float32).eps`. Used in two places the reference
# leaves implicit: the standard scaler's denominator, and `nn.RMSNorm`'s
# epsilon, which defaults to the input dtype's eps when not given.
TABPFN26_F32_EPS <- 1.1920928955078125e-07


# ---------------------------------------------------------------------------
# NaN-aware reductions
# ---------------------------------------------------------------------------
# Deliberately not reusing `torch_nanmean_sb()` / `torch_nanstd_sb()` from
# the v2.5 backend: those fold Inf into the NaN mask unconditionally,
# whereas v2.6 has `include_inf` as a per-call choice and its std ignores
# Inf handling entirely. The difference is invisible after imputation but
# not before it.

# Mean over dim 1, ignoring NaN (and, optionally, Inf).
# @keywords internal
tabpfn26_nanmean <- function(x, include_inf = FALSE) {
  mask <- if (include_inf) {
    torch::torch_isfinite(x)$logical_not()
  } else {
    torch::torch_isnan(x)
  }
  zeros <- torch::torch_zeros_like(x)
  n_valid <- torch::torch_where(mask, zeros, torch::torch_ones_like(x))$sum(dim = 1L)
  value_sum <- torch::torch_where(mask, zeros, x)$sum(dim = 1L)
  value_sum / torch::torch_clamp(n_valid, min = 1)
}

# Standard deviation over dim 1, ignoring NaN, with sklearn's (N-1)
# correction.
# @keywords internal
tabpfn26_nanstd <- function(x) {
  mask <- torch::torch_isnan(x)
  zeros <- torch::torch_zeros_like(x)
  n_valid <- torch::torch_where(mask, zeros, torch::torch_ones_like(x))$sum(dim = 1L)
  value_sum <- torch::torch_where(mask, zeros, x)$sum(dim = 1L)
  mean <- value_sum / torch::torch_clamp(n_valid, min = 1)
  diff <- x - mean$unsqueeze(1L)
  sq_diff <- torch::torch_where(mask, zeros, diff * diff)$sum(dim = 1L)
  torch::torch_sqrt(sq_diff / torch::torch_clamp(n_valid - 1, min = 1))
}

# The signed indicator channel: -2 for NaN, +2 for +Inf, +4 for -Inf.
# @keywords internal
tabpfn26_nan_inf_indicator <- function(x) {
  is_nan <- torch::torch_isnan(x)
  is_inf <- torch::torch_isinf(x)
  pos_inf <- is_inf$logical_and(torch::torch_sign(x) == 1)
  neg_inf <- is_inf$logical_and(torch::torch_sign(x) == -1)
  is_nan$to(dtype = x$dtype)  * TABPFN26_NAN_INDICATOR +
    pos_inf$to(dtype = x$dtype) * TABPFN26_POS_INF_INDICATOR +
    neg_inf$to(dtype = x$dtype) * TABPFN26_NEG_INF_INDICATOR
}

# Replace NaN/Inf with the per-feature mean of the first `n_train` rows.
# Returns the imputed tensor and the mask, which the classifier target
# path needs to know which positions to `ceil()`.
#
# `x` is always 3-D here -- (rows, batch * groups, features) for the
# predictors and (rows, batch, 1) for the target.
# @keywords internal
tabpfn26_impute_with_mean <- function(x, n_train) {
  fit_rows <- if (n_train > 0L) x[1:n_train, , ] else x
  feature_means <- tabpfn26_nanmean(fit_rows, include_inf = TRUE)
  bad <- torch::torch_isnan(x)$logical_or(torch::torch_isinf(x))
  list(
    x = torch::torch_where(bad, feature_means$unsqueeze(1L)$expand_as(x), x),
    mask = bad
  )
}


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

#' TabPFN v2.6 multi-head attention
#'
#' Four bias-free linears named exactly as the checkpoint stores them.
#' One module covers both uses because they differ only in how
#' `forward()` is called:
#'
#' * `single_eval_pos = NULL` -- attention between the features of one
#'   row. Ordinary self-attention; every feature sees every other.
#' * `single_eval_pos = n` -- attention between the cells of one column.
#'   Keys and values come from the first `n` rows only (thinking rows +
#'   training rows), which is the implicit mask: test rows cannot see
#'   each other or themselves, and so need no explicit mask. Test queries
#'   additionally attend to the *first key/value head only*, broadcast
#'   across all query heads -- multi-query attention, which is what makes
#'   the reference's KV cache one head wide.
#'
#' @param embedding_dim Model width.
#' @param n_heads Number of heads; must divide `embedding_dim`.
#' @keywords internal
tabpfn26_attention <- torch::nn_module(
  "TabPFN26Attention",

  initialize = function(embedding_dim, n_heads) {
    if (embedding_dim %% n_heads != 0L) {
      cli::cli_abort(
        "embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})."
      )
    }
    self$n_heads  <- as.integer(n_heads)
    self$head_dim <- as.integer(embedding_dim / n_heads)
    inner <- self$n_heads * self$head_dim

    self$q_projection   <- torch::nn_linear(embedding_dim, inner, bias = FALSE)
    self$k_projection   <- torch::nn_linear(embedding_dim, inner, bias = FALSE)
    self$v_projection   <- torch::nn_linear(embedding_dim, inner, bias = FALSE)
    self$out_projection <- torch::nn_linear(inner, embedding_dim, bias = FALSE)
  },

  # (B, S, H, D) -> (B, H, S, D), the layout SDPA wants.
  to_heads = function(x, n_seq) {
    b <- x$size(1)
    x$view(c(b, n_seq, self$n_heads, self$head_dim))$permute(c(1L, 3L, 2L, 4L))
  },

  # @param x `(B, S, E)`.
  # Fold the head axis back and project out. Shared by both directions.
  from_heads = function(ctx, n_seq) {
    b <- ctx$size(1)
    self$out_projection(
      ctx$permute(c(1L, 3L, 2L, 4L))$reshape(c(b, n_seq, self$n_heads * self$head_dim))
    )
  },

  # Attention between the features of one row: ordinary self-attention,
  # every feature sees every other. `forward` is this direction because it
  # is the one the chunked evaluator calls as a plain function.
  # @param x `(B * R, C, E)`.
  forward = function(x) {
    S <- x$size(2)
    q <- self$to_heads(self$q_projection(x), S)
    k <- self$to_heads(self$k_projection(x), S)
    v <- self$to_heads(self$v_projection(x), S)
    ctx <- torch:::torch_scaled_dot_product_attention(
      query = q, key = k, value = v, dropout_p = 0
    )
    self$from_heads(ctx, S)
  },

  # Attention between the cells of one column. Returns `list(out, kv)`.
  #
  # @param x `(B * C, R, E)`. In the cached path `R` counts test rows only.
  # @param single_eval_pos Rows from this index on are test rows. `0` in
  #   the cached path, where every row is a test row.
  # @param cached_kv `list(key, value)` from a previous build pass, holding
  #   the thinking + training rows' projections for the single multi-query
  #   head. When supplied the K/V projections are skipped entirely -- that
  #   is the whole saving.
  # @param return_kv Also return that entry, for building a cache.
  attend_cells = function(x, single_eval_pos, cached_kv = NULL,
                          return_kv = FALSE) {
    dims <- x$size()
    B <- dims[1]; S <- dims[2]
    H <- self$n_heads; D <- self$head_dim
    q <- self$to_heads(self$q_projection(x), S)

    if (!is.null(cached_kv)) {
      # Every row is a test row attending to the cached single head.
      n_kv <- cached_kv$key$size(3)
      k <- cached_kv$key$expand(c(B, H, n_kv, D))
      v <- cached_kv$value$expand(c(B, H, n_kv, D))
      ctx <- torch:::torch_scaled_dot_product_attention(
        query = q, key = k, value = v, dropout_p = 0
      )
      return(list(out = self$from_heads(ctx, S), kv = NULL))
    }

    n_kv <- if (is.null(single_eval_pos) || single_eval_pos >= S) S
            else as.integer(single_eval_pos)
    kv_src <- if (n_kv == S) x else x[, 1:n_kv, ]
    k <- self$to_heads(self$k_projection(kv_src), n_kv)
    v <- self$to_heads(self$v_projection(kv_src), n_kv)

    kv <- NULL
    if (isTRUE(return_kv)) {
      # Only head 0's K/V is worth keeping: test rows attend to that head
      # alone, so the other heads' projections can never be read back.
      kv <- list(key   = k[, 1, , ]$unsqueeze(2L)$detach()$contiguous(),
                 value = v[, 1, , ]$unsqueeze(2L)$detach()$contiguous())
    }

    if (n_kv == S) {
      ctx <- torch:::torch_scaled_dot_product_attention(
        query = q, key = k, value = v, dropout_p = 0
      )
    } else {
      # Train queries: ordinary multi-head. Test queries: multi-query, so
      # head 0's K/V is broadcast across all heads. Float32 has no GQA
      # kernel, and the reference falls back to `repeat_interleave` there,
      # which is what `$expand()` reproduces.
      n_train <- as.integer(single_eval_pos)
      k1 <- k[, 1, , ]$unsqueeze(2L)$expand(c(B, H, n_kv, D))
      v1 <- v[, 1, , ]$unsqueeze(2L)$expand(c(B, H, n_kv, D))
      ctx_train <- torch:::torch_scaled_dot_product_attention(
        query = q[, , 1:n_train, ], key = k, value = v, dropout_p = 0
      )
      ctx_test <- torch:::torch_scaled_dot_product_attention(
        query = q[, , (n_train + 1L):S, ], key = k1, value = v1, dropout_p = 0
      )
      ctx <- torch::torch_cat(list(ctx_train, ctx_test), dim = 3L)
    }
    list(out = self$from_heads(ctx, S), kv = kv)
  }
)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

#' One TabPFN v2.6 block
#'
#' Three post-norm sublayers: attention between a row's features,
#' attention between a column's cells, then a per-token MLP. Each adds its
#' own residual and is followed by an RMSNorm.
#'
#' @param embedding_dim Model width.
#' @param n_heads Number of attention heads.
#' @param mlp_hidden_dim Feedforward width (`2 * embedding_dim` in the
#'   released checkpoints).
#' @param eps RMSNorm epsilon. `NULL` uses float32's, which is what
#'   `nn.RMSNorm` falls back to when none is given.
#' @keywords internal
tabpfn26_block <- torch::nn_module(
  "TabPFN26Block",

  # `eps` defaults to NULL rather than to the constant: an `nn_module`
  # method's *default arguments* are evaluated somewhere that cannot see
  # the package namespace, so a bare package name there resolves under
  # `load_all()` and then fails in the installed package. The body can
  # see it; the signature cannot.
  initialize = function(embedding_dim, n_heads, mlp_hidden_dim, eps = NULL) {
    if (is.null(eps)) eps <- TABPFN26_F32_EPS
    self$per_sample_attention_between_features <-
      tabpfn26_attention(embedding_dim, n_heads)
    self$per_column_attention_between_cells <-
      tabpfn26_attention(embedding_dim, n_heads)

    self$layernorm_mha1 <- rms_norm(embedding_dim, eps = eps)
    self$layernorm_mha2 <- rms_norm(embedding_dim, eps = eps)
    self$layernorm_mlp  <- rms_norm(embedding_dim, eps = eps)

    self$mlp <- torch::nn_sequential(
      torch::nn_linear(embedding_dim, mlp_hidden_dim, bias = FALSE),
      torch::nn_gelu(),
      torch::nn_linear(mlp_hidden_dim, embedding_dim, bias = FALSE)
    )
  },

  # @param x `(B, R, C, E)` -- rows (thinking + train + test) by columns
  #   (feature groups + the target column).
  # @param single_eval_pos Rows from this index on are test rows.
  # @param cached_kv A block's entry from a [tabpfn26_kv_cache()], or NULL.
  # @param return_kv Collect this block's key/value projections.
  # @param save_peak_memory_factor Chunk count for [chunked_evaluate()].
  # @return `list(state, kv)`.
  forward = function(x, single_eval_pos, dump_prefix = NULL,
                     cached_kv = NULL, return_kv = FALSE,
                     save_peak_memory_factor = NULL) {
    dims <- x$size()
    B <- dims[1]; R <- dims[2]; C <- dims[3]; E <- dims[4]
    spmf <- save_peak_memory_factor

    # --- Between a row's features: the rows fold into the batch, so this
    # is independent per row and can be chunked along `B * R`.
    x <- chunked_evaluate(self$per_sample_attention_between_features, x, spmf,
                          residual = TRUE, batch_dims = 2L)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_attn_features"), x)
    # The norms treat every cell on its own, so they fold all three.
    x <- chunked_evaluate(self$layernorm_mha1, x, spmf,
                          residual = FALSE, batch_dims = 3L)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_norm_1"), x)

    # --- Between a column's cells: the columns fold into the batch. ---
    x <- x$transpose(2L, 3L)$contiguous()                       # (B, C, R, E)
    kv <- NULL
    if (isTRUE(return_kv) || !is.null(cached_kv)) {
      # Building or using a cache bypasses chunking: the key/value tensors
      # have to be produced (or read) whole, not a slice at a time.
      res <- self$per_column_attention_between_cells$attend_cells(
        x$reshape(c(B * C, R, E)), single_eval_pos = single_eval_pos,
        cached_kv = cached_kv, return_kv = return_kv
      )
      x <- x + res$out$reshape(c(B, C, R, E))
      kv <- res$kv
    } else {
      x <- chunked_evaluate(
        function(chunk) {
          self$per_column_attention_between_cells$attend_cells(
            chunk, single_eval_pos = single_eval_pos)$out
        },
        x, spmf, residual = TRUE, batch_dims = 2L
      )
    }
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_attn_items"), x)
    x <- chunked_evaluate(self$layernorm_mha2, x, spmf,
                          residual = FALSE, batch_dims = 3L)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_norm_2"), x)
    x <- x$transpose(2L, 3L)$contiguous()                       # (B, R, C, E)

    # --- Per-token MLP. ---
    x <- chunked_evaluate(self$mlp, x, spmf, residual = TRUE, batch_dims = 3L)
    if (!is.null(dump_prefix)) dump_if_enabled(paste0(dump_prefix, "_post_mlp"), x)
    list(state = chunked_evaluate(self$layernorm_mlp, x, spmf,
                                  residual = FALSE, batch_dims = 3L),
         kv = kv)
  }
)


# ---------------------------------------------------------------------------
# Thinking rows
# ---------------------------------------------------------------------------

#' Learnable rows prepended to the dataset
#'
#' 64 extra rows of learned embeddings, shared across all columns, giving
#' the model scratch space before it sees the data. They count as training
#' rows, so `single_eval_pos` shifts by their number.
#' @keywords internal
tabpfn26_thinking_rows <- torch::nn_module(
  "TabPFN26ThinkingRows",

  initialize = function(num_thinking_rows, embedding_dim) {
    self$num_thinking_rows <- as.integer(num_thinking_rows)
    self$row_token_values_TE <- torch::nn_parameter(
      torch::torch_empty(self$num_thinking_rows, embedding_dim)
    )
    torch::nn_init_normal_(self$row_token_values_TE)
  },

  # @param x `(B, Ri, C, E)`.
  forward = function(x, single_eval_pos) {
    dims <- x$size()
    B <- dims[1]; C <- dims[3]; E <- dims[4]
    toks <- self$row_token_values_TE$unsqueeze(1L)$unsqueeze(3L)$
      expand(c(B, self$num_thinking_rows, C, E))
    list(
      state = torch::torch_cat(list(toks, x), dim = 2L),
      single_eval_pos = as.integer(single_eval_pos) + self$num_thinking_rows
    )
  }
)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

#' Everything the training rows contribute to a prediction
#'
#' TabPFN has no weights to fit, so "training" is just conditioning: the
#' training rows sit in the context of every forward pass and the test rows
#' attend to them. That makes prediction cost grow with the *training* set,
#' and makes it grow again for every batch of test rows, since the whole
#' context is rebuilt each time.
#'
#' It does not have to be. The training rows reach the test rows through
#' exactly two channels, and both can be computed once:
#'
#' * the key/value projections of the between-cells attention, one entry
#'   per block. Only the first head is kept, because test rows attend to
#'   that head alone -- the multi-query trick in
#'   [tabpfn26_attention()]'s `attend_cells()` exists for precisely this.
#' * the preprocessing statistics fitted on the training rows: which
#'   columns survived as non-constant, the standard scaler's mean and
#'   standard deviation, and which features within each group carry
#'   information.
#'
#' Plus the embedded all-NaN target, which is the same for every test row.
#'
#' **This is not free of consequences.** Two of those statistics -- the
#' constant-column mask and the within-group informative mask -- are
#' computed over train *and* test rows in an ordinary forward pass, and
#' over the training rows alone when a cache is built. On data where those
#' agree, a cached prediction is bit-identical to an uncached one; where
#' they disagree, it is a different prediction. The clearest case is a
#' column that is constant across the training rows and varies across the
#' test rows: uncached, it survives and is used; cached, it was dropped
#' before the cache existed. That is the reference's own behaviour, and
#' the reason the cache is opt-in rather than automatic.
#'
#' @param kv List of `list(key, value)`, one per block.
#' @param feature_state Fitted preprocessing statistics.
#' @param test_y_embedding `(B, E)` embedded all-NaN target.
#' @param n_train Number of training rows it was built from.
#' @keywords internal
tabpfn26_kv_cache <- function(kv, feature_state, test_y_embedding, n_train) {
  structure(
    list(kv = kv, feature_state = feature_state,
         test_y_embedding = test_y_embedding, n_train = as.integer(n_train)),
    class = "tabpfn26_kv_cache"
  )
}

#' @export
print.tabpfn26_kv_cache <- function(x, ...) {
  n_bytes <- sum(vapply(x$kv, function(e) {
    sum(vapply(e, function(t) prod(as.numeric(t$size())) * 4, numeric(1)))
  }, numeric(1)))
  cli::cli_text("{.strong TabPFN v2.6 KV cache}")
  cli::cli_bullets(c(
    "*" = "built from {.val {x$n_train}} training row{?s}",
    "*" = "{length(x$kv)} block{?s}, {round(n_bytes / 1e6, 1)} MB of key/value projections"
  ))
  invisible(x)
}


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------

#' TabPFN v2.6 network
#'
#' Mirrors `TabPFNV2p6.forward` in the Python reference, minus the KV
#' cache and the memory-saving chunked evaluation -- both are performance
#' paths that produce the same numbers as the plain forward.
#'
#' @param config Parsed `config.json` with `arch == "tabpfn_v2_6"`.
#' @keywords internal
tabpfn_v2_6_transformer <- torch::nn_module(
  "TabPFNv26",

  initialize = function(config) {
    E <- as.integer(config$embedding_dim)
    self$embedding_dim      <- E
    self$features_per_group <- as.integer(config$features_per_group %||% 3L)
    self$head               <- config$head
    self$task_type <- if (identical(config$head, "classifier"))
      "multiclass" else "regression"

    # ENCODING_SIZE_MULTIPLIER = 2: every feature contributes its value and
    # its NaN/Inf indicator.
    encoding_size <- 2L * self$features_per_group
    encoder_type  <- config$encoder_type %||% "linear"
    self$feature_group_embedder <- if (identical(encoder_type, "mlp")) {
      torch::nn_sequential(
        torch::nn_linear(encoding_size,
                         as.integer(config$encoder_mlp_hidden_dim %||% 1024L),
                         bias = FALSE),
        torch::nn_gelu(),
        torch::nn_linear(as.integer(config$encoder_mlp_hidden_dim %||% 1024L),
                         E, bias = FALSE)
      )
    } else {
      torch::nn_linear(encoding_size, E, bias = FALSE)
    }

    self$target_embedder <- torch::nn_linear(2L, E, bias = TRUE)

    self$add_thinking_rows <- tabpfn26_thinking_rows(
      num_thinking_rows = as.integer(config$num_thinking_rows %||% 64L),
      embedding_dim     = E
    )

    ff <- as.integer(config$mlp_hidden_dim %||% (2L * E))
    self$blocks <- torch::nn_module_list(
      lapply(seq_len(as.integer(config$n_layers)), function(i) {
        tabpfn26_block(
          embedding_dim  = E,
          n_heads        = as.integer(config$n_heads),
          mlp_hidden_dim = ff
        )
      })
    )

    n_out <- if (identical(self$head, "classifier")) {
      as.integer(config$n_out_classes %||% 10L)
    } else {
      as.integer(config$n_bar_bins %||% 5000L)
    }
    self$n_out <- n_out
    self$output_projection <- torch::nn_sequential(
      torch::nn_linear(E, ff, bias = TRUE),
      torch::nn_gelu(),
      torch::nn_linear(ff, n_out, bias = TRUE)
    )

    # "subspace" feature positional embedding: a fixed per-column vector of
    # width E/4, projected up to E.
    self$pos_emb_dim <- as.integer(E %/% 4L)
    self$feature_positional_embedding_embeddings <-
      torch::nn_linear(self$pos_emb_dim, E, bias = TRUE)

    if (identical(self$head, "regressor")) {
      self$criterion <- bar_distribution_criterion(
        n_bar_bins = as.integer(config$n_bar_bins %||% 5000L)
      )
    }
    # Read by the shared predictors. v2.5 has the cache but not the
    # chunked forward, so the two are advertised separately.
    self$needs_column_embeddings <- TRUE
    self$supports_kv_cache <- TRUE
    self$supports_chunked_eval <- TRUE
  },

  # @param x_train `(B, n_train, F)`; @param y_train `(B, n_train)` or
  #   `(B, n_train, 1)`; @param x_test `(B, n_test, F)`. With `kv_cache`
  #   supplied, `x_train` and `y_train` are ignored -- everything the
  #   training rows contribute is already in the cache -- and `x_test`
  #   holds the rows to predict.
  # @param column_embeddings Optional `(2000, E/4)` pre-generated subspace
  #   embeddings; see [load_column_embeddings()].
  # @param kv_cache A [tabpfn26_kv_cache()] to predict against.
  # @param return_kv_cache Build and return one. `x_test` may be empty,
  #   which is the usual way to build from training rows alone.
  # @param save_peak_memory_factor Chunk count for [chunked_evaluate()].
  forward = function(x_train, y_train, x_test, column_embeddings = NULL,
                     kv_cache = NULL, return_kv_cache = FALSE,
                     save_peak_memory_factor = NULL) {
    if (!is.null(kv_cache)) {
      return(self$forward_cached(x_test, kv_cache,
                                 column_embeddings = column_embeddings,
                                 save_peak_memory_factor = save_peak_memory_factor))
    }
    device <- x_train$device
    dims_train <- x_train$size()
    B <- dims_train[1]; n_train <- dims_train[2]
    n_test <- x_test$size(2)
    n_total <- n_train + n_test

    if (B != 1L) {
      cli::cli_abort(c(
        "The TabPFN v2.6 backend runs one dataset at a time.",
        i = "Got a batch of {B}; call it once per dataset."
      ))
    }
    if (identical(self$task_type, "multiclass")) {
      if (as.logical((y_train > (self$n_out - 1L))$any()$cpu())) {
        cli::cli_abort(c(
          "Target out of range for a {self$n_out}-class head.",
          i = "Labels must be ordinal-encoded in {.val {0}}..{.val {self$n_out - 1L}}."
        ))
      }
    }

    # The reference works in (rows, batch, columns) throughout.
    x_RiBC <- torch::torch_cat(list(x_train, x_test), dim = 2L)$
      transpose(1L, 2L)$contiguous()
    dump_if_enabled("input_x_raw", x_RiBC)

    feat <- self$embed_features(
      x_RiBC, n_train = n_train, column_embeddings = column_embeddings,
      device = device
    )
    embedded_y <- self$embed_targets(y_train, n_total = n_total, n_train = n_train)

    # The target rides along as one extra column.
    state <- torch::torch_cat(
      list(feat$embedded, embedded_y$unsqueeze(3L)), dim = 3L
    )                                                     # (B, Ri, G + 1, E)
    dump_if_enabled("embedded_input_pre_thinking", state)

    if (as.logical(torch::torch_isnan(state)$any()$cpu())) {
      cli::cli_abort("NaN in the encoded input; this is a bug in the preprocessing.")
    }

    thinking <- self$add_thinking_rows(state, single_eval_pos = n_train)
    state <- thinking$state
    single_eval_pos <- thinking$single_eval_pos
    n_rows_total <- n_total + self$add_thinking_rows$num_thinking_rows
    dump_if_enabled("embedded_input_post_thinking", state)

    n_blocks <- length(self$blocks)
    dump_at <- c(1L, (n_blocks %/% 2L) + 1L, n_blocks)
    kv_out <- if (isTRUE(return_kv_cache)) vector("list", n_blocks) else NULL
    for (i in seq_len(n_blocks)) {
      res <- self$blocks[[i]](
        state, single_eval_pos = single_eval_pos,
        dump_prefix = if (i == 1L) "layer0_internal" else NULL,
        return_kv = isTRUE(return_kv_cache),
        save_peak_memory_factor = save_peak_memory_factor
      )
      state <- res$state
      if (isTRUE(return_kv_cache)) kv_out[[i]] <- res$kv
      collect_between_layers(state)
      if (i %in% dump_at) {
        dump_if_enabled(
          if (i == 1L) "layer0_out"
          else if (i == n_blocks) "layer_final_out" else "layer_mid_out",
          state
        )
      }
    }

    out <- if (n_test > 0L) {
      self$decode(state, test_start = single_eval_pos, n_rows_total = n_rows_total)
    } else {
      list(logits = NULL, test_hidden = NULL)
    }

    res <- list(
      logits          = out$logits,
      test_hidden     = out$test_hidden,
      encoder_out     = state,
      single_eval_pos = single_eval_pos,
      n_thinking_rows = self$add_thinking_rows$num_thinking_rows
    )
    if (isTRUE(return_kv_cache)) {
      # The embedded target of a test row is the same for every test row --
      # an all-NaN label, imputed from the training statistics -- so one
      # copy is enough. Embed a single padded row and keep the last.
      test_y <- self$embed_targets(y_train, n_total = n_train + 1L,
                                   n_train = n_train)[, n_train + 1L, ]$detach()
      res$kv_cache <- tabpfn26_kv_cache(
        kv = kv_out, feature_state = feat$state,
        test_y_embedding = test_y, n_train = n_train
      )
    }
    res
  },

  #' Predict test rows against a prebuilt cache.
  #'
  #' The training rows never enter this pass. Their contribution reaches
  #' the test rows through exactly two channels, and the cache holds both:
  #' the key/value projections every test row attends to, and the
  #' preprocessing statistics fitted on them.
  #' @keywords internal
  forward_cached = function(x_test, cache, column_embeddings = NULL,
                            save_peak_memory_factor = NULL) {
    device <- x_test$device
    B <- x_test$size(1); n_test <- x_test$size(2)
    if (B != 1L) {
      cli::cli_abort("The TabPFN v2.6 backend runs one dataset at a time.")
    }

    x_RiBC <- x_test$transpose(1L, 2L)$contiguous()
    feat <- self$embed_features(
      x_RiBC, n_train = 0L, column_embeddings = column_embeddings,
      device = device, state = cache$feature_state
    )
    # Every test row carries the same embedded all-NaN target.
    test_y <- cache$test_y_embedding$to(device = device)$
      unsqueeze(2L)$expand(c(B, n_test, self$embedding_dim))
    state <- torch::torch_cat(list(feat$embedded, test_y$unsqueeze(3L)), dim = 3L)

    for (i in seq_along(self$blocks)) {
      state <- self$blocks[[i]](
        state, single_eval_pos = 0L, cached_kv = cache$kv[[i]],
        save_peak_memory_factor = save_peak_memory_factor
      )$state
      collect_between_layers(state)
    }

    # No thinking or training rows here: every row is a test row.
    out <- self$decode(state, test_start = 0L, n_rows_total = n_test)
    list(logits = out$logits, test_hidden = out$test_hidden,
         encoder_out = state, single_eval_pos = 0L,
         n_thinking_rows = self$add_thinking_rows$num_thinking_rows)
  },

  #' Project the target column of the test rows to outputs.
  #' @keywords internal
  decode = function(state, test_start, n_rows_total) {
    test_hidden <- state[, (test_start + 1L):n_rows_total, -1, ]  # (B, n_test, E)
    dump_if_enabled("test_hidden", test_hidden)
    # The reference projects in (rows, batch, embedding) order; keep the
    # same layout so the matmul sees the same shapes.
    logits <- self$output_projection(test_hidden$transpose(1L, 2L))$
      transpose(1L, 2L)                                           # (B, n_test, n_out)
    dump_if_enabled("logits", logits)
    list(logits = logits, test_hidden = test_hidden)
  },

  #' Preprocess and embed the feature columns.
  #'
  #' Mirrors `_preprocess_and_embed_features`: drop constant columns,
  #' group features, impute and flag NaN/Inf, standard-scale on the train
  #' rows, rescale each group by how many of its features are informative,
  #' then embed and add the column positional embedding.
  #' @keywords internal
  embed_features = function(x_RiBC, n_train, column_embeddings, device,
                            state = NULL) {
    dims <- x_RiBC$size()
    Ri <- dims[1]; B <- dims[2]
    using_state <- !is.null(state)

    # --- Constant columns carry no information and are dropped. The mask
    # is computed over train *and* test rows, as in the reference -- which
    # is why it has to be captured for the cached path rather than
    # recomputed there from test rows alone.
    keep <- if (using_state) state$keep else NULL
    if (is.null(keep)) {
      keep <- if (Ri > 1L) {
        as.logical(as.array(
          (x_RiBC[2:Ri, , ] == x_RiBC[1, , ]$unsqueeze(1L))$all(dim = 1L)$
            logical_not()[1, ]$cpu()
        ))
      } else {
        rep(TRUE, x_RiBC$size(3))
      }
      if (!any(keep)) {
        cli::cli_abort(c(
          "Every predictor is constant across the supplied rows.",
          i = "There is nothing for the model to condition on."
        ))
      }
    }
    if (!all(keep)) {
      idx <- torch::torch_tensor(which(keep), dtype = torch::torch_long(),
                                 device = device)
      x_RiBC <- x_RiBC$index_select(dim = 3L, index = idx)
    }

    # --- Group features, zero-padding the last group.
    g <- self$features_per_group
    n_col <- x_RiBC$size(3)
    pad <- (g - (n_col %% g)) %% g
    if (pad > 0L) {
      x_RiBC <- torch::torch_cat(
        list(x_RiBC, torch::torch_zeros(c(Ri, B, pad), device = device,
                                        dtype = x_RiBC$dtype)),
        dim = -1L
      )
    }
    n_groups <- as.integer((n_col + pad) / g)
    x <- x_RiBC$reshape(c(Ri, B * n_groups, g))

    nan_indicators <- tabpfn26_nan_inf_indicator(x)
    dump_if_enabled("preproc_x_nan_indicators", nan_indicators)

    # --- Impute, then scale. Imputing with the train mean leaves that mean
    # unchanged, which is why the reference fits the scaler afterwards --
    # and why the cached path can reuse the scaler's mean to impute with.
    if (using_state) {
      bad <- torch::torch_isnan(x)$logical_or(torch::torch_isinf(x))
      x <- torch::torch_where(bad, state$mean$unsqueeze(1L)$expand_as(x), x)
      mean <- state$mean; std <- state$std
    } else {
      x <- tabpfn26_impute_with_mean(x, n_train = n_train)$x
      fit_rows <- if (n_train > 0L) x[1:n_train, , ] else x
      mean <- tabpfn26_nanmean(fit_rows)
      std  <- tabpfn26_nanstd(fit_rows)
      std  <- torch::torch_where(std == 0, torch::torch_ones_like(std), std)
      if (fit_rows$size(1) == 1L) std <- torch::torch_ones_like(std)
    }
    dump_if_enabled("preproc_x_after_nan_handle", x)

    x <- (x - mean$unsqueeze(1L)) / (std$unsqueeze(1L) + TABPFN26_F32_EPS)
    x <- torch::torch_clamp(x, min = -100, max = 100)
    dump_if_enabled("preproc_x_clipped", x)

    # --- Rescale each group by sqrt(g / informative features in it), and
    # zero the uninformative ones. Unlike v2.5 the mask spans all rows,
    # so it too is captured rather than recomputed on test-only data.
    if (using_state) {
      non_const <- state$non_const
    } else if (Ri > 1L) {
      non_const <- (x[2:Ri, , ] == x[1, , ]$unsqueeze(1L))$
        to(dtype = torch::torch_long())$sum(dim = 1L) != (Ri - 1L)
    } else {
      non_const <- torch::torch_zeros_like(x[1, , ], dtype = torch::torch_bool())
    }
    n_used <- torch::torch_clamp(
      non_const$to(dtype = torch::torch_long())$sum(dim = -1L, keepdim = TRUE),
      min = 1
    )
    scale <- g / n_used$to(dtype = x$dtype)
    x <- x * torch::torch_sqrt(scale)$unsqueeze(1L)
    x <- torch::torch_where(non_const$unsqueeze(1L)$expand_as(x), x,
                            torch::torch_zeros_like(x))
    dump_if_enabled("preproc_x_after_norm_groups_2", x)

    # --- Embed.
    x <- torch::torch_cat(list(x, nan_indicators), dim = -1L)
    dump_if_enabled("encoder_linear_in", x)
    x <- self$feature_group_embedder(x)                      # (Ri, B * G, E)
    dump_if_enabled("embedded_x_pre_pos", x)

    embedded <- x$view(c(Ri, B, n_groups, self$embedding_dim))$
      transpose(1L, 2L)                                      # (B, Ri, G, E)

    pos <- self$column_positional_embedding(
      n_groups, column_embeddings, device = device, dtype = embedded$dtype
    )
    dump_if_enabled("positional_embedding_linear_out", pos)
    list(
      embedded = embedded + pos$unsqueeze(1L)$unsqueeze(1L),
      # Everything fitted here that a test-only pass could not recompute
      # for itself. `mean` doubles as the imputation value, which is exact:
      # imputing with the train mean leaves that mean where it was.
      state = list(keep = keep, mean = mean, std = std, non_const = non_const)
    )
  },

  #' Preprocess and embed the target column.
  #'
  #' Mirrors `_preprocess_and_embed_targets`. Test rows have no label, so
  #' they arrive as NaN, get flagged in the indicator channel, and are
  #' imputed with the training mean. For classification the imputed value
  #' is then rounded up, which is what the reference does for backwards
  #' compatibility -- note it rounds only the imputed positions.
  #' @keywords internal
  embed_targets = function(y_train, n_total, n_train) {
    B <- y_train$size(1)
    y <- y_train$reshape(c(B, n_train, 1L))$transpose(1L, 2L)$contiguous()
    if (n_total > n_train) {
      y <- torch::torch_cat(
        list(y, torch::torch_full(c(n_total - n_train, B, 1L), NaN,
                                  device = y$device, dtype = y$dtype)),
        dim = 1L
      )
    }

    nan_indicators <- tabpfn26_nan_inf_indicator(y)
    imputed <- tabpfn26_impute_with_mean(y, n_train = n_train)
    y <- imputed$x
    if (identical(self$task_type, "multiclass")) {
      y <- torch::torch_where(imputed$mask, y$ceil(), y)
    }
    dump_if_enabled("preproc_y_after_nan_handle", y)
    dump_if_enabled("preproc_y_nan_indicators", nan_indicators)

    y <- torch::torch_cat(list(y, nan_indicators), dim = -1L)
    dump_if_enabled("y_encoder_linear_in", y)
    embedded <- self$target_embedder(y)                      # (Ri, B, E)
    dump_if_enabled("embedded_y", embedded)
    embedded$transpose(1L, 2L)                               # (B, Ri, E)
  },

  #' The per-column "subspace" positional embedding.
  #'
  #' The reference draws these from a seeded generator and then overwrites
  #' the first 2000 rows with a buffer shipped in the package, because a
  #' seeded draw is not reproducible across devices. We only have the
  #' buffer, so columns past its length would need a draw R cannot match;
  #' the model tops out well below that in practice, and asking for more
  #' is refused rather than silently approximated.
  #' @keywords internal
  column_positional_embedding = function(n_groups, column_embeddings, device, dtype) {
    if (is.null(column_embeddings)) {
      cli::cli_abort(
        "TabPFN v2.6 needs the pre-generated column embeddings; \\
         see {.fn load_column_embeddings}."
      )
    }
    if (column_embeddings$size(2) != self$pos_emb_dim) {
      cli::cli_abort(c(
        "The column embeddings are {column_embeddings$size(2)} wide but this \\
         model needs {self$pos_emb_dim}.",
        i = "The width is {.code embedding_dim / 4}."
      ))
    }
    available <- column_embeddings$size(1)
    if (n_groups > available) {
      cli::cli_abort(c(
        "{n_groups} feature groups exceeds the {available} pre-generated \\
         column embeddings.",
        i = "The reference falls back to a device-dependent random draw \\
             beyond this point, which cannot be reproduced here.",
        i = "Reduce the predictor count to at most \\
             {available * self$features_per_group}."
      ))
    }
    embs <- column_embeddings[1:n_groups, ]$to(device = device, dtype = dtype)
    self$feature_positional_embedding_embeddings(embs)       # (G, E)
  }
)


# ---------------------------------------------------------------------------
# Backend hooks
# ---------------------------------------------------------------------------

#' @keywords internal
tabpfn26_build <- function(config, task) {
  cli::cli_alert_info(
    "Building tabpfn_v2_6 ({.val {config$head}}, \\
     {config$n_layers} layers, emb={config$embedding_dim})..."
  )
  tabpfn_v2_6_transformer(config)
}

#' @keywords internal
tabpfn26_detect <- function(config) {
  identical(config$arch, "tabpfn_v2_6")
}


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

#' Stage list for the TabPFN v2.6 diagram
#'
#' The same two-way shape as v2.5 -- attend across a row's features, then
#' down a column's cells -- restated with RMSNorm, no biases anywhere,
#' and an embedder that may be an MLP rather than a single linear. The
#' visible difference in the diagram is the target: v2.6 embeds it as a
#' column of its own and reads the prediction back out of that column.
#'
#' @param config The checkpoint's parsed `config.json`.
#' @param task `"classification"` or `"regression"`.
#' @keywords internal
tabpfn26_describe <- function(config, task) {
  E   <- as.integer(config$embedding_dim)
  H   <- as.integer(config$n_heads)
  L   <- as.integer(config$n_layers)
  ff  <- as.integer(config$mlp_hidden_dim %||% (2L * E))
  grp <- as.integer(config$features_per_group %||% 3L)
  thk <- as.integer(config$num_thinking_rows %||% 64L)
  mlp_enc <- identical(config$encoder_type %||% "linear", "mlp")
  clf <- identical(config$head, "classifier")
  n_out <- if (clf) as.integer(config$n_out_classes %||% 10L)
           else as.integer(config$n_bar_bins %||% 5000L)
  blk <- "blocks.0"

  stages <- list(
    arch_input_stage(),
    arch_stage(
      "embed", "Feature-group embedder", kind = "embed", group = "Embed",
      detail = if (mlp_enc)
        sprintf("MLP(%d -> %d -> %d), no bias: %d values + %d NaN flags",
                2L * grp, as.integer(config$encoder_mlp_hidden_dim %||% 1024L),
                E, grp, grp)
      else sprintf("Linear(%d -> %d), no bias: %d values + %d NaN flags",
                   2L * grp, E, grp, grp),
      shape = "(B, n, F, E)", prefix = "feature_group_embedder"),
    arch_stage(
      "y_embed", "Target embedder", kind = "embed", group = "Embed",
      detail = "Linear(2 -> E); the target rides along as its own column",
      prefix = "target_embedder"),
    arch_stage(
      "pos", "Feature positional embedding", kind = "embed", group = "Embed",
      # The subspace is 48-dimensional whatever the model width is; only
      # the projection out of it scales with E.
      detail = sprintf("subspace: fixed 48-d per column, Linear(48 -> %d)", E),
      prefix = "feature_positional_embedding_embeddings"),
    arch_stage(
      "think", "Thinking rows", kind = "embed", group = "Embed",
      detail = sprintf("%d learned rows prepended to the context", thk),
      shape = sprintf("(B, %d + n, F + 1, E)", thk),
      prefix = "add_thinking_rows"),
    arch_stage(
      "blocks", "Block", kind = "attention", group = "Transformer",
      repeats = L,
      detail = sprintf("post-norm RMSNorm; %d heads of %d; MLP %d -> %d",
                       H, as.integer(E / H), E, ff),
      shape = sprintf("(B, %d + n, F + 1, E)", thk), prefix = "blocks",
      children = list(
        arch_stage("a_feat", "Attention between features", kind = "attention",
                   axis = "features",
                   prefix = paste0(blk, ".per_sample_attention_between_features")),
        arch_stage("rms1", "RMSNorm", kind = "norm",
                   prefix = paste0(blk, ".layernorm_mha1")),
        arch_stage("a_cell", "Attention between cells", kind = "attention",
                   axis = "cells",
                   prefix = paste0(blk, ".per_column_attention_between_cells")),
        arch_stage("rms2", "RMSNorm", kind = "norm",
                   prefix = paste0(blk, ".layernorm_mha2")),
        arch_stage("mlp", "MLP, GELU, no bias", kind = "ffn",
                   prefix = paste0(blk, ".mlp")),
        arch_stage("rms3", "RMSNorm", kind = "norm",
                   prefix = paste0(blk, ".layernorm_mlp"))
      )),
    arch_stage(
      "out", "Output projection", kind = "decode", group = "Decode",
      detail = sprintf("target column only: %d -> %d -> %d, GELU", E, ff, n_out),
      shape = sprintf("(n_test, %d)", n_out),
      prefix = "output_projection"),
    if (clf)
      arch_output_stage("Class logits",
                        detail = sprintf("%d slots; softmax over the classes seen", n_out))
    else
      arch_output_stage("Bar distribution",
                        detail = sprintf("%d bins over the scaled target", n_out))
  )

  list(
    title = sprintf("TabPFN v2.6 - %s", if (clf) "classifier" else "regressor"),
    subtitle = "Prior-Labs  -  per-feature transformer, two-way attention",
    facts = c(
      "Layers"              = L,
      "Embedding width"     = E,
      "Attention heads"     = sprintf("%d x %d", H, as.integer(E / H)),
      "MLP width"           = ff,
      "Features per group"  = grp,
      "Thinking rows"       = thk,
      "Attention per layer" = "2 (features, cells)",
      "Output width"        = n_out,
      "Normalisation"       = "RMSNorm, post",
      "KV cache"            = "1 head wide; not exact"
    ),
    stages = Filter(Negate(is.null), stages)
  )
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#' @keywords internal
register_tabpfn26_backend <- function() {
  register_backend(
    name          = "tabpfn26",
    build         = tabpfn26_build,
    describe      = tabpfn26_describe,
    # R torch reproduces PyTorch's dotted paths for `nn_sequential` and
    # `nn_module_list`, and every module here is named after the
    # checkpoint, so nothing needs rewriting.
    translate_key = identity,
    detect        = tabpfn26_detect,
    task_of       = tabpfn_task_of,
    classifier    = tabpfn_classifier,
    regressor     = tabpfn_regressor,
    peak_terms    = tabpfn_peak_terms,
    # NaN and Inf are first-class inputs: imputed with the train mean and
    # flagged in a dedicated indicator channel.
    handles_missing = TRUE,
    description   = "TabPFN v2.6 (Prior-Labs)",
    parity        = "tabpfn 8.2.0 (PyPI)"
  )
}
