# Top-level `per_feature_transformer`.
#
# The forward pass mirrors `PerFeatureTransformer.forward` in the
# Python reference at
# `tabpfn/architectures/base/transformer.py`:
#
#   1. Pad the feature count to a multiple of `features_per_group`
#      (3), reshape to (B, N, F_groups, 3).
#   2. Encode X -> (B, N, F_groups, emb).
#   3. Encode y -> (B, N, emb); test-position y is NaN (handled by the
#      y encoder's NaN step).
#   4. Add "subspace" feature positional embedding to X only.
#   5. Concatenate y as an extra feature-axis token:
#      (B, N, F_groups + 1, emb).
#   6. Prepend thinking tokens along the sample axis; bump
#      `single_eval_pos` by `num_thinking_rows`.
#   7. Run the transformer encoder (post-norm, stateless LayerNorm).
#   8. Slice the test positions of the y-token column and pass through
#      `decoder_dict$standard` to get the output logits.

# ---------------------------------------------------------------------------
# Thinking tokens
# ---------------------------------------------------------------------------

#' Learnable "thinking tokens" prepended along the sample axis
#' @keywords internal
add_thinking_tokens <- torch::nn_module(
  "AddThinkingTokens",

  initialize = function(num_thinking_rows, embedding_dim) {
    self$num_thinking_rows <- as.integer(num_thinking_rows)
    self$row_token_values <- torch::nn_parameter(
      torch::torch_empty(self$num_thinking_rows, embedding_dim)
    )
    torch::nn_init_normal_(self$row_token_values)
  },

  # @param embedded_input `(B, N, F_groups, emb)`.
  # @param single_eval_pos Integer — number of train rows before tokens.
  forward = function(embedded_input, single_eval_pos) {
    dims <- embedded_input$size()
    B <- dims[1]; num_features <- dims[3]; E <- dims[4]

    # (num_thinking_rows, emb) -> (1, num_thinking_rows, 1, emb) ->
    # (B, num_thinking_rows, num_features, emb)
    toks <- self$row_token_values$unsqueeze(1L)$unsqueeze(3L)
    toks <- toks$expand(c(B, self$num_thinking_rows, num_features, E))

    list(
      state = torch::torch_cat(list(toks, embedded_input), dim = 2L),
      single_eval_pos = as.integer(single_eval_pos) + self$num_thinking_rows
    )
  }
)


# ---------------------------------------------------------------------------
# Items-axis attention mask (blocks test columns for every query)
# ---------------------------------------------------------------------------

#' Build additive (1, 1, n_total, n_total) attention mask
#'
#' Rows `[single_eval_pos+1, n_total]` are test rows; blocking those
#' *columns* for everyone prevents test→test leakage and also prevents
#' train rows from accidentally attending to test. Columns
#' `[1, single_eval_pos]` (= thinking tokens + train rows) remain open.
#'
#' @keywords internal
build_items_attn_mask <- function(single_eval_pos, n_total, device = "cpu") {
  mask <- torch::torch_zeros(c(n_total, n_total), device = device)
  if (single_eval_pos < n_total) {
    neg_inf <- torch::torch_tensor(-Inf, device = device)
    mask[, (single_eval_pos + 1L):n_total] <- neg_inf
  }
  mask$unsqueeze(1L)$unsqueeze(1L)
}


# ---------------------------------------------------------------------------
# Decoder head (matches `decoder_dict.standard` in the ckpt)
# ---------------------------------------------------------------------------

#' Standard decoder: Linear(emb→384) → GELU → Linear(384→n_out)
#'
#' Lives at `decoder_dict.standard` in the state dict — indices 0
#' (first linear) and 2 (second linear) — with a GELU at index 1 that
#' has no parameters.
#' @keywords internal
standard_decoder <- torch::nn_module(
  "StandardDecoder",

  initialize = function(embedding_dim, hidden_dim, n_out, activation = "gelu") {
    self$steps <- torch::nn_module_list(list(
      torch::nn_linear(embedding_dim, hidden_dim, bias = TRUE),
      if (activation == "gelu") torch::nn_gelu() else torch::nn_relu(),
      torch::nn_linear(hidden_dim, n_out, bias = TRUE)
    ))
  },

  forward = function(x) {
    for (i in seq_along(self$steps)) {
      x <- self$steps[[i]](x)
    }
    x
  }
)


#' Wraps `{standard: standard_decoder}` so the state_dict prefix is
#' `decoder_dict.standard.*`.
#' @keywords internal
decoder_dict_module <- torch::nn_module(
  "DecoderDict",
  initialize = function(embedding_dim, hidden_dim, n_out, activation = "gelu") {
    self$standard <- standard_decoder(embedding_dim, hidden_dim, n_out, activation)
  },
  forward = function(x) self$standard(x)
)


# ---------------------------------------------------------------------------
# Criterion (regressor only) — bar distribution borders buffer
# ---------------------------------------------------------------------------

#' Registers `borders` and `losses_per_bucket` as buffers so keys
#' `criterion.borders` and `criterion.losses_per_bucket` are consumed
#' by the loader.
#' @keywords internal
bar_distribution_criterion <- torch::nn_module(
  "BarDistributionCriterion",

  initialize = function(n_bar_bins) {
    self$register_buffer("borders", torch::torch_zeros(n_bar_bins + 1L))
    self$register_buffer("losses_per_bucket", torch::torch_zeros(n_bar_bins))
  },

  forward = function(...) {
    cli::cli_abort("bar_distribution_criterion has no forward; use the decoder.")
  }
)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

#' Everything the training rows contribute to a v2 / v2.5 prediction
#'
#' The same idea as [tabpfn26_kv_cache()], and it lands more cleanly here.
#' In this architecture every statistic the network fits comes from the
#' training rows alone -- the imputation mean, the z-normalisation mean and
#' standard deviation, and the per-group non-constant mask (see
#' [preprocess_x_for_encoder()]). Nothing is fitted over train and test
#' together, so there is no case where conditioning once and reusing it
#' answers a different question than a full forward pass would. v2.6 moved
#' two of those masks to a train+test fit and pays for it; v2.5 does not.
#'
#' What is stored, per layer: head 0 of the between-items key/value
#' projections over the thinking tokens and training rows. Test rows
#' attend to that head alone -- the multi-query path in
#' [mha_fused_qkv()] -- so nothing else can be read back.
#'
#' @param kv List of `list(key, value)`, one per layer.
#' @param feature_state Fitted preprocessing statistics.
#' @param test_y_embedding `(B, emb)` embedding of an absent label.
#' @param n_train Number of training rows it was built from.
#' @param n_feature_groups Feature-group count it was built for; test rows
#'   with a different one cannot use it.
#' @keywords internal
tabpfn_kv_cache <- function(kv, feature_state, test_y_embedding, n_train,
                            n_feature_groups) {
  structure(
    list(kv = kv, feature_state = feature_state,
         test_y_embedding = test_y_embedding, n_train = as.integer(n_train),
         n_feature_groups = as.integer(n_feature_groups)),
    class = "tabpfn_kv_cache"
  )
}

#' @export
print.tabpfn_kv_cache <- function(x, ...) {
  n_bytes <- sum(vapply(x$kv, function(e) {
    sum(vapply(e, function(t) prod(as.numeric(t$size())) * 4, numeric(1)))
  }, numeric(1)))
  cli::cli_text("{.strong TabPFN v2/v2.5 KV cache}")
  cli::cli_bullets(c(
    "*" = "built from {.val {x$n_train}} training row{?s}",
    "*" = "{length(x$kv)} layer{?s}, {round(n_bytes / 1e6, 1)} MB of key/value projections"
  ))
  invisible(x)
}


# ---------------------------------------------------------------------------
# Top-level transformer
# ---------------------------------------------------------------------------

#' Per-feature transformer (top-level)
#' @keywords internal
per_feature_transformer <- torch::nn_module(
  "PerFeatureTransformer",

  initialize = function(config) {
    self$embedding_dim       <- as.integer(config$embedding_dim)
    self$features_per_group  <- as.integer(config$features_per_group %||% 3L)
    self$head                <- config$head
    self$n_thinking_rows     <- 64L   # matches [64, 192] row_token_values

    self$encoder <- input_encoder(
      embedding_dim     = self$embedding_dim,
      features_per_group = self$features_per_group,
      head              = self$head,
      mlp_hidden_dim    = config$encoder_mlp_hidden_dim %||%
                           config$mlp_hidden_dim %||% 1024L
    )

    self$y_encoder <- y_encoder(
      embedding_dim = self$embedding_dim,
      head          = self$head
    )

    self$add_thinking_tokens <- add_thinking_tokens(
      num_thinking_rows = self$n_thinking_rows,
      embedding_dim     = self$embedding_dim
    )

    # "subspace" positional embedding: Linear(48 -> 192) applied to a
    # random per-feature 48-dim tensor (seeded, fixed at inference).
    self$feature_positional_embedding_kind <- config$feature_positional_embedding %||% "subspace"
    self$feature_positional_embedding_embeddings <- torch::nn_linear(48L, self$embedding_dim, bias = TRUE)

    self$transformer_encoder <- tabpfn_layer_stack(
      n_layers       = config$n_layers,
      embedding_dim  = self$embedding_dim,
      n_heads        = config$n_heads,
      mlp_hidden_dim = config$mlp_hidden_dim,
      layer_norm_eps = config$layer_norm_eps %||% 1e-5,
      activation     = config$activation %||% "gelu"
    )

    n_out <- if (self$head == "classifier") {
      as.integer(config$n_out_classes %||% 10L)
    } else {
      as.integer(config$n_bar_bins %||% 5000L)
    }
    self$decoder_dict <- decoder_dict_module(
      embedding_dim = self$embedding_dim,
      hidden_dim    = 384L,
      n_out         = n_out,
      activation    = config$activation %||% "gelu"
    )

    if (self$head == "regressor") {
      self$criterion <- bar_distribution_criterion(
        n_bar_bins = as.integer(config$n_bar_bins %||% 5000L)
      )
    }
    # Read by the shared predictors.
    self$needs_column_embeddings <- TRUE
    self$supports_kv_cache <- TRUE
    self$supports_chunked_eval <- TRUE
    # And it is exactly equivalent here, not merely cheaper: every
    # statistic this architecture fits comes from the training rows alone
    # (see `preprocess_x_for_encoder()`), so conditioning once cannot
    # answer a different question. v2.6 fits two of its masks over train
    # and test together and so cannot claim this.
    self$kv_cache_is_exact <- TRUE
  },

  # @param x_train `(B, n_train, F)`.
  # @param y_train `(B, n_train)` or `(B, n_train, 1)`.
  # @param x_test  `(B, n_test,  F)`.
  # @param column_embeddings Optional `(2000, 48)` tensor of pre-generated
  #'   subspace embeddings. If NULL, random draws are used (less stable).
  # @param kv_cache A [tabpfn_kv_cache()] to predict against. When given,
  #   `x_train` and `y_train` are ignored and `x_test` holds the rows to
  #   predict.
  # @param return_kv_cache Build and return one. `x_test` may be empty,
  #   which is how a cache is built from training rows alone.
  forward = function(x_train, y_train, x_test, column_embeddings = NULL,
                     kv_cache = NULL, return_kv_cache = FALSE,
                     save_peak_memory_factor = NULL) {
    if (!is.null(kv_cache)) {
      return(self$forward_cached(x_test, kv_cache,
                                 column_embeddings = column_embeddings,
                                 save_peak_memory_factor =
                                   save_peak_memory_factor))
    }
    device <- x_train$device

    if (y_train$dim() == 2L) y_train <- y_train$unsqueeze(-1L)

    dims_train <- x_train$size()
    dims_test  <- x_test$size()
    B <- dims_train[1]; n_train <- dims_train[2]; F_raw <- dims_train[3]
    n_test <- dims_test[2]
    n_total <- n_train + n_test

    # --- checkpoint 1: raw input X (pre-pad, pre-grouping) ---
    dump_if_enabled(
      "input_x_raw",
      torch::torch_cat(list(x_train, x_test), dim = 2L)
    )

    # --- Pad features to multiple of features_per_group ---
    g <- self$features_per_group
    missing_to_next <- (g - (F_raw %% g)) %% g
    if (missing_to_next > 0L) {
      pad_train <- torch::torch_zeros(c(B, n_train, missing_to_next),
                                       device = device, dtype = x_train$dtype)
      pad_test  <- torch::torch_zeros(c(B, n_test,  missing_to_next),
                                       device = device, dtype = x_test$dtype)
      x_train <- torch::torch_cat(list(x_train, pad_train), dim = -1L)
      x_test  <- torch::torch_cat(list(x_test,  pad_test),  dim = -1L)
    }
    F_padded <- F_raw + missing_to_next
    F_groups <- as.integer(F_padded / g)

    # Concatenate along sample axis BEFORE encoding (so both train and
    # test go through the same encoder / positional embedding).
    y_test <- torch::torch_full(c(B, n_test, 1L), NaN,
                                 device = device, dtype = y_train$dtype)
    x_all <- torch::torch_cat(list(x_train, x_test), dim = 2L)   # (B, N, F_padded)
    y_all <- torch::torch_cat(list(y_train, y_test), dim = 2L)   # (B, N, 1)

    # Reshape to (B, N, F_groups, g)
    x_all <- x_all$view(c(B, n_total, F_groups, g))
    dump_if_enabled("input_x_grouped", x_all)

    # ===== X preprocessing + encoding =====
    # Rearrange (B, N, F_groups, g) -> (N, B*F_groups, g) to match the
    # Python einops "b s f n -> s (b f) n" convention.
    x_sbg <- x_all$permute(c(2L, 1L, 3L, 4L))$contiguous()$
      view(c(n_total, B * F_groups, g))

    pp <- preprocess_x_for_encoder(x_sbg, single_eval_pos = n_train)
    feature_state <- pp$state
    x_cat <- torch::torch_cat(list(pp$main, pp$nan_indicators), dim = -1L)
    dump_if_enabled("encoder_linear_in", x_cat)   # (N, B*F_groups, 2g)
    # Apply the final (state-dict-loaded) linear/MLP encoder step.
    x_enc <- self$encoder$steps[[6]](x_cat)   # (N, B*F_groups, emb)
    # Match Python's `LinearInputEncoderStep` output layout (S, BG, emb).
    dump_if_enabled("embedded_x_pre_pos", x_enc)
    # Rearrange back: (N, B*F_groups, emb) -> (B, N, F_groups, emb)
    embedded_x <- x_enc$view(c(n_total, B, F_groups, self$embedding_dim))$
      permute(c(2L, 1L, 3L, 4L))$contiguous()

    # ===== y preprocessing + encoding =====
    if (identical(self$head, "classifier")) {
      yp <- preprocess_y_clf_for_encoder(y_all, single_eval_pos = n_train)
      y_final_idx <- 3L   # y_encoder.steps[[3]] = Linear (classifier)
    } else {
      yp <- preprocess_y_reg_for_encoder(y_all, single_eval_pos = n_train)
      y_final_idx <- 2L   # y_encoder.steps[[2]] = Linear (regressor)
    }
    y_cat <- torch::torch_cat(list(yp$main, yp$nan_indicators), dim = -1L)
    # Permute (B, S, 2) -> (S, B, 2) so the Linear sees Python's layout.
    y_cat_sbe <- y_cat$permute(c(2L, 1L, 3L))$contiguous()
    dump_if_enabled("y_encoder_linear_in", y_cat_sbe)
    embedded_y_sbe <- self$y_encoder$steps[[y_final_idx]](y_cat_sbe)   # (S, B, emb)
    dump_if_enabled("embedded_y", embedded_y_sbe)
    embedded_y <- embedded_y_sbe$permute(c(2L, 1L, 3L))$contiguous()   # (B, S, emb)

    # --- Subspace feature positional embedding on X only ---
    pos_emb <- self$compute_feature_positional(
      F_groups = F_groups,
      column_embeddings = column_embeddings,
      device = device,
      dtype  = embedded_x$dtype
    )
    dump_if_enabled("positional_embedding", pos_emb)
    dump_if_enabled("positional_embedding_linear_out", pos_emb)
    embedded_x <- embedded_x + pos_emb$unsqueeze(1L)$unsqueeze(1L)

    # --- Concat y as last feature-axis token -> (B, N, F_groups+1, emb) ---
    embedded_input <- torch::torch_cat(
      list(embedded_x, embedded_y$unsqueeze(3L)),
      dim = 3L
    )
    dump_if_enabled("embedded_input_pre_thinking", embedded_input)

    # --- Thinking tokens: prepended along sample axis ---
    thinking_out <- self$add_thinking_tokens(embedded_input, single_eval_pos = n_train)
    embedded_input  <- thinking_out$state
    single_eval_pos <- thinking_out$single_eval_pos
    n_total_tok     <- n_total + self$n_thinking_rows
    dump_if_enabled("embedded_input_post_thinking", embedded_input)
    # Same data, named to match Python's add_thinking_tokens_out hook.
    dump_if_enabled("add_thinking_tokens_out", embedded_input)

    # --- Run encoder stack (multi-query split inside each layer) ---
    h <- embedded_input
    n_layers <- length(self$transformer_encoder$layers)
    dump_layers <- c(1L, (n_layers %/% 2L) + 1L, n_layers)
    kv_out <- if (isTRUE(return_kv_cache)) vector("list", n_layers) else NULL
    for (i in seq_len(n_layers)) {
      dump_prefix <- if (i == 1L) "layer0_internal" else NULL
      res <- self$transformer_encoder$layers[[i]](
        h, single_eval_pos = single_eval_pos, dump_prefix = dump_prefix,
        return_kv = isTRUE(return_kv_cache),
        save_peak_memory_factor = save_peak_memory_factor
      )
      h <- res$state
      if (isTRUE(return_kv_cache)) kv_out[[i]] <- res$kv
      collect_between_layers(h)
      if (i %in% dump_layers) {
        nm <- if (i == 1L) "layer0_out"
              else if (i == n_layers) "layer_final_out"
              else "layer_mid_out"
        dump_if_enabled(nm, h)
      }
    }
    encoder_out <- h

    # --- Extract test outputs: y-token column (index -1), test rows only ---
    # encoder_out: (B, n_total_tok, F_groups+1, emb)
    out <- if (n_test > 0L) {
      self$decode(encoder_out, test_start = single_eval_pos,
                  n_rows_total = n_total_tok)
    } else {
      list(logits = NULL, test_hidden = NULL)
    }
    logits <- out$logits
    test_hidden <- out$test_hidden

    if (isTRUE(return_kv_cache)) {
      # The embedded target of a test row is the same for every test row:
      # an absent label, imputed from the training statistics. Embed one
      # padded row and keep it.
      test_y <- self$embed_test_target(y_train, n_train = n_train)
      return(list(
        logits = logits, test_hidden = test_hidden, encoder_out = encoder_out,
        single_eval_pos = single_eval_pos,
        n_thinking_rows = self$n_thinking_rows,
        kv_cache = tabpfn_kv_cache(
          kv = kv_out, feature_state = feature_state,
          test_y_embedding = test_y, n_train = n_train,
          n_feature_groups = F_groups
        )
      ))
    }

    list(
      logits          = logits,
      test_hidden     = test_hidden,
      encoder_out     = encoder_out,
      single_eval_pos = single_eval_pos,
      n_thinking_rows = self$n_thinking_rows
    )
  },

  #' Project the y-token column of the test rows to logits.
  #' @keywords internal
  decode = function(encoder_out, test_start, n_rows_total) {
    test_hidden <- encoder_out[, (test_start + 1L):n_rows_total, -1, ]
    dump_if_enabled("test_hidden", test_hidden)
    logits <- self$decoder_dict(test_hidden)
    dump_if_enabled("logits", logits)
    list(logits = logits, test_hidden = test_hidden)
  },

  #' Embed the target column of a row that has no label.
  #'
  #' Every test row gets the same one: the y pipeline imputes an absent
  #' label with the training mean, so the value does not depend on which
  #' test row it belongs to. Built by running the ordinary y path over the
  #' training labels plus a single NaN row and keeping that row.
  #' @keywords internal
  embed_test_target = function(y_train, n_train) {
    B <- y_train$size(1)
    y_pad <- torch::torch_cat(
      list(y_train, torch::torch_full(c(B, 1L, 1L), NaN,
                                      device = y_train$device,
                                      dtype = y_train$dtype)),
      dim = 2L
    )
    yp <- if (identical(self$head, "classifier")) {
      preprocess_y_clf_for_encoder(y_pad, single_eval_pos = n_train)
    } else {
      preprocess_y_reg_for_encoder(y_pad, single_eval_pos = n_train)
    }
    idx <- if (identical(self$head, "classifier")) 3L else 2L
    y_cat <- torch::torch_cat(list(yp$main, yp$nan_indicators), dim = -1L)
    emb <- self$y_encoder$steps[[idx]](y_cat$permute(c(2L, 1L, 3L))$contiguous())
    emb[n_train + 1L, , ]$detach()                                   # (B, emb)
  },

  #' Predict test rows against a prebuilt cache.
  #'
  #' The training rows are not in this pass at all. Everything they
  #' contribute arrives through the cache: the per-layer key/value
  #' projections the test rows attend to, the preprocessing statistics
  #' fitted on them, and the embedded absent-label token.
  #' @keywords internal
  forward_cached = function(x_test, cache, column_embeddings = NULL,
                            save_peak_memory_factor = NULL) {
    device <- x_test$device
    dims <- x_test$size()
    B <- dims[1]; n_test <- dims[2]; F_raw <- dims[3]
    if (B != 1L) {
      cli::cli_abort("The TabPFN backend runs one dataset at a time.")
    }

    g <- self$features_per_group
    missing_to_next <- (g - (F_raw %% g)) %% g
    if (missing_to_next > 0L) {
      x_test <- torch::torch_cat(
        list(x_test, torch::torch_zeros(c(B, n_test, missing_to_next),
                                        device = device, dtype = x_test$dtype)),
        dim = -1L
      )
    }
    F_groups <- as.integer((F_raw + missing_to_next) / g)
    if (F_groups != cache$n_feature_groups) {
      cli::cli_abort(c(
        "This cache was built for {cache$n_feature_groups} feature group{?s}, \\
         but these rows have {F_groups}.",
        i = "The cache is tied to the training matrix it was built from."
      ))
    }

    x_sbg <- x_test$view(c(B, n_test, F_groups, g))$
      permute(c(2L, 1L, 3L, 4L))$contiguous()$view(c(n_test, B * F_groups, g))
    # `single_eval_pos` is unused when the fitted statistics are supplied.
    pp <- preprocess_x_for_encoder(x_sbg, single_eval_pos = 0L,
                                   state = cache$feature_state)
    x_cat <- torch::torch_cat(list(pp$main, pp$nan_indicators), dim = -1L)
    x_enc <- self$encoder$steps[[6]](x_cat)
    embedded_x <- x_enc$view(c(n_test, B, F_groups, self$embedding_dim))$
      permute(c(2L, 1L, 3L, 4L))$contiguous()

    pos_emb <- self$compute_feature_positional(
      F_groups = F_groups, column_embeddings = column_embeddings,
      device = device, dtype = embedded_x$dtype
    )
    embedded_x <- embedded_x + pos_emb$unsqueeze(1L)$unsqueeze(1L)

    embedded_y <- cache$test_y_embedding$to(device = device)$
      unsqueeze(2L)$expand(c(B, n_test, self$embedding_dim))
    h <- torch::torch_cat(list(embedded_x, embedded_y$unsqueeze(3L)), dim = 3L)

    # No thinking tokens and no training rows: every row is a test row.
    for (i in seq_along(self$transformer_encoder$layers)) {
      h <- self$transformer_encoder$layers[[i]](
        h, single_eval_pos = 0L, cached_kv = cache$kv[[i]],
        save_peak_memory_factor = save_peak_memory_factor
      )$state
      collect_between_layers(h)
    }

    out <- self$decode(h, test_start = 0L, n_rows_total = n_test)
    list(logits = out$logits, test_hidden = out$test_hidden,
         encoder_out = h, single_eval_pos = 0L,
         n_thinking_rows = self$n_thinking_rows)
  },

  #' Compute the subspace positional embedding for this forward pass.
  #'
  #' Mirrors the Python logic in `add_embeddings`:
  #'   1. Draw `embs` of shape (F_groups, 48) from a seeded Generator.
  #'   2. Overwrite the first min(F_groups, 2000) rows with the
  #'      pregenerated `column_embeddings` buffer (for determinism).
  #'   3. Project via `feature_positional_embedding_embeddings`
  #'      (Linear 48 -> emb).
  #'
  #' For Phase 2 we use a zero embedding when no `column_embeddings`
  #' is provided; Phase 3/4 will wire in the real buffer and seeded RNG.
  #' @keywords internal
  compute_feature_positional = function(F_groups, column_embeddings, device, dtype) {
    if (is.null(column_embeddings)) {
      embs <- torch::torch_zeros(c(F_groups, 48L), device = device, dtype = dtype)
    } else {
      n_pre <- min(F_groups, column_embeddings$size(1))
      embs <- torch::torch_zeros(c(F_groups, 48L), device = device, dtype = dtype)
      embs[1:n_pre, ] <- column_embeddings[1:n_pre, ]$to(device = device, dtype = dtype)
    }
    self$feature_positional_embedding_embeddings(embs)   # (F_groups, emb)
  }
)
