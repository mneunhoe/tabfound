# Google TabFM 1.0.0 backend.
#
# Three stages, twice over:
#
#   cell_embedder     per-cell Fourier features + the target embedding
#   col_embedder      set-transformer along the ROW axis, per column
#   + cls_tokens      prepended along the column axis
#   row_interactor    encoder along the COLUMN axis, per row (RoPE here)
#   col_embedder_2    the column stage again
#   row_interactor_2  the row stage again, truncated to the CLS columns
#   icl_predictor     24-block encoder over rows, no positional encoding
#
# The checkpoint ships as safetensors + config.json on the Hub, so unlike
# TabPFN this backend needs no offline conversion. The weights are ~1.6 B
# parameters (6.5 GB in float32) and are published under a
# non-commercial license, separate from the Apache-2.0 source.
#
# Reference: https://github.com/google-research/tabfm
#            tabfm/src/pytorch/model.py

# ---------------------------------------------------------------------------
# Cell embedder
# ---------------------------------------------------------------------------

#' Group each column with the columns at offsets 1, 2, 4, ... away
#'
#' Produces `(B, T, H, G)` from `(B, T, H)` by stacking `G` cyclically
#' shifted copies of the column axis, at offsets `2^i - 1`. Each cell is
#' therefore embedded together with a few of its neighbours, which is
#' what lets the model see interactions before any attention runs.
#' @keywords internal
tabfm_group_features <- function(x, feature_group_size) {
  h <- x$size(3)
  dev <- x$device
  parts <- lapply(seq_len(feature_group_size), function(i) {
    offset <- 2^(i - 1L) - 1L
    # 0-based (idx + offset) %% h, then +1 for R's 1-based index_select.
    idx <- ((seq_len(h) - 1L) + offset) %% h + 1L
    torch::torch_index_select(
      x, dim = -1L,
      index = torch::torch_tensor(as.integer(idx), dtype = torch::torch_long(),
                                  device = dev)
    )
  })
  torch::torch_stack(parts, dim = -1L)
}


#' Per-cell Fourier embedding plus the target embedding
#'
#' Each grouped cell value is expanded into `sin`/`cos` at `num_freq`
#' learned frequencies, projected to the model width, and summed over the
#' group. The training rows additionally get the embedded target added
#' in; test rows do not, which is the only thing that distinguishes them.
#'
#' The Fourier expansion runs in float32: the arguments `value * freq`
#' reach ~30, and computing `sin`/`cos` at that magnitude in a narrower
#' dtype loses the low bits that the projection then amplifies.
#' @keywords internal
tabfm_cell_embedder <- torch::nn_module(
  "TabfmCellEmbedder",

  initialize = function(embedding_dim, max_classes, feature_group_size = 3L,
                        num_freq = 32L, is_classifier = TRUE) {
    self$embedding_dim <- as.integer(embedding_dim)
    self$fgs <- as.integer(feature_group_size)
    self$is_classifier <- isTRUE(is_classifier)

    self$register_buffer("fourier_frequencies",
                         torch::torch_zeros(self$fgs, as.integer(num_freq)))
    self$register_buffer("fourier_frequencies_cat",
                         torch::torch_zeros(self$fgs, as.integer(num_freq)))
    self$in_linear <- torch::nn_linear(as.integer(num_freq) * 2L,
                                       self$embedding_dim, bias = TRUE)
    self$in_linear_cat <- torch::nn_linear(as.integer(num_freq) * 2L,
                                           self$embedding_dim, bias = TRUE)

    self$y_embedder_lookup <- if (self$is_classifier) {
      torch::nn_embedding(as.integer(max_classes), self$embedding_dim)
    } else {
      # Regression embeds the scalar target through a small MLP; the
      # hidden width of 6 is fixed in the reference.
      mlp_stack(1L, 6L, self$embedding_dim, activation = "gelu_tanh")
    }
  },

  # @param x `(B, T, H)`.
  # @param y `(B, T)`.
  # @param train_size `(B)` long — rows before this index carry a label.
  # @param cat_mask `(B, H)` logical, or `NULL`. Cells in a flagged column
  #   are expanded against the *categorical* Fourier frequencies and
  #   projected by `in_linear_cat` instead. An all-`FALSE` mask gives
  #   exactly the same answer as `NULL`, which is why this went unnoticed
  #   while the parity harness only ever ran the bare network — the
  #   sklearn wrapper always passes a mask, and it is only all-`FALSE`
  #   when nothing was declared categorical.
  forward = function(x, y, train_size, cat_mask = NULL) {
    dt <- x$dtype
    g  <- tabfm_group_features(x, self$fgs)$unsqueeze(-1)$
      to(dtype = torch::torch_float32())            # (B, T, H, G, 1)
    ff <- self$fourier_frequencies$to(dtype = torch::torch_float32())

    gf <- g * ff                                     # (B, T, H, G, num_freq)
    feats <- torch::torch_cat(list(gf$sin(), gf$cos()), dim = -1L)$to(dtype = dt)
    num_out <- self$in_linear(feats)                 # (B, T, H, G, E)

    cell <- if (is.null(cat_mask)) {
      num_out$sum(dim = -2L)
    } else {
      ffc <- self$fourier_frequencies_cat$to(dtype = torch::torch_float32())
      gfc <- g * ffc
      feats_cat <- torch::torch_cat(list(gfc$sin(), gfc$cos()),
                                    dim = -1L)$to(dtype = dt)
      cat_out <- self$in_linear_cat(feats_cat)
      # The mask is grouped the same way the values are, so a cell picks
      # the categorical branch when *its own* column is categorical --
      # neighbours in the group keep their own branch.
      cmg <- tabfm_group_features(
        cat_mask$unsqueeze(2L)$to(dtype = torch::torch_float32()), self$fgs
      )$to(dtype = torch::torch_bool())$unsqueeze(-1)  # (B, 1, H, G, 1)
      torch::torch_where(cmg, cat_out, num_out)$sum(dim = -2L)
    }

    y_emb <- if (self$is_classifier) {
      n_emb <- self$y_embedder_lookup$weight$size(1)
      y_idx <- y$to(dtype = torch::torch_long())$clamp(min = 0L, max = n_emb - 1L)
      # R torch's nn_embedding indexes from 1.
      self$y_embedder_lookup(y_idx + 1L)
    } else {
      self$y_embedder_lookup(y$unsqueeze(-1)$to(dtype = cell$dtype))
    }

    t <- x$size(2)
    pos <- torch::torch_arange(0L, t - 1L, dtype = torch::torch_long(),
                               device = x$device)$unsqueeze(1L)
    tm <- (pos < train_size$unsqueeze(2L))$unsqueeze(-1)$unsqueeze(-1)
    torch::torch_where(tm, cell + y_emb$unsqueeze(3L), cell)
  }
)


# ---------------------------------------------------------------------------
# Column and row stages
# ---------------------------------------------------------------------------

#' Column-wise set transformer
#'
#' Reshapes `(B, T, HC, E)` so each column of each batch element becomes
#' an independent sequence over rows, runs the set transformer, and
#' projects back. Attention is masked to the training rows, so test rows
#' contribute nothing to a column's summary.
#' @keywords internal
tabfm_col_embedding <- torch::nn_module(
  "TabfmColEmbedding",

  initialize = function(embedding_dim, num_blocks, n_heads, dim_ff, num_inds) {
    self$tf_col <- set_transformer(num_blocks, embedding_dim, n_heads,
                                   dim_ff, num_inds)
    self$out_w <- torch::nn_linear(embedding_dim, embedding_dim, bias = TRUE)
    self$ln_w  <- rms_norm(embedding_dim)
  },

  # Fold the column axis into the batch axis and build the training-row
  # mask. `train_size = NULL` means every row is a query row, which is the
  # cached case: there is nothing left to mask.
  # @keywords internal
  prepare = function(x, train_size) {
    b <- x$size(1); t <- x$size(2); hc <- x$size(3); e <- x$size(4)
    src <- x$permute(c(1L, 3L, 2L, 4L))$contiguous()$reshape(c(b * hc, t, e))

    mask <- NULL
    if (!is.null(train_size)) {
      ts <- torch::torch_repeat_interleave(train_size, as.integer(hc))
      pos <- torch::torch_arange(0L, t - 1L, dtype = torch::torch_long(),
                                 device = x$device)$unsqueeze(1L)
      mask <- (pos < ts$unsqueeze(2L))$unsqueeze(2L)$unsqueeze(3L)
    }
    list(src = src, mask = mask, b = b, t = t, hc = hc, e = e)
  },

  # @keywords internal
  project_out = function(out, p) {
    out <- self$ln_w(self$out_w(out))
    out$reshape(c(p$b, p$hc, p$t, p$e))$permute(c(1L, 3L, 2L, 4L))$contiguous()
  },

  forward = function(x, train_size) {
    p <- self$prepare(x, train_size)
    self$project_out(self$tf_col(p$src, attn_mask = p$mask), p)
  },

  # Run the training rows through, keeping each block's inducing summary.
  # @return `list(out, hidden)`.
  build_hidden = function(x, train_size) {
    p <- self$prepare(x, train_size)
    built <- self$tf_col$build_hidden(p$src, attn_mask = p$mask)
    list(out = self$project_out(built$src, p), hidden = built$hidden)
  },

  #' Embed test rows against prebuilt block summaries.
  #' @keywords internal
  forward_cached = function(x, hidden) {
    p <- self$prepare(x, NULL)
    self$project_out(self$tf_col(p$src, hidden = hidden), p)
  }
)


#' Row-wise encoder over the column axis
#'
#' Each row becomes a sequence over its columns, with RoPE supplying
#' column position. `output_full = FALSE` keeps only the leading CLS
#' columns and flattens them, which is how the variable-width table
#' becomes a fixed-width row representation.
#' @keywords internal
tabfm_row_interaction <- torch::nn_module(
  "TabfmRowInteraction",

  initialize = function(embedding_dim, num_blocks, n_heads, dim_ff, num_cls,
                        rope_base = 100000.0, output_full = TRUE) {
    self$tf_row <- encoder_stack(num_blocks, embedding_dim, n_heads, dim_ff,
                                 rope_base = rope_base)
    self$out_ln <- rms_norm(embedding_dim)
    self$num_cls <- as.integer(num_cls)
    self$output_full <- isTRUE(output_full)
  },

  forward = function(x) {
    b <- x$size(1); t <- x$size(2); hc <- x$size(3); e <- x$size(4)
    src <- x$reshape(c(b * t, hc, e))
    out <- self$tf_row(src)

    if (self$output_full) {
      self$out_ln(out)$reshape(c(b, t, hc, e))
    } else {
      # Slice first, normalize second -- the reference normalizes only
      # the CLS columns it keeps.
      self$out_ln(out[, 1:self$num_cls, ])$reshape(c(b, t, self$num_cls * e))
    }
  }
)


# ---------------------------------------------------------------------------
# In-context learning stage
# ---------------------------------------------------------------------------

#' One-hot encode class ids and project
#'
#' Labels outside `[0, num_classes)` — including the `-100` sentinel used
#' for unlabelled rows — encode to an all-zero vector, so they contribute
#' only the projection's bias.
#' @keywords internal
tabfm_onehot_linear <- torch::nn_module(
  "TabfmOneHotAndLinear",

  initialize = function(num_classes, embedding_dim) {
    self$num_classes <- as.integer(num_classes)
    self$projection <- torch::nn_linear(self$num_classes, embedding_dim,
                                        bias = TRUE)
  },

  forward = function(y) {
    K <- self$num_classes
    y_long <- y$to(dtype = torch::torch_long())
    valid <- (y_long >= 0L) & (y_long < K)
    safe <- torch::torch_where(valid, y_long, torch::torch_zeros_like(y_long))

    dt <- self$projection$weight$dtype
    oh <- torch::torch_zeros(c(y$size(1), y$size(2), K), dtype = dt,
                             device = y$device)
    oh <- oh$scatter(dim = -1L, index = (safe + 1L)$unsqueeze(-1),
                     src = torch::torch_ones_like(oh))
    oh <- oh * valid$unsqueeze(-1)$to(dtype = dt)
    self$projection(oh)
  }
)


#' In-context learning stage
#'
#' Adds the encoded target to the training rows' representations, runs a
#' plain encoder (no positional encoding — the rows are a set), and
#' decodes to per-class logits or a scalar.
#' @keywords internal
tabfm_ic_learning <- torch::nn_module(
  "TabfmICLearning",

  initialize = function(embedding_dim, num_blocks, n_heads, max_classes,
                        dim_ff, decoder_hidden, is_classifier = TRUE) {
    self$tf_icl <- encoder_stack(num_blocks, embedding_dim, n_heads, dim_ff,
                                 rope_base = NULL)
    self$ln <- rms_norm(embedding_dim)
    self$is_classifier <- isTRUE(is_classifier)

    if (self$is_classifier) {
      self$y_encoder <- tabfm_onehot_linear(max_classes, embedding_dim)
      self$decoder <- mlp_stack(embedding_dim, decoder_hidden,
                                as.integer(max_classes), activation = "gelu_tanh")
    } else {
      self$y_encoder <- mlp_stack(1L, decoder_hidden, embedding_dim,
                                  activation = "gelu_tanh")
      self$decoder <- mlp_stack(embedding_dim, decoder_hidden, 1L,
                                activation = "gelu_tanh")
    }
  },

  # The training-row mask, and the representations with the encoded
  # target added to those rows.
  # @keywords internal
  prepare = function(reps, y, train_size) {
    t <- reps$size(2)
    pos <- torch::torch_arange(0L, t - 1L, dtype = torch::torch_long(),
                               device = reps$device)$unsqueeze(1L)
    tm <- pos < train_size$unsqueeze(2L)                       # (B, T)

    y_enc <- if (self$is_classifier) {
      self$y_encoder(y)
    } else {
      self$y_encoder(y$unsqueeze(-1)$to(dtype = reps$dtype))
    }
    list(r = reps + y_enc * tm$unsqueeze(-1)$to(dtype = reps$dtype), tm = tm)
  },

  forward = function(reps, y, train_size) {
    p <- self$prepare(reps, y, train_size)
    out <- self$tf_icl(p$r, attn_mask = p$tm$unsqueeze(2L)$unsqueeze(3L))
    self$decoder(self$ln(out))
  },

  #' Per-block key/value projections over the training rows.
  #' @keywords internal
  build_kv = function(reps, y, train_size) {
    self$tf_icl$build_kv(self$prepare(reps, y, train_size)$r)
  },

  #' Decode test rows against a prebuilt per-block key/value cache.
  #' @keywords internal
  forward_cached = function(reps, cached_kv) {
    # No target is added. The uncached pass multiplies the encoded target
    # by a zero mask for these rows, which leaves the representation
    # exactly as it was.
    out <- self$tf_icl(reps, cached_kv = cached_kv)
    self$decoder(self$ln(out))
  }
)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

#' TabFM (top level)
#' @keywords internal
tabfm_model <- torch::nn_module(
  "TabFM",

  initialize = function(config) {
    e  <- as.integer(config$embed_dim)
    ff <- e * as.integer(config$ff_factor)
    num_cls <- as.integer(config$row_num_cls)
    icl_dim <- e * num_cls
    is_clf <- isTRUE(config$is_classifier)

    self$max_classes <- as.integer(config$max_classes)
    self$is_classifier <- is_clf
    self$num_cls <- num_cls

    self$cell_embedder <- tabfm_cell_embedder(
      embedding_dim = e, max_classes = self$max_classes,
      feature_group_size = as.integer(config$feature_group_size %||% 3L),
      num_freq = as.integer(config$num_freq %||% 32L),
      is_classifier = is_clf
    )
    self$col_embedder <- tabfm_col_embedding(
      e, as.integer(config$col_num_blocks), as.integer(config$col_nhead),
      ff, as.integer(config$col_num_inds)
    )
    self$col_embedder_2 <- tabfm_col_embedding(
      e, as.integer(config$col_num_blocks), as.integer(config$col_nhead),
      ff, as.integer(config$col_num_inds)
    )
    self$row_interactor <- tabfm_row_interaction(
      e, as.integer(config$row_num_blocks), as.integer(config$row_nhead),
      ff, num_cls, output_full = TRUE
    )
    self$row_interactor_2 <- tabfm_row_interaction(
      e, as.integer(config$row_num_blocks), as.integer(config$row_nhead),
      ff, num_cls, output_full = FALSE
    )
    self$cls_tokens <- torch::nn_parameter(torch::torch_zeros(num_cls, e))

    # `decoder_hidden` is null in the published configs; the reference
    # then falls back to twice the ICL width.
    dec_hidden <- config$decoder_hidden
    if (is.null(dec_hidden) || is.na(dec_hidden)) dec_hidden <- icl_dim * 2L

    self$icl_predictor <- tabfm_ic_learning(
      embedding_dim = icl_dim,
      num_blocks = as.integer(config$icl_num_blocks),
      n_heads = as.integer(config$icl_nhead),
      max_classes = self$max_classes,
      dim_ff = icl_dim * as.integer(config$ff_factor),
      decoder_hidden = as.integer(dec_hidden),
      is_classifier = is_clf
    )

    # Read by the predictors; see `tabfm_kv_cache()`.
    self$supports_kv_cache <- TRUE
    self$kv_cache_is_exact <- TRUE
  },

  # `nan_to_num(x, nan = -100)` in the reference. Written out because
  # torch's R binding does not export it, and adding a second `:::`
  # dependency for a one-liner is not worth it.
  # @keywords internal
  fill_missing = function(x) {
    torch::torch_where(
      x$isnan(), torch::torch_full_like(x, -100.0), x
    )$to(dtype = self$cls_tokens$dtype)
  },

  # Prepend the CLS columns the row stage reads out through.
  # @keywords internal
  add_cls = function(emb) {
    b <- emb$size(1); t <- emb$size(2); e <- emb$size(4)
    cls <- self$cls_tokens$unsqueeze(1L)$unsqueeze(1L)$
      expand(c(b, t, self$num_cls, e))
    torch::torch_cat(list(cls, emb), dim = 3L)
  },

  # Everything the training rows contribute to a prediction.
  #
  # @param x `(B, train_size, H)` — the training rows only.
  # @param y `(B, train_size)`; @param train_size `(B)` long.
  # @param cat_mask As in `forward()`. A cache is only valid for the mask
  #   it was built with, since the mask changes how cells are embedded.
  build_kv_cache = function(x, y, train_size, cat_mask = NULL) {
    x <- self$fill_missing(x)
    emb <- self$cell_embedder(x, y, train_size, cat_mask)
    c1 <- self$col_embedder$build_hidden(emb, train_size)
    emb <- self$row_interactor(self$add_cls(c1$out))
    c2 <- self$col_embedder_2$build_hidden(emb, train_size)
    reps <- self$row_interactor_2(c2$out)
    tabfm_kv_cache(
      col_hidden   = list(c1$hidden, c2$hidden),
      icl_kv       = self$icl_predictor$build_kv(reps, y, train_size),
      n_train      = x$size(2),
      n_features   = x$size(3),
      has_cat_mask = !is.null(cat_mask)
    )
  },

  # @param x `(B, T, H)` features, train rows first.
  # @param y `(B, T)` targets; entries at or after `train_size` are
  #   ignored and conventionally set to the -100 sentinel.
  # @param train_size `(B)` long.
  # @param cat_mask `(B, H)` logical marking categorical columns, or
  #   `NULL`. See [tabfm_cell_embedder()].
  # @return `(B, T, max_classes)` logits, or `(B, T, 1)` for regression.
  #
  # The reference's `forward` takes one further argument, `d` -- a
  # per-member count of *active* columns, used to wrap feature groups and
  # to mask attention over zero-padded ones. It is not implemented here
  # because nothing this port builds can produce a padded member: padding
  # only appears when ensemble members have different widths, which in
  # turn only happens under the `ensemble()` preset's feature crosses and
  # SVD features (see `tabfm_ensemble_fit()`, which rejects them). With
  # uniform widths `d` equals the column count for every member, and the
  # reference's own arithmetic then reduces to this one exactly.
  forward = function(x, y, train_size, cat_mask = NULL, kv_cache = NULL) {
    if (!is.null(kv_cache)) {
      return(self$forward_cached(x, kv_cache, cat_mask = cat_mask))
    }
    x <- self$fill_missing(x)

    # Stage outputs are dumped when TABFOUND_DUMP_DIR is set, under the
    # same names the Python parity script hooks. Six stages is enough to
    # bisect any mismatch to one module.
    emb <- self$cell_embedder(x, y, train_size, cat_mask)
    dump_if_enabled("tabfm_cell", emb)
    emb <- self$col_embedder(emb, train_size)
    dump_if_enabled("tabfm_col1", emb)

    emb <- self$row_interactor(self$add_cls(emb))
    dump_if_enabled("tabfm_row1", emb)
    emb <- self$col_embedder_2(emb, train_size)
    dump_if_enabled("tabfm_col2", emb)
    reps <- self$row_interactor_2(emb)
    dump_if_enabled("tabfm_reps", reps)

    out <- self$icl_predictor(reps, y, train_size)
    dump_if_enabled("tabfm_logits", out)
    out
  },

  #' @keywords internal
  forward_cached = function(x, kv_cache, cat_mask = NULL) {
    if (x$size(3) != kv_cache$n_features) {
      cli::cli_abort(
        "This cache was built for {kv_cache$n_features} feature{?s}; \\
         got {x$size(3)}."
      )
    }
    if (!identical(!is.null(cat_mask), kv_cache$has_cat_mask)) {
      cli::cli_abort(
        "This cache was built {if (kv_cache$has_cat_mask) 'with' else 'without'} \\
         a categorical mask; the mask decides which Fourier basis a cell \\
         is embedded against, so the two cannot be mixed."
      )
    }
    x <- self$fill_missing(x)
    # Every row here is a test row: no labels, and `train_size = 0` so the
    # cell embedder adds no target embedding to any of them.
    y <- torch::torch_full(c(x$size(1), x$size(2)), -100.0,
                           dtype = x$dtype, device = x$device)
    train_size <- torch::torch_zeros(x$size(1), dtype = torch::torch_long(),
                                     device = x$device)

    emb <- self$cell_embedder(x, y, train_size, cat_mask)
    dump_if_enabled("tabfm_cell", emb)
    emb <- self$col_embedder$forward_cached(emb, kv_cache$col_hidden[[1]])
    dump_if_enabled("tabfm_col1", emb)

    emb <- self$row_interactor(self$add_cls(emb))
    dump_if_enabled("tabfm_row1", emb)
    emb <- self$col_embedder_2$forward_cached(emb, kv_cache$col_hidden[[2]])
    dump_if_enabled("tabfm_col2", emb)
    reps <- self$row_interactor_2(emb)
    dump_if_enabled("tabfm_reps", reps)

    out <- self$icl_predictor$forward_cached(reps, kv_cache$icl_kv)
    dump_if_enabled("tabfm_logits", out)
    out
  }
)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

#' Everything the training rows contribute to a TabFM prediction
#'
#' TabFM restricts context to the training rows by **masking**: the two
#' column stages mask the inducing attention, and the ICL stage masks its
#' keys. A masked position contributes exactly zero, so a training row's
#' path through the network never picks up anything from a test row --
#' which is what makes the training half computable once and reusable.
#'
#' Two kinds of thing are stored:
#'
#' * `col_hidden` — for each of the two column stages, one
#'   `(B * HC, num_inds, E)` inducing summary per block. Small: the set
#'   transformer's whole point is that this bottleneck does not grow with
#'   the number of rows.
#' * `icl_kv` — one `list(key, value)` per ICL block, over the training
#'   rows. This one does grow with the training set.
#'
#' Both row stages need nothing cached: each row is its own sequence over
#' its own columns, so they never looked across rows at all.
#'
#' Reusing the cache asks the same question a full pass asks, and gets an
#' answer that agrees to within float32 rounding rather than exactly. The
#' reason is the masking: a cached attention reduces over `n_train`
#' positions where the uncached one reduces over `T` positions of which
#' `T - n_train` contribute exactly zero. The sums are equal in exact
#' arithmetic and grouped differently in float32. Measured on the test
#' networks the gap is around `1e-6` of the logit scale -- the same order
#' `predict_chunk_size` already moves things by.
#'
#' @param col_hidden Length-2 list, one entry per column stage.
#' @param icl_kv Per-block ICL key/value projections.
#' @param n_train,n_features What it was built from.
#' @param has_cat_mask Whether a categorical mask was in force. Cells are
#'   embedded against a different Fourier basis with one, so a cache
#'   cannot cross that boundary.
#' @keywords internal
tabfm_kv_cache <- function(col_hidden, icl_kv, n_train, n_features,
                           has_cat_mask) {
  structure(
    list(col_hidden = col_hidden, icl_kv = icl_kv,
         n_train = as.integer(n_train), n_features = as.integer(n_features),
         has_cat_mask = isTRUE(has_cat_mask)),
    class = "tabfm_kv_cache"
  )
}

#' @export
print.tabfm_kv_cache <- function(x, ...) {
  n_bytes <- .tensor_bytes(x$col_hidden) + .tensor_bytes(x$icl_kv)
  cli::cli_text("{.strong TabFM KV cache}")
  cli::cli_bullets(c(
    "*" = "built from {.val {x$n_train}} training row{?s}, \\
           {.val {x$n_features}} feature{?s}",
    "*" = "{length(x$col_hidden[[1]])} block{?s} per column stage, \\
           {length(x$icl_kv)} ICL block{?s}, \\
           {round(n_bytes / 1e6, 1)} MB"
  ))
  invisible(x)
}


# ---------------------------------------------------------------------------
# Backend hooks
# ---------------------------------------------------------------------------

#' @keywords internal
tabfm_build <- function(config, task) {
  cli::cli_alert_info(
    "Building TabFM ({.val {if (isTRUE(config$is_classifier)) 'classifier' else 'regressor'}}, \\
     emb={config$embed_dim}, icl={config$icl_num_blocks} blocks)..."
  )
  tabfm_model(config)
}

#' @keywords internal
tabfm_detect <- function(config) {
  # The published config has no architecture field, so detection keys off
  # the fields themselves. Stage widths alone are not enough -- TabICL is
  # a close cousin and carries the same `icl_num_blocks`, `col_num_inds`,
  # `row_num_cls` and `embed_dim`. What is TabFM-only is the Fourier cell
  # embedder (`num_freq`) and the boolean task flag; TabICL instead marks
  # the task with `max_classes` and tags `arch` during conversion.
  is.null(config$arch) &&
    !is.null(config$num_freq) && !is.null(config$is_classifier) &&
    !is.null(config$embed_dim) && !is.null(config$icl_num_blocks)
}

#' @keywords internal
tabfm_task_of <- function(config) {
  if (is.null(config$is_classifier)) return(NULL)
  if (isTRUE(config$is_classifier)) "classification" else "regression"
}

#' @keywords internal
tabfm_subfolder_for <- function(task) {
  switch(task, classification = "classification", regression = "regression", NULL)
}


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

# Assemble the (x, y, train_size) triple the network expects from a
# train/test split. Unlabelled rows carry the -100 sentinel, which the
# cell embedder and the ICL y-encoder both treat as "no label".
# @keywords internal
.tabfm_batch <- function(X_train, y_train, X_test, cat_mask, device) {
  x <- rbind(as.matrix(X_train), as.matrix(X_test))
  storage.mode(x) <- "double"
  n_train <- nrow(X_train)
  y <- c(as.numeric(y_train), rep(-100, nrow(X_test)))
  list(
    x = as_float_tensor(x, device = device)$unsqueeze(1L),
    y = as_float_tensor(matrix(y, nrow = 1L), device = device),
    # Shape (B,), not (B, 1): R has no scalar type, so a length-1 vector
    # already becomes a 1-D tensor and an extra unsqueeze would silently
    # add a broadcast axis to every downstream mask.
    train_size = torch::torch_tensor(as.integer(n_train),
                                     dtype = torch::torch_long(),
                                     device = device),
    cat_mask = .tabfm_cat_tensor(cat_mask, device),
    n_train = n_train
  )
}

# An all-FALSE mask and no mask compute the same thing, and no mask skips
# the categorical Fourier expansion entirely.
# @keywords internal
.tabfm_cat_tensor <- function(cat_mask, device) {
  if (!any(cat_mask)) return(NULL)
  torch::torch_tensor(matrix(as.logical(cat_mask), nrow = 1L),
                      dtype = torch::torch_bool(), device = device)
}

# Run one ensemble member over a test chunk, cached or not, and return its
# `(n_test, out_dim)` output block.
#
# Members differ in their row subsample as well as their preprocessing
# (`max_num_rows`), so each one's cache is built from that member's own
# training rows rather than from a shared tensor.
# @keywords internal
.tabfm_member_out <- function(net, dev, mem, n_test, cat_mask, cache_store, i) {
  n_tr <- nrow(mem$X) - n_test
  X_train <- mem$X[seq_len(n_tr), , drop = FALSE]
  X_test  <- mem$X[-seq_len(n_tr), , drop = FALSE]
  if (is.null(cache_store)) {
    b <- .tabfm_batch(X_train, mem$y, X_test, cat_mask, dev)
    out <- torch::with_no_grad({ net(b$x, b$y, b$train_size, b$cat_mask) })
    return(list(out = out, block = out[1, (b$n_train + 1L):out$size(2), ]))
  }
  cache <- member_cache(cache_store, i, function() {
    b <- .tabfm_batch(X_train, mem$y, X_train[0L, , drop = FALSE], cat_mask, dev)
    torch::with_no_grad({
      net$build_kv_cache(b$x, b$y, b$train_size, b$cat_mask)
    })
  })
  x_te <- as_float_tensor(as.matrix(X_test), device = dev)$unsqueeze(1L)
  out <- torch::with_no_grad({
    net(x_te, NULL, NULL, .tabfm_cat_tensor(cat_mask, dev), kv_cache = cache)
  })
  list(out = out, block = out[1, , ])
}

# Options shared by both TabFM predictors, matching `TabFMClassifier` /
# `TabFMRegressor` defaults.
# @keywords internal
.tabfm_opts <- function(n_estimators, norm_methods, feat_shuffle_method,
                        class_shift, outlier_threshold, max_num_features,
                        max_num_rows, random_state, quantile_subsample,
                        n_feature_crosses, n_svd_features) {
  if (!identical(n_feature_crosses, 0) || !identical(n_svd_features, 0)) {
    cli::cli_abort(c(
      "Feature crosses and SVD features are not implemented.",
      i = "They belong to the reference's {.code TabFMClassifier.ensemble()} \\
           preset, which also fits NNLS ensemble weights and a probability \\
           calibrator on out-of-fold predictions -- machinery this backend \\
           does not have. The default preset is fully supported."
    ))
  }
  list(n_estimators = as.integer(n_estimators),
       norm_methods = norm_methods %||% c("none", "power"),
       feat_shuffle_method = feat_shuffle_method,
       class_shift = isTRUE(class_shift),
       outlier_threshold = outlier_threshold,
       max_num_features = max_num_features,
       max_num_rows = max_num_rows,
       random_state = random_state,
       quantile_subsample = quantile_subsample)
}

# Impute, then build the ensemble.
#
# The reference's `TransformToNumerical` mean-imputes numeric columns
# when it is handed a data frame, and is the identity on a bare array --
# in which case the very next step, `UniqueFeatureFilter`, rejects the
# NaN outright. An R matrix corresponds to the frame path (that is what
# `tabfound()` builds one from), so imputing is the faithful reading, and
# it is also the only one that works: `CustomStandardScaler` takes a
# plain column mean, so one missing value would turn a whole column NaN.
# @keywords internal
.tabfm_prepare <- function(X, y, task, cat_features, opts) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  imputer <- fit_simple_imputer(X)
  if (!any(imputer$keep)) cli::cli_abort("Every predictor is entirely missing.")
  X <- transform_simple_imputer(X, imputer)
  # The imputer can drop an all-missing column, which renumbers
  # everything to its right -- so the declared indices have to be carried
  # through it on a mask rather than reused as they were given.
  cat_kept <- NULL
  if (!is.null(cat_features) && length(cat_features)) {
    mask <- rep(FALSE, length(imputer$keep))
    mask[cat_features] <- TRUE
    cat_kept <- which(mask[imputer$keep])
  }
  gen <- tabfm_ensemble_fit(
    X, y, task = task, n_estimators = opts$n_estimators,
    norm_methods = opts$norm_methods,
    feat_shuffle_method = opts$feat_shuffle_method,
    class_shift = opts$class_shift, cat_features = cat_kept,
    outlier_threshold = opts$outlier_threshold,
    max_num_features = opts$max_num_features,
    max_num_rows = opts$max_num_rows,
    random_state = opts$random_state,
    quantile_subsample = opts$quantile_subsample
  )
  list(imputer = imputer, gen = gen)
}

#' Build the TabFM classifier predictor
#'
#' Reproduces `TabFMClassifier` under its default preset: an ensemble of
#' `n_estimators` views differing in normalisation, feature order and a
#' cyclic shift of the class labels, averaged as logits and passed
#' through one temperature-scaled softmax.
#'
#' @param ctx Loaded-model context from [load_backend_model()].
#' @param n_estimators Number of ensemble members.
#' @param norm_methods Normalisation methods to spread across members;
#'   `NULL` means `c("none", "power")`.
#' @param feat_shuffle_method `"random"` or `"none"`.
#' @param class_shift Rotate the class labels per member.
#' @param categorical_features Integer vector of 1-based column indices
#'   to treat as categorical. These reach the network as a `cat_mask`,
#'   which routes their cells through a separate Fourier basis — so
#'   declaring them changes the prediction, not just the bookkeeping.
#' @param outlier_threshold Z-score threshold for the outlier clipper.
#' @param max_num_features,max_num_rows Per-member subsampling caps, or
#'   `NULL` for none.
#' @param softmax_temperature Logits are divided by this before the
#'   softmax.
#' @param average_logits Average logits and take one softmax, rather than
#'   averaging per-member probabilities.
#' @param random_state Seed for the ensemble construction.
#' @param quantile_subsample See [fit_sk_quantile_transformer()].
#' @param n_feature_crosses,n_svd_features Accepted only as `0`; see the
#'   error they raise otherwise.
#' @param predict_chunk_size Max test rows per forward pass.
#' @param kv_cache Condition on the training rows once per `predict()`
#'   call and reuse that across every chunk and ensemble member, instead
#'   of re-encoding the whole training set for each. Off by default. With
#'   32 members and a default chunk of 512 rows this is the difference
#'   between one training pass per member and one per member per chunk,
#'   which on a 1.6 B-parameter network is most of the wall clock. It
#'   answers the same question an uncached pass does -- nothing in this
#'   architecture lets a training row see a test row -- to within float32
#'   rounding. See [tabfm_kv_cache()].
#' @param trace_dir Optional directory for the parity harness.
#' @keywords internal
tabfm_classifier <- function(ctx, n_estimators = 32L, norm_methods = NULL,
                             feat_shuffle_method = "random",
                             class_shift = TRUE,
                             categorical_features = NULL,
                             outlier_threshold = 4.0,
                             max_num_features = 500L,
                             max_num_rows = NULL,
                             softmax_temperature = 0.9,
                             average_logits = TRUE,
                             random_state = 42L,
                             quantile_subsample = NULL,
                             n_feature_crosses = 0,
                             n_svd_features = 0,
                             predict_chunk_size = 512L, kv_cache = FALSE,
                             trace_dir = NULL) {
  net <- ctx$net; dev <- ctx$device
  opts <- .tabfm_opts(n_estimators, norm_methods, feat_shuffle_method,
                      class_shift, outlier_threshold, max_num_features,
                      max_num_rows, random_state, quantile_subsample,
                      n_feature_crosses, n_svd_features)

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    if (is.factor(y)) {
      levels_ <- levels(y); y_int <- as.integer(y) - 1L
    } else {
      levels_ <- sort(unique(y)); y_int <- match(y, levels_) - 1L
    }
    if (length(levels_) > net$max_classes) {
      cli::cli_abort(
        "TabFM supports at most {net$max_classes} classes; got {length(levels_)}."
      )
    }
    prep <- .tabfm_prepare(X, y_int, "classification", categorical_features, opts)
    c(prep, list(class_levels = levels_, n_train = nrow(X)))
  }

  .chunk <- function(state, X_test_chunk, caches) {
    X_test <- transform_simple_imputer(as.matrix(X_test_chunk), state$imputer)
    members <- tabfm_ensemble_transform(state$gen, X_test)
    n_cls <- length(state$class_levels)
    acc <- NULL
    for (i in seq_along(members)) {
      mem <- members[[i]]
      res <- .tabfm_member_out(net, dev, mem, nrow(X_test), mem$cat_mask,
                               caches, i)
      if (!is.null(trace_dir)) trace_tensor(trace_dir, "logits", res$out)
      logits <- as.matrix(res$block[, 1:n_cls]$cpu())
      # Training used `(y + shift) %% K`, so the model's column for
      # original class k is `(k + shift) %% K`.
      logits <- logits[, ((seq_len(n_cls) - 1L + mem$shift) %% n_cls) + 1L,
                       drop = FALSE]
      contrib <- if (average_logits) logits else
        .softmax_rows(logits, softmax_temperature)
      acc <- if (is.null(acc)) contrib else acc + contrib
    }
    avg <- acc / length(members)
    if (average_logits) .softmax_rows(avg, softmax_temperature) else avg
  }

  predict_fn <- function(state, newdata, type = "class", ...) {
    caches <- if (isTRUE(kv_cache)) member_cache_store() else NULL
    probs <- chunk_apply(as.matrix(newdata), predict_chunk_size,
                         function(chunk) .chunk(state, chunk, caches))
    colnames(probs) <- as.character(state$class_levels)
    if (type == "prob") return(probs)
    state$class_levels[max.col(probs, ties.method = "first")]
  }

  list(fit = fit_fn, predict = predict_fn)
}


#' Build the TabFM regressor predictor
#'
#' TabFM's regression head emits a single value per row, so there is no
#' predictive distribution to draw quantiles from and `type = "quantiles"`
#' is not offered rather than being approximated. Member predictions are
#' averaged in standardized target space and inverse-transformed once,
#' as the reference does.
#'
#' @inheritParams tabfm_classifier
#' @keywords internal
tabfm_regressor <- function(ctx, n_estimators = 32L, norm_methods = NULL,
                            feat_shuffle_method = "random",
                            categorical_features = NULL,
                            outlier_threshold = 4.0,
                            max_num_features = 500L,
                            max_num_rows = NULL,
                            random_state = 42L,
                            quantile_subsample = NULL,
                            n_feature_crosses = 0,
                            n_svd_features = 0,
                            predict_chunk_size = 512L, kv_cache = FALSE,
                            trace_dir = NULL) {
  net <- ctx$net; dev <- ctx$device
  opts <- .tabfm_opts(n_estimators, norm_methods, feat_shuffle_method,
                      FALSE, outlier_threshold, max_num_features,
                      max_num_rows, random_state, quantile_subsample,
                      n_feature_crosses, n_svd_features)

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    # The network works in standardized target space; `TabFMRegressor`
    # fits a StandardScaler on y and inverse-transforms its output.
    scaler <- fit_target_scaler(y)
    prep <- .tabfm_prepare(X, apply_target_scaler(y, scaler), "regression",
                           categorical_features, opts)
    c(prep, list(scaler = scaler, n_train = nrow(X)))
  }

  .chunk <- function(state, X_test_chunk, caches) {
    X_test <- transform_simple_imputer(as.matrix(X_test_chunk), state$imputer)
    members <- tabfm_ensemble_transform(state$gen, X_test)
    acc <- NULL
    for (i in seq_along(members)) {
      mem <- members[[i]]
      res <- .tabfm_member_out(net, dev, mem, nrow(X_test), mem$cat_mask,
                               caches, i)
      if (!is.null(trace_dir)) trace_tensor(trace_dir, "logits", res$out)
      preds <- as.numeric(res$block[, 1]$cpu())
      acc <- if (is.null(acc)) preds else acc + preds
    }
    matrix(invert_target_scaler(acc / length(members), state$scaler), ncol = 1L)
  }

  predict_fn <- function(state, newdata, type = "mean", ...) {
    caches <- if (isTRUE(kv_cache)) member_cache_store() else NULL
    as.numeric(chunk_apply(as.matrix(newdata), predict_chunk_size,
                           function(chunk) .chunk(state, chunk, caches)))
  }

  list(fit = fit_fn, predict = predict_fn, types = "mean")
}


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

#' Stage list for the TabFM diagram
#'
#' TabFM alternates: summarise the columns, mix the row, summarise the
#' columns again, mix the row again, then learn in context over the rows.
#' Running the column and row stages twice is what distinguishes it from
#' TabICL's single pass through each, and the second row stage is the one
#' that collapses a variable-width table to a fixed-width vector.
#'
#' @param config The checkpoint's parsed `config.json`.
#' @param task `"classification"` or `"regression"`.
#' @keywords internal
tabfm_describe <- function(config, task) {
  E     <- as.integer(config$embed_dim)
  n_cls <- as.integer(config$row_num_cls)
  D     <- E * n_cls
  ff    <- as.integer(config$ff_factor)
  grp   <- as.integer(config$feature_group_size %||% 3L)
  nfreq <- as.integer(config$num_freq %||% 32L)
  n_col <- as.integer(config$col_num_blocks)
  n_row <- as.integer(config$row_num_blocks)
  n_icl <- as.integer(config$icl_num_blocks)
  h_col <- as.integer(config$col_nhead)
  h_row <- as.integer(config$row_nhead)
  h_icl <- as.integer(config$icl_nhead)
  n_ind <- as.integer(config$col_num_inds)
  clf   <- isTRUE(config$is_classifier)
  n_out <- if (clf) as.integer(config$max_classes) else 1L

  col_children <- function(pfx) list(
    arch_stage("ind", "Inducing points attend to the rows", kind = "attention",
               axis = "rows", prefix = paste0(pfx, ".tf_col.blocks.0.mab1")),
    arch_stage("back", "Rows attend back to the inducing points",
               kind = "attention", axis = "inducing",
               prefix = paste0(pfx, ".tf_col.blocks.0.mab2")),
    arch_stage("iv", "Learned inducing points", kind = "embed",
               prefix = paste0(pfx, ".tf_col.blocks.0.ind_vectors")),
    arch_stage("out", "Output projection and RMSNorm", kind = "norm",
               prefix = c(paste0(pfx, ".out_w"), paste0(pfx, ".ln_w")))
  )
  row_children <- function(pfx) list(
    arch_stage("attn", "Attention over a row's columns", kind = "attention",
               axis = "columns",
               prefix = paste0(pfx, ".tf_row.blocks.0")),
    # TabFM's RoPE is a fixed buffer, not a learned table as in TabICL
    # and TabPFN v3, so it owns no parameters and gets no block of its
    # own -- the parent's detail line is where it is stated.
    arch_stage("oln", "RMSNorm", kind = "norm",
               prefix = paste0(pfx, ".out_ln"))
  )

  stages <- list(
    arch_input_stage(detail = "numeric matrix; NaN is mapped to the sentinel -100"),
    arch_stage(
      "cell", "Fourier cell embedder", kind = "embed", group = "Embed",
      detail = sprintf("sin/cos at %d learned frequencies per cell, summed over a group of %d; Linear(%d -> %d)",
                       nfreq, grp, 2L * nfreq, E),
      shape = "(B, n, F, E)", prefix = "cell_embedder"),
    arch_stage(
      "cls", "CLS columns", kind = "embed", group = "Embed",
      detail = sprintf("%d learned columns prepended to every row", n_cls),
      prefix = "cls_tokens"),
    arch_stage(
      "col1", "Column set transformer", kind = "attention",
      group = "Pass 1", repeats = n_col, axis = "rows",
      detail = sprintf("induced self-attention, %d inducing points; %d heads; masked to the training rows",
                       n_ind, h_col),
      shape = "(B, n, F, E)", prefix = "col_embedder",
      children = col_children("col_embedder")),
    arch_stage(
      "row1", "Row encoder", kind = "attention", group = "Pass 1",
      repeats = n_row, axis = "columns",
      detail = sprintf("RoPE over the column order; %d heads; all columns kept",
                       h_row),
      shape = "(B, n, F, E)", prefix = "row_interactor",
      children = row_children("row_interactor")),
    arch_stage(
      "col2", "Column set transformer", kind = "attention",
      group = "Pass 2", repeats = n_col, axis = "rows",
      detail = "same shape as pass 1, its own weights",
      shape = "(B, n, F, E)", prefix = "col_embedder_2",
      children = col_children("col_embedder_2")),
    arch_stage(
      "row2", "Row encoder, CLS readout", kind = "attention", group = "Pass 2",
      repeats = n_row, axis = "columns",
      detail = sprintf("keeps the %d CLS columns and flattens them: a table of any width becomes %d numbers",
                       n_cls, D),
      shape = sprintf("(B, n, %d)", D), prefix = "row_interactor_2",
      children = row_children("row_interactor_2")),
    arch_stage(
      "icl_y", "Target encoder (per row)", kind = "embed",
      group = "In-context",
      detail = if (clf) sprintf("one-hot over %d classes -> Linear(%d)", n_out, D)
               else sprintf("MLP(1 -> %d)", D),
      prefix = c("icl_predictor.y_encoder", "icl_predictor.ln")),
    arch_stage(
      "icl", "In-context learning block", kind = "attention",
      group = "In-context", repeats = n_icl, axis = "rows",
      detail = sprintf("no positional encoding - the rows are a set; %d heads of %d; SwiGLU %d -> %d",
                       h_icl, as.integer(D / h_icl), D, D * ff),
      shape = sprintf("(B, n, %d)", D), prefix = "icl_predictor.tf_icl",
      children = list(
        arch_stage("iattn", "Attention over rows, training-row keys",
                   kind = "attention", axis = "rows",
                   prefix = "icl_predictor.tf_icl.blocks.0.attn"),
        arch_stage("iln", "RMSNorm x4 (sandwich)", kind = "norm",
                   prefix = c("icl_predictor.tf_icl.blocks.0.pre_attn_ln",
                              "icl_predictor.tf_icl.blocks.0.post_attn_ln",
                              "icl_predictor.tf_icl.blocks.0.pre_ff_ln",
                              "icl_predictor.tf_icl.blocks.0.post_ff_ln")),
        arch_stage("imlp", "SwiGLU feed-forward", kind = "ffn",
                   prefix = c("icl_predictor.tf_icl.blocks.0.linear1",
                              "icl_predictor.tf_icl.blocks.0.linear1_gate",
                              "icl_predictor.tf_icl.blocks.0.linear2"))
      )),
    arch_stage(
      "dec", "Decoder MLP", kind = "decode", group = "Decode",
      detail = sprintf("%d -> %s -> %d, GELU", D,
                       format(config$decoder_hidden %||% (D * 2L)), n_out),
      shape = sprintf("(n_test, %d)", n_out), prefix = "icl_predictor.decoder"),
    if (clf)
      arch_output_stage("Class logits",
                        detail = sprintf("%d slots; softmax over the classes seen", n_out))
    else
      arch_output_stage("Scalar prediction",
                        detail = "one number per row, on the scaled target")
  )

  list(
    title = sprintf("TabFM - %s", if (clf) "classifier" else "regressor"),
    subtitle = "Google  -  column and row stages run twice, then an in-context stack",
    facts = c(
      "Column blocks"      = sprintf("%d, twice", n_col),
      "Row blocks"         = sprintf("%d, twice", n_row),
      "ICL blocks"         = n_icl,
      "Embedding width"    = E,
      "ICL width"          = sprintf("%d (%d CLS x %d)", D, n_cls, E),
      "ICL heads"          = sprintf("%d x %d", h_icl, as.integer(D / h_icl)),
      "Inducing points"    = n_ind,
      "Feature group size" = grp,
      "Fourier frequencies" = nfreq,
      "Feed-forward"       = sprintf("SwiGLU, %d x width", ff),
      "Output width"       = n_out,
      "Normalisation"      = "RMSNorm, sandwich",
      "KV cache"           = "exact"
    ),
    symbols = c(arch_default_symbols(), D = sprintf("ICL width (%d)", D)),
    stages = Filter(Negate(is.null), stages)
  )
}


# ---------------------------------------------------------------------------
# Memory scaling
# ---------------------------------------------------------------------------

#' TabFM's peak-memory shapes
#'
#' The same three-stage shape as TabICL with two differences that both
#' cost memory. The cell embedder gives every *cell* its own token rather
#' than one per group of three, so the widest activation is `rows x
#' columns x embed_dim` -- three times TabICL's at the same dimensions --
#' and the 1.6 B float32 parameters are 6.4 GB resident before a single
#' activation exists.
#'
#' Two knobs move real bytes here and are honoured: `max_num_features`
#' caps the columns each member sees, and `max_num_rows` subsamples its
#' context.
#'
#' @inheritParams mitra_peak_terms
#' @keywords internal
tabfm_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  n_est <- max(1, as.numeric(opts$n_estimators %||% 1))
  cap_p <- suppressWarnings(as.numeric(opts$max_num_features %||% Inf))
  cap_n <- suppressWarnings(as.numeric(opts$max_num_rows %||% Inf))
  p <- min(n_features, if (is.finite(cap_p)) cap_p else n_features)
  n_ctx <- min(n_context, if (is.finite(cap_n)) cap_n else n_context)

  .icl_family_terms(
    n_context    = n_ctx,
    n_query      = .resident_query(n_query, opts),
    n_features   = p,
    embed_dim    = as.numeric(config$embed_dim),
    group_size   = as.numeric(config$feature_group_size %||% 3),
    n_cls        = as.numeric(config$row_num_cls %||% 4),
    col_heads    = as.numeric(config$col_nhead %||% 8),
    row_heads    = as.numeric(config$row_nhead %||% 8),
    icl_heads    = as.numeric(config$icl_nhead %||% 8),
    col_inducing = as.numeric(config$col_num_inds %||% 128),
    icl_blocks   = as.numeric(config$icl_num_blocks %||% 12),
    # One token per cell, not per feature group -- this is the difference.
    cell_tokens  = p,
    kv_cache     = isTRUE(opts$kv_cache),
    n_estimators = n_est
  )
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#' @keywords internal
register_tabfm_backend <- function() {
  register_backend(
    name          = "tabfm",
    describe      = tabfm_describe,
    build         = tabfm_build,
    translate_key = identity,
    detect        = tabfm_detect,
    task_of       = tabfm_task_of,
    subfolder_for = tabfm_subfolder_for,
    classifier    = tabfm_classifier,
    regressor     = tabfm_regressor,
    peak_terms    = tabfm_peak_terms,
    aliases       = c("tabfm-1.0.0" = "google/tabfm-1.0.0-pytorch"),
    # `nan_to_num(x, nan = -100)` at the top of the forward pass.
    handles_missing = TRUE,
    description   = "TabFM 1.0.0 column/row/ICL transformer (Google Research)",
    parity        = "tabfm (PyPI)"
  )
}
