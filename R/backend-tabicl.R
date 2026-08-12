# TabICL v2 backend (soda-inria).
#
# Three stages:
#
#   col_embedder    per-column set transformer over ROWS, target-aware
#   row_interactor  encoder over COLUMNS per row (RoPE), read out via CLS
#   icl_predictor   encoder over ROWS, context restricted to labelled rows
#
# Architecturally a cousin of TabFM, but the details differ throughout:
# affine LayerNorm instead of RMSNorm, a plain GELU feed-forward instead
# of SwiGLU, a packed q/k/v projection instead of separate ones, a
# scalable softmax on the length-varying attentions, and context
# restriction by slicing rather than masking.
#
# Checkpoints are `torch.save` dicts; convert once with
# `inst/python/tabicl_convert_ckpt.py`.
#
# Reference: https://github.com/soda-inria/tabicl (tabicl/_model/)

# ---------------------------------------------------------------------------
# Column embedding
# ---------------------------------------------------------------------------

#' Group each column with the columns 1, 2, 4, ... positions away
#'
#' `(B, T, H)` -> `(B, T, H, size)` by stacking cyclically shifted copies
#' at offsets `2^i`.
#'
#' Note the offsets are `2^i`, not TabFM's `2^i - 1`: TabICL's first
#' group member is the *next* column, not the column itself. Getting this
#' wrong produces a model that runs and returns plausible numbers.
#' @keywords internal
tabicl_group_features <- function(x, group_size) {
  h <- x$size(3)
  dev <- x$device
  parts <- lapply(seq_len(group_size), function(i) {
    offset <- 2^(i - 1L)
    idx <- ((seq_len(h) - 1L) + offset) %% h + 1L
    torch::torch_index_select(
      x, dim = -1L,
      index = torch::torch_tensor(as.integer(idx), dtype = torch::torch_long(),
                                  device = dev)
    )
  })
  torch::torch_stack(parts, dim = -1L)
}


#' Linear projection that passes the sentinel through untouched
#'
#' Rows whose entire input equals `-100` keep that value on the output
#' instead of being projected. The reserved CLS-token columns are filled
#' with the sentinel, and this is what stops them being treated as data.
#' @keywords internal
tabicl_skippable_linear <- torch::nn_module(
  "TabiclSkippableLinear",

  initialize = function(in_features, out_features, skip_value = -100.0) {
    self$layer <- torch::nn_linear(in_features, out_features, bias = TRUE)
    self$skip_value <- skip_value
  },

  forward = function(x) {
    out <- self$layer(x)
    is_skip <- (x == self$skip_value)$all(dim = -1L)
    if (!as.logical(is_skip$any()$item())) return(out)
    torch::torch_where(is_skip$unsqueeze(-1L),
                       torch::torch_full_like(out, self$skip_value), out)
  }
)


#' Column-wise embedding
#'
#' Each column becomes an independent sequence over rows. Cells are
#' projected, the embedded target is added to the labelled rows, and a
#' set transformer summarises the column's distribution — with the
#' inducing points restricted to labelled rows so test values cannot leak
#' into a column's statistics.
#'
#' The released checkpoints have `col_affine = FALSE`, so the set
#' transformer's output *is* the embedding; the affine variant (which
#' would use the output as per-column scale and shift) has no weights in
#' these checkpoints and is not built.
#' @keywords internal
tabicl_col_embedding <- torch::nn_module(
  "TabiclColEmbedding",

  initialize = function(config) {
    e <- as.integer(config$embed_dim)
    self$embed_dim <- e
    self$group_size <- as.integer(config$col_feature_group_size %||% 3L)
    self$feature_group <- !identical(config$col_feature_group, FALSE)
    self$target_aware <- isTRUE(config$col_target_aware)
    self$max_classes <- as.integer(config$max_classes %||% 0L)
    self$reserve_cls_tokens <- as.integer(config$row_num_cls %||% 0L)

    in_dim <- if (self$feature_group) self$group_size else 1L
    self$in_linear <- tabicl_skippable_linear(in_dim, e)

    self$tf_col <- tabicl_set_transformer(
      num_blocks = as.integer(config$col_num_blocks),
      embedding_dim = e,
      n_heads = as.integer(config$col_nhead),
      dim_ff = e * as.integer(config$ff_factor),
      num_inds = as.integer(config$col_num_inds),
      ssmax = !identical(config$col_ssmax, "none"),
      bias_free_ln = isTRUE(config$bias_free_ln),
      activation = config$activation %||% "gelu"
    )

    if (self$target_aware) {
      self$y_encoder <- if (self$max_classes > 0L) {
        # One-hot then linear; stored as a plain Linear(max_classes, E).
        torch::nn_linear(self$max_classes, e, bias = TRUE)
      } else {
        torch::nn_linear(1L, e, bias = TRUE)
      }
    }
  },

  # Steps 1-3 of `forward()`: group the columns and reserve the CLS
  # slots, stopping short of the projection.
  #
  # This is where the pipeline can be cut. What comes out is
  # `(B, HC, T, group_size)` -- three floats per cell -- where the
  # projection that follows makes it `(B, HC, T, embed_dim)`, forty-odd
  # times wider. Anything that wants to work a column slice at a time has
  # to start on this side of that.
  # @keywords internal
  group_cells = function(x) {
    xg <- if (self$feature_group) {
      tabicl_group_features(x, self$group_size)
    } else {
      x$unsqueeze(-1L)
    }
    if (self$reserve_cls_tokens > 0L) {
      pad <- torch::torch_full(
        c(xg$size(1), xg$size(2), self$reserve_cls_tokens, xg$size(4)),
        -100.0, dtype = xg$dtype, device = xg$device
      )
      xg <- torch::torch_cat(list(pad, xg), dim = 3L)
    }
    xg$permute(c(1L, 3L, 2L, 4L))$contiguous()               # (B, HC, T, size)
  },

  # Steps 1-4: the grouping above, then the per-cell projection. Nothing
  # here looks past a single row, so it is the same whether the batch
  # holds training rows, test rows, or both.
  # @keywords internal
  embed_cells = function(x) {
    features <- self$group_cells(x)
    list(features = features, src = self$in_linear(features))
  },

  # Add the embedded target to the first `train_size` rows (step 5).
  # @keywords internal
  add_target = function(src, features, y_train) {
    if (!self$target_aware) return(src)
    train_size <- y_train$size(2)
    hc <- features$size(2)
    y_emb <- if (self$max_classes > 0L) {
      # R torch's one-hot indexes from 1, so shift the 0-based class
      # ids before encoding.
      oh <- torch::nnf_one_hot(
        y_train$to(dtype = torch::torch_long()) + 1L,
        num_classes = self$max_classes
      )$to(dtype = src$dtype)
      self$y_encoder(oh)
    } else {
      self$y_encoder(y_train$unsqueeze(-1L)$to(dtype = src$dtype))
    }
    # (B, train_size, E) -> (B, HC, train_size, E)
    y_emb <- y_emb$unsqueeze(2L)$expand(
      c(y_emb$size(1), hc, y_emb$size(2), y_emb$size(3))
    )
    head <- src[, , 1:train_size, ] + y_emb
    if (train_size < src$size(3)) {
      torch::torch_cat(list(head, src[, , (train_size + 1L):src$size(3), ]),
                       dim = 3L)
    } else {
      head
    }
  },

  # Run the labelled rows through, keeping each block's summary.
  #
  # @param col_chunk_size Columns per pass, or `NULL` for all at once.
  #   Every column is embedded on its own -- the set transformer folds
  #   `(B, HC)` into its batch -- so this changes nothing about the
  #   answer and bounds the `(B, HC, T, embed_dim)` tensor the pre-pass
  #   would otherwise hold whole. That tensor is the ceiling: a row-
  #   chunked forward cannot start until the summaries exist, so on a
  #   backend where the pre-pass dominates, chunking rows alone buys
  #   nothing. Measured on TabFM at 4,000 x 90, the pre-pass alone was
  #   36.4 GB of a 35.6 GB chunked forward.
  # @param want_out Keep the labelled rows' column embeddings. A cache
  #   build needs them -- they feed the row interactor -- but a chunked
  #   forward wants only the summaries, and accumulating the full-width
  #   `(B, HC, T, E)` output just to discard it defeats the point of
  #   chunking the columns in the first place.
  # @return `list(out, hidden)`; `out` is `NULL` when not wanted.
  build_hidden = function(x_train, y_train, col_chunk_size = NULL,
                          want_out = TRUE) {
    features <- self$group_cells(x_train)
    hc <- features$size(2)
    cc <- suppressWarnings(as.integer(col_chunk_size %||% NA_integer_))
    if (is.na(cc) || cc < 1L || cc >= hc) {
      src <- self$add_target(self$in_linear(features), features, y_train)
      built <- self$tf_col$build_hidden(src)
      return(list(
        out = if (isTRUE(want_out))
          built$src$permute(c(1L, 3L, 2L, 4L))$contiguous() else NULL,
        hidden = built$hidden))
    }

    outs <- list()
    hids <- NULL
    for (s in seq(1L, hc, by = cc)) {
      len <- min(s + cc - 1L, hc) - s + 1L
      f <- features$narrow(2L, s, len)
      built <- self$tf_col$build_hidden(
        self$add_target(self$in_linear(f), f, y_train)
      )
      if (isTRUE(want_out)) outs[[length(outs) + 1L]] <- built$src
      # Concatenated on the column axis, in slice order, so the whole
      # lines up with what an unchunked pass would have produced. Get
      # this wrong and every column carries another column's summary,
      # which is finite, plausible and silent.
      if (is.null(hids)) {
        hids <- lapply(built$hidden, list)
      } else {
        for (i in seq_along(hids)) {
          hids[[i]][[length(hids[[i]]) + 1L]] <- built$hidden[[i]]
        }
      }
      collect_between_layers(built$src)
    }
    out <- if (!isTRUE(want_out)) NULL else {
      src <- if (length(outs) == 1L) outs[[1L]] else
        torch::torch_cat(outs, dim = 2L)
      src$permute(c(1L, 3L, 2L, 4L))$contiguous()
    }
    list(
      out = out,
      hidden = lapply(hids, function(h) {
        if (length(h) == 1L) h[[1L]] else torch::torch_cat(h, dim = 2L)
      })
    )
  },

  #' Column-embed rows against a prebuilt set of block summaries.
  #'
  #' @param y_train Targets for the *leading* rows of `x`, when some of
  #'   them are labelled. A prediction against a cache passes `NULL` --
  #'   every row there is a test row. A row-chunked uncached pass does
  #'   not: a chunk that straddles the train/test boundary carries both
  #'   kinds, and the labelled ones must get the target embedding the
  #'   unchunked pass would have given them. Silently omitting it leaves
  #'   the shapes intact and the answer wrong.
  #' @keywords internal
  forward_cached = function(x, hidden, y_train = NULL) {
    ce <- self$embed_cells(x)
    src <- if (is.null(y_train)) ce$src
           else self$add_target(ce$src, ce$features, y_train)
    src <- self$tf_col(src, hidden = hidden)
    src$permute(c(1L, 3L, 2L, 4L))$contiguous()
  },

  # @param x `(B, T, H)`.
  # @param y_train `(B, train_size)`.
  forward = function(x, y_train) {
    ce <- self$embed_cells(x)
    src <- self$add_target(ce$src, ce$features, y_train)     # (B, HC, T, E)
    src <- self$tf_col(src, train_size = y_train$size(2))
    src$permute(c(1L, 3L, 2L, 4L))$contiguous()              # (B, T, HC, E)
  }
)


# ---------------------------------------------------------------------------
# Row interaction
# ---------------------------------------------------------------------------

#' Row-wise interaction, read out through CLS tokens
#'
#' Each row is a sequence over its columns, with RoPE supplying column
#' position. The reserved leading columns are overwritten with learned
#' CLS tokens; all blocks but the last run as self-attention, and the
#' last attends *from* the CLS columns only, so the output is a fixed
#' `num_cls * embed_dim` vector per row regardless of table width.
#' @keywords internal
tabicl_row_interaction <- torch::nn_module(
  "TabiclRowInteraction",

  initialize = function(config) {
    e <- as.integer(config$embed_dim)
    self$embed_dim <- e
    self$num_cls <- as.integer(config$row_num_cls)
    self$cls_tokens <- torch::nn_parameter(
      torch::torch_zeros(self$num_cls, e)
    )
    self$tf_row <- tabicl_encoder(
      num_blocks = as.integer(config$row_num_blocks),
      embedding_dim = e,
      n_heads = as.integer(config$row_nhead),
      dim_ff = e * as.integer(config$ff_factor),
      ssmax = FALSE,
      bias_free_ln = isTRUE(config$bias_free_ln),
      activation = config$activation %||% "gelu",
      rope_base = as.numeric(config$row_rope_base %||% 100000),
      rope_interleaved = isTRUE(config$row_rope_interleaved)
    )
    self$out_ln <- affine_layer_norm(e, eps = 1e-5, bias = !isTRUE(config$bias_free_ln))
  },

  # @param embeddings `(B, T, HC, E)`.
  forward = function(embeddings) {
    b <- embeddings$size(1); t <- embeddings$size(2)
    hc <- embeddings$size(3); e <- embeddings$size(4)

    cls <- self$cls_tokens$unsqueeze(1L)$unsqueeze(1L)$
      expand(c(b, t, self$num_cls, e))
    x <- torch::torch_cat(
      list(cls, embeddings[, , (self$num_cls + 1L):hc, ]), dim = 3L
    )

    rp <- self$tf_row$rope
    n_blocks <- length(self$tf_row$blocks)
    for (i in seq_len(n_blocks - 1L)) {
      x <- self$tf_row$blocks[[i]](x, rope = rp)
      collect_between_layers(x)
    }
    # Last block: queries are the CLS columns, keys/values the whole row.
    cls_out <- self$tf_row$blocks[[n_blocks]](
      x[, , 1:self$num_cls, ], x, x, rope = rp
    )
    cls_out <- self$out_ln(cls_out)
    cls_out$reshape(c(b, t, self$num_cls * e))
  }
)


# ---------------------------------------------------------------------------
# In-context learning
# ---------------------------------------------------------------------------

#' Dataset-wise in-context learning
#'
#' Adds the embedded target to the labelled rows, then runs an encoder in
#' which every row queries only the labelled rows. Classification decodes
#' to `max_classes` logits; regression to `num_quantiles` quantile levels.
#' @keywords internal
tabicl_ic_learning <- torch::nn_module(
  "TabiclICLearning",

  initialize = function(config) {
    e <- as.integer(config$embed_dim)
    d <- e * as.integer(config$row_num_cls)
    self$d_model <- d
    self$max_classes <- as.integer(config$max_classes %||% 0L)
    self$is_classifier <- self$max_classes > 0L
    out_dim <- if (self$is_classifier) self$max_classes
               else as.integer(config$num_quantiles)
    self$out_dim <- out_dim

    self$tf_icl <- tabicl_encoder(
      num_blocks = as.integer(config$icl_num_blocks),
      embedding_dim = d,
      n_heads = as.integer(config$icl_nhead),
      dim_ff = d * as.integer(config$ff_factor),
      ssmax = !identical(config$icl_ssmax, "none"),
      bias_free_ln = isTRUE(config$bias_free_ln),
      activation = config$activation %||% "gelu",
      rope_base = NULL
    )
    self$ln <- affine_layer_norm(d, eps = 1e-5, bias = !isTRUE(config$bias_free_ln))
    self$y_encoder <- if (self$is_classifier) {
      torch::nn_linear(self$max_classes, d, bias = TRUE)
    } else {
      torch::nn_linear(1L, d, bias = TRUE)
    }
    self$decoder <- torch::nn_sequential(
      torch::nn_linear(d, d * 2L),
      torch::nn_gelu(),
      torch::nn_linear(d * 2L, out_dim)
    )
  },

  # Add the embedded target to the labelled rows.
  # @keywords internal
  add_target = function(reps, y_train) {
    train_size <- y_train$size(2)
    y_emb <- if (self$is_classifier) {
      oh <- torch::nnf_one_hot(
        y_train$to(dtype = torch::torch_long()) + 1L,
        num_classes = self$max_classes
      )$to(dtype = reps$dtype)
      self$y_encoder(oh)
    } else {
      self$y_encoder(y_train$unsqueeze(-1L)$to(dtype = reps$dtype))
    }

    head <- reps[, 1:train_size, ] + y_emb
    if (train_size < reps$size(2)) {
      torch::torch_cat(list(head, reps[, (train_size + 1L):reps$size(2), ]),
                       dim = 2L)
    } else {
      head
    }
  },

  #' Per-block key/value projections over the labelled rows.
  #' @keywords internal
  build_kv = function(reps_train, y_train) {
    self$tf_icl$build_kv(self$add_target(reps_train, y_train))
  },

  #' Decode test rows against a prebuilt per-block key/value cache.
  #' @keywords internal
  forward_cached = function(reps, cached_kv, save_peak_memory_factor = NULL) {
    # No target is added: every row here is a test row, and the uncached
    # pass leaves those untouched too.
    src <- self$tf_icl(reps, cached_kv = cached_kv,
                       save_peak_memory_factor = save_peak_memory_factor)
    self$decoder(self$ln(src))
  },

  # @param reps `(B, T, d_model)`; @param y_train `(B, train_size)`.
  forward = function(reps, y_train, save_peak_memory_factor = NULL) {
    r <- self$add_target(reps, y_train)
    src <- self$tf_icl(r, train_size = y_train$size(2),
                       save_peak_memory_factor = save_peak_memory_factor)
    self$decoder(self$ln(src))
  }
)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

#' TabICL (top level)
#' @keywords internal
tabicl_model <- torch::nn_module(
  "TabICL",

  initialize = function(config) {
    self$max_classes <- as.integer(config$max_classes %||% 0L)
    self$is_classifier <- self$max_classes > 0L
    self$num_quantiles <- as.integer(config$num_quantiles %||% 0L)
    self$col_embedder <- tabicl_col_embedding(config)
    self$row_interactor <- tabicl_row_interaction(config)
    self$icl_predictor <- tabicl_ic_learning(config)
    # Read by the predictors; see `tabicl_kv_cache()` for why the cache is
    # exact rather than merely cheaper here.
    self$supports_kv_cache <- TRUE
    self$supports_stage_chunking <- TRUE
    self$supports_chunked_eval <- TRUE
    self$kv_cache_is_exact <- TRUE
  },

  # Everything the labelled rows contribute to a prediction.
  #
  # @param x `(B, train_size, H)` — the labelled rows only.
  # @param y_train `(B, train_size)`.
  build_kv_cache = function(x, y_train, col_chunk_size = NULL) {
    built <- self$col_embedder$build_hidden(x, y_train, col_chunk_size)
    reps <- self$row_interactor(built$out)
    tabicl_kv_cache(
      col_hidden = built$hidden,
      icl_kv     = self$icl_predictor$build_kv(reps, y_train),
      n_train    = y_train$size(2),
      n_features = x$size(3)
    )
  },

  #' The column and row stages, optionally a chunk of rows at a time
  #'
  #' What this avoids is the `(B, T, HC, E)` column embedding, which
  #' carries every row at every column at full embedding width and is the
  #' term that decides how large a table fits. What survives the loop is
  #' `(B, T, num_cls * E)` -- a fixed handful of vectors per row however
  #' wide the table.
  #'
  #' The prefix is row-independent once the column stage's per-block
  #' summaries exist, which is what makes the loop legitimate: a row's
  #' path through the column stage depends on the summaries and on
  #' itself, and the row stage is a sequence over one row's columns.
  #'
  #' @section The invariant this must not break:
  #' Only the inducing half of a column block carries scalable softmax,
  #' and its scale depends on the *source* length -- the number of rows
  #' being read. So the summaries are built once, from all the labelled
  #' rows, and chunks only ever pass through the second half, whose keys
  #' and values are the fixed inducing set. If a chunk ever reached the
  #' first half, the answer would depend on the chunk size, and it would
  #' look entirely reasonable.
  #'
  #' @param x `(B, T, H)`, labelled rows first.
  #' @param y_train `(B, train_size)`, or `NULL` when every row is a test
  #'   row (the cache-consuming path).
  #' @param col_hidden Per-block summaries from a cache, or `NULL` to
  #'   build them from the labelled rows.
  #' @param row_chunk_size Rows per pass, or `NULL` for all at once.
  #' @param col_chunk_size Columns per pass of the summary pre-pass. The
  #'   row loop cannot start until the summaries exist, so on a wide
  #'   table this is what decides whether row chunking helps at all.
  #' @return `(B, T, num_cls * E)` row representations.
  #' @keywords internal
  stages_0_to_1 = function(x, y_train, col_hidden = NULL,
                           row_chunk_size = NULL, col_chunk_size = NULL) {
    n_rows <- x$size(2)
    n_train <- if (is.null(y_train)) 0L else y_train$size(2)
    size <- suppressWarnings(as.integer(row_chunk_size %||% NA_integer_))
    use_chunks <- !is.na(size) && size >= 1L && size < n_rows

    hidden <- col_hidden
    if (use_chunks && is.null(hidden)) {
      if (n_train <= 0L) {
        cli::cli_abort("Row chunking needs either labelled rows or a cache.")
      }
      hidden <- self$col_embedder$build_hidden(
        x[, 1:n_train, ], y_train, col_chunk_size, want_out = FALSE
      )$hidden
    }

    if (!use_chunks) {
      emb <- if (is.null(hidden)) self$col_embedder(x, y_train)
             else self$col_embedder$forward_cached(x, hidden)
      dump_if_enabled("tabicl_col", emb)
      return(self$row_interactor(emb))
    }

    parts <- vector("list", length(seq(1L, n_rows, by = size)))
    j <- 0L
    for (s in seq(1L, n_rows, by = size)) {
      j <- j + 1L
      len <- min(s + size - 1L, n_rows) - s + 1L
      # How many rows of *this* chunk are labelled. The boundary falls
      # inside a chunk in general, and a chunk past it has none.
      n_tr_chunk <- max(0L, min(n_train - (s - 1L), len))
      y_chunk <- if (n_tr_chunk > 0L) y_train$narrow(2L, s, n_tr_chunk) else NULL
      emb <- self$col_embedder$forward_cached(
        x$narrow(2L, s, len), hidden, y_chunk
      )
      parts[[j]] <- self$row_interactor(emb)
      collect_between_layers(emb)
    }
    if (j == 1L) parts[[1L]] else torch::torch_cat(parts, dim = 2L)
  },

  # @param x `(B, T, H)` with the labelled rows first.
  # @param y_train `(B, train_size)`.
  # @param kv_cache A [tabicl_kv_cache()] to predict against. When given,
  #   `y_train` is ignored and every row of `x` is a row to predict.
  # @param row_chunk_size Rows per pass through the column and row
  #   stages; see `stages_0_to_1()`.
  # @return `(B, T, out_dim)` — logits over classes, or quantile levels.
  forward = function(x, y_train, kv_cache = NULL, row_chunk_size = NULL,
                     save_peak_memory_factor = NULL, col_chunk_size = NULL) {
    if (!is.null(kv_cache)) {
      return(self$forward_cached(x, kv_cache, row_chunk_size,
                                 save_peak_memory_factor))
    }
    reps <- self$stages_0_to_1(x, y_train, row_chunk_size = row_chunk_size,
                               col_chunk_size = col_chunk_size)
    dump_if_enabled("tabicl_reps", reps)
    out <- self$icl_predictor(reps, y_train, save_peak_memory_factor)
    dump_if_enabled("tabicl_logits", out)
    out
  },

  #' @keywords internal
  forward_cached = function(x, kv_cache, row_chunk_size = NULL,
                            save_peak_memory_factor = NULL) {
    if (x$size(3) != kv_cache$n_features) {
      cli::cli_abort(
        "This cache was built for {kv_cache$n_features} feature{?s}; \\
         got {x$size(3)}."
      )
    }
    reps <- self$stages_0_to_1(x, y_train = NULL,
                               col_hidden = kv_cache$col_hidden,
                               row_chunk_size = row_chunk_size)
    dump_if_enabled("tabicl_reps", reps)
    out <- self$icl_predictor$forward_cached(reps, kv_cache$icl_kv,
                                             save_peak_memory_factor)
    dump_if_enabled("tabicl_logits", out)
    out
  }
)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

#' Everything the labelled rows contribute to a TabICL prediction
#'
#' TabICL restricts context to the labelled rows by **slicing**, not
#' masking: the column stage's inducing points read `src[, , 1:train_size,
#' ]`, and every ICL block's keys are `norm1(q)[, 1:train_size, ]`. A
#' labelled row therefore never sees a test row at any stage, so the whole
#' labelled-row half of the forward pass is a function of the training
#' data alone and can be computed once. The cache holds the same
#' quantities the uncached pass computes, arrived at through the same code
#' path -- it is built by slicing too, which matters because the fused and
#' split projections in [mha_fused_inproj()] disagree in the last bit and
#' a stack of blocks amplifies that.
#'
#' What it is not is bit-identical, and neither is `predict_chunk_size`:
#' dropping the training rows from the batch changes the shapes every
#' matmul sees, and float32 matmuls are not invariant to that. The
#' residual is the same size either way -- around `1e-7` on a logit, on
#' the tiny networks the tests use.
#'
#' Two things are stored:
#'
#' * `col_hidden` — one `(B, HC, num_inds, E)` summary per column-stage
#'   block. This is the bottleneck the set transformer squeezes the rows
#'   through, so it is small however many training rows there were.
#' * `icl_kv` — one `list(key, value)` per ICL block, over the labelled
#'   rows. This one does grow with the training set.
#'
#' The row stage needs nothing: it attends across a single row's columns,
#' so it never looked at another row to begin with.
#'
#' @param col_hidden Per-block column-stage summaries.
#' @param icl_kv Per-block ICL key/value projections.
#' @param n_train Number of labelled rows it was built from.
#' @param n_features Feature count it was built for; rows of a different
#'   width cannot use it.
#' @keywords internal
tabicl_kv_cache <- function(col_hidden, icl_kv, n_train, n_features) {
  structure(
    list(col_hidden = col_hidden, icl_kv = icl_kv,
         n_train = as.integer(n_train), n_features = as.integer(n_features)),
    class = "tabicl_kv_cache"
  )
}

#' @export
print.tabicl_kv_cache <- function(x, ...) {
  n_bytes <- .tensor_bytes(x$col_hidden) + .tensor_bytes(x$icl_kv)
  cli::cli_text("{.strong TabICL KV cache}")
  cli::cli_bullets(c(
    "*" = "built from {.val {x$n_train}} labelled row{?s}, \\
           {.val {x$n_features}} feature{?s}",
    "*" = "{length(x$col_hidden)} column block{?s}, \\
           {length(x$icl_kv)} ICL block{?s}, \\
           {round(n_bytes / 1e6, 1)} MB"
  ))
  invisible(x)
}


# ---------------------------------------------------------------------------
# Backend hooks
# ---------------------------------------------------------------------------

# The only structural difference from the checkpoint layout is the
# `SkippableLinear` wrapper, which holds its `nn_linear` in a child named
# `layer`. Everything else -- the packed `in_proj_weight`, the `nn_sequential`
# indices in `decoder` and `ssmax_layer.*_mlp`, the RoPE `freqs` parameter --
# maps across unchanged.
# @keywords internal
tabicl_translate_key <- function(key) {
  sub("^col_embedder\\.in_linear\\.", "col_embedder.in_linear.layer.", key)
}

#' @keywords internal
tabicl_build <- function(config, task) {
  cli::cli_alert_info(
    "Building TabICL ({.val {config$head}}, emb={config$embed_dim}, \\
     icl={config$icl_num_blocks} blocks)..."
  )
  tabicl_model(config)
}

#' @keywords internal
tabicl_detect <- function(config) identical(config$arch, "tabicl")

#' @keywords internal
tabicl_task_of <- function(config) {
  switch(config$head %||% "",
         classifier = "classification",
         regressor  = "regression",
         NULL)
}


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

# Assemble the (x, y_train) pair the network expects. Unlike TabFM, the
# target tensor covers only the labelled rows -- TabICL derives
# `train_size` from its length.
# @keywords internal
.tabicl_batch <- function(X_train, y_train, X_test, device) {
  x <- rbind(as.matrix(X_train), as.matrix(X_test))
  storage.mode(x) <- "double"
  list(
    x = as_float_tensor(x, device = device)$unsqueeze(1L),
    y = as_float_tensor(matrix(as.numeric(y_train), nrow = 1L), device = device),
    n_train = nrow(X_train)
  )
}

# Run one ensemble member, cached or not, and return its `(n_test, out_dim)`
# output block. `cache_store` is NULL when caching is off.
#
# The two branches differ only in what the network is handed: with a cache
# the training rows are left out of the batch entirely, so the forward
# pass is over the test chunk alone and its whole output is wanted.
# @keywords internal
.tabicl_member_out <- function(net, dev, mem, n_train, cache_store, i,
                               row_chunk_size = NULL,
                               save_peak_memory_factor = NULL,
                               col_chunk_size = NULL) {
  X_train <- mem$X[seq_len(n_train), , drop = FALSE]
  X_test  <- mem$X[-seq_len(n_train), , drop = FALSE]
  if (is.null(cache_store)) {
    b <- .tabicl_batch(X_train, mem$y, X_test, dev)
    out <- torch::with_no_grad({
      net(b$x, b$y, row_chunk_size = row_chunk_size,
          save_peak_memory_factor = save_peak_memory_factor,
          col_chunk_size = col_chunk_size)
    })
    return(out[1, (b$n_train + 1L):out$size(2), ])
  }
  cache <- member_cache(cache_store, i, function() {
    b <- .tabicl_batch(X_train, mem$y, X_train[0L, , drop = FALSE], dev)
    torch::with_no_grad({ net$build_kv_cache(b$x, b$y, col_chunk_size) })
  })
  x_te <- as_float_tensor(as.matrix(X_test), device = dev)$unsqueeze(1L)
  out <- torch::with_no_grad({
    net(x_te, NULL, kv_cache = cache, row_chunk_size = row_chunk_size,
        save_peak_memory_factor = save_peak_memory_factor)
  })
  out[1, , ]
}

# Shared fit for both TabICL predictors: impute, then build the ensemble.
#
# The imputation is not optional dressing. TabICL's network has no
# missing-value channel at all -- a `NaN` in propagates to `NaN` logits
# out -- and the reference's own wrapper is what keeps that from
# happening, by running a mean `SimpleImputer` over the numeric columns
# before the ensemble generator ever sees the data.
# @keywords internal
.tabicl_prepare <- function(X, y, classification, opts) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  imputer <- fit_simple_imputer(X)
  if (!any(imputer$keep)) {
    cli::cli_abort("Every predictor is entirely missing.")
  }
  X <- transform_simple_imputer(X, imputer)
  gen <- tabicl_ensemble_fit(
    X, y, classification = classification,
    n_estimators = opts$n_estimators, norm_methods = opts$norm_methods,
    feat_shuffle_method = opts$feat_shuffle_method,
    class_shuffle_method = opts$class_shuffle_method,
    outlier_threshold = opts$outlier_threshold,
    random_state = opts$random_state,
    quantile_subsample = opts$quantile_subsample
  )
  list(imputer = imputer, gen = gen)
}

# Default options shared by the classifier and the regressor, matching
# `TabICLClassifier` / `TabICLRegressor`.
# @keywords internal
.tabicl_opts <- function(n_estimators, norm_methods, feat_shuffle_method,
                         class_shuffle_method, outlier_threshold, random_state,
                         quantile_subsample) {
  list(n_estimators = as.integer(n_estimators),
       norm_methods = norm_methods %||% c("none", "power"),
       feat_shuffle_method = feat_shuffle_method,
       class_shuffle_method = class_shuffle_method,
       outlier_threshold = outlier_threshold,
       random_state = random_state,
       quantile_subsample = quantile_subsample)
}

#' Build the TabICL classifier predictor
#'
#' Reproduces `TabICLClassifier`: the input is imputed, an ensemble of
#' `n_estimators` differently normalised, differently ordered and
#' differently labelled views is built, each is run through the network,
#' and the results are averaged once the class relabelling is undone.
#'
#' Running a single un-preprocessed forward pass — which is what this
#' backend used to do — is not a cheaper version of this. The network is
#' trained on data that has been through this pipeline, so feeding it raw
#' columns is out of distribution, and one member is one draw from an
#' ensemble whose whole point is that its members disagree.
#'
#' @param ctx Loaded-model context from [load_backend_model()].
#' @param n_estimators Number of ensemble members.
#' @param norm_methods Normalisation methods to spread across members:
#'   any of `"none"`, `"power"`, `"quantile"`, `"quantile_rtdl"`,
#'   `"robust"`. `NULL` means `c("none", "power")`.
#' @param feat_shuffle_method,class_shuffle_method Permutation strategies
#'   for feature order and class labels; see [py_shuffler()].
#' @param outlier_threshold Z-score threshold for the outlier clipper.
#' @param softmax_temperature Logits are divided by this before the
#'   softmax.
#' @param average_logits Average member logits and then take one softmax
#'   (the reference's default, better calibrated), rather than averaging
#'   each member's probabilities.
#' @param random_state Seed for the ensemble construction. Draws come
#'   from a port of CPython's generator, so this reproduces the
#'   reference's members exactly.
#' @param quantile_subsample Row cap for the quantile normalisers; see
#'   [fit_sk_quantile_transformer()].
#' @param predict_chunk_size Max test rows per forward pass. Test rows
#'   only ever attend to training rows, so chunking cannot change the
#'   answer.
#' @param kv_cache Condition on the training rows once per `predict()`
#'   call and reuse that across every chunk and ensemble member, instead
#'   of re-encoding the whole training set for each. Off by default. It
#'   answers the same question an uncached pass does, rather than an
#'   approximation of it: TabICL restricts context by slicing the
#'   labelled rows out, so a labelled row never sees a test row and
#'   nothing the cache holds could have depended on one. The arithmetic
#'   differs only by the float32 noise a change in batch shape already
#'   produces -- see [tabicl_kv_cache()].
#' @keywords internal
tabicl_classifier <- function(ctx, n_estimators = 8L, norm_methods = NULL,
                              feat_shuffle_method = "latin",
                              class_shuffle_method = "shift",
                              outlier_threshold = 4.0,
                              softmax_temperature = 0.9,
                              average_logits = TRUE,
                              random_state = 42L,
                              quantile_subsample = NULL,
                              predict_chunk_size = 1024L,
                              kv_cache = FALSE,
                              row_chunk_size = NULL,
                              save_peak_memory_factor = NULL,
                              col_chunk_size = NULL) {
  net <- ctx$net; dev <- ctx$device
  opts <- .tabicl_opts(n_estimators, norm_methods, feat_shuffle_method,
                       class_shuffle_method, outlier_threshold, random_state,
                       quantile_subsample)

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    if (is.factor(y)) {
      levels_ <- levels(y); y_int <- as.integer(y) - 1L
    } else {
      levels_ <- sort(unique(y)); y_int <- match(y, levels_) - 1L
    }
    if (length(levels_) > net$max_classes) {
      cli::cli_abort(
        "This TabICL checkpoint supports at most {net$max_classes} classes; \\
         got {length(levels_)}."
      )
    }
    prep <- .tabicl_prepare(X, y_int, TRUE, opts)
    c(prep, list(class_levels = levels_, n_train = nrow(X)))
  }

  .chunk <- function(state, X_test_chunk, caches) {
    X_test <- transform_simple_imputer(as.matrix(X_test_chunk), state$imputer)
    members <- tabicl_ensemble_transform(state$gen, X_test)
    n_cls <- length(state$class_levels)
    acc <- NULL
    for (i in seq_along(members)) {
      mem <- members[[i]]
      blk <- .tabicl_member_out(net, dev, mem, state$n_train, caches, i,
                                row_chunk_size, save_peak_memory_factor,
                                col_chunk_size)
      logits <- as.matrix(blk[, 1:n_cls]$cpu())
      # Column `p` of the member's output is P(permuted class p), and the
      # permutation mapped original class k to `class_shuffle[k]`, so
      # reindexing by the permutation itself puts things back -- no
      # inversion needed.
      logits <- logits[, mem$class_shuffle + 1L, drop = FALSE]
      contrib <- if (average_logits) logits else
        .softmax_rows(logits, softmax_temperature)
      acc <- if (is.null(acc)) contrib else acc + contrib
    }
    avg <- acc / length(members)
    if (average_logits) avg <- .softmax_rows(avg, softmax_temperature)
    avg / rowSums(avg)
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

# Row-wise temperature-scaled softmax, max-subtracted as the reference
# does it.
# @keywords internal
.softmax_rows <- function(x, temperature = 1) {
  x <- x / temperature
  x <- x - apply(x, 1L, max)
  e <- exp(x)
  e / rowSums(e)
}


#' Build the TabICL regressor predictor
#'
#' TabICL's regression head emits a whole quantile grid (999 levels for
#' the released checkpoint), and the reference does not read summaries
#' straight off it. It wraps the grid in a distribution first —
#' sorting to remove quantile crossing, then interpolating between the
#' levels and extrapolating past the outermost ones with exponential
#' tails. `type = "mean"` is the mean of that distribution's sorted grid,
#' *not* its median, and `type = "quantiles"` interpolates rather than
#' picking the nearest available level.
#'
#' @inheritParams tabicl_classifier
#' @keywords internal
tabicl_regressor <- function(ctx, n_estimators = 8L, norm_methods = NULL,
                             feat_shuffle_method = "latin",
                             outlier_threshold = 4.0,
                             random_state = 42L,
                             quantile_subsample = NULL,
                             predict_chunk_size = 1024L,
                             kv_cache = FALSE,
                             row_chunk_size = NULL,
                             save_peak_memory_factor = NULL,
                             col_chunk_size = NULL) {
  net <- ctx$net; dev <- ctx$device
  n_q <- as.integer(ctx$config$num_quantiles)
  # Level k of the grid is the (k / (n_q + 1))-th quantile.
  levels_ <- seq_len(n_q) / (n_q + 1)
  opts <- .tabicl_opts(n_estimators, norm_methods, feat_shuffle_method,
                       "none", outlier_threshold, random_state,
                       quantile_subsample)

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    # The network works in standardized target space; `TabICLRegressor`
    # fits a StandardScaler on y and inverse-transforms its output.
    scaler <- fit_target_scaler(y)
    prep <- .tabicl_prepare(X, apply_target_scaler(y, scaler), FALSE, opts)
    c(prep, list(scaler = scaler, n_train = nrow(X)))
  }

  .chunk <- function(state, X_test_chunk, type, quantiles, caches) {
    X_test <- transform_simple_imputer(as.matrix(X_test_chunk), state$imputer)
    members <- tabicl_ensemble_transform(state$gen, X_test)
    acc <- NULL
    for (i in seq_along(members)) {
      blk <- .tabicl_member_out(net, dev, members[[i]], state$n_train, caches,
                                i, row_chunk_size, save_peak_memory_factor,
                                col_chunk_size)
      grid <- as.matrix(blk$cpu())
      stat <- quantile_dist_stat(grid, levels_, type, quantiles)
      stat <- invert_target_scaler(stat, state$scaler)
      acc <- if (is.null(acc)) stat else acc + stat
    }
    acc / length(members)
  }

  predict_fn <- function(state, newdata, type = "mean",
                         quantiles = c(0.1, 0.5, 0.9), ...) {
    if (identical(type, "grid")) type <- "raw_quantiles"
    caches <- if (isTRUE(kv_cache)) member_cache_store() else NULL
    res <- chunk_apply(as.matrix(newdata), predict_chunk_size,
                       function(chunk) .chunk(state, chunk, type, quantiles,
                                              caches))
    if (type %in% c("mean", "median")) return(as.numeric(res))
    if (type == "raw_quantiles") return(res)
    colnames(res) <- paste0("q", format(quantiles, trim = TRUE,
                                        drop0trailing = TRUE))
    res
  }

  list(fit = fit_fn, predict = predict_fn,
       types = c("mean", "median", "quantiles", "grid"),
       quantile_levels = levels_)
}


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

#' Stage list for the TabICL diagram
#'
#' Three stages, each attending over a different axis: a set transformer
#' summarises every column over the rows, an encoder with RoPE mixes a
#' row's columns down to four CLS tokens, and a deep stack learns in
#' context over the rows at that flattened width. The labelled rows are
#' held apart by *slicing* at every stage, never by masking.
#'
#' @param config The checkpoint's parsed `config.json`.
#' @param task `"classification"` or `"regression"`.
#' @keywords internal
tabicl_describe <- function(config, task) {
  E      <- as.integer(config$embed_dim)
  n_cls  <- as.integer(config$row_num_cls)
  D      <- E * n_cls
  ff     <- as.integer(config$ff_factor %||% 2L)
  grp    <- as.integer(config$col_feature_group_size %||% 3L)
  n_col  <- as.integer(config$col_num_blocks)
  n_row  <- as.integer(config$row_num_blocks)
  n_icl  <- as.integer(config$icl_num_blocks)
  h_col  <- as.integer(config$col_nhead)
  h_row  <- as.integer(config$row_nhead)
  h_icl  <- as.integer(config$icl_nhead)
  n_ind  <- as.integer(config$col_num_inds)
  clf    <- as.integer(config$max_classes %||% 0L) > 0L
  n_out  <- if (clf) as.integer(config$max_classes)
            else as.integer(config$num_quantiles)
  ssmax  <- !identical(config$icl_ssmax, "none")

  stages <- list(
    arch_input_stage(detail = "numeric matrix; the wrapper imputes first"),
    arch_stage(
      "in_linear", "Cell projection", kind = "embed",
      group = "Stage 1 - per column",
      detail = sprintf("Linear(%d -> %d) over a column and its %d neighbours; the %d CLS slots pass through untouched",
                       grp, E, grp - 1L, n_cls),
      shape = "(B, F, n, E)", prefix = "col_embedder.in_linear"),
    arch_stage(
      "col_y", "Target encoder (per column)", kind = "embed",
      group = "Stage 1 - per column",
      detail = "added on the labelled rows, so a column's summary is target-aware",
      prefix = "col_embedder.y_encoder"),
    arch_stage(
      "col_tf", "Column set transformer", kind = "attention",
      group = "Stage 1 - per column", repeats = n_col, axis = "rows",
      detail = sprintf("induced self-attention, %d inducing points drawn from the labelled rows only; %d heads",
                       n_ind, h_col),
      shape = "(B, n, F, E)", prefix = "col_embedder.tf_col",
      children = list(
        arch_stage("ind", "Inducing points attend to the rows",
                   kind = "attention", axis = "rows",
                   prefix = "col_embedder.tf_col.blocks.0.multihead_attn1"),
        arch_stage("back", "Rows attend back to the inducing points",
                   kind = "attention", axis = "inducing",
                   prefix = "col_embedder.tf_col.blocks.0.multihead_attn2"),
        arch_stage("iv", "Learned inducing points", kind = "embed",
                   prefix = "col_embedder.tf_col.blocks.0.ind_vectors")
      )),
    arch_stage(
      "cls", "CLS columns", kind = "embed", group = "Stage 2 - per row",
      detail = sprintf("%d learned columns overwrite the reserved slots", n_cls),
      prefix = "row_interactor.cls_tokens"),
    arch_stage(
      "row_tf", "Row encoder", kind = "attention", group = "Stage 2 - per row",
      repeats = n_row, axis = "columns",
      detail = sprintf("RoPE over the column order; %d heads; the last block queries from the CLS columns alone",
                       h_row),
      shape = sprintf("(B, n, %d)", D),
      prefix = c("row_interactor.tf_row", "row_interactor.out_ln"),
      children = list(
        arch_stage("attn", "Attention over a row's columns", kind = "attention",
                   axis = "columns",
                   prefix = "row_interactor.tf_row.blocks.0"),
        arch_stage("rope", "RoPE frequencies", kind = "embed",
                   prefix = "row_interactor.tf_row.rope"),
        arch_stage("oln", "LayerNorm", kind = "norm",
                   prefix = "row_interactor.out_ln")
      )),
    arch_stage(
      "icl_y", "Target encoder (per row)", kind = "embed",
      group = "Stage 3 - in-context",
      detail = sprintf("added on the labelled rows at the flattened width %d", D),
      prefix = c("icl_predictor.y_encoder", "icl_predictor.ln")),
    arch_stage(
      "icl", "In-context learning block", kind = "attention",
      group = "Stage 3 - in-context", repeats = n_icl, axis = "rows",
      detail = sprintf("pre-norm; %d heads of %d; keys and values are the labelled rows only%s",
                       h_icl, as.integer(D / h_icl),
                       if (ssmax) "; scalable softmax" else ""),
      shape = sprintf("(B, n, %d)", D), prefix = "icl_predictor.tf_icl",
      children = list(
        arch_stage("iattn", "Attention over rows, labelled-row keys",
                   kind = "attention", axis = "rows",
                   prefix = "icl_predictor.tf_icl.blocks.0.attn"),
        arch_stage("iln", "LayerNorm x2", kind = "norm",
                   prefix = c("icl_predictor.tf_icl.blocks.0.norm1",
                              "icl_predictor.tf_icl.blocks.0.norm2")),
        arch_stage("imlp", "MLP, GELU", kind = "ffn",
                   prefix = c("icl_predictor.tf_icl.blocks.0.linear1",
                              "icl_predictor.tf_icl.blocks.0.linear2"))
      )),
    arch_stage(
      "dec", "Decoder MLP", kind = "decode", group = "Decode",
      detail = sprintf("%d -> %d -> %d, GELU", D, D * 2L, n_out),
      shape = sprintf("(n_test, %d)", n_out), prefix = "icl_predictor.decoder"),
    if (clf)
      arch_output_stage("Class logits",
                        detail = sprintf("%d slots; softmax over the classes seen", n_out))
    else
      arch_output_stage("Quantile levels",
                        detail = sprintf("%d quantiles of the target", n_out))
  )

  list(
    title = sprintf("TabICL v2 - %s", if (clf) "classifier" else "regressor"),
    subtitle = "soda-inria  -  column set transformer -> row encoder -> in-context stack",
    facts = c(
      "Column blocks"      = n_col,
      "Row blocks"         = n_row,
      "ICL blocks"         = n_icl,
      "Embedding width"    = E,
      "ICL width"          = sprintf("%d (%d CLS x %d)", D, n_cls, E),
      "ICL heads"          = sprintf("%d x %d", h_icl, as.integer(D / h_icl)),
      "Inducing points"    = n_ind,
      "Feature group size" = grp,
      "Feed-forward"       = sprintf("%d x width", ff),
      "Output width"       = n_out,
      "Attention scaling"  = if (ssmax) "learned (scalable softmax)" else "1/sqrt(d)",
      "Train/test split"   = "by slicing, not masking",
      "KV cache"           = "exact"
    ),
    symbols = c(arch_default_symbols(), D = sprintf("ICL width (%d)", D)),
    stages = Filter(Negate(is.null), stages)
  )
}


# ---------------------------------------------------------------------------
# Memory scaling
# ---------------------------------------------------------------------------

#' TabICL's peak-memory shapes
#'
#' Three stages with three different peaks, so the estimate takes the
#' maximum: the column embedder is wide (`rows x column-groups x
#' embed_dim`) but kept linear in rows by its inducing points, the row
#' interactor is the same width over a short sequence, and the in-context
#' stage is quadratic in rows but only `embed_dim * row_num_cls` wide.
#' Summing them would roughly treble the answer.
#'
#' @inheritParams mitra_peak_terms
#' @keywords internal
tabicl_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  n_est <- max(1, as.numeric(opts$n_estimators %||% 1))
  .icl_family_terms(
    n_context    = n_context,
    n_query      = .resident_query(n_query, opts),
    n_features   = n_features,
    embed_dim    = as.numeric(config$embed_dim),
    group_size   = as.numeric(config$col_feature_group_size %||% 3),
    n_cls        = as.numeric(config$row_num_cls %||% 4),
    col_heads    = as.numeric(config$col_nhead %||% 8),
    row_heads    = as.numeric(config$row_nhead %||% 8),
    icl_heads    = as.numeric(config$icl_nhead %||% 8),
    col_inducing = as.numeric(config$col_num_inds %||% 128),
    icl_blocks   = as.numeric(config$icl_num_blocks %||% 12),
    kv_cache     = isTRUE(opts$kv_cache),
    n_estimators = n_est,
    # TabICL gained both axes with the stage-chunking port; neither is on
    # unless asked for, which is what `Inf` comes back as.
    row_chunk    = .stage_row_chunk(opts, config),
    col_chunk    = .stage_col_chunk(opts, config),
    group_channels = as.numeric(config$col_feature_group_size %||% 3)
  )
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#' @keywords internal
register_tabicl_backend <- function() {
  register_backend(
    name          = "tabicl",
    build         = tabicl_build,
    describe      = tabicl_describe,
    translate_key = tabicl_translate_key,
    detect        = tabicl_detect,
    task_of       = tabicl_task_of,
    classifier    = tabicl_classifier,
    regressor     = tabicl_regressor,
    peak_terms    = tabicl_peak_terms,
    # The *network* still has no missing-value handling -- NaN in, NaN
    # logits out. The predictors do: they run the reference wrapper's
    # mean `SimpleImputer` before the ensemble, so the network never sees
    # one. `tabfound()` therefore leaves imputation to the backend rather
    # than applying its own, which would be a different (unverified)
    # imputer in front of the verified one.
    handles_missing = TRUE,
    description   = "TabICL v2 column/row/ICL transformer (soda-inria)",
    parity        = "tabicl (PyPI)"
  )
}
