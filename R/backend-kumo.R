# NVIDIA Kumo Tabular backend.
#
# A port of `sdm.models.KumoTabular` from NVIDIA's
# `structured-data-models` 0.2.1 (Apache-2.0); weights from
# `nvidia/Kumo-Tabular` (OpenMDW-1.1), converted by
# `inst/python/kumo_convert_ckpt.py`. One architecture in three sizes:
#
#   cells  [..., R, C]         -> Fourier cell embedding  [..., R, C, D]
#   + target embedding on the context rows' cells, K readout tokens per row
#   L x { column block: inducing points summarise each column's context
#         rows, every row reads the summary;
#         row block:    self-attention over the K + C tokens of a row }
#   readout tokens -> RMSNorm -> [..., R, K*D] -> row_project -> [..., R, E]
#   ICL: + target embedding on context rows; every row attends to the
#        context rows; the last layer computes query rows only
#   head(RMSNorm(x_query)) -> 10 logits or 999 quantiles
#
# The blocks live in `R/backend-kumo-layers.R`. Module names equal the
# checkpoint keys.


#' Kumo row embedding
#'
#' @param config A converted Kumo config.
#' @keywords internal
kumo_row_embedding <- torch::nn_module(
  "KumoRowEmbedding",

  initialize = function(config) {
    D <- as.integer(config$cell_channels)
    H <- as.integer(config$num_embedding_heads)
    L <- as.integer(config$num_embedding_layers)
    K <- as.integer(config$num_readout_tokens)
    self$K <- K
    self$is_classifier <- as.integer(config$num_classes %||% 0L) > 0L

    self$cell_embedding <- kumo_cell_embedding(
      D, as.integer(config$group_size), as.integer(config$num_frequencies)
    )
    if (self$is_classifier) {
      self$y_emb <- torch::nn_embedding(as.integer(config$num_classes), D)
    } else {
      self$y_lin <- torch::nn_linear(1L, D, bias = FALSE)
    }
    self$readout_token <- torch::nn_parameter(torch::torch_zeros(K, D))
    self$col_blocks <- torch::nn_module_list(lapply(seq_len(L), function(i) {
      kumo_induced_block(D, H, as.integer(config$num_inducing_points))
    }))
    hidden <- as.integer(config$gate_hidden %||% 64L)
    self$row_blocks <- torch::nn_module_list(lapply(seq_len(L), function(i) {
      kumo_block(D, H,
                 query_scaling = kumo_gated_log_scale(D %/% H, H, hidden),
                 rope = TRUE)
    }))
    self$norm <- kumo_rms_norm(D)
  },

  # The target's embedding for the context rows, `[..., Rt, D]`. `y` is
  # 0-based class codes (long) for the classifier, standardized values
  # for the regressor.
  target_embedding = function(y) {
    if (self$is_classifier) self$y_emb(y + 1L)
    else self$y_lin(y$unsqueeze(-1L))
  },

  # @param x `[..., R, C]`, NaN for missing; the first `length(y)` rows
  #   are context.
  # @param y `[..., Rt]`, or `NULL` when every row is a query row and
  #   `cache` stands in for the context.
  # @param categorical_mask `[..., C]` boolean tensor.
  # @param cache A [kumo_kv_cache()]: the context's column means and each
  #   column block's projected inducing summary.
  # @param record Return those for the rows given (which must all be
  #   context) alongside the output: `list(reps, col_kv, impute_mean)`.
  # @param rows,factor Row chunking of the steps whose keys are fixed --
  #   each column block's output step (keys: the inducing summary) and
  #   each row block (a row attends only to itself). `rows` is a chunk
  #   size, `factor` a number of chunks; the smaller chunk wins.
  # @param cols Columns per pass through each column block. Columns are
  #   that block's batch axis, so slicing them changes nothing but the
  #   float32 rounding a different batch size brings.
  # @return `[..., R, K * D]`.
  forward = function(x, y, categorical_mask, cache = NULL, record = FALSE,
                     rows = NULL, factor = NULL, cols = NULL) {
    Rt <- if (is.null(y)) 0L else y$size(y$dim())
    K <- self$K
    nd <- x$dim()

    cells <- self$cell_embedding(x, categorical_mask, train_size = Rt,
                                 impute_mean = cache$impute_mean)
    dump_if_enabled("kumo_cells", cells)
    if (Rt > 0L) {
      cells$narrow(nd - 1L, 1L, Rt)$add_(
        self$target_embedding(y)$unsqueeze(-2L)$to(dtype = cells$dtype)
      )
    }
    readout <- self$readout_token$to(dtype = cells$dtype)$expand(
      c(cells$shape[seq_len(nd - 1L)], K, self$readout_token$size(2))
    )
    seq <- torch::torch_cat(list(readout, cells), dim = -2L)  # [..., R, K+C, D]
    rm(cells, readout)

    L <- length(self$row_blocks)
    col_kv <- vector("list", L)
    for (i in seq_len(L)) {
      blk <- self$col_blocks[[i]]
      tok <- seq$size(nd)
      cs <- seq$narrow(nd, K + 1L, tok - K)$transpose(-2L, -3L)  # [..., C, R, D]
      kv_i <- cache$col_kv[[i]]
      res <- .kumo_split_apply(function(sl, c0, n) {
        kv <- if (is.null(kv_i)) {
          if (is.null(rows) && is.null(factor) && !isTRUE(record)) {
            # The reference's own call: inducing step and output step in
            # one, from the context rows of this very tensor.
            return(list(out = blk(sl, sl$narrow(nd, 1L, Rt)), kv = NULL))
          }
          ob <- blk$output_block
          ob$attn$project_kv(ob$key_value_norm(blk$induce(sl$narrow(nd, 1L, Rt))))
        } else {
          lapply(kv_i, function(t) t$narrow(nd - 1L, c0, n))
        }
        list(out = .kumo_split_apply(function(q, ...) blk$output_block(q, kv),
                                     sl, nd, rows, factor),
             kv = kv)
      }, cs, nd - 1L, cols, NULL, combine = function(parts) {
        list(out = torch::torch_cat(lapply(parts, `[[`, "out"), dim = nd - 1L),
             kv = if (is.null(parts[[1]]$kv)) NULL else
               lapply(c(key = "key", value = "value"), function(k) {
                 torch::torch_cat(lapply(parts, function(p) p$kv[[k]]), dim = nd - 1L)
               }))
      })
      cs <- res$out
      col_kv[[i]] <- res$kv
      dump_if_enabled(sprintf("kumo_col%d", i - 1L), cs)
      seq <- torch::torch_cat(
        list(seq$narrow(nd, 1L, K), cs$transpose(-2L, -3L)), dim = -2L
      )
      rb <- self$row_blocks[[i]]
      last <- i == L
      seq <- .kumo_split_apply(function(sq, ...) {
        rb(if (last) sq$narrow(nd, 1L, K) else sq, sq)
      }, seq, nd - 1L, rows, factor)
      dump_if_enabled(sprintf("kumo_row%d", i - 1L), seq)
    }
    reps <- self$norm(seq)$flatten(start_dim = -2L)
    if (!isTRUE(record)) return(reps)
    list(reps = reps, col_kv = col_kv,
         impute_mean = self$cell_embedding$impute_mean(x, Rt))
  }
)

# Apply `f` to slices of `x` along `axis` and stitch the results. `size`
# is a slice length, `factor` a number of slices; with neither, or with
# slices no smaller than `x`, it is one call on `x` itself -- the same
# call, so a chunking that does nothing changes nothing. `f` receives the
# slice, its 1-based start and its length.
# @keywords internal
.kumo_split_apply <- function(f, x, axis, size = NULL, factor = NULL,
                              combine = NULL) {
  n <- x$size(axis)
  s <- .kumo_chunk_len(n, size, factor)
  if (is.na(s) || s >= n) return(f(x, 1L, n))
  starts <- seq(1L, n, by = s)
  parts <- lapply(starts, function(a) {
    len <- min(s, n - a + 1L)
    f(x$narrow(axis, a, len), a, len)
  })
  if (is.null(combine)) torch::torch_cat(parts, dim = axis) else combine(parts)
}

.kumo_chunk_len <- function(n, size = NULL, factor = NULL) {
  cand <- c(if (!is.null(size)) as.integer(size),
            if (!is.null(factor)) as.integer(ceiling(n / as.integer(factor))))
  cand <- cand[!is.na(cand) & cand >= 1L]
  if (!length(cand) || n == 0L) NA_integer_ else min(cand)
}


#' Kumo in-context-learning block
#'
#' @param config A converted Kumo config.
#' @keywords internal
kumo_icl_block <- torch::nn_module(
  "KumoICLBlock",

  initialize = function(config) {
    E <- as.integer(config$icl_channels)
    He <- as.integer(config$num_icl_heads)
    n_out <- as.integer(config$num_classes %||% 0L)
    self$is_classifier <- n_out > 0L
    if (!self$is_classifier) n_out <- as.integer(config$num_quantiles)
    # Medium and large let the query rows attend with only the first few
    # key/value heads (grouped-query attention); the context rows always
    # use all of them. Not derivable from the weights -- the converter
    # copies it from the reference's size table.
    kvh <- config$num_icl_key_value_heads_for_query
    self$kv_heads <- if (is.null(kvh) || is.list(kvh) || !length(kvh)) NULL
                     else as.integer(kvh)

    if (self$is_classifier) {
      self$y_emb <- torch::nn_embedding(n_out, E)
    } else {
      self$y_lin <- torch::nn_linear(1L, E, bias = FALSE)
    }
    self$layers <- torch::nn_module_list(lapply(
      seq_len(as.integer(config$num_icl_layers)),
      function(j) kumo_block(E, He, query_scaling = kumo_log_scale(He))
    ))
    self$norm <- kumo_rms_norm(E)
    self$head <- torch::nn_sequential(
      torch::nn_linear(E, 2L * E),
      torch::nn_gelu(),
      torch::nn_linear(2L * E, n_out)
    )
  },

  # @param x `[..., R, E]`, context rows first.
  # @param y `[..., Rt]`, as for [kumo_row_embedding]; `NULL` with
  #   `icl_kv`, when every row is a query row.
  # @param icl_kv Per layer, the context's key/value projections (already
  #   cut to the query heads).
  # @param record Every row is context: return those projections instead
  #   of predicting.
  # @param rows,factor Row chunking of the passes whose keys are fixed.
  # @return `[..., R - Rt, n_out]`, or with `record` the per-layer list.
  forward = function(x, y, icl_kv = NULL, record = FALSE, rows = NULL,
                     factor = NULL) {
    Rt <- if (is.null(y)) 0L else y$size(y$dim())
    nd <- x$dim()
    R <- x$size(nd - 1L)
    ax <- nd - 1L
    if (Rt > 0L) {
      emb <- if (self$is_classifier) self$y_emb(y + 1L)
             else self$y_lin(y$unsqueeze(-1L))
      x <- torch::torch_cat(list(x$narrow(ax, 1L, Rt) + emb$to(dtype = x$dtype),
                                 x$narrow(ax, Rt + 1L, R - Rt)), dim = ax)
    }
    L <- length(self$layers)
    against <- function(layer, kv) {
      force(layer); force(kv)
      function(q, ...) layer(q, kv)
    }
    cut <- function(kv) {
      if (is.null(self$kv_heads)) return(kv)
      lapply(kv, function(t) t$narrow(t$dim() - 1L, 1L, self$kv_heads)$contiguous())
    }

    if (!is.null(icl_kv)) {
      for (j in seq_len(L)) {
        x <- .kumo_split_apply(against(self$layers[[j]], icl_kv[[j]]), x, ax,
                               rows, factor)
      }
      return(self$head(self$norm(x)))
    }

    if (isTRUE(record)) {
      out <- vector("list", L)
      for (j in seq_len(L)) {
        layer <- self$layers[[j]]
        kv <- layer$attn$project_kv(layer$key_value_norm(x))
        out[[j]] <- cut(kv)
        if (j < L) x <- .kumo_split_apply(against(layer, kv), x, ax, rows, factor)
      }
      return(out)
    }

    # Only a chunking that actually splits something leaves the
    # reference's one-call path: `factor = 1` is the plain forward.
    s_ <- .kumo_chunk_len(R - Rt, rows, factor)
    chunked <- !is.na(s_) && s_ < R - Rt
    for (j in seq_len(L)) {
      layer <- self$layers[[j]]
      last <- j == L
      ctx <- x$narrow(ax, 1L, Rt)
      qry <- x$narrow(ax, Rt + 1L, R - Rt)
      if (is.null(self$kv_heads) && !chunked) {
        # One call over every row, as the reference makes it: the same
        # call shapes are what keep the rounding the same.
        x <- layer(if (last) qry else x, ctx)
        next
      }
      # Context rows attend with every head; query rows with the first
      # `kv_heads` of the same projections (all of them on small).
      kv <- layer$attn$project_kv(layer$key_value_norm(ctx))
      xq <- .kumo_split_apply(against(layer, cut(kv)), qry, ax, rows, factor)
      x <- if (last) xq else torch::torch_cat(
        list(.kumo_split_apply(against(layer, kv), ctx, ax, rows, factor), xq),
        dim = ax)
    }
    self$head(self$norm(x))
  }
)


#' Kumo Tabular network
#'
#' @param config A converted Kumo config (`arch = "kumo_tabular"`).
#' @keywords internal
kumo_model <- torch::nn_module(
  "KumoTabular",

  initialize = function(config) {
    self$config <- config
    self$row_embedding <- kumo_row_embedding(config)
    kd <- as.integer(config$num_readout_tokens) * as.integer(config$cell_channels)
    E <- as.integer(config$icl_channels)
    self$row_project <- if (kd != E) torch::nn_linear(kd, E) else torch::nn_identity()
    self$icl_block <- kumo_icl_block(config)
    # Read by the predictors and the cache machinery.
    self$supports_kv_cache <- TRUE
    self$supports_stage_chunking <- TRUE
    self$supports_chunked_eval <- TRUE
    self$kv_cache_is_exact <- TRUE
  },

  .prepare = function(x, y, categorical_mask) {
    if (!inherits(categorical_mask, "torch_tensor")) {
      categorical_mask <- torch::torch_tensor(as.logical(categorical_mask),
                                              dtype = torch::torch_bool(),
                                              device = x$device)
    }
    nd <- x$dim()
    mask <- categorical_mask$to(dtype = torch::torch_bool())$expand(
      c(x$shape[seq_len(nd - 2L)], x$size(nd))
    )
    if (!is.null(y)) {
      y <- if (self$row_embedding$is_classifier) y$to(dtype = torch::torch_long())
           else y$to(dtype = x$dtype)
    }
    list(y = y, mask = mask)
  },

  # @param x `[..., R, C]` float, NaN for missing, context rows first.
  # @param y `[..., Rt]`: 0-based class codes for the classifier,
  #   standardized targets for the regressor. Its length is the number
  #   of context rows.
  # @param categorical_mask Logical vector of length `C`, or a boolean
  #   tensor broadcastable to `[..., C]`.
  # @param kv_cache A [kumo_kv_cache()]; then every row of `x` is a query
  #   row and `y` is ignored.
  # @param row_chunk_size Rows per pass. Set below the row count, the
  #   context is encoded on its own into a cache and the query rows then
  #   stream through against it, so only one chunk of them is in flight.
  # @param save_peak_memory_factor Split every step whose keys are fixed
  #   into this many chunks of rows.
  # @param col_chunk_size Columns per pass through each column block.
  # @return `[..., R - Rt, 10]` logits or `[..., R - Rt, 999]` quantiles.
  forward = function(x, y, categorical_mask, kv_cache = NULL,
                     row_chunk_size = NULL, save_peak_memory_factor = NULL,
                     col_chunk_size = NULL) {
    if (!is.null(kv_cache)) {
      return(self$forward_cached(x, kv_cache, row_chunk_size,
                                 save_peak_memory_factor, col_chunk_size))
    }
    p <- self$.prepare(x, y, categorical_mask)
    nd <- x$dim()
    R <- x$size(nd - 1L)
    Rt <- p$y$size(p$y$dim())
    if (!is.na(.kumo_chunk_len(R, row_chunk_size)) && row_chunk_size < R) {
      cache <- self$build_kv_cache(x$narrow(nd - 1L, 1L, Rt), p$y, p$mask,
                                   row_chunk_size, save_peak_memory_factor,
                                   col_chunk_size)
      return(self$forward_cached(x$narrow(nd - 1L, Rt + 1L, R - Rt), cache,
                                 row_chunk_size, save_peak_memory_factor,
                                 col_chunk_size))
    }
    reps <- self$row_embedding(x, p$y, p$mask, factor = save_peak_memory_factor,
                               cols = col_chunk_size)
    dump_if_enabled("kumo_reps", reps)
    out <- self$icl_block(self$row_project(reps), p$y,
                          factor = save_peak_memory_factor)
    dump_if_enabled("kumo_logits", out)
    out
  },

  # Everything the context rows contribute to a prediction.
  #
  # @param x `[..., Rt, C]`, the context rows only.
  # @return A [kumo_kv_cache()].
  build_kv_cache = function(x, y, categorical_mask, row_chunk_size = NULL,
                            save_peak_memory_factor = NULL,
                            col_chunk_size = NULL) {
    p <- self$.prepare(x, y, categorical_mask)
    enc <- self$row_embedding(x, p$y, p$mask, record = TRUE,
                              rows = row_chunk_size,
                              factor = save_peak_memory_factor,
                              cols = col_chunk_size)
    icl_kv <- self$icl_block(self$row_project(enc$reps), p$y, record = TRUE,
                             rows = row_chunk_size,
                             factor = save_peak_memory_factor)
    kumo_kv_cache(enc$impute_mean, enc$col_kv, icl_kv,
                  n_train = x$size(x$dim() - 1L), n_features = x$size(x$dim()),
                  categorical_mask = as.logical(
                    p$mask$reshape(c(-1L, x$size(x$dim())))[1, ]$cpu()))
  },

  # Predict query rows against a [kumo_kv_cache()].
  forward_cached = function(x, kv_cache, row_chunk_size = NULL,
                            save_peak_memory_factor = NULL,
                            col_chunk_size = NULL) {
    nd <- x$dim()
    if (x$size(nd) != kv_cache$n_features) {
      cli::cli_abort(
        "This cache was built for {kv_cache$n_features} feature{?s}; \\
         got {x$size(nd)}."
      )
    }
    p <- self$.prepare(x, NULL, kv_cache$categorical_mask)
    # Query rows see only the context, never each other, so they can go
    # through a chunk at a time with nothing but the cache in common.
    .kumo_split_apply(function(xq, a, n) {
      reps <- self$row_embedding(xq, NULL, p$mask, cache = kv_cache,
                                 factor = save_peak_memory_factor,
                                 cols = col_chunk_size)
      self$icl_block(self$row_project(reps), NULL, icl_kv = kv_cache$icl_kv,
                     factor = save_peak_memory_factor)
    }, x, nd - 1L, row_chunk_size)
  }
)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

#' Everything the context rows contribute to a Kumo Tabular prediction
#'
#' Context rows never attend to query rows anywhere in the network: the
#' cell embedding imputes from context means, each column block's inducing
#' points read the context rows only, a row block is attention within one
#' row, and in the ICL stack only context rows are keys. So the context
#' half of the forward pass is a function of the context alone, and this
#' is it -- what `sdm`'s `fit()` records and its `predict()` replays:
#'
#' * `impute_mean` -- the context column means, `[..., 1, C]`;
#' * `col_kv` -- per column block, the output step's key/value projection
#'   of the `M` inducing vectors, `[..., C, M, H, hd]` (fixed size,
#'   however many context rows);
#' * `icl_kv` -- per ICL layer, the context rows' keys and values, cut to
#'   the heads the query rows use (two on medium and large).
#'
#' Predicting against it differs from the one-shot forward only by the
#' float32 rounding that a different batch shape brings (the reference's
#' own two paths differ by up to 2.4e-6 on the parity fixtures).
#'
#' @param impute_mean,col_kv,icl_kv See above.
#' @param n_train Context rows it was built from.
#' @param n_features Columns it was built for.
#' @param categorical_mask The member's categorical flags.
#' @keywords internal
kumo_kv_cache <- function(impute_mean, col_kv, icl_kv, n_train, n_features,
                          categorical_mask) {
  structure(
    list(impute_mean = impute_mean, col_kv = col_kv, icl_kv = icl_kv,
         n_train = as.integer(n_train), n_features = as.integer(n_features),
         categorical_mask = as.logical(categorical_mask)),
    class = "kumo_kv_cache"
  )
}

#' @export
print.kumo_kv_cache <- function(x, ...) {
  n_bytes <- .tensor_bytes(x$impute_mean) + .tensor_bytes(x$col_kv) +
    .tensor_bytes(x$icl_kv)
  cli::cli_text("{.strong Kumo Tabular KV cache}")
  cli::cli_bullets(c(
    "*" = "built from {.val {x$n_train}} context row{?s}, \\
           {.val {x$n_features}} column{?s}",
    "*" = "{length(x$col_kv)} column block{?s}, \\
           {length(x$icl_kv)} ICL layer{?s}, \\
           {round(n_bytes / 1e6, 1)} MB"
  ))
  invisible(x)
}


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

# One member's network output for the query rows. Without a cache the
# context rows and the chunk are transformed together (every transform is
# row-wise once fitted) and go through one forward pass, as `sdm`'s
# `KumoTabular.forward()` makes it. With one, the member's context is
# encoded once per store -- what the reference's `fit()` records -- and
# the chunk alone runs against it.
#
# With more classes than the network separates, the member's codebook
# turns one task into `T`: the rows are repeated along a leading task
# axis, each task sees its own relabelled context, and the `T` logit sets
# are folded back into `K` class scores by [.kumo_ecoc_scores()]. The
# cache then carries the task axis too.
# @keywords internal
.kumo_member_out <- function(net, dev, rec, X_train, y_train, X_query, m,
                             cache_store = NULL, chunks = list()) {
  run <- function(...) {
    do.call(net, c(list(...), list(row_chunk_size = chunks$row_chunk_size,
                                   save_peak_memory_factor = chunks$save_peak_memory_factor,
                                   col_chunk_size = chunks$col_chunk_size)))
  }
  book <- rec$target$codebooks[[m]]
  tasks <- if (is.null(book)) 1L else nrow(book)
  lead <- function(x) x$to(device = dev)$unsqueeze(1L)$expand(c(tasks, -1L, -1L))
  target <- function() {
    yt <- kumo_recipe_target(rec, y_train, m)
    if (is.null(book)) return(yt$to(device = dev)$unsqueeze(1L))
    code <- as.integer(yt)
    torch::torch_tensor(book[, code + 1L, drop = FALSE], dtype = torch::torch_long(),
                        device = dev)
  }
  fold <- function(out) {
    if (is.null(book)) out[1, , ]$cpu() else .kumo_ecoc_scores(out, book)$cpu()
  }
  if (is.null(cache_store)) {
    mem <- kumo_recipe_member(rec, rbind(X_train, X_query), m)
    out <- torch::with_no_grad(run(lead(mem$x), target(), mem$categorical_mask))
    return(fold(out))
  }
  cache <- member_cache(cache_store, m, function() {
    mem <- kumo_recipe_member(rec, X_train, m)
    torch::with_no_grad(net$build_kv_cache(
      lead(mem$x), target(), mem$categorical_mask, chunks$row_chunk_size,
      chunks$save_peak_memory_factor, chunks$col_chunk_size
    ))
  })
  mem <- kumo_recipe_member(rec, X_query, m)
  out <- torch::with_no_grad(run(lead(mem$x), NULL, mem$categorical_mask,
                                 kv_cache = cache))
  fold(out)
}

#' Fold error-correcting sub-task logits into class scores
#'
#' `sdm.models.ECOC.forward()`'s last step: each sub-task's logits become
#' log-probabilities, class `k` reads its own symbol's in every task, and
#' the score is their mean over the tasks in which `k` is not merged into
#' the rest symbol.
#'
#' @param logits `[T, n, max_classes]`.
#' @param book Integer codebook `[T, K]`, 0-based.
#' @return `[n, K]`.
#' @keywords internal
.kumo_ecoc_scores <- function(logits, book) {
  rest <- logits$size(3L) - 1L
  idx <- torch::torch_tensor(array(book, c(nrow(book), 1L, ncol(book))),
                             dtype = torch::torch_long(), device = logits$device)
  scores <- torch::nnf_log_softmax(logits, dim = -1L)$gather(
    -1L, (idx + 1L)$expand(c(logits$size(1L), logits$size(2L), ncol(book)))
  )
  active <- idx != rest
  scores$masked_fill(active$logical_not(), 0)$sum(dim = 1L) / active$sum(dim = 1L)
}

# 1-based declared column indices -> one logical per column.
# @keywords internal
.kumo_categorical <- function(categorical_features, p) {
  idx <- as.integer(categorical_features %||% integer())
  if (any(idx < 1L | idx > p)) {
    cli::cli_abort("{.arg categorical_features} names columns outside the {p} predictors.")
  }
  seq_len(p) %in% idx
}

#' Build the Kumo Tabular classifier predictor
#'
#' Reproduces `sdm.models.KumoTabular(task = "classification")` called
#' with `num_estimators = n_estimators` and a generator seeded at `seed`:
#' the default recipe builds `n_estimators` members (see
#' [kumo_recipe_fit()]), each runs through the network, and the member
#' logits are averaged in class order and passed through a softmax.
#'
#' The library's own default is one member; the model card and NVIDIA's
#' benchmark wrappers use 8, which is the default here.
#'
#' Classes that do not occur in the training rows cannot be predicted --
#' the recipe drops them, as the reference's does -- and come back with
#' probability zero. Above 10 classes the network's 10 outputs are not
#' enough, and each member runs error-correcting output codes as the
#' reference does: its own random codebook splits the task into `T`
#' ten-way sub-tasks (8 for 15 or 20 classes, 28 for 250), which run as one
#' batch, and each class is scored by its mean log-probability over the
#' sub-tasks that give it a symbol of its own. Memory and time grow by
#' `T`.
#'
#' Columns are categorical only when declared: `sdm`'s type inference
#' keeps numeric columns numerical. [tabfound()] declares a frame's
#' factor columns for you.
#'
#' @param ctx Loaded-model context from [load_backend_model()].
#' @param n_estimators Ensemble members.
#' @param seed Seed for the recipe's generator. Draws are libtorch's own
#'   stream, so on the same platform they are the reference's.
#' @param predict_chunk_size Max query rows per forward pass. Query rows
#'   attend only to context rows, so chunking changes nothing but float32
#'   rounding.
#' @param kv_cache Encode each member's context once per `predict()` call
#'   and run every chunk against it, rather than re-encoding it with each
#'   chunk -- what the reference's `fit()` / `predict()` split does.
#'   [tabfound_cache()] keeps it across calls and sessions. Off by
#'   default; the answer moves only by float32 rounding (see
#'   [kumo_kv_cache()]).
#' @param save_peak_memory_factor Split every step whose keys are fixed
#'   (each column block's output step, each row block, the query rows of
#'   the ICL stack) into this many chunks of rows.
#' @param row_chunk_size Rows in flight per pass: the context is encoded
#'   on its own with its row-independent steps a chunk at a time, then the
#'   query rows stream through against it.
#' @param col_chunk_size Columns per pass through each column block.
#' @param categorical_features 1-based indices of categorical columns.
#' @keywords internal
kumo_classifier <- function(ctx, n_estimators = 8L, seed = 42L,
                            predict_chunk_size = 1024L, kv_cache = FALSE,
                            save_peak_memory_factor = NULL,
                            row_chunk_size = NULL, col_chunk_size = NULL,
                            categorical_features = NULL) {
  net <- ctx$net; dev <- ctx$device
  max_classes <- as.integer(ctx$config$num_classes)
  chunks <- list(row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size,
                 save_peak_memory_factor = save_peak_memory_factor)

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    if (is.factor(y)) {
      levels_ <- levels(y); y_int <- as.integer(y) - 1L
    } else {
      levels_ <- sort(unique(y)); y_int <- match(y, levels_) - 1L
    }
    if (anyNA(y_int)) cli::cli_abort("{.arg y} has missing values.")
    categorical <- .kumo_categorical(categorical_features, ncol(X))
    rec <- kumo_recipe_fit(X, y_int, TRUE, categorical = categorical,
                           n_members = n_estimators, seed = seed,
                           max_classes = max_classes)
    list(X_train = X, y_train_int = y_int, class_levels = levels_,
         n_train = nrow(X), categorical = categorical, recipe = rec)
  }

  .chunk <- function(state, X_query, caches) {
    rec <- state$recipe
    K <- length(rec$target$classes)
    outs <- lapply(seq_len(rec$n_members), function(m) {
      if (m > 1L) collect_between_chunks()
      o <- .kumo_member_out(net, dev, rec, state$X_train, state$y_train_int,
                            X_query, m, caches, chunks)
      kumo_recipe_unshift(rec, o[, 1:K, drop = FALSE], m)
    })
    p <- as.matrix(kumo_recipe_reduce(outs, TRUE))
    full <- matrix(0, nrow(X_query), length(state$class_levels))
    full[, rec$target$classes + 1L] <- p
    full
  }

  predict_fn <- function(state, newdata, type = "class", ...) {
    caches <- .kumo_store(state, kv_cache)
    probs <- chunk_apply(as.matrix(newdata), predict_chunk_size,
                         function(chunk) .chunk(state, chunk, caches))
    colnames(probs) <- as.character(state$class_levels)
    if (type == "prob") return(probs)
    state$class_levels[max.col(probs, ties.method = "first")]
  }

  list(fit = fit_fn, predict = predict_fn,
       build_cache = function(state) .kumo_build_all(state, function(st, one, store) {
         .chunk(st, one, store)
       }))
}

# A cache carried on the fitted state was built once, possibly in another
# session; otherwise build lazily, per `predict()` call, when asked to.
# @keywords internal
.kumo_store <- function(state, kv_cache) {
  member_cache_store_from(state$kv_caches) %||%
    (if (isTRUE(kv_cache)) member_cache_store() else NULL)
}

# Building every member's cache is what one prediction over a single
# query row does, so ask for that: the store comes back full, and the
# row's own answer is thrown away.
# @keywords internal
.kumo_build_all <- function(state, chunk_fn) {
  store <- member_cache_store()
  one <- state$X_train[1, , drop = FALSE]
  chunk_fn(state, one, store)
  member_cache_list(store)
}

#' Build the Kumo Tabular regressor predictor
#'
#' Reproduces `sdm.models.KumoTabular(task = "regression")`: each member
#' predicts 999 quantiles (levels 0.001 to 0.999) of its standardized,
#' randomly signed target; the sign and the scaling are undone, each
#' member's grid is sorted, and the members are combined by a trimmed mean
#' that drops `floor(0.2 * n_estimators)` from each tail, level by level.
#' `type = "mean"` is the mean of that grid -- the point estimate NVIDIA's
#' AutoGluon wrapper reports; `"median"` and `"quantiles"` interpolate it.
#'
#' @inheritParams kumo_classifier
#' @keywords internal
kumo_regressor <- function(ctx, n_estimators = 8L, seed = 42L,
                           predict_chunk_size = 1024L, kv_cache = FALSE,
                           save_peak_memory_factor = NULL,
                           row_chunk_size = NULL, col_chunk_size = NULL,
                           categorical_features = NULL) {
  net <- ctx$net; dev <- ctx$device
  n_q <- as.integer(ctx$config$num_quantiles)
  chunks <- list(row_chunk_size = row_chunk_size, col_chunk_size = col_chunk_size,
                 save_peak_memory_factor = save_peak_memory_factor)
  levels_ <- seq_len(n_q) / (n_q + 1)

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    y <- as.numeric(y)
    if (anyNA(y)) cli::cli_abort("{.arg y} has missing values.")
    categorical <- .kumo_categorical(categorical_features, ncol(X))
    rec <- kumo_recipe_fit(X, y, FALSE, categorical = categorical,
                           n_members = n_estimators, seed = seed)
    list(X_train = X, y_train = y, n_train = nrow(X),
         y_mean = rec$target$mean, y_std = rec$target$scale,
         categorical = categorical, recipe = rec)
  }

  .grid <- function(state, X_query, caches = NULL) {
    rec <- state$recipe
    outs <- lapply(seq_len(rec$n_members), function(m) {
      if (m > 1L) collect_between_chunks()
      o <- .kumo_member_out(net, dev, rec, state$X_train, state$y_train,
                            X_query, m, caches, chunks)
      kumo_recipe_inverse(rec, o, m)
    })
    as.matrix(kumo_recipe_reduce(outs, FALSE))
  }

  predict_fn <- function(state, newdata, type = "mean",
                         quantiles = c(0.1, 0.5, 0.9), ...) {
    if (identical(type, "grid")) type <- "raw_quantiles"
    caches <- .kumo_store(state, kv_cache)
    res <- chunk_apply(as.matrix(newdata), predict_chunk_size, function(chunk) {
      g <- .grid(state, chunk, caches)
      quantile_dist_stat(g, levels_, type, quantiles)
    })
    if (type %in% c("mean", "median")) return(as.numeric(res))
    if (type == "raw_quantiles") return(res)
    colnames(res) <- paste0("q", format(quantiles, trim = TRUE,
                                        drop0trailing = TRUE))
    res
  }

  list(fit = fit_fn, predict = predict_fn,
       build_cache = function(state) .kumo_build_all(state, function(st, one, store) {
         .grid(st, one, store)
       }),
       types = c("mean", "median", "quantiles", "grid"),
       quantile_levels = levels_)
}


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------

#' @keywords internal
# The children of one Kumo transformer block, in the order it runs them:
# keys/values and then queries are normed apart, then attention, then the
# MLP, whose own RMSNorm is its first layer. Both sit inside residual
# connections.
# @keywords internal
kumo_block_children <- function(pfx, id, label, axis, attn) {
  p <- function(x) paste0(pfx, ".", x)
  list(
    arch_residual(
      arch_stage(paste0(id, "_nkv"), "RMSNorm, keys and values", kind = "norm",
                 prefix = p("key_value_norm")),
      arch_stage(paste0(id, "_nq"), "RMSNorm, queries", kind = "norm",
                 prefix = p("query_norm")),
      arch_stage(id, label, kind = "attention", axis = axis, attn = attn,
                 prefix = p("attn"))
    ),
    arch_residual(
      arch_stage(paste0(id, "_n2"), "RMSNorm", kind = "norm", prefix = p("mlp.0")),
      arch_stage(paste0(id, "_mlp"), "MLP, GELU", kind = "ffn",
                 prefix = p(c("mlp.1", "mlp.3")))
    )
  )
}

# @keywords internal
kumo_col_children <- function(pfx = "row_embedding.col_blocks.0") {
  c(
    list(arch_stage("iv", "Learned inducing points", kind = "embed",
                    role = "input", feeds = "ind",
                    prefix = paste0(pfx, ".inducing_points"))),
    kumo_block_children(paste0(pfx, ".inducing_block"), "ind",
                        "Inducing points attend to the context rows",
                        axis = "rows",
                        attn = arch_attn("context rows", qk_norm = TRUE,
                                         scaling = "log-n")),
    kumo_block_children(paste0(pfx, ".output_block"), "back",
                        "Rows attend to the inducing summary",
                        axis = "inducing",
                        attn = arch_attn("inducing points", qk_norm = TRUE))
  )
}

# A repeated stack of blocks, then the norm it ends with.
# @keywords internal
kumo_stack_children <- function(pfx, id, label, axis, final_norm, attn) {
  c(
    kumo_block_children(pfx, id, label, axis, attn),
    list(arch_stage("onorm", "RMSNorm, after the stack", kind = "norm",
                    role = "final", prefix = final_norm))
  )
}

kumo_describe <- function(config, task) {
  D   <- as.integer(config$cell_channels)
  E   <- as.integer(config$icl_channels)
  K   <- as.integer(config$num_readout_tokens)
  H   <- as.integer(config$num_embedding_heads)
  He  <- as.integer(config$num_icl_heads)
  L   <- as.integer(config$num_embedding_layers)
  Li  <- as.integer(config$num_icl_layers)
  M   <- as.integer(config$num_inducing_points)
  G   <- as.integer(config$group_size %||% 3L)
  Fq  <- as.integer(config$num_frequencies %||% 32L)
  clf <- as.integer(config$num_classes %||% 0L) > 0L
  n_out <- if (clf) as.integer(config$num_classes) else as.integer(config$num_quantiles)
  kvh <- config$num_icl_key_value_heads_for_query
  kvh <- if (is.null(kvh) || is.list(kvh) || !length(kvh)) NULL else as.integer(kvh)
  proj <- K * D != E
  yname <- if (clf) "y_emb" else "y_lin"

  stages <- list(
    arch_input_stage(detail = "numeric matrix with NaN kept; categoricals as codes plus a per-column flag"),
    arch_stage(
      "cells", "Cell embedding", kind = "embed", group = "Row embedding",
      detail = sprintf("each cell with two neighbours (offsets 0, 1, 3, wrapping): sin/cos of %d frequencies per slot, numerical or categorical; context-mean imputation plus a projection of the missingness",
                       Fq),
      shape = sprintf("(R, C, %d)", D), prefix = "row_embedding.cell_embedding"),
    arch_stage(
      "row_y", "Target embedding (cells)", kind = "embed", group = "Row embedding",
      detail = "added to every cell of the context rows",
      join = "add", join_rows = "context rows",
      prefix = paste0("row_embedding.", yname)),
    arch_stage(
      "readout", "Readout tokens", kind = "embed", group = "Row embedding",
      detail = sprintf("%d learned tokens prepended to every row; the column blocks skip them", K),
      join = "prepend",
      prefix = "row_embedding.readout_token"),
    arch_stage(
      "col", "Column block", kind = "attention", group = "Row embedding",
      repeats = L, axis = "rows", interleave = "embedding",
      detail = sprintf("induced attention over a column's cell tokens: %d inducing points read the context rows (log-n query scaling), every row reads them back; %d heads",
                       M, H),
      shape = sprintf("(C, R, %d)", D), prefix = "row_embedding.col_blocks",
      children = kumo_col_children()),
    arch_stage(
      "row", "Row block", kind = "attention", group = "Row embedding",
      repeats = L, axis = "columns", interleave = "embedding",
      detail = sprintf("self-attention over a row's %d readout + C cell tokens; RoPE, gated log-n query scaling; the last block queries from the readout tokens only; %d heads",
                       K, H),
      shape = sprintf("(R, %d + C, %d)", K, D),
      # Without a projection (small), the in-context stack reads the
      # flattened readout tokens straight out of the last row block.
      collapses = !proj,
      prefix = c("row_embedding.row_blocks", "row_embedding.norm"),
      children = kumo_stack_children("row_embedding.row_blocks.0", "rattn", "Attention over a row's tokens", "columns", "row_embedding.norm",
                                     attn = arch_attn("all", queries = "readout tokens",
                                                      qk_norm = TRUE,
                                                      scaling = "gated log-n"))),
    if (proj) arch_stage(
      "proj", "Row projection", kind = "embed", group = "In-context",
      detail = sprintf("Linear(%d -> %d) from the flattened readout tokens", K * D, E),
      prefix = "row_project", collapses = TRUE),
    arch_stage(
      "icl_y", "Target embedding (rows)", kind = "embed", group = "In-context",
      detail = "added to the context rows",
      join = "add", join_rows = "context rows",
      prefix = paste0("icl_block.", yname)),
    arch_stage(
      "icl", "In-context learning block", kind = "attention", group = "In-context",
      repeats = Li, axis = "rows",
      detail = sprintf("every row attends to the context rows, log-n query scaling; %d heads of %d%s",
                       He, E %/% He,
                       if (is.null(kvh)) "" else
                         sprintf("; query rows use %d of the key/value heads", kvh)),
      shape = sprintf("(R, %d)", E), prefix = c("icl_block.layers", "icl_block.norm"),
      children = kumo_stack_children("icl_block.layers.0", "iattn", "Attention over context rows", "rows", "icl_block.norm",
                                     attn = arch_attn("context rows", queries = "query rows",
                                                      qk_norm = TRUE, scaling = "log-n",
                                                      kv_heads_test = kvh))),
    arch_stage(
      "head", "Decoder MLP", kind = "decode", group = "Decode",
      detail = sprintf("%d -> %d -> %d, GELU", E, 2L * E, n_out),
      shape = sprintf("(n_test, %d)", n_out), prefix = "icl_block.head"),
    if (clf)
      arch_output_stage("Class logits",
                        detail = sprintf("%d slots; error-correcting codes above %d classes", n_out, n_out))
    else
      arch_output_stage("Quantile levels",
                        detail = sprintf("%d quantiles of the standardized target", n_out))
  )

  list(
    title = sprintf("Kumo Tabular %s - %s", config$size %||% "",
                    if (clf) "classifier" else "regressor"),
    subtitle = "NVIDIA  -  cell embedding -> column/row blocks -> in-context stack",
    facts = c(
      # One column block and one row block per embedding layer.
      "Column-stage blocks" = L,
      "Row-stage blocks" = L,
      "ICL layers"       = Li,
      "Embedding width"  = D,
      "ICL width"        = if (proj) sprintf("%d (projected from %d x %d)", E, K, D)
                           else sprintf("%d (%d readout x %d)", E, K, D),
      "ICL heads"        = sprintf("%d x %d", He, E %/% He),
      "Embedding heads"  = H,
      "Query KV heads"   = if (is.null(kvh)) "all" else kvh,
      "Inducing points"  = M,
      "Features per group" = G,
      "Fourier frequencies" = Fq,
      "Output width"     = n_out,
      "Missing values"   = "handled by the network"
    ),
    symbols = c(arch_default_symbols(), R = "rows", C = "columns"),
    stages = Filter(Negate(is.null), stages)
  )
}


# ---------------------------------------------------------------------------
# Memory scaling
# ---------------------------------------------------------------------------

#' Kumo Tabular's peak-memory shapes
#'
#' The same three stages as TabICL -- a column stage kept linear in rows
#' by inducing points, a row stage over the `C + K` tokens of a row, an
#' in-context stage over rows -- under the same stage names, so the shared
#' assembly ([.peak_from_terms()]) and the calibration harness read them
#' unchanged. One token per column (no grouping), and the in-context
#' width is `icl_channels`, which on medium is not `K * D`.
#'
#' Where Kumo differs from the shared formula is what row chunking keeps:
#'
#' * Unchunked, one forward holds every row: `n_context` plus one
#'   `predict_chunk_size` of query rows.
#' * With `row_chunk_size` below that, the context is encoded on its own
#'   and the query rows stream against it. The row tensor
#'   `[n_context, C + K, D]` is not chunked -- only each block's work on
#'   it is -- so it is a whole-table `res` term, as is the in-context
#'   stack's `[n_context, E]`. The inducing step still reads every
#'   context row for each column slice (the pre-pass), and the context's
#'   cache -- per ICL layer the keys and values of every context row --
#'   lives until the query rows are through, so one member's cache is
#'   persistent even with `kv_cache = FALSE`.
#' * `col_chunk_size` slices the column axis of every column block,
#'   chunked or not.
#'
#' * Past 10 classes (`opts$n_classes`), every member runs its `T`
#'   error-correcting sub-tasks as one batch, so every term -- the cache
#'   included -- is `T` times as large.
#'
#' Columns are capped at the recipe's 500; categorical count columns,
#' which can add up to one per categorical, are not known here.
#'
#' @inheritParams tabicl_peak_terms
#' @keywords internal
kumo_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  n_est <- max(1, as.numeric(opts$n_estimators %||% 1))
  D  <- as.numeric(config$cell_channels)
  E  <- as.numeric(config$icl_channels %||% (4 * D))
  K  <- as.numeric(config$num_readout_tokens %||% 4)
  L  <- as.numeric(config$num_embedding_layers %||% 4)
  Li <- as.numeric(config$num_icl_layers %||% 12)
  M  <- as.numeric(config$num_inducing_points %||% 128)
  He <- as.numeric(config$num_icl_heads %||% 8)
  kvh <- config$num_icl_key_value_heads_for_query
  kvh <- if (is.null(kvh) || is.list(kvh) || !length(kvh)) He else as.numeric(kvh)

  tok <- max(1, min(n_features, 500))
  hc  <- tok + K
  n_q <- .resident_query(n_query, opts)
  n   <- n_context + n_q
  # Kumo has no checkpoint default for either knob: absent is off.
  rows <- .stage_row_chunk(opts, list(inference_row_chunk_size = Inf))
  cc   <- min(tok, .stage_col_chunk(opts, config))
  chunked <- is.finite(rows) && rows < n
  rc <- if (chunked) rows else n

  # Per member: the inducing summaries' keys and values per column block,
  # the context means, and the context rows' keys and values per ICL
  # layer, cut to the heads the query rows use.
  cache <- L * tok * M * D * 2 + tok + Li * 2 * n_context * E * kvh / He

  # Error-correcting sub-tasks, as `.kumo_ecoc_codebook()` counts them.
  K  <- suppressWarnings(as.numeric(opts$n_classes %||% NA))
  Kn <- as.numeric(config$num_classes %||% 0)
  T_ <- if (Kn > 0 && isTRUE(K > Kn)) {
    max(ceiling(K / (Kn - 1)), 4 * ceiling(log(K) / log(Kn)))
  } else 1
  cache <- T_ * cache

  stages <- list(
    list(name = "column summaries", act = 0, res = 0, att = 0,
         prepass = if (chunked) T_ * n_context * cc * D else 0),
    list(name = "column embedding", act = T_ * rc * cc * D,
         res = if (chunked) T_ * n_context * hc * D else 0, att = 0),
    list(name = "row interaction", act = T_ * rc * hc * D,
         res = if (chunked) T_ * n_context * hc * D else 0, att = 0),
    list(name = "in-context learning", act = T_ * rc * E,
         res = if (chunked) T_ * n_context * E else 0, att = 0, copies = "icl")
  )
  persistent <- if (isTRUE(opts$kv_cache)) n_est * cache
                else if (chunked) cache else 0
  list(stages = stages, persistent = persistent, n_estimators = n_est)
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#' @keywords internal
kumo_build <- function(config, task) {
  cli::cli_alert_info(
    "Building Kumo Tabular {.val {config$size}} ({.val {config$head}}, \\
     D={config$cell_channels}, E={config$icl_channels}, \\
     {config$num_icl_layers} ICL layers)..."
  )
  kumo_model(config)
}

#' @keywords internal
kumo_detect <- function(config) identical(config$arch, "kumo_tabular")

#' @keywords internal
kumo_task_of <- function(config) {
  switch(config$head %||% "",
         classifier = "classification",
         regressor  = "regression",
         NULL)
}

register_kumo_backend <- function() {
  register_backend(
    name          = "kumo",
    build         = kumo_build,
    detect        = kumo_detect,
    task_of       = kumo_task_of,
    classifier    = kumo_classifier,
    regressor     = kumo_regressor,
    describe      = kumo_describe,
    peak_terms    = kumo_peak_terms,
    # The network conditions on missingness itself: context-mean
    # imputation inside the cell embedding plus a learned projection of
    # the missingness pattern.
    kv_cache_capable   = TRUE,
    handles_missing    = TRUE,
    imputes_internally = FALSE,
    description   = "NVIDIA Kumo Tabular (small/medium/large) cell/row/ICL transformer",
    parity        = "structured-data-models 0.2.1 (PyPI)"
  )
}
