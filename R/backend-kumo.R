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
  # @param y `[..., Rt]`.
  # @param categorical_mask `[..., C]` boolean tensor.
  # @return `[..., R, K * D]`.
  forward = function(x, y, categorical_mask) {
    Rt <- y$size(y$dim())
    K <- self$K
    nd <- x$dim()

    cells <- self$cell_embedding(x, categorical_mask, train_size = Rt)
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
    for (i in seq_len(L)) {
      tok <- seq$size(nd)
      cols <- seq$narrow(nd, K + 1L, tok - K)$transpose(-2L, -3L)  # [..., C, R, D]
      cols <- self$col_blocks[[i]](cols, cols$narrow(nd, 1L, Rt))
      dump_if_enabled(sprintf("kumo_col%d", i - 1L), cols)
      seq <- torch::torch_cat(
        list(seq$narrow(nd, 1L, K), cols$transpose(-2L, -3L)), dim = -2L
      )
      q <- if (i == L) seq$narrow(nd, 1L, K) else seq
      seq <- self$row_blocks[[i]](q, seq)
      dump_if_enabled(sprintf("kumo_row%d", i - 1L), seq)
    }
    self$norm(seq)$flatten(start_dim = -2L)
  }
)


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
  # @param y `[..., Rt]`, as for [kumo_row_embedding].
  # @return `[..., R - Rt, n_out]`.
  forward = function(x, y) {
    Rt <- y$size(y$dim())
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
    for (j in seq_len(L)) {
      layer <- self$layers[[j]]
      last <- j == L
      ctx <- x$narrow(ax, 1L, Rt)
      qry <- x$narrow(ax, Rt + 1L, R - Rt)
      if (is.null(self$kv_heads)) {
        # One call over every row, as the reference makes it: the same
        # call shapes are what keep the rounding the same.
        x <- layer(if (last) qry else x, ctx)
        next
      }
      # Context rows attend with every head; query rows with the first
      # `kv_heads` of the same projections.
      kv <- layer$attn$project_kv(layer$key_value_norm(ctx))
      kvq <- lapply(kv, function(t) {
        t$narrow(t$dim() - 1L, 1L, self$kv_heads)$contiguous()
      })
      xq <- layer(qry, kvq)
      x <- if (last) xq else torch::torch_cat(list(layer(ctx, kv), xq), dim = ax)
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
  },

  # @param x `[..., R, C]` float, NaN for missing, context rows first.
  # @param y `[..., Rt]`: 0-based class codes for the classifier,
  #   standardized targets for the regressor. Its length is the number
  #   of context rows.
  # @param categorical_mask Logical vector of length `C`, or a boolean
  #   tensor broadcastable to `[..., C]`.
  # @return `[..., R - Rt, 10]` logits or `[..., R - Rt, 999]` quantiles.
  forward = function(x, y, categorical_mask) {
    if (!inherits(categorical_mask, "torch_tensor")) {
      categorical_mask <- torch::torch_tensor(as.logical(categorical_mask),
                                              dtype = torch::torch_bool(),
                                              device = x$device)
    }
    nd <- x$dim()
    categorical_mask <- categorical_mask$to(dtype = torch::torch_bool())$expand(
      c(x$shape[seq_len(nd - 2L)], x$size(nd))
    )
    y <- if (self$row_embedding$is_classifier) y$to(dtype = torch::torch_long())
         else y$to(dtype = x$dtype)

    reps <- self$row_embedding(x, y, categorical_mask)
    dump_if_enabled("kumo_reps", reps)
    out <- self$icl_block(self$row_project(reps), y)
    dump_if_enabled("kumo_logits", out)
    out
  }
)


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

# One member's network output for the query rows: the context rows and the
# chunk are transformed together (every transform is row-wise once
# fitted), the target is the member's view of it, and the whole thing is
# one forward pass, as `sdm`'s `KumoTabular.forward()` makes it.
# @keywords internal
.kumo_member_out <- function(net, dev, rec, X_train, y_train, X_query, m) {
  mem <- kumo_recipe_member(rec, rbind(X_train, X_query), m)
  yt <- kumo_recipe_target(rec, y_train, m)
  out <- torch::with_no_grad(
    net(mem$x$to(device = dev)$unsqueeze(1L), yt$to(device = dev)$unsqueeze(1L),
        mem$categorical_mask)
  )
  out[1, , ]$cpu()
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
#' probability zero. More than 10 classes needs the reference's
#' error-correcting output codes, which are not ported yet.
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
#' @param categorical_features 1-based indices of categorical columns.
#' @keywords internal
kumo_classifier <- function(ctx, n_estimators = 8L, seed = 42L,
                            predict_chunk_size = 1024L,
                            categorical_features = NULL) {
  net <- ctx$net; dev <- ctx$device
  max_classes <- as.integer(ctx$config$num_classes)

  fit_fn <- function(X, y) {
    if (length(y) != nrow(X)) cli::cli_abort("length(y) must equal nrow(X).")
    if (is.factor(y)) {
      levels_ <- levels(y); y_int <- as.integer(y) - 1L
    } else {
      levels_ <- sort(unique(y)); y_int <- match(y, levels_) - 1L
    }
    if (anyNA(y_int)) cli::cli_abort("{.arg y} has missing values.")
    observed <- length(unique(y_int))
    if (observed > max_classes) {
      cli::cli_abort(c(
        "Kumo Tabular predicts at most {max_classes} classes per pass; \\
         the training rows have {observed}.",
        i = "Above that the reference switches to error-correcting output \\
             codes, which tabfound does not implement yet."
      ))
    }
    categorical <- .kumo_categorical(categorical_features, ncol(X))
    rec <- kumo_recipe_fit(X, y_int, TRUE, categorical = categorical,
                           n_members = n_estimators, seed = seed)
    list(X_train = X, y_train_int = y_int, class_levels = levels_,
         n_train = nrow(X), categorical = categorical, recipe = rec)
  }

  .chunk <- function(state, X_query) {
    rec <- state$recipe
    K <- length(rec$target$classes)
    outs <- lapply(seq_len(rec$n_members), function(m) {
      if (m > 1L) collect_between_chunks()
      o <- .kumo_member_out(net, dev, rec, state$X_train, state$y_train_int,
                            X_query, m)
      kumo_recipe_unshift(rec, o[, 1:K, drop = FALSE], m)
    })
    p <- as.matrix(kumo_recipe_reduce(outs, TRUE))
    full <- matrix(0, nrow(X_query), length(state$class_levels))
    full[, rec$target$classes + 1L] <- p
    full
  }

  predict_fn <- function(state, newdata, type = "class", ...) {
    probs <- chunk_apply(as.matrix(newdata), predict_chunk_size,
                         function(chunk) .chunk(state, chunk))
    colnames(probs) <- as.character(state$class_levels)
    if (type == "prob") return(probs)
    state$class_levels[max.col(probs, ties.method = "first")]
  }

  list(fit = fit_fn, predict = predict_fn)
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
                           predict_chunk_size = 1024L,
                           categorical_features = NULL) {
  net <- ctx$net; dev <- ctx$device
  n_q <- as.integer(ctx$config$num_quantiles)
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

  .grid <- function(state, X_query) {
    rec <- state$recipe
    outs <- lapply(seq_len(rec$n_members), function(m) {
      if (m > 1L) collect_between_chunks()
      o <- .kumo_member_out(net, dev, rec, state$X_train, state$y_train,
                            X_query, m)
      kumo_recipe_inverse(rec, o, m)
    })
    as.matrix(kumo_recipe_reduce(outs, FALSE))
  }

  predict_fn <- function(state, newdata, type = "mean",
                         quantiles = c(0.1, 0.5, 0.9), ...) {
    if (identical(type, "grid")) type <- "raw_quantiles"
    res <- chunk_apply(as.matrix(newdata), predict_chunk_size, function(chunk) {
      g <- .grid(state, chunk)
      quantile_dist_stat(g, levels_, type, quantiles)
    })
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

#' @keywords internal
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
      prefix = paste0("row_embedding.", yname)),
    arch_stage(
      "readout", "Readout tokens", kind = "embed", group = "Row embedding",
      detail = sprintf("%d learned tokens prepended to every row", K),
      prefix = "row_embedding.readout_token"),
    arch_stage(
      "col", "Column block", kind = "attention", group = "Row embedding",
      repeats = L, axis = "rows",
      detail = sprintf("induced attention over a column's rows: %d inducing points read the context rows (log-n query scaling), every row reads them back; %d heads",
                       M, H),
      shape = sprintf("(C, R, %d)", D), prefix = "row_embedding.col_blocks",
      children = list(
        arch_stage("ind", "Inducing points attend to the context rows",
                   kind = "attention", axis = "rows",
                   prefix = "row_embedding.col_blocks.0.inducing_block"),
        arch_stage("back", "Rows attend to the inducing summary",
                   kind = "attention", axis = "inducing",
                   prefix = "row_embedding.col_blocks.0.output_block"),
        arch_stage("iv", "Learned inducing points", kind = "embed",
                   prefix = "row_embedding.col_blocks.0.inducing_points")
      )),
    arch_stage(
      "row", "Row block", kind = "attention", group = "Row embedding",
      repeats = L, axis = "columns",
      detail = sprintf("self-attention over a row's %d readout + C cell tokens; RoPE, gated log-n query scaling; the last block queries from the readout tokens only; %d heads",
                       K, H),
      shape = sprintf("(R, %d + C, %d)", K, D),
      prefix = c("row_embedding.row_blocks", "row_embedding.norm"),
      children = list(
        arch_stage("rattn", "Attention over a row's tokens", kind = "attention",
                   axis = "columns", prefix = "row_embedding.row_blocks.0.attn"),
        arch_stage("rmlp", "MLP, GELU", kind = "ffn",
                   prefix = "row_embedding.row_blocks.0.mlp"),
        arch_stage("rnorm", "RMSNorm", kind = "norm",
                   prefix = c("row_embedding.row_blocks.0.query_norm",
                              "row_embedding.row_blocks.0.key_value_norm",
                              "row_embedding.norm"))
      )),
    if (proj) arch_stage(
      "proj", "Row projection", kind = "embed", group = "In-context",
      detail = sprintf("Linear(%d -> %d) from the flattened readout tokens", K * D, E),
      prefix = "row_project"),
    arch_stage(
      "icl_y", "Target embedding (rows)", kind = "embed", group = "In-context",
      detail = "added to the context rows",
      prefix = paste0("icl_block.", yname)),
    arch_stage(
      "icl", "In-context learning block", kind = "attention", group = "In-context",
      repeats = Li, axis = "rows",
      detail = sprintf("every row attends to the context rows, log-n query scaling; %d heads of %d%s",
                       He, E %/% He,
                       if (is.null(kvh)) "" else
                         sprintf("; query rows use %d of the key/value heads", kvh)),
      shape = sprintf("(R, %d)", E), prefix = c("icl_block.layers", "icl_block.norm"),
      children = list(
        arch_stage("iattn", "Attention over context rows", kind = "attention",
                   axis = "rows", prefix = "icl_block.layers.0.attn"),
        arch_stage("imlp", "MLP, GELU", kind = "ffn",
                   prefix = "icl_block.layers.0.mlp"),
        arch_stage("inorm", "RMSNorm", kind = "norm",
                   prefix = c("icl_block.layers.0.query_norm",
                              "icl_block.layers.0.key_value_norm",
                              "icl_block.norm"))
      )),
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
      "Embedding layers" = L,
      "ICL layers"       = Li,
      "Cell width"       = D,
      "ICL width"        = if (proj) sprintf("%d (projected from %d x %d)", E, K, D)
                           else sprintf("%d (%d readout x %d)", E, K, D),
      "Heads"            = sprintf("%d embedding, %d ICL", H, He),
      "Query KV heads"   = if (is.null(kvh)) "all" else kvh,
      "Inducing points"  = M,
      "Cell group"       = sprintf("%d slots x %d frequencies", G, Fq),
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
#' The same three-stage shape as TabICL -- a column stage kept linear in
#' rows by inducing points, a row stage over a short sequence, an
#' in-context stage quadratic in rows -- so it goes through the shared
#' formula with one token per column (no grouping into fewer tokens) and
#' the readout tokens as the row stage's extra slots.
#'
#' The constants in `inst/memory/coefs/kumo.json` are TabICL's, borrowed
#' until `inst/memory/calibrate.R kumo` has been run; the preflight says
#' so when it prints.
#'
#' @inheritParams tabicl_peak_terms
#' @keywords internal
kumo_peak_terms <- function(n_context, n_query, n_features, opts, config) {
  n_est <- max(1, as.numeric(opts$n_estimators %||% 1))
  .icl_family_terms(
    n_context    = n_context,
    n_query      = .resident_query(n_query, opts),
    n_features   = n_features,
    embed_dim    = as.numeric(config$cell_channels),
    group_size   = 1,
    n_cls        = as.numeric(config$num_readout_tokens %||% 4),
    col_heads    = as.numeric(config$num_embedding_heads %||% 4),
    row_heads    = as.numeric(config$num_embedding_heads %||% 4),
    icl_heads    = as.numeric(config$num_icl_heads %||% 8),
    col_inducing = as.numeric(config$num_inducing_points %||% 128),
    icl_blocks   = as.numeric(config$num_icl_layers %||% 12),
    kv_cache     = isTRUE(opts$kv_cache),
    n_estimators = n_est
  )
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
    handles_missing    = TRUE,
    imputes_internally = FALSE,
    description   = "NVIDIA Kumo Tabular (small/medium/large) cell/row/ICL transformer",
    parity        = "structured-data-models 0.2.1 (PyPI)"
  )
}
