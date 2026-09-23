# Parity comparison utilities.
#
# Grades an R run against a reference dump produced by one of the
# `*_reference.py` scripts in this directory. The comparison is staged so
# a failure localises itself:
#
#   preprocess  -- did R build the same per-member model input?
#   forward     -- given identical input, did the network produce the
#                  same logits?
#   predict     -- did the ensemble combination + post-processing agree?
#
# A mismatch at `preprocess` makes the later stages meaningless, so the
# report shows all three rather than short-circuiting.

#' Drop length-1 dimensions
#'
#' The reference dumps carry the batch axis in whatever position the
#' Python module used -- `(n_test, 1, n_out)` for the decoder output,
#' where the R module produces `(1, n_test, n_out)`. That is a layout
#' difference, not a numeric one, so both sides are squeezed before
#' comparison. Any *real* shape disagreement still trips the check,
#' because a squeezed shape mismatch is still a mismatch.
#' @keywords internal
drop_singleton <- function(a) {
  d <- dim(a)
  if (is.null(d)) return(as.array(a))
  keep <- d[d != 1L]
  if (!length(keep)) keep <- 1L
  array(as.numeric(a), dim = keep)
}

#' Compare two tensors elementwise
#'
#' @return A list with `n`, `max_abs`, `mean_abs`, `max_rel`, and `shape_ok`.
#' @keywords internal
tensor_diff <- function(a, b) {
  a <- drop_singleton(as.array(a)); b <- drop_singleton(as.array(b))
  if (!identical(dim(a), dim(b))) {
    return(list(shape_ok = FALSE, n = NA_integer_,
                max_abs = NA_real_, mean_abs = NA_real_, max_rel = NA_real_,
                shape_a = paste(dim(a), collapse = "x"),
                shape_b = paste(dim(b), collapse = "x")))
  }
  av <- as.numeric(a); bv <- as.numeric(b)

  # Missing values are part of what the model sees -- a NaN cell must
  # stay a NaN cell -- so agreeing NaNs count as an exact match, while a
  # NaN on one side only is the worst possible mismatch.
  both_na <- is.na(av) & is.na(bv)
  one_na  <- xor(is.na(av), is.na(bv))

  d <- abs(av - bv)
  d[both_na] <- 0
  d[one_na]  <- Inf
  denom <- pmax(abs(bv), 1e-12)
  rel <- d / denom
  rel[both_na] <- 0

  # Elementwise relative error is useless for a tensor that crosses
  # zero, and absolute error is useless for one whose entries span three
  # orders of magnitude -- TabFM's decoder emits logits down to -800 for
  # the unused class slots. `max_scaled` normalises the largest absolute
  # error by the reference tensor's own scale, which is the measure that
  # stays meaningful for both.
  scale <- max(abs(bv[is.finite(bv)]), na.rm = TRUE)
  if (!is.finite(scale) || scale == 0) scale <- 1

  list(shape_ok = TRUE, n = length(d),
       max_abs = max(d), mean_abs = mean(d[is.finite(d)]),
       max_rel = max(rel), max_scaled = max(d) / scale,
       n_na_mismatch = sum(one_na))
}

#' Load a single-tensor safetensors file written by either harness
#' @keywords internal
read_t <- function(path) {
  if (!file.exists(path)) return(NULL)
  as.array(safetensors::safe_load_file(path, framework = "torch")$t)
}

#' Compare every `member_NN/<name>.safetensors` present in both trees
#' @keywords internal
compare_member_trees <- function(py_dir, r_dir, names) {
  members <- sort(list.files(py_dir, pattern = "^member_[0-9]+$"))
  rows <- list()
  for (m in members) {
    for (nm in names) {
      p <- read_t(file.path(py_dir, m, paste0(nm, ".safetensors")))
      r <- read_t(file.path(r_dir,  m, paste0(nm, ".safetensors")))
      if (is.null(p) || is.null(r)) next
      d <- tensor_diff(r, p)
      # How many *rows* disagree, not just how many cells. A flipped
      # fingerprint corrupts exactly one row; genuine numeric drift
      # touches all of them. The two look identical in `max_abs`.
      bad_rows <- NA_integer_
      if (isTRUE(d$shape_ok) && length(dim(p)) == 2L) {
        cell <- abs(drop_singleton(r) - drop_singleton(p))
        cell[is.na(cell)] <- 0
        bad_rows <- sum(apply(cell, 1L, max) > 1e-6)
      }
      rows[[length(rows) + 1L]] <- data.frame(
        member = m, tensor = nm,
        shape_ok = d$shape_ok, n = d$n,
        max_abs = d$max_abs, max_rel = d$max_rel,
        max_scaled = d$max_scaled %||% NA_real_,
        bad_rows = bad_rows, n_rows = if (is.null(dim(p))) NA_integer_ else dim(p)[1],
        stringsAsFactors = FALSE
      )
    }
  }
  if (!length(rows)) return(NULL)
  do.call(rbind, rows)
}


#' Run the TabPFN backend against a stored Python reference
#'
#' @param reference_dir Directory written by `tabpfn_reference.py`.
#' @param fixture_dir Directory of parity fixtures.
#' @param fixture Fixture name.
#' @param model_dir Converted TabPFN artifacts (`model.safetensors` +
#'   `config.json`) for the matching head.
#' @param trace_dir Where the R run writes its per-member trace. A
#'   tempdir by default.
#' @return A list with `summary` (data.frame of stage-level diffs) and
#'   `members` (per-member, per-tensor diffs).
#' @keywords internal
parity_tabpfn <- function(reference_dir, fixture_dir, fixture, label = fixture,
                          model_dir, trace_dir = tempfile("tabfound-trace-")) {
  stopifnot(dir.exists(reference_dir))
  meta <- jsonlite::fromJSON(file.path(reference_dir, "reference.json"))
  fx   <- read_parity_fixture(fixture_dir, fixture)
  dir.create(trace_dir, recursive = TRUE, showWarnings = FALSE)

  # Older reference dumps predate the flag; those runs used 1.0 for the
  # classifier and the 0.9 default for the regressor.
  temp <- meta$softmax_temperature %||%
    (if (identical(meta$task, "classification")) 1 else 0.9)

  rows <- list()
  add <- function(stage, d, note = "") {
    rows[[length(rows) + 1L]] <<- data.frame(
      fixture = label, stage = stage,
      n = d$n, max_abs = d$max_abs, max_rel = d$max_rel,
      max_scaled = d$max_scaled %||% NA_real_,
      note = note, stringsAsFactors = FALSE
    )
  }

  # What the reference was *told* is categorical, so both sides start from
  # the same declaration by construction rather than by two harnesses
  # agreeing to read the same manifest field. A dump with no such record
  # predates the field; fall back to the fixture's own declaration.
  declared <- if ("declared_categorical" %in% names(meta)) {
    as.integer(unlist(meta$declared_categorical)) + 1L
  } else {
    fx$categorical_features
  }

  # Which columns each side then decided were categorical. Everything
  # downstream is conditional on this, so a disagreement here would show
  # up as a diffuse preprocessing mismatch rather than as itself.
  if (!is.null(meta$inferred_categorical)) {
    r_cat <- detect_categorical_features(fx$x_train, declared)
    py_cat <- as.integer(unlist(meta$inferred_categorical)) + 1L
    add("schema:categorical",
        tensor_diff(matrix(as.numeric(r_cat), ncol = 1L),
                    matrix(as.numeric(py_cat), ncol = 1L)))
  }

  if (identical(meta$task, "classification")) {
    mod <- tabular_classifier(model_dir, device = "cpu",
                              categorical_features = declared,
                              ensemble_configs_dir = reference_dir,
                              softmax_temperature = temp,
                              trace_dir = trace_dir)
    mod <- fit(mod, fx$x_train, as.integer(fx$y_train))
    r_probs  <- predict(mod, fx$x_test, type = "prob")
    py_probs <- read_t(file.path(reference_dir, "final_probs.safetensors"))
    add("predict_proba", tensor_diff(r_probs, py_probs))
  } else {
    mod <- tabular_regressor(model_dir, device = "cpu",
                             categorical_features = declared,
                             ensemble_configs_dir = reference_dir,
                             softmax_temperature = temp,
                             trace_dir = trace_dir)
    mod <- fit(mod, fx$x_train, fx$y_train)
    r_mean  <- predict(mod, fx$x_test)
    py_mean <- read_t(file.path(reference_dir, "final_preds.safetensors"))
    add("predict_mean", tensor_diff(matrix(r_mean, ncol = 1L),
                                    matrix(as.numeric(py_mean), ncol = 1L)))
    py_q <- read_t(file.path(reference_dir, "final_quantiles.safetensors"))
    if (!is.null(py_q) && !is.null(meta$quantiles)) {
      r_q <- predict(mod, fx$x_test, type = "quantiles",
                     quantiles = as.numeric(meta$quantiles))
      add("predict_quantiles", tensor_diff(r_q, py_q))
    }
  }

  # The KV cache, against this backend's own plain forward. There is no
  # reference number to compare against here -- the estimator does not
  # expose a bare cached forward -- but there is a meaningful bound: the
  # uncached path already moves by *something* when the same rows are run
  # in a different-sized batch, because float32 attention reduces in a
  # different order. The cache changes the batch shape too, so its shift
  # is graded against that, not against zero.
  # Only for architectures whose cache is exactly equivalent. v2.6's is
  # not -- it freezes two train+test-fitted masks -- and is graded in
  # `parity_tabpfn26()` against the reference's own cached run instead.
  if (isTRUE(mod$model$kv_cache_is_exact)) {
    net <- mod$model
    col <- load_column_embeddings()
    x_tr <- as_float_tensor(fx$x_train)$unsqueeze(1L)
    x_te <- as_float_tensor(fx$x_test)$unsqueeze(1L)
    y_tr <- as_float_tensor(matrix(as.numeric(fx$y_train), ncol = 1L))$
      squeeze(-1L)$unsqueeze(1L)
    n_te <- x_te$size(2)

    plain <- torch::with_no_grad(
      net(x_tr, y_tr, x_te, column_embeddings = col))$logits
    cache <- torch::with_no_grad(tabpfn_forward(
      net, x_tr, y_tr, torch::torch_zeros(c(1L, 0L, x_tr$size(3))), col,
      return_kv_cache = TRUE))$kv_cache
    cached <- torch::with_no_grad(tabpfn_forward(
      net, NULL, NULL, x_te, col, kv_cache = cache))$logits

    # The same rows, uncached, in a half-sized batch: the floor below
    # which "the cache changed the answer" is not a meaningful claim.
    half <- max(1L, n_te %/% 2L)
    batch_shift <- max(abs(as.array(
      plain[, 1:half, ] -
        torch::with_no_grad(net(x_tr, y_tr, x_te[, 1:half, , drop = FALSE],
                                column_embeddings = col))$logits)))
    shift <- max(abs(as.array(cached) - as.array(plain)))
    # Graded against the batching shift rather than a chosen constant: the
    # row passes either because the cache is exact, or because it moved
    # the answer no more than running the same rows in a different-sized
    # batch already does. Both are statements about float32 reduction
    # order, and neither is a statement about the cache being wrong.
    ratio <- if (batch_shift > 0) shift / batch_shift
             else if (shift == 0) 0 else Inf
    add("cache:vs-batching",
        list(n = length(as.array(cached)), max_abs = shift, max_rel = ratio,
             max_scaled = shift / max(abs(as.array(plain)))),
        note = sprintf("re-batching alone moves them %.3e", batch_shift))
  }

  members <- compare_member_trees(
    reference_dir, trace_dir,
    names = c("X_train", "X_test", "y_train", "logits")
  )
  if (!is.null(members)) {
    for (nm in unique(members$tensor)) {
      sub <- members[members$tensor == nm, , drop = FALSE]
      stage <- switch(nm,
                      X_train = "preprocess:X_train",
                      X_test  = "preprocess:X_test",
                      y_train = "preprocess:y_train",
                      logits  = "forward:logits")
      note <- if (!all(sub$shape_ok)) "SHAPE MISMATCH"
              else if (any(sub$bad_rows > 0, na.rm = TRUE))
                sprintf("%d/%d rows differ",
                        sum(sub$bad_rows, na.rm = TRUE),
                        sum(sub$n_rows, na.rm = TRUE))
              else ""
      # `max_scaled` too: `forward:logits` declares a scale-relative
      # tolerance precisely because unnormalised logits span orders of
      # magnitude, and without it that stage can only ever be graded on
      # absolute error.
      add(stage,
          list(n = sum(sub$n), max_abs = max(sub$max_abs),
               max_rel = max(sub$max_rel), max_scaled = max(sub$max_scaled)),
          note = note)
    }
  }

  list(summary = do.call(rbind, rows), members = members, meta = meta,
       trace_dir = trace_dir)
}


#' Run the TabFM backend against a stored Python reference
#'
#' TabFM's parity is staged by *pipeline stage* rather than by ensemble
#' member: the network is one forward pass, so the useful bisection is
#' cell embedder -> column stage -> row stage -> ... -> decoder. Both
#' sides dump the same six tensors.
#'
#' @param reference_dir Directory written by `tabfm_reference.py`.
#' @param fixture_dir,fixture Parity fixture location and name.
#' @param model_root TabFM Hub snapshot root. The backend picks the
#'   `classification/` or `regression/` subfolder itself, so passing the
#'   root also exercises that resolution.
#' @keywords internal

# ---------------------------------------------------------------------------
# Bare-network runs
# ---------------------------------------------------------------------------
#
# The TabFM / TabICL / Mitra reference dumps are of the *network*:
# `<backend>_reference.py` feeds the fixture's raw columns straight in.
# Their R predictors no longer do that -- they preprocess and ensemble,
# because their sklearn wrappers do -- so a stage-by-stage comparison has
# to bypass them and call the network directly, or it would be comparing
# the last ensemble member's activations against an un-preprocessed pass
# and calling the difference a regression.
#
# The wrapper layer those predictors implement is graded separately, and
# without any checkpoint, by `ensemble_reference.py`.

# @keywords internal
.bare_net <- function(model_dir, backend, task, subfolder = NULL) {
  ctx <- load_backend_model(model_dir, task = task, backend = backend,
                            device = "cpu", subfolder = subfolder)
  ctx$net$eval()
  ctx
}

# @keywords internal
.bare_tabicl <- function(ctx, fx, meta) {
  net <- ctx$net
  x <- rbind(as.matrix(fx$x_train), as.matrix(fx$x_test))
  storage.mode(x) <- "double"
  n_train <- nrow(fx$x_train)
  y <- as.numeric(fx$y_train)
  scaler <- NULL
  if (!identical(meta$task, "classification")) {
    # The reference standardizes y before the forward pass and inverts
    # on the way out; mirror it so the compared surface is the same.
    scaler <- fit_target_scaler(y)
    y <- apply_target_scaler(y, scaler)
  }
  out <- torch::with_no_grad({
    net(as_float_tensor(x, device = ctx$device)$unsqueeze(1L),
        as_float_tensor(matrix(y, nrow = 1L), device = ctx$device))
  })
  test <- out[1, (n_train + 1L):out$size(2), ]
  if (identical(meta$task, "classification")) {
    n_cls <- length(unique(as.integer(fx$y_train)))
    logits <- test[, 1:n_cls] / (meta$softmax_temperature %||% 0.9)
    as.matrix(torch::nnf_softmax(logits, dim = -1L)$cpu())
  } else {
    invert_target_scaler(as.matrix(test$cpu()), scaler)
  }
}

# @keywords internal
.bare_tabfm <- function(ctx, fx, meta) {
  net <- ctx$net
  x <- rbind(as.matrix(fx$x_train), as.matrix(fx$x_test))
  storage.mode(x) <- "double"
  n_train <- nrow(fx$x_train)
  y_train <- as.numeric(fx$y_train)
  scaler <- NULL
  if (!identical(meta$task, "classification")) {
    scaler <- fit_target_scaler(y_train)
    y_train <- apply_target_scaler(y_train, scaler)
  }
  y <- c(y_train, rep(-100, nrow(fx$x_test)))
  out <- torch::with_no_grad({
    net(as_float_tensor(x, device = ctx$device)$unsqueeze(1L),
        as_float_tensor(matrix(y, nrow = 1L), device = ctx$device),
        torch::torch_tensor(as.integer(n_train), dtype = torch::torch_long(),
                            device = ctx$device))
  })
  test <- out[1, (n_train + 1L):out$size(2), ]
  if (identical(meta$task, "classification")) {
    n_cls <- length(unique(as.integer(fx$y_train)))
    as.matrix(torch::nnf_softmax(test[, 1:n_cls], dim = -1L)$cpu())
  } else {
    matrix(invert_target_scaler(as.numeric(test[, 1]$cpu()), scaler), ncol = 1L)
  }
}

# @keywords internal
.bare_mitra <- function(ctx, fx, meta) {
  net <- ctx$net
  xs <- as.matrix(fx$x_train); storage.mode(xs) <- "double"
  xq <- as.matrix(fx$x_test);  storage.mode(xq) <- "double"
  y <- as.numeric(fx$y_train)
  scaler <- NULL
  if (!identical(meta$task, "classification")) {
    # Mitra maps the target to [0, 1] by min-max, not by standardizing.
    scaler <- fit_minmax_scaler(y)
    y <- apply_minmax_scaler(y, scaler)
  }
  out <- torch::with_no_grad({
    net(as_float_tensor(xs, device = ctx$device)$unsqueeze(1L),
        as_float_tensor(matrix(y, nrow = 1L), device = ctx$device),
        as_float_tensor(xq, device = ctx$device)$unsqueeze(1L))
  })
  if (identical(meta$task, "classification")) {
    n_cls <- length(unique(as.integer(fx$y_train)))
    as.matrix(torch::nnf_softmax(out[1, , 1:n_cls], dim = -1L)$cpu())
  } else {
    matrix(invert_minmax_scaler(as.numeric(out[1, , 1]$cpu()), scaler), ncol = 1L)
  }
}

parity_tabfm <- function(reference_dir, fixture_dir, fixture, label = fixture,
                         model_root,
                         trace_dir = tempfile("tabfound-tabfm-")) {
  stopifnot(dir.exists(reference_dir))
  meta <- jsonlite::fromJSON(file.path(reference_dir, "reference.json"))
  fx   <- read_parity_fixture(fixture_dir, fixture)
  dir.create(trace_dir, recursive = TRUE, showWarnings = FALSE)

  rows <- list()
  add <- function(stage, d, note = "") {
    rows[[length(rows) + 1L]] <<- data.frame(
      fixture = label, stage = stage, n = d$n,
      max_abs = d$max_abs, max_rel = d$max_rel,
      max_scaled = d$max_scaled %||% NA_real_,
      note = note, stringsAsFactors = FALSE
    )
  }

  # The stage dumps come out of the model itself via the env-var hook,
  # so the R run has to be wrapped rather than passed a directory.
  old <- Sys.getenv("TABFOUND_DUMP_DIR", unset = NA)
  Sys.setenv(TABFOUND_DUMP_DIR = trace_dir)
  on.exit({
    if (is.na(old)) Sys.unsetenv("TABFOUND_DUMP_DIR")
    else Sys.setenv(TABFOUND_DUMP_DIR = old)
  }, add = TRUE)

  ctx <- .bare_net(model_root, "tabfm", meta$task,
                   subfolder = tabfm_subfolder_for(meta$task))
  r_out <- .bare_tabfm(ctx, fx, meta)
  if (identical(meta$task, "classification")) {
    py <- read_t(file.path(reference_dir, "final_probs.safetensors"))
    add("net:probs", tensor_diff(r_out, py))
  } else {
    py <- read_t(file.path(reference_dir, "final_preds.safetensors"))
    add("net:mean", tensor_diff(r_out, matrix(as.numeric(py), ncol = 1L)))
  }

  for (nm in c("tabfm_cell", "tabfm_col1", "tabfm_row1",
               "tabfm_col2", "tabfm_reps", "tabfm_logits")) {
    r <- read_t(file.path(trace_dir, paste0(nm, ".safetensors")))
    p <- read_t(file.path(reference_dir, paste0(nm, ".safetensors")))
    if (is.null(r) || is.null(p)) next
    d <- tensor_diff(r, p)
    add(paste0("stage:", sub("^tabfm_", "", nm)), d,
        note = if (isTRUE(d$shape_ok)) "" else "SHAPE MISMATCH")
  }

  list(summary = do.call(rbind, rows), meta = meta, trace_dir = trace_dir)
}


#' Run the TabICL backend against a stored Python reference
#'
#' Staged by pipeline stage, like TabFM: column embedder -> row
#' interactor -> ICL decoder, plus the final prediction.
#'
#' @param reference_dir Directory written by `tabicl_reference.py`.
#' @param fixture_dir,fixture Parity fixture location and name.
#' @param model_dir Converted TabICL artifacts for the matching task.
#' @keywords internal
parity_tabicl <- function(reference_dir, fixture_dir, fixture, label = fixture,
                          model_dir,
                          trace_dir = tempfile("tabfound-tabicl-")) {
  stopifnot(dir.exists(reference_dir))
  meta <- jsonlite::fromJSON(file.path(reference_dir, "reference.json"))
  fx   <- read_parity_fixture(fixture_dir, fixture)
  dir.create(trace_dir, recursive = TRUE, showWarnings = FALSE)

  rows <- list()
  add <- function(stage, d, note = "") {
    rows[[length(rows) + 1L]] <<- data.frame(
      fixture = label, stage = stage, n = d$n,
      max_abs = d$max_abs, max_rel = d$max_rel,
      max_scaled = d$max_scaled %||% NA_real_,
      note = note, stringsAsFactors = FALSE
    )
  }

  old <- Sys.getenv("TABFOUND_DUMP_DIR", unset = NA)
  Sys.setenv(TABFOUND_DUMP_DIR = trace_dir)
  on.exit({
    if (is.na(old)) Sys.unsetenv("TABFOUND_DUMP_DIR")
    else Sys.setenv(TABFOUND_DUMP_DIR = old)
  }, add = TRUE)

  ctx <- .bare_net(model_dir, "tabicl", meta$task)
  r_out <- .bare_tabicl(ctx, fx, meta)
  if (identical(meta$task, "classification")) {
    py <- read_t(file.path(reference_dir, "final_probs.safetensors"))
    add("net:probs", tensor_diff(r_out, py))
  } else {
    py <- read_t(file.path(reference_dir, "final_quantiles.safetensors"))
    add("net:quantiles", tensor_diff(r_out, py))
  }

  for (nm in c("tabicl_col", "tabicl_reps", "tabicl_logits")) {
    r <- read_t(file.path(trace_dir, paste0(nm, ".safetensors")))
    p <- read_t(file.path(reference_dir, paste0(nm, ".safetensors")))
    if (is.null(r) || is.null(p)) next
    d <- tensor_diff(r, p)
    add(paste0("stage:", sub("^tabicl_", "", nm)), d,
        note = if (isTRUE(d$shape_ok)) "" else "SHAPE MISMATCH")
  }

  list(summary = do.call(rbind, rows), meta = meta, trace_dir = trace_dir)
}


#' Run the Mitra backend against a stored Python reference
#'
#' @param reference_dir Directory written by `mitra_reference.py`.
#' @param fixture_dir,fixture Parity fixture location and name.
#' @param model_dir Mitra artifacts (`model.safetensors` + `config.json`).
#' @keywords internal
parity_mitra <- function(reference_dir, fixture_dir, fixture, label = fixture,
                         model_dir,
                         trace_dir = tempfile("tabfound-mitra-")) {
  stopifnot(dir.exists(reference_dir))
  meta <- jsonlite::fromJSON(file.path(reference_dir, "reference.json"))
  fx   <- read_parity_fixture(fixture_dir, fixture)
  dir.create(trace_dir, recursive = TRUE, showWarnings = FALSE)

  rows <- list()
  add <- function(stage, d, note = "") {
    rows[[length(rows) + 1L]] <<- data.frame(
      fixture = label, stage = stage, n = d$n,
      max_abs = d$max_abs, max_rel = d$max_rel,
      max_scaled = d$max_scaled %||% NA_real_,
      note = note, stringsAsFactors = FALSE
    )
  }

  old <- Sys.getenv("TABFOUND_DUMP_DIR", unset = NA)
  Sys.setenv(TABFOUND_DUMP_DIR = trace_dir)
  on.exit({
    if (is.na(old)) Sys.unsetenv("TABFOUND_DUMP_DIR")
    else Sys.setenv(TABFOUND_DUMP_DIR = old)
  }, add = TRUE)

  ctx <- .bare_net(model_dir, "mitra", meta$task)
  r_out <- .bare_mitra(ctx, fx, meta)
  if (identical(meta$task, "classification")) {
    py <- read_t(file.path(reference_dir, "final_probs.safetensors"))
    add("net:probs", tensor_diff(r_out, py))
  } else {
    py <- read_t(file.path(reference_dir, "final_preds.safetensors"))
    add("net:mean", tensor_diff(r_out, matrix(as.numeric(py), ncol = 1L)))
  }

  for (nm in c("mitra_quantile", "mitra_embedded", "mitra_encoded",
               "mitra_logits")) {
    r <- read_t(file.path(trace_dir, paste0(nm, ".safetensors")))
    p <- read_t(file.path(reference_dir, paste0(nm, ".safetensors")))
    if (is.null(r) || is.null(p)) next
    d <- tensor_diff(r, p)
    add(paste0("stage:", sub("^mitra_", "", nm)), d,
        note = if (isTRUE(d$shape_ok)) "" else "SHAPE MISMATCH")
  }

  list(summary = do.call(rbind, rows), meta = meta, trace_dir = trace_dir)
}


#' Run the TabPFN v2.6 backend against a stored Python reference
#'
#' Graded in two independent steps, because v2.6 moved the preprocessing
#' inside the architecture and there is no longer a useful boundary
#' between "the pipeline" and "the network":
#'
#' * `forward:logits` -- the R network's raw output against the
#'   reference network's, both fed identical bytes.
#' * `decode:*` -- the R softmax / bar-distribution decoding applied to
#'   the *reference's own* logits. A failure here is a decoding bug and
#'   nothing else, since the input is shared.
#'
#' The classifier adds an end-to-end `predict_proba:single` stage, which
#' the regressor has no counterpart to: `tabular_regressor()` standardises
#' the target before the forward pass and the reference is fed the raw
#' target, so the two runs are not comparable at the logits level. Its
#' decoding is covered by `decode:mean` / `decode:quantiles` instead.
#'
#' @param reference_dir Directory written by `tabpfn26_reference.py`.
#' @param fixture_dir,fixture Parity fixture location and name.
#' @param model_dir Converted v2.6 artifacts for the matching head.
#' @keywords internal
parity_tabpfn26 <- function(reference_dir, fixture_dir, fixture,
                            label = fixture, model_dir) {
  stopifnot(dir.exists(reference_dir))
  meta <- jsonlite::fromJSON(file.path(reference_dir, "reference.json"))
  fx   <- read_parity_fixture(fixture_dir, fixture)
  py   <- safetensors::safe_load_file(
    file.path(reference_dir, "forward.safetensors"), framework = "torch")

  rows <- list()
  add <- function(stage, d, note = "") {
    rows[[length(rows) + 1L]] <<- data.frame(
      fixture = label, stage = stage, n = d$n,
      max_abs = d$max_abs, max_rel = d$max_rel,
      max_scaled = d$max_scaled %||% NA_real_,
      note = note, stringsAsFactors = FALSE
    )
  }

  task <- meta$task
  ctx <- load_backend_model(model_dir, task = task, backend = "tabpfn26",
                            device = "cpu")
  net <- ctx$net
  col <- load_column_embeddings()

  # Same bytes into both networks: the reference dump carries the exact
  # tensors it was given, so nothing can drift through the fixture reader.
  out <- torch::with_no_grad(net(
    py$x_train$unsqueeze(1L), py$y_train$unsqueeze(1L), py$x_test$unsqueeze(1L),
    column_embeddings = col
  ))
  add("forward:logits", tensor_diff(as.array(out$logits[1, , ]),
                                    as.array(py$logits)))

  # The two inference-cost paths, graded twice over.
  #
  # `forward:*` compares R's mechanism against the reference's run of the
  # same mechanism, and inherits the plain forward's cross-implementation
  # float32 gap -- so it carries the same tolerance as `forward:logits`.
  #
  # `selfcheck:*` compares R's mechanism against R's *own* plain forward.
  # That one has no cross-implementation gap to hide in: chunking splits
  # work that was already independent, and the cache holds everything the
  # training rows contribute, so a bit of difference is a bug. It is the
  # sharper of the two and the reason both are here.
  r_plain <- as.array(out$logits[1, , ])
  if (!is.null(py$logits_chunked)) {
    chunked <- as.array(torch::with_no_grad(net(
      py$x_train$unsqueeze(1L), py$y_train$unsqueeze(1L), py$x_test$unsqueeze(1L),
      column_embeddings = col,
      save_peak_memory_factor = as.integer(meta$save_peak_memory_factor %||% 8L)
    ))$logits[1, , ])
    add("forward:chunked", tensor_diff(chunked, as.array(py$logits_chunked)))
    add("selfcheck:chunked", tensor_diff(chunked, r_plain))
  }
  if (!is.null(py$logits_cached)) {
    no_rows <- torch::torch_zeros(c(1L, 0L, py$x_train$size(2)))
    cache <- torch::with_no_grad(net(
      py$x_train$unsqueeze(1L), py$y_train$unsqueeze(1L), no_rows,
      column_embeddings = col, return_kv_cache = TRUE
    ))$kv_cache
    cached <- as.array(torch::with_no_grad(net(
      NULL, NULL, py$x_test$unsqueeze(1L),
      column_embeddings = col, kv_cache = cache
    ))$logits[1, , ])
    add("forward:cached", tensor_diff(cached, as.array(py$logits_cached)))

    # Whether the cache changes the answer at all is a property of the
    # data, not of the port: it freezes the constant-column and
    # informative-feature masks on the training rows, and on data where
    # those already agree with the train+test ones, nothing moves. So the
    # self-check is only an assertion where the *reference's* own cache is
    # exact. Where it is not, the shift is reported next to the
    # reference's own -- the real assertion there is `forward:cached`,
    # which says the two caches agree with each other.
    ref_shift <- max(abs(as.array(py$logits_cached) - as.array(py$logits)))
    if (ref_shift == 0) {
      add("selfcheck:cached", tensor_diff(cached, r_plain))
    } else {
      shift <- max(abs(cached - r_plain))
      add("cache:shifts-prediction",
          list(n = length(cached), max_abs = shift, max_rel = NA_real_,
               max_scaled = shift / max(abs(r_plain))),
          note = sprintf("expected: reference shifts by %.3e", ref_shift))
    }
  }

  temp <- meta$softmax_temperature %||% 0.9
  tempered <- py$logits / temp

  if (identical(task, "classification")) {
    n_classes <- as.integer(meta$n_classes)
    r_probs <- torch::nnf_softmax(tempered[, 1:n_classes], dim = -1L)
    add("decode:probs", tensor_diff(as.array(r_probs), as.array(py$probs)))

    mod <- fit(tabular_classifier(model_dir, backend = "tabpfn26",
                                  device = "cpu", softmax_temperature = temp),
               fx$x_train, as.integer(fx$y_train))
    add("predict_proba:single",
        tensor_diff(predict(mod, fx$x_test, type = "prob"),
                    as.array(py$probs)))
  } else {
    borders <- net$criterion$borders
    add("decode:mean",
        tensor_diff(matrix(as.numeric(as.array(
                      bar_logits_to_mean(tempered, borders))), ncol = 1L),
                    matrix(as.numeric(as.array(py$bar_mean)), ncol = 1L)))
    add("decode:quantiles",
        tensor_diff(as.array(bar_logits_to_quantiles(
                      tempered, borders, as.numeric(meta$quantiles))),
                    as.array(py$bar_quantiles)))
  }

  list(summary = do.call(rbind, rows), meta = meta)
}



#' Compare the R TabPFN v3 backend against a stored Python dump
#'
#' Same shape as [parity_tabpfn26()] -- a bare forward pass graded
#' against the reference's, then the two inference-cost paths, then the
#' decoding graded on the reference's own logits -- with one difference
#' in what the self-checks can assert.
#'
#' v2.6's cache can legitimately change the prediction, so its self-check
#' is conditional. v3's cannot: every statistic the architecture fits
#' comes from the training rows alone, so the cached and uncached paths
#' are the same computation and the self-check is unconditional. What it
#' is not is *bitwise*, which the v2.6 stages are: both the cache and the
#' chunking change the batch shapes the attention kernel sees, and in
#' float32 that moves the last bits. Hence the `-fp32` stage names, which
#' carry a float32-noise tolerance rather than zero.
#'
#' TabPFN v3.5 is graded by the same function. Its forward pass differs
#' from v3's -- a Fourier cell embedder over an in-context ECDF, QK-norm,
#' one multitask checkpoint -- but none of that changes what a comparison
#' has to *do*: the stages, the paths and the self-check semantics are
#' identical, so the backend is a parameter rather than a second copy.
#' See [parity_tabpfn35()].
#'
#' @param reference_dir Directory written by `tabpfn3_reference.py`
#'   (or `tabpfn35_reference.py`).
#' @param fixture_dir,fixture Parity fixture location and name.
#' @param model_dir Converted artifacts for the matching head.
#' @param backend Which backend to load `model_dir` through.
#' @keywords internal
parity_tabpfn3 <- function(reference_dir, fixture_dir, fixture,
                           label = fixture, model_dir,
                           backend = "tabpfn3") {
  stopifnot(dir.exists(reference_dir))
  meta <- jsonlite::fromJSON(file.path(reference_dir, "reference.json"))
  fx   <- read_parity_fixture(fixture_dir, fixture)
  py   <- safetensors::safe_load_file(
    file.path(reference_dir, "forward.safetensors"), framework = "torch")

  rows <- list()
  add <- function(stage, d, note = "") {
    rows[[length(rows) + 1L]] <<- data.frame(
      fixture = label, stage = stage, n = d$n,
      max_abs = d$max_abs, max_rel = d$max_rel,
      max_scaled = d$max_scaled %||% NA_real_,
      note = note, stringsAsFactors = FALSE
    )
  }

  task <- meta$task
  ctx <- load_backend_model(model_dir, task = task, backend = backend,
                            device = "cpu")
  net <- ctx$net

  # Same bytes into both networks: the reference dump carries the exact
  # tensors it was given, so nothing can drift through the fixture reader.
  x_train <- py$x_train$unsqueeze(1L)
  x_test  <- py$x_test$unsqueeze(1L)
  y_train <- py$y_train$unsqueeze(1L)

  # `row_chunk_size = NULL` is the unchunked pass, asked for by name. The
  # network's own default is the checkpoint's 2048/4, matching what the
  # reference does when nobody passes `performance_options` -- so on a
  # fixture above 2048 rows, leaving this out would compare a chunked R
  # pass against an unchunked Python one and call the difference a
  # mismatch. The Python harness names its paths the same way.
  out <- torch::with_no_grad(net(x_train, y_train, x_test,
                                 row_chunk_size = NULL))
  r_plain <- as.array(out$logits[1, , ])
  add("forward:logits", tensor_diff(r_plain, as.array(py$logits)))

  # Neither of v3's cheap paths is bitwise: both change the batch shapes
  # the attention kernel sees, which in float32 moves the last bits. How
  # far they move it grows with the table, so a fixed bound is the wrong
  # instrument -- on 2,664 rows the reference's *own* chunked pass sits
  # 1.5e-5 of scale from its own plain one, above the 1e-5 these rows
  # used to carry. Graded as a ratio instead: the port may reorganise the
  # arithmetic, but no more than the reference reorganises it. The
  # absolute bound is kept as an alternative so a fixture small enough
  # for both sides to be near-exact still passes on its own terms.
  py_shift <- function(nm) {
    if (is.null(py[[nm]])) return(NA_real_)
    max(abs(as.array(py[[nm]]) - as.array(py$logits)))
  }
  add_selfcheck <- function(stage, r_alt, ref_name) {
    shift <- max(abs(r_alt - r_plain))
    ref <- py_shift(ref_name)
    ratio <- if (!is.na(ref) && ref > 0) shift / ref
             else if (shift == 0) 0 else Inf
    add(stage,
        list(n = length(r_alt), max_abs = shift, max_rel = ratio,
             max_scaled = shift / max(abs(r_plain))),
        note = sprintf("reference's own gap %.3e", ref))
  }

  if (!is.null(py$logits_chunked)) {
    chunked <- as.array(torch::with_no_grad(net(
      x_train, y_train, x_test, row_chunk_size = NULL,
      save_peak_memory_factor = as.integer(meta$save_peak_memory_factor %||% 8L)
    ))$logits[1, , ])
    add("forward:chunked", tensor_diff(chunked, as.array(py$logits_chunked)))
    add_selfcheck("selfcheck:chunked-fp32", chunked, "logits_chunked")
  }

  # The row/column chunking of stages 0-2, at the chunk sizes the
  # reference used. Below `row_chunk_size` rows neither side's loop runs
  # more than once, so the row is the plain forward under another name
  # and is labelled "-inactive"; above it, this is the mechanism.
  if (!is.null(py$logits_stage_chunked)) {
    stage_chunked <- as.array(torch::with_no_grad(net(
      x_train, y_train, x_test,
      row_chunk_size = as.integer(meta$row_chunk_size %||% 2048L),
      col_chunk_size = as.integer(meta$col_chunk_size %||% 4L)
    ))$logits[1, , ])
    active <- isTRUE(meta$stage_chunking_active)
    add(if (active) "forward:stage-chunked" else "forward:stage-chunked-inactive",
        tensor_diff(stage_chunked, as.array(py$logits_stage_chunked)),
        note = sprintf("row %s / col %s", meta$row_chunk_size %||% "?",
                       meta$col_chunk_size %||% "?"))
    # And against R's own plain pass, ratio-graded like the other two
    # mechanisms: the port may reorganise the arithmetic, but no more
    # than the reference reorganises it for the same reason.
    if (active) {
      add_selfcheck("selfcheck:stage-chunked-fp32", stage_chunked,
                    "logits_stage_chunked")
    } else {
      # Nothing was chunked, so this must be bit-for-bit the plain pass.
      add("selfcheck:stage-chunked",
          tensor_diff(stage_chunked, r_plain),
          note = "chunk size exceeds the row count; one iteration")
    }
  }

  if (!is.null(py$logits_cached)) {
    no_rows <- torch::torch_zeros(c(1L, 0L, py$x_train$size(2)))
    cache <- torch::with_no_grad(net(
      x_train, y_train, no_rows, return_kv_cache = TRUE,
      row_chunk_size = NULL))$kv_cache
    cached <- as.array(torch::with_no_grad(net(
      NULL, NULL, x_test, kv_cache = cache, row_chunk_size = NULL))$logits[1, , ])
    add("forward:cached", tensor_diff(cached, as.array(py$logits_cached)))
    add_selfcheck("selfcheck:cached-fp32", cached, "logits_cached")
  }

  temp <- meta$softmax_temperature %||% 0.9
  tempered <- py$logits / temp

  if (identical(task, "classification")) {
    n_classes <- as.integer(meta$n_classes)
    r_probs <- torch::nnf_softmax(tempered[, 1:n_classes], dim = -1L)
    add("decode:probs", tensor_diff(as.array(r_probs), as.array(py$probs)))

    mod <- fit(tabular_classifier(model_dir, backend = backend,
                                  device = "cpu", softmax_temperature = temp),
               fx$x_train, as.integer(fx$y_train))
    add("predict_proba:single",
        tensor_diff(predict(mod, fx$x_test, type = "prob"),
                    as.array(py$probs)))
  } else {
    borders <- .regression_borders_of(net)
    add("decode:mean",
        tensor_diff(matrix(as.numeric(as.array(
                      bar_logits_to_mean(tempered, borders))), ncol = 1L),
                    matrix(as.numeric(as.array(py$bar_mean)), ncol = 1L)))
    add("decode:quantiles",
        tensor_diff(as.array(bar_logits_to_quantiles(
                      tempered, borders, as.numeric(meta$quantiles))),
                    as.array(py$bar_quantiles)))
  }

  list(summary = do.call(rbind, rows), meta = meta)
}


#' Compare the R TabPFN v3.5 backend against a stored Python dump
#'
#' The same comparison [parity_tabpfn3()] performs, against a reference
#' written by `tabpfn35_reference.py` and loaded through the `tabpfn35`
#' backend. What v3.5 adds to the architecture -- the in-context ECDF the
#' cell embedder reads, and one checkpoint serving both tasks -- shows up
#' in the reference dump rather than in the grading: the ECDF is part of
#' the forward pass being compared, and the task is decided by the
#' fixture's name on both sides.
#'
#' @inheritParams parity_tabpfn3
#' @keywords internal
parity_tabpfn35 <- function(reference_dir, fixture_dir, fixture,
                            label = fixture, model_dir) {
  parity_tabpfn3(reference_dir, fixture_dir, fixture, label = label,
                 model_dir = model_dir, backend = "tabpfn35")
}


#' Per-stage pass tolerances
#'
#' A single global tolerance does not work here, because the stages
#' compare quantities on very different scales. Preprocessing is
#' deterministic arithmetic on float64 and must agree essentially
#' exactly. Logits are unnormalised and O(10), accumulated across 18-24
#' float32 layers, so they are graded on relative error -- and a uniform
#' shift in logits is invisible after softmax anyway. Probabilities and
#' predictions are the numbers a user actually sees and get the strict
#' absolute bound.
#'
#' A stage passes if *either* the absolute or the relative bound holds.
#' @keywords internal
parity_tolerances <- function() {
  data.frame(
    stage = c("preprocess:X_train", "preprocess:X_test", "preprocess:y_train",
              "forward:logits", "predict_proba", "predict_mean",
              "predict_quantiles",
              # The bare-network finals for TabFM / TabICL / Mitra. Same
              # bounds their `predict_*` rows used to carry -- they grade
              # the same quantity, just without the wrapper in the way.
              # The wrapper is graded separately by
              # `ensemble_reference.py`, with no checkpoint needed.
              "net:probs", "net:mean", "net:quantiles",
              # TabFM stages. These are raw activations, graded on
              # absolute error: they pass through zero constantly, so
              # relative error is meaningless near the crossings.
              "stage:cell", "stage:col1", "stage:row1",
              "stage:col2", "stage:reps", "stage:logits",
              # TabICL stages.
              "stage:col",
              # Mitra stages. The quantile embedding and the packed
              # input are deterministic arithmetic and come out exact.
              "stage:quantile", "stage:embedded", "stage:encoded",
              # TabPFN v2.6. `decode:*` share their input with the
              # reference, so only the decoding arithmetic differs and
              # they must agree tightly. `predict_proba:single` is one
              # forward pass rather than the 4-member ensemble the v2.5
              # `predict_proba` averages, so 24 layers of float32
              # rounding reach it undamped -- 1e-5 is genuinely too
              # tight for it, and 1e-4 is still far below anything that
              # could change a predicted label.
              "decode:probs", "decode:mean", "decode:quantiles",
              "predict_proba:single",
              # Against the reference's own run of the same mechanism:
              # the plain forward's cross-implementation gap, so the plain
              # forward's tolerance.
              "forward:chunked", "forward:cached",
              # Against R's own plain forward: nothing to hide in, so
              # exact.
              "selfcheck:chunked", "selfcheck:cached",
              # v3's counterparts. Both mechanisms are the same
              # computation as the plain forward, but neither is bitwise:
              # they change the batch shapes the attention kernel sees,
              # which in float32 moves the last bits -- and by more the
              # bigger the table, which is why `tol_rel` here is a
              # *ratio* to the reference's own gap between the same two
              # paths rather than an error. Passes at no more than twice
              # the reorganisation the reference itself does, or on the
              # 1e-5 scale bound, which is what a small fixture clears on
              # its own terms.
              # The scale bound on these three is 5e-5, not the 1e-5 the
              # other self-checks carry, because at this fixture size it
              # was never the binding instrument for either backend:
              # measured on `clf_large` (2,664 rows), R's own chunked-vs-
              # plain gap is 2.1e-5 of scale on v3 and 2.7e-5 on v3.5,
              # both above 1e-5, and both passed only on the ratio. The
              # ratio itself tracks the ICL width -- 1.44 at v3's 512,
              # 2.03 at v3.5's 1024 -- because the same reassociation
              # spans twice as many float32 terms. R and Python chunk
              # identically (`ceil(n / factor)` then `split`), so there is
              # no structural difference for a tighter bound to catch: one
              # would show up as orders of magnitude, not as a factor of
              # two. `forward:chunked`, which is the actual
              # cross-implementation claim, passes at 2.7e-4 against 1e-3.
              "selfcheck:chunked-fp32", "selfcheck:cached-fp32",
              # v3's stage-0-2 row/column chunking. Cross-implementation,
              # so the plain forward's bounds; and against R's own plain
              # pass, the same ratio treatment as the other two
              # mechanisms. Below the chunk size the loop runs once and
              # the "-inactive" pair must be exact, which is what catches
              # a driver that copies or reorders even when it has nothing
              # to do.
              "forward:stage-chunked", "forward:stage-chunked-inactive",
              "selfcheck:stage-chunked-fp32", "selfcheck:stage-chunked",
              # Informational: on data where the cache legitimately
              # changes the answer, how much it changed it. Graded by
              # `forward:cached` instead.
              "cache:shifts-prediction",
              # v2/v2.5's cache, graded against the shift that changing
              # the test batch size already causes: exact, or no worse
              # than that. `tol_rel` here is a ratio, not an error.
              "cache:vs-batching"),
    tol_abs = c(1e-6, 1e-6, 1e-6, 1e-3, 1e-5, 1e-4, 1e-4,
                1e-5, 1e-4, 1e-4,
                1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3,
                1e-4,
                1e-6, 1e-6, 1e-4,
                1e-6, 1e-5, 1e-5, 1e-4,
                1e-3, 1e-3, 0, 0, 0, 0, 0, 1e-3, 0, 0, Inf, 0),
    tol_rel = c(1e-6, 1e-6, 1e-6, 1e-3, 1e-4, 1e-4, 1e-4,
                1e-4, 1e-4, 1e-4,
                1e-6, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3,
                1e-3,
                1e-6, 1e-6, 1e-3,
                1e-6, 1e-5, 1e-5, 1e-4,
                1e-3, 1e-3, 0, 0, 2, 2, 0, 1e-3, 2, 0, Inf, 1.5),
    # Scale-relative bound, used only where a tensor's dynamic range
    # makes the other two meaningless. NA disables it.
    tol_scaled = c(NA, NA, NA, 1e-4, NA, NA, NA,
                   NA, NA, NA,
                   NA, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
                   1e-4,
                   NA, NA, 1e-4,
                   NA, 1e-5, 1e-5, NA,
                   1e-4, 1e-4, NA, NA, 5e-5, 5e-5, 1e-4, 1e-4, 5e-5, NA, NA, NA),
    stringsAsFactors = FALSE
  )
}

#' Add a `pass` column to a parity summary
#' @keywords internal
grade_parity <- function(s) {
  tol <- parity_tolerances()
  m <- match(s$stage, tol$stage)
  ta <- ifelse(is.na(m), 1e-5, tol$tol_abs[m])
  tr <- ifelse(is.na(m), 1e-5, tol$tol_rel[m])
  ts <- if (is.null(tol$tol_scaled)) rep(NA_real_, length(m))
        else ifelse(is.na(m), NA_real_, tol$tol_scaled[m])
  s$tol_abs <- ta
  scaled_ok <- !is.na(ts) & !is.null(s$max_scaled) &
    !is.na(s$max_scaled) & s$max_scaled <= ts
  s$pass <- !is.na(s$max_abs) &
    (s$max_abs <= ta | s$max_rel <= tr | scaled_ok)
  s
}

#' Pretty-print a parity summary
#' @keywords internal
print_parity <- function(res) {
  s <- grade_parity(res$summary)
  s <- s[order(s$stage), , drop = FALSE]
  has_scaled <- !is.null(s$max_scaled) && any(!is.na(s$max_scaled))
  cat(sprintf("\n%-22s %-22s %10s %12s %12s %12s  %s\n",
              "FIXTURE", "STAGE", "N", "MAX ABS", "MAX REL",
              if (has_scaled) "MAX/SCALE" else "", ""))
  for (i in seq_len(nrow(s))) {
    sc <- if (has_scaled && !is.na(s$max_scaled[i]))
      sprintf("%12.3e", s$max_scaled[i]) else strrep(" ", 12)
    cat(sprintf("%-22s %-22s %10s %12.3e %12.3e %s  %s %s\n",
                s$fixture[i], s$stage[i], format(s$n[i], big.mark = ","),
                s$max_abs[i], s$max_rel[i], sc,
                if (isTRUE(s$pass[i])) "PASS" else "FAIL", s$note[i]))
  }
  invisible(s)
}
