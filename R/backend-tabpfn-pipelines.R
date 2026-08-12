# TabPFN ensemble member pipelines and ensemble prediction.
#
# Sits between the raw user X/y and the network: each ensemble member
# gets its own view of the data (preprocessing preset + column shuffle +
# class rotation), runs its own forward pass, and the member outputs are
# averaged back in the original label / target space.
#
# The individual transforms are backend-agnostic and live in
# `R/prep-transforms.R`; what is TabPFN-specific is which presets exist,
# the order they compose in, and how member outputs are combined.

# ---------------------------------------------------------------------------
# One member's preprocessing pipeline
# ---------------------------------------------------------------------------
#
# The reference assembles a member's pipeline from its config
# (`create_preprocessing_pipeline`) in a fixed order, and every step is
# optional:
#
#   polynomial features -> remove constant -> primary reshape
#     -> encode categorical -> SVD -> fingerprint -> shuffle
#
# So this is one composer driven by the member's config rather than one
# function per preset. That is not just tidier: TabPFN v2.6 uses the same
# primary transform (`quantile_uni`) for both of its member types and
# distinguishes them by the *other* fields, which a name-keyed dispatch
# cannot see.

#' Resolve `append_original = "auto"`
#'
#' The reference appends the transformed block to the originals only when
#' the table is narrow enough that doubling its width is affordable --
#' under 500 columns, and no more than half the per-estimator budget.
#' @keywords internal
resolve_append_original <- function(append_original, n_features,
                                    max_features_per_estimator = 500L) {
  if (identical(append_original, "auto")) {
    return(n_features < 500L &&
             n_features <= (as.numeric(max_features_per_estimator) / 2))
  }
  isTRUE(append_original)
}

#' Fit a member's primary column transform
#'
#' Fitted on train, applied to train and to every prediction chunk. The
#' fit is the expensive half -- the quantile transformer sorts every
#' column of the training matrix -- and it depends on nothing but the
#' training rows, so it is separated from the application and done once.
#'
#' @return A list with `kind` and whatever that kind needs;
#'   [transform_primary_reshape()] consumes it.
#' @keywords internal
fit_primary_reshape <- function(X_train, preset) {
  # Every column categorical and the member not transforming those: there
  # is nothing left to reshape.
  if (ncol(X_train) == 0L || identical(preset, "none")) {
    return(list(kind = "identity"))
  }
  if (identical(preset, "squashing_scaler_default") ||
      identical(preset, "squashing_scaler_max10")) {
    return(list(kind = "squashing", fit = fit_squashing_scaler(
      X_train,
      max_absolute_value = if (identical(preset, "squashing_scaler_max10")) 10 else 3
    )))
  }
  if (startsWith(preset, "quantile_uni")) {
    return(list(kind = "quantile", fit = fit_quantile_transformer(
      X_train,
      n_quantiles = quantile_preset_n_quantiles(preset, nrow(X_train)),
      extrapolate_ratio = quantile_preset_extrapolate_ratio(preset)
    )))
  }
  cli::cli_abort(c(
    "Preprocessing preset {.val {preset}} is not implemented.",
    i = "Implemented: {.val {c('none', 'squashing_scaler_default',
                               'squashing_scaler_max10', 'quantile_uni',
                               'quantile_uni_coarse', 'quantile_uni_fine',
                               'quantile_uni_extrapolate')}}."
  ))
}

#' @rdname fit_primary_reshape
#' @keywords internal
transform_primary_reshape <- function(X, fitted) {
  switch(fitted$kind,
         identity  = X,
         squashing = transform_squashing_scaler(X, fitted$fit),
         quantile  = transform_quantile_transformer(X, fitted$fit),
         cli::cli_abort("Unknown primary reshape {.val {fitted$kind}}."))
}

#' Apply a member's primary column transform
#'
#' Fit-and-transform-both, for callers that have the pair in hand.
#' @keywords internal
apply_primary_reshape <- function(X_train, X_test, preset) {
  fitted <- fit_primary_reshape(X_train, preset)
  list(train = transform_primary_reshape(X_train, fitted),
       test  = transform_primary_reshape(X_test, fitted))
}

#' Which categorical columns an encoder actually touches
#'
#' The `ordinal` family comes in three widths. Plain `ordinal` encodes
#' every categorical column. The `_common_categories` variants first drop
#' columns whose rarest level is thin, on the grounds that a level seen
#' fewer than ten times cannot teach the model anything and its code would
#' just be noise; `_very_common_categories` additionally drops columns
#' with too many levels for the row count.
#'
#' @param X_train Numeric matrix, used for the level counts.
#' @param cat_ix Integer vector of 1-based categorical columns.
#' @param name The member's `categorical_name`.
#' @return The subset of `cat_ix` the encoder will act on.
#' @keywords internal
select_encoded_categoricals <- function(X_train, cat_ix, name) {
  if (!length(cat_ix)) return(integer())
  X <- as.matrix(X_train)
  rest <- sub("^ordinal", "", name)
  if (startsWith(rest, "_very_common_categories")) {
    keep <- vapply(cat_ix, function(j) {
      col <- X[, j]
      n_unique <- length(unique(col[!is.na(col)])) + as.integer(anyNA(col))
      least_common_category_count(col) >= 10L && n_unique < (nrow(X) %/% 10L)
    }, logical(1))
    return(cat_ix[keep])
  }
  if (startsWith(rest, "_common_categories")) {
    keep <- vapply(cat_ix, function(j) {
      least_common_category_count(X[, j]) >= 10L
    }, logical(1))
    return(cat_ix[keep])
  }
  cat_ix
}


#' Apply a member's categorical encoding
#'
#' `numeric` and `none` leave the matrix alone -- the difference between
#' them is upstream, in whether the primary transform was applied to the
#' categorical columns. The `ordinal` family encodes the selected columns
#' and **moves them to the front**, because the reference builds this out
#' of a `ColumnTransformer` with `remainder = "passthrough"` and that is
#' the layout it produces. Getting the reordering wrong is silent: the
#' widths still match and the network still runs.
#'
#' @param X_train,X_test Numeric matrices.
#' @param cat_ix Integer vector of 1-based categorical columns.
#' @param cfg The member config; reads `categorical_name` and, for the
#'   `_shuffled` variants, `cat_mappings`.
#' @param draw_missing When the member shuffles codes but carries no
#'   mappings, draw them with R's RNG instead of refusing. Only
#'   [generate_ensemble_configs_native()] does this -- it is how the
#'   generator learns which columns the encoder selects and how many
#'   categories each has, without reimplementing the pipeline.
#' @return `list(X_train, X_test, cat_ix, cat_mappings)`.
#' @keywords internal
apply_categorical_encoding <- function(X_train, X_test, cat_ix, cfg,
                                       draw_missing = FALSE) {
  fitted <- fit_categorical_encoding(X_train, cat_ix, cfg, draw_missing)
  list(X_train = transform_categorical_encoding(X_train, fitted),
       X_test  = transform_categorical_encoding(X_test, fitted),
       cat_ix  = fitted$cat_ix,
       cat_mappings = fitted$cat_mappings)
}

#' @rdname apply_categorical_encoding
#' @return For `fit_categorical_encoding()`, the encoder plus the column
#'   indices [transform_categorical_encoding()] needs; the fit reads the
#'   training rows (level counts, category sets) and nothing else.
#' @keywords internal
fit_categorical_encoding <- function(X_train, cat_ix, cfg,
                                     draw_missing = FALSE) {
  name <- cfg$categorical_name %||% "numeric"
  if (name %in% c("numeric", "none")) {
    return(list(kind = "identity", cat_ix = cat_ix, cat_mappings = NULL))
  }
  if (identical(name, "onehot")) {
    cli::cli_abort(c(
      "One-hot categorical encoding is not implemented.",
      i = "Its width depends on the data, so a member's shuffle \\
           permutation cannot be sized ahead of it.",
      i = "No released TabPFN checkpoint asks for it -- v2.5 and v2.6 both \\
           use {.val numeric} and \\
           {.val ordinal_very_common_categories_shuffled}."
    ))
  }
  if (!startsWith(name, "ordinal")) {
    cli::cli_abort("Unknown categorical encoding {.val {name}}.")
  }

  sel <- select_encoded_categoricals(X_train, cat_ix, name)
  if (!length(sel)) {
    return(list(kind = "identity", cat_ix = integer(), cat_mappings = list()))
  }

  enc <- fit_ordinal_encoder(X_train, sel)
  mappings <- NULL
  if (endsWith(name, "_shuffled")) {
    mappings <- cfg$cat_mappings
    if (is.null(mappings) && isTRUE(draw_missing)) {
      mappings <- lapply(enc$n_categories, function(k) sample.int(k) - 1L)
    }
    if (is.null(mappings)) {
      cli::cli_abort(c(
        "This member shuffles categorical codes but carries no mappings.",
        i = "The permutations come from the reference's NumPy generator, \\
             so they have to travel with the config -- regenerate the dump, \\
             or use {.fn generate_ensemble_configs_native}."
      ))
    }
    if (length(mappings) != length(sel)) {
      cli::cli_abort(c(
        "This member carries {length(mappings)} code mapping{?s} for \\
         {length(sel)} encoded column{?s}.",
        i = "The config and the data disagree about which columns are \\
             categorical."
      ))
    }
    for (k in seq_along(sel)) {
      if (length(mappings[[k]]) < enc$n_categories[k]) {
        cli::cli_abort(c(
          "Column {sel[k]} has {enc$n_categories[k]} categories but its \\
           mapping is {length(mappings[[k]])} long."
        ))
      }
    }
  }

  # `[encoded, remainder]` -- the remainder keeps its original order.
  list(
    kind = "ordinal",
    enc = enc, mappings = mappings,
    rest = setdiff(seq_len(ncol(X_train)), sel),
    cat_ix = seq_along(sel),
    cat_mappings = mappings
  )
}

#' @rdname apply_categorical_encoding
#' @keywords internal
transform_categorical_encoding <- function(X, fitted) {
  if (identical(fitted$kind, "identity")) return(X)
  cbind(transform_ordinal_encoder(X, fitted$enc, fitted$mappings),
        X[, fitted$rest, drop = FALSE])
}


#' Build one ensemble member's model inputs
#'
#' [fit_member_pipeline()] does everything that depends on the training
#' rows -- every step's fit, and the transformed training matrix -- and
#' [transform_member_pipeline()] replays it on query rows.
#'
#' The split is what keeps prediction linear in the number of chunks.
#' Every step here is fitted on train and applied to both, and none of it
#' depends on the query rows, so a fit inside the chunk loop is executed
#' `n_members * n_chunks` times where `n_members` would do: at 8 members
#' and 20K test rows that is 160 quantile transformers (a per-column sort
#' of the whole training matrix each) in place of 8. `apply_member_pipeline()`
#' remains as the fit-and-transform-both convenience for callers holding
#' a single pair -- the config generator, and the tests.
#'
#' @param X_train,X_test Numeric matrices of raw inputs.
#' @param cfg One member spec from [load_ensemble_configs_from_dump()] or
#'   [generate_ensemble_configs_native()]. A `NULL` `shuffle_perm` skips
#'   the final shuffle, which is how the generator asks how wide a member
#'   ends up before drawing a permutation of that width.
#' @param categorical_features Integer vector of 1-based categorical
#'   column indices in `X_train`, from [detect_categorical_features()].
#'   Empty means every column is numeric, which reduces the whole
#'   categorical path to a no-op.
#' @param draw_missing Passed to [apply_categorical_encoding()].
#' @return `list(X_train, X_test)`, ready for the network.
#' @keywords internal
apply_member_pipeline <- function(X_train, X_test, cfg,
                                  categorical_features = integer(),
                                  draw_missing = FALSE) {
  fitted <- fit_member_pipeline(X_train, cfg, categorical_features,
                                draw_missing)
  list(X_train = fitted$X_train,
       X_test  = transform_member_pipeline(X_test, fitted),
       cat_mappings = fitted$cat_mappings)
}

#' @rdname apply_member_pipeline
#' @return For `fit_member_pipeline()`, the transformed training matrix
#'   plus the fitted steps; feed both to [transform_member_pipeline()].
#' @keywords internal
fit_member_pipeline <- function(X_train, cfg,
                                categorical_features = integer(),
                                draw_missing = FALSE) {
  X_tr <- as.matrix(X_train)
  cat_ix <- as.integer(categorical_features)
  steps <- list()

  # --- Polynomial features (TabPFN v2.6's regressor; "no" elsewhere).
  # The step rescales the base columns in place and appends the products
  # as numeric ones, so `cat_ix` still points where it did. Rescaling a
  # categorical column is monotone, so the ordinal encoder downstream
  # recovers the same codes from it.
  poly <- cfg$polynomial_features %||% "no"
  if (!identical(poly, "no")) {
    if (is.null(cfg$poly_factor_1)) {
      cli::cli_abort(c(
        "This member asks for polynomial features but carries no factor indices.",
        i = "Which column pairs get multiplied is drawn from the reference's \\
             NumPy generator, so it has to come from the config -- regenerate \\
             the dump with a {.pkg tabfound} recent enough to write \\
             {.file poly_factors.safetensors}."
      ))
    }
    pf <- fit_polynomial_features(X_tr, cfg$poly_factor_1, cfg$poly_factor_2)
    X_tr <- transform_polynomial_features(X_tr, pf)
    steps$poly <- pf
  }

  # --- RemoveConstant, fitted on train. Surviving categorical columns
  # keep their identity, at their new positions.
  keep_cols <- remove_constant_features_fit(X_tr)
  X_tr <- X_tr[, keep_cols, drop = FALSE]
  cat_ix <- match(cat_ix[cat_ix %in% keep_cols], keep_cols)
  steps$keep_cols <- keep_cols

  max_feats <- cfg$max_features_per_estimator %||% 500L
  if (ncol(X_tr) > max_feats) {
    cli::cli_abort(c(
      "{ncol(X_tr)} columns exceeds this member's budget of {max_feats}.",
      i = "Past the budget the reference subsamples features per estimator, \\
           which is not implemented here.",
      i = "Reduce the predictor count, or predict without \\
           {.arg ensemble_configs_dir}."
    ))
  }

  # --- Primary reshape. Four layouts, decided by whether the transformed
  # block is appended to the originals and whether the categorical columns
  # go through the transform at all. Where they do not, they are passed
  # through *in front of* the transformed block rather than left in place;
  # where the block replaces the originals, the transform sees the
  # categorical columns first and the output inherits that order.
  n_in <- ncol(X_tr)
  append <- resolve_append_original(cfg$append_original, n_in, max_feats)
  # `numeric` is the one encoding that means "these are just numbers":
  # only then does the primary transform touch the categorical columns.
  apply_to_cat <- identical(cfg$categorical_name %||% "numeric", "numeric")
  num_ix <- setdiff(seq_len(n_in), cat_ix)

  trans_ix <- if (apply_to_cat) c(cat_ix, num_ix) else num_ix
  pass_ix  <- if (append) seq_len(n_in) else if (apply_to_cat) integer() else cat_ix
  cat_ix   <- if (append) cat_ix
              else if (apply_to_cat) integer()
              else seq_along(cat_ix)

  steps$trans_ix <- trans_ix
  steps$pass_ix  <- pass_ix
  steps$reshape  <- fit_primary_reshape(X_tr[, trans_ix, drop = FALSE],
                                        cfg$preset)
  X_tr <- cbind(X_tr[, pass_ix, drop = FALSE],
                transform_primary_reshape(X_tr[, trans_ix, drop = FALSE],
                                          steps$reshape))

  # --- EncodeCategorical.
  enc <- fit_categorical_encoding(X_tr, cat_ix, cfg, draw_missing)
  X_tr <- transform_categorical_encoding(X_tr, enc)
  steps$encode <- enc

  # --- SVD global transformer, appended.
  gt <- cfg$global_transformer_name
  if (!is.null(gt) && !identical(gt, "None")) {
    svd_fit <- fit_transform_svd_features(X_tr, global_name = gt)
    if (!isTRUE(svd_fit$is_no_op)) {
      X_tr <- cbind(X_tr, transform_svd_features(X_tr, svd_fit))
      steps$svd <- svd_fit
    }
  }

  # --- Fingerprint: one hash column per row, salted with the train shape.
  # The salt is the *train* shape wherever it is applied, so it is fixed
  # here and the query rows inherit it.
  if (!identical(cfg$add_fingerprint %||% TRUE, FALSE)) {
    steps$fingerprint_salt <- nrow(X_tr) * ncol(X_tr)
    X_tr <- cbind(X_tr, apply_fingerprint(X_tr, steps$fingerprint_salt,
                                          is_test = FALSE))
  }

  # --- Shuffle. `cat_mappings` comes back out so the generator can record
  # what it drew; the network itself is not told which columns are
  # categorical (both architectures ignore that argument), so the schema
  # stops mattering here.
  steps$shuffle_perm <- cfg$shuffle_perm
  if (!is.null(cfg$shuffle_perm)) {
    X_tr <- apply_feature_shift(X_tr, cfg$shuffle_perm)
  }

  c(list(X_train = X_tr, cat_mappings = enc$cat_mappings), steps)
}

#' @rdname apply_member_pipeline
#' @param fitted The result of [fit_member_pipeline()].
#' @param X Query rows to put through the fitted pipeline.
#' @keywords internal
transform_member_pipeline <- function(X, fitted) {
  X <- as.matrix(X)
  if (!is.null(fitted$poly)) X <- transform_polynomial_features(X, fitted$poly)
  X <- X[, fitted$keep_cols, drop = FALSE]
  X <- cbind(X[, fitted$pass_ix, drop = FALSE],
             transform_primary_reshape(X[, fitted$trans_ix, drop = FALSE],
                                       fitted$reshape))
  X <- transform_categorical_encoding(X, fitted$encode)
  if (!is.null(fitted$svd)) X <- cbind(X, transform_svd_features(X, fitted$svd))
  if (!is.null(fitted$fingerprint_salt)) {
    X <- cbind(X, apply_fingerprint(X, fitted$fingerprint_salt, is_test = TRUE))
  }
  if (!is.null(fitted$shuffle_perm)) X <- apply_feature_shift(X, fitted$shuffle_perm)
  X
}


#' Fit a Yeo-Johnson + StandardScaler pipeline on y (lambda supplied).
#' Returns (m, s) of the StandardScaler stage.
#' @keywords internal
fit_target_transform_state <- function(z_y, lambda) {
  yt <- yeojohnson_forward(z_y, lambda)
  m  <- mean(yt)
  s  <- sqrt(mean((yt - m) ^ 2))   # population std (ddof=0), matches sklearn
  list(lambda = lambda, m = m, s = s)
}

#' Apply the fitted target_transform to (z-standardized) y.
#' @keywords internal
apply_target_transform <- function(z_y, tt_state) {
  yt <- yeojohnson_forward(z_y, tt_state$lambda)
  (yt - tt_state$m) / tt_state$s
}

#' Inverse-transform the z-norm borders through the fitted target_transform.
#' Returns borders in z-y space, with optional logit_cancel_mask.
#' @keywords internal
inverse_target_transform_borders <- function(znorm_borders, tt_state) {
  # Pipeline inverse: first undo StandardScaler, then undo yeojohnson.
  zy_post_yj <- as.numeric(znorm_borders) * tt_state$s + tt_state$m
  zy         <- yeojohnson_inverse(zy_post_yj, tt_state$lambda)
  res <- list(borders_t = zy, logit_cancel_mask = NULL, descending = FALSE)

  broken_mask <- !is.finite(zy) |
    zy > REGRESSION_NAN_BORDER_LIMIT_UPPER |
    zy < REGRESSION_NAN_BORDER_LIMIT_LOWER
  if (any(broken_mask)) {
    cnc <- .cancel_nan_borders(zy, broken_mask)
    res$borders_t <- cnc$borders
    res$logit_cancel_mask <- cnc$logit_cancel_mask
  }
  res$borders_t <- .repair_borders(res$borders_t)

  res$descending <- all(rev(order(res$borders_t)) == seq_along(res$borders_t))
  if (res$descending) {
    res$borders_t <- rev(res$borders_t)
    if (!is.null(res$logit_cancel_mask)) res$logit_cancel_mask <- rev(res$logit_cancel_mask)
  }
  res
}

#' Load an ensemble config set from a Python-dumped directory.
#'
#' The directory must contain `ensemble_configs.json` and one
#' `member_<NN>/shuffle_perm.safetensors` per member, as produced by
#' `inst/parity/tabpfn_reference.py`. Everything the reference draws from
#' its NumPy generator -- the column shuffle, the class rotation, the
#' polynomial factor pairs, the target-transform lambda -- is read from
#' the dump rather than recomputed, because reproducing PCG64 in R is not
#' worth the days it would take.
#'
#' @param dump_dir Path to a dumped ensemble directory.
#' @return A list of member specs, each carrying the preprocessing
#'   settings ([apply_member_pipeline()] reads them) plus `class_perm`
#'   (0-indexed) and `target_transform_lambda`.
#' @keywords internal
load_ensemble_configs_from_dump <- function(dump_dir) {
  require_suggested("jsonlite")
  require_suggested("safetensors")
  cfg <- jsonlite::fromJSON(file.path(dump_dir, "ensemble_configs.json"),
                            simplifyVector = FALSE)
  read_t1 <- function(path) {
    if (!file.exists(path)) return(NULL)
    safetensors::safe_load_file(path, framework = "torch")
  }
  lapply(seq_along(cfg), function(i) {
    c <- cfg[[i]]
    pc <- c$preprocess_config
    member_dir <- file.path(dump_dir, sprintf("member_%02d", i - 1L))
    perm_py <- as.integer(read_t1(
      file.path(member_dir, "shuffle_perm.safetensors"))$t$cpu())

    lambdas <- read_t1(file.path(member_dir, "target_transform_lambdas.safetensors"))
    poly <- read_t1(file.path(member_dir, "poly_factors.safetensors"))
    cat_maps <- read_t1(file.path(member_dir, "cat_mappings.safetensors"))

    list(
      preset                     = pc$name,
      categorical_name           = pc$categorical_name %||% "numeric",
      # Keyed `col_00`, `col_01`, ... in encoded-column order, so sorting
      # the names recovers that order.
      cat_mappings               = if (is.null(cat_maps)) NULL else
                                    lapply(sort(names(cat_maps)),
                                           function(k) as.integer(cat_maps[[k]]$cpu())),
      append_original            = pc$append_original %||% FALSE,
      max_features_per_estimator = as.integer(pc$max_features_per_estimator %||% 500L),
      global_transformer_name    = pc$global_transformer_name,
      polynomial_features        = c$polynomial_features %||% "no",
      # 0-indexed in the dump, like every other index the reference writes.
      poly_factor_1              = if (is.null(poly)) NULL
                                    else as.integer(poly$factor_1$cpu()) + 1L,
      poly_factor_2              = if (is.null(poly)) NULL
                                    else as.integer(poly$factor_2$cpu()) + 1L,
      class_perm                 = if (is.null(c$class_permutation)) NULL
                                    else as.integer(c$class_permutation$vals),
      shuffle_perm               = perm_py + 1L,
      target_transform_lambda    = if (is.null(lambdas)) NULL
                                    else as.numeric(lambdas$t$cpu()),
      # Honour the member's own setting rather than assuming TRUE: the
      # fingerprint column can be switched off (`FINGERPRINT_FEATURE`
      # in the reference's inference config), and the parity harness
      # uses that to separate hash chaos from real numeric drift.
      add_fingerprint            = !identical(c$add_fingerprint_feature, FALSE)
    )
  })
}

# One forward call for every TabPFN network.
#
# The generations accept different arguments -- v2 has neither optional
# path, v2.5 has the KV cache, v2.6 has both, and v3 has both but no
# column embeddings at all (RoPE over the feature axis replaced them) --
# and naming an argument a network does not declare is an R error rather
# than a no-op. So the call is assembled from what the network advertises.
# @keywords internal
tabpfn_forward <- function(net, x_train, y_train, x_test, col_emb,
                           kv_cache = NULL, return_kv_cache = FALSE,
                           save_peak_memory_factor = NULL,
                           row_chunk_size = NA_integer_,
                           col_chunk_size = NA_integer_) {
  args <- list(x_train, y_train, x_test)
  if (isTRUE(net$needs_column_embeddings)) args$column_embeddings <- col_emb
  if (isTRUE(net$supports_kv_cache)) {
    args$kv_cache <- kv_cache
    args$return_kv_cache <- return_kv_cache
  }
  if (isTRUE(net$supports_chunked_eval)) {
    args$save_peak_memory_factor <- save_peak_memory_factor
  }
  # `NA` means "the checkpoint's own", so passing it through is how a
  # caller who said nothing still gets the reference's defaults.
  if (isTRUE(net$supports_stage_chunking)) {
    args$row_chunk_size <- row_chunk_size
    args$col_chunk_size <- col_chunk_size
  }
  do.call(net, args)
}


#' Fit each member's pipeline once, for one `predict()` call
#'
#' The counterpart of [member_cache_store()], and for the same reason:
#' what a member's pipeline is fitted on -- the training rows and that
#' member's config -- does not change between prediction chunks, so
#' fitting it inside the chunk loop multiplies the cost by the chunk
#' count. Built per `predict()` call rather than kept on the fitted
#' object, so nothing outlives the answer or goes stale against a refit.
#'
#' The KV-cache builder and the prediction path share one store, which
#' also removes the duplicate fit between them.
#'
#' @param X_train Raw training predictors.
#' @param configs Member specs.
#' @param categorical_features Passed to [fit_member_pipeline()].
#' @return A function of the member index returning its fitted pipeline.
#' @keywords internal
member_pipeline_store <- function(X_train, configs,
                                  categorical_features = integer()) {
  X_tr <- as.matrix(X_train); storage.mode(X_tr) <- "double"
  cache <- new.env(parent = emptyenv())
  function(i) {
    key <- as.character(i)
    hit <- cache[[key]]
    if (!is.null(hit)) return(hit)
    fitted <- fit_member_pipeline(X_tr, configs[[i]], categorical_features)
    assign(key, fitted, envir = cache)
    fitted
  }
}

#' Build one KV cache per ensemble member
#'
#' Every member sees a differently preprocessed view of the same training
#' rows, so every member needs its own cache. Building them costs one
#' forward pass per member over the training rows; after that a prediction
#' costs only the test rows, however many batches they arrive in.
#'
#' Only backends whose network advertises `supports_kv_cache` can do this.
#'
#' @param net,col_emb Loaded network and column embeddings.
#' @param X_train Raw training predictors.
#' @param y_train Raw training target, in whatever form `member_y` expects.
#' @param configs Member specs.
#' @param member_y Function `(cfg, y_train)` returning the target this
#'   member conditions on -- a class rotation for the classifier, a
#'   possibly transformed target for the regressor.
#' @param categorical_features Passed to [apply_member_pipeline()].
#' @return List of caches, one per member.
#' @keywords internal
build_member_kv_caches <- function(net, col_emb, X_train, y_train, configs,
                                   member_y, device = "cpu",
                                   categorical_features = integer(),
                                   save_peak_memory_factor = NULL,
                                   row_chunk_size = NA_integer_,
                                   col_chunk_size = NA_integer_,
                                   pipelines = NULL) {
  X_tr <- as.matrix(X_train); storage.mode(X_tr) <- "double"
  # A member's pipeline fits on the training rows only -- there is no
  # test side to this call at all.
  pipelines <- pipelines %||%
    member_pipeline_store(X_tr, configs, categorical_features)
  lapply(seq_along(configs), function(i) {
    cfg <- configs[[i]]
    mem <- pipelines(i)
    x_tr <- as_float_tensor(mem$X_train, device = device)$unsqueeze(1L)
    y_tr <- as_float_tensor(
      matrix(as.numeric(member_y(cfg, y_train)), ncol = 1L), device = device
    )$squeeze(-1L)$unsqueeze(1L)
    no_rows <- torch::torch_zeros(c(1L, 0L, x_tr$size(3)), device = device)
    torch::with_no_grad(
      tabpfn_forward(net, x_tr, y_tr, no_rows, col_emb, return_kv_cache = TRUE,
                     save_peak_memory_factor = save_peak_memory_factor,
                     row_chunk_size = row_chunk_size,
                     col_chunk_size = col_chunk_size)
    )$kv_cache
  })
}


#' Run a full ensemble prediction (regressor), averaging bar-dist probs.
#'
#' Every member's output is re-binned onto the shared border grid before
#' averaging, including members whose target transform is the identity.
#' That looks redundant -- the source and destination grids are the same
#' tensor -- but it is not: a `FullSupportBarDistribution` puts half-normal
#' tails on its outer buckets, so evaluating its CDF at its own borders
#' re-quantises the tail mass rather than reproducing it. Skipping the
#' round trip for identity members shifts the predicted mean by ~1e-4 of
#' its own scale, which is 500x the error that remains once it is done.
#'
#' @param net       Loaded `per_feature_transformer` (regressor head).
#' @param col_emb   Pre-generated column embedding tensor.
#' @param X_train,X_test Numeric matrices of raw inputs.
#' @param y_train   Numeric vector of raw regression targets.
#' @param configs   List of regressor ensemble-member specs; each must
#'   have `target_transform = NULL`.
#' @param borders   `torch_tensor` of length `n_bar_bins + 1` -- the
#'   model's `criterion$borders`, which live in the **z-normalized
#'   target space** (after Python's `(y - mean) / std` step). The
#'   function un-standardizes at the end.
#' @return List with `mean`, `quantiles`, etc. on the RAW y scale.
#' @keywords internal
apply_ensemble_predict_regressor <- function(net, col_emb,
                                              X_train, X_test, y_train,
                                              configs, borders,
                                              quantiles = c(0.1, 0.5, 0.9),
                                              device = "cpu",
                                              softmax_temperature = 1,
                                              trace_dir = NULL,
                                              categorical_features = integer(),
                                              kv_caches = NULL,
                                              save_peak_memory_factor = NULL,
                                              row_chunk_size = NA_integer_,
                                              col_chunk_size = NA_integer_,
                                              pipelines = NULL) {
  X_tr <- as.matrix(X_train); X_te <- as.matrix(X_test)
  storage.mode(X_tr) <- "double"; storage.mode(X_te) <- "double"
  pipelines <- pipelines %||%
    member_pipeline_store(X_tr, configs, categorical_features)
  y_raw <- as.numeric(y_train)
  y_mean <- mean(y_raw)
  y_std  <- sqrt(mean((y_raw - y_mean) ^ 2)) + 1e-20   # population std + eps
  y_z    <- (y_raw - y_mean) / y_std

  n_bins <- as.integer(borders$size(1)) - 1L
  acc_probs <- NULL

  for (i in seq_along(configs)) {
    cfg <- configs[[i]]
    # Between members: this iteration's transients are dead but R's gc
    # cannot see the torch allocations holding them.
    if (i > 1L) collect_between_chunks()
    mem <- pipelines(i)

    # Per-member y: optional target_transform (yeojohnson + StandardScaler)
    tt_state <- NULL
    y_for_model <- y_z
    if (!is.null(cfg$target_transform_lambda)) {
      tt_state <- fit_target_transform_state(y_z, cfg$target_transform_lambda[1])
      y_for_model <- apply_target_transform(y_z, tt_state)
    }

    # With a cache the network never looks at the training rows, so
    # transforming and uploading them is work whose whole result is
    # discarded -- which is the cost `kv_cache = TRUE` exists to avoid.
    # The trace hook is the one consumer that still wants them.
    cache <- if (is.null(kv_caches)) NULL else kv_caches[[i]]
    want_train <- is.null(cache) || !is.null(trace_dir)
    X_tr_t <- if (want_train) {
      torch::torch_tensor(mem$X_train, dtype = torch::torch_float(),
                          device = device)$unsqueeze(1L)
    }
    X_te_t <- torch::torch_tensor(transform_member_pipeline(X_te, mem),
                                   dtype = torch::torch_float(),
                                   device = device)$unsqueeze(1L)
    y_tr_t <- if (want_train) {
      torch::torch_tensor(y_for_model, dtype = torch::torch_float(),
                          device = device)$unsqueeze(1L)
    }
    out <- torch::with_no_grad({
      tabpfn_forward(net, if (is.null(cache)) X_tr_t else NULL,
                     if (is.null(cache)) y_tr_t else NULL, X_te_t, col_emb,
                     kv_cache = cache,
                     save_peak_memory_factor = save_peak_memory_factor,
                     row_chunk_size = row_chunk_size,
                     col_chunk_size = col_chunk_size)
    })
    logits <- out$logits[1, , ]                                 # (n_test, n_bins)
    # The reference divides the raw decoder output by the temperature
    # before anything else touches it -- before the border translation,
    # not just before the softmax.
    if (softmax_temperature != 1) logits <- logits / softmax_temperature
    trace_member(trace_dir, i, X_tr_t$squeeze(1L), X_te_t$squeeze(1L),
                 y_tr_t$squeeze(1L), out$logits$squeeze(1L))

    # Translate this member's bar distribution onto the shared z-y border
    # grid. A member without a target transform is not a special case that
    # can skip the round trip -- see this function's note -- it simply
    # translates from the destination grid to itself.
    logits_c <- logits
    borders_t <- borders
    if (!is.null(tt_state)) {
      tb <- inverse_target_transform_borders(borders, tt_state)
      borders_t <- torch::torch_tensor(
        tb$borders_t, dtype = torch::torch_float(), device = logits$device
      )
      if (!is.null(tb$logit_cancel_mask)) {
        mask_t <- torch::torch_tensor(
          tb$logit_cancel_mask, dtype = torch::torch_bool(),
          device = logits$device
        )
        logits_c <- torch::torch_where(mask_t, torch::torch_full_like(logits, -Inf), logits)
      }
    }
    probs <- translate_probs_across_borders_r(logits_c, frm = borders_t, to = borders)
    acc_probs <- if (is.null(acc_probs)) probs else acc_probs + probs
  }
  avg_probs <- acc_probs / length(configs)

  # Post-process on the RAW y scale. The reference does
  #   logits = (sum_members probs) / n ; logits = logits.log()
  #   criterion.mean(logits) / criterion.icdf(logits, q)
  # on a bar distribution whose borders were rescaled to raw y. Taking
  # the log and letting the head softmax it back is a round trip, but it
  # is the reference's round trip -- and the head is not a plain
  # weighted midpoint (see `bar_logits_to_mean()`'s half-normal tails),
  # so the averaged probabilities have to go through it.
  raw_borders <- borders * y_std + y_mean
  pseudo_logits <- torch::torch_log(avg_probs)

  pred_mean <- bar_logits_to_mean(pseudo_logits, raw_borders)
  q <- bar_logits_to_quantiles(pseudo_logits, raw_borders, quantiles)

  list(
    mean       = as.numeric(pred_mean$cpu()),
    quantiles  = as.matrix(q$cpu()),
    avg_probs  = as.matrix(avg_probs$cpu())
  )
}


#' Run a full ensemble prediction (classifier), averaging probs across members.
#'
#' @param net       Loaded `per_feature_transformer` (classifier head).
#' @param col_emb   Pre-generated column embedding tensor (from
#'   `load_column_embeddings()`).
#' @param X_train,X_test Numeric matrices of raw inputs.
#' @param y_train   Integer vector of 0-indexed class labels.
#' @param configs   List of ensemble-member specs, e.g. from
#'   `load_ensemble_configs_from_dump()`.
#' @param n_classes Integer number of original classes.
#' @return Numeric matrix `(n_test, n_classes)` of averaged probabilities.
#' @keywords internal
apply_ensemble_predict_classifier <- function(net, col_emb,
                                               X_train, X_test, y_train,
                                               configs, n_classes,
                                               device = "cpu",
                                               softmax_temperature = 1,
                                               trace_dir = NULL,
                                               categorical_features = integer(),
                                               kv_caches = NULL,
                                               save_peak_memory_factor = NULL,
                                               row_chunk_size = NA_integer_,
                                               col_chunk_size = NA_integer_,
                                               pipelines = NULL) {
  X_tr <- as.matrix(X_train); X_te <- as.matrix(X_test)
  storage.mode(X_tr) <- "double"; storage.mode(X_te) <- "double"
  pipelines <- pipelines %||%
    member_pipeline_store(X_tr, configs, categorical_features)
  y_int <- as.integer(y_train)
  acc <- array(0, dim = c(nrow(X_te), n_classes))

  for (i in seq_along(configs)) {
    cfg <- configs[[i]]
    if (i > 1L) collect_between_chunks()
    mem <- pipelines(i)
    y_perm <- cfg$class_perm[y_int + 1L]
    # See the regressor: with a cache the training side is discarded.
    cache <- if (is.null(kv_caches)) NULL else kv_caches[[i]]
    want_train <- is.null(cache) || !is.null(trace_dir)
    X_tr_t <- if (want_train) {
      torch::torch_tensor(mem$X_train, dtype = torch::torch_float(),
                          device = device)$unsqueeze(1L)
    }
    X_te_t <- torch::torch_tensor(transform_member_pipeline(X_te, mem),
                                   dtype = torch::torch_float(),
                                   device = device)$unsqueeze(1L)
    y_tr_t <- if (want_train) {
      torch::torch_tensor(as.numeric(y_perm), dtype = torch::torch_float(),
                          device = device)$unsqueeze(1L)
    }
    out <- torch::with_no_grad({
      tabpfn_forward(net, if (is.null(cache)) X_tr_t else NULL,
                     if (is.null(cache)) y_tr_t else NULL, X_te_t, col_emb,
                     kv_cache = cache,
                     save_peak_memory_factor = save_peak_memory_factor,
                     row_chunk_size = row_chunk_size,
                     col_chunk_size = col_chunk_size)
    })
    logits <- out$logits[1, , 1:n_classes]
    if (softmax_temperature != 1) logits <- logits / softmax_temperature
    trace_member(trace_dir, i, X_tr_t$squeeze(1L), X_te_t$squeeze(1L),
                 y_tr_t$squeeze(1L), out$logits$squeeze(1L))
    probs  <- as.matrix(torch::nnf_softmax(logits, dim = -1L)$cpu())
    acc <- acc + apply_class_permutation_inverse(probs, cfg$class_perm + 1L)
  }
  acc / length(configs)
}


#' Mirror the Python reference's per-member trace layout
#'
#' Writes `member_NN/{X_train,X_test,y_train,logits}.safetensors` under
#' `dir`, using the same 0-padded, 0-based member numbering the Python
#' harness uses, so the two trees can be diffed file-for-file. No-op when
#' `dir` is `NULL`.
#' @keywords internal
trace_member <- function(dir, i, X_train, X_test, y_train, logits) {
  if (is.null(dir)) return(invisible())
  require_suggested("safetensors")
  d <- file.path(dir, sprintf("member_%02d", as.integer(i) - 1L))
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  save1 <- function(t, nm) {
    safetensors::safe_save_file(
      list(t = t$detach()$to(dtype = torch::torch_float())$contiguous()$cpu()),
      file.path(d, paste0(nm, ".safetensors"))
    )
  }
  save1(X_train, "X_train"); save1(X_test, "X_test")
  save1(y_train, "y_train"); save1(logits,  "logits")
  invisible()
}


