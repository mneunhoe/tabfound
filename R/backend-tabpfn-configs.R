# R-side wrapper around `inst/python/generate_configs.py`.
#
# Config generation for ensembling requires reproducing numpy's PCG64
# RNG sequence exactly — a multi-day port. Instead we call a thin Python
# helper once per (dataset, n_estimators, random_state) tuple. Inference
# itself is still pure-R (no reticulate at predict time).

#' Generate ensemble configs matching Python's TabPFN
#'
#' Invokes the shipped Python helper (`inst/python/generate_configs.py`)
#' which uses the installed `tabpfn` package to produce the exact
#' per-member specs (shuffle permutations, class permutations, target
#' transform lambdas). The output directory is consumable by
#' [tabular_classifier()] / [tabular_regressor()]'s
#' `ensemble_configs_dir` argument.
#'
#' @param ckpt Path to the original `.ckpt` file (not the converted
#'   safetensors).
#' @param X_train Numeric matrix `(n, p)`.
#' @param y_train Integer/factor labels (classifier) or numeric target
#'   (regressor).
#' @param n_estimators Integer ensemble size.
#' @param head `"classifier"` or `"regressor"`.
#' @param random_state Integer seed. Default 0.
#' @param output_dir Directory to write configs to.
#' @param python_bin Path to the Python executable. Must have the
#'   `tabpfn` package installed (as used for `convert_ckpt.py`).
#' @param no_target_transform Regressor only: pass `TRUE` to disable
#'   target-transform members (still matches Python's output when
#'   `inference_config={"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,)}`).
#' @return `output_dir` invisibly.
#' @export
generate_ensemble_configs <- function(ckpt,
                                       X_train, y_train,
                                       n_estimators,
                                       head = c("classifier", "regressor"),
                                       random_state = 0L,
                                       output_dir,
                                       python_bin,
                                       no_target_transform = FALSE) {
  head <- match.arg(head)
  require_suggested("safetensors")
  stopifnot(file.exists(ckpt))
  stopifnot(!missing(python_bin) && file.exists(python_bin))

  # Stash inputs as a safetensors split file for the Python helper.
  tmp_split <- tempfile(fileext = ".safetensors")
  on.exit(unlink(tmp_split), add = TRUE)

  X_arr <- as.matrix(X_train); storage.mode(X_arr) <- "double"
  if (head == "classifier") {
    if (is.factor(y_train)) {
      y_num <- as.integer(y_train) - 1L
    } else if (is.numeric(y_train)) {
      y_num <- as.integer(y_train)
    } else {
      y_num <- match(y_train, sort(unique(y_train))) - 1L
    }
    y_t <- torch::torch_tensor(as.numeric(y_num), dtype = torch::torch_float())
  } else {
    y_t <- torch::torch_tensor(as.numeric(y_train), dtype = torch::torch_float())
  }
  safetensors::safe_save_file(
    list(
      x_train = torch::torch_tensor(X_arr, dtype = torch::torch_float()),
      y_train = y_t
    ),
    tmp_split
  )

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  script_path <- tabfound_file("python", "tabpfn_generate_configs.py")
  if (!nzchar(script_path)) {
    cli::cli_abort("Cannot locate {.file tabpfn_generate_configs.py}.")
  }

  cmd_args <- c(
    shQuote(script_path),
    "--ckpt", shQuote(normalizePath(ckpt)),
    "--split", shQuote(normalizePath(tmp_split)),
    "--head", head,
    "--n", as.integer(n_estimators),
    "--seed", as.integer(random_state),
    "--out", shQuote(normalizePath(output_dir, mustWork = FALSE))
  )
  if (isTRUE(no_target_transform)) cmd_args <- c(cmd_args, "--no-target-transform")

  status <- system2(python_bin, args = cmd_args,
                    stdout = TRUE, stderr = TRUE)
  attr_status <- attr(status, "status")
  if (!is.null(attr_status) && attr_status != 0L) {
    cli::cli_abort(c(
      "generate_configs.py failed (exit {attr_status}):",
      set_names(utils::tail(status, 10), rep(" ", min(10, length(status))))
    ))
  }
  for (line in status) message(line)
  invisible(output_dir)
}
# Pure-R ensemble-config generation.
#
# Unlike `generate_ensemble_configs()` (which shells out to Python to
# match the Python PCG64 RNG bit-for-bit), this produces a VALID
# ensemble using R's native RNG -- good accuracy bump, no Python
# dependency. Output directory layout is the same, so
# `tabular_classifier(ensemble_configs_dir = ...)` works unchanged.
#
# Output is not bit-identical to Python's TabPFNClassifier / Regressor
# at a given seed. For verified Python parity, use
# `generate_ensemble_configs()` instead.


#' Fit yeo-johnson lambda via MLE over a bounded interval.
#'
#' Maximum-likelihood estimate following scipy's `_yeojohnson_normmax`,
#' using R's `optimize()`. The optimum won't be bit-identical to
#' scipy's brent-based result but is numerically equivalent within
#' ~1e-6 for well-behaved data.
#'
#' @param y Numeric vector (typically z-standardized).
#' @param interval Optimization bounds. Defaults to the `(-2, 2)` range
#'   used by sklearn's `PowerTransformer`.
#' @return Fitted lambda.
#' @keywords internal
fit_yeojohnson_lambda <- function(y, interval = c(-2, 2)) {
  y <- as.numeric(y)
  n <- length(y)
  neg_ll <- function(lambda) {
    yt <- yeojohnson_forward(y, lambda)
    v  <- mean((yt - mean(yt)) ^ 2)     # population variance (ddof=0)
    if (!is.finite(v) || v < .Machine$double.eps) return(Inf)
    ll <- -n / 2 * log(v) +
      (lambda - 1) * sum(sign(y) * log1p(abs(y)))
    -ll
  }
  optimize(neg_ll, interval = interval)$minimum
}




# ---------------------------------------------------------------------------
# Pure-R ensemble-config generation
# ---------------------------------------------------------------------------

# The per-version member menus, as the checkpoints' own `inference_config`
# blocks define them. A member is fully described by its preprocessing
# config; the ensemble is that menu cycled to `n_estimators`, which is
# what `generate_*_ensemble_configs` does on the reference side.
# @keywords internal
.native_presets <- function(variant, head) {
  # `categorical_name` is as load-bearing as the primary transform: it
  # decides both whether the categorical columns go through that transform
  # and whether they are ordinal-encoded afterwards.
  v25_squash <- list(preset = "squashing_scaler_default", append_original = FALSE,
                     max_features_per_estimator = 500L,
                     categorical_name = "ordinal_very_common_categories_shuffled",
                     global_transformer_name = "svd_quarter_components",
                     polynomial_features = "no")
  v26_plain <- list(preset = "quantile_uni", append_original = FALSE,
                    max_features_per_estimator = 680L,
                    categorical_name = "numeric",
                    global_transformer_name = NULL)
  v26_svd <- list(preset = "quantile_uni", append_original = "auto",
                  max_features_per_estimator = 500L,
                  categorical_name = "ordinal_very_common_categories_shuffled",
                  global_transformer_name = "svd_quarter_components")

  if (identical(variant, "v2.5")) {
    return(switch(
      head,
      classifier = list(v25_squash,
                        list(preset = "none", append_original = FALSE,
                             max_features_per_estimator = 500L,
                             categorical_name = "numeric",
                             global_transformer_name = NULL,
                             polynomial_features = "no")),
      regressor  = list(list(preset = "quantile_uni_coarse",
                             append_original = "auto",
                             max_features_per_estimator = 500L,
                             categorical_name = "numeric",
                             global_transformer_name = NULL,
                             polynomial_features = "no"),
                        v25_squash)
    ))
  }
  # v2.6. The classifier runs no polynomial features; the regressor adds up
  # to 10 pairwise products before anything else.
  poly <- if (identical(head, "classifier")) "no" else 10L
  # The classifier's second member fixes `append_original` off, the
  # regressor's leaves it on "auto" -- a difference between the two
  # checkpoints' inference configs, not an oversight.
  if (identical(head, "classifier")) v26_svd$append_original <- FALSE
  lapply(list(v26_plain, v26_svd), function(p) { p$polynomial_features <- poly; p })
}


# The reference's factor-pair draw, on R's RNG: sample a first factor with
# replacement, then a second one at or after it that this first factor has
# not already been paired with. Not bit-identical to NumPy's -- nothing
# here is -- but the same construction, so the pairs are distinct and
# cover the upper triangle including the squares.
# @keywords internal
.draw_poly_factors <- function(n_features, max_features) {
  n_poly <- n_polynomial_features(n_features, max_features)
  if (n_poly == 0L) return(list(f1 = integer(), f2 = integer()))
  pick <- function(v) v[sample.int(length(v), 1L)]

  f1 <- sample.int(n_features, n_poly, replace = TRUE)
  f2 <- rep(NA_integer_, n_poly)
  for (i in seq_len(n_poly)) {
    # Bounded: every retry moves to a first factor that still has a free
    # partner, and `n_poly` never exceeds the number of pairs available.
    repeat {
      used <- f2[!is.na(f2) & f1 == f1[i]]
      avail <- setdiff(f1[i]:n_features, used)
      if (length(avail)) { f2[i] <- pick(avail); break }
      f1[i] <- pick(seq_len(n_features))
    }
  }
  list(f1 = f1, f2 = f2)
}


#' Generate ensemble configs in pure R (no Python required).
#'
#' Produces configs in the same directory layout as
#' [generate_ensemble_configs()] but uses R's base RNG for the per-member
#' draws. Output is NOT bit-identical to Python's `TabPFNClassifier` /
#' `TabPFNRegressor` at a given seed, but:
#'
#' * ensemble members cycle over the preset list the requested version's
#'   checkpoint actually ships -- `squashing_scaler_default` / `none` for
#'   TabPFN v2.5's classifier, two flavours of `quantile_uni` for v2.6;
#' * each member receives a unique shuffle permutation and (classifier)
#'   class permutation derived from `random_state`;
#' * v2.6 regressor members get their polynomial factor pairs drawn the
#'   way the reference draws them;
#' * half of the v2.5 regressor members can optionally carry a
#'   Yeo-Johnson target transform fitted via [fit_yeojohnson_lambda()].
#'   v2.6 has none: its checkpoint sets the target transform to the
#'   identity.
#'
#' In practice this gives the bulk of the ensembling accuracy benefit
#' while keeping the inference path fully pure-R.
#'
#' The member width is measured by running the member's own pipeline
#' rather than predicted from a formula, so a permutation can never come
#' out the wrong length as the pipeline grows.
#'
#' @param X_train Numeric matrix `(n, p)`. Used to size the per-member
#'   column permutations and to fit the y target transform.
#' @param y_train Integer/factor labels (classifier) or numeric target
#'   (regressor).
#' @param n_estimators Integer ensemble size. Typical values: 4, 8.
#' @param head `"classifier"` or `"regressor"`.
#' @param categorical_features Integer vector of 1-based column indices to
#'   declare categorical, or `NULL` to infer. Must match what the model
#'   will be given at `fit()` time -- the configs record a code
#'   permutation per encoded column, and those are sized from this.
#' @param variant Which TabPFN generation's member menu to use: `"v2.5"`
#'   (the default, for the `tabpfn` backend) or `"v2.6"` (for
#'   `tabpfn26`). Passing the wrong one produces a working but
#'   off-distribution ensemble, so match it to the weights.
#' @param random_state Integer seed. Default 0.
#' @param output_dir Directory to write configs to (created if missing).
#' @param add_target_transform Regressor only, and ignored for `"v2.6"`.
#'   If `TRUE` (default), every second v2.5 member includes a
#'   Yeo-Johnson target transform with MLE-fitted lambda.
#' @return `output_dir`, invisibly.
#' @examples
#' \dontrun{
#' d <- tempfile()
#' generate_ensemble_configs_native(X, y, n_estimators = 4,
#'                                  head = "classifier", variant = "v2.6",
#'                                  output_dir = d)
#' clf <- tabular_classifier("tabpfn-v2.6-classifier", ensemble_configs_dir = d)
#' }
#' @export
generate_ensemble_configs_native <- function(X_train, y_train,
                                             n_estimators = 4L,
                                             head = c("classifier", "regressor"),
                                             categorical_features = NULL,
                                             variant = c("v2.5", "v2.6"),
                                             random_state = 0L,
                                             output_dir,
                                             add_target_transform = TRUE) {
  head <- match.arg(head)
  variant <- match.arg(variant)
  require_suggested("safetensors")
  require_suggested("jsonlite")

  X_mat <- as.matrix(X_train); storage.mode(X_mat) <- "double"
  n_features <- ncol(X_mat)
  presets <- .native_presets(variant, head)
  cat_ix <- detect_categorical_features(X_mat, categorical_features)

  n_classes <- NULL
  if (head == "classifier") {
    n_classes <- if (is.factor(y_train)) nlevels(y_train) else length(unique(y_train))
  }

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  cfg_list <- vector("list", n_estimators)

  for (i in seq_len(n_estimators)) {
    spec <- presets[[((i - 1L) %% length(presets)) + 1L]]

    # A stable per-member seed. The multiplicative stride keeps nearby
    # `random_state` values from producing near-identical ensembles.
    seed_i <- as.integer(random_state) * 7919L + as.integer(i) * 97L
    set.seed(seed_i)

    cfg_i <- c(spec, list(add_fingerprint = TRUE, shuffle_perm = NULL))
    if (!identical(spec$polynomial_features, "no")) {
      pf <- .draw_poly_factors(n_features, spec$polynomial_features)
      cfg_i$poly_factor_1 <- pf$f1
      cfg_i$poly_factor_2 <- pf$f2
    }

    # Run the member's own pipeline to learn two things a formula cannot
    # tell us: how wide it ends up, and -- once the categorical columns
    # have been through the primary transform and the encoder's own
    # selection filter -- how many categories each encoded column has.
    sized <- apply_member_pipeline(X_mat, X_mat[1L, , drop = FALSE], cfg_i,
                                   categorical_features = cat_ix,
                                   draw_missing = TRUE)
    n_post <- ncol(sized$X_train)
    cfg_i$cat_mappings <- sized$cat_mappings
    shuffle_perm_0idx <- sample.int(n_post) - 1L

    member_dir <- file.path(output_dir, sprintf("member_%02d", i - 1L))
    dir.create(member_dir, showWarnings = FALSE, recursive = TRUE)
    safetensors::safe_save_file(
      list(t = torch::torch_tensor(as.integer(shuffle_perm_0idx),
                                   dtype = torch::torch_long())),
      file.path(member_dir, "shuffle_perm.safetensors")
    )
    if (length(cfg_i$cat_mappings %||% list())) {
      safetensors::safe_save_file(
        stats::setNames(
          lapply(cfg_i$cat_mappings, function(m)
            torch::torch_tensor(as.integer(m), dtype = torch::torch_long())),
          sprintf("col_%02d", seq_along(cfg_i$cat_mappings) - 1L)
        ),
        file.path(member_dir, "cat_mappings.safetensors")
      )
    }
    if (!is.null(cfg_i$poly_factor_1)) {
      safetensors::safe_save_file(
        list(factor_1 = torch::torch_tensor(cfg_i$poly_factor_1 - 1L,
                                            dtype = torch::torch_long()),
             factor_2 = torch::torch_tensor(cfg_i$poly_factor_2 - 1L,
                                            dtype = torch::torch_long())),
        file.path(member_dir, "poly_factors.safetensors")
      )
    }

    out_i <- list(
      preprocess_config = list(
        name                       = spec$preset,
        categorical_name           = spec$categorical_name,
        append_original            = spec$append_original,
        max_features_per_estimator = spec$max_features_per_estimator,
        global_transformer_name    = spec$global_transformer_name,
        max_onehot_cardinality     = NULL,
        differentiable             = FALSE
      ),
      add_fingerprint_feature = TRUE,
      polynomial_features     = spec$polynomial_features,
      feature_shift_count     = seed_i %% 1000000L,
      feature_shift_decoder   = "shuffle",
      subsample_ix            = NULL,
      outlier_removal_std     = NULL,
      `_model_index`          = 0L
    )

    if (head == "classifier") {
      class_perm_0idx <- sample.int(n_classes) - 1L
      out_i$class_permutation <- list(
        `__ndarray__` = TRUE, dtype = "int64",
        shape = list(as.integer(n_classes)),
        vals  = as.integer(class_perm_0idx)
      )
    }

    if (head == "regressor" && identical(variant, "v2.5") &&
        isTRUE(add_target_transform) && (i %% 2L == 0L)) {
      y_raw  <- as.numeric(y_train)
      y_mean <- mean(y_raw)
      y_std  <- sqrt(mean((y_raw - y_mean) ^ 2))
      lambda <- fit_yeojohnson_lambda((y_raw - y_mean) / y_std)
      safetensors::safe_save_file(
        list(t = torch::torch_tensor(lambda, dtype = torch::torch_double())),
        file.path(member_dir, "target_transform_lambdas.safetensors")
      )
      out_i$target_transform <- sprintf("yeojohnson_lambda=%.6f", lambda)
    }

    cfg_list[[i]] <- out_i
  }

  jsonlite::write_json(
    cfg_list, file.path(output_dir, "ensemble_configs.json"),
    auto_unbox = TRUE, pretty = TRUE, null = "null"
  )
  invisible(output_dir)
}
