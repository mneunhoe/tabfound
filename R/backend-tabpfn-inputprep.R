# Per-inference preprocessing pipeline.
#
# These functions replicate the *stateless* (in the state_dict sense —
# they have no learnable parameters) encoder steps from the Python
# TabPFN pipeline. They fit per-call on the training-row slice and
# transform the full (train + test) tensor.
#
# X pipeline (applied inside per_feature_transformer$forward before
# the final linear encoder step):
#   1. RemoveEmpty       — mask constant columns on train, zero them out
#   2. NaN handling      — replace NaN/Inf with per-feature train mean
#                           and build a signed-sentinel indicator tensor
#   3. FeatureTransform  — soft outlier clip + z-normalize (train stats)
#   4. NormalizeFeatureGroups — scale non-constant cols by
#                               sqrt(g / n_used_per_group)
#
# y pipeline (classifier):
#   1. MulticlassTargetEncoder — (y > sorted_unique_train_ys).sum(-1)
#   2. NaN handling            — same as above
#
# All tensors operate on `(seq_len, batch_or_batch*groups, feat)` shape
# so we can compute axis-0 stats consistently with the Python code.

# Sentinel values used by the Python NanHandlingEncoderStep to mark
# NaN / +Inf / -Inf positions in the indicator channel.
NAN_INDICATOR     <- -2.0
POS_INF_INDICATOR <-  2.0
NEG_INF_INDICATOR <-  4.0


#' Column-wise mean ignoring NaN (counts +Inf/-Inf as NaN too)
#' @keywords internal
torch_nanmean_sb <- function(x, include_inf = TRUE) {
  # x: (S, B, H).  Returns (B, H).
  mask <- torch::torch_isnan(x)
  if (include_inf) mask <- mask$logical_or(torch::torch_isinf(x))
  safe_x <- torch::torch_where(mask, torch::torch_zeros_like(x), x)
  n_valid <- (1 - mask$to(dtype = x$dtype))$sum(dim = 1L)
  n_valid <- torch::torch_clamp(n_valid, min = 1)
  safe_x$sum(dim = 1L) / n_valid
}

#' Column-wise std ignoring NaN/Inf (unbiased, like torch.std default)
#' @keywords internal
torch_nanstd_sb <- function(x) {
  mask <- torch::torch_isnan(x)$logical_or(torch::torch_isinf(x))
  safe_x <- torch::torch_where(mask, torch::torch_zeros_like(x), x)
  n_valid <- (1 - mask$to(dtype = x$dtype))$sum(dim = 1L)
  n_valid <- torch::torch_clamp(n_valid, min = 2)
  mean <- safe_x$sum(dim = 1L) / n_valid
  diff <- safe_x - mean$unsqueeze(1L)
  diff <- torch::torch_where(mask, torch::torch_zeros_like(diff), diff)
  ((diff * diff)$sum(dim = 1L) / (n_valid - 1))$sqrt()
}


# ---------------------------------------------------------------------------
# X pipeline
# ---------------------------------------------------------------------------

#' Preprocess X and produce main + nan_indicators tensors for the X encoder
#'
#' Matches the actual `nn.Sequential` of the
#' `Prior-Labs/tabpfn_2_5` default classifier ckpt:
#'
#'   0. RemoveEmpty                  (no-op for non-padded inputs with B=1)
#'   1. NaN handling                 (impute with train mean + signed indicator)
#'   2. NormalizeFeatureGroups #1    (CONFIGURED OFF in this ckpt — no-op)
#'   3. FeatureTransform             (z-normalize only — `remove_outliers=False`)
#'   4. NormalizeFeatureGroups #2    (active — scale non-const cols by sqrt(g/n_used))
#'
#' If you load a different ckpt with different flags, the
#' configuration must be threaded through here — see
#' `inst/python/dump_py_forward.py` for how to introspect them.
#'
#' Every statistic this fits comes from the training rows alone -- the
#' imputation mean, the z-normalisation mean and standard deviation, and
#' the per-group non-constant mask. Nothing here looks at the test rows,
#' which is what makes a v2.5 KV cache exactly equivalent to a full
#' forward rather than approximately so. Pass `state` to reuse them on a
#' batch that has no training rows in it.
#'
#' @param x Tensor of shape `(S, BG, g)`.
#' @param single_eval_pos Integer number of training rows.
#' @param state Fitted statistics from an earlier call, or `NULL` to fit.
#' @return A list with `$main`, `$nan_indicators` (each `(S, BG, g)`) and
#'   `$state`.
#' @keywords internal
preprocess_x_for_encoder <- function(x, single_eval_pos, state = NULL) {
  dims <- x$size()
  S <- dims[1]; BG <- dims[2]; g <- dims[3]
  using_state <- !is.null(state)

  # --- 1a: signed-sentinel indicator from original values -----------------
  nan_mask <- torch::torch_isnan(x)
  pos_inf  <- torch::torch_isinf(x)$logical_and(x > 0)
  neg_inf  <- torch::torch_isinf(x)$logical_and(x < 0)
  nan_indicators <-
    nan_mask$to(dtype = x$dtype) * NAN_INDICATOR +
    pos_inf$to(dtype  = x$dtype) * POS_INF_INDICATOR +
    neg_inf$to(dtype  = x$dtype) * NEG_INF_INDICATOR

  # --- 1b: NaN/Inf imputation with per-feature train mean -----------------
  feature_means <- if (using_state) state$feature_means else
    torch_nanmean_sb(x[1:single_eval_pos, , ], include_inf = TRUE)
  bad <- nan_mask$logical_or(torch::torch_isinf(x))
  mean_expanded <- feature_means$unsqueeze(1L)$expand(c(S, BG, g))
  x <- torch::torch_where(bad, mean_expanded, x)
  dump_if_enabled("preproc_x_after_nan_handle", x)
  dump_if_enabled("preproc_x_nan_indicators", nan_indicators)

  # Step 2 (NormalizeFeatureGroups #1) is a no-op in this ckpt.
  dump_if_enabled("preproc_x_after_norm_groups_1", x)

  # --- 3: FeatureTransform — z-normalize only (no outlier clip) -----------
  if (using_state) {
    mean <- state$mean; std <- state$std
  } else {
    mean <- torch_nanmean_sb(x[1:single_eval_pos, , ])
    std  <- torch_nanstd_sb(x[1:single_eval_pos, , ])
    std  <- torch::torch_where(std == 0, torch::torch_ones_like(std), std)
  }
  x <- (x - mean$unsqueeze(1L)) / (std$unsqueeze(1L) + 1e-16)
  x <- torch::torch_clamp(x, min = -100, max = 100)
  dump_if_enabled("preproc_x_clipped", x)        # historical name, kept for compare

  # --- 4: NormalizeFeatureGroups #2 ---------------------------------------
  ng <- .apply_normalize_feature_groups(x, single_eval_pos = single_eval_pos,
                                        state = state)
  x <- ng$x
  dump_if_enabled("preproc_x_after_norm_groups_2", x)

  list(main = x, nan_indicators = nan_indicators,
       state = list(feature_means = feature_means, mean = mean, std = std,
                    non_const = ng$non_const))
}


# Apply one NormalizeFeatureGroups step (refits its own non-constant mask).
# Scales non-constant columns by sqrt(g / n_used_per_group); zeros out
# constant columns.
# @keywords internal
.apply_normalize_feature_groups <- function(x, single_eval_pos, state = NULL) {
  dims <- x$size()
  S <- dims[1]; BG <- dims[2]; g <- dims[3]

  non_const <- if (!is.null(state)) state$non_const else {
    train <- x[1:single_eval_pos, , ]
    first_row     <- train[1, , ]$unsqueeze(1L)
    const_matches <- (train[2:single_eval_pos, , ] == first_row)$sum(dim = 1L)
    const_matches != (single_eval_pos - 1L)                       # (BG, g)
  }

  n_used <- torch::torch_clamp(
    non_const$to(dtype = x$dtype)$sum(dim = -1L, keepdim = TRUE),
    min = 1
  )
  scale <- torch::torch_sqrt(
    torch::torch_tensor(g, dtype = x$dtype, device = x$device) / n_used
  )
  x <- x * scale$unsqueeze(1L)

  keep <- non_const$unsqueeze(1L)$expand(c(S, BG, g))
  list(x = torch::torch_where(keep, x, torch::torch_zeros_like(x)),
       non_const = non_const)
}


# Soft outlier-clip used by FeatureTransformEncoderStep:
#   lower, upper = mean ± n_sigma*std  (iteratively tightened)
#   X = max(-log(1+|X|) + lower, X)
#   X = min( log(1+|X|) + upper, X)
.apply_soft_outlier_clip <- function(x, single_eval_pos, n_sigma = 4.0) {
  train <- x[1:single_eval_pos, , ]
  mask0 <- torch::torch_isnan(train)$logical_or(torch::torch_isinf(train))
  clean <- torch::torch_where(mask0, torch::torch_zeros_like(train), train)
  mean0 <- torch_nanmean_sb(train)
  std0  <- torch_nanstd_sb(train)
  cut   <- std0 * n_sigma
  lower <- mean0 - cut
  upper <- mean0 + cut

  # Tighten bounds by re-computing stats after removing initial outliers.
  clean2 <- torch::torch_where(
    (clean > upper$unsqueeze(1L))$logical_or(clean < lower$unsqueeze(1L)),
    torch::torch_full_like(clean, NaN),
    clean
  )
  mean2 <- torch_nanmean_sb(clean2)
  std2  <- torch_nanstd_sb(clean2)
  cut2  <- std2 * n_sigma
  lower <- mean2 - cut2
  upper <- mean2 + cut2

  # Apply smoothed bounds to the full tensor (train + test).
  abs_x <- x$abs()
  x <- torch::torch_maximum(-torch::torch_log1p(abs_x) + lower$unsqueeze(1L), x)
  x <- torch::torch_minimum( torch::torch_log1p(abs_x) + upper$unsqueeze(1L), x)
  x
}


# ---------------------------------------------------------------------------
# y pipeline (classifier)
# ---------------------------------------------------------------------------

#' Preprocess y for the classifier y encoder
#'
#' Pipeline order (matches Python `y_encoder.children()`):
#'   0. NaN handling — signed-sentinel indicator + impute with train mean
#'                     of the *raw* y values
#'   1. MulticlassClassificationTargetEncoder
#'        — `(y > sorted_unique_train_ys).sum(-1)`; runs on the
#'        post-NaN-imputed values, so test rows get an actual rank
#'        based on the imputed mean
#'
#' Order matters: imputing NaN with mean of raw labels (e.g. 1.01 for
#' iris) and THEN ranking gives a different test-position rank than
#' ranking first then imputing with mean-of-ranks. Reproducing the
#' Python order is required for parity.
#'
#' @param y Tensor `(B, N, 1)` — test positions should be NaN.
#' @param single_eval_pos Integer number of training rows.
#' @return `list(main, nan_indicators)`, each `(B, N, 1)`.
#' @keywords internal
preprocess_y_clf_for_encoder <- function(y, single_eval_pos) {
  dims <- y$size()
  B <- dims[1]; N <- dims[2]

  # --- Step 0a: signed-sentinel indicator from the original values --------
  nan_mask <- torch::torch_isnan(y)
  pos_inf  <- torch::torch_isinf(y)$logical_and(y > 0)
  neg_inf  <- torch::torch_isinf(y)$logical_and(y < 0)
  nan_indicators <-
    nan_mask$to(dtype = y$dtype) * NAN_INDICATOR +
    pos_inf$to(dtype  = y$dtype) * POS_INF_INDICATOR +
    neg_inf$to(dtype  = y$dtype) * NEG_INF_INDICATOR

  # --- Step 0b: impute NaN/Inf with train mean (per-batch) ----------------
  train_y <- y[, 1:single_eval_pos, , drop = FALSE]
  feature_means <- torch_nanmean_sb(train_y$permute(c(2L, 1L, 3L)))   # (B, 1)
  feature_means_exp <- feature_means$unsqueeze(1L)$expand(c(B, N, 1L))
  bad <- nan_mask$logical_or(torch::torch_isinf(y))
  y_main <- torch::torch_where(bad, feature_means_exp, y)
  dump_if_enabled("preproc_y_after_nan_handle", y_main)

  # --- Step 1: MulticlassClassificationTargetEncoder ----------------------
  # For each batch, count how many unique training y values each element
  # strictly exceeds.
  y_out <- y_main$clone()
  for (b in seq_len(B)) {
    y_train_b <- y[b, 1:single_eval_pos, 1]   # raw train labels (NaN-free)
    uniq_r <- sort(unique(as.numeric(y_train_b$cpu())))
    unique_ys <- torch::torch_tensor(uniq_r, dtype = y$dtype, device = y$device)
    y_b <- y_main[b, , 1]
    rank <- (y_b$unsqueeze(-1L) > unique_ys$unsqueeze(1L))$
      to(dtype = y$dtype)$sum(dim = -1L)
    y_out[b, , 1] <- rank
  }
  dump_if_enabled("preproc_y_after_multiclass", y_out)
  dump_if_enabled("preproc_y_nan_indicators", nan_indicators)

  list(main = y_out, nan_indicators = nan_indicators)
}


#' Preprocess y for the regressor y encoder
#'
#' Python's regressor y_encoder has only 2 steps (no multiclass encoder):
#'   0. NaN handling — signed-sentinel indicator + impute with train mean
#'   1. LinearInputEncoderStep — concat main + nan_indicators, project to emb
#'
#' This function produces the `main` and `nan_indicators` tensors to be
#' concatenated along the last dim before the final linear.
#'
#' @param y Tensor `(B, N, 1)` — test positions should be NaN.
#' @param single_eval_pos Integer number of training rows.
#' @return `list(main, nan_indicators)`, each `(B, N, 1)`.
#' @keywords internal
preprocess_y_reg_for_encoder <- function(y, single_eval_pos) {
  dims <- y$size()
  B <- dims[1]; N <- dims[2]

  nan_mask <- torch::torch_isnan(y)
  pos_inf  <- torch::torch_isinf(y)$logical_and(y > 0)
  neg_inf  <- torch::torch_isinf(y)$logical_and(y < 0)
  nan_indicators <-
    nan_mask$to(dtype = y$dtype) * NAN_INDICATOR +
    pos_inf$to(dtype  = y$dtype) * POS_INF_INDICATOR +
    neg_inf$to(dtype  = y$dtype) * NEG_INF_INDICATOR

  train_y <- y[, 1:single_eval_pos, , drop = FALSE]
  feature_means <- torch_nanmean_sb(train_y$permute(c(2L, 1L, 3L)))   # (B, 1)
  feature_means_exp <- feature_means$unsqueeze(1L)$expand(c(B, N, 1L))
  bad <- nan_mask$logical_or(torch::torch_isinf(y))
  y_out <- torch::torch_where(bad, feature_means_exp, y)

  dump_if_enabled("preproc_y_after_nan_handle", y_out)
  dump_if_enabled("preproc_y_nan_indicators", nan_indicators)

  list(main = y_out, nan_indicators = nan_indicators)
}
