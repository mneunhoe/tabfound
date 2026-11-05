# The hooks an add-on needs to reweight or select a TabPFN ensemble after
# the fact: configs in memory, member weights, and the members' own
# outputs before they are combined.
#
# The contract that matters is that none of it moves the default: an
# unweighted ensemble built from a list predicts what the same ensemble
# built from a directory predicts, bit for bit. Everything else is
# checked against that -- a weighted combine against hand-weighting
# `predict_members()`, and `predict_members()` against `predict()`.

skip_if_not_installed("torch")


# ---------------------------------------------------------------------------
# Without weights
# ---------------------------------------------------------------------------

test_that("an ensemble comes from a directory or a list, not both", {
  expect_error(.resolve_ensemble("some/dir", list(list()), NULL),
               "not both")
  expect_error(.resolve_ensemble(NULL, list(), NULL), "non-empty list")
  expect_identical(.resolve_ensemble(NULL, NULL, NULL),
                   list(configs = NULL, weights = NULL))
})

test_that("member weights are checked against the ensemble", {
  cfgs <- list(list(a = 1), list(a = 2), list(a = 3))
  expect_error(.resolve_ensemble(NULL, NULL, 1), "needs an ensemble")
  expect_error(.resolve_ensemble(NULL, cfgs, c(1, 1)), "one per member")
  expect_error(.resolve_ensemble(NULL, cfgs, c(1, -1, 1)), "non-negative")
  expect_error(.resolve_ensemble(NULL, cfgs, c(1, NA, 1)), "finite")
  expect_error(.resolve_ensemble(NULL, cfgs, c(0, 0, 0)), "positive sum")
})

test_that("a member weighted zero is dropped before anything is built", {
  cfgs <- list(list(a = 1), list(a = 2), list(a = 3))
  r <- .resolve_ensemble(NULL, cfgs, c(2, 0, 1))
  expect_identical(r$configs, cfgs[c(1L, 3L)])
  expect_identical(r$weights, c(2, 1))
})

test_that("the checkpoint's member menu is exported, and a custom one is checked", {
  menu <- native_ensemble_presets("v2.6", "regressor")
  expect_identical(menu, .native_presets("v2.6", "regressor"))
  for (p in menu) expect_true(all(.preset_fields %in% names(p)))

  bad <- menu
  bad[[2]]$categorical_name <- NULL
  expect_error(.check_presets(bad), "categorical_name")
  expect_error(.check_presets(list()), "non-empty")
})

test_that("configs in memory are the configs on disk, read back", {
  skip_if_not_installed("safetensors")
  skip_if_not_installed("jsonlite")
  set.seed(1)
  X <- cbind(matrix(rnorm(80 * 3), ncol = 3), sample(0:2, 80, TRUE))
  y <- factor(sample(c("a", "b", "c"), 80, TRUE))

  # One directory for both runs, on purpose: v2.5's first member writes
  # categorical mappings and v2.6's does not, so a file left over from the
  # first run would be read as the second's.
  d <- withr::local_tempdir()
  for (variant in c("v2.5", "v2.6")) {
    generate_ensemble_configs_native(X, y, n_estimators = 3L, variant = variant,
                                     categorical_features = 4L,
                                     random_state = 4L, output_dir = d)
    from_disk <- load_ensemble_configs_from_dump(d)
    in_memory <- generate_ensemble_configs_native(
      X, y, n_estimators = 3L, variant = variant, categorical_features = 4L,
      random_state = 4L)
    expect_identical(in_memory, from_disk)
  }
})

test_that("a custom menu changes the members drawn", {
  skip_if_not_installed("safetensors")
  skip_if_not_installed("jsonlite")
  set.seed(2)
  X <- matrix(rnorm(60 * 4), ncol = 4)
  y <- factor(sample(c("a", "b"), 60, TRUE))

  menu <- native_ensemble_presets("v2.6", "classifier")
  menu[[1]]$preset <- "none"
  menu[[1]]$global_transformer_name <- "svd_quarter_components"
  cfg <- generate_ensemble_configs_native(X, y, n_estimators = 2L,
                                          variant = "v2.6", presets = menu)
  expect_identical(cfg[[1]]$preset, "none")
  expect_identical(cfg[[1]]$global_transformer_name, "svd_quarter_components")
  expect_identical(cfg[[2]]$preset, menu[[2]]$preset)
})


# ---------------------------------------------------------------------------
# Against real weights
# ---------------------------------------------------------------------------

members_data <- function(task, seed = 11L) {
  set.seed(seed)
  n_tr <- 120L; n_te <- 40L; p <- 4L
  X <- matrix(rnorm((n_tr + n_te) * p), ncol = p)
  eta <- as.numeric(X %*% c(1, -0.5, 0.25, 0))
  y <- if (task == "classification") {
    factor(cut(eta + rnorm(length(eta), sd = 0.3), c(-Inf, -0.5, 0.5, Inf),
               labels = c("lo", "mid", "hi")))
  } else {
    eta + rnorm(length(eta), sd = 0.3)
  }
  list(X_tr = X[seq_len(n_tr), ], y_tr = y[seq_len(n_tr)],
       X_te = X[n_tr + seq_len(n_te), ])
}

tempered_probs <- function(logits, temperature) {
  z <- logits / temperature
  e <- exp(z - apply(z, 1L, max))
  e / rowSums(e)
}

test_that("classifier: a list, unit weights and a directory predict the same, bit for bit", {
  d <- Sys.getenv("TABFOUND_TABPFN_CLF_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN_CLF_DIR not configured")
  dat <- members_data("classification")

  cfg_dir <- withr::local_tempdir()
  generate_ensemble_configs_native(dat$X_tr, dat$y_tr, n_estimators = 4L,
                                   random_state = 3L, output_dir = cfg_dir)
  cfg <- load_ensemble_configs_from_dump(cfg_dir)

  p_dir <- predict(fit(tabular_classifier(d, ensemble_configs_dir = cfg_dir),
                       dat$X_tr, dat$y_tr), dat$X_te, type = "prob")
  p_list <- predict(fit(tabular_classifier(d, ensemble_configs = cfg),
                        dat$X_tr, dat$y_tr), dat$X_te, type = "prob")
  p_unit <- predict(fit(tabular_classifier(d, ensemble_configs = cfg,
                                           member_weights = rep(1, 4)),
                        dat$X_tr, dat$y_tr), dat$X_te, type = "prob")
  expect_identical(p_list, p_dir)
  expect_identical(p_unit, p_dir)
})

test_that("classifier: predict_members() reproduces predict(), weighted or not", {
  d <- Sys.getenv("TABFOUND_TABPFN_CLF_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN_CLF_DIR not configured")
  dat <- members_data("classification")
  cfg <- generate_ensemble_configs_native(dat$X_tr, dat$y_tr, n_estimators = 4L,
                                          random_state = 3L)

  clf <- fit(tabular_classifier(d, ensemble_configs = cfg), dat$X_tr, dat$y_tr)
  m <- predict_members(clf, dat$X_te)
  expect_length(m$logits, 4L)
  expect_identical(dim(m$logits[[1]]), c(nrow(dat$X_te), 3L))
  expect_identical(colnames(m$logits[[1]]), levels(dat$y_tr))
  expect_null(m$member_weights)
  expect_identical(m$softmax_temperature, 0.9)

  member_p <- lapply(m$logits, tempered_probs, m$softmax_temperature)
  expect_equal(Reduce(`+`, member_p) / 4,
               unname(predict(clf, dat$X_te, type = "prob")),
               tolerance = 1e-5, ignore_attr = TRUE)

  # The weighted combine is the weighted mean of the same members, and a
  # zero-weight member is simply absent.
  w <- c(3, 1, 0, 2)
  wclf <- fit(tabular_classifier(d, ensemble_configs = cfg, member_weights = w),
              dat$X_tr, dat$y_tr)
  by_hand <- Reduce(`+`, Map(`*`, member_p, w)) / sum(w)
  expect_equal(unname(predict(wclf, dat$X_te, type = "prob")), by_hand,
               tolerance = 1e-5, ignore_attr = TRUE)
  wm <- predict_members(wclf, dat$X_te)
  expect_length(wm$logits, 3L)
  expect_equal(wm$member_weights, c(3, 1, 2) / 6)
  expect_equal(wm$logits, m$logits[c(1L, 2L, 4L)], tolerance = 1e-6)
})

test_that("predict_members() is the same whether or not the rows are chunked", {
  d <- Sys.getenv("TABFOUND_TABPFN_CLF_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN_CLF_DIR not configured")
  dat <- members_data("classification")
  cfg <- generate_ensemble_configs_native(dat$X_tr, dat$y_tr, n_estimators = 2L)

  whole <- predict_members(fit(tabular_classifier(d, ensemble_configs = cfg),
                               dat$X_tr, dat$y_tr), dat$X_te)
  parts <- predict_members(fit(tabular_classifier(d, ensemble_configs = cfg,
                                                  predict_chunk_size = 15L),
                               dat$X_tr, dat$y_tr), dat$X_te)
  expect_identical(lapply(parts$logits, dim), lapply(whole$logits, dim))
  expect_equal(parts$logits, whole$logits, tolerance = 1e-5)
})

test_that("predict_members() needs an ensemble", {
  d <- Sys.getenv("TABFOUND_TABPFN_CLF_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN_CLF_DIR not configured")
  dat <- members_data("classification")
  clf <- fit(tabular_classifier(d), dat$X_tr, dat$y_tr)
  expect_error(predict_members(clf, dat$X_te), "needs an ensemble")
})

test_that("regressor: unit weights change nothing, and predict_members() reproduces the mean", {
  d <- Sys.getenv("TABFOUND_TABPFN_REG_DIR", unset = "")
  skip_if(!nzchar(d) || !dir.exists(d), "TABFOUND_TABPFN_REG_DIR not configured")
  dat <- members_data("regression")
  # Four v2.5 members: two of them carry a Yeo-Johnson target transform,
  # so the border translation is exercised, not only the identity.
  cfg <- generate_ensemble_configs_native(dat$X_tr, dat$y_tr, n_estimators = 4L,
                                          head = "regressor", random_state = 5L)

  reg <- fit(tabular_regressor(d, ensemble_configs = cfg), dat$X_tr, dat$y_tr)
  unit <- fit(tabular_regressor(d, ensemble_configs = cfg,
                                member_weights = rep(1, 4)),
              dat$X_tr, dat$y_tr)
  mu <- predict(reg, dat$X_te)
  expect_identical(predict(unit, dat$X_te), mu)
  q <- c(0.1, 0.9)
  expect_identical(predict(unit, dat$X_te, type = "quantiles", quantiles = q),
                   predict(reg, dat$X_te, type = "quantiles", quantiles = q))

  m <- predict_members(reg, dat$X_te)
  expect_length(m$probs, 4L)
  expect_identical(length(m$borders), ncol(m$probs[[1]]) + 1L)
  for (p in m$probs) expect_equal(rowSums(p), rep(1, nrow(p)), tolerance = 1e-4)

  as_logits <- function(p) {
    torch::torch_log(torch::torch_tensor(p, dtype = torch::torch_float()))
  }
  borders <- torch::torch_tensor(m$borders, dtype = torch::torch_float())
  avg <- Reduce(`+`, m$probs) / 4
  expect_equal(as.numeric(bar_logits_to_mean(as_logits(avg), borders)), mu,
               tolerance = 1e-4)

  w <- c(1, 2, 0, 1)
  wreg <- fit(tabular_regressor(d, ensemble_configs = cfg, member_weights = w),
              dat$X_tr, dat$y_tr)
  wavg <- Reduce(`+`, Map(`*`, m$probs, w)) / sum(w)
  expect_equal(predict(wreg, dat$X_te),
               as.numeric(bar_logits_to_mean(as_logits(wavg), borders)),
               tolerance = 1e-4)
})
