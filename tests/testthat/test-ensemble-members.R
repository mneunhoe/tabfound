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


# ---------------------------------------------------------------------------
# The other backends: TabICL, TabFM, Mitra, Kumo
# ---------------------------------------------------------------------------
#
# Their members are generated from the backend's own settings, so the
# hooks are `member_weights =` and `predict_members()` alone. Randomly
# initialised networks answer every question here: the arithmetic that
# combines members does not depend on the weights being trained.

softmax_rows_ref <- function(l) {
  e <- exp(l - apply(l, 1L, max))
  e / rowSums(e)
}

# The model's own combine, redone from `predict_members()`, so a test can
# check that the members are the pieces the prediction is made of.
combine_members <- function(m, w = NULL) {
  n <- length(m$member_index)
  w <- if (is.null(w)) rep(1 / n, n) else w / sum(w)
  wsum <- function(xs) Reduce(`+`, Map(`*`, xs, w))
  stat <- function(g) quantile_grid_stat(g, m$quantile_levels, "mean")
  switch(m$combine,
    mean_of_softmax = wsum(lapply(m$logits, function(l)
      softmax_rows_ref(l / m$softmax_temperature))),
    softmax_of_mean_logits = softmax_rows_ref(wsum(m$logits) / m$softmax_temperature),
    mean_of_grid_stats = wsum(lapply(m$grids, stat)),
    trimmed_mean_grid = {
      a <- simplify2array(m$grids)                    # n x levels x members
      cut <- floor(0.2 * n)
      stat(apply(a, c(1L, 2L), function(v) mean(sort(v)[(cut + 1L):(n - cut)])))
    },
    mean_grid = stat(wsum(m$grids)),
    mean = wsum(m$mean))
}

# One entry per backend and head: a predictor from a tiny network, and
# what its prediction is compared on.
backend_cases <- function() {
  list(
    tabicl_clf = list(make = function(...) tabicl_classifier(
      fake_ctx(tiny_tabicl()), n_estimators = 4L, predict_chunk_size = 9L, ...),
      task = "classification"),
    tabicl_reg = list(make = function(...) tabicl_regressor(
      fake_ctx(tiny_tabicl("regressor"), config = list(num_quantiles = 9L)),
      n_estimators = 4L, predict_chunk_size = 9L, ...),
      task = "regression"),
    tabfm_clf = list(make = function(...) tabfm_classifier(
      fake_ctx(tiny_tabfm()), n_estimators = 4L, predict_chunk_size = 9L, ...),
      task = "classification"),
    tabfm_reg = list(make = function(...) tabfm_regressor(
      fake_ctx(tiny_tabfm(FALSE)), n_estimators = 4L, predict_chunk_size = 9L, ...),
      task = "regression"),
    mitra_clf = list(make = function(...) mitra_classifier(
      fake_ctx(tiny_mitra(n_layers = 2L)), n_estimators = 4L,
      predict_chunk_size = 9L, ...),
      task = "classification"),
    mitra_reg = list(make = function(...) mitra_regressor(
      fake_ctx(tiny_mitra("REGRESSION", n_layers = 2L)), n_estimators = 4L,
      predict_chunk_size = 9L, ...),
      task = "regression"),
    kumo_clf = list(make = function(...) kumo_classifier(
      fake_ctx(tiny_kumo(), config = list(num_classes = 10L)),
      n_estimators = 5L, predict_chunk_size = 9L, ...),
      task = "classification"),
    # Five members, so the trimmed mean drops one from each tail.
    kumo_reg = list(make = function(...) kumo_regressor(
      fake_ctx(tiny_kumo("regressor"), config = list(num_quantiles = 9L)),
      n_estimators = 5L, predict_chunk_size = 9L, ...),
      task = "regression")
  )
}

backend_data <- function(task, seed = 51L) {
  set.seed(seed)
  X <- matrix(stats::rnorm(30L * 4L), ncol = 4L)
  y <- if (task == "classification") {
    factor(rep(c("a", "b", "c"), length.out = 30L))
  } else {
    as.numeric(X[, 1] * 2 + stats::rnorm(30L, sd = 0.3))
  }
  list(X = X, y = y, X_new = matrix(stats::rnorm(20L * 4L), ncol = 4L))
}

.pred <- function(spec, st, X, task) {
  if (task == "classification") spec$predict(st, X, type = "prob") else
    spec$predict(st, X)
}

predict_members_spec <- function(spec, d) {
  spec$predict_members(spec$fit(d$X, d$y), d$X_new)
}

test_that("every backend: unit weights leave the prediction unchanged", {
  for (nm in names(backend_cases())) {
    case <- backend_cases()[[nm]]
    d <- backend_data(case$task)
    plain <- case$make()
    n <- length(predict_members_spec(plain, d)$member_index)
    unit <- case$make(member_weights = rep(1, n))
    # Kumo's regressor is the exception: with weights its trimmed mean
    # becomes a plain one, so unit weights change it by design.
    if (nm == "kumo_reg") next
    a <- .pred(unit, unit$fit(d$X, d$y), d$X_new, case$task)
    b <- .pred(plain, plain$fit(d$X, d$y), d$X_new, case$task)
    # Kumo combines on the device, in float32: a weighted sum is not the
    # same reduction as `mean()`, so unit weights agree to rounding only.
    # Without weights its combine is the unchanged `mean()`. Every other
    # backend scales by exactly 1 and divides by the same count.
    if (nm == "kumo_clf") expect_equal(a, b, tolerance = 1e-6, label = nm) else
      expect_identical(a, b, label = nm)
  }
})

test_that("every backend: predict_members() reproduces predict()", {
  for (nm in names(backend_cases())) {
    case <- backend_cases()[[nm]]
    d <- backend_data(case$task)
    spec <- case$make()
    st <- spec$fit(d$X, d$y)
    m <- spec$predict_members(st, d$X_new)
    expect_null(m$member_weights, label = nm)
    by_hand <- combine_members(m)
    expect_equal(unname(as.matrix(by_hand)),
                 unname(as.matrix(.pred(spec, st, d$X_new, case$task))),
                 tolerance = 1e-5, label = nm)
    if (case$task == "classification") {
      expect_identical(colnames(m$logits[[1]]), levels(d$y), label = nm)
    }
  }
})

test_that("every backend: weights give the weighted combine, and zero-weight members are not run", {
  for (nm in names(backend_cases())) {
    case <- backend_cases()[[nm]]
    d <- backend_data(case$task)
    plain <- case$make()
    m <- plain$predict_members(plain$fit(d$X, d$y), d$X_new)
    n <- length(m$member_index)
    w <- c(3, 0, 1, rep(2, n - 3L))

    weighted <- case$make(member_weights = w)
    st <- weighted$fit(d$X, d$y)
    wm <- weighted$predict_members(st, d$X_new)
    expect_identical(wm$member_index, which(w > 0), label = nm)
    expect_equal(wm$member_weights, w[w > 0], label = nm)

    # The weighted model's own members are the unweighted model's, minus
    # the one weighted zero; and its prediction is their weighted combine.
    field <- intersect(c("logits", "grids", "mean"), names(m))
    expect_equal(wm[[field]], m[[field]][w > 0], tolerance = 1e-5, label = nm)
    expect_equal(unname(as.matrix(.pred(weighted, st, d$X_new, case$task))),
                 unname(as.matrix(combine_members(wm, wm$member_weights))),
                 tolerance = 1e-5, label = nm)
  }
})

test_that("every backend: member weights are checked against the ensemble size", {
  for (nm in names(backend_cases())) {
    case <- backend_cases()[[nm]]
    expect_error(case$make(member_weights = c(1, 1)), "one per member", label = nm)
    expect_error(case$make(member_weights = rep(0, 5)), "positive sum", label = nm)
  }
})

test_that("predict_members() goes through the model API for every backend", {
  # The exported generic: fitted-state checks, newdata preparation, and
  # weights normalised to sum to one.
  d <- backend_data("classification")
  ctx <- fake_ctx(tiny_tabicl())
  ctx$backend <- list(name = "tabicl"); ctx$model_ref <- "tiny"
  spec <- tabicl_classifier(ctx, n_estimators = 3L, member_weights = c(2, 1, 1))
  model <- .new_tabfound_model(ctx, spec, "classification", list())
  fitted <- fit(model, d$X, d$y)
  m <- predict_members(fitted, d$X_new)
  expect_equal(m$member_weights, c(0.5, 0.25, 0.25))
  expect_length(m$logits, 3L)
  expect_identical(m$combine, "softmax_of_mean_logits")
})
