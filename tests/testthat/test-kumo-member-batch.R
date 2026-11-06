# Kumo members stacked on the network's leading axis (issue #97): a group
# of members shares one forward pass instead of one forward per member.

skip_if_not_installed("torch")

mb_matrix <- function(n = 40L, p = 5L, seed = 61L) {
  set.seed(seed)
  X <- matrix(stats::rnorm(n * p), n, p)
  X[3, 2] <- NA
  # A small-integer column, declared categorical below, so the members'
  # flags differ with their column permutations.
  X[, 4] <- sample(0:3, n, replace = TRUE)
  X
}

mb_spec <- function(head = "classifier", n_estimators = 3L, ...) {
  ctor <- if (head == "classifier") kumo_classifier else kumo_regressor
  cfg <- if (head == "classifier") list(num_classes = 10L) else list(num_quantiles = 9L)
  do.call(ctor, list(fake_ctx(tiny_kumo(head), config = cfg),
                     n_estimators = n_estimators, predict_chunk_size = 1024L,
                     categorical_features = 4L, ...))
}

mb_cases <- function(X) {
  list(
    regressor = list(head = "regressor", y = as.numeric(X[, 1] * 2 + X[, 3]),
                     type = "mean"),
    binary = list(head = "classifier",
                  y = factor(rep(c("a", "b"), length.out = nrow(X))), type = "prob"),
    ecoc = list(head = "classifier",
                y = factor(rep(letters[1:14], length.out = nrow(X))), type = "prob")
  )
}

expect_rel_close <- function(got, want, tol = 1e-4, info = NULL) {
  got <- unlist(got); want <- unlist(want)
  expect_identical(length(got), length(want), info = info)
  scale <- max(1, max(abs(want[is.finite(want)])))
  ok <- is.finite(want)
  expect_identical(is.finite(got), ok, info = info)
  expect_lt(max(abs(got[ok] - want[ok])) / scale, tol, label = info)
}

test_that("batched members agree with one member at a time", {
  X <- mb_matrix(); X[3, 2] <- 0
  X_new <- mb_matrix(n = 11L, seed = 62L)
  for (nm in names(mb_cases(X))) {
    cs <- mb_cases(X)[[nm]]
    seq_spec <- mb_spec(cs$head, member_batch_size = 1L)
    state <- seq_spec$fit(X, cs$y)
    ref <- seq_spec$predict(state, X_new, type = cs$type)
    m_ref <- seq_spec$predict_members(state, X_new)
    for (g in 2:3) {
      bat <- mb_spec(cs$head, member_batch_size = g)
      expect_rel_close(bat$predict(state, X_new, type = cs$type), ref,
                       info = paste(nm, g))
      m_got <- bat$predict_members(state, X_new)
      expect_rel_close(m_got[[1]], m_ref[[1]], info = paste(nm, g, "members"))
    }
    if (nm == "ecoc") expect_identical(dim(ref), c(11L, 14L))
  }
})

test_that("batched members agree with row-chunked passes and mixed flags", {
  # Row chunking encodes the context into a cache inside the one forward,
  # so the group's per-member categorical flags must survive into it.
  X <- mb_matrix(n = 40L)
  X_new <- mb_matrix(n = 9L, seed = 63L)
  for (nm in c("regressor", "ecoc")) {
    cs <- mb_cases(X)[[nm]]
    knobs <- list(row_chunk_size = 16L, col_chunk_size = 2L)
    seq_spec <- do.call(mb_spec, c(list(cs$head, member_batch_size = 1L), knobs))
    bat <- do.call(mb_spec, c(list(cs$head, member_batch_size = 3L), knobs))
    state <- seq_spec$fit(X, cs$y)
    flags <- lapply(1:3, function(m) kumo_recipe_member(state$recipe, X, m)$categorical_mask)
    expect_gt(length(unique(flags)), 1L)
    expect_rel_close(bat$predict(state, X_new, type = cs$type),
                     seq_spec$predict(state, X_new, type = cs$type), info = nm)
  }
})

test_that("a group of one is today's member-at-a-time path, bit for bit", {
  X <- mb_matrix(); X_new <- mb_matrix(n = 8L, seed = 64L)
  y <- as.numeric(X[, 1])
  spec <- mb_spec("regressor", member_batch_size = 1L)
  state <- spec$fit(X, y)
  grouped <- 0L
  orig <- .kumo_group_out
  local_mocked_bindings(.kumo_group_out = function(...) {
    grouped <<- grouped + 1L
    orig(...)
  })
  got <- spec$predict(state, X_new, type = "grid")
  expect_identical(grouped, 0L)
  # Against the per-member function the loop has always called, on the
  # same (seeded) weights.
  ctx_net <- tiny_kumo("regressor")
  want <- lapply(1:3, function(m) {
    kumo_recipe_inverse(state$recipe,
                        .kumo_member_out(ctx_net, "cpu", state$recipe, state$X_train,
                                         state$y_train, X_new, m), m)
  })
  expect_identical(got, as.matrix(kumo_recipe_reduce(want, FALSE)))
  # And `.kumo_group_out()` of one member is that same call.
  one <- .kumo_group_out(ctx_net, "cpu", state$recipe, state$X_train, state$y_train,
                         X_new, 2L)
  expect_identical(as.array(one[[1]]),
                   as.array(.kumo_member_out(ctx_net, "cpu", state$recipe,
                                             state$X_train, state$y_train, X_new, 2L)))
})

test_that("a cache keeps the member-at-a-time loop whatever the group size", {
  X <- mb_matrix(); X_new <- mb_matrix(n = 8L, seed = 65L)
  y <- factor(rep(c("a", "b", "c"), length.out = nrow(X)))
  sizes <- integer()
  orig <- .kumo_group_out
  local_mocked_bindings(.kumo_group_out = function(net, dev, rec, X_train, y_train,
                                                   X_query, ms, chunks = list()) {
    sizes <<- c(sizes, length(ms))
    orig(net, dev, rec, X_train, y_train, X_query, ms, chunks)
  })
  spec <- mb_spec(kv_cache = TRUE, member_batch_size = 3L)
  state <- spec$fit(X, y)
  spec$predict(state, X_new, type = "prob")
  state$kv_caches <- spec$build_cache(state)
  spec$predict(state, X_new, type = "prob")
  expect_length(sizes, 0L)
  # Uncached, the same size groups all three.
  mb_spec(member_batch_size = 3L)$predict(state[setdiff(names(state), "kv_caches")],
                                          X_new, type = "prob")
  expect_identical(sizes, 3L)
})

test_that("members split into even groups of at most the batch size", {
  expect_identical(.kumo_member_groups(8L, 1L), as.list(1:8))
  expect_identical(.kumo_member_groups(8L, 8L), list(1:8))
  expect_identical(.kumo_member_groups(8L, 5L), list(1:4, 5:8))
  expect_identical(lengths(.kumo_member_groups(7L, 3L)), c(3L, 2L, 2L))
  expect_identical(.kumo_member_groups(3L, 10L), list(1:3))
  expect_identical(unlist(.kumo_member_groups(11L, 4L)), 1:11)
})

test_that("the group size is the largest the memory estimate calls ok", {
  ctx <- list(config = mem_config_kumo("small"), backend = get_backend("kumo"),
              device = "cpu")
  pick <- function(...) .kumo_member_batch(NULL, ctx, "regression", 1000, 1024, 10,
                                           list(n_estimators = 8L,
                                                predict_chunk_size = 1024L), 8L)
  asked <- list()
  fake <- function(limit, avail = 30e9) {
    function(model, ..., member_batch_size, available = NULL) {
      asked[[length(asked) + 1L]] <<- list(g = member_batch_size, available = available)
      list(verdict = if (member_batch_size <= limit) "ok" else "tight",
           system_avail_bytes = avail, total_peak_bytes = member_batch_size)
    }
  }
  local_mocked_bindings(estimate_peak_memory = fake(8))
  expect_identical(pick(), 8L)
  local_mocked_bindings(estimate_peak_memory = fake(3))
  expect_identical(pick(), 3L)
  local_mocked_bindings(estimate_peak_memory = fake(2))
  asked <- list()
  expect_identical(pick(), 2L)
  # 8 first, against a fresh probe; then only the sizes that change the
  # grouping (4, 3, 2), all against that one probe.
  expect_identical(vapply(asked, `[[`, numeric(1), "g"), c(8, 4, 3, 2))
  expect_null(asked[[1]]$available)
  expect_identical(asked[[2]]$available, 30e9)
  local_mocked_bindings(estimate_peak_memory = fake(1))
  expect_identical(pick(), 1L)
  # Nothing to judge by: one at a time.
  local_mocked_bindings(estimate_peak_memory = fake(8, avail = NA_real_))
  expect_identical(pick(), 1L)
  local_mocked_bindings(estimate_peak_memory = function(...) stop("no constants"))
  expect_identical(pick(), 1L)
  # Forced, and capped at the members run; one member is always one.
  expect_identical(.kumo_member_batch(4L, ctx, "regression", 1, 1, 1, list(), 8L), 4L)
  expect_identical(.kumo_member_batch(16L, ctx, "regression", 1, 1, 1, list(), 8L), 8L)
  expect_identical(.kumo_member_batch(NULL, ctx, "regression", 1, 1, 1, list(), 1L), 1L)
})

test_that("no group is stacked where one member's pass is already large", {
  ctx <- list(config = mem_config_kumo("small"), backend = get_backend("kumo"),
              device = "cpu")
  asked <- 0L
  local_mocked_bindings(estimate_peak_memory = function(...) {
    asked <<- asked + 1L
    list(verdict = "ok", system_avail_bytes = 1e12, total_peak_bytes = 1)
  })
  opts <- list(n_estimators = 8L, predict_chunk_size = 1024L)
  pick <- function(n_context, n_query = 1024, o = opts) {
    .kumo_member_batch(NULL, ctx, "classification", n_context, n_query, 10, o, 8L)
  }
  expect_identical(pick(2000), 8L)
  expect_identical(asked, 1L)
  # Context plus the resident chunk past the threshold: one at a time,
  # without asking.
  expect_identical(pick(3000), 1L)
  expect_identical(pick(2000, n_query = 1e5, o = utils::modifyList(opts,
                   list(predict_chunk_size = 4096L))), 1L)
  expect_identical(asked, 1L)
  # Sub-tasks already fill the pass: 14 classes are 8 per member.
  expect_identical(pick(200, o = c(opts, n_classes = 14L)), 1L)
  expect_identical(pick(200, o = c(opts, n_classes = 3L)), 8L)
  # Forced sizes are not held to it.
  expect_identical(.kumo_member_batch(4L, ctx, "regression", 1e5, 1024, 10, opts, 8L), 4L)
})

test_that("the real estimate picks a group that its own model calls ok", {
  ctx <- list(config = mem_config_kumo("small"), backend = get_backend("kumo"),
              device = "cpu")
  opts <- list(n_estimators = 8L, predict_chunk_size = 1024L)
  local_mocked_bindings(.system_memory = function() {
    list(total = 64e9, available = 16e9, source = "test")
  })
  est <- function(g, ...) estimate_peak_memory(ctx$config, 1000, 1024, 10,
                                               backend = "kumo", task = "regression",
                                               member_batch_size = g, n_estimators = 8L,
                                               predict_chunk_size = 1024L, ...)
  g <- .kumo_member_batch(NULL, ctx, "regression", 1000, 1024, 10, opts, 8L)
  expect_gt(g, 1L)
  expect_identical(est(g)$verdict, "ok")
  if (g < 8L) {
    nxt <- min(which(vapply(seq_len(8L), function(h) {
      max(lengths(.kumo_member_groups(8L, h)))
    }, integer(1)) > g))
    expect_false(identical(est(nxt)$verdict, "ok"))
  }
})

test_that("the Kumo estimate grows with the member batch, uncached only", {
  cfg <- mem_config_kumo("small")
  est <- function(g, ...) estimate_peak_memory(cfg, 4000, 1024, 10, backend = "kumo",
                                               task = "regression", n_estimators = 8L,
                                               member_batch_size = g,
                                               available = 64e9, ...)
  # Four members stacked are four members' live set in one pass ...
  expect_equal(max(est(4)$stage_bytes) / max(est(1)$stage_bytes), 4, tolerance = 1e-8)
  # ... and two passes rather than eight for the allocator's envelope.
  co <- .memory_coefs("kumo")
  env <- function(n) min(1 + co$ensemble_log2_factor * log2(n), co$ensemble_cap)
  expect_equal(est(4)$transient_bytes / est(1)$transient_bytes,
               4 * env(2) / env(8), tolerance = 1e-8)
  # Absent is one member at a time.
  expect_identical(
    estimate_peak_memory(cfg, 4000, 1024, 10, backend = "kumo", task = "regression",
                         n_estimators = 8L, available = 64e9)$total_peak_bytes,
    est(1)$total_peak_bytes)
  # Capped at the members run, and ignored where a cache is in use.
  expect_identical(est(16)$total_peak_bytes, est(8)$total_peak_bytes)
  expect_identical(est(4, kv_cache = TRUE)$total_peak_bytes,
                   est(1, kv_cache = TRUE)$total_peak_bytes)
  # A row-chunked pass's own context cache is the group's.
  p1 <- est(1, row_chunk_size = 512L)$persistent_bytes
  p4 <- est(4, row_chunk_size = 512L)$persistent_bytes
  expect_gt(p4, p1)
})

test_that("member_batch_size is NULL or a whole number of at least 1", {
  expect_null(.check_member_batch_size(NULL))
  expect_identical(.check_member_batch_size(4), 4L)
  for (bad in list(0L, -1L, 1.5, NA_integer_, Inf, "2", c(1L, 2L), TRUE)) {
    expect_error(mb_spec(member_batch_size = bad), "member_batch_size")
  }
  expect_error(mb_spec("regressor", member_batch_size = 0L), "member_batch_size")
})

test_that("the estimate covers every measured member batch", {
  # The sweep in inst/memory/measurements/member-batch-kumo.json: 8 members
  # on the released small and medium weights, a group of G per pass.
  path <- tabfound_file("memory", "measurements", "member-batch-kumo.json")
  skip_if(!nzchar(path), "no stored member-batch sweep")
  blob <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  est <- function(p, g) {
    size <- if (grepl("medium", p$model_id)) "medium" else "small"
    do.call(estimate_peak_memory, c(
      list(mem_config_kumo(size), p$n_context, p$n_query, p$n_features,
           backend = "kumo", task = p$task, n_estimators = p$n_estimators,
           member_batch_size = g, available = 64e9),
      if (!is.null(p$n_classes)) list(n_classes = p$n_classes)
    ))$total_peak_bytes
  }
  checked <- 0L
  for (p in blob$points) {
    if (p$member_batch_size == 1L) next
    label <- sprintf("%s n=%d G=%d calls=%d", p$model_id, p$n_context,
                     p$member_batch_size, p$predict_calls)
    if (!identical(p$status, "ok")) {
      # Stopped at the harness's cap: the estimate must at least be above
      # where it was stopped.
      expect_gt(est(p, p$member_batch_size), p$lower_bound_bytes, label = label)
      next
    }
    # Above the measured peak. Not also "grown from G = 1 by at least as
    # much": the one-member baselines are the noisiest points in the sweep
    # (4.2 GB at 2,000 context rows against 5.1 GB at 1,000), so growth
    # from them says more about the allocator's arena that run than about
    # the factor.
    expect_gt(est(p, p$member_batch_size), p$peak_bytes, label = label)
    checked <- checked + 1L
  }
  expect_gt(checked, 5L)
})
