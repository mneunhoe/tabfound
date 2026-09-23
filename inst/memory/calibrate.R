#!/usr/bin/env Rscript
#
# Measure what a fit/predict cycle really costs, and fit the constants
# that `estimate_peak_memory()` ships with.
#
#   Rscript inst/memory/calibrate.R tabicl tabpfn3        # measure, report
#   Rscript inst/memory/calibrate.R tabicl --write        # ...and write coefs
#   Rscript inst/memory/calibrate.R --all --grid full
#
# Measurements land in `inst/memory/measurements/<backend>.json` whether
# or not `--write` is given; the coefficient files are only touched with
# it. The measurements are the durable artefact -- like the parity
# fixtures, they are taken once and asserted thereafter, by
# `tests/testthat/test-memory-calibration.R`, which needs no weights.
#
# Requires the backend's weights to be downloaded. Points that kill their
# worker are recorded as such rather than being lost: those are the
# observations the whole exercise is about.

suppressPackageStartupMessages({
  library(tabfound)
  library(jsonlite)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

# Where this script lives, so it can find its worker. `commandArgs()`
# carries `--file=` under `Rscript`, and nothing under `sys.source()` --
# which is a supported way in, so the fallback has to be real rather than
# an error handler that never fires. (`normalizePath(NA)` warns and
# returns NA rather than throwing, so a `tryCatch` here silently produced
# a worker path of "NA/calibrate-worker.R".)
HERE <- local({
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(arg)) return(dirname(normalizePath(sub("^--file=", "", arg[1]))))
  if (dir.exists("inst/memory")) return(normalizePath("inst/memory"))
  installed <- tabfound:::tabfound_file("memory")
  if (nzchar(installed) && dir.exists(installed)) return(installed)
  normalizePath(".")
})
WORKER <- file.path(HERE, "calibrate-worker.R")


# ---------------------------------------------------------------------------
# Watching a process's memory
# ---------------------------------------------------------------------------

# Linux keeps a true high-water mark in `/proc/<pid>/status`, which is
# exact and free. macOS does not, so the only way to see a peak there is
# to look often enough to catch it -- a sampling loop, with the sampling
# error that implies. Both are reported with the method that produced
# them, because a 20 ms sampler can miss a spike and a reader of these
# numbers should know which kind they have.
have_vmhwm <- function(pid) file.exists(sprintf("/proc/%d/status", pid))

read_vmhwm <- function(pid) {
  txt <- tryCatch(readLines(sprintf("/proc/%d/status", pid), warn = FALSE),
                  error = function(e) character())
  line <- grep("^VmHWM:", txt, value = TRUE)
  if (!length(line)) return(NA_real_)
  as.numeric(gsub("[^0-9]", "", line[1])) * 1024
}

read_rss <- function(pid) {
  out <- suppressWarnings(tryCatch(
    system2("ps", c("-o", "rss=", "-p", pid), stdout = TRUE, stderr = FALSE),
    error = function(e) character()
  ))
  if (!length(out) || !nzchar(trimws(out[1]))) return(NA_real_)
  as.numeric(trimws(out[1])) * 1024
}

alive <- function(pid) {
  isTRUE(is.finite(suppressWarnings(read_rss(pid))))
}


# ---------------------------------------------------------------------------
# One grid point
# ---------------------------------------------------------------------------

# Launch the worker, watch it until it stops, and report what it cost.
# `interval` is the sampling period on platforms without VmHWM; 20 ms is
# a compromise between catching a transient and spending the machine's
# time on `ps` instead of on the model.
measure_point <- function(point, model_dir, task, opts, interval = 0.01,
                          timeout = 3600) {
  dir <- tempfile("tabfound-cal-"); dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  spec_file <- file.path(dir, "spec.json")
  spec <- list(
    model_dir = model_dir, task = task, opts = opts,
    n_context = point$n_context, n_query = point$n_query,
    n_features = point$n_features, device = "cpu", seed = 1L,
    pid_file = file.path(dir, "pid"),
    marker_file = file.path(dir, "markers"),
    result_file = file.path(dir, "result.json")
  )
  write_json(spec, spec_file, auto_unbox = TRUE)

  system2("Rscript", c(shQuote(WORKER), shQuote(spec_file)),
          stdout = file.path(dir, "out"), stderr = file.path(dir, "err"),
          wait = FALSE)

  # The worker announces itself; until it has, there is nothing to watch.
  deadline <- Sys.time() + 120
  while (!file.exists(spec$pid_file) && Sys.time() < deadline) Sys.sleep(0.05)
  if (!file.exists(spec$pid_file)) {
    return(c(point, list(status = "no_start", peak_bytes = NA_real_)))
  }
  pid <- as.integer(readLines(spec$pid_file, warn = FALSE)[1])

  vmhwm <- have_vmhwm(pid)
  peak <- 0
  loaded_rss <- NA_real_
  seen_loaded <- FALSE
  samples <- 0L
  started <- Sys.time()
  repeat {
    # One read per turn of the loop, used both to record the sample and
    # to notice the process has gone: asking twice would halve the real
    # sampling rate for nothing.
    rss <- read_rss(pid)
    if (!isTRUE(is.finite(rss))) break
    if (difftime(Sys.time(), started, units = "secs") > timeout) break
    samples <- samples + 1L
    peak <- max(peak, rss)
    # The resident size once the weights are in is the baseline every
    # activation sits on top of: R, libtorch and the checkpoint, before
    # the first activation exists.
    if (!seen_loaded && file.exists(spec$marker_file)) {
      if (any(grepl("^loaded ", readLines(spec$marker_file, warn = FALSE)))) {
        loaded_rss <- rss
        seen_loaded <- TRUE
      }
    }
    Sys.sleep(interval)
  }
  if (vmhwm) peak <- read_vmhwm(pid)

  res <- if (file.exists(spec$result_file)) {
    fromJSON(spec$result_file, simplifyVector = TRUE)
  } else NULL

  # A worker that vanished without a result did not necessarily run out
  # of memory -- it may never have got as far as trying. The difference
  # matters: a crash boundary is a finding, and a broken environment is a
  # bug, and treating the second as the first would put a fictional
  # ceiling into the constants. (Observed, not hypothetical: reinstalling
  # the package mid-run turned six points into "died".) The marker file
  # is the evidence -- it only reaches `work_start` if the weights loaded
  # and the data was built.
  markers <- if (file.exists(spec$marker_file)) {
    readLines(spec$marker_file, warn = FALSE)
  } else character()
  reached_work <- any(grepl("^work_start ", markers))
  status <- if (!is.null(res)) res$status
            else if (reached_work) "died"
            else "error"
  c(point, list(
    status = status,
    peak_bytes = if (is.finite(peak) && peak > 0) peak else NA_real_,
    loaded_bytes = loaded_rss,
    elapsed_sec = res$elapsed_sec %||% NA_real_,
    method = if (vmhwm) "vmhwm" else "rss-sampling",
    samples = samples,
    stderr = if (!identical(status, "ok")) {
      paste(utils::tail(readLines(file.path(dir, "err"), warn = FALSE), 5),
            collapse = " | ")
    } else ""
  ))
}


# A peak is a high-water mark, so sampling can only ever *miss* it --
# never overstate it. A run that finishes in a second gives the 10 ms
# sampler only a couple of hundred looks, so short points are measured
# more than once and the largest peak kept: more looks can only find
# more. Long points are left alone; they have thousands of samples and
# they are the expensive ones.
#
# This is worth doing and it is *not* what makes the small end of the
# grid non-monotone. At 8 features TabPFN v2.5 measures 2.1, 1.1, 1.2 and
# 1.9 GB across 800 to 6,400 rows, and repeating each three times moved
# those by less than 0.2 GB. Nothing a model can do fits a peak that
# falls by half when the input doubles: below a gigabyte or two the
# number is the process -- R, libtorch, and whatever arena the allocator
# happened to take -- not the activation. Those points still constrain
# the intercept, which is why they are fitted; what they cannot do is
# grade the slope, which is why `test-memory-calibration.R` holds its
# accuracy band to the points where the transient is the larger term.
measure_point_repeated <- function(point, model_dir, task, opts,
                                   repeats = 3L, short_sec = 8) {
  got <- measure_point(point, model_dir, task, opts)
  if (!identical(got$status, "ok")) return(got)
  elapsed <- got$elapsed_sec %||% NA_real_
  if (!isTRUE(is.finite(elapsed)) || elapsed > short_sec) return(got)
  for (i in seq_len(repeats - 1L)) {
    again <- measure_point(point, model_dir, task, opts)
    if (!identical(again$status, "ok")) next
    if (isTRUE(again$peak_bytes > got$peak_bytes)) got <- again
  }
  got$repeats <- repeats
  got
}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

# Small on purpose. The point of the grid is to separate the terms --
# rows from columns, and both from the constant -- not to cover the space.
# Extrapolation is what the holdout checks, and extrapolation is the whole
# job: the sizes that matter are the ones too big to measure safely.
GRIDS <- list(
  # Quick shape check while working on the harness itself. Too small to
  # fit constants from: at these sizes an idle R + libtorch process is
  # most of what the sampler sees.
  small = list(n_context = c(200, 400, 800, 1600),
               n_features = c(8, 32), n_query = 200),
  # The working default. Large enough that the activations, rather than
  # the process baseline, are what moves between points.
  full  = list(n_context = c(800, 1600, 3200, 6400),
               n_features = c(8, 32, 90), n_query = 500),
  # For the backends cheap enough to push further.
  wide  = list(n_context = c(800, 1600, 3200, 6400, 12800),
               n_features = c(8, 32, 90, 250), n_query = 500),
  # Reaches past TabPFN v3's 2,048-row stage chunk, which nothing below
  # it does. Without points on both sides of that boundary the chunked
  # activation term and the whole-table `res` term are the same straight
  # line and the fit cannot tell them apart -- so this is the grid the v3
  # constants have to come from, not a nicety.
  deep  = list(n_context = c(512, 1024, 2048, 4096, 8192, 16384, 32768),
               n_features = c(16, 50, 120), n_query = 500)
)

build_grid <- function(which = "small") {
  g <- GRIDS[[which]]
  if (is.null(g)) stop("unknown grid: ", which)
  pts <- expand.grid(n_context = g$n_context, n_features = g$n_features,
                     KEEP.OUT.ATTRS = FALSE)
  pts$n_query <- g$n_query
  # Cheapest first, so a run that has to be interrupted still leaves a
  # usable spread rather than one corner of the design.
  pts <- pts[order(pts$n_context * pts$n_features), , drop = FALSE]
  lapply(seq_len(nrow(pts)), function(i) as.list(pts[i, ]))
}


# ---------------------------------------------------------------------------
# Fitting the constants
# ---------------------------------------------------------------------------

# The shapes the estimator would use for this point, in elements. Asking
# the backend directly is what keeps the fitted constants meaningful:
# they are the constants for *these* formulas, and a change to either
# invalidates the other.
terms_for <- function(backend, point, opts, config) {
  tabfound:::.peak_terms_for(get_backend(backend), point$n_context,
                             point$n_query, point$n_features, opts, config)
}

# `peak = intercept + weights + persistent + 4 * max_s(b*act_s + c*att_s)`
#
# The max over stages is what makes this worth more than an `lm()`: which
# stage dominates can itself depend on the coefficients. Residuals are
# taken in log space because peaks here span two orders of magnitude and
# a gigabyte of error means something very different at each end.
#
# A coarse sweep comes first and the optimiser only polishes. Going
# straight to `optim()` looked fine and was not: the objective has long
# flat valleys, and from a single start it settled on an intercept of
# 100 MB where the data wanted 1.7 GB, then inflated the slope to
# compensate. The sweep is cheap and the failure was silent, which is the
# worst combination to leave in.
#
# `intercept` absorbs what is resident before any activation exists and
# is not in the checkpoint: R itself, libtorch, its allocator arenas. It
# cannot be below what the worker reported once the weights were in.
fit_coefs <- function(rows, baseline = 0) {
  measured <- vapply(rows, function(r) r$peak_bytes, numeric(1))
  stopifnot(length(rows) >= 3L)
  fixed <- vapply(rows, function(r) r$weights_bytes + r$persistent_bytes,
                  numeric(1))
  n_stage <- length(rows[[1]]$act)
  act <- matrix(unlist(lapply(rows, function(r) r$act)), ncol = n_stage,
                byrow = TRUE)
  att <- matrix(unlist(lapply(rows, function(r) r$att)), ncol = n_stage,
                byrow = TRUE)
  res <- matrix(unlist(lapply(rows, function(r) r$res %||% rep(0, n_stage))),
                ncol = n_stage, byrow = TRUE)
  pre <- matrix(unlist(lapply(rows, function(r) r$prepass %||% rep(0, n_stage))),
                ncol = n_stage, byrow = TRUE)
  # Which stage is the in-context one, so its own copy count can be
  # fitted. Identifiable only when some point has it dominating, which in
  # practice means a chunked grid: without one the column stage is always
  # the largest and this parameter has nothing to say.
  icl_at <- rows[[1]]$copies_icl %||% rep(FALSE, n_stage)

  # The same arithmetic `estimate_peak_memory()` will do -- literally, via
  # the package's own assembly function -- so the constants cannot be
  # fitted to one expression and applied to another.
  predict_with <- function(theta) {
    co <- list(intercept_bytes = theta[1], act_copies = theta[2],
               att_copies = theta[3], res_copies = theta[4],
               prepass_copies = theta[5], icl_copies = theta[6],
               safety_factor = 1)
    vapply(seq_along(rows), function(i) {
      stages <- lapply(seq_len(n_stage), function(k) {
        list(name = "", act = act[i, k], att = att[i, k], res = res[i, k],
             prepass = pre[i, k],
             copies = if (isTRUE(icl_at[k])) "icl" else "")
      })
      tabfound:::.peak_from_terms(stages, 0, fixed[i], co, spmf = 1,
                                  n_estimators = 1)$peak
    }, numeric(1))
  }
  obj <- function(theta) {
    if (any(theta < 0)) return(1e12)
    p <- predict_with(theta)
    if (any(!is.finite(p)) || any(p <= 0)) return(1e12)
    sum((log(p) - log(measured))^2)
  }

  # `res` only exists where a stage keeps whole-table tensors outside its
  # chunk loop, which today is TabPFN v3 alone. Where every `res` is zero
  # the fourth coefficient is unidentifiable, so it is pinned rather than
  # left to wander into whatever the optimiser's flat valley suggests.
  # A coefficient for a term that is identically zero is not a fit, it is
  # whatever the sweep happened to land on -- and written to a shipped
  # file it reads as a finding. Both are pinned when their term is absent.
  has_res <- any(res > 0)
  has_att <- any(att > 0)
  # The pre-pass only exists where some point was measured chunked. Where
  # none was, it is pinned at the forward's count -- the conservative
  # fallback the estimator uses anyway.
  has_pre <- any(pre > 0)
  res_seed <- if (has_res) c(1, 2, 4, 8) else 2
  att_seed <- if (has_att) c(0, 0.05, 0.1, 0.25, 0.5, 1, 2) else 0
  sweep <- expand.grid(
    a = seq(baseline, max(baseline * 4, 4e9), length.out = 41),
    b = seq(1, 400, length.out = 60),
    cc = att_seed,
    dd = if (has_res) c(1, 2, 4, 8, 64, 256) else 2,
    ee = if (has_pre) c(2, 8, 32, 80) else 0,
    ff = if (has_pre) c(80, 200, 400) else 0
  )
  vals <- mapply(function(a, b, cc, dd, ee, ff) obj(c(a, b, cc, dd, ee, ff)),
                 sweep$a, sweep$b, sweep$cc, sweep$dd, sweep$ee, sweep$ff)
  seed <- as.numeric(sweep[which.min(vals), ])
  if (!has_pre) seed[6] <- seed[2]   # the shared count, i.e. no split

  # With no `res` term in the data the fourth parameter is not just
  # unidentifiable, it is degenerate: handing L-BFGS-B a coordinate whose
  # lower and upper bounds are equal perturbed the other three enough to
  # miss a synthetic round-trip by 2.4%. So it is held out of the search
  # entirely and pasted back afterwards.
  best <- tryCatch(
    {
      live <- c(TRUE, TRUE, has_att, has_res, has_pre, has_pre)
      fit <- stats::optim(
        seed[live],
        function(th) { full <- seed; full[live] <- th; obj(full) },
        method = "L-BFGS-B",
        lower = c(baseline, 0.5, 0, 0, 0, 0.5)[live],
        # `res` used to stop at 64 -- a number chosen when the only
        # points were unchunked and it never went near it. A chunked
        # grid pins it there, which is the optimiser saying the bound is
        # the constraint rather than the data.
        upper = c(8e9, 1000, 100, 1000, 1000, 1000)[live]
      )
      par <- seed; par[live] <- fit$par
      fit$par <- par
      fit
    },
    error = function(e) NULL
  )
  theta <- if (!is.null(best) && best$value < min(vals)) best$par else seed

  pred <- predict_with(theta)

  # Where the activation is smaller than the intercept, what was measured
  # is the process rather than the model's subject. Those points set the
  # *floor* -- the estimate never drops below what this backend costs to
  # do nothing -- and are kept out of the safety factor, which exists to
  # cover the slope. Lumping them in lifts every estimate by the ratio of
  # the worst baseline point, which on TabPFN v2.5 was a factor of two.
  variable <- pred - theta[1]
  baseline_only <- variable < theta[1]
  floor_bytes <- if (any(baseline_only)) max(measured[baseline_only]) else 0
  slope_pts <- if (all(baseline_only)) rep(TRUE, length(measured))
               else !baseline_only

  list(
    censor_floor_lift = function(censored, weights_bytes) {
      # Censored points measured what the machine would give them, not
      # what they wanted, so they are lower bounds -- and an estimate
      # below a lower bound is a false reassurance about a run that was
      # already too big. The report calls these out; this is what lets
      # the margin answer for them.
      out <- 1
      cc <- list(intercept_bytes = theta[1], act_copies = theta[2],
                 att_copies = theta[3], res_copies = theta[4],
                 prepass_copies = theta[5], icl_copies = theta[6],
                 safety_factor = 1)
      for (p in censored) {
        ns <- length(p$act)
        st <- lapply(seq_len(ns), function(k) {
          list(act = p$act[k], att = p$att[k], res = p$res[k],
               prepass = (p$prepass %||% rep(0, ns))[k],
               copies = if (isTRUE((p$copies_icl %||% rep(FALSE, ns))[k]))
                 "icl" else "")
        })
        e <- tabfound:::.peak_from_terms(st, weights_bytes,
                                         p$persistent_bytes, cc, spmf = 1)$peak
        if (isTRUE(e > 0)) out <- max(out, p$peak_bytes / e)
      }
      out
    },
    intercept_bytes = theta[1],
    act_copies = theta[2],
    att_copies = theta[3],
    res_copies = theta[4],
    prepass_copies = if (has_pre) theta[5] else NULL,
    icl_copies = if (has_pre) theta[6] else NULL,
    floor_bytes = floor_bytes,
    # The guard must fail safe. Whatever least squares says, lift it until
    # no measured point sits above its own estimate: an over-estimate
    # costs a warning and an under-estimate costs a session.
    safety_factor = max(1, max(measured[slope_pts] / pred[slope_pts])),
    ratios = pred / measured,
    predicted = pred,
    measured = measured,
    predict_with = predict_with,
    theta = theta
  )
}

# Fit without the largest context sizes, then look at what the fit says
# about them. Extrapolation is the entire job -- the sizes that matter
# are the ones too big to measure safely -- so a model that only
# interpolates is no use, and this is the check that would catch it.
holdout_check <- function(rows, baseline = 0) {
  if (length(rows) < 4L) return(NULL)
  ctx <- vapply(rows, function(r) r$n_context, numeric(1))
  cut <- sort(unique(ctx))
  if (length(cut) < 3L) return(NULL)
  cut <- cut[length(cut) - 1L]          # hold out the largest size
  train <- rows[ctx <= cut]
  test  <- rows[ctx > cut]
  if (!length(test) || length(train) < 3L) return(NULL)
  co <- fit_coefs(train, baseline = baseline)
  pred <- co$predict_with(co$theta)     # refit shape, applied below
  # Rebuild the prediction for the held-out points from the same theta.
  fixed <- vapply(test, function(r) r$weights_bytes + r$persistent_bytes,
                  numeric(1))
  n_stage <- length(test[[1]]$act)
  a <- matrix(unlist(lapply(test, function(r) r$act)), ncol = n_stage,
              byrow = TRUE)
  b <- matrix(unlist(lapply(test, function(r) r$att)), ncol = n_stage,
              byrow = TRUE)
  rr <- matrix(unlist(lapply(test, function(r) r$res %||% rep(0, n_stage))),
               ncol = n_stage, byrow = TRUE)
  cc <- list(intercept_bytes = co$theta[1], act_copies = co$theta[2],
             att_copies = co$theta[3], res_copies = co$theta[4],
             safety_factor = 1)
  p <- vapply(seq_along(test), function(i) {
    stages <- lapply(seq_len(n_stage), function(k) {
      list(name = "", act = a[i, k], att = b[i, k], res = rr[i, k])
    })
    tabfound:::.peak_from_terms(stages, 0, fixed[i], cc, spmf = 1,
                                n_estimators = 1)$peak
  }, numeric(1))
  list(held_out_from = cut + 1,
       points = test,
       predicted = p * co$safety_factor,
       measured = vapply(test, function(r) r$peak_bytes, numeric(1)))
}


# ---------------------------------------------------------------------------
# Driving one backend
# ---------------------------------------------------------------------------

model_for <- function(backend) {
  cat_ <- tabfound:::.model_catalog()
  # `"both"` is TabPFN v3.5's multitask checkpoint: one artifact serves
  # classification and regression, so it is the classifier for this
  # purpose as much as anything is.
  ids <- names(cat_)[vapply(cat_, function(e) {
    identical(e$backend, backend) &&
      (identical(e$task, "classification") || identical(e$task, "both"))
  }, logical(1))]
  ids <- Filter(function(id) tabfound:::.model_is_downloaded(id), ids)
  if (!length(ids)) return(NULL)
  id <- ids[1]
  entry <- cat_[[id]]
  d <- tabfound:::.model_dir(id)
  list(id = id,
       dir = if (is.null(entry$subfolder)) d else file.path(d, entry$subfolder))
}

calibrate_backend <- function(backend, grid = "small", opts = list(),
                              chunked = FALSE) {
  m <- model_for(backend)
  if (is.null(m)) {
    message("  no downloaded classifier for ", backend, " -- skipping")
    return(NULL)
  }
  # One member. Members run sequentially, so a single one has the same
  # peak and a thirty-second grid point instead of a twenty-minute one.
  # Not every backend takes the argument -- TabPFN's ensemble comes from
  # a dumped config directory rather than a count -- so only what the
  # predictor actually declares is passed on.
  opts <- utils::modifyList(list(n_estimators = 1L), opts)
  accepted <- names(formals(get_backend(backend)$classifier))
  cat("  options: ",
      paste(names(opts), unlist(opts), sep = "=", collapse = " "), "\n",
      sep = "")
  opts <- opts[intersect(names(opts), accepted)]
  config <- tabfound:::read_model_config(file.path(m$dir, "config.json"))
  weights <- as.numeric(file.size(file.path(m$dir, "model.safetensors")))

  rows <- list()
  for (point in build_grid(grid)) {
    cat(sprintf("  %-6s ctx=%5d p=%3d ... ", backend, point$n_context,
                point$n_features))
    utils::flush.console()
    got <- measure_point_repeated(point, m$dir, "classification", opts)
    tm <- terms_for(backend, point, .resolve_opts(backend, opts), config)
    got$act <- vapply(tm$stages, function(s) s$act %||% 0, numeric(1))
    got$att <- vapply(tm$stages, function(s) s$att %||% 0, numeric(1))
    got$res <- vapply(tm$stages, function(s) s$res %||% 0, numeric(1))
    got$prepass <- vapply(tm$stages, function(s) s$prepass %||% 0, numeric(1))
    got$copies_icl <- vapply(tm$stages,
                             function(s) identical(s$copies %||% "", "icl"),
                             logical(1))
    # Recorded with the point, because a refit has to re-derive the terms
    # in the configuration the peak was measured under and cannot know it
    # otherwise.
    got$row_chunk_size <- opts$row_chunk_size %||% NA_integer_
    got$col_chunk_size <- opts$col_chunk_size %||% NA_integer_
    got$weights_bytes <- weights
    got$persistent_bytes <- point$n_context * point$n_features * 8 * 2 +
      4 * (tm$persistent %||% 0)
    rows[[length(rows) + 1L]] <- got
    cat(sprintf("%-5s %8s  %5.1fs\n", got$status,
                tabfound:::.fmt_bytes(got$peak_bytes),
                got$elapsed_sec %||% NA_real_))
    # A point that dies is either the finding or a bug in the harness,
    # and the difference is in the worker's last words.
    if (!identical(got$status, "ok") && nzchar(got$stderr %||% "")) {
      cat("         ", substr(got$stderr, 1, 200), "\n", sep = "")
    }
  }

  list(backend = backend, model_id = m$id, rows = rows,
       weights_bytes = weights)
}

# The predictor's own defaults, merged with anything the caller set --
# the same resolution `estimate_peak_memory()` does, so the shapes the
# fit is against are the shapes the estimator will compute later.
.memory_coefs_or_stop <- function(backend) {
  co <- tabfound:::.memory_coefs(backend, device = "cpu")
  if (is.null(co)) stop("no coefficients for ", backend)
  co
}

.resolve_opts <- function(backend, opts) {
  tabfound:::.resolve_memory_opts(get_backend(backend), "classification",
                                  opts, list())
}


# ---------------------------------------------------------------------------
# How an ensemble moves the peak
# ---------------------------------------------------------------------------

# Members run sequentially, so one member's activations are all that is
# ever live -- and the resident high-water mark still climbs, because a
# freed arena is not a returned one. This measures the climb at fixed
# dimensions so the estimator can carry a term for it instead of an
# assumption.
#
#   Rscript inst/memory/calibrate.R tabicl --ensemble
ENSEMBLE_SIZES <- c(1L, 2L, 4L, 8L, 16L, 32L)

# Deliberately small: the point is the ratio between member counts, and a
# large point would hit the machine's ceiling before the sweep finished
# and censor exactly the numbers being compared.
# What `save_peak_memory_factor` actually buys, measured rather than
# assumed. The estimator models it as
# `act_copies * (spmf_floor + (1 - spmf_floor) / k)`, so the sweep fits
# one number: the share of the resident activation the factor cannot
# reach. Its default is 1 -- "buys nothing" -- so a backend that has
# never been swept is never promised a saving.
#
# The point has to be large enough that the activation, not the process,
# is what moves between factors, which is why this is not the ensemble
# point.
SPMF_FACTORS <- c(1L, 2L, 4L, 8L, 16L, 32L)
SPMF_POINT <- list(n_context = 4000, n_query = 500, n_features = 50)

read_spmf_sweep <- function(backend) {
  path <- file.path(HERE, "measurements", paste0("spmf-", backend, ".json"))
  if (!file.exists(path)) {
    message("  no stored spmf sweep at ", path)
    return(NULL)
  }
  blob <- fromJSON(path, simplifyVector = FALSE)
  list(backend = blob$backend, model_id = blob$model_id, point = blob$point,
       rows = lapply(blob$points, function(r) lapply(r, unlist)))
}

spmf_sweep <- function(backend, point = SPMF_POINT) {
  m <- model_for(backend)
  if (is.null(m)) {
    message("  no downloaded classifier for ", backend, " -- skipping")
    return(NULL)
  }
  if (!"save_peak_memory_factor" %in%
        names(formals(get_backend(backend)$classifier))) {
    message("  ", backend, " has no save_peak_memory_factor -- skipping")
    return(NULL)
  }
  rows <- list()
  for (k in SPMF_FACTORS) {
    cat(sprintf("  %-8s spmf=%2d ... ", backend, k))
    utils::flush.console()
    opts <- if (k == 1L) list() else list(save_peak_memory_factor = k)
    r <- measure_point_repeated(point, m$dir, "classification", opts)
    r$spmf <- k
    rows[[length(rows) + 1L]] <- r
    cat(sprintf("%-5s %8s  %5.1fs
", r$status,
                tabfound:::.fmt_bytes(r$peak_bytes), r$elapsed_sec %||% NA))
  }
  list(backend = backend, model_id = m$id, point = point, rows = rows)
}

# Fit the one number the sweep is for. The peak at factor `k` is
# `fixed + variable * (fl + (1 - fl) / k)`, so with `fixed` taken from
# the backend's own fitted intercept the ratios pin `fl` directly. Taken
# as the *largest* `fl` any factor implies, because the estimator must
# not promise more than the worst case observed.
fit_spmf_floor <- function(sweep, co) {
  ok <- Filter(function(r) identical(r$status, "ok"), sweep$rows)
  base <- Filter(function(r) isTRUE(r$spmf == 1L), ok)
  if (!length(base) || length(ok) < 3L) return(NULL)
  fixed <- co$intercept_bytes + (co$weights_fallback_bytes %||% 0)
  v0 <- base[[1]]$peak_bytes - fixed
  if (!isTRUE(v0 > 0)) return(NULL)
  fls <- vapply(Filter(function(r) r$spmf > 1L, ok), function(r) {
    ratio <- (r$peak_bytes - fixed) / v0
    k <- r$spmf
    # ratio = fl + (1 - fl)/k  =>  fl = (ratio - 1/k) / (1 - 1/k)
    # Allowed above 1: a factor that costs more than it saves is a real
    # measurement, and a floor pinned at 1 would report it as neutral.
    max(0, (ratio - 1 / k) / (1 - 1 / k))
  }, numeric(1))
  if (!length(fls)) return(NULL)
  max(fls)
}

# What the summary pre-pass costs, measured rather than inherited from
# the forward's copy count. A row-chunked run cannot start until the
# column stage's summaries exist, and building them holds a tensor as
# wide as the whole table -- so on a chunked run it is often the peak,
# and it is a narrower operation than the forward the `act_copies` fit
# describes. One sweep, one number: `prepass_copies`.
#
# Wide on purpose. The pre-pass scales with columns, so a point with few
# of them cannot separate it from anything else.
CHUNK_POINT <- list(n_context = 8000, n_query = 500, n_features = 200)
CHUNK_ROW <- 2048L

read_chunk_sweep <- function(backend) {
  path <- file.path(HERE, "measurements", paste0("chunks-", backend, ".json"))
  if (!file.exists(path)) {
    message("  no stored chunk sweep at ", path)
    return(NULL)
  }
  blob <- fromJSON(path, simplifyVector = FALSE)
  list(backend = blob$backend, model_id = blob$model_id, point = blob$point,
       rows = lapply(blob$points, function(r) lapply(r, unlist)))
}

chunk_sweep <- function(backend, point = CHUNK_POINT) {
  m <- model_for(backend)
  if (is.null(m)) {
    message("  no downloaded classifier for ", backend, " -- skipping")
    return(NULL)
  }
  fm <- names(formals(get_backend(backend)$classifier))
  if (!all(c("row_chunk_size", "col_chunk_size") %in% fm)) {
    message("  ", backend, " has no stage chunking -- skipping")
    return(NULL)
  }
  # The "off" setting *omits* the key rather than passing NULL. The spec
  # goes to the worker as JSON, and a NULL there comes back as an empty
  # list, which the predictor then tries to read as a chunk size -- a
  # worker that dies before it allocates anything, recorded as a crash
  # boundary that never happened. Omitting it lets the predictor's own
  # default (NULL, meaning off) stand.
  settings <- list(
    list(label = "row only", opts = list(row_chunk_size = CHUNK_ROW)),
    list(label = "row + col 8", opts = list(row_chunk_size = CHUNK_ROW,
                                            col_chunk_size = 8L))
  )
  rows <- list()
  for (st in settings) {
    cat(sprintf("  %-8s %-12s ... ", backend, st$label))
    utils::flush.console()
    r <- measure_point_repeated(point, m$dir, "classification", st$opts)
    r$row_chunk_size <- CHUNK_ROW
    r$col_chunk_size <- st$opts$col_chunk_size %||% NA_integer_
    r$label <- st$label
    rows[[length(rows) + 1L]] <- r
    cat(sprintf("%-5s %8s  %5.1fs\n", r$status,
                tabfound:::.fmt_bytes(r$peak_bytes), r$elapsed_sec %||% NA))
  }
  list(backend = backend, model_id = m$id, point = point, rows = rows)
}

# Fit the one number. With the row chunk fixed, the only term that moves
# between the two settings is the pre-pass, so the difference in measured
# peak over the difference in its shape is the copy count directly.
fit_prepass_copies <- function(sweep, backend, co) {
  ok <- Filter(function(r) identical(r$status, "ok"), sweep$rows)
  if (length(ok) < 2L) return(NULL)
  m <- model_for(backend)
  config <- tabfound:::read_model_config(file.path(m$dir, "config.json"))
  shape <- function(r) {
    opts <- .resolve_opts(backend, list(
      n_estimators = 1L, row_chunk_size = r$row_chunk_size,
      col_chunk_size = if (is.na(r$col_chunk_size)) NULL else r$col_chunk_size
    ))
    tm <- terms_for(backend, sweep$point, opts, config)
    idx <- which(vapply(tm$stages, function(s) identical(s$name, "column summaries"),
                        logical(1)))
    if (!length(idx)) return(0)
    tm$stages[[idx]]$prepass %||% 0
  }
  a <- ok[[1]]; b <- ok[[2]]
  d_shape <- shape(a) - shape(b)
  d_peak <- a$peak_bytes - b$peak_bytes
  if (!isTRUE(d_shape > 0) || !isTRUE(d_peak > 0)) return(NULL)
  list(copies = d_peak / (d_shape * 4), points = ok)
}

# Two measurements of a chunked run are two more points the estimate must
# not fall below. A difference-based fit pins how the peak moves with the
# column chunk and says nothing about where it starts, and the constants
# it starts from were fitted on *unchunked* runs -- which on TabICL left
# the chunked estimate at 0.6x of measured, the one direction a guard
# must never err in. The safety factor is the mechanism the package
# already uses for exactly this, so extend it rather than invent
# something: whatever it takes for these points to be covered too.
lift_safety_for_chunks <- function(sweep, backend, co, prepass_copies) {
  m <- model_for(backend)
  config <- tabfound:::read_model_config(file.path(m$dir, "config.json"))
  w <- as.numeric(file.size(file.path(m$dir, "model.safetensors")))
  if (!isTRUE(is.finite(w))) w <- co$weights_fallback_bytes %||% 0
  cc <- co
  cc$prepass_copies <- prepass_copies
  cc$safety_factor <- 1
  worst <- 1
  for (r in Filter(function(r) identical(r$status, "ok"), sweep$rows)) {
    opts <- .resolve_opts(backend, list(
      n_estimators = 1L, row_chunk_size = r$row_chunk_size,
      col_chunk_size = if (is.na(r$col_chunk_size)) NULL else r$col_chunk_size
    ))
    tm <- terms_for(backend, sweep$point, opts, config)
    persistent <- sweep$point$n_context * sweep$point$n_features * 8 * 2 +
      4 * (tm$persistent %||% 0)
    got <- tabfound:::.peak_from_terms(tm$stages, w, persistent, cc,
                                       spmf = 1, n_estimators = 1)$peak
    if (isTRUE(got > 0)) worst <- max(worst, r$peak_bytes / got)
  }
  worst
}

ENSEMBLE_POINT <- list(n_context = 800, n_query = 500, n_features = 32)

read_ensemble_sweep <- function(backend) {
  path <- file.path(HERE, "measurements", paste0("ensemble-", backend, ".json"))
  if (!file.exists(path)) {
    message("  no stored ensemble sweep at ", path)
    return(NULL)
  }
  blob <- fromJSON(path, simplifyVector = FALSE)
  list(backend = blob$backend, model_id = blob$model_id, point = blob$point,
       rows = lapply(blob$points, function(r) lapply(r, unlist)))
}

ensemble_sweep <- function(backend, point = ENSEMBLE_POINT) {
  m <- model_for(backend)
  if (is.null(m)) {
    message("  no downloaded classifier for ", backend, " -- skipping")
    return(NULL)
  }
  if (!"n_estimators" %in% names(formals(get_backend(backend)$classifier))) {
    message("  ", backend, " has no n_estimators argument -- skipping")
    return(NULL)
  }
  rows <- list()
  for (n in ENSEMBLE_SIZES) {
    cat(sprintf("  %-8s members=%2d ... ", backend, n))
    utils::flush.console()
    r <- measure_point(point, m$dir, "classification", list(n_estimators = n))
    r$n_estimators <- n
    rows[[length(rows) + 1L]] <- r
    cat(sprintf("%-5s %8s  %5.1fs\n", r$status,
                tabfound:::.fmt_bytes(r$peak_bytes), r$elapsed_sec %||% NA))
  }
  list(backend = backend, model_id = m$id, point = point, rows = rows)
}

# How much bigger does the transient have to be to explain the climb?
#
# Only measured *differences* go in. Subtracting a modelled floor
# (intercept + weights) looked simpler and was wrong: the intercept is
# fitted on the size grid and does not describe this much smaller point,
# so for TabFM it left a 0.5 GB "transient" and an implied factor of 14.
# `peak(n) - peak(1)` cancels every constant, whatever it is, and the
# scale it is expressed in is the estimator's own predicted transient --
# which is exactly what the factor will multiply.
ensemble_factor <- function(sweep, co, weights_bytes, config) {
  ok <- Filter(function(r) {
    identical(r$status, "ok") && isTRUE(r$peak_bytes < censor_threshold())
  }, sweep$rows)
  if (length(ok) < 3L) return(NULL)
  n <- vapply(ok, function(r) r$n_estimators, numeric(1))
  peak <- vapply(ok, function(r) r$peak_bytes, numeric(1))
  if (!any(n == 1)) return(NULL)

  opts <- .resolve_opts(sweep$backend, list(n_estimators = 1L))
  tm <- terms_for(sweep$backend, sweep$point, opts, config)
  t1 <- tabfound:::.peak_from_terms(tm$stages, 0, 0, co, spmf = 1,
                                    n_estimators = 1)$transient
  if (!isTRUE(t1 > 0)) return(NULL)

  ratios <- 1 + (peak - peak[n == 1][1]) / t1
  k <- max(0, max((ratios[n > 1] - 1) / log2(n[n > 1])))
  # TabICL keeps climbing across the whole sweep; TabFM stops after two
  # members and stays flat. One slope cannot describe both, and the slope
  # TabFM's first step implies would inflate its 32-member default -- the
  # published one -- by more than double. So the envelope is a capped
  # line: the cap is the largest ratio actually observed, with a margin.
  cap <- max(ratios) * 1.2
  list(k = k, cap = cap, n = n, ratios = ratios,
       envelope = pmin(1 + k * log2(n), cap))
}

report_ensemble <- function(sweep, ef) {
  cat("\n", sweep$backend, " -- ensemble sweep at ctx=", sweep$point$n_context,
      " p=", sweep$point$n_features, "\n", sep = "")
  if (is.null(ef)) {
    cat("  too few usable points to fit a factor\n")
    return(invisible(NULL))
  }
  cat("  members  transient x  envelope\n")
  for (i in seq_along(ef$n)) {
    cat(sprintf("  %7d  %10.2f  %8.2f\n", ef$n[i], ef$ratios[i],
                ef$envelope[i]))
  }
  cat(sprintf("  ensemble_log2_factor = %.3f, ensemble_cap = %.2f\n",
              ef$k, ef$cap))
}


# ---------------------------------------------------------------------------
# Reporting and writing
# ---------------------------------------------------------------------------

# A point that came within reach of the machine's own ceiling did not
# measure what it wanted, it measured what it could get: the OS started
# reclaiming, or the process died partway up. Those observations are
# censored -- they are lower bounds, not values -- and fitting to them
# flattens the curve exactly where it matters. They are kept in the
# record and checked against afterwards, not fitted.
CENSOR_FRACTION <- as.numeric(Sys.getenv("TABFOUND_CENSOR_FRACTION", "0.6"))

censor_threshold <- function() {
  total <- tabfound:::.system_memory()$total
  if (!isTRUE(is.finite(total))) return(Inf)
  CENSOR_FRACTION * total
}

usable_points <- function(rows) {
  thr <- censor_threshold()
  Filter(function(r) {
    identical(r$status, "ok") && isTRUE(r$peak_bytes < thr)
  }, rows)
}

censored_points <- function(rows) {
  thr <- censor_threshold()
  Filter(function(r) {
    identical(r$status, "died") || (identical(r$status, "ok") &&
                                    isTRUE(r$peak_bytes >= thr))
  }, rows)
}

# Points that never got as far as running. Nothing to learn from, and
# they must not be mistaken for a ceiling.
failed_points <- function(rows) {
  Filter(function(r) r$status %in% c("error", "no_start"), rows)
}

report <- function(cal, co, holdout = NULL) {
  cat("\n", cal$backend, " -- fitted constants\n", sep = "")
  cat(sprintf("  intercept %s  act_copies %.1f  att_copies %.2f  safety %.2f\n",
              tabfound:::.fmt_bytes(co$intercept_bytes), co$act_copies,
              co$att_copies, co$safety_factor))
  cat(sprintf("    res_copies %.2f  floor %s\n", co$res_copies %||% NA_real_,
              tabfound:::.fmt_bytes(co$floor_bytes %||% 0)))
  cat("  point                measured   estimated   ratio\n")
  ok <- usable_points(cal$rows)
  for (i in seq_along(ok)) {
    r <- ok[[i]]
    cat(sprintf("  ctx=%5d p=%3d %10s %11s   %5.2f\n",
                r$n_context, r$n_features,
                tabfound:::.fmt_bytes(co$measured[i]),
                tabfound:::.fmt_bytes(co$predicted[i] * co$safety_factor),
                co$ratios[i] * co$safety_factor))
  }
  if (!is.null(holdout)) {
    cat(sprintf("  extrapolation: fitted below ctx=%d, then asked about\n",
                holdout$held_out_from))
    for (i in seq_along(holdout$points)) {
      r <- holdout$points[[i]]
      cat(sprintf("  ctx=%5d p=%3d %10s %11s   %5.2f  (held out)\n",
                  r$n_context, r$n_features,
                  tabfound:::.fmt_bytes(holdout$measured[i]),
                  tabfound:::.fmt_bytes(holdout$predicted[i]),
                  holdout$predicted[i] / holdout$measured[i]))
    }
  }
  bad <- failed_points(cal$rows)
  if (length(bad)) {
    cat("  did not run (not a memory result -- check the environment):\n")
    for (r in bad) {
      cat(sprintf("  ctx=%5d p=%3d  %s\n", r$n_context, r$n_features,
                  substr(r$stderr %||% "", 1, 120)))
    }
  }
  cens <- censored_points(cal$rows)
  if (length(cens)) {
    cat("  censored (recorded, not fitted):\n")
    for (r in cens) {
      # The fit has to say these are trouble, or it has learned nothing
      # from the one outcome that matters.
      ns <- length(r$act)
      stages <- lapply(seq_len(ns), function(k) {
        list(name = "", act = r$act[k], att = r$att[k],
             res = (r$res %||% rep(0, ns))[k],
             prepass = (r$prepass %||% rep(0, ns))[k],
             copies = if (isTRUE((r$copies_icl %||% rep(FALSE, ns))[k]))
               "icl" else "")
      })
      # `co` already carries the safety factor, so `.peak_from_terms()`
      # has applied it. Multiplying again squared it and reported a
      # censored point as UNDER while the same point cleared its floor
      # comfortably in the fit -- two numbers for one quantity, and the
      # louder one was wrong.
      p <- tabfound:::.peak_from_terms(
        stages, r$weights_bytes, r$persistent_bytes, co, spmf = 1,
        n_estimators = 1
      )$peak
      under <- isTRUE(p < r$peak_bytes)
      cat(sprintf("  ctx=%5d p=%3d %10s   estimate %s%s\n",
                  r$n_context, r$n_features,
                  if (identical(r$status, "ok")) "at ceiling" else r$status,
                  tabfound:::.fmt_bytes(p),
                  if (under) "  <-- UNDER the observed floor" else ""))
    }
  }
}

# The stored measurements, in the shape `fit_coefs()` wants. The terms
# are recomputed rather than read back: they are a property of today's
# `peak_terms()`, and re-deriving them is exactly what makes a re-fit
# after a formula change mean something.
# One reader for a stored chunk size, shared with the test suite: see
# `tabfound:::.chunk_field()` for the four spellings it has to accept.
chunk_field <- function(v) tabfound:::.chunk_field(v)

# The options a stored point was measured under. Absent fields mean the
# backend's own defaults, which is what every grid predating the chunked
# one was run with.
point_opts <- function(backend, r) {
  extra <- list(n_estimators = 1L)
  rc <- chunk_field(r$row_chunk_size)
  cc <- chunk_field(r$col_chunk_size)
  if (!all(is.na(rc))) extra$row_chunk_size <- rc
  if (!all(is.na(cc))) extra$col_chunk_size <- cc
  .resolve_opts(backend, extra)
}

read_measurements <- function(backend) {
  path <- file.path(HERE, "measurements", paste0(backend, ".json"))
  if (!file.exists(path)) {
    message("  no stored measurements at ", path)
    return(NULL)
  }
  blob <- fromJSON(path, simplifyVector = FALSE)
  m <- model_for(backend)
  config <- if (!is.null(m)) {
    tabfound:::read_model_config(file.path(m$dir, "config.json"))
  } else {
    message("  weights are gone, so terms cannot be recomputed")
    return(NULL)
  }
  rows <- lapply(blob$points, function(r) {
    r <- lapply(r, function(v) if (is.list(v)) unlist(v) else v)
    # Each point carries the chunking it was measured under. A grid run
    # with `row_chunk_size` set is a different shape from one run
    # without, and re-deriving both against a single `opts` would fit the
    # constants to a configuration half the points were never in.
    tm <- terms_for(backend, r, point_opts(backend, r), config)
    r$act <- vapply(tm$stages, function(s) s$act %||% 0, numeric(1))
    r$att <- vapply(tm$stages, function(s) s$att %||% 0, numeric(1))
    r$res <- vapply(tm$stages, function(s) s$res %||% 0, numeric(1))
    r$prepass <- vapply(tm$stages, function(s) s$prepass %||% 0, numeric(1))
    r$copies_icl <- vapply(tm$stages,
                           function(s) identical(s$copies %||% "", "icl"),
                           logical(1))
    r$weights_bytes <- blob$weights_bytes
    r$persistent_bytes <- r$n_context * r$n_features * 8 * 2 +
      4 * (tm$persistent %||% 0)
    r
  })
  # A grid measured with chunking on lives beside the default one and is
  # fitted *with* it: one set of constants has to describe both, because
  # a user can ask for either.
  #
  # Merging them did not work at first, and what it took is worth
  # recording, because the failure looked like a missing term and was
  # not. With a single `act_copies` the fit pinned `res_copies` at its
  # upper bound, put two censored points below their own observed floor,
  # and landed the held-out chunked point at 0.48 of measured. Two things
  # were wrong. One `act_copies` assumes every stage holds the same
  # number of live copies -- fine while the column stage always dominates,
  # wrong the moment chunking hands that role to the in-context stage,
  # which measures 186 copies against the column stage's 140. And
  # `res_copies` had an upper bound of 64 chosen when nothing came near
  # it; a chunked grid wants 94, and a parameter sitting exactly on its
  # bound is the optimiser reporting the bound, not the data.
  chunked_path <- file.path(HERE, "measurements",
                            paste0(backend, "-chunked.json"))
  if (isTRUE(getOption("tabfound.calibrate_merge_chunked", TRUE)) &&
      file.exists(chunked_path)) {
    cb <- fromJSON(chunked_path, simplifyVector = FALSE)
    extra <- lapply(cb$points, function(r) {
      r <- lapply(r, function(v) if (is.list(v)) unlist(v) else v)
      tm <- terms_for(backend, r, point_opts(backend, r), config)
      r$act <- vapply(tm$stages, function(s) s$act %||% 0, numeric(1))
      r$att <- vapply(tm$stages, function(s) s$att %||% 0, numeric(1))
      r$res <- vapply(tm$stages, function(s) s$res %||% 0, numeric(1))
      r$prepass <- vapply(tm$stages, function(s) s$prepass %||% 0, numeric(1))
      r$copies_icl <- vapply(tm$stages,
                             function(s) identical(s$copies %||% "", "icl"),
                             logical(1))
      r$weights_bytes <- cb$weights_bytes
      r$persistent_bytes <- r$n_context * r$n_features * 8 * 2 +
        4 * (tm$persistent %||% 0)
      r
    })
    message(sprintf("  plus %d point%s measured with chunking on",
                    length(extra), if (length(extra) == 1L) "" else "s"))
    rows <- c(rows, extra)
  }

  # And the `--chunks` sweep's points, which are the only measurements of
  # a *column*-chunked run. Without them nothing in the fit exercises
  # `col_chunk`, and the two settings that do came out at 0.78 and 0.96
  # of measured -- under, on the configuration the knob exists for.
  cs_path <- file.path(HERE, "measurements", paste0("chunks-", backend, ".json"))
  if (file.exists(cs_path)) {
    cs <- fromJSON(cs_path, simplifyVector = FALSE)
    pt <- cs$point
    extra <- lapply(Filter(function(r) identical(r$status, "ok"), cs$points),
                    function(r) {
      r <- lapply(r, function(v) if (is.list(v)) unlist(v) else v)
      r$n_context <- pt$n_context; r$n_query <- pt$n_query
      r$n_features <- pt$n_features
      tm <- terms_for(backend, r, point_opts(backend, r), config)
      ns <- length(tm$stages)
      r$act <- vapply(tm$stages, function(s) s$act %||% 0, numeric(1))
      r$att <- vapply(tm$stages, function(s) s$att %||% 0, numeric(1))
      r$res <- vapply(tm$stages, function(s) s$res %||% 0, numeric(1))
      r$prepass <- vapply(tm$stages, function(s) s$prepass %||% 0, numeric(1))
      r$copies_icl <- vapply(tm$stages,
                             function(s) identical(s$copies %||% "", "icl"),
                             logical(1))
      r$weights_bytes <- blob$weights_bytes
      r$persistent_bytes <- r$n_context * r$n_features * 8 * 2 +
        4 * (tm$persistent %||% 0)
      r
    })
    if (length(extra)) {
      message(sprintf("  plus %d column-chunked point%s", length(extra),
                      if (length(extra) == 1L) "" else "s"))
      rows <- c(rows, extra)
    }
  }

  list(backend = backend, model_id = blob$model_id, rows = rows,
       weights_bytes = blob$weights_bytes)
}

write_measurements <- function(cal, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write_json(list(
    backend = cal$backend,
    model_id = cal$model_id,
    measured_on = format(Sys.time(), "%Y-%m-%d"),
    platform = paste(Sys.info()[["sysname"]], Sys.info()[["machine"]]),
    r_version = as.character(getRversion()),
    torch_version = as.character(utils::packageVersion("torch")),
    machine_total_bytes = tabfound:::.system_memory()$total,
    weights_bytes = cal$weights_bytes,
    points = lapply(cal$rows, function(r) {
      r$stderr <- NULL
      # Normalised on the way out, so "not set" is written as `null` however
      # it was read -- rather than drifting to `{}` after every refit.
      if ("row_chunk_size" %in% names(r)) r$row_chunk_size <- chunk_field(r$row_chunk_size)
      if ("col_chunk_size" %in% names(r)) r$col_chunk_size <- chunk_field(r$col_chunk_size)
      r
    })
  ), path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("  wrote ", path, "\n", sep = "")
}

# The ensemble sweep updates two fields and leaves the rest of the file
# alone: it is a different measurement from the size grid and should not
# quietly overwrite its results.
write_ensemble_coefs <- function(backend, ef, path) {
  blob <- fromJSON(path, simplifyVector = FALSE)
  blob$devices$cpu$float32$ensemble_log2_factor <- ef$k
  blob$devices$cpu$float32$ensemble_cap <- ef$cap
  blob$devices$cpu$float32$ensemble_source <- "measured"
  write_json(blob, path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
  cat("  wrote ensemble factor to ", path, "\n", sep = "")
}

write_coefs <- function(cal, co, path) {
  old <- if (file.exists(path)) fromJSON(path, simplifyVector = TRUE)
         else list()
  blob <- list(
    backend = cal$backend,
    source = "measured",
    measured_on = format(Sys.time(), "%Y-%m-%d"),
    notes = sprintf(
      paste("Fitted by inst/memory/calibrate.R against %d measured points",
            "on %s (%s), model %s. Constants are for the peak_terms()",
            "formulas as they stand; changing either invalidates the",
            "other. safety_factor is whatever it takes for no measured",
            "point to sit above its own estimate."),
      sum(vapply(cal$rows, function(r) identical(r$status, "ok"), logical(1))),
      paste(Sys.info()[["sysname"]], Sys.info()[["machine"]]),
      cal$rows[[1]]$method, cal$model_id),
    devices = list(cpu = list(float32 = list(
      intercept_bytes = co$intercept_bytes,
      act_copies = co$act_copies,
      att_copies = co$att_copies,
      res_copies = co$res_copies,
      # Fitted here when the grid holds chunked points, and carried
      # across from the `--chunks` sweep otherwise. Dropping it on a size
      # refit silently returned the pre-pass to the forward's copy count,
      # which the `res` term then tried to absorb -- it pinned at its
      # upper bound, which is what that looks like from outside.
      prepass_copies = co$prepass_copies %||%
        old$devices$cpu$float32$prepass_copies,
      icl_copies = co$icl_copies %||% old$devices$cpu$float32$icl_copies,
      floor_bytes = co$floor_bytes,
      prep_state_factor = old$devices$cpu$float32$prep_state_factor %||% 1,
      # Set by `--spmf`, likewise a separate sweep. 1 means "the factor
      # buys nothing", which is what an unswept backend must be assumed
      # to deliver.
      spmf_floor = old$devices$cpu$float32$spmf_floor %||% 1,
      # Set by `--ensemble`, which is a separate sweep; carried across a
      # size calibration rather than reset by it.
      ensemble_log2_factor =
        old$devices$cpu$float32$ensemble_log2_factor %||% 0,
      ensemble_cap = old$devices$cpu$float32$ensemble_cap %||% 1,
      ensemble_source =
        old$devices$cpu$float32$ensemble_source %||% "unmeasured",
      safety_factor = co$safety_factor,
      weights_fallback_bytes =
        old$devices$cpu$float32$weights_fallback_bytes %||% cal$weights_bytes
    )))
  )
  write_json(blob, path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
  cat("  wrote ", path, "\n", sep = "")
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  do_write <- "--write" %in% args
  grid <- if ("--grid" %in% args) args[which(args == "--grid") + 1L]
          else "small"
  backends <- setdiff(args, c("--write", "--all", "--refit", "--estimators",
                              "--ensemble", "--spmf", "--chunks",
                              "--chunked", "--grid", grid))
  if ("--all" %in% args || !length(backends)) backends <- list_backends()$name

  refit <- "--refit" %in% args
  ensemble <- "--ensemble" %in% args
  spmf <- "--spmf" %in% args
  chunks <- "--chunks" %in% args
  # A size grid measured with stage chunking *on*. Backends whose default
  # is chunking off -- TabICL and TabFM -- otherwise ship constants that
  # have never seen a chunked run, and the estimate for one is then an
  # extrapolation covered only by the safety factor.
  chunked <- "--chunked" %in% args
  # Calibration runs one ensemble member, because members are sequential
  # and the estimator counts the transient once. `--estimators N` is how
  # that claim gets checked against a real run rather than assumed.
  extra_opts <- list()
  if ("--estimators" %in% args) {
    n <- as.integer(args[which(args == "--estimators") + 1L])
    extra_opts$n_estimators <- n
    args <- setdiff(args, as.character(n))
  }
  if (chunked) extra_opts$row_chunk_size <- CHUNK_ROW

  for (backend in backends) {
    cat("\n== ", backend, " ==\n", sep = "")
    if (chunks) {
      # Superseded where a chunked size grid exists: that grid fits the
      # same two coefficients from twenty-odd points instead of two, and
      # letting the sweep write afterwards would replace the better fit
      # with the worse one purely by running order.
      if (file.exists(file.path(HERE, "measurements",
                                paste0(backend, "-chunked.json")))) {
        cat("  a chunked size grid is present -- it fits the pre-pass\n")
        cat("  from more points than this sweep can; skipping.\n")
        next
      }
      sw <- if (refit) read_chunk_sweep(backend) else chunk_sweep(backend)
      if (is.null(sw)) next
      co <- .memory_coefs_or_stop(backend)
      fit <- fit_prepass_copies(sw, backend, co)
      pc <- if (is.null(fit)) NULL else fit$copies
      safety <- if (is.null(pc)) NULL else
        lift_safety_for_chunks(sw, backend, co, pc)
      write_json(c(sw[c("backend", "model_id", "point")],
                   list(measured_on = format(Sys.time(), "%Y-%m-%d"),
                        machine_total_bytes = tabfound:::.system_memory()$total,
                        prepass_copies = if (is.null(pc)) NA_real_ else pc,
                        safety_factor = if (is.null(safety)) NA_real_ else safety,
                        points = lapply(sw$rows, function(r) {
                          r$stderr <- NULL; r
                        }))),
                 file.path(HERE, "measurements",
                           paste0("chunks-", backend, ".json")),
                 auto_unbox = TRUE, digits = NA, pretty = TRUE)
      cat(sprintf("\n  %s -- pre-pass copies %s (forward's is %.1f)\n", backend,
                  if (is.null(pc)) "not identifiable" else sprintf("%.1f", pc),
                  co$act_copies))
      if (!is.null(safety)) {
        cat(sprintf("    safety factor %.2f -> %.2f to cover the chunked points\n",
                    co$safety_factor, max(co$safety_factor, safety)))
      }
      if (do_write && !is.null(pc)) {
        path <- file.path(HERE, "coefs", paste0(backend, ".json"))
        blob <- fromJSON(path, simplifyVector = FALSE)
        blob$devices$cpu$float32$prepass_copies <- pc
        blob$devices$cpu$float32$safety_factor <-
          max(blob$devices$cpu$float32$safety_factor %||% 1, safety)
        write_json(blob, path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
        cat("  wrote ", path, "\n", sep = "")
      } else if (!is.null(pc)) {
        cat("  (not written -- pass --write to update the coefficient file)\n")
      }
      next
    }
    if (spmf) {
      sw <- if (refit) read_spmf_sweep(backend) else spmf_sweep(backend)
      if (is.null(sw)) next
      co <- .memory_coefs_or_stop(backend)
      fl <- fit_spmf_floor(sw, co)
      write_json(c(sw[c("backend", "model_id", "point")],
                   list(measured_on = format(Sys.time(), "%Y-%m-%d"),
                        machine_total_bytes = tabfound:::.system_memory()$total,
                        spmf_floor = if (is.null(fl)) NA_real_ else fl,
                        points = lapply(sw$rows, function(r) {
                          r$stderr <- NULL; r
                        }))),
                 file.path(HERE, "measurements",
                           paste0("spmf-", backend, ".json")),
                 auto_unbox = TRUE, digits = NA, pretty = TRUE)
      cat(sprintf("\n  %s -- save_peak_memory_factor floor %s\n", backend,
                  if (is.null(fl)) "not identifiable" else sprintf("%.3f", fl)))
      if (!is.null(fl)) {
        for (r in Filter(function(r) identical(r$status, "ok"), sw$rows)) {
          cat(sprintf("    k=%-3d %8s\n", r$spmf,
                      tabfound:::.fmt_bytes(r$peak_bytes)))
        }
      }
      if (do_write && !is.null(fl)) {
        path <- file.path(HERE, "coefs", paste0(backend, ".json"))
        blob <- fromJSON(path, simplifyVector = FALSE)
        blob$devices$cpu$float32$spmf_floor <- fl
        write_json(blob, path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
        cat("  wrote ", path, "\n", sep = "")
      } else if (!is.null(fl)) {
        cat("  (not written -- pass --write to update the coefficient file)\n")
      }
      next
    }
    if (ensemble) {
      sw <- if (refit) read_ensemble_sweep(backend) else ensemble_sweep(backend)
      if (is.null(sw)) next
      co <- .memory_coefs_or_stop(backend)
      m <- model_for(backend)
      w <- as.numeric(file.size(file.path(m$dir, "model.safetensors")))
      cfg <- tabfound:::read_model_config(file.path(m$dir, "config.json"))
      ef <- ensemble_factor(sw, co, w, cfg)
      # Written on a refit too: the points are unchanged but the derived
      # factor is what a refit exists to update.
      write_json(c(sw[c("backend", "model_id", "point")],
                   list(measured_on = format(Sys.time(), "%Y-%m-%d"),
                        machine_total_bytes = tabfound:::.system_memory()$total,
                        weights_bytes = w,
                        ensemble_log2_factor =
                          if (is.null(ef)) NA_real_ else ef$k,
                        ensemble_cap =
                          if (is.null(ef)) NA_real_ else ef$cap,
                        points = lapply(sw$rows, function(r) {
                          r$stderr <- NULL; r
                        }))),
                 file.path(HERE, "measurements",
                           paste0("ensemble-", backend, ".json")),
                 auto_unbox = TRUE, digits = NA, pretty = TRUE)
      report_ensemble(sw, ef)
      if (do_write && !is.null(ef)) {
        write_ensemble_coefs(backend, ef,
                             file.path(HERE, "coefs", paste0(backend, ".json")))
      } else if (!is.null(ef)) {
        cat("  (not written -- pass --write to update the coefficient file)\n")
      }
      next
    }
    # Re-fitting from stored measurements is the common case once the
    # numbers exist: changing a `peak_terms()` formula invalidates the
    # constants but not the measurements, and re-measuring costs hours.
    cal <- if (refit) read_measurements(backend) else
      calibrate_backend(backend, grid = grid, opts = extra_opts)
    if (is.null(cal)) next
    ok <- usable_points(cal$rows)
    if (length(ok) < 3L) {
      message("  only ", length(ok), " usable point(s) -- not fitting")
      write_measurements(cal, file.path(HERE, "measurements",
                                        paste0(backend, ".json")))
      next
    }
    baseline <- stats::median(
      vapply(ok, function(r) r$loaded_bytes %||% 0, numeric(1)), na.rm = TRUE
    )
    if (!isTRUE(is.finite(baseline))) baseline <- 0
    # Written before anything is reported: the measurements cost hours and
    # a formatting bug in the report must not be able to lose them.
    # Written on a refit too. The measured peaks are untouched -- they
    # are the durable artefact -- but the `act`/`att`/`res` vectors
    # beside them are today's `peak_terms()` output, and a refit exists
    # precisely because that changed. Leaving the old vectors in place
    # would leave `test-memory-calibration.R` comparing the constants
    # against shapes they were not fitted to, which is the one thing
    # that file is there to catch.
    #
    # Only the rows that came from *this* file, though. `read_measurements()`
    # can merge a chunked grid in for fitting, and writing the merged set
    # back appended 42 chunked points to the unchunked file the first
    # time this ran -- a refit silently tripling its own input.
    own <- Filter(function(r) {
      v <- chunk_field(r$row_chunk_size)
      if (chunked) !all(is.na(v)) else all(is.na(v))
    }, cal$rows)
    # A refit that would write away every point it just read is a bug in
    # the filter above, not a result. Measurements cost hours and the
    # write below is unconditional, so refuse rather than truncate.
    if (!length(own) && length(cal$rows)) {
      stop("refusing to write 0 of ", length(cal$rows), " measured points ",
           "to ", backend, ".json -- the row filter matched nothing, which ",
           "means the stored chunk sizes were not understood.", call. = FALSE)
    }
    write_measurements(
      list(backend = cal$backend, model_id = cal$model_id,
           weights_bytes = cal$weights_bytes, rows = own),
      file.path(HERE, "measurements",
                paste0(backend, if (chunked) "-chunked" else "", ".json")))
    co <- fit_coefs(ok, baseline = baseline)
    # Lift the margin until the censored points are covered too. They are
    # lower bounds on runs that were already at the machine's limit, and
    # an estimate under one of those is the guard reassuring somebody
    # about the exact run it exists to stop.
    cen <- censored_points(cal$rows)
    if (length(cen)) {
      co$safety_factor <- max(
        co$safety_factor,
        co$censor_floor_lift(cen, cal$weights_bytes)
      )
    }
    report(cal, co, holdout_check(ok, baseline = baseline))
    if (do_write) {
      write_coefs(cal, co, file.path(HERE, "coefs", paste0(backend, ".json")))
    } else {
      cat("  (not written -- pass --write to update the coefficient file)\n")
    }
  }
  invisible(NULL)
}

# Top-level only. `sys.nframe()` is 0 under `Rscript calibrate.R` and
# non-zero under `sys.source()`, which is how the pieces above can be
# reused -- from a test, or to measure one point by hand -- without
# kicking off a full calibration as a side effect.
if (sys.nframe() == 0L) main()
