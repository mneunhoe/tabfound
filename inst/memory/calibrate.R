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
               n_features = c(8, 32, 90, 250), n_query = 500)
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

  # The same arithmetic `estimate_peak_memory()` will do -- literally, via
  # the package's own assembly function -- so the constants cannot be
  # fitted to one expression and applied to another.
  predict_with <- function(theta) {
    co <- list(intercept_bytes = theta[1], act_copies = theta[2],
               att_copies = theta[3], safety_factor = 1)
    vapply(seq_along(rows), function(i) {
      stages <- lapply(seq_len(n_stage), function(k) {
        list(name = "", act = act[i, k], att = att[i, k])
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

  sweep <- expand.grid(
    a = seq(baseline, max(baseline * 4, 4e9), length.out = 41),
    b = seq(1, 400, length.out = 60),
    cc = c(0, 0.05, 0.1, 0.25, 0.5, 1, 2)
  )
  vals <- mapply(function(a, b, cc) obj(c(a, b, cc)),
                 sweep$a, sweep$b, sweep$cc)
  seed <- as.numeric(sweep[which.min(vals), ])

  best <- tryCatch(
    stats::optim(seed, obj, method = "L-BFGS-B",
                 lower = c(baseline, 0.5, 0), upper = c(8e9, 1000, 100)),
    error = function(e) NULL
  )
  theta <- if (!is.null(best) && best$value < min(vals)) best$par else seed

  pred <- predict_with(theta)
  list(
    intercept_bytes = theta[1],
    act_copies = theta[2],
    att_copies = theta[3],
    # The guard must fail safe. Whatever least squares says, lift it until
    # no measured point sits above its own estimate: an over-estimate
    # costs a warning and an under-estimate costs a session.
    safety_factor = max(1, max(measured / pred)),
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
  cc <- list(intercept_bytes = co$theta[1], act_copies = co$theta[2],
             att_copies = co$theta[3], safety_factor = 1)
  p <- vapply(seq_along(test), function(i) {
    stages <- lapply(seq_len(n_stage), function(k) {
      list(name = "", act = a[i, k], att = b[i, k])
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
  ids <- names(cat_)[vapply(cat_, function(e) {
    identical(e$backend, backend) && identical(e$task, "classification")
  }, logical(1))]
  ids <- Filter(function(id) tabfound:::.model_is_downloaded(id), ids)
  if (!length(ids)) return(NULL)
  id <- ids[1]
  entry <- cat_[[id]]
  d <- tabfound:::.model_dir(id)
  list(id = id,
       dir = if (is.null(entry$subfolder)) d else file.path(d, entry$subfolder))
}

calibrate_backend <- function(backend, grid = "small", opts = list()) {
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
    got <- measure_point(point, m$dir, "classification", opts)
    tm <- terms_for(backend, point, .resolve_opts(backend, opts), config)
    got$act <- vapply(tm$stages, function(s) s$act %||% 0, numeric(1))
    got$att <- vapply(tm$stages, function(s) s$att %||% 0, numeric(1))
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
      stages <- lapply(seq_along(r$act), function(k) {
        list(name = "", act = r$act[k], att = r$att[k])
      })
      p <- tabfound:::.peak_from_terms(
        stages, r$weights_bytes, r$persistent_bytes, co, spmf = 1,
        n_estimators = 1
      )$peak
      under <- isTRUE(p * co$safety_factor < r$peak_bytes)
      cat(sprintf("  ctx=%5d p=%3d %10s   estimate %s%s\n",
                  r$n_context, r$n_features,
                  if (identical(r$status, "ok")) "at ceiling" else r$status,
                  tabfound:::.fmt_bytes(p * co$safety_factor),
                  if (under) "  <-- UNDER the observed floor" else ""))
    }
  }
}

# The stored measurements, in the shape `fit_coefs()` wants. The terms
# are recomputed rather than read back: they are a property of today's
# `peak_terms()`, and re-deriving them is exactly what makes a re-fit
# after a formula change mean something.
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
  opts <- .resolve_opts(backend, list(n_estimators = 1L))
  rows <- lapply(blob$points, function(r) {
    r <- lapply(r, function(v) if (is.list(v)) unlist(v) else v)
    tm <- terms_for(backend, r, opts, config)
    r$act <- vapply(tm$stages, function(s) s$act %||% 0, numeric(1))
    r$att <- vapply(tm$stages, function(s) s$att %||% 0, numeric(1))
    r$weights_bytes <- blob$weights_bytes
    r$persistent_bytes <- r$n_context * r$n_features * 8 * 2 +
      4 * (tm$persistent %||% 0)
    r
  })
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
      r
    })
  ), path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
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
      prep_state_factor = old$devices$cpu$float32$prep_state_factor %||% 1,
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
                              "--ensemble", "--grid", grid))
  if ("--all" %in% args || !length(backends)) backends <- list_backends()$name

  refit <- "--refit" %in% args
  ensemble <- "--ensemble" %in% args
  # Calibration runs one ensemble member, because members are sequential
  # and the estimator counts the transient once. `--estimators N` is how
  # that claim gets checked against a real run rather than assumed.
  extra_opts <- list()
  if ("--estimators" %in% args) {
    n <- as.integer(args[which(args == "--estimators") + 1L])
    extra_opts$n_estimators <- n
    args <- setdiff(args, as.character(n))
  }

  for (backend in backends) {
    cat("\n== ", backend, " ==\n", sep = "")
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
      if (!refit) {
        write_measurements(cal, file.path(HERE, "measurements",
                                          paste0(backend, ".json")))
      }
      next
    }
    baseline <- stats::median(
      vapply(ok, function(r) r$loaded_bytes %||% 0, numeric(1)), na.rm = TRUE
    )
    if (!isTRUE(is.finite(baseline))) baseline <- 0
    # Written before anything is reported: the measurements cost hours and
    # a formatting bug in the report must not be able to lose them.
    if (!refit) {
      write_measurements(cal, file.path(HERE, "measurements",
                                        paste0(backend, ".json")))
    }
    co <- fit_coefs(ok, baseline = baseline)
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
