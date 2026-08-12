# The shipped constants, against the measurements they were fitted to.
#
# Measured once by `inst/memory/calibrate.R`, asserted thereafter -- the
# parity fixtures' bargain, for a different kind of number. Nothing here
# needs weights or torch: the measurements carry the dimensions, the
# shapes the backend produced for them, and what the process actually
# cost.
#
# The one assertion that matters more than accuracy is direction. An
# estimate above the truth costs a warning. An estimate below it costs
# the session, because there is no error to catch when libtorch runs out.

skip_if_not_installed("jsonlite")

measurements_dir <- tabfound:::tabfound_file("memory", "measurements")
skip_if(!nzchar(measurements_dir) || !dir.exists(measurements_dir),
        "no stored memory measurements")

# `ensemble-*.json`, `spmf-*.json` and `chunks-*.json` are knob sweeps:
# one dimension held fixed while a setting varies, which is a different
# measurement with a different schema; they are checked separately below.
measurement_files <- list.files(measurements_dir, pattern = "\\.json$",
                                full.names = TRUE)
measurement_files <- measurement_files[
  !startsWith(basename(measurement_files), "ensemble-") &
    !startsWith(basename(measurement_files), "spmf-") &
    # `chunks-*.json` alone: those hold one point's dimensions at the
    # top level and vary a setting beneath, so there is nothing per-row
    # to replay. `<backend>-chunked.json` *is* a size grid -- measured
    # with stage chunking on, fitted together with the default one -- and
    # belongs here.
    !startsWith(basename(measurement_files), "chunks-")
]
skip_if(!length(measurement_files), "no stored memory measurements")

# The same censoring rule the harness applies: a point that came within
# reach of the machine's ceiling measured what it could get, not what it
# wanted, so it is a lower bound rather than a value.
censor_threshold <- function(blob) 0.6 * (blob$machine_total_bytes %||% Inf)

estimate_for <- function(point, blob, co) {
  # `res` is the whole-table term a chunked stage keeps outside its loop.
  # Older measurement files predate it; treat its absence as zero rather
  # than dropping it silently, which would make the estimate here differ
  # from the one the fit was graded on.
  ns <- length(point$act)
  res <- point$res %||% rep(0, ns)
  pre <- point$prepass %||% rep(0, ns)
  icl <- point$copies_icl %||% rep(FALSE, ns)
  stages <- lapply(seq_len(ns), function(k) {
    list(name = "", act = point$act[[k]], att = point$att[[k]],
         res = res[[k]], prepass = pre[[k]],
         copies = if (isTRUE(icl[[k]])) "icl" else "")
  })
  tabfound:::.peak_from_terms(stages, blob$weights_bytes,
                              point$persistent_bytes, co, spmf = 1)$peak
}

# The same assembly `estimate_for()` does, kept apart so a test can ask
# which term dominates without re-deriving it.
decompose_for <- function(point, blob, co) {
  ns <- length(point$act)
  res <- point$res %||% rep(0, ns)
  pre <- point$prepass %||% rep(0, ns)
  icl <- point$copies_icl %||% rep(FALSE, ns)
  stages <- lapply(seq_len(ns), function(k) {
    list(name = "", act = point$act[[k]], att = point$att[[k]],
         res = res[[k]], prepass = pre[[k]],
         copies = if (isTRUE(icl[[k]])) "icl" else "")
  })
  got <- tabfound:::.peak_from_terms(stages, blob$weights_bytes,
                                     point$persistent_bytes, co, spmf = 1)
  fixed <- co$intercept_bytes + blob$weights_bytes + point$persistent_bytes
  list(transient = got$transient, fixed = fixed,
       modelled = (fixed + got$transient) * co$safety_factor,
       floor = co$floor_bytes %||% 0)
}

read_blob <- function(path) {
  blob <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  blob$points <- lapply(blob$points, function(p) {
    p$act <- unlist(p$act); p$att <- unlist(p$att)
    if (!is.null(p$res)) p$res <- unlist(p$res)
    if (!is.null(p$prepass)) p$prepass <- unlist(p$prepass)
    if (!is.null(p$copies_icl)) p$copies_icl <- unlist(p$copies_icl)
    p
  })
  blob
}

`%||%` <- function(x, y) if (is.null(x)) y else x


for (path in measurement_files) {
  blob <- read_blob(path)
  backend <- blob$backend
  co <- tabfound:::.memory_coefs(backend, device = "cpu")
  thr <- censor_threshold(blob)
  fitted_points <- Filter(function(p) {
    identical(p$status, "ok") && isTRUE(p$peak_bytes < thr)
  }, blob$points)
  censored <- Filter(function(p) {
    !identical(p$status, "ok") || isTRUE(p$peak_bytes >= thr)
  }, blob$points)

  test_that(paste0(backend, ": estimates never fall below what was measured"), {
    skip_if(is.null(co), paste("no coefficients for", backend))
    skip_if(!length(fitted_points), "no uncensored measurements")
    for (p in fitted_points) {
      est <- estimate_for(p, blob, co)
      # The safety factor is set to make the tightest point exactly equal,
      # so this comparison sits on a float boundary by construction.
      expect_gte(est * (1 + 1e-9), p$peak_bytes)
    }
  })

  # The accuracy band is about the *slope* -- does the model track what
  # an activation costs -- so it is held to the points where the
  # activation is what was measured. Below a gigabyte or two the number
  # is the process: R, libtorch, and whichever arena the allocator took,
  # which is why the small end of a grid can fall by half when the input
  # doubles (TabPFN v2.5 at 8 features: 2.1, 1.1, 1.2, 1.9 GB across 800
  # to 6,400 rows, reproducible across three repeats each). No model fits
  # that, and a band that includes it grades the allocator rather than
  # the estimator. Those points still constrain the intercept, and the
  # never-below check above still covers every one of them -- which is
  # the property that actually protects a session.
  activation_dominated <- Filter(function(p) {
    d <- decompose_for(p, blob, co)
    # The transient has to be the larger term, and the model -- not the
    # floor -- has to be what the estimate actually returns. A point
    # sitting on the floor is being told what the backend costs to do
    # nothing, which is a true statement about the machine and no
    # statement at all about the scaling this band grades.
    isTRUE(d$transient > d$fixed) && isTRUE(d$modelled > d$floor)
  }, fitted_points)

  # Two separate properties, asserted separately, because rolling them
  # into one number said little about either. The *fit* is how far a
  # single slope over-predicts the activation, graded with the safety
  # factor divided back out. The *safety factor* is how far that fit then
  # has to be lifted so no measured point sits above its estimate -- and
  # since it is defined as the worst under-prediction, bounding it is
  # exactly bounding the other tail. Asserting both directions on the fit
  # would just be asserting the margin twice.
  raw_ratio <- function(p) {
    estimate_for(p, blob, co) / co$safety_factor / p$peak_bytes
  }

  test_that(paste0(backend, ": the fit stays within 1.5x above measured"), {
    skip_if(is.null(co), paste("no coefficients for", backend))
    skip_if(!length(activation_dominated),
            "no point where the transient is the larger term")
    ratios <- vapply(activation_dominated, raw_ratio, numeric(1))
    expect_lte(max(ratios), 1.5)
  })

  test_that(paste0(backend, ": the fail-safe margin stays bounded"), {
    skip_if(is.null(co), paste("no coefficients for", backend))
    expect_gte(co$safety_factor, 1)
    # 2.5 rather than 2 because of TabICL, at 2.25. Its size grid is
    # measured with stage chunking *off* -- that is its default -- and
    # the same constants then have to cover chunked runs, where the peak
    # is a different shape. The margin is what closes that gap, and the
    # real fix is a chunked calibration grid rather than a looser bound;
    # until then this records what the data costs rather than hiding it.
    expect_lte(co$safety_factor, 2.5)
  })

  test_that(paste0(backend, ": points that died estimate above the ceiling"), {
    skip_if(is.null(co), paste("no coefficients for", backend))
    died <- Filter(function(p) identical(p$status, "died"), censored)
    skip_if(!length(died), "nothing died on the calibration machine")
    for (p in died) {
      # The run got at least this far before the allocator gave up, so
      # anything less than that as an estimate is a false reassurance.
      expect_gt(estimate_for(p, blob, co) * (1 + 1e-9), p$peak_bytes)
    }
  })

  test_that(paste0(backend, ": the shipped constants are the measured ones"), {
    skip_if(is.null(co), paste("no coefficients for", backend))
    # A coefficient file still saying "anchor" while measurements exist
    # means a calibration was run and never written.
    expect_identical(co$source, "measured")
  })
}


# ---------------------------------------------------------------------------
# The ensemble sweeps
# ---------------------------------------------------------------------------

test_that("the shipped ensemble factor covers every member count measured", {
  sweeps <- list.files(measurements_dir, pattern = "^ensemble-.*\\.json$",
                       full.names = TRUE)
  skip_if(!length(sweeps), "no stored ensemble sweeps")
  for (path in sweeps) {
    blob <- jsonlite::fromJSON(path, simplifyVector = FALSE)
    co <- tabfound:::.memory_coefs(blob$backend, device = "cpu")
    skip_if(is.null(co), paste("no coefficients for", blob$backend))
    n <- vapply(blob$points, function(p) p$n_estimators, numeric(1))
    # The envelope has to sit at or above the shape it was fitted to, at
    # every member count, or the factor is not an envelope.
    env <- pmin(1 + co$ensemble_log2_factor * log2(n), co$ensemble_cap)
    expect_true(all(diff(env) >= 0), info = blob$backend)
    expect_gte(max(env), 1)
    expect_equal(env[n == 1][1], 1)
  }
})


# ---------------------------------------------------------------------------
# The save_peak_memory_factor sweeps
# ---------------------------------------------------------------------------

test_that("the shipped spmf floor never promises more than the sweep gave", {
  # This is the assertion that would have caught the modelling error the
  # sweep exists to fix. `1 + (act_copies - 1) / k` promised a factor of
  # 32 would cut Mitra's activation by 24x; it cuts it by 2.8x. Promising
  # a saving that does not arrive is the one direction a memory guard
  # must never err in, so the shipped floor has to sit at or above what
  # every measured factor implies.
  sweeps <- list.files(measurements_dir, pattern = "^spmf-.*\\.json$",
                       full.names = TRUE)
  skip_if(!length(sweeps), "no stored spmf sweeps")
  for (path in sweeps) {
    blob <- jsonlite::fromJSON(path, simplifyVector = FALSE)
    co <- tabfound:::.memory_coefs(blob$backend, device = "cpu")
    skip_if(is.null(co), paste("no coefficients for", blob$backend))
    ok <- Filter(function(p) identical(p$status, "ok"), blob$points)
    base <- Filter(function(p) isTRUE(p$spmf == 1L), ok)
    skip_if(!length(base), "no unchunked point in the sweep")

    # Above 1 is legal and meaningful: it says the factor costs more
    # than it saves on this backend, which TabICL measurably does.
    expect_gte(co$spmf_floor, 0)

    fixed <- co$intercept_bytes + (co$weights_fallback_bytes %||% 0)
    v0 <- base[[1]]$peak_bytes - fixed
    skip_if(!isTRUE(v0 > 0), "the unchunked point is below the fixed cost")
    for (p in Filter(function(p) p$spmf > 1L, ok)) {
      implied <- (p$peak_bytes - fixed) / v0
      modelled <- co$spmf_floor + (1 - co$spmf_floor) / p$spmf
      # A floor of 1 is "the factor buys nothing", and the sweep that
      # produced it is a run-to-run comparison of a quantity that varies
      # by a percent or two -- so a point measuring marginally *worse*
      # than unchunked is noise, not a promise broken. 5% of slack, and
      # only where nothing was promised in the first place.
      slack <- 1 + 1e-9
      expect_gte(modelled * slack, implied,
                 label = sprintf("%s at k=%d", blob$backend, p$spmf))
    }
  }
})


# ---------------------------------------------------------------------------
# The stage-chunk sweeps
# ---------------------------------------------------------------------------

test_that("the shipped constants cover the chunked points too", {
  # The size grids are measured in each backend's *default* chunking
  # configuration, so on a backend where that default is "off" the
  # constants have never seen a chunked run. These two points are the
  # only measurements of one, and the estimate has to sit above them for
  # the same reason it sits above every other measurement.
  sweeps <- list.files(measurements_dir, pattern = "^chunks-.*\\.json$",
                       full.names = TRUE)
  skip_if(!length(sweeps), "no stored chunk sweeps")
  for (path in sweeps) {
    blob <- jsonlite::fromJSON(path, simplifyVector = FALSE)
    co <- tabfound:::.memory_coefs(blob$backend, device = "cpu")
    skip_if(is.null(co), paste("no coefficients for", blob$backend))
    ok <- Filter(function(p) identical(p$status, "ok"), blob$points)
    skip_if(!length(ok), "every point in this sweep was censored")
    # The pre-pass is charged its own copy count, and it should come out
    # well below the forward's: it is one column-stage stack, not a whole
    # pipeline. TabPFN v3 measures 3.9 against 78.8, TabICL 71.2 against
    # 137.7.
    if (!is.null(co$prepass_copies)) {
      expect_gt(co$prepass_copies, 0)
      expect_lt(co$prepass_copies, co$act_copies)
    }
  }
})


# ---------------------------------------------------------------------------
# The formulas the constants belong to
# ---------------------------------------------------------------------------

# `act` and `att` in the measurements are what `peak_terms()` produced on
# the day of the measurement. The constants are constants *for those
# formulas*: change a token count and the fitted slope stops meaning
# anything, silently, because everything still runs. This is the test
# that says so -- it needs the weights, since a config is the only place
# the dimensions live.
test_that("peak_terms() still produces the shapes the constants assume", {
  checked <- 0L
  for (path in measurement_files) {
    blob <- read_blob(path)
    backend <- blob$backend
    id <- blob$model_id
    if (is.null(id) || !isTRUE(tabfound:::.model_is_downloaded(id))) next
    entry <- tabfound:::.model_catalog()[[id]]
    dir <- tabfound:::.model_dir(id)
    if (!is.null(entry$subfolder)) dir <- file.path(dir, entry$subfolder)
    config <- tabfound:::read_model_config(file.path(dir, "config.json"))
    for (p in blob$points) {
      # In the chunking the point was measured under, not the backend's
      # default: a grid run with `row_chunk_size` set is a different
      # shape, and asking for the default one here would compare it
      # against terms it was never fitted to.
      extra <- list(n_estimators = 1L)
      rc <- unlist(p$row_chunk_size); cc <- unlist(p$col_chunk_size)
      if (!is.null(rc) && !all(is.na(rc))) extra$row_chunk_size <- as.integer(rc)
      if (!is.null(cc) && !all(is.na(cc))) extra$col_chunk_size <- as.integer(cc)
      opts <- tabfound:::.resolve_memory_opts(
        get_backend(backend), "classification", extra, list()
      )
      tm <- tabfound:::.peak_terms_for(get_backend(backend), p$n_context,
                                       p$n_query, p$n_features, opts, config)
      act <- vapply(tm$stages, function(s) s$act, numeric(1))
      expect_equal(unname(act), unname(p$act),
                   info = sprintf("%s ctx=%d p=%d", backend, p$n_context,
                                  p$n_features))
      checked <- checked + 1L
    }
  }
  skip_if(checked == 0L, "no calibrated model's weights are on this machine")
  expect_gt(checked, 0L)
})
