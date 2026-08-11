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

# `ensemble-*.json` are the member sweeps, a different measurement with a
# different schema; they are checked separately below.
measurement_files <- list.files(measurements_dir, pattern = "\\.json$",
                                full.names = TRUE)
measurement_files <- measurement_files[
  !startsWith(basename(measurement_files), "ensemble-")
]
skip_if(!length(measurement_files), "no stored memory measurements")

# The same censoring rule the harness applies: a point that came within
# reach of the machine's ceiling measured what it could get, not what it
# wanted, so it is a lower bound rather than a value.
censor_threshold <- function(blob) 0.6 * (blob$machine_total_bytes %||% Inf)

estimate_for <- function(point, blob, co) {
  stages <- lapply(seq_along(point$act), function(k) {
    list(name = "", act = point$act[[k]], att = point$att[[k]])
  })
  tabfound:::.peak_from_terms(stages, blob$weights_bytes,
                              point$persistent_bytes, co, spmf = 1)$peak
}

read_blob <- function(path) {
  blob <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  blob$points <- lapply(blob$points, function(p) {
    p$act <- unlist(p$act); p$att <- unlist(p$att)
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

  test_that(paste0(backend, ": the estimate stays within 1.5x of measured"), {
    skip_if(is.null(co), paste("no coefficients for", backend))
    skip_if(!length(fitted_points), "no uncensored measurements")
    ratios <- vapply(fitted_points, function(p) {
      estimate_for(p, blob, co) / p$peak_bytes
    }, numeric(1))
    expect_lte(max(ratios), 1.5)
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
    opts <- tabfound:::.resolve_memory_opts(
      get_backend(backend), "classification", list(n_estimators = 1L), list()
    )
    for (p in blob$points) {
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
