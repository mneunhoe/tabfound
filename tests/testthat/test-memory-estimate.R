# The memory preflight, tested without a single checkpoint.
#
# Everything here runs off configs (see `helper-memory-configs.R`) and an
# injected `available`, so the assertions are about the estimator rather
# than about whatever the test machine happens to have free.

skip_if_not_installed("jsonlite")

peak <- function(cfg, n_context = MEM_FOLD$n_context,
                 n_query = MEM_FOLD$n_query,
                 n_features = MEM_FOLD$n_features,
                 available = MEM_IDLE, ...) {
  estimate_peak_memory(cfg, n_context, n_query, n_features,
                       available = available, ...)
}


# ---------------------------------------------------------------------------
# Probing the machine
# ---------------------------------------------------------------------------

test_that("vm_stat output parses to the reclaimable page classes", {
  txt <- c(
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)",
    "Pages free:                              100000.",
    "Pages active:                            900000.",
    "Pages inactive:                           50000.",
    "Pages speculative:                        10000.",
    "Pages wired down:                        400000.",
    "Pages purgeable:                           5000."
  )
  # free + inactive + speculative + purgeable, at 16 kB a page. Active
  # and wired pages are not reclaimable and must not be counted.
  expect_equal(tabfound:::.parse_vm_stat(txt),
               (100000 + 50000 + 10000 + 5000) * 16384)
})

test_that("vm_stat parsing survives output it does not recognise", {
  expect_true(is.na(tabfound:::.parse_vm_stat("something else entirely")))
})

test_that("meminfo prefers MemAvailable and falls back when it is absent", {
  lines <- c("MemTotal:       16384000 kB",
             "MemFree:          512000 kB",
             "MemAvailable:    8192000 kB",
             "Cached:          4096000 kB")
  got <- tabfound:::.parse_meminfo(lines)
  expect_equal(unname(got[["total"]]), 16384000 * 1024)
  expect_equal(unname(got[["available"]]), 8192000 * 1024)

  old <- tabfound:::.parse_meminfo(lines[c(1, 2, 4)])
  expect_equal(unname(old[["available"]]), (512000 + 4096000) * 1024)
})


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

test_that("every backend ships coefficients and describes its scaling", {
  for (nm in list_backends()$name) {
    bk <- get_backend(nm)
    expect_true(is.function(bk$peak_terms), info = nm)
    co <- tabfound:::.memory_coefs(nm, device = "cpu")
    expect_false(is.null(co), info = nm)
    expect_true(is.finite(co$act_copies) && co$act_copies >= 1, info = nm)
    expect_true(is.finite(co$safety_factor) && co$safety_factor >= 1, info = nm)
  }
})

test_that("coefficients round-trip out of the shipped JSON", {
  co <- tabfound:::.memory_coefs("mitra", device = "cpu")
  raw <- jsonlite::fromJSON(
    tabfound:::tabfound_file("memory", "coefs", "mitra.json")
  )
  expect_equal(co$act_copies, raw$devices$cpu$float32$act_copies)
  expect_equal(co$intercept_bytes, raw$devices$cpu$float32$intercept_bytes)
  expect_identical(co$source, "measured")
})

test_that("an uncalibrated device is refused, not guessed at", {
  # A CUDA estimate built from CPU constants would be worse than none:
  # this is a guard, and a confidently wrong number is the failure mode
  # it exists to prevent.
  expect_null(tabfound:::.memory_coefs("mitra", device = "cuda"))
  expect_error(peak(mem_config_mitra(), device = "cuda"),
               "No memory constants")
})


# ---------------------------------------------------------------------------
# Shape of the estimate
# ---------------------------------------------------------------------------

test_that("the estimate is a breakdown that adds up", {
  e <- peak(mem_config_tabicl())
  expect_s3_class(e, "tabfound_memory_estimate")
  expect_gte(e$total_peak_bytes,
             e$weights_bytes + e$persistent_bytes + e$transient_bytes)
  expect_true(all(is.finite(c(e$weights_bytes, e$persistent_bytes,
                              e$transient_bytes, e$total_peak_bytes))))
  expect_identical(e$backend, "tabicl")
})

test_that("the transient peak is the largest stage, not the sum of them", {
  # At one member, where the ensemble multiplier is 1 and the transient
  # is the largest stage unmodified.
  e <- peak(mem_config_tabicl(), n_estimators = 1)
  expect_length(e$stage_bytes, 3L)
  expect_equal(e$transient_bytes, unname(max(e$stage_bytes)))
  expect_lt(e$transient_bytes, sum(e$stage_bytes))
})

test_that("weights come exactly from the config's recorded shapes", {
  cfg <- mem_config_tabpfn25()
  e <- peak(cfg)
  expected <- (64 * 192 + 3 * 3 * 64 * 192) * 4
  expect_equal(e$weights_bytes, expected)
  expect_identical(e$weights_source, "config shapes")
})

test_that("a config with no shapes falls back and says so", {
  e <- peak(mem_config_mitra())
  expect_identical(e$weights_source, "backend constant (approximate)")
  expect_equal(e$weights_bytes,
               tabfound:::.memory_coefs("mitra", "cpu")$weights_fallback_bytes)
})

test_that("printing reports the breakdown, the verdict and the advice", {
  # cli writes through its own connection, so capture it the cli way.
  ok <- cli::cli_fmt(print(peak(mem_config_tabicl(), 200, 100, 10)))
  expect_true(any(grepl("memory preflight", ok)))
  expect_true(any(grepl("transient", ok)))
  expect_true(any(grepl("OK", ok)))

  no <- cli::cli_fmt(print(peak(mem_config_mitra())))
  expect_true(any(grepl("EXCEEDS", no)))
  expect_true(any(grepl("n_context", no)))
})


# ---------------------------------------------------------------------------
# Monotonicity: more of anything is never less memory
# ---------------------------------------------------------------------------

test_that("the estimate rises with every dimension, on every backend", {
  configs <- list(tabpfn = mem_config_tabpfn25(),
                  tabpfn3 = mem_config_tabpfn3(),
                  tabicl = mem_config_tabicl(),
                  tabfm = mem_config_tabfm(),
                  mitra = mem_config_mitra())
  for (nm in names(configs)) {
    cfg <- configs[[nm]]
    base <- peak(cfg, 2000, 500, 50)$total_peak_bytes
    expect_gt(peak(cfg, 4000, 500, 50)$total_peak_bytes, base)   # context
    expect_gt(peak(cfg, 2000, 500, 90)$total_peak_bytes, base)   # features
    # Query rows only count up to the chunk size, so lift that too.
    expect_gt(peak(cfg, 2000, 900, 50,
                   predict_chunk_size = 4096)$total_peak_bytes,
              peak(cfg, 2000, 500, 50,
                   predict_chunk_size = 4096)$total_peak_bytes)
  }
})


# ---------------------------------------------------------------------------
# The knobs
# ---------------------------------------------------------------------------

test_that("ensemble members raise the peak far less than they multiply", {
  # The naive reading -- eight members, eight times the memory -- would
  # make every TabFM estimate absurd. The measured growth is closer to
  # log2 of the member count; see `.peak_from_terms()`.
  one   <- peak(mem_config_tabicl(), n_estimators = 1)
  eight <- peak(mem_config_tabicl(), n_estimators = 8)
  expect_gt(eight$transient_bytes, one$transient_bytes)
  expect_lt(eight$transient_bytes, 4 * one$transient_bytes)
  expect_gt(eight$persistent_bytes, one$persistent_bytes)
})

test_that("kv_cache moves bytes to persistent, and scales with members", {
  off <- peak(mem_config_tabicl(), kv_cache = FALSE, n_estimators = 4)
  on  <- peak(mem_config_tabicl(), kv_cache = TRUE,  n_estimators = 4)
  expect_gt(on$persistent_bytes, off$persistent_bytes)
  expect_equal(on$transient_bytes, off$transient_bytes)

  on1 <- peak(mem_config_tabicl(), kv_cache = TRUE, n_estimators = 1)
  expect_gt(on$persistent_bytes, on1$persistent_bytes)
})

test_that("Mitra's cache costs gigabytes, and the estimate says so", {
  # Keys and values for every layer, each the full features x rows x dim.
  # This is why the cache is off by default, and why a preflight that
  # ignored it would understate a cached run by an order of magnitude.
  off <- peak(mem_config_mitra(), 1500, 200, 32, kv_cache = FALSE)
  on  <- peak(mem_config_mitra(), 1500, 200, 32, kv_cache = TRUE)
  # Keys and values for twelve layers, each the full features x rows x
  # dim. Asserted in gigabytes rather than as a verdict flip: the verdict
  # boundary moves with every recalibration, the order of magnitude does
  # not.
  expect_gt(on$persistent_bytes - off$persistent_bytes, 2e9)
  expect_gt(on$persistent_bytes, 10 * off$persistent_bytes)
  expect_gt(on$total_peak_bytes, off$total_peak_bytes)
  expect_true(any(grepl("kv_cache", on$suggestions)))
})

test_that("save_peak_memory_factor lowers the transient term", {
  plain   <- peak(mem_config_tabpfn3())
  chunked <- peak(mem_config_tabpfn3(), save_peak_memory_factor = 4)
  expect_lt(chunked$transient_bytes, plain$transient_bytes)
  # It chunks the temporaries, not the state tensor they are made from,
  # so it can never take the transient to zero.
  expect_gt(chunked$transient_bytes, plain$transient_bytes / 4)
})

test_that("predict_chunk_size caps the query rows that are ever resident", {
  small <- peak(mem_config_tabicl(), 2000, 100000, 90, predict_chunk_size = 512)
  big   <- peak(mem_config_tabicl(), 2000, 100000, 90,
                predict_chunk_size = 8192)
  expect_lt(small$transient_bytes, big$transient_bytes)
  # 100,000 query rows with a 512-row chunk cost no more than 512 do.
  expect_equal(
    small$transient_bytes,
    peak(mem_config_tabicl(), 2000, 512, 90,
         predict_chunk_size = 512)$transient_bytes
  )
})

test_that("TabFM honours the caps that shrink what each member sees", {
  base <- peak(mem_config_tabfm(), 6426, 714, 200)
  cols <- peak(mem_config_tabfm(), 6426, 714, 200, max_num_features = 50)
  rows <- peak(mem_config_tabfm(), 6426, 714, 200, max_num_rows = 1000)
  expect_lt(cols$transient_bytes, base$transient_bytes)
  expect_lt(rows$transient_bytes, base$transient_bytes)
})


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
#
# These began as the hand-off's must-pass cases, written from recollected
# runs before anything was measured. `inst/memory/calibrate.R` has since
# measured the same backends on the same machine, and disagrees with
# several of them -- TabICL at fold size costs 24.5 GB with one ensemble
# member, not the ~3 GB the anchored constants implied. Where measurement
# and recollection conflict, these follow the measurement, and
# `inst/memory/README.md` records the conflict rather than papering over
# it.

test_that("the backends rank the way their architectures say they should", {
  # Mitra never narrows the table: its state stays rows x features x dim
  # from first layer to last, where TabICL and TabPFN v3 carry one vector
  # per feature *group* and then per row. That ordering is the most
  # robust thing the estimator knows, and it should survive any
  # recalibration.
  dims <- list(2000, 714, 90)
  peak_of <- function(cfg) {
    do.call(estimate_peak_memory,
            c(list(cfg), dims, list(available = MEM_IDLE)))$total_peak_bytes
  }
  expect_gt(peak_of(mem_config_mitra()), peak_of(mem_config_tabicl()))
  expect_gt(peak_of(mem_config_tabicl()), peak_of(mem_config_tabpfn3()))
})

test_that("fold-sized runs on a 48 GB machine are not called comfortable", {
  # Measured, one ensemble member, 6,426 x 90: TabICL 24.5 GB, TabPFN v3
  # 24.7 GB, TabPFN v2.6 19.4 GB. None of that is comfortable against
  # 40 GB free, and the run that prompted this work was killed at these
  # dimensions when a neighbour took 7 GB.
  for (cfg in list(mem_config_tabicl(), mem_config_tabpfn3(),
                   mem_config_tabpfn25())) {
    e <- peak(cfg, 6426, 714, 90, available = MEM_IDLE)
    expect_false(e$verdict == "ok")
  }
})

test_that("Mitra at fold size is refused outright", {
  expect_identical(peak(mem_config_mitra(), 6426)$verdict, "exceeds")
})

test_that("small tables stay comfortable on every backend", {
  # The other half of not crying wolf: a table the size these models were
  # designed for has to come back clean, or the guard is noise.
  for (cfg in list(mem_config_tabpfn25(), mem_config_tabpfn3(),
                   mem_config_tabicl())) {
    expect_identical(peak(cfg, 200, 100, 10, available = MEM_IDLE)$verdict,
                     "ok")
  }
})

test_that("the verdict comes from available memory, not from total", {
  # The distinction the SIGKILL turned on: same run, same peak, different
  # answer depending on what the machine has free right now.
  cfg <- mem_config_tabpfn3()
  idle <- peak(cfg, 1600, 500, 32, available = MEM_IDLE, total = 48e9)
  busy <- peak(cfg, 1600, 500, 32, available = MEM_BUSY, total = 48e9)
  expect_identical(idle$verdict, "ok")
  expect_false(busy$verdict == "ok")
  expect_equal(idle$total_peak_bytes, busy$total_peak_bytes)
})

test_that("an ensemble raises the estimate, sub-linearly", {
  # Measured on TabICL and TabFM: members run sequentially, so the live
  # set is one member's, but the resident high-water mark climbs anyway.
  # It must climb -- and it must not climb like `n_estimators x`.
  one   <- peak(mem_config_tabicl(), 1600, 500, 32, n_estimators = 1)
  eight <- peak(mem_config_tabicl(), 1600, 500, 32, n_estimators = 8)
  expect_gt(eight$transient_bytes, one$transient_bytes)
  expect_lt(eight$transient_bytes, 8 * one$transient_bytes)
})

test_that("an unprobeable machine gives no verdict rather than a wrong one", {
  e <- peak(mem_config_tabicl(), available = NA_real_)
  expect_identical(e$verdict, "unknown")
  expect_length(e$suggestions, 0L)
})


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

test_that("suggestions name the term that actually dominates", {
  mitra <- peak(mem_config_mitra(), 6426)
  expect_true(any(grepl("n_context", mitra$suggestions)))
  expect_true(any(grepl("Mitra", mitra$suggestions)))

  # The v3 port's missing stage chunking is a known gap; when v3 is the
  # one being refused, the estimate has to say so.
  v3 <- peak(mem_config_tabpfn3(), 200000, 714, 500, available = 4e9)
  expect_false(v3$verdict == "ok")
  expect_true(any(grepl("chunking", v3$suggestions)))
})

test_that("a comfortable run is not given advice it does not need", {
  expect_length(peak(mem_config_tabpfn3(), 200, 100, 10)$suggestions, 0L)
})


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------

test_that("the envelope is the largest context that still earns the verdict", {
  # The boundary, checked from both sides: the answer fits and one row
  # more does not.
  env <- memory_envelope(mem_config_tabicl(), n_features = 50,
                         available = 32e9, n_query = 200)
  n <- env$max_context
  expect_gt(n, 0)
  at   <- peak(mem_config_tabicl(), n,     200, 50, available = 32e9)
  over <- peak(mem_config_tabicl(), n + 1, 200, 50, available = 32e9)
  expect_identical(at$verdict, "ok")
  expect_false(over$verdict == "ok")
})

test_that("a looser verdict admits a larger table", {
  ok <- memory_envelope(mem_config_tabicl(), 50, 32e9, n_query = 200)
  tight <- memory_envelope(mem_config_tabicl(), 50, 32e9, n_query = 200,
                           verdict = "tight")
  expect_gt(tight$max_context, ok$max_context)
})

test_that("the envelope shrinks as columns and members grow", {
  wide <- memory_envelope(mem_config_tabicl(), c(10, 100), 32e9,
                          n_query = 200)
  expect_gt(wide$max_context[1], wide$max_context[2])

  one   <- memory_envelope(mem_config_tabicl(), 50, 32e9, n_query = 200,
                           n_estimators = 1)
  eight <- memory_envelope(mem_config_tabicl(), 50, 32e9, n_query = 200,
                           n_estimators = 8)
  expect_gt(one$max_context, eight$max_context)
})

test_that("both axes may vary at once, giving one row per combination", {
  env <- memory_envelope(mem_config_tabicl(), n_features = c(10, 50),
                         available = c(16e9, 32e9), n_query = 200)
  expect_equal(nrow(env), 4L)
  expect_named(env, c("n_features", "available_bytes", "max_context",
                      "peak_bytes"))
  # More memory never buys less table.
  for (p in c(10, 50)) {
    rows <- env[env$n_features == p, ]
    expect_false(is.unsorted(rows$max_context[order(rows$available_bytes)]))
  }
})

test_that("a machine too small for even one row reports zero, not an error", {
  env <- memory_envelope(mem_config_tabfm(), n_features = 50,
                         available = 2e9, n_query = 200)
  expect_equal(env$max_context, 0)
  expect_true(is.na(env$peak_bytes))
})

test_that("the envelope works from the shipped architecture table alone", {
  # The pre-download case: no weights, no config on disk.
  skip_if(is.null(tabfound:::.shipped_architecture("tabicl-v2-classifier")),
          "no shipped architecture table")
  withr::local_options(tabfound.home = tempfile())
  env <- memory_envelope("tabicl-v2-classifier", n_features = 50,
                         available = 32e9, n_query = 200)
  expect_gt(env$max_context, 0)
})


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

test_that("a fit with nothing to predict yet is still answerable", {
  # What the guard asks at `fit()` time before any query rows exist.
  e <- peak(mem_config_tabicl(), 1000, 0, 20)
  expect_true(is.finite(e$total_peak_bytes))
  expect_gt(e$total_peak_bytes, 0)
  expect_lt(e$transient_bytes,
            peak(mem_config_tabicl(), 1000, 500, 20)$transient_bytes)
})

test_that("a single row and a single column do not divide by zero", {
  for (cfg in list(mem_config_tabicl(), mem_config_tabpfn25(),
                   mem_config_tabpfn3(), mem_config_mitra(),
                   mem_config_tabfm())) {
    e <- peak(cfg, 1, 1, 1)
    expect_true(is.finite(e$total_peak_bytes))
    expect_gt(e$total_peak_bytes, 0)
  }
})

test_that("dimensions far past anything runnable stay finite", {
  # The answer is useless but it has to be a number: a verdict of
  # "exceeds" is what a user asking about a million rows should get, not
  # an `NaN` or an error from the guard.
  e <- peak(mem_config_tabicl(), 1e6, 1e5, 500)
  expect_true(is.finite(e$total_peak_bytes))
  expect_identical(e$verdict, "exceeds")
})

test_that("regressors are estimated as well as classifiers", {
  cfg <- mem_config_tabicl()
  cfg$head <- "regressor"
  cfg$max_classes <- 0
  cfg$num_quantiles <- 999
  e <- estimate_peak_memory(cfg, 1000, 200, 20, task = "regression",
                            available = MEM_IDLE)
  expect_identical(e$task, "regression")
  expect_gt(e$total_peak_bytes, 0)
})

test_that("an artifact directory is accepted, and its weights read exactly", {
  id <- "tabicl-v2-classifier"
  skip_if(!tabfound:::.model_is_downloaded(id), "artifacts not downloaded")
  dir <- tabfound:::.model_dir(id)
  from_dir <- estimate_peak_memory(dir, 1000, 200, 20, available = MEM_IDLE)
  from_id  <- estimate_peak_memory(id, 1000, 200, 20, available = MEM_IDLE)
  expect_equal(from_dir$total_peak_bytes, from_id$total_peak_bytes)
  expect_identical(from_dir$weights_source, "config shapes")
})


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

test_that("bad dimensions are refused", {
  expect_error(peak(mem_config_tabicl(), n_context = -1), "non-negative")
  expect_error(peak(mem_config_tabicl(), n_features = NA), "non-negative")
})

test_that("a model id with nothing downloaded says what to do about it", {
  # Whichever ids this machine happens to be missing: the preflight must
  # refuse rather than start a multi-gigabyte download to answer a
  # question about whether the run would fit.
  ids <- list_models()
  missing <- ids$id[!ids$downloaded]
  skip_if(!length(missing), "every catalogue model is downloaded here")
  expect_error(
    estimate_peak_memory(missing[1], 1000, 100, 10),
    "No local artifacts"
  )
})
