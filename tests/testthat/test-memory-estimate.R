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
  # Four now, not three: the summary pre-pass is its own stage, because
  # a row-chunked run cannot start until it finishes and it is bounded by
  # the *column* chunk rather than the row one.
  expect_length(e$stage_bytes, 4L)
  expect_equal(e$transient_bytes, unname(max(e$stage_bytes)))
  expect_lt(e$transient_bytes, sum(e$stage_bytes))
  # And it is zero unless the forward is chunked, since an unchunked pass
  # computes its summaries inline rather than in a pass of its own.
  expect_identical(unname(e$stage_bytes[["column summaries"]]), 0)
  chunked <- peak(mem_config_tabicl(), n_estimators = 1,
                  row_chunk_size = 512L)
  expect_gt(chunked$stage_bytes[["column summaries"]], 0)
})

test_that("the TabPFN v3 retrieval decoder is a counted stage of its own", {
  e <- peak(mem_config_tabpfn3(), 50000, 1024, 20, n_estimators = 1,
            n_classes = 10)
  dec <- e$stage_bytes[["retrieval decoder"]]
  # Three `(H, 128, N)` score tensors, the keys and a one-hot ten wide, at
  # four bytes and no fitted multiple: the copies are in the count.
  expect_equal(unname(dec), 4 * 6 * 50000 * (3 * 128 + 64 + 10))
  # Bounded by the block of test rows, not by how many there are.
  more <- peak(mem_config_tabpfn3(), 50000, 4096, 20, n_estimators = 1,
               n_classes = 10)
  expect_equal(more$stage_bytes[["retrieval decoder"]], dec)
  # The one-hot is narrowed to the classes present, so more classes cost
  # more, up to the head's width.
  wide <- peak(utils::modifyList(mem_config_tabpfn3(), list(max_num_classes = 160)),
               50000, 1024, 20, n_estimators = 1, n_classes = 500)
  expect_equal(unname(wide$stage_bytes[["retrieval decoder"]]),
               4 * 6 * 50000 * (3 * 128 + 64 + 160))

  # A regressor has no decoder to count.
  reg <- utils::modifyList(mem_config_tabpfn3(), list(head = "regressor"))
  expect_false("retrieval decoder" %in%
                 names(peak(reg, 50000, 1024, 20, task = "regression")$stage_bytes))
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

test_that("the printed lines add up to the printed peak", {
  # The process intercept, safety factor and floor make the peak more than
  # weights + persistent + transient; the printout names the difference
  # rather than leaving the reader to wonder where 13 GB went.
  e <- peak(mem_config_tabfm(), 60, 1, 2)
  extra <- e$total_peak_bytes -
    (e$weights_bytes + e$persistent_bytes + e$transient_bytes)
  expect_gt(extra, 0)
  out <- cli::cli_fmt(print(e))
  line <- grep("overhead", out, value = TRUE)
  expect_length(line, 1L)
  expect_match(line, tabfound:::.fmt_bytes(extra), fixed = TRUE)
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
  #
  # How much less is measured, and depends on where. TabICL's sweep at
  # 800 x 32 climbs to 4.1x the modelled one-member transient by eight
  # members -- a near-constant 2 GB of allocator growth on top of a small
  # transient -- while the same sweep at 4,000 x 50 moves by 3% out to 32
  # members (`measurements/ensemble-tabicl-4000x50.json`). The shipped
  # factor is the small point's, because it is the one the large point
  # does not contradict.
  one   <- peak(mem_config_tabicl(), n_estimators = 1)
  eight <- peak(mem_config_tabicl(), n_estimators = 8)
  expect_gt(eight$transient_bytes, one$transient_bytes)
  expect_lt(eight$transient_bytes, 0.6 * 8 * one$transient_bytes)
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

test_that("a TabPFN cache built per call is one member's, a kept one every member's", {
  # The TabPFN predictors build a member's cache, run it over every chunk
  # and drop it before the next, so `n_estimators` multiplies the cache
  # only when `tabfound_cache()` has kept all of them on the model.
  for (cfg in list(mem_config_tabpfn25(), mem_config_tabpfn3())) {
    off   <- peak(cfg, kv_cache = FALSE, n_estimators = 8)
    one   <- peak(cfg, kv_cache = TRUE,  n_estimators = 1)
    eight <- peak(cfg, kv_cache = TRUE,  n_estimators = 8)
    kept  <- peak(cfg, kv_cache = TRUE,  n_estimators = 8, kv_cache_kept = TRUE)
    cache_one <- one$persistent_bytes - peak(cfg, n_estimators = 1)$persistent_bytes
    expect_gt(cache_one, 0)
    expect_equal(eight$persistent_bytes - off$persistent_bytes, cache_one)
    expect_equal(kept$persistent_bytes - off$persistent_bytes, 8 * cache_one)
  }
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

test_that("save_peak_memory_factor lowers the transient where it measurably does", {
  # What the factor buys is a *measured* per-backend constant now, not an
  # assumed `1 + (act_copies - 1) / k`. Mitra's sweep says a factor of 32
  # takes its activation to 0.65 of the unchunked figure; the estimate
  # has to move with it.
  plain   <- peak(mem_config_mitra())
  chunked <- peak(mem_config_mitra(), save_peak_memory_factor = 4)
  expect_lt(chunked$transient_bytes, plain$transient_bytes)
  # It chunks the temporaries, not the state tensor they are made from,
  # so it can never take the transient to zero.
  expect_gt(chunked$transient_bytes, plain$transient_bytes / 4)
})

test_that("it promises nothing where the sweep found nothing", {
  # TabPFN v3's stage chunking has already taken the transient away by
  # the time the factor could act on it: measured across k = 1..32 at
  # 4,000 x 50 the peak moves from 5.8 GB to 5.1 GB, which is noise.
  # Its fitted floor is 1.005, so the estimate must not fall.
  expect_gte(tabfound:::.memory_coefs("tabpfn3", "cpu")$spmf_floor, 1)
  plain   <- peak(mem_config_tabpfn3())
  chunked <- peak(mem_config_tabpfn3(), save_peak_memory_factor = 8)
  expect_gte(chunked$transient_bytes, plain$transient_bytes)
})

test_that("a factor that costs more than it saves is modelled as costing", {
  # TabICL's in-context stage never materialises its scores, so there is
  # no large transient to chunk and the loop's own temporaries dominate:
  # the sweep measures the factor making it about a fifth *worse*. A
  # floor pinned at 1 would report that as neutral, which would send a
  # user reaching for a knob that hurts them.
  expect_gt(tabfound:::.memory_coefs("tabicl", "cpu")$spmf_floor, 1)
  plain   <- peak(mem_config_tabicl())
  chunked <- peak(mem_config_tabicl(), save_peak_memory_factor = 8)
  expect_gt(chunked$transient_bytes, plain$transient_bytes)
})

test_that("an unswept backend is assumed to gain nothing from the factor", {
  # The default has to be the pessimistic one. A backend nobody has
  # swept must not be promised a saving on the strength of a formula.
  for (nm in list_backends()$name) {
    co <- tabfound:::.memory_coefs(nm, device = "cpu")
    expect_gte(co$spmf_floor, 0)
  }
  # And the default, for a coefficient file written before the sweep
  # existed, is "no saving" rather than "some saving".
  cfg <- withr::local_tempdir()
  dir.create(file.path(cfg, "memory", "coefs"), recursive = TRUE)
  jsonlite::write_json(
    list(backend = "stub", source = "test",
         devices = list(cpu = list(float32 = list(act_copies = 10)))),
    file.path(cfg, "memory", "coefs", "stub.json"), auto_unbox = TRUE)
  local_mocked_bindings(
    tabfound_file = function(...) file.path(cfg, ...),
    .package = "tabfound"
  )
  expect_equal(tabfound:::.memory_coefs("stub", "cpu")$spmf_floor, 1)
})

test_that("col_chunk_size bounds the pre-pass and nothing else", {
  # The two axes bound different things, and a test that cannot tell them
  # apart would pass on a model that had wired them together. The row
  # chunk bounds the forward loop; the column chunk bounds the summary
  # pre-pass the loop cannot start without.
  cfg <- mem_config_tabicl()
  rows_only <- peak(cfg, 20000, 500, 200, row_chunk_size = 2048L)
  both      <- peak(cfg, 20000, 500, 200, row_chunk_size = 2048L,
                    col_chunk_size = 8L)
  expect_gt(rows_only$stage_bytes[["column summaries"]],
            both$stage_bytes[["column summaries"]])
  # The forward loop's own stages are untouched by the column chunk.
  expect_equal(rows_only$stage_bytes[["column embedding"]],
               both$stage_bytes[["column embedding"]])
  expect_lt(both$total_peak_bytes, rows_only$total_peak_bytes)

  # And with no row chunking there is no pre-pass to bound, so the column
  # chunk changes nothing at all.
  a <- peak(cfg, 20000, 500, 200, row_chunk_size = NULL)
  b <- peak(cfg, 20000, 500, 200, row_chunk_size = NULL, col_chunk_size = 8L)
  expect_equal(a$total_peak_bytes, b$total_peak_bytes)
})

test_that("TabICL is sized at one token per column plus the CLS slots", {
  # `group_cells()` stacks cyclic shifts as channels, so all p columns
  # survive the grouping, and the row_num_cls sentinel columns ride along
  # through the column stage. Sizing it as ceiling(p / 3) tokens is what
  # the old fit's act_copies of 141 was quietly absorbing.
  cfg <- mem_config_tabicl()
  unchunked <- list(n_estimators = 1, row_chunk_size = NULL)
  unchunked["row_chunk_size"] <- list(NULL)
  tm <- tabicl_peak_terms(4000, 200, 10, unchunked, cfg)
  st <- stats::setNames(tm$stages, vapply(tm$stages, `[[`, "", "name"))
  n <- 4000 + 200
  expect_equal(st[["column embedding"]]$act, n * (10 + 4) * 128)
  expect_equal(st[["row interaction"]]$act, n * (10 + 4) * 128)
  expect_equal(st[["in-context learning"]]$act, n * 4 * 128)

  # The row-chunked residue: the grouped input, three channels per token.
  ch <- tabicl_peak_terms(4000, 200, 10,
                          list(n_estimators = 1, row_chunk_size = 1024L), cfg)
  expect_equal(ch$stages[[2]]$res, n * ((10 + 4) * 3 + 4 * 128))
})

test_that("the pre-pass is charged its own copy count, not the forward's", {
  # Reusing `act_copies` here put TabICL's chunked estimate at 112 GB
  # against 36.9 GB measured. It is a narrower operation than the forward
  # that constant was fitted to -- one column-stage stack, not a whole
  # pipeline -- and on v3 the sweep says so: 3.9 against 78.8.
  for (nm in c("tabicl", "tabpfn3")) {
    co <- tabfound:::.memory_coefs(nm, device = "cpu")
    expect_false(is.null(co$prepass_copies), info = nm)
    expect_gt(co$prepass_copies, 0)
  }
  expect_lt(tabfound:::.memory_coefs("tabpfn3", device = "cpu")$prepass_copies,
            tabfound:::.memory_coefs("tabpfn3", device = "cpu")$act_copies)
  # TabICL's pre-pass is its whole column stage over every context row:
  # the same stack, at the same width, as the column stage of an
  # unchunked forward. Once both are sized at one token per column
  # (#78) they come out at the same order -- 34 against 28.5 -- and the
  # 71.2-against-137.7 split the old token count produced was the
  # forward's term absorbing a threefold undercount.
  co <- tabfound:::.memory_coefs("tabicl", device = "cpu")
  expect_lt(co$prepass_copies, 2 * co$act_copies)
  # Unswept, it falls back to the forward's count -- which over-states
  # the pre-pass and is therefore the right way to be wrong.
  co <- tabfound:::.memory_coefs("mitra", device = "cpu")
  expect_null(co$prepass_copies)
})

test_that("suggest_chunk_sizes reaches for the column axis when rows run out", {
  cfg <- mem_config_tabicl()
  roomy <- suggest_chunk_sizes(cfg, n_context = 6426, n_features = 90,
                               available = 40e9)
  expect_true(roomy$feasible)
  # A table wide enough that the pre-pass is what binds gets a column
  # chunk as well as a row one.
  wide <- suggest_chunk_sizes(cfg, n_context = 20000, n_features = 400,
                              available = 40e9, verdict = "tight")
  expect_true(is.na(roomy$col_chunk_size) ||
                roomy$col_chunk_size >= wide$col_chunk_size)
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
  # Re-measured 2026-08-11, one ensemble member, 6,426 x 90 -- the
  # Muchlinski fold that prompted this whole preflight, and which was
  # killed at these dimensions when a neighbour took 7 GB:
  #
  #   TabICL 25.7 GB, TabPFN v2.5 19.7 GB, TabPFN v2.6 12.7 GB
  #
  # None of that is comfortable against 40 GB free.
  for (cfg in list(mem_config_tabicl(), mem_config_tabpfn25())) {
    e <- peak(cfg, 6426, 714, 90, available = MEM_IDLE)
    expect_false(e$verdict == "ok")
  }
})

test_that("TabPFN v3 has earned its way out of that list", {
  # The same fold on the same machine measured 24.7 GB before v3 carried
  # the reference's stage chunking and before the layer stacks let go of
  # their intermediates; it now measures 8.3 GB. That is the difference
  # between a run the guard has to warn about and one it should wave
  # through, and it is the single clearest thing the chunking bought.
  e <- peak(mem_config_tabpfn3(), 6426, 714, 90, available = MEM_IDLE)
  expect_identical(e$verdict, "ok")
  expect_lt(e$total_peak_bytes, 20e9)

  # Turning the chunking off has to put it back where it was: a knob that
  # changes nothing when you turn it off is not the knob doing the work.
  off <- peak(mem_config_tabpfn3(), 6426, 714, 90, available = MEM_IDLE,
              row_chunk_size = NULL)
  expect_gt(off$total_peak_bytes, e$total_peak_bytes)
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

  # v3's stage chunking is the knob that changes how the peak *grows*
  # with rows, where `save_peak_memory_factor` only changes what it grows
  # from -- so when v3 is the one being refused, the estimate has to
  # offer it, and offer it first of the two.
  v3 <- peak(mem_config_tabpfn3(), 200000, 714, 500, available = 4e9)
  expect_false(v3$verdict == "ok")
  row_at <- grep("row_chunk_size", v3$suggestions)
  spmf_at <- grep("save_peak_memory_factor", v3$suggestions)
  expect_length(row_at, 1L)
  expect_true(row_at < spmf_at)

  # Turning it off is the one setting worth naming as such.
  off <- peak(mem_config_tabpfn3(), 200000, 714, 500, available = 4e9,
              row_chunk_size = NULL)
  expect_true(any(grepl("Leave .*row_chunk_size", off$suggestions)))

  # v2.6 has no such knob, and should say which generation does rather
  # than offering one it cannot honour.
  v26 <- peak(mem_config_tabpfn26(), 200000, 714, 500, available = 4e9)
  expect_false(any(grepl("row_chunk_size", v26$suggestions)))
  expect_true(any(grepl("v3", v26$suggestions)))
})

test_that("a comfortable run is not given advice it does not need", {
  expect_length(peak(mem_config_tabpfn3(), 200, 100, 10)$suggestions, 0L)
})


# ---------------------------------------------------------------------------
# Kumo Tabular
# ---------------------------------------------------------------------------

kumo_terms <- function(size = "small", n_context = 20000, n_query = 500,
                       n_features = 100, ...) {
  opts <- utils::modifyList(
    list(n_estimators = 1, kv_cache = FALSE, predict_chunk_size = 1024,
         row_chunk_size = NULL, col_chunk_size = NULL), list(...))
  for (nm in names(list(...))) if (is.null(list(...)[[nm]])) opts[nm] <- list(NULL)
  st <- kumo_peak_terms(n_context, n_query, n_features, opts,
                        mem_config_kumo(size))
  st$by <- stats::setNames(st$stages, vapply(st$stages, `[[`, "", "name"))
  st
}

test_that("Kumo row chunking bounds the blocks and keeps the row tensor", {
  off <- kumo_terms()
  on  <- kumo_terms(row_chunk_size = 1024L)
  # Unchunked: every row in one pass, nothing outside a loop, no cache.
  expect_equal(off$by[["row interaction"]]$act, 20500 * 104 * 128)
  expect_equal(off$by[["row interaction"]]$res, 0)
  expect_equal(off$by[["column summaries"]]$prepass, 0)
  expect_equal(off$persistent, 0)
  # Chunked: the blocks see a chunk of rows; the row tensor itself and the
  # in-context stack's input are not chunked, the inducing step reads
  # every context row, and one member's cache outlives the context pass.
  expect_equal(on$by[["row interaction"]]$act, 1024 * 104 * 128)
  expect_equal(on$by[["row interaction"]]$res, 20000 * 104 * 128)
  expect_equal(on$by[["in-context learning"]]$act, 1024 * 512)
  expect_equal(on$by[["in-context learning"]]$res, 20000 * 512)
  expect_equal(on$by[["column summaries"]]$prepass, 20000 * 100 * 128)
  expect_gt(on$persistent, 12 * 2 * 20000 * 512)
  # A chunk at or above the row count is the unchunked pass.
  expect_identical(kumo_terms(row_chunk_size = 1e6L)$stages, off$stages)
})

test_that("Kumo's column chunk bounds the column blocks with or without rows", {
  off <- kumo_terms(col_chunk_size = 8L)
  expect_equal(off$by[["column embedding"]]$act, 20500 * 8 * 128)
  expect_equal(off$by[["row interaction"]]$act, 20500 * 104 * 128)
  on <- kumo_terms(row_chunk_size = 1024L, col_chunk_size = 8L)
  expect_equal(on$by[["column summaries"]]$prepass, 20000 * 8 * 128)
})

test_that("Kumo's caches use the in-context width and the query heads", {
  # Medium projects 1024 readout numbers to E = 512, and its query rows
  # read 2 of the 8 key/value heads: the cache is a quarter of the full
  # projections.
  small <- kumo_terms("small", kv_cache = TRUE)
  med   <- kumo_terms("medium", kv_cache = TRUE)
  expect_equal(med$by[["in-context learning"]]$act, 20500 * 512)
  expect_equal(small$persistent - 12 * 2 * 20000 * 512,
               4 * 100 * 128 * 128 * 2 + 100)
  expect_equal(med$persistent - 24 * 2 * 20000 * 512 / 4,
               6 * 100 * 256 * 256 * 2 + 100)
  # One cache per member, all alive through `predict()`.
  expect_equal(kumo_terms(kv_cache = TRUE, n_estimators = 8)$persistent,
               8 * small$persistent)
  # Columns past the recipe's cap never reach the network.
  expect_identical(kumo_terms(n_features = 2000)$stages,
                   kumo_terms(n_features = 500)$stages)
})

test_that("Kumo past 10 classes pays for every ECOC sub-task", {
  ten <- kumo_terms(n_classes = 10)
  # 15 classes: max(ceil(15 / 9), 4 * ceil(log10(15))) = 8 sub-tasks.
  many <- kumo_terms(n_classes = 15, kv_cache = TRUE)
  for (nm in names(ten$by)) {
    expect_equal(many$by[[nm]]$act, 8 * ten$by[[nm]]$act, info = nm)
  }
  expect_equal(many$persistent, 8 * kumo_terms(kv_cache = TRUE)$persistent)
  expect_identical(kumo_terms(n_classes = 250)$by[["row interaction"]]$act,
                   28 * ten$by[["row interaction"]]$act)
  # And the estimate passes it through.
  a <- peak(mem_config_kumo(), 2000, 500, 20)
  b <- peak(mem_config_kumo(), 2000, 500, 20, n_classes = 15)
  expect_gt(b$total_peak_bytes, a$total_peak_bytes)
})

test_that("Kumo's advice names a smaller size, then its own knobs", {
  big <- peak(mem_config_kumo("large"), 50000, 1000, 100, available = 8e9)
  expect_false(big$verdict == "ok")
  size_at <- grep("Kumo Tabular medium or small", big$suggestions)
  ctx_at <- grep("Reduce .*n_context", big$suggestions)
  expect_length(size_at, 1L)
  expect_true(size_at < ctx_at)
  expect_true(any(grepl("Set .*row_chunk_size", big$suggestions)))
  expect_true(any(grepl("save_peak_memory_factor", big$suggestions)))

  med <- peak(mem_config_kumo("medium"), 50000, 1000, 100, available = 8e9)
  expect_true(any(grepl("Kumo Tabular small", med$suggestions)))
  # Small is already the smallest; with row chunking on there is no
  # chunking advice left to give but a smaller chunk.
  sm <- peak(mem_config_kumo("small"), 50000, 1000, 100, available = 8e9,
             row_chunk_size = 1024L)
  expect_false(any(grepl("Kumo Tabular", sm$suggestions)))
  expect_false(any(grepl("Set .*row_chunk_size", sm$suggestions)))
})

test_that("suggest_chunk_sizes searches Kumo's chunks", {
  # Not "has no stage chunking": Kumo is on the list, and a run that does
  # not fit unchunked gets a chunk size that does.
  # 20 GB rather than 40: since Kumo's layer loops collect (#99) the
  # unchunked estimate at this size is about 27 GB.
  plain <- peak(mem_config_kumo("small"), 6000, 1000, 50, available = 20e9)
  expect_false(plain$verdict %in% c("ok", "tight"))
  got <- suggest_chunk_sizes(mem_config_kumo("small"), n_context = 6000,
                             n_features = 50, available = 20e9,
                             verdict = "tight")
  expect_null(got$note)
  expect_true(got$feasible)
  expect_false(is.na(got$row_chunk_size))
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

test_that("an undownloaded model is estimated from the shipped table", {
  # The preflight must never start a multi-gigabyte download to answer a
  # question about whether the run would fit -- which is the whole reason
  # `architectures.json` exists. A catalogue id not on this machine but
  # in the table is answered from the table, and says so.
  ids <- list_models()
  missing <- ids$id[!ids$downloaded]
  known <- Filter(function(i) !is.null(.shipped_architecture(i)), missing)
  skip_if(!length(known), "every table entry is downloaded here")
  est <- estimate_peak_memory(known[[1]], 1000, 100, 10)
  expect_identical(est$weights_source, "shipped architecture table")
  expect_true(is.finite(est$total_peak_bytes))
})


test_that("a model nobody has and nobody catalogued says what to do", {
  expect_error(estimate_peak_memory("no-such-model", 1000, 100, 10),
               "No local artifacts")
})


test_that("a family name resolves with a task and is refused without one", {
  # `"tabpfn"` names two checkpoints, one per task. The task was already
  # an argument; it just was not reaching the resolver, so every family
  # name failed as though it were unknown.
  est <- estimate_peak_memory("tabpfn", 1000, 100, 10, task = "classification")
  expect_true(is.finite(est$total_peak_bytes))
  expect_identical(est$backend, "tabpfn")
  expect_error(estimate_peak_memory("tabpfn", 1000, 100, 10), "model family")
})


test_that("the shipped architecture table covers what it can", {
  tbl <- .shipped_architecture("tabpfn-v2.5-classifier")
  expect_false(is.null(tbl))
  # Every catalogue entry whose artifacts anyone has converted should be
  # in there; the two that need a fresh download are the known gap.
  ids <- list_models()$id
  covered <- vapply(ids, function(i) !is.null(.shipped_architecture(i)),
                    logical(1))
  expect_setequal(ids[!covered], c("tabfm-1.0.0-regressor", "mitra-regressor"))
})
