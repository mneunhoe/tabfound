# Parity driver.
#
#   Rscript inst/parity/run-parity.R [backend] [--regenerate]
#
# Without `--regenerate` this only replays stored Python reference dumps
# against the current R code, so it needs no Python. With
# `--regenerate` it re-runs the reference implementation first, which
# needs the venv in `TABFOUND_REF_PYTHON` (default `.venvs/ref/bin/python`).
#
# Model artifacts are located via environment variables so the harness
# does not hard-code anyone's checkout:
#
#   TABFOUND_TABPFN_CLF_DIR   converted classifier (model.safetensors + config.json)
#   TABFOUND_TABPFN_REG_DIR   converted regressor
#   TABFOUND_TABPFN_CLF_CKPT  raw .ckpt, only needed with --regenerate
#   TABFOUND_TABPFN_REG_CKPT
#   TABFOUND_TABPFN26_CLF_DIR converted TabPFN v2.6 classifier
#   TABFOUND_TABPFN26_REG_DIR converted TabPFN v2.6 regressor
#   TABFOUND_TABPFN26_CLF_CKPT  raw .ckpt, only needed with --regenerate
#   TABFOUND_TABPFN26_REG_CKPT
#   TABFOUND_TABPFN3_CLF_DIR  converted TabPFN v3 classifier
#   TABFOUND_TABPFN3_REG_DIR  converted TabPFN v3 regressor
#   TABFOUND_TABPFN3_CLF_CKPT   raw .ckpt, only needed with --regenerate
#   TABFOUND_TABPFN3_REG_CKPT
#   TABFOUND_TABFM_DIR        TabFM Hub snapshot root (holds classification/
#                             and regression/ subfolders)
#   TABFOUND_TABICL_CLF_DIR   converted TabICL classifier
#   TABFOUND_TABICL_REG_DIR   converted TabICL regressor
#   TABFOUND_TABICL_CLF_CKPT  raw .ckpt, only needed with --regenerate
#   TABFOUND_TABICL_REG_CKPT
#   TABFOUND_MITRA_CLF_DIR    Mitra classifier snapshot
#   TABFOUND_MITRA_REG_DIR    Mitra regressor snapshot
#   TABFOUND_MITRA_SHIM       dir holding the `mitrapkg` shim (regenerate only)

suppressMessages({
  library(torch)
  library(withr)
  pkgload::load_all(".", quiet = TRUE)
})

here <- "inst/parity"
source(file.path(here, "fixtures.R"))
source(file.path(here, "compare.R"))

args        <- commandArgs(trailingOnly = TRUE)
regenerate  <- "--regenerate" %in% args
backends    <- setdiff(args, "--regenerate")
if (!length(backends)) backends <- "tabpfn"

FIXTURE_DIR <- file.path(here, "fixtures")
REF_ROOT    <- file.path(here, "reference")
PY          <- Sys.getenv("TABFOUND_REF_PYTHON", ".venvs/ref/bin/python")

if (!dir.exists(FIXTURE_DIR)) {
  cat("Generating fixtures...\n")
  write_parity_fixtures(FIXTURE_DIR)
}

# `<fixture>[_auto][_nofp]` -> the fixture it is built from.
.base_fixture <- function(fx) sub("(_auto)?(_nofp)?$", "", fx)

# `--categorical` for the variants that declare, nothing for the `_auto`
# ones (which is what makes them test inference rather than declaration)
# or for fixtures with no categorical columns at all.
.categorical_flag <- function(fx) {
  if (grepl("_auto(_nofp)?$", fx)) return(character())
  cats <- read_parity_fixture(FIXTURE_DIR, .base_fixture(fx))$categorical_features
  if (!length(cats)) return(character())
  c("--categorical", paste(cats - 1L, collapse = ","))
}

env_dir <- function(var, what) {
  v <- Sys.getenv(var, unset = "")
  if (!nzchar(v) || !dir.exists(v)) {
    cat(sprintf("  SKIP: %s is unset or missing (%s)\n", var, what))
    return(NULL)
  }
  v
}


# ---------------------------------------------------------------------------
# TabPFN
# ---------------------------------------------------------------------------

run_tabpfn <- function() {
  cat("\n=== backend: tabpfn ===\n")
  clf_dir <- env_dir("TABFOUND_TABPFN_CLF_DIR", "converted classifier artifacts")
  reg_dir <- env_dir("TABFOUND_TABPFN_REG_DIR", "converted regressor artifacts")

  # `_nofp` variants run the same fixture with the fingerprint feature
  # switched off on both sides. The fingerprint is a SHA-256 over the
  # row's float64 bytes, so a sub-ULP difference in any upstream cell
  # replaces one row's value with an unrelated draw from [0, 1]. Keeping
  # both variants separates "the pipeline drifted" from "a hash flipped".
  fixtures <- list(
    clf_iris                = clf_dir,
    clf_iris_nofp           = clf_dir,
    clf_binary_missing      = clf_dir,
    clf_binary_missing_nofp = clf_dir,
    reg_iris                = reg_dir,
    reg_iris_nofp           = reg_dir,
    reg_skewed              = reg_dir,
    reg_skewed_nofp         = reg_dir,
    # Declared categorical columns, which the ordinal encoder reaches.
    clf_categorical_nofp    = clf_dir,
    reg_categorical_nofp    = reg_dir
  )

  if (regenerate) {
    ckpts <- c(clf = Sys.getenv("TABFOUND_TABPFN_CLF_CKPT", ""),
               reg = Sys.getenv("TABFOUND_TABPFN_REG_CKPT", ""))
    for (fx in names(fixtures)) {
      head <- if (startsWith(fx, "clf")) "clf" else "reg"
      extra <- if (endsWith(fx, "_nofp")) "--no-fingerprint" else character()
      base_fx <- .base_fixture(fx)
      if (!nzchar(ckpts[[head]]) || !file.exists(ckpts[[head]])) {
        cat(sprintf("  SKIP regenerate %s: checkpoint not found\n", fx)); next
      }
      out <- file.path(REF_ROOT, "tabpfn", fx)
      unlink(out, recursive = TRUE)
      cat(sprintf("  regenerating %s ...\n", fx))
      st <- system2(PY, c(file.path(here, "tabpfn_reference.py"),
                          "--fixture-dir", FIXTURE_DIR,
                          "--fixture", base_fx,
                          "--ckpt", ckpts[[head]],
                          "--n-estimators", "4",
                          extra,
                          .categorical_flag(fx),
                          "--out", out),
                    stdout = TRUE, stderr = TRUE)
      status <- attr(st, "status")
      if (!is.null(status) && status != 0L) {
        cat(paste(tail(st, 15), collapse = "\n"), "\n")
        cat(sprintf("  FAILED to regenerate %s\n", fx))
      }
    }
  }

  all_rows <- list()
  for (fx in names(fixtures)) {
    model_dir <- fixtures[[fx]]
    ref <- file.path(REF_ROOT, "tabpfn", fx)
    if (is.null(model_dir) || !dir.exists(ref)) {
      cat(sprintf("  SKIP %s (no model dir or no reference dump)\n", fx)); next
    }
    res <- parity_tabpfn(ref, FIXTURE_DIR, .base_fixture(fx), fx, model_dir)
    all_rows[[fx]] <- print_parity(res)
  }
  combined <- do.call(rbind, all_rows)
  if (is.null(combined)) return(invisible(NULL))

  # A fixture that only fails with fingerprints on, while its `_nofp`
  # twin passes the same stage, is failing for a reason we understand:
  # the fingerprint is a SHA-256 over float64 bytes, so a last-bit
  # difference in any upstream cell replaces one row's value outright.
  # Mark those `KNOWN` rather than FAIL -- but only when the twin
  # actually passed, so a real regression still shows up.
  combined$known <- FALSE
  for (i in which(!combined$pass)) {
    fxi <- combined$fixture[i]
    if (endsWith(fxi, "_nofp")) next
    twin <- combined$pass[combined$fixture == paste0(fxi, "_nofp") &
                          combined$stage == combined$stage[i]]
    if (length(twin) == 1L && isTRUE(twin)) combined$known[i] <- TRUE
  }
  invisible(combined)
}


# ---------------------------------------------------------------------------
# TabPFN v2.6
# ---------------------------------------------------------------------------

run_tabpfn26 <- function() {
  cat("\n=== backend: tabpfn26 ===\n")
  clf_dir <- env_dir("TABFOUND_TABPFN26_CLF_DIR", "converted v2.6 classifier")
  reg_dir <- env_dir("TABFOUND_TABPFN26_REG_DIR", "converted v2.6 regressor")

  # No `_nofp` twins here: this harness runs the bare network, which has
  # no fingerprint feature to switch off. `clf_binary_missing` is the
  # important one -- a constant column and 25 NaNs exercise the
  # constant-column removal, the mean imputation and the NaN indicator
  # channel that v2.6 moved inside the architecture.
  fixtures <- list(
    clf_iris           = clf_dir,
    clf_binary_missing = clf_dir,
    clf_tiny           = clf_dir,
    reg_iris           = reg_dir,
    reg_skewed         = reg_dir,
    reg_tiny           = reg_dir
  )

  if (regenerate) {
    ckpts <- c(clf = Sys.getenv("TABFOUND_TABPFN26_CLF_CKPT", ""),
               reg = Sys.getenv("TABFOUND_TABPFN26_REG_CKPT", ""))
    for (fx in names(fixtures)) {
      head <- if (startsWith(fx, "clf")) "clf" else "reg"
      if (!nzchar(ckpts[[head]]) || !file.exists(ckpts[[head]])) {
        cat(sprintf("  SKIP regenerate %s: checkpoint not found\n", fx)); next
      }
      out <- file.path(REF_ROOT, "tabpfn26", fx)
      unlink(out, recursive = TRUE)
      cat(sprintf("  regenerating %s ...\n", fx))
      st <- system2(PY, c(file.path(here, "tabpfn26_reference.py"),
                          "--fixture-dir", FIXTURE_DIR, "--fixture", fx,
                          "--ckpt", ckpts[[head]], "--out", out),
                    stdout = TRUE, stderr = TRUE)
      status <- attr(st, "status")
      if (!is.null(status) && status != 0L) {
        cat(paste(tail(st, 15), collapse = "\n"), "\n")
        cat(sprintf("  FAILED to regenerate %s\n", fx))
      }
    }
  }

  # The ensemble runs through the full estimator on both sides, so it
  # reuses the v2.5 comparison wholesale -- same dump layout, same stages,
  # same `_nofp` twins. What differs is only which presets the members
  # ask for, and those come out of the dump.
  ens_fixtures <- list(
    clf_iris                = clf_dir,
    clf_iris_nofp           = clf_dir,
    clf_binary_missing      = clf_dir,
    clf_binary_missing_nofp = clf_dir,
    reg_iris                = reg_dir,
    reg_iris_nofp           = reg_dir,
    reg_skewed              = reg_dir,
    reg_skewed_nofp         = reg_dir,
    # Categorical columns: declared (so the ordinal encoder reaches them)
    # and undeclared (so only the low-cardinality one is inferred).
    clf_categorical           = clf_dir,
    clf_categorical_nofp      = clf_dir,
    clf_categorical_auto_nofp = clf_dir,
    reg_categorical           = reg_dir,
    reg_categorical_nofp      = reg_dir,
    reg_categorical_auto_nofp = reg_dir
  )

  if (regenerate) {
    ckpts <- c(clf = Sys.getenv("TABFOUND_TABPFN26_CLF_CKPT", ""),
               reg = Sys.getenv("TABFOUND_TABPFN26_REG_CKPT", ""))
    for (fx in names(ens_fixtures)) {
      head <- if (startsWith(fx, "clf")) "clf" else "reg"
      if (!nzchar(ckpts[[head]]) || !file.exists(ckpts[[head]])) next
      out <- file.path(REF_ROOT, "tabpfn26-ensemble", fx)
      unlink(out, recursive = TRUE)
      cat(sprintf("  regenerating ensemble %s ...\n", fx))
      st <- system2(PY, c(file.path(here, "tabpfn_reference.py"),
                          "--fixture-dir", FIXTURE_DIR,
                          "--fixture", .base_fixture(fx),
                          "--ckpt", ckpts[[head]],
                          "--n-estimators", "4",
                          if (endsWith(fx, "_nofp")) "--no-fingerprint" else character(),
                          .categorical_flag(fx),
                          "--out", out),
                    stdout = TRUE, stderr = TRUE)
      status <- attr(st, "status")
      if (!is.null(status) && status != 0L) {
        cat(paste(tail(st, 15), collapse = "\n"), "\n")
        cat(sprintf("  FAILED to regenerate ensemble %s\n", fx))
      }
    }
  }

  all_rows <- list()
  for (fx in names(fixtures)) {
    ref <- file.path(REF_ROOT, "tabpfn26", fx)
    md  <- fixtures[[fx]]
    if (is.null(md) || !file.exists(file.path(ref, "reference.json"))) {
      cat(sprintf("  SKIP %s (no reference dump or model dir)\n", fx)); next
    }
    res <- parity_tabpfn26(ref, FIXTURE_DIR, fx, fx, md)
    all_rows[[fx]] <- print_parity(res)
  }
  ens_rows <- list()
  for (fx in names(ens_fixtures)) {
    ref <- file.path(REF_ROOT, "tabpfn26-ensemble", fx)
    md  <- ens_fixtures[[fx]]
    if (is.null(md) || !file.exists(file.path(ref, "reference.json"))) {
      cat(sprintf("  SKIP ensemble %s (no reference dump or model dir)\n", fx)); next
    }
    res <- parity_tabpfn(ref, FIXTURE_DIR, .base_fixture(fx),
                         paste0("ens:", fx), md)
    ens_rows[[fx]] <- print_parity(res)
  }

  combined <- do.call(rbind, c(all_rows, ens_rows))
  if (is.null(combined)) return(invisible(NULL))
  combined$known <- FALSE
  # Same rule as v2.5: a fingerprinted fixture that fails a stage its
  # `_nofp` twin passes is failing because a SHA-256 flipped, not because
  # the pipeline drifted.
  for (i in which(!combined$pass)) {
    fxi <- combined$fixture[i]
    if (!startsWith(fxi, "ens:") || endsWith(fxi, "_nofp")) next
    twin <- combined$pass[combined$fixture == paste0(fxi, "_nofp") &
                          combined$stage == combined$stage[i]]
    if (length(twin) == 1L && isTRUE(twin)) combined$known[i] <- TRUE
  }
  invisible(combined)
}

# ---------------------------------------------------------------------------
# TabPFN v3
# ---------------------------------------------------------------------------

run_tabpfn3 <- function() {
  cat("\n=== backend: tabpfn3 ===\n")
  clf_dir <- env_dir("TABFOUND_TABPFN3_CLF_DIR", "converted v3 classifier")
  reg_dir <- env_dir("TABFOUND_TABPFN3_REG_DIR", "converted v3 regressor")

  # The bare network only, as for v2.6: v3 keeps the NaN/Inf handling and
  # the standard scaler inside `forward()`, so this is the surface the R
  # port reimplements. There is no ensemble section because the estimator
  # side is unchanged from v2 -- the members, the presets and the target
  # transforms are the same machinery, already graded there.
  #
  # `clf_binary_missing` is the one that matters most: a constant column
  # plus 25 NaNs exercise the indicator channel, the mean imputation and
  # the zero-variance branch of the scaler in one pass.
  fixtures <- list(
    clf_iris           = clf_dir,
    clf_binary_missing = clf_dir,
    clf_tiny           = clf_dir,
    reg_iris           = reg_dir,
    reg_skewed         = reg_dir,
    reg_tiny           = reg_dir
  )

  if (regenerate) {
    ckpts <- c(clf = Sys.getenv("TABFOUND_TABPFN3_CLF_CKPT", ""),
               reg = Sys.getenv("TABFOUND_TABPFN3_REG_CKPT", ""))
    for (fx in names(fixtures)) {
      head <- if (startsWith(fx, "clf")) "clf" else "reg"
      if (!nzchar(ckpts[[head]]) || !file.exists(ckpts[[head]])) {
        cat(sprintf("  SKIP regenerate %s: checkpoint not found\n", fx)); next
      }
      out <- file.path(REF_ROOT, "tabpfn3", fx)
      unlink(out, recursive = TRUE)
      cat(sprintf("  regenerating %s ...\n", fx))
      st <- system2(PY, c(file.path(here, "tabpfn3_reference.py"),
                          "--fixture-dir", FIXTURE_DIR, "--fixture", fx,
                          "--ckpt", ckpts[[head]], "--out", out),
                    stdout = TRUE, stderr = TRUE)
      status <- attr(st, "status")
      if (!is.null(status) && status != 0L) {
        cat(paste(tail(st, 15), collapse = "\n"), "\n")
        cat(sprintf("  FAILED to regenerate %s\n", fx))
      }
    }
  }

  all_rows <- list()
  for (fx in names(fixtures)) {
    ref <- file.path(REF_ROOT, "tabpfn3", fx)
    md  <- fixtures[[fx]]
    if (is.null(md) || !file.exists(file.path(ref, "reference.json"))) {
      cat(sprintf("  SKIP %s (no reference dump or model dir)\n", fx)); next
    }
    res <- parity_tabpfn3(ref, FIXTURE_DIR, fx, fx, md)
    all_rows[[fx]] <- print_parity(res)
  }

  combined <- do.call(rbind, all_rows)
  if (is.null(combined)) return(invisible(NULL))
  combined$known <- FALSE
  invisible(combined)
}



# ---------------------------------------------------------------------------
# TabFM
# ---------------------------------------------------------------------------

run_tabfm <- function() {
  cat("\n=== backend: tabfm ===\n")
  root <- env_dir("TABFOUND_TABFM_DIR", "TabFM Hub snapshot root")
  if (is.null(root)) return(invisible(NULL))

  fixtures <- list(clf_tiny = "classification", clf_iris = "classification",
                   reg_tiny = "regression")

  if (regenerate) {
    for (fx in names(fixtures)) {
      wd <- file.path(root, fixtures[[fx]])
      if (!dir.exists(wd)) {
        cat(sprintf("  SKIP regenerate %s: %s not found\n", fx, wd)); next
      }
      out <- file.path(REF_ROOT, "tabfm", fx)
      unlink(out, recursive = TRUE)
      cat(sprintf("  regenerating %s ...\n", fx))
      st <- system2(PY, c(file.path(here, "tabfm_reference.py"),
                          "--fixture-dir", FIXTURE_DIR, "--fixture", fx,
                          "--weights-dir", wd, "--out", out),
                    stdout = TRUE, stderr = TRUE)
      status <- attr(st, "status")
      if (!is.null(status) && status != 0L) {
        cat(paste(tail(st, 15), collapse = "\n"), "\n")
        cat(sprintf("  FAILED to regenerate %s\n", fx))
      }
    }
  }

  all_rows <- list()
  for (fx in names(fixtures)) {
    ref <- file.path(REF_ROOT, "tabfm", fx)
    wd  <- file.path(root, fixtures[[fx]])
    if (!file.exists(file.path(ref, "reference.json")) || !dir.exists(wd)) {
      cat(sprintf("  SKIP %s (no reference dump or weights)\n", fx)); next
    }
    res <- parity_tabfm(ref, FIXTURE_DIR, fx, fx, root)
    all_rows[[fx]] <- print_parity(res)
  }
  combined <- do.call(rbind, all_rows)
  if (!is.null(combined)) combined$known <- FALSE
  invisible(combined)
}


# ---------------------------------------------------------------------------
# TabICL
# ---------------------------------------------------------------------------

run_tabicl <- function() {
  cat("\n=== backend: tabicl ===\n")
  clf_dir <- env_dir("TABFOUND_TABICL_CLF_DIR", "converted TabICL classifier")
  reg_dir <- env_dir("TABFOUND_TABICL_REG_DIR", "converted TabICL regressor")

  # `clf_binary_missing` is deliberately absent: TabICL's network has no
  # missing-value handling of its own (no nan_to_num, no imputation), so
  # unimputed NaN propagates to NaN logits. Its sklearn wrapper imputes
  # first. Comparing the raw network on that fixture would only confirm
  # that both sides produce NaN.
  fixtures <- list(clf_tiny = clf_dir, clf_iris = clf_dir,
                   reg_tiny = reg_dir, reg_iris = reg_dir)

  if (regenerate) {
    ckpts <- c(clf = Sys.getenv("TABFOUND_TABICL_CLF_CKPT", ""),
               reg = Sys.getenv("TABFOUND_TABICL_REG_CKPT", ""))
    for (fx in names(fixtures)) {
      head <- if (startsWith(fx, "clf")) "clf" else "reg"
      if (!nzchar(ckpts[[head]]) || !file.exists(ckpts[[head]])) {
        cat(sprintf("  SKIP regenerate %s: checkpoint not found\n", fx)); next
      }
      out <- file.path(REF_ROOT, "tabicl", fx)
      unlink(out, recursive = TRUE)
      cat(sprintf("  regenerating %s ...\n", fx))
      st <- system2(PY, c(file.path(here, "tabicl_reference.py"),
                          "--fixture-dir", FIXTURE_DIR, "--fixture", fx,
                          "--ckpt", ckpts[[head]], "--out", out),
                    stdout = TRUE, stderr = TRUE)
      status <- attr(st, "status")
      if (!is.null(status) && status != 0L) {
        cat(paste(tail(st, 15), collapse = "\n"), "\n")
        cat(sprintf("  FAILED to regenerate %s\n", fx))
      }
    }
  }

  all_rows <- list()
  for (fx in names(fixtures)) {
    ref <- file.path(REF_ROOT, "tabicl", fx)
    md <- fixtures[[fx]]
    if (is.null(md) || !file.exists(file.path(ref, "reference.json"))) {
      cat(sprintf("  SKIP %s (no reference dump or model dir)\n", fx)); next
    }
    res <- parity_tabicl(ref, FIXTURE_DIR, fx, fx, md)
    all_rows[[fx]] <- print_parity(res)
  }
  combined <- do.call(rbind, all_rows)
  if (!is.null(combined)) combined$known <- FALSE
  invisible(combined)
}


# ---------------------------------------------------------------------------
# Mitra
# ---------------------------------------------------------------------------

run_mitra <- function() {
  cat("\n=== backend: mitra ===\n")
  clf_dir <- env_dir("TABFOUND_MITRA_CLF_DIR", "Mitra classifier snapshot")
  reg_dir <- env_dir("TABFOUND_MITRA_REG_DIR", "Mitra regressor snapshot")

  # No `clf_binary_missing`: Mitra's quantile-rank embedding runs
  # `torch.quantile` and `searchsorted`, neither of which has a defined
  # answer for NaN, so unimputed input propagates to NaN output. Its
  # wrapper imputes first. The behaviour is pinned by a test instead.
  fixtures <- list(clf_tiny = clf_dir, clf_iris = clf_dir,
                   reg_tiny = reg_dir, reg_iris = reg_dir)

  if (regenerate) {
    shim <- Sys.getenv("TABFOUND_MITRA_SHIM", "")
    if (!nzchar(shim)) {
      cat("  SKIP regenerate: TABFOUND_MITRA_SHIM not set\n")
    } else {
      for (fx in names(fixtures)) {
        wd <- fixtures[[fx]]
        if (is.null(wd)) next
        out <- file.path(REF_ROOT, "mitra", fx)
        unlink(out, recursive = TRUE)
        cat(sprintf("  regenerating %s ...\n", fx))
        st <- withr::with_envvar(
          c(PYTHONPATH = shim),
          system2(PY, c(file.path(here, "mitra_reference.py"),
                        "--fixture-dir", FIXTURE_DIR, "--fixture", fx,
                        "--weights-dir", wd, "--out", out),
                  stdout = TRUE, stderr = TRUE)
        )
        status <- attr(st, "status")
        if (!is.null(status) && status != 0L) {
          cat(paste(tail(st, 15), collapse = "\n"), "\n")
          cat(sprintf("  FAILED to regenerate %s\n", fx))
        }
      }
    }
  }

  all_rows <- list()
  for (fx in names(fixtures)) {
    ref <- file.path(REF_ROOT, "mitra", fx)
    md <- fixtures[[fx]]
    if (is.null(md) || !file.exists(file.path(ref, "reference.json"))) {
      cat(sprintf("  SKIP %s (no reference dump or model dir)\n", fx)); next
    }
    res <- parity_mitra(ref, FIXTURE_DIR, fx, fx, md)
    all_rows[[fx]] <- print_parity(res)
  }
  combined <- do.call(rbind, all_rows)
  if (!is.null(combined)) combined$known <- FALSE
  invisible(combined)
}


results <- list()
for (b in backends) {
  results[[b]] <- switch(
    b,
    tabpfn   = run_tabpfn(),
    tabpfn26 = run_tabpfn26(),
    tabpfn3  = run_tabpfn3(),
    tabfm  = run_tabfm(),
    tabicl = run_tabicl(),
    mitra  = run_mitra(),
    { cat(sprintf("No parity runner for backend '%s' yet.\n", b)); NULL }
  )
}

combined <- do.call(rbind, Filter(Negate(is.null), results))
if (!is.null(combined)) {
  dir.create(file.path(here, "results"), showWarnings = FALSE)
  utils::write.csv(combined, file.path(here, "results", "parity.csv"),
                   row.names = FALSE)
  known <- if (is.null(combined$known)) rep(FALSE, nrow(combined)) else combined$known
  n_fail  <- sum(!combined$pass & !known)
  n_known <- sum(known)
  cat(sprintf("\n%d/%d checks within tolerance (%d known fingerprint-sensitive).\n",
              sum(combined$pass), nrow(combined), n_known))
  if (n_known > 0L) {
    kn <- combined[known, c("fixture", "stage", "max_abs", "note")]
    cat("\nKnown fingerprint-sensitive (the _nofp twin passes the same stage):\n")
    for (i in seq_len(nrow(kn))) {
      cat(sprintf("  %-20s %-20s max_abs %.3e  %s\n",
                  kn$fixture[i], kn$stage[i], kn$max_abs[i], kn$note[i]))
    }
  }
  if (n_fail > 0L) quit(status = 1L)
}
