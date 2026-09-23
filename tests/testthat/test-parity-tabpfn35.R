# Parity against the PyPI `tabpfn` package's v3.5 architecture.
#
# Replays stored reference dumps -- no Python needed here. Regenerate them
# with `Rscript inst/parity/run-parity.R tabpfn35 --regenerate`, which
# wants the 9.0.0 reference in `.venvs/ref35` (see `py_for()` there);
# `.venvs/ref` stays on 8.2.0, which is what the v2.6 and v3 dumps were
# cut against.
#
# Needs the converted artifacts, so the test skips unless
# TABFOUND_TABPFN35_DIR points at them. One variable, not the
# classifier/regressor pair the older generations need: v3.5 is a single
# multitask checkpoint and the same directory serves both halves.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

parity_dir <- tabfound_file("parity")
skip_if(!nzchar(parity_dir), "parity harness not installed")

source(file.path(parity_dir, "fixtures.R"), local = TRUE)
source(file.path(parity_dir, "compare.R"), local = TRUE)

FIXTURE_DIR <- file.path(parity_dir, "fixtures")
REF_DIR     <- file.path(parity_dir, "reference", "tabpfn35")
skip_if(!dir.exists(REF_DIR), "no stored TabPFN v3.5 reference dumps")

model_dir <- local({
  d <- Sys.getenv("TABFOUND_TABPFN35_DIR", unset = "")
  if (!nzchar(d) || !dir.exists(d)) NULL else d
})

# `clf_binary_missing` is the one that matters most: a constant column and
# 25 NaNs put the indicator channel, the mean imputation, the
# zero-variance branch of the in-architecture scaler -- and, new in v3.5,
# the ECDF ranking of a column whose cells were imputed -- all on the same
# pass. A constant column is also the degenerate case for the ECDF: every
# row ties with every other, so there is nothing to interpolate between.
#
# `clf_large` is the only one big enough for the reference's own
# `_stages_0_to_2` row/column chunking to fire (2,664 rows against its
# 2,048-row default), so it is the only one that says anything about the
# chunked path -- and the only one where the ECDF context is built from
# more rows than a single chunk holds.
fixtures <- c("clf_iris", "clf_binary_missing", "clf_tiny", "clf_large",
              "reg_iris", "reg_skewed", "reg_tiny")

for (fx in fixtures) {
  local({
    fixture <- fx
    test_that(paste0("tabfound matches Python TabPFN v3.5 on ", fixture), {
      ref <- file.path(REF_DIR, fixture)
      skip_if(!dir.exists(ref), paste("no reference dump for", fixture))
      skip_if(is.null(model_dir), "TabPFN v3.5 artifacts not configured")

      res <- parity_tabpfn35(ref, FIXTURE_DIR, fixture, fixture, model_dir)
      graded <- grade_parity(res$summary)

      failed <- graded[!graded$pass, , drop = FALSE]
      expect_equal(
        nrow(failed), 0L,
        info = if (nrow(failed)) paste(
          sprintf("%s: max_abs=%.3e max_rel=%.3e (tol %.1e)",
                  failed$stage, failed$max_abs, failed$max_rel,
                  failed$tol_abs),
          collapse = "; ") else ""
      )
    })
  })
}


test_that("the stored dumps record the reference they were cut against", {
  # The dumps are the only record of which Python produced them, and v3.5
  # is the first generation that needs a *different* interpreter from the
  # rest. A dump regenerated against the wrong venv would still load and
  # still compare -- it would just be grading the port against the wrong
  # model -- so check the provenance rather than trusting the filename.
  for (fx in fixtures) {
    ref <- file.path(REF_DIR, fx, "reference.json")
    if (!file.exists(ref)) next
    meta <- jsonlite::fromJSON(ref)
    expect_identical(meta$architecture, "tabpfn_v3_5")
    expect_identical(meta$head, "multitask")
    expect_true(startsWith(meta$tabpfn_version, "9."))
  }
})
