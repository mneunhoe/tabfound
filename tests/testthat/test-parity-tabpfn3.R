# Parity against the PyPI `tabpfn` package's v3 architecture.
#
# Replays stored reference dumps -- no Python needed here. Regenerate them
# with `Rscript inst/parity/run-parity.R tabpfn3 --regenerate`.
#
# Needs the converted checkpoints, so the test skips unless
# TABFOUND_TABPFN3_CLF_DIR / TABFOUND_TABPFN3_REG_DIR point at them.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

parity_dir <- tabfound_file("parity")
skip_if(!nzchar(parity_dir), "parity harness not installed")

source(file.path(parity_dir, "fixtures.R"), local = TRUE)
source(file.path(parity_dir, "compare.R"), local = TRUE)

FIXTURE_DIR <- file.path(parity_dir, "fixtures")
REF_DIR     <- file.path(parity_dir, "reference", "tabpfn3")
skip_if(!dir.exists(REF_DIR), "no stored TabPFN v3 reference dumps")

model_dir_for <- function(fixture) {
  var <- if (startsWith(fixture, "clf")) "TABFOUND_TABPFN3_CLF_DIR"
         else "TABFOUND_TABPFN3_REG_DIR"
  d <- Sys.getenv(var, unset = "")
  if (!nzchar(d) || !dir.exists(d)) NULL else d
}

# `clf_binary_missing` is the one that matters most: a constant column and
# 25 NaNs put the indicator channel, the mean imputation and the
# zero-variance branch of the in-architecture scaler all on the same pass.
fixtures <- c("clf_iris", "clf_binary_missing", "clf_tiny",
              "reg_iris", "reg_skewed", "reg_tiny")

for (fx in fixtures) {
  local({
    fixture <- fx
    test_that(paste0("tabfound matches Python TabPFN v3 on ", fixture), {
      ref <- file.path(REF_DIR, fixture)
      skip_if(!dir.exists(ref), paste("no reference dump for", fixture))
      md <- model_dir_for(fixture)
      skip_if(is.null(md), "converted TabPFN v3 artifacts not configured")

      res <- parity_tabpfn3(ref, FIXTURE_DIR, fixture, fixture, md)
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
