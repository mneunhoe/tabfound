# End-to-end parity against the PyPI `tabpfn` package.
#
# Replays stored reference dumps -- no Python needed here. Regenerate the
# dumps with `Rscript inst/parity/run-parity.R tabpfn --regenerate`.
#
# Needs the converted checkpoints, so the test skips unless
# TABFOUND_TABPFN_CLF_DIR / TABFOUND_TABPFN_REG_DIR point at them.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")
skip_if_not_installed("digest")

parity_dir <- tabfound_file("parity")
skip_if(!nzchar(parity_dir), "parity harness not installed")

source(file.path(parity_dir, "fixtures.R"), local = TRUE)
source(file.path(parity_dir, "compare.R"), local = TRUE)

FIXTURE_DIR <- file.path(parity_dir, "fixtures")
REF_DIR     <- file.path(parity_dir, "reference", "tabpfn")
skip_if(!dir.exists(REF_DIR), "no stored TabPFN reference dumps")

model_dir_for <- function(fixture) {
  var <- if (startsWith(fixture, "clf")) "TABFOUND_TABPFN_CLF_DIR"
         else "TABFOUND_TABPFN_REG_DIR"
  d <- Sys.getenv(var, unset = "")
  if (!nzchar(d) || !dir.exists(d)) NULL else d
}

# The `_nofp` variants are the deterministic contract: with the
# fingerprint feature off, every stage must agree. The fingerprinted
# variants are checked too, but only up to the point the fingerprint
# perturbs -- see `inst/parity/README.md`.
strict_fixtures <- c("clf_iris_nofp", "clf_binary_missing_nofp",
                     "reg_iris_nofp", "reg_skewed_nofp",
                     # v2.5's members ask for
                     # `ordinal_very_common_categories_shuffled` too, so
                     # the categorical path is not a v2.6-only concern.
                     "clf_categorical_nofp", "reg_categorical_nofp")

for (fx in strict_fixtures) {
  local({
    fixture <- fx
    test_that(paste0("tabfound matches Python TabPFN on ", fixture), {
      ref <- file.path(REF_DIR, fixture)
      skip_if(!dir.exists(ref), paste("no reference dump for", fixture))
      md <- model_dir_for(fixture)
      skip_if(is.null(md), "converted TabPFN artifacts not configured")

      res <- parity_tabpfn(ref, FIXTURE_DIR, sub("_nofp$", "", fixture),
                           fixture, md)
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

      # Preprocessing is float64 arithmetic on both sides and has no
      # excuse to differ at all; keep that stricter than the graded
      # tolerance so drift shows up here first.
      prep <- graded[startsWith(graded$stage, "preprocess:X"), , drop = FALSE]
      expect_true(all(prep$max_abs == 0),
                  info = "member inputs must be bit-identical")
    })
  })
}


test_that("classifier fixtures also match with the fingerprint feature on", {
  md <- model_dir_for("clf_iris")
  skip_if(is.null(md), "converted TabPFN artifacts not configured")
  for (fixture in c("clf_iris", "clf_binary_missing")) {
    ref <- file.path(REF_DIR, fixture)
    skip_if(!dir.exists(ref), paste("no reference dump for", fixture))
    res <- parity_tabpfn(ref, FIXTURE_DIR, fixture, fixture, md)
    graded <- grade_parity(res$summary)
    expect_true(all(graded$pass),
                info = paste(fixture, "stages:",
                             paste(graded$stage[!graded$pass], collapse = ", ")))
  }
})


test_that("the v2.5 KV cache costs no more accuracy than re-batching does", {
  # Everything this architecture fits comes from the training rows alone,
  # so the cache cannot answer a different question -- unlike v2.6's,
  # which freezes two train+test-fitted masks. What is left is float32
  # reduction order, and the honest bound for that is the shift the
  # *uncached* path already shows when the same rows run in a different
  # sized batch. The harness computes both and grades one against the
  # other; here we assert it did.
  for (fixture in c("clf_iris_nofp", "reg_iris_nofp", "clf_categorical_nofp")) {
    ref <- file.path(REF_DIR, fixture)
    skip_if(!dir.exists(ref), paste("no reference dump for", fixture))
    md <- model_dir_for(fixture)
    skip_if(is.null(md), "converted TabPFN artifacts not configured")

    res <- parity_tabpfn(ref, FIXTURE_DIR, sub("_nofp$", "", fixture), fixture, md)
    row <- grade_parity(res$summary)
    row <- row[row$stage == "cache:vs-batching", , drop = FALSE]
    expect_identical(nrow(row), 1L, info = fixture)
    expect_true(row$pass, info = paste(fixture, row$note))
  }
})
