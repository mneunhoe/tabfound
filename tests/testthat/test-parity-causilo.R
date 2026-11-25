# Network parity against `causilo` 1.5.0.
#
# Replays stored reference dumps -- no Python needed here. Regenerate with
# `Rscript inst/parity/run-parity.R causilo --regenerate` (needs
# `.venvs/causilo`). The dumps are of the published checkpoints, so the
# test skips unless TABFOUND_CAUSILO_CLF_DIR / TABFOUND_CAUSILO_REG_DIR
# point at them (`download_model("causilo-1.5-classifier")` puts them in
# the store).

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

parity_dir <- tabfound_file("parity")
skip_if(!nzchar(parity_dir), "parity harness not installed")

source(file.path(parity_dir, "fixtures.R"), local = TRUE)
source(file.path(parity_dir, "compare.R"), local = TRUE)

FIXTURE_DIR <- file.path(parity_dir, "fixtures")
REF_DIR     <- file.path(parity_dir, "reference", "causilo")
skip_if(!dir.exists(REF_DIR), "no stored Causilo reference dumps")

model_dir_for <- function(fixture) {
  var <- if (startsWith(fixture, "clf")) "TABFOUND_CAUSILO_CLF_DIR"
         else "TABFOUND_CAUSILO_REG_DIR"
  d <- Sys.getenv(var, unset = "")
  if (!nzchar(d) || !dir.exists(d)) NULL else d
}

for (fx in c("clf_tiny", "clf_binary_missing", "clf_categorical", "reg_tiny")) {
  local({
    fixture <- fx
    test_that(paste0("tabfound matches causilo's network on ", fixture), {
      ref <- file.path(REF_DIR, fixture)
      skip_if(!file.exists(file.path(ref, "reference.json")),
              paste("no reference dump for", fixture))
      md <- model_dir_for(fixture)
      skip_if(is.null(md), "Causilo 1.5 artifacts not configured")

      res <- parity_causilo(ref, FIXTURE_DIR, fixture, fixture, md)
      graded <- grade_parity(res$summary)
      # The final plus col0, the last row layer, the last prediction
      # layer and the head.
      expect_gte(nrow(graded), 5L)

      failed <- graded[!graded$pass, , drop = FALSE]
      expect_equal(
        nrow(failed), 0L,
        info = if (nrow(failed)) paste(
          sprintf("%s: max_abs=%.3e max_scaled=%.3e",
                  failed$stage, failed$max_abs, failed$max_scaled),
          collapse = "; ") else ""
      )
    })
  })
}
