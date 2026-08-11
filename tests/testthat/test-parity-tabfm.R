# End-to-end parity against the PyPI `tabfm` package.
#
# Replays stored reference dumps — no Python needed here. Regenerate with
# `Rscript inst/parity/run-parity.R tabfm --regenerate`.
#
# Needs the ~6.5 GB Hub snapshot, so the test skips unless
# TABFOUND_TABFM_DIR points at its root.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

parity_dir <- tabfound_file("parity")
skip_if(!nzchar(parity_dir), "parity harness not installed")

source(file.path(parity_dir, "fixtures.R"), local = TRUE)
source(file.path(parity_dir, "compare.R"), local = TRUE)

FIXTURE_DIR <- file.path(parity_dir, "fixtures")
REF_DIR     <- file.path(parity_dir, "reference", "tabfm")
skip_if(!dir.exists(REF_DIR), "no stored TabFM reference dumps")

model_root <- Sys.getenv("TABFOUND_TABFM_DIR", unset = "")
skip_if(!nzchar(model_root) || !dir.exists(model_root),
        "TABFOUND_TABFM_DIR not configured")

for (fx in c("clf_tiny", "clf_iris", "reg_tiny")) {
  local({
    fixture <- fx
    test_that(paste0("tabfound matches Python TabFM on ", fixture), {
      ref <- file.path(REF_DIR, fixture)
      skip_if(!file.exists(file.path(ref, "reference.json")),
              paste("no reference dump for", fixture))

      res <- parity_tabfm(ref, FIXTURE_DIR, fixture, fixture, model_root)
      graded <- grade_parity(res$summary)

      failed <- graded[!graded$pass, , drop = FALSE]
      expect_equal(
        nrow(failed), 0L,
        info = if (nrow(failed)) paste(
          sprintf("%s: max_abs=%.3e max_scaled=%.3e",
                  failed$stage, failed$max_abs, failed$max_scaled),
          collapse = "; ") else ""
      )

      # The cell embedder is deterministic arithmetic with no attention
      # in it, so it has no excuse to be anything but exact. Anything
      # else means the Fourier expansion or the feature grouping drifted.
      cell <- graded[graded$stage == "stage:cell", , drop = FALSE]
      if (nrow(cell)) {
        expect_identical(cell$max_abs[1], 0,
                         label = "cell embedder must be bit-identical")
      }
    })
  })
}


test_that("chunking test rows does not change predictions", {
  # Every test row's prediction depends only on the training context:
  # the column stage masks its inducing points to the training rows, the
  # row stages treat each row independently, and the ICL stage masks
  # attention to the training positions. Chunking is therefore exact in
  # exact arithmetic -- if a future change breaks that dependency
  # structure, chunk size would start moving predictions by a lot.
  #
  # In float32 it is not bit-exact: a different sequence length gives the
  # matmul kernels a different tiling and so a different summation order.
  # That shows up at ~2e-6, the same order as the gap to the Python
  # reference itself, which is why the bound below is 1e-5 and not 0.
  ref <- file.path(REF_DIR, "clf_tiny")
  skip_if(!file.exists(file.path(ref, "reference.json")), "no reference dump")

  fx <- read_parity_fixture(FIXTURE_DIR, "clf_tiny")
  ctx <- load_backend_model(model_root, task = "classification",
                            backend = "tabfm", device = "cpu")

  build <- function(chunk) {
    spec <- tabfm_classifier(ctx, predict_chunk_size = chunk)
    state <- spec$fit(fx$x_train, as.integer(fx$y_train))
    spec$predict(state, fx$x_test, "prob")
  }
  p_whole <- build(1024L)
  p_split <- build(5L)

  expect_equal(dim(p_whole), dim(p_split))
  expect_lt(max(abs(p_whole - p_split)), 1e-5)
})
