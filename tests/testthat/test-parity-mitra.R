# End-to-end parity against AutoGluon's Mitra.
#
# Replays stored reference dumps — no Python needed here. Regenerate with
# `Rscript inst/parity/run-parity.R mitra --regenerate`.
#
# Skips unless TABFOUND_MITRA_CLF_DIR / TABFOUND_MITRA_REG_DIR point at
# the Hub snapshots.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

parity_dir <- tabfound_file("parity")
skip_if(!nzchar(parity_dir), "parity harness not installed")

source(file.path(parity_dir, "fixtures.R"), local = TRUE)
source(file.path(parity_dir, "compare.R"), local = TRUE)

FIXTURE_DIR <- file.path(parity_dir, "fixtures")
REF_DIR     <- file.path(parity_dir, "reference", "mitra")
skip_if(!dir.exists(REF_DIR), "no stored Mitra reference dumps")

model_dir_for <- function(fixture) {
  var <- if (startsWith(fixture, "clf")) "TABFOUND_MITRA_CLF_DIR"
         else "TABFOUND_MITRA_REG_DIR"
  d <- Sys.getenv(var, unset = "")
  if (!nzchar(d) || !dir.exists(d)) NULL else d
}

for (fx in c("clf_tiny", "clf_iris", "reg_tiny", "reg_iris")) {
  local({
    fixture <- fx
    test_that(paste0("tabfound matches Mitra on ", fixture), {
      ref <- file.path(REF_DIR, fixture)
      skip_if(!file.exists(file.path(ref, "reference.json")),
              paste("no reference dump for", fixture))
      md <- model_dir_for(fixture)
      skip_if(is.null(md), "Mitra artifacts not configured")

      res <- parity_mitra(ref, FIXTURE_DIR, fixture, fixture, md)
      graded <- grade_parity(res$summary)

      failed <- graded[!graded$pass, , drop = FALSE]
      expect_equal(
        nrow(failed), 0L,
        info = if (nrow(failed)) paste(
          sprintf("%s: max_abs=%.3e max_scaled=%.3e",
                  failed$stage, failed$max_abs, failed$max_scaled),
          collapse = "; ") else ""
      )

      # The quantile embedding and the packed input are deterministic
      # arithmetic with no attention in them; anything but exact means
      # the bucketing or the packing drifted.
      for (st in c("stage:quantile", "stage:embedded")) {
        row <- graded[graded$stage == st, , drop = FALSE]
        if (nrow(row)) expect_identical(row$max_abs[1], 0, label = st)
      }
    })
  })
}


test_that("one missing value silently deletes a whole Mitra feature column", {
  # Not a NaN-propagation story, which is what it looks like from the
  # outside: `torch.quantile` gives all-NaN quantiles for the affected
  # column, `searchsorted` then buckets every value to 0, the variance is
  # zero, and the zero-variance guard flattens the column. The output is
  # perfectly finite and the feature has vanished.
  #
  # This is the reference's own behaviour, verified against it -- not an
  # artefact of the port. Nothing in the normal flow reaches it: the
  # predictors run `mitra_preprocessor_fit()` first, which mean-imputes
  # exactly as AutoGluon's own preprocessor does, so the network never
  # sees a NaN. The behaviour is still pinned, because a caller invoking
  # this function directly can still hit it and the failure is silent.
  x_support <- torch::torch_tensor(
    array(c(1, 2, NaN, 4, 5, 6, 7, 8), dim = c(1, 4, 2))
  )
  x_query <- torch::torch_tensor(array(c(1.5, 3.5, 6.5, 7.5), dim = c(1, 2, 2)))
  out <- mitra_quantile_embedding(x_support, x_query)

  sup <- as.array(out$support)
  expect_false(anyNA(sup))                  # no NaN escapes
  expect_true(all(sup[1, , 1] == 0))        # ...the column is just gone
  expect_gt(max(abs(sup[1, , 2])), 0.1)     # the clean column survives

  # End to end the column survives, because the predictor imputes before
  # the network gets a chance to lose it. Comparing against a fit on the
  # already-imputed matrix is the check that matters: the two must agree,
  # or the imputation is happening somewhere other than where it should.
  md <- model_dir_for("clf_tiny")
  skip_if(is.null(md), "Mitra artifacts not configured")
  fx <- read_parity_fixture(FIXTURE_DIR, "clf_tiny")
  X <- fx$x_train
  X_na <- X
  X_na[3, 2] <- NA_real_
  X_imp <- X_na
  X_imp[3, 2] <- mean(X_na[-3, 2])

  fit_on <- function(x) {
    predict(fit(tabular_classifier(md, backend = "mitra", device = "cpu",
                                   random_mirror_x = FALSE), x,
                as.integer(fx$y_train)),
            fx$x_test, type = "prob")
  }
  p_na <- fit_on(X_na)
  expect_false(anyNA(p_na))
  expect_equal(p_na, fit_on(X_imp), tolerance = 1e-6)

  # ...and the backend now says so, so `tabfound()` stops layering its own
  # imputer in front of the parity-checked one.
  expect_true(isTRUE(get_backend("mitra")$handles_missing))
})
