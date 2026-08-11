# End-to-end parity against the PyPI `tabicl` package.
#
# Replays stored reference dumps — no Python needed here. Regenerate with
# `Rscript inst/parity/run-parity.R tabicl --regenerate`.
#
# Needs the converted checkpoints, so the test skips unless
# TABFOUND_TABICL_CLF_DIR / TABFOUND_TABICL_REG_DIR point at them.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

parity_dir <- tabfound_file("parity")
skip_if(!nzchar(parity_dir), "parity harness not installed")

source(file.path(parity_dir, "fixtures.R"), local = TRUE)
source(file.path(parity_dir, "compare.R"), local = TRUE)

FIXTURE_DIR <- file.path(parity_dir, "fixtures")
REF_DIR     <- file.path(parity_dir, "reference", "tabicl")
skip_if(!dir.exists(REF_DIR), "no stored TabICL reference dumps")

model_dir_for <- function(fixture) {
  var <- if (startsWith(fixture, "clf")) "TABFOUND_TABICL_CLF_DIR"
         else "TABFOUND_TABICL_REG_DIR"
  d <- Sys.getenv(var, unset = "")
  if (!nzchar(d) || !dir.exists(d)) NULL else d
}

for (fx in c("clf_tiny", "clf_iris", "reg_tiny", "reg_iris")) {
  local({
    fixture <- fx
    test_that(paste0("tabfound matches Python TabICL on ", fixture), {
      ref <- file.path(REF_DIR, fixture)
      skip_if(!file.exists(file.path(ref, "reference.json")),
              paste("no reference dump for", fixture))
      md <- model_dir_for(fixture)
      skip_if(is.null(md), "converted TabICL artifacts not configured")

      res <- parity_tabicl(ref, FIXTURE_DIR, fixture, fixture, md)
      graded <- grade_parity(res$summary)

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


test_that("TabICL's network still propagates NaN -- its predictor imputes", {
  # Both halves of this matter. The *network* has no nan_to_num and no
  # imputation of its own, so a NaN fed straight to it comes back as NaN
  # logits. The *predictor* never does that: it runs the reference
  # wrapper's mean `SimpleImputer` first, which is the only reason the
  # backend is usable on real data at all.
  md <- model_dir_for("clf_tiny")
  skip_if(is.null(md), "converted TabICL artifacts not configured")

  fx <- read_parity_fixture(FIXTURE_DIR, "clf_tiny")
  X <- fx$x_train
  X_na <- X
  X_na[3, 2] <- NA_real_

  # Bare network: NaN in, NaN out.
  ctx <- load_backend_model(md, task = "classification", backend = "tabicl",
                            device = "cpu")
  x <- rbind(X_na, fx$x_test); storage.mode(x) <- "double"
  bare <- torch::with_no_grad({
    ctx$net(as_float_tensor(x, device = ctx$device)$unsqueeze(1L),
            as_float_tensor(matrix(as.numeric(fx$y_train), nrow = 1L),
                            device = ctx$device))
  })
  expect_true(anyNA(as.array(bare$cpu())))

  # Through the predictor: finite, and equal to a fit on the matrix with
  # the same column mean already filled in.
  X_imp <- X_na
  X_imp[3, 2] <- mean(X_na[-3, 2])
  fit_on <- function(x) {
    predict(fit(tabular_classifier(md, backend = "tabicl", device = "cpu"), x,
                as.integer(fx$y_train)),
            fx$x_test, type = "prob")
  }
  p_na <- fit_on(X_na)
  expect_false(anyNA(p_na))
  expect_equal(p_na, fit_on(X_imp), tolerance = 1e-6)
})
