# Parity against the PyPI `tabpfn` package's v2.6 architecture.
#
# Replays stored reference dumps -- no Python needed here. Regenerate them
# with `Rscript inst/parity/run-parity.R tabpfn26 --regenerate`.
#
# Needs the converted checkpoints, so the test skips unless
# TABFOUND_TABPFN26_CLF_DIR / TABFOUND_TABPFN26_REG_DIR point at them.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

parity_dir <- tabfound_file("parity")
skip_if(!nzchar(parity_dir), "parity harness not installed")

source(file.path(parity_dir, "fixtures.R"), local = TRUE)
source(file.path(parity_dir, "compare.R"), local = TRUE)

FIXTURE_DIR <- file.path(parity_dir, "fixtures")
REF_DIR     <- file.path(parity_dir, "reference", "tabpfn26")
skip_if(!dir.exists(REF_DIR), "no stored TabPFN v2.6 reference dumps")

model_dir_for <- function(fixture) {
  var <- if (startsWith(fixture, "clf")) "TABFOUND_TABPFN26_CLF_DIR"
         else "TABFOUND_TABPFN26_REG_DIR"
  d <- Sys.getenv(var, unset = "")
  if (!nzchar(d) || !dir.exists(d)) NULL else d
}

# `clf_binary_missing` is the one that matters most: v2.6 moved
# constant-column removal, mean imputation and the NaN/Inf indicator
# channel inside the architecture, and that fixture has all three.
fixtures <- c("clf_iris", "clf_binary_missing", "clf_tiny",
              "reg_iris", "reg_skewed", "reg_tiny")

for (fx in fixtures) {
  local({
    fixture <- fx
    test_that(paste0("tabfound matches Python TabPFN v2.6 on ", fixture), {
      ref <- file.path(REF_DIR, fixture)
      skip_if(!dir.exists(ref), paste("no reference dump for", fixture))
      md <- model_dir_for(fixture)
      skip_if(is.null(md), "converted TabPFN v2.6 artifacts not configured")

      res <- parity_tabpfn26(ref, FIXTURE_DIR, fixture, fixture, md)
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


ENS_DIR <- file.path(parity_dir, "reference", "tabpfn26-ensemble")

# The `_nofp` variants are the deterministic contract: with the
# fingerprint feature off, every stage must agree. The fingerprinted twins
# are graded in the driver, where a failure can be checked against its
# twin; here only the deterministic ones are asserted.
for (fx in c("clf_iris_nofp", "clf_binary_missing_nofp",
             "reg_iris_nofp", "reg_skewed_nofp",
             # Declared categorical columns, and the same fixture with
             # nothing declared so only the low-cardinality one is inferred.
             "clf_categorical_nofp", "clf_categorical_auto_nofp",
             "reg_categorical_nofp", "reg_categorical_auto_nofp")) {
  local({
    fixture <- fx
    test_that(paste0("tabfound matches the Python v2.6 ensemble on ", fixture), {
      ref <- file.path(ENS_DIR, fixture)
      skip_if(!dir.exists(ref), paste("no ensemble reference for", fixture))
      md <- model_dir_for(fixture)
      skip_if(is.null(md), "converted TabPFN v2.6 artifacts not configured")

      res <- parity_tabpfn(ref, FIXTURE_DIR,
                           sub("(_auto)?(_nofp)?$", "", fixture), fixture, md)
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

      # Per-member model inputs must be bit-identical, not merely close.
      # This is the stage that proves the whole preprocessing chain --
      # quantile transform, polynomial products, SVD, fingerprint,
      # shuffle -- reproduces the reference rather than approximating it.
      pre <- graded[startsWith(graded$stage, "preprocess:X"), , drop = FALSE]
      expect_gt(nrow(pre), 0L)
      expect_true(all(pre$max_abs == 0))
    })
  })
}


test_that("chunked evaluation never moves a bit, on any fixture", {
  # The sharpest assertion in the harness: `save_peak_memory_factor`
  # reorganises work that was already independent, so the output must be
  # bit-identical to the unchunked pass -- not close, identical.
  md_c <- model_dir_for("clf_iris"); md_r <- model_dir_for("reg_iris")
  skip_if(is.null(md_c) || is.null(md_r),
          "converted TabPFN v2.6 artifacts not configured")
  seen <- 0L
  for (fx in fixtures) {
    ref <- file.path(REF_DIR, fx)
    if (!dir.exists(ref)) next
    res <- parity_tabpfn26(ref, FIXTURE_DIR, fx, fx, model_dir_for(fx))
    row <- res$summary[res$summary$stage == "selfcheck:chunked", , drop = FALSE]
    if (!nrow(row)) next
    seen <- seen + 1L
    expect_identical(row$max_abs, 0, info = fx)
  }
  expect_gt(seen, 0L)
})


test_that("the KV cache is exact wherever the reference's is", {
  # Where the cache changes the answer it is a property of the data, not
  # of the port -- it fixes the constant-column and informative-feature
  # masks on the training rows. So the assertion is conditional: the
  # harness only emits `selfcheck:cached` for fixtures where the
  # reference's own cache is exact, and there ours must be too. The
  # fixtures where it is not emit `cache:shifts-prediction` instead, and
  # `forward:cached` is what grades those.
  md <- model_dir_for("clf_iris")
  skip_if(is.null(md), "converted TabPFN v2.6 artifacts not configured")
  exact <- 0L; shifted <- 0L
  for (fx in fixtures) {
    ref <- file.path(REF_DIR, fx)
    if (!dir.exists(ref)) next
    res <- parity_tabpfn26(ref, FIXTURE_DIR, fx, fx, model_dir_for(fx))
    st <- res$summary
    if (nrow(st[st$stage == "selfcheck:cached", , drop = FALSE])) {
      exact <- exact + 1L
      expect_identical(st$max_abs[st$stage == "selfcheck:cached"], 0, info = fx)
    }
    if (nrow(st[st$stage == "cache:shifts-prediction", , drop = FALSE])) {
      shifted <- shifted + 1L
      # It has to actually shift -- a zero here would mean the harness
      # thinks the reference diverged while ours did not, i.e. the cache
      # is not being used at all.
      expect_gt(st$max_abs[st$stage == "cache:shifts-prediction"], 0)
    }
    # Either way, R's cache must agree with the reference's cache.
    expect_true(all(grade_parity(st)$pass[st$stage == "forward:cached"]), info = fx)
  }
  expect_gt(exact, 0L)
  expect_gt(shifted, 0L)
})


test_that("the v2.6 decoding path is exact given the reference's own logits", {
  # Sharper than the graded tolerance: `decode:*` feeds both sides the
  # same logits, so only the softmax / bar-distribution arithmetic can
  # differ, and it should not differ at all. A regression here is a real
  # bug, not accumulated float32 drift, so it is asserted separately
  # rather than hidden behind the stage tolerance.
  fixture <- "reg_iris"
  ref <- file.path(REF_DIR, fixture)
  skip_if(!dir.exists(ref), "no reference dump")
  md <- model_dir_for(fixture)
  skip_if(is.null(md), "converted TabPFN v2.6 artifacts not configured")

  res <- parity_tabpfn26(ref, FIXTURE_DIR, fixture, fixture, md)
  decode <- res$summary[startsWith(res$summary$stage, "decode:"), , drop = FALSE]
  expect_gt(nrow(decode), 0L)
  expect_true(all(decode$max_abs < 1e-6))
})
