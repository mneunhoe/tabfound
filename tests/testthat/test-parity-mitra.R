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


# ---------------------------------------------------------------------------
# Within-sublayer chunking
# ---------------------------------------------------------------------------

# A randomly-initialised model at a fraction of the width, so the
# chunking can be checked without the 303 MB checkpoint.
tiny_mitra <- function(task = "CLASSIFICATION") {
  net <- mitra_model(list(dim = 32L, dim_output = 4L, n_layers = 2L,
                          n_heads = 4L, task = task))
  net$eval()
  net
}

test_that("chunking a Mitra layer computes the same thing at every factor", {
  # All four sublayers are independent across rows -- a query row attends
  # to the support and never to another query row, the feature attention
  # is a sequence per row, and both MLPs are elementwise -- so this
  # reorganises work that was already separate.
  #
  # It is *not* bit-identical, and that is worth being precise about,
  # because v2.6's equivalent is. The observation attention puts rows in
  # the sequence position, so a chunk changes the query length the kernel
  # sees; v2.6 splits a fold of leading dimensions and leaves every
  # attention's own shape alone. What comes out is float32 reduction
  # order: 5.4e-7 of the logits' scale on the real checkpoint, against
  # 1.4e-5 for the stage chunking on TabPFN v3.
  net <- tiny_mitra()
  set.seed(71)
  xs <- torch::torch_randn(c(1L, 40L, 6L))
  ys <- torch::torch_randint(0L, 3L, c(1L, 40L))$to(dtype = torch::torch_float())
  xq <- torch::torch_randn(c(1L, 11L, 6L))

  base <- as.array(torch::with_no_grad(net(xs$clone(), ys, xq$clone())))
  scale <- max(abs(base))
  for (k in c(2L, 3L, 8L, 40L, 100L)) {
    got <- as.array(torch::with_no_grad(
      net(xs$clone(), ys, xq$clone(), save_peak_memory_factor = k)))
    expect_equal(dim(got), dim(base))
    expect_lt(max(abs(got - base)), 1e-5 * scale)
  }
})

test_that("chunking a cached Mitra prediction agrees with the uncached one", {
  net <- tiny_mitra()
  set.seed(72)
  xs <- torch::torch_randn(c(1L, 40L, 6L))
  ys <- torch::torch_randint(0L, 3L, c(1L, 40L))$to(dtype = torch::torch_float())
  xq <- torch::torch_randn(c(1L, 11L, 6L))

  cache <- torch::with_no_grad(net$build_kv_cache(xs$clone(), ys))
  base <- as.array(torch::with_no_grad(
    net(NULL, NULL, xq$clone(), kv_cache = cache)))
  scale <- max(abs(base))
  for (k in c(2L, 3L, 8L)) {
    got <- as.array(torch::with_no_grad(
      net(NULL, NULL, xq$clone(), kv_cache = cache,
          save_peak_memory_factor = k)))
    expect_lt(max(abs(got - base)), 1e-5 * scale)
  }
})

test_that("a Mitra layer's chunking does not depend on the row count dividing", {
  # 17 rows into 5 chunks is 4+4+4+4+1: the ragged last chunk is where a
  # driver that assumes an even split falls over, and it is one row wide,
  # which is where a squeezed dimension would.
  net <- tiny_mitra()
  set.seed(73)
  xs <- torch::torch_randn(c(1L, 17L, 5L))
  ys <- torch::torch_randint(0L, 3L, c(1L, 17L))$to(dtype = torch::torch_float())
  xq <- torch::torch_randn(c(1L, 3L, 5L))
  base <- as.array(torch::with_no_grad(net(xs$clone(), ys, xq$clone())))
  got <- as.array(torch::with_no_grad(
    net(xs$clone(), ys, xq$clone(), save_peak_memory_factor = 5L)))
  expect_equal(dim(got), dim(base))
  expect_lt(max(abs(got - base)), 1e-5 * max(abs(base)))
})

test_that("chunked_evaluate_axis writes back through a non-leading axis", {
  # The whole memory saving rests on `torch_split()` returning views
  # along the split axis, so that `add_()` reaches the original storage
  # instead of a copy. If that ever stopped holding, the results would
  # still be right and the saving would silently be gone.
  x <- torch::torch_zeros(c(2L, 6L, 3L))
  out <- chunked_evaluate_axis(function(z) torch::torch_ones_like(z),
                               x, factor = 3L, axis = 2L)
  expect_true(as.logical((out == 1)$all()))
  # `x` is the same object, mutated -- not a fresh tensor.
  expect_true(as.logical((x == 1)$all()))

  # And with no factor it is the ordinary out-of-place expression.
  y <- torch::torch_zeros(c(2L, 6L, 3L))
  out2 <- chunked_evaluate_axis(function(z) torch::torch_ones_like(z),
                                y, factor = NULL, axis = 2L)
  expect_true(as.logical((out2 == 1)$all()))
  expect_true(as.logical((y == 0)$all()))
})
