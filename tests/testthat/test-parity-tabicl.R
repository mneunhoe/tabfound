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


# ---------------------------------------------------------------------------
# Stage chunking
# ---------------------------------------------------------------------------

# A randomly-initialised model at a fraction of the width: the chunking
# is a question about the driver, not about the weights.
tiny_tabicl <- function(task = "classification") {
  net <- tabicl_model(list(
    arch = "tabicl", head = task,
    max_classes = if (task == "classification") 6L else 0L,
    num_quantiles = if (task == "classification") 0L else 5L,
    embed_dim = 16L, ff_factor = 2L, activation = "gelu",
    col_num_blocks = 2L, col_nhead = 2L, col_num_inds = 4L,
    col_feature_group = TRUE, col_feature_group_size = 3L,
    col_target_aware = TRUE, col_ssmax = "ssmax",
    row_num_blocks = 2L, row_nhead = 2L, row_num_cls = 2L,
    row_rope_base = 100000, row_rope_interleaved = FALSE,
    icl_num_blocks = 2L, icl_nhead = 2L, icl_ssmax = "ssmax",
    bias_free_ln = FALSE
  ))
  net$eval()
  net
}

test_that("a chunk that cannot bite reproduces the plain pass exactly", {
  # The property the whole driver rests on. A loop that copies, reorders
  # or re-derives something when it has nothing to divide fails here and
  # nowhere else.
  net <- tiny_tabicl()
  set.seed(51)
  n_tr <- 30L; n_te <- 9L
  x <- torch::torch_randn(c(1L, n_tr + n_te, 7L))
  y <- torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())
  plain <- as.array(torch::with_no_grad(net(x, y)))
  for (rc in c(n_tr + n_te, n_tr + n_te + 1L, 10000L)) {
    expect_identical(
      as.array(torch::with_no_grad(net(x, y, row_chunk_size = rc))), plain)
  }
})

test_that("stage chunking agrees with the plain pass at every chunk size", {
  net <- tiny_tabicl()
  set.seed(52)
  n_tr <- 30L; n_te <- 9L
  x <- torch::torch_randn(c(1L, n_tr + n_te, 7L))
  y <- torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())
  plain <- as.array(torch::with_no_grad(net(x, y)))
  scale <- max(abs(plain))
  # 8, 16 and 25 all put the train/test boundary at row 30 *inside* a
  # chunk, which is the only place the target embedding can go to the
  # wrong rows. Omitting it there leaves every shape intact.
  for (rc in c(1L, 7L, 8L, 16L, 25L, 30L, 32L)) {
    got <- as.array(torch::with_no_grad(net(x, y, row_chunk_size = rc)))
    expect_equal(dim(got), dim(plain))
    expect_lt(max(abs(got - plain)), 1e-4 * scale)
  }
})

test_that("the target embedding follows the rows it belongs to", {
  # Sharper than the tolerance check above: if a straddling chunk skipped
  # `add_target`, the labelled rows in it would be embedded as if
  # unlabelled, and a chunk size that splits the training rows would
  # disagree with one that does not. Both must match the plain pass.
  net <- tiny_tabicl()
  set.seed(53)
  n_tr <- 24L; n_te <- 8L
  x <- torch::torch_randn(c(1L, n_tr + n_te, 6L))
  y <- torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())
  plain <- as.array(torch::with_no_grad(net(x, y)))
  inside  <- as.array(torch::with_no_grad(net(x, y, row_chunk_size = 10L)))
  aligned <- as.array(torch::with_no_grad(net(x, y, row_chunk_size = 24L)))
  scale <- max(abs(plain))
  expect_lt(max(abs(inside - plain)), 1e-4 * scale)
  expect_lt(max(abs(aligned - plain)), 1e-4 * scale)
  expect_lt(max(abs(inside - aligned)), 1e-4 * scale)
})

test_that("chunking a cached TabICL prediction agrees with the uncached one", {
  net <- tiny_tabicl()
  set.seed(54)
  n_tr <- 30L; n_te <- 11L
  x <- torch::torch_randn(c(1L, n_tr + n_te, 6L))
  y <- torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())
  cache <- torch::with_no_grad(net$build_kv_cache(x[, 1:n_tr, ], y))
  xq <- x[, (n_tr + 1L):(n_tr + n_te), ]
  base <- as.array(torch::with_no_grad(net(xq, NULL, kv_cache = cache)))
  scale <- max(abs(base))
  for (rc in c(1L, 4L, 11L, 100L)) {
    got <- as.array(torch::with_no_grad(
      net(xq, NULL, kv_cache = cache, row_chunk_size = rc)))
    expect_lt(max(abs(got - base)), 1e-4 * scale)
  }
  # Every row here is a test row, so the summaries come from the cache
  # and a chunk larger than the input is the cached pass unchanged.
  expect_identical(
    as.array(torch::with_no_grad(
      net(xq, NULL, kv_cache = cache, row_chunk_size = 100L))), base)
})

test_that("the regressor chunks the same way the classifier does", {
  net <- tiny_tabicl("regressor")
  set.seed(55)
  n_tr <- 26L; n_te <- 7L
  x <- torch::torch_randn(c(1L, n_tr + n_te, 5L))
  y <- torch::torch_randn(c(1L, n_tr))
  plain <- as.array(torch::with_no_grad(net(x, y)))
  got <- as.array(torch::with_no_grad(net(x, y, row_chunk_size = 9L)))
  expect_equal(dim(got), dim(plain))
  expect_lt(max(abs(got - plain)), 1e-4 * max(abs(plain)))
})


test_that("chunking TabICL's in-context queries computes the same thing", {
  # The ICL block restricts its keys to the labelled rows by slicing the
  # *whole* normalized input, so the key/value side is fixed however the
  # queries are split -- which is what makes query chunking legitimate
  # here and not for plain self-attention, where the keys are the
  # queries. The block ignores the factor on that branch rather than
  # quietly changing what each row attends over.
  net <- tiny_tabicl()
  set.seed(56)
  n_tr <- 28L; n_te <- 9L
  x <- torch::torch_randn(c(1L, n_tr + n_te, 6L))
  y <- torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())
  base <- as.array(torch::with_no_grad(net(x, y)))
  scale <- max(abs(base))
  for (k in c(2L, 3L, 8L, 64L)) {
    got <- as.array(torch::with_no_grad(
      net(x, y, save_peak_memory_factor = k)))
    expect_equal(dim(got), dim(base))
    expect_lt(max(abs(got - base)), 1e-4 * scale)
  }

  # And against a cache, where the keys come from the cache instead.
  cache <- torch::with_no_grad(net$build_kv_cache(x[, 1:n_tr, ], y))
  xq <- x[, (n_tr + 1L):(n_tr + n_te), ]
  cb <- as.array(torch::with_no_grad(net(xq, NULL, kv_cache = cache)))
  for (k in c(2L, 8L)) {
    got <- as.array(torch::with_no_grad(
      net(xq, NULL, kv_cache = cache, save_peak_memory_factor = k)))
    expect_lt(max(abs(got - cb)), 1e-4 * max(abs(cb)))
  }
})

test_that("the two TabICL mechanisms compose", {
  net <- tiny_tabicl()
  set.seed(57)
  n_tr <- 28L; n_te <- 9L
  x <- torch::torch_randn(c(1L, n_tr + n_te, 6L))
  y <- torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())
  base <- as.array(torch::with_no_grad(net(x, y)))
  got <- as.array(torch::with_no_grad(
    net(x, y, row_chunk_size = 11L, save_peak_memory_factor = 4L)))
  expect_lt(max(abs(got - base)), 1e-4 * max(abs(base)))
})


test_that("column-chunking the summary pre-pass changes nothing", {
  # Every column is embedded on its own -- the set transformer folds
  # `(B, HC)` into its batch -- so this is a reordering, not a different
  # computation. The concatenation order is the correctness property: a
  # permutation gives every column another column's summary, and every
  # number downstream stays finite and plausible.
  net <- tiny_tabicl()
  set.seed(58)
  n_tr <- 24L; p <- 9L
  x <- torch::torch_randn(c(1L, n_tr, p))
  y <- torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())

  ref <- torch::with_no_grad(net$col_embedder$build_hidden(x, y))
  for (cc in c(1L, 3L, 4L, 100L)) {
    got <- torch::with_no_grad(net$col_embedder$build_hidden(x, y, cc))
    expect_length(got$hidden, length(ref$hidden))
    for (i in seq_along(ref$hidden)) {
      expect_equal(as.integer(got$hidden[[i]]$size()),
                   as.integer(ref$hidden[[i]]$size()))
      expect_lt(max(abs(as.array(got$hidden[[i]]) - as.array(ref$hidden[[i]]))),
                1e-4 * max(abs(as.array(ref$hidden[[i]]))))
    }
    expect_lt(max(abs(as.array(got$out) - as.array(ref$out))),
              1e-4 * max(abs(as.array(ref$out))))
  }

  # A chunked forward wants the summaries and not the full-width output,
  # and accumulating that output would defeat the point: measured on a
  # 12,000 x 300 table it was the difference between 36.9 GB and 29.8 GB.
  lean <- torch::with_no_grad(
    net$col_embedder$build_hidden(x, y, 4L, want_out = FALSE))
  expect_null(lean$out)
  for (i in seq_along(ref$hidden)) {
    expect_lt(max(abs(as.array(lean$hidden[[i]]) - as.array(ref$hidden[[i]]))),
              1e-4 * max(abs(as.array(ref$hidden[[i]]))))
  }
})

test_that("both TabICL chunk axes compose", {
  net <- tiny_tabicl()
  set.seed(59)
  n_tr <- 30L; n_te <- 9L
  x <- torch::torch_randn(c(1L, n_tr + n_te, 9L))
  y <- torch::torch_randint(0L, 3L, c(1L, n_tr))$to(dtype = torch::torch_float())
  plain <- as.array(torch::with_no_grad(net(x, y)))
  scale <- max(abs(plain))
  for (cc in c(1L, 4L)) {
    got <- as.array(torch::with_no_grad(
      net(x, y, row_chunk_size = 11L, col_chunk_size = cc)))
    expect_lt(max(abs(got - plain)), 1e-4 * scale)
  }
})
