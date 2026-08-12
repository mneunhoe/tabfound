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


# ---------------------------------------------------------------------------
# Stage chunking
# ---------------------------------------------------------------------------

# A randomly-initialised model at a fraction of the width. TabFM's real
# checkpoint is 1.64 B parameters and 6.6 GB, so the questions that are
# about the driver rather than the weights get asked here.
tiny_tabfm <- function(is_classifier = TRUE) {
  net <- tabfm_model(list(
    embed_dim = 16L, ff_factor = 2L, feature_group_size = 1L, num_freq = 4L,
    col_num_blocks = 2L, col_nhead = 2L, col_num_inds = 4L,
    row_num_blocks = 2L, row_nhead = 2L, row_num_cls = 2L,
    icl_num_blocks = 2L, icl_nhead = 2L,
    decoder_hidden = 16L, is_classifier = is_classifier,
    max_classes = if (is_classifier) 4L else 1L
  ))
  net$eval()
  net
}

tiny_tabfm_batch <- function(n_tr, n_te, p, seed = 1L) {
  set.seed(seed)
  x <- torch::torch_randn(c(1L, n_tr + n_te, p))
  y <- torch::torch_full(c(1L, n_tr + n_te), -100.0)
  y[1, 1:n_tr] <- torch::torch_randint(0L, 3L, n_tr)$to(dtype = torch::torch_float())
  list(x = x, y = y,
       train_size = torch::torch_tensor(n_tr, dtype = torch::torch_long())$
         reshape(1L))
}

test_that("a TabFM chunk that cannot bite reproduces the plain pass exactly", {
  net <- tiny_tabfm()
  b <- tiny_tabfm_batch(30L, 9L, 6L, seed = 91L)
  plain <- as.array(torch::with_no_grad(net(b$x, b$y, b$train_size)))
  for (rc in c(39L, 40L, 1000L)) {
    expect_identical(
      as.array(torch::with_no_grad(
        net(b$x, b$y, b$train_size, row_chunk_size = rc))), plain)
  }
})

test_that("TabFM stage chunking agrees with the plain pass at every size", {
  # 8, 16 and 25 put the train/test boundary at row 30 inside a chunk.
  # TabFM restricts context by masking rather than slicing, so the
  # chunk-relative train count feeds a *mask* rather than a slice -- a
  # different mechanism reaching the same place, and the same way to get
  # it wrong.
  net <- tiny_tabfm()
  b <- tiny_tabfm_batch(30L, 9L, 6L, seed = 92L)
  plain <- as.array(torch::with_no_grad(net(b$x, b$y, b$train_size)))
  scale <- max(abs(plain))
  for (rc in c(1L, 7L, 8L, 16L, 25L, 30L, 32L)) {
    got <- as.array(torch::with_no_grad(
      net(b$x, b$y, b$train_size, row_chunk_size = rc)))
    expect_equal(dim(got), dim(plain))
    expect_lt(max(abs(got - plain)), 1e-4 * scale)
  }
})

test_that("chunking a cached TabFM prediction agrees with the uncached one", {
  net <- tiny_tabfm()
  b <- tiny_tabfm_batch(30L, 11L, 6L, seed = 93L)
  cache <- torch::with_no_grad(net$build_kv_cache(
    b$x[, 1:30, ], b$y[, 1:30], b$train_size))
  xq <- b$x[, 31:41, ]
  base <- as.array(torch::with_no_grad(net(xq, NULL, NULL, kv_cache = cache)))
  scale <- max(abs(base))
  for (rc in c(1L, 4L, 11L, 100L)) {
    got <- as.array(torch::with_no_grad(
      net(xq, NULL, NULL, kv_cache = cache, row_chunk_size = rc)))
    expect_lt(max(abs(got - base)), 1e-4 * scale)
  }
})

test_that("the TabFM regressor chunks the same way the classifier does", {
  net <- tiny_tabfm(is_classifier = FALSE)
  b <- tiny_tabfm_batch(26L, 7L, 5L, seed = 94L)
  plain <- as.array(torch::with_no_grad(net(b$x, b$y, b$train_size)))
  got <- as.array(torch::with_no_grad(
    net(b$x, b$y, b$train_size, row_chunk_size = 9L)))
  expect_equal(dim(got), dim(plain))
  expect_lt(max(abs(got - plain)), 1e-4 * max(abs(plain)))
})


test_that("column-chunking TabFM's pre-pass changes nothing", {
  # The pre-pass is TabFM's ceiling: a row-chunked forward cannot start
  # until the summaries exist, and measured at 4,000 x 90 the cumulative
  # peak went 12.9 GB after the cell embedder to 29.2 GB after this
  # stage. Chunking it by column is what makes 8,000 x 90 run at all.
  net <- tiny_tabfm()
  b <- tiny_tabfm_batch(30L, 0L, 8L, seed = 95L)
  ref <- torch::with_no_grad(
    net$col_embedder$build_hidden(
      net$cell_embedder(net$fill_missing(b$x), b$y, b$train_size, NULL),
      b$train_size))
  emb <- net$cell_embedder(net$fill_missing(b$x), b$y, b$train_size, NULL)
  for (cc in c(1L, 3L, 4L, 100L)) {
    got <- torch::with_no_grad(
      net$col_embedder$build_hidden(emb, b$train_size, cc))
    expect_equal(as.integer(got$out$size()), as.integer(ref$out$size()))
    expect_lt(max(abs(as.array(got$out) - as.array(ref$out))),
              1e-4 * max(abs(as.array(ref$out))))
    expect_length(got$hidden, length(ref$hidden))
    for (i in seq_along(ref$hidden)) {
      # Unfolded from the `(b, hc)` attention batch before concatenating
      # and folded back after, so the shape has to survive the round trip
      # exactly.
      expect_equal(as.integer(got$hidden[[i]]$size()),
                   as.integer(ref$hidden[[i]]$size()))
      expect_lt(max(abs(as.array(got$hidden[[i]]) - as.array(ref$hidden[[i]]))),
                1e-4 * max(abs(as.array(ref$hidden[[i]]))))
    }
  }
})

test_that("both TabFM chunk axes compose", {
  net <- tiny_tabfm()
  b <- tiny_tabfm_batch(30L, 9L, 8L, seed = 96L)
  plain <- as.array(torch::with_no_grad(net(b$x, b$y, b$train_size)))
  scale <- max(abs(plain))
  for (cc in c(1L, 4L)) {
    got <- as.array(torch::with_no_grad(
      net(b$x, b$y, b$train_size, row_chunk_size = 8L, col_chunk_size = cc)))
    expect_lt(max(abs(got - plain)), 1e-4 * scale)
  }
})
