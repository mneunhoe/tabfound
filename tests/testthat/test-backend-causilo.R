# Causilo 1.5: network structure and weight-free layer parity.
#
# Nothing here needs a checkpoint. The structure tests derive the
# checkpoint's key list independently of the R module tree and check the
# two agree, at the published size (against the key list read from the
# published safetensors headers, stored in `fixtures/`) and at the tiny
# size the parity reference uses. The parity tests replay tiny random
# networks built by `causilo`'s own classes:
#
#   .venvs/causilo/bin/python -I inst/parity/causilo_layers_reference.py \
#       --out inst/parity/layers/causilo_layers.safetensors
#
# Real weights are graded in `test-parity-causilo.R`.

skip_if_not_installed("torch")

ref_path <- tabfound_file("parity", "layers", "causilo_layers.safetensors.gz")

causilo_ref <- function() {
  skip_if(!nzchar(ref_path), "Causilo layer reference not found")
  skip_if_not_installed("safetensors")
  list(t = read_reference_tensors(ref_path),
       meta = jsonlite::fromJSON(sub("\\.safetensors\\.gz$", ".json", ref_path)))
}

# The checkpoint's tensor names and shapes for a config, written from the
# reference's module definitions (`causilo/v1_5/nn/`), not from the R
# modules.
causilo_expected_keys <- function(m) {
  W <- m$width; P <- 4L * W; out <- if (m$max_classes) 10L else 999L
  tc <- if (m$max_classes) 10L else 1L
  lin <- function(pfx, i, o) {
    stats::setNames(list(c(o, i), o), paste0(pfx, c(".weight", ".bias")))
  }
  ff <- function(pfx, w) c(lin(paste0(pfx, ".value"), w, 2L * w),
                           lin(paste0(pfx, ".gate"), w, 2L * w),
                           lin(paste0(pfx, ".output"), 2L * w, w))
  attn <- function(pfx, w, heads, scale) {
    hw <- w %/% heads
    c(lin(paste0(pfx, ".projection"), w, 3L * w), lin(paste0(pfx, ".output"), w, w),
      if (scale) c(lin(paste0(pfx, ".scaling.length.0"), 1L, 64L),
                   lin(paste0(pfx, ".scaling.length.2"), 64L, heads * hw),
                   lin(paste0(pfx, ".scaling.content.0"), hw, 64L),
                   lin(paste0(pfx, ".scaling.content.2"), 64L, hw)))
  }
  norm <- function(pfx, w) stats::setNames(list(w), paste0(pfx, ".weight"))
  alayer <- function(pfx, w, heads, scale) {
    c(attn(paste0(pfx, ".attention"), w, heads, scale), ff(paste0(pfx, ".feedforward"), w),
      norm(paste0(pfx, ".attention_norm"), w), norm(paste0(pfx, ".feedforward_norm"), w))
  }
  k <- list("feature_embedding.frequencies" = c(1L, 16L),
            "feature_embedding.rank_frequencies" = c(1L, 4L),
            "feature_embedding.missing" = c(1L, W))
  k <- c(k, lin("feature_embedding.projection", 41L, W),
         lin("feature_embedding.target.projection", tc, W))
  for (i in seq_len(m$column_depth) - 1L) {
    p <- sprintf("columns.layers.%d", i)
    k <- c(k, alayer(paste0(p, ".gather"), W, 4L, TRUE),
           alayer(paste0(p, ".broadcast"), W, 4L, FALSE),
           stats::setNames(list(c(256L, W)), paste0(p, ".latents")))
    r <- sprintf("row.layers.%d", i)
    k <- c(k, attn(paste0(r, ".gather"), W, 8L, TRUE), attn(paste0(r, ".broadcast"), W, 8L, TRUE),
           ff(paste0(r, ".feedforward"), W), norm(paste0(r, ".gather_norm"), W),
           norm(paste0(r, ".broadcast_norm"), W), norm(paste0(r, ".feedforward_norm"), W))
  }
  k <- c(k, list("row.latents" = c(4L, W)), norm("row.output_norm", W))
  for (j in seq_len(m$prediction_depth) - 1L) {
    k <- c(k, alayer(sprintf("prediction.layers.%d", j), P, 8L, TRUE))
  }
  k <- c(k, lin("prediction.target.projection", tc, P),
         norm("prediction.head.normalize", P),
         lin("prediction.head.hidden", P, 2L * P), lin("prediction.head.output", 2L * P, out))
  k[order(names(k))]
}

r_keys <- function(net) {
  sd <- net$state_dict()
  k <- lapply(sd, function(t) as.integer(t$shape))
  k[order(names(k))]
}

test_that("the derived key list is the published checkpoints'", {
  path <- test_path("fixtures", "causilo-1.5-keys.tsv")
  pub <- utils::read.delim(path, comment.char = "#", colClasses = "character")
  for (task in c("classification", "regression")) {
    col <- if (task == "classification") "classifier" else "regressor"
    want <- stats::setNames(lapply(strsplit(pub[[col]], "x", fixed = TRUE), as.integer),
                            pub$name)
    m <- .causilo_model_config(mem_config_causilo(task))
    got <- causilo_expected_keys(m)
    expect_identical(names(got), sort(names(want)), info = task)
    expect_identical(got, want[names(got)], info = task)
    expect_length(got, 1024L)
  }
})

test_that("the module tree is the checkpoint key list", {
  for (task in c("classification", "regression")) {
    for (dims in list(c(16L, 2L, 2L), c(24L, 3L, 1L))) {
      cfg <- tiny_causilo_config(task, width = dims[[1]], column_depth = dims[[2]],
                                 prediction_depth = dims[[3]])
      net <- suppressMessages(causilo_build(cfg, task))
      expect_identical(r_keys(net), causilo_expected_keys(.causilo_model_config(cfg)),
                       info = paste(task, paste(dims, collapse = "/")))
    }
  }
  # The task must match the checkpoint.
  expect_error(suppressMessages(causilo_build(tiny_causilo_config("regression"),
                                              "classification")), "regression model")
})

test_that("unit RMS and the bounded multiplier match their definitions", {
  x <- torch::torch_tensor(rbind(c(3, 4), c(0, 0), c(1e-30, 0)))
  y <- as.matrix(causilo_unit_rms(x))
  expect_equal(y[1, ], c(3, 4) / sqrt(12.5), tolerance = 1e-6)
  # An all-zero head stays zero instead of dividing by zero.
  expect_identical(y[2, ], c(0, 0))
  expect_true(all(is.finite(y)))
  v <- c(-100, -1, 0, 0.5, 100)
  b <- as.numeric(causilo_bounded_multiplier(torch::torch_tensor(v), 8))
  expect_equal(b, exp(log(8) * tanh(v / log(8))), tolerance = 1e-6)
  expect_true(all(b >= 1 / 8 - 1e-6 & b <= 8 + 1e-6))
})

test_that("mid-ranks count ties, ignore missing cells and saturate", {
  # One column, 5 context rows (two tied 2s, a missing cell), 3 queries.
  cells <- torch::torch_tensor(c(1, 2, 2, NaN, 3, 2, 0, NaN))$view(c(1L, 8L, 1L))
  ctx <- causilo_ecdf_context(cells, 5L)
  expect_identical(as.numeric(ctx), c(1, 2, 2, 3, Inf))
  r <- as.numeric(causilo_ecdf_ranks(cells, ctx))
  # Four finite context values: a 2 has one below and two equal.
  expect_equal(r, c(0.5, 2, 2, NaN, 3.5, 2, 0, NaN) / 4)
  # A column with no observed context value ranks nothing.
  empty <- torch::torch_tensor(c(NaN, NaN, 1))$view(c(1L, 3L, 1L))
  expect_true(all(is.nan(as.numeric(causilo_ecdf_ranks(empty, causilo_ecdf_context(empty, 2L))))))
})

# Float32 through 2 x (column + row) + 2 prediction layers against a
# different libtorch build.
expect_close <- function(r, p, tol = 2e-5, info = NULL) {
  r <- as.array(r$cpu()); p <- as.array(p$cpu())
  expect_identical(dim(r), dim(p), info = info)
  expect_identical(is.nan(r), is.nan(p), info = info)
  ok <- !is.nan(p)
  scale <- max(1, max(abs(p[ok])))
  expect_lte(max(abs(r[ok] - p[ok])) / scale, tol, label = info)
}

test_that("functional pieces match causilo's", {
  ref <- causilo_ref()$t
  expect_close(causilo_unit_rms(ref[["unit_rms.x"]]), ref[["unit_rms.y"]], 1e-6, "unit_rms")
  expect_close(causilo_bounded_multiplier(ref[["bounded.x"]], 8), ref[["bounded.y8"]], 1e-6, "b8")
  expect_close(causilo_bounded_multiplier(ref[["bounded.x"]], 2), ref[["bounded.y2"]], 1e-6, "b2")
  cells <- ref[["ecdf.cells"]]
  ctx <- causilo_ecdf_context(cells, 6L)
  expect_true(torch::torch_equal(ctx, ref[["ecdf.context"]]))
  r <- causilo_ecdf_ranks(cells, ctx)
  expect_identical(as.array(r$isnan()), as.array(ref[["ecdf.ranks"]]$isnan()))
  expect_true(torch::torch_allclose(r$nan_to_num(), ref[["ecdf.ranks"]]$nan_to_num(),
                                    rtol = 0, atol = 0))
})

# A tiny net carrying the reference's weights: the classifier's, with the
# regressor's own task-shaped tensors on top.
causilo_ref_net <- function(ref, tag, meta) {
  task <- if (tag == "clf") "classification" else "regression"
  net <- suppressMessages(causilo_model(tiny_causilo_config(task)))
  w <- list()
  for (src in unique(c("clf", tag))) {
    keys <- grep(paste0("^", src, "\\.w\\."), names(ref), value = TRUE)
    w[sub(paste0("^", src, "\\.w\\."), "", keys)] <- ref[keys]
  }
  expect_setequal(names(w), names(net$state_dict()))
  suppressMessages(load_state_dict(net, w, names(w)))
  net$eval()
  net
}

for (tag in c("clf", "reg")) local({
  tg <- tag
  test_that(paste("a tiny", tg, "network matches causilo's stage by stage"), {
    r <- causilo_ref()
    ref <- r$t
    net <- causilo_ref_net(ref, tg, r$meta)
    dump <- withr::local_tempdir()
    withr::local_envvar(TABFOUND_DUMP_DIR = dump)
    out <- torch::with_no_grad(net(ref[["net.table"]], ref[[paste0(tg, ".targets")]]))
    rd <- function(nm) safetensors::safe_load_file(
      file.path(dump, paste0("causilo_", nm, ".safetensors")), framework = "torch")$t
    expect_close(rd("embed"), ref[[paste0(tg, ".embed")]], info = "embed")
    for (i in 0:1) {
      expect_close(rd(paste0("col", i)), ref[[sprintf("%s.col%d", tg, i)]], info = paste("col", i))
      expect_close(rd(paste0("row", i)), ref[[sprintf("%s.row%d", tg, i)]], info = paste("row", i))
      expect_close(rd(paste0("pred", i)), ref[[sprintf("%s.pred%d", tg, i)]], info = paste("pred", i))
    }
    want <- ref[[paste0(tg, ".out")]]
    got <- if (tg == "clf") out[, , 1:10] else out
    expect_close(got, want, info = "out")
  })
})

test_that("query rows do not see each other", {
  # Every query row reads only context-derived state, so a query row's
  # output is the same predicted alone or with others.
  net <- tiny_causilo("classification")
  x <- torch::torch_randn(c(1L, 14L, 3L))
  y <- torch::torch_randint(0L, 3L, c(1L, 10L))$to(dtype = torch::torch_float32())
  all <- torch::with_no_grad(net(x, y))
  idx <- torch::torch_tensor(c(1:10, 13L), dtype = torch::torch_long())
  one <- torch::with_no_grad(net(x$index_select(2L, idx), y))
  expect_lt(as.numeric((all[1, 3, ] - one[1, 1, ])$abs()$max()), 1e-5)
})

test_that("the backend is registered with the capabilities it has", {
  bk <- get_backend("causilo")
  expect_true(bk$handles_missing)
  expect_false(bk$imputes_internally)
  expect_identical(bk$parity, "causilo 1.5.0 (PyPI)")
  expect_identical(detect_backend(mem_config_causilo("regression"))$name, "causilo")
})
