# Shared transformer layers, checked against the reference
# implementation's own classes with identical weights and inputs.
# Regenerate with:
#
#   .venvs/ref/bin/python inst/parity/layers_reference.py \
#       --out inst/parity/layers/layers.safetensors
#
# No model weights involved, so these run anywhere the package does —
# and a broken RMSNorm or a mis-indexed RoPE fails here in milliseconds
# rather than as a vague drift inside a 1.6 B-parameter forward pass.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

ref_path <- tabfound_file("parity", "layers", "layers.safetensors.gz")
skip_if(!nzchar(ref_path), "layer reference not found")

ref <- read_reference_tensors(ref_path)
meta <- jsonlite::fromJSON(sub("\\.safetensors\\.gz$", ".json", ref_path))

# Copy reference weights into an R module by name. The R module trees are
# built to mirror the reference exactly, so this is a plain name match —
# if it ever needs a fixup, that is itself a finding.
load_ref <- function(module, prefix) {
  params <- c(module$parameters, module$buffers)
  keys <- grep(paste0("^", prefix, "\\."), names(ref), value = TRUE)
  keys <- setdiff(keys, paste0(prefix, ".", c("x", "y", "q", "k", "v", "mask",
                                              "y_self", "y_cross", "y_masked",
                                              "out", "freqs")))
  n <- 0L
  for (k in keys) {
    path <- sub(paste0("^", prefix, "\\."), "", k)
    if (!path %in% names(params)) next
    params[[path]]$set_data(ref[[k]])
    n <- n + 1L
  }
  n
}

expect_close <- function(a, b, tol = 1e-5, label = "") {
  av <- as.numeric(as.array(a)); bv <- as.numeric(as.array(b))
  expect_equal(length(av), length(bv))
  expect_lt(max(abs(av - bv)), tol, label = label)
}


test_that("rms_norm matches the reference RMSNorm", {
  m <- rms_norm(meta$rmsnorm_dim)
  expect_gt(load_ref(m, "rmsnorm"), 0L)
  expect_close(m(ref$rmsnorm.x), ref$rmsnorm.y)
})


test_that("rope rotates using the loaded buffer, not a recomputed one", {
  # TabFM's convention: adjacent-channel pairing, position on axis 2 of
  # (B, T, N, Dh).
  m <- rope(meta$rope_dim, meta$rope_base, interleaved = TRUE, seq_axis = 2L)
  # The reference dump perturbs `freqs` away from the closed form on
  # purpose. Without loading it, this test fails — which is the whole
  # point: TabFM's frequencies were computed in bfloat16 at train time
  # and cannot be regenerated in float32.
  before <- as.numeric(as.array(m$buffers$freqs))
  m$buffers$freqs$set_data(ref$rope.freqs)
  expect_gt(max(abs(before - as.numeric(as.array(ref$rope.freqs)))), 1e-6)
  expect_close(m(ref$rope.x), ref$rope.y)
})


test_that("mha_qk_norm matches, with and without a mask", {
  m <- mha_qk_norm(meta$attn_dim, meta$attn_heads, use_rope = FALSE)
  expect_gt(load_ref(m, "attn"), 0L)

  expect_close(m(ref$attn.q, ref$attn.k, ref$attn.v), ref$attn.y)

  mask <- ref$attn.mask$to(dtype = torch::torch_bool())
  expect_close(m(ref$attn.q, ref$attn.k, ref$attn.v, attn_mask = mask),
               ref$attn.y_masked)
  # Masking must actually change the answer, or the test proves nothing.
  expect_gt(max(abs(as.numeric(as.array(ref$attn.y)) -
                      as.numeric(as.array(ref$attn.y_masked)))), 1e-4)
})


test_that("mab matches for self- and cross-attention", {
  m <- mab(meta$attn_dim, meta$attn_heads, meta$mab_ff)
  expect_gt(load_ref(m, "mab"), 0L)
  expect_close(m(ref$mab.x), ref$mab.y_self)
  expect_close(m(ref$mab.x, ref$attn.k, ref$attn.v), ref$mab.y_cross)
})


test_that("isab matches, and the mask reaches mab1 only", {
  m <- isab(meta$attn_dim, meta$attn_heads, meta$mab_ff, meta$isab_num_inds)
  expect_gt(load_ref(m, "isab"), 0L)
  expect_close(m(ref$isab.x), ref$isab.y)
  mask <- ref$isab.mask$to(dtype = torch::torch_bool())
  expect_close(m(ref$isab.x, attn_mask = mask), ref$isab.y_masked)
})


test_that("mlp_stack matches TabFM's tanh-approximation gelu", {
  m <- mlp_stack(meta$mlp_in, meta$mlp_hidden, meta$mlp_out,
                 activation = "gelu_tanh")
  expect_gt(load_ref(m, "mlp"), 0L)
  expect_close(m(ref$mlp.x), ref$mlp.y)

  # Asking for the exact form here must NOT reproduce the reference --
  # that is the whole point of keeping the two names apart. TabFM is a
  # JAX port and needs the approximation; TabICL is PyTorch-native and
  # needs the exact form.
  m2 <- mlp_stack(meta$mlp_in, meta$mlp_hidden, meta$mlp_out,
                  activation = "gelu")
  load_ref(m2, "mlp")
  expect_gt(max(abs(as.numeric(as.array(m2(ref$mlp.x))) -
                      as.numeric(as.array(ref$mlp.y)))), 1e-5)

  x <- torch::torch_tensor(c(-2, -0.5, 0.5, 2))
  expect_gt(max(abs(as.numeric(apply_activation(x, "gelu_tanh")$cpu()) -
                      as.numeric(apply_activation(x, "gelu")$cpu()))), 1e-5)
})


test_that("one-hot y encoder zeroes out-of-range labels", {
  m <- tabfm_onehot_linear(meta$onehot_classes, meta$onehot_dim)
  expect_gt(load_ref(m, "onehot"), 0L)
  # The fixture includes -100 (the unlabelled sentinel), -1 and 9, all of
  # which must encode to the all-zero vector and so contribute only bias.
  expect_close(m(ref$onehot.y), ref$onehot.out)
})


# --- TabICL layers ---------------------------------------------------------

test_that("ssmax_qa_mlp scales queries by context length", {
  m <- ssmax_qa_mlp(meta$iclssmax_nh, meta$iclssmax_hd)
  expect_gt(load_ref(m, "iclssmax"), 0L)
  expect_close(m(ref$iclssmax.q, 7L), ref$iclssmax.y_n7)
  expect_close(m(ref$iclssmax.q, 64L), ref$iclssmax.y_n64)

  # The whole point of a scalable softmax is that the scale moves with
  # the source length, so the two must differ.
  expect_gt(max(abs(as.numeric(as.array(ref$iclssmax.y_n7)) -
                      as.numeric(as.array(ref$iclssmax.y_n64)))), 1e-4)
})


test_that("rope in non-interleaved mode matches TabICL's", {
  m <- rope(meta$iclrope_dim, meta$iclrope_base, interleaved = FALSE,
            learnable = TRUE)
  m$parameters$freqs$set_data(ref$iclrope.freqs)
  expect_close(m(ref$iclrope.x), ref$iclrope.y)

  # Half-split and adjacent-pair rotation are genuinely different
  # operations; using TabFM's convention here would silently produce a
  # plausible but wrong result.
  m_int <- rope(meta$iclrope_dim, meta$iclrope_base, interleaved = TRUE,
                learnable = TRUE)
  m_int$parameters$freqs$set_data(ref$iclrope.freqs)
  expect_gt(max(abs(as.numeric(as.array(m_int(ref$iclrope.x))) -
                      as.numeric(as.array(ref$iclrope.y)))), 1e-3)
})


for (tag in c("iclmab", "iclmab_nobias")) {
  local({
    prefix <- tag
    bias_free <- grepl("nobias", prefix)
    test_that(paste0("tabicl_mab matches (", prefix, ")"), {
      m <- tabicl_mab(meta$iclmab_dim, meta$iclmab_heads,
                      meta$iclmab_dim * 2L, ssmax = TRUE,
                      bias_free_ln = bias_free, activation = "gelu")
      expect_gt(load_ref(m, prefix), 0L)
      # LayerNorm bias must be present exactly when the config says so.
      expect_identical("norm1.bias" %in% names(m$parameters), !bias_free)

      expect_close(m(ref[[paste0(prefix, ".x")]]),
                   ref[[paste0(prefix, ".y_self")]])
      expect_close(m(ref[[paste0(prefix, ".x")]],
                     ref[[paste0(prefix, ".k")]],
                     ref[[paste0(prefix, ".k")]]),
                   ref[[paste0(prefix, ".y_cross")]])
      # train_size restricts keys/values to a prefix of the query.
      expect_close(m(ref[[paste0(prefix, ".x")]], train_size = 3L),
                   ref[[paste0(prefix, ".y_trainsize")]])
    })
  })
}


test_that("tabicl_isab matches, with and without the train-size slice", {
  m <- tabicl_isab(meta$iclmab_dim, meta$iclmab_heads, meta$iclmab_dim * 2L,
                   meta$iclisab_num_inds, ssmax = TRUE, bias_free_ln = FALSE,
                   activation = "gelu")
  expect_gt(load_ref(m, "iclisab"), 0L)
  expect_close(m(ref$iclisab.x), ref$iclisab.y)
  expect_close(m(ref$iclisab.x, train_size = 4L), ref$iclisab.y_trainsize)

  # Only the first attention carries a scalable softmax; the second sees
  # a fixed-length source (the inducing points) and does not need one.
  expect_true("multihead_attn1.attn.ssmax_layer.base_mlp.0.weight" %in%
                names(m$parameters))
  expect_false(any(grepl("multihead_attn2.attn.ssmax_layer",
                         names(m$parameters))))
})


# --- Mitra layers ----------------------------------------------------------

test_that("the quantile-rank embedding matches, including constant columns", {
  skip_if(!isTRUE(meta$has_mitra), "Mitra layer reference not generated")
  # This transformation carries no parameters, so a mismatch here is
  # pure arithmetic: quantile computation, bucketing convention, or the
  # order of the centre/scale steps.
  out <- mitra_quantile_embedding(ref$mitraq.x_support, ref$mitraq.x_query)
  expect_close(out$support, ref$mitraq.support, tol = 1e-6)
  expect_close(out$query,   ref$mitraq.query,   tol = 1e-6)

  # Column 5 of the fixture is constant, so its variance is zero and the
  # division would produce NaN; both sides must send it to exactly 0.
  sup <- as.array(out$support)
  expect_true(all(sup[, , 5] == 0))
  expect_false(anyNA(sup))
})


test_that("a Mitra layer matches for both support and query", {
  skip_if(!isTRUE(meta$has_mitra), "Mitra layer reference not generated")
  m <- mitra_layer(meta$mitralayer_dim, meta$mitralayer_heads)
  expect_gt(load_ref(m, "mitralayer"), 0L)

  got <- m(ref$mitralayer.support, ref$mitralayer.query)
  expect_close(got$support, ref$mitralayer.support_out)
  expect_close(got$query,   ref$mitralayer.query_out)

  # The query rows attend to the support rows but never to each other,
  # which is what makes predictions independent across test rows.
  # Perturbing one query row must leave the others alone.
  q2 <- as.array(ref$mitralayer.query)
  q2[1, 1, , ] <- q2[1, 1, , ] + 5
  got2 <- m(ref$mitralayer.support, torch::torch_tensor(q2))
  a <- as.array(got$query); b <- as.array(got2$query)
  expect_gt(max(abs(a[1, 1, , ] - b[1, 1, , ])), 1e-3)
  expect_lt(max(abs(a[1, -1, , ] - b[1, -1, , ])), 1e-5)
})


# --- SDPA wrapper and its fallback (C6) -----------------------------------

test_that("the pure-torch SDPA fallback agrees with the fused kernel", {
  skip_if_not_installed("torch")
  set.seed(4)
  q <- torch::torch_randn(c(2, 3, 5, 8))
  k <- torch::torch_randn(c(2, 3, 7, 8))
  v <- torch::torch_randn(c(2, 3, 7, 8))

  fused <- withr::with_options(list(tabfound.sdpa = "torch"), sdpa(q, k, v))
  plain <- withr::with_options(list(tabfound.sdpa = "r"), sdpa(q, k, v))
  expect_equal(as.array(fused), as.array(plain), tolerance = 1e-6)

  # An additive mask, as the feature-group attentions use.
  m <- torch::torch_zeros(c(5, 7))
  m[, 6:7] <- -Inf
  fused_m <- withr::with_options(list(tabfound.sdpa = "torch"),
                                 sdpa(q, k, v, attn_mask = m))
  plain_m <- withr::with_options(list(tabfound.sdpa = "r"),
                                 sdpa(q, k, v, attn_mask = m))
  expect_equal(as.array(fused_m), as.array(plain_m), tolerance = 1e-6)
  # The masked positions really were excluded.
  expect_false(isTRUE(all.equal(as.array(fused), as.array(fused_m))))

  # A boolean mask keeps the TRUE positions.
  b <- torch::torch_ones(c(5, 7), dtype = torch::torch_bool())
  b[, 1:2] <- FALSE
  expect_equal(
    as.array(withr::with_options(list(tabfound.sdpa = "torch"),
                                 sdpa(q, k, v, attn_mask = b))),
    as.array(withr::with_options(list(tabfound.sdpa = "r"),
                                 sdpa(q, k, v, attn_mask = b))),
    tolerance = 1e-6
  )

  # An explicit scale, as TabICL's scalable softmax passes.
  expect_equal(
    as.array(withr::with_options(list(tabfound.sdpa = "torch"),
                                 sdpa(q, k, v, scale = 1))),
    as.array(withr::with_options(list(tabfound.sdpa = "r"),
                                 sdpa(q, k, v, scale = 1))),
    tolerance = 1e-6
  )
})


test_that("a whole attention block gives the same answer either way", {
  skip_if_not_installed("torch")
  torch::torch_manual_seed(11)
  att <- mha_fused_qkv(embedding_dim = 12L, n_heads = 3L)
  x <- torch::torch_randn(c(2, 6, 12))
  a <- withr::with_options(list(tabfound.sdpa = "torch"),
                           torch::with_no_grad(att(x)))
  b <- withr::with_options(list(tabfound.sdpa = "r"),
                           torch::with_no_grad(att(x)))
  # The whole point of reaching for the fused kernel is float32 rounding,
  # so this is an agreement test, not an identity one.
  expect_equal(as.array(a), as.array(b), tolerance = 1e-5)
})
