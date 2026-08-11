# Architecture description and diagrams.
#
# The networks here are built from configs and left at their random
# initialisation: nothing in this file runs a forward pass, and none of
# it needs a checkpoint. Widths are cut right down so a full six-backend
# sweep stays cheap -- the stage list does not depend on how wide the
# model is.

skip_if_not_installed("torch")

# A model object of the shape `tabfound_architecture()` expects, without
# the 600 MB of weights a real one carries.
arch_test_model <- function(config, backend, task = "classification") {
  net <- suppressMessages(get_backend(backend)$build(config, task))
  structure(
    list(spec = NULL, state = NULL, model = net, config = config,
         device = "cpu", backend = backend, task = task,
         model_ref = list(model = "test", backend = backend, device = "cpu",
                          args = list())),
    class = c(if (task == "classification") "tabfound_classifier"
              else "tabfound_regressor", "tabfound_model")
  )
}

arch_test_configs <- function() {
  list(
    tabpfn = list(
      config = list(arch = "per_feature_transformer", head = "classifier",
                    n_layers = 2L, embedding_dim = 12L, n_heads = 3L,
                    mlp_hidden_dim = 24L, features_per_group = 3L,
                    n_out_classes = 10L),
      task = "classification"),
    tabpfn26 = list(
      config = list(arch = "tabpfn_v2_6", head = "regressor", n_layers = 2L,
                    embedding_dim = 12L, n_heads = 3L, mlp_hidden_dim = 24L,
                    features_per_group = 3L, num_thinking_rows = 4L,
                    encoder_type = "linear", n_bar_bins = 16L),
      task = "regression"),
    tabpfn3 = list(
      config = list(arch = "tabpfn_v3", head = "classifier", embed_dim = 8L,
                    feat_agg_num_cls_tokens = 2L, ff_factor = 2L,
                    feature_group_size = 3L, use_nan_indicators = TRUE,
                    dist_embed_num_blocks = 1L, dist_embed_num_heads = 2L,
                    dist_embed_num_inducing_points = 4L,
                    feat_agg_num_blocks = 1L, feat_agg_num_heads = 2L,
                    nlayers = 2L, icl_num_heads = 2L, icl_num_kv_heads_test = 1L,
                    decoder_head_dim = 4L, decoder_num_heads = 2L,
                    decoder_use_softmax_scaling = TRUE,
                    softmax_scaling_mlp_hidden_dim = 4L, max_num_classes = 6L,
                    n_out = 6L, n_bar_bins = 16L),
      task = "classification"),
    tabicl = list(
      config = list(arch = "tabicl", head = "classifier", max_classes = 6L,
                    num_quantiles = 9L, embed_dim = 8L, col_num_blocks = 1L,
                    col_nhead = 2L, col_num_inds = 4L, col_feature_group = "same",
                    col_feature_group_size = 3L, col_target_aware = TRUE,
                    col_ssmax = "qassmax-mlp-elementwise", row_num_blocks = 2L,
                    row_nhead = 2L, row_num_cls = 2L, row_rope_base = 100000,
                    row_rope_interleaved = FALSE, icl_num_blocks = 2L,
                    icl_nhead = 2L, icl_ssmax = "qassmax-mlp-elementwise",
                    ff_factor = 2L, activation = "gelu", bias_free_ln = FALSE),
      task = "classification"),
    tabfm = list(
      config = list(arch = "tabfm", embed_dim = 8L, ff_factor = 2L,
                    row_num_cls = 2L, col_num_blocks = 1L, col_nhead = 2L,
                    col_num_inds = 4L, row_num_blocks = 1L, row_nhead = 2L,
                    icl_num_blocks = 2L, icl_nhead = 2L, max_classes = 6L,
                    feature_group_size = 3L, num_freq = 4L,
                    is_classifier = TRUE, decoder_hidden = NULL),
      task = "classification"),
    mitra = list(
      config = list(dim = 8L, dim_output = 6L, n_heads = 2L, n_layers = 2L,
                    task = "CLASSIFICATION"),
      task = "classification")
  )
}


test_that("every registered backend describes its architecture", {
  for (nm in list_backends()$name) {
    expect_false(is.null(get_backend(nm)$describe),
                 info = paste(nm, "has no describe hook"))
  }
})


# The point of the audit: a stage list is a hand-written summary of a
# network, and this is what stops it going stale. A backend that grows a
# submodule fails here until its description grows one too.
test_that("each backend's stages claim every parameter exactly once", {
  for (nm in names(arch_test_configs())) {
    cs <- arch_test_configs()[[nm]]
    arch <- tabfound_architecture(arch_test_model(cs$config, nm, cs$task))
    audit <- arch_audit_coverage(arch)
    expect_identical(audit$unclaimed, character(),
                     info = paste(nm, "unclaimed:",
                                  paste(audit$unclaimed, collapse = ", ")))
    expect_identical(audit$duplicated, character(),
                     info = paste(nm, "duplicated:",
                                  paste(audit$duplicated, collapse = ", ")))
  }
})


test_that("stage parameter counts add up to the network's own total", {
  for (nm in names(arch_test_configs())) {
    cs <- arch_test_configs()[[nm]]
    arch <- tabfound_architecture(arch_test_model(cs$config, nm, cs$task))
    per_stage <- vapply(arch$stages, `[[`, numeric(1), "params")
    expect_equal(sum(per_stage, na.rm = TRUE), arch$n_params,
                 info = nm)
  }
})


test_that("every sublayer prefix names something that exists", {
  for (nm in names(arch_test_configs())) {
    cs <- arch_test_configs()[[nm]]
    arch <- tabfound_architecture(arch_test_model(cs$config, nm, cs$task))
    for (s in arch$stages) {
      for (ch in s$children) {
        if (!length(ch$prefix)) next
        expect_gt(ch$params, 0)
      }
    }
  }
})


test_that("the description reports the head it was built for", {
  cs <- arch_test_configs()$tabpfn26
  arch <- tabfound_architecture(arch_test_model(cs$config, "tabpfn26", "regression"))
  expect_identical(arch$task, "regression")
  expect_match(arch$title, "regressor")
  expect_true(any(grepl("Bar distribution",
                        vapply(arch$stages, `[[`, character(1), "label"))))
})


test_that("as.data.frame gives one row per stage, and more with detail", {
  cs <- arch_test_configs()$tabicl
  arch <- tabfound_architecture(arch_test_model(cs$config, "tabicl", cs$task))
  d1 <- as.data.frame(arch)
  d2 <- as.data.frame(arch, detail = "full")
  expect_identical(nrow(d1), length(arch$stages))
  expect_gt(nrow(d2), nrow(d1))
  expect_true(all(d1$kind %in% names(ARCH_KINDS)))
  expect_true(all(is.na(d1$axis) | d1$axis %in% ARCH_AXES))
})


test_that("a model with no config, and a non-model, are refused", {
  bare <- structure(list(config = NULL, model = NULL, backend = "tabpfn",
                         task = "classification"),
                    class = c("tabfound_classifier", "tabfound_model"))
  expect_error(tabfound_architecture(bare), "no configuration")
  expect_error(tabfound_architecture(iris), "loaded tabular foundation model")
})


test_that("a backend without a describe hook says so rather than drawing", {
  on.exit(rm("nodesc", envir = .tabfound_backends), add = TRUE)
  register_backend(name = "nodesc", build = function(config, task) NULL,
                   description = "no description")
  m <- structure(list(config = list(a = 1), model = NULL, backend = "nodesc",
                      task = "classification"),
                 class = c("tabfound_classifier", "tabfound_model"))
  expect_error(tabfound_architecture(m), "does not describe its architecture")
})


# ---------------------------------------------------------------------------
# Layout and rendering
# ---------------------------------------------------------------------------

test_that("the scene fits inside its own canvas", {
  cs <- arch_test_configs()$tabpfn3
  arch <- tabfound_architecture(arch_test_model(cs$config, "tabpfn3", cs$task))
  for (detail in c("overview", "full")) {
    for (theme in c("light", "dark")) {
      sc <- arch_scene(arch, detail = detail, theme = theme)
      xs <- unlist(lapply(sc$items, function(e)
        switch(e$type, rect = c(e$x, e$x + e$w), line = c(e$x1, e$x2),
               poly = e$x, text = e$x)))
      ys <- unlist(lapply(sc$items, function(e)
        switch(e$type, rect = c(e$y, e$y + e$h), line = c(e$y1, e$y2),
               poly = e$y, text = e$y)))
      expect_gte(min(xs), 0)
      expect_lte(max(xs), sc$width)
      expect_gte(min(ys), 0)
      expect_lte(max(ys), sc$height)
    }
  }
})


test_that("blocks are laid out top to bottom without overlapping", {
  cs <- arch_test_configs()$tabicl
  arch <- tabfound_architecture(arch_test_model(cs$config, "tabicl", cs$task))
  sc <- arch_scene(arch, detail = "overview")
  g <- arch_geom()
  # One card per stage: full block width, at the block column's left
  # edge. The offset copies behind a repeated stack are the same width
  # but start further right, so they drop out here.
  x_box <- g$pad + g$gutter_l + g$gap
  cards <- Filter(function(e)
    e$type == "rect" && isTRUE(all.equal(e$w, g$box_w)) &&
      isTRUE(all.equal(e$x, x_box)), sc$items)
  expect_identical(length(cards), length(arch$stages))
  tops <- vapply(cards, `[[`, numeric(1), "y")
  bots <- tops + vapply(cards, `[[`, numeric(1), "h")
  expect_true(all(diff(tops) > 0))
  expect_true(all(bots[-length(bots)] <= tops[-1]))
})


test_that("SVG output is well formed and carries the diagram's text", {
  cs <- arch_test_configs()$mitra
  arch <- tabfound_architecture(arch_test_model(cs$config, "mitra", cs$task))
  svg <- arch_render_svg(arch_scene(arch))
  expect_match(svg[2], "^<svg xmlns=")
  expect_identical(tail(svg, 1), "</svg>")
  expect_true(any(grepl("Quantile-rank embedding", svg, fixed = TRUE)))
  # Every element opened is closed on its own line.
  body <- svg[grepl("^  <", svg)]
  expect_true(all(grepl("/>$|</(text|tspan)>$", body)))
})


test_that("SVG escapes the characters that would break the document", {
  esc <- arch_svg_item(sc_text(0, 0, "a & b <c> \"d\"", col = "#000000"))
  expect_match(esc, "a &amp; b &lt;c&gt;")
  expect_false(grepl("<c>", esc, fixed = TRUE))
})


test_that("the two themes differ in surface and share the layout", {
  cs <- arch_test_configs()$mitra
  arch <- tabfound_architecture(arch_test_model(cs$config, "mitra", cs$task))
  a <- arch_scene(arch, theme = "light")
  b <- arch_scene(arch, theme = "dark")
  expect_false(identical(a$bg, b$bg))
  expect_identical(a$height, b$height)
  expect_identical(length(a$items), length(b$items))
})


test_that("plot_architecture writes each format", {
  skip_on_cran()
  cs <- arch_test_configs()$mitra
  m <- arch_test_model(cs$config, "mitra", cs$task)
  dir <- withr::local_tempdir()

  for (ext in c("svg", "png", "pdf")) {
    f <- file.path(dir, paste0("arch.", ext))
    expect_identical(suppressMessages(plot_architecture(m, f)), f)
    expect_true(file.exists(f))
    expect_gt(file.size(f), 1000)
  }
  # The format follows the extension, and an unknown one is an error
  # rather than a silently mislabelled file.
  expect_error(plot_architecture(m, file.path(dir, "arch.jpeg")),
               "Cannot tell the output format")
  expect_error(plot_architecture(m, format = "png"), "needs a .*file")
})


test_that("an already-built description can be drawn without rebuilding", {
  skip_on_cran()
  cs <- arch_test_configs()$mitra
  arch <- tabfound_architecture(arch_test_model(cs$config, "mitra", cs$task))
  f <- file.path(withr::local_tempdir(), "arch.svg")
  suppressMessages(plot_architecture(arch, f))
  expect_true(any(grepl("Mitra", readLines(f, warn = FALSE))))
})
