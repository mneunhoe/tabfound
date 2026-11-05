# Diagram styles and the helpers they share.
#
# Each style is a scene builder registered in `arch_styles()`. The scene
# tests run over every registered style, so a new one is covered by
# being registered; what is specific to a style is tested on its own.

skip_if_not_installed("torch")

arch_test_arch <- function(name) {
  cs <- arch_test_configs()[[name]]
  tabfound_architecture(arch_test_model(cs$config, name, cs$task))
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

test_that("cards is registered and is the default", {
  expect_true("cards" %in% names(arch_styles()))
  expect_identical(formals(arch_scene)$style, "cards")
  arch <- arch_test_arch("mitra")
  expect_identical(arch_scene(arch), arch_scene(arch, style = "cards"))
})


test_that("an unknown style is an error that names the choices", {
  arch <- arch_test_arch("mitra")
  expect_error(arch_scene(arch, style = "nope"), "must be one of.*cards")
  expect_error(plot_architecture(arch, style = c("cards", "cards")),
               "must be one of")
})


test_that("the default style comes from the tabfound.arch_style option", {
  skip_on_cran()
  arch <- arch_test_arch("mitra")
  withr::local_options(tabfound.arch_style = "nope")
  expect_error(plot_architecture(arch, tempfile(fileext = ".svg")),
               "must be one of")
})


test_that("every style's scene fits its canvas in both themes", {
  for (style in names(arch_styles())) {
    for (nm in names(arch_test_configs())) {
      arch <- arch_test_arch(nm)
      for (detail in c("overview", "full")) {
        a <- arch_scene(arch, style = style, detail = detail, theme = "light")
        b <- arch_scene(arch, style = style, detail = detail, theme = "dark")
        info <- paste(style, nm, detail)
        xs <- unlist(lapply(a$items, function(e)
          switch(e$type, rect = c(e$x, e$x + e$w), line = c(e$x1, e$x2),
                 poly = e$x, text = e$x)))
        ys <- unlist(lapply(a$items, function(e)
          switch(e$type, rect = c(e$y, e$y + e$h), line = c(e$y1, e$y2),
                 poly = e$y, text = e$y)))
        expect_true(min(xs) >= 0 && max(xs) <= a$width, info = info)
        expect_true(min(ys) >= 0 && max(ys) <= a$height, info = info)
        expect_identical(a$height, b$height, info = info)
        expect_identical(length(a$items), length(b$items), info = info)
      }
    }
  }
})


# ---------------------------------------------------------------------------
# cards
# ---------------------------------------------------------------------------

# Refactoring a style must not move a single byte of its output. A
# deliberate change to the cards look regenerates these fixtures (the
# last one: adding the key).
test_that("cards output is byte-identical to its fixtures", {
  skip_on_cran()
  arch <- arch_test_arch("tabicl")
  dir <- withr::local_tempdir()
  cases <- list(
    list(file = "arch-cards-tabicl.svg", detail = "overview", theme = "light"),
    list(file = "arch-cards-tabicl-full-dark.svg", detail = "full",
         theme = "dark")
  )
  for (cs in cases) {
    out <- file.path(dir, cs$file)
    suppressMessages(plot_architecture(arch, out, style = "cards",
                                       detail = cs$detail, theme = cs$theme))
    expect_identical(unname(tools::md5sum(out)),
                     unname(tools::md5sum(test_path("fixtures", cs$file))),
                     info = cs$file)
  }
})


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

test_that("text items carry a font family, and mono is its older spelling", {
  expect_identical(sc_text(0, 0, "a")$family, "sans")
  expect_identical(sc_text(0, 0, "a", mono = TRUE)$family, "mono")
  expect_identical(sc_text(0, 0, "a", family = "serif")$family, "serif")
  expect_error(sc_text(0, 0, "a", family = "fantasy"))

  svg <- arch_svg_item(sc_text(0, 0, "a", family = "serif"))
  expect_match(svg, "Palatino", fixed = TRUE)
  expect_match(arch_svg_item(sc_text(0, 0, "a", mono = TRUE)), "Menlo",
               fixed = TRUE)
})


test_that("serif text draws on a graphics device", {
  skip_on_cran()
  scene <- list(width = 200, height = 50, bg = "#ffffff", items = list(
    sc_text(10, 15, "Serif", family = "serif"),
    sc_text(10, 35, "Serif italic", family = "serif", italic = TRUE)
  ))
  f <- tempfile(fileext = ".pdf")
  on.exit(unlink(f))
  expect_no_warning(suppressMessages(arch_write_scene(scene, f)))
  expect_gt(file.size(f), 500)
})


test_that("text widths depend on the family", {
  w <- function(...) arch_text_width("Column-wise embedding", 10, ...)
  expect_equal(w(), w(family = "sans"))
  expect_equal(w(mono = TRUE), w(family = "mono"))
  expect_false(isTRUE(all.equal(w(), w(family = "serif"))))
  expect_gt(w(family = "serif", bold = TRUE), w(family = "serif"))
})


test_that("wrapping keeps every word and each line fits", {
  text <- paste("Column-wise embedding with induced self-attention over",
                "the rows of each column, sharing weights across columns")
  lines <- arch_wrap_text(text, 120, 10.5)
  expect_gt(length(lines), 1L)
  expect_identical(paste(lines, collapse = " "), text)
  expect_true(all(arch_text_width(lines, 10.5) <= 120))

  serif <- arch_wrap_text(text, 120, 10.5, family = "serif", italic = TRUE)
  expect_identical(paste(serif, collapse = " "), text)
  expect_true(all(arch_text_width(serif, 10.5, family = "serif",
                                  italic = TRUE) <= 120))
})


test_that("wrapping with a line limit marks what it dropped", {
  text <- paste(rep("word", 40), collapse = " ")
  lines <- arch_wrap_text(text, 100, 10, max_lines = 2)
  expect_length(lines, 2L)
  expect_match(lines[2], "\\.\\.\\.$")
  expect_true(all(arch_text_width(lines, 10) <= 100))

  # A word wider than the line is cut rather than overflowing it.
  long <- arch_wrap_text("Supercalifragilisticexpialidocious", 60, 10)
  expect_length(long, 1L)
  expect_true(arch_text_width(long, 10) <= 60)

  expect_identical(arch_wrap_text("", 100, 10), character())
  expect_identical(arch_wrap_text("fits", 100, 10, max_lines = 1), "fits")
})


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

test_that("arch_viridis matches viridisLite to within a small colour difference", {
  skip_if_not_installed("viridisLite")
  lab <- function(hex) {
    grDevices::convertColor(t(grDevices::col2rgb(hex)) / 255,
                            from = "sRGB", to = "Lab")
  }
  for (range in list(c(0, 1), c(0, 0.55), c(0.35, 1))) {
    ours <- arch_viridis(41, range[1], range[2])
    ref  <- substr(viridisLite::viridis(41, begin = range[1], end = range[2]),
                   1, 7)
    delta_e <- sqrt(rowSums((lab(ours) - lab(ref))^2))
    expect_lt(max(delta_e), 2)
  }
  expect_identical(arch_viridis(0), character())
  expect_length(arch_viridis(5), 5L)
  expect_identical(arch_viridis(1, begin = 0.35), arch_viridis(3, 0.35, 1)[1])
})


# ---------------------------------------------------------------------------
# The description: collapse flag and fact vocabulary
# ---------------------------------------------------------------------------

test_that("at most one stage collapses the table, and before the ICL stack", {
  expected <- list(tabicl = "row_tf", tabfm = "row2", tabpfn3 = "agg",
                   tabpfn35 = "agg", tabpfn = character(),
                   tabpfn26 = character(), mitra = character())
  for (nm in names(arch_test_variants())) {
    v <- arch_test_variants()[[nm]]
    arch <- arch_test_variant_arch(v)
    ids <- vapply(arch$stages, `[[`, character(1), "id")
    hit <- ids[vapply(arch$stages, function(s) isTRUE(s$collapses), logical(1))]
    want <- if (identical(v$backend, "kumo")) {
      if (identical(nm, "kumo small")) "row" else "proj"
    } else expected[[v$backend]]
    expect_identical(hit, want, info = nm)
    if (length(hit) && "icl" %in% ids) {
      expect_lt(match(hit, ids), match("icl", ids))
    }
  }
  df <- as.data.frame(arch_test_arch("tabicl"))
  expect_identical(df$stage[df$collapses], "row_tf")
})


test_that("every backend names its facts from the shared vocabulary", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    expect_true(length(arch$facts) > 0L, info = nm)
    expect_identical(setdiff(names(arch$facts), ARCH_FACT_NAMES), character(),
                     info = nm)
    expect_false(anyDuplicated(names(arch$facts)) > 0L, info = nm)
  }
  expect_identical(arch_ff_fact(768 * 4, 768), "4 x width (3072)")
  expect_identical(arch_ff_fact(384, 192), "2 x width (384)")
  expect_identical(arch_ff_fact(1000, 768), "1.3 x width (1000)")
})


# ---------------------------------------------------------------------------
# The key: every diagram explains itself
# ---------------------------------------------------------------------------

test_that("every symbol drawn in a shape has a definition", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    expect_identical(setdiff(arch_shape_symbols(arch), names(arch$symbols)),
                     character(), info = nm)
  }
  # The plain words a shape may spell out are not symbols.
  expect_identical(arch_tokens("(n rows, p features)"),
                   c("n", "rows", "p", "features"))
  expect_identical(arch_tokens("(B, 64 + n, F + 1, E) (n_test, 10)"),
                   c("B", "n", "F", "E", "n_test"))
})


test_that("every abbreviation drawn is in the glossary or is a proper name", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    expect_identical(arch_unexplained_terms(arch, unname(ARCH_KINDS)),
                     character(), info = nm)
  }
  fake <- list(title = "A MoE model", facts = character(), stages = list())
  expect_identical(arch_unexplained_terms(fake), "MoE")
  fake$title <- "MLPs over RoPE, then TabPFN"
  expect_identical(arch_unexplained_terms(fake), character())
  expect_named(arch_used_terms(fake), c("RoPE", "MLP"))
})


test_that("a backend's glossary entries win over the shared ones", {
  gl <- arch_merge_glossary(c(ICL = "reworded", GQA = "grouped-query attention"))
  expect_identical(gl[["ICL"]], "reworded")
  expect_identical(gl[["GQA"]], "grouped-query attention")
  expect_identical(sum(names(gl) == "ICL"), 1L)
  expect_identical(arch_merge_glossary(), ARCH_GLOSSARY)
})


# What a scene draws above its key must be in it -- every glossary term
# in its text, every symbol in its monospaced shapes -- and the key must
# draw each entry whole.
expect_key_whole <- function(sc, gl, info) {
  items <- Filter(function(e) e$type == "text", sc$items)
  txt <- vapply(items, `[[`, character(1), "text")
  at <- match("Key", txt)
  expect_false(is.na(at), info = info)
  body <- txt[seq_len(at - 1L)]
  key <- txt[seq(at, length(txt))]

  terms <- names(gl)[vapply(names(gl), function(t)
    any(grepl(arch_term_pattern(t), body, perl = TRUE)), logical(1))]
  listed <- unlist(lapply(sc$key, function(g) names(g$items)))
  # A diff lists a symbol whose meanings differ as "F (A)" and "F (B)".
  listed <- c(listed, sub(" \\([AB]\\)$", "", listed))
  expect_true(all(terms %in% listed),
              info = paste(info, setdiff(terms, listed)))
  mono <- vapply(items[seq_len(at - 1L)], `[[`, character(1),
                 "family") == "mono"
  shape_syms <- setdiff(arch_tokens(body[mono]), ARCH_SHAPE_WORDS)
  expect_true(all(shape_syms %in% listed),
              info = paste(info, setdiff(shape_syms, listed)))

  # Never truncated; wrapped lines are consecutive items, so joining
  # them restores every meaning.
  expect_false(any(grepl("...", key, fixed = TRUE)), info = info)
  joined <- paste(key, collapse = " ")
  for (g in sc$key) {
    for (t in names(g$items)) {
      expect_true(t %in% key, info = paste(info, t))
      expect_true(grepl(g$items[[t]], joined, fixed = TRUE),
                  info = paste(info, t))
    }
    for (nt in g$notes) {
      expect_true(grepl(nt$text, joined, fixed = TRUE),
                  info = paste(info, nt$text))
    }
  }
  svg <- arch_render_svg(sc)
  expect_identical(sum(startsWith(svg, "  <title>")), 1L, info = info)
  expect_identical(sum(startsWith(svg, "  <desc>")), 1L, info = info)
}

test_that("the key is drawn whole on every style, theme and setting", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    for (style in names(arch_styles())) {
      for (theme in c("light", "dark")) {
        for (detail in c("overview", "full")) {
          for (facts in c(TRUE, FALSE)) {
            sc <- arch_scene(arch, style = style, detail = detail,
                             theme = theme, show_facts = facts)
            expect_key_whole(sc, arch$glossary,
                             paste(nm, style, theme, detail, facts))
          }
        }
      }
    }
  }
})


test_that("the key wraps long entries instead of cutting them", {
  th <- list(ink = "#000000", ink2 = "#333333", ink3 = "#666666")
  long <- paste(rep("a fairly long definition", 12), collapse = " ")
  groups <- list(
    list(head = "SYMBOLS IN SHAPES", items = c(B = long, n = "rows"), mono = TRUE),
    list(head = "HOW TO READ", notes = list(arch_note(long, "#ff0000")))
  )
  k <- arch_key_items(groups, 0, 0, 400, th)
  txt <- vapply(Filter(function(e) e$type == "text", k$items), `[[`,
                character(1), "text")
  expect_false(any(grepl("...", txt, fixed = TRUE)))
  joined <- paste(txt, collapse = " ")
  expect_identical(lengths(regmatches(joined, gregexpr(long, joined, fixed = TRUE))),
                   2L)
  expect_gt(length(txt), 8L)
  expect_true(any(vapply(k$items, function(e) e$type == "rect", logical(1))))
  # Narrow keys use one column, wide ones two: the second entry sits
  # under the first or beside it.
  narrow <- arch_key_items(groups[1], 0, 0, 400, th)
  wide <- arch_key_items(groups[1], 0, 0, 800, th)
  x_of <- function(k, s) Filter(function(e) e$type == "text" &&
                                  identical(e$text, s), k$items)[[1]]$x
  expect_identical(x_of(narrow, "n"), x_of(narrow, "B"))
  expect_gt(x_of(wide, "n"), x_of(wide, "B"))

  desc <- arch_key_text(groups)
  expect_match(desc, "^Symbols in shapes: B: a fairly")
  expect_match(desc, "How to read: ")
})


# ---------------------------------------------------------------------------
# tufte
# ---------------------------------------------------------------------------

tufte_dots <- function(sc) {
  Filter(function(e) e$type == "rect" && isTRUE(all.equal(e$w, 7)) &&
           isTRUE(all.equal(e$r, 3.5)), sc$items)
}

test_that("tufte places one dot per stage, further right for more parameters", {
  for (nm in c("tabicl", "mitra", "tabpfn35")) {
    arch <- arch_test_arch(nm)
    sc <- arch_scene(arch, style = "tufte")
    params <- vapply(arch$stages, function(s)
      if (is.na(s$params)) 0 else s$params, numeric(1))
    params <- params[params > 0]
    dots <- tufte_dots(sc)
    expect_length(dots, length(params))
    x <- vapply(dots, `[[`, numeric(1), "x")
    # Position is monotone in log10(params): equal counts share an x.
    o <- order(params)
    expect_true(all(diff(x[o]) >= -1e-9), info = nm)
    expect_equal(diff(x[o]) > 1e-9, diff(log10(params[o])) > 1e-9,
                 info = nm)
  }
})


test_that("the tufte axis is a range frame over the data's decades", {
  expect_identical(arch_axis_decades(c(512, 25.7e6)), 3:7)
  expect_identical(arch_axis_decades(c(1000, 1e6)), 3:6)
  expect_identical(arch_axis_decades(c(300, 900)), integer())
  expect_identical(arch_decade_label(c(0, 2, 3, 5, 6, 9)),
                   c("1", "100", "1k", "100k", "1M", "1B"))

  px <- function(n) log10(n) * 100
  th <- arch_tufte_theme()
  items <- arch_log_axis_items(c(512, 25.7e6), px, 0, th)
  frame <- items[[1]]
  expect_equal(c(frame$x1, frame$x2), px(c(512, 25.7e6)))
  ticks <- Filter(function(e) e$type == "line" && e$y2 > e$y1, items)
  expect_equal(vapply(ticks, `[[`, numeric(1), "x1"), px(10^(3:7)))
  # Decades 100 px apart all get a label; one value range with no
  # decade inside it labels its ends instead.
  labels <- vapply(Filter(function(e) e$type == "text", items), `[[`,
                   character(1), "text")
  expect_identical(labels, c("1k", "10k", "100k", "1M", "10M"))
  ends <- arch_log_axis_items(c(300, 900), px, 0, th)
  expect_identical(vapply(Filter(function(e) e$type == "text", ends), `[[`,
                          character(1), "text"), c("300", "900"))
  # Crowded decades are labelled every other one.
  tight <- arch_log_axis_items(c(1e3, 1e7), function(n) log10(n) * 20, 0, th)
  expect_identical(vapply(Filter(function(e) e$type == "text", tight), `[[`,
                          character(1), "text"), c("1k", "100k", "10M"))
})


test_that("tufte stage labels stay clear of their tags", {
  g <- arch_tufte_geom()
  x_text <- g$pad + g$phase_w + g$gap
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    for (detail in c("overview", "full")) {
      sc <- arch_scene(arch, style = "tufte", detail = detail)
      txt <- Filter(function(e) e$type == "text", sc$items)
      tags <- Filter(function(e) identical(e$anchor, "end") &&
                       isTRUE(all.equal(e$x, x_text + g$text_w)), txt)
      for (tg in tags) {
        left <- tg$x - arch_text_width(tg$text, tg$size, family = "serif",
                                       italic = TRUE)
        same_row <- Filter(function(e) identical(e$anchor, "start") &&
                             e$x >= x_text && e$x < tg$x &&
                             abs(e$y - tg$y) < 1, txt)
        for (lab in same_row) {
          right <- lab$x + arch_text_width(lab$text, lab$size,
                                           family = lab$family)
          expect_lte(right, left, label = paste(nm, detail, lab$text))
        }
      }
    }
  }
})


test_that("tufte drops the dot plot without parameters, and sets serif type", {
  arch <- arch_test_arch("tabicl")
  sc <- arch_scene(arch, style = "tufte", show_params = FALSE)
  expect_length(tufte_dots(sc), 0L)
  txt <- vapply(Filter(function(e) e$type == "text", sc$items), `[[`,
                character(1), "text")
  expect_false(any(grepl("total$|log scale", txt)))

  sc <- arch_scene(arch, style = "tufte", detail = "full")
  fam <- vapply(Filter(function(e) e$type == "text", sc$items), `[[`,
                character(1), "family")
  expect_setequal(unique(fam), c("serif", "mono"))
  # Shapes are the only monospaced text outside the key's symbols.
  shapes <- unlist(lapply(arch$stages, `[[`, "shape"))
  mono <- vapply(Filter(function(e) e$type == "text" && e$family == "mono",
                        sc$items), `[[`, character(1), "text")
  expect_true(all(mono %in% c(shapes, names(arch$symbols))))
  expect_true(any(grepl("Palatino", arch_render_svg(sc), fixed = TRUE)))
})


test_that("tufte colours each phase in stage order from the viridis slice", {
  arch <- arch_test_arch("tabicl")
  groups <- unique(unlist(lapply(arch$stages, `[[`, "group")))
  for (theme in c("light", "dark")) {
    th <- arch_tufte_theme(theme)
    sc <- arch_scene(arch, style = "tufte", theme = theme)
    rules <- Filter(function(e) e$type == "line" && isTRUE(all.equal(e$lwd, 2.5)),
                    sc$items)
    expect_identical(vapply(rules, `[[`, character(1), "col"),
                     arch_viridis(length(groups), th$viridis[1], th$viridis[2]))
  }
})


# ---------------------------------------------------------------------------
# icicle
# ---------------------------------------------------------------------------

test_that("sublayers counted over all blocks never claim more than their stage", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    for (s in arch$stages) {
      if (!length(s$children)) next
      all_blocks <- sum(vapply(s$children, function(ch)
        if (is.na(ch$params_all)) 0 else ch$params_all, numeric(1)))
      expect_lte(all_blocks, s$params, label = paste(nm, s$id))
    }
  }
  # A shared table is counted once, a per-block one in every block.
  idx <- c("s.blocks.0.attn.w" = 2, "s.blocks.1.attn.w" = 2,
           "s.blocks.11.attn.w" = 2, "s.rope" = 5, "s.blocks.0.attnx" = 7)
  expect_identical(arch_count_params_all_blocks(idx, "s.blocks.0.attn"), 6)
  expect_identical(arch_count_params_all_blocks(idx, "s.rope"), 5)
})


icicle_bands <- function(sc) {
  g <- arch_icicle_geom()
  rects <- Filter(function(e) e$type == "rect" &&
                    isTRUE(all.equal(e$h, g$band_h)), sc$items)
  ys <- sort(unique(vapply(rects, `[[`, numeric(1), "y")))
  lapply(stats::setNames(ys, c("phase", "stage", "sublayer")[seq_along(ys)]),
         function(y) Filter(function(e) isTRUE(all.equal(e$y, y)), rects))
}

test_that("icicle bands share one linear scale and nest exactly", {
  g <- arch_icicle_geom()
  x0 <- g$pad + g$gutter
  wid <- g$width - g$pad - x0
  # A cell's slot runs from its x to the next cell's; the drawn rect is
  # a little narrower, to leave a gap, and never under 0.8 px.
  starts <- function(band) sort(vapply(band, `[[`, numeric(1), "x"))
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    bands <- icicle_bands(arch_scene(arch, style = "icicle"))
    params <- vapply(arch$stages, `[[`, numeric(1), "params")
    params <- params[!is.na(params) & params > 0]

    st <- starts(bands$stage)
    slots <- diff(c(st, x0 + wid))
    expect_equal(st[1], x0, info = nm)
    expect_equal(slots / wid, params / sum(params), tolerance = 1e-9,
                 info = nm)
    # Every stage boundary is also a sublayer boundary, and the sublayer
    # band spans the same width: each stage's sublayers fill it exactly.
    sub <- starts(bands$sublayer)
    expect_equal(sub[1], x0, info = nm)
    expect_true(all(vapply(st, function(x) any(abs(sub - x) < 1e-6),
                           logical(1))), info = nm)
    expect_lte(max(sub), x0 + wid)
    # Phases start on stage boundaries.
    ph <- starts(bands$phase)
    expect_true(all(vapply(ph, function(x) any(abs(st - x) < 1e-6),
                           logical(1))), info = nm)
  }
})


test_that("every icicle block with parameters is labelled or listed", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    sc <- arch_scene(arch, style = "icicle")
    txt <- vapply(Filter(function(e) e$type == "text", sc$items), `[[`,
                  character(1), "text")
    # The "too narrow" list wraps at spaces; joining restores it.
    joined <- paste(txt, collapse = " ")
    for (s in arch$stages) {
      if (is.na(s$params) || s$params <= 0) next
      expect_true(s$label %in% txt ||
                    grepl(paste0(s$label, " ("), joined, fixed = TRUE),
                  info = paste(nm, s$label))
    }
    phases <- unique(unlist(lapply(arch$stages, function(s)
      if (!is.na(s$params) && s$params > 0) s$group)))
    for (p in phases) {
      expect_true(p %in% txt || grepl(paste0(p, " ("), joined, fixed = TRUE),
                  info = paste(nm, p))
    }
  }
})


test_that("icicle label text keeps 4.5:1 contrast on every fill", {
  for (theme in c("light", "dark")) {
    th <- arch_icicle_theme(theme)
    fills <- c(th$phase, th$other, th$kind,
               vapply(th$kind, arch_mix, character(1), th$surface, 0.75))
    for (f in fills) {
      expect_gte(arch_contrast(f, arch_text_on(f)), 4.5,
                 label = paste(theme, f))
    }
  }
  expect_equal(arch_contrast("#000000", "#ffffff"), 21)
  expect_identical(arch_text_on("#ffffff"), "#111111")
  expect_identical(arch_text_on("#0072B2"), "#ffffff")
})


# ---------------------------------------------------------------------------
# glyphs
# ---------------------------------------------------------------------------

# The glyph rects drawn for each stage, found by the stage label's row.
glyph_rects <- function(arch, sc) {
  g <- arch_glyphs_geom()
  txt <- Filter(function(e) e$type == "text" && isTRUE(all.equal(e$x, g$x_text)) &&
                  isTRUE(e$bold), sc$items)
  tops <- vapply(txt, `[[`, numeric(1), "y") - 7
  cells <- Filter(function(e) e$type == "rect" && e$x < g$x_text &&
                    e$h < g$cell_h + 0.5 && e$w > 7.5, sc$items)
  lapply(seq_along(tops), function(i) {
    hi <- if (i < length(tops)) tops[i + 1L] else Inf
    Filter(function(e) e$y >= tops[i] - 1 && e$y < hi - 1, cells)
  })
}

test_that("glyphs collapse to one wide column after the collapsing stage", {
  g <- arch_glyphs_geom()
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    sc <- arch_scene(arch, style = "glyphs")
    rects <- glyph_rects(arch, sc)
    expect_length(rects, length(arch$stages))
    wide <- vapply(rects, function(r)
      length(r) > 0L && all(vapply(r, `[[`, numeric(1), "w") > g$wide_w - 2),
      logical(1))
    at <- which(vapply(arch$stages, function(s) isTRUE(s$collapses), logical(1)))
    expected <- if (length(at)) seq_along(arch$stages) > at else
      logical(length(arch$stages))
    # The prediction glyph keeps only its test rows, so judge it by
    # width like the rest.
    expect_identical(wide, expected, info = nm)
  }
})


test_that("a glyph shows every axis its sublayers attend over", {
  th <- arch_glyphs_theme()
  arch <- arch_test_arch("mitra")
  layer <- Filter(function(s) identical(s$id, "layers"), arch$stages)[[1]]
  expect_null(layer$axis)
  ax <- arch_glyph_axes(layer)
  expect_true(ax$rows && ax$cols && !ax$induce)
  fills <- vapply(arch_glyph_items(layer, 0, 0, FALSE, arch_glyphs_geom(),
                                   th)$items, function(e)
    if (e$type == "rect") e$fill else NA_character_, character(1))
  expect_true(all(c(th$rows, th$cols) %in% fills))

  tabicl <- arch_test_arch("tabicl")
  col_tf <- Filter(function(s) identical(s$id, "col_tf"), tabicl$stages)[[1]]
  expect_true(arch_glyph_axes(col_tf)$induce)
  # TabPFN v2.6's attention between cells runs down a column.
  expect_true(arch_glyph_axes(list(axis = "cells", children = list()))$rows)
})


test_that("glyph highlights keep 3:1 contrast against the table", {
  for (theme in c("light", "dark")) {
    th <- arch_glyphs_theme(theme)
    for (h in c("rows", "cols", "induce", "out", "embed")) {
      for (f in c("train", "test", "surface")) {
        expect_gte(arch_contrast(th[[h]], th[[f]]), 3,
                   label = paste(theme, h, "on", f))
      }
    }
  }
})



# ---------------------------------------------------------------------------
# plot_architecture_diff()
# ---------------------------------------------------------------------------

test_that("a model diffed with itself is matched and shared throughout", {
  arch <- arch_test_arch("tabpfn35")
  pairs <- arch_align_stages(arch$stages, arch$stages)
  expect_length(pairs, length(arch$stages))
  expect_true(all(vapply(pairs, arch_pair_status, character(1)) == "matched"))
  facts <- arch_diff_facts(arch$facts, arch$facts)
  expect_true(all(vapply(facts, `[[`, character(1), "status") == "shared"))
})


test_that("TabICL against TabPFN v3 pairs the stages that correspond", {
  cs <- arch_test_configs()
  icl <- tabfound_architecture(arch_test_model(
    utils::modifyList(cs$tabicl$config, list(icl_num_blocks = 3L)),
    "tabicl", cs$tabicl$task))
  pfn <- arch_test_arch("tabpfn3")
  pairs <- arch_align_stages(icl$stages, pfn$stages)
  id <- function(p, side) if (is.null(p[[side]])) NA_character_ else p[[side]]$id
  status <- stats::setNames(vapply(pairs, arch_pair_status, character(1)),
                            vapply(pairs, function(p)
                              paste(id(p, "a"), id(p, "b")), character(1)))
  expect_identical(status[["cls NA"]], "only A")
  expect_identical(status[["NA onorm"]], "only B")
  expect_identical(status[["icl icl"]], "3 vs 2")
  expect_identical(status[["dec dec"]], "matched")
  # Every stage of each model appears once, in its own order.
  a_ids <- Filter(Negate(is.na), vapply(pairs, id, character(1), "a"))
  b_ids <- Filter(Negate(is.na), vapply(pairs, id, character(1), "b"))
  expect_identical(unname(a_ids), vapply(icl$stages, `[[`, character(1), "id"))
  expect_identical(unname(b_ids), vapply(pfn$stages, `[[`, character(1), "id"))

  facts <- arch_diff_facts(icl$facts, pfn$facts)
  st <- stats::setNames(vapply(facts, `[[`, character(1), "status"),
                        vapply(facts, `[[`, character(1), "name"))
  expect_identical(st[["ICL layers"]], "different")
  expect_identical(st[["Embedding width"]], "shared")
  expect_identical(st[["Train/test split"]], "one side")
  # Shared and differing facts first, then A's own, then B's.
  first_one_side <- match("one side", st)
  expect_true(all(st[seq_len(first_one_side - 1L)] != "one side"))
})


test_that("a symbol the two models define differently is keyed once per model", {
  th <- arch_diff_theme()
  sym <- arch_diff_symbols(c(B = "batch", F = "feature groups"),
                           c(B = "batch", F = "features", S = "support rows"),
                           th)
  expect_identical(names(sym$items), c("B", "F (A)", "F (B)", "S"))
  expect_identical(unname(sym$cols), c(th$a, th$b))

  sc <- arch_diff_scene(arch_test_arch("tabicl"), arch_test_arch("mitra"))
  syms <- names(sc$key[[1]]$items)
  expect_true(all(c("F (A)", "F (B)") %in% syms))
  expect_false("F" %in% syms)
})


test_that("the diff draws its key whole, and its text is readable", {
  for (pr in list(c("tabicl", "tabpfn3"), c("tabicl", "mitra"),
                  c("kumo", "tabfm"))) {
    A <- arch_test_arch(pr[1])
    B <- arch_test_arch(pr[2])
    for (theme in c("light", "dark")) {
      sc <- arch_diff_scene(A, B, theme)
      expect_key_whole(sc, arch_merge_glossary(), paste(pr, collapse = " "))
    }
  }
  for (theme in c("light", "dark")) {
    th <- arch_diff_theme(theme)
    for (col in c("a", "b", "diff", "same", "ink2", "ink3")) {
      for (bg in c("surface", "a_bg", "b_bg")) {
        expect_gte(arch_contrast(th[[col]], th[[bg]]), 4.5,
                   label = paste(theme, col, "on", bg))
      }
    }
  }
})


test_that("plot_architecture_diff takes models and descriptions in any mix", {
  skip_on_cran()
  cs <- arch_test_configs()$mitra
  m <- arch_test_model(cs$config, "mitra", cs$task)
  arch <- arch_test_arch("tabicl")
  f <- file.path(withr::local_tempdir(), "diff.svg")
  expect_identical(suppressMessages(plot_architecture_diff(arch, m, f)), f)
  svg <- readLines(f, warn = FALSE, encoding = "UTF-8")
  expect_match(svg[3], "<title>TabICL v2 vs Mitra</title>", fixed = TRUE)
  expect_true(any(grepl("only A", svg, fixed = TRUE)))
  expect_error(plot_architecture_diff(arch, "nope"), "Cannot describe")
  expect_error(plot_architecture_diff(arch, m, format = "png"), "needs a .*file")
})


# ---------------------------------------------------------------------------
# Descriptions checked against the forward code
# ---------------------------------------------------------------------------

stage_ids <- function(arch) vapply(arch$stages, `[[`, character(1), "id")
stage_by_id <- function(arch, id) arch$stages[[match(id, stage_ids(arch))]]

test_that("Kumo's column and row blocks alternate, one of each per layer", {
  for (nm in c("kumo classification", "kumo regression", "kumo small")) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    ids <- stage_ids(arch)
    expect_identical(stage_by_id(arch, "col")$interleave, "embedding",
                     info = nm)
    expect_identical(stage_by_id(arch, "row")$interleave, "embedding",
                     info = nm)
    expect_identical(match("row", ids) - match("col", ids), 1L, info = nm)
    expect_match(stage_by_id(arch, "readout")$detail, "column blocks skip them",
                 fixed = TRUE)
  }
  arch <- arch_test_arch("kumo")
  expect_identical(arch_repeat_tag(stage_by_id(arch, "col")),
                   "× 2, alternating")
  for (style in names(arch_styles())) {
    notes <- unlist(lapply(arch_scene(arch, style = style)$key, function(g)
      vapply(g$notes, `[[`, character(1), "text")))
    if (identical(style, "icicle")) next   # draws no repeat tags
    expect_true(any(startsWith(notes, "alternating: Column block and Row block")),
                info = style)
  }
  diff <- arch_diff_scene(arch, arch)
  notes <- vapply(diff$key[[3]]$notes, `[[`, character(1), "text")
  expect_identical(sum(startsWith(notes, "alternating:")), 1L)
})


test_that("interleaved stages must be consecutive and repeat alike", {
  st <- function(id, rep, il) arch_stage(id, id, repeats = rep, interleave = il)
  expect_true(arch_check_interleave(
    list(st("a", 2L, "g"), st("b", 2L, "g"), st("c", 1L, NULL)), "x"))
  expect_error(arch_check_interleave(
    list(st("a", 2L, "g"), st("c", 1L, NULL), st("b", 2L, "g")), "x"),
    "interleaves")
  expect_error(arch_check_interleave(
    list(st("a", 2L, "g"), st("b", 3L, "g")), "x"), "interleaves")
  expect_error(arch_check_interleave(list(st("a", 2L, "g")), "x"),
               "interleaves")
})


test_that("TabFM's CLS columns join after the first column stage", {
  for (nm in c("tabfm classification", "tabfm regression")) {
    ids <- stage_ids(arch_test_variant_arch(arch_test_variants()[[nm]]))
    expect_true(match("col1", ids) < match("cls", ids), info = nm)
    expect_true(match("cls", ids) < match("row1", ids), info = nm)
  }
})


test_that("TabICL's and TabFM's final norm belongs to the decoder", {
  for (nm in c("tabicl", "tabfm")) {
    arch <- arch_test_arch(nm)
    expect_true("icl_predictor.ln" %in% stage_by_id(arch, "dec")$prefix,
                info = nm)
    expect_false("icl_predictor.ln" %in% stage_by_id(arch, "icl_y")$prefix,
                 info = nm)
    expect_identical(arch_audit_coverage(arch)$duplicated, character(),
                     info = nm)
  }
})


test_that("TabPFN v2.x describe their encoders and embeddings as built", {
  reg <- arch_test_variant_arch(arch_test_variants()[["tabpfn regression"]])
  clf <- arch_test_variant_arch(arch_test_variants()[["tabpfn classification"]])
  expect_match(stage_by_id(reg, "encoder")$detail, "^MLP\\(6 -> ")
  expect_match(stage_by_id(clf, "encoder")$detail, "^Linear\\(6 -> 12\\)")
  expect_match(stage_by_id(clf, "y_encoder")$detail, "extra column",
               fixed = TRUE)
  # The test config is 12 wide, so the subspace is 3-d, not 48.
  v26 <- arch_test_arch("tabpfn26")
  expect_match(stage_by_id(v26, "pos")$detail, "fixed 3-d per column",
               fixed = TRUE)
  net <- arch_test_model(arch_test_configs()$tabpfn26$config, "tabpfn26",
                         "regression")$model
  expect_identical(
    net$feature_positional_embedding_embeddings$weight$size(2), 3L)
})


# ---------------------------------------------------------------------------
# schematic
# ---------------------------------------------------------------------------

schematic_plus_count <- function(sc) {
  r <- arch_schematic_geom()$plus_r
  sum(vapply(sc$items, function(e) e$type == "rect" &&
               isTRUE(all.equal(e$w, 2 * r)) && isTRUE(all.equal(e$r, r)),
             logical(1)))
}

test_that("the schematic draws one plus per residual connection and per added stage", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    sc <- arch_scene(arch, style = "schematic")
    units <- sum(vapply(arch$stages, function(s) {
      r <- vapply(s$children, `[[`, integer(1), "residual")
      length(unique(r[!is.na(r)]))
    }, integer(1)))
    adds <- sum(vapply(arch$stages, function(s) identical(s$join, "add"),
                       logical(1)))
    expect_identical(schematic_plus_count(sc), units + adds, info = nm)
  }
})

test_that("the schematic marks each repeated container once", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    sc <- arch_scene(arch, style = "schematic")
    txt <- vapply(Filter(function(e) e$type == "text", sc$items), `[[`,
                  character(1), "text")
    marks <- sum(grepl("^[0-9]+ ×$", txt))
    # A container per stage with sublayers, interleaved ones sharing one;
    # only stacks of more than one are marked.
    with_kids <- Filter(function(s) length(s$children) > 0L, arch$stages)
    groups <- unique(vapply(with_kids, function(s) s$interleave %||% s$id,
                            character(1)))
    reps <- vapply(groups, function(g) {
      s <- Filter(function(s) identical(s$interleave %||% s$id, g), with_kids)[[1]]
      s$repeats
    }, integer(1))
    expect_identical(marks, sum(reps > 1L), info = nm)
  }
})

test_that("schematic side inputs sit left of the flow and callouts right of it", {
  g <- arch_schematic_geom()
  left_of_flow <- g$xc - g$box_w / 2 - g$cpad - 10 - 20
  arch <- arch_test_variant_arch(arch_test_variants()[["kumo classification"]])
  sc <- arch_scene(arch, style = "schematic")
  side <- Filter(function(e) e$type == "rect" && isTRUE(all.equal(e$x, g$side_x)),
                 sc$items)
  expect_gt(length(side), 0L)
  expect_true(all(vapply(side, function(e) e$x + e$w, numeric(1)) < left_of_flow))
  calls <- Filter(function(e) e$type == "text" && e$x >= g$call_x, sc$items)
  expect_gt(length(calls), 0L)
  flow_right <- g$xc + g$box_w / 2 + g$skip + 16
  expect_true(g$call_x > flow_right)
})

test_that("attention steps carry their variant in the schematic", {
  arch <- arch_test_arch("tabpfn")
  sc <- arch_scene(arch, style = "schematic")
  txt <- vapply(Filter(function(e) e$type == "text", sc$items), `[[`,
                character(1), "text")
  expect_true("keys: context rows · test rows: 1 KV head (MQA)" %in% txt)
  expect_identical(arch_attn_line(arch_attn()), NULL)
  expect_identical(arch_attn_line(arch_attn("training rows", kv_heads_test = 2)),
                   "keys: training rows · test rows: 2 KV heads (GQA)")
})

test_that("a description without residual connections is drawn, and says so", {
  arch <- arch_test_arch("tabicl")
  arch$stages <- lapply(arch$stages, function(s) {
    s$children <- lapply(s$children, function(ch) { ch$residual <- NULL; ch })
    s
  })
  sc <- arch_scene(arch, style = "schematic")
  notes <- vapply(sc$key[[3]]$notes, `[[`, character(1), "text")
  expect_true(any(grepl("not recorded", notes, fixed = TRUE)))
  expect_identical(schematic_plus_count(sc),
                   sum(vapply(arch$stages, function(s) identical(s$join, "add"),
                              logical(1))))
})

test_that("every backend's accent is readable in both themes", {
  for (theme in c("light", "dark")) {
    th <- arch_schematic_theme(theme)
    for (b in names(ARCH_ACCENTS)) {
      arch <- list(backend = b, title = "x", subtitle = NULL, facts = character(),
                   stages = list(arch_stage("in", "Input", kind = "input")),
                   symbols = character(), glossary = ARCH_GLOSSARY)
      sc <- arch_scene(arch, style = "schematic", theme = theme)
      title <- Filter(function(e) e$type == "text" && identical(e$text, "x"),
                      sc$items)[[1]]
      expect_gte(arch_contrast(title$col, th$surface), 4.5,
                 label = paste(theme, b))
      expect_gte(arch_contrast(arch_readable(title$col, th$attn), th$attn), 4.5,
                 label = paste(theme, b, "on attention"))
    }
  }
  expect_identical(arch_readable("#000000", "#ffffff"), "#000000")
  expect_gte(arch_contrast(arch_readable("#ffff00", "#ffffff"), "#ffffff"), 4.5)
})

schematic_zooms <- function(sc) {
  Filter(function(e) e$type == "rect" && isTRUE(e$dash) &&
           isTRUE(all.equal(e$r, 8)), sc$items)
}

test_that("full detail adds one zoom box per distinct feed-forward", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    ffns <- unlist(lapply(arch$stages, function(s) lapply(
      Filter(function(ch) identical(ch$kind, "ffn"), s$children),
      function(ch) paste(unlist(ch$ffn), collapse = "/"))))
    expect_length(schematic_zooms(arch_scene(arch, style = "schematic")), 0L)
    full <- arch_scene(arch, style = "schematic", detail = "full")
    expect_length(schematic_zooms(full), length(unique(ffns)))
  }
})

test_that("a zoom box shows the feed-forward's widths and steps", {
  txt_of <- function(sc) vapply(Filter(function(e) e$type == "text", sc$items),
                                `[[`, character(1), "text")
  tabfm <- arch_scene(arch_test_arch("tabfm"), style = "schematic",
                      detail = "full")
  txt <- txt_of(tabfm)
  expect_true("SiLU" %in% txt)
  expect_true("×" %in% txt)
  expect_true("8 → 16 → 8" %in% txt)
  expect_true("SiLU" %in% names(tabfm$key[[2]]$items))
  # Biases left out of a feed-forward are said so.
  pfn <- txt_of(arch_scene(arch_test_arch("tabpfn"), style = "schematic",
                           detail = "full"))
  expect_true("12 → 24 → 12, no bias" %in% pfn)
})


# ---------------------------------------------------------------------------
# Your own colours
# ---------------------------------------------------------------------------

BRANDS <- list(cream = c("#f4f1ea", "#1f4e79"), navy = c("#0b1f3a", "#f2a900"),
               white = c("#ffffff", "#e4002b"), charcoal = c("#2b2b2b", "#00a3a1"))

test_that("leaving the colours out changes nothing", {
  arch <- arch_test_arch("tabicl")
  for (style in names(arch_styles())) {
    expect_identical(arch_scene(arch, style = style),
                     arch_scene(arch, style = style, background = NULL,
                                highlight = NULL), info = style)
  }
})

test_that("a background reaches every style and keeps the text readable", {
  arch <- arch_test_arch("tabicl")
  for (b in names(BRANDS)) {
    bg <- BRANDS[[b]][1]
    for (style in names(arch_styles())) {
      sc <- arch_scene(arch, style = style, background = bg,
                       highlight = BRANDS[[b]][2])
      info <- paste(b, style)
      expect_identical(sc$bg, bg, info = info)
      expect_identical(sc$theme$surface, bg, info = info)
      # The title and the key's body text stay readable on it.
      expect_gte(arch_contrast(sc$theme$ink, bg), 4.5, label = info)
      expect_gte(arch_contrast(sc$theme$ink2, bg), 4.5, label = info)
      txt <- Filter(function(e) e$type == "text", sc$items)
      expect_gte(arch_contrast(txt[[1]]$col, bg), 4.5, label = paste(info, "title"))
    }
    d <- arch_diff_scene(arch, arch_test_arch("mitra"),
                         theme = arch_theme_for(bg),
                         colours = list(background = bg, highlight = BRANDS[[b]][2]))
    expect_identical(d$bg, bg, info = b)
    for (h in c("a", "b", "diff", "same", "ink2", "ink3")) {
      for (f in c("surface", "a_bg", "b_bg")) {
        expect_gte(arch_contrast(d$theme[[h]], d$theme[[f]]), 4.5,
                   label = paste(b, h, "on", f))
      }
    }
  }
})

test_that("a theme carried to another background keeps its contrasts", {
  for (theme in c("light", "dark")) {
    th <- arch_theme(theme)
    for (bg in c("#f4f1ea", "#ffffff", "#0b1f3a", "#2b2b2b")) {
      new <- arch_adapt_theme(th, bg)
      for (nm in c("ink", "ink2", "ink3", "panel", "rule")) {
        best <- max(arch_contrast("#000000", bg), arch_contrast("#ffffff", bg))
        want <- min(arch_contrast(th[[nm]], th$surface), best)
        expect_gte(arch_contrast(new[[nm]], bg), want - 0.05,
                   label = paste(theme, bg, nm))
      }
      expect_identical(new$hue, th$hue)
    }
  }
  expect_identical(arch_adapt_theme(arch_theme(), NULL), arch_theme())
})

test_that("the highlight lands where each style says it does", {
  arch <- arch_test_arch("tabicl")
  hl <- "#1f4e79"
  near <- function(col) arch_hue_distance(col, hl) < 0.03
  colours <- list(background = "#f4f1ea", highlight = hl)
  expect_true(near(arch_cards_colours(arch_theme(), colours)$hue[["attention"]]))
  expect_true(near(arch_glyphs_colours(arch_glyphs_theme(), colours)$rows))
  expect_true(all(vapply(arch_tufte_phase_colours(3, arch_adapt_theme(
    arch_tufte_theme(), colours$background), colours), near, logical(1))))
  ice <- arch_scene(arch, style = "icicle", highlight = hl)
  fills <- vapply(Filter(function(e) e$type == "rect", ice$items),
                  function(e) e$fill %||% NA_character_, character(1))
  expect_true(hl %in% fills)
  sch <- arch_scene(arch, style = "schematic", highlight = hl)
  title <- Filter(function(e) e$type == "text", sch$items)[[1]]
  expect_true(near(title$col))
  # The diff's model B moves away from a highlight that would crowd it.
  th <- arch_diff_colours(arch_diff_theme(), list(highlight = "#f2a900"))
  expect_gt(arch_hue_distance(th$a, th$b), 0.2)
})

test_that("a dark background picks the dark theme unless told otherwise", {
  arch <- arch_test_arch("mitra")
  f <- file.path(withr::local_tempdir(), "a.svg")
  suppressMessages(plot_architecture(arch, f, background = "#0b1f3a"))
  expect_match(paste(readLines(f, warn = FALSE), collapse = ""),
               "fill=\"#0b1f3a\"", fixed = TRUE)
  expect_identical(arch_theme_for("#0b1f3a"), "dark")
  expect_identical(arch_theme_for("#f4f1ea"), "light")
  expect_identical(arch_scene(arch, background = "#0b1f3a")$theme$name, "dark")
  expect_identical(arch_scene(arch, background = "#0b1f3a",
                              theme = "light")$theme$name, "light")
  expect_identical(arch_colour("navy", "x"), "#000080")
  expect_error(plot_architecture(arch, f, background = "not a colour"),
               "must be one colour")
  expect_error(plot_architecture_diff(arch, arch, f, highlight = rep("red", 4)),
               "must be one to 3 colours")
  expect_error(plot_architecture(arch, f, highlight = rep("red", 5)),
               "must be one to 4 colours")
})

test_that("the diff takes one highlight per model and one for what differs", {
  hl <- c("#440154", "#21908c", "#8f7a00")
  near <- function(a, b) arch_hue_distance(a, b) < 0.03
  th <- arch_diff_colours(arch_diff_theme(),
                          list(background = "#ffffff", highlight = hl))
  for (i in seq_along(hl)) {
    h <- c("a", "b", "diff")[i]
    expect_true(near(th[[h]], hl[i]), label = h)
  }
  # B given, it is used as given rather than picked to stand apart.
  th2 <- arch_diff_colours(arch_diff_theme(), list(highlight = hl[1:2]))
  expect_true(near(th2$b, hl[2]))
  expect_identical(th2$diff, arch_diff_colours(arch_diff_theme(),
                                               list(highlight = hl[1]))$diff)
  # What matches stays grey, darkened only to read on the new fills.
  expect_lt(grDevices::rgb2hsv(grDevices::col2rgb(th$same))["s", 1], 0.1)
  f <- file.path(withr::local_tempdir(), "d.svg")
  arch <- arch_test_arch("tabicl")
  expect_no_error(suppressMessages(
    plot_architecture_diff(arch, arch_test_arch("mitra"), f, highlight = hl)))
})

test_that("the glyphs take one highlight per thing they pick out", {
  arch <- arch_test_arch("kumo")
  hl <- c("#440154", "#31688e", "#35b779", "#8f7a00")
  near <- function(a, b) arch_hue_distance(a, b) < 0.03
  th <- arch_glyphs_colours(arch_glyphs_theme(),
                            list(background = "#ffffff", highlight = hl))
  for (i in seq_along(hl)) {
    h <- c("rows", "cols", "induce", "out")[i]
    expect_true(near(th[[h]], hl[i]), label = h)
  }
  # Fewer than four leave the rest as the theme has them.
  th2 <- arch_glyphs_colours(arch_glyphs_theme(), list(highlight = hl[1:2]))
  expect_identical(th2$induce, arch_glyphs_colours(arch_glyphs_theme(),
                                                   list(highlight = hl[1]))$induce)
  expect_true(near(th2$cols, hl[2]))
  # Tufte takes them as its phase colours, in stage order, and blends
  # between them when there are more phases than colours.
  surf <- arch_adapt_theme(arch_tufte_theme(), "#ffffff")
  cols <- arch_tufte_phase_colours(3, surf, list(background = "#ffffff",
                                                 highlight = hl))
  for (i in 1:3) expect_true(near(cols[i], hl[i]), label = paste("phase", i))
  more <- arch_tufte_phase_colours(6, surf, list(highlight = hl[1:2]))
  expect_length(unique(more), 6)
  expect_true(near(more[1], hl[1]))
  expect_true(near(more[6], hl[2]))
  # Colours you choose need not run dark to light, so the key says less.
  tufte_text <- function(sc) paste(vapply(
    Filter(function(e) e$type == "text", sc$items), `[[`, character(1), "text"),
    collapse = "\n")
  expect_match(tufte_text(arch_scene(arch, style = "tufte", highlight = hl)),
               "one colour each in stage order")
  expect_match(tufte_text(arch_scene(arch, style = "tufte")),
               "from first (dark) to last (light)", fixed = TRUE)
  # The schematic has one accent and draws the first, as if given alone.
  expect_identical(arch_scene(arch, style = "schematic", highlight = hl),
                   arch_scene(arch, style = "schematic", highlight = hl[1]))
})

test_that("cards and icicle take one highlight per coloured kind", {
  arch <- arch_test_arch("tabicl")
  hl <- c("#440154", "#31688e", "#35b779", "#8f7a00")
  near <- function(a, b) arch_hue_distance(a, b) < 0.03
  th <- arch_cards_colours(arch_theme(), list(background = "#ffffff",
                                              highlight = hl))
  ice <- arch_scene(arch, style = "icicle", highlight = hl)
  fills <- vapply(Filter(function(e) e$type == "rect", ice$items),
                  function(e) e$fill %||% NA_character_, character(1))
  for (i in seq_along(hl)) {
    k <- ARCH_HIGHLIGHT_KINDS[i]
    expect_true(near(th$hue[[k]], hl[i]), label = paste("cards", k))
    expect_gte(arch_contrast(th$hue[[k]], th$surface), 3)
  }
  # TabICL has attention, embedding and feed-forward blocks to colour.
  expect_true(all(hl[1:3] %in% fills))
  # Fewer leave the rest as the theme has them.
  th1 <- arch_cards_colours(arch_theme(), list(highlight = hl[1]))
  expect_identical(th1$hue[["embed"]], arch_theme()$hue[["embed"]])
  # The neutral kinds stay neutral.
  expect_identical(th$hue[["norm"]], arch_cards_colours(
    arch_theme(), list(background = "#ffffff"))$hue[["norm"]])
})
