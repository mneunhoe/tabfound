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
  fake <- list(title = "A GQA model", facts = character(), stages = list())
  expect_identical(arch_unexplained_terms(fake), "GQA")
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


test_that("the key is drawn whole on every style, theme and setting", {
  scene_text <- function(sc) {
    vapply(Filter(function(e) e$type == "text", sc$items), `[[`,
           character(1), "text")
  }
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    sym <- arch_used_symbols(arch)
    expect_true(length(sym) > 0L, info = nm)
    for (style in names(arch_styles())) {
      for (theme in c("light", "dark")) {
        for (detail in c("overview", "full")) {
          for (facts in c(TRUE, FALSE)) {
            info <- paste(nm, style, theme, detail, facts)
            sc <- arch_scene(arch, style = style, detail = detail,
                             theme = theme, show_facts = facts)
            txt <- scene_text(sc)
            at <- match("Key", txt)
            expect_false(is.na(at), info = info)
            key <- txt[seq(at, length(txt))]
            # Never truncated.
            expect_false(any(grepl("...", key, fixed = TRUE)), info = info)
            # Every symbol and term, with its whole meaning: wrapped
            # lines are consecutive items, so joining them restores it.
            joined <- paste(key, collapse = " ")
            terms <- c(sym, arch_used_terms(arch, unname(ARCH_KINDS)))
            for (t in names(terms)) {
              expect_true(t %in% key, info = paste(info, t))
              expect_true(grepl(terms[[t]], joined, fixed = TRUE),
                          info = paste(info, t))
            }
            svg <- arch_render_svg(sc)
            expect_identical(sum(startsWith(svg, "  <title>")), 1L, info = info)
            expect_identical(sum(startsWith(svg, "  <desc>")), 1L, info = info)
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
