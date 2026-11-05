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

# The fixtures were written by `main` before the styles existed. Moving
# the cards scene into its own builder must not move a single byte; a
# deliberate change to the cards look regenerates them.
test_that("cards output is byte-identical to the pre-styles diagrams", {
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
