# Diagram layout.
#
# Laying out and drawing are kept apart. This file turns a
# `tabfound_arch` into a *scene*: a flat list of rectangles, texts, lines
# and polygons with pixel coordinates, plus the canvas size. Nothing here
# knows what an SVG or a graphics device is. `arch-render.R` has two
# renderers that walk the same scene, which is why the SVG, PNG and PDF
# outputs are the same picture rather than three drifting near-copies.
#
# Coordinates are CSS pixels with the origin at the top left and y
# increasing downwards, i.e. SVG's own system. The grid renderer flips
# the y axis once, in the viewport, rather than per item.
#
# A *style* is a scene builder: the same description drawn a different
# way. The builders are registered in `arch_styles()`, and each lives in
# its own `arch-style-*.R` file; what they share -- the primitives, the
# text metrics, wrapping and colour -- is here.

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

# The registry. A function rather than a list constant because the
# builders live in files that collate after this one, so a list built at
# load time would find them undefined. Every builder takes
# `(arch, detail, theme, show_shapes, show_params, show_facts)`, with
# `detail` and `theme` already matched, and ignores what it cannot draw.
# @keywords internal
arch_styles <- function() {
  list(cards = arch_scene_cards)
}

# @keywords internal
arch_match_style <- function(style) {
  styles <- names(arch_styles())
  if (!is.character(style) || length(style) != 1L || !style %in% styles) {
    cli::cli_abort(c(
      "{.arg style} must be one of {.or {.val {styles}}}.",
      x = "Got {.val {style}}."
    ))
  }
  style
}

#' Lay a described architecture out into a renderer-independent scene
#'
#' @param arch A `tabfound_arch` from [tabfound_architecture()].
#' @param style A style registered in `arch_styles()`.
#' @param detail `"overview"` or `"full"`.
#' @param theme `"light"` or `"dark"`.
#' @param show_shapes,show_params Draw the right-margin annotations.
#' @param show_facts Draw the fact sheet at the foot of the diagram.
#' @return A list with `width`, `height`, `bg` and `items`.
#' @keywords internal
arch_scene <- function(arch, style = "cards", detail = c("overview", "full"),
                       theme = c("light", "dark"),
                       show_shapes = TRUE, show_params = TRUE,
                       show_facts = TRUE) {
  build <- arch_styles()[[arch_match_style(style)]]
  build(arch, detail = match.arg(detail), theme = match.arg(theme),
        show_shapes = show_shapes, show_params = show_params,
        show_facts = show_facts)
}

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

# Mix two hex colours. Used only to derive a block's fill from its accent
# hue, which is a lightness step within one hue -- not a second
# categorical decision.
# @keywords internal
arch_mix <- function(a, b, w) {
  ca <- grDevices::col2rgb(a)[, 1]
  cb <- grDevices::col2rgb(b)[, 1]
  grDevices::rgb(t(round(ca * w + cb * (1 - w))), maxColorValue = 255)
}

# Viridis without the dependency. The anchors are
# `viridisLite::viridis(17)`, interpolated in Lab, which stays within a
# small colour difference of the real map everywhere. Not
# `hcl.colors("viridis")`: that is a different curve, and the contrast
# limits the styles pick `begin` and `end` from were checked against the
# real one.
ARCH_VIRIDIS <- c(
  "#440154", "#48186A", "#472D7B", "#424086", "#3B528B", "#33638D",
  "#2C728E", "#26828E", "#21908C", "#1F9F88", "#27AD81", "#3EBC74",
  "#5DC863", "#82D34D", "#AADC32", "#D5E21A", "#FDE725"
)

# @keywords internal
arch_viridis <- function(n, begin = 0, end = 1) {
  if (n < 1L) return(character())
  ramp <- grDevices::colorRamp(ARCH_VIRIDIS, space = "Lab")
  at <- if (n == 1L) begin else seq(begin, end, length.out = n)
  grDevices::rgb(ramp(at), maxColorValue = 255)
}

# ---------------------------------------------------------------------------
# Scene primitives
# ---------------------------------------------------------------------------

# @keywords internal
sc_rect <- function(x, y, w, h, fill = NA, stroke = NA, lwd = 1, r = 0,
                    dash = FALSE) {
  list(type = "rect", x = x, y = y, w = w, h = h, fill = fill,
       stroke = stroke, lwd = lwd, r = r, dash = dash)
}

# `y` is the vertical centre of the text, not its baseline: centring is
# what every caller here wants, and it is the one convention both
# renderers can honour exactly. `family` is `"sans"`, `"serif"` or
# `"mono"`; `mono = TRUE` is the older spelling of `family = "mono"`.
# @keywords internal
sc_text <- function(x, y, text, size = 12, col = "#000000", anchor = "start",
                    bold = FALSE, italic = FALSE, mono = FALSE,
                    family = if (mono) "mono" else "sans") {
  list(type = "text", x = x, y = y, text = text, size = size, col = col,
       anchor = anchor, bold = bold, italic = italic,
       family = arch_match_family(family))
}

# @keywords internal
arch_match_family <- function(family) {
  match.arg(family, c("sans", "serif", "mono"))
}

# @keywords internal
sc_line <- function(x1, y1, x2, y2, col = "#000000", lwd = 1, dash = FALSE) {
  list(type = "line", x1 = x1, y1 = y1, x2 = x2, y2 = y2, col = col,
       lwd = lwd, dash = dash)
}

# @keywords internal
sc_poly <- function(x, y, fill) list(type = "poly", x = x, y = y, fill = fill)

# A downward flow arrow between two blocks.
# @keywords internal
sc_arrow <- function(x, y1, y2, col) {
  list(
    sc_line(x, y1, x, y2 - 5, col = col, lwd = 1.4),
    sc_poly(c(x - 4, x + 4, x), c(y2 - 6, y2 - 6, y2), fill = col)
  )
}

# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

# Neither renderer can ask a font how wide a string is before it is
# drawn -- the SVG writer has no font at all, and the grid path would
# need an open device. A per-character average is enough here, because it
# is used only to decide when to ellipsize inside a fixed-width box. The
# factors are measured off a rendered diagram in the sans stack the SVG
# asks for, rounded up so the estimate errs towards truncating. The serif
# factors take the wider of Palatino and its Georgia fallback, measured
# over the gallery's labels, plus the same 5% margin; italic sans is
# close enough to upright that it shares the regular factor.
# @keywords internal
ARCH_CHAR_W <- c(regular = 0.485, bold = 0.515, mono = 0.60,
                 serif = 0.48, serif_bold = 0.555, serif_italic = 0.49)

# @keywords internal
arch_char_w <- function(bold, mono = FALSE, family = if (mono) "mono" else "sans",
                        italic = FALSE) {
  key <- switch(arch_match_family(family),
    mono  = "mono",
    sans  = if (bold) "bold" else "regular",
    serif = if (bold) "serif_bold" else if (italic) "serif_italic" else "serif"
  )
  ARCH_CHAR_W[[key]]
}

# @keywords internal
arch_text_width <- function(text, size, bold = FALSE, mono = FALSE,
                            family = if (mono) "mono" else "sans",
                            italic = FALSE) {
  nchar(text, type = "chars") * size * arch_char_w(bold, family = family,
                                                    italic = italic)
}

# @keywords internal
arch_fit_text <- function(text, max_px, size, bold = FALSE, mono = FALSE,
                          family = if (mono) "mono" else "sans",
                          italic = FALSE) {
  if (is.null(text) || !nzchar(text)) return(text)
  cw <- arch_char_w(bold, family = family, italic = italic)
  if (nchar(text, type = "chars") * size * cw <= max_px) return(text)
  # Three dots rather than U+2026: the PDF device's font encodings have no
  # ellipsis glyph and warn when asked for one, and nothing is gained by
  # making the two outputs differ over it.
  keep <- max(1L, floor(max_px / (size * cw)) - 3L)
  paste0(substr(text, 1L, keep), "...")
}

# Greedy word wrap into lines no wider than `max_px`. A word too long for
# a line on its own is cut with `arch_fit_text()`. With `max_lines`, the
# text that does not fit is dropped and the last line ends in "...", so
# a caller that must not lose text leaves it at `Inf`.
# @keywords internal
arch_wrap_text <- function(text, max_px, size, family = "sans", bold = FALSE,
                           italic = FALSE, max_lines = Inf) {
  if (is.null(text) || !nzchar(text)) return(character())
  fits <- function(s) {
    arch_text_width(s, size, bold, family = family, italic = italic) <= max_px
  }
  words <- strsplit(trimws(text), "\\s+")[[1L]]
  lines <- character()
  cur <- ""
  for (w in words) {
    cand <- if (nzchar(cur)) paste(cur, w) else w
    if (fits(cand)) {
      cur <- cand
      next
    }
    if (nzchar(cur)) lines <- c(lines, cur)
    cur <- if (fits(w)) w else
      arch_fit_text(w, max_px, size, bold, family = family, italic = italic)
  }
  lines <- c(lines, cur)
  if (length(lines) > max_lines) {
    last <- paste0(lines[max_lines], "...")
    if (!fits(last)) {
      last <- arch_fit_text(paste0(lines[max_lines], " ..."), max_px, size,
                            bold, family = family, italic = italic)
    }
    lines <- c(lines[seq_len(max_lines - 1L)], last)
  }
  lines
}

# Only the symbols that actually appear in a shape annotation. A legend
# for letters the diagram never uses is noise, and the backends declare
# a superset so a shape can be reworded without touching the legend.
# @keywords internal
arch_used_symbols <- function(arch) {
  if (!length(arch$symbols)) return(arch$symbols)
  shapes <- paste(unlist(lapply(arch$stages, function(s)
    c(s$shape, unlist(lapply(s$children, `[[`, "shape"))))), collapse = " ")
  keep <- vapply(names(arch$symbols),
                 function(nm) grepl(paste0("\\b", nm, "\\b"), shapes),
                 logical(1))
  arch$symbols[keep]
}
