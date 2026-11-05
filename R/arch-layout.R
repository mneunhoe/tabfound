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
  list(cards = arch_scene_cards, tufte = arch_scene_tufte,
       icicle = arch_scene_icicle)
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

# WCAG relative luminance and contrast ratio, for choosing text that
# stays readable on a coloured fill.
# @keywords internal
arch_luminance <- function(hex) {
  v <- grDevices::col2rgb(hex)[, 1] / 255
  v <- ifelse(v <= 0.03928, v / 12.92, ((v + 0.055) / 1.055)^2.4)
  sum(c(0.2126, 0.7152, 0.0722) * v)
}

# @keywords internal
arch_contrast <- function(a, b) {
  l <- sort(c(arch_luminance(a), arch_luminance(b)), decreasing = TRUE)
  (l[1] + 0.05) / (l[2] + 0.05)
}

# Whichever of `dark` and `light` reads better on `fill`.
# @keywords internal
arch_text_on <- function(fill, dark = "#111111", light = "#ffffff") {
  if (arch_contrast(fill, dark) >= arch_contrast(fill, light)) dark else light
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

# ---------------------------------------------------------------------------
# Shared parts for the styles that are not cards
# ---------------------------------------------------------------------------

# The fact sheet as a booktabs table: a rule above, a rule below, no
# fill and no rules between rows. Two columns past five facts. `label_x`
# right-aligns a "Facts" label in the margin when given. Returns the
# items and the y the table ends at.
# @keywords internal
arch_fact_table_items <- function(facts, x, y, w, th, family = "serif",
                                  size = 11.5, label_x = NULL) {
  n <- length(facts)
  if (!n) return(list(items = list(), y = y))
  out <- list()
  add <- function(e) out[[length(out) + 1L]] <<- e
  line <- 19
  ncol <- if (n > 5L) 2L else 1L
  nrow <- ceiling(n / ncol)
  gap <- 18
  colw <- (w - (ncol - 1L) * gap) / ncol

  add(sc_line(x, y, x + w, y, col = th$ink2, lwd = 0.8))
  if (!is.null(label_x)) {
    add(sc_text(label_x, y + 14, "Facts", size = size, col = th$ink2,
                italic = TRUE, anchor = "end", family = family))
  }
  for (k in seq_len(n)) {
    fx <- x + ((k - 1L) %/% nrow) * (colw + gap)
    fy <- y + 14 + ((k - 1L) %% nrow) * line
    name <- names(facts)[k]
    name_w <- arch_text_width(name, size, family = family)
    add(sc_text(fx, fy, name, size = size, col = th$ink2, family = family))
    value <- arch_fit_text(as.character(facts[[k]]), colw - name_w - 12, size,
                           family = family)
    add(sc_text(fx + colw, fy, value, size = size, col = th$ink,
                anchor = "end", family = family))
  }
  y <- y + 14 + nrow * line - 6
  add(sc_line(x, y, x + w, y, col = th$ink2, lwd = 0.8))
  list(items = out, y = y)
}

# "100", "1k", ..., "1B": short labels for powers of ten.
# @keywords internal
arch_decade_label <- function(d) {
  units <- c("", "k", "M", "B", "T")
  i <- pmin(d %/% 3, length(units) - 1L)
  paste0(10^(d - 3 * i), units[i + 1L])
}

# A range-frame log axis: the line runs from the smallest value to the
# largest, not across the whole plot, and ticks sit only on the decades
# the values span. `px` maps a value to x.
# @keywords internal
arch_log_axis_items <- function(values, px, y, th, family = "serif") {
  values <- values[is.finite(values) & values > 0]
  if (!length(values)) return(list())
  out <- list(sc_line(px(min(values)), y, px(max(values)), y, col = th$ink3,
                      lwd = 0.8))
  decades <- arch_axis_decades(values)
  at <- 10^decades
  labels <- arch_decade_label(decades)
  # Every decade gets a tick; when decades are too narrow for their
  # labels to sit apart, only every second (or third) is labelled.
  if (length(at) > 1L) {
    step <- ceiling(34 / (px(at[2]) - px(at[1])))
    labels[(seq_along(labels) - 1L) %% step != 0L] <- ""
  }
  # Values within one decade cross no tick; label the ends instead, so
  # the axis still says what it measures.
  if (!length(at)) {
    at <- range(values)
    labels <- vapply(at, function(v) gsub(" ", "", arch_format_count(v)),
                     character(1))
  }
  for (k in seq_along(at)) {
    xd <- px(at[k])
    out[[length(out) + 1L]] <- sc_line(xd, y, xd, y + 4, col = th$ink3,
                                       lwd = 0.8)
    if (nzchar(labels[k])) {
      out[[length(out) + 1L]] <- sc_text(xd, y + 14, labels[k], size = 10,
                                         col = th$ink3, anchor = "middle",
                                         family = family)
    }
  }
  out
}

# The decades inside the range of `values`: a tick outside the range
# frame would dangle past the end of its line.
# @keywords internal
arch_axis_decades <- function(values) {
  lo <- ceiling(log10(min(values)) - 1e-9)
  hi <- floor(log10(max(values)) + 1e-9)
  if (lo > hi) integer() else seq.int(lo, hi)
}

# A range-frame linear axis from 0 to `max`, ticked at `pretty()` values.
# @keywords internal
arch_linear_axis_items <- function(max, px, y, th, family = "sans") {
  if (!is.finite(max) || max <= 0) return(list())
  out <- list(sc_line(px(0), y, px(max), y, col = th$ink3, lwd = 0.8))
  for (v in pretty(c(0, max), n = 5)) {
    if (v > max) next
    out <- c(out, list(
      sc_line(px(v), y, px(v), y + 4, col = th$ink3, lwd = 0.8),
      sc_text(px(v), y + 15,
              if (v == 0) "0" else gsub(" ", "", arch_format_count(v)),
              size = 10, col = th$ink3, anchor = "middle", family = family)
    ))
  }
  out
}
