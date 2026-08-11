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

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

# Three hues carry identity -- embedding, attention, feed-forward -- and
# everything else is neutral. That is not minimalism for its own sake:
# only the first three slots of the reference categorical palette clear
# the colour-vision-deficiency floors when every pair can appear
# together, which is the case in a diagram where blocks are read against
# each other rather than in sequence. Every block is also labelled in
# words and listed in the legend, so colour never carries meaning alone.
# @keywords internal
arch_theme <- function(theme = c("light", "dark")) {
  theme <- match.arg(theme)
  if (identical(theme, "light")) {
    list(
      name    = "light",
      surface = "#fcfcfb",
      panel   = "#f0efec",
      ink     = "#0b0b0b",
      ink2    = "#52514e",
      ink3    = "#7d7b74",
      rule    = "#dbd9d3",
      tint    = 0.13,
      hue = c(input = "#7d7b74", embed = "#2a78d6", attention = "#eb6834",
              ffn = "#1baf7a", norm = "#a3a199", decode = "#52514e",
              output = "#7d7b74")
    )
  } else {
    list(
      name    = "dark",
      surface = "#1a1a19",
      panel   = "#252523",
      ink     = "#ffffff",
      ink2    = "#c3c2b7",
      ink3    = "#8d8b82",
      rule    = "#3a3a37",
      tint    = 0.22,
      hue = c(input = "#8d8b82", embed = "#3987e5", attention = "#d95926",
              ffn = "#199e70", norm = "#6b6a63", decode = "#c3c2b7",
              output = "#8d8b82")
    )
  }
}

# Mix two hex colours. Used only to derive a block's fill from its accent
# hue, which is a lightness step within one hue -- not a second
# categorical decision.
# @keywords internal
arch_mix <- function(a, b, w) {
  ca <- grDevices::col2rgb(a)[, 1]
  cb <- grDevices::col2rgb(b)[, 1]
  grDevices::rgb(t(round(ca * w + cb * (1 - w))), maxColorValue = 255)
}

# @keywords internal
arch_kind_fill <- function(th, kind) arch_mix(th$hue[[kind]], th$surface, th$tint)

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
# renderers can honour exactly.
# @keywords internal
sc_text <- function(x, y, text, size = 12, col = "#000000", anchor = "start",
                    bold = FALSE, italic = FALSE, mono = FALSE) {
  list(type = "text", x = x, y = y, text = text, size = size, col = col,
       anchor = anchor, bold = bold, italic = italic, mono = mono)
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
# asks for, rounded up so the estimate errs towards truncating.
# @keywords internal
ARCH_CHAR_W <- c(regular = 0.485, bold = 0.515, mono = 0.60)

# @keywords internal
arch_char_w <- function(bold, mono) {
  ARCH_CHAR_W[[if (mono) "mono" else if (bold) "bold" else "regular"]]
}

# @keywords internal
arch_text_width <- function(text, size, bold = FALSE, mono = FALSE) {
  nchar(text, type = "chars") * size * arch_char_w(bold, mono)
}

# @keywords internal
arch_fit_text <- function(text, max_px, size, bold = FALSE, mono = FALSE) {
  if (is.null(text) || !nzchar(text)) return(text)
  if (arch_text_width(text, size, bold, mono) <= max_px) return(text)
  # Three dots rather than U+2026: the PDF device's font encodings have no
  # ellipsis glyph and warn when asked for one, and nothing is gained by
  # making the two outputs differ over it.
  keep <- max(1L, floor(max_px / (size * arch_char_w(bold, mono))) - 3L)
  paste0(substr(text, 1L, keep), "...")
}

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# @keywords internal
arch_geom <- function() {
  list(
    pad      = 30,      # canvas margin
    gutter_l = 152,     # phase brackets and their labels
    gutter_r = 190,     # shapes and parameter counts
    gap      = 16,      # between a column and the block column
    box_w    = 472,
    box_h    = 46,      # a block with a label only
    box_h2   = 60,      # a block with a label and a detail line
    arrow    = 24,      # vertical space between blocks
    deck     = 5,       # offset per card in a repeated block's deck
    child_h  = 32,
    child_gap = 7,
    pad_in   = 13       # padding inside an expanded container
  )
}

# @keywords internal
arch_canvas_width <- function(g) {
  g$pad + g$gutter_l + g$gap + g$box_w + g$gap + g$gutter_r + g$pad
}

# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

# Space above the children inside an expanded block: the stage's own
# label, plus its detail line when it has one.
# @keywords internal
arch_header_height <- function(s) if (is.null(s$detail)) 25 else 43

# Height a stage's block occupies, including its deck offsets or its
# expanded children.
# @keywords internal
arch_stage_height <- function(s, detail, g) {
  expanded <- identical(detail, "full") && length(s$children) > 0L
  if (expanded) {
    n <- length(s$children)
    return(2 * g$pad_in + arch_header_height(s) +
             n * g$child_h + (n - 1L) * g$child_gap)
  }
  base <- if (is.null(s$detail)) g$box_h else g$box_h2
  base + if (s$repeats > 1L) 2 * g$deck else 0
}

#' Lay a described architecture out into a renderer-independent scene
#'
#' @param arch A `tabfound_arch` from [tabfound_architecture()].
#' @param detail `"overview"` or `"full"`.
#' @param theme `"light"` or `"dark"`.
#' @param show_shapes,show_params Draw the right-margin annotations.
#' @param show_facts Draw the fact sheet at the foot of the diagram.
#' @return A list with `width`, `height`, `bg` and `items`.
#' @keywords internal
arch_scene <- function(arch, detail = c("overview", "full"),
                       theme = c("light", "dark"),
                       show_shapes = TRUE, show_params = TRUE,
                       show_facts = TRUE) {
  detail <- match.arg(detail)
  th <- arch_theme(theme)
  g  <- arch_geom()

  W  <- arch_canvas_width(g)
  x_box   <- g$pad + g$gutter_l + g$gap
  x_right <- x_box + g$box_w + g$gap
  x_mid   <- x_box + g$box_w / 2

  it <- list()
  push <- function(...) it[[length(it) + 1L]] <<- list(...)[[1L]]
  push_all <- function(lst) for (e in lst) push(e)

  # --- header ---------------------------------------------------------
  y <- g$pad + 12
  push(sc_text(g$pad, y, arch$title, size = 19, col = th$ink, bold = TRUE))
  if (!is.null(arch$subtitle) && nzchar(arch$subtitle)) {
    y <- y + 22
    push(sc_text(g$pad, y, arch$subtitle, size = 11.5, col = th$ink2))
  }
  y <- y + 20
  push(sc_line(g$pad, y, W - g$pad, y, col = th$rule, lwd = 1))
  y <- y + 26

  # --- blocks ---------------------------------------------------------
  spans <- list()   # phase brackets, resolved once all tops are known
  for (i in seq_along(arch$stages)) {
    s <- arch$stages[[i]]
    h <- arch_stage_height(s, detail, g)
    push_all(arch_block_items(s, x_box, y, h, g, th, detail))

    if (show_shapes || show_params) {
      push_all(arch_margin_items(s, x_right, y, h, g, th,
                                 show_shapes, show_params))
    }
    if (!is.null(s$group)) {
      spans[[length(spans) + 1L]] <- list(group = s$group, top = y,
                                          bottom = y + h)
    }
    if (i < length(arch$stages)) {
      push_all(sc_arrow(x_mid, y + h, y + h + g$arrow, th$ink3))
    }
    y <- y + h + g$arrow
  }
  y <- y - g$arrow

  push_all(arch_bracket_items(spans, x_box, g, th))

  # --- legend ---------------------------------------------------------
  y <- y + 30
  legend <- arch_legend_items(arch, g$pad, y, W - 2 * g$pad, th)
  push_all(legend$items)
  y <- legend$y

  # --- fact sheet -----------------------------------------------------
  if (show_facts && length(arch$facts)) {
    y <- y + 18
    facts <- arch_facts_items(arch, g$pad, y, W - 2 * g$pad, th)
    push_all(facts$items)
    y <- facts$y
  }

  list(width = W, height = ceiling(y + g$pad), bg = th$surface, items = it,
       theme = th)
}

# One block: the deck of cards behind it when it repeats, the card
# itself, its accent edge, its text, and its children when expanded.
# @keywords internal
arch_block_items <- function(s, x, y, h, g, th, detail) {
  out <- list()
  accent <- th$hue[[s$kind]]
  fill   <- arch_kind_fill(th, s$kind)
  expanded <- identical(detail, "full") && length(s$children) > 0L
  stacked  <- s$repeats > 1L && !expanded

  # Card height excludes the deck offsets, which sit below and right.
  card_h <- if (expanded) h else h - if (stacked) 2 * g$deck else 0

  if (stacked) {
    for (k in 2:1) {
      out[[length(out) + 1L]] <- sc_rect(
        x + k * g$deck, y + k * g$deck, g$box_w, card_h,
        fill = arch_mix(fill, th$surface, 0.45), stroke = th$rule, lwd = 1, r = 5
      )
    }
  }

  out[[length(out) + 1L]] <- sc_rect(x, y, g$box_w, card_h, fill = fill,
                                     stroke = accent, lwd = 1.3, r = 5)
  # Accent edge. Drawn as a narrow rect rather than a thick stroke so it
  # keeps its width when the diagram is scaled.
  out[[length(out) + 1L]] <- sc_rect(x, y + 1, 5, card_h - 2, fill = accent,
                                     stroke = NA, r = 2)

  # Right-hand tags -- the attention axis and the repeat count -- claim
  # their width first, so the label and detail lines are truncated to
  # what is actually left rather than run underneath them.
  tags <- c(
    if (!is.null(s$axis)) paste0("over ", s$axis),
    if (s$repeats > 1L) paste0("× ", s$repeats)
  )
  tag_w <- if (!length(tags)) 0 else
    max(vapply(tags, arch_text_width, numeric(1), size = 11, bold = TRUE)) + 16

  tx <- x + 17
  text_w <- g$box_w - 34 - tag_w
  if (expanded) {
    out[[length(out) + 1L]] <- sc_text(tx, y + g$pad_in + 6,
      arch_fit_text(s$label, text_w, 13, bold = TRUE), size = 13,
      col = th$ink, bold = TRUE)
    if (!is.null(s$detail)) {
      out[[length(out) + 1L]] <- sc_text(tx, y + g$pad_in + 25,
        arch_fit_text(s$detail, g$box_w - 34, 10.5), size = 10.5,
        col = th$ink2)
    }
    cy <- y + g$pad_in + arch_header_height(s)
    for (ch in s$children) {
      out <- c(out, arch_child_items(ch, x + g$pad_in, cy,
                                     g$box_w - 2 * g$pad_in, g, th))
      cy <- cy + g$child_h + g$child_gap
    }
  } else {
    cy <- y + card_h / 2
    if (is.null(s$detail)) {
      out[[length(out) + 1L]] <- sc_text(tx, cy,
        arch_fit_text(s$label, text_w, 13, bold = TRUE), size = 13,
        col = th$ink, bold = TRUE)
    } else {
      out[[length(out) + 1L]] <- sc_text(tx, cy - 9,
        arch_fit_text(s$label, text_w, 13, bold = TRUE), size = 13,
        col = th$ink, bold = TRUE)
      out[[length(out) + 1L]] <- sc_text(tx, cy + 10,
        arch_fit_text(s$detail, text_w, 10.5), size = 10.5, col = th$ink2)
    }
  }

  if (length(tags)) {
    rx <- x + g$box_w - 17
    ry <- if (expanded) y + g$pad_in + 6 else y + card_h / 2
    if (length(tags) == 1L) {
      out[[length(out) + 1L]] <- sc_text(rx, ry, tags[1], size = 11,
        col = th$ink2, anchor = "end", bold = is.null(s$axis))
    } else {
      out[[length(out) + 1L]] <- sc_text(rx, ry - 9, tags[1], size = 10,
        col = th$ink3, anchor = "end")
      out[[length(out) + 1L]] <- sc_text(rx, ry + 10, tags[2], size = 11.5,
        col = th$ink2, anchor = "end", bold = TRUE)
    }
  }
  out
}

# A sublayer inside an expanded block.
# @keywords internal
arch_child_items <- function(ch, x, y, w, g, th) {
  accent <- th$hue[[ch$kind]]
  out <- list(
    sc_rect(x, y, w, g$child_h, fill = arch_kind_fill(th, ch$kind),
            stroke = arch_mix(accent, th$surface, 0.45), lwd = 1, r = 4),
    sc_rect(x, y + 1, 4, g$child_h - 2, fill = accent, stroke = NA, r = 2)
  )
  cy <- y + g$child_h / 2
  right <- c(if (!is.null(ch$axis)) paste0("over ", ch$axis),
             if (ch$repeats > 1L) paste0("× ", ch$repeats))
  right <- if (length(right)) paste(right, collapse = " · ") else NULL
  rw <- if (is.null(right)) 0 else arch_text_width(right, 9.5) + 14
  out[[length(out) + 1L]] <- sc_text(
    x + 13, cy, arch_fit_text(ch$label, w - 26 - rw, 11.5), size = 11.5,
    col = th$ink)
  if (!is.null(right)) {
    out[[length(out) + 1L]] <- sc_text(x + w - 11, cy, right, size = 9.5,
                                       col = th$ink3, anchor = "end")
  }
  out
}

# The right margin: symbolic output shape above, parameter count below.
# @keywords internal
arch_margin_items <- function(s, x, y, h, g, th, show_shapes, show_params) {
  lines <- c(
    if (show_shapes && !is.null(s$shape)) s$shape,
    if (show_params && !is.na(s$params) && s$params > 0)
      paste0(arch_format_count(s$params), " params")
  )
  if (!length(lines)) return(list())
  cy <- y + (h - if (s$repeats > 1L) 2 * g$deck else 0) / 2
  if (length(lines) == 1L) {
    return(list(sc_text(x, cy, lines[1], size = 10.5, col = th$ink2,
                        mono = !is.null(s$shape))))
  }
  list(
    sc_text(x, cy - 9, lines[1], size = 10.5, col = th$ink2, mono = TRUE),
    sc_text(x, cy + 10, lines[2], size = 10, col = th$ink3)
  )
}

# Phase brackets in the left margin: a rule spanning the stages that
# share a `group`, with the group's name beside it.
# @keywords internal
arch_bracket_items <- function(spans, x_box, g, th) {
  if (!length(spans)) return(list())
  out <- list()
  i <- 1L
  while (i <= length(spans)) {
    j <- i
    while (j < length(spans) && identical(spans[[j + 1L]]$group, spans[[i]]$group)) {
      j <- j + 1L
    }
    top <- spans[[i]]$top
    bot <- spans[[j]]$bottom
    bx <- x_box - 11
    out <- c(out, list(
      sc_line(bx, top, bx, bot, col = th$rule, lwd = 2),
      sc_line(bx, top, bx + 5, top, col = th$rule, lwd = 2),
      sc_line(bx, bot, bx + 5, bot, col = th$rule, lwd = 2)
    ))
    label <- arch_fit_text(spans[[i]]$group, g$gutter_l - 4, 10.5)
    out[[length(out) + 1L]] <- sc_text(bx - 9, (top + bot) / 2, label,
                                       size = 10.5, col = th$ink3,
                                       anchor = "end")
    i <- j + 1L
  }
  out
}

# @keywords internal
arch_legend_items <- function(arch, x, y, w, th) {
  kinds <- unique(vapply(arch$stages, `[[`, character(1), "kind"))
  for (s in arch$stages) {
    for (ch in s$children) kinds <- union(kinds, ch$kind)
  }
  kinds <- names(ARCH_KINDS)[names(ARCH_KINDS) %in% kinds]

  out <- list()
  cx <- x
  cy <- y + 7
  for (k in kinds) {
    lab <- ARCH_KINDS[[k]]
    need <- 15 + arch_text_width(lab, 10.5) + 20
    if (cx + need > x + w) { cx <- x; cy <- cy + 20 }
    out <- c(out, list(
      sc_rect(cx, cy - 5, 11, 11, fill = arch_kind_fill(th, k),
              stroke = th$hue[[k]], lwd = 1.1, r = 2),
      sc_rect(cx, cy - 4, 3, 9, fill = th$hue[[k]], stroke = NA, r = 1),
      sc_text(cx + 17, cy, lab, size = 10.5, col = th$ink2)
    ))
    cx <- cx + need
  }
  list(items = out, y = cy + 10)
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

# @keywords internal
arch_facts_items <- function(arch, x, y, w, th) {
  facts <- arch$facts
  n <- length(facts)
  ncol <- if (n > 5L) 2L else 1L
  nrow <- ceiling(n / ncol)
  line <- 19
  sym <- arch_used_symbols(arch)
  sym <- if (length(sym))
    paste(paste0(names(sym), " = ", unname(sym)), collapse = "   ·   ")
    else NULL
  h <- 16 + 20 + nrow * line + if (is.null(sym)) 4 else 24

  out <- list(sc_rect(x, y, w, h, fill = th$panel, stroke = NA, r = 6))
  out[[length(out) + 1L]] <- sc_text(x + 16, y + 19, "Fact sheet", size = 10.5,
                                     col = th$ink3, bold = TRUE)
  colw <- (w - 32) / ncol
  for (i in seq_len(n)) {
    col <- (i - 1L) %/% nrow
    row <- (i - 1L) %% nrow
    fx <- x + 16 + col * colw
    fy <- y + 40 + row * line
    out[[length(out) + 1L]] <- sc_text(
      fx, fy, arch_fit_text(names(facts)[i], colw * 0.52, 10.5), size = 10.5,
      col = th$ink2)
    out[[length(out) + 1L]] <- sc_text(
      fx + colw - 16, fy,
      arch_fit_text(as.character(facts[[i]]), colw * 0.46, 10.5, bold = TRUE),
      size = 10.5, col = th$ink, anchor = "end", bold = TRUE)
  }
  if (!is.null(sym)) {
    out[[length(out) + 1L]] <- sc_text(
      x + 16, y + h - 13, arch_fit_text(sym, w - 32, 9.5), size = 9.5,
      col = th$ink3, italic = TRUE)
  }
  list(items = out, y = y + h)
}
