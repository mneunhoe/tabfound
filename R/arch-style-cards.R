# The "cards" style: the default architecture diagram.
#
# One rounded card per stage, stacked top to bottom with flow arrows,
# coloured by what the stage does. Repeated stages are drawn as a deck
# with their count, phases are bracketed in the left margin, and the
# output shape and parameter count sit in the right margin. The shared
# scene primitives and text metrics are in `arch-layout.R`.

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


# @keywords internal
arch_kind_fill <- function(th, kind) arch_mix(th$hue[[kind]], th$surface, th$tint)

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

# The theme with your own colours: the background carries the neutrals
# with it; the highlight is the attention colour. On another background
# every hue is kept at 3:1, so no kind of card fades into it.
# @keywords internal
arch_cards_colours <- function(th, colours) {
  if (!is.null(colours$background)) {
    th <- arch_adapt_theme(th, colours$background)
    th$hue <- vapply(th$hue, arch_readable, character(1), th$surface, 3)
  }
  if (!is.null(colours$highlight)) {
    th$hue[["attention"]] <- arch_readable(colours$highlight, th$surface, 3)
  }
  th
}

# The cards scene builder, registered in `arch_styles()`. `detail` and
# `theme` arrive already matched.
# @keywords internal
arch_scene_cards <- function(arch, detail, theme, show_shapes, show_params,
                             show_facts, colours = list()) {
  th <- arch_cards_colours(arch_theme(theme), colours)
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

  # --- key --------------------------------------------------------------
  groups <- arch_key_groups(
    arch, arch_cards_notes(arch, detail, show_shapes, show_params),
    show_shapes = show_shapes, extra = unname(ARCH_KINDS)
  )
  key <- arch_key_items(groups, g$pad, y + 26, W - 2 * g$pad, th)
  push_all(key$items)
  y <- key$y

  list(width = W, height = ceiling(y + g$pad), bg = th$surface, items = it,
       theme = th, title = arch$title,
       key = groups,
       desc = paste(c(arch$subtitle, arch_key_text(groups)), collapse = ". "))
}

# The notation cards draws, and only what this diagram actually uses.
# @keywords internal
arch_cards_notes <- function(arch, detail, show_shapes, show_params) {
  axes <- unlist(lapply(arch$stages, function(s)
    c(s$axis, if (identical(detail, "full"))
      unlist(lapply(s$children, `[[`, "axis")))))
  repeats <- any(vapply(arch$stages, `[[`, integer(1), "repeats") > 1L)
  has_shape <- show_shapes &&
    any(vapply(arch$stages, function(s) !is.null(s$shape), logical(1)))
  margin <- c(if (has_shape) "the tensor shape each stage emits",
              if (show_params) "its exact parameter count")
  c(
    if (repeats) list(arch_note(if (identical(detail, "full"))
      "\u00d7 N: the block is stacked N times; its sublayers are drawn once"
      else "\u00d7 N: the block is stacked N times; the card is drawn once, as a deck")),
    arch_interleave_notes(arch),
    if (length(axes)) list(arch_note(
      "over rows / columns / features: the axis this block's attention runs across")),
    list(arch_note("card colour: what kind of stage it is (see the legend above)")),
    if (length(margin)) list(arch_note(paste0(
      "right margin: ", paste(margin, collapse = ", then ")))),
    if (any(vapply(arch$stages, function(s) !is.null(s$group), logical(1))))
      list(arch_note("left margin: the phase each group of stages belongs to"))
  )
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
    arch_repeat_tag(s)
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
             if (ch$repeats > 1L) paste0("\u00d7 ", ch$repeats))
  right <- if (length(right)) paste(right, collapse = " \u00b7 ") else NULL
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


# @keywords internal
arch_facts_items <- function(arch, x, y, w, th) {
  facts <- arch$facts
  n <- length(facts)
  ncol <- if (n > 5L) 2L else 1L
  nrow <- ceiling(n / ncol)
  line <- 19
  # The symbols used to sit on a line at the foot of this panel, cut to
  # its width; they are in the key now, which never truncates.
  h <- 16 + 20 + nrow * line + 4

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
  list(items = out, y = y + h)
}
