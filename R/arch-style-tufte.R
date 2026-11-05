# The "tufte" style.
#
# The same description as a table with a dot plot beside it, after
# Tufte's rules for statistical graphics: as little non-data ink as the
# reading needs, labels set directly beside what they name rather than
# in a legend, and one quantity -- parameters -- shown by position on a
# common scale, which Cleveland and McGill found people read more
# accurately than length or area.
#
# * Each stage is a row: its name, attention axis and repeat count, its
#   detail line, its output shape, and a dot for its parameter count.
# * The parameter axis is log10 from 100 to 1 B in every diagram, so the
#   dots of two models can be compared by eye. A model with a stage
#   outside that range widens it, and the key then stops claiming the
#   scale is shared.
# * Phases are coloured hairlines in the left margin, labelled directly.
#   The colour runs through viridis in stage order, over the part of the
#   map that keeps 3:1 contrast against the surface.
# * The axis is a range frame: it runs from the smallest dot to the
#   largest, so its length is itself a reading of the spread.

# @keywords internal
arch_tufte_theme <- function(theme = c("light", "dark")) {
  theme <- match.arg(theme)
  if (identical(theme, "light")) {
    list(name = "light", surface = "#fffff8", ink = "#111111",
         ink2 = "#4a4a46", ink3 = "#85847c", rule = "#d9d7cc",
         guide = "#ecebe2", viridis = c(0, 0.55))
  } else {
    list(name = "dark", surface = "#151513", ink = "#f2f1ea",
         ink2 = "#bdbcb2", ink3 = "#8a897f", rule = "#3b3b36",
         guide = "#262622", viridis = c(0.35, 1))
  }
}

# @keywords internal
arch_tufte_geom <- function() {
  list(
    pad = 32,         # canvas margin
    phase_w = 118,    # phase labels, right-aligned against their rule
    gap = 18,         # between columns
    text_w = 330,     # stage name, tags and detail
    shape_w = 128,    # output shape
    plot_w = 168,     # the dot plot
    num_w = 92,       # parameter count and share
    row_gap = 14,     # between stages
    phase_gap = 14,   # extra space where the phase changes
    lo = 2, hi = 9    # the shared log10 domain: 100 to 1 B parameters
  )
}

# The phase colours: viridis in stage order, or, with a highlight, shades
# of it from full strength to light. Either way each keeps 3:1 against
# the background, which a viridis slice chosen for the theme's own
# surface might not on another.
# @keywords internal
arch_tufte_phase_colours <- function(k, th, colours) {
  if (!k) return(character())
  cols <- if (is.null(colours$highlight)) {
    arch_viridis(k, th$viridis[1], th$viridis[2])
  } else {
    w <- if (k == 1L) 1 else seq(1, 0.45, length.out = k)
    vapply(w, function(x) arch_mix(colours$highlight, th$surface, x),
           character(1))
  }
  if (is.null(colours$background) && is.null(colours$highlight)) return(cols)
  vapply(cols, arch_readable, character(1), th$surface, 3, USE.NAMES = FALSE)
}

# @keywords internal
arch_tufte_tags <- function(s) {
  tags <- c(arch_repeat_tag(s),
            if (!is.null(s$axis)) paste0("over ", s$axis))
  if (length(tags)) paste(tags, collapse = "  ") else NULL
}

# The tufte scene builder, registered in `arch_styles()`.
# @keywords internal
arch_scene_tufte <- function(arch, detail, theme, show_shapes, show_params,
                             show_facts, colours = list()) {
  th <- arch_adapt_theme(arch_tufte_theme(theme), colours$background)
  g  <- arch_tufte_geom()
  serif <- "serif"

  x_phase <- g$pad
  x_text  <- x_phase + g$phase_w + g$gap
  x_shape <- x_text + g$text_w + g$gap
  x_plot  <- x_shape + g$shape_w + g$gap
  x_num   <- x_plot + g$plot_w + 10
  W <- x_num + g$num_w + g$pad

  params <- vapply(arch$stages, function(s)
    if (is.na(s$params)) 0 else s$params, numeric(1))
  total <- sum(params)
  kids <- function(s) if (identical(detail, "full")) s$children else list()
  child_params <- unlist(lapply(arch$stages, function(s)
    vapply(kids(s), function(ch) if (is.na(ch$params)) 0 else ch$params,
           numeric(1))))
  drawn <- c(params, child_params)
  drawn <- drawn[drawn > 0]
  # The domain is shared by every diagram unless a value falls outside it.
  lo <- min(g$lo, if (length(drawn)) floor(log10(min(drawn))) else g$lo)
  hi <- max(g$hi, if (length(drawn)) ceiling(log10(max(drawn))) else g$hi)
  shared_scale <- lo == g$lo && hi == g$hi
  px <- function(n) x_plot + (log10(n) - lo) / (hi - lo) * g$plot_w

  groups <- unique(unlist(lapply(arch$stages, `[[`, "group")))
  phase_col <- stats::setNames(
    arch_tufte_phase_colours(length(groups), th, colours), groups)
  col_of <- function(s) if (is.null(s$group)) th$ink3 else phase_col[[s$group]]

  it <- list()
  push <- function(e) it[[length(it) + 1L]] <<- e
  push_all <- function(lst) for (e in lst) push(e)

  # --- title: no rule, whitespace does the separating -------------------
  y <- g$pad + 10
  push(sc_text(g$pad, y, arch$title, size = 22, col = th$ink, family = serif))
  if (!is.null(arch$subtitle) && nzchar(arch$subtitle)) {
    y <- y + 24
    push(sc_text(g$pad, y, arch$subtitle, size = 13, col = th$ink2,
                 italic = TRUE, family = serif))
  }
  y <- y + 40

  # --- column heads, small and grey ---------------------------------------
  if (show_shapes) {
    push(sc_text(x_shape, y, "output shape", size = 10.5, col = th$ink3,
                 italic = TRUE, family = serif))
  }
  if (show_params) {
    push(sc_text(x_plot, y, "parameters, log scale", size = 10.5,
                 col = th$ink3, italic = TRUE, family = serif))
  }
  y <- y + 22

  # --- stages -------------------------------------------------------------
  spans <- list()
  prev_group <- NULL
  for (i in seq_along(arch$stages)) {
    s <- arch$stages[[i]]
    grp <- s$group %||% ""
    if (i > 1L && !identical(grp, prev_group)) y <- y + g$phase_gap
    prev_group <- grp

    top <- y
    ly <- y + 8
    tags <- arch_tufte_tags(s)
    tag_w <- if (is.null(tags)) 0 else
      arch_text_width(tags, 12, family = serif, italic = TRUE) + 12
    push(sc_text(x_text, ly, arch_fit_text(s$label, g$text_w - tag_w, 14,
                                           family = serif),
                 size = 14, col = th$ink, family = serif))
    if (!is.null(tags)) {
      push(sc_text(x_text + g$text_w, ly, tags, size = 12, col = th$ink2,
                   italic = TRUE, anchor = "end", family = serif))
    }
    for (d in arch_wrap_text(s$detail, g$text_w, 11.5, family = serif,
                             max_lines = 3)) {
      ly <- ly + 16
      push(sc_text(x_text, ly, d, size = 11.5, col = th$ink2, family = serif))
    }

    for (ch in kids(s)) {
      ly <- ly + 18
      ktags <- arch_tufte_tags(ch)
      ktag_w <- if (is.null(ktags)) 0 else
        arch_text_width(ktags, 10.5, family = serif, italic = TRUE) + 12
      push(sc_text(x_text + 14, ly,
                   arch_fit_text(ch$label, g$text_w - 14 - ktag_w, 11.5,
                                 family = serif),
                   size = 11.5, col = th$ink2, family = serif))
      if (!is.null(ktags)) {
        push(sc_text(x_text + g$text_w, ly, ktags, size = 10.5, col = th$ink3,
                     italic = TRUE, anchor = "end", family = serif))
      }
      if (show_params && !is.na(ch$params) && ch$params > 0) {
        push(sc_line(x_plot, ly, px(ch$params), ly, col = th$guide, lwd = 0.6))
        push(sc_rect(px(ch$params) - 2, ly - 2, 4, 4, fill = th$surface,
                     stroke = col_of(s), lwd = 1, r = 2))
        push(sc_text(x_num, ly, paste0(arch_format_count(ch$params),
                                       if (s$repeats > 1L) " / block"),
                     size = 10.5, col = th$ink3, family = serif))
      }
    }
    bottom <- ly + 8

    if (show_shapes && !is.null(s$shape)) {
      push(sc_text(x_shape, y + 8, s$shape, size = 11, col = th$ink2,
                   family = "mono"))
    }
    if (show_params && params[i] > 0) {
      dy <- y + 8
      push(sc_line(x_plot, dy, px(params[i]), dy, col = th$guide, lwd = 0.8))
      push(sc_rect(px(params[i]) - 3.5, dy - 3.5, 7, 7, fill = col_of(s),
                   stroke = NA, r = 3.5))
      push(sc_text(x_num, dy, arch_format_count(params[i]), size = 12,
                   col = th$ink, family = serif))
      share <- params[i] / total
      if (share >= 0.01) {
        push(sc_text(x_num + 52, dy, sprintf("%.0f%%", 100 * share),
                     size = 10.5, col = th$ink3, family = serif))
      }
    }
    if (nzchar(grp)) {
      spans[[length(spans) + 1L]] <- list(group = grp, top = top,
                                          bottom = bottom)
    }
    y <- bottom + g$row_gap
  }
  y <- y - g$row_gap

  push_all(arch_tufte_phase_items(spans, phase_col, x_text, g, th))

  # --- the axis -----------------------------------------------------------
  if (show_params && total > 0) {
    y <- y + 20
    # The range frame spans every dot drawn, sublayers' included.
    push_all(arch_log_axis_items(drawn, px, y, th, family = serif))
    push(sc_text(x_num, y, paste(arch_format_count(total), "total"),
                 size = 10.5, col = th$ink2, family = serif))
    y <- y + 16
  }

  # --- facts --------------------------------------------------------------
  if (show_facts && length(arch$facts)) {
    y <- y + 26
    facts <- arch_fact_table_items(arch$facts, x_text, y, W - g$pad - x_text,
                                   th, family = serif,
                                   label_x = x_phase + g$phase_w)
    push_all(facts$items)
    y <- facts$y
  }

  # --- key ----------------------------------------------------------------
  notes <- arch_tufte_notes(arch, detail, show_shapes, show_params,
                            shared_scale)
  groups_key <- arch_key_groups(arch, notes, show_shapes = show_shapes)
  # Whitespace, not a second rule, separates it from the fact table.
  key <- arch_key_items(groups_key, x_text, y + 22, W - g$pad - x_text, th,
                        family = serif, rule = FALSE)
  push_all(key$items)
  y <- key$y

  list(width = W, height = ceiling(y + g$pad), bg = th$surface, items = it,
       theme = th, title = arch$title,
       key = groups_key,
       desc = paste(c(arch$subtitle, arch_key_text(groups_key)),
                    collapse = ". "))
}

# One coloured hairline per run of stages in the same phase, with the
# phase's name set beside it in the margin.
# @keywords internal
arch_tufte_phase_items <- function(spans, phase_col, x_text, g, th) {
  out <- list()
  i <- 1L
  while (i <= length(spans)) {
    j <- i
    while (j < length(spans) &&
           identical(spans[[j + 1L]]$group, spans[[i]]$group)) {
      j <- j + 1L
    }
    top <- spans[[i]]$top
    bot <- spans[[j]]$bottom
    bx <- x_text - 10
    out[[length(out) + 1L]] <- sc_line(bx, top + 1, bx, bot - 1,
                                       col = phase_col[[spans[[i]]$group]],
                                       lwd = 2.5)
    label <- arch_wrap_text(spans[[i]]$group, g$phase_w - 6, 11.5,
                            family = "serif", italic = TRUE, max_lines = 2)
    for (k in seq_along(label)) {
      out[[length(out) + 1L]] <- sc_text(
        bx - 8, top + 8 + (k - 1L) * 15, label[k], size = 11.5, col = th$ink2,
        italic = TRUE, anchor = "end", family = "serif")
    }
    i <- j + 1L
  }
  out
}

# The notation tufte draws, and only what this diagram actually uses.
# @keywords internal
arch_tufte_notes <- function(arch, detail, show_shapes, show_params,
                             shared_scale) {
  full <- identical(detail, "full")
  axes <- unlist(lapply(arch$stages, function(s)
    c(s$axis, if (full) unlist(lapply(s$children, `[[`, "axis")))))
  repeats <- any(vapply(arch$stages, `[[`, integer(1), "repeats") > 1L)
  has_shape <- show_shapes &&
    any(vapply(arch$stages, function(s) !is.null(s$shape), logical(1)))
  has_groups <- any(vapply(arch$stages, function(s) !is.null(s$group),
                           logical(1)))
  has_children <- full && show_params &&
    any(vapply(arch$stages, function(s) length(s$children) > 0L, logical(1)))
  c(
    if (repeats) list(arch_note("\u00d7 N: the block is stacked N times")),
    arch_interleave_notes(arch),
    if (length(axes)) list(arch_note(
      "over rows / columns / features: the axis this block's attention runs across")),
    if (show_params) list(arch_note(paste0(
      "dot: the parameters in that stage, on a log scale",
      if (shared_scale) " shared by every tabfound diagram",
      "; the axis runs from the smallest dot to the largest; % is the stage's share of the total"))),
    if (has_children) list(arch_note(
      "hollow dot: one sublayer's parameters; \"/ block\" counts one copy of a stacked block")),
    if (has_groups) list(arch_note(
      "margin colour: the phase, from first (dark) to last (light)")),
    if (has_shape) list(arch_note(
      "output shape: the tensor each stage emits, in the symbols above"))
  )
}
