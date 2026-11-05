# The "glyphs" style.
#
# What each stage does to the table, drawn as a small table. Every stage
# gets a glyph -- training rows above, test rows below, columns across --
# with the part its attention runs over picked out: a column for
# attention across rows, a row for attention across columns or
# features, a cluster of dots for inducing points. Once a stage has
# merged a row's columns into one vector, later glyphs are a single
# wide column, so the moment a model stops seeing features is visible
# at a glance.
#
# The highlights are the Okabe-Ito colours, which stay distinct under
# the common colour-vision deficiencies, shifted only as far as needed
# to keep 3:1 contrast against both cell fills in each theme. Every
# highlight is also named in the key, so colour never carries meaning
# alone.

# @keywords internal
arch_glyphs_theme <- function(theme = c("light", "dark")) {
  theme <- match.arg(theme)
  if (identical(theme, "light")) {
    list(name = "light", surface = "#fbfbf9", ink = "#141414",
         ink2 = "#55554f", ink3 = "#7d7c75", rule = "#dcdbd5",
         train = "#e4e3dc", test = "#f2f1ec", embed = "#7f7f79",
         rows = "#0072B2", cols = "#D35D00", induce = "#00916A",
         out = "#B16991")
  } else {
    list(name = "dark", surface = "#161618", ink = "#f2f1ea",
         ink2 = "#bdbcb2", ink3 = "#8a897f", rule = "#3a3a37",
         train = "#2c2c30", test = "#222226", embed = "#77777b",
         rows = "#56B4E9", cols = "#E69F00", induce = "#009E73",
         out = "#CC79A7")
  }
}

# @keywords internal
arch_glyphs_geom <- function() {
  list(
    pad = 40,
    x_text = 150,     # stage text, right of the glyph column
    text_w = 420,
    shape_w = 150,
    num_w = 80,
    stage_gap = 22,
    # The glyph: 5 columns of 10 px, 4 training and 2 test rows of 7 px.
    cols = 5, train_rows = 4, test_rows = 2, cell_w = 10, cell_h = 7,
    cell_gap = 2, split = 4, wide_w = 54
  )
}

# The axes a stage's attention runs over, its sublayers' included:
# Mitra's layer has none of its own but attends across rows and across
# features in its children. "cells" is attention down a column, so it is
# drawn as rows.
# @keywords internal
arch_glyph_axes <- function(s) {
  axes <- unique(c(s$axis, unlist(lapply(s$children, `[[`, "axis"))))
  list(rows = any(axes %in% c("rows", "cells")),
       cols = any(axes %in% c("columns", "features")),
       induce = "inducing" %in% axes)
}

# One glyph at (x, y). Returns its items and height.
# @keywords internal
arch_glyph_items <- function(s, x, y, collapsed, g, th) {
  nr <- g$train_rows + g$test_rows
  nc <- if (collapsed) 1L else g$cols
  cw <- if (collapsed) g$wide_w else g$cell_w
  ax <- arch_glyph_axes(s)
  hl_col <- if (collapsed) 1L else 3L
  hl_row <- 2L
  row_y <- function(r) y + (r - 1L) * (g$cell_h + g$cell_gap) +
    if (r > g$train_rows) g$split else 0
  col_x <- function(k) x + (k - 1L) * (cw + g$cell_gap)

  out <- list()
  add <- function(e) out[[length(out) + 1L]] <<- e
  # An embedding outlines every cell -- each becomes a vector -- rather
  # than filling it: a whole glyph in a fill dark enough to read would
  # outweigh everything around it.
  embed <- identical(s$kind, "embed")
  for (r in seq_len(nr)) {
    test <- r > g$train_rows
    for (k in seq_len(nc)) {
      fill <- if (test) th$test else th$train
      hl <- FALSE
      if (ax$rows && k == hl_col) { fill <- th$rows; hl <- TRUE }
      if (ax$cols && !collapsed && r == hl_row) { fill <- th$cols; hl <- TRUE }
      if (s$kind %in% c("decode", "output") && test) { fill <- th$out; hl <- TRUE }
      # The prediction has no training rows left to show.
      if (identical(s$kind, "output") && !test) next
      if (embed && !hl) {
        add(sc_rect(col_x(k) + 0.5, row_y(r) + 0.5, cw - 1, g$cell_h - 1,
                    fill = fill, stroke = th$embed, lwd = 1, r = 1))
      } else {
        add(sc_rect(col_x(k), row_y(r), cw, g$cell_h, fill = fill,
                    stroke = NA, r = 1))
      }
    }
  }
  h <- row_y(nr) + g$cell_h - y

  if (ax$induce) {
    ix <- col_x(nc) + cw + 10
    for (k in 1:3) {
      add(sc_rect(ix, y + 6 + (k - 1L) * 12, 7, 7, fill = th$induce,
                  stroke = NA, r = 3.5))
    }
    add(sc_line(col_x(hl_col) + cw + 1, y + h / 2, ix - 2, y + h / 2,
                col = th$induce, lwd = 1))
  }
  # Attention across rows: test rows read the training rows above them.
  if (ax$rows) {
    axx <- col_x(hl_col) - 5
    add(sc_line(axx, row_y(g$train_rows + 1L) + g$cell_h / 2, axx,
                row_y(1L) + 5, col = th$rows, lwd = 1.2))
    add(sc_poly(c(axx - 3, axx + 3, axx),
                c(row_y(1L) + 6, row_y(1L) + 6, row_y(1L)), fill = th$rows))
  }
  list(items = out, h = h)
}

# @keywords internal
arch_glyphs_tags <- function(s) {
  tags <- c(arch_repeat_tag(s),
            if (!is.null(s$axis)) paste("over", s$axis))
  if (length(tags)) paste(tags, collapse = "  \u00b7  ") else NULL
}

# The glyphs scene builder, registered in `arch_styles()`. With
# `detail = "full"` the sublayers are listed as text under each stage;
# the glyph already shows every axis they attend over.
# @keywords internal
arch_scene_glyphs <- function(arch, detail, theme, show_shapes, show_params,
                              show_facts) {
  th <- arch_glyphs_theme(theme)
  g  <- arch_glyphs_geom()
  x_shape <- g$x_text + g$text_w + 20
  x_num <- x_shape + g$shape_w
  W <- x_num + g$num_w + g$pad
  full <- identical(detail, "full")

  it <- list()
  push <- function(e) it[[length(it) + 1L]] <<- e
  push_all <- function(lst) for (e in lst) push(e)

  y <- g$pad
  push(sc_text(g$pad, y, arch$title, size = 19, col = th$ink, bold = TRUE))
  if (!is.null(arch$subtitle) && nzchar(arch$subtitle)) {
    y <- y + 22
    push(sc_text(g$pad, y, arch$subtitle, size = 11.5, col = th$ink2))
  }
  y <- y + 20
  push(sc_line(g$pad, y, W - g$pad, y, col = th$rule, lwd = 1))
  y <- y + 26

  collapsed <- FALSE
  for (s in arch$stages) {
    gl <- arch_glyph_items(s, g$pad, y, collapsed, g, th)
    push_all(gl$items)

    tags <- arch_glyphs_tags(s)
    tag_w <- if (is.null(tags)) 0 else arch_text_width(tags, 11.5) + 12
    push(sc_text(g$x_text, y + 7,
                 arch_fit_text(s$label, g$text_w - tag_w, 13.5, bold = TRUE),
                 size = 13.5, col = th$ink, bold = TRUE))
    if (!is.null(tags)) {
      push(sc_text(g$x_text + g$text_w, y + 7, tags, size = 11.5,
                   col = th$ink2, anchor = "end"))
    }
    ty <- y + 7
    for (d in arch_wrap_text(s$detail, g$text_w, 11, max_lines = 2)) {
      ty <- ty + 16
      push(sc_text(g$x_text, ty, d, size = 11, col = th$ink2))
    }
    if (full) {
      for (ch in s$children) {
        ty <- ty + 16
        ctags <- arch_glyphs_tags(ch)
        ctag_w <- if (is.null(ctags)) 0 else arch_text_width(ctags, 10.5) + 12
        push(sc_text(g$x_text + 12, ty,
                     arch_fit_text(ch$label, g$text_w - 12 - ctag_w, 11),
                     size = 11, col = th$ink2))
        if (!is.null(ctags)) {
          push(sc_text(g$x_text + g$text_w, ty, ctags, size = 10.5,
                       col = th$ink3, anchor = "end"))
        }
      }
    }
    if (show_shapes && !is.null(s$shape)) {
      push(sc_text(x_shape, y + 7, s$shape, size = 11, col = th$ink2,
                   family = "mono"))
    }
    if (show_params && !is.na(s$params) && s$params > 0) {
      push(sc_text(x_num, y + 7, arch_format_count(s$params), size = 12,
                   col = th$ink))
    }
    y <- y + max(gl$h, ty - y + 9) + g$stage_gap
    # The collapsing stage still takes the table in; the glyphs after it
    # see one vector per row.
    if (isTRUE(s$collapses)) collapsed <- TRUE
  }
  y <- y - g$stage_gap

  if (show_facts && length(arch$facts)) {
    y <- y + 34
    facts <- arch_fact_table_items(arch$facts, g$x_text, y,
                                   W - g$pad - g$x_text, th, family = "sans",
                                   size = 11, label_x = g$x_text - 12)
    push_all(facts$items)
    y <- facts$y
  }

  notes <- arch_glyphs_notes(arch, detail, show_shapes, show_params, th)
  groups_key <- arch_key_groups(arch, notes, show_shapes = show_shapes)
  key <- arch_key_items(groups_key, g$pad, y + 30, W - 2 * g$pad, th)
  push_all(key$items)
  y <- key$y

  list(width = W, height = ceiling(y + g$pad), bg = th$surface, items = it,
       theme = th, title = arch$title, key = groups_key,
       desc = paste(c(arch$subtitle, arch_key_text(groups_key)),
                    collapse = ". "))
}

# The notation glyphs draws, and only what this diagram actually uses.
# @keywords internal
arch_glyphs_notes <- function(arch, detail, show_shapes, show_params, th) {
  st <- arch$stages
  ax <- lapply(st, arch_glyph_axes)
  any_of <- function(f) any(vapply(st, f, logical(1)))
  kinds <- vapply(st, `[[`, character(1), "kind")
  full <- identical(detail, "full")
  axes <- unlist(lapply(st, function(s)
    c(s$axis, if (full) unlist(lapply(s$children, `[[`, "axis")))))
  has_shape <- show_shapes && any_of(function(s) !is.null(s$shape))
  margin <- c(if (has_shape) "the tensor shape each stage emits",
              if (show_params) "its exact parameter count")
  c(
    list(arch_note(
      "each table: training rows above, test rows below, columns across",
      th$train)),
    if (any(vapply(ax, `[[`, logical(1), "rows"))) list(arch_note(paste(
      "attention across rows, within a column; the arrow points from the",
      "test rows to the training rows they read"), th$rows)),
    if (any(vapply(ax, `[[`, logical(1), "cols"))) list(arch_note(
      "attention across columns or features, within a row", th$cols)),
    if (any(vapply(ax, `[[`, logical(1), "induce"))) list(arch_note(
      "inducing points: a small learned set that summarises the rows",
      th$induce)),
    if ("embed" %in% kinds) list(arch_note(
      "outlined cells: an embedding, where every cell becomes a vector",
      th$embed, outline = TRUE)),
    if (any(kinds %in% c("decode", "output"))) list(arch_note(
      "the prediction for the test rows", th$out)),
    if (any_of(function(s) isTRUE(s$collapses))) list(arch_note(paste(
      "one wide column: after this point the table's columns are merged",
      "into one vector per row"))),
    if (any_of(function(s) s$repeats > 1L)) list(arch_note(
      "\u00d7 N: the block is stacked N times")),
    arch_interleave_notes(arch),
    if (length(axes)) list(arch_note(
      "over rows / columns / features: the axis this block's attention runs across")),
    if (length(margin)) list(arch_note(paste0(
      "right columns: ", paste(margin, collapse = ", then "))))
  )
}
