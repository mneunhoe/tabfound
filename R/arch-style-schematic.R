# The "schematic" style.
#
# How each block runs, in the convention of the papers: the data flows
# upwards from the input table to the prediction, every repeated block
# is opened up into its steps in the order they run, and the residual
# connections are drawn -- a line up the right-hand side that rejoins the
# stream at a circled plus. It follows the original Transformer figure
# and Sebastian Raschka's LLM Architecture Gallery, drawn from this
# package's own descriptions.
#
# * Repeated blocks sit in a container tinted with the model's accent
#   colour, marked "N x". Interleaved blocks share one container.
# * What feeds a step without being a step -- a RoPE table, inducing
#   points, CLS tokens, QK-norm -- is a small box at the left with an
#   arrow into it. A stage that joins the stream (a target encoder, CLS
#   tokens) is a box at the left whose arrow meets the stream at a
#   circled plus (added) or a pill (concatenated or prepended).
# * Attention steps are dark, with a line saying where their keys come
#   from and how they differ from plain attention.
# * Facts point at the part they describe, with dotted leaders and the
#   numbers in the accent colour.
#
# Layout is computed with u, height above the foot of the flow, and the
# items are turned the right way up at the end: the flow is easiest to
# build in the order it runs.

# @keywords internal
arch_schematic_theme <- function(theme = c("light", "dark")) {
  theme <- match.arg(theme)
  if (identical(theme, "light")) {
    list(name = "light", surface = "#ffffff", outer = "#f2f2f0",
         outer_stroke = "#d6d6d1", box = "#ffffff", box_stroke = "#3a3a38",
         ink = "#111111", ink2 = "#4d4d4a", ink3 = "#6b6b66",
         line = "#333333", attn = "#3a3a3a", attn_ink = "#ffffff",
         rule = "#d6d6d1")
  } else {
    list(name = "dark", surface = "#161618", outer = "#212123",
         outer_stroke = "#47474a", box = "#2b2b2e", box_stroke = "#c9c9c4",
         ink = "#f2f1ea", ink2 = "#c2c1b8", ink3 = "#9a998f",
         line = "#d0d0cb", attn = "#e4e3dd", attn_ink = "#111111",
         rule = "#3a3a37")
  }
}

# One accent per backend, so a model is always the same colour. These are
# starting hues; `arch_readable()` moves each just far enough to keep
# 4.5:1 as text on whatever it is drawn on.
ARCH_ACCENTS <- c(
  tabpfn = "#0b5fad", tabpfn26 = "#3f51b5", tabpfn3 = "#00796b",
  tabpfn35 = "#2e7d32", tabicl = "#ad1457", tabfm = "#c43e00",
  mitra = "#6a1b9a", kumo = "#7a5c00"
)

# `col` mixed towards black or white, whichever reads better on `bg`,
# just until it reaches `ratio` against it.
# @keywords internal
arch_readable <- function(col, bg, ratio = 4.5) {
  if (arch_contrast(col, bg) >= ratio) return(col)
  toward <- if (arch_luminance(bg) > 0.18) "#000000" else "#ffffff"
  for (w in seq(0.98, 0, by = -0.02)) {
    c2 <- arch_mix(col, toward, w)
    if (arch_contrast(c2, bg) >= ratio) return(c2)
  }
  toward
}

# @keywords internal
arch_schematic_geom <- function() {
  list(
    width = 940, pad = 36,
    side_x = 36, side_w = 176,    # side inputs, left of the stream
    xc = 420,                     # the stream's centre line
    box_w = 280, box_h = 30,      # a step
    gap = 22,                     # an arrow between steps
    skip = 26,                    # the residual line, right of the boxes
    plus_r = 8,                   # the circled plus
    cpad = 14,                    # inside a container
    call_x = 700                  # callouts
  )
}

# The line under an attention step's name: what makes it differ from
# plain attention over every token.
# @keywords internal
arch_attn_line <- function(a) {
  if (is.null(a)) return(NULL)
  bits <- c(
    if (!identical(a$keys, "all")) paste("keys:", a$keys),
    if (!is.null(a$queries)) paste("last block queries:", a$queries),
    if (!identical(a$scaling, "1/sqrt(d)")) a$scaling,
    if (!is.null(a$kv_heads_test))
      if (a$kv_heads_test == 1L) "test rows: 1 KV head (MQA)"
      else sprintf("test rows: %d KV heads (GQA)", a$kv_heads_test)
  )
  if (length(bits)) paste(bits, collapse = " \u00b7 ") else NULL
}

# The schematic scene builder, registered in `arch_styles()`.
# @keywords internal
arch_scene_schematic <- function(arch, detail, theme, show_shapes,
                                 show_params, show_facts) {
  th <- arch_schematic_theme(theme)
  g  <- arch_schematic_geom()
  W  <- g$width
  accent0 <- ARCH_ACCENTS[[arch$backend]] %||% "#0b5fad"
  accent <- arch_readable(if (identical(theme, "dark"))
                            arch_mix(accent0, "#ffffff", 0.6) else accent0,
                          th$surface)
  tint <- arch_mix(accent, th$outer, if (identical(theme, "dark")) 0.22 else 0.14)
  attn_accent <- arch_readable(accent, th$attn)

  xc <- g$xc
  bx <- xc - g$box_w / 2
  x_skip <- bx + g$box_w + g$skip
  c_left <- bx - g$cpad - 10
  c_right <- x_skip + 16

  # --- emitters in u-space: u is height above the foot, y = -u ---------
  flow <- list()
  under <- list()           # containers, drawn beneath the flow
  emit <- function(e) flow[[length(flow) + 1L]] <<- e
  emit_under <- function(e) under[[length(under) + 1L]] <<- e
  rect_u <- function(x, u0, w, h, ...) sc_rect(x, -(u0 + h), w, h, ...)
  text_u <- function(x, u, txt, ...) sc_text(x, -u, txt, ...)
  line_u <- function(x1, u1, x2, u2, ...) sc_line(x1, -u1, x2, -u2, ...)
  arrow_up <- function(x, u1, u2) {
    emit(line_u(x, u1, x, u2 - 5, col = th$line, lwd = 1.2))
    emit(sc_poly(c(x - 4, x + 4, x), -c(u2 - 6, u2 - 6, u2), fill = th$line))
  }
  arrow_right <- function(x1, x2, u) {
    emit(line_u(x1, u, x2 - 5, u, col = th$line, lwd = 1.1))
    emit(sc_poly(c(x2 - 6, x2 - 6, x2), -c(u - 4, u + 4, u), fill = th$line))
  }
  arrow_left <- function(x1, x2, u) {
    emit(line_u(x1, u, x2 + 5, u, col = th$line, lwd = 1.1))
    emit(sc_poly(c(x2 + 6, x2 + 6, x2), -c(u - 4, u + 4, u), fill = th$line))
  }
  plus_u <- function(u) {
    r <- g$plus_r
    emit(rect_u(xc - r, u - r, 2 * r, 2 * r, fill = th$box,
                stroke = th$line, lwd = 1.2, r = r))
    emit(line_u(xc - r + 3, u, xc + r - 3, u, col = th$line, lwd = 1.2))
    emit(line_u(xc, u - r + 3, xc, u + r - 3, col = th$line, lwd = 1.2))
  }
  pill_u <- function(u, txt) {
    w <- arch_text_width(txt, 10, bold = TRUE) + 14
    emit(rect_u(xc - w / 2, u - 9, w, 18, fill = th$box, stroke = th$line,
                lwd = 1.1, r = 9))
    emit(text_u(xc, u, txt, size = 10, col = th$ink, bold = TRUE,
                anchor = "middle"))
  }

  anchors <- list()         # where callouts can point, by stage/child
  side <- list()            # side inputs, placed once the flow is known
  shapes <- list()          # shape labels on the arrows
  drawn_extra <- character()  # text the description does not hold
  any_unit <- FALSE

  # A step box (plain or attention) with its bottom at u; returns its
  # height.
  step_box <- function(ch, u, stage_id) {
    is_attn <- identical(ch$kind, "attention")
    sub <- if (is_attn) arch_attn_line(ch$attn) else NULL
    sub_lines <- if (is.null(sub)) character()
                 else arch_wrap_text(sub, g$box_w - 20, 10)
    h <- if (length(sub_lines)) g$box_h + 4 + 13 * length(sub_lines) else g$box_h
    fill <- if (is_attn) th$attn else th$box
    ink <- if (is_attn) th$attn_ink else th$ink
    emit(rect_u(bx, u, g$box_w, h, fill = fill,
                stroke = if (is_attn) NA else th$box_stroke, lwd = 1, r = 6))
    top_text <- u + h - 15
    emit(text_u(xc, top_text, arch_fit_text(ch$label, g$box_w - 20, 11.5),
                size = 11.5, col = ink, anchor = "middle"))
    for (k in seq_along(sub_lines)) {
      emit(text_u(xc, top_text - 4 - 13 * k, sub_lines[k], size = 10,
                  col = attn_accent, anchor = "middle"))
    }
    drawn_extra <<- c(drawn_extra, sub_lines)
    anchors[[paste(stage_id, ch$id)]] <<- list(x = bx + g$box_w, u = u + h / 2)
    # QK-norm is drawn the way the papers do: a box feeding the attention.
    if (is_attn && isTRUE(ch$attn$qk_norm)) {
      side[[length(side) + 1L]] <<- list(label = "QK-norm", u = u + h / 2,
                                         x_to = bx)
      drawn_extra <<- c(drawn_extra, "QK-norm")
    }
    h
  }

  # A stage drawn as a plain box on the stream.
  plain_box <- function(s, u) {
    h <- g$box_h + if (is.null(s$detail)) 0 else 0
    emit(rect_u(bx, u, g$box_w, h, fill = th$box, stroke = th$box_stroke,
                lwd = 1, r = if (s$kind %in% c("input", "output")) 0 else 6))
    emit(text_u(xc, u + h / 2, arch_fit_text(s$label, g$box_w - 20, 12),
                size = 12, col = th$ink, anchor = "middle"))
    right <- bx + g$box_w
    if (show_params && !is.na(s$params) && s$params > 0) {
      count <- arch_format_count(s$params)
      emit(text_u(right + 8, u + h / 2, count, size = 10, col = th$ink3))
      right <- right + 8 + arch_text_width(count, 10)
    }
    anchors[[s$id]] <<- list(x = right, u = u + h / 2)
    h
  }

  # A run of stages that repeat together, in one tinted container.
  container <- function(stages, u0) {
    reps <- stages[[1L]]$repeats
    u <- u0
    for (si in seq_along(stages)) {
      s <- stages[[si]]
      if (length(stages) > 1L && si > 1L) {
        # Interleaved: each later stage's steps under a small heading.
        emit(text_u(c_left + 10, u + 10, s$label, size = 10.5, col = th$ink2,
                    italic = TRUE))
      }
      steps <- Filter(function(ch) identical(ch$role, "step"), s$children)
      inputs <- Filter(function(ch) identical(ch$role, "input"), s$children)
      at <- list()
      i <- 1L
      while (i <= length(steps)) {
        unit <- steps[[i]]$residual %||% NA_integer_
        run <- i
        if (!is.na(unit)) {
          while (run < length(steps) &&
                 identical(steps[[run + 1L]]$residual, unit)) run <- run + 1L
        }
        # Every group is entered by an arrow; its residual line leaves
        # the flow halfway up it.
        arrow_up(xc, u, u + g$gap)
        u_split <- u + g$gap / 2
        u <- u + g$gap
        for (k in i:run) {
          if (k > i) {
            arrow_up(xc, u, u + 14)
            u <- u + 14
          }
          h <- step_box(steps[[k]], u, s$id)
          at[[steps[[k]]$id]] <- u + h / 2
          u <- u + h
        }
        if (!is.na(unit)) {
          any_unit <<- TRUE
          arrow_up(xc, u, u + 14)
          u_plus <- u + 14 + g$plus_r
          plus_u(u_plus)
          # The skip line: out of the stream below the group, up the
          # right-hand side, back in at the plus.
          emit(line_u(xc, u_split, x_skip, u_split, col = th$line, lwd = 1.1))
          emit(line_u(x_skip, u_split, x_skip, u_plus, col = th$line, lwd = 1.1))
          arrow_left(x_skip, xc + g$plus_r, u_plus)
          u <- u_plus + g$plus_r
        }
        i <- run + 1L
      }
      for (inp in inputs) {
        target <- at[[inp$feeds]] %||% (u - 10)
        side[[length(side) + 1L]] <<- list(label = inp$label, u = target,
                                           x_to = c_left)
      }

    }
    u <- u + g$cpad
    header_u <- u + 10
    label <- paste(vapply(stages, `[[`, character(1), "label"), collapse = ", then ")
    emit(text_u(c_left + 10, header_u, arch_fit_text(label, c_right - c_left - 90, 11.5,
                                                     bold = TRUE),
                size = 11.5, col = th$ink, bold = TRUE))
    if (show_params) {
      total <- sum(vapply(stages, function(s) if (is.na(s$params)) 0 else s$params,
                          numeric(1)))
      if (total > 0) {
        emit(text_u(c_right - 10, header_u, arch_format_count(total), size = 10,
                    col = th$ink3, anchor = "end"))
      }
    }
    top <- header_u + 14
    emit_under(rect_u(c_left, u0, c_right - c_left, top - u0, fill = tint,
                      stroke = NA, r = 10))
    if (reps > 1L) {
      emit(text_u(c_left - 10, u0 + 16, paste0(reps, " \u00d7"), size = 15,
                  col = accent, bold = TRUE, anchor = "end"))
    }
    for (s in stages) {
      anchors[[s$id]] <<- list(x = c_right, u = header_u)
      anchors[[paste0(s$id, "@repeat")]] <<- list(x = c_right, u = u0 + 16)
    }
    top
  }

  # A stage that joins the stream: a box at the left, meeting the stream
  # at a plus (added) or a pill (concatenated, prepended).
  join_point <- function(s, u) {
    mid <- u + 12
    if (identical(s$join, "add")) plus_u(mid) else
      pill_u(mid, if (identical(s$join, "concat")) "concat" else "prepend")
    half <- if (identical(s$join, "add")) g$plus_r
            else arch_text_width(s$join, 10, bold = TRUE) / 2 + 7
    side[[length(side) + 1L]] <<- list(label = s$label, u = mid,
                                       x_to = xc - half, note = s$join_rows)
    # A callout about the stage points at where it joins, on the flow.
    anchors[[s$id]] <<- list(x = xc + half, u = mid)
    mid + 12
  }

  # --- the flow, bottom to top ------------------------------------------
  st <- arch$stages
  u <- 0
  i <- 1L
  while (i <= length(st)) {
    s <- st[[i]]
    if (i > 1L && is.null(s$join) && length(s$children)) {
      # A plain line: the container's own entry arrow carries the head.
      emit(line_u(xc, u, xc, u + g$gap, col = th$line, lwd = 1.2))
      if (show_shapes && !is.null(st[[i - 1L]]$shape)) {
        shapes[[length(shapes) + 1L]] <- list(text = st[[i - 1L]]$shape,
                                              u = u + g$gap / 2)
      }
      u <- u + g$gap
    } else if (i > 1L && is.null(s$join)) {
      arrow_up(xc, u, u + g$gap)
      if (show_shapes && !is.null(st[[i - 1L]]$shape)) {
        shapes[[length(shapes) + 1L]] <- list(text = st[[i - 1L]]$shape,
                                              u = u + g$gap / 2)
      }
      u <- u + g$gap
    } else if (i > 1L) {
      emit(line_u(xc, u, xc, u + 4, col = th$line, lwd = 1.2))
      u <- u + 4
    }
    if (!is.null(s$join)) {
      u <- join_point(s, u)
      i <- i + 1L
      next
    }
    if (length(s$children)) {
      group <- list(s)
      if (!is.null(s$interleave)) {
        while (i + length(group) <= length(st) &&
               identical(st[[i + length(group)]]$interleave, s$interleave)) {
          group[[length(group) + 1L]] <- st[[i + length(group)]]
        }
      }
      u <- container(group, u)
      # Steps that run once after the stack.
      for (gs in group) {
        for (fin in Filter(function(ch) identical(ch$role, "final"), gs$children)) {
          arrow_up(xc, u, u + 14)
          u <- u + 14
          u <- u + step_box(fin, u, gs$id)
        }
      }
      i <- i + length(group)
      next
    }
    u <- u + plain_box(s, u)
    i <- i + 1L
  }
  flow_top <- u

  # --- side inputs, spread so they do not overlap -------------------------
  if (length(side)) {
    ord <- order(vapply(side, `[[`, numeric(1), "u"))
    side <- side[ord]
    prev <- -Inf
    for (k in seq_along(side)) {
      side[[k]]$at <- max(side[[k]]$u, prev + 34)
      prev <- side[[k]]$at
    }
    for (sd in side) {
      h <- 26
      emit(rect_u(g$side_x, sd$at - h / 2, g$side_w, h, fill = th$box,
                  stroke = th$box_stroke, lwd = 1, r = 6))
      emit(text_u(g$side_x + g$side_w / 2, sd$at,
                  arch_fit_text(sd$label, g$side_w - 12, 10.5), size = 10.5,
                  col = th$ink, anchor = "middle"))
      # Elbow to the target's height when the box was pushed off it.
      x_mid <- g$side_x + g$side_w + 12
      emit(line_u(g$side_x + g$side_w, sd$at, x_mid, sd$at, col = th$line, lwd = 1.1))
      emit(line_u(x_mid, sd$at, x_mid, sd$u, col = th$line, lwd = 1.1))
      arrow_right(x_mid, sd$x_to, sd$u)
      if (!is.null(sd$note)) {
        emit(text_u(x_mid + 6, sd$u - 10, sd$note, size = 9.5, col = th$ink2,
                    italic = TRUE))
        drawn_extra <- c(drawn_extra, sd$note)
      }
    }
    flow_top <- max(flow_top, prev + 20)
  }

  # --- shapes on the arrows -------------------------------------------------
  for (sh in shapes) {
    emit(text_u(xc + 8, sh$u, sh$text, size = 10, col = th$ink2, family = "mono"))
  }

  # --- callouts ---------------------------------------------------------------
  callouts <- list()
  if (show_facts) {
    for (f in names(arch$facts)) {
      a <- arch_fact_anchor(arch, f)
      if (is.null(a)) next
      key <- if (identical(a$at, "repeat")) paste0(a$stage, "@repeat")
             else if (!is.null(a$child)) paste(a$stage, a$child) else a$stage
      pos <- anchors[[key]] %||% anchors[[a$stage]]
      if (is.null(pos)) next
      callouts[[key]] <- callouts[[key]] %||% list(pos = pos, facts = character())
      callouts[[key]]$facts[[f]] <- as.character(arch$facts[[f]])
    }
  }
  placed <- character()
  if (length(callouts)) {
    ord <- order(vapply(callouts, function(c) c$pos$u, numeric(1)))
    prev_top <- -Inf
    for (cl in callouts[ord]) {
      n <- length(cl$facts)
      h <- 15 * n
      centre <- max(cl$pos$u, prev_top + h / 2 + 10)
      prev_top <- centre + h / 2
      leader_x <- g$call_x - 8
      emit(line_u(cl$pos$x + 4, cl$pos$u, leader_x, centre, col = th$ink3,
                  lwd = 1, dash = TRUE))
      for (k in seq_len(n)) {
        yk <- centre + h / 2 - 7.5 - 15 * (k - 1L)
        nm <- names(cl$facts)[k]
        nm_w <- arch_text_width(nm, 11, bold = TRUE)
        emit(text_u(g$call_x, yk, nm, size = 11, col = th$ink, bold = TRUE))
        # Bold sans runs a little wider than its estimate; 10 px keeps
        # the value clear of its name.
        emit(text_u(g$call_x + nm_w + 10, yk,
                    arch_fit_text(cl$facts[[k]], W - g$pad - g$call_x - nm_w - 10, 11,
                                  bold = TRUE),
                    size = 11, col = accent, bold = TRUE))
      }
      placed <- c(placed, names(cl$facts))
    }
    flow_top <- max(flow_top, prev_top)
  }

  # --- the outer container, then everything turned the right way up ------
  outer <- rect_u(c_left - 20, -16, c_right - c_left + 40, flow_top + 32,
                  fill = th$outer, stroke = th$outer_stroke, lwd = 1, r = 14)
  body <- c(list(outer), under, flow)

  it <- list()
  push <- function(e) it[[length(it) + 1L]] <<- e
  push_all <- function(lst) for (e in lst) push(e)
  y <- g$pad + 4
  push(sc_text(g$pad, y, arch$title, size = 20, col = accent, bold = TRUE))
  if (!is.null(arch$subtitle) && nzchar(arch$subtitle)) {
    y <- y + 24
    push(sc_text(g$pad, y, arch$subtitle, size = 11.5, col = th$ink2))
  }
  shift <- y + 34 + flow_top + 16
  push_all(arch_shift_items(body, shift))
  y <- shift + 16 + 10

  # --- facts no callout points at -----------------------------------------
  rest <- arch$facts[!names(arch$facts) %in% placed]
  if (show_facts && length(rest)) {
    y <- y + 24
    facts <- arch_fact_table_items(rest, c_left - 20, y, W - g$pad - (c_left - 20),
                                   th, family = "sans", size = 11,
                                   label_x = c_left - 32)
    push_all(facts$items)
    y <- facts$y
  }

  # --- key ----------------------------------------------------------------
  notes <- arch_schematic_notes(arch, th, accent, any_unit,
                                show_shapes && length(shapes) > 0L,
                                length(callouts) > 0L,
                                fallback = !arch_schematic_has_flow(arch))
  groups_key <- arch_key_groups(arch, notes, show_shapes = show_shapes,
                                extra = drawn_extra, show_details = FALSE)
  key <- arch_key_items(groups_key, g$pad, y + 30, W - 2 * g$pad, th)
  push_all(key$items)
  y <- key$y

  list(width = W, height = ceiling(y + g$pad), bg = th$surface, items = it,
       theme = th, title = arch$title, key = groups_key,
       desc = paste(c(arch$subtitle, arch_key_text(groups_key)),
                    collapse = ". "))
}

# Move every item down by `dy`, turning u-space (y = -u) into the page.
# @keywords internal
arch_shift_items <- function(items, dy) {
  lapply(items, function(e) {
    switch(e$type,
      rect = { e$y <- e$y + dy; e },
      text = { e$y <- e$y + dy; e },
      line = { e$y1 <- e$y1 + dy; e$y2 <- e$y2 + dy; e },
      poly = { e$y <- e$y + dy; e },
      e)
  })
}

# Whether the description records residual connections: descriptions
# built before they did lack the field on their children.
# @keywords internal
arch_schematic_has_flow <- function(arch) {
  kids <- unlist(lapply(arch$stages, `[[`, "children"), recursive = FALSE)
  !length(kids) || all(vapply(kids, function(ch) "residual" %in% names(ch),
                              logical(1)))
}

# The notation the schematic draws, and only what this diagram uses.
# @keywords internal
arch_schematic_notes <- function(arch, th, accent, any_unit, any_shape,
                                 any_callout, fallback) {
  st <- arch$stages
  joins <- vapply(st, function(s) s$join %||% NA_character_, character(1))
  kids <- unlist(lapply(st, `[[`, "children"), recursive = FALSE)
  has <- function(f) any(vapply(kids, f, logical(1)))
  c(
    list(arch_note("the data flows upwards, from the input table to the prediction")),
    if (any_unit) list(arch_note(paste(
      "a line up the right-hand side ending in a circled plus: a residual",
      "connection, adding the input of the steps it passes back to their output"))),
    if (fallback) list(arch_note(
      "residual connections are not recorded for this model, so none are drawn")),
    if ("add" %in% joins) list(arch_note(
      "a box at the left meeting the flow at a circled plus: that stage's output is added to the flow")),
    if (any(c("concat", "prepend") %in% joins)) list(arch_note(paste(
      "concat / prepend: a stage's output joins as an extra column, or as",
      "learned rows or columns put in front"))),
    if (any(vapply(st, function(s) s$repeats > 1L, logical(1)))) list(arch_note(
      "N \u00d7: the steps in the tinted container are stacked N times", accent)),
    arch_interleave_notes(arch),
    if (has(function(ch) identical(ch$role, "input")) ||
        has(function(ch) isTRUE(ch$attn$qk_norm))) list(arch_note(paste(
      "small boxes at the left feed the step their arrow points to"))),
    if (has(function(ch) identical(ch$kind, "attention"))) list(arch_note(paste(
      "dark boxes are attention; the line under the name says where its keys",
      "come from, which tokens query in the last block, how it is scaled,",
      "and how many key/value heads the test rows read"), th$attn)),
    if (any_shape) list(arch_note(
      "beside the arrows: the tensor shape each stage emits, in the symbols above")),
    if (any_callout) list(arch_note(paste(
      "dotted lines: facts about the part they point to, with the numbers in",
      "the model's colour")))
  )
}
