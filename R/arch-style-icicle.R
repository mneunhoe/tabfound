# The "icicle" style.
#
# Where the parameters are, as lengths. Three bands run across the page
# on one linear scale: the phases, the stages within them, and the
# sublayers within those, each summed over every block of its stack. A
# stage's width is its share of the model, so a 25 M in-context stack
# and a 512-parameter token table are drawn in proportion -- which no
# other style does. The layout is Kruskal and Landwehr's icicle plot,
# turned on its side.
#
# * Every band sums to the total, and each stage's sublayers sum to the
#   stage. Parameters a stage owns that no listed sublayer claims are
#   drawn as "other"; a stage that lists no sublayers fills its own
#   width in the sublayer band too.
# * A block is labelled only where its label fits. Everything left
#   unlabelled is listed under the bands with its count, so nothing is
#   dropped silently.
# * Stages are coloured by kind (Okabe-Ito, as in glyphs); sublayers
#   take their own kind's colour mixed towards the surface. Label text
#   is dark or light, whichever reads better on the fill.
# * Stages with no parameters (the input, a weightless embedding, the
#   prediction) have no width and do not appear.

# @keywords internal
arch_icicle_theme <- function(theme = c("light", "dark")) {
  theme <- match.arg(theme)
  if (identical(theme, "light")) {
    list(name = "light", surface = "#ffffff", ink = "#151515",
         ink2 = "#5a5a55", ink3 = "#7a7a73", rule = "#dddcd6",
         phase = "#4a4a46", other = "#e3e2dc",
         kind = c(input = "#d9d8d2", embed = "#0072B2", attention = "#D55E00",
                  ffn = "#009E73", norm = "#b8b7b0", decode = "#CC79A7",
                  output = "#d9d8d2"))
  } else {
    list(name = "dark", surface = "#161618", ink = "#f2f1ea",
         ink2 = "#bdbcb2", ink3 = "#8f8e85", rule = "#3a3a37",
         phase = "#bdbcb2", other = "#3a3a37",
         kind = c(input = "#4a4a47", embed = "#56B4E9", attention = "#E69F00",
                  ffn = "#009E73", norm = "#6b6a63", decode = "#CC79A7",
                  output = "#4a4a47"))
  }
}

# @keywords internal
arch_icicle_geom <- function() {
  list(width = 940, pad = 40, gutter = 110, band_h = 46, band_gap = 3)
}

# The sublayer cells of one stage: each child over all its blocks, then
# whatever the stage owns beyond them. Old descriptions without
# `params_all` fall back to per-block counts times the repeats.
# @keywords internal
arch_icicle_sublayers <- function(s, th) {
  total <- s$params
  if (!length(s$children)) {
    return(list(list(label = NULL, value = total, kind = s$kind,
                     fill = arch_mix(th$kind[[s$kind]], th$surface, 0.75))))
  }
  cells <- list()
  for (ch in s$children) {
    v <- ch$params_all %||% (ch$params * s$repeats)
    if (is.na(v) || v <= 0) next
    cells[[length(cells) + 1L]] <- list(
      label = ch$label, value = v, kind = ch$kind,
      fill = arch_mix(th$kind[[ch$kind]], th$surface, 0.75))
  }
  claimed <- sum(vapply(cells, `[[`, numeric(1), "value"))
  # A description whose sublayers claim more than their stage is wrong;
  # shrink them to fit rather than draw past the stage's edge. The tests
  # check no backend does this.
  if (claimed > total && claimed > 0) {
    for (k in seq_along(cells)) cells[[k]]$value <- cells[[k]]$value * total / claimed
    claimed <- total
  }
  if (total - claimed > 0) {
    cells[[length(cells) + 1L]] <- list(label = "other", value = total - claimed,
                                        kind = NULL, fill = th$other)
  }
  cells
}

# The icicle scene builder, registered in `arch_styles()`. It always
# draws the sublayer band and draws no shapes, so `detail`, `show_shapes`
# and `show_params` do not apply.
# @keywords internal
arch_scene_icicle <- function(arch, detail, theme, show_shapes, show_params,
                              show_facts) {
  th <- arch_icicle_theme(theme)
  g  <- arch_icicle_geom()
  W  <- g$width
  x0 <- g$pad + g$gutter
  wid <- W - x0 - g$pad

  stages <- Filter(function(s) !is.na(s$params) && s$params > 0, arch$stages)
  total <- sum(vapply(stages, `[[`, numeric(1), "params"))
  sx <- function(v) if (total > 0) v / total * wid else 0

  it <- list()
  push <- function(e) it[[length(it) + 1L]] <<- e
  push_all <- function(lst) for (e in lst) push(e)
  narrow <- list(phase = character(), stage = character(),
                 sublayer = character())

  # --- header ---------------------------------------------------------
  y <- g$pad
  push(sc_text(g$pad, y, arch$title, size = 19, col = th$ink, bold = TRUE))
  if (!is.null(arch$subtitle) && nzchar(arch$subtitle)) {
    y <- y + 22
    push(sc_text(g$pad, y, arch$subtitle, size = 11.5, col = th$ink2))
  }
  y <- y + 30

  bands <- c(phase = "phase", stage = "stage",
             sublayer = "sublayers, all blocks")
  band_y <- y + (seq_along(bands) - 1L) * (g$band_h + g$band_gap)
  names(band_y) <- names(bands)
  for (b in names(bands)) {
    push(sc_text(x0 - 12, band_y[[b]] + g$band_h / 2, bands[[b]], size = 11,
                 col = th$ink3, anchor = "end", italic = TRUE))
  }

  # One block: its fill, then its label and count. A block too narrow
  # for its whole label but wide enough to say something gets the label
  # cut short; either way, a label not drawn in full goes in the band's
  # "too narrow" list with its count, so no name is lost.
  cell <- function(band, x, w, fill, label, sub, list_as = label) {
    by <- band_y[[band]]
    push(sc_rect(x, by, max(w - 1.5, 0.8), g$band_h, fill = fill, stroke = NA))
    if (is.null(label)) return(invisible())
    tc <- arch_text_on(fill, dark = "#111111", light = "#ffffff")
    room <- w - 12
    whole <- arch_text_width(label, 11.5, bold = TRUE) <= room
    if (whole || room >= 48) {
      shown <- if (whole) label else arch_fit_text(label, room, 11.5, bold = TRUE)
      sub_fits <- arch_text_width(sub, 10.5) <= room
      push(sc_text(x + 6, by + if (sub_fits) 15 else g$band_h / 2, shown,
                   size = 11.5, col = tc, bold = TRUE))
      if (sub_fits) push(sc_text(x + 6, by + 32, sub, size = 10.5, col = tc))
    }
    if (!whole) {
      narrow[[band]] <<- c(narrow[[band]], paste0(list_as, " (", sub, ")"))
    }
  }
  pct <- function(v) {
    share <- 100 * v / total
    if (share < 0.5) "<1%" else sprintf("%.0f%%", share)
  }

  # --- bands ----------------------------------------------------------
  x <- x0
  phase_x <- x0
  phase <- NULL
  phase_sum <- 0
  flush_phase <- function() {
    if (!is.null(phase) && nzchar(phase) && phase_sum > 0) {
      cell("phase", phase_x, sx(phase_sum), th$phase, phase,
           paste0(arch_format_count(phase_sum), ", ", pct(phase_sum)))
    }
  }
  for (s in stages) {
    grp <- s$group %||% ""
    if (!identical(grp, phase)) {
      flush_phase()
      phase <- grp
      phase_x <- x
      phase_sum <- 0
    }
    phase_sum <- phase_sum + s$params
    cell("stage", x, sx(s$params), th$kind[[s$kind]], s$label,
         paste0(arch_format_count(s$params),
                if (s$repeats > 1L) paste0(", ", s$repeats, " blocks")))
    cx <- x
    for (sub in arch_icicle_sublayers(s, th)) {
      label <- sub$label
      # A sublayer is labelled with its own name; the narrow list also
      # needs its stage, to tell one stage's "LayerNorm" from another's.
      cell("sublayer", cx, sx(sub$value), sub$fill, label,
           arch_format_count(sub$value),
           list_as = paste0(s$label, ": ", label))
      cx <- cx + sx(sub$value)
    }
    x <- x + sx(s$params)
  }
  flush_phase()

  # --- axis -------------------------------------------------------------
  y <- band_y[["sublayer"]] + g$band_h + 14
  push_all(arch_linear_axis_items(total, function(v) x0 + sx(v), y, th))
  push(sc_text(x0 - 12, y, "parameters", size = 11, col = th$ink3,
               anchor = "end", italic = TRUE))
  y <- y + 26

  # --- what was too narrow to label -------------------------------------
  heads <- c(phase = "phases", stage = "stages", sublayer = "sublayers")
  first <- TRUE
  for (b in names(narrow)) {
    if (!length(narrow[[b]])) next
    y <- y + if (first) 16 else 6
    if (first) {
      push(sc_text(g$pad, y, "Too narrow to label", size = 11, col = th$ink3,
                   bold = TRUE))
      y <- y + 20
      first <- FALSE
    }
    push(sc_text(x0 - 12, y, heads[[b]], size = 11, col = th$ink3,
                 anchor = "end", italic = TRUE))
    lines <- arch_wrap_text(paste(narrow[[b]], collapse = "  ·  "), wid, 11)
    for (l in lines) {
      push(sc_text(x0, y, l, size = 11, col = th$ink2))
      y <- y + 16
    }
  }

  # --- facts --------------------------------------------------------------
  if (show_facts && length(arch$facts)) {
    y <- y + 24
    facts <- arch_fact_table_items(arch$facts, x0, y, wid, th, family = "sans",
                                   size = 11, label_x = x0 - 12)
    push_all(facts$items)
    y <- facts$y
  }

  # --- key ------------------------------------------------------------------
  kinds <- unique(c(vapply(stages, `[[`, character(1), "kind"),
                    unlist(lapply(stages, function(s)
                      vapply(s$children, `[[`, character(1), "kind")))))
  kinds <- names(ARCH_KINDS)[names(ARCH_KINDS) %in% kinds]
  notes <- c(
    list(arch_note(paste(
      "three bands: the phases, then the stages in them, then each stage's",
      "sublayers summed over all of its stacked blocks"))),
    list(arch_note(paste(
      "width: parameters, on one linear scale for the whole figure; a stage",
      "with no listed sublayers fills its own width in the sublayer band"))),
    lapply(kinds, function(k) arch_note(paste0(
      tolower(ARCH_KINDS[[k]]), " (its sublayers in a muted shade)"),
      th$kind[[k]])),
    if (any(vapply(stages, function(s) any(vapply(arch_icicle_sublayers(s, th),
          function(c) identical(c$label, "other"), logical(1))), logical(1))))
      list(arch_note(
        "other: parameters in a stage that none of its listed sublayers claims",
        th$other)),
    if (any(vapply(stages, function(s) is.null(s$group), logical(1))))
      list(arch_note("a gap in the phase band: stages that belong to no phase")),
    if (length(stages) < length(arch$stages))
      list(arch_note(paste(
        "stages with no parameters (the input, a weightless embedding, the",
        "prediction) have no width and are not drawn")))
  )
  groups_key <- arch_key_groups(arch, notes, show_shapes = FALSE,
                                show_details = FALSE)
  key <- arch_key_items(groups_key, g$pad, y + 30, W - 2 * g$pad, th)
  push_all(key$items)
  y <- key$y

  list(width = W, height = ceiling(y + g$pad), bg = th$surface, items = it,
       theme = th, title = arch$title,
       key = groups_key,
       desc = paste(c(arch$subtitle, arch_key_text(groups_key)),
                    collapse = ". "))
}
