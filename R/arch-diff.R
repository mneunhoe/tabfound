# Comparing two architectures.
#
# `plot_architecture_diff()` sets two models side by side, stage against
# stage, then fact against fact. Stages are paired by a longest common
# subsequence over their kind and attention axis -- the same kind of
# block doing the same kind of attention, in the same order -- so the
# pairing is structural, not semantic: TabPFN v3's many-class decoder
# pairs with TabICL's decoder MLP because both are decoders. Facts are
# paired by name, which is why the backends share one vocabulary.

# @keywords internal
arch_diff_theme <- function(theme = c("light", "dark")) {
  theme <- match.arg(theme)
  if (identical(theme, "light")) {
    list(name = "light", surface = "#ffffff", ink = "#151515",
         ink2 = "#5a5a55", ink3 = "#6d6d68", rule = "#e2e1db",
         a = "#007a5e", b = "#9e5f00", a_bg = "#eaf5f1", b_bg = "#fbf1e3",
         diff = "#c0392b", same = "#6e6e68")
  } else {
    list(name = "dark", surface = "#161618", ink = "#f2f1ea",
         ink2 = "#bdbcb2", ink3 = "#95948b", rule = "#3a3a37",
         a = "#3fbf9b", b = "#f0a840", a_bg = "#14302a", b_bg = "#33270f",
         diff = "#ff7a6b", same = "#a3a29a")
  }
}

# The theme with your own colours: the highlights, in order, are model
# A, model B and what differs. The card fills follow their model's
# colour onto the new background, and every text colour keeps 4.5:1 on
# the surface and on both fills.
# @keywords internal
arch_diff_colours <- function(th, colours) {
  if (is.null(colours$background) && is.null(colours$highlight)) return(th)
  hues <- c("a", "b", "diff", "same", "a_bg", "b_bg")
  th <- arch_adapt_theme(th, colours$background, keep = hues)
  hl <- colours$highlight
  if (length(hl) == 1L) {
    # Model B must stay apart from A: it takes whichever of the theme's
    # two model colours is further round the colour wheel from yours.
    candidates <- c(th$b, th$a)
    th$b <- candidates[[which.max(vapply(candidates, arch_hue_distance,
                                         numeric(1), hl))]]
  }
  th[c("a", "b", "diff")[seq_along(hl)]] <- as.list(hl)
  w <- if (identical(th$name, "dark")) 0.2 else 0.1
  th$a_bg <- arch_mix(th$a, th$surface, w)
  th$b_bg <- arch_mix(th$b, th$surface, w)
  for (h in c("a", "b", "diff", "same", "ink2", "ink3")) {
    th[[h]] <- arch_readable(th[[h]], c(th$surface, th$a_bg, th$b_bg), 4.5)
  }
  th
}

# How far apart two colours are round the colour wheel, from 0 to 0.5.
# @keywords internal
arch_hue_distance <- function(a, b) {
  h <- grDevices::rgb2hsv(grDevices::col2rgb(c(a, b)))["h", ]
  d <- abs(h[1] - h[2])
  min(d, 1 - d)
}

# @keywords internal
arch_diff_geom <- function() {
  list(pad = 40, col_w = 330, gutter = 150, card_h = 40, card_gap = 8)
}

# Pair two stage lists by a longest common subsequence over kind and
# axis. Returns a list of pairs, each with `a` and `b` (either may be
# NULL), in an order consistent with both models. Ties are broken the
# same way every time: walking back from the end, a stage only B has is
# set aside before one only A has, so in each unmatched run A's stages
# come first.
# @keywords internal
arch_align_stages <- function(sa, sb) {
  key <- function(s) paste(s$kind, s$axis %||% "")
  ka <- vapply(sa, key, character(1))
  kb <- vapply(sb, key, character(1))
  n <- length(ka)
  m <- length(kb)
  L <- matrix(0L, n + 1L, m + 1L)
  for (i in seq_len(n)) {
    for (j in seq_len(m)) {
      L[i + 1L, j + 1L] <- if (ka[i] == kb[j]) L[i, j] + 1L
                           else max(L[i, j + 1L], L[i + 1L, j])
    }
  }
  out <- list()
  i <- n
  j <- m
  while (i > 0L || j > 0L) {
    if (i > 0L && j > 0L && ka[i] == kb[j]) {
      out <- c(list(list(a = sa[[i]], b = sb[[j]])), out)
      i <- i - 1L
      j <- j - 1L
    } else if (j > 0L && (i == 0L || L[i + 1L, j] >= L[i, j + 1L])) {
      out <- c(list(list(a = NULL, b = sb[[j]])), out)
      j <- j - 1L
    } else {
      out <- c(list(list(a = sa[[i]], b = NULL)), out)
      i <- i - 1L
    }
  }
  out
}

# What the chip between a pair says.
# @keywords internal
arch_pair_status <- function(p) {
  if (is.null(p$a)) return("only B")
  if (is.null(p$b)) return("only A")
  if (p$a$repeats == p$b$repeats) "matched"
  else paste(p$a$repeats, "vs", p$b$repeats)
}

# Facts paired by name: those both models report first, then A's own,
# then B's, each in the order of the shared vocabulary.
# @keywords internal
arch_diff_facts <- function(fa, fb) {
  ord <- function(x) x[order(match(x, ARCH_FACT_NAMES), na.last = TRUE)]
  na <- names(fa)
  nb <- names(fb)
  nms <- c(ord(intersect(na, nb)), ord(setdiff(na, nb)), ord(setdiff(nb, na)))
  lapply(nms, function(n) {
    va <- if (n %in% na) as.character(fa[[n]]) else NULL
    vb <- if (n %in% nb) as.character(fb[[n]]) else NULL
    status <- if (is.null(va) || is.null(vb)) "one side"
              else if (identical(va, vb)) "shared" else "different"
    list(name = n, a = va, b = vb, status = status)
  })
}

# The union of two models' symbols. A symbol both use with different
# meanings is listed once per model, as "F (A)" and "F (B)", coloured.
# @keywords internal
arch_diff_symbols <- function(sa, sb, th) {
  items <- character()
  cols <- character()
  for (nm in union(names(sa), names(sb))) {
    in_a <- nm %in% names(sa)
    in_b <- nm %in% names(sb)
    if (in_a && in_b && !identical(sa[[nm]], sb[[nm]])) {
      items[[paste0(nm, " (A)")]] <- sa[[nm]]
      items[[paste0(nm, " (B)")]] <- sb[[nm]]
      cols[[paste0(nm, " (A)")]] <- th$a
      cols[[paste0(nm, " (B)")]] <- th$b
    } else {
      items[[nm]] <- if (in_a) sa[[nm]] else sb[[nm]]
    }
  }
  list(items = items, cols = cols)
}

# @keywords internal
arch_short_title <- function(arch) {
  sub(" - (classifier|regressor)$", "", arch$title)
}

# @keywords internal
arch_total_params <- function(arch) {
  sum(vapply(arch$stages, function(s) if (is.na(s$params)) 0 else s$params,
             numeric(1)))
}

# The diff scene.
# @keywords internal
arch_diff_scene <- function(A, B, theme = "light", colours = list()) {
  th <- arch_diff_colours(arch_diff_theme(theme), colours)
  g <- arch_diff_geom()
  xa <- g$pad
  xb <- xa + g$col_w + g$gutter
  xm <- xa + g$col_w + g$gutter / 2
  W <- xb + g$col_w + g$pad

  it <- list()
  push <- function(e) it[[length(it) + 1L]] <<- e
  push_all <- function(lst) for (e in lst) push(e)

  title <- paste(arch_short_title(A), "vs", arch_short_title(B))
  y <- g$pad
  push(sc_text(xa, y, title, size = 19, col = th$ink, bold = TRUE))
  y <- y + 22
  push(sc_text(xa, y, paste("stages paired by kind and attention axis;",
                            "facts paired by name"),
               size = 11.5, col = th$ink2))
  y <- y + 32
  for (side in list(list(x = xa, arch = A, col = th$a, tag = "MODEL A"),
                    list(x = xb, arch = B, col = th$b, tag = "MODEL B"))) {
    push(sc_text(side$x, y, side$tag, size = 10, col = side$col, bold = TRUE))
    push(sc_text(side$x, y + 17,
                 arch_fit_text(paste0(arch_short_title(side$arch), "  \u00b7  ",
                                      arch_format_count(arch_total_params(side$arch))),
                               g$col_w, 13, bold = TRUE),
                 size = 13, col = th$ink, bold = TRUE))
  }
  y <- y + 40

  card <- function(x, s, bg, accent) {
    push(sc_rect(x, y, g$col_w, g$card_h, fill = bg, stroke = NA, r = 6))
    push(sc_rect(x, y, 3, g$card_h, fill = accent, stroke = NA, r = 1.5))
    count <- if (!is.na(s$params) && s$params > 0) arch_format_count(s$params)
    count_w <- if (is.null(count)) 0 else arch_text_width(count, 11.5) + 12
    push(sc_text(x + 12, y + 13,
                 arch_fit_text(s$label, g$col_w - 22 - count_w, 12, bold = TRUE),
                 size = 12, col = th$ink, bold = TRUE))
    if (!is.null(count)) {
      push(sc_text(x + g$col_w - 10, y + 13, count, size = 11.5, col = th$ink,
                   anchor = "end"))
    }
    tags <- arch_glyphs_tags(s)
    tag_w <- if (is.null(tags)) 0 else arch_text_width(tags, 10.5) + 12
    if (!is.null(tags)) {
      push(sc_text(x + 12, y + 29, tags, size = 10.5, col = th$ink2))
    }
    if (!is.null(s$shape)) {
      push(sc_text(x + g$col_w - 10, y + 29,
                   arch_fit_text(s$shape, g$col_w - 22 - tag_w, 10.5,
                                 family = "mono"),
                   size = 10.5, col = th$ink2, anchor = "end", family = "mono"))
    }
  }
  chip <- function(text, col) {
    w <- arch_text_width(text, 10, bold = TRUE) + 14
    push(sc_rect(xm - w / 2, y + 11, w, 18, fill = th$surface, stroke = col,
                 lwd = 1, r = 9))
    push(sc_text(xm, y + 20, text, size = 10, col = col, anchor = "middle",
                 bold = TRUE))
  }

  pairs <- arch_align_stages(A$stages, B$stages)
  for (p in pairs) {
    if (!is.null(p$a)) card(xa, p$a, th$a_bg, th$a)
    if (!is.null(p$b)) card(xb, p$b, th$b_bg, th$b)
    status <- arch_pair_status(p)
    chip(status, switch(status, "only A" = th$a, "only B" = th$b,
                        matched = th$same, th$diff))
    y <- y + g$card_h + g$card_gap
  }

  # --- facts --------------------------------------------------------------
  facts <- arch_diff_facts(A$facts, B$facts)
  if (length(facts)) {
    y <- y + 26
    push(sc_text(xa, y, "Facts", size = 14, col = th$ink, bold = TRUE))
    y <- y + 22
    value_w <- g$col_w - 110
    for (f in facts) {
      push(sc_line(xa, y - 10, W - g$pad, y - 10, col = th$rule, lwd = 0.8))
      push(sc_text(xa, y, arch_fit_text(f$name, 180, 11.5), size = 11.5,
                   col = th$ink2))
      push(sc_text(xa + 190, y, arch_fit_text(f$a %||% "--", value_w, 11.5),
                   size = 11.5, col = if (is.null(f$a)) th$ink3 else th$a))
      push(sc_text(xb, y, arch_fit_text(f$b %||% "--", value_w, 11.5),
                   size = 11.5, col = if (is.null(f$b)) th$ink3 else th$b))
      push(sc_text(W - g$pad, y, f$status, size = 10.5,
                   col = switch(f$status, shared = th$same,
                                different = th$diff, th$ink3),
                   anchor = "end", bold = identical(f$status, "different")))
      y <- y + 22
    }
  }

  # --- key ----------------------------------------------------------------
  status <- vapply(pairs, arch_pair_status, character(1))
  notes <- c(
    list(arch_note("model A, its stages and its fact values", th$a),
         arch_note("model B, its stages and its fact values", th$b),
         arch_note(paste(
           "stages are paired by kind and attention axis, in order, not by",
           "name: a pair is the same kind of block in the same place, not",
           "necessarily the same design"))),
    if ("matched" %in% status) list(arch_note(
      "matched: same kind and axis, and the same number of stacked blocks",
      th$same)),
    if (any(grepl(" vs ", status, fixed = TRUE))) list(arch_note(
      "N vs M: same kind and axis, with N stacked blocks in A and M in B",
      th$diff)),
    if (any(status %in% c("only A", "only B"))) list(arch_note(
      "only A / only B: a stage with no counterpart in the other model")),
    if (length(facts)) list(arch_note(paste(
      "facts: shared (the same value), different, or one side (reported",
      "by one model only, shown as --)"))),
    # Both models' interleaving notes, once each.
    local({
      alt <- c(arch_interleave_notes(A), arch_interleave_notes(B))
      alt[!duplicated(vapply(alt, `[[`, character(1), "text"))]
    }),
    list(arch_note(paste(
      "on each card: \u00d7 N stacked blocks and the axis its attention runs",
      "over, then its output shape; top right, its exact parameter count")))
  )
  # Cards draw labels, tags and shapes, not detail lines.
  note_text <- vapply(notes, `[[`, character(1), "text")
  sym <- arch_diff_symbols(arch_used_symbols(A, details = FALSE),
                           arch_used_symbols(B, details = FALSE), th)
  terms_a <- arch_used_terms(A, note_text, details = FALSE)
  terms_b <- arch_used_terms(B, note_text, details = FALSE)
  terms <- c(terms_a, terms_b[!names(terms_b) %in% names(terms_a)])
  terms <- terms[order(match(names(terms), names(ARCH_GLOSSARY)))]
  groups_key <- list(
    list(head = "SYMBOLS IN SHAPES", items = sym$items, mono = TRUE,
         term_col = sym$cols),
    list(head = "ABBREVIATIONS", items = terms),
    list(head = "HOW TO READ", notes = notes)
  )
  key <- arch_key_items(groups_key, xa, y + 30, W - 2 * g$pad, th)
  push_all(key$items)
  y <- key$y

  list(width = W, height = ceiling(y + g$pad), bg = th$surface, items = it,
       theme = th, title = title, key = groups_key,
       desc = paste(c(paste("Model A:", A$title), paste("Model B:", B$title),
                      arch_key_text(groups_key)), collapse = ". "))
}

# @keywords internal
arch_as_arch <- function(x, arg) {
  if (inherits(x, "tabfound_arch")) return(x)
  tryCatch(tabfound_architecture(x), error = function(e) {
    cli::cli_abort(c("Cannot describe {.arg {arg}}.", x = conditionMessage(e)),
                   call = NULL)
  })
}

#' Compare two models' architectures
#'
#' Draws two architectures side by side: stage against stage, with a chip
#' between each pair saying whether they match, then fact against fact.
#' Stages are paired by a longest common subsequence over their kind and
#' the axis their attention runs over, so the pairing is structural, not
#' semantic -- a decoder pairs with a decoder whatever each is called.
#' Facts are paired by name. Like every diagram here, it ends in a key
#' that explains its symbols, abbreviations and notation.
#'
#' @param a,b The two models: each a model from [tabular_classifier()] or
#'   [tabular_regressor()], a [tabfound()] fit, or a `tabfound_arch` from
#'   [tabfound_architecture()], in any mix. `a` is drawn on the left.
#' @inheritParams plot_architecture
#' @param highlight Up to three colours, in this order: model A (teal by
#'   default), model B (amber) and what differs (red). Given one, model B
#'   takes amber or teal, whichever is further from it. Matched stages
#'   and shared facts stay grey.
#' @return Invisibly: the output path when `file` is given, otherwise a
#'   list of the two `tabfound_arch` objects that were drawn.
#' @examples
#' \dontrun{
#' icl <- tabular_classifier("tabicl-v2-classifier")
#' pfn <- tabular_classifier("tabpfn-v3-classifier")
#' plot_architecture_diff(icl, pfn, "tabicl-vs-tabpfn3.svg")
#' plot_architecture_diff(icl, pfn, "tabicl-vs-tabpfn3-viridis.svg",
#'                        background = "white",
#'                        highlight = grDevices::hcl.colors(3, "viridis"))
#' }
#' @seealso [plot_architecture()] to draw one model.
#' @export
plot_architecture_diff <- function(a, b, file = NULL, format = NULL,
                                   theme = c("light", "dark"), scale = 1,
                                   background = NULL, highlight = NULL) {
  background <- arch_colour(background, "background")
  highlight <- arch_colour(highlight, "highlight", n = 3L)
  theme <- if (missing(theme) && !is.null(background)) arch_theme_for(background)
           else match.arg(theme)
  A <- arch_as_arch(a, "a")
  B <- arch_as_arch(b, "b")
  scene <- arch_diff_scene(A, B, theme,
                           colours = list(background = background,
                                          highlight = highlight))
  out <- arch_write_scene(scene, file, format, scale)
  invisible(out %||% list(a = A, b = B))
}
