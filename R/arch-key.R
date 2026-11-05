# The key.
#
# Every diagram explains itself: each symbol in a shape annotation, each
# abbreviation in its text and each piece of notation the style draws is
# defined on the image, so a diagram shared on its own still reads. The
# key is drawn at the foot of every style, cannot be turned off and is
# never truncated -- entries wrap onto as many lines as they need and
# the canvas grows to fit.
#
# "Used" is judged by one tokeniser for both sides, so what the key lists
# and what the tests check is drawn are the same set.

# ---------------------------------------------------------------------------
# What a diagram uses
# ---------------------------------------------------------------------------

# Plain words a shape may spell out in full, as in "(n rows, p features)".
# They are read as text, not as symbols needing a definition.
ARCH_SHAPE_WORDS <- c("rows", "features")

# Identifier-like tokens: a letter or underscore, then letters, digits or
# underscores. `n_test` is one token; `64` and `+` are none.
# @keywords internal
arch_tokens <- function(text) {
  text <- text[!is.na(text)]
  if (!length(text)) return(character())
  unique(unlist(regmatches(text, gregexpr("\\b[A-Za-z_][A-Za-z0-9_]*\\b",
                                          text, perl = TRUE))))
}

# Every shape annotation in the description, sublayers included.
# @keywords internal
arch_shapes <- function(arch) {
  unlist(lapply(arch$stages, function(s)
    c(s$shape, unlist(lapply(s$children, `[[`, "shape")))))
}

# The tokens in the shapes that must each have a definition.
# @keywords internal
arch_shape_symbols <- function(arch) {
  setdiff(arch_tokens(arch_shapes(arch)), ARCH_SHAPE_WORDS)
}

# Every piece of description text a diagram may draw. A style that does
# not draw detail lines or shapes says so, and their words drop out.
# @keywords internal
arch_drawn_text <- function(arch, details = TRUE, shapes = TRUE) {
  stage_text <- function(s) {
    c(s$label, if (details) s$detail, s$group, if (shapes) s$shape,
      unlist(lapply(s$children, stage_text)))
  }
  c(arch$title, arch$subtitle, names(arch$facts),
    vapply(arch$facts, as.character, character(1)),
    unlist(lapply(arch$stages, stage_text)))
}

# Only the symbols the diagram uses, in the order the backend declares
# them. A legend for letters the diagram never draws is noise, and the
# backends declare a superset so a shape can be reworded without
# touching the key. Details count too: "Linear(2 -> E)" uses `E`.
# @keywords internal
arch_used_symbols <- function(arch, shapes = TRUE, details = TRUE) {
  sym <- arch$symbols
  if (!length(sym)) return(sym)
  prose <- unlist(lapply(arch$stages, function(s)
    c(s$label, if (details) s$detail, unlist(lapply(s$children, function(ch)
      c(ch$label, if (details) ch$detail))))))
  used <- union(if (shapes) arch_shape_symbols(arch), arch_tokens(prose))
  sym[names(sym) %in% used]
}

# @keywords internal
arch_term_pattern <- function(term) paste0("\\b", term, "s?\\b")

# The glossary entries whose term appears as a whole word, optionally
# plural, in the diagram's text or in `extra` (the style's own notes and
# legend).
# @keywords internal
arch_used_terms <- function(arch, extra = character(), details = TRUE,
                            shapes = TRUE) {
  gl <- arch$glossary %||% ARCH_GLOSSARY
  txt <- paste(c(arch_drawn_text(arch, details, shapes), extra),
               collapse = "\n")
  hit <- vapply(names(gl), function(t) grepl(arch_term_pattern(t), txt,
                                             perl = TRUE), logical(1))
  gl[hit]
}

# Tokens that look like abbreviations -- two or more capitals, as in
# "ICL", "RoPE", "LayerNorm" -- and that neither the glossary nor the
# proper-name list covers. Empty for a description that explains itself.
# @keywords internal
arch_unexplained_terms <- function(arch, extra = character()) {
  gl <- names(arch$glossary %||% ARCH_GLOSSARY)
  tok <- arch_tokens(c(arch_drawn_text(arch), extra))
  tok <- tok[grepl("^[A-Za-z]*[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*$", tok)]
  base <- sub("s$", "", tok)
  tok[!(tok %in% c(gl, ARCH_PROPER_NAMES) | base %in% gl)]
}

# ---------------------------------------------------------------------------
# Building and drawing it
# ---------------------------------------------------------------------------

# One line of a style's "How to read" group. `swatch` draws a colour chip
# before it, for notes that explain a colour; `outline = TRUE` draws the
# chip as an outline, for a colour the diagram uses on edges.
# @keywords internal
arch_note <- function(text, swatch = NULL, outline = FALSE) {
  list(text = text, swatch = swatch, outline = outline)
}

# The three groups of a single model's key. `notes` is the style's list
# of `arch_note()`s; `extra` is any other text the style draws (its
# legend), which is scanned for abbreviations along with the notes.
# `show_details = FALSE` is for a style that draws no detail lines.
# @keywords internal
arch_key_groups <- function(arch, notes = list(), show_shapes = TRUE,
                            extra = character(), show_details = TRUE) {
  note_text <- vapply(notes, `[[`, character(1), "text")
  list(
    list(head = "SYMBOLS IN SHAPES",
         items = arch_used_symbols(arch, show_shapes, show_details),
         mono = TRUE),
    list(head = "ABBREVIATIONS",
         items = arch_used_terms(arch, c(note_text, extra), show_details,
                                 show_shapes)),
    list(head = "HOW TO READ", notes = notes)
  )
}

# Draw the key at (x, y), `w` wide. `th` needs `ink`, `ink2` and `ink3`.
# Terms and meanings sit in two columns when there is room; notes always
# run the full width. `family` sets the text face to match the style;
# symbols stay monospaced, as they are in the shapes. `rule = FALSE`
# drops the hairline above it, for a style whose last element already
# ends in one. Returns the items and the y the key ends at.
# @keywords internal
arch_key_items <- function(groups, x, y, w, th, family = "sans",
                           rule = TRUE) {
  out <- list()
  add <- function(e) out[[length(out) + 1L]] <<- e
  size <- 11
  line_h <- 15
  row_gap <- 5

  if (rule) add(sc_line(x, y, x + w, y, col = th$ink3, lwd = 0.8))
  y <- y + 20
  add(sc_text(x, y, "Key", size = 12, col = th$ink, bold = TRUE,
              family = family))
  y <- y + 22

  for (g in groups) {
    is_notes <- !is.null(g$notes)
    n <- if (is_notes) length(g$notes) else length(g$items)
    if (!n) next
    add(sc_text(x, y, g$head, size = 10.5, col = th$ink3, bold = TRUE,
                family = family))
    y <- y + 18

    if (is_notes) {
      swatches <- any(vapply(g$notes, function(nt) !is.null(nt$swatch),
                             logical(1)))
      tx <- x + if (swatches) 16 else 0
      for (nt in g$notes) {
        if (!is.null(nt$swatch)) {
          add(if (isTRUE(nt$outline)) {
            sc_rect(x + 0.5, y - 4.5, 9, 9, fill = NA, stroke = nt$swatch,
                    lwd = 1.2, r = 2)
          } else {
            sc_rect(x, y - 5, 10, 10, fill = nt$swatch, stroke = NA, r = 2)
          })
        }
        lines <- arch_wrap_text(nt$text, x + w - tx, size, family = family)
        for (k in seq_along(lines)) {
          add(sc_text(tx, y + (k - 1L) * line_h, lines[k], size = size,
                      col = th$ink2, family = family))
        }
        y <- y + length(lines) * line_h + row_gap
      }
    } else {
      ncol <- if (w >= 640) 2L else 1L
      gap <- 24
      colw <- (w - (ncol - 1L) * gap) / ncol
      terms <- names(g$items)
      mono <- isTRUE(g$mono)
      term_family <- if (mono) "mono" else family
      term_w <- max(vapply(terms, arch_text_width, numeric(1), size = size,
                           bold = !mono, family = term_family)) + 14
      rows <- ceiling(n / ncol)
      wrapped <- lapply(unname(g$items), arch_wrap_text,
                        max_px = colw - term_w, size = size, family = family)
      # Items run down the first column, then the second; a row is as
      # tall as the taller of its two entries.
      row_of <- (seq_len(n) - 1L) %% rows + 1L
      col_of <- (seq_len(n) - 1L) %/% rows
      heights <- vapply(seq_len(rows), function(r)
        max(lengths(wrapped[row_of == r])) * line_h, numeric(1))
      tops <- y + c(0, cumsum(heights + row_gap))[seq_len(rows)]
      for (i in seq_len(n)) {
        cx <- x + col_of[i] * (colw + gap)
        ty <- tops[row_of[i]]
        # A diff colours a term that belongs to one model only.
        term_col <- if (terms[i] %in% names(g$term_col)) g$term_col[[terms[i]]]
                    else th$ink
        add(sc_text(cx, ty, terms[i], size = size, col = term_col,
                    bold = !mono, family = term_family))
        for (k in seq_along(wrapped[[i]])) {
          add(sc_text(cx + term_w, ty + (k - 1L) * line_h, wrapped[[i]][k],
                      size = size, col = th$ink2, family = family))
        }
      }
      y <- y + sum(heights + row_gap)
    }
    y <- y + 10
  }
  list(items = out, y = y)
}

# The key as one line of plain text, for the SVG's `<desc>`.
# @keywords internal
arch_key_text <- function(groups) {
  parts <- vapply(groups, function(g) {
    entries <- if (!is.null(g$notes)) {
      vapply(g$notes, `[[`, character(1), "text")
    } else if (length(g$items)) {
      paste0(names(g$items), ": ", unname(g$items))
    } else {
      character()
    }
    if (!length(entries)) return(NA_character_)
    head <- paste0(toupper(substr(g$head, 1, 1)), tolower(substring(g$head, 2)))
    paste0(head, ": ", paste(entries, collapse = "; "), ".")
  }, character(1))
  paste(parts[!is.na(parts)], collapse = " ")
}
