# Rendering.
#
# Two renderers walk the scene built in `arch-layout.R`.
#
# * `arch_render_svg()` writes the SVG by hand. No dependency, no device,
#   text stays text (selectable, searchable, restyleable), and the output
#   is byte-stable, which is what lets the tests assert on it.
# * `arch_render_grid()` draws into whatever graphics device is open, so
#   PNG and PDF come from `grDevices` with nothing else installed, and an
#   interactive `plot()` lands in the plot pane.
#
# Both consume the same coordinates, so the three formats are the same
# picture. Sizes are CSS pixels throughout; the grid path converts once,
# at 96 dpi.

# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

ARCH_FONT <- paste(
  "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue'",
  "Arial, sans-serif", sep = ", "
)
ARCH_FONT_MONO <- "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# @keywords internal
svg_esc <- function(x) {
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;", x, fixed = TRUE)
  x <- gsub(">", "&gt;", x, fixed = TRUE)
  gsub("\"", "&quot;", x, fixed = TRUE)
}

# Two decimals, no padding, no scientific notation: readable, and
# byte-stable across platforms so the tests can assert on the output.
# @keywords internal
svg_num <- function(x) {
  format(round(as.numeric(x), 2), trim = TRUE, scientific = FALSE,
         drop0trailing = TRUE)
}

# @keywords internal
svg_attr <- function(...) {
  a <- list(...)
  a <- a[!vapply(a, function(v) is.null(v) || is.na(v[1]), logical(1))]
  paste0(names(a), "=\"", unlist(a), "\"", collapse = " ")
}

# @keywords internal
arch_svg_item <- function(e) {
  switch(
    e$type,
    rect = paste0("  <rect ", svg_attr(
      x = svg_num(e$x), y = svg_num(e$y),
      width = svg_num(e$w), height = svg_num(e$h),
      rx = if (e$r > 0) svg_num(e$r) else NULL,
      fill = if (is.na(e$fill)) "none" else e$fill,
      stroke = if (is.na(e$stroke)) NULL else e$stroke,
      `stroke-width` = if (is.na(e$stroke)) NULL else svg_num(e$lwd),
      `stroke-dasharray` = if (!is.na(e$stroke) && isTRUE(e$dash)) "4 3" else NULL
    ), "/>"),
    line = paste0("  <line ", svg_attr(
      x1 = svg_num(e$x1), y1 = svg_num(e$y1),
      x2 = svg_num(e$x2), y2 = svg_num(e$y2),
      stroke = e$col, `stroke-width` = svg_num(e$lwd),
      `stroke-linecap` = "round",
      `stroke-dasharray` = if (isTRUE(e$dash)) "4 3" else NULL
    ), "/>"),
    poly = paste0("  <polygon ", svg_attr(
      points = paste(paste0(svg_num(e$x), ",", svg_num(e$y)), collapse = " "),
      fill = e$fill
    ), "/>"),
    text = paste0("  <text ", svg_attr(
      x = svg_num(e$x), y = svg_num(e$y), dy = "0.35em",
      `font-family` = if (isTRUE(e$mono)) ARCH_FONT_MONO else ARCH_FONT,
      `font-size` = paste0(svg_num(e$size), "px"),
      `font-weight` = if (isTRUE(e$bold)) "600" else NULL,
      `font-style` = if (isTRUE(e$italic)) "italic" else NULL,
      `text-anchor` = switch(e$anchor, start = NULL, middle = "middle",
                             end = "end"),
      fill = e$col
    ), ">", svg_esc(e$text), "</text>"),
    ""
  )
}

# @keywords internal
arch_render_svg <- function(scene) {
  body <- vapply(scene$items, arch_svg_item, character(1))
  c(
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
    sprintf(paste0("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%s\" ",
                   "height=\"%s\" viewBox=\"0 0 %s %s\">"),
            svg_num(scene$width), svg_num(scene$height),
            svg_num(scene$width), svg_num(scene$height)),
    sprintf("  <rect width=\"%s\" height=\"%s\" fill=\"%s\"/>",
            svg_num(scene$width), svg_num(scene$height), scene$bg),
    body,
    "</svg>"
  )
}

# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

# CSS pixels are 1/96 in; graphics-device font sizes are points, 1/72 in.
# @keywords internal
px_to_pt <- function(px) px * 72 / 96

# Non-ASCII glyphs are the right choice for the SVG, which declares its
# own encoding, but a graphics device in a non-UTF-8 locale draws them as
# `<U+2014>` escapes. Rather than give up the typography everywhere, the
# grid path falls back to ASCII when -- and only when -- the session
# cannot render it.
ARCH_GLYPH_ASCII <- c(
  "—" = " - ",   # em dash
  "×" = "x",     # multiplication sign
  "·" = "-",     # middle dot
  "→" = "->",    # rightwards arrow
  "≤" = "<="
)

# @keywords internal
arch_plain <- function(s) {
  if (isTRUE(l10n_info()[["UTF-8"]])) return(s)
  for (g in names(ARCH_GLYPH_ASCII)) {
    s <- gsub(g, ARCH_GLYPH_ASCII[[g]], s, fixed = TRUE, useBytes = FALSE)
  }
  s
}

# @keywords internal
arch_render_grid <- function(scene) {
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(xscale = c(0, scene$width),
                                    yscale = c(0, scene$height)))
  grid::grid.rect(gp = grid::gpar(fill = scene$bg, col = NA))
  # Scene coordinates run downwards from the top left; grid's run upwards
  # from the bottom left. Flipping here, once per item, rather than by
  # reversing the viewport's scale: a reversed scale leaves `just` and
  # `height` pointing the wrong way, which silently stacks every box in
  # the opposite direction.
  for (e in scene$items) arch_grid_item(e, scene$height)
  grid::popViewport()
  invisible(NULL)
}

# @keywords internal
arch_grid_item <- function(e, H) {
  nu <- function(v) grid::unit(v, "native")
  fy <- function(v) grid::unit(H - v, "native")
  switch(
    e$type,
    rect = {
      gp <- grid::gpar(
        fill = if (is.na(e$fill)) NA else e$fill,
        col  = if (is.na(e$stroke)) NA else e$stroke,
        lwd  = e$lwd, lty = if (isTRUE(e$dash)) "22" else "solid"
      )
      # `grid.roundrect` takes a single radius in absolute units, which is
      # what the scene stores anyway.
      if (e$r > 0) {
        grid::grid.roundrect(x = nu(e$x), y = fy(e$y), width = nu(e$w),
                             height = nu(e$h), just = c("left", "top"),
                             r = grid::unit(e$r, "bigpts"), gp = gp)
      } else {
        grid::grid.rect(x = nu(e$x), y = fy(e$y), width = nu(e$w),
                        height = nu(e$h), just = c("left", "top"), gp = gp)
      }
    },
    line = grid::grid.lines(
      x = nu(c(e$x1, e$x2)), y = fy(c(e$y1, e$y2)),
      gp = grid::gpar(col = e$col, lwd = e$lwd, lineend = "round",
                      lty = if (isTRUE(e$dash)) "22" else "solid")
    ),
    poly = grid::grid.polygon(x = nu(e$x), y = fy(e$y),
                              gp = grid::gpar(fill = e$fill, col = NA)),
    text = grid::grid.text(
      label = arch_plain(e$text), x = nu(e$x), y = fy(e$y),
      just = c(switch(e$anchor, start = "left", middle = "centre", end = "right"),
               "centre"),
      gp = grid::gpar(
        col = e$col, fontsize = px_to_pt(e$size),
        fontface = if (isTRUE(e$bold) && isTRUE(e$italic)) "bold.italic"
                   else if (isTRUE(e$bold)) "bold"
                   else if (isTRUE(e$italic)) "italic" else "plain",
        fontfamily = if (isTRUE(e$mono)) "mono" else "sans"
      )
    ),
    invisible(NULL)
  )
}

# ---------------------------------------------------------------------------
# User-facing
# ---------------------------------------------------------------------------

# @keywords internal
arch_format_from_file <- function(file) {
  ext <- tolower(tools::file_ext(file))
  if (ext %in% c("svg", "png", "pdf")) return(ext)
  cli::cli_abort(c(
    "Cannot tell the output format from {.file {file}}.",
    i = "Use a {.val .svg}, {.val .png} or {.val .pdf} extension, or pass \\
         {.arg format}."
  ))
}

#' Draw a model's architecture
#'
#' Renders the architecture of a loaded tabular foundation model as a
#' stacked block diagram: one block per stage, coloured by what the stage
#' does, tagged with the axis its attention runs over, annotated with the
#' tensor shape it emits and its exact parameter count, and bracketed
#' into phases in the left margin. Repeated blocks are drawn once, as a
#' deck, with their repeat count; `detail = "full"` opens them up to show
#' the sublayers instead.
#'
#' Everything drawn comes from the model itself -- the stage list from
#' the backend's own description of its network, the parameter counts
#' read off the loaded weights -- so the picture cannot disagree with the
#' checkpoint it was made from.
#'
#' @param object A model from [tabular_classifier()] or
#'   [tabular_regressor()], or a [tabfound()] fit. A
#'   `tabfound_arch` from [tabfound_architecture()] is also accepted, so a
#'   description can be inspected and then drawn.
#' @param file Output path. The extension picks the format. With `NULL`
#'   (the default) the diagram is drawn on the current graphics device.
#' @param format `"svg"`, `"png"` or `"pdf"`. Inferred from `file` when
#'   not given.
#' @param detail `"overview"` draws each repeated stack as one block;
#'   `"full"` expands it into its sublayers.
#' @param theme `"light"` or `"dark"`.
#' @param scale Multiplies the rendered size. Only affects raster output
#'   (`png`), where it raises the resolution; SVG and PDF are vector and
#'   scale on their own.
#' @param show_shapes,show_params Draw the right-margin tensor-shape and
#'   parameter-count annotations.
#' @param show_facts Draw the fact sheet under the diagram.
#' @return Invisibly: the output path when `file` is given, otherwise the
#'   `tabfound_arch` that was drawn.
#' @examples
#' \dontrun{
#' clf <- tabular_classifier("path/to/tabpfn-v2.5-clf")
#' plot_architecture(clf)                                # current device
#' plot_architecture(clf, "tabpfn.svg")
#' plot_architecture(clf, "tabpfn.pdf", detail = "full")
#' plot_architecture(clf, "tabpfn-dark.png", theme = "dark", scale = 2)
#' }
#' @seealso [tabfound_architecture()] for the description behind the
#'   picture.
#' @export
plot_architecture <- function(object, file = NULL, format = NULL,
                              detail = c("overview", "full"),
                              theme = c("light", "dark"), scale = 1,
                              show_shapes = TRUE, show_params = TRUE,
                              show_facts = TRUE) {
  detail <- match.arg(detail)
  theme  <- match.arg(theme)
  arch <- if (inherits(object, "tabfound_arch")) object
          else tabfound_architecture(object)

  scene <- arch_scene(arch, detail = detail, theme = theme,
                      show_shapes = show_shapes, show_params = show_params,
                      show_facts = show_facts)

  if (is.null(file)) {
    if (!is.null(format) && !identical(format, "device")) {
      cli::cli_abort("{.arg format} needs a {.arg file} to write to.")
    }
    arch_render_grid(scene)
    return(invisible(arch))
  }

  fmt <- format %||% arch_format_from_file(file)
  fmt <- match.arg(fmt, c("svg", "png", "pdf"))

  switch(
    fmt,
    # `useBytes` so the UTF-8 glyphs reach the file intact even when the
    # session's locale is not UTF-8; the document declares its encoding.
    svg = writeLines(arch_render_svg(scene), file, useBytes = TRUE),
    png = {
      grDevices::png(file, width = scene$width * scale,
                     height = scene$height * scale, res = 96 * scale,
                     bg = scene$bg)
      on.exit(grDevices::dev.off(), add = TRUE)
      arch_render_grid(scene)
    },
    pdf = {
      grDevices::pdf(file, width = scene$width / 96, height = scene$height / 96,
                     bg = scene$bg)
      on.exit(grDevices::dev.off(), add = TRUE)
      arch_render_grid(scene)
    }
  )
  cli::cli_alert_success("Wrote {.file {file}} ({scene$width} x {scene$height} px).")
  invisible(file)
}

#' Draw a model's architecture
#'
#' A thin wrapper on [plot_architecture()], so `plot()` on a loaded model
#' does the obvious thing.
#'
#' @param x A `tabfound_model`.
#' @param ... Passed to [plot_architecture()].
#' @return Invisibly, the `tabfound_arch` that was drawn.
#' @export
plot.tabfound_model <- function(x, ...) plot_architecture(x, ...)
