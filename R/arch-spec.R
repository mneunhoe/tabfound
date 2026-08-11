# Architecture description.
#
# A diagram is worth no more than the description behind it, so the
# description is the primary object and the picture is one rendering of
# it. `tabfound_architecture()` turns a loaded model into a
# `tabfound_arch`: an ordered list of stages, each with a kind, the axis
# its attention runs over, a symbolic shape annotation, and the
# parameter-name prefixes it owns.
#
# Parameter counts are read off the loaded network, never recomputed from
# the config, so a diagram cannot quietly disagree with the weights it
# claims to describe. `arch_audit_coverage()` checks the other half of
# that: every parameter in the network is claimed by exactly one stage,
# which is what stops a stage list going stale when an architecture
# gains a submodule.
#
# Each backend supplies its stage list through the registry's `describe`
# hook, so adding a backend still means one `register_backend()` call.

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# The kinds a stage can have. They drive colour and the legend, and they
# are deliberately few: an architecture diagram that needs twenty colours
# has stopped explaining anything.
ARCH_KINDS <- c(
  input     = "Data",
  embed     = "Embedding / encoder",
  attention = "Attention",
  ffn       = "Feed-forward",
  norm      = "Normalisation",
  decode    = "Decoder head",
  output    = "Prediction"
)

# The axis an attention sublayer runs over. This is the one thing that
# most distinguishes tabular foundation models from each other and from
# language models, so it gets its own slot rather than being buried in
# prose.
ARCH_AXES <- c("features", "columns", "rows", "cells", "inducing", "classes")

#' Describe one stage of an architecture
#'
#' @param id Short identifier, unique within the architecture.
#' @param label Human-facing name, shown inside the block.
#' @param kind One of `names(ARCH_KINDS)`.
#' @param detail Optional second line, e.g. the layer widths.
#' @param axis Optional attention axis; one of `ARCH_AXES`.
#' @param shape Optional symbolic output shape, e.g. `"(B, n, F, E)"`.
#' @param group Optional name of the phase this stage belongs to.
#'   Consecutive stages sharing a group are bracketed together in the
#'   diagram's left margin.
#' @param repeats Integer. How many times this block is stacked.
#' @param prefix Character vector of parameter-name prefixes this stage
#'   owns. Used to read exact parameter counts off the network and to
#'   check that every parameter is accounted for.
#' @param children Optional list of `arch_stage()`s, drawn when
#'   `detail = "full"`. Their `prefix` entries point at the *first*
#'   member of a repeated stack, so their counts are per block.
#' @return A `tabfound_arch_stage`.
#' @keywords internal
arch_stage <- function(id, label, kind = "embed", detail = NULL, axis = NULL,
                       shape = NULL, group = NULL, repeats = 1L,
                       prefix = character(), children = list()) {
  kind <- match.arg(kind, names(ARCH_KINDS))
  if (!is.null(axis)) axis <- match.arg(axis, ARCH_AXES)
  structure(
    list(id = id, label = label, kind = kind, detail = detail, axis = axis,
         shape = shape, group = group, repeats = as.integer(repeats),
         prefix = as.character(prefix), children = children,
         params = NA_real_),
    class = "tabfound_arch_stage"
  )
}

# Convenience: the same stage list every backend starts and ends with.
# @keywords internal
arch_input_stage <- function(shape = "(n rows, p features)",
                             detail = "numeric matrix, NA allowed") {
  arch_stage("input", "Input table", kind = "input", detail = detail,
             shape = shape)
}

# @keywords internal
arch_output_stage <- function(label, detail = NULL, shape = NULL) {
  arch_stage("output", label, kind = "output", detail = detail, shape = shape)
}

# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------

# Every parameter in the network, as name -> element count. Buffers are
# excluded: they are fixed tables (RoPE frequencies aside, which this
# package registers as parameters because the checkpoints store them
# that way), not learned weights.
# @keywords internal
arch_param_index <- function(net) {
  if (is.null(net)) return(NULL)
  ps <- net$named_parameters()
  if (!length(ps)) return(stats::setNames(numeric(), character()))
  stats::setNames(
    vapply(ps, function(p) as.numeric(p$numel()), numeric(1)),
    names(ps)
  )
}

# Which parameter names a prefix claims. A prefix matches a name exactly,
# or matches it up to a dot -- so `"encoder"` claims `"encoder.5.weight"`
# but not `"encoder_2.weight"`.
# @keywords internal
arch_prefix_hits <- function(names_, prefix) {
  if (!length(prefix)) return(logical(length(names_)))
  hit <- logical(length(names_))
  for (p in prefix) {
    hit <- hit | names_ == p | startsWith(names_, paste0(p, "."))
  }
  hit
}

# @keywords internal
arch_count_params <- function(index, prefix) {
  if (is.null(index) || !length(prefix)) return(NA_real_)
  sum(index[arch_prefix_hits(names(index), prefix)])
}

# Fill in `params` throughout a stage tree.
# @keywords internal
arch_fill_params <- function(stages, index) {
  lapply(stages, function(s) {
    s$params <- arch_count_params(index, s$prefix)
    if (length(s$children)) s$children <- arch_fill_params(s$children, index)
    s
  })
}

#' Parameters no stage claims, and parameters claimed twice
#'
#' A stage list is a hand-written summary of a network, and hand-written
#' summaries rot. This is what the tests assert on: after a describe hook
#' runs, every parameter should be claimed exactly once.
#'
#' @param arch A `tabfound_arch`.
#' @return A list with `unclaimed` and `duplicated` parameter names.
#' @keywords internal
arch_audit_coverage <- function(arch) {
  index <- arch$param_index
  if (is.null(index)) return(list(unclaimed = character(), duplicated = character()))
  nms <- names(index)
  counts <- integer(length(nms))
  for (s in arch$stages) {
    counts <- counts + as.integer(arch_prefix_hits(nms, s$prefix))
  }
  list(unclaimed = nms[counts == 0L], duplicated = nms[counts > 1L])
}

# ---------------------------------------------------------------------------
# The architecture object
# ---------------------------------------------------------------------------

#' Describe a loaded model's architecture
#'
#' Turns a loaded tabular foundation model into a structured description
#' of its architecture: the ordered stages it runs, the axis each
#' attention sublayer works over, the tensor shapes between stages, and
#' the exact parameter count of every stage read off the network itself.
#'
#' [plot_architecture()] renders this to SVG, PNG or PDF; printing it
#' gives the same information as text, and `as.data.frame()` gives one
#' row per stage.
#'
#' @param object A model from [tabular_classifier()] or
#'   [tabular_regressor()] -- fitted or not -- or a [tabfound()] fit.
#' @return An object of class `tabfound_arch`: a list with `backend`,
#'   `task`, `title`, `subtitle`, `facts`, `n_params` and `stages`.
#' @examples
#' \dontrun{
#' clf <- tabular_classifier("path/to/tabpfn-v2.5-clf")
#' arch <- tabfound_architecture(clf)
#' arch
#' as.data.frame(arch)
#' }
#' @seealso [plot_architecture()]
#' @export
tabfound_architecture <- function(object) {
  ctx <- arch_context(object)
  bk  <- get_backend(ctx$backend)

  if (is.null(bk$describe)) {
    cli::cli_abort(c(
      "The {.val {bk$name}} backend does not describe its architecture.",
      i = "Give {.fn register_backend} a {.arg describe} function to \\
           enable {.fn plot_architecture}."
    ))
  }

  desc <- bk$describe(ctx$config, ctx$task)
  index <- arch_param_index(ctx$net)
  stages <- arch_fill_params(desc$stages, index)

  # The parameter count belongs in the subtitle but is not the backend's
  # to know: it comes from the weights, which the describe hook never
  # sees. Append it here so a description built from a checkpoint always
  # reports that checkpoint's size.
  subtitle <- paste(
    c(desc$subtitle,
      if (!is.null(index)) paste(arch_format_count(sum(index)), "parameters")),
    collapse = "  ·  "
  )

  structure(
    list(
      backend     = bk$name,
      task        = ctx$task,
      title       = desc$title,
      subtitle    = subtitle,
      facts       = desc$facts %||% character(),
      symbols     = desc$symbols %||% arch_default_symbols(),
      stages      = stages,
      n_params    = if (is.null(index)) NA_real_ else sum(index),
      param_index = index,
      config      = ctx$config
    ),
    class = "tabfound_arch"
  )
}

# Pull the network, config, backend name and task out of whatever the
# caller handed us. A `tabfound_fit` wraps the model; everything else
# carries the fields directly.
# @keywords internal
arch_context <- function(object) {
  if (inherits(object, "tabfound_arch")) {
    cli::cli_abort("{.arg object} is already an architecture description.")
  }
  if (inherits(object, "tabfound_fit")) {
    inner <- object$model %||% object$fit %||% object$spec
    if (!inherits(inner, "tabfound_model")) {
      cli::cli_abort("This {.cls tabfound_fit} carries no model to describe.")
    }
    object <- inner
  }
  if (!inherits(object, "tabfound_model")) {
    cli::cli_abort(c(
      "{.arg object} must be a loaded tabular foundation model.",
      i = "Use {.fn tabular_classifier} or {.fn tabular_regressor}, or \\
           pass a {.fn tabfound} fit."
    ))
  }
  if (is.null(object$config)) {
    cli::cli_abort("This model carries no configuration to describe.")
  }
  list(net = object$model, config = object$config,
       backend = object$backend, task = object$task)
}

# The shape symbols every backend's annotations are written in. Backends
# may add their own.
# @keywords internal
arch_default_symbols <- function() {
  c(B = "ensemble members (batch)",
    n = "rows in the context (train + test)",
    p = "raw features",
    F = "feature groups",
    E = "embedding width")
}

# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

# @keywords internal
arch_format_count <- function(n) {
  if (length(n) != 1L || is.na(n)) return("--")
  if (n >= 1e9) return(sprintf("%.2f B", n / 1e9))
  if (n >= 1e6) return(sprintf("%.1f M", n / 1e6))
  if (n >= 1e3) return(sprintf("%.1f k", n / 1e3))
  format(n, big.mark = ",", trim = TRUE, scientific = FALSE)
}

#' @export
print.tabfound_arch <- function(x, ...) {
  cli::cli_h1(x$title)
  if (!is.null(x$subtitle)) cli::cli_text("{.emph {x$subtitle}}")
  cli::cli_text("{arch_format_count(x$n_params)} parameters in {length(x$stages)} stages")
  cli::cli_par()
  for (s in x$stages) {
    rep_txt <- if (s$repeats > 1L) cli::col_yellow(sprintf(" x%d", s$repeats)) else ""
    axis_txt <- if (!is.null(s$axis)) cli::col_grey(sprintf(" [over %s]", s$axis)) else ""
    par_txt <- if (is.na(s$params) || s$params == 0)
      "" else cli::col_grey(sprintf("  %s", arch_format_count(s$params)))
    cli::cli_text("{cli::symbol$bullet} {.strong {s$label}}{rep_txt}{axis_txt}{par_txt}")
    if (!is.null(s$detail)) cli::cli_text("  {cli::col_grey(s$detail)}")
    if (!is.null(s$shape)) cli::cli_text("  {cli::col_blue(s$shape)}")
  }
  cli::cli_end()
  if (length(x$facts)) {
    cli::cli_h3("Facts")
    for (nm in names(x$facts)) cli::cli_text("{nm}: {.val {x$facts[[nm]]}}")
  }
  invisible(x)
}

#' @param row.names,optional Ignored; present for S3 consistency.
#' @param detail `"overview"` for one row per stage, `"full"` to also
#'   include each repeated block's sublayers.
#' @rdname tabfound_architecture
#' @export
as.data.frame.tabfound_arch <- function(x, row.names = NULL, optional = FALSE,
                                        detail = c("overview", "full"), ...) {
  detail <- match.arg(detail)
  rows <- list()
  add <- function(s, parent = NA_character_) {
    rows[[length(rows) + 1L]] <<- data.frame(
      stage   = s$id,
      parent  = parent,
      label   = s$label,
      kind    = s$kind,
      axis    = s$axis %||% NA_character_,
      repeats = s$repeats,
      detail  = s$detail %||% NA_character_,
      shape   = s$shape %||% NA_character_,
      params  = s$params,
      stringsAsFactors = FALSE
    )
    if (identical(detail, "full")) for (ch in s$children) add(ch, parent = s$id)
  }
  for (s in x$stages) add(s)
  do.call(rbind, rows)
}
