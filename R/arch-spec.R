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
#' @param collapses `TRUE` on the stage after which the table has no
#'   feature axis left: each row is one vector from here on. At most one
#'   stage per architecture sets it; models that keep their cells to the
#'   end set none.
#' @param interleave Optional name shared by consecutive repeated stages
#'   whose blocks take turns -- one of each per repeat -- rather than
#'   running as one stack after another. They must have the same
#'   `repeats`.
#' @param role For a child: `"step"` (the default) for one step of the
#'   block's forward pass, listed in the order the steps run; `"input"`
#'   for a parameter table or module that another step consumes -- a
#'   RoPE table, inducing points, CLS tokens -- named by `feeds`; or
#'   `"final"` for a step that runs once after the whole stack rather
#'   than in every block, such as a stack's output norm.
#' @param feeds For `role = "input"`: the id of the step child it feeds.
#' @return A `tabfound_arch_stage`.
#' @keywords internal
arch_stage <- function(id, label, kind = "embed", detail = NULL, axis = NULL,
                       shape = NULL, group = NULL, repeats = 1L,
                       prefix = character(), children = list(),
                       collapses = FALSE, interleave = NULL,
                       role = c("step", "input", "final"), feeds = NULL) {
  kind <- match.arg(kind, names(ARCH_KINDS))
  role <- match.arg(role)
  if (identical(role, "input") != !is.null(feeds)) {
    cli::cli_abort("{.arg feeds} goes with {.code role = \"input\"}, and only with it.")
  }
  if (!is.null(axis)) axis <- match.arg(axis, ARCH_AXES)
  structure(
    list(id = id, label = label, kind = kind, detail = detail, axis = axis,
         shape = shape, group = group, repeats = as.integer(repeats),
         prefix = as.character(prefix), children = children,
         collapses = isTRUE(collapses), interleave = interleave,
         role = role, feeds = feeds, params = NA_real_),
    class = "tabfound_arch_stage"
  )
}

# The fact names every backend draws from. Sharing
# them is what lets two models' fact sheets be lined up against each
# other: "ICL blocks" in one and "ICL layers" in another would read as
# two different facts. A backend may use any subset, and the tests check
# that it uses nothing else.
ARCH_FACT_NAMES <- c(
  "Layers", "Column-stage blocks", "Row-stage blocks", "ICL layers",
  "Embedding width", "ICL width", "Attention heads", "ICL heads",
  "Embedding heads", "Query KV heads", "Inducing points",
  "Features per group", "Fourier frequencies", "Feed-forward",
  "Thinking rows", "Attention per layer", "Cell channels", "Output width",
  "Heads in checkpoint", "Preprocessing", "Normalisation",
  "Attention scaling", "Train/test split", "Missing values", "KV cache"
)

# A feed-forward width as a multiple of the width it expands, written the
# same way everywhere so equal values compare equal: "4 x width (768)".
# @keywords internal
arch_ff_fact <- function(hidden, width) {
  ratio <- format(round(hidden / width, 2), trim = TRUE, drop0trailing = TRUE)
  sprintf("%s x width (%d)", ratio, as.integer(hidden))
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

# Fill in `params` throughout a stage tree. A repeated stage's children
# point at its first block, so their `params` are per block; each child
# also gets `params_all`, its count over every block. That is not
# `params * repeats`: a child can own parts that are not per block -- a
# RoPE table or an output norm shared by the whole stack.
# @keywords internal
arch_fill_params <- function(stages, index, repeats = 1L) {
  lapply(stages, function(s) {
    s$params <- arch_count_params(index, s$prefix)
    s$params_all <- if (repeats > 1L) {
      arch_count_params_all_blocks(index, s$prefix)
    } else {
      s$params
    }
    if (length(s$children)) {
      s$children <- arch_fill_params(s$children, index, s$repeats)
    }
    s
  })
}

# Count a first-block prefix over every block: its first `.0` segment --
# the block index -- matches any index, so "layers.0.attn" also claims
# "layers.11.attn". Later `0`s are the block's own layers ("mlp.0", the
# first layer of a sequential) and stay as they are.
# @keywords internal
arch_count_params_all_blocks <- function(index, prefix) {
  if (is.null(index) || !length(prefix)) return(NA_real_)
  nms <- names(index)
  hit <- logical(length(nms))
  for (p in prefix) {
    # Prefix segments are module and parameter names -- letters, digits
    # and underscores -- so only the dots between them need escaping.
    seg <- strsplit(p, ".", fixed = TRUE)[[1L]]
    first <- match("0", seg)
    if (!is.na(first)) seg[first] <- "[0-9]+"
    rx <- paste0("^", paste(seg, collapse = "\\."), "(\\.|$)")
    hit <- hit | grepl(rx, nms)
  }
  sum(index[hit])
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
#'   `task`, `title`, `subtitle`, `facts`, `symbols`, `glossary`,
#'   `n_params` and `stages`.
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
  arch_check_interleave(desc$stages, bk$name)
  index <- arch_param_index(ctx$net)
  stages <- arch_fill_params(desc$stages, index)

  # The parameter count belongs in the subtitle but is not the backend's
  # to know: it comes from the weights, which the describe hook never
  # sees. Append it here so a description built from a checkpoint always
  # reports that checkpoint's size.
  subtitle <- paste(
    c(desc$subtitle,
      if (!is.null(index)) paste(arch_format_count(sum(index)), "parameters")),
    collapse = "  \u00b7  "
  )

  structure(
    list(
      backend     = bk$name,
      task        = ctx$task,
      title       = desc$title,
      subtitle    = subtitle,
      facts       = desc$facts %||% character(),
      symbols     = desc$symbols %||% arch_default_symbols(),
      glossary    = arch_merge_glossary(desc$glossary),
      stages      = stages,
      n_params    = if (is.null(index)) NA_real_ else sum(index),
      param_index = index,
      config      = ctx$config
    ),
    class = "tabfound_arch"
  )
}

# Stages that interleave take turns, so they must sit next to each other
# and repeat the same number of times. A describe hook that breaks this
# is describing something its network does not do.
# @keywords internal
arch_check_interleave <- function(stages, backend) {
  groups <- vapply(stages, function(s) s$interleave %||% NA_character_,
                   character(1))
  for (g in unique(groups[!is.na(groups)])) {
    at <- which(groups %in% g)
    reps <- vapply(stages[at], `[[`, integer(1), "repeats")
    if (length(at) < 2L || any(diff(at) != 1L) || length(unique(reps)) != 1L) {
      cli::cli_abort(c(
        "The {.val {backend}} description interleaves {.val {g}} wrongly.",
        i = "Interleaved stages must be consecutive, at least two, and \\
             repeat the same number of times."
      ))
    }
  }
  invisible(TRUE)
}

# The tag for a stage's repeat count: "× 4", or "× 4, alternating" for a
# stage whose blocks take turns with another's.
# @keywords internal
arch_repeat_tag <- function(s) {
  if (s$repeats <= 1L) return(NULL)
  paste0("\u00d7 ", s$repeats, if (!is.null(s$interleave)) ", alternating")
}

# The key note for interleaved stages, naming each group's stages; empty
# when nothing interleaves.
# @keywords internal
arch_interleave_notes <- function(arch) {
  groups <- unique(unlist(lapply(arch$stages, `[[`, "interleave")))
  lapply(groups, function(g) {
    labels <- vapply(Filter(function(s) identical(s$interleave, g),
                            arch$stages), `[[`, character(1), "label")
    arch_note(paste0(
      "alternating: ", paste(labels, collapse = " and "),
      " take turns, one of each per repeat, rather than one stack after ",
      "the other"))
  })
}

# A backend's own entries win over the shared ones.
# @keywords internal
arch_merge_glossary <- function(extra = NULL) {
  if (!length(extra)) return(ARCH_GLOSSARY)
  c(extra, ARCH_GLOSSARY[!names(ARCH_GLOSSARY) %in% names(extra)])
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
    E = "embedding width",
    n_test = "test rows (the rows being predicted)")
}

# Abbreviations a diagram may draw, with the definition its key shows. A
# backend can add or reword entries through a `glossary` element in its
# describe hook's return value. Only the terms a diagram actually draws
# go in its key, so a long list here costs nothing.
ARCH_GLOSSARY <- c(
  ICL       = "in-context learning: test rows attend to the labelled training rows",
  CLS       = "learned summary token that reads out a row or column",
  RoPE      = "rotary position embedding",
  MLP       = "multilayer perceptron (the feed-forward sublayer)",
  GELU      = "Gaussian error linear unit (activation)",
  SwiGLU    = "gated linear unit with a Swish activation",
  LayerNorm = "layer normalisation",
  RMSNorm   = "root-mean-square normalisation",
  QK        = "query and key (as in QK-norm)",
  KV        = "keys and values (as in KV cache, KV heads)",
  `NaN`     = "not-a-number: undefined values in the input",
  `NA`      = "missing values in the input",
  ECDF      = "empirical cumulative distribution function"
)

# Names that look like abbreviations but need no definition: models,
# publishers and libraries. The tests fail on a capitalised abbreviation
# that is in neither this list nor the glossary, which is how a new
# "GQA" in a label gets caught before it ships unexplained.
ARCH_PROPER_NAMES <- c(
  "TabPFN", "TabICL", "TabFM", "NVIDIA", "AutoGluon"
)

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
    rep_txt <- if (s$repeats > 1L) cli::col_yellow(sprintf(
      " x%d%s", s$repeats, if (!is.null(s$interleave)) ", alternating" else "")) else ""
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
      collapses = isTRUE(s$collapses),
      interleave = s$interleave %||% NA_character_,
      params  = s$params,
      stringsAsFactors = FALSE
    )
    if (identical(detail, "full")) for (ch in s$children) add(ch, parent = s$id)
  }
  for (s in x$stages) add(s)
  do.call(rbind, rows)
}
