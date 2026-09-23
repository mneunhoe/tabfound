# Render the architecture gallery.
#
#   Rscript inst/gallery/make-gallery.R [outdir]
#
# Produces one diagram per backend for `vignettes/architecture-gallery.qmd`.
# The vignette ships the results rather than building them, because
# instantiating all six networks at their published widths costs a few
# hundred million parameters and a couple of minutes -- too much to pay
# on every `R CMD build`. Run this when a description changes.
#
# **No weights are read.** A diagram needs two things: the config, which
# fixes the architecture, and the *shapes* of the parameters, which the
# config fixes too. So each network is built and left at its random
# initialisation. Every number on these diagrams -- widths, head counts,
# per-stage and total parameter counts -- is what the trained checkpoint
# has, because none of them depends on the values in the tensors.
#
# Configs are resolved in this order, per model:
#
#   1. a directory named by the model's environment variable, e.g.
#      `TABFOUND_GALLERY_TABICL`, holding `config.json`;
#   2. the local model store, if `download_model(<id>)` has been run;
#   3. the publisher's `config.json` on the Hub, for the two models that
#      ship one (TabFM and Mitra) -- a few hundred bytes, not the weights.
#
# A model that none of the three finds is skipped with a message naming
# what would fetch it. The vignette says which diagrams it has.

suppressMessages(library(tabfound))

`%||%` <- function(x, y) if (is.null(x)) y else x

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args)) args[[1]] else "vignettes/figures"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# What to draw
# ---------------------------------------------------------------------------

GALLERY <- list(
  list(key = "tabpfn",   backend = "tabpfn",   task = "classification",
       id = "tabpfn-v2.5-classifier", env = "TABFOUND_GALLERY_TABPFN"),
  list(key = "tabpfn26", backend = "tabpfn26", task = "classification",
       id = "tabpfn-v2.6-classifier", env = "TABFOUND_GALLERY_TABPFN26"),
  list(key = "tabpfn3",  backend = "tabpfn3",  task = "classification",
       id = "tabpfn-v3-classifier",   env = "TABFOUND_GALLERY_TABPFN3"),
  # v3.5 has no per-task checkpoint, so `task` here picks which of the
  # two heads the diagram draws rather than which artifacts to read.
  list(key = "tabpfn35", backend = "tabpfn35", task = "classification",
       id = "tabpfn-v3.5",            env = "TABFOUND_GALLERY_TABPFN35"),
  list(key = "tabicl",   backend = "tabicl",   task = "classification",
       id = "tabicl-v2-classifier",   env = "TABFOUND_GALLERY_TABICL"),
  # TabFM and Mitra publish their config next to the weights, so the
  # gallery can be built for them without downloading anything large.
  list(key = "tabfm",    backend = "tabfm",    task = "classification",
       id = "tabfm-1.0.0-classifier", env = "TABFOUND_GALLERY_TABFM",
       repo = "google/tabfm-1.0.0-pytorch", path = "classification/config.json"),
  list(key = "mitra",    backend = "mitra",    task = "classification",
       id = "mitra-classifier",       env = "TABFOUND_GALLERY_MITRA",
       repo = "autogluon/mitra-classifier", path = "config.json")
)

# Extra renderings the vignette uses to show the options off.
EXTRAS <- list(
  list(key = "tabpfn3", suffix = "-full", detail = "full"),
  list(key = "tabpfn35", suffix = "-full", detail = "full"),
  list(key = "tabicl",  suffix = "-dark", theme  = "dark")
)

# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

config_path <- function(spec) {
  dir <- Sys.getenv(spec$env, unset = "")
  if (nzchar(dir) && file.exists(file.path(dir, "config.json"))) {
    return(file.path(dir, "config.json"))
  }
  store <- file.path(tabfound_home(), "models", spec$id)
  if (file.exists(file.path(store, "config.json"))) {
    return(file.path(store, "config.json"))
  }
  if (!is.null(spec$repo) && requireNamespace("hfhub", quietly = TRUE)) {
    got <- tryCatch(hfhub::hub_download(spec$repo, spec$path),
                    error = function(e) NULL)
    if (!is.null(got)) return(got)
  }
  NULL
}

skip_note <- function(spec) {
  if (!is.null(spec$repo)) {
    sprintf("could not reach %s; set %s to a directory holding its config.json",
            spec$repo, spec$env)
  } else {
    sprintf("run download_model(\"%s\"), or set %s to a converted directory",
            spec$id, spec$env)
  }
}

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

built <- list()
for (spec in GALLERY) {
  path <- config_path(spec)
  if (is.null(path)) {
    message("skipping ", spec$key, ": ", skip_note(spec))
    next
  }
  config <- jsonlite::fromJSON(path, simplifyVector = TRUE)
  net <- suppressMessages(
    get_backend(spec$backend)$build(config, spec$task)
  )
  # The same object `tabular_classifier()` hands to `plot()`, minus the
  # predictor closures, which no diagram consults.
  model <- structure(
    list(spec = NULL, state = NULL, model = net, config = config,
         device = "cpu", backend = spec$backend, task = spec$task,
         model_ref = list(model = spec$id, backend = spec$backend,
                          device = "cpu", args = list())),
    class = c("tabfound_classifier", "tabfound_model")
  )

  arch <- tabfound_architecture(model)
  plot_architecture(arch, file.path(outdir, paste0(spec$key, ".svg")))

  for (ex in Filter(function(e) identical(e$key, spec$key), EXTRAS)) {
    plot_architecture(
      arch, file.path(outdir, paste0(spec$key, ex$suffix, ".svg")),
      detail = ex$detail %||% "overview", theme = ex$theme %||% "light"
    )
  }

  built[[spec$key]] <- data.frame(
    key = spec$key, title = arch$title, params = arch$n_params,
    stages = length(arch$stages), stringsAsFactors = FALSE
  )
  rm(net, model, arch)
  gc(verbose = FALSE)
}

summary_df <- do.call(rbind, built)
if (!is.null(summary_df)) {
  message("\nBuilt ", nrow(summary_df), " diagram(s) in ", outdir, ":")
  print(summary_df, row.names = FALSE)
}
