#!/usr/bin/env Rscript
#
# Build `architectures.json` from whatever checkpoints are on this
# machine.
#
#   Rscript inst/memory/collect-architectures.R
#
# `estimate_peak_memory()` needs a model's *dimensions*, never its
# weights. Those dimensions live in `config.json`, which lives inside the
# download -- so without this file the one question the preflight most
# needs to answer ("is it worth pulling 6.5 GB?") could only be answered
# after pulling 6.5 GB.
#
# So the architecture fields are extracted once, here, and shipped: a few
# hundred bytes per model against the gigabytes they describe. The
# parameter shapes are dropped and replaced by the one number they were
# being summed for.
#
# Run this after adding a checkpoint to the catalogue. Ids whose weights
# are not on this machine are left as they were, so a partial run is
# additive rather than destructive.

suppressPackageStartupMessages({
  library(tabfound)
  library(jsonlite)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

HERE <- local({
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(arg)) dirname(normalizePath(sub("^--file=", "", arg[1])))
  else if (dir.exists("inst/memory")) normalizePath("inst/memory")
  else normalizePath(".")
})
out_path <- file.path(HERE, "architectures.json")

existing <- if (file.exists(out_path)) {
  fromJSON(out_path, simplifyVector = FALSE)
} else list()

catalog <- tabfound:::.model_catalog()
added <- character()

for (id in names(catalog)) {
  if (!isTRUE(tabfound:::.model_is_downloaded(id))) next
  entry <- catalog[[id]]
  dir <- tabfound:::.model_dir(id)
  if (!is.null(entry$subfolder)) dir <- file.path(dir, entry$subfolder)

  config <- tabfound:::read_model_config(file.path(dir, "config.json"))
  weights <- as.numeric(file.size(file.path(dir, "model.safetensors")))

  # The shapes are hundreds of entries and exist only to be summed; the
  # sum is what the estimator wants, so keep that and drop them.
  config$state_dict_shapes <- NULL
  config$state_dict_keys <- NULL
  config$source_sha256 <- NULL

  existing[[id]] <- list(
    backend = entry$backend,
    task = entry$task,
    weights_bytes = weights,
    config = config
  )
  added <- c(added, id)
}

if (!length(added)) {
  message("No downloaded models found; nothing to collect.")
} else {
  write_json(list(
    collected_on = format(Sys.Date()),
    note = paste("Architecture dimensions only -- no weights, no parameter",
                 "shapes. Regenerate with",
                 "inst/memory/collect-architectures.R."),
    models = existing
  ), out_path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
  cat("wrote ", out_path, "\n", sep = "")
  cat("  ", length(existing), " models known; refreshed: ",
      paste(added, collapse = ", "), "\n", sep = "")
}
