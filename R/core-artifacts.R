# Locating and loading model artifacts.
#
# Every backend consumes the same two artifacts:
#   model.safetensors  — the tensors, keyed by their original state_dict names
#   config.json        — hyperparameters plus the ordered `state_dict_keys`
#
# Some publishers ship these directly (Google's TabFM has
# `classification/` and `regression/` subfolders on the Hub); others
# ship PyTorch pickles that must be converted once, offline, by a
# script in `inst/python/`.

#' Resolve a model reference to on-disk artifact paths
#'
#' @param model Character. Either a local directory containing
#'   `model.safetensors` + `config.json`, or a HuggingFace repo id.
#' @param subfolder Optional character. Sub-directory within the local
#'   directory or Hub repo holding the artifacts (e.g. `"classification"`
#'   for `google/tabfm-1.0.0-pytorch`).
#' @param weights_file,config_file File names to look for.
#' @return A list with `weights` and `config` paths.
#' @keywords internal
resolve_artifacts <- function(model,
                              subfolder = NULL,
                              weights_file = "model.safetensors",
                              config_file  = "config.json") {
  require_suggested("jsonlite")
  require_suggested("safetensors")

  rel <- function(f) if (is.null(subfolder)) f else file.path(subfolder, f)

  if (dir.exists(model)) {
    weights_path <- file.path(model, rel(weights_file))
    config_path  <- file.path(model, rel(config_file))
    if (!file.exists(weights_path) || !file.exists(config_path)) {
      cli::cli_abort(c(
        "Local directory {.path {model}} is missing model artifacts.",
        i = "Expected {.path {rel(weights_file)}} and {.path {rel(config_file)}}.",
        i = "Checkpoints published as PyTorch pickles need a one-time \\
             conversion; see {.path inst/python/}."
      ))
    }
    return(list(weights = weights_path, config = config_path))
  }

  # A string that is clearly a filesystem path but is not a directory is
  # a typo, not a Hub repo id. Saying so beats letting hfhub fail with
  # "missing commit header" three frames deeper.
  looks_like_path <- grepl("^(/|~|\\.\\.?/)", model) ||
    grepl("[\\\\]", model) || file.exists(model)
  if (looks_like_path) {
    cli::cli_abort(c(
      "{.path {model}} is not a directory.",
      i = "Pass a directory containing {.path {weights_file}} and \\
           {.path {config_file}}, a HuggingFace repo id, or a registered alias."
    ))
  }

  require_suggested("hfhub")
  list(
    weights = hfhub::hub_download(model, rel(weights_file)),
    config  = hfhub::hub_download(model, rel(config_file))
  )
}


#' Read a config.json into a plain list
#'
#' `jsonlite::fromJSON()` simplifies length-1 arrays to scalars, which is
#' what the backends want, but it also simplifies a list of strings to a
#' character vector — fine for `state_dict_keys`.
#'
#' @keywords internal
read_model_config <- function(path) {
  require_suggested("jsonlite")
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}


#' Load a safetensors file as a named list of tensors
#' @keywords internal
read_safetensors <- function(path, device = "cpu") {
  require_suggested("safetensors")
  obj <- safetensors::safe_load_file(path, framework = "torch")
  # `safe_load_file` returns extra metadata attributes alongside the
  # tensors; keep only entries that really are tensors.
  keep <- vapply(obj, function(x) inherits(x, "torch_tensor"), logical(1))
  obj[keep]
}
