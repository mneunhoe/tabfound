# Getting model weights onto the machine.
#
# Two of the four families publish artifacts this package can read
# directly (TabFM and Mitra ship `model.safetensors` + `config.json` on
# the Hub). The other two ship PyTorch pickles, which have to be
# converted once -- and cannot be converted from R: the checkpoints are
# nested Python dicts holding a state_dict alongside hyperparameters, and
# `torch::load_state_dict()` refuses them ("Expected Tensor but got
# GenericDict"). So conversion shells out to the scripts in
# `inst/python/`, exactly as the manual instructions always did. This
# file automates the sequence rather than replacing it.
#
# Everything lands in one local store, `tabfound_home()`, so a model is
# downloaded and converted at most once per machine and the second call
# is instant.
#
# On where that store lives: not inside the installed package. A package
# must not write to its own installation directory -- CRAN policy
# forbids it, and it fails outright on the common setups where the
# library is read-only or shared between users. `tools::R_user_dir()` is
# R's designated per-user, per-package location and is what
# `tabfound_home()` returns. Point `options(tabfound.home = ...)` or
# `TABFOUND_HOME` somewhere else if you want a different location.
#
# Nothing here downloads without being told to. `load_backend_model()`
# asks first; see `.prompt_download()`.


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

# One entry per downloadable artifact set. `convert` names the script in
# `inst/python/` that turns a checkpoint into artifacts, or is NULL when
# the publisher already ships them.
#
# `size_mb` is the on-disk size of the *converted* result, measured, and
# used only to make the consent prompt concrete when the Hub cannot be
# reached for the real number. NA where it has not been measured.
# @keywords internal
.model_catalog <- function() {
  list(
    "tabpfn-v2.5-classifier" = list(
      backend = "tabpfn", task = "classification",
      family = c("tabpfn", "tabpfn-v2.5"),
      repo = "Prior-Labs/tabpfn_2_5",
      file = "tabpfn-v2.5-classifier-v2.5_default.ckpt",
      convert = "tabpfn", head = "classifier", size_mb = 41,
      license = "Prior Labs License (see the repo)",
      description = "TabPFN v2.5 classifier"
    ),
    "tabpfn-v2.5-regressor" = list(
      backend = "tabpfn", task = "regression",
      family = c("tabpfn", "tabpfn-v2.5"),
      repo = "Prior-Labs/tabpfn_2_5",
      file = "tabpfn-v2.5-regressor-v2.5_default.ckpt",
      convert = "tabpfn", head = "regressor", size_mb = 39,
      license = "Prior Labs License (see the repo)",
      description = "TabPFN v2.5 regressor"
    ),
    "tabpfn-v2.6-classifier" = list(
      backend = "tabpfn26", task = "classification", family = "tabpfn-v2.6",
      repo = "Prior-Labs/tabpfn_2_6",
      file = "tabpfn-v2.6-classifier-v2.6_default.ckpt",
      convert = "tabpfn", head = "classifier", size_mb = 43,
      license = "tabpfn-2.6-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v2.6 classifier"
    ),
    "tabpfn-v2.6-regressor" = list(
      backend = "tabpfn26", task = "regression", family = "tabpfn-v2.6",
      repo = "Prior-Labs/tabpfn_2_6",
      file = "tabpfn-v2.6-regressor-v2.6_default.ckpt",
      convert = "tabpfn", head = "regressor", size_mb = 52,
      license = "tabpfn-2.6-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v2.6 regressor"
    ),
    "tabpfn-v3-classifier" = list(
      backend = "tabpfn3", task = "classification", family = "tabpfn-v3",
      repo = "Prior-Labs/tabpfn_3",
      file = "tabpfn-v3-classifier-v3_default.ckpt",
      convert = "tabpfn", head = "classifier", size_mb = 203,
      license = "tabpfn-3-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v3 classifier"
    ),
    "tabpfn-v3-regressor" = list(
      backend = "tabpfn3", task = "regression", family = "tabpfn-v3",
      repo = "Prior-Labs/tabpfn_3",
      file = "tabpfn-v3-regressor-v3_default.ckpt",
      convert = "tabpfn", head = "regressor", size_mb = 223,
      license = "tabpfn-3-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v3 regressor"
    ),
    # The six specialised v3 checkpoints. Architecturally they are the
    # *same model* as the defaults -- identical `config`, identical
    # state-dict keys and shapes, verified checkpoint by checkpoint -- so
    # they need no backend of their own and load through `tabpfn3` like
    # any other. What differs is the weights (four of them) and the
    # `inference_config` recipe the publisher bundles alongside.
    #
    # They carry no `family`, so `"tabpfn-v3"` keeps meaning the default
    # and these are reachable by exact id only.
    "tabpfn-v3-classifier-binary" = list(
      backend = "tabpfn3", task = "classification",
      repo = "Prior-Labs/tabpfn_3",
      file = "tabpfn-v3-classifier-v3_20260417_binary.ckpt",
      convert = "tabpfn", head = "classifier", size_mb = 203,
      license = "tabpfn-3-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v3 classifier, tuned for binary targets under 200k rows"
    ),
    "tabpfn-v3-classifier-multiclass" = list(
      backend = "tabpfn3", task = "classification",
      repo = "Prior-Labs/tabpfn_3",
      file = "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt",
      convert = "tabpfn", head = "classifier", size_mb = 203,
      license = "tabpfn-3-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v3 classifier, tuned for multiclass targets under 200k rows"
    ),
    # The two `_ood` checkpoints hold *byte-identical weights* to the
    # corresponding defaults -- SHA-256 over the full state dict matches.
    # Only the bundled preprocessing recipe differs, which this package
    # reads from an ensemble dump rather than from the checkpoint. They
    # are catalogued so the publisher's names resolve, but downloading one
    # when you already have the default fetches the same tensors again.
    "tabpfn-v3-classifier-ood" = list(
      backend = "tabpfn3", task = "classification",
      repo = "Prior-Labs/tabpfn_3",
      file = "tabpfn-v3-classifier-v3_20260506_ood.ckpt",
      convert = "tabpfn", head = "classifier", size_mb = 203,
      license = "tabpfn-3-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v3 classifier, OOD preprocessing (weights identical to the default)"
    ),
    "tabpfn-v3-regressor-mediumdata" = list(
      backend = "tabpfn3", task = "regression",
      repo = "Prior-Labs/tabpfn_3",
      file = "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt",
      convert = "tabpfn", head = "regressor", size_mb = 223,
      license = "tabpfn-3-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v3 regressor, tuned for under 100k rows"
    ),
    "tabpfn-v3-regressor-ood" = list(
      backend = "tabpfn3", task = "regression",
      repo = "Prior-Labs/tabpfn_3",
      file = "tabpfn-v3-regressor-v3_20260506_ood.ckpt",
      convert = "tabpfn", head = "regressor", size_mb = 223,
      license = "tabpfn-3-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v3 regressor, OOD preprocessing (weights identical to the default)"
    ),
    "tabpfn-v3-regressor-timeseries" = list(
      backend = "tabpfn3", task = "regression",
      repo = "Prior-Labs/tabpfn_3",
      file = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
      convert = "tabpfn", head = "regressor", size_mb = 223,
      license = "tabpfn-3-license-v1.0: non-commercial (see the repo)",
      description = "TabPFN v3 regressor, fine-tuned on synthetic time series"
    ),
    "tabicl-v2-classifier" = list(
      backend = "tabicl", task = "classification", family = "tabicl",
      repo = "jingang/TabICL",
      file = "tabicl-classifier-v2-20260212.ckpt",
      convert = "tabicl", size_mb = 106,
      license = "see the repo",
      description = "TabICL v2 classifier"
    ),
    "tabicl-v2-regressor" = list(
      backend = "tabicl", task = "regression", family = "tabicl",
      repo = "jingang/TabICL",
      file = "tabicl-regressor-v2-20260212.ckpt",
      convert = "tabicl", size_mb = 109,
      license = "see the repo",
      description = "TabICL v2 regressor"
    ),
    "tabfm-1.0.0-classifier" = list(
      backend = "tabfm", task = "classification", family = "tabfm-1.0.0",
      repo = "google/tabfm-1.0.0-pytorch", subfolder = "classification",
      convert = NULL, size_mb = NA_real_,
      license = "NON-COMMERCIAL (weights); Apache-2.0 (source)",
      description = "TabFM 1.0.0 classifier"
    ),
    "tabfm-1.0.0-regressor" = list(
      backend = "tabfm", task = "regression", family = "tabfm-1.0.0",
      repo = "google/tabfm-1.0.0-pytorch", subfolder = "regression",
      convert = NULL, size_mb = NA_real_,
      license = "NON-COMMERCIAL (weights); Apache-2.0 (source)",
      description = "TabFM 1.0.0 regressor"
    ),
    "mitra-classifier" = list(
      backend = "mitra", task = "classification", family = "mitra",
      repo = "autogluon/mitra-classifier", convert = NULL,
      size_mb = NA_real_, license = "Apache-2.0",
      description = "Mitra classifier"
    ),
    "mitra-regressor" = list(
      backend = "mitra", task = "regression", family = "mitra",
      repo = "autogluon/mitra-regressor", convert = NULL,
      size_mb = NA_real_, license = "Apache-2.0",
      description = "Mitra regressor"
    )
  )
}

# Resolve a user-supplied string to a catalogue id. Accepts an exact id,
# or a family name plus the task being loaded -- so `"tabpfn"` means the
# classifier under `tabular_classifier()` and the regressor under
# `tabular_regressor()`. Returns NULL when the string is not ours, which
# is how a plain path or Hub repo id passes straight through.
#
# An entry may answer to more than one family name: `"tabpfn"` stays
# pinned to v2.5 so existing code keeps loading the same weights, and
# `"tabpfn-v2.5"` / `"tabpfn-v2.6"` / `"tabpfn-v3"` name a version
# explicitly.
# @keywords internal
.catalog_id <- function(model, task = NULL) {
  if (!is.character(model) || length(model) != 1L) return(NULL)
  cat_ <- .model_catalog()
  if (model %in% names(cat_)) return(model)
  if (is.null(task)) return(NULL)
  hit <- vapply(cat_, function(e) {
    model %in% e$family && identical(e$task, task)
  }, logical(1))
  if (sum(hit) == 1L) names(cat_)[hit] else NULL
}


# ---------------------------------------------------------------------------
# The local store
# ---------------------------------------------------------------------------

#' Where downloaded models are kept
#'
#' Defaults to `tools::R_user_dir("tabfound", "cache")`, R's designated
#' per-user location for package data. Override with
#' `options(tabfound.home = )` or the `TABFOUND_HOME` environment
#' variable.
#'
#' It is deliberately not inside the installed package: a package writing
#' to its own installation directory violates CRAN policy and breaks on
#' read-only or shared libraries.
#'
#' @param create Create the directory if it does not exist.
#' @return The path, as a string.
#' @examples
#' tabfound_home()
#' @export
tabfound_home <- function(create = FALSE) {
  path <- getOption("tabfound.home")
  if (is.null(path)) path <- Sys.getenv("TABFOUND_HOME", unset = "")
  if (!nzchar(path)) path <- tools::R_user_dir("tabfound", "cache")
  path <- path.expand(path)
  if (isTRUE(create) && !dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  path
}

# @keywords internal
.model_dir <- function(id) file.path(tabfound_home(), "models", id)

# @keywords internal
.model_is_downloaded <- function(id) {
  d <- .model_dir(id)
  e <- .model_catalog()[[id]]
  sub <- e$subfolder
  rel <- function(f) if (is.null(sub)) f else file.path(sub, f)
  file.exists(file.path(d, rel("model.safetensors"))) &&
    file.exists(file.path(d, rel("config.json")))
}

#' List the models this package can download
#'
#' @return A data frame with one row per downloadable artifact set:
#'   `id`, `backend`, `task`, `downloaded`, `size_mb` (on disk once
#'   downloaded, otherwise the expected size), `license` and `repo`.
#' @examples
#' list_models()
#' @seealso [download_model()], [tabfound_home()]
#' @export
list_models <- function() {
  cat_ <- .model_catalog()
  got  <- vapply(names(cat_), .model_is_downloaded, logical(1))
  size <- vapply(names(cat_), function(id) {
    if (.model_is_downloaded(id)) {
      f <- list.files(.model_dir(id), recursive = TRUE, full.names = TRUE)
      round(sum(file.size(f)) / 1e6, 1)
    } else {
      as.numeric(cat_[[id]]$size_mb)
    }
  }, numeric(1))
  data.frame(
    id         = names(cat_),
    backend    = vapply(cat_, `[[`, character(1), "backend"),
    task       = vapply(cat_, `[[`, character(1), "task"),
    downloaded = unname(got),
    size_mb    = unname(size),
    license    = vapply(cat_, `[[`, character(1), "license"),
    repo       = vapply(cat_, `[[`, character(1), "repo"),
    row.names  = NULL, stringsAsFactors = FALSE
  )
}


# ---------------------------------------------------------------------------
# Python, for the two families that need converting
# ---------------------------------------------------------------------------

# Find an interpreter that can actually run the converter -- which means
# one with torch and safetensors importable, not merely one that exists.
# Checking now beats a traceback from a subprocess later.
# @keywords internal
.find_python <- function(python = NULL) {
  cands <- unique(Filter(nzchar, c(
    python,
    getOption("tabfound.python"),
    Sys.getenv("TABFOUND_PYTHON", unset = ""),
    Sys.getenv("TABFOUND_REF_PYTHON", unset = ""),
    ".venvs/ref/bin/python",
    unname(Sys.which("python3")),
    unname(Sys.which("python"))
  )))
  for (p in cands) {
    ok <- tryCatch(
      system2(p, c("-c", shQuote("import torch, safetensors")),
              stdout = FALSE, stderr = FALSE) == 0L,
      error = function(e) FALSE, warning = function(w) FALSE
    )
    if (isTRUE(ok)) return(p)
  }
  cli::cli_abort(c(
    "No Python interpreter with {.pkg torch} and {.pkg safetensors} was found.",
    x = "These weights are published as PyTorch pickles, which cannot be \\
         read from R -- converting them needs Python once.",
    i = "Install them ({.code pip install torch safetensors}) and point \\
         {.envvar TABFOUND_PYTHON} or {.code options(tabfound.python=)} at \\
         that interpreter.",
    i = "Tried: {.path {cands}}"
  ))
}

# @keywords internal
.converter_script <- function(which) {
  path <- tabfound_file("python", sprintf("%s_convert_ckpt.py", which))
  if (!nzchar(path)) {
    cli::cli_abort("Could not locate the {.val {which}} converter in {.path inst/python/}.")
  }
  path
}

# @keywords internal
.convert_ckpt <- function(entry, src, dest, python, quiet = FALSE) {
  script <- .converter_script(entry$convert)
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  args <- if (identical(entry$convert, "tabpfn")) {
    c(script, "--src", src,
      "--dst-weights", file.path(dest, "model.safetensors"),
      "--dst-config",  file.path(dest, "config.json"),
      "--head", entry$head)
  } else {
    c(script, "--src", src, "--dst", dest)
  }
  if (!quiet) cli::cli_alert_info("Converting with {.path {basename(python)}}...")
  out <- suppressWarnings(
    system2(python, shQuote(args), stdout = TRUE, stderr = TRUE)
  )
  status <- attr(out, "status") %||% 0L
  if (!identical(as.integer(status), 0L)) {
    cli::cli_abort(c(
      "Conversion failed.",
      x = "{.path {basename(script)}} exited with status {status}.",
      set_names(utils::tail(out, 8), rep("*", length(utils::tail(out, 8))))
    ))
  }
  invisible(out)
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

# Hard-link if we can (the Hub cache is usually on the same filesystem,
# and a 6.5 GB model should not be stored twice), copy if we cannot.
# @keywords internal
.place_file <- function(from, to) {
  dir.create(dirname(to), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(to)) unlink(to)
  ok <- suppressWarnings(tryCatch(file.link(from, to), error = function(e) FALSE))
  if (!isTRUE(ok)) {
    ok <- file.copy(from, to, overwrite = TRUE)
  }
  if (!isTRUE(ok)) cli::cli_abort("Could not place {.path {basename(from)}} into the model store.")
  invisible(to)
}

# Ask the Hub how big this really is. Best-effort: offline is normal.
# @keywords internal
.remote_size_mb <- function(entry) {
  if (!requireNamespace("hfhub", quietly = TRUE)) return(NA_real_)
  info <- tryCatch(hfhub::hub_repo_info(entry$repo, files_metadata = TRUE),
                   error = function(e) NULL)
  if (is.null(info) || is.null(info$siblings)) return(NA_real_)
  wanted <- if (!is.null(entry$file)) entry$file else {
    rel <- function(f) if (is.null(entry$subfolder)) f else
      paste(entry$subfolder, f, sep = "/")
    c(rel("model.safetensors"), rel("config.json"))
  }
  sizes <- vapply(info$siblings, function(s) {
    if (!is.null(s$rfilename) && s$rfilename %in% wanted && !is.null(s$size)) {
      as.numeric(s$size)
    } else 0
  }, numeric(1))
  total <- sum(sizes)
  if (total <= 0) NA_real_ else round(total / 1e6, 1)
}

#' Download and convert a model's weights
#'
#' Fetches a model's artifacts from the Hub into the local store
#' ([tabfound_home()]) and, for the families that publish PyTorch
#' pickles, runs the one-time conversion into
#' `model.safetensors` + `config.json`. After this the model loads with
#' no network and no Python.
#'
#' TabFM and Mitra ship readable artifacts and need no conversion.
#' TabPFN and TabICL do, and that step needs a Python with `torch` and
#' `safetensors` — the checkpoints are nested Python pickles that R's
#' torch bindings cannot read. Point `options(tabfound.python = )` or
#' `TABFOUND_PYTHON` at a suitable interpreter if the default search does
#' not find one.
#'
#' @param model A catalogue id from [list_models()], or a family name
#'   (`"tabpfn"`, `"tabpfn-v2.5"`, `"tabpfn-v2.6"`, `"tabpfn-v3"`,
#'   `"tabicl"`, `"tabfm-1.0.0"`, `"mitra"`) together with `task`.
#'   `"tabpfn"` stays pinned to v2.5; ask for a later generation by name.
#'   The catalogue carries each family's default checkpoint only; the
#'   specialised v3 variants (`_binary`, `_multiclass`, `_ood`,
#'   `_mediumdata`, `_timeseries`) can be converted by hand with
#'   `inst/python/tabpfn_convert_ckpt.py` and loaded from a local path.
#' @param task `"classification"` or `"regression"`, when `model` names a
#'   family rather than a specific artifact set.
#' @param force Re-download and re-convert even if the model is already
#'   in the store.
#' @param python Path to the interpreter used for conversion. Defaults to
#'   a search; see Details.
#' @param quiet Suppress progress messages.
#' @return The directory holding the artifacts, invisibly.
#' @examples
#' \dontrun{
#' download_model("tabpfn-v2.5-classifier")
#' download_model("tabicl", task = "regression")
#'
#' # Afterwards, loading needs neither network nor Python:
#' clf <- tabular_classifier("tabpfn-v2.5-classifier")
#' }
#' @seealso [list_models()], [tabfound_home()], [remove_model()]
#' @export
download_model <- function(model, task = NULL, force = FALSE, python = NULL,
                           quiet = FALSE) {
  id <- .catalog_id(model, task)
  if (is.null(id)) {
    cli::cli_abort(c(
      "{.val {model}} is not a downloadable model.",
      i = "See {.fn list_models} for what can be downloaded.",
      i = "For a family name, pass {.arg task} as well."
    ))
  }
  entry <- .model_catalog()[[id]]
  dest  <- .model_dir(id)

  if (.model_is_downloaded(id) && !isTRUE(force)) {
    if (!quiet) cli::cli_alert_success("{.val {id}} is already in {.path {dest}}.")
    return(invisible(dest))
  }
  require_suggested("hfhub")

  if (!quiet) cli::cli_alert_info("Downloading {.val {id}} from {.val {entry$repo}}...")
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)

  if (is.null(entry$convert)) {
    rel <- function(f) if (is.null(entry$subfolder)) f else
      file.path(entry$subfolder, f)
    for (f in c("model.safetensors", "config.json")) {
      got <- hfhub::hub_download(entry$repo, rel(f))
      .place_file(got, file.path(dest, rel(f)))
    }
  } else {
    src <- hfhub::hub_download(entry$repo, entry$file)
    py  <- .find_python(python)
    .convert_ckpt(entry, src, dest, py, quiet = quiet)
  }

  # Provenance, so a store directory can say where it came from.
  writeLines(
    jsonlite::toJSON(list(
      id = id, repo = entry$repo, file = entry$file %||% NA_character_,
      subfolder = entry$subfolder %||% NA_character_,
      converted = !is.null(entry$convert),
      tabfound_version = as.character(utils::packageVersion("tabfound")),
      downloaded_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
    ), auto_unbox = TRUE, pretty = TRUE),
    file.path(dest, "SOURCE.json")
  )

  if (!.model_is_downloaded(id)) {
    cli::cli_abort(c(
      "Download finished but {.path {dest}} has no usable artifacts.",
      i = "Expected {.path model.safetensors} and {.path config.json}."
    ))
  }
  if (!quiet) {
    mb <- round(sum(file.size(list.files(dest, recursive = TRUE,
                                         full.names = TRUE))) / 1e6, 1)
    cli::cli_alert_success("{.val {id}} ready in {.path {dest}} ({mb} MB).")
  }
  invisible(dest)
}

#' Delete a downloaded model
#'
#' @param model A catalogue id from [list_models()].
#' @param quiet Suppress the confirmation message.
#' @return The removed directory, invisibly.
#' @export
remove_model <- function(model, quiet = FALSE) {
  id <- .catalog_id(model)
  if (is.null(id)) cli::cli_abort("{.val {model}} is not a catalogue model id.")
  d <- .model_dir(id)
  if (!dir.exists(d)) {
    if (!quiet) cli::cli_alert_info("{.val {id}} is not downloaded.")
    return(invisible(d))
  }
  unlink(d, recursive = TRUE)
  if (!quiet) cli::cli_alert_success("Removed {.path {d}}.")
  invisible(d)
}


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------

# Ask before pulling hundreds of megabytes (or, for TabFM, gigabytes)
# over someone's network. Three policies, via
# `options(tabfound.download=)`:
#
#   "ask"    (default) prompt when interactive, refuse otherwise
#   "always" download without asking -- for scripts that want it
#   "never"  refuse, and say what to run
#
# Refusing in a non-interactive session is deliberate. A prompt nobody
# can answer would either hang a batch job or, worse, be silently
# defaulted to yes.
# @keywords internal
.prompt_download <- function(id, entry) {
  policy <- getOption("tabfound.download", "ask")
  hint <- sprintf('download_model("%s")', id)

  if (identical(policy, "never")) {
    cli::cli_abort(c(
      "{.val {id}} is not downloaded, and {.code options(tabfound.download = \"never\")} is set.",
      i = "Run {.code {hint}} to fetch it."
    ))
  }
  if (identical(policy, "always")) return(TRUE)

  size <- .remote_size_mb(entry)
  if (is.na(size)) size <- as.numeric(entry$size_mb)
  size_txt <- if (is.na(size)) "size unknown"
              else if (size >= 1000) sprintf("~%.1f GB", size / 1000)
              else sprintf("~%.0f MB", size)

  if (!interactive()) {
    cli::cli_abort(c(
      "{.val {id}} is not downloaded.",
      i = "This session is not interactive, so I will not download \\
           {size_txt} without being asked to.",
      i = "Run {.code {hint}} first, or set \\
           {.code options(tabfound.download = \"always\")}."
    ))
  }

  cli::cli_inform(c(
    "!" = "{.val {id}} is not downloaded yet.",
    "*" = "source: {.val {entry$repo}}",
    "*" = "size: {size_txt}",
    "*" = "licence: {.emph {entry$license}}",
    "*" = "destination: {.path {.model_dir(id)}}",
    if (!is.null(entry$convert))
      c("*" = "needs a one-time conversion step, which requires Python \\
               with {.pkg torch} + {.pkg safetensors}")
  ))
  isTRUE(utils::askYesNo("Download it now?", default = FALSE))
}

# The hook `load_backend_model()` calls. Turns a catalogue id or family
# name into a local directory, downloading it first if the user agrees.
# Anything not in the catalogue falls through to the alias table, so
# paths and Hub repo ids behave exactly as before.
# @keywords internal
resolve_model_source <- function(model, task = NULL) {
  id <- .catalog_id(model, task)
  if (is.null(id)) return(expand_model_alias(model))
  if (.model_is_downloaded(id)) return(.model_dir(id))
  if (!.prompt_download(id, .model_catalog()[[id]])) {
    cli::cli_abort(c(
      "Cannot load {.val {id}}: the weights are not on this machine.",
      i = "Run {.code download_model(\"{id}\")} when you are ready."
    ))
  }
  download_model(id)
  .model_dir(id)
}
