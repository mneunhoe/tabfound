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
    # TabPFN v3.5. Two things set these apart from every entry above.
    #
    # `task = "both"`: from v3.5 on, one checkpoint serves classification
    # and regression. The head is chosen per forward pass rather than
    # baked into the weights, so splitting this into a classifier id and a
    # regressor id would download the same 876 MB twice. `.catalog_id()`
    # matches `"both"` against either task.
    #
    # `convert = "tabpfn35"`: the publisher ships safetensors, not a
    # pickled `.ckpt`, with the real config in the file's own
    # `__metadata__` header -- the repo's top-level `config.json` holds
    # only the model name. So the converter lifts the header out and
    # copies the tensors rather than round-tripping them through torch,
    # and there is no `head` to cross-check.
    "tabpfn-v3.5" = list(
      backend = "tabpfn35", task = "both", family = "tabpfn-v3.5",
      repo = "Prior-Labs/tabpfn_3_5",
      file = "tabpfn-v3.5-20260909.safetensors",
      convert = "tabpfn35", size_mb = 876,
      license = "tabpfn-3-5-license-v1.0: research, evaluation and internal benchmarking; production use needs a commercial licence (see the repo)",
      description = "TabPFN v3.5, multitask (classification and regression)"
    ),
    # As with the specialised v3 checkpoints, the two variants carry no
    # `family`, so `"tabpfn-v3.5"` keeps meaning the default and these are
    # reachable by exact id only. Both are the same architecture as the
    # default and load through `tabpfn35`; only the config differs.
    "tabpfn-v3.5-fast" = list(
      backend = "tabpfn35", task = "both",
      repo = "Prior-Labs/tabpfn_3_5",
      file = "tabpfn-v3.5-fast-20260909.safetensors",
      convert = "tabpfn35", size_mb = 334,
      license = "tabpfn-3-5-license-v1.0: research, evaluation and internal benchmarking; production use needs a commercial licence (see the repo)",
      description = "TabPFN v3.5 Fast (alpha), 8 ICL layers instead of 24"
    ),
    "tabpfn-v3.5-multiclass" = list(
      backend = "tabpfn35", task = "both",
      repo = "Prior-Labs/tabpfn_3_5",
      file = "tabpfn-v3.5-20260909_multiclass.safetensors",
      convert = "tabpfn35", size_mb = 876,
      license = "tabpfn-3-5-license-v1.0: research, evaluation and internal benchmarking; production use needs a commercial licence (see the repo)",
      description = "TabPFN v3.5, experimental multiclass variant (4 aggregation heads)"
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
# `"tabpfn-v2.5"` / `"tabpfn-v2.6"` / `"tabpfn-v3"` / `"tabpfn-v3.5"` name
# a version explicitly.
#
# An entry whose `task` is `"both"` answers to either task: from TabPFN
# v3.5 on, one checkpoint serves classification and regression, so the
# same artifacts are what `tabular_classifier()` and `tabular_regressor()`
# each resolve to.
# @keywords internal
.catalog_id <- function(model, task = NULL) {
  if (!is.character(model) || length(model) != 1L) return(NULL)
  cat_ <- .model_catalog()
  if (model %in% names(cat_)) return(model)
  if (is.null(task)) return(NULL)
  hit <- vapply(cat_, function(e) {
    model %in% e$family &&
      (identical(e$task, task) || identical(e$task, "both"))
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

# What a completed download left behind, keyed by path relative to the
# model directory. Recorded in `SOURCE.json` so an interrupted download
# can be told from a finished one.
# @keywords internal
.artifact_sizes <- function(dest) {
  rel <- list.files(dest, recursive = TRUE)
  rel <- rel[basename(rel) != "SOURCE.json"]
  if (!length(rel)) return(list())
  stats::setNames(as.list(as.numeric(file.size(file.path(dest, rel)))), rel)
}

# @keywords internal
.model_is_downloaded <- function(id) {
  d <- .model_dir(id)
  e <- .model_catalog()[[id]]
  sub <- e$subfolder
  rel <- function(f) if (is.null(sub)) f else file.path(sub, f)
  if (!file.exists(file.path(d, rel("model.safetensors"))) ||
      !file.exists(file.path(d, rel("config.json")))) {
    return(FALSE)
  }
  # Names alone pass forever once an interrupted download has created the
  # files: a truncated `model.safetensors` is still a `model.safetensors`.
  # Sizes are recorded on completion, so compare against them when they
  # are there (they are not, for stores written before this check).
  src <- file.path(d, "SOURCE.json")
  if (!file.exists(src) || !requireNamespace("jsonlite", quietly = TRUE)) {
    return(TRUE)
  }
  recorded <- tryCatch(jsonlite::fromJSON(src)$files, error = function(e) NULL)
  if (is.null(recorded) || !length(recorded)) return(TRUE)
  got <- .artifact_sizes(d)
  for (nm in names(recorded)) {
    want <- as.numeric(recorded[[nm]])
    have <- got[[nm]]
    if (is.null(have) || !isTRUE(all.equal(as.numeric(have), want))) return(FALSE)
  }
  TRUE
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

# ---------------------------------------------------------------------------
# Hub access: tokens, offline, and what a 401 actually means
# ---------------------------------------------------------------------------
#
# Everything that reaches the Hub goes through `.hub_download()`. It
# exists for three failures that all present as something other than what
# they are.

# `hfhub::hub_headers()` reads `HUGGING_FACE_HUB_TOKEN` and
# `HUGGINGFACE_HUB_TOKEN`, and nothing else. The Python ecosystem has
# moved on: `HF_TOKEN` is what the docs tell you to set, and
# `huggingface-cli login` writes a file rather than an environment
# variable at all. So access granted through the normal flow is invisible
# from R, and a gated repo 401s -- which hfhub reports as "Connection
# error... cannot find the requested files in the disk cache".
# @keywords internal
.hf_token <- function() {
  for (v in c("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")) {
    tok <- Sys.getenv(v, unset = "")
    if (nzchar(tok)) return(tok)
  }
  home <- Sys.getenv("HF_HOME", unset = "")
  files <- c(if (nzchar(home)) file.path(home, "token"),
             path.expand("~/.cache/huggingface/token"))
  for (p in files) {
    if (file.exists(p)) {
      tok <- trimws(paste(readLines(p, warn = FALSE), collapse = ""))
      if (nzchar(tok)) return(tok)
    }
  }
  ""
}

# Put the resolved token where hfhub will look, for one call only.
# @keywords internal
.with_hf_token <- function(expr) {
  tok <- .hf_token()
  if (nzchar(tok) && !nzchar(Sys.getenv("HUGGING_FACE_HUB_TOKEN", unset = ""))) {
    old <- Sys.getenv("HUGGING_FACE_HUB_TOKEN", unset = NA_character_)
    Sys.setenv(HUGGING_FACE_HUB_TOKEN = tok)
    on.exit({
      if (is.na(old)) Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
      else Sys.setenv(HUGGING_FACE_HUB_TOKEN = old)
    }, add = TRUE)
  }
  force(expr)
}

# @keywords internal
.hf_offline <- function() {
  opt <- getOption("tabfound.offline", NULL)
  if (!is.null(opt)) return(isTRUE(opt))
  env <- Sys.getenv("HF_HUB_OFFLINE", unset = "")
  nzchar(env) && !identical(env, "0") && !identical(tolower(env), "false")
}

#' Fetch one file from the Hub, cache-first
#'
#' Every Hub-referenced load pays a network round trip before consulting
#' the cache -- a HEAD when online, a timeout when not -- even though the
#' file is already on disk. So ask the cache first and only go to the
#' network when it misses. `options(tabfound.offline = TRUE)` (or
#' `HF_HUB_OFFLINE`) says never to go at all.
#'
#' @param repo Hub repo id.
#' @param file Path within the repo.
#' @return A local path.
#' @keywords internal
.hub_download <- function(repo, file) {
  require_suggested("hfhub")
  cached <- tryCatch(
    hfhub::hub_download(repo, file, local_files_only = TRUE),
    error = function(e) NULL
  )
  if (!is.null(cached) && file.exists(cached)) return(cached)

  if (.hf_offline()) {
    cli::cli_abort(c(
      "{.path {file}} from {.val {repo}} is not in the Hub cache, and \\
       tabfound is in offline mode.",
      i = "Unset {.envvar HF_HUB_OFFLINE} / \\
           {.code options(tabfound.offline = FALSE)} to fetch it.",
      i = "The cache is {.path {Sys.getenv('HUGGINGFACE_HUB_CACHE',
                                           '~/.cache/huggingface/hub')}}."
    ))
  }
  .with_hf_token(tryCatch(
    hfhub::hub_download(repo, file),
    error = function(e) .hub_download_abort(e, repo, file)
  ))
}

# hfhub reports an authorization failure as a cache miss, which sends
# everybody looking in the wrong place. Say what it is and what to do.
# @keywords internal
.hub_download_abort <- function(e, repo, file) {
  msg <- conditionMessage(e)
  looks_gated <- grepl("401|403|cannot find the requested files|Connection error",
                       msg, ignore.case = TRUE)
  if (!looks_gated) {
    cli::cli_abort(c("Could not download {.path {file}} from {.val {repo}}.",
                     x = msg))
  }
  have_token <- nzchar(.hf_token())
  cli::cli_abort(c(
    "Could not download {.path {file}} from {.val {repo}}.",
    x = "This usually means the repo is gated and the request was \\
         unauthorized -- hfhub reports that as a cache miss.",
    i = if (have_token)
          "A token was found and sent. Accept the model's terms at \\
           {.url https://huggingface.co/{repo}} with the same account."
        else
          "No token was found. Set {.envvar HF_TOKEN}, or run \\
           {.code huggingface-cli login}, after accepting the terms at \\
           {.url https://huggingface.co/{repo}}.",
    i = "Or convert a checkpoint you already have locally: \\
         {.path inst/python/tabpfn_convert_ckpt.py}.",
    x = msg
  ))
}


# Ask the Hub how big this really is. Best-effort: offline is normal.
# @keywords internal
.remote_size_mb <- function(entry) {
  if (.hf_offline()) return(NA_real_)
  if (!requireNamespace("hfhub", quietly = TRUE)) return(NA_real_)
  info <- .with_hf_token(tryCatch(
    hfhub::hub_repo_info(entry$repo, files_metadata = TRUE),
    error = function(e) NULL
  ))
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
#' TabPFN up to v3 and TabICL do, and that step needs a Python with
#' `torch` and `safetensors` — the checkpoints are nested Python pickles
#' that R's torch bindings cannot read. TabPFN v3.5 is published as
#' safetensors, so its conversion only lifts the config out of the file's
#' header and needs `safetensors` alone. Point
#' `options(tabfound.python = )` or `TABFOUND_PYTHON` at a suitable
#' interpreter if the default search does not find one.
#'
#' @param model A catalogue id from [list_models()], or a family name
#'   (`"tabpfn"`, `"tabpfn-v2.5"`, `"tabpfn-v2.6"`, `"tabpfn-v3"`,
#'   `"tabpfn-v3.5"`, `"tabicl"`, `"tabfm-1.0.0"`, `"mitra"`) together
#'   with `task`. `"tabpfn"` stays pinned to v2.5; ask for a later
#'   generation by name. A family name resolves to that family's default
#'   checkpoint; the specialised variants are catalogued too, but by exact
#'   id only — the v3 ones as `tabpfn-v3-classifier-binary`,
#'   `-multiclass`, `-ood`, `tabpfn-v3-regressor-mediumdata`, `-ood`,
#'   `-timeseries`, and the v3.5 ones as `tabpfn-v3.5-fast` and
#'   `tabpfn-v3.5-multiclass`.
#'
#'   `task` is ignored for TabPFN v3.5, whose single multitask checkpoint
#'   serves both.
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
      got <- .hub_download(entry$repo, rel(f))
      .place_file(got, file.path(dest, rel(f)))
    }
  } else {
    src <- .hub_download(entry$repo, entry$file)
    py  <- .find_python(python)
    .convert_ckpt(entry, src, dest, py, quiet = quiet)
  }

  # Provenance, so a store directory can say where it came from -- plus
  # the file sizes, which is what makes "is it downloaded?" answerable
  # after an interrupted run.
  writeLines(
    jsonlite::toJSON(list(
      id = id, repo = entry$repo, file = entry$file %||% NA_character_,
      subfolder = entry$subfolder %||% NA_character_,
      converted = !is.null(entry$convert),
      tabfound_version = as.character(utils::packageVersion("tabfound")),
      downloaded_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
      files = .artifact_sizes(dest)
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
