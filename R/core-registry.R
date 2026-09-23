# Backend registry.
#
# A backend packages everything model-specific behind a fixed interface,
# so `tabular_classifier()` / `tabular_regressor()` never branch on the
# model family. Adding a new open-weight model means writing one
# `register_backend()` call — no changes to the core.

.tabfound_backends <- new.env(parent = emptyenv())

#' Register a model backend
#'
#' @param name Character. Unique backend id, e.g. `"tabpfn"`.
#' @param build Function `(config, task)` returning a constructed
#'   `nn_module` with randomly initialised weights.
#' @param translate_key Function mapping a checkpoint key to the R
#'   `named_parameters()` path. Defaults to identity.
#' @param detect Function `(config)` returning `TRUE` when this backend
#'   recognises the config. Used to infer the backend when the caller
#'   does not name one.
#' @param task_of Function `(config)` returning `"classification"` or
#'   `"regression"`.
#' @param subfolder_for Function `(task)` returning the artifact
#'   subfolder for a task, or `NULL`.
#' @param classifier,regressor Functions `(ctx, ...)` that take the
#'   loaded-model context (see [load_backend_model()]) and return the
#'   user-facing predictor object.
#' @param describe Optional function `(config, task)` returning
#'   `list(title, subtitle, facts, symbols, stages)`, where `stages` is a
#'   list of [arch_stage()]s. This is what [tabfound_architecture()] and
#'   [plot_architecture()] draw; a backend without one cannot be
#'   diagrammed.
#' @param peak_terms Optional function
#'   `(n_context, n_query, n_features, opts, config)` returning the shapes
#'   that decide peak memory, in tensor *elements* rather than bytes:
#'   `list(stages = list(list(name, act, att), ...), persistent, n_estimators)`.
#'   The core turns those into bytes with fitted per-backend constants --
#'   see [estimate_peak_memory()] and `inst/memory/`. `act` is a stage's
#'   largest activation, `att` the attention scores it materialises; the
#'   peak is the maximum over stages, never the sum, because only one is
#'   live at a time. A backend without this cannot be preflighted.
#' @param aliases Named character vector mapping user-friendly model
#'   shorthands to local paths or Hub repo ids.
#' @param description One-line human description.
#' @param kv_cache_capable Logical. Can this backend hand back a
#'   conditioned state for the training rows that a later `predict()` can
#'   start from? That is what [tabfound_cache()] persists. An
#'   architecture that re-fits statistics over train and test together
#'   has nothing to hand over.
#' @param handles_missing Logical. Does the **network** tolerate
#'   `NA`/`NaN` in the predictors? TabPFN encodes them explicitly, as an
#'   is-missing channel beside the value, and TabFM maps them to a
#'   sentinel. TabICL propagates them to `NaN` output and Mitra silently
#'   collapses the column, so for those it is `FALSE`.
#' @param imputes_internally Logical. Does the **predictor** fill them in
#'   before the network sees them? TabICL, Mitra and TabFM all run their
#'   reference wrapper's mean imputer as the first preprocessing step.
#'
#'   The two flags used to be one, which conflated guarantees that are
#'   not the same. Either means [tabfound()] need not impute, so
#'   `na_action = "auto"` reads their disjunction; but a network that
#'   *conditions on* missingness treats it as information, while a
#'   wrapper that mean-fills has thrown that information away before the
#'   model is reached. For imputation and synthesis, where the
#'   missingness pattern is the object of study, that difference is the
#'   whole question -- see [tabfound_impute()].
#' @param parity Optional character. Reference Python package + version
#'   this backend has been checked against.
#' @return Invisibly, the backend spec.
#' @export
register_backend <- function(name,
                             build,
                             translate_key = identity,
                             detect = function(config) FALSE,
                             task_of = function(config) NULL,
                             subfolder_for = function(task) NULL,
                             classifier = NULL,
                             regressor = NULL,
                             describe = NULL,
                             peak_terms = NULL,
                             aliases = character(),
                             description = "",
                             kv_cache_capable = FALSE,
                             handles_missing = FALSE,
                             imputes_internally = FALSE,
                             parity = NA_character_) {
  spec <- list(
    name = name, build = build, translate_key = translate_key,
    detect = detect, task_of = task_of, subfolder_for = subfolder_for,
    classifier = classifier, regressor = regressor, describe = describe,
    peak_terms = peak_terms,
    aliases = aliases, description = description,
    kv_cache_capable = isTRUE(kv_cache_capable),
    handles_missing = isTRUE(handles_missing),
    imputes_internally = isTRUE(imputes_internally), parity = parity
  )
  class(spec) <- "tabfound_backend"
  assign(name, spec, envir = .tabfound_backends)
  invisible(spec)
}

#' List registered backends
#'
#' @return A data.frame with one row per backend. `missing` says what
#'   happens to `NA` predictors: `"encoded"` when the network conditions
#'   on missingness itself, `"imputed"` when the predictor fills it in
#'   first (the reference wrapper's mean imputer), `"encoded+imputed"`
#'   when both, and `"unhandled"` when neither -- see
#'   [register_backend()] for why the two are not the same guarantee.
#' @export
list_backends <- function() {
  nms <- ls(.tabfound_backends)
  if (length(nms) == 0L) {
    return(data.frame(name = character(), description = character(),
                      missing = character(), parity = character(),
                      stringsAsFactors = FALSE))
  }
  specs <- lapply(nms, get_backend)
  data.frame(
    name        = vapply(specs, `[[`, character(1), "name"),
    description = vapply(specs, `[[`, character(1), "description"),
    missing     = vapply(specs, function(s) .missing_handling(s), character(1)),
    parity      = vapply(specs, function(s) as.character(s$parity), character(1)),
    stringsAsFactors = FALSE
  )
}

# @keywords internal
.missing_handling <- function(spec) {
  net <- isTRUE(spec$handles_missing)
  wrp <- isTRUE(spec$imputes_internally)
  if (net && wrp) "encoded+imputed" else if (net) "encoded" else
    if (wrp) "imputed" else "unhandled"
}

# Whether `tabfound()` needs to impute before handing data over: either
# route means the network never sees an `NA`.
# @keywords internal
.backend_covers_missing <- function(spec) {
  isTRUE(spec$handles_missing) || isTRUE(spec$imputes_internally)
}

#' Retrieve a registered backend by name
#' @param name Character backend id.
#' @export
get_backend <- function(name) {
  if (!exists(name, envir = .tabfound_backends, inherits = FALSE)) {
    known <- ls(.tabfound_backends)
    cli::cli_abort(c(
      "Unknown backend {.val {name}}.",
      i = "Registered backends: {.val {known}}."
    ))
  }
  get(name, envir = .tabfound_backends, inherits = FALSE)
}

#' Expand a model alias to a path or Hub repo id
#'
#' Checks every backend's alias table. Returns the input unchanged when
#' it is not an alias.
#' @keywords internal
expand_model_alias <- function(model) {
  for (nm in ls(.tabfound_backends)) {
    al <- get_backend(nm)$aliases
    if (length(al) && model %in% names(al)) return(unname(al[[model]]))
  }
  model
}

#' Infer which backend recognises a config
#' @keywords internal
detect_backend <- function(config) {
  nms <- ls(.tabfound_backends)
  hits <- Filter(function(nm) isTRUE(get_backend(nm)$detect(config)), nms)
  if (length(hits) == 1L) return(get_backend(hits))
  if (length(hits) == 0L) {
    cli::cli_abort(c(
      "No registered backend recognised this {.file config.json}.",
      i = "Registered backends: {.val {nms}}.",
      i = "Pass {.arg backend} explicitly to override detection."
    ))
  }
  cli::cli_abort(c(
    "{length(hits)} backends claim this config: {.val {hits}}.",
    i = "Pass {.arg backend} explicitly to disambiguate."
  ))
}


#' Resolve a model's artifacts, config and backend, without loading weights
#'
#' The first half of [load_backend_model()], split out so a caller can ask
#' what a checkpoint *is* -- its backend, and the `inference_config` recipe
#' it carries -- before committing to building it. [tabfound()] needs that
#' to decide whether text and date columns are expanded, and it has to
#' decide before it can say which columns are categorical, which it must
#' say before the model is constructed.
#'
#' Reads only `config.json`. A catalogue id still resolves to the local
#' store, asking to download first if needed, exactly as loading would --
#' so nothing is fetched here that loading would not fetch a moment later.
#'
#' @inheritParams load_backend_model
#' @return `list(paths, config, backend)`.
#' @keywords internal
.resolve_backend_config <- function(model, task, backend = NULL, subfolder = NULL) {
  # A catalogue id or family name resolves to the local store, asking
  # first if the weights are not there yet. Everything else -- a path, a
  # Hub repo id, a registered alias -- passes straight through.
  model  <- resolve_model_source(model, task)

  # The subfolder may depend on the backend, and the backend may need
  # the config — which lives inside the subfolder. Resolve in two steps:
  # if the caller named a backend we can ask it directly; otherwise try
  # the top level first, then each backend's subfolder guess.
  bk <- if (!is.null(backend)) get_backend(backend) else NULL
  sub <- subfolder %||% (if (!is.null(bk)) bk$subfolder_for(task) else NULL)

  paths <- tryCatch(
    resolve_artifacts(model, subfolder = sub),
    error = function(e) {
      if (!is.null(sub) || !is.null(bk)) stop(e)
      # Try each registered backend's subfolder convention.
      for (nm in ls(.tabfound_backends)) {
        cand <- get_backend(nm)$subfolder_for(task)
        if (is.null(cand)) next
        got <- tryCatch(resolve_artifacts(model, subfolder = cand),
                        error = function(e2) NULL)
        if (!is.null(got)) return(got)
      }
      stop(e)
    }
  )

  cli::cli_alert_info("Reading configuration...")
  config <- read_model_config(paths$config)

  if (is.null(bk)) bk <- detect_backend(config)

  cfg_task <- bk$task_of(config)
  if (!is.null(cfg_task) && !identical(cfg_task, task)) {
    cli::cli_abort(c(
      "These artifacts describe a {.val {cfg_task}} model, not a {.val {task}} one.",
      i = "Use {.fn {if (cfg_task == 'classification') 'tabular_classifier' else 'tabular_regressor'}} instead."
    ))
  }

  list(paths = paths, config = config, backend = bk)
}

#' Build and weight-load a model through its backend
#'
#' Shared plumbing for [tabular_classifier()] and [tabular_regressor()]:
#' resolve artifacts, read the config, pick a backend, construct the
#' network, copy the weights in, move to device, set eval mode.
#'
#' @param model Local directory, Hub repo id, or registered alias.
#' @param task `"classification"` or `"regression"`.
#' @param backend Optional backend name; inferred from the config when
#'   `NULL`.
#' @param device Device string.
#' @param subfolder Optional artifact subfolder, overriding the
#'   backend's default for this task.
#' @return A list with `net`, `config`, `backend`, `device`, `task` and
#'   `model_ref` (the resolved model string, kept so a saved model can be
#'   rebuilt -- see [tabfound_save()]).
#' @keywords internal
load_backend_model <- function(model, task, backend = NULL,
                               device = "cpu", subfolder = NULL) {
  device <- resolve_device(device)
  # Keep the caller's original string: it is what `tabfound_save()`
  # records, so a reloaded model re-resolves through the same path rather
  # than hard-coding today's cache location.
  model_ref <- model
  resolved <- .resolve_backend_config(model, task, backend, subfolder)
  paths <- resolved$paths
  config <- resolved$config
  bk <- resolved$backend

  cli::cli_alert_info("Loading weights ({.val {bk$name}} backend)...")
  weights <- read_safetensors(paths$weights)

  net <- bk$build(config, task)
  # Converted checkpoints record the ordered key list in the config;
  # checkpoints published as safetensors directly (TabFM) do not, so
  # fall back to the file's own keys.
  keys <- config$state_dict_keys %||% names(weights)
  load_state_dict(net, weights, keys, translate_fn = bk$translate_key)

  # `set_data()` has copied everything wanted out of `weights`, but the
  # list is still a live reference to a second full copy of the
  # checkpoint -- 6.5 GB for TabFM -- and R will not free it until
  # something collects. Do that before moving the network to the device,
  # which is the other peak. Costs a few milliseconds against a load that
  # already took seconds.
  rm(weights)
  gc(full = FALSE)

  net$to(device = device)
  net$eval()
  # Inference only: nothing here ever calls backward, and an unwrapped
  # forward pass would otherwise quietly build an autograd graph over
  # every activation of a multi-gigabyte network. `with_no_grad` at the
  # call sites is the belt; this is the braces.
  for (p in net$parameters) p$requires_grad_(FALSE)

  list(net = net, config = config, backend = bk, device = device, task = task,
       model_ref = model_ref)
}
