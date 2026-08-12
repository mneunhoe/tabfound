# Persisting a fitted context's KV cache.
#
# Fitting one of these models stores rows; the expensive part of a
# prediction is conditioning the network on them, and `kv_cache = TRUE`
# already does that once per `predict()` instead of once per chunk. What
# it cannot do is survive the call -- so every session, every script run,
# every restart pays the same conditioning cost again.
#
# The stated reason was that torch tensors do not survive `saveRDS()`.
# True, and irrelevant: safetensors is already a dependency, and a KV
# cache is a shallow tree of tensors with a few integers attached.
#
# So the split here is by what each format is good at. Tensors go to
# `.safetensors`, which needs a flat `name -> tensor` map, so the tree is
# walked into dotted paths (`kv.3.key`). Everything else -- the list
# structure itself, the class attributes, `n_train`, `n_features` -- is
# ordinary R data and goes in the RDS as a *skeleton*: the same tree with
# each tensor replaced by its path. Grafting the two back together
# restores the object, classes and all.

# The sentinel that marks a tensor's place in the skeleton. Prefixed so
# it cannot collide with a string a backend might legitimately store.
.KV_TENSOR_TAG <- "<tabfound:tensor>"

# @keywords internal
.kv_flatten <- function(x, prefix = "kv") {
  tensors <- list()
  walk <- function(node, path) {
    if (inherits(node, "torch_tensor")) {
      tensors[[path]] <<- node
      return(structure(path, class = "tabfound_tensor_ref"))
    }
    if (is.list(node)) {
      nms <- names(node)
      out <- lapply(seq_along(node), function(i) {
        key <- if (!is.null(nms) && nzchar(nms[[i]])) nms[[i]] else as.character(i)
        walk(node[[i]], paste(path, key, sep = "."))
      })
      names(out) <- nms
      # Classes and any other attributes are plain R data; keep them so a
      # restored cache is the same S3 object the backend built.
      attributes(out) <- c(attributes(out), attributes(node)[
        setdiff(names(attributes(node)), c("names"))])
      return(out)
    }
    node
  }
  skeleton <- walk(x, prefix)
  list(skeleton = skeleton, tensors = tensors)
}

# @keywords internal
.kv_graft <- function(skeleton, tensors) {
  walk <- function(node) {
    if (inherits(node, "tabfound_tensor_ref")) {
      path <- as.character(node)
      t <- tensors[[path]]
      if (is.null(t)) {
        cli::cli_abort(c(
          "The saved cache is missing the tensor {.field {path}}.",
          i = "The bundle's {.file .safetensors} file does not match its \\
               {.file state.rds}; re-save the model."
        ))
      }
      return(t)
    }
    if (is.list(node)) {
      out <- lapply(node, walk)
      attributes(out) <- c(attributes(out), attributes(node)[
        setdiff(names(attributes(node)), c("names"))])
      return(out)
    }
    node
  }
  walk(skeleton)
}

# safetensors keys are flat strings; a member index has to live in the
# key rather than in a directory.
# @keywords internal
.kv_write <- function(caches, dir) {
  require_suggested("safetensors")
  flat <- .kv_flatten(caches, "kv")
  if (!length(flat$tensors)) return(list(skeleton = flat$skeleton, file = NA_character_))
  file <- file.path(dir, "cache.safetensors")
  # `$clone()`, not just `$contiguous()`. A row slice of a bigger tensor
  # is already contiguous -- it just starts partway into the storage --
  # and safetensors writes from the start of the storage, so such a
  # tensor comes back holding its neighbour's numbers at the right shape.
  # Silent, and it cost an afternoon: exactly one tensor of a TabPFN
  # cache (`test_y_embedding`) is that kind of slice. Cloning gives every
  # tensor its own storage at offset zero.
  tensors <- lapply(flat$tensors,
                    function(t) t$detach()$cpu()$contiguous()$clone())
  safetensors::safe_save_file(tensors, file)
  list(skeleton = flat$skeleton, file = basename(file))
}

# @keywords internal
.kv_read <- function(skeleton, file, device = "cpu") {
  if (is.null(skeleton)) return(NULL)
  if (is.na(file) || !file.exists(file)) {
    cli::cli_abort(c(
      "This bundle records a KV cache but {.path {basename(file)}} is missing.",
      i = "Save the model again, or load it without the cache."
    ))
  }
  require_suggested("safetensors")
  tensors <- safetensors::safe_load_file(file, framework = "torch")
  tensors <- tensors[vapply(tensors, function(x) inherits(x, "torch_tensor"),
                            logical(1))]
  if (!identical(device, "cpu")) {
    tensors <- lapply(tensors, function(t) t$to(device = device))
  }
  .kv_graft(skeleton, tensors)
}


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

# A cache is a function of the training context, and nothing about a
# tensor says which context it came from. A cache used against the wrong
# one produces numbers rather than an error -- the shapes still line up
# whenever the feature count does -- so the fingerprint is checked, not
# assumed.
# @keywords internal
.kv_guard_of <- function(state) {
  X <- state$X_train %||% state$X_support
  # Not every backend keeps the training matrix on the state -- TabICL
  # and TabFM keep a fitted imputer and a member generator instead. What
  # they all keep is *something* derived from the training rows and made
  # of plain data, which is what the fingerprint is taken over.
  parts <- list(X,
                state$y_train_int %||% state$y_train %||% state$y,
                state$imputer$means, state$imputer$keep,
                state$class_levels, state$y_mean, state$y_std)
  list(
    n_train    = as.integer(state$n_train %||% NROW(X) %||% 0L),
    n_features = as.integer(if (!is.null(X)) NCOL(X)
                            else length(state$imputer$keep) %||% 0L),
    digest     = .context_digest(parts)
  )
}

# `digest` is Suggests, so fall back to something cheap and still
# sensitive to a changed context rather than refusing to check at all.
# @keywords internal
.context_digest <- function(parts) {
  parts <- Filter(Negate(is.null), parts)
  if (!length(parts)) return(NA_character_)
  flat <- lapply(parts, function(p) {
    if (is.numeric(p) || is.logical(p)) as.numeric(p) else as.character(p)
  })
  if (requireNamespace("digest", quietly = TRUE)) {
    return(digest::digest(flat, algo = "xxhash64"))
  }
  num <- unlist(Filter(is.numeric, flat), use.names = FALSE)
  if (!length(num)) return(NA_character_)
  paste(length(num), sum(num, na.rm = TRUE),
        sum(num * seq_along(num), na.rm = TRUE), sep = "-")
}

# @keywords internal
.kv_guard_check <- function(state) {
  want <- state$kv_guard
  if (is.null(want) || is.null(state$kv_caches)) return(invisible(TRUE))
  got <- .kv_guard_of(state)
  same <- identical(want$n_train, got$n_train) &&
    identical(want$n_features, got$n_features) &&
    (is.na(want$digest) || is.na(got$digest) ||
       identical(want$digest, got$digest))
  if (!same) {
    cli::cli_abort(c(
      "This model's cached context does not match its training rows.",
      x = "The cache was built on {want$n_train} row{?s} x \\
           {want$n_features} feature{?s}; the model now holds \\
           {got$n_train} x {got$n_features}.",
      i = "Rebuild it with {.fn tabfound_cache}, or drop it with \\
           {.code tabfound_cache(object, build = FALSE)}."
    ))
  }
  invisible(TRUE)
}


# ---------------------------------------------------------------------------
# The user-facing half
# ---------------------------------------------------------------------------

#' Precompute a fitted model's KV cache
#'
#' Conditioning the network on the training rows is the expensive half of
#' a prediction, and it does not depend on what you are predicting. With
#' `kv_cache = TRUE` a backend already does it once per `predict()` call
#' rather than once per chunk; this does it once and *keeps* it, on the
#' model object, so every later call -- and, via [tabfound_save()], every
#' later session -- starts from the conditioned state.
#'
#' The cache is a function of the fitted context. It is fingerprinted
#' against those rows when it is built and checked when it is used, so a
#' cache cannot silently outlive the data it came from.
#'
#' Not every backend has one: the cache is what a network can carry
#' forward from the training rows, and an architecture that re-fits
#' statistics over train and test together has nothing to hand over. Ask
#' for one where there is none and you get an error saying so.
#'
#' @param object A fitted `tabfound_model`.
#' @param build `TRUE` (default) to build the cache, `FALSE` to drop one
#'   the object already carries.
#' @return A copy of `object` carrying (or no longer carrying) the cache.
#' @seealso [tabfound_save()], which writes the cache alongside the state.
#' @examples
#' \dontrun{
#' clf <- fit(tabular_classifier("path/to/model", kv_cache = TRUE), X, y)
#' clf <- tabfound_cache(clf)
#' tabfound_save(clf, "clf-cached")     # a directory bundle
#' }
#' @export
tabfound_cache <- function(object, build = TRUE) {
  if (!inherits(object, "tabfound_model")) {
    cli::cli_abort("{.arg object} must be a {.cls tabfound_model}.")
  }
  .require_fitted(object)
  if (!isTRUE(build)) {
    object$state$kv_caches <- NULL
    object$state$kv_guard  <- NULL
    return(object)
  }
  builder <- object$spec$build_cache
  if (!is.function(builder)) {
    capable <- .kv_capable_backends()
    cli::cli_abort(c(
      "The {.val {object$backend}} backend cannot precompute a KV cache.",
      i = "Backends that can: {.val {capable}}."
    ))
  }
  caches <- builder(object$state)
  if (is.null(caches)) {
    cli::cli_abort(c(
      "The {.val {object$backend}} backend returned no cache for this fit.",
      i = "Some architectures only cache under particular options; see \\
           the backend's own documentation."
    ))
  }
  object$state$kv_caches <- caches
  object$state$kv_guard  <- .kv_guard_of(object$state)
  object
}

# @keywords internal
.kv_capable_backends <- function() {
  nms <- ls(.tabfound_backends)
  keep <- vapply(nms, function(nm) {
    bk <- get_backend(nm)
    isTRUE(bk$kv_cache_capable)
  }, logical(1))
  nms[keep]
}

#' Is a cache attached, and what did it cost to build?
#' @param object A `tabfound_model`.
#' @return `TRUE` when the fitted object carries a precomputed KV cache.
#' @export
has_cache <- function(object) {
  inherits(object, "tabfound_model") && !is.null(object$state$kv_caches)
}
