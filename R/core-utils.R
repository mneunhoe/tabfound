# Backend-agnostic helpers: device resolution, tensor coercion, the
# activation-dump hook used by the parity harness, and dependency checks.

#' Resolve a device string, falling back to CPU when unavailable
#'
#' @param device Character. One of `"cpu"`, `"cuda"`, `"mps"`.
#' @return A validated device string. Falls back to `"cpu"` with a
#'   warning if the requested device is not available in the current
#'   torch install.
#' @keywords internal
resolve_device <- function(device = c("cpu", "cuda", "mps")) {
  device <- match.arg(device)
  if (device == "cuda" && !torch::cuda_is_available()) {
    cli::cli_alert_warning("CUDA not available, falling back to CPU.")
    return("cpu")
  }
  if (device == "mps" && !torch::backends_mps_is_available()) {
    cli::cli_alert_warning("MPS not available, falling back to CPU.")
    return("cpu")
  }
  device
}

#' Convert an R matrix / data.frame to a float torch tensor
#'
#' @param x Numeric matrix or data.frame.
#' @param device Character device string.
#' @return A 2-D `torch_tensor` of `torch_float()`.
#' @keywords internal
as_float_tensor <- function(x, device = "cpu") {
  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  storage.mode(x) <- "double"
  torch::torch_tensor(x, dtype = torch::torch_float(), device = device)
}

#' Dump a named tensor when the parity harness is active
#'
#' No-op unless `TABFOUND_DUMP_DIR` is set. Writes
#' `$TABFOUND_DUMP_DIR/<name>.safetensors` so a forward pass can be
#' diffed tensor-by-tensor against a reference Python dump. See
#' `inst/parity/`.
#'
#' `TABPFN_DUMP_DIR` is honoured as a deprecated alias so dumps taken
#' with the predecessor `tabpfn` package still work.
#'
#' @keywords internal
dump_if_enabled <- function(name, tensor) {
  dump_dir <- Sys.getenv("TABFOUND_DUMP_DIR", unset = "")
  if (!nzchar(dump_dir)) {
    dump_dir <- Sys.getenv("TABPFN_DUMP_DIR", unset = "")
  }
  if (!nzchar(dump_dir)) return(invisible())
  if (!dir.exists(dump_dir)) dir.create(dump_dir, recursive = TRUE)
  path <- file.path(dump_dir, paste0(name, ".safetensors"))
  safetensors::safe_save_file(
    list(t = tensor$detach()$contiguous()$cpu()),
    path
  )
  invisible()
}

#' Evaluate a function over a tensor in chunks, writing back in place
#'
#' A port of the reference's `chunked_evaluate_maybe_inplace`. Every
#' sublayer of a TabPFN block is independent across some leading set of
#' dimensions -- a per-token MLP is independent across rows *and* columns,
#' attention between a row's features is independent across rows -- so the
#' work can be split along those and done a slice at a time. What that
#' buys is peak memory, not speed: attention materialises an
#' `(n, n)` score matrix internally, and doing it in `k` pieces makes that
#' transient `k` times smaller.
#'
#' The result is written back into `x` rather than accumulated into a new
#' tensor, which is the other half of the saving. That only works if
#' `x$flatten()` returns a view instead of a copy, which needs the folded
#' dimensions to be contiguous -- hence the `contiguous()` call, a no-op
#' when `x` already is.
#'
#' Chunking changes nothing about the arithmetic: each slice sees exactly
#' the values it would have seen inside the whole.
#'
#' @param f Function of one tensor, returning a tensor of the same shape.
#' @param x Input tensor.
#' @param factor Number of chunks, or `NULL` to evaluate in one go.
#' @param residual Return `x + f(x)` rather than `f(x)`.
#' @param batch_dims How many leading dimensions of `x` to fold together
#'   and split along.
#' @param ... Passed to `f`.
#' @keywords internal
chunked_evaluate <- function(f, x, factor, residual, batch_dims, ...) {
  shape <- x$size()
  if (is.null(factor)) {
    res <- f(x$flatten(start_dim = 1L, end_dim = as.integer(batch_dims)), ...)$
      view(shape)
    return(if (isTRUE(residual)) x + res else res)
  }

  x <- x$contiguous()
  flat <- x$flatten(start_dim = 1L, end_dim = as.integer(batch_dims))
  n <- flat$size(1)
  split_size <- as.integer(ceiling(n / as.integer(factor)))
  for (chunk in torch::torch_split(flat, split_size, dim = 1L)) {
    if (isTRUE(residual)) chunk$add_(f(chunk, ...)) else chunk$copy_(f(chunk, ...))
  }
  x
}


#' Check that a suggested package is installed, abort with an install hint if not
#' @keywords internal
require_suggested <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cli::cli_abort(c(
      "Package {.pkg {pkg}} is required for this action.",
      i = "Install with {.code install.packages(\"{pkg}\")}"
    ))
  }
  invisible(TRUE)
}

#' Construct a named character vector allowing duplicate names
#'
#' `stats::setNames()` is fine for this, but cli's bullet lists need
#' repeated `"*"` names, which reads more clearly with a helper.
#' @keywords internal
set_names <- function(x, nms) {
  names(x) <- nms
  x
}

#' Locate a file shipped in `inst/`, falling back to a source checkout
#'
#' During development the package is often loaded with `pkgload::load_all()`
#' from a working directory somewhere inside the repo, where
#' `system.file()` already resolves. This helper additionally walks up
#' from the working directory so scripts run from `inst/parity/` or a
#' test directory still find the file.
#'
#' @param ... Path components below `inst/`.
#' @return An absolute path, or `""` when nothing was found.
#' @keywords internal
tabfound_file <- function(...) {
  path <- system.file(..., package = "tabfound", mustWork = FALSE)
  if (nzchar(path) && file.exists(path)) return(path)

  d <- getwd()
  for (i in 1:6) {
    cand <- file.path(d, "inst", ...)
    if (file.exists(cand)) return(cand)
    parent <- dirname(d)
    if (identical(parent, d)) break
    d <- parent
  }
  ""
}


#' Apply a function to row-chunks of a matrix and rbind the results
#'
#' In-context models re-encode the whole training context for every
#' forward pass, so test rows are processed in bounded batches rather
#' than all at once.
#'
#' @param X Matrix of test rows.
#' @param chunk_size Max rows per call.
#' @param fn Function taking a row-subset matrix and returning a matrix
#'   with one row per input row.
#' @keywords internal
chunk_apply <- function(X, chunk_size, fn) {
  n <- nrow(X)
  if (n <= chunk_size) return(fn(X))
  starts <- seq(1L, n, by = as.integer(chunk_size))
  do.call(rbind, lapply(starts, function(s) {
    e <- min(s + as.integer(chunk_size) - 1L, n)
    fn(X[s:e, , drop = FALSE])
  }))
}

#' A place to keep one KV cache per ensemble member, for one `predict()`
#'
#' Every backend that ensembles conditions on the *same* training rows in
#' each member -- what differs is the preprocessing, so each member needs
#' its own cache, and each of those is worth exactly as many rebuilds as
#' there are prediction chunks: one.
#'
#' The store is created per `predict()` call rather than kept on the
#' fitted object, which is what makes the cache invisible: nothing
#' survives the call, so nothing can go stale against a refitted model or
#' hold a several-hundred-megabyte tensor alive after the answer is out.
#' @keywords internal
member_cache_store <- function() new.env(parent = emptyenv())

#' Fetch member `i`'s cache, building it on first use
#' @param store A [member_cache_store()], or `NULL` to skip caching.
#' @param i Member index.
#' @param build Zero-argument function returning the cache.
#' @keywords internal
member_cache <- function(store, i, build) {
  if (is.null(store)) return(NULL)
  key <- as.character(i)
  hit <- store[[key]]
  if (!is.null(hit)) return(hit)
  cache <- build()
  assign(key, cache, envir = store)
  cache
}

#' Float32 footprint of a tensor, or of an arbitrarily nested list of them
#'
#' What the `print()` methods for the KV caches report. A cache is the one
#' object in this package whose size is a decision the user makes, so it
#' should not take a debugger to find out how large one got.
#'
#' @param x A tensor, or a list (of lists) of tensors. `NULL` counts as 0.
#' @keywords internal
.tensor_bytes <- function(x) {
  if (is.null(x)) return(0)
  if (inherits(x, "torch_tensor")) return(prod(as.numeric(x$size())) * 4)
  if (is.list(x)) return(sum(vapply(x, .tensor_bytes, numeric(1))))
  0
}

#' Write a single tensor into a parity trace directory
#'
#' Companion to [dump_if_enabled()] for backends whose trace is a handful
#' of named tensors rather than a per-member tree.
#' @keywords internal
trace_tensor <- function(dir, name, tensor) {
  if (is.null(dir)) return(invisible())
  require_suggested("safetensors")
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  safetensors::safe_save_file(
    list(t = tensor$detach()$to(dtype = torch::torch_float())$contiguous()$cpu()),
    file.path(dir, paste0(name, ".safetensors"))
  )
  invisible()
}


#' Read a safetensors file that may be gzipped
#'
#' The parity reference dumps ship gzipped. That is not for size — it is
#' because `file(1)` mistakes a raw safetensors header for a DOS `.COM`
#' executable when the header length happens to start with the wrong
#' byte, and `R CMD check` shells out to `file`. Gzip's magic number is
#' unambiguous.
#'
#' @param path Path to a `.safetensors` or `.safetensors.gz` file.
#' @keywords internal
read_reference_tensors <- function(path) {
  require_suggested("safetensors")
  if (!grepl("\\.gz$", path)) {
    return(safetensors::safe_load_file(path, framework = "torch"))
  }
  tmp <- tempfile(fileext = ".safetensors")
  on.exit(unlink(tmp), add = TRUE)
  con_in <- gzfile(path, "rb"); on.exit(close(con_in), add = TRUE)
  con_out <- file(tmp, "wb");   on.exit(close(con_out), add = TRUE)
  repeat {
    chunk <- readBin(con_in, "raw", 1e6)
    if (!length(chunk)) break
    writeBin(chunk, con_out)
  }
  close(con_out); close(con_in)
  on.exit(unlink(tmp), add = FALSE)
  safetensors::safe_load_file(tmp, framework = "torch")
}
