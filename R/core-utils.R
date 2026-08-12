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


# Where collecting starts paying. The cost is fixed per forward pass --
# one collection per block, a few milliseconds each -- while the saving
# grows with the table, so the two cross. Measured on the v3 ICL stack
# (24 blocks, 512 wide), peak with and against without, and the wall
# clock beside it:
#
#   rows   state    saved      time
#   1,000   2.0 MB  1,003 MB   +25%     <- not worth it
#   2,000   3.9 MB  2,001 MB    +9%
#   4,000   7.8 MB  2,351 MB    -4%     <- faster: less allocator pressure
#   8,000  15.6 MB  3,118 MB    -5%
#  16,000  31.2 MB  3,274 MB    +2%
#
# 4 MB puts the line between the first two rows. Anyone who would rather
# have that first gigabyte back can say so with the option.
COLLECT_MIN_BYTES <- 4194304

#' Let go of a layer's intermediates before starting the next one
#'
#' R torch frees a tensor's memory when R's garbage collector runs, not
#' when the last reference to it leaves scope. A deep stack evaluated in
#' one call therefore accumulates every block's intermediates until
#' something triggers a collection, which is why a 24-block stack can
#' hold 3.3 GB to carry a 31 MB state.
#'
#' torch has a knob for this -- `options(torch.threshold_call_gc = )`,
#' 4000 by default, which is why the accumulation stops just short of
#' 4 GB -- but lowering it is the worse instrument. The allocator fires
#' its collection *inside* a block, where the intermediates are still
#' referenced by the frame that is running and cannot be freed. Measured
#' on the v3 ICL stack at 16,000 rows: 5,671 MB at the default, 3,000 MB
#' at a threshold of 50 and 15% slower for it, against **2,426 MB** for a
#' collection placed between the blocks, which cost nothing measurable.
#'
#' Frequency is not negotiable either -- the saving is monotone in it.
#' At 16,000 rows, collecting every block gives 2,426 MB, every fourth
#' 4,166 MB, and every eighth 5,753 MB, which is no better than never.
#'
#' @param x The tensor the loop carries from one layer to the next; its
#'   size is what decides whether a collection is worth it. `NULL` means
#'   "cannot tell", which under `"auto"` means no.
#' @return Invisibly, whether a collection happened.
#' @section Options:
#' `tabfound.collect_between_layers` is `"auto"` (default: collect once
#' the carried tensor is worth it), `TRUE` (always) or `FALSE` (never).
#' `tabfound.collect_min_bytes` moves the `"auto"` threshold.
#' @keywords internal
collect_between_layers <- function(x = NULL) {
  mode <- getOption("tabfound.collect_between_layers", "auto")
  if (isFALSE(mode)) return(invisible(FALSE))
  if (!isTRUE(mode)) {
    if (is.null(x)) return(invisible(FALSE))
    bytes <- tryCatch(x$numel() * x$element_size(), error = function(e) 0)
    min_bytes <- getOption("tabfound.collect_min_bytes", COLLECT_MIN_BYTES)
    if (!isTRUE(bytes >= min_bytes)) return(invisible(FALSE))
  }
  # `full = FALSE` deliberately: a full collection is 45x the cost
  # (44.6 ms against 1.0 ms here) and the tensors this is for are freed
  # by the level-0 pass all the same.
  gc(full = FALSE)
  invisible(TRUE)
}


#' Let go of a chunk's or a member's intermediates before the next one
#'
#' The same mechanism as [collect_between_layers()] with different
#' arithmetic. Between layers the collection has to earn its
#' millisecond, because a small carried state means small savings --
#' hence the size test. Between prediction chunks and between ensemble
#' members there is nothing to weigh: each iteration is a whole forward
#' pass over the training context, so the fixed cost is never the
#' deciding term, and each leaves behind everything that pass allocated.
#'
#' This is where the accumulation actually bites. Every backend already
#' collects between layers; the loop *above* those -- eight members, each
#' running the full stack, then the next chunk doing it again -- held all
#' of it until something else triggered a collection.
#'
#' @return Invisibly, whether a collection happened. Honours the same
#'   `tabfound.collect_between_layers` option, so one switch turns all of
#'   it off.
#' @keywords internal
collect_between_chunks <- function() {
  if (isFALSE(getOption("tabfound.collect_between_layers", "auto"))) {
    return(invisible(FALSE))
  }
  gc(full = FALSE)
  invisible(TRUE)
}

#' Evaluate a function over one axis of a tensor in chunks, in place
#'
#' The same trade as [chunked_evaluate()] — peak memory for a little loop
#' overhead, with the result written back into `x` rather than
#' accumulated — but splitting a named axis instead of a fold of the
#' leading ones.
#'
#' That distinction matters where the sublayer's own reshape needs the
#' leading dimensions kept apart. Mitra's observation attention folds
#' `(batch, features)` into the attention batch and puts *rows* in the
#' sequence position, so its key/value cache is indexed by
#' `batch * features`; flattening `(batch, rows)` first, as
#' [chunked_evaluate()] does, would hand a chunk spanning two datasets to
#' a cache that expects one. Splitting the row axis leaves that fold
#' intact.
#'
#' `torch_split()` returns views along any axis, so the in-place
#' `add_()` reaches the original storage whichever axis is chosen.
#'
#' @param f Function of one tensor, returning a tensor of the same shape.
#' @param x Input tensor.
#' @param factor Number of chunks, or `NULL` to evaluate in one go.
#' @param axis 1-based axis to split along.
#' @param residual Write `x + f(x)` rather than `f(x)`.
#' @param ... Passed to `f`.
#' @keywords internal
chunked_evaluate_axis <- function(f, x, factor, axis, residual = TRUE, ...) {
  if (is.null(factor)) {
    res <- f(x, ...)
    return(if (isTRUE(residual)) x + res else res)
  }
  x <- x$contiguous()
  n <- x$size(as.integer(axis))
  split_size <- as.integer(ceiling(n / as.integer(factor)))
  for (chunk in torch::torch_split(x, split_size, dim = as.integer(axis))) {
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
  do.call(rbind, lapply(seq_along(starts), function(k) {
    # A chunk's transients are dead the moment it returns, but they are
    # torch allocations, which R's collector cannot see and will not free
    # before the next chunk has allocated its own. This is the same
    # collection [collect_between_layers()] does inside a stack, one level
    # up: between chunks rather than between layers.
    if (k > 1L) collect_between_chunks()
    s <- starts[[k]]
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

#' A store pre-filled from a persisted cache
#'
#' The lazy backends build member caches on first use, which is right
#' when the cache lives for one `predict()` call. A cache restored from
#' disk is already built, so it is loaded into a store of the same shape
#' and the builders never run.
#'
#' @param caches A named list keyed by member index, as
#'   [tabfound_cache()] stores it, or `NULL`.
#' @return A [member_cache_store()], or `NULL`.
#' @keywords internal
member_cache_store_from <- function(caches) {
  if (is.null(caches)) return(NULL)
  store <- member_cache_store()
  for (nm in names(caches)) assign(nm, caches[[nm]], envir = store)
  store
}

#' Everything a member-cache store holds, keyed by member index
#' @keywords internal
member_cache_list <- function(store) {
  if (is.null(store)) return(NULL)
  out <- as.list(store)
  out[order(as.integer(names(out)))]
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


#' Set libtorch's thread count
#'
#' libtorch defaults to one intra-op thread per core, which is right for
#' a single model in a single process and catastrophic inside a
#' `parallel` / `future` worker pool: `k` workers each spawning `n_cores`
#' threads oversubscribe the machine by `k`-fold, and the run gets slower
#' the more workers you add.
#'
#' This is the knob for that. The rule of thumb for `k` R workers is
#' `tabfound_threads(max(1, parallel::detectCores() %/% k))`, called
#' inside each worker -- the setting is per process.
#'
#' The other half of the advice is not to fork at all: multiple-imputation
#' chains and ensemble members are already sequential calls into a
#' multi-threaded library, so the parallelism is better left to torch
#' than taken from it. Forking a process that has already initialised
#' libtorch is its own hazard -- the child inherits a thread pool it
#' cannot use -- so a worker pool must be created *before* the first
#' torch call, or with `future::plan(multisession)` rather than
#' `multicore`.
#'
#' Set it **before the first forward pass**. libtorch's native backend
#' refuses both counts once its parallel region has started, and says so
#' on stderr from C++ rather than through an R condition, so a late call
#' looks as though it worked. Reading back the value is the check.
#'
#' @param n Number of intra-op threads. `NULL` reports the current
#'   setting without changing it.
#' @param interop Number of inter-op threads, or `NULL` to leave it.
#' @return Invisibly, the intra-op thread count in effect afterwards.
#' @examples
#' \dontrun{
#' tabfound_threads()        # report
#' tabfound_threads(4L)      # four intra-op threads in this process
#' }
#' @export
tabfound_threads <- function(n = NULL, interop = NULL) {
  current <- function() {
    tryCatch(torch::torch_get_num_threads(), error = function(e) NA_integer_)
  }
  if (!is.null(n)) {
    n <- as.integer(n)
    if (is.na(n) || n < 1L) cli::cli_abort("{.arg n} must be a positive integer.")
    tryCatch(torch::torch_set_num_threads(n),
             error = function(e) cli::cli_warn(c(
               "Could not set the thread count: {conditionMessage(e)}"
             )))
  }
  if (!is.null(interop)) {
    interop <- as.integer(interop)
    if (is.na(interop) || interop < 1L) {
      cli::cli_abort("{.arg interop} must be a positive integer.")
    }
    tryCatch(torch::torch_set_num_interop_threads(interop),
             error = function(e) cli::cli_warn(c(
               "Could not set the inter-op thread count.",
               i = "libtorch only allows this before the parallel region \\
                    starts -- set it before the first forward pass."
             )))
  }
  invisible(current())
}
