# Memory preflight: predict the peak before allocating anything.
#
# The failure this exists to prevent is not an R error. A libtorch
# allocation failure at fold size kills the R process outright -- no
# condition, no traceback, an empty terminal -- so there is nothing to
# catch and nothing to report after the fact. The only defence is to
# answer the question *before* the first tensor exists:
#
#   will fit() + predict() at these dimensions, with these options, fit
#   in the memory this machine currently has available?
#
# Answering it needs three things, and this file holds all three:
#
#   1. what the machine has right now  -- `.system_memory()`
#   2. what the model will ask for     -- the backend's `peak_terms()`
#                                         plus fitted coefficients
#   3. what to do about the gap        -- verdict + suggestions
#
# The estimate is deliberately a breakdown rather than one number. Where
# the bytes go is the actionable part: weights are fixed, persistent
# state answers to `kv_cache`, and the transient peak answers to context
# size, chunking and `save_peak_memory_factor`.


# ---------------------------------------------------------------------------
# What the machine has
# ---------------------------------------------------------------------------

# Total is fixed for the life of the session; available is not, and the
# difference between them is the whole point. A TabICL run that fits
# comfortably on an idle machine gets killed by the OS when another
# process is holding 7 GB: peak *headroom* decides survival, not peak
# demand against the sticker number.
.mem_cache <- new.env(parent = emptyenv())

#' Free and total physical memory, in bytes
#'
#' macOS has no `MemAvailable`. The closest honest approximation is the
#' page classes the kernel can hand out without swapping -- free,
#' inactive, speculative and purgeable -- which is what `vm_stat` reports
#' and what this reads. Linux states `MemAvailable` directly and it is
#' the better number, so it is used as-is where present.
#'
#' Every path is allowed to fail. A machine this does not recognise
#' returns `NA` availability, which downgrades the guard to silence
#' rather than to a wrong answer.
#'
#' @return `list(total, available, source)`, bytes, `NA` when unknown.
#' @keywords internal
.system_memory <- function() {
  total <- .mem_cache$total
  if (is.null(total)) {
    total <- .system_memory_total()
    assign("total", total, envir = .mem_cache)
  }
  # Availability moves, so it cannot be cached for the session -- but on
  # macOS reading it means shelling out to `vm_stat`, and a chained-
  # equations loop asks twice per univariate fit. A short time-to-live
  # keeps the number honest at a fraction of the calls; the guard's
  # thresholds are order-of-magnitude judgements, not knife edges.
  ttl <- getOption("tabfound.memory_probe_ttl", 2)
  now <- as.numeric(Sys.time())
  hit <- .mem_cache$avail
  if (!is.null(hit) && is.finite(ttl) && (now - hit$at) < ttl) {
    return(list(total = total, available = hit$bytes, source = hit$source))
  }
  av <- .system_memory_available()
  assign("avail", list(bytes = av$bytes, source = av$source, at = now),
         envir = .mem_cache)
  list(total = total, available = av$bytes, source = av$source)
}

# @keywords internal
.system_memory_total <- function() {
  if (requireNamespace("ps", quietly = TRUE)) {
    got <- tryCatch(as.numeric(ps::ps_system_memory()$total),
                    error = function(e) NA_real_)
    if (isTRUE(is.finite(got))) return(got)
  }
  sys <- Sys.info()[["sysname"]]
  out <- tryCatch({
    if (identical(sys, "Darwin")) {
      as.numeric(system2("sysctl", c("-n", "hw.memsize"), stdout = TRUE))
    } else if (identical(sys, "Linux")) {
      .parse_meminfo(readLines("/proc/meminfo", warn = FALSE))[["total"]]
    } else if (identical(sys, "Windows")) {
      .parse_wmic(system2("wmic", c("OS", "get", "TotalVisibleMemorySize",
                                    "/Value"), stdout = TRUE))
    } else NA_real_
  }, error = function(e) NA_real_, warning = function(w) NA_real_)
  if (length(out) != 1L || !is.finite(out)) NA_real_ else as.numeric(out)
}

# @keywords internal
.system_memory_available <- function() {
  if (requireNamespace("ps", quietly = TRUE)) {
    got <- tryCatch(as.numeric(ps::ps_system_memory()$avail),
                    error = function(e) NA_real_)
    if (isTRUE(is.finite(got))) return(list(bytes = got, source = "ps"))
  }
  sys <- Sys.info()[["sysname"]]
  out <- tryCatch({
    if (identical(sys, "Darwin")) {
      list(bytes = .parse_vm_stat(system2("vm_stat", stdout = TRUE)),
           source = "vm_stat")
    } else if (identical(sys, "Linux")) {
      info <- .parse_meminfo(readLines("/proc/meminfo", warn = FALSE))
      list(bytes = info[["available"]], source = "/proc/meminfo")
    } else if (identical(sys, "Windows")) {
      txt <- system2("wmic", c("OS", "get", "FreePhysicalMemory", "/Value"),
                     stdout = TRUE)
      list(bytes = .parse_wmic(txt), source = "wmic")
    } else list(bytes = NA_real_, source = NA_character_)
  }, error = function(e) list(bytes = NA_real_, source = NA_character_),
     warning = function(w) list(bytes = NA_real_, source = NA_character_))
  if (length(out$bytes) != 1L || !is.finite(out$bytes)) {
    return(list(bytes = NA_real_, source = NA_character_))
  }
  out
}

#' Sum the reclaimable page classes out of `vm_stat` output
#'
#' Split out from the probe so it can be tested against captured text
#' rather than against whatever the test machine happens to be doing.
#' @param txt Character vector, the lines `vm_stat` printed.
#' @keywords internal
.parse_vm_stat <- function(txt) {
  hdr <- grep("page size of", txt, value = TRUE)
  page <- if (length(hdr)) {
    as.numeric(sub(".*page size of ([0-9]+) bytes.*", "\\1", hdr[1]))
  } else 4096
  field <- function(label) {
    line <- grep(paste0("^", label, ":"), txt, value = TRUE)
    if (!length(line)) return(0)
    as.numeric(gsub("[^0-9]", "", line[1]))
  }
  # Free plus the classes the kernel can reclaim without touching swap.
  # Wired and compressed pages are not on that list.
  pages <- field("Pages free") + field("Pages inactive") +
    field("Pages speculative") + field("Pages purgeable")
  if (!is.finite(pages) || pages <= 0) return(NA_real_)
  pages * page
}

#' Pull `MemTotal` / `MemAvailable` out of `/proc/meminfo`
#' @param lines Character vector of file lines.
#' @keywords internal
.parse_meminfo <- function(lines) {
  field <- function(label) {
    line <- grep(paste0("^", label, ":"), lines, value = TRUE)
    if (!length(line)) return(NA_real_)
    # Values are in kB.
    as.numeric(gsub("[^0-9]", "", line[1])) * 1024
  }
  avail <- field("MemAvailable")
  if (!is.finite(avail)) {
    # Pre-3.14 kernels have no MemAvailable; free + cached is the usual
    # stand-in and errs low, which is the safe direction here.
    avail <- sum(c(field("MemFree"), field("Cached")), na.rm = TRUE)
    if (avail == 0) avail <- NA_real_
  }
  c(total = field("MemTotal"), available = avail)
}

# `wmic` prints `Name=value`, in kB.
# @keywords internal
.parse_wmic <- function(txt) {
  line <- grep("=", txt, value = TRUE)
  if (!length(line)) return(NA_real_)
  val <- as.numeric(gsub("[^0-9]", "", line[1]))
  if (!is.finite(val)) NA_real_ else val * 1024
}


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

# The scaling model is `peak = intercept + b * T_act + c * T_att`, fitted
# per backend, with the backend contributing only the shapes `T_act` and
# `T_att` and this file contributing the constants. The constants are
# dtype- and device-dependent but machine-independent, so they ship with
# the package -- measured once, asserted thereafter, like the parity
# fixtures.
#
# `b` reads as "how many live copies of the largest activation a stage
# holds at once": the state tensor, its normed copy, the residual, the
# attention output, and an MLP intermediate that is `ff_factor` times
# wider. `c` is the same for materialised attention scores.

.coef_cache <- new.env(parent = emptyenv())

#' Fitted peak-memory coefficients for one backend
#'
#' @param backend Backend name.
#' @param device `"cpu"`, `"cuda"`, `"mps"`.
#' @param dtype Currently `"float32"` throughout.
#' @return A list of constants, or `NULL` when none are shipped.
#' @keywords internal
.memory_coefs <- function(backend, device = "cpu", dtype = "float32") {
  key <- paste(backend, device, dtype, sep = "/")
  hit <- .coef_cache[[key]]
  if (!is.null(hit)) return(hit)

  path <- tabfound_file("memory", "coefs", paste0(backend, ".json"))
  if (!nzchar(path)) return(NULL)
  blob <- tryCatch(jsonlite::fromJSON(path, simplifyVector = TRUE),
                   error = function(e) NULL)
  if (is.null(blob)) return(NULL)

  entry <- blob$devices[[device]][[dtype]]
  if (is.null(entry)) {
    # Constants for another device are not transferable -- an estimate is
    # better than nothing only when it is about the right machine.
    return(NULL)
  }
  out <- list(
    intercept_bytes        = entry$intercept_bytes %||% 0,
    act_copies             = entry$act_copies %||% 12,
    att_copies             = entry$att_copies %||% 0,
    res_copies             = entry$res_copies %||% 2,
    floor_bytes            = entry$floor_bytes %||% 0,
    spmf_floor             = entry$spmf_floor %||% 1,
    prepass_copies         = entry$prepass_copies,
    icl_copies             = entry$icl_copies,
    prep_state_factor      = entry$prep_state_factor %||% 1,
    safety_factor          = entry$safety_factor %||% 1,
    ensemble_log2_factor   = entry$ensemble_log2_factor %||% 0,
    ensemble_cap           = entry$ensemble_cap %||% Inf,
    weights_fallback_bytes = entry$weights_fallback_bytes %||% NA_real_,
    source                 = blob$source %||% "unknown",
    notes                  = blob$notes %||% ""
  )
  assign(key, out, envir = .coef_cache)
  out
}


# ---------------------------------------------------------------------------
# Resolving what is being asked about
# ---------------------------------------------------------------------------

# `estimate_peak_memory()` takes whatever the user has to hand: a fitted
# or unfitted model, a directory of artifacts, a config already read, or
# a catalogue id. Only the first knows its weights exactly; the rest are
# resolved as far as they can be and the shortfall is recorded rather
# than hidden.
# @keywords internal
.resolve_memory_target <- function(model, backend = NULL, device = NULL,
                                   task = NULL) {
  if (inherits(model, "tabfound_model")) {
    return(list(
      backend = get_backend(model$backend),
      config  = model$config,
      device  = device %||% model$device,
      weights = .model_weight_bytes(model),
      weights_source = "loaded weights",
      opts    = model$model_ref$args %||% list(),
      task    = model$task
    ))
  }

  if (is.list(model)) {
    config <- model
    bk <- if (!is.null(backend)) get_backend(backend)
          else detect_backend(config)
    return(list(
      backend = bk, config = config, device = device %||% "cpu",
      weights = .config_weight_bytes(config),
      weights_source = if (is.null(config$state_dict_shapes)) "backend fallback"
                       else "config shapes",
      opts = list(), task = bk$task_of(config)
    ))
  }

  if (!is.character(model) || length(model) != 1L) {
    cli::cli_abort(c(
      "{.arg model} must be a {.cls tabfound_model}, a config list, an \\
       artifact directory, or a model id.",
      i = "Got {.cls {class(model)[1]}}."
    ))
  }

  dir <- .local_artifact_dir(model, task)
  if (is.null(dir)) {
    # Nothing downloaded. The dimensions are still knowable -- they are a
    # property of the published architecture, not of the copy on this
    # disk -- so the shipped table answers it, and the whole point of a
    # preflight survives: you can ask whether a 6.5 GB download is worth
    # making before you make it.
    known <- .shipped_architecture(model, task)
    if (!is.null(known)) {
      bk <- if (!is.null(backend)) get_backend(backend)
            else get_backend(known$backend)
      return(list(
        backend = bk, config = known$config, device = device %||% "cpu",
        weights = known$weights_bytes,
        weights_source = "shipped architecture table",
        opts = list(), task = known$task %||% bk$task_of(known$config)
      ))
    }
    # A family name that names two catalogue entries is not missing, it
    # is ambiguous, and saying so beats sending the caller to a download
    # command that would ask them the same question.
    fam <- .catalog_family_matches(model)
    if (is.null(task) && length(fam) > 1L) {
      cli::cli_abort(c(
        "{.val {model}} is a model family, not a single checkpoint.",
        i = "It covers {.val {fam}}.",
        i = "Pass {.arg task} ({.val classification} or {.val regression}), \\
             or name the checkpoint."
      ))
    }
    cli::cli_abort(c(
      "No local artifacts for {.val {model}}, and it is not in the \\
       shipped architecture table.",
      i = "Run {.code download_model(\"{model}\")} first, or pass the \\
           {.file config.json} contents as {.arg model}.",
      i = "{.fn estimate_peak_memory} needs the config's dimensions; it \\
           never needs the weights."
    ))
  }
  paths  <- resolve_artifacts(dir$path, subfolder = dir$subfolder)
  config <- read_model_config(paths$config)
  bk <- if (!is.null(backend)) get_backend(backend) else detect_backend(config)
  w <- .config_weight_bytes(config)
  src <- "config shapes"
  if (!is.finite(w)) {
    w <- tryCatch(as.numeric(file.size(paths$weights)),
                  error = function(e) NA_real_)
    src <- "artifact file size"
  }
  list(backend = bk, config = config, device = device %||% "cpu",
       weights = w, weights_source = src, opts = list(),
       task = bk$task_of(config))
}

# Look for artifacts already on disk, without triggering a download. A
# preflight that downloads 6.5 GB to tell you the run will not fit has
# missed the point.
# @keywords internal
.local_artifact_dir <- function(model, task = NULL) {
  if (dir.exists(model)) {
    if (file.exists(file.path(model, "config.json"))) {
      return(list(path = model, subfolder = NULL))
    }
    # Not every backend puts its artifacts at the top of the directory.
    # TabFM ships a `classification/` and a `regression/` tree under one
    # snapshot, which is why passing that snapshot's path used to fail
    # here while passing the model *id* worked -- the catalogue carries
    # the subfolder and a bare path does not. Ask the backends what they
    # would look for, rather than teaching this function one layout.
    for (nm in list_backends()$name) {
      sub_for <- get_backend(nm)$subfolder_for
      if (!is.function(sub_for)) next
      for (task in c("classification", "regression")) {
        sub <- tryCatch(sub_for(task), error = function(e) NULL)
        if (is.null(sub)) next
        if (file.exists(file.path(model, sub, "config.json"))) {
          return(list(path = model, subfolder = sub))
        }
      }
    }
    return(list(path = model, subfolder = NULL))
  }
  # As in `.shipped_architecture()`: a family name needs the task, and
  # resolves without one only when it covers a single checkpoint.
  id <- tryCatch(.catalog_id(model, task), error = function(e) NULL)
  if (is.null(id) && is.null(task)) {
    hits <- .catalog_family_matches(model)
    if (length(hits) == 1L) id <- hits
  }
  if (is.null(id) || !isTRUE(.model_is_downloaded(id))) return(NULL)
  list(path = .model_dir(id), subfolder = .model_catalog()[[id]]$subfolder)
}

# Architecture dimensions for models that are catalogued but not
# downloaded. Collected once from real `config.json` files by
# `inst/memory/collect-architectures.R`; a few hundred bytes each.
# @keywords internal
.arch_cache <- new.env(parent = emptyenv())

# @keywords internal
.shipped_architecture <- function(model, task = NULL) {
  if (is.null(.arch_cache$table)) {
    path <- tabfound_file("memory", "architectures.json")
    tbl <- if (nzchar(path)) {
      tryCatch(jsonlite::fromJSON(path, simplifyVector = TRUE)$models,
               error = function(e) list())
    } else list()
    assign("table", tbl, envir = .arch_cache)
  }
  # A family name (`"tabpfn"`, `"tabfm-1.0.0"`) only resolves once the
  # task is known, because a family ships one checkpoint per task. The
  # task was already an argument of `estimate_peak_memory()`; it just was
  # not reaching here, which is why every family name failed.
  id <- tryCatch(.catalog_id(model, task), error = function(e) NULL) %||%
    if (is.null(task)) {
      hits <- .catalog_family_matches(model)
      if (length(hits) == 1L) hits else model
    } else model
  entry <- .arch_cache$table[[id]]
  if (is.null(entry)) return(NULL)
  entry
}

# Every catalogue id a family name covers. One means the name is
# unambiguous even without a task; two means the caller has to choose.
# @keywords internal
.catalog_family_matches <- function(model) {
  if (!is.character(model) || length(model) != 1L) return(character())
  ids <- c(
    tryCatch(.catalog_id(model, "classification"), error = function(e) NULL),
    tryCatch(.catalog_id(model, "regression"), error = function(e) NULL)
  )
  unique(Filter(Negate(is.null), ids))
}

# Exact, from the tensors themselves.
#
# `$parameters` materialises the whole module tree as an R list --
# thousands of entries for these networks -- and the answer cannot change
# for a given network, so it is memoised against the module itself. That
# matters in a chained-equations loop, which asks `m * maxit * p` times.
# @keywords internal
.weight_bytes_cache <- new.env(parent = emptyenv())

# @keywords internal
.model_weight_bytes <- function(object) {
  net <- object$model
  if (is.null(net)) return(NA_real_)
  key <- tryCatch(format(net$.__enclos_env__), error = function(e) NULL)
  if (!is.null(key) && !is.null(.weight_bytes_cache[[key]])) {
    return(.weight_bytes_cache[[key]])
  }
  params <- tryCatch(net$parameters, error = function(e) NULL)
  # No parameters at all means the tensors are gone (a `saveRDS()` round
  # trip) or never existed. Zero would be a lie; NA sends the caller on
  # to the next source.
  if (!length(params)) return(NA_real_)
  got <- tryCatch(
    sum(vapply(params, function(p) prod(as.numeric(p$size())), numeric(1))) * 4,
    error = function(e) NA_real_
  )
  got <- if (!isTRUE(is.finite(got))) NA_real_ else got
  # A dangling-pointer network gives NA every time and is not worth
  # remembering; a real answer is.
  if (!is.null(key) && !is.na(got)) assign(key, got, envir = .weight_bytes_cache)
  got
}

# Exact, from the config, with no weights on disk at all. The converted
# checkpoints record every parameter's shape, which is the whole point of
# carrying `state_dict_shapes` -- a pre-download check can be exact.
# @keywords internal
.config_weight_bytes <- function(config) {
  shapes <- config$state_dict_shapes
  if (is.null(shapes) || !length(shapes)) return(NA_real_)
  sum(vapply(shapes, function(s) prod(as.numeric(unlist(s))), numeric(1))) * 4
}


# Backend knobs that change the peak. The user passes some; the rest come
# from the predictor's own defaults, which is where `n_estimators = 32`
# for TabFM and `= 1` for Mitra actually live.
# @keywords internal
.resolve_memory_opts <- function(backend, task, user_opts, extra) {
  fn <- if (identical(task, "regression")) backend$regressor
        else backend$classifier
  defaults <- if (is.function(fn)) {
    f <- formals(fn)
    f <- f[setdiff(names(f), c("ctx", "..."))]
    lapply(f, function(v) tryCatch(eval(v), error = function(e) NULL))
  } else list()
  opts <- utils::modifyList(defaults, user_opts %||% list())
  extra <- extra %||% list()
  opts <- utils::modifyList(opts, extra)
  # `modifyList()` *removes* an element it is handed as NULL rather than
  # storing one. For `row_chunk_size` that is the wrong reading: NULL is
  # a value there -- "run every row in one pass" -- and dropping it would
  # silently estimate the chunked run instead, which is the one case a
  # caller asking this question most needs to see.
  for (nm in names(extra)) {
    if (is.null(extra[[nm]])) opts[nm] <- list(NULL)
  }
  opts
}


# ---------------------------------------------------------------------------
# Shapes the backends build their terms from
# ---------------------------------------------------------------------------

# Rows resident in a stage-chunked pass through stages 0-2.
#
# The three states of `row_chunk_size` as the estimator sees them: absent
# or `NA` is the checkpoint's own value, `NULL` is the unchunked pass,
# and a number is itself. Mirrors `.tabpfn3_chunk_arg()` on the network
# side -- they have to agree, or the estimate is about a different run
# than the one that will happen.
# @keywords internal
.stage_row_chunk <- function(opts, config) {
  if ("row_chunk_size" %in% names(opts) && is.null(opts$row_chunk_size)) {
    return(Inf)
  }
  rc <- suppressWarnings(as.numeric(opts$row_chunk_size %||% NA))
  if (isTRUE(is.finite(rc)) && rc >= 1) return(rc)
  as.numeric(config$inference_row_chunk_size %||% 2048)
}

# Columns resident in the summary pre-pass, same three states.
#
# Kept separate from the row chunk because they bound different things: a
# row chunk bounds the forward loop, a column chunk bounds the pre-pass
# the loop cannot start without. Only TabPFN v3 carries a default in its
# config; on TabICL and TabFM the knob exists but is off unless asked
# for, and `Inf` is what "off" means here.
# @keywords internal
.stage_col_chunk <- function(opts, config) {
  if ("col_chunk_size" %in% names(opts) && is.null(opts$col_chunk_size)) {
    return(Inf)
  }
  cc <- suppressWarnings(as.numeric(opts$col_chunk_size %||% NA))
  if (isTRUE(is.finite(cc)) && cc >= 1) return(cc)
  as.numeric(config$inference_col_chunk_size %||% Inf)
}

#' Query rows resident in one forward pass
#'
#' Test rows go through in bounded batches ([chunk_apply()]), so a
#' prediction over 100,000 rows never has 100,000 rows in flight --
#' `predict_chunk_size` of them do. This is why the chunk size is a real
#' lever on the peak and why the estimate must not use `n_query` raw.
#' @keywords internal
.resident_query <- function(n_query, opts) {
  chunk <- suppressWarnings(as.numeric(opts$predict_chunk_size %||% Inf))
  if (!isTRUE(is.finite(chunk)) || chunk < 1) chunk <- Inf
  min(n_query, chunk)
}

#' Terms for the column -> row -> in-context stack
#'
#' TabICL, TabFM and TabPFN v3 are the same three-stage shape with
#' different widths: embed each column against the rows, let a row's
#' columns interact, then attend across rows in context. The stages have
#' quite different peaks -- the first two are wide and shallow, the third
#' narrow and quadratic in rows -- so the maximum matters and the sum
#' would be badly wrong.
#'
#' @param n_context,n_query,n_features Dimensions; `n_query` should
#'   already be the resident chunk (see [.resident_query()]).
#' @param embed_dim Model width.
#' @param group_size Features per column token.
#' @param n_cls Number of CLS tokens the row stage appends, which is also
#'   the factor between the embedding width and the in-context width.
#' @param col_heads,row_heads,icl_heads Head counts per stage.
#' @param col_inducing Inducing points in the column stage, which is what
#'   keeps it linear in rows rather than quadratic.
#' @param icl_blocks Blocks in the in-context stack, for the cache size.
#' @param cell_tokens Column tokens per row, when the backend embeds
#'   whole cells instead of feature groups (TabFM). Defaults to the
#'   number of feature groups.
#' @param kv_cache,n_estimators Whether caches are kept, and how many.
#' @keywords internal
.icl_family_terms <- function(n_context, n_query, n_features, embed_dim,
                              group_size, n_cls, col_heads, row_heads,
                              icl_heads, col_inducing, icl_blocks,
                              cell_tokens = NULL, kv_cache = FALSE,
                              n_estimators = 1, masked_attention = FALSE,
                              row_chunk = Inf, group_channels = 0,
                              col_chunk = Inf) {
  fg  <- max(1, ceiling(n_features / max(1, group_size)))
  tok <- cell_tokens %||% fg
  hc  <- fg + n_cls
  n   <- n_context + n_query
  d_icl <- embed_dim * n_cls

  # Rows resident in stages 0-2 at once. Chunking is what makes this a
  # constant rather than the row count.
  chunked <- isTRUE(is.finite(row_chunk)) && row_chunk < n
  rc <- if (chunked) row_chunk else n

  # What outlives the row loop when there is one: the grouped input every
  # chunk is sliced from, `group_channels` floats per column per row, and
  # the row embeddings the chunks accumulate into. Both are linear in the
  # row count and both are far narrower than the embedded tensor the loop
  # exists to avoid -- 6 and 4 channels against 128 -- which is the whole
  # point. Carried as `res` rather than `act` because they are one copy
  # apiece, not the dozens a live block holds.
  res <- if (chunked) n * (tok * group_channels + n_cls * embed_dim) else 0

  # The summary pre-pass, which exists only when the forward is chunked:
  # it reads *every* context row -- that is what a summary is -- while a
  # slice of the columns, so the row chunk does not bound it and the
  # column chunk does. Leaving it out was how the estimator came to
  # report a chunked TabFM as cheap while its pre-pass alone measured
  # 36.4 GB of a 35.6 GB run.
  #
  # It carries its own copy count rather than the forward's. A stage's
  # `act_copies` is fitted to whichever stage dominates a whole forward
  # pass -- every live tensor in the pipeline at that moment -- and the
  # pre-pass is a narrower thing: one column-stage stack and nothing
  # else. Measured on TabICL at 12,000 x 300, charging it the forward's
  # 138 copies put the estimate at 112 GB against 36.9 GB observed.
  prepass <- if (chunked) n_context * min(tok, col_chunk) * embed_dim else 0

  stages <- list(
    list(name = "column summaries",
         act = 0, res = 0, att = 0, prepass = prepass),
    list(name = "column embedding",
         act = rc * tok * embed_dim, res = res,
         att = 0),
    list(name = "row interaction",
         act = rc * hc * embed_dim, res = res,
         att = 0),
    list(name = "in-context learning",
         act = n * d_icl, res = 0, copies = "icl",
         # Only a *masked* attention pays for its score matrix. Torch's
         # fused SDPA picks a memory-efficient kernel when no mask is
         # passed and never materialises the `(n, n)` scores: measured at
         # `(1, 8, n, 64)`, peak RSS goes 353 -> 382 -> 448 MB across
         # n = 4,000, 8,000, 16,000, where the scores alone would be 488,
         # 1,953 and 7,812 MB, and a hand-written attention at n = 8,000
         # takes 6,235 MB. With a mask it is quadratic again, though at
         # roughly a sixth of a float32 score matrix.
         #
         # Of the six backends only TabFM masks here -- it restricts
         # context by masking where TabICL and the TabPFN family slice.
         att = if (isTRUE(masked_attention)) icl_heads * n_context * n else 0)
  )

  # A cache is built once per ensemble member and every one of them stays
  # alive until `predict()` returns, so this term is the one place
  # `n_estimators` genuinely multiplies.
  persistent <- if (isTRUE(kv_cache)) {
    n_estimators * (n_context * tok * embed_dim +
                    icl_blocks * 2 * n_context * d_icl)
  } else 0

  list(stages = stages, persistent = persistent, n_estimators = n_estimators)
}


# float32 throughout; the coef files carry a dtype key for when that
# stops being true.
DTYPE_BYTES <- 4

#' Turn a backend's shapes into bytes, given fitted constants
#'
#' The one place the scaling model is actually evaluated. Split out
#' because `inst/memory/calibrate.R` fits against exactly this expression
#' and `tests/testthat/test-memory-calibration.R` replays stored
#' measurements through it: if the assembly here and the assembly the
#' constants were fitted to ever drift apart, the constants stop meaning
#' anything, and nothing would say so.
#'
#' Only one stage is live at a time, so the transient is the largest, not
#' the sum. `save_peak_memory_factor` divides `act_copies - 1` rather than
#' `act_copies`: it chunks the temporaries but not the state tensor they
#' are made from.
#'
#' @section Ensembles:
#' Members run one after another, so the *live set* is one member's. The
#' resident high-water mark is not: measured on TabICL, going from one
#' member to sixteen at fixed dimensions took the peak from 3.1 GB to
#' 8.7 GB, in steps that look like an allocator taking a bigger arena and
#' not giving it back. TabFM behaves the same way. So the transient is
#' multiplied by `1 + ensemble_log2_factor * log2(n_estimators)` --
#' growth, but nothing like the `n_estimators x` a naive reading would
#' give, and flat for the single-member default.
#'
#' This is the term the hand-off that commissioned this work explicitly
#' said not to include. It was right about the live set and wrong about
#' resident memory, which is what the OS kills on; the measurement is in
#' `inst/memory/measurements/ensemble-*.json`.
#'
#' @param stages List of `list(name, act, att)`, in tensor elements.
#' @param weights,persistent Bytes.
#' @param co Coefficients from [.memory_coefs()].
#' @param spmf `save_peak_memory_factor`, or 1.
#' @param n_estimators Ensemble members the call will run.
#' @keywords internal
.peak_from_terms <- function(stages, weights, persistent, co, spmf = 1,
                             n_estimators = 1) {
  # What `save_peak_memory_factor` reaches, and what it does not.
  #
  # This used to be `1 + (act_copies - 1) / spmf`, which says exactly one
  # copy of the activation survives chunking and the other ninety-nine
  # divide. Measured, that is far too generous: on Mitra at 2,000 x 50 a
  # factor of 8 takes the peak from 25.1 GB to 11.2 GB -- 2.2x, where
  # that form predicts 7.5x. `spmf_floor` is the share the factor cannot
  # touch, fitted per backend from a sweep, and it defaults to **1** --
  # "buys nothing" -- because an unmeasured backend must not be promised
  # a saving it may not deliver. A guard that over-estimates costs a
  # warning; one that under-estimates costs the session.
  # Not clamped at 1: a backend where the loop's own temporaries cost
  # more than the transient it removes has a floor *above* 1, and saying
  # so is more use than pretending the knob is free. TabICL measures 1.22
  # -- using the factor there costs about a fifth more memory.
  fl <- max(0, co$spmf_floor %||% 1)
  act_mult <- co$act_copies * (fl + (1 - fl) / spmf)
  att_mult <- co$att_copies / spmf
  # `res` is not divided by `spmf` and not multiplied by `act_copies`:
  # these are the few whole-table tensors a chunked stage keeps outside
  # its loop, so neither kind of chunking touches them. Its own small
  # multiple covers the copy a `torch_cat` makes at the end.
  res_mult <- co$res_copies %||% 2
  # The summary pre-pass, where one exists, is charged its own count --
  # defaulting to the forward's, which over-estimates and is therefore
  # the safe thing to do until a sweep says otherwise.
  pre_mult <- (co$prepass_copies %||% co$act_copies) / spmf
  # One `act_copies` per backend assumes every stage holds the same
  # number of live copies of its activation. They do not, and it only
  # showed once chunking changed *which* stage dominates: the constant
  # was fitted where the column stage was largest, and a chunked run
  # hands it to the in-context stage instead. So that stage carries its
  # own, defaulting to the shared one where nothing has measured it.
  icl_copies <- co$icl_copies %||% co$act_copies
  icl_mult <- 1 + (icl_copies - 1) / spmf
  stage_bytes <- vapply(stages, function(s) {
    mult <- if (identical(s$copies %||% "", "icl")) icl_mult else act_mult
    DTYPE_BYTES * (mult * (s$act %||% 0) + att_mult * (s$att %||% 0) +
                   res_mult * (s$res %||% 0) + pre_mult * (s$prepass %||% 0))
  }, numeric(1))
  names(stage_bytes) <- vapply(stages, function(s) s$name %||% "", character(1))
  n_est <- max(1, suppressWarnings(as.numeric(n_estimators)))
  if (!isTRUE(is.finite(n_est))) n_est <- 1
  # Capped, because the two backends measured disagree about the shape:
  # TabICL climbs across the whole sweep and TabFM saturates after two
  # members. The line covers the first, the cap keeps it from inflating
  # the second's 32-member default beyond anything ever observed.
  ens <- min(1 + (co$ensemble_log2_factor %||% 0) * log2(n_est),
             co$ensemble_cap %||% Inf)
  transient <- max(c(stage_bytes, 0)) * ens
  modelled <- (co$intercept_bytes + weights + persistent + transient) *
    co$safety_factor
  list(
    stage_bytes = stage_bytes,
    ensemble_multiplier = ens,
    transient = transient,
    # Two regimes, because the measurements show two. Above a gigabyte or
    # so the peak is the activation and the model tracks it. Below that
    # it is the process -- R, libtorch, and whichever arena the allocator
    # took -- and it does not fall with the input: TabPFN v2.5 at 8
    # features measures 2.1 GB at 800 rows and 1.1 GB at 1,600,
    # reproducibly. `floor_bytes` is the largest peak observed where the
    # activation was negligible, so the estimate never drops below what
    # this backend costs to do nothing.
    peak = max(modelled, co$floor_bytes %||% 0)
  )
}


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------

#' Estimate peak memory for a fit / predict pair, before allocating
#'
#' Predicts the high-water mark of a `fit()` + `predict()` cycle at given
#' dimensions and compares it with what the machine has free *now*.
#' Nothing here builds a tensor, and nothing here needs the weights: an
#' artifact directory or the contents of a `config.json` is enough, so
#' the check can run before a multi-gigabyte download.
#'
#' The answer is a breakdown, because which term dominates decides what
#' to do about it:
#'
#' * `weights_bytes` -- the checkpoint, resident for the object's life.
#'   Fixed. Only a smaller model changes it.
#' * `persistent_bytes` -- the fitted context and, when `kv_cache = TRUE`,
#'   one cache per ensemble member held for the whole `predict()` call.
#'   `kv_cache` does not save memory; it *moves* it here from the
#'   transient term, and multiplies it by `n_estimators`.
#' * `transient_bytes` -- the forward pass. The largest single stage, not
#'   the sum of them, and counted once for the whole ensemble: members
#'   run sequentially, so `n_estimators` does not multiply this.
#'
#' @section Accuracy:
#' The shipped constants are anchored to observed working and failing
#' runs rather than fitted to a measurement grid -- see
#' `inst/memory/README.md` and the `calibration` field of the result. They
#' are set to err high: for a guard, a false "tight" costs a warning and
#' a false "ok" costs the session.
#'
#' @param model A `tabfound_model`, an artifact directory, a model id
#'   with artifacts already downloaded, or a `config.json` read into a
#'   list.
#' @param n_context Training rows the model will be fitted on.
#' @param n_query Rows to predict. Only the largest chunk is ever
#'   resident, so a `predict_chunk_size` below this caps the transient
#'   term.
#' @param n_features Predictor columns.
#' @param ... Backend knobs that change the peak, overriding both the
#'   predictor's defaults and (for a model object) the arguments it was
#'   built with: `n_estimators`, `kv_cache`, `save_peak_memory_factor`,
#'   `predict_chunk_size`.
#' @param backend Optional backend name, when detection from the config
#'   is ambiguous.
#' @param device Device the run will use. Constants are per-device;
#'   only `"cpu"` ships today.
#' @param task `"classification"` or `"regression"`, when `model` is a
#'   config that does not say.
#' @param available,total Override the machine probe, in bytes. Mostly
#'   for asking "would this fit on a 16 GB laptop?" and for tests.
#' @return An object of class `tabfound_memory_estimate`: a list with
#'   `weights_bytes`, `persistent_bytes`, `transient_bytes`,
#'   `total_peak_bytes`, `system_total_bytes`, `system_avail_bytes`,
#'   `verdict` (`"ok"`, `"tight"`, `"exceeds"`, or `"unknown"` when the
#'   machine could not be probed) and `suggestions`.
#' @examples
#' \dontrun{
#' # Before downloading anything, from a config you already have:
#' estimate_peak_memory(jsonlite::fromJSON("config.json"),
#'                      n_context = 6426, n_query = 714, n_features = 90)
#'
#' # Would it fit on a 16 GB laptop with half of it free?
#' estimate_peak_memory(clf, 6426, 714, 90, available = 8e9)
#' }
#' @seealso [list_models()] for what a download would cost on disk.
#' @export
estimate_peak_memory <- function(model, n_context, n_query = 0L, n_features,
                                 ..., backend = NULL, device = NULL,
                                 task = NULL, available = NULL, total = NULL) {
  require_suggested("jsonlite")
  n_context  <- as.numeric(n_context)
  n_query    <- as.numeric(n_query)
  n_features <- as.numeric(n_features)
  if (!all(is.finite(c(n_context, n_query, n_features))) ||
      any(c(n_context, n_query, n_features) < 0)) {
    cli::cli_abort("{.arg n_context}, {.arg n_query} and {.arg n_features} \\
                    must be non-negative numbers.")
  }

  tgt  <- .resolve_memory_target(model, backend = backend, device = device,
                                 task = task)
  task <- task %||% tgt$task %||% "classification"
  opts <- .resolve_memory_opts(tgt$backend, task, tgt$opts, list(...))
  dev  <- tgt$device %||% "cpu"

  terms <- .peak_terms_for(tgt$backend, n_context, n_query, n_features,
                           opts, tgt$config)
  co <- .memory_coefs(tgt$backend$name, device = dev)
  if (is.null(co)) {
    # Constants for another device are not transferable, and a
    # confidently wrong number is worse here than no number: this is a
    # guard. GPU/MPS constants can be added to the coef files without
    # touching this function.
    cli::cli_abort(c(
      "No memory constants for {.val {tgt$backend$name}} on \\
       {.val {dev}}.",
      i = "Only {.val cpu} is calibrated today; see {.file inst/memory/}."
    ))
  }

  spmf <- suppressWarnings(as.numeric(opts$save_peak_memory_factor %||% 1))
  if (!isTRUE(is.finite(spmf)) || spmf < 1) spmf <- 1

  # The fitted context is R doubles, not tensors, and each ensemble member
  # keeps its own preprocessor state alongside it.
  n_est <- max(1, suppressWarnings(as.numeric(.opt_n_estimators(opts, terms))))
  if (!isTRUE(is.finite(n_est))) n_est <- 1
  context_bytes <- n_context * n_features * 8 *
    (1 + n_est * co$prep_state_factor)
  persistent <- context_bytes + DTYPE_BYTES * (terms$persistent %||% 0)

  weights <- tgt$weights
  weights_source <- tgt$weights_source
  if (!isTRUE(is.finite(weights))) {
    weights <- co$weights_fallback_bytes
    weights_source <- "backend constant (approximate)"
  }
  if (!isTRUE(is.finite(weights))) {
    weights <- 0
    weights_source <- "unknown (not counted)"
  }

  assembled <- .peak_from_terms(terms$stages, weights, persistent, co, spmf,
                                n_estimators = n_est)
  stage_bytes <- assembled$stage_bytes
  transient <- assembled$transient
  peak <- assembled$peak

  sysmem <- .system_memory()
  sys_total <- total     %||% sysmem$total
  sys_avail <- available %||% sysmem$available

  verdict <- .memory_verdict(peak, sys_avail)
  out <- list(
    backend            = tgt$backend$name,
    device             = dev,
    task               = task,
    dims               = c(n_context = n_context, n_query = n_query,
                           n_features = n_features),
    opts               = opts[intersect(names(opts),
                                        c("n_estimators", "kv_cache",
                                          "save_peak_memory_factor",
                                          "predict_chunk_size",
                                          "row_chunk_size", "col_chunk_size"))],
    weights_bytes      = weights,
    weights_source     = weights_source,
    persistent_bytes   = persistent,
    transient_bytes    = transient,
    stage_bytes        = stage_bytes,
    total_peak_bytes   = peak,
    system_total_bytes = sys_total,
    system_avail_bytes = sys_avail,
    system_source      = if (is.null(available)) sysmem$source else "supplied",
    verdict            = verdict,
    calibration        = co$source,
    suggestions        = .memory_suggestions(verdict, tgt$backend$name, opts,
                                             weights, persistent, transient)
  )
  class(out) <- "tabfound_memory_estimate"
  out
}

# TabPFN's ensemble size is the length of its loaded config dump rather
# than an integer argument, so the backend reports it through the terms.
# @keywords internal
.opt_n_estimators <- function(opts, terms) {
  terms$n_estimators %||% opts$n_estimators %||% 1
}

# Ask the backend for its shapes, with a clear failure when it has none.
# @keywords internal
.peak_terms_for <- function(backend, n_context, n_query, n_features, opts,
                            config) {
  fn <- backend$peak_terms
  if (!is.function(fn)) {
    cli::cli_abort(c(
      "Backend {.val {backend$name}} does not describe its memory scaling.",
      i = "A backend contributes {.arg peak_terms} to {.fn register_backend}; \\
           without it there is nothing to estimate from."
    ))
  }
  terms <- fn(n_context = n_context, n_query = n_query,
              n_features = n_features, opts = opts, config = config)
  if (!is.list(terms) || !length(terms$stages)) {
    cli::cli_abort(
      "{.val {backend$name}}'s {.arg peak_terms} returned no stages."
    )
  }
  terms
}

# Leave the machine room to breathe. The SIGKILL that motivated all this
# happened on a box with 41 GB of headroom on paper: consuming everything
# that is nominally free is how a run gets chosen by the OS reaper, so
# the verdict is taken against available memory minus a reserve.
# @keywords internal
.memory_verdict <- function(peak, available) {
  if (!isTRUE(is.finite(available))) return("unknown")
  reserve <- min(2e9, 0.25 * available)
  eff <- max(available - reserve, 0)
  if (eff <= 0) return("exceeds")
  if (peak <= 0.5 * eff) "ok" else if (peak <= eff) "tight" else "exceeds"
}


# ---------------------------------------------------------------------------
# Saying something useful about it
# ---------------------------------------------------------------------------

# Suggestions are ordered by how much they actually move: the dominant
# term first. Telling someone to turn off a KV cache they are not using,
# or to chunk a run whose weights are the problem, is noise.
# @keywords internal
.memory_suggestions <- function(verdict, backend, opts, weights,
                                persistent, transient) {
  if (identical(verdict, "ok") || identical(verdict, "unknown")) {
    return(character())
  }
  s <- character()

  if (transient >= max(weights, persistent)) {
    s <- c(s, "Reduce {.arg n_context}: the transient peak grows with \\
                the context, and subsampling it is the one change that \\
                always helps.")
    if (isTRUE(opts$kv_cache)) {
      s <- c(s, "Set {.code kv_cache = FALSE}: it holds one cache per \\
                  ensemble member for the whole {.fn predict} call.")
    }
    chunk <- opts$predict_chunk_size
    if (!is.null(chunk) && is.finite(chunk) && chunk > 128) {
      s <- c(s, "Lower {.arg predict_chunk_size} (now {chunk}): only one \\
                  chunk of query rows is resident at a time.")
    }
    # Ordered by how much each moves. On v3 the stage chunking is the
    # dominant one -- it bounds the `(rows, columns, embedding)` tensor
    # itself, where `save_peak_memory_factor` only bounds the temporaries
    # made from it -- so it goes first, and it is the one knob that
    # changes how the peak grows with rows rather than by how much.
    if (identical(backend, "tabpfn3")) {
      rc <- suppressWarnings(as.numeric(opts$row_chunk_size %||% NA))
      if (is.null(opts$row_chunk_size)) {
        # Explicitly off, which is the one setting that makes the peak
        # grow with every row rather than with a chunk of them.
        s <- c(s, "Leave {.arg row_chunk_size} at its default instead of \\
                    {.code NULL}: stages 0-2 then hold one chunk of rows \\
                    rather than all of them.")
      } else if (!isTRUE(is.finite(rc))) {
        # The default -- the checkpoint's own, 2048 on every released v3.
        s <- c(s, "Lower {.arg row_chunk_size} below the checkpoint's \\
                    default: stages 0-2 hold one chunk of rows at a time, \\
                    so it is the term that decides how the peak grows.")
      } else if (rc > 256) {
        s <- c(s, "Lower {.arg row_chunk_size} (now {rc}): stages 0-2 hold \\
                    one chunk of rows at a time, so it is the term that \\
                    decides how the peak grows.")
      }
    }
    if (backend %in% c("tabpfn26", "tabpfn3") &&
        is.null(opts$save_peak_memory_factor)) {
      s <- c(s, "Set {.arg save_peak_memory_factor} (try {.val 4}): it \\
                  splits each sublayer's work, for the same answer.")
    }
    if (identical(backend, "tabpfn26")) {
      s <- c(s, "This generation has no stage chunking -- only TabPFN v3 \\
                  bounds the row axis of its embedding tensor. At these \\
                  dimensions v3 will cost far less.")
    }
  }

  if (persistent > transient && isTRUE(opts$kv_cache)) {
    s <- c(s, "Set {.code kv_cache = FALSE}: at these dimensions the caches \\
                are the largest term, and they scale with {.arg n_estimators}.")
  }

  if (weights >= max(transient, persistent)) {
    s <- c(s, "The checkpoint itself is the largest term here; only a \\
                smaller model changes that.")
  }

  if (identical(backend, "mitra")) {
    s <- c(s, "Mitra attends across rows *and* columns, so its activation is \\
                the steepest in the package; its published target is small \\
                tables. TabPFN v3 or TabICL will cost far less at these \\
                dimensions.")
  } else if (identical(backend, "tabfm")) {
    s <- c(s, "TabFM keeps 1.6 B float32 parameters resident before any \\
                activation exists; TabPFN v3 or TabICL will cost far less.")
  }
  # Resolve the templates here, where the values they mention are in
  # scope, rather than leaving live braces in a string that gets printed
  # somewhere else entirely.
  env <- environment()
  vapply(s, function(txt) {
    cli::format_inline(txt, .envir = env, keep_whitespace = FALSE)
  }, character(1), USE.NAMES = FALSE)
}

# @keywords internal
.fmt_bytes <- function(x) {
  if (!isTRUE(is.finite(x))) return("unknown")
  if (x >= 1e9) return(paste0(format(round(x / 1e9, 1), nsmall = 1), " GB"))
  if (x >= 1e6) return(paste0(round(x / 1e6), " MB"))
  paste0(round(x / 1e3), " kB")
}

#' @export
print.tabfound_memory_estimate <- function(x, ...) {
  # cli treats `{.name(...)}` as a style, so every formatted number is
  # built here and interpolated as a plain value.
  b <- lapply(x[c("weights_bytes", "persistent_bytes", "transient_bytes",
                  "total_peak_bytes", "system_avail_bytes",
                  "system_total_bytes")], .fmt_bytes)
  stage <- names(which.max(x$stage_bytes))

  cli::cli_text("{.strong memory preflight} <{x$backend}> on {.val {x$device}}")
  cli::cli_text("{.emph {x$dims[['n_context']]} context x \\
                 {x$dims[['n_features']]} features, \\
                 {x$dims[['n_query']]} to predict}")
  sym <- switch(x$verdict, ok = "v", tight = "!", exceeds = "x", "i")
  cli::cli_bullets(c(
    "*" = "weights {.strong {b$weights_bytes}} ({x$weights_source})",
    "*" = "persistent {.strong {b$persistent_bytes}}",
    "*" = "transient {.strong {b$transient_bytes}}, \\
           largest stage {.val {stage}}",
    ">" = "peak {.strong {b$total_peak_bytes}}",
    "i" = "available {.strong {b$system_avail_bytes}} of {b$system_total_bytes}"
  ))
  cli::cli_bullets(set_names(
    paste0("verdict: ", toupper(x$verdict)), sym
  ))
  if (length(x$suggestions)) {
    cli::cli_bullets(set_names(x$suggestions, rep("i", length(x$suggestions))))
  }
  if (identical(x$calibration, "anchor")) {
    cli::cli_text("{.emph Constants are anchored to observed runs, not yet \\
                   fitted to a measurement grid.}")
  }
  invisible(x)
}


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------

#' How large a table can this model actually take?
#'
#' The inverse of [estimate_peak_memory()]: instead of asking what a
#' given table costs, ask how big a table fits in a given amount of
#' memory. Answered by bisection over the estimator, which is monotone in
#' every dimension, so there is exactly one boundary to find.
#'
#' Both `n_features` and `available` may be vectors; the result is one
#' row per combination, which is what makes it a table. Nothing here
#' needs weights — a catalogued model resolves against the shipped
#' architecture dimensions, so this works before any download.
#'
#' @section What "fits" means:
#' `verdict = "ok"` returns the largest context the estimator calls
#' comfortable, which leaves roughly half the usable headroom unspent.
#' That is the number to plan around. `verdict = "tight"` returns the
#' largest it would not refuse — closer to the true ceiling, and closer
#' to the point where another process arriving takes the run down with
#' it. The gap between them is the margin the guard is asking for.
#'
#' @param model A `tabfound_model`, a catalogued model id, an artifact
#'   directory, or a `config.json` read into a list.
#' @param n_features Predictor columns. May be a vector.
#' @param available Memory the run will have, in bytes. May be a vector.
#'   Defaults to what this machine has free now.
#' @param n_query Rows to predict in the same call.
#' @param verdict Largest context that still earns at least this verdict:
#'   `"ok"` (default) or `"tight"`.
#' @param max_context Upper bound for the search.
#' @param ... Backend knobs, exactly as [estimate_peak_memory()] takes
#'   them: `n_estimators`, `kv_cache`, `predict_chunk_size`,
#'   `save_peak_memory_factor`.
#' @return A data frame with `n_features`, `available_bytes`,
#'   `max_context` and `peak_bytes` (what that context is estimated to
#'   cost). `max_context` is `0` when even one row does not fit.
#' @examples
#' \dontrun{
#' # How far does TabICL get on a 16 GB laptop with 10 GB free?
#' memory_envelope("tabicl-v2-classifier", n_features = c(10, 50, 100),
#'                 available = 10e9)
#'
#' # The same question across a range of machines.
#' memory_envelope("tabpfn-v3-classifier", n_features = 50,
#'                 available = c(8, 16, 32, 64) * 1e9)
#' }
#' @seealso [estimate_peak_memory()]
#' @export
memory_envelope <- function(model, n_features, available = NULL,
                            n_query = 1000, verdict = c("ok", "tight"),
                            max_context = 1e6, ...) {
  verdict <- match.arg(verdict)
  accept <- if (identical(verdict, "ok")) "ok" else c("ok", "tight")
  available <- available %||% .system_memory()$available
  if (!length(available) || !any(is.finite(available))) {
    cli::cli_abort(c(
      "Could not work out how much memory to plan against.",
      i = "Pass {.arg available} in bytes."
    ))
  }

  grid <- expand.grid(n_features = n_features, available = available,
                      KEEP.OUT.ATTRS = FALSE)
  out <- lapply(seq_len(nrow(grid)), function(i) {
    p <- grid$n_features[i]
    a <- grid$available[i]
    fits <- function(n) {
      est <- estimate_peak_memory(model, n_context = n, n_query = n_query,
                                  n_features = p, available = a, ...)
      list(ok = est$verdict %in% accept, peak = est$total_peak_bytes)
    }
    if (!fits(1)$ok) {
      return(data.frame(n_features = p, available_bytes = a,
                        max_context = 0, peak_bytes = NA_real_))
    }
    # Monotone in `n_context`, so a plain bisection is exact to the row.
    lo <- 1; hi <- max_context
    if (fits(hi)$ok) {
      return(data.frame(n_features = p, available_bytes = a,
                        max_context = hi, peak_bytes = fits(hi)$peak))
    }
    while (hi - lo > 1) {
      mid <- floor((lo + hi) / 2)
      if (fits(mid)$ok) lo <- mid else hi <- mid
    }
    data.frame(n_features = p, available_bytes = a, max_context = lo,
               peak_bytes = fits(lo)$peak)
  })
  do.call(rbind, out)
}


#' Pick stage-chunk sizes that fit the memory you have
#'
#' The reference implementation halves its chunk size and retries when an
#' allocation fails. R cannot do that: a libtorch allocation failure kills
#' the process outright, with no condition to catch and nothing left to
#' retry from — which is the premise the whole preflight rests on. So the
#' chunk size has to be chosen *before* the run, from the estimate, and
#' this is the function that does it.
#'
#' Only TabPFN v3 has stage chunking. For any other backend this returns
#' the run's estimate with `row_chunk_size = NA`, which is the honest
#' answer: there is no knob here to turn.
#'
#' @section What it searches:
#' Row chunks from the checkpoint's own default downwards, halving, and
#' then column chunks the same way if the row axis alone does not get
#' there. The largest pair that fits wins — smaller chunks cost loop
#' overhead and buy nothing once the estimate clears.
#'
#' The order is not arbitrary. A row chunk bounds the forward loop; a
#' column chunk bounds the summary pre-pass the loop cannot start
#' without. On a table that is long rather than wide the first is what
#' binds, and on a wide one the second takes over — measured on TabICL at
#' 12,000 × 300, row chunking alone leaves 36.9 GB and adding a column
#' chunk of 8 takes it to 29.8 GB.
#'
#' A `NULL` row chunk is never suggested: it is the unchunked pass, and
#' if that fitted there would be nothing to ask.
#'
#' @param model A `tabfound_model`, a catalogued model id, an artifact
#'   directory, or a `config.json` read into a list.
#' @param n_context,n_query,n_features Dimensions of the intended run.
#' @param available Memory to plan against, in bytes. Defaults to what
#'   this machine has free now.
#' @param verdict Accept a chunk size that earns at least this verdict:
#'   `"ok"` (default) or `"tight"`.
#' @param min_row_chunk Smallest row chunk worth suggesting.
#' @param ... Other backend knobs, as [estimate_peak_memory()] takes them.
#' @return A list with `row_chunk_size`, `col_chunk_size`, `peak_bytes`,
#'   `verdict` and `feasible`. When nothing fits, `feasible` is `FALSE`
#'   and the sizes are the smallest tried — the run is too big for this
#'   machine whatever the chunking, and `n_context` is the thing to
#'   change.
#' @examples
#' \dontrun{
#' # 200,000 rows on a laptop with 8 GB free: how small do the chunks
#' # have to be?
#' suggest_chunk_sizes("tabpfn-v3-classifier", n_context = 2e5,
#'                     n_features = 50, available = 8e9)
#' }
#' @seealso [estimate_peak_memory()], [memory_envelope()]
#' @export
suggest_chunk_sizes <- function(model, n_context, n_query = 1000, n_features,
                                available = NULL, verdict = c("ok", "tight"),
                                min_row_chunk = 64L, ...) {
  verdict <- match.arg(verdict)
  accept <- if (identical(verdict, "ok")) "ok" else c("ok", "tight")

  est <- function(rc) {
    estimate_peak_memory(model, n_context = n_context, n_query = n_query,
                         n_features = n_features, available = available,
                         row_chunk_size = rc, ...)
  }
  est2 <- function(rc, cc) {
    estimate_peak_memory(model, n_context = n_context, n_query = n_query,
                         n_features = n_features, available = available,
                         row_chunk_size = rc, col_chunk_size = cc, ...)
  }

  # `NA` is "the checkpoint's own", which is where the search starts.
  top <- est(NA_integer_)
  if (!isTRUE(top$backend %in% c("tabpfn3", "tabicl", "tabfm"))) {
    return(list(row_chunk_size = NA_integer_, col_chunk_size = NA_integer_,
                peak_bytes = top$total_peak_bytes, verdict = top$verdict,
                feasible = top$verdict %in% accept,
                note = sprintf("the %s backend has no stage chunking",
                               top$backend)))
  }

  # The search starts at whatever the run would use if nobody said
  # anything, which is the checkpoint's own value -- not the literal `NA`
  # that stands for it in `opts`.
  start <- suppressWarnings(as.numeric(top$opts$row_chunk_size %||% NA))
  if (!isTRUE(is.finite(start))) {
    start <- .stage_row_chunk(list(), .resolve_memory_target(model)$config)
  }
  halving <- function(from, floor_at) {
    out <- as.integer(max(floor_at, from))
    while (out[length(out)] > floor_at) {
      out <- c(out, as.integer(max(floor_at, out[length(out)] %/% 2L)))
    }
    unique(out)
  }
  sizes <- halving(start, min_row_chunk)

  for (rc in sizes) {
    e <- est(rc)
    if (e$verdict %in% accept) {
      return(list(row_chunk_size = rc, col_chunk_size = NA_integer_,
                  peak_bytes = e$total_peak_bytes, verdict = e$verdict,
                  feasible = TRUE, note = NULL))
    }
  }

  # The rows are as small as they go and it still does not fit, so what
  # is left is the pre-pass. Search the columns at the smallest row chunk.
  rc <- sizes[length(sizes)]
  cols <- halving(n_features, 1L)
  for (cc in cols) {
    e <- est2(rc, cc)
    if (e$verdict %in% accept) {
      return(list(row_chunk_size = rc, col_chunk_size = cc,
                  peak_bytes = e$total_peak_bytes, verdict = e$verdict,
                  feasible = TRUE, note = NULL))
    }
  }
  smallest <- est2(rc, cols[length(cols)])
  list(row_chunk_size = rc, col_chunk_size = cols[length(cols)],
       peak_bytes = smallest$total_peak_bytes, verdict = smallest$verdict,
       feasible = FALSE,
       note = "no chunk size fits; reduce n_context or free memory")
}


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

# What `fit()` and `predict()` consult. It has one job and one hard rule:
# it must never be the reason a run fails. Anything it cannot work out --
# an uncalibrated device, a backend without `peak_terms()`, a machine it
# cannot probe -- means silence, not an error and not a guess.
#
# `options(tabfound.memory_guard =)`
#   "warn"  (default) a `cli` warning naming the estimate, the available
#           memory and the knobs; the run proceeds.
#   "error" refuse, with the same message.
#   "off"   silence.

.guard_seen <- new.env(parent = emptyenv())

#' Forget which memory warnings have already been given
#'
#' The guard warns once per distinct situation, so a chained-equations
#' loop that fits the same model two hundred times says it once. This
#' clears that record; mostly for tests.
#' @keywords internal
reset_memory_guard <- function() {
  rm(list = ls(.guard_seen), envir = .guard_seen)
  invisible(NULL)
}

# @keywords internal
.memory_guard_mode <- function() {
  mode <- getOption("tabfound.memory_guard", "warn")
  if (!is.character(mode) || length(mode) != 1L ||
      !mode %in% c("warn", "error", "off")) {
    return("warn")
  }
  mode
}

#' Check a fit / predict call against available memory before it runs
#'
#' @param object A `tabfound_model`.
#' @param n_context,n_query,n_features Dimensions of the call.
#' @param stage `"fit"` or `"predict"`, which changes only the wording.
#' @return Invisibly, the estimate when one was made, else `NULL`.
#' @keywords internal
.memory_guard <- function(object, n_context, n_query, n_features,
                          stage = c("fit", "predict")) {
  stage <- match.arg(stage)
  mode <- .memory_guard_mode()
  if (identical(mode, "off")) return(invisible(NULL))

  # The estimate is a function of these inputs, so a situation already
  # judged is a situation that need not be judged again -- and the
  # judging is not free: it walks the module tree and probes the system.
  # A chained-equations loop presents the same handful of situations
  # thousands of times over.
  #
  # Only under `"warn"`, which is the mode that is about *saying*
  # something once. `"error"` is a live gate and re-checks every call,
  # because what it guards against is the memory available now.
  in_key <- paste(object$backend, object$device, stage,
                  n_context, n_query, n_features, sep = "/")
  if (identical(mode, "warn") && !is.null(.guard_seen[[in_key]])) {
    return(invisible(.guard_seen[[in_key]]))
  }

  est <- tryCatch(
    estimate_peak_memory(object, n_context = n_context, n_query = n_query,
                         n_features = n_features),
    error = function(e) NULL,
    warning = function(w) NULL
  )
  if (is.null(est) || est$verdict %in% c("ok", "unknown")) {
    # Remembering the quiet verdicts too: those are the ones an MI loop
    # meets over and over, and re-deriving "ok" costs the same as
    # deriving it.
    if (identical(mode, "warn")) {
      assign(in_key, est %||% NA, envir = .guard_seen)
    }
    return(invisible(est))
  }

  msg <- .memory_guard_message(est, stage)
  if (identical(mode, "error")) {
    cli::cli_abort(c(msg, i = "Set {.code options(tabfound.memory_guard = \\
                               \"warn\")} to make this a warning instead."))
  }

  # One warning per distinct situation. A chained-equations loop calls
  # `fit()` once per variable per iteration, and two hundred copies of
  # the same paragraph would bury the one thing worth reading.
  assign(in_key, est, envir = .guard_seen)

  cli::cli_warn(c(msg, i = "This is said once per situation; \\
                            {.code options(tabfound.memory_guard = \"off\")} \\
                            silences it."))
  invisible(est)
}

# @keywords internal
.memory_guard_message <- function(est, stage) {
  peak  <- .fmt_bytes(est$total_peak_bytes)
  avail <- .fmt_bytes(est$system_avail_bytes)
  what <- if (identical(stage, "fit")) {
    "Fitting this model and predicting from it"
  } else {
    "This prediction"
  }
  head <- if (identical(est$verdict, "exceeds")) {
    cli::format_inline(
      "{what} needs about {peak}, against {avail} available."
    )
  } else {
    cli::format_inline(
      "{what} needs about {peak}, against {avail} available -- \\
       little headroom."
    )
  }
  # Naming the failure mode is the point. An R user reasonably expects a
  # condition they could have caught; there is none, and knowing that is
  # what makes the warning worth reading rather than worth suppressing.
  body <- cli::format_inline(
    "{.val {est$backend}} at {est$dims[['n_context']]} context row{?s} x \\
     {est$dims[['n_features']]} feature{?s}. Running out here kills the R \\
     process from inside libtorch: no error, no traceback, no output."
  )
  c(head, x = body, set_names(est$suggestions,
                              rep("i", length(est$suggestions))))
}

# The query rows a later `predict()` will have in flight, which is what
# `fit()` has to be checked against: the fit itself allocates almost
# nothing, and the peak that matters arrives one call later.
# @keywords internal
.predict_chunk_of <- function(object) {
  chunk <- tryCatch({
    bk <- get_backend(object$backend)
    opts <- .resolve_memory_opts(bk, object$task,
                                 object$model_ref$args %||% list(), list())
    suppressWarnings(as.numeric(opts$predict_chunk_size %||% NA))
  }, error = function(e) NA_real_)
  if (!isTRUE(is.finite(chunk))) 0 else chunk
}
