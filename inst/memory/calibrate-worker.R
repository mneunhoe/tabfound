#!/usr/bin/env Rscript
#
# One grid point, in its own process. Driven by `calibrate.R`; not meant
# to be run by hand.
#
# It gets a process to itself for two reasons. Resident memory is a
# high-water mark that never comes back down within a process, so a
# second point measured in the same one would inherit the first's floor.
# And the interesting points are the ones that die: an allocation failure
# here aborts *this* process, which the driver observes and records, and
# leaves the harness itself running.
#
# Usage: Rscript calibrate-worker.R <spec.json>

`%||%` <- function(x, y) if (is.null(x)) y else x

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: calibrate-worker.R <spec.json>")
spec <- jsonlite::fromJSON(args[[1]], simplifyVector = TRUE)

# The driver watches this file to learn which process to sample, and the
# marker file below to learn which part of its life was the measurement.
writeLines(as.character(Sys.getpid()), spec$pid_file)

mark <- function(what) {
  cat(sprintf("%s %.6f\n", what, as.numeric(Sys.time())),
      file = spec$marker_file, append = TRUE)
}

suppressPackageStartupMessages(library(tabfound))
# The thing being calibrated must not be in the way of the measurement.
options(tabfound.memory_guard = "off")

set.seed(spec$seed %||% 1L)

n_ctx <- as.integer(spec$n_context)
n_qry <- as.integer(spec$n_query)
n_ftr <- as.integer(spec$n_features)

X <- matrix(stats::rnorm(n_ctx * n_ftr), n_ctx, n_ftr)
y <- factor(sample(c("a", "b"), n_ctx, replace = TRUE))
X_new <- matrix(stats::rnorm(n_qry * n_ftr), n_qry, n_ftr)

opts <- as.list(spec$opts %||% list())
is_reg <- identical(spec$task, "regression")
ctor <- if (is_reg) tabular_regressor else tabular_classifier
if (is_reg) y <- as.numeric(y)

mark("load_start")
model <- suppressMessages(
  do.call(ctor, c(list(model = spec$model_dir, device = spec$device %||% "cpu"),
                  opts))
)
# Touch a parameter so the weights are certainly resident, not merely
# mapped: a lazily paged checkpoint would flatter every measurement.
invisible(sum(as.numeric(model$model$parameters[[1]]$sum()$cpu())))
mark("loaded")

t0 <- Sys.time()
mark("work_start")
model <- fit(model, X, y)
invisible(predict(model, X_new, type = "prob"))
mark("work_end")
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

jsonlite::write_json(
  list(status = "ok", elapsed_sec = elapsed),
  spec$result_file, auto_unbox = TRUE
)
