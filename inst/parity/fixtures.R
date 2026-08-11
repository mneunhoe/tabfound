# Deterministic parity fixtures.
#
# Both the R package and the reference Python implementation must see
# *identical bytes*, so fixtures are generated once here, cast to
# float32, and written to safetensors. Regenerating them invalidates
# every stored reference prediction — bump `FIXTURE_VERSION` and re-run
# the Python side when you do.
#
#   Rscript inst/parity/fixtures.R [outdir]

FIXTURE_VERSION <- 2L

#' Build the fixture list
#'
#' Each fixture is a list with `x_train`, `y_train`, `x_test`, `task`
#' (`"classification"` or `"regression"`) and a short `note` describing
#' what it exercises.
#' @keywords internal
build_parity_fixtures <- function() {
  fx <- list()

  # --- 1. iris, 3-class, all-numeric, no missings -------------------------
  set.seed(1)
  n  <- nrow(iris)
  tr <- sort(sample.int(n, 100L))
  te <- setdiff(seq_len(n), tr)
  fx$clf_iris <- list(
    task    = "classification",
    note    = "iris, 3 classes, 4 numeric features, no NaN",
    x_train = as.matrix(iris[tr, 1:4]),
    y_train = as.integer(iris[tr, 5]) - 1L,
    x_test  = as.matrix(iris[te, 1:4])
  )

  # --- 2. binary, missing values, a constant column ----------------------
  set.seed(42)
  n_tr <- 80L; n_te <- 40L; p <- 6L
  X <- matrix(rnorm((n_tr + n_te) * p), ncol = p)
  X[, 5] <- 1.0                                   # constant column
  X[cbind(sample.int(n_tr + n_te, 25L),
          sample.int(p - 1L, 25L, replace = TRUE))] <- NA_real_
  lin <- 1.3 * X[, 1] - 0.8 * X[, 2] + 0.5 * X[, 3] * X[, 4]
  lin[is.na(lin)] <- 0
  y <- as.integer(lin > median(lin, na.rm = TRUE))
  fx$clf_binary_missing <- list(
    task    = "classification",
    note    = "binary, 6 features incl. a constant column and 25 NaNs",
    x_train = X[seq_len(n_tr), , drop = FALSE],
    y_train = y[seq_len(n_tr)],
    x_test  = X[n_tr + seq_len(n_te), , drop = FALSE]
  )

  # --- 3. regression on iris ---------------------------------------------
  fx$reg_iris <- list(
    task    = "regression",
    note    = "iris Sepal.Length from 3 numeric features",
    x_train = as.matrix(iris[tr, 2:4]),
    y_train = as.numeric(iris[tr, 1]),
    x_test  = as.matrix(iris[te, 2:4])
  )

  # --- 4. synthetic nonlinear regression, skewed target ------------------
  set.seed(7)
  n_tr <- 120L; n_te <- 60L; p <- 5L
  X <- matrix(rnorm((n_tr + n_te) * p), ncol = p)
  y <- exp(0.7 * X[, 1] + 0.4 * sin(3 * X[, 2]) + 0.2 * X[, 3] * X[, 4]) +
    rnorm(n_tr + n_te, sd = 0.05)
  fx$reg_skewed <- list(
    task    = "regression",
    note    = "synthetic nonlinear, log-normal-ish target (exercises target transforms)",
    x_train = X[seq_len(n_tr), , drop = FALSE],
    y_train = y[seq_len(n_tr)],
    x_test  = X[n_tr + seq_len(n_te), , drop = FALSE]
  )

  # --- 5/6. Small fixtures for expensive backends ------------------------
  # TabFM is a 1.6 B-parameter model; a full-size fixture makes each
  # parity iteration a multi-minute affair. These are deliberately tiny
  # so the network can be checked cheaply, and the larger fixtures above
  # then confirm nothing depends on the size.
  set.seed(2024)
  n_tr <- 48L; n_te <- 12L; p <- 4L
  X <- matrix(rnorm((n_tr + n_te) * p), ncol = p)
  lin <- 1.1 * X[, 1] - 0.9 * X[, 2] + 0.6 * X[, 3]
  fx$clf_tiny <- list(
    task    = "classification",
    note    = "48/12 rows, 4 features, 3 classes -- cheap enough for a 1.6B model",
    x_train = X[seq_len(n_tr), , drop = FALSE],
    y_train = as.integer(cut(lin[seq_len(n_tr)], 3L)) - 1L,
    x_test  = X[n_tr + seq_len(n_te), , drop = FALSE]
  )
  fx$reg_tiny <- list(
    task    = "regression",
    note    = "48/12 rows, 4 features, continuous target",
    x_train = X[seq_len(n_tr), , drop = FALSE],
    y_train = lin[seq_len(n_tr)] + rnorm(n_tr, sd = 0.1),
    x_test  = X[n_tr + seq_len(n_te), , drop = FALSE]
  )

  # --- 7. mixed categorical / numeric -------------------------------------
  # The categorical path is only reachable when a column is *declared*
  # categorical or has fewer than four distinct values, so this fixture
  # carries both: a 3-level column the reference infers on its own, a
  # 6-level and a 12-level one that only a declaration reaches, and a
  # high-cardinality one that stays numeric however it is labelled. The
  # `_very_common_categories` filter needs levels seen at least ten times
  # and fewer than n/10 of them, so with 200 train rows the 12-level
  # column is deliberately on the wrong side of that cut.
  set.seed(99)
  n_tr <- 200L; n_te <- 80L; n <- n_tr + n_te
  g3  <- sample.int(3L,  n, replace = TRUE) - 1L
  g6  <- sample.int(6L,  n, replace = TRUE) - 1L
  g12 <- sample.int(12L, n, replace = TRUE) - 1L
  num <- rnorm(n)
  X <- cbind(g3, num, g6, g12, rnorm(n))
  X[cbind(sample.int(n, 15L), 2L)] <- NA_real_
  lin <- 0.9 * g3 - 0.5 * (g6 %% 2L) + 0.8 * num + 0.3 * X[, 5]
  fx$clf_categorical <- list(
    task    = "classification",
    note    = "3 categorical columns (3, 6 and 12 levels) + 2 numeric, 1 with NAs",
    x_train = X[seq_len(n_tr), , drop = FALSE],
    y_train = as.integer(lin[seq_len(n_tr)] > median(lin[seq_len(n_tr)])),
    x_test  = X[n_tr + seq_len(n_te), , drop = FALSE],
    # 1-based, as everything R-side is; the Python harness converts.
    categorical_features = c(1L, 3L, 4L)
  )
  fx$reg_categorical <- list(
    task    = "regression",
    note    = "same columns as clf_categorical, continuous target",
    x_train = X[seq_len(n_tr), , drop = FALSE],
    y_train = lin[seq_len(n_tr)] + rnorm(n_tr, sd = 0.1),
    x_test  = X[n_tr + seq_len(n_te), , drop = FALSE],
    categorical_features = c(1L, 3L, 4L)
  )

  fx
}


#' Write fixtures to `<outdir>/<name>.safetensors` plus a manifest
#' @keywords internal
write_parity_fixtures <- function(outdir) {
  stopifnot(requireNamespace("safetensors", quietly = TRUE),
            requireNamespace("jsonlite", quietly = TRUE))
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
  fx <- build_parity_fixtures()

  manifest <- list(version = FIXTURE_VERSION, fixtures = list())
  for (nm in names(fx)) {
    f <- fx[[nm]]
    tens <- list(
      x_train = torch::torch_tensor(f$x_train, dtype = torch::torch_float()),
      x_test  = torch::torch_tensor(f$x_test,  dtype = torch::torch_float()),
      y_train = if (f$task == "classification")
        torch::torch_tensor(as.integer(f$y_train), dtype = torch::torch_long())
      else
        torch::torch_tensor(as.numeric(f$y_train), dtype = torch::torch_float())
    )
    safetensors::safe_save_file(tens, file.path(outdir, paste0(nm, ".safetensors")))
    manifest$fixtures[[nm]] <- list(
      task = f$task, note = f$note,
      n_train = nrow(f$x_train), n_test = nrow(f$x_test),
      n_features = ncol(f$x_train),
      # 1-based. Absent means "nothing declared"; the model still infers.
      categorical_features = f$categorical_features %||% integer()
    )
    cat(sprintf("  %-20s %s  (%d train / %d test / %d feat)\n",
                nm, f$task, nrow(f$x_train), nrow(f$x_test), ncol(f$x_train)))
  }
  jsonlite::write_json(manifest, file.path(outdir, "manifest.json"),
                       auto_unbox = TRUE, pretty = TRUE)
  invisible(fx)
}


#' Read a fixture back as plain R objects
#' @keywords internal
read_parity_fixture <- function(dir, name) {
  stopifnot(requireNamespace("safetensors", quietly = TRUE))
  t <- safetensors::safe_load_file(file.path(dir, paste0(name, ".safetensors")),
                                   framework = "torch")
  # Which columns the fixture declares categorical lives in the manifest,
  # not the tensor file, so both harnesses read it from one place.
  cats <- integer()
  mf <- file.path(dir, "manifest.json")
  if (file.exists(mf) && requireNamespace("jsonlite", quietly = TRUE)) {
    m <- jsonlite::fromJSON(mf, simplifyVector = FALSE)
    cats <- as.integer(unlist(m$fixtures[[name]]$categorical_features %||% list()))
  }
  list(
    x_train = as.matrix(t$x_train),
    x_test  = as.matrix(t$x_test),
    y_train = as.numeric(t$y_train),
    categorical_features = cats
  )
}


if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  outdir <- if (length(args)) args[[1]] else
    file.path(dirname(sys.frame(1)$ofile %||% "inst/parity"), "fixtures")
  if (is.na(outdir) || !nzchar(outdir)) outdir <- "inst/parity/fixtures"
  suppressMessages(library(torch))
  cat("Writing parity fixtures to", outdir, "\n")
  write_parity_fixtures(outdir)
}
