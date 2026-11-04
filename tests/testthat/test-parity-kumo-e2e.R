# End-to-end parity against `sdm.models.KumoTabular` (small checkpoints).
#
# The whole predictor: the recipe, the 8-member ensemble from a seeded
# generator, the network, the class realignment and the trimmed mean over
# members. Plus the edge cases: an all-constant table (the network sees no
# feature columns at all), a column missing in every training row, an
# empty query, and a class that never occurs in the training rows. And 15
# classes, past the network's 10, where each member runs error-correcting
# output codes; its codebooks are the ones the reference's forward pass
# actually drew.
#
# Regenerate with `inst/parity/kumo_e2e_reference.py`. Skips unless
# TABFOUND_KUMO_CLF_DIR / TABFOUND_KUMO_REG_DIR point at the converted
# small artifacts.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

ref_dir <- tabfound_file("parity", "reference", "kumo_e2e")
skip_if(!nzchar(ref_dir), "no stored Kumo end-to-end reference")
clf_dir <- Sys.getenv("TABFOUND_KUMO_CLF_DIR", unset = "")
reg_dir <- Sys.getenv("TABFOUND_KUMO_REG_DIR", unset = "")
skip_if(!dir.exists(clf_dir) || !dir.exists(reg_dir),
        "converted Kumo small artifacts not configured")

ref  <- read_reference_tensors(file.path(ref_dir, "e2e.safetensors"))
meta <- jsonlite::fromJSON(file.path(ref_dir, "reference.json"),
                           simplifyVector = FALSE)
as_mat <- function(t) {
  m <- as.array(t$to(dtype = torch::torch_float64()))
  if (!is.matrix(m)) m <- matrix(m, ncol = 1L)
  m
}

clf <- suppressMessages(tabular_classifier(clf_dir, device = "cpu"))
reg <- suppressMessages(tabular_regressor(reg_dir, device = "cpu"))

# Float32 through eight members, against a different libtorch build. The
# members' inputs agree to a float32 ulp (`test-prep-kumo.R`), the network
# to ~1e-6 (`test-parity-kumo.R`); what reaches the output is the network
# gap, averaged.
TOL_PROBA <- 1e-5
TOL_REG   <- 1e-4   # relative to the target's scale

for (nm in names(meta$cases)) {
  local({
    name <- nm
    cs <- meta$cases[[name]]
    test_that(paste0("tabfound matches sdm's KumoTabular end to end on ", name), {
      p <- paste0(name, ".")
      X_train <- as_mat(ref[[paste0(p, "x_train")]])
      X_test <- as_mat(ref[[paste0(p, "x_test")]])
      if (!length(X_test)) X_test <- X_train[0, , drop = FALSE]
      y <- as.numeric(as_mat(ref[[paste0(p, "y_train")]]))
      want <- as_mat(ref[[paste0(p, "out")]])
      cats <- unlist(cs$categorical)
      if (identical(cs$task, "classification")) {
        lv <- 0:max(2L, as.integer(max(y)))
        m <- tabfound:::.rebuild_predictor(clf, categorical_features = cats)
        m <- fit(m, X_train, factor(y, levels = lv))
        got <- predict(m, X_test, type = "prob")
        expect_identical(dim(got), c(nrow(X_test), length(lv)))
        books <- grep(paste0("^", name, "\\.codebook"), names(ref), value = TRUE)
        mine <- m$state$recipe$target$codebooks
        expect_identical(length(mine), length(books))
        for (i in seq_along(books)) {
          want_cb <- as_mat(ref[[paste0(p, "codebook", i - 1L)]])
          storage.mode(want_cb) <- "integer"
          expect_identical(mine[[i]], want_cb)
        }
        # The reference has a column only for classes seen in training;
        # tabfound returns every level, the unseen ones at zero.
        seen <- as.character(unlist(cs$classes))
        expect_true(all(got[, setdiff(colnames(got), seen)] == 0))
        got <- got[, seen, drop = FALSE]
        if (length(got)) expect_lt(max(abs(got - want)), TOL_PROBA)
      } else {
        m <- tabfound:::.rebuild_predictor(reg, categorical_features = cats)
        m <- fit(m, X_train, y)
        got <- predict(m, X_test, type = "grid")
        expect_identical(dim(got), dim(want))
        expect_lt(max(abs(got - want)) / sd(y), TOL_REG)
        # The point estimate is the mean of the grid.
        expect_equal(predict(m, X_test, type = "mean"), rowMeans(got),
                     tolerance = 1e-12)
      }
    })
  })
}


# The cache and the three chunking knobs change batch shapes, never which
# rows see which; graded against the reference's own one-shot forward,
# they must clear the same bounds as the plain path.
VARIANTS <- list(
  cached  = list(kv_cache = TRUE),
  chunked = list(row_chunk_size = 64L, save_peak_memory_factor = 3L,
                 col_chunk_size = 2L)
)
for (nm in c("clf_binary_missing", "clf_categorical", "reg_skewed",
             "reg_categorical", "clf_many_classes")) {
  for (vn in names(VARIANTS)) {
    local({
      name <- nm; variant <- VARIANTS[[vn]]; label <- vn
      cs <- meta$cases[[name]]
      test_that(sprintf("the %s Kumo path matches sdm on %s", label, name), {
        p <- paste0(name, ".")
        X_train <- as_mat(ref[[paste0(p, "x_train")]])
        X_test <- as_mat(ref[[paste0(p, "x_test")]])
        y <- as.numeric(as_mat(ref[[paste0(p, "y_train")]]))
        want <- as_mat(ref[[paste0(p, "out")]])
        args <- c(list(categorical_features = unlist(cs$categorical),
                       device = "cpu"), variant)
        if (identical(cs$task, "classification")) {
          m <- suppressMessages(do.call(tabular_classifier, c(list(clf_dir), args)))
          m <- fit(m, X_train, factor(y, levels = 0:max(2L, as.integer(max(y)))))
          got <- predict(m, X_test, type = "prob")[, as.character(unlist(cs$classes)),
                                                    drop = FALSE]
          expect_lt(max(abs(got - want)), TOL_PROBA)
        } else {
          m <- suppressMessages(do.call(tabular_regressor, c(list(reg_dir), args)))
          m <- fit(m, X_train, y)
          got <- predict(m, X_test, type = "grid")
          expect_lt(max(abs(got - want)) / sd(y), TOL_REG)
        }
      })
    })
  }
}
