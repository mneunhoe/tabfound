# End-to-end parity against `TabICLClassifier` / `TabICLRegressor`.
#
# The whole estimator, not the bare network: imputation, the unique-value
# filter, the 8-member ensemble, class-shuffle inversion and the quantile
# head. Includes the cases tabicl 2.2.0 changed -- an entirely missing
# training column, an all-constant table, an all-missing table and a
# test batch with an entirely missing column.
#
# Regenerate with `inst/parity/tabicl_e2e_reference.py`. Skips unless
# TABFOUND_TABICL_CLF_DIR / TABFOUND_TABICL_REG_DIR point at converted
# checkpoints.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

ref_dir <- tabfound_file("parity", "reference", "tabicl_e2e")
skip_if(!nzchar(ref_dir), "no stored TabICL end-to-end reference")

clf_dir <- Sys.getenv("TABFOUND_TABICL_CLF_DIR", unset = "")
reg_dir <- Sys.getenv("TABFOUND_TABICL_REG_DIR", unset = "")
skip_if(!dir.exists(clf_dir) || !dir.exists(reg_dir),
        "converted TabICL artifacts not configured")

ref  <- read_reference_tensors(file.path(ref_dir, "e2e.safetensors"))
meta <- jsonlite::fromJSON(file.path(ref_dir, "reference.json"))
as_mat <- function(t) {
  m <- as.array(t$to(dtype = torch::torch_float64()))
  if (!is.matrix(m)) m <- matrix(m, ncol = 1L)
  m
}

y_cls <- factor(as.vector(as_mat(ref$y_cls)), levels = 0:2)
y_reg <- as.vector(as_mat(ref$y_reg))
clf <- tabular_classifier(clf_dir, device = "cpu")
reg <- tabular_regressor(reg_dir, device = "cpu")

# float32 forward passes through eight independently preprocessed members.
TOL_PROBA <- 1e-5
TOL_REG   <- 1e-4   # relative to the target's scale

for (case in meta$cases) {
  local({
    cs <- case
    test_that(paste0("TabICL end to end matches tabicl ", meta$tabicl_version,
                     " on '", cs, "'"), {
      X  <- as_mat(ref[[paste0(cs, "_X")]])
      Xt <- as_mat(ref[[paste0(cs, "_Xt")]])

      proba <- predict(fit(clf, X, y_cls), Xt, type = "prob")
      expect_equal(unname(as.matrix(proba)),
                   as_mat(ref[[paste0(cs, "_proba")]]),
                   tolerance = TOL_PROBA, ignore_attr = TRUE)

      r <- fit(reg, X, y_reg)
      scale <- stats::sd(y_reg)
      expect_lt(max(abs(predict(r, Xt) - as.vector(as_mat(ref[[paste0(cs, "_mean")]])))),
                TOL_REG * scale)
      expect_lt(max(abs(predict(r, Xt, type = "median") -
                          as.vector(as_mat(ref[[paste0(cs, "_median")]])))),
                TOL_REG * scale)
      q <- predict(r, Xt, type = "quantiles", quantiles = meta$quantiles)
      expect_lt(max(abs(unname(as.matrix(q)) - as_mat(ref[[paste0(cs, "_quantiles")]]))),
                TOL_REG * scale)
    })
  })
}
