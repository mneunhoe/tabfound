# Causilo 1.5 preprocessing and ensemble, against `causilo`'s own
# `PreparedDataset` (weight-free).
#
#   .venvs/causilo/bin/python -I inst/parity/causilo_ensemble_reference.py \
#       --fixture-dir inst/parity/fixtures \
#       --out inst/parity/ensemble/causilo_ensemble.safetensors

skip_if_not_installed("torch")

ens_path <- tabfound_file("parity", "ensemble", "causilo_ensemble.safetensors.gz")

causilo_ens <- function() {
  skip_if(!nzchar(ens_path), "Causilo ensemble reference not found")
  skip_if_not_installed("safetensors")
  list(t = read_reference_tensors(ens_path),
       meta = jsonlite::fromJSON(sub("\\.safetensors\\.gz$", ".json", ens_path),
                                 simplifyVector = FALSE))
}

test_that("every member's input matches the reference's", {
  ref <- causilo_ens()
  worst <- 0
  for (nm in names(ref$meta$cases)) {
    cs <- ref$meta$cases[[nm]]
    t <- function(k) as.array(ref$t[[paste0(nm, ".", k)]])
    xtr <- t("x_train"); xte <- t("x_test"); ytr <- as.numeric(t("y_train"))
    clf <- identical(cs$task, "classification")
    categorical <- seq_len(ncol(xtr)) %in% unlist(cs$categorical)
    rec <- causilo_recipe_fit(xtr, ytr, clf, categorical, cs$n_estimators, cs$seed)
    expect_identical(rec$plan$count, as.integer(cs$count), info = nm)
    expect_identical(rec$plan$views, as.integer(unlist(cs$views)), info = nm)
    expect_identical(rec$plan$trim, as.integer(cs$trim), info = nm)
    expect_identical(rec$columns$order, as.integer(unlist(cs$column_order)), info = nm)
    expect_identical(rec$columns$groups, as.integer(unlist(cs$groups)), info = nm)
    inp <- causilo_member_inputs(rec, causilo_recipe_apply(rec, xte),
                                 seq_len(rec$plan$count))
    want_x <- t("members_x"); want_y <- t("members_y")
    got_x <- as.array(inp$x); got_y <- as.array(inp$y)
    expect_identical(dim(got_x), dim(want_x), info = nm)
    expect_identical(is.nan(got_x), is.nan(want_x), info = nm)
    ok <- !is.nan(want_x)
    # float32 views of float64 statistics: equal up to a float32 ulp or two.
    d <- max(c(0, abs(got_x[ok] - want_x[ok]) / pmax(1, abs(want_x[ok]))))
    worst <- max(worst, d)
    expect_lte(d, 1e-6, label = paste(nm, "x"))
    expect_equal(got_y, matrix(want_y, nrow = nrow(got_y)), tolerance = 1e-6,
                 info = nm)
  }
  expect_lt(worst, 1e-6)
})

test_that("non-finite features are refused and odd tables survive", {
  X <- cbind(a = c(1, 2, Inf), b = 1:3)
  expect_error(causilo_recipe_fit(X, 1:3, FALSE, c(FALSE, FALSE), 1L, 42L), "infinite")
  # Nothing varies: no usable column.
  expect_error(causilo_recipe_fit(cbind(rep(1, 4)), 1:4, FALSE, FALSE, 1L, 42L),
               "varies")
  # A single training row keeps every column, ordered by the bytes of the
  # value (little-endian: 2.0 is 00..00 40, 1.0 is 00..f0 3f).
  rec <- causilo_recipe_fit(cbind(1, 2), 3, FALSE, c(FALSE, FALSE), 8L, 42L)
  expect_identical(rec$columns$order, 2:1)
})

test_that("the plan balances signs and keeps member 0 unchanged", {
  p <- causilo_ensemble_plan(5L, 3L, 8L, 42L)
  expect_true(all(p$signs[1, ] == 1))
  expect_true(all(colSums(p$signs < 0) == 4L))
  expect_identical(p$shifts[1], 0L)
  expect_identical(p$views, c(0L, 0L, 0L, 0L, 1L, 1L, 2L, 2L))
  r <- causilo_ensemble_plan(3L, NULL, 5L, 42L)
  expect_identical(r$target_signs, c(1, -1, 1, -1, 1))
  expect_identical(r$trim, 1L)
  expect_identical(r$views, c(0L, 0L, 0L, 1L, 1L))
  # Few features cap the count.
  expect_identical(causilo_ensemble_plan(1L, 2L, 8L, 42L)$count, 4L)
})

test_that("fsum is exactly rounded", {
  expect_identical(tabfound:::.causilo_fsum(c(1, 1e100, 1, -1e100)), 2)
  expect_identical(tabfound:::.causilo_fsum(rep(0.1, 10)), 1)
  expect_identical(tabfound:::.causilo_fsum(numeric()), 0)
})
