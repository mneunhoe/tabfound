# The Kumo Tabular recipe and ensemble, against `sdm`'s own
# `RecipeExecution`, with no checkpoint:
#
#   .venvs/sdm/bin/python inst/parity/kumo_ensemble_reference.py \
#       --fixture-dir inst/parity/fixtures \
#       --out inst/parity/ensemble/kumo_ensemble.safetensors
#
# The reference was run with 8 members and a generator seeded at 42. The
# R recipe draws from libtorch's generator in the same order, so on the
# platform the dump was cut on the members agree to the last bit or to
# float64 summation noise; elsewhere (an Intel build using MKL's Bernoulli
# stream) the signs could differ, and the test says so.

skip_if_not_installed("torch")
skip_if_not_installed("safetensors")

ref_path <- tabfound_file("parity", "ensemble", "kumo_ensemble.safetensors.gz")
skip_if(!nzchar(ref_path), "Kumo ensemble reference not found")
ref <- read_reference_tensors(ref_path)
meta <- jsonlite::fromJSON(sub("\\.safetensors\\.gz$", ".json", ref_path),
                           simplifyVector = FALSE)
E <- meta$members

case_input <- function(name) {
  cs <- meta$cases[[name]]
  p <- paste0(name, ".")
  list(cs = cs,
       X_train = as.matrix(ref[[paste0(p, "x_train")]]),
       X_test = as.matrix(ref[[paste0(p, "x_test")]]),
       y = as.numeric(ref[[paste0(p, "y_train")]]),
       clf = identical(cs$task, "classification"),
       cat = seq_len(ncol(as.matrix(ref[[paste0(p, "x_train")]]))) %in%
         unlist(cs$categorical))
}

fit_case <- function(name) {
  i <- case_input(name)
  rec <- kumo_recipe_fit(i$X_train, i$y, i$clf, categorical = i$cat,
                         n_members = E, seed = meta$seed)
  c(i, list(rec = rec))
}

for (nm in names(meta$cases)) {
  local({
    name <- nm
    test_that(paste0("Kumo recipe members match sdm on ", name), {
      f <- fit_case(name)
      cs <- f$cs
      n_ctx <- cs$n_dump_ctx
      rows <- c(seq_len(n_ctx), cs$n_train + seq_len(cs$n_test))
      X_all <- rbind(f$X_train, f$X_test)
      for (m in seq_len(E)) {
        p <- sprintf("%s.m%d.", name, m - 1L)
        mem <- kumo_recipe_member(f$rec, X_all, m)
        expect_identical(mem$columns, unlist(cs$members[[m]]$columns),
                         label = sprintf("%s member %d columns", name, m))
        expect_identical(mem$categorical_mask,
                         as.logical(as.integer(ref[[paste0(p, "mask")]])),
                         label = sprintf("%s member %d mask", name, m))
        got <- as.matrix(mem$x)[rows, , drop = FALSE]
        want <- as.matrix(ref[[paste0(p, "x")]])
        expect_identical(dim(got), dim(want))
        d <- abs(got - want); d[is.nan(got) & is.nan(want)] <- 0
        expect_true(identical(is.nan(got), is.nan(want)),
                    label = sprintf("%s member %d NaN pattern", name, m))
        # float32 outputs of a float64 pipeline: equal, or one float32
        # ulp where float64 summation order differs.
        expect_lte(max(d, 0), 2^-23 * max(1, max(abs(want), na.rm = TRUE)),
                   label = sprintf("%s member %d x", name, m))

        # Class codes exactly. A regression target is standardized in
        # float32, and R's libtorch sums float32 in a different order from
        # the reference's (167.122925 against 167.122910 over one fixture's
        # 120 targets), so its statistics -- and the values -- can sit one
        # float32 ulp apart.
        yt <- as.numeric(kumo_recipe_target(f$rec, f$y, m))
        yw <- as.numeric(ref[[paste0(p, "y")]])
        if (f$clf) {
          expect_identical(yt, yw, label = sprintf("%s member %d y", name, m))
        } else {
          expect_lte(max(abs(yt - yw)), 2^-22 * max(1, abs(yw)),
                     label = sprintf("%s member %d y", name, m))
        }
        if (f$clf) {
          K <- length(f$rec$target$classes)
          rot <- (seq_len(K) - 1L + f$rec$target$shifts[m]) %% K + 1L
          expect_identical(as.integer(f$rec$target$classes[rot]),
                           as.integer(unlist(cs$members[[m]]$classes)),
                           label = sprintf("%s member %d class order", name, m))
          # Past 10 classes, the member's ECOC codebook: integer symbols
          # from the same stream, so exact or wrong.
          if (!is.null(ref[[paste0(p, "codebook")]])) {
            want_cb <- as.matrix(ref[[paste0(p, "codebook")]])
            storage.mode(want_cb) <- "integer"
            expect_identical(f$rec$target$codebooks[[m]], want_cb,
                             label = sprintf("%s member %d codebook", name, m))
          } else {
            expect_null(f$rec$target$codebooks)
          }
        }
      }
    })

    test_that(paste0("Kumo output reduction matches sdm on ", name), {
      f <- fit_case(name)
      outs <- lapply(seq_len(E), function(m) {
        o <- ref[[sprintf("%s.m%d.out", name, m - 1L)]]
        if (f$clf) kumo_recipe_unshift(f$rec, o, m)
        else kumo_recipe_inverse(f$rec, o, m)
      })
      got <- as.matrix(kumo_recipe_reduce(outs, f$clf))
      want <- as.matrix(ref[[paste0(name, ".final")]])
      expect_equal(got, want, tolerance = 1e-6, ignore_attr = TRUE)
    })
  })
}


# --- Properties that hold whatever the reference says -----------------------

test_that("the trimmed mean drops floor(0.2 E) members from each tail", {
  for (E_ in c(1L, 4L, 5L, 8L, 16L)) {
    vals <- lapply(seq_len(E_), function(e) torch::torch_full(c(1L, 1L), e^2))
    got <- as.numeric(kumo_recipe_reduce(vals, FALSE))
    cut <- floor(0.2 * E_)
    v <- sort((seq_len(E_))^2)
    want <- mean(v[(cut + 1L):(E_ - cut)])
    expect_equal(got, want, tolerance = 1e-6, label = paste("E =", E_))
  }
})

test_that("class shifts come from randperm pools of size K", {
  rec <- kumo_recipe_fit(matrix(rnorm(30), 10), rep(0:2, length.out = 10), TRUE,
                         n_members = 9L, seed = 1L)
  s <- rec$target$shifts
  for (b in 0:2) expect_setequal(s[b * 3L + 1:3], 0:2)
  # Members' views decode to the same labels.
  for (m in 1:9) {
    y <- as.integer(kumo_recipe_target(rec, rep(0:2, length.out = 10), m))
    expect_identical((y + s[m]) %% 3L, rep(0:2, length.out = 10))
  }
})

test_that("an unseen or missing category codes to -1 and is not a NaN", {
  X <- cbind(c(1, 2, 2, 1, 3, 1), c(10, 20, 10, 20, 10, 20))
  rec <- kumo_recipe_fit(X, c(0, 1, 0, 1, 0, 1), TRUE, categorical = c(TRUE, FALSE),
                         n_members = 1L)
  mem <- kumo_recipe_member(rec, rbind(X, c(7, 10), c(NA, 20)), 1L)
  expect_false(anyNA(as.matrix(mem$x)))
})

test_that("the recipe state is plain R data", {
  rec <- kumo_recipe_fit(matrix(rnorm(40), 20), rnorm(20), FALSE, n_members = 4L)
  bad <- rapply(rec, function(x) inherits(x, "torch_tensor"), how = "unlist")
  expect_false(any(bad))
})

test_that("an ECOC codebook gives every class its own symbol somewhere", {
  for (K in c(11L, 15L, 40L, 250L)) {
    g <- .kumo_generator(K)
    cb <- .kumo_ecoc_codebook(K, g)
    T_ <- max(ceiling(K / 9), 4 * ceiling(log10(K)))
    expect_identical(dim(cb), c(as.integer(T_), K))
    for (t in seq_len(T_)) {
      own <- cb[t, ] != 9L
      # Nine classes per task, holding the symbols 0..8 once each.
      expect_identical(sum(own), 9L)
      expect_setequal(cb[t, own], 0:8)
    }
    expect_true(all(colSums(cb != 9L) >= 1L), label = paste("K =", K))
  }
  # Drawn after the rest of the recipe, so ten classes or fewer draw none
  # and leave every other draw where it was.
  X <- matrix(rnorm(60), 30)
  a <- kumo_recipe_fit(X, rep(0:9, 3), TRUE, n_members = 2L)
  expect_null(a$target$codebooks)
  b <- kumo_recipe_fit(X, rep(0:14, 2), TRUE, n_members = 2L)
  expect_length(b$target$codebooks, 2L)
  expect_false(identical(b$target$codebooks[[1]], b$target$codebooks[[2]]))
})
