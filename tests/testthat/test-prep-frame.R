# Frame-level expansion: layout against the reference, and the wiring.
#
# The first half checks `fit_frame_expansion()` against what TabPFN v3.5's
# estimator produces from the same mixed frame -- `DateTransformer` then
# `TextTransformer` -- at fit and on unseen rows. The second half checks
# that `tabfound()` and the engine API put the expansion where it belongs:
# after hardhat, before encoding, with declared categoricals moved to where
# their columns now sit, and nothing at all for a frame with no text or
# dates.

skip_if_not_installed("safetensors")
skip_if_not_installed("jsonlite")

ref_path <- tabfound_file("parity", "textdate", "textdate.safetensors.gz")
skip_if(!nzchar(ref_path), "text/date reference not installed")
ref <- read_reference_tensors(ref_path)
scl <- jsonlite::fromJSON(sub("\\.safetensors\\.gz$", ".json", ref_path),
                          simplifyVector = FALSE)
as_mat <- function(t) as.matrix(as.array(t$cpu()))
unnull <- function(x, na) vapply(x, function(v) if (is.null(v)) na else v, na)

# The reference's frame, rebuilt from the raw inputs it recorded: epoch
# seconds for the timestamps (a naive pandas timestamp is a UTC instant in
# R), days for the dates, seconds for the durations.
reference_frame <- function() {
  epoch <- unnull(scl$frame_in_ts_epoch, NA_real_)
  ts <- as.POSIXct(epoch, origin = "1970-01-01", tz = "UTC")
  ny <- ts; attr(ny, "tzone") <- "America/New_York"
  data.frame(
    a = unnull(scl$frame_in_a, NA_real_),
    dur = as.difftime(unnull(scl$frame_in_dur_seconds, NA_real_), units = "secs"),
    low = unnull(scl$frame_in_low, NA_character_),
    ts = ts, ny = ny,
    d = as.Date(unnull(scl$frame_in_d_days, NA_real_), origin = "1970-01-01"),
    note = unnull(scl$frame_in_note, NA_character_),
    stringsAsFactors = FALSE
  )
}


test_that("the expanded frame has the reference's layout and names", {
  frame <- reference_frame()
  n_fit <- scl$frame_n_fit
  fitted <- fit_frame_expansion(frame[seq_len(n_fit), ], TRUE, TRUE)
  # Kept columns in order, then the three date blocks, then the text block.
  expect_identical(names(fitted$data), unlist(scl$frame_output_names))
  # Durations are converted in place, a low-cardinality string stays a
  # categorical, and only `note` is text.
  expect_identical(fitted$state$durations, "dur")
  expect_identical(names(fitted$state$text), "note")
  expect_true(is.factor(fitted$data$low))
})

test_that("its values match the reference at fit and on new rows", {
  frame <- reference_frame()
  n_fit <- scl$frame_n_fit
  fitted <- fit_frame_expansion(frame[seq_len(n_fit), ], TRUE, TRUE)
  new <- transform_frame_expansion(frame[-seq_len(n_fit), ], fitted$state)

  num <- unlist(scl$frame_numeric_names)
  text_cols <- grep("^note_", num, value = TRUE)
  other <- setdiff(num, text_cols)
  for (which in c("fit", "new")) {
    got <- unname(as.matrix(if (which == "fit") fitted$data[num] else new[num]))
    colnames(got) <- num
    want <- as_mat(ref[[paste0("frame_", which)]])
    colnames(want) <- num
    # Numbers, durations and calendar features: exact, missing included.
    expect_identical(is.na(got[, other]), is.na(want[, other]), info = which)
    d <- abs(got[, other] - want[, other]); d[is.na(d)] <- 0
    expect_identical(max(d), 0, info = which)
    # Text: tight against the float64 twin, and no further from the real
    # float32 output than the twin is.
    twin <- as_mat(ref[[paste0("frame_note_twin_", which)]])
    expect_lt(max(abs(got[, text_cols] - twin)), 1e-9, label = paste("twin", which))
    own_gap <- max(abs(twin - want[, text_cols]))
    expect_lte(max(abs(got[, text_cols] - want[, text_cols])), own_gap * 1.01 + 1e-9)
  }
})

test_that("text detection follows the reference's rule", {
  vals <- scl$detect_values
  detect <- as.data.frame(lapply(vals, unnull, na = NA_character_),
                          stringsAsFactors = FALSE)
  expected <- unlist(scl$detect_columns)[unlist(scl$detect_text_positions) + 1L]
  expect_identical(.text_columns(detect, 30L), expected)
  # `boundary` has exactly 30 distinct values counting its missing one: not
  # more than 30, so not text. Numbers stored as text are not text.
  expect_false("boundary" %in% .text_columns(detect, 30L))
  expect_false("numbers_as_text" %in% .text_columns(detect, 30L))
  # A declared categorical is never text.
  expect_false("text" %in% .text_columns(detect, 30L, declared = "text"))
  # A factor is R's `category`, and is never text either.
  detect$text <- factor(detect$text)
  expect_identical(.text_columns(detect, 30L), character(0))
})

test_that("pandas' numeric spellings are pandas', not R's", {
  expect_identical(.is_numeric_spelling(c("1", " 2 ", "3e2", "-inf", "Inf")),
                   rep(TRUE, 5))
  # R reads both of these as numbers; pandas reads neither.
  expect_identical(.is_numeric_spelling(c("NaN", "0x10", "1,000", "1_000", "abc")),
                   rep(FALSE, 5))
})

test_that("'auto' follows the checkpoint's recipe, and is off without one", {
  on <- list(inference_config = list(TRANSFORM_TEXT = TRUE, TRANSFORM_DATES = TRUE))
  expect_true(.resolve_transform_flag("auto", on, "TRANSFORM_TEXT", "transform_text"))
  expect_false(.resolve_transform_flag("auto", list(), "TRANSFORM_TEXT", "transform_text"))
  expect_false(.resolve_transform_flag(FALSE, on, "TRANSFORM_TEXT", "transform_text"))
  expect_error(.preprocess_options(transform_text = "yes"), "auto")
})


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

wiring_frame <- function(n = 60) {
  set.seed(2)
  words <- replicate(200, paste(sample(letters, sample(3:8, 1), TRUE), collapse = ""))
  data.frame(
    lvl = sample(1:4, n, TRUE),
    when = as.Date("2024-01-01") + sample(0:300, n, TRUE),
    note = vapply(seq_len(n), function(i) paste(sample(words, 4), collapse = " "), ""),
    x = rnorm(n),
    y = rnorm(n),
    stringsAsFactors = FALSE
  )
}

# The fake backend's config, with or without the v3.5 recipe, so "auto" can
# be exercised both ways without real weights.
with_recipe <- function(dir, on) {
  cfg <- list(arch = "faketest", state_dict_keys = c("weight", "bias"))
  if (on) cfg$inference_config <- list(TRANSFORM_TEXT = TRUE, TRANSFORM_DATES = TRUE)
  jsonlite::write_json(cfg, file.path(dir, "config.json"), auto_unbox = TRUE)
  dir
}

test_that("tabfound() expands when the recipe says so, and moves declarations", {
  dir <- with_recipe(local_fake_backend(), on = TRUE)
  df <- wiring_frame()
  fit <- tabfound(y ~ ., data = df, model = dir, backend = "faketest",
                  categorical_features = 1L)
  exp <- fit$expansion
  expect_s3_class(exp, "tabfound_expansion")
  expect_identical(names(exp$dates), "when")
  expect_identical(names(exp$text), "note")
  # The stub regressor fits one coefficient per model column, plus an
  # intercept: 2 kept + 9 date features (a date has no time of day: year,
  # epoch seconds, day of year, and month, day and weekday pairs) + 30
  # text components.
  expect_length(fit$inner$state$b, 1L + 2L + 9L + 30L)
  # `lvl` was declared as column 1 and is still column 1 -- but the check
  # that matters is that the declaration names the same column.
  expect_identical(exp$output_names[fit$inner$model_ref$args$categorical_features], "lvl")

  p <- predict(fit, df[1:5, ])
  expect_identical(nrow(p), 5L)
})

test_that("with the recipe off, tabfound() behaves as it always has", {
  dir <- with_recipe(local_fake_backend(), on = FALSE)
  df <- wiring_frame()
  fit <- tabfound(y ~ ., data = df, model = dir, backend = "faketest")
  # A date is one number and the text column a categorical, as before.
  expect_length(fit$inner$state$b, 1L + 4L)
  expect_length(fit$expansion$dates, 0L)
  expect_length(fit$expansion$text, 0L)
  expect_identical(fit$inner$model_ref$args$categorical_features,
                   which(fit$expansion$output_names == "note"))
  # Unseen strings in a categorical warn and become NA, and say how to
  # read them as text instead.
  new <- df[1:3, ]; new$note <- c("never seen before", "nor this", "or this")
  expect_warning(predict(fit, new), "transform_text")
})

test_that("a frame with no text or dates is not touched at all", {
  dir <- with_recipe(local_fake_backend(), on = TRUE)
  df <- data.frame(a = rnorm(40), b = factor(sample(c("u", "v"), 40, TRUE)),
                   y = rnorm(40))
  fit <- tabfound(y ~ ., data = df, model = dir, backend = "faketest")
  expect_null(fit$expansion)
  expect_length(fit$inner$state$b, 1L + 2L)
})

test_that("the engine API expands at fit and replays at predict", {
  dir <- with_recipe(local_fake_backend(), on = TRUE)
  df <- wiring_frame()
  X <- df[setdiff(names(df), "y")]
  # Expanded as text, `note` is no longer a categorical, so the engine API's
  # usual warning about undeclared ones has nothing to say.
  expect_no_warning(m <- fit(tabular_regressor(dir, backend = "faketest"), X, df$y))
  expect_identical(names(m$state$expansion$text), "note")
  expect_length(m$state$b, 1L + 2L + 9L + 30L)
  expect_length(predict(m, X[1:4, ]), 4L)
  # A matrix cannot carry text or dates, so once they were expanded it is
  # refused rather than guessed into the expanded layout.
  expect_error(predict(m, as.matrix(X[1:4, c("lvl", "x")])), "data frame")
  # An explicit FALSE wins over the recipe.
  # Off, it is one -- and the warning it has always had comes back.
  expect_warning(
    off <- fit(tabular_regressor(dir, backend = "faketest", transform_text = FALSE,
                                 transform_dates = FALSE), X, df$y),
    "ordinal codes"
  )
  expect_length(off$state$b, 1L + 4L)
})

test_that("a saved fit replays its expansion after loading", {
  dir <- with_recipe(local_fake_backend(), on = TRUE)
  df <- wiring_frame()
  fit <- tabfound(y ~ ., data = df, model = dir, backend = "faketest")
  f <- withr::local_tempfile(fileext = ".rds")
  tabfound_save(fit, f)
  back <- tabfound_load(f)
  expect_identical(predict(back, df[1:6, ]), predict(fit, df[1:6, ]))
})
