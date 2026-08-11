# The guard that `fit()` and `predict()` consult.
#
# Availability is mocked throughout, so these assert what the guard does
# with an answer rather than what this machine happens to have free.

skip_if_not_installed("jsonlite")

# A 48 GB machine with `free` GB actually available.
local_machine <- function(free_gb, env = parent.frame()) {
  testthat::local_mocked_bindings(
    .system_memory = function() {
      list(total = 48e9, available = free_gb * 1e9, source = "mock")
    },
    .env = env
  )
}

# The Muchlinski fold that killed the session the hand-off was written
# about, and a table small enough that nothing could object to it.
big_X   <- function() matrix(0, nrow = 6426, ncol = 90)
small_X <- function() matrix(0, nrow = 50, ncol = 4)
some_y  <- function(X) factor(rep(c("a", "b"), length.out = nrow(X)))

mitra_model <- function(...) {
  mem_stub_model("mitra", mem_config_mitra(), ...)
}

setup_guard <- function(env = parent.frame()) {
  tabfound:::reset_memory_guard()
  withr::defer(tabfound:::reset_memory_guard(), envir = env)
}


# ---------------------------------------------------------------------------
# What it does when the run will not fit
# ---------------------------------------------------------------------------

test_that("fit() warns before a run that cannot fit, and still proceeds", {
  setup_guard(); local_machine(36)
  m <- mitra_model()
  X <- big_X()
  expect_warning(out <- fit(m, X, some_y(X)), "needs about")
  # A warning, not a refusal: the default must not take the decision away.
  expect_true(is_fitted(out))
})

test_that("the warning names the estimate, the headroom and a way out", {
  setup_guard(); local_machine(36)
  m <- mitra_model()
  X <- big_X()
  w <- tryCatch(fit(m, X, some_y(X)), warning = function(w) w)
  txt <- paste(conditionMessage(w), collapse = " ")
  expect_match(txt, "GB")                    # the estimate
  expect_match(txt, "available")             # the headroom
  expect_match(txt, "n_context")             # a knob that would help
  expect_match(txt, "libtorch")              # why there is no error to catch
  expect_match(txt, "memory_guard")          # how to silence it
})

test_that("predict() is checked against the fitted context and the new rows", {
  setup_guard(); local_machine(36)
  m <- mitra_model(n_train = 6426)
  expect_warning(predict(m, matrix(0, nrow = 714, ncol = 90)), "needs about")
})

test_that("a comfortable run says nothing at all", {
  setup_guard(); local_machine(36)
  m <- mitra_model()
  X <- small_X()
  expect_no_warning(fit(m, X, some_y(X)))
})

test_that("the same run on a busier machine is the one that gets flagged", {
  setup_guard(); local_machine(40)
  m <- mitra_model()
  X <- matrix(0, nrow = 700, ncol = 32)
  expect_no_warning(fit(m, X, some_y(X)))

  setup_guard(); local_machine(9)
  expect_warning(fit(m, X, some_y(X)), "little headroom|needs about")
})


# ---------------------------------------------------------------------------
# The option
# ---------------------------------------------------------------------------

test_that("\"error\" refuses the run", {
  setup_guard(); local_machine(36)
  withr::local_options(tabfound.memory_guard = "error")
  m <- mitra_model()
  X <- big_X()
  expect_error(fit(m, X, some_y(X)), "needs about")
  expect_error(fit(m, X, some_y(X)), "memory_guard")
})

test_that("\"off\" is silent", {
  setup_guard(); local_machine(1)
  withr::local_options(tabfound.memory_guard = "off")
  m <- mitra_model()
  X <- big_X()
  expect_no_warning(out <- fit(m, X, some_y(X)))
  expect_true(is_fitted(out))
})

test_that("a nonsense option value falls back to warning rather than failing", {
  setup_guard(); local_machine(36)
  withr::local_options(tabfound.memory_guard = "yes please")
  m <- mitra_model()
  X <- big_X()
  expect_warning(fit(m, X, some_y(X)), "needs about")
})


# ---------------------------------------------------------------------------
# Not being a nuisance
# ---------------------------------------------------------------------------

test_that("a repeated situation is warned about once", {
  # `mi()` fits the same model once per variable per iteration. Two
  # hundred copies of the same paragraph would bury it.
  setup_guard(); local_machine(36)
  m <- mitra_model()
  X <- big_X()
  y <- some_y(X)
  expect_warning(fit(m, X, y), "needs about")
  expect_no_warning(fit(m, X, y))
  expect_no_warning(fit(m, X, y))

  # Different dimensions are a different situation, and get said.
  expect_warning(fit(m, X[, 1:60], y), "needs about")
})

test_that("resetting makes it speak up again", {
  setup_guard(); local_machine(36)
  m <- mitra_model()
  X <- big_X()
  expect_warning(fit(m, X, some_y(X)), "needs about")
  tabfound:::reset_memory_guard()
  expect_warning(fit(m, X, some_y(X)), "needs about")
})


# ---------------------------------------------------------------------------
# Never the reason a run fails
# ---------------------------------------------------------------------------

test_that("a backend it knows nothing about is passed over in silence", {
  # `stub` is not registered, so there is no way to estimate anything.
  # Silence is the only acceptable answer; an error here would break a
  # run the guard exists to protect.
  setup_guard(); local_machine(36)
  m <- stub_model("classification")
  X <- big_X()
  expect_no_warning(out <- fit(m, X, some_y(X)))
  expect_true(is_fitted(out))
  expect_no_warning(predict(out, X))
})

test_that("an unprobeable machine is passed over in silence", {
  setup_guard()
  testthat::local_mocked_bindings(
    .system_memory = function() {
      list(total = NA_real_, available = NA_real_, source = NA_character_)
    }
  )
  m <- mitra_model()
  X <- big_X()
  expect_no_warning(fit(m, X, some_y(X)))
})

test_that("an uncalibrated device is passed over in silence", {
  # GPU constants are explicitly out of scope; refusing to estimate must
  # not turn into refusing to run.
  setup_guard(); local_machine(36)
  m <- mitra_model()
  m$device <- "cuda"
  X <- big_X()
  expect_no_warning(fit(m, X, some_y(X)))
})

test_that("the formula interface is covered by the same guard", {
  # The design promise is that the check lives on the shared fit/predict
  # path rather than in each backend, so `tabfound()` gets it for free.
  # Worth asserting against a real model rather than a stub, because the
  # thing being checked is the wiring.
  id <- "tabicl-v2-classifier"
  skip_if(!tabfound:::.model_is_downloaded(id), "artifacts not downloaded")
  setup_guard(); local_machine(4)
  d <- data.frame(matrix(stats::rnorm(2000 * 20), nrow = 2000))
  d$y <- factor(rep(c("a", "b"), length.out = 2000))
  expect_warning(
    suppressMessages(tabfound(y ~ ., data = d, model = id, n_estimators = 1L)),
    "needs about"
  )
})

test_that("regressors are guarded too, not just classifiers", {
  setup_guard(); local_machine(36)
  m <- mem_stub_model("mitra", mem_config_mitra(), task = "regression",
                      n_train = 6426)
  expect_warning(predict(m, matrix(0, nrow = 714, ncol = 90), type = "mean"),
                 "needs about")
})

test_that("the guard costs little enough to sit in a chained-equations loop", {
  setup_guard(); local_machine(36)
  m <- mitra_model()
  X <- small_X()
  y <- some_y(X)
  elapsed <- system.time(for (i in 1:50) fit(m, X, y))[["elapsed"]]
  expect_lt(elapsed, 5)
})
