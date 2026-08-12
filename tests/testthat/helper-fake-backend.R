# A complete, weightless backend: registry entry plus artifacts on disk.
#
# `helper-stub-model.R` fakes the *object* `tabular_classifier()` returns,
# which is enough for anything that only calls `fit()` and `predict()`.
# The formula interface is not that: `tabfound()` resolves a model
# reference, reads a config, builds a network and loads weights into it,
# so testing it -- and testing that a `tabfound_fit` survives a save/load
# round trip -- needs a real trip through the registry.
#
# So this writes a two-parameter checkpoint to a temp directory and
# registers a backend that recognises it. Everything downstream of the
# load is the stub specs. Registration is undone when the calling test
# ends, because other tests iterate over every registered backend and
# require hooks (`describe`, `peak_terms`) that this one has no business
# having.

local_fake_backend <- function(env = parent.frame(),
                               handles_missing = FALSE) {
  skip_if_not_installed("torch")
  skip_if_not_installed("safetensors")
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("withr")

  register_backend(
    name    = "faketest",
    build   = function(config, task) torch::nn_linear(2L, 1L),
    detect  = function(config) identical(config$arch, "faketest"),
    task_of = function(config) NULL,
    classifier = function(ctx, categorical_features = NULL, ...) {
      stub_classifier_spec()
    },
    regressor = function(ctx, categorical_features = NULL, ...) {
      stub_regressor_spec()
    },
    description     = "weightless test backend",
    handles_missing = handles_missing
  )
  withr::defer(rm("faketest", envir = .tabfound_backends), envir = env)

  dir <- withr::local_tempdir(.local_envir = env)
  safetensors::safe_save_file(
    list(weight = torch::torch_zeros(c(1L, 2L)),
         bias   = torch::torch_zeros(1L)),
    file.path(dir, "model.safetensors")
  )
  jsonlite::write_json(
    list(arch = "faketest", state_dict_keys = c("weight", "bias")),
    file.path(dir, "config.json"),
    auto_unbox = TRUE
  )
  dir
}
