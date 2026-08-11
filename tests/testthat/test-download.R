# Model download, conversion and the consent gate.
#
# Nothing here touches the network. The catalogue, the store layout, the
# resolution rules and the consent policies are all testable offline, and
# the one test that exercises the real converter runs against a
# checkpoint already on the machine (skipped when there is none).

test_that("the catalogue resolves ids and family + task", {
  m <- list_models()
  expect_s3_class(m, "data.frame")
  expect_true(all(c("id", "backend", "task", "downloaded", "repo") %in% names(m)))
  expect_true(all(m$task %in% c("classification", "regression")))
  expect_setequal(unique(m$backend),
                  c("tabpfn", "tabpfn26", "tabpfn3", "tabicl", "tabfm", "mitra"))

  # An exact id resolves without a task.
  expect_identical(tabfound:::.catalog_id("tabpfn-v2.5-classifier"),
                   "tabpfn-v2.5-classifier")
  # A family name needs one, and picks the matching head.
  expect_identical(tabfound:::.catalog_id("tabpfn", "classification"),
                   "tabpfn-v2.5-classifier")
  expect_identical(tabfound:::.catalog_id("tabpfn", "regression"),
                   "tabpfn-v2.5-regressor")
  expect_null(tabfound:::.catalog_id("tabpfn"))

  # Adding v2.6 must not repoint anyone's existing `"tabpfn"`: the bare
  # family name stays on v2.5, and a version is asked for by name.
  expect_identical(tabfound:::.catalog_id("tabpfn-v2.5", "classification"),
                   "tabpfn-v2.5-classifier")
  expect_identical(tabfound:::.catalog_id("tabpfn-v2.6", "classification"),
                   "tabpfn-v2.6-classifier")
  expect_identical(tabfound:::.catalog_id("tabpfn-v2.6", "regression"),
                   "tabpfn-v2.6-regressor")
  expect_identical(tabfound:::.catalog_id("tabpfn-v3", "classification"),
                   "tabpfn-v3-classifier")
  expect_identical(tabfound:::.catalog_id("tabpfn-v3", "regression"),
                   "tabpfn-v3-regressor")
  expect_identical(m$backend[m$id == "tabpfn-v2.6-classifier"], "tabpfn26")
  expect_identical(m$backend[m$id == "tabpfn-v3-regressor"], "tabpfn3")

  # The six specialised v3 checkpoints are addressable by exact id only.
  # They carry no family, so `"tabpfn-v3"` must keep meaning the default
  # -- otherwise `.catalog_id()` would see several claimants for one
  # family + task and silently return NULL.
  spec <- c("tabpfn-v3-classifier-binary", "tabpfn-v3-classifier-multiclass",
            "tabpfn-v3-classifier-ood", "tabpfn-v3-regressor-mediumdata",
            "tabpfn-v3-regressor-ood", "tabpfn-v3-regressor-timeseries")
  expect_true(all(spec %in% m$id))
  expect_true(all(m$backend[match(spec, m$id)] == "tabpfn3"))
  for (id in spec) expect_identical(tabfound:::.catalog_id(id), id)
  # Anything not ours passes through as NULL, which is what lets a plain
  # path or Hub repo id reach the old code path untouched.
  expect_null(tabfound:::.catalog_id("/some/local/dir", "classification"))
  expect_null(tabfound:::.catalog_id("someone/some-repo", "classification"))
})


test_that("the store location is redirectable", {
  d <- withr::local_tempdir()
  withr::local_options(tabfound.home = d)
  expect_identical(tabfound_home(), path.expand(d))
  expect_match(tabfound:::.model_dir("mitra-classifier"), "models/mitra-classifier$")

  # Never inside the installed package: writing there breaks read-only
  # and shared libraries, and CRAN forbids it.
  withr::local_options(tabfound.home = NULL)
  withr::local_envvar(TABFOUND_HOME = "")
  expect_false(grepl(system.file(package = "tabfound"), tabfound_home(), fixed = TRUE))
})


test_that("a model is only 'downloaded' when both artifacts are present", {
  d <- withr::local_tempdir()
  withr::local_options(tabfound.home = d)
  id <- "tabpfn-v2.5-classifier"
  expect_false(tabfound:::.model_is_downloaded(id))

  dir.create(tabfound:::.model_dir(id), recursive = TRUE)
  writeLines("{}", file.path(tabfound:::.model_dir(id), "config.json"))
  expect_false(tabfound:::.model_is_downloaded(id))   # weights still missing
  writeLines("x", file.path(tabfound:::.model_dir(id), "model.safetensors"))
  expect_true(tabfound:::.model_is_downloaded(id))
  expect_true(list_models()$downloaded[list_models()$id == id])

  # Subfolder-shaped entries (TabFM) must be looked for in the subfolder.
  id2 <- "tabfm-1.0.0-classifier"
  dir.create(file.path(tabfound:::.model_dir(id2), "classification"), recursive = TRUE)
  writeLines("{}", file.path(tabfound:::.model_dir(id2), "classification", "config.json"))
  expect_false(tabfound:::.model_is_downloaded(id2))
  writeLines("x", file.path(tabfound:::.model_dir(id2), "classification",
                            "model.safetensors"))
  expect_true(tabfound:::.model_is_downloaded(id2))
})


test_that("nothing downloads without consent", {
  d <- withr::local_tempdir()
  withr::local_options(tabfound.home = d)

  # Non-interactive is the case that matters for scripts and CI: refuse,
  # and say exactly what to run. A prompt nobody can answer would either
  # hang the job or get silently defaulted to yes.
  withr::local_options(tabfound.download = "ask")
  expect_error(
    tabfound:::resolve_model_source("tabpfn-v2.5-classifier", "classification"),
    "not interactive|not downloaded"
  )

  withr::local_options(tabfound.download = "never")
  expect_error(
    tabfound:::resolve_model_source("tabicl", "regression"),
    'download_model\\("tabicl-v2-regressor"\\)'
  )
  expect_false(dir.exists(file.path(d, "models")))
})


test_that("an already-downloaded model resolves with no prompt and no network", {
  d <- withr::local_tempdir()
  withr::local_options(tabfound.home = d, tabfound.download = "never")
  id <- "mitra-regressor"
  dir.create(tabfound:::.model_dir(id), recursive = TRUE)
  writeLines("{}", file.path(tabfound:::.model_dir(id), "config.json"))
  writeLines("x", file.path(tabfound:::.model_dir(id), "model.safetensors"))

  # "never" would abort if consent were consulted at all.
  expect_identical(tabfound:::resolve_model_source(id, "regression"),
                   tabfound:::.model_dir(id))
  expect_identical(tabfound:::resolve_model_source("mitra", "regression"),
                   tabfound:::.model_dir(id))
})


test_that("non-catalogue models are untouched by the download layer", {
  withr::local_options(tabfound.download = "never")
  # A registered alias still expands the way it always did.
  expect_identical(tabfound:::resolve_model_source("tabfm-nonesuch", "classification"),
                   "tabfm-nonesuch")
  expect_identical(tabfound:::resolve_model_source("/tmp/whatever", "classification"),
                   "/tmp/whatever")
})


test_that("download_model rejects things that are not in the catalogue", {
  expect_error(download_model("not-a-model"), "not a downloadable model")
  expect_error(download_model("tabpfn"), "not a downloadable model")  # needs task
  expect_error(remove_model("not-a-model"), "not a catalogue model id")
})


test_that("remove_model deletes only what is there", {
  d <- withr::local_tempdir()
  withr::local_options(tabfound.home = d)
  id <- "mitra-classifier"
  expect_silent(remove_model(id, quiet = TRUE))       # absent: a no-op
  dir.create(tabfound:::.model_dir(id), recursive = TRUE)
  writeLines("x", file.path(tabfound:::.model_dir(id), "model.safetensors"))
  remove_model(id, quiet = TRUE)
  expect_false(dir.exists(tabfound:::.model_dir(id)))
})


test_that("a Python without torch is rejected before the subprocess runs", {
  # `.find_python()` checks that torch and safetensors import, not merely
  # that an interpreter exists -- otherwise the failure surfaces as a
  # traceback from a subprocess several frames later.
  withr::local_options(tabfound.python = NULL)
  withr::local_envvar(TABFOUND_PYTHON = "/nonexistent/python",
                      TABFOUND_REF_PYTHON = "/nonexistent/python")
  sys_py <- Sys.which("python3")
  has_torch <- nzchar(sys_py) &&
    system2(sys_py, c("-c", shQuote("import torch, safetensors")),
            stdout = FALSE, stderr = FALSE) == 0L
  skip_if(has_torch, "system python has torch; cannot test the failure path")
  expect_error(tabfound:::.find_python(), "torch.*safetensors|No Python interpreter")
})


test_that("the real converter produces loadable artifacts", {
  # Runs the shipped Python converter on a checkpoint that is already on
  # this machine -- no download. This is the part of the pipeline most
  # likely to rot: it depends on an external script, an interpreter, and
  # the checkpoint's internal layout.
  ckpt <- Sys.getenv("TABFOUND_TABPFN_CLF_CKPT", unset = "")
  skip_if(!nzchar(ckpt) || !file.exists(ckpt),
          "TABFOUND_TABPFN_CLF_CKPT not set to a local checkpoint")
  py <- tryCatch(tabfound:::.find_python(), error = function(e) NULL)
  skip_if(is.null(py), "no Python with torch + safetensors")

  dest <- withr::local_tempdir()
  entry <- tabfound:::.model_catalog()[["tabpfn-v2.5-classifier"]]
  tabfound:::.convert_ckpt(entry, ckpt, dest, py, quiet = TRUE)

  expect_true(file.exists(file.path(dest, "model.safetensors")))
  expect_true(file.exists(file.path(dest, "config.json")))
  cfg <- read_model_config(file.path(dest, "config.json"))
  expect_true(length(cfg$state_dict_keys) > 0)
  # The artifacts must satisfy the same resolver the loader uses.
  paths <- resolve_artifacts(dest)
  expect_true(file.exists(paths$weights))
  expect_identical(detect_backend(cfg)$name, "tabpfn")
})


test_that("the converter tells the two TabPFN architectures apart", {
  # One script converts both v2.5 and v2.6, dispatching on the ckpt's own
  # `architecture_name`. Emitting the wrong `arch` would route the
  # weights to the wrong module tree, which fails loudly at load time --
  # but only if this dispatch is right, so it is pinned here.
  ckpt <- Sys.getenv("TABFOUND_TABPFN26_CLF_CKPT", unset = "")
  skip_if(!nzchar(ckpt) || !file.exists(ckpt),
          "TABFOUND_TABPFN26_CLF_CKPT not set to a local checkpoint")
  py <- tryCatch(tabfound:::.find_python(), error = function(e) NULL)
  skip_if(is.null(py), "no Python with torch + safetensors")

  dest <- withr::local_tempdir()
  entry <- tabfound:::.model_catalog()[["tabpfn-v2.6-classifier"]]
  tabfound:::.convert_ckpt(entry, ckpt, dest, py, quiet = TRUE)

  cfg <- read_model_config(file.path(dest, "config.json"))
  expect_identical(cfg$arch, "tabpfn_v2_6")
  expect_identical(detect_backend(cfg)$name, "tabpfn26")
  expect_identical(tabpfn_task_of(cfg), "classification")
  expect_true(file.exists(resolve_artifacts(dest)$weights))

  # Converting it as the wrong head must fail rather than write a config
  # that describes a model the checkpoint is not.
  wrong <- entry; wrong$head <- "regressor"
  expect_error(
    tabfound:::.convert_ckpt(wrong, ckpt, withr::local_tempdir(), py, quiet = TRUE),
    "Conversion failed"
  )
})
