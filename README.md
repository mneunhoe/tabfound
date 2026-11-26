# tabfound

Tabular foundation models for R.

A tabular foundation model is a neural network that has been pretrained on
millions of synthetic datasets. You do not train it on your data. You
*condition* it on your data: the training rows go in alongside the rows you
want predictions for, and a single forward pass returns a full predictive
distribution. On small and medium tables (up to a few thousand rows), these
models are often competitive with a tuned gradient-boosted tree, with no
tuning at all.

`tabfound` runs several published models (TabPFN, TabICL, TabFM, Mitra,
Kumo Tabular, Causilo)
behind one interface: a formula, a data frame, and `predict()`. Everything
runs in R on [`torch`](https://torch.mlverse.org/), and Python is never
needed to fit or predict. Beyond prediction, the package uses the models'
predictive distributions for **multiple imputation** (with `mice` and
`Amelia` interop) and **synthetic data** (with `synthpop` interop).

```r
library(tabfound)

fit <- tabfound(Species ~ ., data = iris[train, ], model = "tabpfn-v2.5")
predict(fit, iris[test, ])                  # tibble: .pred_class
predict(fit, iris[test, ], type = "prob")   # .pred_setosa, .pred_versicolor, ...

fit <- tabfound(mpg ~ ., data = mtcars[train, ], model = "tabpfn-v2.5")
predict(fit, mtcars[test, ])                                   # mean
predict(fit, mtcars[test, ], type = "quantiles",
        quantiles = c(0.05, 0.5, 0.95))                        # intervals
```

`mode` is inferred from the outcome: a factor gives a classifier and a
number gives a regressor. The first time you use a model, you are asked
whether to download it.

## Installation

```r
# install.packages("remotes")
remotes::install_github("mneunhoe/tabfound")
# or: devtools::install_github("mneunhoe/tabfound")
```

`tabfound` needs `torch`. If you have never used it, run
`torch::install_torch()` once after installing. Imputation and synthesis
also use `mice`, `Amelia` and `synthpop` if they are installed.

## Getting a model

```r
list_models()                         # what exists, its size and licence, and what you have
download_model("tabpfn-v2.5-classifier")
```

Weights are downloaded once and stored in `tabfound_home()`, outside your R
library. After that they load by name, offline.

Wherever a model is asked for, you can pass:

- a **family name** such as `"tabpfn-v2.5"`, `"tabicl"` or `"mitra"`. The
  task (classification or regression) decides which checkpoint is used.
- a **catalogue id** such as `"tabpfn-v2.5-classifier"`, from
  `list_models()`.
- a **path** to a local directory that holds the weights.

**TabPFN and TabICL need Python once.** Their weights are published in a
PyTorch format that R cannot read, so `download_model()` converts them. For
that one step it needs a Python interpreter with `torch` and `safetensors`
installed. Point `options(tabfound.python = )` at one if it is not found.
Python is never used again afterwards. TabFM, Mitra, Kumo Tabular and
Causilo need no conversion and no Python at all.

### Licences

The package is MIT, but each model's weights have their own licence. Check
it before you publish or deploy:

- **Permissive:** TabICL (BSD-3-Clause), Mitra (Apache-2.0) and Kumo
  Tabular (OpenMDW-1.1).
- **Research and evaluation only:** TabPFN (v2.5, v2.6, v3 and v3.5), TabFM
  and Causilo. Commercial or production use needs a separate licence from
  the publisher.

Causilo's licence, the Nums AI Model License, is the most specific of
these. It permits non-commercial research, testing, evaluation and
benchmarking, and publishing the results. A separate licence from Nums AI
is needed for commercial or production use of the model *or of its
predictions*, and for offering the model as a hosted or API service,
whether paid or free.

These are summaries, not legal advice, and the full licence text in each
model's repository is what counts. `list_models()` shows each licence, and
the download prompt repeats it.

## Which model?

No single model is best. If you are not sure where to start:

| if you want | use |
|---|---|
| a small, fast default | `"tabpfn-v2.5"` |
| multiple imputation | `"tabpfn-v2.5"`, the best-tested option (see [below](#multiple-imputation)) |
| a permissive licence | `"kumo-tabular-small"`, `"mitra"` or `"tabicl"` |
| a larger table | `"tabpfn-v3"`, which chunks rows and columns (see [Large tables](#large-tables)) |
| free-text and date columns handled for you | `"tabpfn-v3.5"` |

### All models

| model | name | download | licence | missing values | regression output |
|---|---|---|---|---|---|
| TabPFN v2.5 | `"tabpfn-v2.5"` | ~40 MB per task | non-commercial | native | distribution |
| TabPFN v2.6 | `"tabpfn-v2.6"` | ~50 MB per task | non-commercial | native | distribution |
| TabPFN v3 | `"tabpfn-v3"` | ~200 MB per task | non-commercial | native | distribution |
| TabPFN v3.5 | `"tabpfn-v3.5"` | 876 MB, both tasks | non-commercial | native | distribution |
| TabICL v2 | `"tabicl"` | ~110 MB per task | BSD-3-Clause | mean-imputed | distribution |
| TabFM 1.0 | `"tabfm-1.0.0"` | ~6.6 GB per task | non-commercial | mean-imputed | point only |
| Mitra | `"mitra"` | ~300 MB per task | Apache-2.0 | mean-imputed | point only |
| Kumo Tabular | `"kumo-tabular-small"`, `-medium`, `-large` | 110 / 246 / 855 MB per task | OpenMDW-1.1 | native | distribution |
| Causilo 1.5 | `"causilo"` | 1.1 GB per task | Nums AI Model License (non-commercial) | native | distribution |

"Distribution" means `predict()` can return quantiles and intervals as well
as the mean. "Point only" means it returns the mean alone.

### Notes on each model

- **TabPFN v2.5.** Small and fast. Weights:
  [Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5).
- **TabPFN v2.6.** The successor to v2.5 at about the same size, with its
  preprocessing moved inside the network. Weights:
  [Prior-Labs/tabpfn_2_6](https://huggingface.co/Prior-Labs/tabpfn_2_6).
- **TabPFN v3.** Six specialised variants: binary, multiclass, OOD, medium
  data and time series. Weights:
  [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3).
- **TabPFN v3.5.** The newest TabPFN, with one checkpoint for both tasks.
  There is also a 334 MB `"tabpfn-v3.5-fast"` and a multiclass variant.
  Weights:
  [Prior-Labs/tabpfn_3_5](https://huggingface.co/Prior-Labs/tabpfn_3_5).
- **TabICL v2.** **Not recommended for imputation** (see
  [below](#multiple-imputation)). Weights:
  [jingang/TabICL](https://huggingface.co/jingang/TabICL).
- **TabFM 1.0.** Large (1.6 B parameters) and slow on CPU. Weights:
  [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch).
- **Mitra.** 72 M parameters. Weights:
  [autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier),
  [autogluon/mitra-regressor](https://huggingface.co/autogluon/mitra-regressor).
- **Kumo Tabular.** NVIDIA's model in three sizes. `"kumo-tabular"` means
  large, which is slow on CPU. It averages an ensemble of 8 members by
  default. Weights:
  [nvidia/Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular)
  (`v1.0.1`), fetched as a bit-identical safetensors copy from
  [mneunhoe/Kumo-Tabular-safetensors](https://huggingface.co/mneunhoe/Kumo-Tabular-safetensors).
- **Causilo 1.5.** 275 M parameters and slow on CPU. It averages an
  ensemble of 8 members by default; `light = TRUE` uses one, which is the
  published "Causilo 1.5 (Light)". Two limits to know about:
  - Above 10 classes it uses error-correcting codes with tabfound's own
    codebook, so its numbers there differ from the Python reference's.
  - It has not been validated for imputation.

  Weights: [nums-ai/causilo-1.5](https://huggingface.co/nums-ai/causilo-1.5).

All the models are inference-only. Each one reproduces its Python
reference's own preprocessing and ensembling, and gets the same predictions
as that reference to floating-point precision (see
[Validation](#validation)).

## Your data

`tabfound()` takes an ordinary data frame and handles column types for
you:

- **Factors, characters and logicals** are encoded, and the model is
  *told* which columns are categorical. A five-level factor is not treated
  as the numbers 1 to 5.
- **Missing values (`NA`)** are fine in predictors. Each model handles them
  the way its reference does: TabPFN, Kumo Tabular and Causilo model missingness
  explicitly, while the others mean-impute first. `Inf` is an error. Recode it to `NA` or to
  a finite value.
- **Dates and free text.** On TabPFN v3.5, dates are expanded into calendar
  features (year, day of year, cyclical month and weekday, …), and
  high-cardinality text columns become 30 text-similarity features. The
  other models can do the same with `transform_dates = TRUE` and
  `transform_text = TRUE`.
- **New data** is checked against the training data. A missing column or
  an unseen factor level is an error that names the column or level.

The x/y form works too: `tabfound(x = df[, -1], y = df$outcome, model = ...)`.

## Multiple imputation

Every model returns a predictive distribution, not just a point estimate,
so it can serve as the imputation model in chained equations. Each missing
cell gets a draw.

```r
mods <- tabfound_models(classifier = "tabpfn-v2.5-classifier",
                        regressor  = "tabpfn-v2.5-regressor")
imp  <- tabfound_impute(airquality, m = 5, models = mods, maxit = 3, seed = 1)

library(mice)
summary(pool(with(imp, lm(Ozone ~ Wind + Temp))))  # Rubin's rules
plot(as_mids(imp))                                 # convergence traces
amp <- as_amelia(imp)                              # Amelia tools
```

If you would rather keep your existing `mice` workflow, you can use the
package as a method inside it:

```r
options(tabfound.models = mods)
mice(airquality, method = "tabfound", m = 5)
```

Results of the simulation study in [`inst/simulation/`](inst/simulation/README.md)
(missingness depends on the outcome, so complete-case analysis is biased):

- **TabPFN removes the bias.** Bias goes from −0.111 (complete cases) to
  −0.008, with 0.965 coverage. That is on par with a correctly specified
  parametric imputer and with `mice`'s PMM, and TabPFN was told nothing
  about the model. Its intervals are somewhat narrower than PMM's.
- **TabICL does not.** It leaves bias of −0.061 with 0.86 coverage. Use
  TabPFN for imputation.
- **`proper = FALSE` is the default**, even though Rubin's theory calls for
  a proper imputer. Setting `proper = TRUE` bootstraps the context. That
  restores nominal coverage but multiplies the bias about eightfold on
  these models. Turn it on if coverage matters more to you than the point
  estimate.

Full walk-through: `vignette("multiple-imputation", "tabfound")`.

## Synthetic data

The same distributions can generate a whole synthetic dataset, one variable
at a time, the way `synthpop::syn()` does:

```r
sds <- tabfound_syn(iris, m = 5, models = mods, seed = 1)

library(synthpop)
compare(sds, iris)
utility.gen(sds, iris)
summary(lm.synds(Sepal.Length ~ Petal.Length, as_synds(sds)))

# or inside synthpop:
options(tabfound.models = mods)
syn(iris, method = "tabfound", m = 5)
```

These three choices should be reported, not left at their defaults without
thought:

- **`draw_mode`**:
  - `"predictive"` (the default) draws novel values.
  - `"pmm"` returns real observed values, like CART. It has the highest
    replication risk.
  - `"rank"` reproduces each marginal exactly.
- **`proper = TRUE`** is the default here, matching synthpop's definition.
  In the imputation simulation, the same correction cost a lot of bias. Its
  effect on synthesis has not been measured yet, so compare against
  `proper = FALSE`.
- **This is not disclosure control.** The real data is the model's context
  at generation time, and the method makes no differential-privacy claim.

Full walk-through, with a utility/disclosure comparison against CART:
`vignette("synthetic-data", "tabfound")`.

## Large tables

Prediction cost grows with the number of *training* rows, because every
forward pass sees all of them. Two things to know:

**Running out of memory kills the R session** without an error message.
`fit()` and `predict()` therefore estimate the memory they need beforehand
and warn you if it is too much. You can run the same check before
downloading anything:

```r
estimate_peak_memory("tabpfn-v3-classifier",
                     n_context = 20000, n_query = 5000, n_features = 50)
suggest_chunk_sizes("tabpfn-v3-classifier",
                    n_context = 50000, n_features = 50, available = 32e9)
```

**Useful arguments** for `tabular_classifier()` / `tabular_regressor()`,
or `...` in `tabfound()`:

- `kv_cache = TRUE` processes the training rows once per `predict()` call
  instead of once per batch. It is often about twice as fast. The TabPFN
  models already do this on their own when a call spans several batches,
  wherever it leaves the answer unchanged; `chunk_cache = FALSE` turns
  that off.
- `row_chunk_size`, `col_chunk_size` (TabPFN v3/v3.5, TabICL, TabFM,
  Kumo Tabular, Causilo) and
  `save_peak_memory_factor` (most models) trade speed for lower peak
  memory.
- `device = "cuda"` or `"mps"` runs on a GPU.
- `tabfound_threads(n)` sets the number of threads for this process. When
  running in parallel workers, set it to `parallel::detectCores() %/% k`
  inside each worker, before any model is used.

How large a table fits on an 8, 16, 32 or 64 GB machine, for each model:
`vignette("how-big-a-table", "tabfound")`.

## Saving a fitted model

`saveRDS()` does **not** work for torch objects. Use:

```r
tabfound_save(fit, "fit.tabfound")
fit <- tabfound_load("fit.tabfound")
```

This saves the training data and a reference to the weights, which is a
few kilobytes. `tabfound_cache(fit)` precomputes the model's pass over the
training rows and stores that too, so later sessions start predicting
immediately. The saved result is a larger directory, but predictions are
identical.

## Validation

Each model is a port of the published model *and* of the Python package
around it (preprocessing, ensembling and output), and both parts are tested
against the original on shared data. Predicted probabilities typically
agree to about 1e-6. This is the floating-point noise of the network
itself. Details, numbers and how to rerun the checks:
[`inst/parity/README.md`](inst/parity/README.md).

The imputation and synthesis methods have no reference implementation. They
are evaluated by simulation instead (see above).

## Learn more

- `vignette("tabfound")`: a longer tour, with worked examples and a
  comparison to `lm()`
- `vignette("multiple-imputation")`, `vignette("synthetic-data")`
- `vignette("how-big-a-table")`: memory and table size, per model
- `vignette("advanced")`: ensembles, the matrix interface, reproducing the
  Python reference exactly, and known limitations per model
- `vignette("architecture-gallery")`: diagrams of every architecture in
  five styles, side-by-side comparisons of two models, and how to add a
  new model

## Citation

```r
citation("tabfound")
```

Please also cite the model you use. `citation("tabfound")` includes the
references for TabPFN, TabICL and Causilo, and NVIDIA's announcement of
Kumo Tabular (there is no paper yet). For TabFM and Mitra, see the model card
on Hugging Face.

## License

MIT for the package. The model weights have their own licences (see
[Licences](#licences)).
