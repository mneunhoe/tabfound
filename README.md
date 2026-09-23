# tabfound

Tabular foundation models for R.

A tabular foundation model is a neural network that has been pretrained on
millions of synthetic datasets. You do not train it on your data. You
*condition* it on your data: the training rows go in alongside the rows you
want predictions for, and a single forward pass returns a full predictive
distribution. On small and medium tables (up to a few thousand rows), these
models are often competitive with a tuned gradient-boosted tree, with no
tuning at all.

`tabfound` runs several published models (TabPFN, TabICL, TabFM, Mitra)
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

Weights are stored in `tabfound_home()`, outside your R library, and are
downloaded once. Then they load by name, offline. You can pass a family
name (`"tabpfn-v2.5"`, `"tabicl"`, `"mitra"`) and the task decides which
checkpoint is used. A local directory path works wherever a name does.

**One-time Python step.** TabPFN and TabICL publish their weights in a
PyTorch format that R cannot read. `download_model()` converts them once,
and for that step it needs a Python interpreter with `torch` and
`safetensors` installed. Point `options(tabfound.python = )` at one if it
is not found. TabFM and Mitra need no conversion. After this step, Python
is never used again.

> **Check the licence before you publish or deploy.** The package is MIT,
> but the model weights are not. TabPFN v2.6, v3 and v3.5 and TabFM are
> licensed for **research and evaluation only**, not commercial or
> production use. `list_models()` shows each licence, and the download
> prompt repeats it.

## Which model?

No single model is best. Here is how they differ in the ways that usually
matter:

| model | name | size | licence | notes |
|---|---|---|---|---|
| TabPFN v2.5 | `"tabpfn-v2.5"` | ~40 MB per task | Prior Labs licence | Small and fast. The best-tested option for imputation (see below). |
| TabPFN v2.6 | `"tabpfn-v2.6"` | ~50 MB per task | non-commercial | |
| TabPFN v3 | `"tabpfn-v3"` | ~200 MB per task | non-commercial | Scales to larger tables. Six specialised variants (binary, OOD, time series, …). |
| TabPFN v3.5 | `"tabpfn-v3.5"` | 876 MB, both tasks | research only | Newest. Handles free-text and date columns natively. |
| TabICL v2 | `"tabicl"` | ~110 MB per task | see repo | Regression returns a full quantile grid. **Not recommended for imputation.** |
| TabFM 1.0 | `"tabfm-1.0.0"` | ~6.6 GB per task | non-commercial | Large (1.6 B parameters) and slow on CPU. Regression is point prediction only. |
| Mitra | `"mitra"` | ~300 MB per task | Apache-2.0 | The only permissive licence. Best suited to small tables. Regression is point prediction only. |

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
  the way its reference does: TabPFN models missingness explicitly, while
  the others mean-impute first. `Inf` is an error. Recode it to `NA` or to
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
  instead of once per batch. It is often about twice as fast.
- `row_chunk_size`, `col_chunk_size` (TabPFN v3/v3.5, TabICL, TabFM) and
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
- `vignette("architecture-gallery")`: diagrams of every architecture, and
  how to add a new model

## Citation

```r
citation("tabfound")
```

Please also cite the model you use. `citation("tabfound")` includes the
references for TabPFN and TabICL. For TabFM and Mitra, see the model card on
Hugging Face.

## License

MIT for the package. The model weights have their own licences (see
[Getting a model](#getting-a-model)).
