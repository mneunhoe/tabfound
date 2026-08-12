# tabfound

Tabular foundation models in pure R [`torch`](https://torch.mlverse.org/).
No Python runtime at predict time.

`tabfound` offers a unified R package for different Tabular Foundation Models
port into a package where a model family is a *backend*: a description of
how to build a network from a config, how checkpoint keys map onto R
module paths, how to preprocess the design matrix, how to build and
combine an ensemble of transformed views of it, and how to turn the
network's output into predictions. One user-facing API serves all of
them.

Every backend is a port of a published model *and of the estimator
around it* — the preprocessing and ensembling its reference wrapper
does, not just its forward pass — and both halves are checked against
the original.

```r
# The formula interface. Factors, logicals, characters, Dates and
# missing values are handled for you; `mode` is inferred from the
# outcome's type.
fit <- tabfound(Species ~ ., data = iris[train, ], model = "path/to/model")
predict(fit, iris[test, ])                    # tibble: .pred_class
predict(fit, iris[test, ], type = "prob")     # .pred_setosa, ...

# ...or supply x and y directly.
fit <- tabfound(iris[train, 1:4], iris[train, 5], model = "path/to/model")
```

Underneath sits the engine, if you want the numeric-matrix interface:

```r
# TabPFN, from converted local artifacts. `fit()` returns a fitted copy;
# it does not modify its input.
clf <- tabular_classifier("path/to/converted/tabpfn-v2.5-clf", device = "cpu")
clf <- fit(clf, iris[train, 1:4], iris[train, 5])
predict(clf, iris[test, 1:4])                   # class labels
predict(clf, iris[test, 1:4], type = "prob")    # or predict_proba(clf, ...)

reg <- fit(tabular_regressor("path/to/converted/tabpfn-v2.5-reg"), X, y)
predict(reg, X_new)
predict(reg, X_new, type = "quantiles", quantiles = c(0.1, 0.5, 0.9))
predict(reg, X_new, type = "sample", n_samples = 100)

# TabFM, straight from the Hub -- same API, no conversion step. The
# backend is inferred from config.json and the task subfolder is picked
# for you.
clf <- fit(tabular_classifier("tabfm-1.0.0"), X, y)
predict(clf, X_new, type = "prob")

# TabICL. Its regressor predicts a full 999-level quantile grid rather
# than a point estimate, so summaries are read off the distribution --
# `mean` averages the whole grid, `median` inverts its CDF, and they are
# not the same number on a skewed one.
reg <- fit(tabular_regressor("path/to/converted/tabicl-v2-reg"), X, y)
predict(reg, X_new, type = "quantiles", quantiles = c(0.05, 0.5, 0.95))
predict(reg, X_new, type = "median")
predict(reg, X_new, type = "grid")              # all 999 levels

# All three of these ensemble by default, as their Python wrappers do.
# The knobs are the wrappers' own, and `random_state` reproduces the
# reference's members exactly.
clf <- fit(tabular_classifier("path/to/converted/tabicl-v2-cla",
                              n_estimators = 16, norm_methods = c("none", "power"),
                              random_state = 42), X, y)

# Mitra, also straight from the Hub.
clf <- fit(tabular_classifier("mitra-classifier"), X, y)
predict(clf, X_new, type = "prob")

# Saving. `saveRDS()` cannot capture torch weights -- use these instead.
tabfound_save(clf, "clf.tabfound")
clf <- tabfound_load("clf.tabfound")
```

## Status

| backend | status | verified against |
|---|---|---|
| `tabpfn` (TabPFN v2 / v2.5, Prior-Labs) | working — classifier and regressor, single pass, n-member ensemble, KV cache, chunked forward | `tabpfn` 8.2.0 on PyPI |
| `tabpfn26` (TabPFN v2.6, Prior-Labs) | working — classifier and regressor, single pass and n-member ensemble | `tabpfn` 8.2.0 on PyPI |
| `tabpfn3` (TabPFN v3, Prior-Labs) | working — classifier and regressor, single pass and n-member ensemble, KV cache, chunked forward, row/column stage chunking | `tabpfn` 8.2.0 on PyPI |
| `tabfm` (Google TabFM 1.0.0) | working — classifier and regressor, preprocessing + 32-member ensemble, KV cache, row stage chunking | `tabfm` on PyPI, float32 |
| `tabicl` (TabICL v2, soda-inria) | working — classifier and regressor, preprocessing + 8-member ensemble, KV cache, row stage chunking | `tabicl` 2.1.1 on PyPI |
| `mitra` (Mitra, AutoGluon) | working — classifier and regressor, preprocessing + ensemble, KV cache, chunked forward (inference only, no fine-tuning) | AutoGluon `Tab2D` |

`list_backends()` reports what is registered in your install.

## The formula interface

`tabfound()` is the layer an R user wants: a formula or an `x`/`y` pair,
a data frame with whatever column types it happens to have, and a
`predict()` method that returns a tibble.

It delegates the hard part — remembering what the training data looked
like and re-applying that to new data — to
[hardhat](https://hardhat.tidymodels.org/), which also validates new
data and fails precisely: a missing column names the column, a novel
factor level names the level.

The Python wrappers solve the same problem with a `TransformToNumerical`
step that sniffs pandas dtypes to guess which columns are categorical. R
does not need to guess. `is.factor()` and `inherits(x, "Date")` are
exact, so the encoder dispatches on real types:

| R type | encoded as |
|---|---|
| numeric, integer | unchanged |
| factor, ordered | 0-based ordinal codes (not one-hot — none of the three references one-hot encode features) |
| character | coerced to factor, then as above |
| logical | coerced to factor, then 0/1 |
| Date, POSIXct, difftime | its numeric representation |

Encoding a factor to codes is only half the job: the model still has to be
*told* those columns are categorical, or it will treat the codes as a
quantity with an order and a scale. `tabfound()` passes the factor columns'
positions down automatically. This matters more than it looks — the
reference's own inference only catches columns with fewer than four
distinct values, so a five-level factor is invisible to it, and R knows
the answer exactly where pandas has to guess:

```r
# Declared for you, from the frame's factor columns.
fit <- tabfound(churn ~ ., data = d, model = "tabpfn-v2.6-classifier")

# Or by hand, on the matrix interface, as 1-based column positions.
clf <- tabular_classifier("tabpfn-v2.6-classifier", categorical_features = c(1, 3))
```

What that changes: for members whose config asks for it, declared columns
skip the primary transform (a quantile transform on category codes is
meaningless), get ordinal-encoded with their codes randomly permuted per
member, and move to the front of the matrix. Backends with no categorical
handling of their own warn rather than silently ignoring the declaration.

Logicals are coerced deliberately rather than incidentally: the formula
path runs a `model.matrix` expansion that would turn a bare logical into
two dummy columns while leaving a factor alone, so coercing first is
what keeps the formula and `x`/`y` interfaces producing the same result.

**Missing values** are handled per backend, because the backends
genuinely differ — and in every case the handling is the reference
implementation's own. TabPFN encodes missingness as an explicit
indicator channel and TabFM maps it to a sentinel. TabICL's *network*
has none of that and propagates `NaN` straight to its output, and
Mitra's silently deletes any column containing one; both of their
predictors impute first, exactly as their Python wrappers do, so neither
failure mode is reachable through the API. All four therefore take `NA`
directly and `na_action = "auto"` passes it through to them.

Which route a backend takes is recorded, and is worth knowing rather
than glossing: `list_backends()$missing` reads `"encoded"` for TabPFN,
`"imputed"` for TabICL and Mitra, `"encoded+imputed"` for TabFM. The two
are not the same guarantee. A network that encodes missingness
*conditions on* it — the fact that a value was absent is available to
the model, and to anything you ask the model about it. A wrapper that
mean-fills has replaced that fact with a column mean before the network
is reached. For prediction the distinction rarely matters; for
imputation and synthesis, where the missingness pattern is the object of
study, it is most of the question, and it is one reason the MI
simulation finds TabPFN and TabICL so far apart.

TabICL's and TabFM's imputation is `SimpleImputer(strategy = "mean")`,
fitted on the training rows: new data is filled with the *training*
column means, and a column that is entirely missing is dropped rather
than invented. Mitra's is AutoGluon's own, which is the same idea.

`Inf` is **not** treated as missing and is rejected with an error naming
the columns. That is the reference's line too — sklearn takes `NaN`
where it advertises it and refuses infinities — and it is the right
line: an infinity cannot be imputed, and left alone it poisons a column
mean and every scaler downstream of it without producing a single `NaN`
to notice. Replace them with `NA` to have them imputed, or with a finite
bound if that is what they encode. Mitra is the exception and maps them
to zero, because its own preprocessor does.

`na_action` still takes `"auto"` (the default), `"pass"`, `"impute"` or
`"fail"`, and `"impute"` remains available if you would rather this
package's plain column-mean fill ran instead — but it is *not* a port of
any reference implementation and is not covered by the parity harness,
so the backend's own is the better default. Forcing `"pass"` on a
backend that declares it cannot cope still warns rather than silently
returning `NaN`.

`recipes` is not wired up yet, and neither is a parsnip engine.

## Multiple imputation

`predict()` throws the model's uncertainty away at the last step.
Multiple imputation is the case where that uncertainty *is* the product,
and these models supply it directly: TabPFN's regression head is a bar
distribution you can sample from, TabICL's is a 999-level quantile grid
you can invert, and every classifier emits a category distribution.

```r
mods <- tabfound_models(classifier = "path/to/tabpfn-v2.5-clf",
                        regressor  = "path/to/tabpfn-v2.5-reg")

imp <- tabfound_impute(airquality, m = 5, models = mods, maxit = 3, seed = 1)
```

Chained equations, one draw from the predictive distribution per missing
cell. Column types pick the head the way they do everywhere else, with
one exception: numeric 0/1 columns take the classifier path and come back
as 0/1, because a continuous draw is not a valid value of a dummy. The
classifier is loaded lazily, so data with no incomplete factors never
pays for it.

The result pools with whatever you already use:

```r
library(mice)
summary(pool(with(imp, lm(Ozone ~ Wind + Temp))))   # with() returns a mira

mids <- as_mids(imp)          # the full mice toolchain: densityplot(), complete(), ...
plot(mids)                    # convergence traces, recorded per sweep
mice::complete(imp, "long")   # registered on mice's own generic

amp <- as_amelia(imp)         # Amelia::mi.meld() and friends
```

`where` and `post` are arguments here too — `where` picks the cells to
draw (marking an *observed* cell overimputes it, which is how you check
the model against values you already have), and `post` takes a named list
of functions applied to each variable's draws, the place to squeeze a
value into a plausible range. mice's `post` is a string evaluated inside
its sampler; a function does the same job without reaching into the loop.

Or let mice drive and call in here per variable, which buys mice's
`blocks`, `ignore` and its own diagnostics:

```r
options(tabfound.models = mods)
mice(airquality, method = "tabfound", m = 5)
```

Unlike the rest of the package this has no reference implementation to be
verified against, so the parity harness has nothing to say about it — it
is a use of the fitted models, not a claim about them. What stands in for
parity is a simulation where the right answer is known by construction,
in [`inst/simulation/`](inst/simulation/README.md): missingness that
depends on the outcome, so that a complete-case analysis is provably
biased and a correct imputation puts the estimate back.

```bash
Rscript inst/simulation/run-mi-sim.R --backend=lm --reps=200      # 13s, no weights
Rscript inst/simulation/run-mi-sim.R --backend=tabpfn --reps=200
```

On that design (200 reps, `n = 400`, ~42% missing) complete-case analysis
carries a bias of −0.111 with 0.78 coverage on the coefficient of
interest. **TabPFN** imputation cuts that to −0.008 with 0.965 coverage —
matching a correctly specified parametric imputer and `mice`'s PMM for
bias removal, having been told nothing about the data-generating process.
Its intervals run systematically narrower than PMM's, which on one
coefficient tips into undercoverage (0.900 vs 0.950), so PMM remains the
safer default if you want intervals you need not think about. **TabICL
does not do the job**: −0.061 with 0.86 coverage, and on the categorical
coefficient it removes essentially none of the bias. Use the TabPFN
backend for imputation; the numbers and the open questions are in
[`inst/simulation/README.md`](inst/simulation/README.md).

**Properness.** Between-imputation variability comes from the posterior
predictive. A PFN draws each query row independently given a fixed
context, so *m* chains conditioned on the same observed rows differ only
in the noise of the draw: parameter uncertainty is missing, which is the
defect that makes an imputer improper under Rubin's rules.
`proper = TRUE` applies the usual correction — resample the context once
per imputation — and it is **off by default**, against the theory and
with the measurements. On the simulation above it moves coverage to
nominal (0.900 → 0.955 on the fully observed covariate) and costs an
order of magnitude in bias (−0.008 → −0.062, against a complete-case
−0.111), widening intervals by up to 92%. The same correction is free for
a correctly specified parametric imputer on the same data. A bootstrap
context is not the same model with different parameters — only 63% of its
rows are distinct, and for a learner whose entire fit *is* its context
that is a third of the training data thrown away. Turn it on if you need
nominal coverage more than the point estimate; the numbers are in
[`inst/simulation/README.md`](inst/simulation/README.md).

There is no equivalent of Amelia's `bounds` — draws come off the
predictive distribution, tails included. Backends with a
point-estimate-only regression head (TabFM, Mitra) have nothing to draw
from and are refused unless you ask for `draw = "residual"`.

Full walk-through: `vignette("multiple-imputation", "tabfound")`.

## Synthetic data

The same predictive distributions, pointed at every cell instead of the
missing ones. `tabfound_syn()` is sequential (fully conditional)
synthesis, the way `synthpop::syn()` does it: each variable is modelled on
the **real** data given the variables before it in the visit sequence, then
drawn at the **synthetic** values of those variables generated earlier in
the same pass. The first variable is a bootstrap of its real marginal.

```r
sds <- tabfound_syn(iris, m = 5, models = mods, seed = 1)
```

One pass, no iteration — *p* fits rather than the *m* × sweeps × *p* the
imputation path needs. And because conditioning on a context is not
training, with `proper = FALSE` all *m* syntheses share one fit and one
forward pass over a stacked query. A tree has to be regrown per synthesis;
this does not.

Everything downstream is synthpop's, because it already owns it:

```r
library(synthpop)
compare(sds, iris)                                    # registered on synthpop's generics
utility.gen(sds, iris)
summary(lm.synds(Sepal.Length ~ Petal.Length, as_synds(sds)))
replicated.uniques(as_synds(sds), iris)$no.uniques
```

Or let synthpop drive the pass and call in here per variable, which buys
`rules`/`rvalues` constraints, `cont.na` handling and `k != n` generation:

```r
options(tabfound.models = mods)
syn(iris, method = "tabfound", m = 5)
```

`syn()` rejects `...` arguments not named for one of its own methods, so
tuning goes through `options(tabfound.syn = list(draw_mode = "pmm"))`.

Three things are decisions rather than details, and all three are meant to
be reported rather than defaulted away:

**The sampler.** A CART leaf is a set of observed values, so `syn.cart`
cannot produce an unobserved value, leave the observed range, or break
integrality. A bar distribution is a continuum and does all three.
`draw_mode` is that tradeoff: `"predictive"` keeps the draw (smooth, novel
values, low replication risk, smeared point masses), `"pmm"` returns a real
donor from the nearest observed values (CART's support semantics exactly,
highest replication risk), `"rank"` maps the draws onto the observed order
statistics (marginal reproduced exactly — which means that part of the
fidelity is by construction). Categoricals need no such choice: a softmax
plus one draw per row already *is* the analogue of a leaf's class
proportions.

**Properness.** These models draw each query row independently given a
fixed context, so between-synthesis variance carries predictive noise and
no parameter uncertainty — the same defect that makes an imputer improper
under Rubin's rules, and one the synthetic-data variance estimators assume
away. `proper = TRUE` bootstraps the context rows once per synthesis,
which is exactly synthpop's definition, and is the default here. This is
the one place `tabfound_syn()` deliberately disagrees with
`synthpop::syn()`.

It is also, on current evidence, the riskier default of the two. The MI
side ran the same correction through a coverage simulation and found it
costs an order of magnitude in bias on these models, because a bootstrap
context keeps only 63% distinct rows and an in-context learner's fit *is*
its context — see the MI section above. Synthesis has not been measured
that way, and its loss function is different (synthpop's estimators
*assume* properness), so the default stands; but expect the utility cost
to be of the same order, and compare against `proper = FALSE` before
concluding anything about fidelity.

**Scope.** The context *is* the real data at generation time. This is a
fully conditional synthesiser, not a disclosure-control method, and no
differential-privacy claim is available from it. `NA` is treated as a
value to reproduce rather than a hole to fill — an explicit level for
categoricals, a two-part model for continuous variables, which `cont_na`
extends to any other spike (`cont_na = list(income = 0)`).

As with imputation, there is no reference implementation to be verified
against and the parity harness has nothing to say about it.

Full walk-through, with a measured utility/disclosure table across the
three sampling modes and `syn.cart`:
`vignette("synthetic-data", "tabfound")`.

## Model objects

Fitted models are plain lists with value semantics. `fit()` returns a new
object, so `m2 <- fit(m1, X, y)` leaves `m1` unfitted, the same contract
as `lm()` and everything else in R. The network itself is shared rather
than copied — it is read-only and can be several gigabytes — while the
fitted context (training rows, class levels, target scaler) is ordinary R
data.

`saveRDS()` does **not** work: a torch module survives the round trip
structurally but its tensors come back as dangling pointers, and the
failure only shows up later, in use. [`tabfound_save()`] /
[`tabfound_load()`] are the supported path, for both object types: the
engine-level `tabfound_model` and the `tabfound_fit` that `tabfound()`
returns, whose hardhat blueprint travels with it so a reloaded fit
re-applies the same encoding and factor levels to new data. They write
only the fitted context plus a reference to the model artifacts — a few
kilobytes — and re-resolve the weights on load, since fitting these
models stores context rows rather than learning parameters. Using a
`saveRDS`-ed object raises an error naming the fix rather than failing
inside torch.

### Fit once, query forever

Conditioning the network on the training rows is the expensive half of a
prediction, and it does not depend on what you are predicting.
`kv_cache = TRUE` already does it once per `predict()` instead of once
per chunk; `tabfound_cache()` does it once and *keeps* the result, on the
object and through a save:

```r
clf <- fit(tabular_classifier(dir, kv_cache = TRUE), X_train, y_train)
clf <- tabfound_cache(clf)          # condition now, once
tabfound_save(clf, "clf-bundle")    # a directory, not a file
```

```r
clf <- tabfound_load("clf-bundle")  # next session, next machine
predict(clf, X_new, type = "prob")  # starts from the conditioned state
```

Measured on TabPFN v2.5, 800 context rows × 6 features, 512 query rows:
0.43 s per uncached call against 0.17 s to build the cache plus 0.13 s
per call — and the reload costs 0.25 s, so it pays for itself on the
first call of the second session. On TabICL (1,000 × 6, 2,000 query
rows) it is 4.9 s against 2.3 s, a 2.1× speedup that persists.
Predictions are **bit-identical** across the session boundary.

What travels is a bundle directory: `state.rds` beside
`cache.safetensors`. The RDS holds the skeleton of the cache — its list
structure, classes and integers, all ordinary R data — and the tensors go
to safetensors under dotted paths (`kv.3.key`), which is precisely the
thing `saveRDS()` cannot carry. A model with no cache is still a single
file, as before.

Two things to know. The cache is **large** — it is the conditioned state
of every ensemble member, 32 MB for that TabPFN fit and 409 MB for the
TabICL one — so this trades disk for latency, deliberately. And it is a
function of the training rows: it is fingerprinted when built and checked
when used, so a cache that has outlived its context raises an error
naming both shapes rather than quietly answering from the wrong data.

`has_cache()` says whether an object carries one;
`tabfound_cache(object, build = FALSE)` drops it.

## Parity

Every backend is checked end-to-end against the reference Python
implementation on shared fixtures, staged so a mismatch localises itself
to preprocessing, the forward pass, or the output head. See
[`inst/parity/README.md`](inst/parity/README.md) for how to run it and
what the current numbers are.

Two things get compared, because two things can be wrong. The
**network** is graded against the reference's own forward pass on
identical bytes. The **estimator around it** — the preprocessing, the
ensemble construction, the combination — is graded separately, because a
port can be exactly right about the architecture and still return the
wrong number.

### The wrapper layer, without a checkpoint

`inst/parity/ensemble_reference.py` grades everything between the raw
columns and the network, and needs no model weights at all, so it runs
in the ordinary test suite rather than only on a machine with 6.5 GB of
checkpoints on it. Six pieces, each depending on the one above it:

| piece | result |
|---|---|
| CPython's `random.Random` | exact on all five seeds, across `random`, `getrandbits`, `_randbelow`, `shuffle`, both branches of `sample`, `choice` |
| `Shuffler` | exact on all 40 (size, method, seed) combinations, including the Latin squares |
| the sklearn transformers | exact to ~1e-13 or better |
| both ensemble generators | every member of six configurations: same permutation, same relabelling, same normalisation, same matrix |
| Mitra's preprocessor | bit-identical after the float32 narrowing |
| TabICL's quantile distribution | exact, spline and exponential tails alike |

The generators are compared member for member, not distributionally.
Which permutation member 3 gets is decided by
`random.Random(random_state)`, so an ensemble that is merely
"statistically equivalent" is a different set of predictions.
`R/py-random.R` is therefore a port of CPython's MT19937 — its
`init_by_array` seeding, its `getrandbits`, its rejection-sampling
`_randbelow` — because R's own `sample()` agrees with none of it.

Two gaps are stated rather than papered over. The Yeo-Johnson lambda
agrees to 2e-8, not bitwise: sklearn hands the search to
`scipy.optimize.fminbound`, which is ported step for step, but the
likelihood goes through a `logsumexp` whose summation order R cannot
share, and Brent's accept/reject test is a strict comparison — so the
two searches occasionally take different final steps and stop within
`xatol` (1.48e-8) of each other. And Mitra's random sign flips cannot be
reproduced at all, because the reference draws them from NumPy's
*global* generator and never seeds it; they differ between two of its
own runs.

### End to end

Against `TabICLClassifier` / `TabICLRegressor` on the released v2
checkpoints, `n_estimators = 8`, through the full public API:

| output | max_abs | of its own scale |
|---|---|---|
| `predict_proba` | 1.6e-6 | 1.6e-6 |
| `predict` (mean) | 4.9e-6 | 2.0e-7 |
| `predict` (median) | 5.5e-6 | 2.2e-7 |
| `predict` (quantiles) | 7.5e-6 | 2.8e-7 |

That is the float32 forward pass's own noise, reached through eight
independently preprocessed members — so the preprocessing, the member
construction, the class-relabelling inversion and the ensemble
combination all agree, not just the network.

### Per backend

**Mitra** (76 M parameters) against AutoGluon's `Tab2D`, 20/20 checks in
tolerance across classification and regression:

- All 392 / 393 checkpoint tensors map onto the R module tree by
  **identity**.
- The quantile-rank embedding and the packed model input are
  **bit-identical** (`max_abs == 0`) on every fixture.
- `predict_proba` agrees to 9e-7, regression `predict` to 1.4e-5.
- Its preprocessor — mean imputation, the constant-column drop, the
  min-max target scaling and its inverse — is bit-identical to
  AutoGluon's.
- Ships `model.safetensors` + `config.json` on the Hub under Apache-2.0,
  so no conversion step and no license caveat.

**TabICL v2** (28 M parameters) against the `tabicl` package, 16/16
checks in tolerance across classification and regression:

- Every stage agrees to ~1e-6 of its own scale; `predict_proba` to 2e-6
  and the regressor's full 999-level quantile grid to 5e-6.
- TabICL's forward pass has two implementations — a plain three-stage
  path and an eval path with chunking and K/V caching. The harness
  records the gap between them, currently **exactly 0**, so the path this
  port implements is the path users get.
- 62 layer-level assertions cover both families, including the
  differences between them: the two GELU flavours and the two RoPE
  pairing conventions are each asserted to disagree.
- The estimator around it is checked end to end, at the numbers in the
  table above — the only one of the three for which weights were
  available locally to do so.

**TabFM 1.0.0** (1.6 B parameters, 913 tensors) against the `tabfm`
package, float32, 21/21 checks in tolerance across classification and
regression:

- All 913 checkpoint tensors map onto the R module tree by **identity** —
  no key rewriting needed.
- The cell embedder is **bit-identical** (`max_abs == 0`): Fourier
  expansion, cyclic feature grouping and target embedding all exact.
- Every later stage agrees to ~1e-6 of its own scale; `predict_proba` to
  3e-6 on 48/12 rows and 7e-6 on iris, and the regression head to 2e-6.
- Layer-level assertions (RMSNorm, RoPE, per-head-norm attention,
  sandwich-norm blocks, induced set attention, tanh-gelu MLP) pass
  against the reference classes **without needing the checkpoint**.
- So does the `cat_mask` check — the flag that routes a declared
  categorical column's cells through a separate Fourier basis. On random
  weights the two implementations agree to 2e-7 with a mask, both give
  exactly the plain forward pass when it is all-`FALSE`, and both move
  the logits by 3.7e-3 when it is not. That last number is why the check
  exists: the mask is not bookkeeping, and a wrapper that builds one
  while the network ignores it looks correct in every stage-level check.

Headline for the TabPFN backend, against `tabpfn` 8.2.0:

- Per-member model inputs are **bit-identical** (`max_abs == 0`) on all
  four fixtures — including one with missing values, a constant column,
  and infinities.
- `predict_proba` agrees to ~2e-6; regression `predict` to ~2e-5 and
  quantiles to ~6e-6.
- The one caveat is TabPFN's row-fingerprint feature, which is a SHA-256
  of the row's float64 bytes and therefore flips wholesale on a 1-ULP
  upstream difference. That is quantified rather than hidden — see the
  parity README.

For the TabPFN **v2.6** backend, against the same package:

- v2.6 moved the preprocessing inside the network, so the *single-pass*
  comparison is network-to-network on identical bytes rather than staged
  around a pipeline. Logits agree to ≤1.8e-5 of their own scale across
  six fixtures, including one with a constant column and 25 missing
  values. Decoding — softmax for the classifier, the bar distribution for
  the regressor — is **bit-identical** (`max_abs == 0`) when both sides
  are handed the same logits.
- The *ensemble* is graded like v2.5's, through the full estimator on
  both sides. Per-member model inputs are **bit-identical** on all four
  fixtures — including the regressor's polynomial-feature expansion, and
  including the fixture with missing values and a constant column.
- `predict_proba` agrees to ~8e-6; regression `predict` to ~3e-5 and
  quantiles to ~4e-5.
- The KV cache and the chunked forward are each graded twice: against the
  reference's own run of the same mechanism, and against R's own plain
  forward. Chunking is **bit-identical to the unchunked pass on every
  fixture**. The cache is bit-identical on the four fixtures where the
  reference's own cache is, and shifts the prediction on the same two it
  shifts for the reference, by the same order of magnitude.
- Two fixtures carry genuine categorical columns — 3, 6 and 12 levels
  alongside numerics — and are run twice: declared, so the ordinal
  encoder reaches them, and undeclared, so only the low-cardinality one
  is inferred. Both sides agree exactly on *which* columns are
  categorical, and the member inputs stay bit-identical through the
  encoding and the reordering it causes.

For the TabPFN **v3** backend, against the same package:

- v3 keeps v2.6's arrangement where the NaN/Inf handling and the standard
  scaler live inside the network, so this is again a network-to-network
  comparison on identical bytes. Across six fixtures — including one with
  a constant column and 25 missing values, and one with 12 rows —
  **all 42 checks pass**: logits agree to ≤2.9e-6 of their own scale, and
  decoding is **bit-identical** (`max_abs == 0`) when both sides are
  handed the same logits.
- End-to-end `predict_proba` from a single forward pass agrees to ~1.6e-6.
- The KV cache and the chunked forward are each graded against the
  reference's own run of the same mechanism *and* against R's own plain
  forward, both to ≤2.9e-6 of scale. Neither is bitwise, and neither can
  be: both change the batch shapes the attention kernel sees. But unlike
  v2.6's, v3's cache is the *same computation* as the uncached pass —
  nothing in the architecture is fitted over train and test together —
  so the self-check is an unconditional assertion rather than a
  conditional one.
- **All eight published v3 checkpoints** were run through the same
  comparison, not just the two defaults: the four classifiers agree to
  ≤1.2e-6 and the four regressors to ≤2.3e-5, on the same input. The
  stored parity tree covers the defaults only, since the other six share
  their architecture exactly and would grade the same code twice.

The shared preprocessing steps are additionally pinned against sklearn
and TabPFN's own step classes in `tests/testthat/test-prep-transforms.R`,
and the wrapper layer in `test-prep-sklearn.R`, `test-prep-ensemble.R`
and `test-py-random.R`. None of those need model weights, so they run
anywhere the package does — which is the point: they cover the code most
likely to drift, on the machines least likely to have a checkpoint.

## Installation

```r
# install.packages("remotes")
remotes::install_local("/path/to/tabfound")
```

Requires `torch`, `R6`, `cli`. For full functionality also `safetensors`,
`jsonlite`, `hfhub`, `digest`.

## Getting the weights

```r
list_models()                              # what exists, and what you have
download_model("tabpfn-v2.5-classifier")   # fetch + convert, once
```

`download_model()` pulls a model into a local store and, for publishers
that ship PyTorch pickles, runs the conversion. Afterwards it loads by
name with no network and no Python:

```r
clf <- tabular_classifier("tabpfn-v2.5-classifier")
fit <- tabfound(Species ~ ., data = iris, model = "tabpfn")   # family + task
```

A family name plus the task picks the head, so `"tabpfn"` is the
classifier under `tabular_classifier()` and the regressor under
`tabular_regressor()`. **`"tabpfn"` stays pinned to v2.5** — adding v2.6
does not repoint existing code onto different weights. Ask for a version
by name:

```r
clf <- tabular_classifier("tabpfn-v2.6-classifier")
fit <- tabfound(Species ~ ., data = iris, model = "tabpfn-v3")
```

TabPFN v2.6 and v3 weights are released under `tabpfn-2.6-license-v1.0`
and `tabpfn-3-license-v1.0`, which permit research, evaluation and
internal benchmarking but **not commercial or production use**.
`list_models()` prints the licence, and the download prompt repeats it.

v3 publishes six specialised checkpoints alongside the two defaults, and
all eight are catalogued. They are **the same architecture** — identical
`config`, identical state-dict keys and shapes, checked checkpoint by
checkpoint — so they load through the same backend with no special
handling:

```r
clf <- tabular_classifier("tabpfn-v3-classifier-binary")
reg <- tabular_regressor("tabpfn-v3-regressor-timeseries")
```

| id | what the publisher tuned it for |
|---|---|
| `tabpfn-v3-classifier-binary` | binary targets, <200k rows |
| `tabpfn-v3-classifier-multiclass` | multiclass targets, <200k rows |
| `tabpfn-v3-classifier-ood` | test inputs outside the training distribution |
| `tabpfn-v3-regressor-mediumdata` | <100k rows, alternative preprocessing |
| `tabpfn-v3-regressor-ood` | test inputs outside the training distribution |
| `tabpfn-v3-regressor-timeseries` | synthetic time series (used by TabPFN-TS-3) |

Two of them are worth knowing about before you spend the bandwidth. The
**`_ood` checkpoints carry byte-identical weights to the corresponding
defaults** — SHA-256 over the full state dict matches. What makes them
OOD-robust is the preprocessing recipe bundled in the checkpoint's
`inference_config` (`squashing_scaler_max10` in place of
`squashing_scaler_default`, and `none` or `quantile_uni_extrapolate` in
place of `quantile_uni` — all three implemented), not the network. Since
this package reads that recipe from an ensemble dump rather than from the
checkpoint, a single-pass prediction from `tabpfn-v3-classifier-ood` is
*the same number* as one from `tabpfn-v3-classifier`. The entries exist so
the publisher's names resolve; they are not four distinct models.

The remaining four do carry distinct weights, and all eight agree with
the Python reference (see the parity section).

Family names stay on the defaults: `"tabpfn-v3"` plus a task gives you
`tabpfn-v3-classifier` or `tabpfn-v3-regressor`, never a variant.

Skip the download step and the **first load asks first**:

```
! "tabfm-1.0.0-regressor" is not downloaded yet.
* source: "google/tabfm-1.0.0-pytorch"
* size: ~6.6 GB
* licence: NON-COMMERCIAL (weights); Apache-2.0 (source)
* destination: '.../models/tabfm-1.0.0-regressor'
Download it now? (yes/No/cancel)
```

Nothing is fetched without an answer. A non-interactive session fails
with the `download_model()` call to run rather than quietly pulling
gigabytes; `options(tabfound.download = "always")` opts into unattended
downloads, `"never"` refuses outright.

Weights live in `tabfound_home()` — `tools::R_user_dir("tabfound",
"cache")` by default, redirectable via `options(tabfound.home = )` or
`TABFOUND_HOME`. Deliberately *not* inside the installed package: writing
there breaks read-only and shared libraries, and CRAN forbids it.

A finished download records its file sizes in `SOURCE.json`, so an
interrupted one is detected and retried rather than passing the
"is it there?" check forever on a truncated file.

**Gated repos and offline use.** `hfhub` reads a token from
`HUGGING_FACE_HUB_TOKEN` / `HUGGINGFACE_HUB_TOKEN` and nowhere else,
which means access granted the normal Python way — `HF_TOKEN`, or
`huggingface-cli login`, which writes `~/.cache/huggingface/token` — is
invisible from R, and the resulting 401 surfaces as *"Connection error…
cannot find the requested files in the disk cache"*. tabfound reads all
four sources and forwards the token for the duration of the call, and
when a fetch fails that way it says the repo is probably gated, whether a
token was found, and points at the local-conversion route.

Every Hub-referenced load consults the cache before the network, so a
cached model costs no round trip (and no timeout when the network is
gone). `options(tabfound.offline = TRUE)` or `HF_HUB_OFFLINE` makes that
the only mode: a cache miss is an error naming the cache directory rather
than a hang.

**Conversion needs Python, once.** TabFM and Mitra ship
`model.safetensors` + `config.json` and need none. TabPFN and TabICL
publish PyTorch pickles that R cannot read — nested Python dicts holding
a state dict alongside hyperparameters, which R's torch bindings reject
outright. Converting them needs `torch` + `safetensors` in some
interpreter; point `options(tabfound.python = )` or `TABFOUND_PYTHON` at
it if the default search misses. Nothing calls Python afterwards.

The manual route still works and is unchanged:

```bash
hf download Prior-Labs/tabpfn_2_5 tabpfn-v2.5-classifier-v2.5_default.ckpt --local-dir ckpts

python inst/python/tabpfn_convert_ckpt.py \
  --src ckpts/tabpfn-v2.5-classifier-v2.5_default.ckpt \
  --dst-weights ckpts/converted/tabpfn-v2.5-clf/model.safetensors \
  --dst-config  ckpts/converted/tabpfn-v2.5-clf/config.json \
  --head classifier
```

The same script converts v2.6 (`Prior-Labs/tabpfn_2_6`,
`tabpfn-v2.6-classifier-v2.6_default.ckpt`) and v3
(`Prior-Labs/tabpfn_3`, `tabpfn-v3-classifier-v3_default.ckpt`); it reads
the checkpoint's own `architecture_name` and writes the matching `arch`
into `config.json`, which is what `tabular_classifier()` dispatches on.

A directory path works anywhere a model name does.

## Ensembling

None of these models is meant to be run once. Each reference estimator
builds several *views* of the same table — a different feature order, a
different class labelling, a different normalisation — runs the network
on each, and combines the results. The two families go about it
differently enough that they are documented separately.

### TabFM, TabICL and Mitra: on by default

These three ensemble as soon as you fit them, because their sklearn
wrappers do. Nothing needs configuring:

```r
clf <- fit(tabular_classifier("path/to/converted/tabicl-v2-cla"), X, y)
```

That is 8 members for TabICL, 32 for TabFM and 1 for Mitra — each
backend's own default. The arguments are the wrappers' own:

```r
clf <- tabular_classifier(
  model_dir,
  n_estimators        = 16,
  norm_methods        = c("none", "power", "quantile"),
  feat_shuffle_method = "latin",     # TabICL; TabFM takes "random"
  softmax_temperature = 0.9,
  average_logits      = TRUE,
  random_state        = 42
)
```

`random_state` is not a nod to reproducibility — it reproduces the
reference's members *exactly*. Which permutation member 3 gets is decided
by `random.Random(random_state)`, so `R/py-random.R` is a port of
CPython's MT19937: its `init_by_array` seeding, its `getrandbits`, its
rejection-sampling `_randbelow`. R's own `sample()` is a different
algorithm on a differently-seeded generator and agrees with none of it.
An ensemble that is merely "statistically equivalent" is a different set
of predictions.

What each view actually differs by:

| | TabICL | TabFM |
|---|---|---|
| feature order | Latin square (every column visits every position across the set) | independent random permutations |
| class labels | an arbitrary permutation | a cyclic rotation by a drawn offset |
| normalisation | `none` and `power` by default; `quantile`, `quantile_rtdl` and `robust` also available | same menu |
| combination | average logits, then one temperature-scaled softmax | same |

Each normalisation method fits one preprocessing pipeline —
`CustomStandardScaler` → the normaliser → a two-stage outlier clipper —
which every member sharing that method reuses. Mitra is the odd one out:
it has no designed set of views at all, only a random per-column sign
flip per member, and AutoGluon defaults to a single member.

Two of the reference's presets are deliberately absent. TabFM's
`ensemble()` preset (feature crosses, SVD features, NNLS ensemble
weights, probability calibration) needs out-of-fold cross-fitting that
this package does not have, so `n_feature_crosses` and `n_svd_features`
raise rather than being silently ignored. Mitra's fine-tuning is
out of scope for an inference-only package, so predictions match a
`MitraClassifier` with `fine_tune = FALSE`.

### TabPFN: from a dumped config directory

A single forward pass is the default. To reproduce the reference
estimator's n-member ensemble, point at a dumped config directory:

```r
clf <- tabular_classifier(model_dir, ensemble_configs_dir = "path/to/dump")
```

`tabpfn`, `tabpfn26` and `tabpfn3` all support it. A member is described
entirely by its config — which primary transform, whether the transformed
block replaces or is appended to the originals, whether an SVD block and
polynomial products are added, the column shuffle, the class rotation —
so one composer serves every member type of every version, and adding a
version means adding entries to a table rather than a code path.

The implemented primary transforms are `none`,
`squashing_scaler_default`, `squashing_scaler_max10`, `quantile_uni`,
`quantile_uni_coarse`, `quantile_uni_fine` and
`quantile_uni_extrapolate`, which between them cover every recipe the
eight published v3 checkpoints and both v2 generations ask for. A dump
naming anything else raises rather than substituting a near neighbour.

Everything the reference draws from its NumPy generator (the shuffle, the
class rotation, the polynomial factor pairs, any target-transform lambda)
is read out of the dump rather than recomputed: reproducing PCG64 in R
is not worth the days it would take.

`generate_ensemble_configs_native()` builds a working ensemble in pure R,
no Python needed. Pass `variant` to match your weights — it decides which
member menu is used, and the two generations do not share one:

```r
d <- tempfile()
generate_ensemble_configs_native(X, y, n_estimators = 4,
                                 head = "classifier", variant = "v2.6",
                                 output_dir = d)
clf <- tabular_classifier("tabpfn-v2.6-classifier", ensemble_configs_dir = d)
```

It is not bit-identical to Python at a given seed, and does not try to
be; for verified parity, dump the configs from the reference with
`generate_ensemble_configs()`.

One catch when combining it with `tabfound()`: the configs record a code
permutation per encoded categorical column, sized from the matrix you pass
here — so that matrix and its `categorical_features` have to be the ones
the model will actually see. `tabfound()` molds the frame through hardhat
first, which reorders columns, so generate from the molded matrix rather
than the raw frame. A mismatch is caught rather than silently mis-encoded:
the pipeline refuses when the config and the data disagree about which
columns are categorical.

`softmax_temperature` defaults to `0.9` across every backend, matching
the reference estimators. Pass `softmax_temperature = 1` for the
untempered distribution.

## Making prediction cheaper

None of these models have weights to fit, so "training" is conditioning:
the training rows sit in the context of every forward pass. That makes
prediction cost grow with the *training* set, and grow again for every
batch of test rows, since the whole context is rebuilt each time. Two
arguments change that.

```r
clf <- tabular_classifier("tabpfn-v2.6-classifier",
                          kv_cache = TRUE,             # condition once
                          save_peak_memory_factor = 8) # smaller transients
```

`kv_cache` works on every backend — `tabpfn` (v2 / v2.5), `tabpfn26`,
`tabpfn3`, `tabicl`, `tabfm` and `mitra`. `save_peak_memory_factor` works
everywhere except `tabfm`; `row_chunk_size` and `col_chunk_size` work on
`tabpfn3`, `tabicl` and `tabfm`. Asking a TabPFN
backend for a path it does not have is an error rather than a silent
no-op.

**`kv_cache`** conditions on the training rows once per `predict()` call
and reuses that for every chunk and every ensemble member. On TabPFN the
training rows reach the test rows through exactly two channels, and both
are computed once: the key/value projections of the between-items
attention (one head per layer — test rows attend to that head alone), and
the preprocessing statistics fitted on the training rows. Measured on this
machine, 1500 train / 3000 test rows in 6 chunks:

| | no cache | `kv_cache = TRUE` |
|---|---|---|
| v2.5, single pass | 3.6 s | **1.9 s** |
| v2.6, single pass | 4.5 s | **2.1 s** |
| v2.6, 4-member ensemble | 10.1 s | **6.4 s** |
| v3, single pass | 3.6 s | **1.8 s** |
| TabICL, single member | 4.5 s | **2.1 s** |
| TabICL, 4-member ensemble | 16.9 s | **7.6 s** |

Mitra and TabFM have no weights on this machine, so they were timed at
their published architecture dimensions with random initialisation —
wall clock depends on tensor shapes, not on the values in them. On 1000
train / 2000 test rows in 4 chunks: Mitra (512-wide, 12 layers) 15.6 s →
**8.2 s**, TabFM (1.6 B parameters) 30.3 s → **15.6 s**. TabPFN v2.5 on
the same shape is 3.8 s → **1.9 s**.

Where the speedup comes from is worth being precise about, because it
tells you when to bother. Uncached, each of `k` chunks re-encodes
`n_train + chunk` rows; cached, the training rows are encoded once and
each chunk carries only its own. So the ceiling is
`k * (n_train + chunk) / (n_train + k * chunk)` — 2.0x at the shape
above, and higher with more chunks or a larger training set. TabPFN,
Mitra and TabFM all land within a few percent of that ceiling, because
nearly all their work is context work. TabICL lands further below it
(1.4x at that shape, 2.1x at 1500/3000 in 6 chunks): its row stage
attends across a single row's columns and its cell projection is
per-row, so a real share of its cost was never the training rows' to
begin with and no cache can remove it.

The generations and architectures also differ in what the cache costs
you, and it is worth knowing which one you are on.

**v2.5 pays nothing.** Every statistic its network fits comes from the
training rows alone — the imputation mean, the z-normalisation mean and
standard deviation, the per-group non-constant mask. Nothing is fitted
over train and test together, so conditioning once cannot answer a
different question. What remains is float32 reduction order, and the
parity harness grades that against a bound rather than a chosen constant:
the shift the *uncached* path already shows when the same rows run in a
different-sized batch. On the fixtures the cache moves things either not
at all or by 0.7–1.2× that pre-existing shift.

**v2.6 can.** It moved two of those masks — the constant-column mask and
the within-group informative mask — to a fit over train *and* test.
Building a cache fits them on the training rows alone. Where those agree,
a cached prediction is bit-identical; where they don't, it is a different
prediction. The clearest case is a column constant across the training
rows that varies across the test rows: uncached it survives and is used,
cached it was dropped before the cache existed. That is the reference's
own behaviour on the same fixtures, and the harness reports it one by one.

**v3 pays nothing either**, and holds more: its cache carries the scaler
statistics, the inducing summary of each distribution-embedder block, the
one-head key/value projections of each of the 24 ICL blocks, and the
training rows' final embeddings, which the classifier's retrieval decoder
reads. Every one of those is a function of the training rows alone, so
the cached and uncached paths are the same computation.

**The other three backends pay nothing either**, for the same structural
reason, but each holds a different shape of thing.

*TabICL* restricts context by **slicing**: the column stage's inducing
points read `src[, , 1:train_size, ]`, and every ICL block's keys are the
first `train_size` positions of its own normalised input. A labelled row
never sees a test row at any stage. The cache holds one inducing summary
per column block — fixed size, however many training rows there were —
and one key/value pair per ICL block, which does scale with the training
set. The row stage needs nothing: it attends across a single row's
columns, so it never looked at another row to begin with. On the run
above the largest change in `predict_proba` was 2.9e-06.

*TabFM* has the same three-stage shape but restricts context by
**masking** rather than slicing, and does the column stage twice. The
cache holds the inducing summaries of both column stages and the ICL
blocks' key/value pairs. Masked positions contribute exactly zero, so the
cached and uncached passes compute the same sum — grouped differently,
which leaves float32 rounding of about 1e-06 of the logit scale. With the
default 32-member ensemble and a 512-row chunk this is the difference
between one training pass per member and one per member per chunk, on a
1.6 B-parameter network.

*Mitra* is the cleanest of the four to cache and the most expensive to
hold. Its row attention is one-directional — support attends to support,
query attends to support — and everything else in a layer acts on one row
at a time, so the support set is never told the query rows exist. A
cached prediction there is **bit-identical**, not merely equivalent: the
query rows are their own tensor either way, so not one operation on them
changes shape. The cost is memory. Mitra has no multi-query path, so
every head has to be kept: `2 * n_layers * F * S * dim` floats, where `F`
counts the target column. `print()` on the cache reports the figure.

Either way it is off by default.

What the cache spares is the *forward pass* over the training rows. It
used to spare only that: the member's preprocessing was still fitted, and
the training matrix still uploaded, once per member per chunk, before
being handed to a network that ignores it. Both are now skipped when a
cache is supplied, which is what makes `kv_cache = TRUE` cost what it
claims to.

**`save_peak_memory_factor`** splits each sublayer's work into that many
chunks, shrinking the temporaries each one materialises. It reorganises
work that was already independent, so the only cost is a little loop
overhead: bit-identical on v2.6, and on v3 not quite — chunking changes
the batch shapes the attention kernel sees — but the difference stays at
float32 noise, ~1e-6 of the logits' own scale.

What it cannot do is change how the peak *grows*. The tensor it slices is
a sublayer's input, not its output, so one copy of the state survives
every factor you pick. Measured on v3 at 50 features, the marginal cost
of a context row goes from 1.59 MB to 0.549 MB at a factor of 8 — a
useful 2.9×, and still a straight line.

**`row_chunk_size` and `col_chunk_size`** (v3 only) are the ones that
bend it. They drive the cell embedding, distribution embedder and column
aggregator a chunk of rows at a time, so the `(rows, columns, embedding)`
tensor — 51 KB per row at 100 features — is never resident whole. What
survives the loop is a quarter of one column's width per row. The column
chunk bounds the pre-pass that computes the distribution embedder's
inducing summaries, which is the one part a row-chunked pass cannot do
for itself.

**TabICL has the row half of the same mechanism**, as `row_chunk_size`,
off by default. Its column stage summarises the labelled rows into a
fixed set of inducing points, after which every row's path through the
column and row stages depends on nothing but itself — the same
row-independent prefix v3 has, and the reason the design transfers. At
Measured, three repeats per point, fresh process each:

| TabICL, 90 features | plain | `row_chunk_size = 2048` |
|---|---|---|
| 6,426 rows (the fold) | 24.5 GB | **12.0 GB** |
| 12,000 rows | 37.6 GB | **19.3 GB** |
| 20,000 rows | — | 35.5 GB |

`col_chunk_size` bounds the other half. A row-chunked pass cannot start
until the column stage's summaries exist, and building them holds a
tensor as wide as the whole table — so on a wide table the summaries,
not the rows, are the ceiling. TabICL at 12,000 × 300:

| | peak |
|---|---|
| no chunking | *does not finish* |
| `row_chunk_size = 2048` | 36.9 GB, 47 s |
| plus `col_chunk_size = 8` | **29.8 GB, 28 s** |

TabFM is the clearest case, because there the pre-pass *is* the peak:
its cumulative cost at 4,000 × 90 goes 12.9 GB after the cell embedder
to 29.2 GB after the column stage. Chunking rows alone buys 6% and still
cannot finish 8,000 × 90; adding `col_chunk_size = 8` finishes it, at the
same peak the failing run reached. Same peak, one dies and one does not —
which is what a ceiling looks like from underneath.

`save_peak_memory_factor` reaches TabICL's in-context stage too, and
measurably buys nothing there — 1.5% at 12,000 rows, and at the sweep
point it is 20% *worse*, because the loop's own temporaries cost more
than the transient it removes. That is the fused attention kernel again:
the in-context stage attends across every row, but unmasked, so it never
materialises the scores and there is nothing large to chunk. The knob is
there for symmetry with the other backends, and the estimator models the
penalty rather than promising a saving.

Both v3 knobs default to the checkpoint's own values, 2048 and 4, which
is what the Python reference does on this architecture and nowhere else.
`NULL` runs every row in one pass. Unlike `save_peak_memory_factor` this is not
bit-identical — on the package's 2,664-row fixture it moves the logits by
1.4e-5 of their own scale, which is *less* than the reference's own
chunked pass moves them.

The two mechanisms attack different terms and neither substitutes for the
other. Measured, v3 at 50 features, peak resident memory:

| n_context | plain | chunked | chunked + the collect below |
|---|---|---|---|
| 8,000 | 17.8 GB | 9.4 GB | **5.6 GB** |
| 16,000 | 21.2 GB | 12.7 GB | **5.8 GB** |

At 32,768 context rows × 120 features the whole `fit()` + `predict()`
measures 18.1 GB, where before any of this the same machine could not
finish 8,192 × 120 at all.

**Letting go between layers.** R torch frees a tensor when R's collector
runs, not when the last reference leaves scope, so a deep stack
accumulates every block's intermediates inside one forward pass — v3's
24-block ICL stack held 3.3 GB to carry a 31 MB state. Every backend's
layer loop now collects between blocks once the tensor it carries is
worth it (4 MB, which the measurements place), which costs 1–7% on small
tables and 2.3–2.9× less memory on large ones:

| backend | dims | without | with |
|---|---|---|---|
| `tabicl` | 6,000 × 50 | 21.3 GB | **12.3 GB**, and 21% faster |
| `tabpfn` (v2.5) | 6,000 × 50 | 15.3 GB | **10.9 GB** |
| `tabfm` | 500 × 20 | 17.4 GB | **12.9 GB** |
| `mitra` | 1,000 × 50 | 17.4 GB | **12.8 GB** |
| `tabpfn26` | 6,000 × 50 | 8.9 GB | **7.8 GB** |

It pays when the peak is accumulation and is a wash when the peak is
genuinely live data — Mitra at 2,000 × 50, whose 36 GB really is resident
2-D attention state, comes out 2.7% worse. `options(tabfound.collect_between_layers =)`
takes `"auto"` (default), `TRUE` or `FALSE`.

The same thing happens one level up, where it had been missed: an
ensemble runs the whole stack per member, and a chunked prediction runs
the whole ensemble per chunk, so both loops were holding every iteration's
dead intermediates while the next one allocated. Every member loop and
every chunk loop now collects too. There is nothing to weigh there — the
iteration that just finished was a full forward pass over the training
context, so a millisecond of collection is never the deciding term — and
the same option switches all of it off.

**Fitting in the right loop.** A member's preprocessing — the quantile
transformer that sorts every column of the training matrix, the ordinal
encoder, the SVD — is fitted on the training rows and depends on nothing
else, but it used to be fitted *inside* the chunk loop: `n_members ×
n_chunks` fits where `n_members` would do. It is now fitted once per
`predict()` call and replayed on each chunk, which is bit-identical
(verified against the parity fixtures and end-to-end on a real
checkpoint) and worth 9–20% at three chunks, more as chunks multiply.
The member views themselves are built one at a time rather than all up
front: for TabICL's eight members on an 8,000 × 30 table that is 15.5 MB
of R matrices down to 1.9 MB, and TabFM has thirty-two of them.

**Mitra is the case all of this was worth doing for.** It attends across
rows *and* columns, so its activation carries every row at full embedding
width on both axes — the steepest curve in the package, and the reason
the preflight exists. All four of its sublayers are independent across
rows, so `save_peak_memory_factor` applies to it too:

| Mitra, 4,000 × 50 | peak |
|---|---|
| no factor | 40.8 GB |
| 2 | 33.6 GB |
| 4 | 24.3 GB |
| 8 | 21.4 GB |
| 32 | **14.8 GB** |

And at the fold from the preflight hand-off — 6,426 × 90, recorded there
as something Mitra "genuinely cannot do" on this machine — a factor of 32
completes in **30.9 GB**.

It is not bit-identical the way v2.6's is: the observation attention puts
rows in the sequence position, so a chunk changes the query length the
kernel sees, where v2.6 splits a fold of leading dimensions and leaves
every attention's own shape alone. The gap is 5.4e-7 of the logits'
scale, an order of magnitude tighter than v3's stage chunking.

**TabPFN v2.5 has it too** — it used to be a v2.6-and-newer path, and
v2.5's layer turns out to have the same alternating shape and the same
independent folds. At the 6,426 × 90 fold it takes the peak from
**15.2 GB to 7.5 GB** at a factor of 2, reproducibly and with identical
probabilities. Larger factors are slightly *worse* than 2 there, which is
the argument for sweeping rather than assuming more is better.

This is the tightest of the four numerically: the chunking splits folds
of *leading* dimensions only, so no attention's own sequence length
changes — only its batch size, and only when the fold does not divide
evenly. Exact where it divides, ~1.6e-7 otherwise.

**What the factor buys is now measured, not assumed.** The estimator used
to model it as `1 + (act_copies - 1) / k` — one copy survives, the rest
divide — which promised 24× at `k = 32` where Mitra delivers 2.8×, and
that is an *under*-estimate, the one direction a guard must never err in.
`inst/memory/calibrate.R --spmf` sweeps the factor and fits the share it
cannot reach: 0.64 for Mitra, 0.85 for v2.6, and **1.00 for v3**, whose
stage chunking has already taken the transient away. An unswept backend
defaults to 1 — no promised saving at all.

### Threads

libtorch runs one intra-op thread per core, which is right for one model
in one process and wrong inside a worker pool: `k` R workers each
spawning `n_cores` threads oversubscribe the machine `k`-fold, and the
run gets *slower* the more workers you add. `tabfound_threads(n)` sets
the count for the calling process — the rule of thumb is
`detectCores() %/% k`, called inside each worker.

Set it before the first forward pass. libtorch's native backend refuses
both thread counts once its parallel region has started and says so on
stderr from C++, not through an R condition, so a late call looks as
though it worked; `tabfound_threads()` with no argument reads the value
back.

The advice underneath the knob is not to fork at all. Chained-equations
chains and ensemble members are already sequential calls into a
multi-threaded library, so the parallelism is better left to torch than
taken from it — and forking a process that has already initialised
libtorch leaves the child with a thread pool it cannot use, so a pool
must be built before the first torch call, or with
`future::plan(multisession)` rather than `multicore`.

### Knowing before you run

The knobs above only help if you find out in time. Running out of memory
inside libtorch does not raise an R error — it kills the process, with no
condition, no traceback and no output — so `fit()` and `predict()`
estimate the peak first and say something while there is still a session
to say it to.

```r
estimate_peak_memory("mitra-classifier",
                     n_context = 6426, n_query = 714, n_features = 90)
#> memory preflight <mitra> on "cpu"
#> 6426 context x 90 features, 714 to predict
#> * weights 303 MB (artifact file size)
#> * persistent 9 MB
#> * transient 136.3 GB, largest stage "attention across rows"
#> > peak 170.9 GB
#> i available 36.7 GB of 51.5 GB
#> x verdict: EXCEEDS
#> i Reduce `n_context`: the transient peak grows with the context...
#> i Mitra attends across rows *and* columns, so its activation is the
#>   steepest in the package; its published target is small tables.
#>   TabPFN v3 or TabICL will cost far less at these dimensions.
```

It works from a `config.json` alone — no weights, so the check can run
before a 6.5 GB download — and takes the same backend arguments the
predictors do, which is how you ask what a knob would buy:

```r
estimate_peak_memory(clf, 6426, 714, 90, kv_cache = TRUE)
estimate_peak_memory(clf, 6426, 714, 90, available = 8e9)  # a 16 GB laptop
```

The Python reference answers the same question by halving its chunk size
and retrying when an allocation fails. R cannot: the failure kills the
process, which is the premise this whole section rests on. So the chunk
size has to be chosen *before* the run, and `suggest_chunk_sizes()` is
what chooses it — searching down from the checkpoint's own default and
returning the largest that still fits, or saying plainly that none does.

```r
suggest_chunk_sizes("tabpfn-v3-classifier", n_context = 50000,
                    n_features = 50, available = 32e9)
#> $row_chunk_size [1] 1024
#> $peak_bytes     [1] 1.31e+10
#> $verdict        [1] "ok"
#> $feasible       [1] TRUE
#> $note           NULL
```

When nothing fits it says so — `feasible = FALSE` and a note pointing at
`n_context`, rather than a chunk size that would only fail later.


Six things it is careful about, because getting them wrong would make
the number useless. A backend's stages have very different peaks, so the
transient term is the largest one, not their sum. `kv_cache` does not
save memory, it *moves* it — out of the transient term, into one cache
per member held for the whole `predict()` call. An ensemble raises the
peak, but nothing like `n_estimators ×`: members run sequentially, so the
live set is one member's, while the resident high-water mark climbs
roughly with `log2(n_estimators)` as the allocator takes bigger arenas and
does not give them back — measured, and the opposite of what the design
brief assumed.

The fourth is the summary pre-pass. A row-chunked forward cannot start
until the column stage's summaries exist, and building them reads every
context row — so on a chunked run it is often the peak, it answers to
`col_chunk_size` rather than `row_chunk_size`, and it is charged its own
measured copy count rather than the forward's. Reusing the forward's put
TabICL at 112 GB against 36.9 GB measured.

The last two are about what attention actually costs. Torch's fused
scaled-dot-product attention takes a memory-efficient kernel when no mask
is passed and **never materialises the `(n, n)` score matrix**: at ICL
shapes, peak resident memory goes 353 → 382 → 448 MB across n = 4,000,
8,000 and 16,000, where the scores alone would be 488 MB, 1.9 GB and
7.8 GB, and a hand-written attention at n = 8,000 takes 6.2 GB. Only
TabFM pays for one, because it is the only backend that restricts context
by masking rather than slicing. And the estimate has a floor as well as a
slope, because the measurements have two regimes: above a gigabyte or so
the peak is the activation and tracks the model, and below it the peak is
the process and does not fall with the input — TabPFN v2.5 at 8 features
measures 2.1 GB at 800 rows and 1.1 GB at 1,600, reproducibly.

The verdict is taken against memory **available now**, not installed. The
run that prompted all this fitted comfortably alone on a 48 GB machine
and was killed by the OS when another process took 7 GB; nothing about
the sticker number could tell those two apart.

`options(tabfound.memory_guard =)` sets what happens when a real call
does not fit: `"warn"` (default), `"error"` to refuse, `"off"` for
silence. The guard is never itself the reason a run fails — an
uncalibrated device, an unknown backend or an unprobeable machine all
mean silence rather than a guess — and it says a given thing once, so
`mi()`'s chained-equations loop does not repeat itself.

The constants behind the estimate are **measured**, by
`inst/memory/calibrate.R`, on a grid of real fit/predict runs whose peak
resident memory was sampled from a watching process. They are set to err
high: for a guard, a false "tight" costs a warning and a false "ok" costs
the session. On the calibration grid no estimate falls below its
measurement, and where the activation is what was measured the fit stays
within 1.5× above it.

Three caveats, all in `inst/memory/README.md` in full. The figures are
larger than intuition suggests — a 6,426 × 90 TabICL fold measures
25.7 GB with a single ensemble member — because what is being predicted
is *resident* memory, which includes what libtorch's allocator has taken
and not returned, and that is what the OS kills on. Nothing beyond 32,768
context rows has been measured, so the envelope past that is
extrapolation. And Mitra and TabFM are poorly constrained: they are big
enough that on a 48 GB machine every informative measurement is at the
ceiling, so their constants extrapolate from a corner — TabFM's grid
stops at 1,600 rows. Re-run the harness on a larger machine before
trusting them.

## Package layout

```
R/core-*.R            device, artifact resolution, download + convert, state-dict loader, registry
R/core-memory.R       memory preflight: availability probe, peak estimator, the fit/predict guard
R/api.R               tabular_classifier() / tabular_regressor() + S3 methods
R/nn-*.R              shared layers: attention variants, norms, RoPE, blocks, MLPs
R/prep-transforms.R   TabPFN's own column transforms (squashing, quantile, SVD, fingerprint)
R/prep-sklearn.R      the stock sklearn transformers the other three wrappers chain
R/prep-ensemble.R     the ensemble generators + Mitra's preprocessor
R/py-random.R         CPython's random.Random, which decides every ensemble member
R/dist-*.R            output heads (TabPFN's bar distribution, TabICL's quantile distribution)
R/mi*.R               multiple imputation + the mice / Amelia bridges
R/syn*.R              sequential synthesis + the synthpop bridges
R/backend-*-layers.R  family-specific blocks where conventions diverge
R/backend-*.R         one model family each (`backend-tabpfn3.R` is TabPFN v3)
inst/memory/          peak-memory calibration harness, measurements + fitted coefficients
inst/parity/          the parity harness
inst/python/          one-time checkpoint conversion
```

Adding a backend means one `register_backend()` call plus its layers —
no changes to the core. See `?register_backend`.

## Known gaps

**TabFM.** The network and the sklearn wrapper's default preset are
both ported and verified: the preprocessing pipeline, the 32-member
ensemble (feature permutations, class shifts, per-member normalisation)
and the `cat_mask` that routes a declared categorical column's cells
through a separate Fourier basis. Not ported: the heavier
`TabFMClassifier.ensemble()` preset — feature crosses, SVD features,
NNLS ensemble weights and probability calibration, which together need
out-of-fold cross-fitting — and `TransformToNumerical`'s datetime
handling. Asking for `n_feature_crosses` or `n_svd_features` errors
rather than silently ignoring them. Its regression head emits a point
estimate, so there is no `$predict_quantiles()`. `kv_cache = TRUE`
conditions on the training rows once per `predict()` instead of once per
chunk per member, which on this network is most of the wall clock.

Practical notes: the checkpoint is ~6.5 GB per task and loads in about
6 s; a forward pass on 60 rows x 4 features takes ~0.5 s on CPU, and the
default ensemble runs 32 of them. Weights are non-commercial licensed.

**TabICL.** Network and wrapper both ported and verified, end to end
against `TabICLClassifier` / `TabICLRegressor`: mean imputation, the
preprocessing pipeline, the 8-member ensemble with Latin-square feature
permutations and shifted class labels, logit averaging, and the quantile
distribution the regressor reads its mean, median and quantiles from,
plus a K/V cache (`kv_cache = TRUE`) that conditions on the labelled rows
once per `predict()`. Not ported: the `TableVectorizer`-style DataFrame
typing (R knows its own column types, so `tabfound()` does not need to
guess) and the many-class hierarchical path for problems with more
classes than the checkpoint supports. The checkpoint is 110 MB and a
forward pass on 60 rows takes ~0.07 s on CPU.

Target standardization is ported for both TabFM and TabICL: their
regressor networks operate on standardized targets and their wrappers
fit a `StandardScaler` on `y`, so `$fit()` does the same and predictions
are inverse-transformed back to the original scale. Skipping this does
not error, it just returns predictions on the wrong scale — on iris it
was the difference between an RMSE of 1.82 and 0.35.

**Mitra.** Network and preprocessor both ported and verified. Not
ported: fine-tuning, which AutoGluon does by default
(`fine_tune_steps = 50`) and which this package, being inference-only,
cannot do — so predictions match a `MitraClassifier` with
`fine_tune=False`, not the out-of-the-box one. Its raw network still has
the sharpest edge in the package — a single `NA` in a column makes that
column's quantiles all-`NaN`, which buckets every value to zero, which
zeroes the column's variance, which the zero-variance guard flattens, so
one missing value silently deletes the entire feature with no `NaN` in
the output and no error — but nothing reachable through the API hits it,
because the predictor mean-imputes first exactly as AutoGluon's
preprocessor does. One more thing is knowingly *not* faithful: the
per-column random sign flips are drawn from a seedable R generator here,
where the reference takes them from NumPy's global generator and never
seeds it, so its own flips differ between two of its own runs.

**TabPFN.**

- `outlier_removal_std` is not implemented. It is a no-op
  on the current fixtures (the classifier default is 12σ, the regressor
  passes `NULL`), but data with extreme outliers will diverge.
- `predict(type = "sample")` is ensemble-aware: it draws from the
  members' averaged bucket probabilities — the same pseudo-logits the
  reference hands its head for `mean()` and `icdf()` — so a draw agrees
  with the quantiles reported beside it. Without ensemble configs it is
  one forward pass, as before.
- A data frame handed to `fit()` on the matrix interface is encoded the
  same way `tabfound()` encodes one (factors to ordinal codes, dates to
  numbers), and anything that cannot be encoded is refused rather than
  coerced. What that path does *not* do is declare which columns are
  categorical — it warns and names them; `tabfound()` declares them from
  the frame. See the formula-interface section.
- One-hot categorical encoding is not implemented. Its width depends on
  the data, which a member's shuffle permutation cannot be sized ahead
  of; no released checkpoint asks for it.
- Text columns are not supported. The reference has a `TEXT` modality
  for high-cardinality strings; here a character column becomes a factor
  and is declared categorical, which is refused above 30 levels.
- The fused SDPA kernel is reached through one wrapper (`sdpa()`) rather
  than eleven `torch:::` call sites. It still prefers torch's unexported
  `torch_scaled_dot_product_attention`, because that is what makes the
  float32 rounding match the reference — but the lookup is soft, and a
  pure public-API fallback takes over when it is absent, agreeing to
  ~1e-6. `options(tabfound.sdpa = "r")` forces the fallback, which is
  how the agreement is tested.

**TabPFN v2.6.**

- A member whose predictor count exceeds its per-estimator budget (680
  and 500 for v2.6's two member types) errors rather than subsampling
  features the way the reference would.
- The KV cache freezes two preprocessing masks on the training rows that
  an ordinary pass fits over train and test together, so on some data it
  is a different prediction rather than a cheaper one. It is opt-in for
  that reason; see below.
- One dataset per call: the reference's batch dimension is not exercised
  by any caller here, and a batch greater than one is rejected rather
  than run through the untested `select_features()` padding path.
- Predictors are capped at 6000 columns (2000 feature groups of 3).
  Beyond that the reference falls back to a seeded random draw for the
  column embeddings, which is device-dependent and cannot be reproduced;
  that case errors rather than guessing.

**TabPFN v3.**

- The row and column chunking of `_stages_0_to_2` **is** reproduced, as
  `row_chunk_size` / `col_chunk_size`, defaulting to the checkpoint's own
  2048 and 4 exactly as the reference does. Three things about it are
  worth knowing. It is not bit-identical — it changes the batch shapes
  the attention kernel sees, and on the package's 2,664-row fixture moves
  the logits by 1.4e-5 of their own scale, less than the reference's own
  chunked pass moves them. The reference's OOM fallback, which halves the
  chunk size and retries, is *not* ported and cannot be: a libtorch
  allocation failure kills the R process outright, so there is nothing to
  catch and nothing to retry from — `suggest_chunk_sizes()` chooses the
  size beforehand instead. And it does not touch the in-context stage,
  which stays quadratic in the context row count: chunking makes a large
  context a wall-clock decision rather than a dead session, and the
  reachable million-row regime is a million rows *to predict* against a
  bounded context, via `kv_cache = TRUE`.
- One dataset per call, as for v2.6: the reference's batch dimension is
  not exercised by any caller here and a batch greater than one is
  rejected.
- No ensemble config generator of its own. `ensemble_configs_dir` and the
  shared member pipeline work as they do for v2, but v3's own
  `PREPROCESS_TRANSFORMS` defaults (squashing scaler + quantile, SVD
  quarter-components, 200/500 features per estimator) have to come from a
  Python dump rather than being generated natively.
- The specialised checkpoints' `FEATURE_SUBSAMPLING_METHOD` settings
  (`"random"`, `"balanced"`) are not implemented; a member whose
  predictor count exceeds its budget errors, as it does for v2.6.
- `torch.compile`, int8 KV-cache quantisation and the MLX/FlashAttention-3
  backends have no counterpart. They are throughput paths, not different
  arithmetic.

## License

MIT for this package. Model weights carry their own licenses — notably
Google's TabFM weights are released under a **non-commercial** license,
separate from its Apache-2.0 source.
