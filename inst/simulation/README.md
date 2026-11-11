# MI simulation harness

`tabfound_impute()` is the one part of the package with no Python
reference to be checked against — it is a *use* of the models, not a port
of anything, so `inst/parity/` has nothing to say about it. This is what
stands in for it: a simulation where the right answer is known by
construction.

```bash
Rscript inst/simulation/run-mi-sim.R --backend=lm --reps=200
Rscript inst/simulation/run-mi-sim.R --backend=tabpfn --reps=200 --save
```

Options: `--backend` (`lm`, `tabpfn`, `tabicl`, `kumo`), `--reps`, `--n`, `--m`,
`--maxit`, `--proper`, `--seed`, `--temperature`, `--save`. Real backends are located
through the same environment variables the parity harness uses
(`TABFOUND_TABPFN_CLF_DIR`, `TABFOUND_TABPFN_REG_DIR`, and the `TABICL`
and `KUMO` equivalents).

`paired-compare.R` reads a saved `-reps.csv` and compares two arms
replication by replication — McNemar on the coverage indicators, paired
*t* on the estimates and the interval widths. All arms see the same data
sets, so the pairing is far sharper than comparing summary columns:

```bash
Rscript inst/simulation/paired-compare.R results/mi-mar-tabpfn-reps.csv
```

## The design

The textbook MAR demonstration. `x1` (continuous) and `g` (a binary
factor) go missing with probability depending on `y`, which is fully
observed. That is missing *at random* given the data — and it is exactly
the case a complete-case analysis gets wrong, because dropping those rows
selects on the outcome.

```
x2 ~ N(0, 1)
x1 = 0.5 x2 + sqrt(0.75) N(0, 1)          cor(x1, x2) = 0.5
g  ~ Bernoulli(0.4)
y  = 1 + 2 x1 - 1 x2 + 1.5 (g == "b") + N(0, 1)

P(x1 missing) = logistic(-0.4 + 0.9 z),   z = scale(y)     ~42%
P(g  missing) = logistic(-1.2 + 0.8 z)                     ~23%
```

`x1` and `x2` are correlated on purpose: an imputation model that ignores
the other covariates is not merely inefficient, it is wrong, which is
what makes the chained part of chained equations do any work. `g` is
there to exercise the classifier head, which is otherwise untested by a
purely continuous design.

Four arms are fitted per replication, all analysing `y ~ x1 + x2 + g`:

| arm | what it is | what it should show |
|---|---|---|
| `FULL` | before deletion | the benchmark: unbiased, coverage ~0.95 |
| `CC` | complete cases | **biased**, coverage well below 0.95 |
| `MI` | `tabfound_impute()` | back at `FULL`'s estimate, wider intervals |
| `PMM` | `mice` defaults | an external yardstick, not a target to beat |

If `CC` is not visibly biased, the mechanism is too weak and the run
proves nothing — read that row first.

## `--backend=lm` comes first

The `lm` arm is a correctly specified imputation model (least squares for
`x1`, logistic regression for `g`) wearing a `tabfound_model`'s object
shape. It needs no weights and no torch, and runs 200 replications in
about 13 seconds.

It tests the **loop**, not any model: a chain that conditioned on the
wrong rows, drew means instead of samples, or mishandled the categorical
path would break here too, and much more legibly than it would under a
600 MB network. Only once the `lm` arm is clean does a foundation-model
arm say anything about the model.

The arm earns its keep. An earlier version used a one-vs-rest linear
probability model for `g` and left a residual bias of −0.05 on that
coefficient — misspecification of the reference imputer, not a defect in
the loop, but it would have been easy to read the other way round.
Replacing it with the logistic fit the DGP actually implies moved that to
+0.009, which is zero within Monte Carlo error.

## Results

200 replications, `n = 400`, `m = 5`, `maxit = 3`, seed 20260807, at the
package defaults (`proper = FALSE` — see *Properness* below for what
happens when that is turned on). All arms see the same data sets,
replication for replication, so the columns are directly comparable.
Monte Carlo error is about 0.006–0.010 on a bias and 0.02 on a coverage.

TabPFN appears twice. `tabfound_models()` builds at
`softmax_temperature = 1` since #111; the runs before that drew at the
constructors' 0.9 (see *Temperature* below). The `tabicl` run and the
*Properness* section predate the change and were made at 0.9.

**`x1`** — continuous, ~42% missing, truth 2.0:

| arm | bias | coverage | CI width |
|---|---|---|---|
| FULL | −0.001 | 0.945 | 0.229 |
| CC | **−0.111** | **0.780** | 0.355 |
| MI, `lm` | +0.003 | 0.935 | 0.283 |
| MI, `tabpfn` (temperature 1, default) | −0.010 | 0.970 | 0.316 |
| MI, `tabpfn` (temperature 0.9) | −0.008 | 0.965 | 0.314 |
| MI, `tabicl` | **−0.061** | **0.860** | 0.314 |
| MI, `kumo` (small) | −0.001 | 0.935 | 0.300 |
| MI, `mice` PMM | +0.013 | 0.945 | 0.311 |

**`g`** — binary factor, ~23% missing, truth 1.5:

| arm | bias | coverage |
|---|---|---|
| FULL | +0.002 | 0.955 |
| CC | **−0.078** | 0.930 |
| MI, `lm` | +0.003 | 0.945 |
| MI, `tabpfn` (temperature 1, default) | −0.015 | 0.925 |
| MI, `tabpfn` (temperature 0.9) | +0.011 | 0.940 |
| MI, `tabicl` | **−0.081** | 0.925 |
| MI, `kumo` (small) | **−0.025** | 0.910 |
| MI, `mice` PMM | +0.013 | 0.955 |

FULL, CC and PMM are identical across the five runs by construction --
they do not depend on which backend was loaded -- so the only row that
moves is `MI`. That is the point of the pairing.

Two conclusions, one of them unwelcome.

**TabPFN works.** At the default temperature it removes 91% of the
complete-case bias on `x1` and 80% on `g`, landing within 0.015 of truth
on every coefficient (93% and 86% at 0.9, within 0.011) —
indistinguishable from the correctly specified `lm` imputer, and from
`mice`'s purpose-built PMM, despite being given nothing about the
data-generating process.

**TabICL does not.** It removes only 45% of the bias on `x1` (−0.061,
roughly 11 Monte Carlo standard errors from zero) and on `g` it removes
essentially none: −0.081 against complete-case's −0.078, meaning the
imputed categories carry almost no information the analysis can use.
Coverage on `x1` drops to 0.860. Anyone reaching for `tabfound_impute()`
should use the TabPFN backend.

The mechanism is not established. A single-data-set diagnostic found
TabICL's draws slightly more over-dispersed than TabPFN's relative to the
true conditional distribution (SD ratio 1.11 vs 1.05), which is the
direction that produces attenuation, but one data set is not enough to
attribute the effect and the two backends' imputation centres correlated
with the truth equally well (0.968 both). Treat this as an open question
rather than a diagnosis. It is consistent with `draw = "grid"` — inverting
a 999-level predicted quantile grid — being a worse route to a posterior
draw than TabPFN's native bar-distribution sampling, but that is a
hypothesis, not a finding — and the Kumo arm below, which takes the same
route and is unbiased on `x1`, now counts against it.

## Kumo Tabular (small)

`--backend=kumo`, the `small` checkpoints (classifier and regressor),
same 200 data sets. Kumo takes the same route TabICL does: it has no
native sampler, so `draw = "auto"` resolves to `"grid"` and continuous
cells are drawn by inverting its 999-level quantile grid. Categorical
cells come from the classifier's probabilities, as for every backend.

**`x1` (the grid route): it works.** Bias −0.001, coverage 0.935 —
the whole complete-case bias removed, where TabICL through the same
`.mi_icdf()` removes 45%. Whatever is wrong with the TabICL arm, it is
not that inverting a 999-level grid is a bad way to draw an imputation.
That narrows TabICL's open question to TabICL's own predictive
distributions.

**`g` (the classifier): a small residual bias.** −0.025 against
complete-case's −0.078, so about two-thirds of the bias removed; roughly
2.3 Monte Carlo standard errors from zero. Coverage 0.910. Kumo applies
no temperature, so the change in #111 does not touch this run.

Paired against PMM (`paired-compare.R results/mi-mar-kumo-reps.csv`):

| coef | coverage (Kumo / PMM) | McNemar *p* | estimate difference | *p* | width difference | *p* |
|---|---|---|---|---|---|---|
| x1 | 0.935 / 0.945 | 0.75 | −0.014 | <0.0001 | −0.011 | 0.040 |
| x2 | 0.915 / 0.950 | 0.070 | −0.013 | <0.0001 | **−0.031** | **<0.0001** |
| g  | 0.910 / 0.955 | 0.067 | **−0.038** | **<0.0001** | **−0.071** | **<0.0001** |

And against TabPFN's MI arm at the default temperature,
`mi-mar-tabpfn-temp1-reps.csv`. The runs share the seed, and their FULL
arms agree on all 600 estimates, so they saw the same data sets and pair
across files (stack the two MI arms under different names and run
`paired-compare.R` on the result):

| coef | coverage (Kumo / TabPFN) | McNemar *p* | estimate difference | *p* | width difference | *p* |
|---|---|---|---|---|---|---|
| x1 | 0.935 / 0.970 | 0.046 | +0.009 | <0.0001 | −0.016 | 0.019 |
| x2 | 0.915 / 0.905 | 0.77 | −0.009 | <0.0001 | −0.009 | 0.14 |
| g  | 0.910 / 0.925 | 0.55 | −0.010 | 0.015 | −0.005 | 0.67 |

No coverage difference survives Bonferroni (0.017) in either table, and
every estimate difference against TabPFN is 0.010 or less. Against
TabPFN at 0.9 the gap on `g` was 0.036; most of that was TabPFN's
sharpened classifier pushing its own `g` estimate up (see *Temperature*),
not Kumo pulling down. Against PMM, Kumo's `g` estimate still sits 0.038
lower, and its intervals are narrower than PMM's everywhere, like
TabPFN's.

**Reading it.** Kumo small is a usable imputer — far better than TabICL,
unbiased on the continuous variable, and within 0.010 of TabPFN on every
coefficient. TabPFN remains the default recommendation, on its smaller
`g` bias and the depth of evidence behind it. Only `small` was run: the
`medium` and `large` checkpoints are not validated here, and nothing
above says how they behave.

Reproduce with:

```bash
Rscript inst/simulation/run-mi-sim.R --backend=kumo --reps=200 --save
Rscript inst/simulation/paired-compare.R results/mi-mar-kumo-reps.csv
```

The run took about 66 minutes on CPU.

## Temperature

TabPFN, the TabICL classifier and TabFM divide their logits by
`softmax_temperature` before the softmax, and their constructors default
to 0.9, as the reference implementations do. That sharpens the
predictive distribution: right for point prediction, wrong for a draw.
Since #111 `tabfound_models()` builds at 1 instead, which reaches
`tabfound_impute()`, `mice.impute.tabfound()` and `tabfound_syn()`;
`tabular_classifier()` / `tabular_regressor()` keep 0.9.

**Calibration, on complete data from this design** (five-fold
out-of-fold predictions, TabPFN v2.5, `n = 400`):

- Regressor (from #111, 12 replications): mean predictive variance over
  out-of-fold mean squared error is 0.93 at 0.9 and 1.04 at 1, and the
  shortfall at 0.9 does not shrink with `n` (0.91–0.93 at `n = 1600`).
- Classifier (12 replications, `g` given `y, x1, x2`): regressing the
  outcome on the predicted logit gives a slope of 0.89 at 0.9 —
  overconfident — and 0.99 at 1. Out-of-fold log loss is lower at 1
  (0.4801 vs 0.4818, paired *p* < 1e-6).

**On the MAR simulation**, paired on the same 200 data sets (MI arm at 1
vs at 0.9):

| coef | coverage (1 / 0.9) | McNemar *p* | estimate difference | *p* | width ratio | mean abs. error (1 / 0.9) |
|---|---|---|---|---|---|---|
| x1 | 0.970 / 0.965 | 1.00 | −0.002 | <0.0001 | 1.006 | 0.0608 / 0.0610 |
| x2 | 0.905 / 0.900 | 1.00 | +0.001 | 0.0007 | 1.001 | 0.0664 / 0.0665 |
| g  | 0.925 / 0.940 | 0.37 | **−0.026** | <0.0001 | 1.021 | 0.1204 / 0.1206 |

The continuous imputations barely move: at this `n` the imputed `x1`'s
spread is a small part of what drives its coefficient. The binary one
does. Sharper category draws put the imputed `g` closer to the modal
class than the data warrant, which pushed its coefficient *up* at 0.9
(+0.011); untempered draws carry the honest misclassification and land
at −0.015. Neither is distinguishable from zero on its own (Monte Carlo
error ≈ 0.010), and the mean absolute error is the same to four digits.

So the simulation does not choose between the two at `n = 400`. The
default follows the calibration: draws should come from the predictive
distribution, and 1 is where both heads are calibrated. The issue's
argument that a fixed variance shortfall biases an imputed regressor by
a fixed amount — growing in standard errors like `sqrt(n)` — applies to
larger samples than this simulation runs.

Reproduce with:

```bash
Rscript inst/simulation/run-mi-sim.R --backend=tabpfn --reps=200 --temperature=1 --save
Rscript inst/simulation/paired-compare.R results/mi-mar-tabpfn-temp1-reps.csv
```

An explicit `--temperature` tags the output (`-temp1`); without it the
script runs at the package default and writes the plain name. That run
took about 17 minutes on CPU.

## TabPFN vs PMM, paired

The two arms worth choosing between. Because they share a data set
replication for replication, `mi-mar-<backend>-reps.csv` supports paired
tests, which are far sharper than eyeballing the summary columns —
McNemar on the coverage indicators, a paired *t* on the estimates and on
the interval widths. At the default temperature
(`mi-mar-tabpfn-temp1-reps.csv`):

| coef | coverage (TabPFN / PMM) | McNemar *p* | estimate difference | *p* | width difference | *p* |
|---|---|---|---|---|---|---|
| x1 | 0.970 / 0.945 | 0.18 | −0.023 | <0.0001 | +0.005 | 0.51 |
| x2 | 0.905 / 0.950 | 0.039 | −0.004 | 0.097 | **−0.022** | **0.002** |
| g  | 0.925 / 0.955 | 0.18 | **−0.028** | **<0.0001** | **−0.066** | **<0.0001** |

At 0.9 (`mi-mar-tabpfn-reps.csv`) the picture was the same, with `x2`
coverage at 0.900 (McNemar *p* = 0.021), `g` widths −0.078 and no
estimate difference on `g`.

**Bias: a tie in practice.** TabPFN's estimate sits 0.023 below PMM's on
`x1` — a real paired difference that happens to favour TabPFN, since
truth is 2 and the two biases are −0.010 and +0.013. On `g` it sits 0.028
below, biases −0.015 and +0.013: the same distance from truth on
opposite sides. Both are unbiased at any scale a user cares about.

**Intervals: TabPFN's are systematically narrower**, significantly so on
`x2` and `g`. Narrower is not automatically worse — on `g` coverage stays
at 0.925. On `x2` it does bite: 0.905 against PMM's 0.950.

Worth stating precisely, because it is close to the line: the `x2`
coverage test is nominally significant (*p* = 0.039 here, 0.021 at 0.9)
but would not survive Bonferroni across the three coefficients
(threshold 0.017). What makes it persuasive is not that test alone but
that it agrees with the width result at the same coefficient
(*p* = 0.002) and points the same way. Read it as good evidence, not
proof. Temperature 1 did not fix it: its `x2` intervals are only 0.1%
wider.

`x2` is the *fully observed* covariate — its coefficient is disturbed
only indirectly, through the imputation of `x1` and `g` — so this is
TabPFN mildly understating between-imputation variance where the
imputation's influence is second-hand. If you want intervals you do not
have to think about, PMM is the safer default today; if you want bias
removal, they are interchangeable.

## Properness: the correction that makes things worse

The undercoverage above has a textbook fix, and it does not work here.
(Measured at temperature 0.9, before #111.)

A PFN draws each query row independently given a fixed context, so the
`m` chains of an imputation differ only in the noise of the draw. No
parameter uncertainty enters, which is exactly what makes an imputer
improper under Rubin's rules — `tabfound_syn()` says so in its own
documentation and resamples the context by default for that reason
(by subsampling, since a bootstrap's duplicated rows make it copy real
records).
`tabfound_impute(proper = TRUE)` offers the same correction: resample the
observed rows once per imputation, then run the chain against that
context. mice's `norm.boot` and `polyreg.boot` are the same idea.

Run both ways on the same 200 data sets (`--proper=TRUE`, everything else
identical), the correction behaves completely differently for the two
imputers.

**The `lm` arm — a correctly specified parametric imputer:**

| coef | bias | CI width | coverage |
|---|---|---|---|
| x1 | +0.003 → +0.003 | 0.283 → 0.325 | 0.935 → 0.950 |
| x2 | −0.005 → −0.007 | 0.291 → 0.323 | **0.910 → 0.935** |
| g  | +0.003 → +0.008 | 0.538 → 0.650 | 0.945 → 0.965 |

Textbook: bias does not move, intervals widen 11–21%, coverage goes up.

**The `tabpfn` arm — the same correction, same data sets:**

| coef | bias | CI width | coverage |
|---|---|---|---|
| x1 | −0.008 → **−0.062** | 0.314 → 0.547 | 0.965 → 0.955 |
| x2 | +0.008 → **+0.041** | 0.308 → 0.436 | **0.900 → 0.955** |
| g  | +0.011 → **−0.069** | 0.559 → 0.792 | 0.940 → 0.955 |

Coverage is repaired and the estimate is wrecked. On `x1` the bias goes
from −0.008 to −0.062 against a complete-case bias of −0.111: better than
deleting the rows, but only just, where the default removes 93% of it.
Paired on the data set, the proper estimates are farther from truth on
every coefficient — mean absolute error 0.061 → 0.092 on `x1`
(*p* = 2e-12), 0.067 → 0.081 on `x2` (*p* = 2e-05), 0.121 → 0.138 on `g`
(*p* = 0.012) — and every interval is wider (all *p* < 1e-9). The
coverage that buys is bought with width, not with accuracy.

**Why.** A bootstrap of the context is not the same model with different
parameters. It is a worse model. Resampling *n* rows with replacement
leaves 63% of them distinct (50% under `proper = "bayes"`, whose
Dirichlet weights concentrate harder), so an in-context learner — whose
entire fit *is* its context — loses a third of its training data and
gains ties it never saw during pre-training. A parametric imputer barely
notices: least squares on 145 distinct rows with multiplicities is still
least squares.

A single-data-set diagnostic points the same way. Under `proper = TRUE`
TabPFN's draws for the deleted cells correlate less with the values that
were actually deleted (0.803 → 0.784 per draw, 0.889 → 0.875 for the
imputation mean) at essentially unchanged dispersion (1.00 → 1.02 of the
true conditional SD). Draws that are no wider but less informative are
the errors-in-variables condition, and attenuation of every coefficient
is what it predicts.

**What ships.** `proper = FALSE` is the default, against the theory and
with the numbers. `proper = TRUE` and `proper = "bayes"` are available
for anyone who needs nominal coverage more than a point estimate.

Two things this does *not* establish. It is one DGP at one sample size;
the balance could differ where the observed context is large enough that
losing 37% of it costs little. And it is not a licence to read the
default's intervals as correct — the `x2` undercoverage at 0.900 is real,
it is just cheaper than the cure. Whether properness can be had for an
in-context learner without degrading the context is open: every
resampling scheme duplicates rows.

Reproduce with:

```bash
Rscript inst/simulation/run-mi-sim.R --backend=tabpfn --reps=200 --proper=TRUE --save
Rscript inst/simulation/paired-compare.R results/mi-mar-tabpfn-proper-reps.csv
```

The variant that resamples per *sweep* rather than per imputation was
measured too (`mi-mar-tabpfn-proper-persweep.csv`) and lands in the same
place — bias −0.060, widths 0.604/0.445/0.710 — so this is not an
artefact of where the resample sits in the loop.

## Reading the numbers

`bias` is against the known truth; `coverage` is the share of replication
CIs containing it; `ci_width` is the mean interval width. Monte Carlo
error on a bias estimate is `emp_sd / sqrt(reps)` — at 200 replications
that is around 0.005–0.010 here, so differences smaller than ~0.02 are
noise. Coverage carries about ±0.02.

Results land in `inst/simulation/results/mi-mar-<backend>.csv` with
`--save`, plus `-reps.csv` with the per-replication estimates. Runs in a
non-default regime tag themselves: `-proper`, `-bayes`, and `-temp<value>`
when `--temperature` is given.
