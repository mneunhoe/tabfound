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

Options: `--backend` (`lm`, `tabpfn`, `tabicl`), `--reps`, `--n`, `--m`,
`--maxit`, `--proper`, `--seed`, `--save`. Real backends are located
through the same environment variables the parity harness uses
(`TABFOUND_TABPFN_CLF_DIR`, `TABFOUND_TABPFN_REG_DIR`, and the `TABICL`
equivalents).

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

**`x1`** — continuous, ~42% missing, truth 2.0:

| arm | bias | coverage | CI width |
|---|---|---|---|
| FULL | −0.001 | 0.945 | 0.229 |
| CC | **−0.111** | **0.780** | 0.355 |
| MI, `lm` | +0.003 | 0.935 | 0.283 |
| MI, `tabpfn` | −0.008 | 0.965 | 0.314 |
| MI, `tabicl` | **−0.061** | **0.860** | 0.314 |
| MI, `mice` PMM | +0.013 | 0.945 | 0.311 |

**`g`** — binary factor, ~23% missing, truth 1.5:

| arm | bias | coverage |
|---|---|---|
| FULL | +0.002 | 0.955 |
| CC | **−0.078** | 0.930 |
| MI, `lm` | +0.003 | 0.945 |
| MI, `tabpfn` | +0.011 | 0.940 |
| MI, `tabicl` | **−0.081** | 0.925 |
| MI, `mice` PMM | +0.013 | 0.955 |

FULL, CC and PMM are identical across the three runs by construction --
they do not depend on which backend was loaded -- so the only row that
moves is `MI`. That is the point of the pairing.

Two conclusions, one of them unwelcome.

**TabPFN works.** It removes 93% of the complete-case bias on `x1` and
86% on `g`, landing within 0.011 of truth on every coefficient with
coverage at or above nominal — indistinguishable from the correctly
specified `lm` imputer, and from `mice`'s purpose-built PMM, despite
being given nothing about the data-generating process.

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
hypothesis, not a finding.

## TabPFN vs PMM, paired

The two arms worth choosing between. Because they share a data set
replication for replication, `mi-mar-<backend>-reps.csv` supports paired
tests, which are far sharper than eyeballing the summary columns —
McNemar on the coverage indicators, a paired *t* on the estimates and on
the interval widths:

| coef | coverage (TabPFN / PMM) | McNemar *p* | width difference | *p* |
|---|---|---|---|---|
| x1 | 0.965 / 0.945 | 0.34 | +0.003 | 0.70 |
| x2 | 0.900 / 0.950 | **0.021** | **−0.023** | **0.001** |
| g  | 0.940 / 0.955 | 0.61 | **−0.078** | <0.0001 |

**Bias: a tie in practice.** TabPFN's estimate sits 0.021 below PMM's on
`x1` — a real paired difference (*p* < 0.001) that happens to favour
TabPFN, since truth is 2 and the two biases are −0.008 and +0.013. On `g`
they are indistinguishable. Both are unbiased at any scale a user cares
about.

**Intervals: TabPFN's are systematically narrower**, significantly so on
`x2` and `g`. Narrower is not automatically worse — on `g` it costs
nothing, coverage stays at 0.940. On `x2` it does bite: 0.900 against
PMM's 0.950.

Worth stating precisely, because it is close to the line: the `x2`
coverage test is nominally significant at *p* = 0.021 but would not
survive Bonferroni across the three coefficients (threshold 0.017). What
makes it persuasive is not that test alone but that it agrees with the
width result at the same coefficient (*p* = 0.001) and points the same
way. Read it as good evidence, not proof.

`x2` is the *fully observed* covariate — its coefficient is disturbed
only indirectly, through the imputation of `x1` and `g` — so this is
TabPFN mildly understating between-imputation variance where the
imputation's influence is second-hand. If you want intervals you do not
have to think about, PMM is the safer default today; if you want bias
removal, they are interchangeable.

## Properness: the correction that makes things worse

The undercoverage above has a textbook fix, and it does not work here.

A PFN draws each query row independently given a fixed context, so the
`m` chains of an imputation differ only in the noise of the draw. No
parameter uncertainty enters, which is exactly what makes an imputer
improper under Rubin's rules — `tabfound_syn()` says so in its own
documentation and bootstraps the context by default for that reason.
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
non-default regime tag themselves: `-proper`, `-bayes`.
