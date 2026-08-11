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
`--maxit`, `--seed`, `--save`. Real backends are located through the same
environment variables the parity harness uses
(`TABFOUND_TABPFN_CLF_DIR`, `TABFOUND_TABPFN_REG_DIR`, and the `TABICL`
equivalents).

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

200 replications, `n = 400`, `m = 5`, `maxit = 3`, seed 20260807. All
arms see the same data sets, replication for replication, so the columns
are directly comparable. Monte Carlo error is about 0.006–0.010 on a
bias and 0.02 on a coverage.

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

## Reading the numbers

`bias` is against the known truth; `coverage` is the share of replication
CIs containing it; `ci_width` is the mean interval width. Monte Carlo
error on a bias estimate is `emp_sd / sqrt(reps)` — at 200 replications
that is around 0.005–0.010 here, so differences smaller than ~0.02 are
noise. Coverage carries about ±0.02.

Results land in `inst/simulation/results/mi-mar-<backend>.csv` with
`--save`.
