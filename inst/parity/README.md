# Parity harness

Every backend in `tabfound` is a reimplementation of a published model.
The only way to know a reimplementation is right is to run the original
on the same bytes and compare. This directory does that.

## Layout

```
fixtures.R               generates the shared datasets (safetensors, float32)
fixtures/                the generated fixtures + manifest.json
tabpfn_reference.py      runs the PyPI `tabpfn` package, dumps reference outputs
tabpfn26_reference.py    runs the bare TabPFN v2.6 network, dumps its forward pass
tabpfn3_reference.py     runs the bare TabPFN v3 network, dumps its forward pass
tabfm_reference.py       runs the PyPI `tabfm` network, dumps per-stage outputs
tabicl_reference.py      runs the PyPI `tabicl` network, dumps per-stage outputs
mitra_reference.py       runs AutoGluon's `Tab2D`, dumps per-stage outputs
transforms_reference.py  pins the individual column transforms (no weights needed)
transforms/              its output -- small, and ships with the package
ensemble_reference.py    pins the wrapper layer: preprocessing + ensembling
ensemble/                its output -- also ships with the package
layers_reference.py      pins the shared transformer layers (no weights needed)
layers/                  its output -- also ships with the package
compare.R                staged comparison + tolerances
run-parity.R             driver
reference/               stored reference dumps (see "Size" below)
results/parity.csv       last run's numbers
```

Both sides read the *same* fixture file, so nothing can drift through a
difference in how the data was constructed.

## Running it

Replay the stored reference dumps (no Python needed):

```bash
Rscript inst/parity/run-parity.R tabpfn
```

Regenerate the reference first (needs the venv and the raw checkpoints):

```bash
Rscript inst/parity/run-parity.R tabpfn --regenerate
```

Configuration is by environment variable so nothing is hard-coded to one
checkout:

| variable | meaning |
|---|---|
| `TABFOUND_TABPFN_CLF_DIR` | converted classifier (`model.safetensors` + `config.json`) |
| `TABFOUND_TABPFN_REG_DIR` | converted regressor |
| `TABFOUND_TABPFN_CLF_CKPT` | raw `.ckpt`, only for `--regenerate` |
| `TABFOUND_TABPFN_REG_CKPT` | raw `.ckpt`, only for `--regenerate` |
| `TABFOUND_TABPFN26_CLF_DIR` / `_REG_DIR` | converted TabPFN v2.6 artifacts |
| `TABFOUND_TABPFN26_CLF_CKPT` / `_REG_CKPT` | raw `.ckpt`, only for `--regenerate` |
| `TABFOUND_TABPFN3_CLF_DIR` / `_REG_DIR` | converted TabPFN v3 artifacts |
| `TABFOUND_TABPFN3_CLF_CKPT` / `_REG_CKPT` | raw `.ckpt`, only for `--regenerate` |
| `TABFOUND_TABFM_DIR` | TabFM Hub snapshot root (holds `classification/` and `regression/`) |
| `TABFOUND_TABICL_CLF_DIR` / `_REG_DIR` | converted TabICL artifacts |
| `TABFOUND_TABICL_CLF_CKPT` / `_REG_CKPT` | raw `.ckpt`, only for `--regenerate` |
| `TABFOUND_MITRA_CLF_DIR` / `_REG_DIR` | Mitra Hub snapshots |
| `TABFOUND_MITRA_SHIM` | directory holding the `mitrapkg` shim, only for `--regenerate` |
| `TABFOUND_REF_PYTHON` | python with the reference packages (default `.venvs/ref/bin/python`) |

Set up the reference environment with:

```bash
uv venv --python 3.12 .venvs/ref && uv pip install --python .venvs/ref/bin/python torch safetensors numpy scikit-learn pandas tabpfn tabicl tabfm
```

### TabPFN v2.6

This one is graded differently, because v2.6 moved the preprocessing
inside the architecture: constant-column removal, NaN/Inf imputation and
flagging, standard scaling and feature-group normalisation all happen in
`forward()`. There is no longer a pipeline/network seam to bisect at, so
`tabpfn26_reference.py` runs the **bare network** rather than the
estimator, and the comparison splits along a different line:

* `forward:logits` — both networks fed the same bytes.
* `decode:probs` / `decode:mean` / `decode:quantiles` — R's decoding
  applied to *the reference's own* logits. Since the input is shared,
  only the decoding arithmetic can differ, and it comes out exact.
* `predict_proba:single` — end to end through the public API
  (classifier only; see below).

The regressor has no end-to-end stage: `tabular_regressor()` standardises
the target before the forward pass and the reference is fed the raw
target, so their logits are not comparable. Its decoding is covered by
`decode:mean` / `decode:quantiles` instead.

There are no `_nofp` twins for the bare-network run — it has no
fingerprint feature to switch off.

The **ensemble** is a separate reference tree
(`reference/tabpfn26-ensemble/`, rows labelled `ens:`) and is graded
exactly like v2.5's: the full estimator on both sides, `_nofp` twins
included, same stages. v2.6's members differ from v2.5's only in what
their configs ask for — `quantile_uni` as the primary transform, and for
the regressor a polynomial-feature expansion in front of everything else
— and the R side reads all of that out of the dump, so one comparison
covers both.

```bash
Rscript inst/parity/run-parity.R tabpfn26
```

### TabPFN v3

Graded exactly like v2.6's bare-network run, and for the same reason: v3
also keeps the NaN/Inf handling and the standard scaler inside
`forward()`, so `tabpfn3_reference.py` runs the network rather than the
estimator. Same six fixtures, same stages.

There is no separate ensemble tree. v3's estimator side is the machinery
already graded under `ens:` for v2.5 and v2.6 — the same member
pipeline, the same presets, the same target transforms — and the R
backend reuses the v2 predictors wholesale, so a third copy of that
comparison would test nothing new. What is v3-specific is the network,
and that is what this tree covers.

```bash
Rscript inst/parity/run-parity.R tabpfn3
```

### The cheap-inference paths

The KV cache is graded differently for the two TabPFN generations,
because it means different things to them.

For **v2 / v2.5** (`cache:vs-batching`) there is no reference number to
compare against -- the estimator exposes no bare cached forward -- but
there is a real bound. Everything this architecture fits comes from the
training rows alone, so the cache cannot change what is computed; only
float32 reduction order can move, and the uncached path already moves by
*something* when the same rows run in a different-sized batch. The
harness measures both and passes the row if the cache is either exact or
no worse than that. The `tol_rel` column for this stage is a ratio, not
an error.

For **v2.6** the cache genuinely can change the answer, so it is graded
against the reference's own cached run instead -- see below.

`save_peak_memory_factor` and the KV cache exist to make inference cost
less, not to change what it computes -- but only one of them fully
succeeds at that, and the harness is built to say which.

Both are graded from two angles:

* `forward:chunked` / `forward:cached` -- R's mechanism against the
  reference's run of the *same* mechanism. Inherits the plain forward's
  cross-implementation float32 gap, so it carries the plain forward's
  tolerance.
* `selfcheck:chunked` / `selfcheck:cached` -- R's mechanism against R's
  own plain forward, at tolerance **zero**. Nothing to hide in here.

`selfcheck:chunked` passes at exactly 0 on every fixture: chunking splits
work that was already independent.

v3 emits `selfcheck:chunked-fp32` / `selfcheck:cached-fp32` instead, at a
`max_scaled` tolerance of 1e-5 rather than zero. Not a weaker claim about
what the mechanisms *do* — v3's cache holds everything the training rows
contribute, so it is unconditionally the same computation — but both it
and the chunking change the batch shapes the attention kernel sees, and
in float32 that moves the last bits. Measured, it moves them by ~1e-6 of
the logits' own scale.

`selfcheck:cached` is emitted only where the *reference's* own cache is
exact. Where it is not, the harness emits `cache:shifts-prediction`
instead, carrying both sides' shift in its note. The cache fixes the
constant-column and informative-feature masks on the training rows, and
an ordinary pass fits them over train and test together; on data where
those differ, the cache is a different prediction by design. Which
fixtures that happens on is a property of the data, and R and the
reference agree on which ones.

### Categorical columns

Two fixtures (`clf_categorical`, `reg_categorical`) carry columns the
model should treat as categories rather than quantities: 3, 6 and 12
levels next to two numerics, one of which has missing values. Each is run
in two variants, because the reference decides categorical-ness two
different ways and both have to agree:

* `<fixture>_nofp` — the columns are **declared** via
  `categorical_features_indices`, so the ordinal encoder reaches all
  three.
* `<fixture>_auto_nofp` — nothing declared, so only the three-level
  column is **inferred** (the rule fires below four distinct values).

The `schema:categorical` stage compares the two sides' decisions directly,
before anything conditional on them runs. Without it a disagreement about
which columns are categorical would surface as a diffuse preprocessing
mismatch rather than as itself.

The category-code permutations the `_shuffled` encoders apply are drawn
from the reference's NumPy generator, so they are dumped per member as
`cat_mappings.safetensors` alongside the shuffle permutation.

### The Mitra shim

Mitra lives inside `autogluon.tabular`, whose dependency tree is large
and almost entirely irrelevant to a forward pass. Rather than install it,
`mitra_reference.py` imports a minimal package shim — four modules copied
verbatim from the AutoGluon repo:

```
mitrapkg/_internal/config/enums.py
mitrapkg/_internal/models/{base,embedding,tab2d}.py
```

plus `einx`. Point `TABFOUND_MITRA_SHIM` at the directory containing
`mitrapkg/`. Copying the reference's own source verbatim is the point:
the comparison is against the real implementation, not a paraphrase.

## The wrapper layer

Everything above this line grades a *network*. TabFM, TabICL and Mitra
are not used as networks: each ships an sklearn estimator that
preprocesses the table, builds several transformed views of it, runs the
network once per view and combines the results. That layer is where a
port can be exactly right about the architecture and still return the
wrong number, and until it was itself compared it was the largest
unchecked surface in this package.

`ensemble_reference.py` pins it, and like `transforms_reference.py` it
needs no checkpoint:

```bash
.venvs/ref/bin/python inst/parity/ensemble_reference.py \
    --out inst/parity/ensemble/ensemble.safetensors \
    --mitra-src <dir holding AutoGluon's mitra/_internal>
gzip -9 inst/parity/ensemble/ensemble.safetensors
```

`--mitra-src` is optional; without it Mitra's preprocessor is skipped and
the corresponding test skips with it. The output is replayed by
`test-py-random.R`, `test-prep-sklearn.R` and `test-prep-ensemble.R`.

Four things are graded, in the order one depends on the next:

| what | why it needs its own comparison |
|---|---|
| `random.Random` | every member's identity comes off this one stream |
| the sklearn estimators | `CustomStandardScaler`, `OutlierRemover`, `UniqueFeatureFilter`, `PowerTransformer`, `QuantileTransformer`, `RobustScaler`, and the pipeline that chains them |
| the ensemble generators | TabICL's and TabFM's, member for member — configuration *and* member matrix |
| Mitra's preprocessor | imputation, constant-column drop, target scaling, mirrors |

**The generator comparison is member for member, not distributional.**
Which permutation member 3 gets is decided by
`random.Random(random_state)`, so an ensemble that is "statistically
equivalent" is simply a different set of predictions. That is why
`R/py-random.R` reproduces CPython's MT19937 — its `init_by_array`
seeding, its `getrandbits`, its rejection-sampling `_randbelow`, and
`shuffle` / `sample` / `choice` on top. R's own `sample()` is a different
algorithm on a differently-seeded generator and agrees with none of it.

Two places are deliberately compared loosely, and both are stated where
they are implemented:

* **The Yeo-Johnson lambda.** sklearn hands the search to
  `scipy.optimize.fminbound`, which is ported step for step, but the
  likelihood goes through a `logsumexp` whose summation order R cannot
  share. It agrees to about an ULP; Brent's accept/reject test is a
  strict comparison, so the two searches occasionally take different
  final steps and stop `xatol` (1.48e-8) apart. Measured: lambdas within
  2e-8, transformed values within 8e-8.
* **Mitra's sign mirrors.** The reference draws them from NumPy's
  *global* generator and never seeds it, so its own mirrors differ
  between two of its own runs. They are compared with the mirrors turned
  off; the R port offers a seed, which the reference does not.

Two ordering conventions differ harmlessly and the tests know it. The
reference groups ensemble members by iterating a `set` of normalisation
method names, whose order moves with Python's hash seed; and Mitra casts
to float32 leaving `transform_X`, where this port narrows one step later
in `as_float_tensor()`. Members are averaged, so the first cannot change
a prediction; the second is the same rounding, so neither can the
second. The comparisons align by method and compare after the narrowing
rather than pretending the differences are not there.

## Five backends, two stagings

The staging follows whatever the model's own structure makes bisectable.

**TabPFN** is an ensemble of independently preprocessed members, so it is
staged by member and by phase: `preprocess:*`, `forward:logits`,
`predict_*`. From v2.6 on the preprocessing moved inside the network, so
the bare-network runs for v2.6 and v3 stage differently again — see
their sections above.

**TabFM**, **TabICL** and **Mitra** are each one forward pass through a
handful of modules, so they are staged by module — `stage:cell`,
`stage:col1`, `stage:row1`, `stage:col2`, `stage:reps`, `stage:logits`
for TabFM; `stage:col`, `stage:reps`, `stage:logits` for TabICL;
`stage:quantile`, `stage:embedded`, `stage:encoded`, `stage:logits` for
Mitra — plus the final prediction. Both sides dump the same tensors: Python via forward hooks,
R via the `TABFOUND_DUMP_DIR` hook built into the model.

Both also get a *layer-level* check that needs no checkpoint at all
(`layers_reference.py`): each layer is instantiated with identical random
weights on both sides and compared directly. That is where a broken
RMSNorm, a mis-indexed RoPE or the wrong GELU surfaces in milliseconds
rather than as a vague drift deep inside a full model. It is also where
the differences *between* the two families get pinned — the same test
file asserts that TabFM's tanh-approximation GELU does **not** reproduce
TabICL's exact one, and that the two RoPE pairing conventions disagree.

## Why the comparison is staged

A single end-to-end number tells you *that* something is wrong, never
*what*. Each run therefore compares four things, in the order the data
flows:

| stage | what it answers |
|---|---|
| `schema:categorical` | did both sides decide the same columns are categorical? |
| `forward:chunked` / `forward:cached` | do the cheap-inference paths match the reference's? |
| `selfcheck:*` | do they match R's own plain forward, exactly? |
| `preprocess:X_train` / `X_test` / `y_train` | did R build the same per-member model input? |
| `forward:logits` | given identical input, did the network agree? |
| `predict_proba` / `predict_mean` / `predict_quantiles` | did the ensemble combination and output head agree? |

Both harnesses write the same `member_NN/{X_train,X_test,y_train,logits}.safetensors`
layout, so the two trees can be diffed file-for-file.

This is not decoration. Every bug found while building the TabPFN
backend was localised by which stage went red first:

- `preprocess` red, `forward` green → constant-column removal dropped
  every column containing a `NaN`, because R's `==` yields `NA` where
  NumPy's yields `False`.
- `preprocess` red on one member only → the SVD step was missing the
  mean-imputation that the reference wraps around its scaler, and the
  sign convention of `svd_flip`.
- `preprocess` and `forward` green, `predict_mean` red → the reference
  divides the decoder output by `softmax_temperature` (default **0.9**,
  for the regressor as well as the classifier) before anything else
  touches it. This was worth ~0.1% on every prediction and is invisible
  end-to-end unless you can see that the logits already matched.
- `preprocess` and `forward` green, `predict_mean` red at ~1e-4 of its
  own scale on *every* regression fixture → the ensemble skipped
  `translate_probs_across_borders` for members whose target transform is
  the identity, on the reasoning that translating a grid to itself is a
  no-op. It is not: a `FullSupportBarDistribution` puts half-normal tails
  on its outer buckets, so evaluating its CDF at its own borders
  re-quantises the tail mass. Doing the round trip unconditionally, as
  the reference does, cut the error 500x. Only visible because the stage
  before it was already exact.
- TabICL's `stage:col` red while its `in_linear` was exact to 1e-7 →
  the wrong GELU. `nn.TransformerEncoderLayer(activation="gelu")` is the
  exact erf form, but the shared helper had been written for TabFM,
  which is a JAX port and wants the tanh approximation. Bisecting one
  block at a time put it inside a single feed-forward.

## Tolerances

One global tolerance would be meaningless across stages on different
scales, so `parity_tolerances()` sets them per stage. A stage passes if
*either* the absolute or the relative bound holds.

| stage | abs | rel | rationale |
|---|---|---|---|
| `schema:categorical` | 1e-6 | 1e-6 | a set of column indices; it either matches or it does not |
| `forward:chunked` / `forward:cached` | 1e-3 | 1e-3 | the plain forward's gap, for the same reason |
| `selfcheck:*` | 0 | 0 | same implementation, same arithmetic; anything else is a bug |
| `selfcheck:*-fp32` (v3) | — | — | 1e-5 on `max_scaled`: same computation, but a different attention batch shape, so float32-level rather than bitwise |
| `cache:vs-batching` | 0 | 1.5x | exact, or no worse than changing the test batch size already is |
| `preprocess:*` | 1e-6 | 1e-6 | float64 arithmetic on both sides; in practice these come out at exactly 0 |
| `forward:logits` | 1e-3 | 1e-3 | unnormalised, O(10), accumulated over 18–24 float32 layers; a uniform shift is invisible after softmax anyway |
| `predict_proba` | 1e-5 | 1e-4 | what the user actually sees |
| `predict_mean` / `predict_quantiles` | 1e-4 | 1e-4 | ditto |
| `decode:*` (v2.6) | 1e-5 | 1e-5 | both sides decode the *same* logits, so only the arithmetic can differ |
| `predict_proba:single` (v2.6) | 1e-4 | 1e-4 | one forward pass, so 24 layers of float32 rounding reach it undamped rather than averaged over four members |
| `stage:cell` | 1e-6 | 1e-6 | deterministic arithmetic, no attention; comes out exactly 0 |
| `stage:*` (other) | 1e-4 | 1e-3 | raw activations |

All `stage:*` rows additionally pass on `max_scaled` at 1e-4, which is
what actually carries them (see below).

### A third measure: `max_scaled`

Elementwise relative error is meaningless for a tensor that crosses zero,
and absolute error is meaningless for one whose entries span three orders
of magnitude. TabFM's decoder emits `max_classes = 10` logits regardless
of how many classes a dataset has, and the unused slots get pushed to
around **-800** — so a `max_abs` of 6.6e-3 there looks alarming and is
in fact 8e-6 of the tensor's own scale.

`max_scaled = max_abs / max(abs(reference))` is reported alongside the
other two, and the activation stages pass on it. Without that column the
sensible reading of `stage:logits` on `clf_iris` would have been "the ICL
stack is drifting"; with it, the reading is "float32, as expected".

## The fingerprint caveat

TabPFN appends a per-row "fingerprint" feature: `SHA-256` of the row's
rounded float64 bytes, mapped into `[0, 1]`. It is deliberately a
chaotic function of the row.

That makes it a hash, not a number. A difference of 1 ULP in *any*
upstream cell — the kind LAPACK and the reference's ARPACK produce
routinely — changes the bytes and therefore replaces that row's
fingerprint with an unrelated draw from `[0, 1]`. Rounding to 12 decimals
absorbs most of it, but a value sitting on a rounding boundary still
flips. This cannot be fixed short of reproducing NumPy's LAPACK calls
bit-for-bit, and it is not a correctness problem: the fingerprint carries
no information, it exists only to let the model tell identical rows
apart.

So every fixture is run twice: once as published, and once as `<name>_nofp`
with `FINGERPRINT_FEATURE=False` on the Python side and
`add_fingerprint_feature` honoured from the config on the R side. The
`_nofp` variants are the deterministic contract and must pass every
stage. A fixture that fails **only** with fingerprints on, while its
`_nofp` twin passes the same stage, is reported as `KNOWN` rather than
`FAIL` — and the report prints how many rows actually differ, so a
genuine regression (all rows) never hides behind a flipped hash (one or
two rows).

## Current status

### TabPFN v2.5 vs `tabpfn` 8.2.0 / torch 2.13.0

```
38/44 checks within tolerance (6 known fingerprint-sensitive)
```

- All four `_nofp` fixtures: every stage passes, with member inputs
  bit-identical (`max_abs == 0`).
- Both classifier fixtures pass with fingerprints on as well.
- The two regressor fixtures with fingerprints on differ on 2 of 200 and
  4 of 480 rows respectively, for the reason above.

### TabPFN v3 vs `tabpfn` 8.2.0 / torch 2.13.0

```
42/42 checks within tolerance
```

- All 510 classifier / 504 regressor tensors map onto the R module tree
  by **identity** — no key rewriting at all.
- `forward:logits` agrees to ≤2.9e-6 of its own scale across all six
  fixtures, including `clf_binary_missing` (a constant column and 25
  NaNs) and `reg_tiny` (12 rows).
- `decode:probs`, `decode:mean` and `decode:quantiles` are **exactly 0**
  on every fixture: handed the same logits, the softmax and the bar
  distribution agree bit for bit.
- `predict_proba:single` — end to end through the public API — agrees to
  1.6e-6.
- The KV cache and the chunked forward agree with the reference's own
  runs of the same mechanisms, and with R's plain forward, to ≤2.9e-6 of
  scale. `reg_skewed`'s cache is exactly 0 against R's plain forward.

**All eight published checkpoints, not just the defaults.** The v3 repo
ships six specialised variants alongside them. Every one has an identical
`config` and identical state-dict keys and shapes, so the stored tree
grades the defaults only — a second copy would exercise the same code.
The others were checked once by hand, on one shared input:

| checkpoint | max_abs vs Python |
|---|---|
| `classifier-v3_default` | 1.2e-6 |
| `classifier-v3_20260417_binary` | 7.2e-7 |
| `classifier-v3_20260417_multiclass` | 6.0e-7 |
| `classifier-v3_20260506_ood` | 1.2e-6 |
| `regressor-v3_default` | 8.6e-6 |
| `regressor-v3_20260417_mediumdata` | 2.2e-5 |
| `regressor-v3_20260506_ood` | 8.6e-6 |
| `regressor-v3_20260506_timeseries` | 4.3e-6 |

The two `_ood` rows match their defaults exactly because the weights are
byte-identical — SHA-256 over the full state dict agrees. Only the
`inference_config` recipe bundled with them differs, and that is
estimator-side, not network-side.

### TabFM 1.0.0 vs `tabfm` / torch 2.13.0, float32

```
21/21 checks within tolerance
```

- All 913 checkpoint tensors map onto the R module tree by **identity**.
- `stage:cell` is **exactly 0** on all three fixtures — the Fourier
  expansion, the cyclic feature grouping and the target embedding are
  bit-identical.
- Every later stage agrees to ~1e-6 of its own scale. `predict_proba`
  agrees to 3e-6 (`clf_tiny`) and 7e-6 (`clf_iris`); the regression head's
  `predict_mean` to 2e-6.
- The layer-level checks (29 assertions, no checkpoint needed) pass.

The reference is run in **float32**. TabFM is designed for bfloat16 and
`tabfm_v1_0_0.load()` casts to it by default; comparing bfloat16 against
R torch's float32 would measure the dtype, not the port.

### TabICL v2 vs `tabicl` 2.1.1 / torch 2.13.0

```
16/16 checks within tolerance
```

- All 391 classifier / 347 regressor tensors map onto the R module tree,
  with one rewrite: `col_embedder.in_linear.*` gains a `.layer`
  component because the R side wraps its linear in a skip-aware module.
- Every stage agrees to ~1e-6 of its own scale. `predict_proba` agrees
  to 2e-6 (`clf_tiny`) and 6e-6 (`clf_iris`); the regressor's full
  999-level quantile grid to 5e-6.
- `TabICL.forward` dispatches on `self.training`: the training path is
  the plain three-stage pipeline, eval routes through an inference
  manager with chunking and K/V caching. The reference script records
  the gap between them, which is currently **exactly 0** — so the plain
  path this port implements is the path users get.
- `clf_binary_missing` is deliberately not run against TabICL: its
  network has no missing-value handling of its own, so unimputed NaN
  propagates to NaN logits. That behaviour is pinned by a test instead.

### Mitra vs `autogluon` Tab2D

```
20/20 checks within tolerance
```

- `stage:quantile` and `stage:embedded` are **exactly 0** on all four
  fixtures: the quantile-rank embedding, the bucketing convention and the
  packing of the target as feature column 1 are all bit-identical.
- `predict_proba` agrees to 9e-7; regression `predict` to 1.4e-5.
- `clf_binary_missing` is not run here either, for a subtler reason than
  TabICL's: Mitra does not propagate `NaN`, it *absorbs* it. One missing
  value makes a column's quantiles all-`NaN`, every value then buckets to
  zero, and the zero-variance guard flattens the column. The output is
  finite and the feature has vanished. Pinned by a test rather than by a
  fixture.

### A note on the regression fixtures

The regressors all operate on rescaled targets, and not in the same way:
TabFM and TabICL standardize, while Mitra maps to `[0, 1]` by min-max
(`normalize_y`). The reference scripts apply whichever the wrapper does
and invert it on the output. That is deliberate: it puts the
standardization inside the compared surface rather than leaving it as an
untested assumption on the R side. It was worth doing — a sanity check on
iris caught the R predictors feeding raw `y` to the network, which turned
an RMSE of 0.35 into 1.82 while every stage-level parity check still
passed. Parity against a network says nothing about the wrapper around
it — which is what the wrapper layer above is for. The target rescaling
was the first piece of that wrapper to get compared; it is no longer the
only one.

### Current status: the wrapper layer

Every comparison in `ensemble_reference.py` passes.

- The `random.Random` stream is exact on all five seeds across
  `random`, `getrandbits`, `_randbelow`, `shuffle`, both branches of
  `sample`, and `choice`.
- `Shuffler` matches on all 40 (size, method, seed) combinations,
  including the Latin squares.
- Every sklearn estimator is exact to ~1e-13 or better, except the
  Yeo-Johnson lambda discussed above.
- All six ensemble configurations — three TabICL, three TabFM —
  reproduce **every member**: same feature permutation, same class
  relabelling, same normalisation, same member matrix to ~4e-8 (the
  Yeo-Johnson gap, on the `power` members only; the `none` members are
  exact to 9e-16).
- Mitra's preprocessor is bit-identical after the float32 narrowing,
  including the imputation, the constant-column drop and the min-max
  target scaling.

End to end against `TabICLClassifier` / `TabICLRegressor` on the released
v2 checkpoints, 60 training rows and 12 test rows, `n_estimators = 8`:

| output | max_abs | of its own scale |
|---|---|---|
| `predict_proba` | 1.6e-6 | 1.6e-6 |
| `predict` (mean) | 4.9e-6 | 2.0e-7 |
| `predict` (median) | 5.5e-6 | 2.2e-7 |
| `predict` (quantiles) | 7.5e-6 | 2.8e-7 |

That is the float32 forward pass's own noise, reached through eight
independently preprocessed members — so the preprocessing, the member
construction, the class-relabelling inversion and the ensemble
combination are all in agreement, not just the network.

TabFM's `cat_mask` — the flag that routes a categorical column's cells
through a separate Fourier basis — is checked against the reference
network directly, on random weights and so without any checkpoint: the
two agree to 2e-7 with a mask, both give exactly the plain forward pass
when the mask is all-`FALSE`, and both move the logits by 3.7e-3 when it
is not. That last number is the point: the mask is not bookkeeping, and
a wrapper that builds one while the network ignores it would have looked
correct in every stage-level check.

## Size

`reference/` is ~92 MB, nearly all of it 5000-bin regression logits:
TabPFN v2.5's per-member dumps (24 MB) and v2.6's ensemble (37 MB)
dominate, with the v2.6 and v3 bare-network trees at 7 MB each and TabFM's
per-stage activations at 11 MB. It is
`.Rbuildignore`d, so it never ships in the package tarball. If you would
rather not track it in git, delete it and regenerate with
`--regenerate`. The small `transforms/` (16 KB), `ensemble/` (130 KB) and
`layers/` (84 KB) references are worth keeping either way — they ship
with the package and are what let `test-prep-transforms.R`,
`test-prep-sklearn.R`, `test-prep-ensemble.R`, `test-py-random.R` and
`test-nn-layers.R` run without any checkpoint at all. `transforms/` is
also where a preprocessing step
gets pinned before anything downstream can use it -- v3's
`quantile_uni_extrapolate` was added there, with a fixture built to carry
out-of-range test values, because without them it and `quantile_uni` are
the same function and the comparison would pin nothing.

## Adding a backend

1. Add fixtures to `fixtures.R` if the model needs shapes the current
   ones do not cover.
2. Write `<backend>_reference.py` that runs the reference package on a
   fixture and dumps the same `member_NN/` layout plus final outputs.
3. Add a `parity_<backend>()` to `compare.R` and a `run_<backend>()`
   branch to `run-parity.R`.
4. Thread a `trace_dir` through the backend's predictor so the R side
   writes the matching tree.
