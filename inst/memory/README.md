# Memory preflight

`estimate_peak_memory()` answers one question before any tensor exists:

> will `fit()` + `predict()` at these dimensions, with these options, fit
> in the memory this machine currently has available?

It has to be answered in advance because it cannot be answered
afterwards. A libtorch allocation failure at fold size kills the R
process outright — no condition, no traceback, no output — so there is
nothing for a handler to catch. Preflight is the whole defence.

## The model

```
peak = max( floor,
            (intercept + weights + persistent
             + ensemble(n) * max over stages(transient)) * safety )
```

* **weights** — the checkpoint, resident for the object's life. Read
  exactly from the loaded module, or from `state_dict_shapes` in a
  converted `config.json` (so a pre-download check is exact too), or from
  the artifact's file size, or last of all from the per-backend constant
  in `coefs/`.
* **intercept** — what is resident before any activation exists and is
  not in the checkpoint: R itself, libtorch, its allocator arenas.
  Between 0.6 GB and 2.7 GB depending on backend; 9.5 GB for TabFM.
* **persistent** — the fitted context, plus one KV cache per ensemble
  member when `kv_cache = TRUE`. `kv_cache` does not save memory; it
  moves it here out of the transient term.
* **transient** — the forward pass. The **maximum** over stages, never
  the sum: only one is live at a time. Each stage contributes four kinds
  of term: `act`, the activation a live block holds, multiplied by
  `act_copies` and divided by `save_peak_memory_factor`; `att`, a
  materialised attention score matrix, which exists only where the
  attention is masked; `res`, whole-table tensors a *chunked* stage keeps
  outside its loop, which neither kind of chunking touches; and
  `prepass`, the summary pass a row-chunked forward cannot start without.
  The pre-pass reads every context row and a slice of the columns, so the
  *column* chunk bounds it and the row chunk does not — and it carries
  its own `prepass_copies`, because it is one column-stage stack rather
  than a whole pipeline. Charging it the forward's count put TabICL's
  chunked estimate at 112 GB against 36.9 GB measured; the sweep puts it
  at 71.2 against a forward's 137.7, and TabPFN v3's at 3.9 against 78.8.
* **floor** — the largest peak observed where the activation was
  negligible. Below a gigabyte or two the number being measured is the
  process, not the model, and it does not fall with the input; the floor
  is how the estimate says so instead of extrapolating a slope into a
  region where there is none.

Each backend contributes only shapes, in tensor *elements*, through
`peak_terms()` in its `register_backend()` call. `coefs/*.json` supply
the constants that turn elements into bytes. `.peak_from_terms()` in
`R/core-memory.R` is the one place the expression is evaluated — the
fitter and the replay test both go through it, so the constants cannot be
fitted to one expression and applied to another.

## What measurement changed

The constants shipped first were *anchored*: set so the estimator landed
on the right side of a handful of remembered runs. `calibrate.R` has
since measured all six backends on the machine those runs came from, and
been re-run after TabPFN v3 gained the reference's stage chunking and
every backend's layer loop started letting go of its intermediates. Five
things came out of it.

**1. Everything costs more than the anchors implied.** TabICL at
6,426 × 90 with one ensemble member measures **25.7 GB**, where the
anchored constants said 2.8 GB. The anchors were only ever constrained by
*"this run worked"*, which turns out to be a very weak statement on a
48 GB machine.

**2. `act_copies` is 60–130, not 12–30.** The resident high-water mark
runs one to two hundred times the largest single activation. That is not
the live set — it is what the allocator has taken from the OS and not
given back. For an OOM guard that is the right quantity: the kernel kills
on resident memory, not on what is reachable.

**3. Ensembles do raise the peak.** The hand-off that commissioned this
work said, in bold, not to multiply by `n_estimators` — members run
sequentially, so the live set is one member's. True of the live set,
false of RSS. Measured on TabICL at 800 × 32, one member to sixteen took
the peak from 3.1 GB to 8.7 GB, in steps that look like an allocator
taking a bigger arena each time it needs one. TabFM does the same and
then saturates. Hence `ensemble_log2_factor` and `ensemble_cap`, and the
sweeps in `measurements/ensemble-*.json`.

**3b. Stages do not hold the same number of copies.** A single
`act_copies` says they do, which was invisible while the column stage
always dominated. Chunking hands that role to the in-context stage, and
the difference shows: on TabICL the in-context stage measures **186**
copies against the column stage's **141**. Fitting one number for both
put `res_copies` on its upper bound, two censored points below their own
observed floor, and the held-out chunked point at 0.48 of measured — a
failure that looked like a missing term and was two wrong constants. The
in-context stage now carries its own `icl_copies`, identifiable only when
some point has it dominating, which in practice means a chunked grid.

The second half of that was a bound, not a model: `res_copies` had an
upper limit of 64 chosen when nothing came near it. A chunked grid wants
**147**. A parameter sitting exactly on its bound is the optimiser
reporting the bound rather than the data, and it is worth reading that
way whenever it happens.

**4. There is no attention score matrix.** This reverses what this file
used to say. Torch's fused SDPA takes a memory-efficient kernel when no
mask is passed and never materialises the `(n, n)` scores: at ICL shapes
`(1, 8, n, 64)`, peak RSS goes 353 → 382 → 448 MB across n = 4,000, 8,000
and 16,000, where the scores alone would be 488 MB, 1.9 GB and 7.8 GB —
and a hand-written attention at n = 8,000 takes 6.2 GB, which is the
control that proves the sampler would have seen one. The earlier
`att_copies` of 0.09–2.0 were a spare linear-in-`n` degree of freedom the
fit had found a use for, not a measurement of anything.

Every `att` term is now zero except TabFM's, which is the only backend
that restricts context by *masking* rather than slicing, and a masked
SDPA does pay — at roughly a sixth of a float32 score matrix. TabFM's own
grid is too small to constrain it, so its fitted `att_copies` is zero
too; the term is in the shape because the mechanism is real, not because
the data has yet said how much it costs.

**5. Below a gigabyte, the measurement is the process.** TabPFN v2.5 at
8 features measures 2.1 GB at 800 rows and **1.1 GB at 1,600** — a peak
that halves when the input doubles, reproducibly across three repeats
each. No model fits that, and letting the safety factor stretch to cover
it lifted every estimate for that backend by a factor of two. Those
points now set `floor_bytes` and are excluded from the safety factor,
which exists to cover the slope.

### Where the estimator and the field reports still disagree

The hand-off records TabICL completing a 6,426 × 90 fold in 66 s at its
default 8 members, on this machine. With the measured constants the
estimator puts that run at ~94 GB and refuses it. Both cannot be right,
and the disagreement is not resolved:

* the 1-member measurement at those dimensions (25.7 GB, re-measured
  2026-08-11) is direct and reproducible;
* the ensemble factor is measured only at 800 × 32, where the whole sweep
  fits under the ceiling — at fold size, every multi-member run is at or
  past it, so there is nothing to check the extrapolation against;
* the same report has that configuration being SIGKILLed as soon as a
  neighbour took 7 GB, which is hard to square with a run that had
  comfortable headroom.

The estimator follows its measurements. Treat multi-member estimates at
large dimensions as the least trustworthy number it produces, and see
"what to do next".

## Reading the numbers

Ratios are `estimate / measured`, and must be ≥ 1: a guard that
under-estimates costs a session, one that over-estimates costs a warning.
`safety_factor` is set to whatever makes the tightest *slope* point
exactly 1.

| backend | points | max context | ratio range | safety | floor |
|---|---|---|---|---|---|
| `tabpfn` | 12 | 6,400 | 1.00–2.00 | 1.42 | 2.1 GB |
| `tabpfn26` | 12 | 6,400 | 1.00–2.22 | 1.28 | 2.4 GB |
| `tabpfn3` | **21** | **32,768** | 1.00–1.59 | 1.33 | 2.4 GB |
| `tabicl` | 12 | 6,400 | 1.00–1.97 | 1.43 | — |
| `mitra` | 11 of 12 | 6,400 | 1.00–2.25 | 1.26 | — |
| `tabfm` | 8 | 1,600 | 1.00–1.51 | 1.27 | 20.4 GB |

The wide end of each range is a small, baseline-dominated point sitting
on the floor, where the estimate is telling you what the backend costs to
do nothing. `test-memory-calibration.R` grades the accuracy band only
where the transient is the larger term and the model rather than the
floor is what the estimate returns, and grades the fail-safe margin
separately — rolling the two into one number said little about either.

TabPFN v3 gets its own grid (`--grid deep`, out to 32,768 rows) because
nothing below 2,048 crosses its own stage-chunk boundary, and without
points on both sides the chunked activation term and the whole-table
`res` term are the same straight line. It is also the only backend whose
largest grid points all completed: before the chunking landed, 8,192 × 120
and 16,384 × 120 were both at this machine's ceiling.

Mitra and TabFM are the weak ones, and for the same reason: they are
large enough that on a 48 GB machine every informative point is at or
past the ceiling. TabFM's grid stops at 1,600 rows and its fitted
`act_copies` is pinned at the lower bound, which is another way of saying
the intercept and the floor explain everything it could measure.
Believe neither without a bigger machine.

## Method

**One process per grid point.** Resident memory is a high-water mark that
never comes back down, so a second point measured in the same process
inherits the first's floor. And the interesting points are the ones that
die: an allocation failure aborts that worker, which the driver records,
and leaves the harness running.

**Peak.** `VmHWM` from `/proc/<pid>/status` where it exists — a true
high-water mark, exact and free. macOS has no equivalent, so a 10 ms
`ps -o rss=` sampling loop, with the sampling error that implies. Each
measurement records which method produced it. Do **not** use `gc()` or
R-side accounting: the allocations belong to libtorch, which R cannot
see.

**Repeats.** A high-water mark can only be missed by sampling, never
overstated, so a point that finishes in under eight seconds is run three
times and the largest peak kept. This is worth doing and it is *not* what
makes the small end of a grid non-monotone — repeating TabPFN v2.5's
8-feature points moved them by less than 0.2 GB and they still fall by
half when the rows double. That is the floor, not the sampler.

**Censoring.** A point whose peak came within 60% of the machine's total
did not measure what it wanted, it measured what it could get. Those are
lower bounds, not values; they are recorded, excluded from the fit, and
checked afterwards — the fitted model has to put every one of them above
its own observed floor. `TABFOUND_CENSOR_FRACTION` moves the threshold.

**Died vs. errored.** A worker that vanishes without a result is only a
crash boundary if it got as far as `work_start` in its marker file.
Anything earlier is a broken environment, reported as `error` and
excluded from everything. This is not hypothetical: reinstalling the
package mid-run turned six TabFM points into "died", and without the
distinction that would have baked a fictional ceiling into the constants.

**Fitting.** Log-space least squares, because peaks span two orders of
magnitude. A coarse sweep first and the optimiser only polishes — going
straight to `optim()` looked fine and was not, settling on a 100 MB
intercept where the data wanted 1.7 GB and inflating the slope to
compensate, silently.

**Absent terms are pinned, not fitted.** Where every `att` or every `res`
in a backend's grid is zero, its coefficient is held out of the search
entirely rather than left to wander. Two reasons: a number fitted against
an identically-zero column is whatever the sweep happened to land on, and
written into a shipped file it reads as a finding; and handing L-BFGS-B a
coordinate whose bounds are equal perturbed the other three enough to
miss a synthetic round-trip by 2.4%.

**Extrapolation.** The largest context size is held out, refitted
without, and predicted. That is the whole job: the sizes that matter are
the ones too big to measure safely.

## Running it

```sh
Rscript inst/memory/calibrate.R tabicl tabpfn3      # measure and report
Rscript inst/memory/calibrate.R tabicl --write      # ...and write coefs
Rscript inst/memory/calibrate.R tabicl --refit      # re-fit stored points
Rscript inst/memory/calibrate.R tabicl --ensemble   # member sweep
Rscript inst/memory/calibrate.R tabicl --spmf       # chunk-factor sweep
Rscript inst/memory/calibrate.R tabicl --chunks     # stage-chunk sweep
Rscript inst/memory/calibrate.R --all --grid full
```

Order matters when re-running several: a size refit recomputes
`safety_factor` from the size grid alone, so `--chunks` has to come after
it or its lift is lost.

Needs the backend's weights downloaded. Measurements are written before
anything is reported, so a formatting bug cannot lose hours of them.
`--refit` re-derives the terms from today's `peak_terms()` and re-fits
without re-measuring, which is what to run after changing a formula.

`tests/testthat/test-memory-calibration.R` replays the stored
measurements against the shipped constants and needs no weights at all.
It asserts the direction (never below measured, on every point), the
accuracy (within 1.5× above, on the points where the activation is what
was measured), that the fail-safe margin stays bounded, and that
`peak_terms()` still produces the shapes the constants were fitted to —
that last one only when the weights happen to be present, since a config
is the only place the dimensions live.

## What to do next

1. **Re-measure Mitra and TabFM on a larger machine.** Everything
   informative about them is censored here. Until then their constants
   describe a corner of the design and extrapolate on faith.
2. **Measure the ensemble factor at working dimensions.** It is the
   single largest multiplier in the model and the only one fitted at one
   small point. This is where the estimator and the field reports part
   company.
3. **`mitra` and the TabPFN family have borrowed ensemble factors**, not
   measured ones — `ensemble_source` in their coefficient files says so.
   Mitra's default is one member and TabPFN's ensemble is config-driven,
   so neither is exercised by default, but a user who sets
   `n_estimators` gets a number nobody has checked.
4. GPU/MPS constants. The device key in each file exists for it; nothing
   is calibrated but `cpu`, and an uncalibrated device is refused rather
   than guessed at.
5. **Measure a chunked grid for the remaining backends.** TabICL has
   one — `--chunked --grid deep`, 21 points, reaching 32,768 × 50 at
   29.5 GB where the default grid tops out much earlier — and it is
   fitted together with the default grid and the `--chunks` sweep's
   column-chunked points, so one set of constants describes all three
   configurations. TabFM has none: its `--chunks` sweep point died at
   both settings on this machine. TabPFN v3 needs none, since chunking is
   its default and its ordinary grid is already chunked.
5. **Push the other five backends past their chunk-free ceiling.** Only
   TabPFN v3 has a grid that reaches 32,768 rows, because only it can. The
   rest still stop at 6,400, so the envelope they describe beyond that is
   extrapolation from a quarter of the range v3's is fitted on.
6. **Re-measure the ensemble sweeps.** They predate the between-layer
   collect, which is precisely a change to how much the allocator holds
   and does not return — the mechanism `ensemble_log2_factor` was
   introduced to describe.

## How the performance knobs work

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
row-independent prefix v3 has, and the reason the design transfers. Measured, three repeats per point, fresh process each:

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

### Collecting between layers

R torch frees a tensor when R's collector
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


## Preflight: what the estimate is careful about

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

### TabPFN v3.5 envelope

**The memory preflight is calibrated to 32,768 rows and should not be
extrapolated past it.** Inside that envelope the constants behave: all
21 measured points sit at or below their own estimate. Outside it they
under-predict, and by more than the other backends do — holding out the
32k row and refitting on the rest puts the estimate at 0.75x the
measured peak at 16 features, where TabPFN v3 under the same treatment
gives 1.01x. So `memory_envelope()` and `suggest_chunk_sizes()` are
advice, not a guarantee, above roughly 32k rows on this backend; leave
headroom or measure. The constants were fitted on the 876 MB default
checkpoint, so the Fast variant is covered conservatively rather than
tightly.
