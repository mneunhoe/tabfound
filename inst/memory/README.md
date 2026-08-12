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
