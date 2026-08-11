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
peak = intercept + weights + persistent + ensemble(n) * max over stages(transient)
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
  the sum: only one is live at a time.

Each backend contributes only shapes, in tensor *elements*, through
`peak_terms()` in its `register_backend()` call. `coefs/*.json` supply
the constants that turn elements into bytes. `.peak_from_terms()` in
`R/core-memory.R` is the one place the expression is evaluated — the
fitter and the replay test both go through it, so the constants cannot be
fitted to one expression and applied to another.

## What measurement changed

The constants shipped first were *anchored*: set so the estimator landed
on the right side of a handful of remembered runs. `calibrate.R` has
since measured all six backends on the machine those runs came from. Four
things came out of it, and three of them contradict what the anchors
assumed.

**1. Everything costs far more than the anchors implied.** TabICL at
6,426 × 90 with one ensemble member measures **24.5 GB**, where the
anchored constants said 2.8 GB. TabPFN v3 measures 24.7 GB against an
anchored 1.8 GB. The anchors were only ever constrained by *"this run
worked"*, which turns out to be a very weak statement on a 48 GB machine.

**2. `act_copies` is 75–212, not 12–30.** The resident high-water mark
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

**4. The attention term is small but real.** `att_copies` came out
between 0.09 and 2.0 rather than the 0 the anchors assumed, so the
quadratic term does carry some weight even though no backend passes an
attention mask to `scaled_dot_product_attention`.

### Where the estimator and the field reports still disagree

The hand-off records TabICL completing a 6,426 × 90 fold in 66 s at its
default 8 members, on this machine. With the measured constants the
estimator puts that run at ~94 GB and refuses it. Both cannot be right,
and the disagreement is not resolved:

* the 1-member measurement at those dimensions (24.5 GB) is direct and
  reproducible;
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
`safety_factor` is set to whatever makes the tightest point exactly 1.

| backend | points | ratio range | held-out ctx=6400 | notes |
|---|---|---|---|---|
| `tabpfn` | 12 | 1.00–1.49 | 1.29–1.71 | |
| `tabpfn26` | 12 | 1.00–1.36 | 1.21–1.33 | |
| `tabpfn3` | 12 | 1.00–1.31 | 1.18–1.34 | |
| `tabicl` | 12 | 1.00–1.38 | 1.17–1.67 | |
| `mitra` | 8 of 12 | 1.00–1.41 | 0.94 | 4 points censored |
| `tabfm` | 7 of 12 | 1.00–1.41 | 0.96 | 5 points censored |

Mitra and TabFM are the weak ones, and for the same reason: they are
large enough that on a 48 GB machine every informative point is at or
past the ceiling. Their fits rest on the small-feature corner and
extrapolate into a region nothing could measure here. Mitra's fit puts
2,000 × 90 at 78 GB; the hand-off reports that run completing in 33 s.
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

**Extrapolation.** The largest context size is held out, refitted
without, and predicted. That is the whole job: the sizes that matter are
the ones too big to measure safely.

## Running it

```sh
Rscript inst/memory/calibrate.R tabicl tabpfn3      # measure and report
Rscript inst/memory/calibrate.R tabicl --write      # ...and write coefs
Rscript inst/memory/calibrate.R tabicl --refit      # re-fit stored points
Rscript inst/memory/calibrate.R tabicl --ensemble   # member sweep
Rscript inst/memory/calibrate.R --all --grid full
```

Needs the backend's weights downloaded. Measurements are written before
anything is reported, so a formatting bug cannot lose hours of them.
`--refit` re-derives the terms from today's `peak_terms()` and re-fits
without re-measuring, which is what to run after changing a formula.

`tests/testthat/test-memory-calibration.R` replays the stored
measurements against the shipped constants and needs no weights at all.
It asserts the direction (never below measured), the accuracy (within
1.5×), and that `peak_terms()` still produces the shapes the constants
were fitted to — that last one only when the weights happen to be
present, since a config is the only place the dimensions live.

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
