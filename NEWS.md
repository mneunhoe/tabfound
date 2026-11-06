# tabfound (development version)

* `predict(type = "quantiles")` and `predict(type = "sample")` on TabPFN
  regressors decode every quantile or draw in one batched pass instead of
  one at a time. With 199 quantiles on 5,000 rows that is about 3.5x
  faster at under half the peak memory, and the results are bit-identical.
  Single-pass sampling now runs the same chunked, cache-aware forward as
  the mean and quantiles, so its peak no longer grows with `newdata` and
  `kv_cache = TRUE` now applies to it. Seeded draws match the old ones up
  to the forward's batch-size rounding (#64).
* A cached TabPFN ensemble (`kv_cache = TRUE`) now builds one member's KV
  cache at a time, runs it over every prediction chunk and drops it before
  the next, so a `predict()` or `predict_members()` call holds one cache
  instead of one per member. Peak memory with eight members falls by
  roughly the size of seven caches (6.4 GB to 3.8 GB at 2,000 x 30 on
  v2.5), and predictions are bit-identical. A cache kept by
  `tabfound_cache()` still holds every member, and `estimate_peak_memory()`
  now counts the cache this way (#65).
* TabPFN `predict()` calls that span two or more `predict_chunk_size`
  chunks now condition on the training rows once for the call instead of
  once per chunk, building a temporary cache one ensemble member at a
  time: 1.2-3x faster, more with larger training sets and more chunks. On
  v2.5 it is bit-identical to the uncached pass, and on v2.6 it is used
  only on chunks where it gives the uncached answer (a chunk whose
  constant or informative masks differ runs uncached). On v3 and v3.5 the
  cached pass agrees to float32 rounding, about 1e-5 on probabilities,
  inside the tolerance row chunking already has. It is built only when
  `estimate_peak_memory()` says it fits. Nothing is kept; `kv_cache = FALSE` still means no persistent
  cache, and the new `chunk_cache = FALSE` turns this off (#66).
* TabPFN ensemble regressors no longer build an `n_test x 5000` R matrix of
  averaged bucket probabilities for `type = "mean"` and `"quantiles"`,
  which only sampling read. Sampling now decodes those probabilities one
  chunk at a time. At 20,000 test rows this cuts peak memory by 1.3 GB for
  the mean and from 20.5 to 6.7 GB for a 100-draw sample. All outputs,
  seeded samples included, are bit-identical (#67).
* Between-layer garbage collections now become full collections once the
  carried state reaches 64 MB (`tabfound.collect_full_min_bytes`), and
  Mitra also collects between its sublayers. A level-0 pass left promoted
  tensors alive, so one Mitra forward at 2,000 x 50 peaked at 26.6 GB and
  is now 3.8 GB. TabICL at 6,000 x 90 went from 15.3 to 5.2 GB, and TabPFN
  v2.5 at 6,000 x 60 from 12.7 to 4.1 GB resident. Predictions are
  unchanged. The size gate also no longer skips states of 2 GB and more,
  which an integer overflow had read as too small to collect (#68).
* TabPFN v3/v3.5 attention now expands the single test-row key/value head
  as a view instead of copying it to every query head, so cached
  predictions no longer pile up a 2·H·N·D copy per layer: up to 1.9x
  faster at small batches and gigabytes less peak memory, with
  bit-identical results (#69).
* The TabPFN v3/v3.5 many-class decoder now attends a block of at most 128
  test rows at a time, collecting between blocks, and narrows its one-hot
  labels to the classes present. It no longer materialises (B, 6, M, N)
  attention scores (about 3.7 GB at M = 1,024, N = 50,000), results are
  bit-identical, and the memory guard now counts the decoder (#70).
* TabPFN v3.5 classifier KV caches now hold the decoder's projected keys
  instead of the training row embeddings, so cached predictions no longer
  recompute `mlp_classification` and `k_projection` over every training
  row on each call (75 ms to 7 ms at 10,000 rows). v3 and v3.5 regressor
  caches no longer store embeddings nothing reads, making caches and
  `tabfound_save()` bundles 14-25% smaller. Caches saved before this still
  load. Results are bit-identical (#71).
* TabICL and TabFM KV-cache builds now honour `row_chunk_size`, and TabFM's
  summary pre-pass row-chunks its first row stage. A cache build used to
  run the column and row stages over every training row at once, so
  turning `kv_cache` on gave back most of the row-chunking saving: at
  6,000 x 90 with `row_chunk_size = 1024`, TabICL's peak RSS goes from
  22 GB to 10 GB, at about 1.4x the build time, the same trade the
  uncached chunked path makes. Results are identical when a chunk doesn't
  bite, and otherwise move by about 1e-6 of scale, as the existing
  chunking does (#72).
* TabICL and Mitra skip final-layer work on context rows nothing reads:
  TabICL's last ICL block and decoder now run on the test rows only, its
  ICL cache build reuses each block's cached keys and values and skips the
  discarded last block, its summary pre-pass skips the last column block's
  unused half, and Mitra's last layer no longer runs the support stream
  past its keys and values. All but the ICL query slicing are
  bit-identical; that moves results by float32 rounding (about 1e-6 of
  scale). TabICL's network now returns the test rows only, as its cached
  path does (#73).
* TabICL's column stage no longer runs the reserved CLS slots through its
  set transformer only for the row stage to overwrite them; the CLS tokens
  are prepended in the row stage instead. That takes 4/(p + 4) off the
  column stage (about 23% of column + row time at p = 3). It regroups
  float32 reductions, moving results by about 2e-6 of scale, within
  parity tolerance. Caches saved by earlier versions still load (#74).
* The TabICL regressor no longer sorts its 999-level quantile grid in R:
  `type = "mean"` skips the sort and the tail fit, and the other types
  sort on the device. About 15x faster post-processing for the mean and
  4-5x for the rest, bit-identical on real weights (the mean can differ by
  at most ~1e-15 where the grid crosses) (#75).
* Expanding a text column at fit is about 1.8x faster: the randomized
  SVD's LU step goes through LAPACK (`Matrix::lu()`) instead of an R loop,
  and the training column's n-grams and projection are computed once
  instead of up to four times. The encoded features move by at most
  ~1e-12 (#76).
* RoPE's split-half rotation (TabPFN v3/v3.5 column aggregation, TabICL's
  row stage) now takes the two channel halves as views instead of copies,
  about 2x faster in isolation and bit-identical in output (#77).
* The memory guard no longer overestimates TabICL by 2-3x on a single
  member. Its terms counted `ceiling(p / 3)` column tokens where TabICL
  carries all p columns plus four CLS slots, and its calibration was
  measured while #68 inflated the peaks. A 6,000 x 90 single-member
  estimate falls from 26.8 to 13.3 GB (measured 10.0 GB) (#78).
* Kumo Tabular's `predict()` no longer re-encodes the training context for
  every `predict_chunk_size` chunk. A call spanning two or more chunks
  builds each member's KV cache for that call, one member at a time, and
  drops it (`chunk_cache = TRUE`, a new argument on the Kumo predictors).
  With 8 members over 4 chunks that is 1.6-2.9x faster at lower peak
  memory (9.1 to 6.2 GB on small at 8,000 context rows), and it agrees
  with the uncached pass to float32 rounding (about 2e-6).
  `kv_cache = TRUE` now also builds one member's cache at a time;
  `kv_cache = FALSE` still keeps nothing on the model, and
  `chunk_cache = FALSE` restores the old behaviour (#96).
* Kumo Tabular's predictors run several ensemble members in one forward
  pass when no cache is in use (`member_batch_size`). By default the group
  is the largest the memory estimate leaves comfortable room for, and
  members are batched only where a single member's pass is small (up to
  about 4,000 context rows). That is 1.2-1.6x faster with 8 members at
  500-2,000 context rows on CPU, and unchanged at larger contexts; answers
  move by float32 rounding only. The row-chunked KV cache build now keeps
  per-member categorical flags (#97).
* Kumo Tabular medium and large no longer pass `enable_gqa` to the
  attention kernel for the query rows; the 2 KV heads are repeated to
  every query head first. On CPU each such call is 3-4x faster, about
  6-13% end to end, and predictions agree to float32 rounding (#98).
* Kumo's layer loops now release each layer's intermediates, as every
  other backend's already did. Peak memory falls 1.4-2.9x (4.1 to 1.4 GB
  on the small regressor at 1,000 context rows) and predictions are
  bit-identical. Kumo's memory-guard constants were re-measured to match,
  so its estimates are lower and it chunks or warns less often (#99).
* Fixed TabICL and TabFM classifiers fitted on a factor with a level no
  training row has -- common in cross-validation folds, subsets, and any
  `tabfound()` fit, since hardhat keeps the outcome's levels. The class
  shifts and permutations were sized by the observed classes but undone
  over the full level set, so TabFM mislabelled or merged classes, and
  could predict the class it never saw, while TabICL failed with a
  `dimnames` error. Only the observed classes now reach the network, as in
  the reference wrappers; an unused level keeps its column, with
  probability 0 (#32).
* Fixed `predict()` on the engine API silently relabelling columns once
  `fit()` had expanded a date or text column: the expanded frame was
  renamed to the training names by position, so reordered columns, or a
  column swapped for an unknown one, passed the schema check and the
  predictions were for the wrong columns. The column names are now
  checked before the expansion is replayed, and a reorder is refused as it
  is on a fit with nothing to expand. The refusal's suggested fix,
  `newdata[, c("a", "b")]`, is now code that runs (#33).
* Fixed `tabfound_syn()` copying a large share of the real rows exactly
  with its default `proper = TRUE`. The proper draw bootstrapped the
  context, and a PFN reads a duplicated context row as a near-certain
  lookup, so each variable came back almost deterministically from the
  real row it was conditioned on: 34 to 45 of 60 rows per synthesis were
  exact copies, against 4 to 5 with `proper = FALSE` (#34). `proper = TRUE`
  now draws a subsample without replacement, as
  `tabfound_impute(proper = "subsample")` does, with the same new
  `proper_frac` (default 0.632), and copies 3 to 5. `proper = "bootstrap"`
  keeps synthpop's definition on request. `proper` is now validated:
  `"bayes"`, which used to run improper without a word, is an error, as is
  any other unsupported value.
* Fixed TabPFN regression ensembles and Kumo Tabular regressors losing
  precision when the target's mean is large next to its spread -- years,
  timestamps, prices. Both rescaled to the raw target in float32: at an
  offset of 1e6, a target of sd 1 got quantiles on a grid of 1/16 and an
  ensemble mean off by about 2 sd, and Kumo rounded `y + 1e7` to whole
  numbers before it was standardized. The ensemble now summarises on the
  z-scale borders and un-normalises in double, as the single-pass path
  and `tabpfn` 9.x do; its means also move closer to the 9.2.0
  reference. Kumo standardizes the target and inverts its quantile grid
  in float64, where `sdm` works in the float32 the target arrives in;
  at ordinary scales that moves a value by a float32 ulp. Shifting `y`
  now shifts every mean, quantile and sample by exactly as much (#35).
* Fixed TabICL and TabFM imputing a categorical column's missing values
  with its mean code (1.6 is not a category) and passing a level unseen
  at fit through as its R code. Declared categorical columns now go
  through the reference wrappers' encoders: codes over the categories
  seen at fit -- sorted for TabICL, in order of appearance for TabFM,
  where a category seen only once also counts as unknown -- with missing
  and unseen values as -1, and the columns moved in front of the numeric
  ones, as `TransformToNumerical` orders them. Only numeric columns are
  mean-imputed. `tabicl_classifier()` and `tabicl_regressor()` gain
  `categorical_features`, so `tabfound()` now declares a frame's factors
  to TabICL too (#36).
* Imputation keeps a variable's type and support. An integer 0/1 column
  with missing values stays an integer in `tabfound_impute()` (and in
  `tabfound_syn()` when it has a single observed value), where it came
  back double, observed cells included. `mice.impute.tabfound()` draws a
  numeric 0/1 dummy from the classifier, as `tabfound_impute()` does,
  rather than drawing continuous values such as 1.03 and -0.03, and an
  ordered target comes back ordered (#37).
* `as_mids()` builds the mids from the imputation's `where`, so cells that
  `where` over-imputes reach `mice::complete()`. It errored when `where`
  marked extra cells in a column with missing values, and returned the
  original values for cells marked in a fully observed column (#38).
* Fixed `tabfound_syn()` returning all-`NA` columns when a single-string
  `method` was combined with a `visit_sequence` that leaves columns out.
  The string was recycled to every column, so the unvisited ones were
  neither synthesised nor carried through (#39); a scalar `"regression"`
  or `"classification"` also errored when an unvisited column was of the
  other kind. The string now applies to the visit sequence only, and the
  other columns are carried through, as they are without `method`.
* Fixed `tabfound_load()` re-reading `ensemble_configs_dir`: a directory
  regenerated or deleted after the save (a `tempfile()` from another
  session, say) gave the reloaded model other members, or failed with
  "lexical error: invalid char in json text". A saved KV cache was still
  accepted, because its guard checked only the training rows, so the
  reloaded bundle mixed the old members' cache with the new configs and
  its probabilities moved by up to 0.74 (#40). The configs are now read
  in when the model is built and saved with it, and the cache guard
  fingerprints the constructor arguments too. Older saves still load
  while their directory exists, and say so clearly when it is gone.
  Alongside: a relative local model directory is saved as an absolute
  path, so a reload after `setwd()` no longer goes to the Hub; saving
  over an earlier save of the other form (file or cached bundle)
  replaces it; and refitting a fitted model no longer reuses the
  categorical positions an earlier fit's text or date expansion moved.
* Fitting Mitra, generating TabPFN ensemble configs, and seeded
  `tabfound_impute()` and `tabfound_syn()` runs no longer reset the
  caller's random number generator. They seed locally and put
  `.Random.seed` back when they finish, so a cross-validation or
  simulation loop around a fit no longer draws the same numbers on every
  iteration. The seeded draws themselves are unchanged (#41).
* Formula interface (`tabfound()`) (#42):
  * A factor made inside the formula (`factor(k)`, `cut(x, 3)`,
    `I(x > 0)`) is refused at fit with advice to make it a column of
    `data`. It used to be one-hot encoded, against the one ordinal
    column per factor these models take, and `predict()` then failed on
    new data that lacked any of its levels.
  * `categorical_features` takes predictor names, and positions now
    count the predictors in the formula's order (`data`'s for `y ~ .`).
    They used to count hardhat's molded columns, which put factors last,
    so `1L` in `y ~ g + x1` declared `x1`.
  * `print()` lists the predictors the model reads; `y ~ . - k` no
    longer lists `k`.
* Quantile predictions: `predict(m, x, "quantiles", c(-0.5, 1.5))` now
  refuses out-of-range levels passed by position, as it already did for
  `quantiles =`. Columns are named per level, so `c(1e-4, 0.5)` gives
  `q0.0001 q0.5` rather than `q1e-04 q5e-01`, and levels that differ past
  the seventh digit no longer share a name. In `tabfound()` predictions,
  `type = "median"` gives `.pred`, `type = "grid"` gives `.pred_q<level>`
  rather than `.pred_1` ... `.pred_999`, and a repeated level is made
  unique instead of aborting. `quantile_grid_stat()` reads a vector as one
  row and checks the grid's width against `levels` (#43).
* TabPFN v2.5 and v2.6 classifiers refuse a target with more classes than
  the checkpoint's 10-class head at `fit()`, as TabICL and Mitra already
  did. v2.5 used to fail only at predict, with a dimnames error (#44).
* A constant regression target no longer crashes v2.5 config generation in
  `generate_ensemble_configs_native()`. A constant has no Yeo-Johnson
  lambda to fit, so its members carry no target transform (#44).
* TabPFN's `predict_members()$member_index` gives the members' positions in
  the ensemble that was passed in, as documented, rather than renumbering
  the members left after zero weights are dropped (#44).
* `register_backend()` checks its name, hooks and aliases, and refuses to
  replace a built-in backend unless `overwrite = TRUE`. During
  auto-detection, a backend whose `detect()` errors is skipped with a
  warning instead of breaking detection for every model (#44).
* Fixed four places where TabPFN's preprocessing drifted from the
  reference (#45). The SVD features gave a column whose spread is
  rounding noise its own tiny scale instead of sklearn's 1, so it swamped
  every component. The quantile reshapes computed their knots with
  `quantile(type = 7)` and `approx()`, which on tied data put values half
  a rank step from NumPy's. The fingerprint hashed R's `NA` bit pattern
  rather than NumPy's NaN, so every row with a missing value got a
  different fingerprint. And string columns were coded in the session's
  collation order, which only matches sklearn's code-point order in the C
  locale, so the codes and the category shuffles drawn from them depended
  on how R was started. String columns are now coded in code-point order
  everywhere, including `tabfound_impute()` and synthesis.
* `tabular_classifier()` and `tabular_regressor()` take column names in
  `categorical_features` as well as positions; names are matched against
  `colnames(X)` at `fit()`. Before, a name warned "NAs introduced by
  coercion" and then aborted as out of range. Names that match no column,
  and declarations that are neither positions nor names, are refused with
  a message saying which (#46).
* `print(estimate_peak_memory())` has an "overhead" line for the process
  baseline and safety margin, so its lines add up to the peak (#46).
* `tabfound_complete(x, "long")$.id` follows mice: an integer for a frame
  whose row names were never set, the row names otherwise. It was always
  character (#46).
* Architecture diagrams:
  * `plot_architecture(style = )` draws the same description five ways:
    `"cards"` (the default), `"tufte"` (a table with a dot per stage on a
    log scale shared by every diagram), `"glyphs"` (each stage as a small
    table showing the axis its attention runs over), `"icicle"` (where the
    parameters are, as lengths) and `"schematic"` (how each block runs, in
    the convention of the papers and Sebastian Raschka's LLM Architecture
    Gallery: data flowing upwards, residual connections drawn, attention
    steps saying where their keys come from, facts pointing at their
    part). `options(tabfound.arch_style = )` sets the default.
  * In the schematic, `detail = "full"` adds a zoom box beside each
    distinct feed-forward step, showing what it computes inside (Linear,
    activation, Linear, or SwiGLU's gated pair) with its widths. Each
    feed-forward sublayer records them through `arch_ffn()`, checked
    against the network's parameter count.
  * `vignette("architecture-gallery")` shows TabICL in every style, and
    the schematic for four models that each show something different.
  * `plot_architecture()` and `plot_architecture_diff()` take
    `background =` and `highlight =`, to match a house style or a slide
    template. The background carries the style's greys, rules and panels
    with it, each kept as far from it as it was from the theme's own, and
    a dark background picks the dark theme. The highlight replaces what
    the style picks out -- the schematic's accent, tufte's phase colours,
    the attention colour in cards and icicle, row attention in glyphs,
    model A in the diff -- adjusted only as far as it must be to stay
    readable. Up to four highlights can be given: cards and icicle take
    them for attention, embeddings, feed-forward layers and the decoder;
    the glyphs for row attention, column attention, inducing points and
    the prediction; tufte as its phase colours; and the schematic uses
    the first, so one vector serves every style. The diff takes up to
    three: model A, model B and what differs.
  * New `plot_architecture_diff()` draws two models side by side, pairing
    their stages by kind and attention axis and their facts by name. It
    takes models, fits or `tabfound_arch` objects in any mix.
  * Every diagram ends in a key that defines each symbol in its shapes,
    each abbreviation in its text and the notation of its style, so a
    figure shared on its own still reads. It replaces the fact sheet's
    symbol line, which was cut off at the panel's width. SVG output also
    carries the title and the key as `<title>` and `<desc>`.
  * Labels no longer collide or get cut off. A key's terms and the
    schematic's callout names are measured letter by letter, so a wide
    one such as "RMSNorm" or "Embedding heads" no longer runs into what
    follows it. A callout value too long for its line wraps under
    itself, and the stage descriptions in glyphs and tufte are drawn in
    full instead of being cut at two or three lines.
* Architecture descriptions (`tabfound_architecture()`) are checked
  against each network's forward pass:
  * Corrections. Kumo Tabular's column and row blocks alternate, one of
    each per layer, rather than running as two stacks; its readout tokens
    skip the column blocks. TabFM's CLS columns join after the first
    column stage. TabICL's and TabFM's final norm belongs to the decoder
    (`decoder(ln(x))`), not the target encoder; TabICL's inducing points
    are learned vectors whose keys are the labelled rows. TabPFN v2.5's
    target joins as an extra column on every row, and its regressor's
    encoder is an MLP; TabPFN v2.6's positional subspace is a quarter of
    the model width (48 only at the published width).
  * A stage's sublayers are the steps of its forward pass in the order
    they run, which a test checks by tracing each network. Norms are
    listed one by one where they run, whole attention-and-MLP blocks are
    split into their steps, and both halves of every induced-attention
    block are shown. This changes `detail = "full"` diagrams and the
    icicle's sublayer band.
  * Descriptions record how each block computes: `arch_residual()` groups
    the steps each residual connection skips; `arch_attn()` says which rows
    supply an attention step's keys, which tokens query in the last block,
    and its QK-norm, scaling and test key/value heads; `join` and
    `join_rows` say how a stage joins the stream; `role` and `feeds` mark
    a sublayer's inputs (RoPE tables, inducing points, CLS tokens) and the
    steps run once after a stack. Stages also gain `collapses` and
    `interleave`, and sublayers `params_all`, their count over every block
    of a stack.
  * Fact names are shared across backends, so the same fact has the same
    name everywhere: "Column-stage blocks", "Row-stage blocks" and "ICL
    layers" replace "Column blocks", "Row blocks", "Aggregator blocks" and
    "ICL blocks"; "Features per group" replaces "Feature group size" and
    Kumo's "Cell group"; "Feed-forward" (as "2 x width (1024)") replaces
    "MLP width"; "Embedding width" replaces Mitra's "Width" and Kumo's
    "Cell width"; Kumo's "Heads" is split into "ICL heads" and "Embedding
    heads". This changes `print()` output and the fact sheet.
  * The `n_test` symbol is defined for every backend and Mitra defines `n`
    and `p`; each description carries a `glossary`, which explains every
    abbreviation a diagram draws.
* Hooks for reweighting and selecting an ensemble after the fact, for
  add-on packages such as `tabfound.thinking`. Every backend has them:
  * `member_weights =` combines the members with a weighted mean. Members
    weighted zero are never run. Without it, every combine is the
    reference's, bit for bit. Kumo's regressor, whose reference combine
    is a trimmed mean, uses a plain weighted mean under weights.
  * New `predict_members()` returns each member's output before it is
    combined, and names how the backend combines them (`$combine`):
    * classifiers: raw logits, with each member's class permutation or
      shift undone;
    * TabPFN regressors: bucket probabilities on the shared grid;
    * TabICL and Kumo regressors: quantile grids;
    * TabFM and Mitra regressors: point predictions.
  * New `quantile_grid_stat()` summarises a TabICL or Kumo quantile grid
    the way those regressors do.
  * TabPFN members are the configs, so `tabular_classifier()` /
    `tabular_regressor()` take them as an in-memory list,
    `ensemble_configs =`, in place of `ensemble_configs_dir =`, and
    `generate_ensemble_configs_native()` returns that list when
    `output_dir` is left out. The other backends generate their members
    from their own settings and seed.
  * New `native_ensemble_presets()` returns the checkpoint's member menu,
    and `generate_ensemble_configs_native(presets = )` draws members from
    an edited one.
* `list_models()` and the download prompt now name TabPFN v2.5's licence,
  `tabpfn-2.5-license-v1.1`, as non-commercial, like v2.6, v3 and v3.5's,
  and TabICL's as BSD-3-Clause; both used to say only "see the repo". The
  README's model table says the same, and links each model's weights on
  Hugging Face.
* Fixed TabPFN regression ensembles losing probability mass at bar
  resolution. Every member's bar distribution is translated onto the
  shared border grid, and that translation took differences of float32
  CDF values and clamped them at zero, so buckets whose two CDF values
  rounded together came out with probability exactly 0 -- 2 to 6% of the
  bars near the truth on TabPFN v2.6, and an infinite log score wherever
  an observed value fell in one. It is now a port of the reference's
  replacement from `tabpfn` 9.1 (PriorLabs/TabPFN#1288): each bucket's
  mass is a sum of non-negative overlap terms, and the two outer source
  buckets are read as half-normal tails. Means, quantiles and samples move
  by little; densities and log scores move a lot. The full-estimator
  parity fixtures are now cut against `tabpfn` 9.2.0 (#27).
* `fit()` now refuses a target with missing or non-finite values, saying
  how many and at which rows. Before, every backend accepted one: a single
  `NA` in a regression target made every prediction `NA`, and a classifier
  fitted on an `NA` label, both without a warning (#25).
* `categorical_features = integer()` now declares that no column is
  categorical, and turns inference off; only `NULL` infers. Before, the
  two were the same, so a count or time step with fewer than four values
  was always taken as categorical on a table of more than 100 rows, and
  there was no way to say otherwise (#24). `tabfound()` on a frame with
  no factor columns still infers, as before. The inference rule is now
  documented in `tabular_classifier()` and `tabular_regressor()`.
* Fixed TabPFN ensembles failing when the configs from
  `generate_ensemble_configs_native()` were fitted on other rows than
  they were generated on -- a fold, a subsample, a time window. `fit()`
  inferred the categorical columns again on its own rows, and a column
  numeric on the whole table could look categorical on a slice of it,
  which ended in an error about missing mappings or a bare `stopifnot()`
  at predict time (#23). Each config now records the categorical columns
  and the predictor count it was generated for, and a fit uses that set.
  A `categorical_features` given to the model as well is checked against
  it, and a mismatch, or a member that does not fit the rows (a column
  constant on them, say), is an error naming the problem. Python dumps
  record no set and infer as before.
* Fixed the same TabPFN ensembles failing on a slice where a member's
  ordinal encoder takes fewer categorical columns than it did on the
  whole table. The `_common_categories` encoders keep a column only when
  its rarest level has at least ten rows (and, for
  `_very_common_categories`, it has fewer than n/10 levels), so a fold
  could drop one, and the fit stopped with "carries N code mappings, but
  encodes M". Configs now record which predictor each category-code
  mapping belongs to, and a fit picks the mappings by predictor. A column
  with fewer levels on the slice than its mapping was drawn for gets the
  mapping cut to a permutation of the codes it has, as the reference's
  own draw on those rows would give. Python dumps carry no such record
  and stay positional.
* Fixed wrong predictions from Kumo Tabular medium and large on Apple's
  `mps` device. libtorch's fused attention kernel on MPS returns wrong
  values for non-contiguous inputs when the query is shorter than the keys
  and the head size is 64 or more; medium and large's readout step is that
  case (small's head size is 32, so it was unaffected). `sdpa()` now makes
  its inputs contiguous on `mps`, and all three sizes match the CPU to
  float32 rounding. Other devices are unchanged.
* Fixed regression `predict()` returning too many values when the test
  rows span more than one `predict_chunk_size` chunk: the per-chunk means
  (or medians) were joined with `rbind()`, which recycled the last chunk,
  so 3,000 rows in chunks of 1,024 gave 3,072 numbers. Affected the Kumo
  Tabular and TabICL regressors' `type = "mean"` and `"median"`;
  classification, quantile output and the other backends were not
  affected. `chunk_apply()` now concatenates vectors and checks
  it returns one result per row.
* New backend `kumo` for NVIDIA Kumo Tabular (small, medium and large;
  classification and regression), ported from `structured-data-models`
  0.2.1 with weights from `nvidia/Kumo-Tabular` `v1.0.1` (OpenMDW-1.1).
  * Six catalogue entries, `kumo-tabular-{small,medium,large}-{classifier,regressor}`;
    the family name `"kumo-tabular"` is large, as in the reference.
    `download_model()` converts the published state dicts once with Python
    (`inst/python/kumo_convert_ckpt.py`), deriving the config from the
    tensor shapes, and now pins a Hub revision.
  * The full default recipe and ensemble (8 members by default, as on the
    model card), with the reference's random draws replayed from
    libtorch's own generator: on Apple arm64 the members are the
    reference's own. Classification averages member logits; regression
    takes a trimmed mean of 999-quantile grids, and `type = "mean"` is the
    grid's mean.
  * More than 10 classes run the reference's error-correcting output
    codes. Classes absent from the training rows come back with
    probability zero.
  * Missing values are handled inside the network (`na_action = "auto"`
    passes them through).
  * `kv_cache`, `tabfound_cache()` / `tabfound_save()` bundles,
    `predict_chunk_size`, `save_peak_memory_factor`, `row_chunk_size` and
    `col_chunk_size` all work; they move outputs by float32 rounding only.
  * Memory preflight calibrated on small; `suggest_chunk_sizes()` covers
    Kumo, and the preflight suggests a smaller Kumo size before a smaller
    context. Medium and large estimates reuse small's constants.
  * Parity: the network, the recipe (weight-free, every member and
    codebook), and end-to-end predictions on the small checkpoints,
    including a 15-class case; see `inst/parity/README.md`.
  * Architecture diagrams for small and medium in
    `vignette("architecture-gallery")`.
* `estimate_peak_memory()` takes `n_classes`; `fit()` and `predict()` pass
  it to the memory guard.
* `sdpa()` passes `enable_gqa` to the fused kernel.
* The TabICL backend now follows `tabicl` 2.2.0 (previously 2.1.1). The
  network is unchanged; the wrapper changes are ported:
  * An entirely missing training column is zero-filled rather than
    dropped, and a table whose every column is constant or missing keeps
    one column and predicts the target's marginal distribution instead of
    raising an error.
  * A config without `row_rope_interleaved` gets the reference's default
    (`TRUE`) rather than `FALSE`.
  * Fixed a crash in the Latin-square feature shuffle with a single
    column.
* New stored end-to-end parity check against `TabICLClassifier` /
  `TabICLRegressor` (`inst/parity/tabicl_e2e_reference.py`,
  `tests/testthat/test-parity-tabicl-e2e.R`), covering the cases above and
  a test batch with an entirely missing column.
* The wrapper-layer parity dump no longer depends on Python's hash seed.
