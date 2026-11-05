# tabfound (development version)

* New architecture style, `plot_architecture(style = "schematic")`: how
  each block runs, in the convention of the papers and Sebastian Raschka's
  LLM Architecture Gallery. The data flows upwards; every repeated block
  is opened into its steps in the order they run, with residual
  connections drawn where each network adds them; attention steps say
  where their keys come from and how they differ from plain attention;
  inputs such as RoPE tables, inducing points and QK-norm feed their step
  from the side; and facts point at the part they describe. Each backend
  has its own accent colour, kept readable in both themes.
* Architecture descriptions record how each block computes, not only
  what it contains: `arch_residual()` groups the steps each residual
  connection skips (pre-norm, post-norm or sandwich, as each network
  runs it); `arch_attn()` says which rows supply an attention step's
  keys, which tokens query in the last block, and whether it uses
  QK-norm, a scaling variant or fewer key/value heads for the test rows;
  and stages that join the stream rather than transform it say so
  (`join`: added, concatenated or prepended, and to which rows). Facts
  can be placed in a diagram through `ARCH_FACT_ANCHORS`. No diagram
  draws these yet; the schematic style will. The glossary gains GQA and
  MQA.
* A stage's sublayers are now the steps of its forward pass, in the order
  they run, checked against each network by a test that traces one
  forward pass. Norms are listed one by one where they run (no more
  "LayerNorm x2" or "RMSNorm x4"), whole attention-and-MLP blocks are
  split into their norms, attention and feed-forward, and both halves of
  every induced-attention block are shown in full. Sublayers gain `role`:
  `"input"` for what a step consumes (RoPE tables, inducing points, CLS
  tokens, named by `feeds`) and `"final"` for a step run once after the
  stack (a stack's output norm). This changes `detail = "full"` diagrams
  and the icicle's sublayer band.
* Hooks for reweighting and selecting a TabPFN ensemble after the fact,
  for add-on packages such as `tabfound.thinking`:
  * `tabular_classifier()` / `tabular_regressor()` take the ensemble as
    an in-memory list, `ensemble_configs =`, in place of
    `ensemble_configs_dir =`, and `generate_ensemble_configs_native()`
    returns that list when `output_dir` is left out.
  * `member_weights =` combines the members with a weighted mean. Members
    weighted zero are never run. Without it the combine is the
    reference's unweighted mean, bit for bit.
  * New `predict_members()` returns each member's output before it is
    combined: the raw logits for a classifier, the bucket probabilities on
    the shared grid for a regressor.
  * New `native_ensemble_presets()` returns the checkpoint's member menu,
    and `generate_ensemble_configs_native(presets = )` draws members from
    an edited one.

* Architecture descriptions corrected against the networks' forward
  passes, so every diagram style shows what the models run:
  * Kumo Tabular's column and row blocks alternate, one of each per layer;
    they were drawn as two stacks run one after the other. Stages gain
    `interleave`, and the diagrams tag such blocks "x N, alternating".
    Its readout tokens are noted as skipping the column blocks.
  * TabFM's CLS columns join after the first column stage, not before it,
    and its cell embedder's target embedding is now mentioned.
  * TabICL's and TabFM's final norm moves from the target encoder to the
    decoder, where it runs (`decoder(ln(x))`); TabICL's inducing points
    are learned vectors whose keys are the labelled rows.
  * TabPFN v2.5's target encoder joins as an extra column on every row,
    not added on the training rows, and its regressor's encoder is an
    MLP. TabPFN v2.6's positional subspace is a quarter of the model
    width (48 only at the published width).
* Architecture diagrams come in four styles: `plot_architecture(style = )`
  takes `"cards"` (the default, unchanged apart from the key below),
  `"tufte"` (a table with a dot per stage on a log scale shared by every
  diagram), `"glyphs"` (each stage as a small table showing the axis its
  attention runs over) and `"icicle"` (where the parameters are, as
  lengths). `options(tabfound.arch_style = )` sets the default.
* New `plot_architecture_diff()` draws two models side by side, pairing
  their stages by kind and attention axis and their facts by name. It
  takes models, fits or `tabfound_arch` objects in any mix.
* Every architecture diagram ends in a key that defines each symbol in
  its shapes, each abbreviation in its text and the notation of its
  style, so a figure shared on its own still reads. It replaces the fact
  sheet's symbol line, which was cut off at the panel's width. SVG output
  also carries the title and the key as `<title>` and `<desc>`.
* Architecture descriptions:
  * Fact names are shared across backends, so the same fact has the same
    name everywhere: "Column-stage blocks", "Row-stage blocks" and "ICL
    layers" replace "Column blocks", "Row blocks", "Aggregator blocks"
    and "ICL blocks"; "Features per group" replaces "Feature group size"
    and Kumo's "Cell group"; "Feed-forward" (as "2 x width (1024)")
    replaces "MLP width"; "Embedding width" replaces Mitra's "Width" and
    Kumo's "Cell width"; Kumo's "Heads" is split into "ICL heads" and
    "Embedding heads". This changes `print()` output and the fact sheet.
  * Stages gain `collapses`, marking the stage after which a model works
    on one vector per row.
  * Sublayers gain `params_all`, their count over every block of a stack,
    which unlike per-block count times repeats is right for parts shared
    by the whole stack (RoPE tables, output norms).
  * The `n_test` symbol is defined for every backend, and Mitra defines
    `n` and `p`; each description carries a `glossary`.
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
