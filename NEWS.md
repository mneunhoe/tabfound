# tabfound (development version)

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
