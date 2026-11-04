# tabfound (development version)

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
