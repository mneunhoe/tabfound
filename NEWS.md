# tabfound (development version)

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
