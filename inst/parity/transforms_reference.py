#!/usr/bin/env python3
"""Reference outputs for the individual column transforms.

The end-to-end parity run needs model weights; this one does not. It
pins each shared preprocessing step against the exact library the R port
was written to match (sklearn, NumPy, or TabPFN's own step classes), so
`tests/testthat/test-prep-transforms.R` can catch a regression in
`R/prep-transforms.R` without downloading a checkpoint.

    .venvs/ref/bin/python inst/parity/transforms_reference.py \
        --out inst/parity/transforms/transforms.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True,
                    help="Path to the .safetensors file; a sibling .json "
                         "holds the scalars.")
args = parser.parse_args()

rng = np.random.RandomState(11)
X = rng.randn(40, 5)
X[:, 4] = 2.5                      # constant column
X[3, 0] = np.nan
X[7, 1] = np.nan
X[9, 2] = np.inf
X_test = rng.randn(15, 5)
X_test[:, 4] = 2.5
X_test[1, 0] = np.nan

out: dict = {"X": X.tolist(), "X_test": X_test.tolist()}

# --- remove constant features -------------------------------------------
sel = np.logical_and(
    (X[0:1, :] == X).mean(axis=0) < 1.0, ~np.all(np.isnan(X), axis=0)
)
out["remove_constant_keep"] = [int(i) + 1 for i in np.where(sel)[0]]  # 1-indexed

# --- squashing scaler ----------------------------------------------------
from tabpfn.preprocessing.steps.squashing_scaler_transformer import (  # noqa: E402
    SquashingScaler,
)

sq = SquashingScaler(max_absolute_value=3.0)
Xs = X[:, sel]
out["squashing_train"] = np.asarray(sq.fit_transform(Xs)).tolist()
out["squashing_test"] = np.asarray(sq.transform(X_test[:, sel])).tolist()

# --- quantile transformer (uniform, adaptive n_quantiles) ----------------
from sklearn.preprocessing import QuantileTransformer  # noqa: E402

# sklearn's QuantileTransformer rejects non-finite input, and in the real
# pipeline it only ever sees data that an earlier step has already made
# finite -- so feed it a NaN-carrying but Inf-free view.
Xq = np.where(np.isinf(Xs), np.nan, Xs)
Xq_test = np.where(np.isinf(X_test[:, sel]), np.nan, X_test[:, sel])
out["quantile_X"] = Xq.tolist()
out["quantile_X_test"] = Xq_test.tolist()
n_q = max(Xq.shape[0] // 10, 2)
qt = QuantileTransformer(n_quantiles=n_q, output_distribution="uniform",
                         subsample=10**9, random_state=0)
out["quantile_n_quantiles"] = int(n_q)
out["quantile_train"] = np.asarray(qt.fit_transform(Xq)).tolist()
out["quantile_test"] = np.asarray(qt.transform(Xq_test)).tolist()

# --- quantile transformer on a constant column ---------------------------
# The fold `0.5 * (interp(x) - interp(-x))` is not the whole story:
# sklearn then pins values equal to the fitted min/max to 0/1, lower last.
# A constant column is both at once, so it comes out 0 rather than 0.5 --
# a half-unit difference that no well-spread column can reveal.
Xconst = np.column_stack([Xq[:, 0], np.full(Xq.shape[0], 2.5)])
qtc = QuantileTransformer(n_quantiles=max(Xconst.shape[0] // 5, 2),
                          output_distribution="uniform",
                          subsample=10**9, random_state=0)
out["quantile_const_X"] = Xconst.tolist()
out["quantile_const_n_quantiles"] = int(max(Xconst.shape[0] // 5, 2))
out["quantile_const_train"] = np.asarray(qtc.fit_transform(Xconst)).tolist()

# --- quantile transformer with OOD extrapolation -------------------------
# `quantile_uni_extrapolate`, which the v3 `_ood` regressor's recipe asks
# for. The ordinary transform clips everything outside the training range
# to 0 or 1, so "just past the edge" and "far outside" become the same
# number; this one continues linearly instead and clips only at
# -ratio / 1+ratio. The test block below is built to *have* out-of-range
# values -- without them the two presets are identical and the fixture
# would pin nothing.
from tabpfn.preprocessing.steps.adaptive_quantile_transformer import (  # noqa: E402
    AdaptiveQuantileTransformer,
    get_extrapolate_ratio_for_preset,
    get_user_n_quantiles_for_preset,
)

Xe = Xq[:, :3]                      # three well-spread columns, NaNs kept
Xe_test = Xq_test[:, :3].copy()
Xe_test[0, 0] = np.nanmin(Xe[:, 0]) - 5.0     # far below
Xe_test[1, 0] = np.nanmin(Xe[:, 0]) - 0.01    # just below
Xe_test[2, 1] = np.nanmax(Xe[:, 1]) + 5.0     # far above
Xe_test[3, 1] = np.nanmax(Xe[:, 1]) + 0.01    # just above
Xe_test[4, 2] = np.nanmin(Xe[:, 2])           # exactly at the boundary
Xe_test[5, 2] = np.nanmax(Xe[:, 2])
# A constant column has no range to extrapolate along and must be skipped.
Xe = np.column_stack([Xe, np.full(Xe.shape[0], 2.5)])
Xe_test = np.column_stack([Xe_test, np.full(Xe_test.shape[0], 7.5)])

n_qe = get_user_n_quantiles_for_preset("quantile_uni_extrapolate", Xe.shape[0])
ratio = get_extrapolate_ratio_for_preset("quantile_uni_extrapolate")
assert ratio == 1.0
qte = AdaptiveQuantileTransformer(
    n_quantiles=n_qe, output_distribution="uniform", subsample=10**9,
    random_state=0, extrapolate_ratio=ratio,
)
out["quantile_extrap_X"] = Xe.tolist()
out["quantile_extrap_X_test"] = Xe_test.tolist()
out["quantile_extrap_n_quantiles"] = int(n_qe)
out["quantile_extrap_ratio"] = float(ratio)
out["quantile_extrap_train"] = np.asarray(qte.fit_transform(Xe)).tolist()
out["quantile_extrap_test"] = np.asarray(qte.transform(Xe_test)).tolist()

# The same fit without extrapolation, so the test can assert the two
# actually differ where it matters and agree everywhere else.
qtp = QuantileTransformer(n_quantiles=n_qe, output_distribution="uniform",
                          subsample=10**9, random_state=0)
qtp.fit(Xe)
out["quantile_extrap_plain_test"] = np.asarray(qtp.transform(Xe_test)).tolist()

# --- polynomial features -------------------------------------------------
from tabpfn.preprocessing.steps.nan_handling_polynomial_features_step import (  # noqa: E402
    NanHandlingPolynomialFeaturesStep,
)

# Fed the NaN-carrying but Inf-free view: the step's StandardScaler
# tolerates NaN and rejects Inf, and in the real pipeline `clean_data`
# has already mapped Inf to NaN before anything here runs.
poly = NanHandlingPolynomialFeaturesStep(max_features=10, random_state=0)
poly._fit(Xq, None)
base_train, added_train, _ = poly._transform(Xq, is_test=False)
base_test, added_test, _ = poly._transform(Xq_test, is_test=True)
# 0-indexed, exactly as the ensemble dump writes them.
out["poly_factor_1"] = np.asarray(poly.poly_factor_1_idx).astype("int64").tolist()
out["poly_factor_2"] = np.asarray(poly.poly_factor_2_idx).astype("int64").tolist()
out["poly_train"] = np.column_stack([base_train, added_train]).tolist()
out["poly_test"] = np.column_stack([base_test, added_test]).tolist()

# --- categorical detection + ordinal encoding ----------------------------
from tabpfn.preprocessing.datamodel import FeatureModality  # noqa: E402
from tabpfn.preprocessing.modality_detection import (  # noqa: E402
    detect_feature_modalities,
)
from tabpfn.preprocessing.steps.encode_categorical_features_step import (  # noqa: E402
    _get_least_common_category_count,
)

# Five columns chosen to land on every side of the rules: 3 levels (auto-
# inferred), 6 and 15 levels (only a declaration reaches them), a
# continuous one, and one whose rarest level appears fewer than ten times
# (which the `common_categories` filters drop).
crng = np.random.RandomState(5)
n_cat = 150
cat_X = np.column_stack([
    crng.randint(0, 3, n_cat).astype(float),
    crng.randint(0, 6, n_cat).astype(float),
    crng.randint(0, 15, n_cat).astype(float),
    crng.randn(n_cat),
    np.where(np.arange(n_cat) < 3, 1.0, 0.0) + crng.randint(0, 2, n_cat) * 2.0,
])
cat_X[7, 1] = np.nan
out["cat_X"] = cat_X.tolist()

for label, declared in (("auto", None), ("declared", [0, 1, 2, 3, 4])):
    schema = detect_feature_modalities(
        cat_X,
        None,
        min_samples_for_inference=100,
        max_unique_for_category=30,
        min_unique_for_numerical=4,
        provided_categorical_indices=declared,
    )
    out[f"cat_detected_{label}"] = [
        int(i) + 1 for i in schema.indices_for(FeatureModality.CATEGORICAL)
    ]

out["cat_least_common"] = [
    int(_get_least_common_category_count(cat_X[:, j])) for j in range(cat_X.shape[1])
]

from sklearn.preprocessing import OrdinalEncoder  # noqa: E402

_enc_cols = [0, 1, 2]
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
out["cat_ordinal_train"] = np.asarray(
    oe.fit_transform(cat_X[:, _enc_cols])
).tolist()
out["cat_ordinal_n_categories"] = [len(c) for c in oe.categories_]
# A test batch with an unseen level in the first column, which must come
# back as NaN rather than erroring or wrapping around.
cat_X_test = cat_X[:10].copy()
cat_X_test[0, 0] = 99.0
cat_X_test[1, 1] = np.nan
out["cat_X_test"] = cat_X_test.tolist()
out["cat_ordinal_test"] = np.asarray(
    oe.transform(cat_X_test[:, _enc_cols])
).tolist()

# Shuffled codes: a fixed permutation per column, applied to non-NaN codes.
_perms = [np.array([2, 0, 1]), np.array([5, 3, 1, 0, 4, 2, 6]), np.arange(15)[::-1]]
# One key per column: the permutations are ragged (one entry per level),
# and the serializer below packs each value into a single rectangular array.
for k, perm in enumerate(_perms):
    out[f"cat_ordinal_perm_{k}"] = perm.astype("int64").tolist()
_shuffled = np.asarray(oe.fit_transform(cat_X[:, _enc_cols])).copy()
for k, perm in enumerate(_perms):
    col = _shuffled[:, k]
    ok = ~np.isnan(col)
    col[ok] = perm[col[ok].astype(int)].astype(col.dtype)
out["cat_ordinal_shuffled_train"] = _shuffled.tolist()

# --- SVD features --------------------------------------------------------
from tabpfn.preprocessing.steps.add_svd_features_step import (  # noqa: E402
    get_svd_features_transformer,
    get_svd_n_components,
)

Xsq = np.asarray(sq.fit_transform(Xs))
n_comp = get_svd_n_components("svd_quarter_components", *Xsq.shape)
svd = get_svd_features_transformer("svd_quarter_components", *Xsq.shape,
                                   random_state=0)
svd.fit(Xsq)
out["svd_n_components"] = int(n_comp)
out["svd_train"] = np.asarray(svd.transform(Xsq)).tolist()
out["svd_test"] = np.asarray(
    svd.transform(np.asarray(sq.transform(X_test[:, sel])))
).tolist()

# --- fingerprint ---------------------------------------------------------
from tabpfn.preprocessing.steps.add_fingerprint_features_step import (  # noqa: E402
    AddFingerprintFeaturesStep,
)

fp = AddFingerprintFeaturesStep()
fp.n_cells_ = Xsq.shape[0] * Xsq.shape[1]
res = fp._transform(Xsq, is_test=False)
out["fingerprint_salt"] = int(fp.n_cells_)
out["fingerprint_train"] = np.asarray(res[1]).ravel().tolist()
Xsq_test = np.asarray(sq.transform(X_test[:, sel]))
out["fingerprint_test"] = np.asarray(
    fp._transform(Xsq_test, is_test=True)[1]
).ravel().tolist()

# --- Yeo-Johnson ---------------------------------------------------------
from tabpfn.preprocessing.steps.safe_power_transformer import (  # noqa: E402
    SafePowerTransformer,
)

y = rng.gamma(2.0, 1.0, size=60)
spt = SafePowerTransformer(standardize=False)
yt = spt.fit_transform(y.reshape(-1, 1)).ravel()
out["yeojohnson_y"] = y.tolist()
out["yeojohnson_lambda"] = float(np.asarray(spt.lambdas_).ravel()[0])
out["yeojohnson_forward"] = yt.tolist()
out["yeojohnson_inverse"] = spt.inverse_transform(
    yt.reshape(-1, 1)
).ravel().tolist()

import tabpfn  # noqa: E402
import sklearn  # noqa: E402

out["_versions"] = {"tabpfn": tabpfn.__version__, "sklearn": sklearn.__version__,
                    "numpy": np.__version__}

# JSON cannot represent NaN or Infinity portably -- Python emits bare
# `NaN`/`Infinity` tokens that strict parsers (including jsonlite) reject
# -- and these fixtures exist precisely to exercise non-finite values. So
# the arrays go to safetensors as float64 and only scalars go to JSON.
tensors = {}
scalars = {}
for k, v in out.items():
    if isinstance(v, list) and v and isinstance(v[0], (list, float, int)):
        arr = np.asarray(v, dtype="float64")
        tensors[k] = torch.from_numpy(np.ascontiguousarray(arr))
    else:
        scalars[k] = v

out_path = Path(args.out)
save_file(tensors, str(out_path))
with open(out_path.with_suffix(".json"), "w") as fh:
    json.dump(scalars, fh, indent=2)
print(f"[py] wrote {len(tensors)} arrays to {out_path} "
      f"and {len(scalars)} scalars to {out_path.with_suffix('.json')}")
# Gzip the result. `file(1)` has a notoriously loose DOS-.COM heuristic --
# a first byte of 0xb8 reads as an x86 MOV -- and safetensors starts with
# a raw little-endian header length, so a benign tensor dump can be
# reported as "COM executable for DOS". `R CMD check` shells out to
# `file` and raises "Found the following executable file". Gzip's magic
# number is unambiguous, and this cannot regress when the contents change.
import gzip, os, shutil  # noqa: E402

with open(out_path, "rb") as fh_in, gzip.open(str(out_path) + ".gz", "wb") as fh_out:
    shutil.copyfileobj(fh_in, fh_out)
os.remove(out_path)
print(f"[py] gzipped -> {out_path}.gz")

