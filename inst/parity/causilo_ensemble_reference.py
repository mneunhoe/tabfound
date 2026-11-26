#!/usr/bin/env python3
"""Weight-free reference for the Causilo 1.5 preprocessing and ensemble.

Runs `causilo`'s own `PreparedDataset.from_inputs` (the estimator's
`fit()` without the network) and `query_table` on a set of small tables,
and dumps what each ensemble member would hand the network: the signed
float32 feature table, context rows first, and the member targets. No
checkpoint is read. `tests/testthat/test-prep-causilo.R` replays it.

Categorical columns are given to the reference as strings whose sorted
order is the order of their integer codes, so the R side, which is handed
the codes, sees the same categories in the same order.

    .venvs/causilo/bin/python -I inst/parity/causilo_ensemble_reference.py \\
        --fixture-dir inst/parity/fixtures \\
        --out inst/parity/ensemble/causilo_ensemble.safetensors
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from causilo.v1_5.data.dataset import PreparedDataset, join_context

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

fixture_dir = Path(args.fixture_dir)
out: dict[str, torch.Tensor] = {}
cases: dict[str, dict] = {}


def label(k: float) -> object:
    # Zero-padded, so string order is code order.
    return np.nan if np.isnan(k) else f"c{int(k):03d}"


def frame(x: np.ndarray, categorical: list[int]) -> pd.DataFrame:
    cols = {}
    for j in range(x.shape[1]):
        if j in categorical:
            cols[f"x{j}"] = pd.Series([label(v) for v in x[:, j]], dtype=object)
        else:
            cols[f"x{j}"] = x[:, j]
    return pd.DataFrame(cols)


def run_case(name, x_train, y_train, x_test, task, n_estimators, categorical=(), seed=42):
    categorical = list(categorical)
    ds = PreparedDataset.from_inputs(
        frame(x_train, categorical), y_train, None, task=task,
        n_estimators=n_estimators, random_state=seed, retain_preprocessing=True,
    )
    query = ds.query_table(frame(x_test, categorical), slice(0, len(x_test)))
    table = join_context(ds.training_table(), query)
    members = slice(None)
    x = ds.signed_views(table, members)
    y = ds.targets_for(members)
    out[f"{name}.x_train"] = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float64))
    out[f"{name}.x_test"] = torch.from_numpy(np.ascontiguousarray(x_test, dtype=np.float64))
    out[f"{name}.y_train"] = torch.from_numpy(np.asarray(y_train, dtype=np.float64))
    out[f"{name}.members_x"] = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    out[f"{name}.members_y"] = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    plan = ds.ensemble
    cases[name] = {
        "task": task, "n_estimators": n_estimators, "seed": seed,
        "categorical": [c + 1 for c in categorical],
        "count": int(len(plan.feature_signs)),
        "views": list(plan.feature_views),
        "trim": int(plan.trim_count),
        "column_order": [int(i) + 1 for i in ds.columns.order],
        "groups": [int(g) + 1 for g in ds.columns.groups],
        "classes": None if task != "classification" else
            [str(c) for c in ds.target_preprocessor.encoder_.classes_],
        "target_limits": None if ds.target_limits is None else list(ds.target_limits),
    }


def fixture(name):
    fx = load_file(str(fixture_dir / f"{name}.safetensors"))
    return (fx["x_train"].double().numpy(), fx["y_train"].double().numpy(),
            fx["x_test"].double().numpy())


manifest = json.loads((fixture_dir / "manifest.json").read_text())["fixtures"]

for fx_name, counts in (("clf_tiny", (8,)), ("clf_binary_missing", (1, 2, 4, 5, 8)),
                        ("reg_tiny", (1, 2, 4, 5, 8)), ("clf_categorical", (8,)),
                        ("reg_categorical", (5,))):
    xtr, ytr, xte = fixture(fx_name)
    info = manifest[fx_name]
    cat = [c - 1 for c in info.get("categorical_features", [])]
    task = info["task"]
    if task == "classification":
        ytr = ytr.astype(np.int64)
    for n_est in counts:
        run_case(f"{fx_name}_e{n_est}", xtr, ytr, xte, task, n_est, cat)

# Edge columns: a duplicated pair, ties, a column whose first value is
# missing, a constant with missing cells, a pure constant (dropped), an
# all-missing column (dropped), a 60-category column (log counts), and
# a -0.0.
rng = np.random.default_rng(7)
n, q = 120, 30
base = rng.normal(size=n + q)
xs = np.column_stack([
    base,                                   # 0
    base,                                   # 1 duplicate of 0
    np.round(rng.normal(size=n + q)),       # 2 ties
    rng.exponential(size=n + q) * 1e3,      # 3 skewed, large
    np.full(n + q, 2.5),                    # 4 constant, dropped
    np.where(rng.random(n + q) < 0.3, np.nan, 1.0),   # 5 constant + missing
    np.full(n + q, np.nan),                 # 6 all missing, dropped
    rng.integers(0, 60, size=n + q).astype(float),    # 7 60 categories
    rng.integers(0, 4, size=n + q).astype(float),     # 8 4 categories
])
xs[0, 2] = np.nan                           # first value missing
xs[3, 3] = -0.0
xs[rng.random(n + q) < 0.1, 0] = np.nan
xs[rng.random(n + q) < 0.1, 7] = np.nan
xs[n + 2, 8] = 9                            # an unseen category in a query
y_cls = (base + rng.normal(size=n + q) > 0).astype(np.int64)[:n] * 2 + rng.integers(0, 2, n)
y_reg = (3 * base + rng.normal(size=n + q))[:n] * 100 + 5e4
for n_est in (1, 2, 5, 8):
    run_case(f"edge_clf_e{n_est}", xs[:n], y_cls, xs[n:], "classification", n_est, [7, 8])
    run_case(f"edge_reg_e{n_est}", xs[:n], y_reg, xs[n:], "regression", n_est, [7, 8])
run_case("edge_clf_seed0", xs[:n], y_cls, xs[n:], "classification", 8, [7, 8], seed=0)

# One retained feature and two classes: at most 4 distinct members, so
# 8 requested fit 4 and the plan takes its rare branch.
xo = rng.normal(size=(40, 2)); xo[:, 1] = 1.0
run_case("one_feature_e8", xo[:30], rng.integers(0, 2, 30), xo[30:], "classification", 8)
run_case("one_feature_reg_e8", xo[:30], rng.normal(size=30), xo[30:], "regression", 8)

path = Path(args.out)
path.parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous().clone() for k, v in out.items()}, str(path))
with open(path, "rb") as src, gzip.GzipFile(str(path) + ".gz", "wb", mtime=0) as dst:
    shutil.copyfileobj(src, dst)
os.remove(path)
meta = {"cases": cases, "causilo_version": importlib.metadata.version("causilo"),
        "numpy_version": np.__version__}
with open(str(path).replace(".safetensors", ".json"), "w") as fh:
    json.dump(meta, fh, indent=1)
print(f"[py] {len(cases)} cases -> {path}.gz")
