#!/usr/bin/env python3
"""End-to-end reference for the TabICL backend: the full public estimator.

`tabicl_reference.py` grades the bare network and `ensemble_reference.py`
the wrapper layer. This grades the two together, through
`TabICLClassifier` / `TabICLRegressor` exactly as a user calls them: mean
imputation, the unique-value filter, the 8-member ensemble, class-shuffle
inversion, logit averaging and the quantile head.

Besides an ordinary table it runs the cases tabicl 2.2.0 changed:

    base            60 x 6 training rows, 12 test rows, a few NaNs
    train_empty     one training column entirely missing
                    (kept and zero-filled since 2.2.0, then dropped as constant)
    all_const       every column constant (one column kept since 2.2.0,
                    so the model falls back to the target's marginal)
    all_missing     every training column entirely missing
    test_empty      one column entirely missing in the *test* batch
                    (2.1.1 masked it batch-wide; 2.2.0 imputes it)

Inputs and outputs go to one safetensors file so the R side reads the
same bytes.

Usage
-----
    .venvs/ref/bin/python inst/parity/tabicl_e2e_reference.py \\
        --clf-ckpt <...>/tabicl-classifier-v2-20260212.ckpt \\
        --reg-ckpt <...>/tabicl-regressor-v2-20260212.ckpt \\
        --out inst/parity/reference/tabicl_e2e
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from tabicl import TabICLClassifier, TabICLRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--clf-ckpt", required=True)
parser.add_argument("--reg-ckpt", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

rng = np.random.default_rng(2202)
n_tr, n_te, p = 60, 12, 6


def table(n):
    X = rng.normal(size=(n, p)) * np.array([1.0, 5.0, 0.3, 2.0, 20.0, 1.0])
    X[:, 3] = np.round(X[:, 3])
    return X


X_base, Xt_base = table(n_tr), table(n_te)
y_cls = (X_base[:, 0] + 0.3 * X_base[:, 1] + rng.normal(size=n_tr) > 0).astype(int)
y_cls[X_base[:, 2] > 0.3] = 2
y_reg = X_base[:, 0] * 2.0 - X_base[:, 3] + rng.normal(size=n_tr)

X_base[[3, 17], 1] = np.nan
Xt_base[5, 4] = np.nan

cases = {}
cases["base"] = (X_base, Xt_base)

X = X_base.copy(); X[:, 2] = np.nan
cases["train_empty"] = (X, Xt_base)

row = np.array([0.4, 1.0, 0.5, 118.2, -3.0, 7.0])
cases["all_const"] = (np.tile(row, (n_tr, 1)), np.tile(row, (n_te, 1)))

cases["all_missing"] = (np.full((n_tr, p), np.nan), Xt_base)

Xt = Xt_base.copy(); Xt[:, 1] = np.nan
cases["test_empty"] = (X_base, Xt)

QUANTILES = [0.1, 0.5, 0.9]
common = dict(n_estimators=8, random_state=42, device="cpu",
              use_amp=False, use_fa3=False, kv_cache=False)

T: dict[str, np.ndarray] = {"y_cls": y_cls.astype(np.float64),
                            "y_reg": y_reg.astype(np.float64)}
meta = {"cases": list(cases), "quantiles": QUANTILES,
        "estimator_kwargs": {k: v for k, v in common.items()},
        "tabicl_version": version("tabicl"),
        "torch_version": torch.__version__}

for name, (Xtr, Xte) in cases.items():
    T[f"{name}_X"] = Xtr
    T[f"{name}_Xt"] = Xte

    clf = TabICLClassifier(model_path=args.clf_ckpt, **common).fit(Xtr, y_cls)
    T[f"{name}_proba"] = clf.predict_proba(Xte).astype(np.float64)

    reg = TabICLRegressor(model_path=args.reg_ckpt, **common).fit(Xtr, y_reg)
    T[f"{name}_mean"] = reg.predict(Xte).astype(np.float64)
    T[f"{name}_median"] = reg.predict(Xte, output_type="median").astype(np.float64)
    T[f"{name}_quantiles"] = reg.predict(
        Xte, output_type="quantiles", alphas=QUANTILES).astype(np.float64)
    print(f"[py] {name}: proba {T[f'{name}_proba'].shape}, "
          f"mean {T[f'{name}_mean'][:3].round(3)}")

save_file({k: np.ascontiguousarray(v) for k, v in T.items()},
          str(OUT / "e2e.safetensors"))
with open(OUT / "reference.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"[py] -> {OUT}")
