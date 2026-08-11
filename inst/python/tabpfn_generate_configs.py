#!/usr/bin/env python3
"""Generate ensemble configs for a specific (X_train, y_train) + n_estimators.

Run once per (dataset, n_estimators, random_state) tuple. The R-side
`load_tabpfn_{classifier,regressor}(..., ensemble_configs_dir=...)` then
consumes the output directory.

This script is a thinner cousin of `dump_ensemble.py` — it only dumps
the config artifacts (shuffle_perm, class_permutation, target_transform
lambdas, ensemble_configs.json). It does NOT run the model and does NOT
dump model outputs.

Inputs are read from a shared safetensors file with keys:
    x_train: float32 (n_train, n_features)
    y_train: float32 (n_train,)   — class labels (int-valued) for classifier
                                   — continuous for regressor

Usage:
    tabpfn/bin/python3 inst/python/generate_configs.py \\
        --ckpt ckpts/tabpfn-v2.5-classifier-v2.5_default.ckpt \\
        --split inst/extdata/debug_iris_split.safetensors \\
        --head classifier --n 4 --seed 0 \\
        --out debug/py_ensemble_classifier
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from tabpfn import TabPFNClassifier, TabPFNRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=Path, required=True,
                    help="Path to the .ckpt model file.")
parser.add_argument("--split", type=Path, required=True,
                    help="Safetensors file with x_train, y_train (at minimum).")
parser.add_argument("--head", choices=["classifier", "regressor"], required=True)
parser.add_argument("--n", type=int, required=True, help="n_estimators.")
parser.add_argument("--seed", type=int, default=0, help="random_state.")
parser.add_argument("--out", type=Path, required=True, help="Output directory.")
parser.add_argument("--no-target-transform", action="store_true",
                    help="Regressor only: disable target_transform members.")
args = parser.parse_args()

if not args.ckpt.exists():
    sys.exit(f"Missing ckpt: {args.ckpt}")
if not args.split.exists():
    sys.exit(f"Missing split: {args.split}")

args.out.mkdir(parents=True, exist_ok=True)

split = safe_load_file(str(args.split))
X_train = split["x_train"].numpy()
dtype_y = "int64" if args.head == "classifier" else "float32"
y_train = split["y_train"].numpy().astype(dtype_y)

# ---------- Build the estimator, fit to trigger preprocessor fit ----------
torch.manual_seed(args.seed)
np.random.seed(args.seed)

Est = TabPFNClassifier if args.head == "classifier" else TabPFNRegressor
est_kwargs = dict(
    model_path=str(args.ckpt.absolute()),
    n_estimators=args.n,
    random_state=args.seed,
    device="cpu",
    inference_precision=torch.float32,
    softmax_temperature=1.0,
)
if args.head == "regressor" and args.no_target_transform:
    est_kwargs["inference_config"] = {"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,)}
est = Est(**est_kwargs)
est.fit(X_train, y_train)

# ---------- Capture shuffle_perm per member (fit+first transform) --------
from tabpfn.preprocessing.steps.shuffle_features_step import ShuffleFeaturesStep

_captured: dict[int, np.ndarray] = {}
_seen_ids: set[int] = set()
_orig_tx = ShuffleFeaturesStep._transform

def _shuffle_tx_traced(self, X, *, is_test=False):
    if id(self) not in _seen_ids:
        _seen_ids.add(id(self))
        perm = self.index_permutation_
        if isinstance(perm, torch.Tensor):
            perm_arr = perm.detach().cpu().numpy().astype("int64")
        else:
            perm_arr = np.array(perm, dtype=np.int64)
        idx = len(_captured)
        _captured[idx] = perm_arr
    return _orig_tx(self, X, is_test=is_test)

ShuffleFeaturesStep._transform = _shuffle_tx_traced

# Running predict() forces the pipelines to transform (= fit, then transform)
# and captures each member's shuffle permutation.
if args.head == "classifier":
    est.predict_proba(X_train[:1])
else:
    est.predict(X_train[:1])

# Restore original method.
ShuffleFeaturesStep._transform = _orig_tx

# ---------- Write per-member shuffle_perm + target_transform lambdas ------
for i, perm in _captured.items():
    d = args.out / f"member_{i:02d}"
    d.mkdir(parents=True, exist_ok=True)
    safe_save_file(
        {"t": torch.from_numpy(perm).contiguous()},
        str(d / "shuffle_perm.safetensors"),
    )

cfgs = getattr(est, "ensemble_configs_", []) or []
for i, c in enumerate(cfgs):
    tt = getattr(c, "target_transform", None)
    if tt is None:
        continue
    try:
        spt = tt.named_steps.get("input_transformer") if hasattr(tt, "named_steps") else tt
        if hasattr(spt, "lambdas_"):
            lmb = np.asarray(spt.lambdas_, dtype=np.float64)
            d = args.out / f"member_{i:02d}"
            d.mkdir(parents=True, exist_ok=True)
            safe_save_file(
                {"t": torch.from_numpy(lmb).contiguous()},
                str(d / "target_transform_lambdas.safetensors"),
            )
    except Exception:
        pass

# ---------- Dump ensemble_configs.json (for class_perm + preset metadata) ---
def _json_safe(o):
    if isinstance(o, np.ndarray):
        return {"__ndarray__": True, "dtype": str(o.dtype), "shape": list(o.shape),
                "vals": o.tolist()}
    if is_dataclass(o):
        return {k: _json_safe(v) for k, v in asdict(o).items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    try:
        json.dumps(o)
        return o
    except Exception:
        return repr(o)

with (args.out / "ensemble_configs.json").open("w") as f:
    json.dump([_json_safe(c) for c in cfgs], f, indent=2)

print(f"Wrote {len(cfgs)} member config(s) to {args.out}")
