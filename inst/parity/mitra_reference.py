#!/usr/bin/env python3
"""Generate reference outputs for the Mitra backend.

Grades the *network* (`Tab2D.forward`), not `MitraClassifier`: the
sklearn wrapper adds its own feature preprocessing and ensembling, which
is a separate port. Dumps the output of each pipeline stage, under the
same names `R/backend-mitra.R` writes when `TABFOUND_DUMP_DIR` is set:

    mitra_quantile  quantile-rank embedding of the support  (B, S, F)
    mitra_embedded  query after packing y with x            (B, Q, F+1, D)
    mitra_encoded   query after the 12 layers               (B, Q, F+1, D)
    mitra_logits    decoder output on the target column     (B, Q, dim_output)

Mitra lives inside `autogluon.tabular`, whose full dependency tree is
large and mostly irrelevant here. `--source-dir` should point at a
directory holding a minimal package shim with the four modules the model
actually needs (`_internal/config/enums.py`, `_internal/models/{base,
embedding,tab2d}.py`), copied verbatim from the AutoGluon repo. See
`inst/parity/README.md`.

Usage
-----
    PYTHONPATH=<shim dir> .venvs/ref/bin/python inst/parity/mitra_reference.py \\
        --fixture-dir inst/parity/fixtures --fixture clf_tiny \\
        --weights-dir <hf snapshot> \\
        --out inst/parity/reference/mitra/clf_tiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from mitrapkg._internal.models.tab2d import Tab2D

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--weights-dir", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
WD = Path(args.weights_dir)

with open(WD / "config.json") as fh:
    cfg = json.load(fh)
TASK = "classification" if cfg["task"].upper() == "CLASSIFICATION" else "regression"

fx = safe_load_file(str(Path(args.fixture_dir) / f"{args.fixture}.safetensors"))
X_train = fx["x_train"].float()
X_test = fx["x_test"].float()
y_train = fx["y_train"].float()
n_train, n_test = X_train.shape[0], X_test.shape[0]

# Mitra's preprocessor maps a regression target to [0, 1] by min-max
# (`normalize_y`) and inverts it on the way out -- min-max, not the
# standardization TabFM and TabICL use. Mirror it so the comparison
# covers the predictor, not just the network on an arbitrary scale.
y_scaler = None
if TASK == "regression":
    y_np = y_train.numpy().astype("float64")
    lo, hi = float(y_np.min()), float(y_np.max())
    rng = hi - lo if hi > lo else 1.0
    y_scaler = (lo, rng)
    y_train = torch.tensor((y_np - lo) / rng, dtype=torch.float32)

x_support = X_train.unsqueeze(0)
x_query = X_test.unsqueeze(0)
y_support = y_train.unsqueeze(0)

# No padding: one table, every row and column valid.
padding_features = torch.zeros((1, X_train.shape[1]), dtype=torch.bool)
padding_obs_support = torch.zeros((1, n_train), dtype=torch.bool)
padding_obs_query = torch.zeros((1, n_test), dtype=torch.bool)

model = Tab2D(
    dim=cfg["dim"], dim_output=cfg["dim_output"], n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"], task=cfg["task"],
    use_pretrained_weights=False, path_to_weights="", device="cpu",
)
model.load_state_dict(safe_load_file(str(WD / "model.safetensors")))
model.eval().float()

captured: dict[str, torch.Tensor] = {}


def _keep(name, t):
    captured[name] = t.detach().clone().float().cpu().contiguous()


# The quantile embedding returns a tuple; keep the support half.
_orig_quantile = model.x_quantile.forward


def _quantile_hook(*a, **k):
    out = _orig_quantile(*a, **k)
    _keep("mitra_quantile", out[0])
    return out


model.x_quantile.forward = _quantile_hook

# The layer stack runs on the packed (y, x) tensor; capture the query
# side going in and coming out.
_orig_layer0 = model.layers[0].forward
_orig_layer_last = model.layers[-1].forward


def _layer0_hook(support, query__, *a, **k):
    _keep("mitra_embedded", query__)
    return _orig_layer0(support, query__, *a, **k)


def _layer_last_hook(support, query__, *a, **k):
    out = _orig_layer_last(support, query__, *a, **k)
    _keep("mitra_encoded", out[1])
    return out


model.layers[0].forward = _layer0_hook
model.layers[-1].forward = _layer_last_hook

with torch.no_grad():
    logits = model(x_support, y_support, x_query,
                   padding_features, padding_obs_support, padding_obs_query)

_keep("mitra_logits", logits if logits.dim() == 3 else logits.unsqueeze(-1))

for name, t in captured.items():
    safe_save_file({"t": t}, str(OUT / f"{name}.safetensors"))

meta = {
    "fixture": args.fixture,
    "task": TASK,
    "torch_version": torch.__version__,
    "n_train": n_train,
    "n_test": n_test,
    "config": cfg,
    "stages": sorted(captured),
    "y_scaler": (None if y_scaler is None
                 else {"min": y_scaler[0], "range": y_scaler[1]}),
}

if TASK == "classification":
    n_classes = int(y_train.max().item()) + 1
    probs = torch.softmax(logits[0, :, :n_classes], dim=-1)
    safe_save_file({"t": probs.contiguous()}, str(OUT / "final_probs.safetensors"))
    meta["n_classes"] = n_classes
else:
    preds = logits[0] * y_scaler[1] + y_scaler[0]
    safe_save_file({"t": preds.contiguous().reshape(-1)},
                   str(OUT / "final_preds.safetensors"))

with open(OUT / "reference.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"[py] {args.fixture}: logits {tuple(logits.shape)} -> {OUT}")
