#!/usr/bin/env python3
"""Generate reference outputs for the TabFM backend from the PyPI `tabfm`.

Calls `TabFM.forward()` directly rather than going through
`TabFMClassifier`. That is deliberate: this script grades the *network*,
so it must not also drag in the sklearn wrapper's preprocessing and
ensembling. Those get their own comparison once they are ported.

Dumps the output of each of the six pipeline stages, under the same
names `R/backend-tabfm.R` writes when `TABFOUND_DUMP_DIR` is set:

    tabfm_cell    cell embedder            (B, T, H, E)
    tabfm_col1    first column stage       (B, T, H, E)
    tabfm_row1    first row stage          (B, T, num_cls + H, E)
    tabfm_col2    second column stage      (B, T, num_cls + H, E)
    tabfm_reps    second row stage         (B, T, num_cls * E)
    tabfm_logits  ICL decoder              (B, T, max_classes | 1)

Everything runs in **float32**. The released model is designed for
bfloat16 and `tabfm_v1_0_0.load()` casts to it by default; comparing a
bfloat16 reference against R torch's float32 would measure the dtype,
not the port.

Usage
-----
    .venvs/ref/bin/python inst/parity/tabfm_reference.py \\
        --fixture-dir inst/parity/fixtures --fixture clf_tiny \\
        --weights-dir <hf snapshot>/classification \\
        --out inst/parity/reference/tabfm/clf_tiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from tabfm.src.pytorch.model import TabFM

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--weights-dir", required=True,
                    help="Directory holding model.safetensors + config.json "
                         "(a `classification/` or `regression/` subfolder of "
                         "the Hub snapshot).")
parser.add_argument("--out", required=True)
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
WD = Path(args.weights_dir)

with open(WD / "config.json") as fh:
    cfg = json.load(fh)
TASK = "classification" if cfg.get("is_classifier", True) else "regression"

fx = safe_load_file(str(Path(args.fixture_dir) / f"{args.fixture}.safetensors"))
X_train = fx["x_train"].float()
X_test = fx["x_test"].float()
y_train = fx["y_train"].float()

n_train, n_test = X_train.shape[0], X_test.shape[0]

# The regressor networks operate in standardized target space -- both
# reference wrappers fit a StandardScaler on y and inverse-transform the
# output. Do the same here so the comparison covers the whole predictor,
# not just the network with an arbitrary target scale.
y_scaler = None
if TASK == "regression":
    y_np = y_train.numpy().astype("float64")
    y_mean = float(y_np.mean())
    y_std = float(y_np.std())          # population std, as sklearn uses
    if y_std == 0.0:
        y_std = 1.0
    y_scaler = (y_mean, y_std)
    y_train = torch.tensor((y_np - y_mean) / y_std, dtype=torch.float32)

# The network takes one concatenated sequence with the labelled rows
# first; unlabelled rows carry the -100 sentinel that both the cell
# embedder and the ICL y-encoder read as "no label".
x = torch.cat([X_train, X_test], dim=0).unsqueeze(0)
y = torch.cat([y_train, torch.full((n_test,), -100.0)], dim=0).unsqueeze(0)
train_size = torch.tensor([n_train], dtype=torch.long)

model = TabFM(**{k: v for k, v in cfg.items()
                 if k in TabFM.__init__.__kwdefaults__})
state = safe_load_file(str(WD / "model.safetensors"))
missing, unexpected = model.load_state_dict(state, strict=True), None
model.eval().float()

# Chunking is documented as exact and is a no-op below these sizes, but
# turn it off anyway so the reference has one fewer moving part.
for m in model.modules():
    for attr in ("row_chunk_size", "col_chunk_size", "ffn_chunk_size"):
        if hasattr(m, attr):
            setattr(m, attr, None)

captured: dict[str, torch.Tensor] = {}


def _hook(name):
    def fn(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        captured[name] = t.detach().clone().float().cpu().contiguous()
    return fn


for name, module in (("tabfm_cell", model.cell_embedder),
                     ("tabfm_col1", model.col_embedder),
                     ("tabfm_row1", model.row_interactor),
                     ("tabfm_col2", model.col_embedder_2),
                     ("tabfm_reps", model.row_interactor_2),
                     ("tabfm_logits", model.icl_predictor)):
    module.register_forward_hook(_hook(name))

with torch.no_grad():
    logits = model(x, y, train_size)

for name, t in captured.items():
    safe_save_file({"t": t}, str(OUT / f"{name}.safetensors"))

safe_save_file({"t": logits.detach().float().cpu().contiguous()},
               str(OUT / "final_logits.safetensors"))

if TASK == "classification":
    n_classes = int(y_train.max().item()) + 1
    probs = torch.softmax(logits[0, n_train:, :n_classes], dim=-1)
    safe_save_file({"t": probs.contiguous()}, str(OUT / "final_probs.safetensors"))
else:
    preds = logits[0, n_train:, 0] * y_scaler[1] + y_scaler[0]
    safe_save_file({"t": preds.contiguous()}, str(OUT / "final_preds.safetensors"))
    n_classes = None
    meta_y_scaler = {"mean": y_scaler[0], "scale": y_scaler[1]}

import tabfm  # noqa: E402

meta = {
    "fixture": args.fixture,
    "task": TASK,
    "torch_version": torch.__version__,
    "tabfm_version": getattr(tabfm, "__version__", "unknown"),
    "dtype": "float32",
    "n_train": n_train,
    "n_test": n_test,
    "n_classes": n_classes,
    "config": cfg,
    "stages": sorted(captured),
    "y_scaler": (None if y_scaler is None
                 else {"mean": y_scaler[0], "scale": y_scaler[1]}),
}
with open(OUT / "reference.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"[py] {args.fixture}: logits {tuple(logits.shape)} -> {OUT}")
