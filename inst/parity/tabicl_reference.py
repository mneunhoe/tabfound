#!/usr/bin/env python3
"""Generate reference outputs for the TabICL backend from the PyPI `tabicl`.

Grades the *network*, not `TabICLClassifier`: the sklearn wrapper adds
its own preprocessing and ensembling, which is a separate port. Dumps the
output of each of the three pipeline stages, under the same names
`R/backend-tabicl.R` writes when `TABFOUND_DUMP_DIR` is set:

    tabicl_col     column embedder   (B, T, C+G, E)
    tabicl_reps    row interactor    (B, T, C*E)
    tabicl_logits  ICL decoder       (B, T, max_classes | num_quantiles)

`TabICL.forward` dispatches on `self.training`: the training path is the
plain three-stage pipeline, while eval routes through an inference
manager that adds chunking and K/V caching. Dropout is 0.0 in the
released configs, so the training path is deterministic and is what gets
dumped here. The script also runs the eval path and records how far the
two agree — if that number ever grows, the "plain path" this port
implements has stopped being the path users actually get.

Usage
-----
    .venvs/ref/bin/python inst/parity/tabicl_reference.py \\
        --fixture-dir inst/parity/fixtures --fixture clf_iris \\
        --ckpt <...>/tabicl-classifier-v2-20260212.ckpt \\
        --out inst/parity/reference/tabicl/clf_iris
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from tabicl._model.tabicl import TabICL

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = dict(ckpt["config"])
TASK = "classification" if int(cfg.get("max_classes", 0)) > 0 else "regression"

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

x = torch.cat([X_train, X_test], dim=0).unsqueeze(0)   # (1, T, H)
y = y_train.unsqueeze(0)                                # (1, train_size)

model = TabICL(**cfg)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.float()

captured: dict[str, torch.Tensor] = {}


def _hook(name):
    def fn(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        captured[name] = t.detach().clone().float().cpu().contiguous()
    return fn


# Hook the decoder rather than `icl_predictor` itself: the predictor
# slices off the training rows before returning, and the R model dumps
# the full sequence, so hooking one level down keeps the two comparable.
for name, module in (("tabicl_col", model.col_embedder),
                     ("tabicl_reps", model.row_interactor),
                     ("tabicl_logits", model.icl_predictor.decoder)):
    module.register_forward_hook(_hook(name))

# Training mode selects the plain three-stage path. Dropout is 0.0 in the
# released configs, so this is deterministic.
model.train()
with torch.no_grad():
    train_out = model(x.clone(), y)

for name, t in captured.items():
    safe_save_file({"t": t}, str(OUT / f"{name}.safetensors"))

# The model's own return value is already sliced to the test rows.
test_logits = train_out.detach().float().cpu().contiguous()
safe_save_file({"t": test_logits.contiguous()}, str(OUT / "final_logits.safetensors"))

meta = {
    "fixture": args.fixture,
    "task": TASK,
    "torch_version": torch.__version__,
    "tabicl_version": __import__("importlib.metadata").metadata.version("tabicl"),
    "n_train": n_train,
    "n_test": n_test,
    "config": {k: v for k, v in cfg.items()},
    "stages": sorted(captured),
}

if TASK == "classification":
    n_classes = int(y_train.max().item()) + 1
    probs = torch.softmax(test_logits[0, :, :n_classes] / 0.9, dim=-1)
    safe_save_file({"t": probs.contiguous()}, str(OUT / "final_probs.safetensors"))
    meta["n_classes"] = n_classes
    meta["softmax_temperature"] = 0.9
else:
    grid = test_logits[0] * y_scaler[1] + y_scaler[0]
    safe_save_file({"t": grid.contiguous()},
                   str(OUT / "final_quantiles.safetensors"))
    meta["num_quantiles"] = int(cfg["num_quantiles"])
    meta["y_scaler"] = {"mean": y_scaler[0], "scale": y_scaler[1]}

# Cross-check: the eval path adds chunking and caching on top of the same
# math. Record the gap so a future divergence is visible rather than
# assumed away.
captured.clear()
model.eval()
with torch.no_grad():
    # tabicl >= 2.2.0 resolves an unset inference device to CUDA, XPU or
    # MPS when present; the weights here live on CPU, so pin it.
    from tabicl import InferenceConfig
    cpu = {"device": "cpu", "use_amp": False, "use_fa3": False}
    icfg = InferenceConfig()
    icfg.update_from_dict({"COL_CONFIG": cpu, "ROW_CONFIG": cpu, "ICL_CONFIG": cpu})
    eval_out = model(x.clone(), y, return_logits=True, inference_config=icfg)
train_test = train_out if train_out.shape[1] == n_test else train_out[:, n_train:]
ev = eval_out
if ev.shape[-1] != train_test.shape[-1]:
    ev = ev[..., : train_test.shape[-1]]
    train_test = train_test[..., : ev.shape[-1]]
gap = float((train_test - ev).abs().max().item())
# JSON has no NaN literal that strict parsers accept. A NaN here means the
# forward pass itself produced NaN -- TabICL's network has no missing-value
# handling of its own, so unimputed input propagates straight through.
meta["train_vs_eval_max_abs"] = gap if np.isfinite(gap) else None
meta["output_has_nan"] = not bool(np.isfinite(
    test_logits.numpy()).all())

with open(OUT / "reference.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"[py] {args.fixture}: logits {tuple(test_logits.shape)} "
      f"| train-vs-eval {meta['train_vs_eval_max_abs']:.3e} -> {OUT}")
