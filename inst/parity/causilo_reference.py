#!/usr/bin/env python3
"""Generate reference outputs for the Causilo 1.5 network from `causilo`.

Grades the *network*, not `CausiloClassifier` / `CausiloRegressor`: the
estimators add the preprocessing recipe and the ensemble, which are graded
separately (`causilo_ensemble_reference.py`, `causilo_e2e_reference.py`).
The model has no `forward()`; it is driven the way the estimators drive
it, through `ModelRunner._predict_direct()` on the CPU in float32 under
`torch.inference_mode()`. The fixture's raw columns go in as one member
(`[1, R, C]`, NaN kept), the target as 0-based class codes or a
population-SD standardized float32 target. Stage outputs are dumped under
the names `R/backend-causilo.R` writes when `TABFOUND_DUMP_DIR` is set:

    causilo_embed   feature embedding, context targets added  [1, C, R, W]
    causilo_col{i}  column layer i                            [1, C, R, W]
    causilo_row{i}  row layer i: latents + features            [1, R, 4+C, W]
                    (the last: latents only)                   [1, R, 4, W]
    causilo_reps    pooled latents                             [1, R, 4W]
    causilo_pred{j} prediction layer j (the last: query rows)  [1, R | Rq, 4W]
    causilo_head    head output, query rows                    [1, Rq, 10 | 999]

(`--stages minimal`, the default, keeps `col0`, the last row layer, the
last prediction layer and the head, which holds `reference/causilo` to a
few MB; `all` keeps every stage.)

Needs `causilo` 1.5.0; see `inst/parity/README.md` for `.venvs/causilo`.

Usage
-----
    .venvs/causilo/bin/python inst/parity/causilo_reference.py \\
        --fixture-dir inst/parity/fixtures --fixture clf_tiny \\
        --model-dir <...>/causilo-1.5-classifier \\
        --out inst/parity/reference/causilo/clf_tiny
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from causilo.v1_5.execution.runner import ModelRunner
from causilo.v1_5.nn.checkpoint import build_model, validate_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--model-dir", required=True,
                    help="Directory with the published model.safetensors + config.json.")
parser.add_argument("--out", required=True)
parser.add_argument("--stages", choices=["minimal", "all"], default="minimal")
args = parser.parse_args()

torch.manual_seed(0)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

fixture_dir = Path(args.fixture_dir)
manifest = json.loads((fixture_dir / "manifest.json").read_text())
info = manifest["fixtures"][args.fixture]
TASK = info["task"]

fx = safe_load_file(str(fixture_dir / f"{args.fixture}.safetensors"))
X_train = fx["x_train"].float()
X_test = fx["x_test"].float()
n_train, n_test = X_train.shape[0], X_test.shape[0]

y_scaler = None
if TASK == "classification":
    y = fx["y_train"].long().float()
    n_classes = int(y.max().item()) + 1
else:
    y_raw = fx["y_train"].double()
    mean = y_raw.mean()
    std = y_raw.std(correction=0)
    if float(std) == 0.0:
        std = torch.ones_like(std)
    y = ((y_raw - mean) / std).float()
    y_scaler = (float(mean), float(std))
    n_classes = 0

table = torch.cat([X_train, X_test], dim=0).unsqueeze(0)   # [1, R, C]
targets = y.unsqueeze(0)                                   # [1, n]

model_dir = Path(args.model_dir)
config = json.loads((model_dir / "config.json").read_text())
model = build_model(validate_checkpoint(config),
                    safe_load_file(str(model_dir / "model.safetensors")))
model.eval()
if (config["task"] == "classification") != (TASK == "classification"):
    raise SystemExit(f"{model_dir} is a {config['task']} checkpoint, fixture is {TASK}")

captured: dict[str, torch.Tensor] = {}


def _hook(name):
    def fn(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        captured[name] = t.detach().clone().float().cpu().contiguous()
    return fn


L = len(model.columns.layers)
Lp = len(model.prediction.layers)
full = args.stages == "all"
keep_col = range(L) if full else [0]
keep_row = range(L) if full else [L - 1]
keep_pred = range(Lp) if full else [Lp - 1]
hooks = [("causilo_head", model.prediction.head)]
if full:
    hooks.append(("causilo_embed", model.feature_embedding))
hooks += [(f"causilo_col{i}", model.columns.layers[i]) for i in keep_col]
hooks += [(f"causilo_row{i}", model.row.layers[i]) for i in keep_row]
hooks += [(f"causilo_pred{j}", model.prediction.layers[j]) for j in keep_pred]
for name, module in hooks:
    module.register_forward_hook(_hook(name))


def _pool_hook(_module, _inputs, output):
    if full:
        captured["causilo_reps"] = output.detach().clone().float().cpu().contiguous()


# `pool_latents` is a method, not a module; wrap it to see its output.
_pool = model.row.pool_latents


def pool_latents(latents):
    out = _pool(latents)
    _pool_hook(None, None, out)
    return out


model.row.pool_latents = pool_latents

runner = ModelRunner(model, torch.device("cpu"))
with torch.inference_mode():
    out = runner._predict_direct(table, targets, num_classes=n_classes, weights=None)

captured["causilo_out"] = out.detach().float().cpu().contiguous()
for name, t in captured.items():
    safe_save_file({"t": t}, str(OUT / f"{name}.safetensors"))

meta = {
    "fixture": args.fixture,
    "task": TASK,
    "torch_version": torch.__version__,
    "causilo_version": importlib.metadata.version("causilo"),
    "n_train": n_train,
    "n_test": n_test,
    "stages": sorted(k for k in captured if k != "causilo_out"),
}
if TASK == "classification":
    probs = torch.softmax(out[0], dim=-1)
    safe_save_file({"t": probs.contiguous()}, str(OUT / "final_probs.safetensors"))
    meta["n_classes"] = n_classes
else:
    grid = out[0].double() * y_scaler[1] + y_scaler[0]
    safe_save_file({"t": grid.float().contiguous()}, str(OUT / "final_quantiles.safetensors"))
    meta["y_scaler"] = {"mean": y_scaler[0], "scale": y_scaler[1]}

with open(OUT / "reference.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"[py] {args.fixture}: out {tuple(out.shape)} -> {OUT}")
