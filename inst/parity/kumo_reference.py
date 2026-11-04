#!/usr/bin/env python3
"""Generate reference outputs for the Kumo Tabular backend from `sdm`.

Grades the *network*, not `sdm.models.KumoTabular`: the public model adds
the preprocessing recipe and the ensemble, which are a separate port.
The network is called the way the recipe calls it -- `x` with NaN kept,
`y` as 0-based class codes or a standardized target, and a per-column
categorical mask -- under `torch.no_grad()`, which selects the in-place
inference path users actually get. Stage outputs are dumped under the
names `R/backend-kumo.R` writes when `TABFOUND_DUMP_DIR` is set:

    kumo_cells    cell embedding, before the target is added  [1, R, C, D]
    kumo_col{i}   column block i                              [1, C, R, D]
    kumo_row{i}   row block i (last: readout tokens only)     [1, R, K+C | K, D]

(`--stages minimal`, the default, keeps only `kumo_col0` / `kumo_row0`.)
    kumo_reps     row embedding                               [1, R, K*D]
    kumo_logits   network output, query rows                  [1, Rq, 10 | 999]

Needs `structured-data-models` (tested with 0.2.1); see
`inst/parity/README.md` for the `.venvs/sdm` setup.

Usage
-----
    .venvs/sdm/bin/python inst/parity/kumo_reference.py \\
        --fixture-dir inst/parity/fixtures --fixture clf_iris \\
        --ckpt <...>/small/classifier.pt --size small \\
        --out inst/parity/reference/kumo/clf_iris
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from sdm.models.kumo.tabular.model import MODEL_KWARGS, _KumoTabular

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--ckpt", required=True, help="The original .pt state dict.")
parser.add_argument("--size", required=True, choices=sorted(MODEL_KWARGS))
parser.add_argument("--out", required=True)
parser.add_argument("--stages", choices=["minimal", "all"], default="minimal",
                    help="`minimal` keeps the first column and row block "
                         "of the embedding; `all` keeps every layer.")
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
n_features = X_train.shape[1]

# The manifest's column indices are R's, 1-based.
cat_cols = [int(c) - 1 for c in info.get("categorical_features", [])]
categorical_mask = torch.zeros(n_features, dtype=torch.bool)
categorical_mask[cat_cols] = True

y_scaler = None
if TASK == "classification":
    y = fx["y_train"].long()
else:
    # The recipe standardizes the target in float32 with the population
    # SD before the network sees it; do the same so the regressor runs on
    # the scale it was trained on.
    y_raw = fx["y_train"].float()
    mean = y_raw.mean()
    std = y_raw.std(correction=0)
    if float(std) == 0.0:
        std = torch.ones_like(std)
    y = (y_raw - mean) / std
    y_scaler = (float(mean), float(std))

x = torch.cat([X_train, X_test], dim=0).unsqueeze(0)   # [1, R, C]
y = y.unsqueeze(0)                                      # [1, Rt]
mask = categorical_mask.unsqueeze(0)                    # [1, C]

model = _KumoTabular(
    num_classes=10 if TASK == "classification" else 0,
    num_quantiles=999 if TASK == "regression" else 0,
    **MODEL_KWARGS[args.size],
)
state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

captured: dict[str, torch.Tensor] = {}


def _hook(name):
    def fn(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        captured[name] = t.detach().clone().float().cpu().contiguous()
    return fn


re_ = model.row_embedding
hooks = [("kumo_cells", re_.cell_embedding), ("kumo_reps", re_)]
# Every layer's dump is [R, C, D] or larger, which is what makes the
# stored references big; the first layer of each kind is where a
# structural mistake shows, so that is what ships by default.
n_keep = len(re_.col_blocks) if args.stages == "all" else 1
hooks += [(f"kumo_col{i}", b) for i, b in enumerate(re_.col_blocks) if i < n_keep]
hooks += [(f"kumo_row{i}", b) for i, b in enumerate(re_.row_blocks) if i < n_keep]
for name, module in hooks:
    module.register_forward_hook(_hook(name))

with torch.no_grad():
    out = model(x.clone(), y, mask)

captured["kumo_logits"] = out.detach().float().cpu().contiguous()
for name, t in captured.items():
    safe_save_file({"t": t}, str(OUT / f"{name}.safetensors"))

meta = {
    "fixture": args.fixture,
    "task": TASK,
    "size": args.size,
    "torch_version": torch.__version__,
    "sdm_version": importlib.metadata.version("structured-data-models"),
    "n_train": n_train,
    "n_test": n_test,
    "categorical_features": [c + 1 for c in cat_cols],
    "stages": sorted(captured),
}
if TASK == "classification":
    n_classes = int(y.max().item()) + 1
    probs = torch.softmax(out[0, :, :n_classes], dim=-1)
    safe_save_file({"t": probs.contiguous()}, str(OUT / "final_probs.safetensors"))
    meta["n_classes"] = n_classes
else:
    grid = out[0] * y_scaler[1] + y_scaler[0]
    safe_save_file({"t": grid.contiguous()}, str(OUT / "final_quantiles.safetensors"))
    meta["y_scaler"] = {"mean": y_scaler[0], "scale": y_scaler[1]}

with open(OUT / "reference.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"[py] {args.fixture} ({args.size}): out {tuple(out.shape)} -> {OUT}")
