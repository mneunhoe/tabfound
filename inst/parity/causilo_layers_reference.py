#!/usr/bin/env python3
"""Weight-free reference outputs for the Causilo 1.5 network.

Builds `causilo`'s own `Model` at a tiny width with random weights, runs
it the way the estimators do (`ModelRunner._predict_direct`, CPU, float32,
inference mode) on a small table that exercises the edge cases, and dumps
the weights, inputs and stage outputs together, so
`tests/testthat/test-backend-causilo.R` can grade the R port with no
checkpoint at all. One classifier and one regressor, two members each.

The input table has: ties, a missing cell in context and query, an
infinite cell, a column whose context is all missing (its ranks are NaN
throughout), and a column that is constant. The 256 column latents and
999 quantiles are module constants, so even this net carries them.

    .venvs/causilo/bin/python -I inst/parity/causilo_layers_reference.py \\
        --out inst/parity/layers/causilo_layers.safetensors
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from causilo.v1_5.execution.runner import ModelRunner
from causilo.v1_5.nn.ecdf import ecdf_ranks, fit_ecdf_context
from causilo.v1_5.nn.layers.functional import unit_rms
from causilo.v1_5.nn.layers.scaling import bounded_multiplier
from causilo.v1_5.nn.model import Model
from causilo.v1_5.nn.model_config import ModelConfig

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()

torch.manual_seed(0)
out: dict[str, torch.Tensor] = {}
meta: dict = {}

# --- functional pieces ------------------------------------------------------
x = torch.randn(3, 2, 5, 8) * 3
x[0, 0, 0] = 0.0                     # an all-zero head: the eps clamp
out["unit_rms.x"] = x
out["unit_rms.y"] = unit_rms(x)
v = torch.linspace(-20, 20, 41)
out["bounded.x"] = v
out["bounded.y8"] = bounded_multiplier(v, 8.0)
out["bounded.y2"] = bounded_multiplier(v, 2.0)

# Ranks: [B=2, C=3, R=9, 1]; 6 context rows.
nan, inf = float("nan"), float("inf")
cells = torch.tensor([
    [[1, 2, 2, nan, 3, 2, 2, 0.5, nan],      # ties, missing, below range
     [nan, nan, nan, nan, nan, nan, 1, 2, nan],  # no observed context
     [5, -inf, 5, 5, inf, 1, 5, 7, -inf]],   # infinite cells
    [[0, 0, 0, 0, 0, 0, 0, 1, -1],           # constant context
     [1, 2, 3, 4, 5, 6, 3.5, 6, 100],
     [-1, nan, -1, 2, 2, 2, 2, -1, 0]],
]).unsqueeze(-1)
ctx = fit_ecdf_context(cells, 6)
out["ecdf.cells"] = cells
out["ecdf.context"] = ctx
out["ecdf.ranks"] = ecdf_ranks(cells, ctx)

# --- tiny networks ----------------------------------------------------------
B, n_ctx, n_query, C = 2, 7, 4, 4
table = torch.randn(B, n_ctx + n_query, C)
table[:, 2, 1] = nan                 # a missing context cell
table[0, n_ctx + 1, 0] = nan         # a missing query cell
table[:, :, 2] = torch.round(table[:, :, 2])   # ties
table[:, :n_ctx, 3] = nan            # no observed context in column 3
table[1, 5, 0] = inf                 # an infinite cell
out["net.table"] = table

# The regressor shares the classifier's weights except the three
# task-shaped tensors, which keeps the stored file small; `reg.w.*` holds
# only those.
shared: dict[str, torch.Tensor] = {}
for tag, classes in (("clf", 10), ("reg", 0)):
    model = Model(ModelConfig(max_classes=classes, width=16, column_depth=2,
                              prediction_depth=2))
    for name, p in model.named_parameters():
        with torch.no_grad():
            if name in shared and shared[name].shape == p.shape:
                p.copy_(shared[name])
            else:
                p.copy_(torch.randn_like(p) * 0.3)
                out[f"{tag}.w.{name}"] = p.detach().clone()
    model.eval()
    if not shared:
        shared = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if classes:
        targets = torch.randint(0, 3, (B, n_ctx)).float()
    else:
        targets = torch.randn(B, n_ctx)
    out[f"{tag}.targets"] = targets

    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _i, o):
            captured[name] = (o[0] if isinstance(o, tuple) else o).detach().clone()
        return fn

    model.feature_embedding.register_forward_hook(hook("embed"))
    for i, layer in enumerate(model.columns.layers):
        layer.register_forward_hook(hook(f"col{i}"))
    for i, layer in enumerate(model.row.layers):
        layer.register_forward_hook(hook(f"row{i}"))
    for j, layer in enumerate(model.prediction.layers):
        layer.register_forward_hook(hook(f"pred{j}"))
    runner = ModelRunner(model, torch.device("cpu"))
    with torch.inference_mode():
        res = runner._predict_direct(table, targets, num_classes=classes, weights=None)
    for k, t in captured.items():
        out[f"{tag}.{k}"] = t
    out[f"{tag}.out"] = res.detach().clone()
    meta[f"{tag}_classes"] = classes

meta.update(n_ctx=n_ctx, n_query=n_query, width=16, column_depth=2,
            prediction_depth=2,
            torch_version=torch.__version__,
            causilo_version=importlib.metadata.version("causilo"))

path = Path(args.out)
path.parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, str(path))
with open(path, "rb") as src, gzip.GzipFile(str(path) + ".gz", "wb", mtime=0) as dst:
    shutil.copyfileobj(src, dst)
os.remove(path)
with open(str(path).replace(".safetensors", ".json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"[py] {len(out)} tensors -> {path}.gz")
