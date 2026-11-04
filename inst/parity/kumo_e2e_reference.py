#!/usr/bin/env python3
"""End-to-end reference for the Kumo Tabular backend.

Runs the public `sdm.models.KumoTabular` -- recipe, ensemble, network,
output reduction -- with `num_estimators = 8` and a `torch.Generator`
seeded at 42, on CPU in float32, for the small checkpoints (loaded from
the original `.pt` files rather than the Hub). Classification
probabilities are put in class order (the reference leaves them in
member 0's rotated order); regression keeps the full 999-quantile grid.

Each case is also run through `fit()` then `predict()`, the reference's
cached path, and the gap to the one-shot forward is recorded.

    .venvs/sdm/bin/python inst/parity/kumo_e2e_reference.py \\
        --fixture-dir inst/parity/fixtures \\
        --clf-ckpt <...>/small/classifier.pt --reg-ckpt <...>/small/regressor.pt \\
        --out inst/parity/reference/kumo_e2e
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from sdm import Stype, TableTensor
from sdm.models import KumoTabular

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--clf-ckpt", required=True)
parser.add_argument("--reg-ckpt", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--members", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

models = {}
for task, ckpt in (("classification", args.clf_ckpt), ("regression", args.reg_ckpt)):
    m = KumoTabular(task, size="small", pretrained=False, device="cpu")
    m.models[task].load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    models[task] = m

out: dict[str, torch.Tensor] = {}
meta: dict = {"members": args.members, "seed": args.seed, "size": "small",
              "torch_version": torch.__version__,
              "sdm_version": importlib.metadata.version("structured-data-models"),
              "cases": {}}


def tables(x_train, y_train, x_test, task, cat):
    def frame(x):
        cols, st = {}, {}
        for j in range(x.shape[1]):
            nm = f"x{j + 1}"
            cols[nm] = x[:, j]
            st[nm] = "categorical" if j in cat else "numerical"
        df = pd.DataFrame(cols)
        for j in cat:
            df[f"x{j + 1}"] = df[f"x{j + 1}"].astype("category")
        return df, st
    dtr, st = frame(x_train)
    dte, _ = frame(x_test)
    for j in cat:
        nm = f"x{j + 1}"
        allc = pd.Index(sorted(set(dtr[nm].cat.categories) | set(dte[nm].cat.categories)))
        dtr[nm] = dtr[nm].cat.set_categories(allc)
        dte[nm] = dte[nm].cat.set_categories(allc)
    xc = TableTensor.from_pandas(dtr, st)
    xq = TableTensor.from_pandas(dte, st)
    if task == "classification":
        yc = TableTensor.from_pandas(
            pd.DataFrame({"y": pd.Series(y_train.astype(np.int64)).astype("category")}),
            {"y": "categorical"})
    else:
        yc = TableTensor.from_pandas(pd.DataFrame({"y": y_train}), {"y": "numerical"})
    return xc, yc, xq


def result(t: TableTensor, task):
    res = t.numerical
    cols = list(t.columns[Stype.numerical])
    if task == "classification":
        order = sorted(range(len(cols)), key=lambda i: int(cols[i]))
        return res[..., order].contiguous(), [int(cols[i]) for i in order]
    return res.contiguous(), None


def run(name, x_train, y_train, x_test, task, cat=()):
    cat = list(cat)
    xc, yc, xq = tables(x_train, y_train, x_test, task, cat)
    m = models[task]
    with torch.no_grad():
        fwd = m(xc, yc, xq, num_estimators=args.members,
                generator=torch.Generator().manual_seed(args.seed))
        m.fit(xc, yc, num_estimators=args.members,
              generator=torch.Generator().manual_seed(args.seed))
        pred = m.predict(xq)
    r_fwd, classes = result(fwd, task)
    r_pred, _ = result(pred, task)
    p = f"{name}."
    out[p + "x_train"] = torch.as_tensor(x_train, dtype=torch.float64)
    out[p + "x_test"] = torch.as_tensor(x_test, dtype=torch.float64)
    out[p + "y_train"] = torch.as_tensor(y_train, dtype=torch.float64)
    out[p + "out"] = r_fwd.float()
    gap = float((r_fwd - r_pred).abs().max()) if r_fwd.numel() else 0.0
    meta["cases"][name] = {"task": task, "categorical": [c + 1 for c in cat],
                           "classes": classes, "fit_predict_gap": gap,
                           "shape": list(r_fwd.shape)}
    print(f"[py] {name}: {tuple(r_fwd.shape)}  fit/predict gap {gap:.3e}")


fixture_dir = Path(args.fixture_dir)
manifest = json.loads((fixture_dir / "manifest.json").read_text())
for fx in ("clf_binary_missing", "clf_categorical", "reg_skewed", "reg_categorical"):
    info = manifest["fixtures"][fx]
    d = load_file(str(fixture_dir / f"{fx}.safetensors"))
    run(fx, d["x_train"].double().numpy(), d["y_train"].double().numpy(),
        d["x_test"].double().numpy(), info["task"],
        [int(c) - 1 for c in info.get("categorical_features", [])])

rng = np.random.default_rng(1)
base = rng.normal(size=(60, 4))
y3 = rng.integers(0, 3, size=60).astype(float)
yr = base[:, 0] * 2 + rng.normal(size=60)

# Every column constant: the recipe drops them all and the network sees
# no features at all.
run("all_const", np.ones((40, 3)), y3[:40], np.ones((5, 3)), "classification")
# A column missing in every training row, present at query time.
xm = base.copy(); xm[:45, 2] = np.nan
run("train_col_missing", xm[:45], yr[:45], xm[45:], "regression")
# Nothing to predict.
run("test_empty", base[:40], y3[:40], base[:0], "classification")
# Class 1 never occurs in the context: the reference has no column for it.
ya = y3[:45].copy(); ya[ya == 1] = 2
run("absent_class", base[:45], ya, base[45:], "classification")

save_file({k: v.contiguous().clone() for k, v in out.items()}, str(OUT / "e2e.safetensors"))
with open(OUT / "reference.json", "w") as fh:
    json.dump(meta, fh, indent=1)
print(f"[py] wrote {len(out)} tensors -> {OUT}")
