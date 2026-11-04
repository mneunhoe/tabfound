#!/usr/bin/env python3
"""Reference members and output reduction for the Kumo Tabular recipe.

Runs `sdm`'s own `RecipeExecution(default_recipe())` -- the exact object
`KumoTabular.forward()` builds -- on fixture and synthetic tables, with
`num_members = 8` and a `torch.Generator` seeded at 42, and dumps every
member's network inputs: `x` (context rows then query rows, float32),
`y`, the categorical mask, the column order, and the class order. Then it
draws each member's ECOC codebook where there are more than 10 classes,
from the same generator after the recipe's draws (where
`KumoTabular.forward()` draws it), and pushes synthetic member outputs
through the reference's
`inverse_transform_target()` and `transform_output()`, so the reduction
(class realignment, trimmed mean) is pinned too.

No model weights are involved, so `tests/testthat/test-prep-kumo.R` runs
anywhere the package does.

    .venvs/sdm/bin/python inst/parity/kumo_ensemble_reference.py \\
        --fixture-dir inst/parity/fixtures \\
        --out inst/parity/ensemble/kumo_ensemble.safetensors
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

from sdm import Stype, TableTensor
from sdm.models import ECOC
from sdm.models.base import _categorical_mask
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.processing.execution import RecipeExecution

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--members", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

E = args.members
out: dict[str, torch.Tensor] = {}
meta: dict = {"members": E, "seed": args.seed, "cases": {}}


def frame(x: np.ndarray, cat: list[int]) -> tuple[pd.DataFrame, dict]:
    cols = {}
    stypes = {}
    for j in range(x.shape[1]):
        name = f"x{j + 1}"
        if j in cat:
            cols[name] = pd.Series(x[:, j]).astype("category")
            stypes[name] = "categorical"
        else:
            cols[name] = x[:, j]
            stypes[name] = "numerical"
    return pd.DataFrame(cols), stypes


def run_case(name: str, x_train: np.ndarray, y_train: np.ndarray,
             x_test: np.ndarray, task: str, cat: list[int]) -> None:
    df_tr, stypes = frame(x_train, cat)
    df_te, _ = frame(x_test, cat)
    # Categories are per frame in pandas; give the query frame the union so
    # unseen values reach AlignCategories as real categories.
    for j in cat:
        nm = f"x{j + 1}"
        allc = pd.Index(sorted(set(df_tr[nm].cat.categories) | set(df_te[nm].cat.categories)))
        df_tr[nm] = df_tr[nm].cat.set_categories(allc)
        df_te[nm] = df_te[nm].cat.set_categories(allc)
    xc = TableTensor.from_pandas(df_tr, stypes)
    xq = TableTensor.from_pandas(df_te, stypes)
    if task == "classification":
        yc = TableTensor.from_pandas(pd.DataFrame({"y": pd.Series(y_train.astype(np.int64)).astype("category")}),
                                     {"y": "categorical"})
    else:
        yc = TableTensor.from_pandas(pd.DataFrame({"y": y_train.astype(np.float64)}),
                                     {"y": "numerical"})

    ex = RecipeExecution(default_recipe())
    g = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        contexts = ex.fit_transform(x=xc, y=yc, related_tables=None,
                                    num_members=E, generator=g)
        queries = ex.transform(x=xq, related_tables=None)
        # What `ECOC.forward()` draws next, member by member, from the same
        # generator -- nothing in between consumes it.
        books = []
        if task == "classification":
            K = contexts[0].y.categorical.categories[0].numel()
            ecoc = ECOC(max_classes=10)
            if K > ecoc.max_classes:
                books = [ecoc._draw_codebook(K, "cpu", g) for _ in range(E)]

    p = f"{name}."
    out[p + "x_train"] = torch.as_tensor(x_train, dtype=torch.float64)
    out[p + "x_test"] = torch.as_tensor(x_test, dtype=torch.float64)
    out[p + "y_train"] = torch.as_tensor(y_train, dtype=torch.float64)
    case = {"task": task, "categorical": [c + 1 for c in cat],
            "n_train": int(x_train.shape[0]), "n_test": int(x_test.shape[0]),
            "members": []}
    member_outs = []
    gen = torch.Generator().manual_seed(7)
    # Only the first context rows ship (R still fits on all of them), and
    # the reduction is checked on a few query rows: the member tensors and
    # the 999-quantile outputs are what make the dump big.
    n_ctx = min(x_train.shape[0], 150)
    n_out = min(x_test.shape[0], 4)
    case["n_dump_ctx"], case["n_dump_out"] = n_ctx, n_out
    for m, (c, q) in enumerate(zip(contexts, queries)):
        x = torch.cat([c.x.numerical[:n_ctx], q.x.numerical], dim=-2).float()
        out[p + f"m{m}.x"] = x.contiguous()
        mask = _categorical_mask([c])
        out[p + f"m{m}.mask"] = mask.to(torch.uint8)
        info = {"columns": list(c.x.columns[Stype.numerical])}
        if task == "classification":
            out[p + f"m{m}.y"] = c.y.categorical.code.squeeze(-1).double()
            classes = c.y.categorical.categories[0].tolist()
            info["classes"] = [int(v) for v in classes]
            if books:
                out[p + f"m{m}.codebook"] = books[m].double()
            logits = torch.randn(n_out, len(classes), generator=gen)
            out[p + f"m{m}.out"] = logits
            member_outs.append(TableTensor(
                columns={Stype.numerical: [str(v) for v in classes]}, numerical=logits))
        else:
            out[p + f"m{m}.y"] = c.y.numerical.squeeze(-1).double()
            qs = torch.randn(n_out, 999, generator=gen)
            out[p + f"m{m}.out"] = qs
            member_outs.append(TableTensor(
                columns={Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]},
                numerical=qs))
        case["members"].append(info)

    with torch.no_grad():
        if task == "regression":
            member_outs = list(ex.inverse_transform_target(member_outs))
        final = ex.transform_output(member_outs)
    res = final.numerical
    cols = list(final.columns[Stype.numerical])
    if task == "classification":
        # The reference leaves the columns in member 0's rotated order;
        # put them in class order, as the AutoGluon wrapper does.
        order = sorted(range(len(cols)), key=lambda i: int(cols[i]))
        res = res[..., order]
        case["output_columns"] = [cols[i] for i in order]
    out[p + "final"] = res.contiguous().float()
    meta["cases"][name] = case
    print(f"[py] {name}: {len(contexts)} members, x {tuple(out[p + 'm0.x'].shape)}")


fixture_dir = Path(args.fixture_dir)
manifest = json.loads((fixture_dir / "manifest.json").read_text())
for fx in ("clf_binary_missing", "clf_categorical", "reg_skewed",
           "reg_categorical", "clf_iris", "clf_many_classes"):
    info = manifest["fixtures"][fx]
    d = load_file(str(fixture_dir / f"{fx}.safetensors"))
    run_case(fx, d["x_train"].double().numpy(), d["y_train"].double().numpy(),
             d["x_test"].double().numpy(), info["task"],
             [int(c) - 1 for c in info.get("categorical_features", [])])

rng = np.random.default_rng(0)

# A categorical with more than 50 levels gets a log-count column; unseen
# and missing query values share the "missing in context" count.
n = 160
hc = rng.integers(0, 70, size=n).astype(float)
hc[rng.choice(n, 6, replace=False)] = np.nan
x = np.column_stack([rng.normal(size=n), hc, rng.integers(0, 3, size=n).astype(float),
                     rng.lognormal(size=n)])
x_te = x[120:].copy()
x_te[0, 1] = 99.0        # unseen
x_te[1, 2] = np.nan
run_case("synth_highcard", x[:120], rng.integers(0, 4, size=120).astype(float),
         x_te, "classification", [1, 2])

# More than 500 columns: SelectColumns keeps the first 500 after the
# member's own shuffle, so members keep different subsets.
x = rng.normal(size=(30, 520))
run_case("synth_wide", x[:24], rng.normal(size=24), x[24:], "regression", [])

# More than 8192 context rows: RankGaussian spaces its knots, on a column
# with many distinct values and pads on one with few.
n = 9100
x = np.column_stack([rng.standard_t(3, size=n), rng.integers(0, 40, size=n).astype(float)])
run_case("synth_tall", x[:9000], rng.normal(size=9000), x[9000:], "regression", [])

# Awkward columns: +-Inf, a constant, an all-missing one, [5, NaN, 5],
# an all-negative skewed one; and a constant target.
n = 40
x = np.column_stack([
    rng.normal(size=n), np.full(n, 3.0), np.full(n, np.nan),
    np.where(np.arange(n) % 7 == 0, np.nan, 5.0), -rng.lognormal(size=n),
    rng.normal(size=n),
])
x[3, 0] = np.inf
x[5, 5] = -np.inf
run_case("synth_edge", x[:30], np.full(30, 2.5), x[30:], "regression", [])

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous().clone() for k, v in out.items()}, str(out_path))
meta["torch_version"] = torch.__version__
meta["sdm_version"] = importlib.metadata.version("structured-data-models")
with open(out_path.with_suffix(".json"), "w") as fh:
    json.dump(meta, fh, indent=1)
with open(out_path, "rb") as fh_in, gzip.open(str(out_path) + ".gz", "wb") as fh_out:
    shutil.copyfileobj(fh_in, fh_out)
os.remove(out_path)
print(f"[py] wrote {len(out)} tensors -> {out_path}.gz")
