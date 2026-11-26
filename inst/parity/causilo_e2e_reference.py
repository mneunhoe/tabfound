#!/usr/bin/env python3
"""End-to-end reference for the Causilo 1.5 backend.

Runs the public `CausiloClassifier` / `CausiloRegressor` (`causilo` 1.5.0)
-- recipe, ensemble plan, network, member combination -- on the CPU in
float32 with `random_state = 42`, from the published checkpoints, for
`n_estimators` 1, 2, 4, 5 and 8 (the view allocation changes with each,
and regression trimming starts at 4). Categorical columns are handed over
as strings whose sorted order is the order of their integer codes (see
`causilo_ensemble_reference.py`), so the R side, given the codes, sees the
same categories. Classifiers record `predict_proba` and `predict_logits`
(above 10 classes, where the reference has no logits, the ECOC codebook
it drew instead),
regressors the mean, median, two sets of quantiles (one symmetric, one
not) and the raw combined grid. Each case is also fitted with
`use_kv_cache=True`, and that output stored beside the plain one (all
but the raw grid, which is kept for the plain fit only, in float32).

    .venvs/causilo/bin/python -I inst/parity/causilo_e2e_reference.py \\
        --fixture-dir inst/parity/fixtures \\
        --clf-dir <...>/causilo-1.5-classifier --reg-dir <...>/causilo-1.5-regressor \\
        --out inst/parity/reference/causilo_e2e
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

from causilo import CausiloClassifier, CausiloRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--clf-dir", required=True)
parser.add_argument("--reg-dir", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
fixture_dir = Path(args.fixture_dir)
manifest = json.loads((fixture_dir / "manifest.json").read_text())["fixtures"]

out: dict[str, torch.Tensor] = {}
cases: dict[str, dict] = {}
Q_SYM = np.array([0.1, 0.5, 0.9])
Q_ASYM = np.array([0.05, 0.25, 0.0005])


def label(k: float) -> object:
    return np.nan if np.isnan(k) else f"c{int(k):03d}"


def frame(x: np.ndarray, categorical: list[int]) -> pd.DataFrame:
    cols = {}
    for j in range(x.shape[1]):
        if j in categorical:
            cols[f"x{j}"] = pd.Series([label(v) for v in x[:, j]], dtype=object)
        else:
            cols[f"x{j}"] = x[:, j]
    return pd.DataFrame(cols)


def t(a) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.float64))).clone()


def run(name, fx_name, n_est):
    fx = load_file(str(fixture_dir / f"{fx_name}.safetensors"))
    info = manifest[fx_name]
    cat = [c - 1 for c in info.get("categorical_features", [])]
    xtr, xte = fx["x_train"].double().numpy(), fx["x_test"].double().numpy()
    ytr = fx["y_train"].double().numpy()
    task = info["task"]
    res = {}
    for cached in (False, True):
        key = "cached" if cached else "plain"
        if task == "classification":
            m = CausiloClassifier(n_estimators=n_est, random_state=42, device="cpu",
                                  checkpoint_path=args.clf_dir, use_kv_cache=cached)
            m.fit(frame(xtr, cat), ytr.astype(np.int64))
            res[f"{key}.proba"] = m.predict_proba(frame(xte, cat))
            codec = m._engine.state.dataset.codec
            if codec is None:
                res[f"{key}.logits"] = m.predict_logits(frame(xte, cat))
            else:
                # Above 10 classes the codebook is drawn with NumPy's PCG64,
                # which tabfound does not replay; keep the one used so the
                # R decoder can be graded on it.
                out[f"{name}.codebook"] = torch.from_numpy(codec.codebook.astype(np.int64)).clone()
            classes = [int(c) for c in m.classes_]
        else:
            m = CausiloRegressor(n_estimators=n_est, random_state=42, device="cpu",
                                 checkpoint_path=args.reg_dir, use_kv_cache=cached)
            m.fit(frame(xtr, cat), ytr)
            X = frame(xte, cat)
            res[f"{key}.mean"] = m.predict(X)
            res[f"{key}.median"] = m.predict(X, output_type="median")
            res[f"{key}.q_sym"] = m.predict(X, output_type="quantiles", quantiles=Q_SYM)
            res[f"{key}.q_asym"] = m.predict(X, output_type="quantiles", quantiles=Q_ASYM)
            res[f"{key}.raw"] = m.predict(X, output_type="raw")
            classes = None
    out[f"{name}.x_train"] = t(xtr)
    out[f"{name}.x_test"] = t(xte)
    out[f"{name}.y_train"] = t(ytr)
    for k, v in res.items():
        # The raw grids are most of the file: keep the plain one, in float32.
        if k == "cached.raw":
            continue
        out[f"{name}.{k}"] = t(v).float() if k == "plain.raw" else t(v)
    gaps = {k.split(".", 1)[1]: float(np.max(np.abs(res[k] - res["plain." + k.split(".", 1)[1]])))
            for k in res if k.startswith("cached.")}
    cases[name] = {"fixture": fx_name, "task": task, "n_estimators": n_est,
                   "categorical": [c + 1 for c in cat], "classes": classes,
                   "cached_vs_plain": gaps}
    print(f"[py] {name}: cached vs plain {gaps}", flush=True)


for fx_name in ("clf_tiny", "clf_binary_missing", "reg_tiny"):
    for n_est in (1, 2, 4, 5, 8):
        run(f"{fx_name}_e{n_est}", fx_name, n_est)
for fx_name in ("clf_categorical", "reg_categorical"):
    for n_est in (1, 8):
        run(f"{fx_name}_e{n_est}", fx_name, n_est)
# 15 observed classes: error-correcting output codes, one member per code
# row whatever n_estimators says.
run("clf_many_classes_e8", "clf_many_classes", 8)

save_file({k: v.contiguous() for k, v in out.items()}, str(OUT / "e2e.safetensors"))
meta = {"cases": cases, "q_sym": Q_SYM.tolist(), "q_asym": Q_ASYM.tolist(),
        "seed": 42, "torch_version": torch.__version__,
        "causilo_version": importlib.metadata.version("causilo")}
with open(OUT / "reference.json", "w") as fh:
    json.dump(meta, fh, indent=1)
