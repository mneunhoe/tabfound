#!/usr/bin/env python3
"""Introspect Python TabPFNClassifier / TabPFNRegressor ensembling.

Runs `TabPFNClassifier(n_estimators=N).predict_proba(X_test)` (or
`TabPFNRegressor` for regression) on the shared iris split and dumps
per-ensemble-member inputs (X_train post-preprocessing, X_test
post-preprocessing, y_train post-preprocessing) and outputs (model
logits) plus the final averaged prediction.

Usage:
    tabpfn/bin/python3 inst/python/dump_ensemble.py --head classifier --n 4
    tabpfn/bin/python3 inst/python/dump_ensemble.py --head regressor  --n 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from tabpfn import TabPFNClassifier, TabPFNRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--head", choices=["classifier", "regressor"], default="classifier")
parser.add_argument("--n", type=int, default=4)
args = parser.parse_args()
HEAD = args.head

REPO = Path(__file__).resolve().parent.parent.parent
DUMP_DIR = REPO / "debug" / f"py_ensemble_{args.head}"
DUMP_DIR.mkdir(parents=True, exist_ok=True)

if args.head == "regressor":
    CKPT = REPO / "ckpts" / "tabpfn-v2.5-regressor-v2.5_default.ckpt"
    SPLIT = REPO / "inst" / "extdata" / "debug_iris_reg_split.safetensors"
else:
    CKPT = REPO / "ckpts" / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    SPLIT = REPO / "inst" / "extdata" / "debug_iris_split.safetensors"

split = safe_load_file(str(SPLIT))
X_train_np = split["x_train"].numpy()
X_test_np  = split["x_test"].numpy()
y_train_np = split["y_train"].numpy().astype(
    "int64" if args.head == "classifier" else "float32"
)
y_test_np  = split["y_test"].numpy().astype(
    "int64" if args.head == "classifier" else "float32"
)

torch.manual_seed(0)
np.random.seed(0)

Est = TabPFNClassifier if args.head == "classifier" else TabPFNRegressor
est_kwargs = dict(
    model_path=str(CKPT.absolute()),
    n_estimators=args.n,
    random_state=0,
    device="cpu",
    inference_precision=torch.float32,
    softmax_temperature=1.0,   # disable temperature scaling for clean comparison
)
# Allow regressor to skip target_transform for partial-ensembling parity tests.
if HEAD == "regressor" and os.environ.get("TABPFN_NO_TARGET_TRANSFORM") == "1":
    est_kwargs["inference_config"] = {"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,)}
est = Est(**est_kwargs)
est.fit(X_train_np, y_train_np)

# --------- Monkey-patch _call_model to capture per-member inputs ----------
import tabpfn.inference as _inf
# TabPFNClassifier / TabPFNRegressor default to `fit_mode="fit_with_cache"` which
# picks `InferenceEngineCachePreprocessing`. Patch the parent so all subclasses
# route through our instrumentation.
_orig_call_model = _inf.MultiDeviceInferenceEngine._call_model \
    if hasattr(_inf.MultiDeviceInferenceEngine, "_call_model") \
    else _inf.InferenceEngineCachePreprocessing._call_model

_member_counter = {"i": 0}
member_records: list[dict] = []

def _instrumented_call_model(
    self, *, device, X_train, X_test, y_train, feature_schema,
    autocast, only_return_standard_out, model_index, save_peak_mem,
    gpu_preprocessor, task_type,
):
    i = _member_counter["i"]
    member_dir = DUMP_DIR / f"member_{i:02d}"
    member_dir.mkdir(parents=True, exist_ok=True)

    def _as_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone().cpu().contiguous().float()
        return torch.as_tensor(np.asarray(x)).float().cpu().contiguous()

    safe_save_file({"t": _as_tensor(X_train)}, str(member_dir / "X_train.safetensors"))
    safe_save_file({"t": _as_tensor(X_test)},  str(member_dir / "X_test.safetensors"))
    safe_save_file({"t": _as_tensor(y_train)}, str(member_dir / "y_train.safetensors"))

    _member_counter["i"] += 1
    out = _orig_call_model(
        self,
        device=device, X_train=X_train, X_test=X_test, y_train=y_train,
        feature_schema=feature_schema, autocast=autocast,
        only_return_standard_out=only_return_standard_out,
        model_index=model_index, save_peak_mem=save_peak_mem,
        gpu_preprocessor=gpu_preprocessor, task_type=task_type,
    )

    # Output tensor (standard decoder) -> save
    if isinstance(out, dict):
        t = out.get("standard")
    else:
        t = out
    if t is not None:
        safe_save_file({"t": _as_tensor(t)}, str(member_dir / "logits.safetensors"))
    return out

_inf.InferenceEngineOnDemand._call_model = _instrumented_call_model
_inf.InferenceEngineCachePreprocessing._call_model = _instrumented_call_model

# Also capture the fitted shuffle permutation + y_train passed through.
from tabpfn.preprocessing.steps.shuffle_features_step import ShuffleFeaturesStep
_orig_shuffle_tx = ShuffleFeaturesStep._transform
_shuffle_counter = {"i": 0}
_shuffle_seen = set()
def _shuffle_tx_traced(self, X, *, is_test=False):
    key = id(self)
    if key not in _shuffle_seen:
        _shuffle_seen.add(key)
        i = _shuffle_counter["i"]
        d = DUMP_DIR / f"member_{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        import numpy as _np
        perm = self.index_permutation_
        if isinstance(perm, torch.Tensor):
            perm_arr = perm.detach().cpu().numpy().astype("int64")
        else:
            perm_arr = _np.array(perm, dtype=_np.int64)
        safe_save_file(
            {"t": torch.from_numpy(perm_arr).contiguous()},
            str(d / "shuffle_perm.safetensors"),
        )
        _shuffle_counter["i"] += 1
    return _orig_shuffle_tx(self, X, is_test=is_test)
ShuffleFeaturesStep._transform = _shuffle_tx_traced

# Dump the regressor's target_transform fitted parameters (yeo-johnson
# lambdas + mean/std of the training target). Only present on some
# regressor members.
try:
    from tabpfn.preprocessing.steps.safe_power_transformer import SafePowerTransformer
    _orig_spt_transform = SafePowerTransformer.transform
    _spt_seen = set()
    _spt_counter = {"i": 0}
    def _spt_transform_traced(self, X):
        key = id(self)
        if key not in _spt_seen:
            _spt_seen.add(key)
            import numpy as _np
            i = _spt_counter["i"]
            d = DUMP_DIR / f"target_transform_{i:02d}"
            d.mkdir(parents=True, exist_ok=True)
            lmbdas = _np.asarray(self.lambdas_, dtype=_np.float64)
            safe_save_file(
                {"t": torch.from_numpy(lmbdas).contiguous()},
                str(d / "lambdas.safetensors"),
            )
            _spt_counter["i"] += 1
        return _orig_spt_transform(self, X)
    SafePowerTransformer.transform = _spt_transform_traced
except Exception as _e:
    pass

# --------- Run predict_proba / predict ------------------------------------
if args.head == "classifier":
    probs = est.predict_proba(X_test_np)
    preds = probs.argmax(axis=1)
    acc = float((preds == y_test_np).mean())
    print(f"Python TabPFNClassifier(n={args.n}) iris acc: {acc*100:.2f}%")
    safe_save_file({"t": torch.from_numpy(probs).float().contiguous()},
                   str(DUMP_DIR / "final_probs.safetensors"))
else:
    preds = est.predict(X_test_np)
    rmse = float(((preds - y_test_np) ** 2).mean() ** 0.5)
    mae  = float(abs(preds - y_test_np).mean())
    print(f"Python TabPFNRegressor(n={args.n}) iris RMSE: {rmse:.4f}  MAE: {mae:.4f}")
    safe_save_file({"t": torch.from_numpy(preds).float().contiguous()},
                   str(DUMP_DIR / "final_preds.safetensors"))

# ------- Also capture the ensemble configs for inspection ----------------
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

configs_path = DUMP_DIR / "ensemble_configs.json"
cfgs = getattr(est, "ensemble_configs_", None)
if cfgs is None:
    cfgs = []
with configs_path.open("w") as f:
    json.dump([_json_safe(c) for c in cfgs], f, indent=2)

# Dump per-member target_transform lambdas (regressor only).
for i, c in enumerate(cfgs):
    tt = getattr(c, "target_transform", None)
    if tt is None:
        continue
    # Pipeline with first step the SafePowerTransformer.
    try:
        spt = tt.named_steps.get("input_transformer") if hasattr(tt, "named_steps") else tt
        if hasattr(spt, "lambdas_"):
            import numpy as _np
            lmb = _np.asarray(spt.lambdas_, dtype=_np.float64)
            d = DUMP_DIR / f"member_{i:02d}"
            d.mkdir(parents=True, exist_ok=True)
            safe_save_file(
                {"t": torch.from_numpy(lmb).contiguous()},
                str(d / "target_transform_lambdas.safetensors"),
            )
    except Exception:
        pass
print(f"Wrote {_member_counter['i']} member(s) + configs to {DUMP_DIR}")
