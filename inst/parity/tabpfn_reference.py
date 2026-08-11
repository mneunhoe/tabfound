#!/usr/bin/env python3
"""Generate reference outputs for the TabPFN backend from the PyPI package.

Runs `TabPFNClassifier` / `TabPFNRegressor` on a parity fixture and dumps
everything the R side needs to reproduce the run and everything needed to
grade it:

    <out>/reference.json                 run metadata + accuracy/RMSE
    <out>/ensemble_configs.json          per-member preprocessing configs
    <out>/final_probs.safetensors        predict_proba output (classifier)
    <out>/final_preds.safetensors        predict output (regressor)
    <out>/member_NN/shuffle_perm.safetensors
    <out>/member_NN/X_train.safetensors  member input AFTER preprocessing
    <out>/member_NN/X_test.safetensors
    <out>/member_NN/y_train.safetensors
    <out>/member_NN/logits.safetensors   raw decoder output for that member
    <out>/member_NN/target_transform_lambdas.safetensors  (regressor, if any)

The member-level dumps let the parity report localise a mismatch:
preprocessing, the forward pass, or the ensemble combination.

Usage
-----
    .venvs/ref/bin/python inst/parity/tabpfn_reference.py \
        --fixture-dir inst/parity/fixtures \
        --fixture clf_iris \
        --ckpt ckpts/tabpfn-v2.5-classifier-v2.5_default.ckpt \
        --n-estimators 4 \
        --out inst/parity/reference/tabpfn/clf_iris

`softmax_temperature=1.0` is set explicitly: the estimator defaults to
0.9, which the R port does not apply, and comparing against the default
would silently measure the temperature rather than the model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

import tabpfn
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing.datamodel import FeatureModality as _FeatureModality


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--task", choices=["classification", "regression"],
                    default=None,
                    help="Inferred from the fixture name prefix when omitted.")
parser.add_argument("--n-estimators", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-target-transform", action="store_true",
                    help="Disable Yeo-Johnson target transform members "
                         "(regressor only).")
parser.add_argument("--softmax-temperature", type=float, default=0.9,
                    help="Both estimators default to 0.9; the R backend "
                         "mirrors that default. Pass 1.0 to compare the "
                         "untempered distribution.")
parser.add_argument("--categorical", default="",
                    help="Comma-separated 0-based column indices to declare "
                         "categorical, as `categorical_features_indices`.")
parser.add_argument("--no-fingerprint", action="store_true",
                    help="Disable the per-row fingerprint feature "
                         "(inference_config FINGERPRINT_FEATURE=False). The "
                         "fingerprint is a SHA-256 of the row's float64 bytes, "
                         "so a sub-ULP difference anywhere upstream flips it to "
                         "an unrelated value; turning it off isolates the rest "
                         "of the pipeline for exact comparison.")
parser.add_argument("--out", required=True)
args = parser.parse_args()

TASK = args.task or ("classification" if args.fixture.startswith("clf")
                     else "regression")
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

fx = safe_load_file(str(Path(args.fixture_dir) / f"{args.fixture}.safetensors"))
X_train = fx["x_train"].numpy().astype("float32")
X_test = fx["x_test"].numpy().astype("float32")
y_train = fx["y_train"].numpy()
y_train = y_train.astype("int64" if TASK == "classification" else "float32")

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# ---------------------------------------------------------------------------
# Tracing hooks
# ---------------------------------------------------------------------------
# The estimator does not expose per-member inputs or the fitted column
# permutation, so we wrap three internals. Signatures drift between
# releases; wrappers forward *args/**kwargs untouched and only read what
# they need.

DUMP = OUT

# Tracing covers exactly one prediction pass. The regressor is called a
# second time for quantiles, which would otherwise append a duplicate set
# of member_NN directories and make the member count disagree with the
# ensemble config count.
_tracing = {"on": True}


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        # clone: .cpu()/.contiguous() are no-ops when already CPU/contiguous,
        # so without it the saved tensor aliases a buffer that gets reused.
        return x.detach().clone().cpu().contiguous().float()
    return torch.as_tensor(np.asarray(x)).float().cpu().contiguous()


import tabpfn.inference as _inf  # noqa: E402

_member_counter = {"i": 0}
_orig_call_model = _inf.InferenceEngineCachePreprocessing._call_model


def _traced_call_model(self, *pargs, **kwargs):
    if not _tracing["on"]:
        return _orig_call_model(self, *pargs, **kwargs)
    i = _member_counter["i"]
    d = DUMP / f"member_{i:02d}"
    d.mkdir(parents=True, exist_ok=True)
    for key in ("X_train", "X_test", "y_train"):
        if key in kwargs:
            safe_save_file({"t": _as_tensor(kwargs[key])}, str(d / f"{key}.safetensors"))
    _member_counter["i"] += 1

    out = _orig_call_model(self, *pargs, **kwargs)

    t = out.get("standard") if isinstance(out, dict) else out
    if t is not None:
        safe_save_file({"t": _as_tensor(t)}, str(d / "logits.safetensors"))
    return out


for _cls_name in ("InferenceEngineCachePreprocessing", "InferenceEngineOnDemand"):
    _cls = getattr(_inf, _cls_name, None)
    if _cls is not None:
        _cls._call_model = _traced_call_model


from tabpfn.preprocessing.steps.shuffle_features_step import (  # noqa: E402
    ShuffleFeaturesStep,
)

_orig_shuffle = ShuffleFeaturesStep._transform
_shuffle_seen: set[int] = set()
_shuffle_counter = {"i": 0}


def _traced_shuffle(self, X, *pargs, **kwargs):
    if id(self) not in _shuffle_seen:
        _shuffle_seen.add(id(self))
        i = _shuffle_counter["i"]
        d = DUMP / f"member_{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        perm = self.index_permutation_
        perm_arr = (perm.detach().cpu().numpy() if isinstance(perm, torch.Tensor)
                    else np.asarray(perm)).astype("int64")
        safe_save_file({"t": torch.from_numpy(perm_arr).contiguous()},
                       str(d / "shuffle_perm.safetensors"))
        _shuffle_counter["i"] += 1
    return _orig_shuffle(self, X, *pargs, **kwargs)


ShuffleFeaturesStep._transform = _traced_shuffle


from tabpfn.preprocessing.steps.nan_handling_polynomial_features_step import (  # noqa: E402
    NanHandlingPolynomialFeaturesStep,
)

# Which column pairs a member multiplies is drawn from the reference's
# NumPy generator, so the R side cannot recompute it -- dump it alongside
# the shuffle permutation. Hooked on `_transform` rather than `_fit` so it
# advances in the same pass, and in the same member order, as the shuffle
# tracer above; the id-set makes the train and test calls count once.
_orig_poly = NanHandlingPolynomialFeaturesStep._transform
_poly_seen: set[int] = set()
_poly_counter = {"i": 0}


def _traced_poly(self, X, *pargs, **kwargs):
    if id(self) not in _poly_seen:
        _poly_seen.add(id(self))
        d = DUMP / f"member_{_poly_counter['i']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        safe_save_file(
            {
                "factor_1": torch.from_numpy(
                    np.asarray(self.poly_factor_1_idx).astype("int64")
                ).contiguous(),
                "factor_2": torch.from_numpy(
                    np.asarray(self.poly_factor_2_idx).astype("int64")
                ).contiguous(),
            },
            str(d / "poly_factors.safetensors"),
        )
        _poly_counter["i"] += 1
    return _orig_poly(self, X, *pargs, **kwargs)


NanHandlingPolynomialFeaturesStep._transform = _traced_poly


from tabpfn.preprocessing.steps.encode_categorical_features_step import (  # noqa: E402
    EncodeCategoricalFeaturesStep,
)

# The `_shuffled` ordinal encoders permute each column's category codes
# with the reference's NumPy generator. Same story as the shuffle and the
# polynomial pairs: dump it, because R cannot redraw it. Keyed in
# encoded-column order so the R side can zip them to its own selection --
# which doubles as a check that both sides selected the same columns.
_orig_encode = EncodeCategoricalFeaturesStep._transform
_encode_seen: set[int] = set()
_encode_counter = {"i": 0}


def _traced_encode(self, X, *pargs, **kwargs):
    if id(self) not in _encode_seen:
        _encode_seen.add(id(self))
        mappings = getattr(self, "random_mappings_", None) or {}
        if mappings:
            d = DUMP / f"member_{_encode_counter['i']:02d}"
            d.mkdir(parents=True, exist_ok=True)
            safe_save_file(
                {
                    f"col_{col:02d}": torch.from_numpy(
                        np.asarray(m).astype("int64")
                    ).contiguous()
                    for col, m in mappings.items()
                },
                str(d / "cat_mappings.safetensors"),
            )
        _encode_counter["i"] += 1
    return _orig_encode(self, X, *pargs, **kwargs)


EncodeCategoricalFeaturesStep._transform = _traced_encode

# ---------------------------------------------------------------------------
# Fit + predict
# ---------------------------------------------------------------------------

inference_config = None
if args.no_fingerprint:
    inference_config = {"FINGERPRINT_FEATURE": False}

cat_indices = (
    [int(i) for i in args.categorical.split(",") if i.strip() != ""]
    if args.categorical
    else None
)

common = dict(
    categorical_features_indices=cat_indices,
    inference_config=inference_config,
    model_path=str(Path(args.ckpt).resolve()),
    n_estimators=args.n_estimators,
    auto_scale_n_estimators=False,
    device="cpu",
    random_state=args.seed,
    fit_mode="fit_preprocessors",
    inference_precision=torch.float32,
)

if TASK == "classification":
    est = TabPFNClassifier(softmax_temperature=args.softmax_temperature,
                           balance_probabilities=False,
                           average_before_softmax=False,
                           **common)
else:
    # The regressor divides the decoder output by the same temperature,
    # before the border translation rather than just before the softmax.
    est = TabPFNRegressor(softmax_temperature=args.softmax_temperature,
                          average_before_softmax=False,
                          **common)

est.fit(X_train, y_train)

meta = {
    "tabpfn_version": tabpfn.__version__,
    "torch_version": torch.__version__,
    "fixture": args.fixture,
    "task": TASK,
    "n_estimators": args.n_estimators,
    "seed": args.seed,
    "ckpt": Path(args.ckpt).name,
    "fingerprint_feature": not args.no_fingerprint,
    "softmax_temperature": args.softmax_temperature,
    "declared_categorical": cat_indices,
    # What the estimator actually decided, which may be a subset of what
    # was declared -- the R side has to agree on this, not on the request.
    "inferred_categorical": [
        int(i)
        for i in est.inferred_feature_schema_.indices_for(
            _FeatureModality.CATEGORICAL
        )
    ],
}

if TASK == "classification":
    probs = est.predict_proba(X_test)
    _tracing["on"] = False
    safe_save_file({"t": torch.from_numpy(np.ascontiguousarray(probs)).float()},
                   str(OUT / "final_probs.safetensors"))
    meta["n_classes"] = int(probs.shape[1])
    print(f"[py] {args.fixture}: predict_proba {probs.shape}")
else:
    preds = est.predict(X_test)
    _tracing["on"] = False
    safe_save_file({"t": torch.from_numpy(np.ascontiguousarray(preds)).float()},
                   str(OUT / "final_preds.safetensors"))
    for q in ([0.1, 0.5, 0.9],):
        try:
            qs = est.predict(X_test, output_type="quantiles", quantiles=q)
            qs = np.stack([np.asarray(v) for v in qs], axis=1)
            safe_save_file({"t": torch.from_numpy(np.ascontiguousarray(qs)).float()},
                           str(OUT / "final_quantiles.safetensors"))
            meta["quantiles"] = q
        except Exception as e:  # pragma: no cover - optional extra
            meta["quantiles_error"] = str(e)
    print(f"[py] {args.fixture}: predict {preds.shape}")

meta["n_members_traced"] = _member_counter["i"]


# ---------------------------------------------------------------------------
# Ensemble configs
# ---------------------------------------------------------------------------

def _json_safe(o):
    if isinstance(o, np.ndarray):
        return {"__ndarray__": True, "dtype": str(o.dtype),
                "shape": list(o.shape), "vals": o.ravel().tolist()}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: _json_safe(getattr(o, f.name))
                for f in dataclasses.fields(o)}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


configs = [m.config for m in est.executor_.ensemble_members]
with open(OUT / "ensemble_configs.json", "w") as fh:
    json.dump([_json_safe(c) for c in configs], fh, indent=2)

# Regressor members may carry a fitted Yeo-Johnson target transform. Read
# the lambdas straight off the fitted pipeline rather than tracing
# `SafePowerTransformer.transform`: sklearn's `PowerTransformer.fit_transform`
# routes through `_fit(force_transform=True)` and never calls `transform`,
# so a transform hook silently records nothing.
n_written = 0
scaler_stats = []
for idx, cfg in enumerate(configs):
    tt = getattr(cfg, "target_transform", None)
    # `None` and an identity `FunctionTransformer` are the same thing here:
    # a member with no target transform. TabPFN v2.6's regressor resolves
    # `REGRESSION_Y_PREPROCESS_TRANSFORMS` to `("none",)`, which is the
    # identity transformer rather than `None`, so both have to be skipped.
    if tt is None or not hasattr(tt, "steps"):
        continue
    spt = tt.steps[0][1]
    lam = np.asarray(getattr(spt, "lambdas_"), dtype="float64")
    d = OUT / f"member_{idx:02d}"
    d.mkdir(parents=True, exist_ok=True)
    safe_save_file({"t": torch.from_numpy(lam).contiguous()},
                   str(d / "target_transform_lambdas.safetensors"))
    n_written += 1
    try:
        std = tt.steps[1][1].steps[2][1]
        scaler_stats.append({"member": idx,
                             "lambda": lam.tolist(),
                             "mean": np.asarray(std.mean_).tolist(),
                             "scale": np.asarray(std.scale_).tolist()})
    except (AttributeError, IndexError):
        pass
meta["n_target_transforms"] = n_written
meta["target_transform_stats"] = scaler_stats

with open(OUT / "reference.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"[py] wrote reference to {OUT}")
