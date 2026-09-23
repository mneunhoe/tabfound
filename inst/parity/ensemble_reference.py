#!/usr/bin/env python3
"""Reference outputs for the preprocessing + ensembling wrappers.

`transforms_reference.py` pins TabPFN's column transforms. This is its
counterpart for the other three backends: the stock sklearn estimators
their wrappers reach for, the `random.Random` stream that decides which
view each ensemble member gets, the ensemble generators themselves, and
Mitra's preprocessor.

Like that one it needs no model weights, so
`tests/testthat/test-prep-sklearn.R`, `test-prep-ensemble.R` and
`test-py-random.R` all run anywhere the package does. That matters more
here than it did there: none of this code is reachable from the
bare-network parity runs, so without these dumps the entire wrapper
layer would only ever be checked when someone has a checkpoint on disk.

    .venvs/ref/bin/python inst/parity/ensemble_reference.py \
        --out inst/parity/ensemble/ensemble.safetensors
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True,
                    help="Path to the .safetensors file; a sibling .json "
                         "holds the scalars and index vectors.")
parser.add_argument("--mitra-src", default=None,
                    help="Directory holding AutoGluon's mitra sources "
                         "(the `_internal/` tree). Mitra's preprocessor is "
                         "skipped when absent.")
args = parser.parse_args()

T: dict[str, torch.Tensor] = {}
S: dict[str, object] = {}

from importlib.metadata import version as _pkg_version  # noqa: E402
S["versions"] = {pkg: _pkg_version(pkg) for pkg in ("tabicl", "tabfm", "scikit-learn", "numpy")}


def put(name, arr):
    T[name] = torch.as_tensor(np.asarray(arr, dtype=np.float64)).contiguous()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
#
# One table, built so that every branch in the pipeline is live: an
# all-positive column and an all-negative one (Yeo-Johnson takes a
# different, log-space route for each), a rounded column full of ties
# (the quantile transformer's tie-averaging), a constant column (the
# unique-value filter, and the epsilon in CustomStandardScaler), a wild
# value (the outlier clipper) and missing values (the imputer).

rng = np.random.default_rng(4242)
n_tr, n_te, p = 60, 15, 7
X = rng.normal(size=(n_tr, p)) * np.array([1.0, 6.0, 0.2, 3.0, 50.0, 1.0, 1.0])
X[:, 1] = np.abs(X[:, 1]) + 0.5
X[:, 2] = -np.abs(X[:, 2]) - 0.5
X[:, 3] = np.round(X[:, 3])
X[:, 6] = 4.0
X[5, 0] = 60.0
X_test = rng.normal(size=(n_te, p)) * np.array([1.0, 6.0, 0.2, 3.0, 50.0, 1.0, 1.0])
X_test[:, 1] = np.abs(X_test[:, 1]) + 0.5
X_test[:, 2] = -np.abs(X_test[:, 2]) - 0.5
X_test[:, 3] = np.round(X_test[:, 3])
X_test[:, 6] = 4.0

X_na = X.copy()
X_na[2, 0] = np.nan
X_na[9, 4] = np.nan
X_test_na = X_test.copy()
X_test_na[1, 0] = np.nan

y_cls = rng.integers(0, 3, size=n_tr)
y_reg = rng.normal(size=n_tr) * 2.0 + 1.0

put("X", X)
put("X_test", X_test)
put("X_na", X_na)
put("X_test_na", X_test_na)
S["y_cls"] = y_cls.tolist()
S["y_reg"] = y_reg.tolist()


# ---------------------------------------------------------------------------
# CPython's random.Random
# ---------------------------------------------------------------------------
#
# The ensemble members are only reproducible if this stream is, so it is
# pinned directly rather than being inferred from the members it drives.
# A break here is unambiguous; a break seen only in the members is not.

py_rand = {}
for seed in [0, 1, 42, 12345, 2 ** 31 - 1]:
    r = random.Random(seed)
    py_rand[f"random_{seed}"] = [r.random() for _ in range(5)]
    r = random.Random(seed)
    py_rand[f"getrandbits_{seed}"] = [r.getrandbits(k)
                                      for k in [1, 3, 8, 17, 32, 32, 5]]
    r = random.Random(seed)
    py_rand[f"randbelow_{seed}"] = [r._randbelow(n)
                                    for n in [2, 3, 7, 10, 64, 1000, 999999]]
    r = random.Random(seed)
    lst = list(range(12))
    r.shuffle(lst)
    py_rand[f"shuffle_{seed}"] = lst
    r = random.Random(seed)
    py_rand[f"sample_pool_{seed}"] = r.sample(list(range(10)), 10)
    r = random.Random(seed)
    py_rand[f"sample_set_{seed}"] = r.sample(list(range(5000)), 30)
    r = random.Random(seed)
    py_rand[f"choice_{seed}"] = [r.choice(list(range(17))) for _ in range(6)]
S["py_random"] = py_rand


# ---------------------------------------------------------------------------
# The individual sklearn estimators
# ---------------------------------------------------------------------------

from scipy import stats  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from tabicl._sklearn.preprocessing import (  # noqa: E402
    CustomStandardScaler,
    OutlierRemover,
    PreprocessingPipeline,
    Shuffler,
    UniqueFeatureFilter,
)

css = CustomStandardScaler().fit(X)
S["css_mean"] = css.mean_.tolist()
S["css_scale"] = css.scale_.tolist()
put("css_train", css.transform(X))
put("css_test", css.transform(X_test))

Xs = css.transform(X)
Xs_test = css.transform(X_test)

orm = OutlierRemover(threshold=4.0).fit(Xs)
S["orm_lower"] = orm.lower_bounds_.tolist()
S["orm_upper"] = orm.upper_bounds_.tolist()
put("orm_train", orm.transform(Xs))
put("orm_test", orm.transform(Xs_test))

# The filter needs a finite input, and a column that is constant *except*
# for a couple of repeated values is the interesting case: it survives.
X_uff = X.copy()
X_uff[:, 5] = np.where(np.arange(n_tr) < 40, 0.0, 1.0)
put("X_uff", X_uff)
S["uff_keep"] = UniqueFeatureFilter().fit(X_uff).features_to_keep_.tolist()

# tabicl >= 2.2.0: an all-constant table keeps its first column rather
# than leaving nothing for the shuffler to permute.
X_const = np.tile(np.array([0.4, 1.0, 0.5, 118.2]), (n_tr, 1))
put("X_const", X_const)
S["uff_keep_const"] = UniqueFeatureFilter().fit(X_const).features_to_keep_.tolist()

# tabicl >= 2.2.0 imputes with `keep_empty_features=True`: an entirely
# missing column is kept and filled with 0 instead of being dropped.
from sklearn.impute import SimpleImputer  # noqa: E402
X_empty = X[:, :3].copy()
X_empty[:, 1] = np.nan
X_empty[[2, 7], 0] = np.nan
X_empty_test = X_test[:, :3].copy()
X_empty_test[[0, 3], 1] = np.nan
X_empty_test[1, 0] = np.nan
put("X_empty", X_empty)
put("X_empty_test", X_empty_test)
si_keep = SimpleImputer(keep_empty_features=True).fit(X_empty)
put("si_keep_train", si_keep.transform(X_empty))
put("si_keep_test", si_keep.transform(X_empty_test))
si_drop = SimpleImputer().fit(X_empty)
put("si_drop_train", si_drop.transform(X_empty))
put("si_drop_test", si_drop.transform(X_empty_test))

pt = PowerTransformer(method="yeo-johnson", standardize=True).fit(Xs)
S["pt_lambdas"] = pt.lambdas_.tolist()
put("pt_train", pt.transform(Xs))
put("pt_test", pt.transform(Xs_test))

qt = QuantileTransformer(output_distribution="normal", random_state=42).fit(Xs)
put("qt_quantiles", qt.quantiles_)
put("qt_train", qt.transform(Xs))
put("qt_test", qt.transform(Xs_test))

qtu = QuantileTransformer(output_distribution="uniform", n_quantiles=19).fit(Xs)
put("qtu_train", qtu.transform(Xs))
put("qtu_test", qtu.transform(Xs_test))

rs = RobustScaler(unit_variance=True).fit(Xs)
S["rs_center"] = rs.center_.tolist()
S["rs_scale"] = rs.scale_.tolist()
put("rs_train", rs.transform(Xs))

# Yeo-Johnson's likelihood and its maximiser, on columns that exercise
# all three branches of the log-likelihood (all-positive, all-negative,
# mixed sign).
yj_cols = {"pos": np.abs(rng.normal(size=40)) + 0.1,
           "neg": -np.abs(rng.normal(size=40)) - 0.1,
           "mix": rng.normal(size=40) * 5,
           "skew": rng.exponential(size=40) * 3}
S["yj_grid"] = [-2.0, -0.5, 0.0, 0.7, 1.0, 3.0]
S["yj_cols"] = {k: v.tolist() for k, v in yj_cols.items()}
S["yj_lambdas"] = {k: float(stats.yeojohnson_normmax(v))
                   for k, v in yj_cols.items()}
S["yj_llf"] = {k: [float(stats.yeojohnson_llf(l, v)) for l in S["yj_grid"]]
               for k, v in yj_cols.items()}

# The whole pipeline, per normalisation method.
S["pipe_methods"] = ["none", "power", "quantile", "robust"]
for m in S["pipe_methods"]:
    pp = PreprocessingPipeline(normalization_method=m, random_state=42).fit(X)
    put(f"pipe_{m}_train", pp.X_transformed_)
    put(f"pipe_{m}_test", pp.transform(X_test))


# ---------------------------------------------------------------------------
# Shufflers
# ---------------------------------------------------------------------------

shuf = {}
for n in [2, 3, 5, 7, 12]:
    for meth in ["none", "shift", "random", "latin"]:
        for seed in [0, 42]:
            pats = Shuffler(n_elements=n, method=meth,
                            random_state=seed).shuffle(8)
            shuf[f"{n}_{meth}_{seed}"] = [list(map(int, p)) for p in pats]
S["shuffler"] = shuf


# ---------------------------------------------------------------------------
# The ensemble generators
# ---------------------------------------------------------------------------
#
# Each member's own (X, y) is dumped, not just the configuration. A
# matching configuration with a mismatched member matrix would mean the
# preprocessing had drifted, and only dumping both tells those apart.

from tabfm.src.classifier_and_regressor import EnsembleGenerator as FMGen  # noqa: E402
from tabicl._sklearn.preprocessing import EnsembleGenerator as ICLGen  # noqa: E402

ens_meta = {}


def dump_icl(tag, Xf, yf, classification, Xt=None, **kw):
    g = ICLGen(classification=classification, **kw)
    g.fit(Xf, yf)
    data = g.transform(X_test if Xt is None else Xt, mode="both")
    members = []
    i = 0
    # The generator groups members by `list(set(methods))`, whose order
    # follows the string hash and so changes with PYTHONHASHSEED from one
    # run of this script to the next. Averaging makes the order irrelevant
    # to any prediction; dump in first-appearance order (the order of
    # `norm_methods`), which is what the R side uses, so the file is stable.
    for m in [m for m in g.norm_methods_ if m in data]:
        Xs_, ys_ = data[m]
        feats = g.feature_shuffles_[m]
        cls = g.class_shuffles_[m] if classification else [None] * len(feats)
        for j in range(Xs_.shape[0]):
            put(f"{tag}_{i:02d}_X", Xs_[j])
            put(f"{tag}_{i:02d}_y", ys_[j])
            members.append({
                "norm": m,
                "feat": list(map(int, feats[j])),
                "class_shuffle": None if cls[j] is None else list(map(int, cls[j])),
            })
            i += 1
    ens_meta[tag] = {"keep": g.unique_filter_.features_to_keep_.tolist(),
                     "members": members, "kwargs": _jsonable(kw),
                     "classification": classification}


def dump_fm(tag, Xf, yf, task, **kw):
    g = FMGen(task=task, **kw)
    g.fit(Xf, yf)
    data = g.transform(X_test)
    Xs_all, ys_all, cat_masks, _, _ = g.prepare_ensemble_tensors(data)
    feats, shifts, norms = [], [], []
    for m in g.ensemble_configs_:
        feats.extend(g.feature_shuffle_patterns_[m])
        shifts.extend(g.class_shift_offsets_[m])
        norms.extend([m] * len(g.ensemble_configs_[m]))
    members = []
    for i in range(Xs_all.shape[0]):
        put(f"{tag}_{i:02d}_X", Xs_all[i])
        put(f"{tag}_{i:02d}_y", ys_all[i])
        members.append({"norm": norms[i], "feat": list(map(int, feats[i])),
                        "shift": int(shifts[i]),
                        "cat_mask": [bool(b) for b in cat_masks[i]]})
    ens_meta[tag] = {"keep": g.unique_filter_.features_to_keep_.tolist(),
                     "members": members, "kwargs": _jsonable(kw),
                     "classification": task == "classification"}


def _jsonable(kw):
    return {k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in kw.items()}


dump_icl("icl_clf", X, y_cls, True, n_estimators=8,
         norm_methods=["none", "power"], feat_shuffle_method="latin",
         class_shuffle_method="shift", random_state=42)
dump_icl("icl_clf_rand", X, y_cls, True, n_estimators=6,
         norm_methods=["none", "quantile"], feat_shuffle_method="random",
         class_shuffle_method="random", random_state=7)
dump_icl("icl_reg", X, y_reg, False, n_estimators=8,
         norm_methods=["none", "power"], feat_shuffle_method="latin",
         random_state=42)
# All-constant input: one column survives, so the Latin square is 1 x 1.
X_const_test = np.tile(np.array([0.4, 1.0, 0.5, 118.2]), (n_te, 1))
put("X_const_test", X_const_test)
dump_icl("icl_clf_const", X_const, y_cls, True, Xt=X_const_test,
         n_estimators=4, norm_methods=["none", "power"],
         feat_shuffle_method="latin", class_shuffle_method="shift",
         random_state=0)
dump_icl("icl_reg_const", X_const, y_reg, False, Xt=X_const_test,
         n_estimators=4, norm_methods=["none", "power"],
         feat_shuffle_method="latin", random_state=0)

dump_fm("fm_clf", X, y_cls, "classification", n_estimators=8,
        norm_methods=["none", "power"], cat_features=[0, 3], random_state=42)
dump_fm("fm_clf_noshift", X, y_cls, "classification", n_estimators=5,
        norm_methods=["none"], class_shift=False, random_state=13)
dump_fm("fm_reg", X, y_reg, "regression", n_estimators=8,
        norm_methods=["none", "power"], random_state=42)
S["ensembles"] = ens_meta


# ---------------------------------------------------------------------------
# TabICL's QuantileDistribution
# ---------------------------------------------------------------------------

from tabicl._model.quantile_dist import QuantileDistribution  # noqa: E402

qd = {}
for tag, n_q, n_rows in [("small", 40, 6), ("full", 999, 4)]:
    base = np.sort(rng.normal(size=(n_rows, n_q)), axis=1)
    # Perturb after sorting so the grid genuinely crosses -- reading
    # summaries off an already-monotone grid would not exercise the fix.
    grid = base + rng.normal(scale=0.05, size=(n_rows, n_q))
    d = QuantileDistribution(torch.tensor(grid, dtype=torch.float64))
    alphas = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9995]
    put(f"qd_{tag}_grid", grid)
    put(f"qd_{tag}_sorted", d.quantiles.numpy())
    put(f"qd_{tag}_mean", d.quantiles.mean(dim=-1).numpy())
    put(f"qd_{tag}_icdf", d.icdf(torch.tensor(alphas, dtype=torch.float64)).numpy())
    qd[tag] = {"n_q": n_q, "alphas": alphas,
               "beta_l": d.beta_l.numpy().tolist(),
               "beta_r": d.beta_r.numpy().tolist(),
               "median": d.icdf(torch.tensor(0.5, dtype=torch.float64)).numpy().tolist()}
S["quantile_dist"] = qd


# ---------------------------------------------------------------------------
# Mitra's preprocessor
# ---------------------------------------------------------------------------
#
# Lives inside `autogluon.tabular`, so it is loaded from a source tree
# rather than installed -- the same arrangement `mitra_reference.py`
# uses. `random_mirror_x` is forced off: the reference draws its sign
# flips from NumPy's *global* generator and never seeds it, so its
# mirrors are not reproducible even between two of its own runs.

if args.mitra_src:
    src_dir = Path(args.mitra_src)
    loguru = types.ModuleType("loguru")
    loguru.logger = types.SimpleNamespace(info=lambda *a, **k: None,
                                          warning=lambda *a, **k: None)
    sys.modules["loguru"] = loguru
    enums = types.ModuleType("enums")
    exec((src_dir / "_internal/config/enums.py").read_text(), enums.__dict__)
    pre = types.ModuleType("preproc")
    pre.Task = enums.Task
    body = (src_dir / "_internal/data/preprocessor.py").read_text()
    exec(body.replace("from ..._internal.config.enums import Task", ""),
         pre.__dict__)

    mitra_meta = {}
    for tag, yv, task, mirror_reg in [
        ("mitra_clf", y_cls, enums.Task.CLASSIFICATION, False),
        ("mitra_reg", y_reg, enums.Task.REGRESSION, False),
    ]:
        pp = pre.Preprocessor(
            dim_embedding=None, n_classes=10, dim_output=10,
            use_quantile_transformer=False, use_feature_count_scaling=False,
            use_random_transforms=False, shuffle_classes=False,
            shuffle_features=False, random_mirror_regression=mirror_reg,
            random_mirror_x=False, task=task)
        pp.fit(X_na.copy(), yv.copy())
        put(f"{tag}_train", pp.transform_X(X_na.copy()))
        put(f"{tag}_test", pp.transform_X(X_test_na.copy()))
        put(f"{tag}_y", pp.transform_y(yv.copy()))
        entry = {"singular": [bool(b) for b in pp.singular_features],
                 "pre_nan_mean": pp.pre_nan_mean.tolist()}
        if task == enums.Task.REGRESSION:
            preds = np.linspace(0.1, 0.9, 9)
            entry["y_min"] = float(pp.y_min)
            entry["y_max"] = float(pp.y_max)
            entry["preds"] = preds.tolist()
            entry["inverse"] = pp.inverse_transform_y(preds.copy()).tolist()
        mitra_meta[tag] = entry
    S["mitra"] = mitra_meta
else:
    print("[py] --mitra-src not given; skipping Mitra's preprocessor")


# ---------------------------------------------------------------------------

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
save_file(T, str(out))
with open(out.with_suffix(".json"), "w") as fh:
    json.dump(S, fh)
print(f"[py] {len(T)} tensors -> {out}")
print(f"[py] scalars -> {out.with_suffix('.json')}")
