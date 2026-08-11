#!/usr/bin/env python3
"""Generate reference outputs for the TabPFN v2.6 backend.

Unlike `tabpfn_reference.py`, which drives the full `TabPFNClassifier`
estimator, this runs the bare `TabPFNV2p6` network on a parity fixture:
one forward pass, no ensembling, no sklearn-side preprocessing. That is
deliberate. In v2.6 the preprocessing that matters -- constant-column
removal, NaN/Inf imputation and flagging, standard scaling, feature-group
normalisation -- moved *inside* the architecture, so a bare forward pass
is exactly the surface the R port reimplements. Comparing estimator
output would fold in the quantile transforms and ensemble members too,
and a mismatch would say nothing about where it came from.

Everything lands in one `<out>/forward.safetensors`:

    x_train, x_test, y_train   the fixture, exactly as fed in
    logits                     raw decoder output, (n_test, n_out)
    probs                      classifier: softmax(logits / temperature)
    bar_mean, bar_quantiles    regressor: the reference bar distribution's
                               own decoding of `logits`

The decoded tensors are there so the R side can be graded in two
independent steps: does its forward pass reproduce `logits`, and does its
bar-distribution / softmax decoding reproduce `bar_mean` and `probs` when
handed *the reference's own* logits. Keeping those separate means a
mismatch names its own cause. Note the regressor is fed the fixture's raw
target here, not the standardised one `tabular_regressor()` uses, so the
logits are only comparable against a matching raw forward pass -- which is
exactly what the harness compares them against.

Usage
-----
    .venvs/ref/bin/python inst/parity/tabpfn26_reference.py \
        --fixture-dir inst/parity/fixtures \
        --fixture clf_iris \
        --ckpt ckpts/tabpfn-v2.6-classifier-v2.6_default.ckpt \
        --out inst/parity/reference/tabpfn26/clf_iris
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

import tabpfn
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.architectures.tabpfn_v2_6 import get_architecture, parse_config

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-dir", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--task", choices=["classification", "regression"], default=None,
                    help="Inferred from the fixture name prefix when omitted.")
parser.add_argument("--softmax-temperature", type=float, default=0.9,
                    help="Divides the logits before decoding. 0.9 is the "
                         "estimator default, which the R backend mirrors.")
parser.add_argument("--quantiles", default="0.1,0.5,0.9")
parser.add_argument("--out", required=True)
args = parser.parse_args()

TASK = args.task or ("classification" if args.fixture.startswith("clf")
                     else "regression")
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

fx = safe_load_file(str(Path(args.fixture_dir) / f"{args.fixture}.safetensors"))
X_train = fx["x_train"].to(torch.float32)
X_test = fx["x_test"].to(torch.float32)
y_train = fx["y_train"].to(torch.float32)

raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
if raw.get("architecture_name") != "tabpfn_v2_6":
    raise SystemExit(
        f"{args.ckpt} is a {raw.get('architecture_name')!r} checkpoint, "
        "not tabpfn_v2_6."
    )

config, _unused = parse_config(raw["config"])
model = get_architecture(config)
model.load_state_dict(
    {k: v for k, v in raw["state_dict"].items() if not k.startswith("criterion.")},
    strict=True,
)
model.eval()

if (model.task_type == "multiclass") != (TASK == "classification"):
    raise SystemExit(
        f"Checkpoint is a {model.task_type} model but the fixture is {TASK}."
    )

# The architecture takes (rows, batch, columns) with the test rows
# appended to the train rows, and labels for the train rows only.
x = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
with torch.no_grad():
    logits = model(x, y_train, only_return_standard_out=True)

    # The two paths that exist purely to make inference cheaper, each run
    # its own way so the R side is graded against the mechanism rather
    # than against the plain pass it is supposed to agree with.
    #
    # `save_peak_memory_factor` splits every sublayer's work into chunks.
    perf = dataclasses.replace(
        model.get_default_performance_options(), save_peak_memory_factor=8
    )
    logits_chunked = model(
        x.clone(), y_train, only_return_standard_out=True, performance_options=perf
    )

    # The KV cache conditions on the training rows once, then predicts the
    # test rows without them present.
    _, cache = model(
        X_train.unsqueeze(1), y_train, only_return_standard_out=True,
        return_kv_cache=True,
    )
    logits_cached = model(
        X_test.unsqueeze(1), y_train, only_return_standard_out=True,
        kv_cache=cache, x_is_test_only=True,
    )

# (n_test, batch, n_out) -> (n_test, n_out); the batch is one dataset.
logits = logits[:, 0, :].contiguous()

tensors = {
    "x_train": X_train.contiguous(),
    "x_test": X_test.contiguous(),
    "y_train": y_train.contiguous(),
    "logits": logits,
    "logits_chunked": logits_chunked[:, 0, :].contiguous(),
    "logits_cached": logits_cached[:, 0, :].contiguous(),
}

tempered = logits / args.softmax_temperature
quantiles = [float(q) for q in args.quantiles.split(",")]
if TASK == "classification":
    # The head is always 10 wide; only the classes present in the fixture
    # are decoded, which is what the estimator does.
    n_classes = int(y_train.max().item()) + 1
    tensors["probs"] = tempered[:, :n_classes].softmax(-1).contiguous()
    n_bar_bins = None
else:
    borders = raw["state_dict"]["criterion.borders"].to(torch.float32)
    bar = FullSupportBarDistribution(borders)
    tensors["bar_mean"] = bar.mean(tempered).contiguous()
    tensors["bar_quantiles"] = torch.stack(
        [bar.icdf(tempered, q) for q in quantiles], dim=-1
    ).contiguous()
    n_classes = None
    n_bar_bins = int(borders.shape[0]) - 1

safe_save_file(tensors, str(OUT / "forward.safetensors"))

meta = {
    "fixture": args.fixture,
    "task": TASK,
    "tabpfn_version": tabpfn.__version__,
    "torch_version": torch.__version__,
    "ckpt": Path(args.ckpt).name,
    "architecture": "tabpfn_v2_6",
    "head": "classifier" if model.task_type == "multiclass" else "regressor",
    "n_out": int(logits.shape[-1]),
    "n_classes": n_classes,
    "n_bar_bins": n_bar_bins,
    "softmax_temperature": args.softmax_temperature,
    "quantiles": quantiles,
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "n_features": int(X_train.shape[1]),
    "save_peak_memory_factor": 8,
}
(OUT / "reference.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
