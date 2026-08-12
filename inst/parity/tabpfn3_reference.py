#!/usr/bin/env python3
"""Generate reference outputs for the TabPFN v3 backend.

Like `tabpfn26_reference.py`, this runs the bare `TabPFNV3` network on a
parity fixture -- one forward pass, no ensembling, no sklearn-side
preprocessing -- rather than driving the `TabPFNClassifier` estimator.
v3 keeps v2.6's arrangement where the preprocessing that matters (NaN/Inf
indicators, mean imputation, standard scaling) lives *inside*
`forward()`, so the bare pass is exactly the surface the R port
reimplements, and a mismatch here names its own cause.

Everything lands in one `<out>/forward.safetensors`:

    x_train, x_test, y_train   the fixture, exactly as fed in
    logits                     raw decoder output, (n_test, n_out)
    logits_chunked             the same with save_peak_memory_factor set
    logits_stage_chunked       the same with the reference's row/column
                               chunking of stages 0-2 turned on
    logits_stage_chunked_spmf  both chunking mechanisms at once
    logits_cached              the same via the explicit KV cache
    probs                      classifier: softmax(logits / temperature)
    bar_mean, bar_quantiles    regressor: the reference bar distribution's
                               own decoding of `logits`

Every pass sets `use_chunkwise_inference` explicitly rather than taking
the default. That matters: v3 is the one architecture whose
`get_default_performance_options()` turns it *on*, so a fixture large
enough to cross `inference_row_chunk_size` would silently change which
path `logits` came from. `logits` is the unchunked pass by construction,
and the chunked one is a separate, named tensor.

The decoded tensors are there so the R side can be graded in two
independent steps: does its forward pass reproduce `logits`, and does its
bar-distribution / softmax decoding reproduce `bar_mean` and `probs` when
handed *the reference's own* logits. Note the regressor is fed the
fixture's raw target here, not the standardised one `tabular_regressor()`
uses, so the logits are only comparable against a matching raw forward
pass -- which is what the harness compares them against.

Usage
-----
    .venvs/ref/bin/python inst/parity/tabpfn3_reference.py \
        --fixture-dir inst/parity/fixtures \
        --fixture clf_iris \
        --ckpt ckpts/tabpfn-v3-classifier-v3_default.ckpt \
        --out inst/parity/reference/tabpfn3/clf_iris
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
from tabpfn.architectures.tabpfn_v3 import get_architecture, parse_config

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
parser.add_argument("--save-peak-memory-factor", type=int, default=8)
parser.add_argument("--row-chunk-size", type=int, default=None,
                    help="Rows per stage-0-2 chunk. Defaults to the "
                         "checkpoint's own `inference_row_chunk_size`.")
parser.add_argument("--col-chunk-size", type=int, default=None,
                    help="Feature groups per inducing-hidden chunk. Defaults "
                         "to the checkpoint's `inference_col_chunk_size`.")
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
if raw.get("architecture_name") != "tabpfn_v3":
    raise SystemExit(
        f"{args.ckpt} is a {raw.get('architecture_name')!r} checkpoint, "
        "not tabpfn_v3."
    )

config, _unused = parse_config(raw["config"])
model = get_architecture(config)
model.load_state_dict(raw["state_dict"], strict=True)
model.eval()

if (model.task_type == "multiclass") != (TASK == "classification"):
    raise SystemExit(
        f"Checkpoint is a {model.task_type} model but the fixture is {TASK}."
    )

if args.row_chunk_size is not None:
    model.inference_row_chunk_size = args.row_chunk_size
if args.col_chunk_size is not None:
    model.inference_col_chunk_size = args.col_chunk_size
ROW_CHUNK = int(model.inference_row_chunk_size)
COL_CHUNK = int(model.inference_col_chunk_size)

# Every pass names its own path. `use_chunkwise_inference` defaults to
# True on v3 alone, so leaving it implicit would make `logits` mean
# different things either side of `inference_row_chunk_size` rows.
BASE = dataclasses.replace(
    model.get_default_performance_options(), use_chunkwise_inference=False
)


def perf(*, spmf=None, stage_chunked=False):
    return dataclasses.replace(
        BASE,
        save_peak_memory_factor=spmf,
        use_chunkwise_inference=stage_chunked,
    )


# The architecture takes (rows, batch, columns) with the test rows
# appended to the train rows, and labels for the train rows only.
x = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
with torch.no_grad():
    logits = model(x, y_train, only_return_standard_out=True,
                   performance_options=perf())

    # The three paths that exist purely to make inference cheaper, each
    # run its own way so the R side is graded against the mechanism
    # rather than against the plain pass it is supposed to agree with.
    #
    # `save_peak_memory_factor` splits every sublayer's work into chunks.
    logits_chunked = model(
        x.clone(), y_train, only_return_standard_out=True,
        performance_options=perf(spmf=args.save_peak_memory_factor),
    )

    # `use_chunkwise_inference` is the other axis: it drives stages 0-2 a
    # row chunk at a time, with the distribution embedder's inducing
    # summaries precomputed over the training rows in column chunks, so
    # the `(1, rows, columns, embed)` tensor is never resident whole. It
    # only does anything above `inference_row_chunk_size` rows -- below
    # that the loop runs once and this is the pass above.
    logits_stage_chunked = model(
        x.clone(), y_train, only_return_standard_out=True,
        performance_options=perf(stage_chunked=True),
    )

    # Both at once, which is what the R port will do by default once it
    # carries the stage chunking.
    logits_stage_chunked_spmf = model(
        x.clone(), y_train, only_return_standard_out=True,
        performance_options=perf(spmf=args.save_peak_memory_factor,
                                 stage_chunked=True),
    )

    # The KV cache conditions on the training rows once, then predicts the
    # test rows without them present. Unlike v2.6's, this cache holds
    # everything the training rows contribute, so it is not merely cheaper
    # -- it is the same computation.
    _, cache = model(
        X_train.unsqueeze(1), y_train, only_return_standard_out=True,
        return_kv_cache=True, performance_options=perf(),
    )
    logits_cached = model(
        X_test.unsqueeze(1), y_train, only_return_standard_out=True,
        kv_cache=cache, x_is_test_only=True, performance_options=perf(),
    )

# (n_test, batch, n_out) -> (n_test, n_out); the batch is one dataset.
logits = logits[:, 0, :].contiguous()

tensors = {
    "x_train": X_train.contiguous(),
    "x_test": X_test.contiguous(),
    "y_train": y_train.contiguous(),
    "logits": logits,
    "logits_chunked": logits_chunked[:, 0, :].contiguous(),
    "logits_stage_chunked": logits_stage_chunked[:, 0, :].contiguous(),
    "logits_stage_chunked_spmf":
        logits_stage_chunked_spmf[:, 0, :].contiguous(),
    "logits_cached": logits_cached[:, 0, :].contiguous(),
}

tempered = logits / args.softmax_temperature
quantiles = [float(q) for q in args.quantiles.split(",")]
if TASK == "classification":
    # The head is `max_num_classes` wide (160 in the released checkpoint);
    # only the classes present in the fixture are decoded, which is what
    # the estimator does.
    n_classes = int(y_train.max().item()) + 1
    tensors["probs"] = tempered[:, :n_classes].softmax(-1).contiguous()
    n_bar_bins = None
else:
    # v3 registers the borders on the module itself rather than hanging
    # them off a `criterion` submodule, and registers them on both heads.
    borders = model.regression_borders.to(torch.float32)
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
    "architecture": "tabpfn_v3",
    "head": "classifier" if model.task_type == "multiclass" else "regressor",
    "n_out": int(logits.shape[-1]),
    "n_classes": n_classes,
    "n_bar_bins": n_bar_bins,
    "softmax_temperature": args.softmax_temperature,
    "quantiles": quantiles,
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "n_features": int(X_train.shape[1]),
    "save_peak_memory_factor": args.save_peak_memory_factor,
    "row_chunk_size": ROW_CHUNK,
    "col_chunk_size": COL_CHUNK,
    # Whether the stage-chunked pass actually took a different path than
    # the plain one. False on every fixture below `row_chunk_size` rows,
    # where the loop runs exactly once.
    "stage_chunking_active": bool(
        ROW_CHUNK < int(X_train.shape[0]) + int(X_test.shape[0])
    ),
}
(OUT / "reference.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
