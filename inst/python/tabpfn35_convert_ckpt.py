#!/usr/bin/env python3
"""Convert a published TabPFN v3.5 `.safetensors` into the package's artifacts.

One-shot, offline. The R package needs only the converted artifacts at
inference time and never invokes Python afterwards.

This is the smallest of the three converters, because Prior Labs publishes
v3.5 as safetensors rather than as a pickled `.ckpt`. There is nothing to
unpickle, no Lightning wrapper to strip and no hyperparameters to infer:
the file's own `__metadata__` header already carries `architecture_name`,
`config` and `inference_config`, each as a JSON string. So the job is to
lift the config out of the header, write it as `config.json` in the shape
the R backend registry expects, and put the tensors where
`resolve_artifacts()` looks for them.

The repo's top-level `config.json` is *not* that config -- it holds only
`{"model_name": "TabPFN-v3.5"}`. The real one is in the header of each
checkpoint, which is what lets the three published files differ
(`nlayers`, `feat_agg_num_heads`, `N_ESTIMATORS`) while sharing a name.

Because the tensors are already float32 safetensors, the fast path copies
the file rather than round-tripping it through torch -- 876 MB for the base
checkpoint. A checkpoint carrying any non-float32 floating tensor falls
back to a load-cast-save pass, so the artifact the R loader sees is float32
either way.

Usage
-----
    python tabpfn35_convert_ckpt.py \\
        --src  tabpfn-v3.5-20260909.safetensors \\
        --dst  ckpts/converted/tabpfn-v3.5

The checkpoint can be fetched with:

    hf download Prior-Labs/tabpfn_3_5 tabpfn-v3.5-20260909.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    sys.stderr.write("ERROR: safetensors is required. `pip install safetensors`.\n")
    raise


ARCHITECTURE_NAME = "tabpfn_v3_5"

# Flags the v3.5 checkpoints still carry in their config but which the
# reference implementation no longer reads: `tabpfn_v3_5.py` mentions none
# of them, so the behaviour they used to gate is now unconditional. Loading
# a checkpoint that turns one off would silently give the wrong answer, the
# same trap the v3 converter guards against for `use_rope` and
# `layernorm_elementwise_affine`, so refuse instead.
ALWAYS_ON_FLAGS = (
    "use_rope",
    "use_qk_norm",
    "use_mlp_heads",
    "use_fourier_cell_embedding",
    "y_encoder_layernorm",
    "layernorm_elementwise_affine",
    "use_nan_indicators",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_header(src: Path) -> tuple[dict, dict, dict]:
    """Return (metadata, config, tensor shapes) without materialising tensors."""
    with safe_open(str(src), framework="pt", device="cpu") as fh:
        metadata = fh.metadata() or {}
        keys = list(fh.keys())
        shapes = {k: list(fh.get_slice(k).get_shape()) for k in keys}
        dtypes = {k: fh.get_slice(k).get_dtype() for k in keys}
    return metadata, {"keys": keys, "shapes": shapes, "dtypes": dtypes}, {}


def _json_field(metadata: dict, name: str):
    raw = metadata.get(name)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        # `architecture_name` is stored as a JSON string, so it decodes to a
        # plain str; anything that is not JSON at all is passed through.
        return raw


def _recast_to_float32(src: Path, dst_weights: Path) -> None:
    """Fallback for a checkpoint holding non-float32 floating tensors."""
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError:
        sys.exit(
            "ERROR: this checkpoint holds non-float32 tensors, which needs "
            "PyTorch to recast. `pip install torch`."
        )
    tensors = {}
    for k, v in load_file(str(src)).items():
        t = v.detach()
        if t.is_floating_point():
            t = t.to(torch.float32)
        tensors[k] = t.contiguous().cpu()
    save_file(tensors, str(dst_weights))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Path to the published .safetensors checkpoint.")
    parser.add_argument("--dst", required=True,
                        help="Output directory for model.safetensors + config.json.")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    metadata, header, _ = _read_header(src)

    arch = _json_field(metadata, "architecture_name")
    if arch != ARCHITECTURE_NAME:
        sys.exit(
            f"{src.name} declares architecture {arch!r}, not "
            f"{ARCHITECTURE_NAME!r}. Convert v3 checkpoints with "
            f"tabpfn_convert_ckpt.py instead."
        )

    config = _json_field(metadata, "config")
    if not isinstance(config, dict):
        sys.exit(f"{src.name} has no 'config' object in its safetensors header.")

    for flag in ALWAYS_ON_FLAGS:
        if flag in config and not config[flag]:
            sys.exit(
                f"{src.name} sets {flag}=false. The v3.5 reference "
                f"implementation no longer reads that flag -- the behaviour is "
                f"unconditional -- so this checkpoint cannot be loaded faithfully."
            )

    max_num_classes = int(config.get("max_num_classes", 0))
    num_buckets = int(config.get("num_buckets", 0))
    if max_num_classes < 2 or num_buckets < 2:
        sys.exit(
            f"{src.name} is not a multitask checkpoint: max_num_classes="
            f"{max_num_classes}, num_buckets={num_buckets}; v3.5 needs both."
        )

    # Cross-check the two head widths against the tensors that realise them,
    # the way the v3 converter checks `output_projection` against `n_out`.
    shapes = header["shapes"]
    for key, want, what in (
        ("heads.output_projection.weight", num_buckets, "num_buckets"),
        ("icl_y_encoder.multiclass.embedding.weight", max_num_classes,
         "max_num_classes"),
        ("heads.regression_borders", num_buckets + 1, "num_buckets + 1"),
    ):
        if key not in shapes:
            sys.exit(f"{src.name} is missing the tensor {key!r}.")
        got = shapes[key][0]
        if got != want:
            sys.exit(
                f"{src.name}: {key} has leading dimension {got}, but the "
                f"config says {what} is {want}."
            )

    config = dict(config)
    config["arch"] = ARCHITECTURE_NAME
    config["tabpfn_version"] = "3.5"
    # One checkpoint serves both tasks, so there is no head to name. The
    # shared `tabpfn_task_of()` switches on exactly "classifier"/"regressor"
    # and returns NULL otherwise, which is the "no opinion" answer
    # `load_backend_model()` needs to let both `tabular_classifier()` and
    # `tabular_regressor()` accept this artifact.
    config["head"] = "multitask"
    config["n_out_classification"] = max_num_classes
    config["n_out_regression"] = num_buckets
    config["n_bar_bins"] = num_buckets
    config["borders_key"] = "heads.regression_borders"
    config["state_dict_keys"] = header["keys"]
    config["state_dict_shapes"] = shapes
    config["source_ckpt"] = src.name
    config["source_sha256"] = _sha256(src)

    # The publisher's preprocessing recipe. Not read by the R backend, which
    # takes its ensemble recipe from a Python dump, but it is the only record
    # of how these weights are meant to be driven -- N_ESTIMATORS in
    # particular differs between the base and fast checkpoints.
    inference_config = _json_field(metadata, "inference_config")
    if isinstance(inference_config, dict):
        config["inference_config"] = inference_config

    dst_weights = dst / "model.safetensors"
    non_f32 = sorted(
        k for k, d in header["dtypes"].items()
        if d in ("BF16", "F16", "F64")
    )
    if non_f32:
        print(f"[py] recasting {len(non_f32)} non-float32 tensors via torch")
        _recast_to_float32(src, dst_weights)
    else:
        # Already float32 safetensors with the right keys: a copy is exact and
        # skips materialising ~0.9 GB of tensors.
        shutil.copyfile(src, dst_weights)

    with open(dst / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    n_params = 0
    for shape in shapes.values():
        n = 1
        for d in shape:
            n *= d
        n_params += n
    print(f"[py] {src.name}: {len(shapes)} tensors, {n_params / 1e6:.1f} M params "
          f"(multitask, {config['nlayers']} ICL layers) -> {dst}")


if __name__ == "__main__":
    main()
