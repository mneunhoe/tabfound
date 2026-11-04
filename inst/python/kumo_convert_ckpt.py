#!/usr/bin/env python3
"""Convert an NVIDIA Kumo Tabular `.pt` into safetensors + config.json.

One-shot, offline. The R package needs only the converted artifacts at
inference time and never invokes Python afterwards.

Kumo Tabular checkpoints are bare `torch.save` state dicts: no config, no
Lightning wrapper. The reference (`structured-data-models`, `sdm`) keeps
the hyperparameters in code (`sdm/models/kumo/tabular/model.py`,
`MODEL_KWARGS`), so this script derives them from the tensor shapes,
matches them against that table to name the size, and writes them out.
The one setting the weights cannot carry -- how many KV heads the query
rows attend with -- comes from the table.

The state dict is verified key for key and shape for shape against the
module tree that size implies, so a checkpoint from a later release that
changed the architecture fails here rather than at load time.

Usage
-----
    python kumo_convert_ckpt.py \\
        --src  small/classifier.pt \\
        --dst  ckpts/converted/kumo-tabular-small-classifier

The checkpoint can be fetched with:

    hf download nvidia/Kumo-Tabular small/classifier.pt --revision v1.0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    sys.stderr.write("ERROR: PyTorch is required. `pip install torch`.\n")
    raise

try:
    from safetensors.torch import save_file
except ImportError:
    sys.stderr.write("ERROR: safetensors is required. `pip install safetensors`.\n")
    raise


# `sdm` 0.2.1, `MODEL_KWARGS`. Only the fields the weights cannot show
# are needed to tell the sizes apart, but all of them are kept so the
# match is checked on every dimension.
SIZES = {
    "small": dict(cell_channels=128, num_embedding_layers=4,
                  num_embedding_heads=4, num_inducing_points=128,
                  icl_channels=512, num_icl_layers=12, num_icl_heads=8,
                  num_icl_key_value_heads_for_query=None),
    "medium": dict(cell_channels=256, num_embedding_layers=6,
                   num_embedding_heads=4, num_inducing_points=256,
                   icl_channels=512, num_icl_layers=24, num_icl_heads=8,
                   num_icl_key_value_heads_for_query=2),
    "large": dict(cell_channels=256, num_embedding_layers=6,
                  num_embedding_heads=4, num_inducing_points=256,
                  icl_channels=1024, num_icl_layers=24, num_icl_heads=16,
                  num_icl_key_value_heads_for_query=2),
}


def fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def count(sd: dict, pattern: str) -> int:
    rx = re.compile(pattern)
    return len({m.group(1) for k in sd if (m := rx.match(k))})


def shape(sd: dict, key: str) -> list[int]:
    if key not in sd:
        fail(f"expected tensor `{key}` is missing; not a Kumo Tabular checkpoint?")
    return list(sd[key].shape)


def derive(sd: dict) -> dict:
    """Every hyperparameter the shapes determine."""
    re_ = "row_embedding."
    D = shape(sd, re_ + "norm.weight")[0]
    E = shape(sd, "icl_block.norm.weight")[0]
    G, F = shape(sd, re_ + "cell_embedding.num_freq")
    K = shape(sd, re_ + "readout_token")[0]
    n_emb = count(sd, r"row_embedding\.col_blocks\.(\d+)\.")
    n_icl = count(sd, r"icl_block\.layers\.(\d+)\.")
    M = shape(sd, re_ + "col_blocks.0.inducing_points")[0]
    H = shape(sd, re_ + "row_blocks.0.attn.sdpa.query_scaling.head_scale")[0]
    He = shape(sd, "icl_block.layers.0.attn.sdpa.query_scaling.head_scale")[0]
    gate = shape(sd, re_ + "row_blocks.0.attn.sdpa.query_scaling.gate.0.weight")[0]
    out = shape(sd, "icl_block.head.2.weight")[0]

    is_clf = (re_ + "y_emb.weight") in sd
    if is_clf == ((re_ + "y_lin.weight") in sd):
        fail("checkpoint has both or neither of `y_emb` and `y_lin`")

    # The RoPE table is a frozen parameter initialised as
    # 1 / theta^(2i / hd); recover theta from its second entry rather than
    # assuming it. The table is float32, so the recovered value is only
    # good to a few digits (99999.97); it is informational -- the R side
    # loads the table itself.
    hd = D // H
    inv = sd[re_ + "row_blocks.0.attn.query_transform.0.inv_freq"].double()
    theta = float((1.0 / inv[1]) ** (hd / 2.0)) if inv.numel() > 1 else 1e5
    theta = float(f"{theta:.4g}")

    return dict(
        cell_channels=D, num_embedding_layers=n_emb, num_embedding_heads=H,
        num_inducing_points=M, group_size=G, num_frequencies=F,
        num_readout_tokens=K, icl_channels=E, num_icl_layers=n_icl,
        num_icl_heads=He, gate_hidden=gate,
        rope_theta=theta,
        num_classes=out if is_clf else 0,
        num_quantiles=0 if is_clf else out,
    )


def expected_shapes(h: dict) -> dict[str, list[int]]:
    """The module tree `sdm` builds for these hyperparameters."""
    D, E, H = h["cell_channels"], h["icl_channels"], h["num_embedding_heads"]
    He, M, K = h["num_icl_heads"], h["num_inducing_points"], h["num_readout_tokens"]
    G, F, gate = h["group_size"], h["num_frequencies"], h["gate_hidden"]
    is_clf = h["num_classes"] > 0
    out = h["num_classes"] or h["num_quantiles"]
    s: dict[str, list[int]] = {}

    def block(p: str, c: int, heads: int | None, gated: bool = False,
              rope: bool = False) -> None:
        s[p + "query_norm.weight"] = [c]
        s[p + "key_value_norm.weight"] = [c]
        s[p + "attn.qkv_lin.weight"] = [3 * c, c]
        s[p + "attn.qkv_lin.bias"] = [3 * c]
        s[p + "attn.out_lin.weight"] = [c, c]
        s[p + "attn.out_lin.bias"] = [c]
        if heads is not None:
            s[p + "attn.sdpa.query_scaling.head_scale"] = [heads]
        hd = c // (heads or 1)
        if gated:
            q = p + "attn.sdpa.query_scaling.gate."
            s[q + "0.weight"] = [gate, hd]
            s[q + "0.bias"] = [gate]
            s[q + "2.weight"] = [hd, gate]
            s[q + "2.bias"] = [hd]
        if rope:
            s[p + "attn.query_transform.0.inv_freq"] = [hd // 2]
            s[p + "attn.key_transform.0.inv_freq"] = [hd // 2]
        s[p + "mlp.0.weight"] = [c]
        s[p + "mlp.1.weight"] = [2 * c, c]
        s[p + "mlp.1.bias"] = [2 * c]
        s[p + "mlp.3.weight"] = [c, 2 * c]
        s[p + "mlp.3.bias"] = [c]

    ce = "row_embedding.cell_embedding."
    s[ce + "num_freq"] = [G, F]
    s[ce + "cat_freq"] = [G, F]
    for lin in ("num_lin", "cat_lin"):
        s[ce + lin + ".weight"] = [D, 2 * F]
        s[ce + lin + ".bias"] = [D]
    s[ce + "nan_lin.weight"] = [D, G]
    if is_clf:
        s["row_embedding.y_emb.weight"] = [h["num_classes"], D]
        s["icl_block.y_emb.weight"] = [h["num_classes"], E]
    else:
        s["row_embedding.y_lin.weight"] = [D, 1]
        s["icl_block.y_lin.weight"] = [E, 1]
    s["row_embedding.readout_token"] = [K, D]
    for i in range(h["num_embedding_layers"]):
        cb = f"row_embedding.col_blocks.{i}."
        s[cb + "inducing_points"] = [M, D]
        block(cb + "inducing_block.", D, H)
        block(cb + "output_block.", D, None)
        block(f"row_embedding.row_blocks.{i}.", D, H, gated=True, rope=True)
    s["row_embedding.norm.weight"] = [D]
    if K * D != E:
        s["row_project.weight"] = [E, K * D]
        s["row_project.bias"] = [E]
    for j in range(h["num_icl_layers"]):
        block(f"icl_block.layers.{j}.", E, He)
    s["icl_block.norm.weight"] = [E]
    s["icl_block.head.0.weight"] = [2 * E, E]
    s["icl_block.head.0.bias"] = [2 * E]
    s["icl_block.head.2.weight"] = [out, 2 * E]
    s["icl_block.head.2.bias"] = [out]
    return s


def match_size(h: dict) -> str:
    for name, ref in SIZES.items():
        if all(h[k] == v for k, v in ref.items() if k in h):
            return name
    dims = {k: h[k] for k in SIZES["small"] if k in h}
    fail(f"the checkpoint's dimensions match no known Kumo Tabular size: {dims}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to the .pt file.")
    parser.add_argument("--dst", required=True,
                        help="Output directory for model.safetensors + config.json.")
    parser.add_argument("--head", choices=["classifier", "regressor"],
                        help="Fail unless the checkpoint is this head.")
    parser.add_argument("--size", choices=sorted(SIZES),
                        help="Fail unless the checkpoint is this size.")
    parser.add_argument("--source-repo", default="nvidia/Kumo-Tabular")
    parser.add_argument("--source-revision", default=None)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # A bare state dict, so `weights_only=True` loads it without
    # unpickling anything but tensors -- the same call `sdm` makes.
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict) or not sd or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor)
            for k, v in sd.items()):
        fail(f"{src} is not a flat str -> Tensor state dict.")

    h = derive(sd)
    size = match_size(h)
    head = "classifier" if h["num_classes"] > 0 else "regressor"
    if args.head and args.head != head:
        fail(f"{src.name} is a {head}, but a {args.head} was asked for.")
    if args.size and args.size != size:
        fail(f"{src.name} is the {size} model, but {args.size} was asked for.")

    want = expected_shapes(h)
    got = {k: list(v.shape) for k, v in sd.items()}
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    wrong = sorted(k for k in set(want) & set(got) if want[k] != got[k])
    if missing or extra or wrong:
        fail("state dict does not match the Kumo Tabular module tree:\n"
             f"  missing: {missing[:8]}\n  unexpected: {extra[:8]}\n"
             f"  wrong shape: {[(k, got[k], want[k]) for k in wrong[:8]]}")

    # The RoPE table is one parameter shared by every row block, and the
    # state dict lists it under each of them. safetensors refuses aliased
    # storage, and the R loader wants one tensor per module path anyway,
    # so every entry gets its own copy.
    tensors = {}
    for k, v in sd.items():
        t = v.detach()
        if t.is_floating_point():
            t = t.to(torch.float32)
        tensors[k] = t.contiguous().cpu().clone()

    config = {
        "arch": "kumo_tabular",
        "head": head,
        "size": size,
        **h,
        "num_icl_key_value_heads_for_query":
            SIZES[size]["num_icl_key_value_heads_for_query"],
        "license": "OpenMDW-1.1",
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "source_ckpt": src.name,
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "state_dict_keys": list(tensors.keys()),
        "state_dict_shapes": {k: list(v.shape) for k, v in tensors.items()},
    }

    dst.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst / "model.safetensors"))
    with open(dst / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    n_params = sum(v.numel() for v in tensors.values())
    print(f"[py] {src.name}: {len(tensors)} tensors, {n_params / 1e6:.1f} M params "
          f"({size} {head}) -> {dst}")


if __name__ == "__main__":
    main()
