#!/usr/bin/env python3
"""Convert a TabICL `.ckpt` into safetensors + config.json.

One-shot, offline. The R package needs only the converted artifacts at
inference time and never invokes Python afterwards.

TabICL checkpoints are plain `torch.save` dicts with two keys,
`state_dict` and `config`, so this is a much smaller job than the TabPFN
converter: no Lightning unwrapping, no key canonicalization, no
hyperparameter inference. The config is copied through as-is, with an
`arch` marker added so the R backend registry can recognise it.

Usage
-----
    python tabicl_convert_ckpt.py \\
        --src  tabicl-classifier-v2-20260212.ckpt \\
        --dst  ckpts/converted/tabicl-v2-clf

The checkpoint can be fetched with:

    hf download jingang/TabICL tabicl-classifier-v2-20260212.ckpt
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="Path to the .ckpt file.")
parser.add_argument("--dst", required=True,
                    help="Output directory for model.safetensors + config.json.")
args = parser.parse_args()

src = Path(args.src)
dst = Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)

# weights_only=False: the checkpoint carries a plain config dict alongside
# the tensors. Only convert checkpoints you trust.
ckpt = torch.load(src, map_location="cpu", weights_only=False)
if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
    sys.exit(f"{src} does not look like a TabICL checkpoint "
             f"(expected a dict with a 'state_dict' key).")

state_dict = ckpt["state_dict"]
config = dict(ckpt.get("config", {}))

tensors = {}
for k, v in state_dict.items():
    t = v.detach()
    if t.is_floating_point():
        t = t.to(torch.float32)
    tensors[k] = t.contiguous().cpu()

# `max_classes == 0` marks a regressor; the head then emits `num_quantiles`
# values per row instead of class logits.
is_classifier = int(config.get("max_classes", 0)) > 0
config["arch"] = "tabicl"
config["head"] = "classifier" if is_classifier else "regressor"
config["state_dict_keys"] = list(tensors.keys())
config["state_dict_shapes"] = {k: list(v.shape) for k, v in tensors.items()}
config["source_ckpt"] = src.name
config["source_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()

save_file(tensors, str(dst / "model.safetensors"))
with open(dst / "config.json", "w") as fh:
    json.dump(config, fh, indent=2)

n_params = sum(v.numel() for v in tensors.values())
print(f"[py] {src.name}: {len(tensors)} tensors, {n_params / 1e6:.1f} M params "
      f"({config['head']}) -> {dst}")
