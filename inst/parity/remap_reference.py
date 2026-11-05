#!/usr/bin/env python3
"""Reference outputs for the regressor's border translation.

Every TabPFN regressor ensemble member's bar distribution is moved onto
the shared z-space grid by `translate_probs_across_borders()`. `tabpfn`
9.1 replaced that function (PriorLabs/TabPFN#1288): each destination
bucket's mass became a sum of non-negative overlap terms, and the two
outer source buckets are read as half-normal tails. This pins
`translate_probs_across_borders_r()` in `R/dist-bar.R` against the
reference function itself, on grids built to stress it, with no
checkpoint needed. Needs `tabpfn` >= 9.1:

    .venvs/ref/bin/python inst/parity/remap_reference.py \
        --out inst/parity/transforms/remap.safetensors
    gzip -9 -f inst/parity/transforms/remap.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tabpfn
import torch
from safetensors.torch import save_file
from tabpfn.utils import translate_probs_across_borders

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True,
                    help="Path to the .safetensors file; a sibling .json "
                         "holds the versions.")
args = parser.parse_args()

rng = np.random.RandomState(1288)

# The destination grid: z-space-like, dense in the middle, as the
# checkpoint's own borders are, and wide enough to reach past every source
# grid below on both sides.
u = np.linspace(-1.0, 1.0, 301)
to = np.sign(u) * (np.abs(u) ** 1.5) * 6.0

grids = {}
# A member with a target transform: the same grid pushed through a
# monotone map, so its buckets bunch up where the destination's do not.
# In float32, as the reference holds borders.
grids["transformed"] = np.sinh(to / 2.0) * 0.8 + 0.1
# Narrow buckets where neighbouring float32 CDF values round together:
# the case the old difference-of-CDFs translation zeroed out.
fine = np.linspace(-0.002, 0.002, 41) + 0.5
grids["narrow"] = np.unique(np.concatenate([np.linspace(-4.0, 4.0, 120), fine]))
# A zero-width interior bucket: a point mass, which goes whole to the
# destination bucket that holds its point.
pm = np.linspace(-3.0, 5.0, 150)
grids["point_mass"] = np.sort(np.concatenate([pm, [pm[70]]]))
# A grid narrower than the destination on both sides, so the destination's
# outer buckets are filled from the half-normal tails alone.
grids["inside"] = np.linspace(-2.0, 2.5, 90)

out: dict[str, torch.Tensor] = {
    "to": torch.tensor(to, dtype=torch.float32),
}
for name, frm in grids.items():
    frm_t = torch.tensor(frm, dtype=torch.float32)
    n = frm_t.shape[0] - 1
    logits = torch.tensor(rng.randn(6, n) * 2.0, dtype=torch.float32)
    # One row peaked hard on a single bucket, one with all its mass in the
    # outer tails: the two shapes a smooth random row never produces.
    logits[4] = -30.0
    logits[4, n // 2] = 10.0
    logits[5] = -30.0
    logits[5, 0] = 5.0
    logits[5, -1] = 5.0
    out[f"{name}_frm"] = frm_t
    out[f"{name}_logits"] = logits
    out[f"{name}_probs"] = translate_probs_across_borders(
        logits, frm=frm_t, to=out["to"]).contiguous()

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
save_file(out, str(out_path))
out_path.with_suffix(".json").write_text(json.dumps(
    {"grids": list(grids), "_versions": {"tabpfn": tabpfn.__version__,
                                         "torch": torch.__version__}},
    indent=2) + "\n")
print(f"wrote {out_path}")
