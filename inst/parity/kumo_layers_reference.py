#!/usr/bin/env python3
"""Weight-free reference outputs for the Kumo Tabular layers.

The Kumo counterpart of `layers_reference.py`, kept separate because it
needs `structured-data-models` (`sdm`), which lives in its own venv.
Instantiates `sdm`'s own classes with small random weights, runs them on
small random inputs, and dumps weights, inputs and outputs together, so
`tests/testthat/test-nn-layers.R` can check the R port with no
checkpoint at all.

Besides the single blocks it dumps two tiny whole networks: a classifier
shaped like the medium model (a `row_project` and grouped-query attention
for the query rows) and a regressor shaped like small (no projection, all
heads). That is the only parity the GQA path gets without downloading
medium's 246 MB.

Every parameter is redrawn: `sdm` zero-initialises the output
projections, MLP outputs and gates, which would make every block an
identity and hide any bug behind it.

    .venvs/sdm/bin/python inst/parity/kumo_layers_reference.py \\
        --out inst/parity/layers/kumo_layers.safetensors
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.cell_embedding import CellEmbedding
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.nn import (
    GatedLogScale,
    InducedTransformerBlock,
    LogScale,
    RMSNorm,
    RotaryEmbedding,
)

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()

torch.manual_seed(0)
out: dict[str, torch.Tensor] = {}
meta: dict = {}


def randomize(module: torch.nn.Module) -> None:
    for name, p in module.named_parameters():
        if name.endswith("inv_freq"):
            continue
        with torch.no_grad():
            if name.endswith("head_scale"):
                p.uniform_(0.5, 1.5)
            elif name.endswith("norm.weight") or name.endswith("mlp.0.weight"):
                p.normal_(mean=1.0, std=0.2)
            else:
                p.normal_(std=0.3)


def record(prefix: str, module: torch.nn.Module) -> None:
    for k, v in module.state_dict().items():
        out[f"{prefix}.{k}"] = v.detach().clone().float().contiguous()


# --- RMSNorm (eps = None -> float32 eps) ----------------------------------
d = 8
ln = RMSNorm(d)
randomize(ln)
x = torch.randn(2, 5, d)
out["kumo_rmsnorm.x"] = x
with torch.no_grad():
    out["kumo_rmsnorm.y"] = ln(x)
record("kumo_rmsnorm", ln)
meta["kumo_rmsnorm_dim"] = d

# --- RoPE (split half, positions on the axis before the heads) -----------
hd = 8
rope = RotaryEmbedding(hd, layout="split_half", theta=100_000, requires_grad=False)
x = torch.randn(3, 6, 2, hd)   # [B, S, H, hd]
out["kumo_rope.x"] = x
with torch.no_grad():
    out["kumo_rope.y"] = rope(x)
record("kumo_rope", rope)
meta["kumo_rope_dim"] = hd

# --- LogScale / GatedLogScale ---------------------------------------------
H = 2
ls = LogScale(num_heads=H)
randomize(ls)
q = torch.randn(3, 5, H, hd)
out["kumo_logscale.x"] = q
with torch.no_grad():
    out["kumo_logscale.y"] = ls(q, key_len=7)
record("kumo_logscale", ls)

gls = GatedLogScale(channels=hd, num_heads=H, hidden_channels=16)
randomize(gls)
out["kumo_gatedlogscale.x"] = q
with torch.no_grad():
    out["kumo_gatedlogscale.y"] = gls(q, key_len=7)
record("kumo_gatedlogscale", gls)
meta["kumo_scale_heads"], meta["kumo_scale_hidden"], meta["kumo_scale_keys"] = H, 16, 7

# --- Transformer block: plain, with RoPE + gated scaling, and GQA --------
C, H = 16, 4
blk = KumoTabularTransformerBlock(C, H, query_scaling=LogScale(num_heads=H))
randomize(blk)
qx = torch.randn(2, 5, C)
kvx = torch.randn(2, 9, C)
out["kumo_block.q"], out["kumo_block.kv"] = qx, kvx
with torch.no_grad():
    out["kumo_block.y"] = blk(qx, kvx)
    # The ICL query pass: K/V from the context with all heads, then only
    # the first two kept for the query rows.
    _, kv = blk(kvx, kvx, return_key_value=True)
    from sdm.cache import KVCacheEntry
    kv2 = KVCacheEntry(kv.key[..., :2, :].contiguous(), kv.value[..., :2, :].contiguous())
    out["kumo_block.y_gqa"] = blk(qx, kv2)
record("kumo_block", blk)
meta["kumo_block_dim"], meta["kumo_block_heads"] = C, H

rope_blk = RotaryEmbedding(C // H, layout="split_half", theta=100_000, requires_grad=False)
rblk = KumoTabularTransformerBlock(
    C, H,
    query_scaling=GatedLogScale(channels=C // H, num_heads=H, hidden_channels=16),
    rope=rope_blk,
)
randomize(rblk)
rx = torch.randn(3, 7, C)
out["kumo_rowblock.x"] = rx
with torch.no_grad():
    out["kumo_rowblock.y"] = rblk(rx, rx)
    out["kumo_rowblock.y_last"] = rblk(rx[..., :2, :], rx)
record("kumo_rowblock", rblk)

# --- Induced block (columns as batch) ------------------------------------
M = 4
ib = InducedTransformerBlock(
    channels=C, num_inducing_points=M,
    inducing_block=KumoTabularTransformerBlock(C, H, query_scaling=LogScale(num_heads=H)),
    output_block=KumoTabularTransformerBlock(C, H, query_scaling=None),
)
randomize(ib)
ix = torch.randn(3, 10, C)   # [cols, rows, C]
out["kumo_induced.x"] = ix
with torch.no_grad():
    out["kumo_induced.y"] = ib(ix, ix[..., :6, :])
record("kumo_induced", ib)
meta["kumo_induced_inducing"], meta["kumo_induced_train"] = M, 6

# --- Cell embedding: NaN, an all-missing column, categoricals, C < 4 -----
D, G, F = 16, 3, 8
ce = CellEmbedding(channels=D, group_size=G, num_frequencies=F)
randomize(ce)
with torch.no_grad():
    ce.nan_lin.weight.normal_(std=0.3)
for tag, ncol in (("", 5), ("_narrow", 2)):
    cx = torch.randn(1, 12, ncol)
    cx[0, 3, 0] = float("nan")
    cx[0, 9, ncol - 1] = float("nan")       # a query row
    if ncol > 2:
        cx[0, :, 2] = float("nan")          # all missing
        cx[0, :, 1] = torch.randint(0, 4, (12,)).float()
    cm = torch.zeros(1, ncol, dtype=torch.bool)
    cm[0, min(1, ncol - 1)] = True
    out[f"kumo_cells{tag}.x"] = cx
    out[f"kumo_cells{tag}.mask"] = cm.to(torch.uint8)
    with torch.no_grad():
        out[f"kumo_cells{tag}.y"] = ce(cx, cm, train_size=8).clone()
record("kumo_cells", ce)
meta["kumo_cells_dim"], meta["kumo_cells_group"], meta["kumo_cells_freq"] = D, G, F
meta["kumo_cells_train"] = 8

# --- Two tiny whole networks ---------------------------------------------
TINY = {
    "clf": dict(num_classes=10, num_quantiles=0, cell_channels=16,
                num_embedding_layers=2, num_embedding_heads=2,
                num_inducing_points=4, group_size=3, num_frequencies=8,
                num_readout_tokens=4, icl_channels=32, num_icl_layers=2,
                num_icl_heads=4, num_icl_key_value_heads_for_query=2),
    "reg": dict(num_classes=0, num_quantiles=9, cell_channels=16,
                num_embedding_layers=2, num_embedding_heads=2,
                num_inducing_points=4, group_size=3, num_frequencies=8,
                num_readout_tokens=2, icl_channels=32, num_icl_layers=2,
                num_icl_heads=4, num_icl_key_value_heads_for_query=None),
}
for tag, kw in TINY.items():
    net = _KumoTabular(**kw)
    randomize(net)
    net.eval()
    nx = torch.randn(2, 14, 5)           # two members
    nx[0, 2, 1] = float("nan")
    nx[1, 11, 4] = float("nan")
    nx[:, :, 3] = torch.randint(0, 3, (2, 14)).float()
    nm = torch.tensor([False, False, False, True, False])
    if tag == "clf":
        ny = torch.randint(0, 3, (2, 9))
    else:
        ny = torch.randn(2, 9)
    out[f"kumo_net_{tag}.x"] = nx
    out[f"kumo_net_{tag}.y"] = ny.float()
    out[f"kumo_net_{tag}.mask"] = nm.to(torch.uint8)
    with torch.no_grad():
        out[f"kumo_net_{tag}.out"] = net(nx.clone(), ny, nm.expand(2, -1))
    record(f"kumo_net_{tag}", net)
    meta[f"kumo_net_{tag}"] = {
        **{k: v for k, v in kw.items()},
        "arch": "kumo_tabular",
        "head": "classifier" if tag == "clf" else "regressor",
        "gate_hidden": 64,
    }

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous().clone() for k, v in out.items()}, str(out_path))

meta["torch_version"] = torch.__version__
meta["sdm_version"] = importlib.metadata.version("structured-data-models")
with open(out_path.with_suffix(".json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"[py] wrote {len(out)} tensors to {out_path}")

# Gzipped for the same reason as layers_reference.py: `file(1)` can call
# a raw safetensors header a DOS executable, which `R CMD check` flags.
with open(out_path, "rb") as fh_in, gzip.open(str(out_path) + ".gz", "wb") as fh_out:
    shutil.copyfileobj(fh_in, fh_out)
os.remove(out_path)
print(f"[py] gzipped -> {out_path}.gz")
