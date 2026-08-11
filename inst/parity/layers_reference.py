#!/usr/bin/env python3
"""Reference outputs for the shared transformer layers.

Instantiates the reference implementation's own layer classes with small
random weights, runs them on small random inputs, and dumps weights,
inputs and outputs together. The R side then builds the same layer, loads
the same weights, and compares.

This is the check that does not need the 6.5 GB checkpoint: a broken
RMSNorm or a mis-indexed RoPE shows up here in milliseconds instead of
as a vague activation drift twenty layers into a full model.

    .venvs/ref/bin/python inst/parity/layers_reference.py \\
        --out inst/parity/layers/layers.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tabfm.src.pytorch.model import (
    MLP,
    RMSNorm,
    RoPE,
    InducedSelfAttentionBlock,
    MultiheadAttention,
    MultiheadAttentionBlock,
    OneHotAndLinear,
)

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()

torch.manual_seed(0)
out: dict[str, torch.Tensor] = {}
meta: dict = {}


def record(prefix: str, module: torch.nn.Module) -> None:
    for k, v in module.state_dict().items():
        out[f"{prefix}.{k}"] = v.detach().clone().float().contiguous()


# --- RMSNorm -------------------------------------------------------------
d = 8
ln = RMSNorm(d)
ln.weight.data.normal_(mean=1.0, std=0.3)
x = torch.randn(2, 5, d)
out["rmsnorm.x"] = x
out["rmsnorm.y"] = ln(x).detach()
record("rmsnorm", ln)
meta["rmsnorm_dim"] = d

# --- RoPE ----------------------------------------------------------------
head_dim, base = 8, 100000.0
rp = RoPE(head_dim, base)
# Perturb the buffer so the test would fail if R recomputed it from
# `base` instead of loading it, which is exactly the mistake to catch.
rp.freqs.data = rp.freqs.data * 1.07
xr = torch.randn(2, 6, 3, head_dim)
out["rope.x"] = xr
out["rope.freqs"] = rp.freqs.detach().clone().float()
out["rope.y"] = rp.rotate(xr).detach()
meta["rope_dim"] = head_dim
meta["rope_base"] = base

# --- MultiheadAttention (per-head norm + learned scale) ------------------
dm, nh = 16, 4
at = MultiheadAttention(dm, nh, rope_base=None)
for p in at.parameters():
    p.data.normal_(std=0.4)
q = torch.randn(2, 5, dm)
k = torch.randn(2, 7, dm)
v = torch.randn(2, 7, dm)
mask = torch.ones(2, 1, 1, 7, dtype=torch.bool)
mask[:, :, :, 5:] = False
out["attn.q"], out["attn.k"], out["attn.v"] = q, k, v
out["attn.mask"] = mask.float()
out["attn.y"] = at(q, k, v).detach()
out["attn.y_masked"] = at(q, k, v, attn_mask=mask).detach()
record("attn", at)
meta["attn_dim"], meta["attn_heads"] = dm, nh

# --- MultiheadAttentionBlock (sandwich norms + SwiGLU) -------------------
ff = dm * 4
blk = MultiheadAttentionBlock(dm, nh, ff, activation="swiglu")
blk.ffn_chunk_size = None
for p in blk.parameters():
    p.data.normal_(std=0.3)
xb = torch.randn(2, 5, dm)
out["mab.x"] = xb
out["mab.y_self"] = blk(xb).detach()
out["mab.y_cross"] = blk(xb, k, v).detach()
record("mab", blk)
meta["mab_ff"] = ff

# --- InducedSelfAttentionBlock -------------------------------------------
num_inds = 3
isb = InducedSelfAttentionBlock(dm, nh, ff, num_inds)
for m in isb.modules():
    if hasattr(m, "ffn_chunk_size"):
        m.ffn_chunk_size = None
for p in isb.parameters():
    p.data.normal_(std=0.3)
xs = torch.randn(2, 6, dm)
smask = torch.ones(2, 1, 1, 6, dtype=torch.bool)
smask[:, :, :, 4:] = False
out["isab.x"] = xs
out["isab.mask"] = smask.float()
out["isab.y"] = isb(xs).detach()
out["isab.y_masked"] = isb(xs, attn_mask=smask).detach()
record("isab", isb)
meta["isab_num_inds"] = num_inds

# --- MLP (gelu-tanh between layers) --------------------------------------
mlp = MLP(6, [11], 4, activation="gelu")
for p in mlp.parameters():
    p.data.normal_(std=0.5)
xm = torch.randn(2, 5, 6)
out["mlp.x"] = xm
out["mlp.y"] = mlp(xm).detach()
record("mlp", mlp)
meta["mlp_in"], meta["mlp_hidden"], meta["mlp_out"] = 6, 11, 4

# --- OneHotAndLinear ------------------------------------------------------
ohl = OneHotAndLinear(4, 5)
for p in ohl.parameters():
    p.data.normal_(std=0.5)
yv = torch.tensor([[0, 1, 3, -100, 9], [2, 2, -1, 0, 1]], dtype=torch.float32)
out["onehot.y"] = yv
out["onehot.out"] = ohl(yv).detach()
record("onehot", ohl)
meta["onehot_classes"], meta["onehot_dim"] = 4, 5

# --- TabICL layers -------------------------------------------------------
# Different family, same idea: build the reference's own classes with
# random weights so the R port can be checked without a checkpoint.
from tabicl._model.layers import (  # noqa: E402
    InducedSelfAttentionBlock as IclISAB,
    MultiheadAttentionBlock as IclMAB,
)
from tabicl._model.rope import RotaryEmbedding as IclRoPE  # noqa: E402
from tabicl._model.ssmax import QASSMaxMLP  # noqa: E402

torch.manual_seed(1)

# Scalable softmax (query-aware, elementwise).
icl_nh, icl_hd = 4, 8
ss = QASSMaxMLP(icl_nh, icl_hd, elementwise=True)
for p_ in ss.parameters():
    p_.data.normal_(std=0.4)
q_ss = torch.randn(2, icl_nh, 5, icl_hd)
out["iclssmax.q"] = q_ss
out["iclssmax.y_n7"] = ss(q_ss, 7).detach()
out["iclssmax.y_n64"] = ss(q_ss, 64).detach()
record("iclssmax", ss)
meta["iclssmax_nh"], meta["iclssmax_hd"] = icl_nh, icl_hd

# Non-interleaved RoPE with a learnable frequency table.
icl_rope = IclRoPE(dim=icl_hd, theta=100000.0, interleaved=False)
icl_rope.freqs.data = icl_rope.freqs.data * 0.93
x_icl_rope = torch.randn(2, icl_nh, 6, icl_hd)
out["iclrope.x"] = x_icl_rope
out["iclrope.freqs"] = icl_rope.freqs.detach().clone().float()
out["iclrope.y"] = icl_rope.rotate_queries_or_keys(x_icl_rope).detach()
meta["iclrope_dim"], meta["iclrope_base"] = icl_hd, 100000.0

# Pre-norm block with packed q/k/v, both LayerNorm variants.
icl_dm = icl_nh * icl_hd
for tag, bias_free in (("iclmab", False), ("iclmab_nobias", True)):
    blk = IclMAB(icl_dm, icl_nh, icl_dm * 2, dropout=0.0, activation="gelu",
                 norm_first=True, bias_free_ln=bias_free,
                 ssmax="qassmax-mlp-elementwise")
    for p_ in blk.parameters():
        p_.data.normal_(std=0.25)
    xb2 = torch.randn(2, 5, icl_dm)
    kb2 = torch.randn(2, 7, icl_dm)
    out[f"{tag}.x"] = xb2
    out[f"{tag}.k"] = kb2
    out[f"{tag}.y_self"] = blk(xb2).detach()
    out[f"{tag}.y_cross"] = blk(xb2, kb2, kb2).detach()
    out[f"{tag}.y_trainsize"] = blk(xb2, train_size=3).detach()
    record(tag, blk)
meta["iclmab_dim"], meta["iclmab_heads"] = icl_dm, icl_nh

# Induced set-attention block, with and without the train-size slice.
icl_isab = IclISAB(icl_dm, icl_nh, icl_dm * 2, num_inds=3, dropout=0.0,
                   activation="gelu", norm_first=True, bias_free_ln=False,
                   ssmax="qassmax-mlp-elementwise")
for p_ in icl_isab.parameters():
    p_.data.normal_(std=0.25)
x_is = torch.randn(2, 6, icl_dm)
out["iclisab.x"] = x_is
out["iclisab.y"] = icl_isab(x_is).detach()
out["iclisab.y_trainsize"] = icl_isab(x_is, train_size=4).detach()
record("iclisab", icl_isab)
meta["iclisab_num_inds"] = 3

# --- Mitra layers --------------------------------------------------------
# Optional: needs the `mitrapkg` shim on PYTHONPATH (see the parity
# README). Skipped rather than fatal so this script still runs without it.
try:
    from mitrapkg._internal.models.embedding import Tab2DQuantileEmbeddingX
    from mitrapkg._internal.models.tab2d import Layer as MitraLayer

    torch.manual_seed(2)

    # The quantile-rank embedding carries no parameters at all, so this
    # pins the arithmetic on its own.
    qe = Tab2DQuantileEmbeddingX(16)
    xs = torch.randn(2, 40, 5)
    xs[:, :, 4] = 3.0                      # a constant column -> zero variance
    xq = torch.randn(2, 7, 5)
    xq[:, :, 4] = 3.0
    pad_obs = torch.zeros(2, 40, dtype=torch.bool)
    pad_feat = torch.zeros(2, 5, dtype=torch.bool)
    out["mitraq.x_support"] = xs.clone()
    out["mitraq.x_query"] = xq.clone()
    qs, qq = qe(xs.clone(), xq.clone(), pad_obs, pad_feat)
    out["mitraq.support"] = qs.detach()
    out["mitraq.query"] = qq.detach()

    mdim, mheads = 32, 4
    ml = MitraLayer(mdim, mheads, use_flash_attn=False)
    for p_ in ml.parameters():
        p_.data.normal_(std=0.2)
    sup = torch.randn(2, 6, 4, mdim)
    qry = torch.randn(2, 3, 4, mdim)
    out["mitralayer.support"] = sup
    out["mitralayer.query"] = qry
    s_out, q_out = ml(sup, qry, None, None, 2, None, None, None)
    out["mitralayer.support_out"] = s_out.detach()
    out["mitralayer.query_out"] = q_out.detach()
    record("mitralayer", ml)
    meta["mitralayer_dim"], meta["mitralayer_heads"] = mdim, mheads
    meta["has_mitra"] = True
except ImportError:
    meta["has_mitra"] = False

from safetensors.torch import save_file  # noqa: E402

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, str(out_path))

meta["torch_version"] = torch.__version__
with open(out_path.with_suffix(".json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"[py] wrote {len(out)} tensors to {out_path}")
# Gzip the result. `file(1)` has a notoriously loose DOS-.COM heuristic --
# a first byte of 0xb8 reads as an x86 MOV -- and safetensors starts with
# a raw little-endian header length, so a benign tensor dump can be
# reported as "COM executable for DOS". `R CMD check` shells out to
# `file` and raises "Found the following executable file". Gzip's magic
# number is unambiguous, and this cannot regress when the contents change.
import gzip, os, shutil  # noqa: E402

with open(out_path, "rb") as fh_in, gzip.open(str(out_path) + ".gz", "wb") as fh_out:
    shutil.copyfileobj(fh_in, fh_out)
os.remove(out_path)
print(f"[py] gzipped -> {out_path}.gz")

