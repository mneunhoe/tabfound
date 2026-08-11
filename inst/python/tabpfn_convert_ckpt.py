#!/usr/bin/env python3
"""Convert a Prior-Labs TabPFN `.ckpt` into safetensors + config.json.

This is a one-shot, offline utility. The R package `tabpfn` only needs
`.safetensors` + `config.json` at inference time and never invokes Python.

Usage
-----
    python convert_ckpt.py \
        --src   /path/to/tabpfn-v2-classifier-v2_default.ckpt \
        --dst-weights out/model.safetensors \
        --dst-config  out/config.json \
        --head  classifier        # or: regressor

Three architectures are handled, told apart by the checkpoint's own
`architecture_name` field:

* ``base`` (TabPFN v2 / v2.5) -> ``arch: "per_feature_transformer"``
* ``tabpfn_v2_6``             -> ``arch: "tabpfn_v2_6"``
* ``tabpfn_v3``               -> ``arch: "tabpfn_v3"``

They share a training pipeline but not a module tree, so the emitted
config names the architecture and the R side dispatches on it.

Design notes
------------
* State-dict keys are preserved as-is. For v2 / v2.5 that means TabPFN's
  custom `MultiHeadAttention` fused per-head parameters `_w_qkv`
  (shape `[3, n_heads, d_k, emb]`) and `_w_out` (`[n_heads, d_v, emb]`)
  rather than `nn.MultiheadAttention`'s `in_proj_weight`; v2.6 uses
  ordinary separate `q/k/v/out_projection` linears.
* Buffers (notably the regressor `criterion.borders`) are preserved verbatim.
* `use_pre_norm` in the emitted v2/v2.5 config is forced to `false`
  regardless of the ckpt's stored hparam, because the reference
  implementation explicitly disables pre-norm (the forward pass raises
  AssertionError when `pre_norm=True`).
* `config.json` contains every hyperparameter the R side needs to
  reconstruct the model, plus the full ordered `state_dict_keys` list
  so the loader can assert nothing was missed.

This script requires `torch` and `safetensors` in the Python env that
runs it (any recent PyTorch is fine). Run it **once per checkpoint**.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import torch
except ImportError as e:
    sys.stderr.write("ERROR: PyTorch is required. `pip install torch`.\n")
    raise

try:
    from safetensors.torch import save_file as safetensors_save_file
except ImportError as e:
    sys.stderr.write(
        "ERROR: safetensors is required. `pip install safetensors`.\n"
    )
    raise


# ---------------------------------------------------------------------------
# Checkpoint unwrapping
# ---------------------------------------------------------------------------

NESTED_STATE_KEYS = ("state_dict", "model_state_dict", "model")


def _unwrap_state_dict(obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Return (state_dict, hyperparameters) from a raw `torch.load` result."""
    hparams: Dict[str, Any] = {}

    if isinstance(obj, dict):
        for hp_key in ("hyper_parameters", "hparams", "config"):
            if hp_key in obj and isinstance(obj[hp_key], dict):
                hparams = dict(obj[hp_key])
                break

        for k in NESTED_STATE_KEYS:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                if all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return dict(sd), hparams
                return _unwrap_state_dict(sd)[0], hparams

        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return dict(obj), hparams

    raise RuntimeError("Could not find a tensor state_dict inside the checkpoint.")


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Key canonicalization
# ---------------------------------------------------------------------------

def _canonicalize(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    sd = _strip_prefix(state_dict, "model.")
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if v.is_floating_point():
            v = v.detach().to(torch.float32).contiguous().cpu()
        else:
            v = v.detach().contiguous().cpu()
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Shape inference (best-effort crosscheck of hparams)
# ---------------------------------------------------------------------------

def _infer_hparams(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer embedding_dim, n_layers, n_heads, mlp_hidden_dim, etc."""
    info: Dict[str, Any] = {}

    layer_idx = -1
    for k in state_dict:
        m = re.match(r"transformer_encoder\.layers\.(\d+)\.", k)
        if m is not None:
            layer_idx = max(layer_idx, int(m.group(1)))
    if layer_idx >= 0:
        info["n_layers"] = layer_idx + 1

    # embedding dim: take the last axis of the first attention _w_qkv we see.
    for k, v in state_dict.items():
        if k.endswith("._w_qkv") and v.ndim == 4:
            info.setdefault("embedding_dim", int(v.shape[-1]))
            info.setdefault("n_heads", int(v.shape[1]))
            break

    # Transformer-layer MLP hidden dim: from
    #   transformer_encoder.layers.0.mlp.linear1.weight  shape (hidden, emb)
    k_txf_mlp = "transformer_encoder.layers.0.mlp.linear1.weight"
    if k_txf_mlp in state_dict and state_dict[k_txf_mlp].ndim == 2:
        info["mlp_hidden_dim"] = int(state_dict[k_txf_mlp].shape[0])

    # Encoder final-step MLP hidden dim (regressor only).
    #   encoder.5.mlp.0.weight  shape (hidden, in_channels)
    k_enc_mlp = "encoder.5.mlp.0.weight"
    if k_enc_mlp in state_dict and state_dict[k_enc_mlp].ndim == 2:
        info["encoder_mlp_hidden_dim"] = int(state_dict[k_enc_mlp].shape[0])

    return info


# ---------------------------------------------------------------------------
# Config emission
# ---------------------------------------------------------------------------

def _hparam_get(hparams: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in hparams:
            return hparams[k]
        for top_v in hparams.values():
            if isinstance(top_v, dict) and k in top_v:
                return top_v[k]
    return default


def _find_borders_key(state_dict: Dict[str, torch.Tensor]) -> str | None:
    candidates = [k for k in state_dict if k.endswith("borders") or k.endswith(".borders")]
    if not candidates:
        return None
    candidates.sort(key=lambda k: (-len(k), k))
    return candidates[0]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_config_v2_6(
    state_dict: Dict[str, torch.Tensor],
    hparams: Dict[str, Any],
    head: str,
    src_path: Path,
) -> Dict[str, Any]:
    """Emit the config for the `tabpfn_v2_6` architecture.

    The v2.6 checkpoints carry a small, complete hyperparameter dict, so
    almost nothing has to be inferred from tensor shapes. What is derived:

    * ``mlp_hidden_dim`` -- the transformer block's feedforward width. The
      reference sets it to ``2 * emsize`` (``TabPFNV2p6.hidden_size``)
      rather than storing it, and the same value is the hidden width of
      ``output_projection``.
    * ``n_out_classes`` / ``n_bar_bins`` -- the output width, cross-checked
      against ``output_projection.2.weight``.
    """
    emsize = int(_hparam_get(hparams, "emsize", default=192))
    max_num_classes = int(_hparam_get(hparams, "max_num_classes", default=0))
    num_buckets = int(_hparam_get(hparams, "num_buckets", default=5000))

    # `max_num_classes > 0` is exactly how the reference's get_architecture()
    # decides classification vs regression, so it is the authority on which
    # head this checkpoint is -- not the caller's flag.
    ckpt_head = "classifier" if max_num_classes > 0 else "regressor"
    if head != ckpt_head:
        raise ValueError(
            f"--head {head!r} contradicts the checkpoint: max_num_classes="
            f"{max_num_classes} means this is a {ckpt_head}."
        )

    n_out = max_num_classes if ckpt_head == "classifier" else num_buckets
    out_key = "output_projection.2.weight"
    if out_key in state_dict and int(state_dict[out_key].shape[0]) != n_out:
        raise ValueError(
            f"{out_key} has {state_dict[out_key].shape[0]} outputs but the "
            f"hyperparameters imply {n_out}."
        )

    layernorm_type = _hparam_get(hparams, "layernorm_type", default="rmsnorm")
    if layernorm_type != "rmsnorm":
        raise ValueError(
            f"Only 'rmsnorm' is implemented for tabpfn_v2_6, got {layernorm_type!r}."
        )

    config: Dict[str, Any] = {
        "arch": "tabpfn_v2_6",
        "tabpfn_version": "2.6",
        "head": ckpt_head,
        "n_layers": int(_hparam_get(hparams, "nlayers", default=24)),
        "embedding_dim": emsize,
        "n_heads": int(_hparam_get(hparams, "nhead", default=3)),
        # TabPFNV2p6.hidden_size, used for both the block MLP and the
        # hidden width of output_projection.
        "mlp_hidden_dim": 2 * emsize,
        "encoder_type": _hparam_get(hparams, "encoder_type", default="linear"),
        "encoder_mlp_hidden_dim": int(
            _hparam_get(hparams, "encoder_mlp_hidden_dim", default=1024)
        ),
        "features_per_group": int(_hparam_get(hparams, "features_per_group", default=3)),
        "num_thinking_rows": int(_hparam_get(hparams, "num_thinking_rows", default=64)),
        "layernorm_type": layernorm_type,
        "layernorm_elementwise_affine": bool(
            _hparam_get(hparams, "layernorm_elementwise_affine", default=True)
        ),
        "activation": "gelu",
        # Only "subspace" is supported by the reference; it raises otherwise.
        "feature_positional_embedding": "subspace",
        "dtype": "float32",
    }

    if ckpt_head == "classifier":
        config["n_out_classes"] = n_out
        config["n_bar_bins"] = None
        config["borders_key"] = None
    else:
        borders_key = _find_borders_key(state_dict)
        config["n_out_classes"] = None
        config["n_bar_bins"] = (
            int(state_dict[borders_key].shape[0]) - 1
            if borders_key is not None
            else num_buckets
        )
        config["borders_key"] = borders_key

    config["source_ckpt"] = src_path.name
    config["source_sha256"] = _file_sha256(src_path)
    config["state_dict_keys"] = list(state_dict.keys())
    config["state_dict_shapes"] = {k: list(v.shape) for k, v in state_dict.items()}
    return config


def build_config_v3(
    state_dict: Dict[str, torch.Tensor],
    hparams: Dict[str, Any],
    head: str,
    src_path: Path,
) -> Dict[str, Any]:
    """Emit the config for the `tabpfn_v3` architecture.

    A v3 checkpoint stores its `TabPFNV3Config` verbatim under the
    top-level ``config`` key, so every field is copied across rather than
    inferred. The fields are passed through under their reference names --
    ``embed_dim``, ``nlayers``, ``icl_num_kv_heads_test`` and the rest --
    because the R backend is a port of that one file and reusing its
    vocabulary keeps the two readable side by side.

    Two things are *not* passthrough:

    * ``head``, which the reference derives rather than stores:
      ``max_num_classes >= 2`` means classification, anything else
      regression.
    * ``n_out``, the decoder's width: ``max_num_classes`` for the
      classifier (the many-class decoder's one-hot axis) and
      ``num_buckets`` for the regressor (the bar distribution's bins).
    """
    def _get(key, default=None):
        return _hparam_get(hparams, key, default=default)

    max_num_classes = int(_get("max_num_classes", 0))
    num_buckets = int(_get("num_buckets", 5000))

    # `get_architecture()` decides the task from max_num_classes alone, so
    # the checkpoint -- not the caller's flag -- is the authority here.
    ckpt_head = "classifier" if max_num_classes >= 2 else "regressor"
    if head != ckpt_head:
        raise ValueError(
            f"--head {head!r} contradicts the checkpoint: max_num_classes="
            f"{max_num_classes} means this is a {ckpt_head}."
        )

    n_out = max_num_classes if ckpt_head == "classifier" else num_buckets
    # The regressor's decoder is an nn.Sequential; its last linear pins n_out.
    out_key = "output_projection.2.weight"
    if out_key in state_dict and int(state_dict[out_key].shape[0]) != n_out:
        raise ValueError(
            f"{out_key} has {state_dict[out_key].shape[0]} outputs but the "
            f"config implies {n_out}."
        )

    if not bool(_get("use_rope", True)):
        # Without RoPE the column aggregator has no `rope.freqs` tensor and
        # the feature axis loses its ordering. No released checkpoint does
        # this; refuse rather than load something the port has never seen.
        raise ValueError("tabpfn_v3 checkpoints with use_rope=false are not supported.")
    if not bool(_get("layernorm_elementwise_affine", True)):
        raise ValueError(
            "tabpfn_v3 checkpoints with layernorm_elementwise_affine=false are "
            "not supported: the R port's RMSNorm always carries a weight."
        )

    config: Dict[str, Any] = {
        "arch": "tabpfn_v3",
        "tabpfn_version": "3",
        "head": ckpt_head,
        "max_num_classes": max_num_classes,
        "num_buckets": num_buckets,
        "n_out": n_out,
        # Stage 0: cell embedding + feature grouping.
        "embed_dim": int(_get("embed_dim", 128)),
        "feature_group_size": int(_get("feature_group_size", 3)),
        "use_nan_indicators": bool(_get("use_nan_indicators", True)),
        # Stage 1: per-column distribution embedder.
        "dist_embed_num_blocks": int(_get("dist_embed_num_blocks", 3)),
        "dist_embed_num_heads": int(_get("dist_embed_num_heads", 8)),
        "dist_embed_num_inducing_points": int(
            _get("dist_embed_num_inducing_points", 128)
        ),
        # Stage 2: cross-feature aggregation onto CLS tokens.
        "feat_agg_num_blocks": int(_get("feat_agg_num_blocks", 3)),
        "feat_agg_num_heads": int(_get("feat_agg_num_heads", 8)),
        "feat_agg_num_cls_tokens": int(_get("feat_agg_num_cls_tokens", 4)),
        "feat_agg_rope_base": float(_get("feat_agg_rope_base", 100_000)),
        "use_rope": True,
        # Stage 3: the in-context-learning transformer.
        "nlayers": int(_get("nlayers", 24)),
        "icl_num_heads": int(_get("icl_num_heads", 8)),
        "icl_num_kv_heads": (
            None if _get("icl_num_kv_heads") is None else int(_get("icl_num_kv_heads"))
        ),
        "icl_num_kv_heads_test": (
            None
            if _get("icl_num_kv_heads_test") is None
            else int(_get("icl_num_kv_heads_test"))
        ),
        # Decoder.
        "decoder_head_dim": int(_get("decoder_head_dim", 64)),
        "decoder_num_heads": int(_get("decoder_num_heads", 6)),
        "decoder_use_softmax_scaling": bool(_get("decoder_use_softmax_scaling", False)),
        # Shared.
        "ff_factor": int(_get("ff_factor", 2)),
        "softmax_scaling_mlp_hidden_dim": int(
            _get("softmax_scaling_mlp_hidden_dim", 64)
        ),
        "layernorm_elementwise_affine": True,
        "activation": "gelu",
        "dtype": "float32",
    }

    # The bar-distribution borders are a registered buffer on both heads,
    # so a classifier carries them too; only the regressor reads them.
    borders_key = "regression_borders"
    if borders_key in state_dict:
        config["borders_key"] = borders_key
        config["n_bar_bins"] = int(state_dict[borders_key].shape[0]) - 1
    else:
        config["borders_key"] = None
        config["n_bar_bins"] = num_buckets

    config["source_ckpt"] = src_path.name
    config["source_sha256"] = _file_sha256(src_path)
    config["state_dict_keys"] = list(state_dict.keys())
    config["state_dict_shapes"] = {k: list(v.shape) for k, v in state_dict.items()}
    return config


def build_config(
    state_dict: Dict[str, torch.Tensor],
    hparams: Dict[str, Any],
    head: str,
    src_path: Path,
) -> Dict[str, Any]:
    inferred = _infer_hparams(state_dict)

    embedding_dim = _hparam_get(hparams, "emsize", "embedding_dim", "d_model") \
        or inferred.get("embedding_dim")
    n_layers = _hparam_get(hparams, "nlayers", "num_layers", "n_layers") \
        or inferred.get("n_layers")
    n_heads = _hparam_get(hparams, "nhead", "num_heads", "n_heads")
    mlp_hidden_dim = _hparam_get(hparams, "dim_feedforward", "mlp_hidden_dim") \
        or inferred.get("mlp_hidden_dim")
    nhid_factor = _hparam_get(hparams, "nhid_factor")
    if mlp_hidden_dim is None and nhid_factor is not None and embedding_dim is not None:
        mlp_hidden_dim = int(nhid_factor) * int(embedding_dim)
    encoder_mlp_hidden_dim = inferred.get("encoder_mlp_hidden_dim")

    tabpfn_version = _hparam_get(hparams, "tabpfn_version", "version", default="2.5")

    config: Dict[str, Any] = {
        "arch": "per_feature_transformer",
        "tabpfn_version": str(tabpfn_version),
        "head": head,
        "n_layers": int(n_layers) if n_layers is not None else None,
        "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
        "n_heads": int(n_heads) if n_heads is not None else None,
        "mlp_hidden_dim": int(mlp_hidden_dim) if mlp_hidden_dim is not None else None,
        "encoder_mlp_hidden_dim": int(encoder_mlp_hidden_dim)
            if encoder_mlp_hidden_dim is not None else None,
        "activation": _hparam_get(hparams, "activation", default="gelu"),
        "layer_norm_eps": float(_hparam_get(hparams, "layer_norm_eps", default=1e-5)),
        # PerFeatureEncoderLayer.forward explicitly raises if pre_norm=True;
        # the reference implementation runs post-norm regardless of the
        # stored hparam. Emit `false` so the R port matches actual behavior.
        "use_pre_norm": False,
        "features_per_group": _hparam_get(hparams, "features_per_group", default=1),
        "max_num_features": _hparam_get(hparams, "max_num_features", default=500),
        "max_num_samples": _hparam_get(hparams, "max_num_samples", default=10000),
        "feature_positional_embedding": _hparam_get(
            hparams, "feature_positional_embedding", default="none"
        ),
        "dtype": "float32",
    }

    if head == "classifier":
        n_out_classes = _hparam_get(hparams, "num_classes", "n_classes", "n_out_classes")
        if n_out_classes is None:
            # Read the final output dim of the standard decoder. The head's
            # last linear is at index `.2` in the nn.Sequential.
            for key in (
                "decoder_dict.standard.2.weight",
                "decoder_dict.classification.2.weight",
            ):
                if key in state_dict and state_dict[key].ndim == 2:
                    n_out_classes = int(state_dict[key].shape[0])
                    break
        config["n_out_classes"] = int(n_out_classes) if n_out_classes is not None else None
        config["n_bar_bins"] = None
        config["borders_key"] = None
    elif head == "regressor":
        borders_key = _find_borders_key(state_dict)
        n_bar_bins: int | None = None
        if borders_key is not None:
            n_bar_bins = int(state_dict[borders_key].shape[0]) - 1
        config["n_bar_bins"] = n_bar_bins
        config["borders_key"] = borders_key
        config["n_out_classes"] = None
    else:
        raise ValueError(f"--head must be 'classifier' or 'regressor', got {head!r}")

    config["source_ckpt"] = src_path.name
    config["source_sha256"] = _file_sha256(src_path)
    config["state_dict_keys"] = list(state_dict.keys())
    config["state_dict_shapes"] = {k: list(v.shape) for k, v in state_dict.items()}
    return config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def convert(src: Path, dst_weights: Path, dst_config: Path, head: str) -> None:
    print(f"Reading checkpoint: {src}")
    raw = torch.load(src, map_location="cpu", weights_only=False)

    architecture = raw.get("architecture_name") if isinstance(raw, dict) else None
    state_dict, hparams = _unwrap_state_dict(raw)
    print(
        f"  found {len(state_dict)} tensors, {len(hparams)} hyperparameters"
        f", architecture={architecture!r}"
    )

    state_dict = _canonicalize(state_dict)
    print(f"  after canonicalization: {len(state_dict)} tensors")

    if architecture == "tabpfn_v3":
        config = build_config_v3(state_dict, hparams, head, src)
    elif architecture == "tabpfn_v2_6":
        config = build_config_v2_6(state_dict, hparams, head, src)
    else:
        config = build_config(state_dict, hparams, head, src)

    # v3 names its shapes differently; nothing to cross-check by these keys.
    if config["arch"] == "tabpfn_v3":
        required = ("nlayers", "embed_dim", "icl_num_heads")
    else:
        required = ("n_layers", "embedding_dim", "n_heads")
    missing = [k for k in required if config.get(k) is None]
    if missing:
        print(
            "WARNING: could not determine: " + ", ".join(missing) +
            " — you may need to fill these in manually in config.json."
        )

    dst_weights.parent.mkdir(parents=True, exist_ok=True)
    dst_config.parent.mkdir(parents=True, exist_ok=True)

    safetensors_save_file(state_dict, str(dst_weights))
    print(f"Wrote weights: {dst_weights}  ({dst_weights.stat().st_size / 1e6:.1f} MB)")

    with dst_config.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote config:  {dst_config}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--src", required=True, type=Path, help="Path to .ckpt")
    parser.add_argument("--dst-weights", required=True, type=Path, help="Output .safetensors")
    parser.add_argument("--dst-config", required=True, type=Path, help="Output config.json")
    parser.add_argument(
        "--head",
        required=True,
        choices=["classifier", "regressor"],
        help="Which output head this checkpoint is for",
    )
    args = parser.parse_args(argv)
    convert(args.src, args.dst_weights, args.dst_config, args.head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
