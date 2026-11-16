"""Tiny synthetic MTP (same code paths, random weights, hidden 32) + synthetic GMTP shards.

Used by selftest_train.py to exercise train_mtp.py end to end on a CPU in seconds
(< 1 GB): data stream, chained loss, optimizer, checkpoints, resume, in-loop eval.
"""
from __future__ import annotations

import types
from pathlib import Path

import torch
import torch.nn as nn

import mtp_net as N
import quant_sim as Q
R = N.R


def _q8(w: torch.Tensor) -> "R.Q8Tensor":
    d, q = Q.q8_0_quant(w)
    t = R.Q8Tensor.__new__(R.Q8Tensor)
    t.qs, t.d, t.shape = q, d.float(), list(w.shape)
    return t


def make_tiny(seed: int = 0, vocab: int = 1000, **cfg_over):
    """cfg_over: [stage2] MtpConfig overrides (e.g. experts=512, used=10 for the structural
    launch profile in profile_ops.py); defaults unchanged."""
    g = torch.Generator().manual_seed(seed)
    kw = dict(hidden=32, hc=4, low_rank=8, heads=2, kv_heads=1, head_dim=16, rotary_dim=8,
              rope_theta=1e4, idx_heads=1, idx_dim=8, idx_top_k=2048, ratio=4, experts=4, used=2,
              expert_ff=32, shared_ff=32, eps=1e-6, vocab=vocab, layer=48)
    kw.update(cfg_over)
    cfg = R.MtpConfig(**kw)
    H, D, LR, hd = cfg.hidden, cfg.D, cfg.low_rank, cfg.head_dim
    shapes = {"enorm": [H], "hnorm": [D], "eh_proj": [H, 2 * H],
              "hc_attn_norm": [D], "hc_attn_down": [LR, D], "hc_attn_up": [D, LR], "hc_attn_inject": [cfg.hc, D],
              "attn_q": [cfg.heads * 2 * hd, H], "attn_k": [cfg.kv_heads * hd, H], "attn_v": [cfg.kv_heads * hd, H],
              "attn_output": [H, cfg.heads * hd], "attn_q_norm": [hd], "attn_k_norm": [hd],
              "indexer_q": [cfg.idx_heads * cfg.idx_dim, H], "indexer_k": [cfg.idx_dim, H],
              "indexer_q_norm": [cfg.idx_dim], "indexer_k_norm": [cfg.idx_dim],
              "hc_ffn_norm": [D], "hc_ffn_down": [LR, D], "hc_ffn_up": [D, LR], "hc_ffn_inject": [cfg.hc, D],
              "router": [cfg.experts, H], "shexp_gate_inp": [H], "shexp_gate": [cfg.shared_ff, H],
              "shexp_up": [cfg.shared_ff, H], "shexp_down": [H, cfg.shared_ff],
              "hc_head_norm": [D], "hc_head_down": [LR, D], "hc_head_up": [D, LR]}
    ref = types.SimpleNamespace(cfg=cfg, q8={}, _experts_q8={})
    for _, attr in R.DENSE_NAMES:
        s = shapes[attr]
        w = (1.0 + 0.1 * torch.randn(*s, generator=g)) if len(s) == 1 else torch.randn(*s, generator=g) / s[-1] ** 0.5
        setattr(ref, attr, nn.Parameter(w))
        ref.q8[attr] = len(s) == 2 and s[-1] % 32 == 0 and attr not in ("router",)
    for attr, s in (("gate_exps", [cfg.experts, cfg.expert_ff, H]), ("up_exps", [cfg.experts, cfg.expert_ff, H]),
                    ("down_exps", [cfg.experts, H, cfg.expert_ff])):
        setattr(ref, attr, nn.Parameter(torch.randn(*s, generator=g) / s[-1] ** 0.5))
        ref.q8[attr] = True
    embd = _q8(torch.randn(vocab, H, generator=g))
    head = _q8(torch.randn(vocab, H, generator=g) / H ** 0.5)
    target = N.FrozenTarget(embd, head, [], {"synthetic": True})
    return ref, target


def write_shards(out: Path, n_records: int = 6, n_tokens: int = 40, seed: int = 0) -> list[Path]:
    """Synthetic GMTP v1 shards (read_gmtp reference writer) with hidden 32 x hc 4."""
    import random
    G = N.read_gmtp
    rnd = random.Random(seed)
    G.ST_SHAPE.update(hidden=32, hc_count=4, vocab=1000, topk=32, max_context=4096)
    paths = []
    for split_ord, name in ((7, "train"), (8, "val")):
        G.ST_RECORDS[:] = [{"id": f"tiny-{name}-{i}", "source_id": f"tiny/{i}.txt",
                            "tokens": [rnd.randrange(1000) for _ in range(n_tokens)],
                            "supervise_from": 0, "metadata": {"kind": "selftest"}} for i in range(n_records)]
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        paths.append(G._write_shard(d, split_ord, False))
    return paths
