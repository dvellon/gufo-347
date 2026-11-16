"""Shared library for stage-1 MTP drafter fine-tuning (train / eval / export).

`MtpNet` is a device-aware, vectorized re-implementation of the validated
reference `../reference/mtp_ref.py` (which ports gufo's oracle
`ReferenceModel::MtpStep` operator for operator). It keeps the reference's
numerics switches (`decode=True` = gufo decode storage: Q8 32-block activation
quantization before every Q8_0 weight, F16 K/V, F16-rounded router and shared
gate) and its chained-drafting semantics (`MtpDrafter.block_unrolled`):

  * chain p (anchor token t_{p+1}) drafts d_{p,1..D};
  * depth 1 row p   : MTP position p,   token t_{p+1},   hidden h_p (trunk final wide residual);
  * depth j>=2 row p: MTP position p+j-1, token d_{p,j-1} (gufo) or t_{p+j} (teacher),
                      hidden = the block's own carried residual of depth j-1 (gufo's
                      MtpForward(hidden_row=-1));
  * attention of depth j row p sees depth-1 K/V of rows <= p (catch-up / DraftReplay
    rows) plus its own chain's rows of depths 2..j (draft KV, rewound after verify);
  * depth j of chain p predicts token p+j+1 and is supervised by the capture's sup
    row of position p+j (target distribution after t_{<=p+j}).

`selftest_net.py` checks MtpNet against mtp_ref (float32 block_unrolled and the
sequential decode-numerics gufo chain).

Only rows whose label exists are computed (depth j: rows p <= n-1-j), so every
computed position is < n <= 2048: always gufo's dense attention regime (QSA
block selection starts at n_kv >= 2052); the indexer is therefore not evaluated.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference"))
sys.path.insert(0, str(HERE.parent / "capture"))
import mtp_ref as R  # noqa: E402
import read_gmtp  # noqa: E402

DRAFT_VOCAB = 49152                        # production 38ff7eeb (promoted 2026-10-26 ~13:55) uses the 49K prefix;
                                           # src/draft_vocab.hpp still says 98304 (pre-promotion; eval --draft-prefix 98304)
DRAFT_EXTRA_ROWS = R.MTP_DRAFT_EXTRA_ROWS  # 1024
DRAFT_WINDOW = R.MTP_DRAFT_VOCAB_WINDOW    # 32
CONF_FLOOR = R.GREEDY_DRAFT_CONFIDENCE_FLOOR  # 0.5
MAX_DRAFTS = 7                             # mtp_policy.hpp kMaxMtpDraftTokens

# Dense tensors that receive gradients. Indexer projections only steer QSA block
# selection beyond 2048 positions; they are never trained and are exported unchanged.
NO_TRAIN = {"indexer_q", "indexer_k", "indexer_q_norm", "indexer_k_norm"}
EXPERTS = [a for _, a in R.EXPERT_NAMES]   # gate_exps, up_exps, down_exps
ATTR_TO_GGUF = {attr: f"blk.48.{suffix}" for suffix, attr in R.DENSE_NAMES + R.EXPERT_NAMES}


# ---------------------------------------------------------------------------
# numerics helpers (device-aware copies of mtp_ref's)
# ---------------------------------------------------------------------------
def half(x):
    return x.to(torch.float16).to(x.dtype)


def q8_act(x):
    return R.q8_act(x)


def rms_norm(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = y.to(x.dtype)
    return y if w is None else y * w.to(x.dtype)


def rope(x, pos, rotary_dim, theta):
    half_dim = rotary_dim // 2
    i = torch.arange(half_dim, dtype=torch.float32, device=x.device)
    freq = torch.pow(torch.tensor(theta, dtype=torch.float32, device=x.device), -2.0 * i / rotary_dim)
    angle = pos.to(torch.float32)[:, None] * freq[None, :]
    c = torch.cos(angle).to(x.dtype)[:, None, :]
    s = torch.sin(angle).to(x.dtype)[:, None, :]
    a, b = x[..., :half_dim], x[..., half_dim:rotary_dim]
    return torch.cat([a * c - b * s, a * s + b * c, x[..., rotary_dim:]], dim=-1)


def q8_fake(w: torch.Tensor) -> torch.Tensor:
    """Q8_0 round trip of a weight (rows of 32-blocks), straight-through gradient.
    Quantization-aware training: the forward sees what export will write."""
    with torch.no_grad():
        wb = w.detach().float().unflatten(-1, (-1, 32))
        amax = wb.abs().amax(-1, keepdim=True)
        d = amax / 127.0
        inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
        q = torch.sign(wb * inv) * torch.floor((wb * inv).abs() + 0.5)
        deq = (q * d.to(torch.float16).float()).flatten(-2).to(w.dtype)
    return w + (deq - w).detach()


# ---------------------------------------------------------------------------
# Frozen target tensors (token_embd, output) and special tokens
# ---------------------------------------------------------------------------
class FrozenTarget:
    """token_embd / output from the trunk (Q8_0), special-token ids.

    Sources: the trunk GGUF directory (host) or a compact bundle written by
    `save_bundle` (cloud: avoids uploading the 111 GB trunk)."""

    def __init__(self, embd: R.Q8Tensor, head: R.Q8Tensor, specials: list[int], source: dict):
        self.embd_q8, self.head_q8 = embd, head
        self.specials = sorted(int(s) for s in specials)
        self.vocab, self.hidden = embd.shape
        self.source = source
        self.embd_dense: Optional[torch.Tensor] = None
        self.head_dense: Optional[torch.Tensor] = None
        self.head_xform = None   # optional draft-only head requantization (lean drafter what-if)

    def head_rows(self, index, device) -> torch.Tensor:
        """Head rows (slice or ids) as f32 with the head transform applied."""
        if self.head_dense is not None:
            return self.head_dense[index if isinstance(index, slice) else index.to(self.head_dense.device)]
        w = self.head_q8.dequant(index if isinstance(index, slice) else index.cpu(), dtype=torch.float32)
        return (self.head_xform(w) if self.head_xform else w).to(device)

    @classmethod
    def from_trunk(cls, trunk_dir: Path) -> "FrozenTarget":
        st = R.SharedTargetTensors(Path(trunk_dir), cache_head=False)
        te = st.embd_file.tensors["token_embd.weight"]
        embd = R.Q8Tensor(st.embd_file.raw("token_embd.weight"), te["shape"])
        head = st.head_q8()
        src = {"trunk_dir": str(trunk_dir), "embd_file": st.embd_file.path.name,
               "head_file": st.head_file.path.name}
        return cls(embd, head, st.special_tokens(), src)

    @classmethod
    def from_bundle(cls, path: Path) -> "FrozenTarget":
        with open(path, "rb") as f:
            magic = f.read(8)
        if magic == BUNDLE_MAGIC:
            return read_raw_bundle(Path(path))
        b = torch.load(path, map_location="cpu", weights_only=True)
        def q(prefix):
            t = R.Q8Tensor.__new__(R.Q8Tensor)
            t.qs, t.d, t.shape = b[prefix + "_qs"], b[prefix + "_d"].float(), list(b[prefix + "_qs"].shape)
            return t
        meta = json.loads(bytes(b["meta"].tolist()).decode())
        return cls(q("embd"), q("head"), b["specials"].tolist(), meta)

    def save_bundle(self, path: Path) -> dict:
        meta = dict(self.source, vocab=self.vocab, hidden=self.hidden, format="gufo-mtp-frozen-target-v1")
        torch.save({"embd_qs": self.embd_q8.qs, "embd_d": self.embd_q8.d.to(torch.float16),
                    "head_qs": self.head_q8.qs, "head_d": self.head_q8.d.to(torch.float16),
                    "specials": torch.tensor(self.specials, dtype=torch.int64),
                    "meta": torch.tensor(list(json.dumps(meta).encode()), dtype=torch.uint8)}, path)
        return meta

    def to_dense(self, device, dtype, embed: bool = True):
        """Dequantized embedding and head on `device` (bf16: 2 x 1.27 GB)."""
        def deq(q, xform=None):
            out = torch.empty(q.shape, dtype=dtype, device=device)
            step = 16384
            for r in range(0, q.shape[0], step):
                if isinstance(q, Q8File):
                    w = q.dequant(slice(r, r + step)).to(device)
                else:
                    qs = q.qs[r:r + step].to(device)
                    d = q.d[r:r + step].to(device)
                    w = (qs.float().unflatten(-1, (-1, 32)) * d.float().unsqueeze(-1)).flatten(-2)
                out[r:r + step] = (xform(w) if xform else w).to(dtype)
            return out
        self.embd_dense = deq(self.embd_q8) if embed else None
        self.head_dense = deq(self.head_q8, self.head_xform)

    def embed(self, ids: torch.Tensor, dtype) -> torch.Tensor:
        if self.embd_dense is not None:
            return self.embd_dense[ids.to(self.embd_dense.device)].to(dtype)
        cpu = ids.cpu()
        return self.embd_q8.dequant(cpu, dtype=torch.float32).to(dtype)

    def head_weight(self, dtype=torch.float32) -> torch.Tensor:
        """Full dequantized head (cached; f32 on CPU = 2.5 GB). Decode-fidelity
        evaluation needs f32: int8 x f16-scale products are not exact in bf16."""
        if self.head_dense is None or self.head_dense.dtype != dtype:
            self.head_dense = self.head_q8.dequant(dtype=dtype)
        return self.head_dense


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------
class Lora(nn.Module):
    """W' = W + (alpha/r) B A; for expert tensors A [E,r,in], B [E,out,r]."""

    def __init__(self, out_f: int, in_f: int, rank: int, alpha: float, experts: int = 0):
        super().__init__()
        lead = (experts,) if experts else ()
        self.A = nn.Parameter(torch.empty(*lead, rank, in_f))
        self.B = nn.Parameter(torch.zeros(*lead, out_f, rank))
        nn.init.normal_(self.A, std=1.0 / math.sqrt(in_f))
        self.scale = alpha / rank

    def delta_x(self, x, expert=None):
        A = self.A if expert is None else self.A[expert]
        B = self.B if expert is None else self.B[expert]
        return F.linear(F.linear(x, A.to(x.dtype)), B.to(x.dtype)) * self.scale

    def merged(self, expert=None) -> torch.Tensor:
        A = self.A if expert is None else self.A[expert]
        B = self.B if expert is None else self.B[expert]
        return (B.float() @ A.float()) * self.scale


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
@dataclass
class NetOptions:
    decode: bool = False            # gufo decode numerics (eval fidelity); False for training
    dense_mode: str = "full"        # full | lora | frozen
    expert_mode: str = "frozen"     # full | lora | frozen
    lora_rank: int = 16
    expert_lora_rank: int = 8
    lora_alpha: float = 16.0
    qat: bool = False               # Q8_0 fake-quant (STE) on Q8 weights in forward
    param_dtype: torch.dtype = torch.float32     # trainable dense master weights
    expert_dtype: torch.dtype = torch.bfloat16   # materialized experts ("full": float32 recommended)
    expert_storage: str = "dense"   # dense (materialized) | q8 (CPU eval: dequant per expert on use)
    attn_chunk: int = 512           # query rows per attention chunk
    moe_impl: str = "loop"          # [stage2] loop (stage 1) | grouped | bmm | auto - see moe_grouped.py
    dropout: float = 0.0            # [ablation] residual dropout on attention / MoE outputs (training only)
    depth_bias: bool = False        # [ablation C3] learned per-depth bias on the fused residual (zero-init)


class MtpNet(nn.Module):
    def __init__(self, ref: R.MtpDrafter, target: FrozenTarget, opt: NetOptions, device="cpu"):
        super().__init__()
        self.cfg = ref.cfg
        self.opt = opt
        self.target = target
        self.q8 = dict(ref.q8)
        self.device_ = torch.device(device)
        for _, attr in R.DENSE_NAMES:
            w = getattr(ref, attr).detach().to(self.device_, opt.param_dtype)
            trainable = opt.dense_mode == "full" or (opt.dense_mode == "lora" and w.dim() == 1)
            trainable = trainable and attr not in NO_TRAIN
            self.register_parameter(attr, nn.Parameter(w, requires_grad=trainable))
        self.lora = nn.ModuleDict()
        if opt.dense_mode == "lora":
            for _, attr in R.DENSE_NAMES:
                w = getattr(self, attr)
                if w.dim() == 2 and attr not in NO_TRAIN:
                    self.lora[attr] = Lora(w.shape[0], w.shape[1], opt.lora_rank, opt.lora_alpha).to(self.device_)
        if opt.depth_bias:                            # [ablation C3] not in the GGUF: eval via --checkpoint only
            self.register_parameter("depth_bias", nn.Parameter(
                torch.zeros(MAX_DRAFTS, ref.cfg.D, dtype=opt.param_dtype, device=self.device_)))
        self._experts_q8 = {}
        for attr in EXPERTS:
            if opt.expert_storage == "q8":
                assert opt.expert_mode == "frozen", "q8 expert storage is eval-only"
                q = ref._experts_q8.get(attr)
                assert q is not None, "reference experts already materialized"
                self._experts_q8[attr] = q
                continue
            if attr in ref._experts_q8:
                q = ref._experts_q8[attr]
                w = torch.empty(q.shape, dtype=opt.expert_dtype, device=self.device_)
                for e in range(q.shape[0]):
                    w[e] = q.dequant(e, dtype=torch.float32).to(self.device_, opt.expert_dtype)
            else:
                w = getattr(ref, attr).detach().to(self.device_, opt.expert_dtype)
            self.register_parameter(attr, nn.Parameter(w, requires_grad=opt.expert_mode == "full"))
            if opt.expert_mode == "lora":
                E, o, i = w.shape
                self.lora[attr] = Lora(o, i, opt.expert_lora_rank, opt.lora_alpha, experts=E).to(self.device_)

    # -- bookkeeping ------------------------------------------------------
    def trainable_named(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def trainable_state(self) -> dict:
        return {n: p.detach().cpu() for n, p in self.trainable_named()}

    def load_trainable(self, state: dict) -> None:
        own = dict(self.named_parameters())
        missing = [n for n, _ in self.trainable_named() if n not in state]
        if missing:
            raise KeyError(f"checkpoint lacks {missing[:5]}")
        with torch.no_grad():
            for n, t in state.items():
                own[n].copy_(t.to(own[n].device, own[n].dtype))

    # -- primitives -------------------------------------------------------
    def _w(self, attr, expert=None, dtype=None):
        if attr in self._experts_q8:
            w = self._experts_q8[attr].dequant(expert, dtype=torch.float32)
        else:
            w = getattr(self, attr)
            w = w if expert is None else w[expert]
        if self.opt.qat and self.q8.get(attr, False):
            w = q8_fake(w)
        return w if dtype is None else w.to(dtype)

    def lin(self, attr, x, expert=None):
        if self.opt.decode and self.q8[attr]:
            x = q8_act(x)
        y = F.linear(x, self._w(attr, expert, x.dtype))
        if attr in self.lora:
            y = y + self.lora[attr].delta_x(x, expert)
        return y

    def grouped_norm(self, res, w):
        c = self.cfg
        x = res.unflatten(-1, (c.hc, c.hidden))
        return (rms_norm(x, None, c.eps) * w.to(res.dtype).view(c.hc, c.hidden)).flatten(-2)

    def hc_mix(self, prefix, res, inject: bool):
        c = self.cfg
        xn = self.grouped_norm(res, getattr(self, prefix + "_norm"))
        lo = F.silu(self.lin(prefix + "_down", xn) / c.hc)
        gate = self.lin(prefix + "_up", lo)
        mixed = (xn.unflatten(-1, (c.hc, c.hidden)) *
                 torch.sigmoid(gate.unflatten(-1, (c.hc, c.hidden)))).sum(-2) / c.hc
        inj = self.lin(prefix + "_inject", xn) if inject else None
        return mixed, inj

    def hc_combine(self, res, block, inject):
        c = self.cfg
        w = 2.0 * torch.sigmoid(inject / c.hc)
        return (res.unflatten(-1, (c.hc, c.hidden)) + block[:, None, :] * w[:, :, None]).flatten(-2)

    def fuse(self, tokens: torch.Tensor, hidden: torch.Tensor, dtype):
        c = self.cfg
        emb = rms_norm(self.target.embed(tokens, dtype).to(hidden.device), self.enorm, c.eps)
        hn = rms_norm(hidden.to(dtype), self.hnorm, c.eps)
        if self.opt.decode and self.q8["eh_proj"]:
            emb, hn = q8_act(emb), q8_act(hn)
        W = self._w("eh_proj", dtype=dtype)
        w_e, w_h = W[:, :c.hidden], W[:, c.hidden:]
        ep = F.linear(emb, w_e)
        hp = F.linear(hn.unflatten(-1, (c.hc, c.hidden)), w_h)
        if "eh_proj" in self.lora:
            lo = self.lora["eh_proj"]
            A, B = lo.A.to(dtype), lo.B.to(dtype)
            ep = ep + F.linear(F.linear(emb, A[:, :c.hidden]), B) * lo.scale
            hp = hp + F.linear(F.linear(hn.unflatten(-1, (c.hc, c.hidden)), A[:, c.hidden:]), B) * lo.scale
        return (hp + ep[:, None, :]).flatten(-2)

    def qkv(self, x, pos):
        c = self.cfg
        n = x.shape[0]
        qg = self.lin("attn_q", x).view(n, c.heads, 2 * c.head_dim)
        q = rms_norm(qg[..., :c.head_dim], self.attn_q_norm, c.eps)
        gate = qg[..., c.head_dim:]
        k = rms_norm(self.lin("attn_k", x).view(n, c.kv_heads, c.head_dim), self.attn_k_norm, c.eps)
        v = self.lin("attn_v", x).view(n, c.kv_heads, c.head_dim)
        q = rope(q, pos, c.rotary_dim, c.rope_theta)
        k = rope(k, pos, c.rotary_dim, c.rope_theta)
        if self.opt.decode:
            k, v = half(k), half(v)
        return q, gate, k, v

    def attend(self, q, gate, K1, V1, chain):
        """q [R,H,d] (row p = chain p); K1/V1 [R1,kvh,d] depth-1 rows (causal: key s <= p);
        chain: list of (k, v) [>=R,kvh,d], row p visible only to chain p."""
        c = self.cfg
        R, G = q.shape[0], c.heads // c.kv_heads
        scale = 1.0 / math.sqrt(c.head_dim)
        qg = q.view(R, c.kv_heads, G, c.head_dim)
        out = []
        step = self.opt.attn_chunk if not torch.is_grad_enabled() else R
        keys = torch.arange(K1.shape[0], device=q.device)
        for r0 in range(0, R, step):
            r1 = min(R, r0 + step)
            qc = qg[r0:r1]
            s1 = torch.einsum("rkgd,skd->kgrs", qc, K1) * scale            # [kvh,G,rc,R1]
            causal = keys[None, :] <= torch.arange(r0, r1, device=q.device)[:, None]
            s1 = s1.masked_fill(~causal, float("-inf"))
            diag = [torch.einsum("rkgd,rkd->kgr", qc, ck[r0:r1]).unsqueeze(-1) * scale for ck, _ in chain]
            s = torch.cat([s1] + diag, -1) if diag else s1
            p = torch.softmax(s.float(), -1).to(q.dtype)
            ctx = torch.einsum("kgrs,skd->rkgd", p[..., :K1.shape[0]], V1)
            for i, (_, cv) in enumerate(chain):
                ctx = ctx + torch.einsum("kgr,rkd->rkgd", p[..., K1.shape[0] + i], cv[r0:r1])
            out.append(ctx.reshape(r1 - r0, c.heads, c.head_dim))
        ctx = torch.cat(out, 0) * torch.sigmoid(gate)
        return self.lin("attn_output", ctx.reshape(R, c.heads * c.head_dim))

    def moe(self, x):
        c = self.cfg
        router = self._w("router", dtype=x.dtype)
        if self.opt.decode:
            router = half(router)
        logits = F.linear(x, router)
        if "router" in self.lora:
            logits = logits + self.lora["router"].delta_x(x)
        probs = torch.softmax(logits.float(), -1)
        top_p, top_i = torch.topk(probs, c.used, dim=-1)
        top_w = (top_p / top_p.sum(-1, keepdim=True).clamp_min(6.103515625e-5)).to(x.dtype)
        flat_e = top_i.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        tok = torch.arange(x.shape[0], device=x.device).repeat_interleave(c.used)[order]
        wts = top_w.reshape(-1)[order]
        xs = x[tok]                                   # expert-major gather
        y_rows = None
        if self.opt.moe_impl != "loop" and not self._experts_q8:   # [stage2] batched experts (moe_grouped.py)
            import moe_grouped                        # None = unsupported here (e.g. f32 on CUDA) -> loop
            y_rows = moe_grouped.expert_rows(self, xs, flat_e[order], self.opt.moe_impl)
        if y_rows is None:                            # stage-1 per-expert loop (unchanged)
            counts = torch.bincount(flat_e, minlength=c.experts).tolist()
            ys = []
            a = 0
            for e, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                xe = xs[a:a + cnt]
                h = F.silu(self.lin("gate_exps", xe, e)) * self.lin("up_exps", xe, e)
                ys.append(self.lin("down_exps", h, e))
                a += cnt
            y_rows = torch.cat(ys, 0)
        # one scatter; per token, contributions are summed in ascending expert order (as mtp_ref)
        out = torch.zeros(x.shape[0], c.hidden, dtype=x.dtype, device=x.device).index_add(
            0, tok, wts[:, None] * y_rows)
        sh = F.silu(self.lin("shexp_gate", x)) * self.lin("shexp_up", x)
        sgw = self._w("shexp_gate_inp", dtype=x.dtype)
        if self.opt.decode:
            sgw = half(sgw)
        sg = torch.sigmoid(F.linear(x, sgw.view(1, -1)))
        return out + sg * self.lin("shexp_down", sh)

    def head_mixed(self, res):
        return self.hc_mix("hc_head", res, inject=False)[0]

    def head_input(self, mixed):
        return q8_act(mixed) if self.opt.decode else mixed

    # -- one depth of the unrolled chain -----------------------------------
    def depth_body(self, tokens, hidden, pos, K1, V1, *chain_flat):
        """Returns (res, k, v) for one depth; K1/V1 None at depth 1 (uses own k/v)."""
        dtype = hidden.dtype
        res = self.fuse(tokens, hidden, dtype)
        if self.opt.depth_bias:                       # [ablation C3] depth j = 1 (K1 None) or len(chain)/2 + 2
            j = 1 if K1 is None else len(chain_flat) // 2 + 2
            res = res + self.depth_bias[j - 1].to(dtype)
        mixed, inject = self.hc_mix("hc_attn", res, True)
        q, gate, k, v = self.qkv(mixed, pos)
        chain = list(zip(chain_flat[0::2], chain_flat[1::2]))
        if K1 is None:                      # depth 1: causal over the catch-up rows themselves
            Kd, Vd = k, v
        else:                               # depth >= 2: catch-up rows <= p + own chain rows incl. itself
            Kd, Vd = K1, V1
            chain = chain + [(k, v)]
        att = self.attend(q, gate, Kd, Vd, chain)
        if self.opt.dropout and self.training and torch.is_grad_enabled():        # [ablation] (checkpoint recompute replays the same RNG)
            att = F.dropout(att, self.opt.dropout)
        res = self.hc_combine(res, att, inject)
        mixed, inject = self.hc_mix("hc_ffn", res, True)
        ffn = self.moe(mixed)
        if self.opt.dropout and self.training and torch.is_grad_enabled():        # [ablation]
            ffn = F.dropout(ffn, self.opt.dropout)
        res = self.hc_combine(res, ffn, inject)
        return res, k, v

    def unrolled(self, hidden: torch.Tensor, tokens: torch.Tensor, depth: int, feed: str,
                 on_depth, checkpoint: bool = False, detach_chain: bool = False):
        """Runs depths 1..depth over a record.

        hidden [n, D] (trunk rows h_0..h_{n-1}); tokens [n] (t_0..t_{n-1}, long, device).
        Depth j computes rows p = 0..n-1-j. `on_depth(j, res_j)` returns the next
        depth's fed tokens for feed='greedy' (else ignored) and computes losses or
        metrics (it may run its own checkpointed head)."""
        n = hidden.shape[0]
        dev = hidden.device
        K1 = V1 = None
        chain_flat: list = []
        res_prev, fed = None, None
        for j in range(1, depth + 1):
            Rj = n - j
            if Rj <= 0:
                break
            pos = torch.arange(Rj, device=dev) + (j - 1)
            if j == 1:
                inp_h, inp_t = hidden[:Rj], tokens[1:1 + Rj]
            else:
                inp_h = res_prev[:Rj]
                if detach_chain:
                    inp_h = inp_h.detach()
                inp_t = tokens[j:j + Rj] if feed == "teacher" else fed[:Rj]
            args = (inp_t, inp_h, pos, K1, V1, *chain_flat)
            if checkpoint and torch.is_grad_enabled():
                res, k, v = torch.utils.checkpoint.checkpoint(self.depth_body, *args, use_reentrant=False)
            else:
                res, k, v = self.depth_body(*args)
            if j == 1:
                K1, V1 = k, v
            else:
                chain_flat += [k, v]
            fed = on_depth(j, res)
            res_prev = res


# ---------------------------------------------------------------------------
# Building the net from the GGUF files
# ---------------------------------------------------------------------------
def load_reference(mtp_gguf: Path, target: FrozenTarget, decode: bool) -> R.MtpDrafter:
    """mtp_ref.MtpDrafter without touching the trunk directory (shared=stub)."""
    class _Shared:  # only .vocab is read by the constructor
        vocab = target.vocab
    return R.MtpDrafter(Path(mtp_gguf), shared=_Shared(), decode=decode, dtype=torch.float32)


def build_net(mtp_gguf: Path, target: FrozenTarget, opt: NetOptions, device="cpu") -> MtpNet:
    ref = load_reference(mtp_gguf, target, opt.decode)
    return MtpNet(ref, target, opt, device)


# ---------------------------------------------------------------------------
# Data: GMTP records
# ---------------------------------------------------------------------------
@dataclass
class RecordData:
    rid: str
    tokens: np.ndarray       # int64 [n]
    hidden_bf16: np.ndarray  # uint16 [n, D] (or float32 for f32 shards)
    sup_ids: np.ndarray      # int64 [n, K]  (row q = position q; -1 where unsupervised)
    sup_lp: np.ndarray       # float32 [n, K]
    supervised: np.ndarray   # bool [n]
    kinds: np.ndarray        # int8 [n] turn-kind code per position (see KIND_CODES)
    meta: dict

    @property
    def n(self):
        return len(self.tokens)

    @property
    def greedy(self) -> np.ndarray:
        """g[q]: target argmax for token q+1 (-1 if unsupervised)."""
        return np.where(self.supervised, self.sup_ids[:, 0], -1)

    def hidden_tensor(self, device, dtype) -> torch.Tensor:
        h = self.hidden_bf16
        if h.dtype == np.uint16:
            t = torch.from_numpy(h.view(np.int16)).view(torch.bfloat16)
        else:
            t = torch.from_numpy(h)
        return t.to(device=device, dtype=dtype, non_blocking=True)


KIND_CODES = {"other": 0, "code": 1, "doc": 2, "diff": 3, "log": 4, "response": 5, "prompt": 6}


def _kind_map(meta: dict, n: int) -> np.ndarray:
    k = np.zeros(n, np.int8)
    md = meta.get("metadata") or {}
    if md.get("kind") == "target-responses":            # on-policy records: prompt vs target-generated response
        for seg in md.get("segments", []) or []:
            s0, r0, e0 = int(seg.get("start", 0)), int(seg.get("response_start", 0)), int(seg.get("end", n))
            k[max(0, s0):min(n, r0)] = KIND_CODES["prompt"]
            k[max(0, r0):min(n, e0)] = KIND_CODES["response"]
        return k
    for turn in (meta.get("metadata") or {}).get("turns", []) or []:
        code = KIND_CODES.get(str(turn.get("kind", "other")), 0)
        b0, b1 = turn.get("body_start"), turn.get("body_end")
        if isinstance(b0, int) and isinstance(b1, int):
            k[max(0, b0):min(n, b1)] = code
    return k


def read_record(shard: "read_gmtp.Shard", rec) -> RecordData:
    n = rec.n_tokens
    tokens = np.asarray(shard.tokens(rec), dtype=np.int64)
    raw = shard.hidden_raw(rec)
    hidden = np.ascontiguousarray(raw).reshape(n, -1)
    K = shard.header.topk
    ids = np.full((n, K), -1, np.int64)
    lp = np.zeros((n, K), np.float32)
    sup = shard.sup(rec)
    sf = rec.supervise_from
    ids[sf:] = sup["ids"].astype(np.int64)
    lp[sf:] = sup["logprob"].astype(np.float32)
    supervised = np.zeros(n, bool)
    supervised[sf:] = True
    return RecordData(rec.metadata.get("id", f"{shard.path.name}#{rec.ordinal}"), tokens, hidden, ids, lp,
                      supervised, _kind_map(rec.metadata, n), rec.metadata)


def resolve_shards(patterns: Sequence[str]) -> list[Path]:
    import glob
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p)) or ([p] if Path(p).exists() else [])
        out += [Path(h) for h in hits if h.endswith(".mtps")]
    seen, uniq = set(), []
    for p in out:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return uniq


# ---------------------------------------------------------------------------
# Small draft head (draft_vocab.hpp) per record
# ---------------------------------------------------------------------------
def small_head_plan(tokens: np.ndarray, specials: Sequence[int], prefix: int = DRAFT_VOCAB):
    """Per anchor p (anchor = t_{p+1}, committed = t_0..t_p):
    fits[p] -> production uses the small head; extra_ids [U] (specials then the
    record's rare tokens in first-use order); allowed[p, U] bool."""
    spec = sorted(int(s) for s in specials)
    special_set = set(spec)
    capacity = DRAFT_EXTRA_ROWS - len(spec)
    n = len(tokens)
    known: dict[int, int] = {}
    rare: list[int] = []
    first_use: list[int] = []
    last_miss = prior_miss = None
    fits = np.zeros(max(n - 1, 0), bool)
    known_count_at = np.zeros(max(n - 1, 0), np.int64)
    common = lambda t: 0 <= t < prefix or t in special_set  # noqa: E731
    for p in range(n - 1):
        t = int(tokens[p])                      # scan committed token p
        if not (common(t) or t in known):
            prior_miss, last_miss = last_miss, p
            if t >= 0 and len(rare) < capacity:
                known[t] = len(rare)
                rare.append(t)
                first_use.append(p)
        scanned = p + 1
        anchor = int(tokens[p + 1])
        quiet = prior_miss is None or prior_miss + DRAFT_WINDOW < scanned
        fits[p] = (common(anchor) or anchor in known) and quiet
        known_count_at[p] = len(rare)
    extra = np.array(spec + rare, np.int64)
    allowed = np.zeros((max(n - 1, 0), len(extra)), bool)
    allowed[:, :len(spec)] = True
    for u in range(len(rare)):
        allowed[first_use[u]:, len(spec) + u] = True
    return fits, extra, allowed


def context_head_plan(tokens: np.ndarray, specials: Sequence[int], prefix: int = 16384):
    """Lean-drafter A2 head: rows [0, prefix) + specials + every token of
    t_0..t_{p+1} (prompt + emitted + anchor); no capacity limit, no fallback."""
    spec = sorted(int(s) for s in specials)
    sset = set(spec)
    n = len(tokens)
    first: dict = {}
    for i, t in enumerate(tokens.tolist()):
        if t >= prefix and t not in sset and t not in first:
            first[t] = i
    rare = sorted(first, key=first.get)
    extra = np.array(spec + rare, np.int64)
    allowed = np.zeros((max(n - 1, 0), len(extra)), bool)
    allowed[:, :len(spec)] = True
    for u, t in enumerate(rare):
        allowed[max(first[t] - 1, 0):, len(spec) + u] = True     # anchor p sees t_0..t_{p+1}
    return np.ones(max(n - 1, 0), bool), extra, allowed


def context_subset_miss(tokens, greedy, supervised, specials, prefix: int = 16384):
    """A2 kill metric: depth-1 target greedy labels g[p+1] outside anchor p's
    context subset. Returns (misses, labels)."""
    sset = set(int(s) for s in specials)
    seen: set = set()
    miss = tot = 0
    n = len(tokens)
    for p in range(n - 1):
        seen.add(int(tokens[p]))
        seen.add(int(tokens[p + 1]))
        if supervised[p + 1] and greedy[p + 1] >= 0:
            y = int(greedy[p + 1])
            tot += 1
            miss += not (y < prefix or y in sset or y in seen)
    return miss, tot


# ---------------------------------------------------------------------------
# Draft heads (argmax + top1-top2 gap), memory-bounded
# ---------------------------------------------------------------------------
def head_chunks(target: "FrozenTarget", device, limit: Optional[int] = None, step: int = 16384):
    """Yields (v0, W[v0:v1]) over the shared head. Uses the device-resident dense
    head when present (GPU), else dequantizes Q8_0 chunks to f32 on the fly (host:
    no 2.5 GB cache; decode fidelity needs f32 anyway)."""
    V = limit or target.vocab
    for v0 in range(0, V, step):
        v1 = min(V, v0 + step)
        yield v0, target.head_rows(slice(v0, v1), device)


@torch.no_grad()
def full_logits(net: "MtpNet", mixed: torch.Tensor) -> torch.Tensor:
    x = net.head_input(mixed)
    return torch.cat([F.linear(x.to(W.dtype), W).float() for _, W in head_chunks(net.target, x.device)], -1)


def _merge_top2(bv, bi, z, offset):
    v, i = torch.topk(z, min(2, z.shape[-1]), -1)
    v = torch.cat([bv, v], -1)
    i = torch.cat([bi, i + offset], -1)
    o = torch.topk(v, 2, -1)
    return o.values, torch.gather(i, -1, o.indices)


@torch.no_grad()
def head_top2(net: MtpNet, mixed: torch.Tensor, fits: torch.Tensor, extra: torch.Tensor,
              allowed: torch.Tensor, mode: str, prefix: int = DRAFT_VOCAB):
    """mode 'small': rows with fits score output rows [0, 98304) + specials + the
    request's tabled rare tokens (production small draft head; empty/untabled
    slots are -inf), other rows the full head; 'full': full head everywhere.
    Returns token ids [R] and gaps [R] (top1 - top2 logit)."""
    x = net.head_input(mixed)
    R = x.shape[0]
    dev = x.device
    small = fits if mode == "small" else torch.zeros_like(fits)
    bv = torch.full((R, 2), float("-inf"), device=dev)
    bi = torch.zeros((R, 2), dtype=torch.long, device=dev)
    sel_s = small.nonzero().flatten()
    sel_f = (~small).nonzero().flatten()
    for v0, W in head_chunks(net.target, dev):
        xs = x.to(W.dtype)
        if sel_f.numel():
            bv[sel_f], bi[sel_f] = _merge_top2(bv[sel_f], bi[sel_f], F.linear(xs[sel_f], W).float(), v0)
        if sel_s.numel() and v0 < prefix:
            Wp = W[:prefix - v0]
            bv[sel_s], bi[sel_s] = _merge_top2(bv[sel_s], bi[sel_s], F.linear(xs[sel_s], Wp).float(), v0)
    if sel_s.numel() and extra.numel():
        We = net.target.head_rows(extra, dev)
        ze = F.linear(x[sel_s].to(We.dtype), We).float().masked_fill(~allowed[sel_s], float("-inf"))
        # extra rows are token ids; encode as vocab ids directly
        v, i = torch.topk(ze, min(2, ze.shape[-1]), -1)
        v = torch.cat([bv[sel_s], v], -1)
        i = torch.cat([bi[sel_s], extra[i]], -1)
        o = torch.topk(v, 2, -1)
        bv[sel_s], bi[sel_s] = o.values, torch.gather(i, -1, o.indices)
    return bi[:, 0], bv[:, 0] - bv[:, 1]


# ---------------------------------------------------------------------------
# Offline acceptance (gufo greedy verification) from captured target labels
# ---------------------------------------------------------------------------
@torch.no_grad()
def draft_record(net: MtpNet, rd: RecordData, depth: int, head: str, device, dtype,
                 max_tokens: Optional[int] = None, prefix: int = DRAFT_VOCAB, context_prefix: int = 16384):
    """gufo greedy chains for every anchor of a record. Returns numpy
    drafts [n-1, depth] (-1 = not computed), gaps [n-1, depth], fits [n-1]."""
    if max_tokens:
        rd = truncate(rd, max_tokens)
    n = rd.n
    hidden = rd.hidden_tensor(device, dtype)
    tokens = torch.from_numpy(rd.tokens).to(device)
    # head: small (production) | full | context (lean-drafter A2: top-K rows + specials + all seen tokens)
    if head == "context":
        fits_np, extra_np, allowed_np = context_head_plan(rd.tokens, net.target.specials, context_prefix)
        mode, pfx = "small", context_prefix
    else:
        fits_np, extra_np, allowed_np = small_head_plan(rd.tokens, net.target.specials, prefix)
        mode, pfx = head, prefix
    fits = torch.from_numpy(fits_np).to(device)
    extra = torch.from_numpy(extra_np).to(device)
    allowed = torch.from_numpy(allowed_np).to(device)
    drafts = np.full((n - 1, depth), -1, np.int64)
    gaps = np.full((n - 1, depth), np.nan, np.float32)

    def on_depth(j, res):
        Rj = res.shape[0]
        ids, g = head_top2(net, net.head_mixed(res), fits[:Rj], extra, allowed[:Rj], mode, pfx)
        drafts[:Rj, j - 1] = ids.cpu().numpy()
        gaps[:Rj, j - 1] = g.cpu().numpy()
        return ids

    net.unrolled(hidden, tokens, depth, "greedy", on_depth)
    return drafts, gaps, fits_np


def truncate(rd: RecordData, n: int) -> RecordData:
    if rd.n <= n:
        return rd
    return RecordData(rd.rid, rd.tokens[:n], rd.hidden_bf16[:n], rd.sup_ids[:n], rd.sup_lp[:n],
                      rd.supervised[:n], rd.kinds[:n], rd.meta)


class AcceptanceStats:
    """Accumulates chain outcomes; computes expected tokens per verify call.

    gufo greedy verification accepts draft j of chain p iff d_{p,1..j} all equal
    the target's greedy tokens. The capture is teacher-forced on corpus text, so
    the target label of depth j is only observable when the text itself followed
    the target's greedy path through position p+j-1 (t_{p+i+1} == g_{p+i},
    i < j): otherwise the captured distribution is conditioned on a different
    prefix than the accepted drafts. Estimators:
      km     product-limit (censoring-aware): r_j = P(accept j | accepted <j,
             observable at j); E = 1 + sum_j prod_{i<=j} r_i     [primary]
      lower  unobservable outcome counted as a rejection (lower bound)
      path   anchors whose text is target-greedy through the whole chain (exact,
             selected subset)
    `stop`: production confidence stop (a draft whose top1-top2 gap <= 0.5 ends
    the chain after itself)."""

    def __init__(self, depth: int):
        self.D = depth
        z = lambda: np.zeros(depth, np.int64)  # noqa: E731
        self.c = {s: {"coh": z(), "succ": z()} for s in ("stop", "nostop")}
        self.lower = {s: np.zeros(depth + 1, np.int64) for s in ("stop", "nostop")}  # accepted-count hist
        self.lower_n = 0
        self.path = {s: np.zeros(depth + 1, np.int64) for s in ("stop", "nostop")}
        self.path_n = 0
        self.anchors = 0
        self.fit_anchors = 0
        self.greedy_match = [0, 0]
        self.by_kind: dict[int, "AcceptanceStats"] = {}

    def add(self, rd: RecordData, drafts: np.ndarray, gaps: np.ndarray, fits: np.ndarray,
            kind_split: bool = True) -> None:
        D, n = self.D, rd.n
        t, g = rd.tokens, rd.greedy
        P = n - 1
        p = np.arange(P)
        j = np.arange(1, D + 1)
        lab = p[:, None] + j[None, :]                                 # label position p+j
        avail = (lab <= n - 1) & (drafts >= 0)
        labc = np.minimum(lab, n - 1)
        avail &= rd.supervised[labc] & rd.supervised[np.minimum(p + 1, n - 1)][:, None]
        acc = avail & (drafts == g[labc])
        # text greedy at position q: t[q+1] == g[q]
        tg = np.zeros(n, bool)
        tg[:-1] = (t[1:] == g[:-1]) & (g[:-1] >= 0)
        # obs[p, j] = all_{i=1..j-1} tg[p+i]
        obs = np.ones((P, D), bool)
        for jj in range(1, D):
            q = np.minimum(p + jj, n - 1)
            obs[:, jj] = obs[:, jj - 1] & tg[q] & (p + jj <= n - 1)
        stopped_after = np.full(P, D, np.int64)                      # chain length with stop
        low_gap = np.nan_to_num(gaps, nan=np.inf) <= CONF_FLOOR
        first = np.where(low_gap.any(1), low_gap.argmax(1) + 1, D)
        stopped_after = np.minimum(stopped_after, first)
        self.anchors += P
        self.fit_anchors += int(fits.sum())
        self.greedy_match[0] += int(tg[:-1][rd.supervised[:-1]].sum())
        self.greedy_match[1] += int(rd.supervised[:-1].sum())
        for s in ("stop", "nostop"):
            reach = j[None, :] <= (stopped_after[:, None] if s == "stop" else D)
            ok = acc & reach
            prev = np.ones(P, bool)
            coh, succ = self.c[s]["coh"], self.c[s]["succ"]
            for jj in range(D):
                cohort = prev & avail[:, jj] & obs[:, jj]
                coh[jj] += int(cohort.sum())
                succ[jj] += int((cohort & ok[:, jj]).sum())
                prev = prev & ok[:, jj]
            full = avail[:, D - 1]                                   # all D labels exist
            run = np.cumprod(ok & obs, 1).sum(1)
            self.lower[s] += np.bincount(run[full], minlength=D + 1)
            pathsel = full & obs[:, D - 1]
            self.path[s] += np.bincount(np.cumprod(ok, 1).sum(1)[pathsel], minlength=D + 1)
            if s == "stop":
                self.lower_n += int(full.sum())
                self.path_n += int(pathsel.sum())
        if kind_split:
            kinds = rd.kinds[np.minimum(p + 1, n - 1)]
            for kc in np.unique(kinds):
                sub = self.by_kind.setdefault(int(kc), AcceptanceStats(D))
                m = kinds == kc
                sub._add_masked(self, rd, drafts, gaps, fits, m)

    def _add_masked(self, parent, rd, drafts, gaps, fits, m):
        # re-run on a masked copy: mark unselected anchors as not computed
        d2 = drafts.copy()
        d2[~m] = -1
        self.add(rd, d2, gaps, fits & m, kind_split=False)
        self.anchors -= int((~m).sum())

    def expected(self, s: str, k: Optional[int] = None) -> dict:
        k = k or self.D
        coh, succ = self.c[s]["coh"][:k], self.c[s]["succ"][:k]
        r = np.where(coh > 0, succ / np.maximum(coh, 1), 0.0)
        surv = np.cumprod(r)
        lo = self.lower[s]
        lower = 1 + (np.arange(self.D + 1).clip(max=k) * lo).sum() / max(lo.sum(), 1)
        pa = self.path[s]
        path = 1 + (np.arange(self.D + 1).clip(max=k) * pa).sum() / max(pa.sum(), 1)
        return {"km": float(1 + surv.sum()), "lower": float(lower), "path": float(path),
                "r": [round(float(x), 4) for x in r], "cohort": coh.tolist()}

    def report(self) -> dict:
        out = {"anchors": self.anchors, "small_head_anchor_frac": self.fit_anchors / max(self.anchors, 1),
               "text_greedy_frac": self.greedy_match[0] / max(self.greedy_match[1], 1),
               "lower_anchors": self.lower_n, "path_anchors": self.path_n}
        for s in ("stop", "nostop"):
            out[s] = {f"k{k}": self.expected(s, k) for k in range(1, self.D + 1)}
            for k in range(1, self.D + 1):
                for key in ("r", "cohort"):
                    if k != self.D:
                        out[s][f"k{k}"].pop(key)
        inv = {v: k for k, v in KIND_CODES.items()}
        out["by_kind"] = {inv.get(k, str(k)): {"anchors": v.anchors,
                                               "stop_k7_km": v.expected("stop")["km"],
                                               "stop_k7_lower": v.expected("stop")["lower"]}
                          for k, v in sorted(self.by_kind.items())}
        return out


def headline(rep: dict, D: int) -> str:
    s = rep["stop"][f"k{D}"]
    ns = rep["nostop"][f"k{D}"]
    return (f"tokens/call k={D}+stop: km {s['km']:.3f} lower {s['lower']:.3f} path {s['path']:.3f} | "
            f"nostop km {ns['km']:.3f} | r(stop) {s['r']} | anchors {rep['anchors']} "
            f"text_greedy {rep['text_greedy_frac']:.3f}")


def file_sha256(path: Path, limit: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def lower_priority() -> str:
    """Below-normal CPU priority for host-side runs (the Halo GPU queue owns the box)."""
    try:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p; k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32); k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
            return "below-normal"
        import os
        os.nice(10)
        return "nice+10"
    except Exception as exc:  # pragma: no cover
        return f"unchanged ({exc})"


def commit_available_gb() -> float:
    """System commit headroom (Windows: ullAvailPageFile) or MemAvailable (Linux)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return min(m.ullAvailPageFile, m.ullAvailPhys) / 2**30
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return float("inf")


def start_memory_watchdog(min_avail_gb: float, poll_s: float = 0.25, on_trip=None):
    """Host safety: the Halo shares memory/commit with the GPU job queue. If system
    headroom drops below `min_avail_gb`, exit at once (code 3) so this process can
    never be the reason a GPU job fails to allocate. Progress is written
    incrementally by the callers, so a rerun resumes."""
    import os
    import threading

    def run():
        import time as _t
        while True:
            avail = commit_available_gb()
            if avail < min_avail_gb:
                print(f"MEMORY_GUARD: available {avail:.1f} GB < {min_avail_gb} GB; exiting", flush=True)
                if on_trip:
                    try:
                        on_trip()
                    except Exception:
                        pass
                os._exit(3)
            _t.sleep(poll_s)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# NetOptions (de)serialization for checkpoints
# ---------------------------------------------------------------------------
_DTYPES = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}


def opts_to_dict(o: NetOptions) -> dict:
    from dataclasses import asdict
    return {k: (str(v) if isinstance(v, torch.dtype) else v) for k, v in asdict(o).items()}


def opts_from_dict(d: dict) -> NetOptions:
    kw = {k: (_DTYPES.get(v, v) if isinstance(v, str) and v.startswith("torch.") else v)
          for k, v in d.items() if k in NetOptions.__dataclass_fields__}
    return NetOptions(**kw)


# ---------------------------------------------------------------------------
# Streamed (low-memory) weights for host runs on the Halo: <3 GB resident.
# Q8_0 tensors stay on disk; every use reads and dequantizes only the needed
# rows/experts with plain file reads (OS file cache, not process memory/commit).
# ---------------------------------------------------------------------------
class Q8File:
    """Drop-in for mtp_ref.Q8Tensor.dequant over a Q8_0 tensor in a GGUF file."""

    def __init__(self, path: Path, abs_offset: int, shape: Sequence[int]):
        import threading
        self.path, self.base, self.shape = Path(path), int(abs_offset), list(shape)
        self.cols = self.shape[-1]
        self.row_bytes = self.cols // 32 * 34
        self.rows_per_lead = int(np.prod(self.shape[1:-1])) if len(self.shape) > 2 else 1
        self._fh = open(self.path, "rb", buffering=0)
        self._lock = threading.Lock()

    def _read_rows(self, r0: int, count: int) -> np.ndarray:
        with self._lock:
            self._fh.seek(self.base + r0 * self.row_bytes)
            buf = self._fh.read(count * self.row_bytes)
        return np.frombuffer(buf, np.uint8).reshape(count, self.row_bytes)

    @staticmethod
    def _deq(raw: np.ndarray, cols: int, dtype) -> torch.Tensor:
        blocks = raw.reshape(raw.shape[0], cols // 32, 34)
        d = torch.from_numpy(blocks[..., :2].copy().view(np.float16).astype(np.float32))
        qs = torch.from_numpy(blocks[..., 2:].copy().view(np.int8))
        return (qs.to(torch.float32) * d).reshape(raw.shape[0], cols).to(dtype)

    def dequant(self, index=None, dtype=torch.float32) -> torch.Tensor:
        if index is None:
            raise MemoryError("Q8File: full dequant refused (streamed mode)")
        if isinstance(index, int) and len(self.shape) == 3:
            r0 = index * self.shape[1]
            return self._deq(self._read_rows(r0, self.shape[1]), self.cols, dtype)
        if isinstance(index, slice):
            r0, r1, _ = index.indices(self.shape[0])
            return self._deq(self._read_rows(r0, r1 - r0), self.cols, dtype)
        ids = np.asarray(index.cpu() if torch.is_tensor(index) else index).reshape(-1)
        rows = np.stack([self._read_rows(int(i), 1)[0] for i in ids]) if len(ids) else \
            np.zeros((0, self.row_bytes), np.uint8)
        return self._deq(rows, self.cols, dtype)


def _q8file(g: "R.GgufFile", name: str) -> Q8File:
    t = g.tensors[name]
    assert t["type"] == R.Q8_0, name
    return Q8File(g.path, g.info["data_offset"] + t["offset"], t["shape"])


class _StreamShared:
    """mtp_ref.SharedTargetTensors stand-in backed by Q8File (for mtp_ref methods)."""

    def __init__(self, target: "FrozenTarget"):
        self.t = target
        self.vocab, self.hidden = target.vocab, target.hidden
        self.cache_head = False

    def embed(self, ids, dtype=torch.float32):
        return self.t.embd_q8.dequant(torch.as_tensor(list(ids)), dtype=dtype)

    def head_q8(self):
        return self.t.head_q8


def load_streamed(mtp_gguf: Path, trunk_dir: Optional[Path], decode: bool, target: Optional["FrozenTarget"] = None):
    """(reference MtpDrafter, FrozenTarget) with experts / embedding / head left on
    disk. Resident: dense blk.48 tensors in f32 (~0.36 GB) + transients.
    Pass `target` (e.g. FrozenTarget.from_bundle) to skip the trunk directory."""
    if target is not None:
        return _streamed_ref(mtp_gguf, target, decode), target
    shards = sorted(Path(trunk_dir).glob("*.gguf"))
    embd = head = None
    specials = None
    for p in shards:
        g = R.GgufFile(p)
        if "token_embd.weight" in g.tensors:
            embd = _q8file(g, "token_embd.weight")
        if "output.weight" in g.tensors:
            head = _q8file(g, "output.weight")
        if specials is None and "-00001-of-" in p.name:
            st = R.SharedTargetTensors.__new__(R.SharedTargetTensors)
            st.vocab_file, st._specials = g, None
            specials = st.special_tokens()
        g._mm.close()
        g._fh.close()
    target = FrozenTarget(embd, head, specials, {"trunk_dir": str(trunk_dir), "streamed": True})
    return _streamed_ref(mtp_gguf, target, decode), target


def _streamed_ref(mtp_gguf: Path, target: "FrozenTarget", decode: bool):
    g = R.GgufFile(Path(mtp_gguf))
    ref = R.MtpDrafter.__new__(R.MtpDrafter)
    nn.Module.__init__(ref)
    ref.cfg = R.MtpConfig.from_gguf(g)
    ref.cfg.vocab = target.vocab
    ref.decode, ref.dtype = decode, torch.float32
    ref.shared = _StreamShared(target)
    p = f"blk.{ref.cfg.layer}."
    ref.q8 = {}
    for suffix, attr in R.DENSE_NAMES:
        ref.q8[attr] = g.tensors[p + suffix]["type"] == R.Q8_0
        ref.register_parameter(attr, nn.Parameter(R._dense(g, p + suffix, torch.float32), requires_grad=False))
    ref._experts_q8 = {}
    for suffix, attr in R.EXPERT_NAMES:
        ref.q8[attr] = True
        ref._experts_q8[attr] = _q8file(g, p + suffix)
    g._mm.close()
    g._fh.close()
    return ref


def process_private_gb() -> float:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return max(getattr(mi, "private", 0), mi.rss) / 2**30
    except Exception:
        return 0.0


def start_rss_watchdog(max_gb: float, poll_s: float = 0.2):
    """Hard cap for host runs (coordinator rule: < 3 GB resident). Exits (code 5)
    as soon as this process' private/resident memory exceeds `max_gb`."""
    import os
    import threading

    def run():
        import time as _t
        peak = 0.0
        while True:
            m = process_private_gb()
            peak = max(peak, m)
            if m > max_gb:
                print(f"RSS_GUARD: {m:.2f} GB > {max_gb} GB; exiting", flush=True)
                os._exit(5)
            _t.sleep(poll_s)

    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------------
# Lean drafter what-if (A3): draft-only requantized MTP layer and head
# ---------------------------------------------------------------------------
class AffineTensor:
    """Requantized expert tensor kept as block codes (int8 + per-32 scale/offset):
    dequant cost and memory like Q8_0 (2.5 GB for all experts)."""

    def __init__(self, src, kind: str, fallback: str):
        import quant_sim as Q
        self.shape = list(src.shape)
        E = self.shape[0]
        self.codes = torch.empty(self.shape, dtype=torch.int8)
        self.scale = torch.empty(*self.shape[:-1], self.shape[-1] // 32)
        self.offset = torch.empty_like(self.scale)
        for e in range(E):
            c, s_, o = Q.encode(src.dequant(e, dtype=torch.float32), kind, fallback)
            self.codes[e], self.scale[e], self.offset[e] = c, s_, o

    def dequant(self, index=None, dtype=torch.float32):
        import quant_sim as Q
        c = self.codes if index is None else self.codes[index]
        s_ = self.scale if index is None else self.scale[index]
        o = self.offset if index is None else self.offset[index]
        return Q.decode(c, s_, o).to(dtype)


@torch.no_grad()
def apply_lean_weights(net, layer: str = "q8_0", head: str = "q8_0", fallback: str = "q5_0") -> dict:
    """Simulate requantizing every Q8_0 tensor of the MTP layer to `layer` (q4_k;
    rows not a multiple of 256 -> `fallback`, as llama-quantize) and the draft-only
    head copy to `head` (q4_0). F32 tensors stay F32; token_embd untouched."""
    import quant_sim as Q
    done = {}
    if layer != "q8_0":
        def fx(w):
            return Q.fake(w.float(), layer, fallback)
        for _, attr in R.DENSE_NAMES:
            if net.q8.get(attr) and attr not in NO_TRAIN:
                w = getattr(net, attr)
                w.data.copy_(fx(w.data).to(w.dtype))
                done[attr] = layer if (layer != "q4_k" or w.shape[-1] % 256 == 0) else fallback
        for attr in EXPERTS:
            if attr in net._experts_q8:
                net._experts_q8[attr] = AffineTensor(net._experts_q8[attr], layer, fallback)
            else:
                w = getattr(net, attr)
                for e in range(w.shape[0]):
                    w.data[e] = fx(w.data[e]).to(w.dtype)
            cols = net.cfg.expert_ff if attr == "down_exps" else net.cfg.hidden
            done[attr] = layer if (layer != "q4_k" or cols % 256 == 0) else fallback
    if head != "q8_0":
        def hx(w):
            return Q.fake(w.float(), head, fallback)
        net.target.head_xform = hx
        if net.target.head_dense is not None:
            hd = net.target.head_dense
            for r in range(0, hd.shape[0], 16384):
                hd[r:r + 16384] = hx(hd[r:r + 16384]).to(hd.dtype)
        done["output(draft copy)"] = head
    return done


# ---------------------------------------------------------------------------
# Raw frozen-target bundle: token_embd + output Q8_0 bytes + specials, streamed
# (no tensor ever held in memory; loads as Q8File views; ~1.36 GB on disk)
# ---------------------------------------------------------------------------
BUNDLE_MAGIC = b"GMTPFRZ1"
BUNDLE_ALIGN = 4096


def write_raw_bundle(trunk_dir: Path, out: Path, chunk: int = 64 << 20) -> dict:
    trunk_dir, out = Path(trunk_dir), Path(out)
    src = {}
    specials = None
    for p in sorted(trunk_dir.glob("*.gguf")):
        g = R.GgufFile(p)
        for name in ("token_embd.weight", "output.weight"):
            if name in g.tensors:
                t = g.tensors[name]
                assert t["type"] == R.Q8_0
                rows, cols = t["shape"]
                src[name] = (p, g.info["data_offset"] + t["offset"], rows, cols)
        if specials is None and "-00001-of-" in p.name:
            st = R.SharedTargetTensors.__new__(R.SharedTargetTensors)
            st.vocab_file, st._specials = g, None
            specials = st.special_tokens()
        g._mm.close()
        g._fh.close()
    (pe, oe, vocab, hidden), (ph, oh, vh, hh) = src["token_embd.weight"], src["output.weight"]
    assert (vocab, hidden) == (vh, hh)
    nbytes = vocab * (hidden // 32 * 34)
    a = lambda x: (x + BUNDLE_ALIGN - 1) // BUNDLE_ALIGN * BUNDLE_ALIGN  # noqa: E731
    embd_off = BUNDLE_ALIGN * 4
    head_off = a(embd_off + nbytes)
    shas = {}
    tmp = out.with_suffix(".partial")
    with open(tmp, "wb") as fo:
        fo.write(bytes(embd_off))
        for key, (path, off), dest in (("token_embd", (pe, oe), embd_off), ("output", (ph, oh), head_off)):
            h = hashlib.sha256()
            fo.seek(dest)
            with open(path, "rb", buffering=0) as fi:
                fi.seek(off)
                left = nbytes
                while left:
                    b = fi.read(min(chunk, left))
                    h.update(b)
                    fo.write(b)
                    left -= len(b)
            shas[key] = h.hexdigest()
        meta = {"format": "gufo-mtp-frozen-target-raw-v1", "vocab": vocab, "hidden": hidden, "type": "Q8_0",
                "embd_offset": embd_off, "head_offset": head_off, "tensor_bytes": nbytes,
                "sha256": shas, "specials": specials,
                "sources": {"token_embd": pe.name, "output": ph.name}}
        mb = json.dumps(meta).encode()
        assert 16 + len(mb) <= embd_off
        fo.seek(0)
        fo.write(BUNDLE_MAGIC + len(mb).to_bytes(8, "little") + mb)
    tmp.replace(out)
    return meta


def read_raw_bundle(path: Path) -> "FrozenTarget":
    with open(path, "rb") as f:
        f.read(8)
        n = int.from_bytes(f.read(8), "little")
        meta = json.loads(f.read(n))
    shape = [meta["vocab"], meta["hidden"]]
    embd = Q8File(path, meta["embd_offset"], shape)
    head = Q8File(path, meta["head_offset"], shape)
    return FrozenTarget(embd, head, meta["specials"], dict(meta, bundle=str(path)))
