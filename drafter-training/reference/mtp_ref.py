"""PyTorch reference of gufo's shared Q8_0 MTP drafter (Qwen3.8-Flash-Next).

Ports, operator for operator, the scalar oracle `ReferenceModel::MtpStep`
(src/models/qwen38_flash_next/reference.cpp) that `qwen38_flash_next_gpu_probe
--mtp-audit` (tests/models/qwen38_flash_next/mtp_audit.cpp) checks the HIP
predictor against, plus the runtime state semantics of
`Executor::MtpForward/MtpBody/MtpHead` (kernels/rocm/executor.cpp) and the
engine's draft cycle (`Session::DraftReplay/DraftCatchUp/PrepareDecode/
FinishDecode`, engine.cpp).

Numerics
  * `decode=True` emulates gufo's decode storage exactly as the oracle's
    `Storage::kDecode`: Q8 (32-block, F16 scale) activation quantization in
    front of every Q8_0 weight, F16 K/V and pooled indexer keys, F16-rounded
    router/shared-gate weights. Use it to compare against gufo.
  * `decode=False` is the plain float formula (`Storage::kFloat32`); use it
    for training.
Weights are always the exact dequantized GGUF values (Q8_0 -> f32).

Frozen vs trainable
  frozen   : token_embd.weight, output.weight (borrowed from the target
             trunk, `qwen4exp.nextn_shared_target_tensors`), never Parameters.
  trainable: every `blk.48.*` tensor of the MTP sidecar (see
             `MtpDrafter.param_report()`); indexer q/k projections only steer
             top-k block selection (>2048 positions) and get no gradient.

Nothing here touches a GPU; tensors live wherever the caller moves them.
"""
from __future__ import annotations

import math
import mmap
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.gufo.gguf import GGML_TYPE_BLOCK, Reader, parse_gguf  # noqa: E402

DEFAULT_MTP = Path(r"models/qwen3.8-flash-next/MTP/"
                   r"mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
DEFAULT_TRUNK_DIR = Path(r"models/qwen3.8-flash-next/UD-Q4_K_XL")

# draft_vocab.hpp / mtp_sampling.hpp
MTP_DRAFT_VOCAB = 98304
MTP_DRAFT_EXTRA_ROWS = 1024
MTP_DRAFT_VOCAB_WINDOW = 32
MTP_CANDIDATES = 64
GREEDY_DRAFT_CONFIDENCE_FLOOR = 0.5

Q8_0, F32, F16, BF16 = 8, 0, 1, 30


# ---------------------------------------------------------------------------
# GGUF access (read-only mmap; only the named tensors are ever touched)
# ---------------------------------------------------------------------------
class GgufFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.info = parse_gguf(self.path)
        self.tensors = {t["name"]: t for t in self.info["tensors"]}
        self._fh = self.path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def kv(self, key, default=None):
        return self.info["user_kv"].get(key, self.info["metadata"].get(key, default))

    def raw(self, name: str, row_start: int = 0, row_count: Optional[int] = None) -> np.ndarray:
        """Bytes of `name`, optionally only GGUF rows [row_start, +row_count)."""
        t = self.tensors[name]
        block, size = GGML_TYPE_BLOCK[t["type"]]
        cols = t["shape"][-1]
        row_bytes = cols // block * size
        rows_total = int(np.prod(t["shape"][:-1])) if len(t["shape"]) > 1 else 1
        if row_count is None:
            row_count = rows_total - row_start
        base = self.info["data_offset"] + t["offset"] + row_start * row_bytes
        return np.frombuffer(self._mm, dtype=np.uint8, count=row_count * row_bytes, offset=base)

    def string_array(self, wanted: str) -> list[str]:
        """Reads one string-array metadata key (parse_gguf skips tokenizer.*)."""
        r = Reader(self._mm)
        r.pos = 4
        r.u32()
        r.u64()
        n_kv = r.u64()
        for _ in range(n_kv):
            key = r.string()
            tag = r.u32()
            if key == wanted:
                assert tag == 9
                atype = r.u32()
                n = r.u64()
                assert atype == 8
                return [r.string() for _ in range(n)]
            r.skip_value(tag)
        raise KeyError(wanted)


class Q8Tensor:
    """Q8_0 storage kept as int8 codes + f32 block scales: [..., rows, cols]."""

    def __init__(self, raw: np.ndarray, shape: Sequence[int]):
        shape = list(shape)
        cols = shape[-1]
        nb = cols // 32
        blocks = raw.reshape(-1, 34)
        d = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(*shape[:-1], nb)
        qs = blocks[:, 2:].copy().view(np.int8).reshape(*shape)
        self.shape = shape
        self.qs = torch.from_numpy(qs)
        self.d = torch.from_numpy(d)

    def dequant(self, index=None, dtype=torch.float32) -> torch.Tensor:
        qs = self.qs if index is None else self.qs[index]
        d = self.d if index is None else self.d[index]
        x = qs.to(dtype).unflatten(-1, (-1, 32)) * d.to(dtype).unsqueeze(-1)
        return x.flatten(-2)


def _dense(g: GgufFile, name: str, dtype=torch.float32) -> torch.Tensor:
    t = g.tensors[name]
    raw = g.raw(name)
    shape = t["shape"]
    if t["type"] == F32:
        a = raw.view(np.float32).reshape(shape)
        return torch.from_numpy(a.copy()).to(dtype)
    if t["type"] == BF16:
        a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return torch.from_numpy(a.copy()).to(dtype)
    if t["type"] == F16:
        return torch.from_numpy(raw.view(np.float16).astype(np.float32).reshape(shape)).to(dtype)
    if t["type"] == Q8_0:
        return Q8Tensor(raw, shape).dequant(dtype=dtype)
    raise ValueError(f"{name}: unsupported type {t['type_name']}")


# ---------------------------------------------------------------------------
# Configuration (config.hpp/config.cpp; values read from the sidecar)
# ---------------------------------------------------------------------------
@dataclass
class MtpConfig:
    hidden: int = 2560
    hc: int = 4
    low_rank: int = 320
    heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 1.0e7
    idx_heads: int = 4
    idx_dim: int = 128
    idx_top_k: int = 2048
    ratio: int = 4
    experts: int = 512
    used: int = 10
    expert_ff: int = 640
    shared_ff: int = 640
    eps: float = 1e-6
    vocab: int = 248320
    layer: int = 48

    @property
    def D(self) -> int:  # HcDim
        return self.hc * self.hidden

    @classmethod
    def from_gguf(cls, g: GgufFile) -> "MtpConfig":
        k = lambda s: g.kv("qwen4exp." + s)  # noqa: E731
        f32 = lambda v: float(np.float32(v))  # noqa: E731
        c = cls(hidden=k("embedding_length"), hc=k("hyper_connection.count"),
                low_rank=k("hyper_connection.low_rank"), heads=k("attention.head_count"),
                kv_heads=k("attention.head_count_kv"), head_dim=k("attention.key_length"),
                rotary_dim=k("rope.dimension_count"), rope_theta=f32(k("rope.freq_base")),
                idx_heads=k("attention.indexer.head_count"),
                idx_dim=k("attention.indexer.key_length"),
                idx_top_k=k("attention.indexer.top_k"), experts=k("expert_count"),
                used=k("expert_used_count"), expert_ff=k("expert_feed_forward_length"),
                shared_ff=k("expert_shared_feed_forward_length"),
                eps=f32(k("attention.layer_norm_rms_epsilon")),
                layer=k("block_count") - 1)
        ratios = [r for r in k("attention.compress_ratios") if r]
        # Config::FromGguf: the MTP uses the trunk's ratio-four indexer.
        c.ratio = ratios[0] if ratios else 4
        assert k("nextn_predict_layers") == 1
        return c


# ---------------------------------------------------------------------------
# Numerics helpers (cpu_ops.cpp, reference.cpp)
# ---------------------------------------------------------------------------
def half(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(x.dtype)


def round_away(x: torch.Tensor) -> torch.Tensor:  # std::round
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def q8_act(x: torch.Tensor) -> torch.Tensor:
    """reference.cpp QuantizedInput: 32-blocks, scale=peak/127 stored as F16."""
    xb = x.unflatten(-1, (-1, 32))
    peak = xb.abs().amax(-1, keepdim=True)
    scale = peak / 127.0
    safe = torch.where(peak == 0, torch.ones_like(scale), scale)
    q = round_away(xb / safe)
    out = torch.where(peak == 0, torch.zeros_like(xb), half(scale) * q)
    return out.flatten(-2)


def rms_norm(x: torch.Tensor, w: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def rope(x: torch.Tensor, pos: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    """cpu::Rope (NeoX pairs i, i+rotary/2 inside each head). x: [n, heads, dim]."""
    half_dim = rotary_dim // 2
    i = torch.arange(half_dim, dtype=torch.float32)
    freq = torch.pow(torch.tensor(theta, dtype=torch.float32), -2.0 * i / rotary_dim)
    angle = pos.to(torch.float32)[:, None] * freq[None, :]
    c = torch.cos(angle).to(x.dtype)[:, None, :]
    s = torch.sin(angle).to(x.dtype)[:, None, :]
    a, b = x[..., :half_dim], x[..., half_dim:rotary_dim]
    return torch.cat([a * c - b * s, a * s + b * c, x[..., rotary_dim:]], dim=-1)


# ---------------------------------------------------------------------------
# Frozen target tensors: token embedding rows and the LM head
# ---------------------------------------------------------------------------
class SharedTargetTensors:
    """token_embd/output borrowed from the trunk shards (read-only, row access)."""

    def __init__(self, trunk_dir: Path = DEFAULT_TRUNK_DIR, cache_head: bool = True):
        shards = sorted(Path(trunk_dir).glob("*.gguf"))
        self.embd_file = self.head_file = self.vocab_file = None
        for p in shards:
            g = GgufFile(p)
            if "token_embd.weight" in g.tensors:
                self.embd_file = g
            if "output.weight" in g.tensors:
                self.head_file = g
            if self.vocab_file is None and "-00001-of-" in p.name:
                self.vocab_file = g
        assert self.embd_file is not None and self.head_file is not None
        te = self.embd_file.tensors["token_embd.weight"]
        self.vocab, self.hidden = te["shape"]
        assert te["type"] == Q8_0 and self.head_file.tensors["output.weight"]["type"] == Q8_0
        self._head_q8: Optional[Q8Tensor] = None
        self._head_f32: Optional[torch.Tensor] = None
        self.cache_head = cache_head
        self._specials: Optional[list[int]] = None

    def embed(self, ids: Sequence[int], dtype=torch.float32) -> torch.Tensor:
        rows = [Q8Tensor(self.embd_file.raw("token_embd.weight", int(t), 1), (1, self.hidden))
                .dequant(dtype=dtype) for t in ids]
        return torch.cat(rows, 0)

    def head_q8(self) -> Q8Tensor:
        if self._head_q8 is None:
            raw = self.head_file.raw("output.weight")
            self._head_q8 = Q8Tensor(raw, (self.vocab, self.hidden))
        return self._head_q8

    def head_weight(self, dtype=torch.float32) -> torch.Tensor:
        """Full dequantized [vocab, hidden] head (2.5 GB in f32)."""
        if self._head_f32 is not None and self._head_f32.dtype == dtype:
            return self._head_f32
        w = self.head_q8().dequant(dtype=dtype)
        if self.cache_head:
            self._head_f32 = w
        return w

    def special_tokens(self) -> list[int]:
        """QwenTokenizer special IDs (tokenizer.cpp 356-370)."""
        if self._specials is None:
            names = {"<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
                     "<think>", "</think>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
                     "<tts_text_bos_single>"}
            toks = self.vocab_file.string_array("tokenizer.ggml.tokens")
            self._specials = [i for i, t in enumerate(toks)
                              if t in names or (len(t) >= 4 and t.startswith("<|") and t.endswith("|>"))]
        return self._specials


# ---------------------------------------------------------------------------
# MTP block
# ---------------------------------------------------------------------------
DENSE_NAMES = [  # (gguf suffix, module attribute)
    ("nextn.enorm.weight", "enorm"), ("nextn.hnorm.weight", "hnorm"),
    ("nextn.eh_proj.weight", "eh_proj"),
    ("hc_attn_norm.weight", "hc_attn_norm"), ("hc_attn_down.weight", "hc_attn_down"),
    ("hc_attn_up.weight", "hc_attn_up"), ("hc_attn_inject.weight", "hc_attn_inject"),
    ("attn_q.weight", "attn_q"), ("attn_k.weight", "attn_k"), ("attn_v.weight", "attn_v"),
    ("attn_output.weight", "attn_output"), ("attn_q_norm.weight", "attn_q_norm"),
    ("attn_k_norm.weight", "attn_k_norm"),
    ("indexer.q_proj.weight", "indexer_q"), ("indexer.k_proj.weight", "indexer_k"),
    ("indexer.q_norm.weight", "indexer_q_norm"), ("indexer.k_norm.weight", "indexer_k_norm"),
    ("hc_ffn_norm.weight", "hc_ffn_norm"), ("hc_ffn_down.weight", "hc_ffn_down"),
    ("hc_ffn_up.weight", "hc_ffn_up"), ("hc_ffn_inject.weight", "hc_ffn_inject"),
    ("ffn_gate_inp.weight", "router"), ("ffn_gate_inp_shexp.weight", "shexp_gate_inp"),
    ("ffn_gate_shexp.weight", "shexp_gate"), ("ffn_up_shexp.weight", "shexp_up"),
    ("ffn_down_shexp.weight", "shexp_down"),
    ("nextn.hc_head_norm.weight", "hc_head_norm"), ("nextn.hc_head_down.weight", "hc_head_down"),
    ("nextn.hc_head_up.weight", "hc_head_up"),
]
EXPERT_NAMES = [("ffn_gate_exps.weight", "gate_exps"), ("ffn_up_exps.weight", "up_exps"),
                ("ffn_down_exps.weight", "down_exps")]


@dataclass
class MtpState:
    """Per-session predictor state (executor.hpp Session::MtpState)."""
    position: int = 0
    k: list = field(default_factory=list)        # [pos] -> [kv_heads, head_dim]
    v: list = field(default_factory=list)
    index_k: list = field(default_factory=list)  # raw indexer keys [idx_dim]
    block_k: list = field(default_factory=list)  # pooled+norm+rope keys
    h: Optional[torch.Tensor] = None             # carried wide residual [D]
    residual_valid: bool = False

    def rewind(self, position: int, ratio: int = 4) -> None:
        """Executor::MtpRewind."""
        if position != self.position:
            self.residual_valid = False
        self.position = position
        del self.k[position:], self.v[position:], self.index_k[position:]
        del self.block_k[position // ratio:]


@dataclass
class MtpOutput:
    logits: Optional[torch.Tensor] = None   # [vocab] or [subset rows]
    token: Optional[int] = None
    candidates: Optional[tuple] = None      # (ids, logits) top-64, descending
    trace: Optional[dict] = None


class MtpDrafter(nn.Module):
    def __init__(self, mtp_path: Path = DEFAULT_MTP, shared: Optional[SharedTargetTensors] = None,
                 decode: bool = True, dtype=torch.float32, trunk_dir: Path = DEFAULT_TRUNK_DIR,
                 materialize_experts: bool = False):
        super().__init__()
        g = GgufFile(mtp_path)
        self.cfg = MtpConfig.from_gguf(g)
        self.decode = decode
        self.dtype = dtype
        self.shared = shared if shared is not None else SharedTargetTensors(trunk_dir)
        self.cfg.vocab = self.shared.vocab
        p = f"blk.{self.cfg.layer}."
        self.q8 = {}  # attribute -> weight is Q8_0 (activation quantized in decode mode)
        for suffix, attr in DENSE_NAMES:
            t = g.tensors[p + suffix]
            self.q8[attr] = t["type"] == Q8_0
            self.register_parameter(attr, nn.Parameter(_dense(g, p + suffix, dtype)))
        self._experts_q8 = {}
        for suffix, attr in EXPERT_NAMES:
            t = g.tensors[p + suffix]
            self.q8[attr] = t["type"] == Q8_0
            q = Q8Tensor(g.raw(p + suffix), t["shape"])
            if materialize_experts:
                self.register_parameter(attr, nn.Parameter(q.dequant(dtype=dtype)))
            else:
                self._experts_q8[attr] = q
        # Frozen target tensors are never registered as parameters.
        c = self.cfg
        assert self.eh_proj.shape == (c.hidden, 2 * c.hidden)

    # -- bookkeeping --------------------------------------------------------
    def materialize_experts(self, dtype=None) -> None:
        dtype = dtype or self.dtype
        for attr, q in list(self._experts_q8.items()):
            self.register_parameter(attr, nn.Parameter(q.dequant(dtype=dtype)))
            del self._experts_q8[attr]

    def param_report(self) -> dict:
        c = self.cfg
        trainable = {attr: int(np.prod(getattr(self, attr).shape)) for _, attr in DENSE_NAMES}
        for _, attr in EXPERT_NAMES:
            trainable[attr] = c.experts * (c.expert_ff * c.hidden)
        frozen = {"token_embd": c.vocab * c.hidden, "output": c.vocab * c.hidden}
        total = sum(trainable.values())
        experts = sum(trainable[a] for _, a in EXPERT_NAMES)
        no_grad = trainable["indexer_q"] + trainable["indexer_k"]
        active = total - experts + c.used * 3 * c.expert_ff * c.hidden
        return {"trainable": trainable, "trainable_total": total, "routed_expert_params": experts,
                "dense_params": total - experts, "selection_only_params(no grad)": no_grad,
                "active_params_per_token": active, "frozen": frozen,
                "frozen_total": sum(frozen.values())}

    # -- primitive ops ------------------------------------------------------
    def _w(self, attr: str, expert: Optional[int] = None) -> torch.Tensor:
        if attr in self._experts_q8:
            return self._experts_q8[attr].dequant(expert, dtype=self.dtype)
        w = getattr(self, attr)
        return w if expert is None else w[expert]

    def lin(self, attr: str, x: torch.Tensor, expert: Optional[int] = None) -> torch.Tensor:
        if self.decode and self.q8[attr]:
            x = q8_act(x)
        return F.linear(x, self._w(attr, expert))

    def grouped_norm(self, res: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        x = res.unflatten(-1, (c.hc, c.hidden))
        return (rms_norm(x, None, c.eps) * w.view(c.hc, c.hidden)).flatten(-2)

    def hc_mix(self, prefix: str, res: torch.Tensor, inject: bool):
        """ReferenceModel::HcMix: res [n, D] -> mixed [n, H], inject [n, hc]."""
        c = self.cfg
        xn = self.grouped_norm(res, getattr(self, prefix + "_norm"))
        lo = F.silu(self.lin(prefix + "_down", xn) / c.hc)
        gate = self.lin(prefix + "_up", lo)
        mixed = (xn.unflatten(-1, (c.hc, c.hidden)) *
                 torch.sigmoid(gate.unflatten(-1, (c.hc, c.hidden)))).sum(-2) / c.hc
        inj = self.lin(prefix + "_inject", xn) if inject else None
        return mixed, inj

    def hc_combine(self, res: torch.Tensor, block: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        w = 2.0 * torch.sigmoid(inject / c.hc)  # [n, hc]
        return (res.unflatten(-1, (c.hc, c.hidden)) + block[:, None, :] * w[:, :, None]).flatten(-2)

    # -- embedding / fusion ---------------------------------------------------
    def fuse(self, tokens: Sequence[int], hidden: torch.Tensor, trace: Optional[dict] = None):
        """enorm(embed(t)) and hnorm(h) through eh_proj: [n, D] fused residual."""
        c = self.cfg
        emb = self.shared.embed(tokens, self.dtype)
        emb = rms_norm(emb, self.enorm, c.eps)
        hn = rms_norm(hidden.to(self.dtype), self.hnorm, c.eps)  # one denominator over D
        if trace is not None:
            trace["norm"] = hn
        if self.decode and self.q8["eh_proj"]:
            emb, hn = q8_act(emb), q8_act(hn)
        w_e, w_h = self.eh_proj[:, :c.hidden], self.eh_proj[:, c.hidden:]
        ep = F.linear(emb, w_e)                                       # [n, H]
        hp = F.linear(hn.unflatten(-1, (c.hc, c.hidden)), w_h)        # [n, hc, H]
        fused = (hp + ep[:, None, :]).flatten(-2)
        if trace is not None:
            trace["fused"] = fused
        return fused

    # -- attention --------------------------------------------------------------
    def qkv(self, x: torch.Tensor, pos: torch.Tensor):
        c = self.cfg
        n = x.shape[0]
        qg = self.lin("attn_q", x).view(n, c.heads, 2 * c.head_dim)
        q = rms_norm(qg[..., :c.head_dim], self.attn_q_norm, c.eps)
        gate = qg[..., c.head_dim:]
        k = rms_norm(self.lin("attn_k", x).view(n, c.kv_heads, c.head_dim), self.attn_k_norm, c.eps)
        v = self.lin("attn_v", x).view(n, c.kv_heads, c.head_dim)
        q = rope(q, pos, c.rotary_dim, c.rope_theta)
        k = rope(k, pos, c.rotary_dim, c.rope_theta)
        if self.decode:
            k, v = half(k), half(v)
        iq = rms_norm(self.lin("indexer_q", x).view(n, c.idx_heads, c.idx_dim), self.indexer_q_norm, c.eps)
        iq = rope(iq, pos, c.rotary_dim, c.rope_theta)
        ik = self.lin("indexer_k", x)  # raw, pooled later
        return q, gate, k, v, iq, ik

    def pool_block(self, raw: torch.Tensor, block: int) -> torch.Tensor:
        """Pooled+normed+roped indexer key of a complete block (reference.cpp 399-417)."""
        c = self.cfg
        pooled = (raw / c.ratio).sum(0)
        pooled = rms_norm(pooled, self.indexer_k_norm, c.eps)
        pooled = rope(pooled.view(1, 1, -1), torch.tensor([block * c.ratio]), c.rotary_dim, c.rope_theta).view(-1)
        return half(pooled) if self.decode else pooled

    def visible_mask(self, iq: torch.Tensor, pos: torch.Tensor, block_k: torch.Tensor, n_kv: int) -> torch.Tensor:
        """[n, n_kv] bool. Dense causal below the QSA budget, else top blocks + tail."""
        c = self.cfg
        budget = c.idx_top_k // c.ratio
        mask = torch.zeros(len(pos), n_kv, dtype=torch.bool)
        for i, p in enumerate(pos.tolist()):
            n = p + 1
            complete = n // c.ratio
            if complete <= budget:
                mask[i, :n] = True
                continue
            score = torch.relu(torch.einsum("hd,bd->hb", iq[i], block_k[:complete])).sum(0)
            order = torch.argsort(score, descending=True, stable=True)[:budget]
            sel = torch.zeros(complete, dtype=torch.bool)
            sel[order] = True
            mask[i, :complete * c.ratio] = sel.repeat_interleave(c.ratio)
            mask[i, complete * c.ratio:n] = True
        return mask

    def attention(self, state: MtpState, x: torch.Tensor, pos0: int) -> torch.Tensor:
        """Gated GQA + QSA over the predictor's own cache; appends this batch's rows."""
        c = self.cfg
        n = x.shape[0]
        pos = torch.arange(pos0, pos0 + n)
        q, gate, k, v, iq, ik = self.qkv(x, pos)
        assert len(state.k) == pos0
        state.k.extend(k.unbind(0))
        state.v.extend(v.unbind(0))
        state.index_k.extend(ik.unbind(0))
        n_kv = pos0 + n
        while len(state.block_k) < n_kv // c.ratio:
            b = len(state.block_k)
            state.block_k.append(self.pool_block(torch.stack(state.index_k[b * c.ratio:(b + 1) * c.ratio]), b))
        K, Vv = torch.stack(state.k), torch.stack(state.v)          # [n_kv, kvh, d]
        bk = torch.stack(state.block_k) if state.block_k else torch.zeros(0, c.idx_dim, dtype=x.dtype)
        mask = self.visible_mask(iq, pos, bk, n_kv)
        group = c.heads // c.kv_heads
        Kh = K.repeat_interleave(group, dim=1).transpose(0, 1)       # [heads, n_kv, d]
        Vh = Vv.repeat_interleave(group, dim=1).transpose(0, 1)
        s = torch.einsum("nhd,hkd->hnk", q, Kh) / math.sqrt(c.head_dim)
        s = s.masked_fill(~mask[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        ctx = torch.einsum("hnk,hkd->nhd", p, Vh) * torch.sigmoid(gate)
        return self.lin("attn_output", ctx.reshape(n, c.heads * c.head_dim))

    # -- MoE ------------------------------------------------------------------
    def moe(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        router = half(self.router) if self.decode else self.router
        probs = torch.softmax(F.linear(x, router), dim=-1)
        top_p, top_i = torch.topk(probs, c.used, dim=-1)
        top_w = top_p / top_p.sum(-1, keepdim=True).clamp_min(6.103515625e-5)
        out = torch.zeros(x.shape[0], c.hidden, dtype=x.dtype)
        for e in torch.unique(top_i).tolist():
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            xe = x[rows]
            h = F.silu(self.lin("gate_exps", xe, e)) * self.lin("up_exps", xe, e)
            out.index_add_(0, rows, top_w[rows, slot, None] * self.lin("down_exps", h, e))
        sh = F.silu(self.lin("shexp_gate", x)) * self.lin("shexp_up", x)
        sgw = half(self.shexp_gate_inp) if self.decode else self.shexp_gate_inp
        sg = torch.sigmoid(F.linear(x, sgw.view(1, -1)))
        return out + sg * self.lin("shexp_down", sh)

    # -- head -----------------------------------------------------------------
    def head_mixed(self, res: torch.Tensor) -> torch.Tensor:
        return self.hc_mix("hc_head", res, inject=False)[0]

    def logits(self, mixed: torch.Tensor, rows: Optional[torch.Tensor] = None,
               chunk: int = 32768) -> torch.Tensor:
        """Full head (rows=None) or the given output() rows (small draft head)."""
        x = q8_act(mixed) if self.decode else mixed
        if rows is not None:
            w = self.shared.head_q8().dequant(rows, dtype=self.dtype)
            return F.linear(x, w)
        if self.shared.cache_head:
            return F.linear(x, self.shared.head_weight(self.dtype))
        q = self.shared.head_q8()
        outs = [F.linear(x, q.dequant(slice(r, r + chunk), dtype=self.dtype))
                for r in range(0, self.cfg.vocab, chunk)]
        return torch.cat(outs, -1)

    # -- one predictor batch (Executor::MtpForward) ------------------------------
    @torch.no_grad()
    def mtp_forward(self, state: MtpState, tokens: Sequence[int], hidden: Optional[torch.Tensor] = None,
                    head: bool = True, kv_only: bool = False, draft_rows: Optional[torch.Tensor] = None,
                    trace: bool = False, stage_inputs: Optional[dict] = None) -> MtpOutput:
        """Runs `tokens` at state.position.. . Row i reads `hidden[i]` (the trunk's
        final wide residual of MTP position state.position+i, whose successor is
        tokens[i]); hidden=None (single token) reads the carried residual.
        draft_rows: optional output() row IDs of the small head (prefix + table);
        returned token/candidates are then remapped to token IDs.
        stage_inputs (validation only, like MtpStep's): observed "fused",
        "attention", "hidden", "head" rows replace this model's own values as the
        inputs of the following stage, isolating arithmetic from Q8 drift."""
        c = self.cfg
        n = len(tokens)
        if hidden is None:
            assert n == 1 and state.residual_valid, "MTP residual requires a full known-hidden forward"
            hidden = state.h[None]
        assert hidden.shape == (n, c.D)
        si = stage_inputs or {}
        tr = {} if trace else None
        res = self.fuse(tokens, hidden, tr)
        if "fused" in si:
            res = si["fused"].view(n, c.D).to(res.dtype)
        pos0 = state.position
        state.residual_valid = False
        mixed, inject = self.hc_mix("hc_attn", res, True)
        res = self.hc_combine(res, self.attention(state, mixed, pos0), inject)
        state.position = pos0 + n
        if kv_only:
            return MtpOutput()
        if tr is not None:
            tr["attention"] = res
        if "attention" in si:
            res = si["attention"].view(n, c.D).to(res.dtype)
        mixed, inject = self.hc_mix("hc_ffn", res, True)
        ffn = self.moe(si["ffn_input"].view(n, c.hidden).to(res.dtype) if "ffn_input" in si else mixed)
        if tr is not None:
            tr.update(ffn_input=mixed, ffn_output=ffn)
        if "ffn_output" in si:
            ffn = si["ffn_output"].view(n, c.hidden).to(res.dtype)
        res = self.hc_combine(res, ffn, inject)
        if tr is not None:
            tr["hidden"] = res
        if "hidden" in si:
            res = si["hidden"].view(n, c.D).to(res.dtype)
        state.h = res[-1].clone()
        state.residual_valid = True
        out = MtpOutput(trace=tr)
        if head:
            hm = self.head_mixed(res[-1:])
            if tr is not None:
                tr["head"] = hm
            if "head" in si:
                hm = si["head"].view(1, c.hidden).to(hm.dtype)
            lg = self.logits(hm, draft_rows)[0]
            vals, idx = torch.sort(lg, descending=True, stable=True)
            ids = idx[:MTP_CANDIDATES] if draft_rows is None else draft_rows[idx[:MTP_CANDIDATES]]
            out.logits = lg
            out.candidates = (ids, vals[:MTP_CANDIDATES])
            out.token = int(ids[0])
        return out

    # -- training / offline evaluation: all anchors in parallel -------------------
    def block_unrolled(self, hidden: torch.Tensor, next_tokens: torch.Tensor, depth: int,
                       feed: str = "greedy", teacher: Optional[torch.Tensor] = None,
                       head_rows: Optional[torch.Tensor] = None, return_logits: bool = True):
        """Teacher-forced catch-up for every row plus `depth`-1 recursive steps for
        every anchor, with gufo's exact cache visibility ("training-time test").

        hidden[T, D]: trunk final wide residuals h_0..h_{T-1};
        next_tokens[T]: t_1..t_T (row p's shifted token, row p = MTP position p).
        Chain p (anchor token t_{p+1}) drafts d_{p,1..depth}; depth-j row of chain
        p sits at position p+j-1 and sees depth-1 K/V of rows <= p plus its own
        chain's depth-2..j rows (Executor::MtpForward with hidden_row -1).
        feed: "greedy" feeds argmax drafts (gufo), "teacher" feeds teacher[p, j-2]
        (= t_{p+j}) for depth j>=2.
        Returns list of per-depth logits [T, rows] (and residual list).
        Dense attention only: exact while T + depth - 1 <= idx_top_k + ratio - 1."""
        c = self.cfg
        T = hidden.shape[0]
        assert T + depth - 1 <= c.idx_top_k + c.ratio - 1, "QSA sparse regime: use mtp_forward chains"
        group = c.heads // c.kv_heads
        scale = 1.0 / math.sqrt(c.head_dim)
        pos1 = torch.arange(T)
        K1 = V1 = None
        chain_kv = []  # per depth >=2: (k [T,kvh,d], v)
        logits_out, res_out = [], []
        tokens = next_tokens
        h = hidden
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        for j in range(1, depth + 1):
            pos = pos1 + (j - 1)
            res = self.fuse(tokens.tolist(), h)
            mixed, inject = self.hc_mix("hc_attn", res, True)
            q, gate, k, v, _, _ = self.qkv(mixed, pos)
            if j == 1:
                K1, V1 = k, v
            else:
                chain_kv.append((k, v))
            kh = K1.repeat_interleave(group, 1)                     # [T, heads, d]
            s1 = torch.einsum("phd,qhd->hpq", q, kh) * scale        # [heads, T, T]
            s1 = s1.masked_fill(~causal[None], float("-inf"))
            diag = [torch.einsum("phd,phd->hp", q, ck.repeat_interleave(group, 1)) * scale
                    for ck, _ in chain_kv]                          # each [heads, T]
            s = torch.cat([s1] + [d[..., None] for d in diag], dim=-1)
            p = torch.softmax(s, dim=-1)
            ctx = torch.einsum("hpq,qhd->phd", p[..., :T], V1.repeat_interleave(group, 1))
            for i, (_, cv) in enumerate(chain_kv):
                ctx = ctx + p[..., T + i].transpose(0, 1)[..., None] * cv.repeat_interleave(group, 1)
            ctx = ctx * torch.sigmoid(gate)
            attn = self.lin("attn_output", ctx.reshape(T, c.heads * c.head_dim))
            res = self.hc_combine(res, attn, inject)
            mixed, inject = self.hc_mix("hc_ffn", res, True)
            res = self.hc_combine(res, self.moe(mixed), inject)
            res_out.append(res)
            lg = self.logits(self.head_mixed(res), head_rows) if return_logits or feed == "greedy" else None
            logits_out.append(lg)
            if j < depth:
                if feed == "greedy":
                    nxt = lg.argmax(-1)
                    tokens = nxt if head_rows is None else head_rows[nxt]
                else:
                    tokens = teacher[:, j - 1]
                h = res
        return logits_out, res_out


# ---------------------------------------------------------------------------
# gufo draft-cycle semantics
# ---------------------------------------------------------------------------
def greedy_chain(model: MtpDrafter, state: MtpState, catchup_tokens: Sequence[int],
                 catchup_hidden: torch.Tensor, k: int, draft_rows: Optional[torch.Tensor] = None,
                 confidence_stop: bool = False) -> tuple[list[int], list[MtpOutput]]:
    """One gufo greedy draft cycle (Session::PrepareDecode, engine.cpp 1171-1260).

    catchup rows = DraftReplay: MTP positions state.position.. consume the kept
    trunk rows and their successor tokens; the final successor is the anchor.
    Returns k drafts (fewer with confidence_stop: a draft whose top1-top2 logit
    gap is <= 0.5 ends the chain after itself)."""
    outs = [model.mtp_forward(state, catchup_tokens, catchup_hidden, draft_rows=draft_rows)]
    drafts = [outs[-1].token]
    while len(drafts) < k:
        if confidence_stop:
            ids, vals = outs[-1].candidates
            if float(vals[0] - vals[1]) <= GREEDY_DRAFT_CONFIDENCE_FLOOR:
                break
        outs.append(model.mtp_forward(state, [drafts[-1]], None, draft_rows=draft_rows))
        drafts.append(outs[-1].token)
    return drafts, outs


class GufoDraftSession:
    """Tracks one request like engine.cpp Session: committed tokens, kept trunk
    rows from hidden_base, the predictor's trailing position, and rewind after
    verification (FinishDecode -> MtpRewind(base))."""

    def __init__(self, model: MtpDrafter):
        self.model = model
        self.state = MtpState()
        self.tokens: list[int] = []
        self.hidden: dict[int, torch.Tensor] = {}  # position -> trunk final residual

    def prefill(self, tokens: Sequence[int], hidden: torch.Tensor) -> None:
        """Target prefill of `tokens` with their final residuals: KV-only catch-up
        of every known successor (Executor::Forward kPrefill, 3302-3310)."""
        base = len(self.tokens)
        self.tokens.extend(int(t) for t in tokens)
        for i in range(len(tokens)):
            self.hidden[base + i] = hidden[i]
        mp = self.state.position
        if len(self.tokens) - 1 > mp:
            rows = list(range(mp, len(self.tokens) - 1))
            self.model.mtp_forward(self.state, [self.tokens[p + 1] for p in rows],
                                   torch.stack([self.hidden[p] for p in rows]), head=False, kv_only=True)

    def cycle(self, anchor: int, k: int, **kw) -> list[int]:
        """Draft k tokens after `anchor` (the next committed token)."""
        mp = self.state.position
        rows = list(range(mp, len(self.tokens)))
        succ = self.tokens[mp + 1:] + [anchor]
        self.base = len(self.tokens)
        drafts, _ = greedy_chain(self.model, self.state, succ,
                                 torch.stack([self.hidden[p] for p in rows]), k, **kw)
        return drafts

    def commit(self, accepted: Sequence[int], hidden: torch.Tensor) -> None:
        """After target verification: `accepted` = anchor + accepted drafts, with
        their trunk final residuals. Rewinds the predictor to the cycle base."""
        for i, t in enumerate(accepted):
            self.hidden[self.base + i] = hidden[i]
        self.tokens.extend(int(t) for t in accepted)
        self.state.rewind(self.base, self.model.cfg.ratio)


class MtpDraftVocab:
    """draft_vocab.hpp policy for the small draft head."""

    def __init__(self, specials: Sequence[int]):
        self.specials = sorted(specials)
        self.special_set = set(specials)
        self.capacity = MTP_DRAFT_EXTRA_ROWS - len(self.specials)
        self.rare: list[int] = []
        self.known: set[int] = set()
        self.scanned = 0
        self.last_miss = self.prior_miss = None

    def common(self, t: int) -> bool:  # Model::InDraftSubset
        return 0 <= t < MTP_DRAFT_VOCAB or t in self.special_set

    def catch_up(self, committed: Sequence[int]) -> None:
        for self.scanned in range(self.scanned, len(committed)):
            t = committed[self.scanned]
            if self.common(t) or t in self.known:
                continue
            self.prior_miss, self.last_miss = self.last_miss, self.scanned
            if t < 0 or len(self.rare) >= self.capacity:
                continue
            self.known.add(t)
            self.rare.append(t)
        self.scanned = len(committed)

    def quiet(self) -> bool:
        return self.prior_miss is None or self.prior_miss + MTP_DRAFT_VOCAB_WINDOW < self.scanned

    def fits(self, anchor: int) -> bool:
        return (self.common(anchor) or anchor in self.known) and self.quiet()

    def rows(self) -> torch.Tensor:
        """Head rows in gufo order: prefix, specials, rare (empty slots omitted:
        they score -inf and can never win)."""
        return torch.cat([torch.arange(MTP_DRAFT_VOCAB),
                          torch.tensor(self.specials + self.rare, dtype=torch.long)])


def offline_acceptance(model: MtpDrafter, hidden: torch.Tensor, tokens: torch.Tensor,
                       target_next: torch.Tensor, depth: int,
                       head_rows: Optional[torch.Tensor] = None) -> dict:
    """Per-depth greedy acceptance with every position as anchor.

    hidden[T, D]: trunk final residuals h_0..h_{T-1}; tokens[T+1]: t_0..t_T;
    target_next[T + depth]: target greedy argmax at each position (prediction of
    the following token) -- equals tokens[1:] for target-greedy text.
    Chain p (anchor t_{p+1}) accepts d_{p,j} iff d_{p,1..j} all equal the
    target's greedy tokens target_next[p+1 .. p+j]."""
    T = hidden.shape[0]
    logits, _ = model.block_unrolled(hidden, tokens[1:T + 1], depth, feed="greedy", head_rows=head_rows)
    drafts = torch.stack([lg.argmax(-1) if head_rows is None else head_rows[lg.argmax(-1)]
                          for lg in logits], 1)                        # [T, depth]
    idx = torch.arange(T)[:, None] + torch.arange(1, depth + 1)[None]
    valid = idx < target_next.shape[0]
    want = torch.where(valid, target_next[idx.clamp(max=target_next.shape[0] - 1)], -1)
    ok = (drafts == want) & valid
    cum = torch.cumprod(ok.int(), 1).bool()
    rows_ok = valid.all(1)
    return {"drafts": drafts, "accept_cum": cum,
            "accept_rate_by_depth": cum[rows_ok].float().mean(0).tolist(),
            "expected_accepted_per_cycle(k=depth)": float(cum[rows_ok].float().sum(1).mean())}
