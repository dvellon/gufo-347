"""GGML block quantizers (ports of ggml-quants.c *_ref, no imatrix), vectorized in torch.

q8_0_bytes / q8_0_from_bytes : exact Q8_0 encoder/decoder (export_gguf writes these bytes)
fake(x, kind)                : quantize -> dequantize round trip, for offline what-if
                               evaluation ("lean drafter" A3: Q4_K layer, Q4_0 head).
Rounding follows ggml: roundf (half away from zero) for Q8_0, (int8_t)(v + 8.5f)
truncation for Q4_0/Q5_0, nearest_int (half to even) for Q4_K/make_qkx2_quants.
Q4_K needs rows that are a multiple of 256; like llama-quantize, other rows fall
back (default Q5_0). Accumulation order inside make_qkx2_quants differs from the C
loop (vectorized sums), so a rare Q4_K block may pick a different (equally
optimal) scale; Q8_0/Q4_0/Q5_0 are bit-exact ports.
"""
from __future__ import annotations

import numpy as np
import torch


def _f16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def _round_away(x: torch.Tensor) -> torch.Tensor:
    # exact for float32 inputs (computed in float64)
    xd = x.double()
    return (torch.sign(xd) * torch.floor(xd.abs() + 0.5)).to(torch.float32)


# ---------------------------------------------------------------------------
# Q8_0
# ---------------------------------------------------------------------------
def q8_0_quant(x: torch.Tensor):
    """x [..., n*32] f32 -> (d_f16 [..., n] float16, qs int8 [..., n*32]) as ggml quantize_row_q8_0_ref."""
    xb = x.float().unflatten(-1, (-1, 32))
    amax = xb.abs().amax(-1, keepdim=True)
    d = amax / 127.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = _round_away(xb * idv).clamp(-128, 127).to(torch.int8)
    return d.squeeze(-1).to(torch.float16), q.flatten(-2)


def q8_0_bytes(x: torch.Tensor) -> bytes:
    """Rows of x (last dim multiple of 32) -> GGUF Q8_0 block bytes (34 B / 32 values)."""
    d, q = q8_0_quant(x.reshape(-1, x.shape[-1]))
    nb = d.shape[-1]
    blocks = np.empty((d.shape[0], nb, 34), np.uint8)
    blocks[..., :2] = d.numpy().view(np.uint8).reshape(d.shape[0], nb, 2)
    blocks[..., 2:] = q.numpy().view(np.uint8).reshape(d.shape[0], nb, 32)
    return blocks.tobytes()


def q8_0_from_bytes(raw: bytes | np.ndarray, cols: int) -> torch.Tensor:
    b = np.frombuffer(raw, np.uint8).reshape(-1, cols // 32, 34)
    d = torch.from_numpy(b[..., :2].copy().view(np.float16).astype(np.float32))
    q = torch.from_numpy(b[..., 2:].copy().view(np.int8)).float()
    return (q * d).reshape(-1, cols)


def q8_0_fake(x):
    d, q = q8_0_quant(x)
    return (q.float().unflatten(-1, (-1, 32)) * d.float().unsqueeze(-1)).flatten(-2)


# ---------------------------------------------------------------------------
# Q4_0 / Q5_0 (signed max, d = max / -2^(b-1))
# ---------------------------------------------------------------------------
def _signed_max(xb):
    idx = xb.abs().argmax(-1, keepdim=True)          # first max |x| (C: strict '<' keeps first)
    return torch.gather(xb, -1, idx)


def q4_0_fake(x):
    xb = x.float().unflatten(-1, (-1, 32))
    mx = _signed_max(xb)
    d = mx / -8.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.trunc(xb * idv + 8.5).clamp(max=15)   # (int8_t)(x0 + 8.5f), MIN(15, .)
    return ((q - 8.0) * _f16(d)).flatten(-2)


def q5_0_fake(x):
    xb = x.float().unflatten(-1, (-1, 32))
    mx = _signed_max(xb)
    d = mx / -16.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.trunc(xb * idv + 16.5).clamp(max=31)
    return ((q - 16.0) * _f16(d)).flatten(-2)


# ---------------------------------------------------------------------------
# Q4_K (super-block 256 = 8 x 32, 6-bit scales and mins)
# ---------------------------------------------------------------------------
def _make_qkx2(x, w, nmax=15, rmin=-1.0, rdelta=0.1, nstep=20):
    """x, w [B, 32] -> (scale [B], the_min [B], L [B, 32]) as make_qkx2_quants(use_mad=false)."""
    mn = x.amin(-1).clamp(max=0.0)
    mx = x.amax(-1)
    sum_w = w.sum(-1)
    sum_x = (w * x).sum(-1)
    flat = mx == mn
    rng = torch.where(flat, torch.ones_like(mx), mx - mn)
    iscale = nmax / rng
    scale = 1.0 / iscale
    L = torch.round(iscale[:, None] * (x - mn[:, None])).clamp(0, nmax)
    best = (w * (scale[:, None] * L + mn[:, None] - x) ** 2).sum(-1)
    b_scale, b_min = scale, mn
    for i in range(nstep + 1):
        # ggml updates `min` on every improvement and reuses it in later steps
        rng_i = torch.where(flat, torch.ones_like(mx), mx - b_min)
        isc = (rmin + rdelta * i + nmax) / rng_i
        La = torch.round(isc[:, None] * (x - b_min[:, None])).clamp(0, nmax)
        sl, sl2, sxl = (w * La).sum(-1), (w * La * La).sum(-1), (w * La * x).sum(-1)
        D = sum_w * sl2 - sl * sl
        Ds = torch.where(D > 0, D, torch.ones_like(D))
        ts = (sum_w * sxl - sum_x * sl) / Ds
        tm = (sl2 * sum_x - sl * sxl) / Ds
        pos = tm > 0
        ts = torch.where(pos, sxl / sl2.clamp(min=1e-30), ts)
        tm = torch.where(pos, torch.zeros_like(tm), tm)
        cur = (w * (ts[:, None] * La + tm[:, None] - x) ** 2).sum(-1)
        upd = (D > 0) & (cur < best)
        best = torch.where(upd, cur, best)
        b_scale = torch.where(upd, ts, b_scale)
        b_min = torch.where(upd, tm, b_min)
        L = torch.where(upd[:, None], La, L)
    scale = torch.where(flat, torch.zeros_like(b_scale), b_scale)
    the_min = torch.where(flat, -mn, -b_min)
    L = torch.where(flat[:, None], torch.zeros_like(L), L)
    return scale, the_min, L


def q4_k_fake(x):
    shape = x.shape
    xb = x.float().reshape(-1, 8, 32)                  # [S, 8, 32]
    S = xb.shape[0]
    flat = xb.reshape(-1, 32)
    av = torch.sqrt((flat * flat).sum(-1) / 32)
    w = av[:, None] + flat.abs()
    sc, mins, L = _make_qkx2(flat, w)
    sc, mins, L = sc.view(S, 8), mins.view(S, 8), L.view(S, 8, 32)
    max_scale = sc.amax(-1).clamp(min=0)
    max_min = mins.amax(-1).clamp(min=0)
    inv_s = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
    inv_m = torch.where(max_min > 0, 63.0 / max_min, torch.zeros_like(max_min))
    ls = torch.round(inv_s[:, None] * sc).clamp(0, 63)
    lm = torch.round(inv_m[:, None] * mins).clamp(0, 63)
    d = _f16(max_scale / 63.0)
    dmin = _f16(max_min / 63.0)
    dd = d[:, None] * ls                                # [S, 8]
    dm = dmin[:, None] * lm
    ddz = torch.where(dd != 0, dd, torch.ones_like(dd))
    l = torch.round((xb + dm[..., None]) / ddz[..., None]).clamp(0, 15)
    l = torch.where((dd != 0)[..., None], l, L)
    y = dd[..., None] * l - dm[..., None]
    return y.reshape(shape)


def encode(x: torch.Tensor, kind: str, fallback: str = "q5_0"):
    """Block codes with dequant = codes * scale - offset per 32-block:
    (codes int8 [..., n], scale f32 [..., n/32], offset f32 [..., n/32])."""
    x = x.float()
    if kind == "q4_k" and x.shape[-1] % 256:
        kind = fallback
    if kind == "q8_0":
        d, qv = q8_0_quant(x)
        return qv, d.float(), torch.zeros_like(d, dtype=torch.float32)
    if kind in ("q4_0", "q5_0"):
        half_range = 8.0 if kind == "q4_0" else 16.0
        top = 15 if kind == "q4_0" else 31
        xb = x.unflatten(-1, (-1, 32))
        d = _signed_max(xb) / -half_range
        idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
        qv = torch.trunc(xb * idv + (half_range + 0.5)).clamp(max=top)
        return (qv - half_range).to(torch.int8).flatten(-2), _f16(d).squeeze(-1), torch.zeros_like(d.squeeze(-1))
    if kind == "q4_k":
        shape = x.shape
        xb = x.reshape(-1, 8, 32)
        S = xb.shape[0]
        flat = xb.reshape(-1, 32)
        av = torch.sqrt((flat * flat).sum(-1) / 32)
        w = av[:, None] + flat.abs()
        sc, mins, L = _make_qkx2(flat, w)
        sc, mins, L = sc.view(S, 8), mins.view(S, 8), L.view(S, 8, 32)
        max_scale = sc.amax(-1).clamp(min=0)
        max_min = mins.amax(-1).clamp(min=0)
        inv_s = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
        inv_m = torch.where(max_min > 0, 63.0 / max_min, torch.zeros_like(max_min))
        ls = torch.round(inv_s[:, None] * sc).clamp(0, 63)
        lm = torch.round(inv_m[:, None] * mins).clamp(0, 63)
        dd = _f16(max_scale / 63.0)[:, None] * ls
        dm = _f16(max_min / 63.0)[:, None] * lm
        ddz = torch.where(dd != 0, dd, torch.ones_like(dd))
        l = torch.round((xb + dm[..., None]) / ddz[..., None]).clamp(0, 15)
        l = torch.where((dd != 0)[..., None], l, L)
        lead = shape[:-1]
        return (l.to(torch.int8).reshape(shape), dd.reshape(*lead, shape[-1] // 32),
                dm.reshape(*lead, shape[-1] // 32))
    raise ValueError(kind)


def decode(codes: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    y = codes.float().unflatten(-1, (-1, 32)) * scale.unsqueeze(-1) - offset.unsqueeze(-1)
    return y.flatten(-2)


def fake(x: torch.Tensor, kind: str, fallback: str = "q5_0") -> torch.Tensor:
    """Quantize->dequantize along the last dim. kind: q8_0 | q4_0 | q5_0 | q4_k."""
    if kind == "q8_0":
        return q8_0_fake(x)
    return decode(*encode(x, kind, fallback))


def selftest() -> dict:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 512, generator=g) * torch.rand(64, 1, generator=g) * 3
    out = {}
    for k in ("q8_0", "q5_0", "q4_0", "q4_k"):
        y = fake(x, k)
        out[k] = float(((y - x).norm() / x.norm()))
    # Q8_0 idempotence: quantizing a Q8_0-dequantized tensor reproduces it
    y = q8_0_fake(x)
    out["q8_0_idempotent"] = bool(torch.equal(q8_0_fake(y), y))
    raw = q8_0_bytes(x)
    out["q8_0_bytes_roundtrip"] = bool(torch.equal(q8_0_from_bytes(raw, 512), y))
    # expected relative RMS: q8 ~0.4%, q5_0 ~2-3%, q4_k ~3-5%, q4_0 ~6-8%
    for k in ("q4_0", "q5_0", "q4_k"):
        ref = {"q4_0": q4_0_fake, "q5_0": q5_0_fake, "q4_k": q4_k_fake}[k](x)
        out[k + "_codes_equal_fake"] = bool(torch.equal(decode(*encode(x, k)), ref))
    out["ordering_ok"] = out["q8_0"] < out["q5_0"] < out["q4_0"] and out["q4_k"] < out["q4_0"]
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=1))
