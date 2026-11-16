"""Stage 2: batched routed-expert GEMMs for MtpNet.moe (replaces the per-expert Python loop).

Stage 1 measured 323 tok/s on the H100 (~8.5x below estimate): MtpNet.moe ran one
Python iteration per active expert (512) with ~23 eager ops each (3 base GEMMs,
3 x LoRA (2 GEMMs + 2 casts + scale + add), silu, mul), times forward + checkpoint
recompute + backward, times 7 depths = ~2/3 of a million launches per record,
plus one host sync (`counts.tolist()`) per MoE call.

Here the same math runs as a handful of grouped GEMMs over the expert-major sorted
rows that MtpNet.moe already builds (`xs = x[tok]`, sorted by expert, stable):

  impl "grouped": torch.nn.functional.grouped_mm (torch._grouped_mm) with int32 offsets
                  (CUDA: bf16 only, SM>=80; CPU: reference fallback, bit-identical to the
                  loop for fp32 and bf16 incl. backward - measured on the Dell, torch 2.14.1).
                  No host sync: offsets stay on the device.
  impl "bmm":     bucketed padded torch.bmm (power-of-two capacity buckets; per bucket
                  one gather of rows and one gather of that bucket's expert weights).
                  Portable fallback for any device/dtype; one host sync per MoE call.
  impl "auto":    same as "grouped". Where grouped_mm is unsupported (e.g. the f32
                  decode-numerics evals on CUDA) both fall back to the stage-1 loop.

Numerics per expert are the loop's: q8_act (decode) before every Q8 weight, the same
weight transform (`MtpNet._w`: QAT fake-quant, dtype cast), y = x W^T [+ (x A^T) B^T * s].
The caller keeps the loop's scatter (one index_add in expert-major row order), so on
the CPU fallback the MoE output is bit-identical to the loop. On CUDA, grouped kernels
sum in a different order than per-expert cuBLAS GEMMs (bf16 rounding-level
differences); `selftest_moe.py --device cuda` measures that on the real weights
(J0 of run_stage2.sh) before any training step.

Only dense (materialized) expert storage is supported; q8 storage (CPU eval) keeps
the loop. NetOptions.moe_impl defaults to "loop", so stage-1 behaviour and old
checkpoints are unchanged.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_GMM = getattr(F, "grouped_mm", None) or getattr(torch, "_grouped_mm", None)
QAT_CHUNK = 64          # experts per q8_fake call (bounds the fp32 transient of full-expert QAT)
GMM_ALIGN = int(os.environ.get("GUFO_GMM_ALIGN", "0"))   # see GroupedPlan (K0 fallback on CUDA: 8)


def grouped_supported(device: torch.device, dtype: torch.dtype) -> bool:
    if _GMM is None:
        return False
    if device.type == "cpu":
        return dtype in (torch.float32, torch.bfloat16)
    if device.type == "cuda":
        return dtype == torch.bfloat16 and torch.cuda.get_device_capability(device)[0] >= 8
    return False


def resolve_impl(impl: str, device: torch.device, dtype: torch.dtype) -> str:
    """grouped/auto fall back to the stage-1 loop where grouped_mm is unsupported (e.g. the
    f32 decode-numerics evals on CUDA), so gate evals keep the stage-1 numerics; bmm only
    when asked for explicitly."""
    if impl in ("grouped", "auto"):
        return "grouped" if grouped_supported(device, dtype) else "loop"
    return impl


# ---------------------------------------------------------------------------
# weights (whole [E, out, in] tensors, same transform as MtpNet._w per expert)
# ---------------------------------------------------------------------------
def all_expert_weight(net, attr: str, dtype) -> torch.Tensor:
    import mtp_net as N
    w = getattr(net, attr)
    if net.opt.qat and net.q8.get(attr, False):
        # q8_fake is per 32-block of a row: chunking over experts is elementwise-identical
        w = torch.cat([N.q8_fake(w[e0:e0 + QAT_CHUNK]) for e0 in range(0, w.shape[0], QAT_CHUNK)], 0)
    return w.to(dtype)


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------
class GroupedPlan:
    """offs: int32 cumulative row counts per expert (device; no host sync).

    align > 0 (env GUFO_GMM_ALIGN, e.g. 8): every expert's row group is zero-padded to a
    multiple of `align` rows before each grouped GEMM (as torchtitan does for its MoE), in
    case the CUDA kernels need 16-byte aligned group offsets where the token dimension is
    the contraction dimension (the LoRA weight gradients). The padded buffer has a static
    upper-bound size (S + (align-1)*E rows) so no host sync is needed; rows past offs[-1]
    are ignored by grouped_mm and never read back. Off by default (CPU verified both ways)."""

    def __init__(self, flat_e: torch.Tensor, experts: int, align: int = 0):
        counts = torch.bincount(flat_e, minlength=experts)
        self.dest = None
        if not align:
            self.offs = torch.cumsum(counts, 0).to(torch.int32)
            return
        padded = (counts + align - 1) // align * align
        start = torch.cumsum(counts, 0) - counts
        pstart = torch.cumsum(padded, 0) - padded
        S = flat_e.numel()
        self.dest = pstart[flat_e] + torch.arange(S, device=flat_e.device) - start[flat_e]
        self.rows = S + (align - 1) * experts
        self.offs = torch.cumsum(padded, 0).to(torch.int32)

    def mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x [S, in] expert-major rows; w [E, out, in] -> [S, out]."""
        if self.dest is None:
            return _GMM(x, w.transpose(-2, -1), offs=self.offs)
        xp = x.new_zeros(self.rows, x.shape[1]).index_copy(0, self.dest, x)
        return _GMM(xp, w.transpose(-2, -1), offs=self.offs).index_select(0, self.dest)


class BucketPlan:
    """Experts grouped by power-of-two capacity (>= min_cap); per bucket: expert ids,
    a row-gather index into [x; 0] (padding -> the zero row), and where the valid
    outputs land. Output rows are restored to the expert-major order with one
    index_select (differentiable)."""

    def __init__(self, flat_e: torch.Tensor, experts: int, min_cap: int = 8):
        dev = flat_e.device
        counts = torch.bincount(flat_e, minlength=experts)
        ch = counts.tolist()                                    # one host sync per MoE call
        starts, a = [], 0
        for c in ch:
            starts.append(a)
            a += c
        self.S = a
        buckets: dict[int, list[int]] = {}
        for e, c in enumerate(ch):
            if c:
                cap = max(min_cap, 1 << (c - 1).bit_length())
                buckets.setdefault(cap, []).append(e)
        self.buckets = []
        order_parts = []
        for cap in sorted(buckets):
            es = buckets[cap]
            gidx, valid = [], []
            for i, e in enumerate(es):
                c = ch[e]
                gidx += list(range(starts[e], starts[e] + c)) + [self.S] * (cap - c)
                valid += list(range(i * cap, i * cap + c))
                order_parts += list(range(starts[e], starts[e] + c))
            self.buckets.append({
                "cap": cap,
                "eids": torch.tensor(es, dtype=torch.long, device=dev),
                "gidx": torch.tensor(gidx, dtype=torch.long, device=dev),
                "valid": torch.tensor(valid, dtype=torch.long, device=dev),
                "all": len(es) == experts,
            })
        # concatenated bucket outputs are in `order_parts` row order; invert to expert-major
        order = torch.tensor(order_parts, dtype=torch.long, device=dev)
        self.inv = torch.empty_like(order)
        self.inv[order] = torch.arange(len(order_parts), device=dev)

    def mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xz = torch.cat([x, x.new_zeros(1, x.shape[1])], 0)
        outs = []
        for b in self.buckets:
            E_b, cap = b["eids"].numel(), b["cap"]
            xb = xz.index_select(0, b["gidx"]).view(E_b, cap, x.shape[1])
            wb = w if b["all"] else w.index_select(0, b["eids"])
            yb = torch.bmm(xb, wb.transpose(1, 2)).reshape(E_b * cap, -1)
            outs.append(yb.index_select(0, b["valid"]))
        return torch.cat(outs, 0).index_select(0, self.inv)


# ---------------------------------------------------------------------------
# the routed-expert FFN over expert-major rows
# ---------------------------------------------------------------------------
def expert_rows(net, xs: torch.Tensor, flat_e_sorted: torch.Tensor, impl: str) -> torch.Tensor:
    """xs [S, H]: rows sorted by expert (MtpNet.moe's `x[tok]`); flat_e_sorted [S]: their
    expert ids (ascending). Returns down(silu(gate(x)) * up(x)) per row, same order, or
    None when `impl` resolves to the loop (caller runs the stage-1 loop)."""
    import mtp_net as N
    c = net.cfg
    impl = resolve_impl(impl, xs.device, xs.dtype)
    if impl == "grouped":
        # grouped_mm needs 16-byte aligned strides: hidden, expert_ff and LoRA ranks x element size
        dims = [c.hidden, c.expert_ff] + [net.lora[a].A.shape[1] for a in N.EXPERTS if a in net.lora]
        if any(d * xs.element_size() % 16 for d in dims):
            impl = "loop"
    if impl == "loop":
        return None
    plan = (GroupedPlan(flat_e_sorted, c.experts, GMM_ALIGN) if impl == "grouped"
            else BucketPlan(flat_e_sorted, c.experts))

    def lin(attr, x):
        if net.opt.decode and net.q8[attr]:
            x = N.q8_act(x)
        y = plan.mm(x, all_expert_weight(net, attr, x.dtype))
        if attr in net.lora:
            lo = net.lora[attr]
            y = y + plan.mm(plan.mm(x, lo.A.to(x.dtype)), lo.B.to(x.dtype)) * lo.scale
        return y

    h = F.silu(lin("gate_exps", xs)) * lin("up_exps", xs)
    return lin("down_exps", h)
