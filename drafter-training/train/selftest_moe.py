"""Stage 2: batched routed experts (moe_grouped.py) vs the stage-1 per-expert loop.

  T  tiny synthetic MTPs (4 and 64 experts): training step in f32 and bf16 (dense full-FT +
     expert LoRA; expert full-FT + QAT + dense LoRA) - per-depth residuals, loss and every
     trainable gradient; decode numerics (no grad) - drafts and gaps.
  R  real weights (--mtp-gguf, --frozen-bundle, --shard), the stage-2 training config
     (dense full-FT, expert LoRA r16 with non-zero B, bf16 experts and activations):
     LAYER  one MoE call on the record's real depth-1 FFN input (identical routing by
            construction): output, input gradient and every MoE parameter gradient;
     CHAIN  one record, train_mtp.DepthLoss, depth --depth, per-depth checkpointing: per-depth
            residuals, loss, every trainable gradient;
     DRAFTS quick-eval drafts (no grad, small head).

Why two levels. The chain is chaotic in bf16: any rounding change flips some top-10-of-512
routing decisions (and draft near-ties), and the 7-depth recurrence amplifies them - on the
real weights a correct but differently-rounded implementation moves depth-7 residuals by
~25 % rel-RMS. So the CHAIN and DRAFTS comparisons are judged against a FLOOR: the stage-1
loop itself rerun with the trunk hidden perturbed by one ulp on 1 % of its entries. The LAYER
check isolates the kernels (same routing) and has fixed tolerances (TOL_LAYER).

Verdict per implementation:
  bit-identical everywhere -> PASS (CPU grouped without alignment must be: torch's CPU
      grouped_mm runs the same per-group GEMMs as the loop);
  else LAYER within TOL_LAYER and CHAIN within max(TOL, 3 x floor) per depth/loss/gradients
      and DRAFTS agreement >= floor agreement - 0.05 -> PASS.
  On CUDA nothing is bit-identical (index_add atomics make even loop-vs-loop differ).

  python selftest_moe.py --tiny-only
  python selftest_moe.py --skip-tiny --shard <val.mtps> --mtp-gguf <sidecar> --frozen-bundle <frozen-target.bin> \
      --rows 48 --depth 7 --threads 4 --rss-cap-gb 9.5 [--device cuda] [--impls grouped] --out moe-check.json
"""
from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import numpy as np
import torch

import moe_grouped
import mtp_net as N
import tiny
import train_mtp as TM

TOL = {  # CHAIN (residual rel-RMS, loss rel, worst gradient rel-L2) when no floor is larger
    torch.float32: (1e-5, 1e-5, 1e-4),
    torch.bfloat16: (3e-2, 1e-2, 5e-2),
}
TOL_LAYER = {  # LAYER (output rel-RMS, worst gradient rel-L2), same routing by construction
    torch.float32: (1e-5, 1e-4),
    torch.bfloat16: (1e-2, 3e-2),
}


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def loss_args():
    return types.SimpleNamespace(head_chunk=512, kl_weight=1.0, ce_weight=0.5, depth_decay=0.85,
                                 offpath_weight=0.5)


def perturbed(rd: "N.RecordData", frac: float = 0.01, seed: int = 0) -> "N.RecordData":
    """One-ulp flip (lowest mantissa bit) on `frac` of the trunk hidden entries."""
    h = rd.hidden_bf16.copy()
    bits = h.view(np.uint16) if h.dtype == np.uint16 else h.view(np.uint32)
    m = np.random.default_rng(seed).random(bits.shape) < frac
    bits[m] ^= 1
    return N.RecordData(rd.rid + "~ulp", rd.tokens, h, rd.sup_ids, rd.sup_lp, rd.supervised, rd.kinds, rd.meta)


def train_pass(net, rd, depth, impl, dev, dtype, W):
    net.opt.moe_impl = impl
    net.zero_grad(set_to_none=True)
    lf = TM.DepthLoss(net, W, loss_args(), np.ones(len(N.KIND_CODES), np.float32))
    lf.record(rd, dev)
    res_all = []

    def od(j, res):
        res_all.append(res.detach().clone())
        return lf(j, res)

    hidden = rd.hidden_tensor(dev, dtype)
    tokens = torch.from_numpy(rd.tokens).to(dev)
    sync(dev)
    t0 = time.time()
    net.unrolled(hidden, tokens, depth, "teacher", od, checkpoint=True)
    loss = lf.loss / max(lf.wsum, 1e-6)
    loss.backward()
    sync(dev)
    dt = time.time() - t0
    # grads stay on the parameters (compare() reads them in place; only the baseline is cloned)
    grads = {n: p.grad.detach() for n, p in net.trainable_named() if p.grad is not None}
    return {"loss": loss.detach(), "res": res_all, "grads": grads, "seconds": dt}


def compare(base, other) -> dict:
    res_rel = [rel(o, b) for o, b in zip(other["res"], base["res"])]
    res_eq = all(torch.equal(o, b) for o, b in zip(other["res"], base["res"]))
    loss_rel = abs(float(other["loss"]) - float(base["loss"])) / max(abs(float(base["loss"])), 1e-30)
    names = sorted(base["grads"])
    missing = [n for n in names if n not in other["grads"]]
    g_rel = {n: rel(other["grads"][n], base["grads"][n]) for n in names if n not in missing}
    g_eq = all(torch.equal(other["grads"][n], base["grads"][n]) for n in names if n not in missing)
    worst = max(g_rel.items(), key=lambda kv: kv[1]) if g_rel else ("-", 0.0)
    return {"bit_identical": bool(res_eq and g_eq and torch.equal(other["loss"], base["loss"]) and not missing),
            "loss_rel": loss_rel, "res_rel_rms_per_depth": [float(f"{x:.3g}") for x in res_rel],
            "grads_compared": len(g_rel), "grads_missing": missing,
            "grad_rel_worst": [worst[0], float(f"{worst[1]:.3g}")],
            "grad_rel_expert_lora": {n: float(f"{v:.3g}") for n, v in g_rel.items() if "exps" in n},
            "seconds": round(other["seconds"], 2)}


def judge_chain(c: dict, floor: dict, dtype) -> bool:
    tol_res, tol_loss, tol_g = TOL[dtype]
    if c["bit_identical"]:
        return True
    if c["grads_missing"]:
        return False
    fr = floor["res_rel_rms_per_depth"]
    ok_res = all(x <= max(tol_res, 3 * f) for x, f in zip(c["res_rel_rms_per_depth"], fr))
    return bool(ok_res and c["loss_rel"] <= max(tol_loss, 3 * floor["loss_rel"])
                and c["grad_rel_worst"][1] <= max(tol_g, 3 * floor["grad_rel_worst"][1]))


def check_impls(net, rd, depth, dev, dtype, W, impls=("grouped", "bmm")) -> dict:
    base = train_pass(net, rd, depth, "loop", dev, dtype, W)
    base["grads"] = {n: g.clone() for n, g in base["grads"].items()}
    print(f"  loop pass {base['seconds']:.1f}s rss {N.process_private_gb():.2f} GB", flush=True)
    rep = {"loop_seconds": round(base["seconds"], 2), "loss": float(base["loss"]),
           "trainable_tensors": len(base["grads"])}
    rep["floor_loop_ulp"] = compare(base, train_pass(net, perturbed(rd), depth, "loop", dev, dtype, W))
    if dev.type == "cuda":
        rep["loop_vs_loop"] = compare(base, train_pass(net, rd, depth, "loop", dev, dtype, W))
    for impl in impls:
        c = compare(base, train_pass(net, rd, depth, impl, dev, dtype, W))
        c["pass"] = judge_chain(c, rep["floor_loop_ulp"], dtype)
        rep[impl] = c
        print(f"  {impl} pass {c['seconds']:.1f}s bit_identical={c['bit_identical']} judged={c['pass']} "
              f"rss {N.process_private_gb():.2f} GB", flush=True)
    net.opt.moe_impl = "loop"
    net.zero_grad(set_to_none=True)
    return rep


def layer_check(net, rd, dev, dtype, impls) -> dict:
    """One MoE call on the record's depth-1 FFN input: identical routing for every impl."""
    c = net.cfg
    with torch.no_grad():
        n = rd.n
        hidden = rd.hidden_tensor(dev, dtype)[:n - 1]
        tok = torch.from_numpy(rd.tokens).to(dev)[1:n]
        res = net.fuse(tok, hidden, dtype)
        mixed, inject = net.hc_mix("hc_attn", res, True)
        q, gate, k, v = net.qkv(mixed, torch.arange(n - 1, device=dev))
        res = net.hc_combine(res, net.attend(q, gate, k, v, []), inject)
        x_in, _ = net.hc_mix("hc_ffn", res, True)
    g = torch.randn(x_in.shape, generator=torch.Generator().manual_seed(3)).to(dev)

    def run(impl):
        net.opt.moe_impl = impl
        net.zero_grad(set_to_none=True)
        x = x_in.detach().clone().requires_grad_(True)
        y = net.moe(x)
        (y.float() * g).sum().backward()
        gr = {nm: p.grad.detach().clone() for nm, p in net.named_parameters() if p.grad is not None}
        return y.detach(), x.grad.detach(), gr

    y0, dx0, g0 = run("loop")
    out = {"rows": int(x_in.shape[0]), "params_with_grad": len(g0)}
    for impl in impls:
        y, dx, gr = run(impl)
        g_rel = {nm: rel(gr[nm], g0[nm]) for nm in g0 if nm in gr}
        worst = max(list(g_rel.items()) + [("x", rel(dx, dx0))], key=lambda kv: kv[1])
        bit = bool(torch.equal(y, y0) and torch.equal(dx, dx0) and all(torch.equal(gr[nm], g0[nm]) for nm in g0)
                   and set(gr) == set(g0))
        tol_y, tol_g = TOL_LAYER[dtype]
        out[impl] = {"bit_identical": bit, "out_rel": float(f"{rel(y, y0):.3g}"),
                     "dx_rel": float(f"{rel(dx, dx0):.3g}"), "grad_rel_worst": [worst[0], float(f"{worst[1]:.3g}")],
                     "grad_rel_lora_exps": {nm: float(f"{v:.3g}") for nm, v in g_rel.items() if "exps" in nm},
                     "pass": bool(bit or (set(gr) == set(g0) and rel(y, y0) <= tol_y and worst[1] <= tol_g))}
    net.opt.moe_impl = "loop"
    net.zero_grad(set_to_none=True)
    return out


def drafts_check(net, rd, depth, dev, dtype, head, impls=("grouped", "bmm")) -> dict:
    net.eval()
    net.opt.moe_impl = "loop"
    d0, g0, _ = N.draft_record(net, rd, depth, head, dev, dtype)
    ok = d0 >= 0
    dp, _, _ = N.draft_record(net, perturbed(rd), depth, head, dev, dtype)
    floor = float((dp[ok] == d0[ok]).mean())
    out = {"floor_loop_ulp_drafts_equal": floor}
    for impl in impls:
        net.opt.moe_impl = impl
        d, g, _ = N.draft_record(net, rd, depth, head, dev, dtype)
        bit = bool(np.array_equal(np.nan_to_num(g), np.nan_to_num(g0)) and np.array_equal(d, d0))
        eq = float((d[ok] == d0[ok]).mean())
        out[impl] = {"drafts_equal": eq, "gaps_bit_identical": bit, "gap_max_abs": float(np.nanmax(np.abs(g - g0))),
                     "pass": bool(bit or eq >= floor - 0.05)}
    net.opt.moe_impl = "loop"
    net.train()
    return out


# ---------------------------------------------------------------------------
def tiny_suite(dev) -> dict:
    tmp = Path(__import__("tempfile").mkdtemp(prefix="moe-tiny-"))
    tr, _ = tiny.write_shards(tmp, n_records=2, n_tokens=96, seed=3)
    s = N.read_gmtp.Shard(tr)
    rd = N.read_record(s, s.records[0])
    out = {}
    for experts, used in ((4, 2), (64, 6)):
        for case in ("f32_densefull_explora", "bf16_densefull_explora", "f32_denselora_expfull_qat"):
            dtype = torch.bfloat16 if case.startswith("bf16") else torch.float32
            ref, target = tiny.make_tiny(7, experts=experts, used=used)
            target.to_dense(dev, dtype)
            dense = "lora" if "denselora" in case else "full"
            expm = "full" if "expfull" in case else "lora"
            o = N.NetOptions(decode=False, dense_mode=dense, expert_mode=expm, lora_rank=4, expert_lora_rank=8,
                             qat="qat" in case, expert_dtype=dtype if expm == "lora" else torch.float32)
            torch.manual_seed(11)
            net = N.MtpNet(ref, target, o, dev)
            for lo in net.lora.values():
                torch.nn.init.normal_(lo.B, std=0.02)
            out[f"E{experts}/{case}"] = {"chain": check_impls(net, rd, 4, dev, dtype, target.head_dense),
                                         "layer": layer_check(net, rd, dev, dtype, ("grouped", "bmm"))}
        # decode numerics (q8_act before every Q8 weight), no grad
        ref, target = tiny.make_tiny(7, experts=experts, used=used)
        target.to_dense(dev, torch.float32)
        net = N.MtpNet(ref, target, N.NetOptions(decode=True, dense_mode="frozen", expert_mode="frozen",
                                                 expert_dtype=torch.float32), dev)
        out[f"E{experts}/decode_drafts"] = drafts_check(net, rd, 4, dev, torch.float32, "full")
    return out


def real_suite(a, dev) -> dict:
    s = N.read_gmtp.Shard(a.shard)
    rd = N.truncate(N.read_record(s, s.records[a.record]), a.rows)
    target = N.FrozenTarget.from_bundle(Path(a.frozen_bundle))
    ref, _ = N.load_streamed(Path(a.mtp_gguf), None, False, target=target)
    target.to_dense(dev, torch.bfloat16, embed=dev.type == "cuda")
    o = N.NetOptions(decode=False, dense_mode="full", expert_mode="lora", expert_lora_rank=16, lora_alpha=16.0,
                     expert_dtype=torch.bfloat16, param_dtype=torch.float32)
    t0 = time.time()
    torch.manual_seed(5)
    net = N.MtpNet(ref, target, o, dev)
    del ref                                               # streamed reference: only the dense copy is needed
    for lo in net.lora.values():
        torch.nn.init.normal_(lo.B, std=1e-3)            # LoRA path contributes (B=0 at init otherwise)
    rep = {"record": rd.rid, "rows": rd.n, "depth": a.depth, "build_s": round(time.time() - t0, 1),
           "rss_gb": round(N.process_private_gb(), 2)}
    rep["layer_bf16"] = layer_check(net, rd, dev, torch.bfloat16, tuple(a.impls))
    print("  layer", json.dumps(rep["layer_bf16"]), flush=True)
    active = []                                           # routing coverage per MoE call (spy, no effect)
    orig_moe = net.moe

    def moe_spy(x):
        with torch.no_grad():
            lg = torch.nn.functional.linear(x, net.router.to(x.dtype))
            active.append(int(torch.unique(torch.topk(lg.float(), net.cfg.used, -1).indices).numel()))
        return orig_moe(x)
    net.moe = moe_spy
    rep["train_bf16"] = check_impls(net, rd, a.depth, dev, torch.bfloat16, target.head_dense, tuple(a.impls))
    net.moe = orig_moe
    rep["active_experts_per_moe_call"] = active[:a.depth]
    rep["rss_gb_after_train"] = round(N.process_private_gb(), 2)
    rep["quick_eval_drafts_bf16"] = drafts_check(net, rd, a.depth, dev, torch.bfloat16, "small", tuple(a.impls))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny-only", action="store_true")
    ap.add_argument("--skip-tiny", action="store_true")
    ap.add_argument("--shard")
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--mtp-gguf")
    ap.add_argument("--frozen-bundle")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rss-cap-gb", type=float, default=0.0)
    ap.add_argument("--impls", nargs="+", default=["grouped", "bmm"],
                    help="compared against the loop in the real-weight suite (bmm holds gathered expert weights "
                         "for backward: ~+5 GB per depth on the real model)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    print("priority", N.lower_priority(), flush=True)
    if a.rss_cap_gb:
        N.start_rss_watchdog(a.rss_cap_gb)
    dev = torch.device(a.device)
    rep = {"torch": torch.__version__, "device": str(dev), "threads": a.threads, "gmm_align": moe_grouped.GMM_ALIGN}
    t0 = time.time()
    if not a.skip_tiny:
        rep["tiny"] = tiny_suite(dev)
        print("tiny", json.dumps(rep["tiny"])[:4000], flush=True)
    if not a.tiny_only:
        rep["real"] = real_suite(a, dev)
        print("real", json.dumps(rep["real"]), flush=True)
    must_be_bit = dev.type == "cpu" and moe_grouped.GMM_ALIGN == 0
    oks = []

    def walk(d, path):
        for k, v in d.items():
            if isinstance(v, dict) and "pass" in v:
                ok = v["pass"] and (v.get("bit_identical", v.get("gaps_bit_identical")) if (must_be_bit and k == "grouped") else True)
                oks.append((f"{path}/{k}", bool(ok)))
            elif isinstance(v, dict):
                walk(v, f"{path}/{k}")
    walk({k: v for k, v in rep.items() if isinstance(v, dict)}, "")
    rep["checks"] = len(oks)
    rep["failed"] = [k for k, ok in oks if not ok]
    rep["verdict"] = "PASS" if oks and not rep["failed"] else "CHECK"
    rep["seconds"] = round(time.time() - t0, 1)
    print("verdict", rep["verdict"], rep["checks"], "checks", rep["seconds"], "s", rep["failed"], flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
