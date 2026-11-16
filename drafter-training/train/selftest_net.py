"""Equivalence check: MtpNet (vectorized trainer/evaluator) vs the validated
reference mtp_ref.py, on real captured trunk rows.

  A. float32, greedy feed: MtpNet.unrolled vs MtpDrafter.block_unrolled
     (logits per depth, drafts).
  B. decode numerics: MtpNet.draft_record(head=full) vs the sequential gufo
     cycle (GufoDraftSession: KV-only prefill catch-up, DraftReplay, chained
     MtpForward(hidden_row=-1)) for a few anchors.
  C. teacher feed + gradient smoke: one backward pass through 3 chained depths
     (LoRA on dense + experts) produces finite, non-zero gradients.

CPU only. Usage:
  python selftest_net.py --shard <data-dir>/capture-v1/shards/val/<shard>.mtps [--rows 48 --depth 3]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import mtp_net as N
R = N.R


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--anchors", type=int, nargs="*", default=[5, 17, 30])
    ap.add_argument("--mtp-gguf", default=str(R.DEFAULT_MTP))
    ap.add_argument("--trunk-dir", default=str(R.DEFAULT_TRUNK_DIR))
    ap.add_argument("--frozen-bundle", default=None)
    ap.add_argument("--rss-cap-gb", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-start-gb", type=float, default=24.0)
    ap.add_argument("--min-avail-gb", type=float, default=10.0)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    print('priority', N.lower_priority())
    avail = N.commit_available_gb()
    if avail < a.min_start_gb:
        print(f"not starting: only {avail:.1f} GB memory headroom (< {a.min_start_gb})")
        return 4
    N.start_memory_watchdog(a.min_avail_gb)
    if a.rss_cap_gb:
        N.start_rss_watchdog(a.rss_cap_gb)
    t0 = time.time()
    shard = N.read_gmtp.Shard(a.shard)
    rd = N.truncate(N.read_record(shard, shard.records[a.record]), a.rows)
    target = (N.FrozenTarget.from_bundle(Path(a.frozen_bundle)) if a.frozen_bundle
              else N.FrozenTarget.from_trunk(Path(a.trunk_dir)))
    shared = N._StreamShared(target)                             # mtp_ref reads the same embedding/head
    hidden = rd.hidden_tensor("cpu", torch.float32)
    tokens = torch.from_numpy(rd.tokens)
    report = {"record": rd.rid, "rows": rd.n, "depth": a.depth}

    # A. float32 greedy unrolled
    # streamed reference: experts/head/embedding read from disk per use (low resident memory)
    ref, _ = N.load_streamed(Path(a.mtp_gguf), None, False, target=target)
    net = N.MtpNet(ref, target, N.NetOptions(decode=False, dense_mode="frozen", expert_storage="q8"))
    T = rd.n - a.depth                       # rows with labels at every depth (dense regime)
    ref_logits, _ = ref.block_unrolled(hidden[:T], tokens[1:T + 1], a.depth, feed="greedy")
    got = {}

    def on_depth(j, res):
        lg = N.full_logits(net, net.head_mixed(res))
        got[j] = lg
        return lg.argmax(-1)

    with torch.no_grad():
        net.unrolled(hidden[:T + a.depth], tokens[:T + a.depth], a.depth, "greedy", on_depth)
    rows_a = []
    for j in range(1, a.depth + 1):
        g, r = got[j][:T], ref_logits[j - 1]
        rows_a.append({"depth": j, "rel_rms": rel(g, r), "max_abs": float((g - r).abs().max()),
                       "argmax_equal": float((g.argmax(-1) == r.argmax(-1)).float().mean())})
    report["A_float32_unrolled"] = rows_a
    print("A", json.dumps(rows_a))

    # B. decode numerics, sequential gufo cycles vs parallel chains
    ref.decode = True                                            # same weights, gufo decode numerics
    ref_d = ref
    net_d = N.MtpNet(ref_d, target, N.NetOptions(decode=True, dense_mode="frozen", expert_storage="q8"))
    drafts, gaps, _ = N.draft_record(net_d, rd, a.depth, "full", "cpu", torch.float32)
    rows_b = []
    for p in a.anchors:
        if p + a.depth > rd.n - 1:
            continue
        s = R.GufoDraftSession(ref_d)
        s.prefill(rd.tokens[:p + 1].tolist(), hidden[:p + 1])
        seq = s.cycle(int(rd.tokens[p + 1]), a.depth)
        rows_b.append({"anchor_p": p, "sequential": seq, "parallel": drafts[p].tolist(),
                       "equal": seq == drafts[p].tolist(), "gaps": [round(float(x), 3) for x in gaps[p]]})
    report["B_decode_chains"] = rows_b
    print("B", json.dumps(rows_b))

    # C. gradient smoke (teacher feed, LoRA dense+experts, float32, q8 expert base kept frozen)
    del net, net_d
    ref.decode = False
    opt = N.NetOptions(decode=False, dense_mode="lora", expert_mode="frozen", lora_rank=4, expert_storage="q8")
    net_t = N.MtpNet(ref, target, opt)
    for lo in net_t.lora.values():
        torch.nn.init.normal_(lo.B, std=1e-3)
    hid_s, tok_s = hidden[:24], tokens[:24]
    losses = []

    def on_depth_t(j, res):
        W = target.head_q8.dequant(slice(0, N.DRAFT_VOCAB), dtype=torch.float32)  # labels < 98304 only
        z = torch.nn.functional.linear(net_t.head_mixed(res), W)
        lab = torch.from_numpy(rd.sup_ids[j:j + res.shape[0], 0])
        keep = lab < N.DRAFT_VOCAB
        losses.append(torch.nn.functional.cross_entropy(z[keep], lab[keep]))
        return None

    net_t.unrolled(hid_s, tok_s, a.depth, "teacher", on_depth_t, checkpoint=True)
    loss = torch.stack(losses).sum()
    loss.backward()
    grads = {n: float(p.grad.norm()) for n, p in net_t.trainable_named() if p.grad is not None}
    report["C_grad"] = {"loss": float(loss), "params_with_grad": len(grads),
                        "trainable": len(net_t.trainable_named()),
                        "all_finite": all(np.isfinite(v) for v in grads.values()),
                        "nonzero": sum(v > 0 for v in grads.values()),
                        "sample": dict(list(grads.items())[:6])}
    print("C", json.dumps(report["C_grad"]))
    ok = (all(r["rel_rms"] < 1e-4 and r["argmax_equal"] == 1.0 for r in rows_a)
          and all(r["equal"] for r in rows_b)
          and report["C_grad"]["all_finite"] and report["C_grad"]["nonzero"] > 0)
    report["verdict"] = "PASS" if ok else "CHECK"
    report["seconds"] = round(time.time() - t0, 1)
    print("verdict", report["verdict"], report["seconds"], "s")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
