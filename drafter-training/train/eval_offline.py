"""Offline acceptance of an MTP head on captured GMTP shards (gufo greedy verification).

For every anchor of every selected record, the gufo draft chain is run in
parallel (mtp_net.MtpNet.unrolled, validated against mtp_ref and the
sequential gufo cycle by selftest_net.py). A chain has up to --depth drafts, the
model's own carried-residual recurrence, and draft KV visible only to its own
chain. Draft j is accepted iff drafts 1..j equal the target's greedy tokens
(capture sup ids[0]). See mtp_net.AcceptanceStats for the censoring-aware
estimators: km (primary), lower (fork counted as reject), path (exact on
target-greedy text).

Gate metric: expected tokens per verify call (1 + accepted drafts) at k = 7,
with the production confidence stop and the production small draft head
(98304 prefix + specials + per-request table), in decode numerics.

Heads: small (production; --draft-prefix 98304, or 49152 for the T1 candidate),
full, or context (lean drafter A2: top --context-prefix rows + specials + every
prompt/emitted token, no fallback).
Weights: --layer-quant q8_0|q4_k (A3: draft-only requantized MTP layer; rows
not a multiple of 256 fall back to --q4-fallback, default q5_0) and
--head-quant q8_0|q4_0 (draft-only head copy).

--lean-suite runs these arms on the same records:
  A  q8 layer,   q8 head,   small head     (production baseline)
  B  q4_k layer, q8 head,   small head     (A3 alone)
  C  q8 layer,   q8 head,   context head   (A2 alone)
  D  q4_k layer, q4_0 head, context head   (A2+A3 bundle)
It also measures the A2 context-subset miss rate. Verdict: KILL if D loses
more than --lean-kill (0.12) tokens/call vs A, or the miss rate is above 1.5%.

Per-record draft arrays go to OUT/[arm/]records/*.npz as soon as each record
finishes, so an interrupted run resumes; `--summarize-only` recombines.

Memory: the Halo has a 3 GB / 2-thread rule, so do not run real weights there.
Use the Dell (CPU, <= 20 GB) or the cloud GPU:
  python eval_offline.py --frozen-bundle model/frozen-target.bin --mtp-gguf model/mtp.gguf \
      --shards "data/val/*.mtps" --records 16 --per-shard 8 --stride 7 --max-tokens 1024 \
      --threads 10 --dense-target --lean-suite --out runs/lean-001
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import mtp_net as N
import stage2_data as S2  # [stage2] --window-long-records
R = N.R

ARMS = {
    "A": dict(layer="q8_0", head_quant="q8_0", head="small"),
    "B": dict(layer="q4_k", head_quant="q8_0", head="small"),
    "C": dict(layer="q8_0", head_quant="q8_0", head="context"),
    "D": dict(layer="q4_k", head_quant="q4_0", head="context"),
}


def build_eval_net(a, layer="q8_0", head_quant="q8_0") -> N.MtpNet:
    target = (N.FrozenTarget.from_bundle(Path(a.frozen_bundle)) if a.frozen_bundle
              else N.FrozenTarget.from_trunk(Path(a.trunk_dir)))
    dev = torch.device(a.device)
    decode = a.numerics == "decode"
    if dev.type == "cuda" or a.dense_target:
        target.to_dense(dev, torch.float32 if decode or dev.type == "cpu" else torch.bfloat16,
                        embed=dev.type == "cuda")
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        o = N.opts_from_dict(ck["net_options"])
        o.decode, o.qat, o.param_dtype, o.expert_storage = decode, False, torch.float32, "dense"
        o.expert_dtype = torch.float32 if decode else torch.bfloat16
        net = N.build_net(Path(a.mtp_gguf), target, o, dev)
        net.load_trainable(ck["trainable"])
    elif a.streamed:
        ref, _ = N.load_streamed(Path(a.mtp_gguf), None, decode, target=target)
        o = N.NetOptions(decode=decode, dense_mode="frozen", expert_mode="frozen", expert_storage="q8")
        net = N.MtpNet(ref, target, o, dev)
    else:
        storage = "q8" if dev.type == "cpu" else "dense"
        o = N.NetOptions(decode=decode, dense_mode="frozen", expert_mode="frozen", expert_storage=storage,
                         expert_dtype=torch.float32 if decode else torch.bfloat16)
        net = N.build_net(Path(a.mtp_gguf), target, o, dev)
    done = N.apply_lean_weights(net, layer, head_quant, a.q4_fallback)
    net.eval()
    net.lean_applied = done
    return net


def select_records(shards: list[Path], count: int, per_shard: int, stride: int):
    out = []
    for p in shards:
        s = N.read_gmtp.Shard(p)
        recs = s.records[::stride][:per_shard] if per_shard else s.records[::stride]
        out += [(p, r.ordinal, r.metadata.get("id", f"{p.name}#{r.ordinal}")) for r in recs]
        if count and len(out) >= count:
            break
    return out[:count] if count else out


def summarize(out_dir: Path, depth: int, anchor_kinds: list[str] | None = None) -> dict:
    """anchor_kinds: keep only chains whose anchor token t_{p+1} has one of these kinds
    (e.g. ["response"] = target-generated positions of on-policy records)."""
    codes = [N.KIND_CODES[k] for k in anchor_kinds] if anchor_kinds else None
    stats = N.AcceptanceStats(depth)
    sst = None                                        # [stage2/R3-14] sampled-path acceptance, when recorded
    n_rec = toks = miss = labels = 0
    for f in sorted((out_dir / "records").glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        n = len(z["tokens"])
        rd = N.RecordData(str(z["rid"]), z["tokens"], np.zeros((n, 0), np.uint16), z["sup_ids0"][:, None],
                          np.zeros((n, 1), np.float32), z["supervised"], z["kinds"], {})
        drafts = z["drafts"][:, :depth].copy()
        fits = z["fits"].copy()
        keep = None
        if codes is not None:
            keep = np.isin(z["kinds"][np.minimum(np.arange(n - 1) + 1, n - 1)], codes)
            drafts[~keep] = -1
            fits &= keep
        stats.add(rd, drafts, z["gaps"][:, :depth].astype(np.float32), fits)
        temps = sorted(float(k[len("alpha_lo_T"):]) for k in z.files if k.startswith("alpha_lo_T"))
        if temps:
            sst = sst or S2.SampledStats(depth, temps)
            sst.add(rd, {T: (z[f"alpha_lo_T{T:g}"][:, :depth], z[f"alpha_hi_T{T:g}"][:, :depth]) for T in temps}, keep)
        n_rec += 1
        toks += n
        if "ctx_miss" in z:
            miss += int(z["ctx_miss"][0])
            labels += int(z["ctx_miss"][1])
    rep = stats.report()
    rep.update(records=n_rec, tokens=toks)
    if sst is not None:
        rep["sampled"] = sst.report()
    if labels:
        rep["context_subset_miss_rate"] = miss / labels
    return rep


def run_arm(a, out: Path, todo, layer, head_quant, head, t0) -> dict:  # noqa: C901
    (out / "records").mkdir(parents=True, exist_ok=True)
    net = None
    dtype = torch.float32 if (a.numerics == "decode" or a.device == "cpu") else torch.bfloat16
    shards: dict = {}
    for i, (path, ordinal, rid) in enumerate(todo):
        dest = out / "records" / f"{i:05d}-{rid.replace('/', '_')}.npz"
        if dest.exists():
            continue
        if a.time_limit_min and (time.time() - t0) / 60 > a.time_limit_min:
            print("TIME_LIMIT reached; stopping before record", i, flush=True)
            break
        if net is None:
            net = build_eval_net(a, layer, head_quant)
            (out / "lean.json").write_text(json.dumps({"layer": layer, "head_quant": head_quant, "head": head,
                                                       "tensors": net.lean_applied}, indent=1))
            print(f"[{out.name}] model ready {time.time() - t0:.0f}s lean={layer}/{head_quant}/{head}", flush=True)
        shard = shards.setdefault(path, N.read_gmtp.Shard(path))
        rd0 = N.read_record(shard, shard.records[ordinal])
        # [stage2] --window-long-records: one npz per response window (window 0 keeps the stage-1 name and is
        # written last, so an existing dest still means "record complete"); off = stage-1 truncation
        parts = S2.windowed_records(rd0, a.max_tokens, a.window_long_records) if a.max_tokens else [rd0]
        for wi in reversed(range(len(parts))):
            rd = parts[wi]
            dest_w = dest if wi == 0 else dest.with_name(dest.name[:-4] + f"-w{wi:03d}.npz")
            t1 = time.time()
            drafts, gaps, fits = N.draft_record(net, rd, a.depth, head, a.device, dtype,
                                                prefix=a.draft_prefix, context_prefix=a.context_prefix)
            cm = N.context_subset_miss(rd.tokens, rd.greedy, rd.supervised, net.target.specials, a.context_prefix)
            extra = {}
            if a.sampled_temps:                       # [stage2/R3-14] sampled-path acceptance bounds per row/depth
                for T, (lo, hi) in S2.sampled_record(net, rd, a.depth, a.device, dtype, tuple(a.sampled_temps)).items():
                    extra[f"alpha_lo_T{T:g}"], extra[f"alpha_hi_T{T:g}"] = lo, hi
            tmp = dest_w.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, rid=np.array(rid if wi == 0 else rd.rid), tokens=rd.tokens.astype(np.int64),
                                sup_ids0=rd.sup_ids[:, 0], supervised=rd.supervised, kinds=rd.kinds,
                                drafts=drafts.astype(np.int64), gaps=gaps.astype(np.float32), fits=fits,
                                ctx_miss=np.array(cm, np.int64), **extra)
            tmp.replace(dest_w)
            print(f"[{out.name}] [{i + 1}/{len(todo)}] {rid} n={rd.n}{f' w{wi}' if len(parts) > 1 else ''} "
                  f"{time.time() - t1:.1f}s", flush=True)
    del net
    rep = summarize(out, a.depth, a.anchor_kinds)
    rep["arm"] = {"layer": layer, "head_quant": head_quant, "head": head}
    (out / "summary.json").write_text(json.dumps(rep, indent=1))
    print(f"[{out.name}]", N.headline(rep, a.depth), flush=True)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", default=[])
    ap.add_argument("--records", type=int, default=4, help="total records (0 = all)")
    ap.add_argument("--per-shard", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--sampled-temps", type=float, nargs="*", default=[],
                    help="[stage2/R3-14] also record sampled-path acceptance sum min(p_T, q_T) per row/depth (T <= 1); "
                         "summary.json gains 'sampled' (E_lo/E_hi overall and per kind). One extra teacher-fed pass")
    ap.add_argument("--window-long-records", action="store_true",
                    help="[stage2] records > --max-tokens with response segments -> response windows (stage2_data.py)")
    ap.add_argument("--depth", type=int, default=N.MAX_DRAFTS)
    ap.add_argument("--head", choices=["small", "full", "context"], default="small")
    ap.add_argument("--draft-prefix", type=int, default=N.DRAFT_VOCAB)
    ap.add_argument("--context-prefix", type=int, default=16384)
    ap.add_argument("--layer-quant", choices=["q8_0", "q4_k", "q5_0", "q4_0"], default="q8_0")
    ap.add_argument("--head-quant", choices=["q8_0", "q4_0", "q5_0"], default="q8_0")
    ap.add_argument("--q4-fallback", default="q5_0")
    ap.add_argument("--lean-suite", action="store_true")
    ap.add_argument("--lean-arms", default="A,B,C,D")
    ap.add_argument("--lean-kill", type=float, default=0.12)
    ap.add_argument("--numerics", choices=["decode", "float"], default="decode")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mtp-gguf", default=str(R.DEFAULT_MTP))
    ap.add_argument("--trunk-dir", default=str(R.DEFAULT_TRUNK_DIR))
    ap.add_argument("--frozen-bundle", default=None)
    ap.add_argument("--dense-target", action="store_true", help="CPU: cache f32 embedding+head (5 GB) for speed")
    ap.add_argument("--streamed", action="store_true", help="CPU low-memory: experts read from disk per use")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rss-cap-gb", type=float, default=0.0)
    ap.add_argument("--time-limit-min", type=float, default=0)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--anchor-kinds", nargs="*", default=None,
                    help="restrict anchors, e.g. --anchor-kinds response (on-policy generated positions)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [x.strip() for x in a.lean_arms.split(",")] if a.lean_suite else [None]
    if a.summarize_only:
        for arm in arms:
            d = out / arm if arm else out
            rep = summarize(d, a.depth, a.anchor_kinds)
            (d / "summary.json").write_text(json.dumps(rep, indent=1))
            print(arm or "", N.headline(rep, a.depth))
        return 0
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
        print("priority", N.lower_priority(), flush=True)
        if a.rss_cap_gb:
            N.start_rss_watchdog(a.rss_cap_gb)
    t0 = time.time()
    todo = select_records(N.resolve_shards(a.shards), a.records, a.per_shard, a.stride)
    (out / "run.json").write_text(json.dumps({"records": [r[2] for r in todo], "args": vars(a)}, indent=1))
    if not a.lean_suite:
        rep = run_arm(a, out, todo, a.layer_quant, a.head_quant, a.head, t0)
        rep["elapsed_s"] = round(time.time() - t0, 1)
        (out / "summary.json").write_text(json.dumps(rep, indent=1))
        return 0
    reps = {}
    for arm in arms:
        cfg = ARMS[arm]
        reps[arm] = run_arm(a, out / arm, todo, cfg["layer"], cfg["head_quant"], cfg["head"], t0)
    D = a.depth
    km = {k: r["stop"][f"k{D}"]["km"] for k, r in reps.items()}
    lo = {k: r["stop"][f"k{D}"]["lower"] for k, r in reps.items()}
    miss = reps.get("A", next(iter(reps.values()))).get("context_subset_miss_rate")
    verdict = {"tokens_per_call_km": km, "tokens_per_call_lower": lo,
               "delta_vs_A_km": {k: round(v - km["A"], 4) for k, v in km.items()} if "A" in km else None,
               "context_subset_miss_rate": miss}
    if "A" in km and "D" in km:
        loss = km["A"] - km["D"]
        kill = loss > a.lean_kill or (miss is not None and miss > 0.015)
        verdict["bundle_loss_km"] = round(loss, 4)
        verdict["verdict"] = ("KILL" if kill else "ALIVE") + (
            f" (bundle loss {loss:.3f} vs limit {a.lean_kill}; miss {miss:.4f} vs 0.015)")
    if "A" in km and "B" in km:
        verdict["A3_alone_loss_km"] = round(km["A"] - km["B"], 4)
        verdict["A3_alone_note"] = "ideas doc kill line for A3 alone: > 0.08 tokens/call"
    verdict["records"] = len(todo)
    verdict["elapsed_s"] = round(time.time() - t0, 1)
    (out / "lean-verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
