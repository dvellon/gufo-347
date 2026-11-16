"""Stage-2 trainer additions on the tiny synthetic MTP (CPU, < 1 GB, ~1-2 min, 2 threads).

  1. stage-1 reproducibility: with no stage-2 flag, the record order and the loss/grad-norm
     trajectory of train_mtp.py equal the stage-1 trainer's (--stage1-train: path to the
     unmodified stage-1 train_mtp.py, e.g. a git/scratch copy) - bit for bit.
  2. --moe-impl grouped: same trajectory as the loop (bit-identical on CPU), and bmm close;
     --no-head-checkpoint: same trajectory as the checkpointed head loss.
  3. response windows: every response row is trained exactly once, windows <= max_tokens,
     context before each response piece, records without segments = stage-1 truncation.
  4. --shard-weights: deterministic, expected repeat counts, resume refused when changed.
  5. end to end: --window-long-records + --shard-weights + --eval-set ...@response (and a
     two-glob set) + --early-stop (best.pt / best.json, early-stop stop reason) + --resume auto;
     --init-trainable warm start.

  python selftest_stage2.py [--stage1-train <dir with the stage-1 train_mtp.py>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import mtp_net as N
import stage2_data as S2
import tiny
import train_mtp


def log(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "train-log.jsonl").read_text().splitlines()]


def steps(run: Path) -> list[tuple]:
    return [(r["step"], r["loss"], r["grad_norm"]) for r in log(run) if r.get("event") == "step"]


def write_response_shards(out: Path, n_records: int, n_tokens: int, seed: int, split_ord: int) -> Path:
    """tiny GMTP shard whose records are on-policy-like: target-responses segments."""
    G = N.read_gmtp
    rnd = random.Random(seed)
    G.ST_SHAPE.update(hidden=32, hc_count=4, vocab=1000, topk=32, max_context=4096)
    recs = []
    for i in range(n_records):
        segs, p = [], 0
        while p < n_tokens - 20:
            pr = rnd.randrange(10, 60)
            rs = rnd.randrange(15, 90)
            segs.append({"start": p, "response_start": min(n_tokens, p + pr), "end": min(n_tokens, p + pr + rs)})
            p = min(n_tokens, p + pr + rs)
        recs.append({"id": f"op-{split_ord}-{i}", "source_id": f"tiny/op{i}", "supervise_from": 0,
                     "tokens": [rnd.randrange(1000) for _ in range(n_tokens)],
                     "metadata": {"kind": "target-responses", "segments": segs}})
    G.ST_RECORDS[:] = recs
    d = out
    d.mkdir(parents=True, exist_ok=True)
    return G._write_shard(d, split_ord, False)


def check_windows() -> dict:
    rnd = random.Random(1)
    worst = {"cases": 0, "ok": True}
    for case in range(300):
        n = rnd.randrange(50, 3000)
        maxt = rnd.choice([64, 128, 512, 2048])
        segs, p = [], 0
        while p < n - 5:
            pr, rs = rnd.randrange(0, 400), rnd.randrange(1, 1500)
            segs.append({"start": p, "response_start": min(n, p + pr), "end": min(n, p + pr + rs)})
            p = min(n, p + pr + rs)
        meta = {"metadata": {"kind": "target-responses", "segments": segs}}
        wins = S2.record_windows(meta, n, maxt)
        trained = np.zeros(n, np.int64)
        for s, e, t in wins:
            assert 0 <= s <= t <= e <= n and e - s <= maxt, (s, e, t, maxt)
            trained[t:e] += 1
        resp = np.zeros(n, bool)
        for r0, e0 in S2.response_spans(meta, n):
            resp[r0:e0] = True
        if n <= maxt:
            ok = wins == [(0, n, 0)]
        else:
            ok = bool((trained[resp] == 1).all() and (trained <= 1).all())
            for s, e, t in wins:                       # context before the first new response row
                first = next((q for q in range(t, e) if resp[q]), None)
                if first is not None and first >= S2.MIN_CONTEXT and first - s < min(S2.MIN_CONTEXT, maxt // 2):
                    ok = False
        worst["cases"] += 1
        worst["ok"] = bool(worst["ok"] and ok)
    # no segments -> stage-1 truncation
    worst["no_segments_truncates"] = S2.record_windows({"metadata": {"kind": "code"}}, 5000, 2048) == [(0, 2048, 0)]
    worst["ok"] = bool(worst["ok"] and worst["no_segments_truncates"])
    return worst


def check_alpha() -> dict:
    """R3-14 metric: alpha_rows bounds vs brute force sum_v min(p_T, q_T) on full small vocabularies."""
    import torch
    g = torch.Generator().manual_seed(0)
    out = {"exact_when_q_eq_p": True, "bounds_hold": True, "cases": 0, "max_gap_T1": 0.0}
    for case in range(200):
        V = int(torch.randint(33, 300, (1,), generator=g))
        K = 32
        logp = torch.log_softmax(torch.randn(4, V, generator=g) * float(torch.rand(1, generator=g) * 4 + 0.2), -1)
        same = case % 4 == 0
        z = (logp * 1.0) if same else torch.randn(4, V, generator=g) * 2
        top = torch.topk(logp, K, -1)
        ids, lp = top.indices, top.values
        valid = torch.ones_like(ids, dtype=torch.bool)
        for T in (1.0, 0.7):
            lse = torch.logsumexp(z / T, -1)
            lo, hi = S2.alpha_rows(z.gather(-1, ids), lse, lp, valid, T)
            pT = torch.softmax(logp / T, -1)
            qT = torch.softmax(z / T, -1)
            exact = torch.minimum(pT, qT).sum(-1)
            out["bounds_hold"] &= bool(((lo <= exact + 1e-5) & (exact <= hi + 1e-5)).all())
            if T == 1.0:
                out["max_gap_T1"] = max(out["max_gap_T1"], float((hi - lo).max()))
            if same and V == K:
                out["exact_when_q_eq_p"] &= bool(torch.allclose(lo, torch.ones(4), atol=1e-5))
        out["cases"] += 1
    # full-vocab top-K (V == K): exact = 1 when q == p, both temperatures
    logp = torch.log_softmax(torch.randn(3, 32, generator=g), -1)
    for T in (1.0, 0.7):
        lo, hi = S2.alpha_rows(logp.clone(), torch.logsumexp(logp / T, -1), logp, torch.ones(3, 32, dtype=torch.bool), T)
        out["exact_when_q_eq_p"] &= bool(torch.allclose(lo, torch.ones(3), atol=1e-5) and torch.allclose(hi, torch.ones(3), atol=1e-5))
    out["ok"] = bool(out["bounds_hold"] and out["exact_when_q_eq_p"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-train", default=None, help="directory holding the unmodified stage-1 train_mtp.py")
    a = ap.parse_args()
    N.lower_priority()
    tmp = Path(tempfile.mkdtemp(prefix="mtp-s2-"))
    res = {}
    tr, va = tiny.write_shards(tmp / "data", n_records=8, n_tokens=48)
    common = ["--device", "cpu", "--tiny-synthetic", "--threads", "2", "--shards", str(tr), "--val-shards", str(va),
              "--depth", "4", "--grad-accum", "2", "--ckpt-every", "2", "--eval-every", "2", "--eval-records", "2",
              "--log-every", "1", "--warmup", "1", "--max-steps", "4", "--resume", "none", "--expert-mode", "lora"]

    # 1. stage-1 reproducibility
    train_mtp.main(common + ["--out", str(tmp / "new-loop")])
    if a.stage1_train:
        spec = importlib.util.spec_from_file_location("train_mtp_stage1", Path(a.stage1_train) / "train_mtp.py")
        T1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(T1)
        T1.main(common + ["--out", str(tmp / "stage1")])
        s_old = T1.RecordStream([str(tr)], 1234, 0, 48)
        s_new = train_mtp.RecordStream([str(tr)], 1234, 0, 48)
        res["stage1_order_equal"] = all(s_old.block_order(e, 0) == s_new.block_order(e, 0) for e in range(3))
        res["stage1_trajectory_equal"] = steps(tmp / "stage1") == steps(tmp / "new-loop")
    # 2. grouped / bmm vs loop, end to end through optimizer steps
    train_mtp.main(common + ["--out", str(tmp / "new-grouped"), "--moe-impl", "grouped"])
    train_mtp.main(common + ["--out", str(tmp / "new-bmm"), "--moe-impl", "bmm"])
    res["grouped_trajectory_bit_identical"] = steps(tmp / "new-grouped") == steps(tmp / "new-loop")
    b, l_ = steps(tmp / "new-bmm"), steps(tmp / "new-loop")
    res["bmm_trajectory_max_rel"] = max(abs(x[1] - y[1]) / abs(y[1]) for x, y in zip(b, l_))
    train_mtp.main(common + ["--out", str(tmp / "new-nohc"), "--no-head-checkpoint"])
    res["no_head_checkpoint_bit_identical"] = steps(tmp / "new-nohc") == steps(tmp / "new-loop")
    # 3. windows
    res["windows"] = check_windows()
    # 4. mixture weights
    op = write_response_shards(tmp / "op", n_records=6, n_tokens=300, seed=4, split_ord=31)
    pats = [str(tr), str(op)]
    w = f"*shard-000031*=2.5,{tr.name}=0.5"
    s1 = train_mtp.RecordStream(pats, 7, 0, 64, shard_weights=w)
    s2 = train_mtp.RecordStream(pats, 7, 0, 64, shard_weights=w)
    o1 = [s1.block_order(e, 0) for e in range(40)]
    res["weights_deterministic"] = o1 == [s2.block_order(e, 0) for e in range(40)]
    mean_op = np.mean([sum(1 for x in o if x[0] == op.name) for o in o1]) / 6
    mean_tr = np.mean([sum(1 for x in o if x[0] == tr.name) for o in o1]) / 8
    res["weights_mean_repeat"] = [round(float(mean_op), 3), round(float(mean_tr), 3)]
    res["weights_ok"] = bool(abs(mean_op - 2.5) < 0.2 and abs(mean_tr - 0.5) < 0.12)
    sw = train_mtp.RecordStream(pats, 7, 0, 64, window_long=True)
    nwin = sum(len(sw.windows[op.name][r]) for r in range(6))
    res["window_entries"] = {"records": 14, "entries_per_epoch": sw.entries_per_epoch, "op_windows": nwin}
    res["window_entries_ok"] = bool(sw.entries_per_epoch == 8 + nwin and nwin > 6)
    # 5. end to end
    run = tmp / "s2"
    e2e = ["--device", "cpu", "--tiny-synthetic", "--threads", "2", "--shards", str(tr), str(op),
           "--depth", "3", "--grad-accum", "2", "--ckpt-every", "2", "--eval-every", "1", "--log-every", "1",
           "--warmup", "1", "--max-tokens", "64", "--window-long-records", "--shard-weights", w,
           "--eval-set", f"onpolicy={op}@response", "--eval-set", f"trainprobe={tr}|{op}", "--eval-set-records", "4",
           "--early-stop", "onpolicy", "--early-stop-patience", "2", "--early-stop-min-delta", "5.0",
           "--moe-impl", "grouped", "--out", str(run)]
    train_mtp.main(e2e + ["--max-steps", "6", "--resume", "none"])
    lg = log(run)
    ends = [r for r in lg if r.get("event") == "end"]
    res["e2e_end_reason"] = ends[-1]["reason"] if ends else None
    res["e2e_early_stop"] = bool(ends) and ends[-1]["reason"].startswith("early stop")
    res["e2e_best"] = json.loads((run / "ckpt" / "best.json").read_text()) if (run / "ckpt" / "best.json").exists() else None
    res["e2e_eval_sets"] = sorted({r["set"] for r in lg if r.get("event") == "eval-set"})
    res["e2e_overfit_gap_logged"] = any(r.get("event") == "overfit-gap" for r in lg)
    res["e2e_window_rows_le_max"] = all(d["rows"] <= 64 for r in lg if r.get("event") == "step" for d in r["depths"])
    # resume with changed weights is refused; with the same flags it continues
    try:
        train_mtp.main([x if x != w else f"*shard-000031*=3" for x in e2e] + ["--max-steps", "8", "--resume", "auto",
                                                                            "--early-stop-patience", "0"])
        res["resume_changed_weights_refused"] = False
    except SystemExit as exc:
        res["resume_changed_weights_refused"] = "differ" in str(exc)
    n_before = len(log(run))
    train_mtp.main(e2e + ["--max-steps", "8", "--resume", "auto", "--early-stop-patience", "0"])
    l2 = log(run)[n_before:]
    res["resume_same_ok"] = any(r.get("event") == "resumed" for r in l2) and [r for r in l2 if r.get("event") == "end"][-1]["step"] == 8
    ok = (res.get("stage1_order_equal", True) and res.get("stage1_trajectory_equal", True)
          and res["grouped_trajectory_bit_identical"] and res["bmm_trajectory_max_rel"] < 1e-4
          and res["no_head_checkpoint_bit_identical"]
          and res["windows"]["ok"] and res["weights_deterministic"] and res["weights_ok"] and res["window_entries_ok"]
          and res["e2e_early_stop"] and res["e2e_best"] is not None and res["e2e_eval_sets"] == ["onpolicy", "trainprobe"]
          and res["e2e_overfit_gap_logged"] and res["e2e_window_rows_le_max"]
          and res["resume_changed_weights_refused"] and res["resume_same_ok"])
    # R3-14: sampled-acceptance bounds, and checkpoint selection by the sampled metric
    res["alpha"] = check_alpha()
    run_s = tmp / "s2-sampled"
    i_out = e2e.index("--out")
    sargs = e2e[:i_out] + e2e[i_out + 2:] + ["--out", str(run_s), "--eval-sampled-temps", "1.0", "0.7",
                                                  "--early-stop-metric", "sampled@1", "--max-steps", "3",
                                                  "--resume", "none", "--early-stop-min-delta", "0.0"]
    sargs[sargs.index("--early-stop-patience") + 1] = "0"
    train_mtp.main(sargs)
    ls = log(run_s)
    evs = [r for r in ls if r.get("event") == "eval-set" and r.get("set") == "onpolicy"]
    bj = json.loads((run_s / "ckpt" / "best.json").read_text()) if (run_s / "ckpt" / "best.json").exists() else {}
    res["sampled_e2e"] = {"evals": len(evs), "best_metric": bj.get("metric"),
                          "E_T1_lo_step0": evs[0].get("sampled@1") if evs else None,
                          "per_kind_keys": sorted(evs[0]["sampled"]["T1"]["by_kind"]) if evs else []}
    res["sampled_ok"] = bool(evs and bj.get("metric") == "sampled@1" and "sampled@0.7" in evs[0]
                             and 1.0 <= evs[0]["sampled@1"] <= 8.0 and "response" in res["sampled_e2e"]["per_kind_keys"])
    ok = ok and res["alpha"]["ok"] and res["sampled_ok"]
    # R4-28: head-aware fine-tune against a Q4_0 draft head (finite, and different from the Q8_0 run)
    train_mtp.main(common + ["--out", str(tmp / "q4head"), "--draft-head-quant", "q4_0"])
    q4 = steps(tmp / "q4head")
    res["q4_head_ok"] = bool(q4 and all(abs(x[1]) < 1e6 and x[1] == x[1] for x in q4) and q4 != steps(tmp / "new-loop"))
    ok = ok and res["q4_head_ok"]
    # warm start from another run's trainable weights (fresh optimizer / cursor)
    ck = sorted((tmp / "new-loop" / "ckpt").glob("step-*.pt"))[-1]
    train_mtp.main(common + ["--out", str(tmp / "warm"), "--init-trainable", str(ck)])
    res["init_trainable_ok"] = any(r.get("event") == "init-trainable" for r in log(tmp / "warm")) and         steps(tmp / "warm")[0][1] != steps(tmp / "new-loop")[0][1]
    ok = ok and res["init_trainable_ok"]
    res["stage1_checked"] = bool(a.stage1_train)
    res["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(res, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
