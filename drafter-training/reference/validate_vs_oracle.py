"""Cross-checks mtp_ref.py against gufo's scalar MTP oracle (CPU only).

  python validate_vs_oracle.py make-inputs out/inputs.bin [--rows 6 --drafts 4]
  cpp/build/mtp_ref_dump.exe <trunk-00001.gguf> <mtp.gguf> out/inputs.bin out/oracle-decode.bin decode
  python validate_vs_oracle.py compare out/inputs.bin out/oracle-decode.bin --mode decode
  python validate_vs_oracle.py unrolled --rows 6 --depth 3     # parallel vs sequential chains

The oracle is ReferenceModel::MtpStep (src/models/qwen38_flash_next/reference.cpp),
the same code qwen38_flash_next_gpu_probe --mtp-audit holds the HIP predictor to
(relative RMS: fusion/attention 2e-3, carry/head-mixer 5e-5).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_ref as M  # noqa: E402

STAGES = ["norm", "fused", "attention", "hidden", "head", "ffn_input", "ffn_output"]
TOKENS = [785, 6722, 315, 9625, 374, 12095, 13, 576, 3170, 279, 12884, 6303, 30, 2014, 1124, 525]


def synth_hidden(shared: M.SharedTargetTensors, tokens, seed=1234) -> torch.Tensor:
    """Deterministic stand-in for trunk final residuals: per-stream scaled
    copies of the token embedding (the trunk's stream initialisation) plus
    structured noise. Only the cross-implementation agreement is measured."""
    g = torch.Generator().manual_seed(seed)
    e = shared.embed(tokens)
    rows = []
    for i in range(len(tokens)):
        streams = []
        for s in range(4):
            noise = torch.randn(e.shape[1], generator=g) * e[i].abs().mean() * (2.0 + s)
            streams.append(e[i] * (8.0 * (1 + 0.5 * s)) + noise)
        rows.append(torch.cat(streams))
    return torch.stack(rows)


def make_inputs(path: Path, rows: int, drafts: int) -> None:
    shared = M.SharedTargetTensors(cache_head=False)
    toks = (TOKENS * ((rows + 1) // len(TOKENS) + 1))[:rows + 1]
    hidden = synth_hidden(shared, toks[:rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", rows, drafts))
        f.write(np.asarray(toks[1:rows + 1], dtype="<i4").tobytes())
        f.write(hidden.numpy().astype("<f4").tobytes())
    print(f"wrote {path} rows={rows} drafts={drafts}")


def read_inputs(path: Path):
    b = path.read_bytes()
    rows, drafts = struct.unpack_from("<ii", b)
    toks = np.frombuffer(b, "<i4", rows, 8).tolist()
    hidden = torch.from_numpy(np.frombuffer(b, "<f4", offset=8 + 4 * rows).copy()).view(rows, -1)
    return rows, drafts, toks, hidden


def read_oracle(path: Path):
    b = path.read_bytes()
    steps, D, H, V = struct.unpack_from("<iiii", b)
    off = 16
    out = []
    for _ in range(steps):
        token, has, argmax = struct.unpack_from("<iii", b, off)
        off += 12
        rec = {"token": token, "argmax": argmax}
        for name, n in zip(STAGES, [D, D, D, D, H, H, H]):
            rec[name] = torch.from_numpy(np.frombuffer(b, "<f4", n, off).copy())
            off += 4 * n
        if has:
            rec["logits"] = torch.from_numpy(np.frombuffer(b, "<f4", V, off).copy())
            off += 4 * V
        out.append(rec)
    assert off == len(b)
    return out


def rel(a: torch.Tensor, b: torch.Tensor):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a.detach(), b.detach()
    return float((a - b).norm() / b.norm().clamp_min(1e-30)), float((a - b).abs().max())


def compare(inputs: Path, oracle: Path, mode: str, report: Path | None, isolated: bool = False) -> int:
    """Free-running (default): both chains evolve independently from the same
    inputs, following the oracle's draft tokens. --isolated: every stage gets
    the oracle's previous-stage output as input (MtpStep stage_inputs), so
    each number measures one operator group, as the GPU audit does."""
    rows, drafts, toks, hidden = read_inputs(inputs)
    ref = read_oracle(oracle)
    model = M.MtpDrafter(decode=(mode == "decode"))
    state = M.MtpState()
    results, token, ok = [], None, True
    for step, r in enumerate(ref):
        catchup = step < rows
        token = toks[step] if catchup else token
        assert token == r["token"], (step, token, r["token"])
        t0 = time.time()
        if catchup:
            h_in = hidden[step:step + 1]
        else:
            h_in = ref[step - 1]["hidden"][None] if isolated else None
        si = {k: r[k] for k in ("fused", "attention", "ffn_input", "ffn_output", "hidden", "head")} \
            if isolated else None
        out = model.mtp_forward(state, [token], h_in, head=step >= rows - 1, trace=True, stage_inputs=si)
        tr = {k: v[-1] for k, v in out.trace.items()}
        if "head" not in tr:
            src = r["hidden"][None] if isolated else out.trace["hidden"][-1:]
            tr["head"] = model.head_mixed(src)[0]
        rec = {"step": step, "catchup": catchup, "token": token, "seconds": round(time.time() - t0, 2)}
        for s in STAGES:
            rec[s] = rel(tr[s], r[s])
        if out.logits is not None:
            lg, rl = out.logits, r["logits"]
            top_py = torch.topk(lg, 64).indices.tolist()
            top_c = torch.topk(rl, 64).indices.tolist()
            rec.update(argmax_py=out.token, argmax_oracle=r["argmax"],
                       argmax_equal=out.token == r["argmax"],
                       logits=rel(lg, rl), top10_overlap=len(set(top_py[:10]) & set(top_c[:10])),
                       top64_overlap=len(set(top_py) & set(top_c)),
                       top1_margin_oracle=float(torch.topk(rl, 2).values[0] - torch.topk(rl, 2).values[1]))
            ok &= rec["argmax_equal"]
            token = r["argmax"]  # follow the oracle chain so stages stay comparable
        results.append(rec)
        print(json.dumps(rec))
    summary = {"mode": mode, "isolated": isolated, "rows": rows, "drafts": drafts,
               "all_argmax_equal": bool(ok),
               "worst_rel": {s: max(x[s][0] for x in results) for s in STAGES},
               "worst_logit_max_abs": max((x["logits"][1] for x in results if "logits" in x), default=None),
               "worst_logit_rel": max((x["logits"][0] for x in results if "logits" in x), default=None)}
    print(json.dumps(summary, indent=1))
    if report:
        report.write_text(json.dumps({"summary": summary, "steps": results}, indent=1))
    return 0 if ok else 1


def unrolled(rows: int, depth: int, mode: str, report: Path | None) -> int:
    """block_unrolled (all anchors, one pass per depth) vs greedy_chain from a
    fresh state per anchor: same drafts and logits."""
    model = M.MtpDrafter(decode=(mode == "decode"))
    toks = TOKENS[:rows + 1]
    hidden = synth_hidden(model.shared, toks[:rows])
    t0 = time.time()
    logits, _ = model.block_unrolled(hidden, torch.tensor(toks[1:rows + 1]), depth, feed="greedy")
    t_par = time.time() - t0
    worst, same = 0.0, True
    recs = []
    for p in range(rows):
        st = M.MtpState()
        drafts, outs = M.greedy_chain(model, st, toks[1:p + 2], hidden[:p + 1], depth)
        par = [int(logits[j][p].argmax()) for j in range(depth)]
        errs = [rel(logits[j][p], outs[j].logits) for j in range(depth)]
        worst = max(worst, max(e[0] for e in errs))
        same &= par == drafts
        recs.append({"anchor_row": p, "sequential": drafts, "parallel": par, "rel": errs})
        print(json.dumps(recs[-1]))
    summary = {"mode": mode, "rows": rows, "depth": depth, "drafts_equal": bool(same),
               "worst_logit_rel": worst, "parallel_seconds": round(t_par, 1)}
    print(json.dumps(summary, indent=1))
    if report:
        report.write_text(json.dumps({"summary": summary, "anchors": recs}, indent=1))
    return 0 if same else 1


def long_context(inputs: Path, oracle: Path, mode: str, report: Path | None) -> int:
    """One batched catch-up over every row (crossing the QSA sparse boundary at
    n_kv > idx_top_k + ratio - 1), then the recursive drafts. Also reruns with
    the indexer disabled to show the sparse rows actually discriminate."""
    rows, drafts, toks, hidden = read_inputs(inputs)
    ref = read_oracle(oracle)
    summary = {"mode": mode, "rows": rows, "drafts": drafts}
    for variant in ("qsa", "dense-control"):
        model = M.MtpDrafter(decode=(mode == "decode"))
        if variant == "dense-control":
            model.cfg.idx_top_k = 1 << 30
        c = model.cfg
        state = M.MtpState()
        t0 = time.time()
        out = model.mtp_forward(state, toks[:rows], hidden[:rows], head=True, trace=True)
        secs = time.time() - t0
        boundary = c.idx_top_k + c.ratio - 1 if variant == "qsa" else M.MtpConfig().idx_top_k + 3
        stats = {"dense": {}, "sparse": {}}
        for stage in ("attention", "hidden"):
            errs = [rel(out.trace[stage][p], ref[p][stage])[0] for p in range(rows)]
            stats["dense"][stage] = max(errs[:boundary])
            stats["sparse"][stage] = max(errs[boundary:]) if rows > boundary else None
        chain = [out.token]
        lg_err = [rel(out.logits, ref[rows - 1]["logits"])]
        for step in range(rows, rows + drafts - 1):
            o = model.mtp_forward(state, [ref[step - 1]["argmax"]], None)
            chain.append(o.token)
            lg_err.append(rel(o.logits, ref[step]["logits"]))
        oracle_chain = [ref[s]["argmax"] for s in range(rows - 1, rows + drafts - 1)]
        summary[variant] = {"catchup_seconds": round(secs, 1), "first_sparse_row": boundary,
                            "worst_rel_by_regime": stats, "chain_py": chain, "chain_oracle": oracle_chain,
                            "chain_equal": chain == oracle_chain, "logit_rel_maxabs": lg_err}
        print(json.dumps({variant: summary[variant]}))
    print(json.dumps(summary, indent=1))
    if report:
        report.write_text(json.dumps(summary, indent=1))
    return 0 if summary["qsa"]["chain_equal"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make-inputs")
    a.add_argument("path", type=Path)
    a.add_argument("--rows", type=int, default=6)
    a.add_argument("--drafts", type=int, default=4)
    b = sub.add_parser("compare")
    b.add_argument("inputs", type=Path)
    b.add_argument("oracle", type=Path)
    b.add_argument("--mode", choices=["decode", "float32"], default="decode")
    b.add_argument("--report", type=Path)
    b.add_argument("--isolated", action="store_true")
    c = sub.add_parser("unrolled")
    c.add_argument("--rows", type=int, default=6)
    c.add_argument("--depth", type=int, default=3)
    c.add_argument("--mode", choices=["decode", "float32"], default="float32")
    c.add_argument("--report", type=Path)
    d = sub.add_parser("long")
    d.add_argument("inputs", type=Path)
    d.add_argument("oracle", type=Path)
    d.add_argument("--mode", choices=["decode", "float32"], default="float32")
    d.add_argument("--report", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(16)
    if args.cmd == "make-inputs":
        make_inputs(args.path, args.rows, args.drafts)
        return 0
    if args.cmd == "compare":
        return compare(args.inputs, args.oracle, args.mode, args.report, args.isolated)
    if args.cmd == "long":
        return long_context(args.inputs, args.oracle, args.mode, args.report)
    return unrolled(args.rows, args.depth, args.mode, args.report)


if __name__ == "__main__":
    sys.exit(main())
