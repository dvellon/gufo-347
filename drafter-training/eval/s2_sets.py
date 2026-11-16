"""Stage-2 record sets: the selection / judge split of the eval 100 and the data-mix subsets of arm 2. CPU only, stdlib only.
NEW file, 2026-10-27 (round-5 package fix; RECIPE.md sec. 2-3). Sidecars (.mtps.json) are read for ids, kinds, sources and
tokens; shards are only touched by `build-sel` / `mix --build`, which call ablations/gmtp_subset.py (byte copy).

Why: best.pt must never be selected on a record that later judges the head. The selection set is a FIXED half of the clean
eval 100 (split by prompt key, so a no-think twin never sits on the other side of its thinking record); the judge is eval-B
(all 30) + the other clean half ("judge-rest") + the live 016 A/B. SPLIT.json is written once on the host (pre-registered,
before any stage-2 number) and only ever SHRUNK on the cloud GPU (records whose source key collides with the final training data).

  # host, once (writes SPLIT.json; refuses to overwrite):
  python s2_sets.py split --eval-sidecars "<data-dir>/capture-onpolicy-v1/eval/shards/test/*.mtps.json" \
      --train-sidecars "<data-dir>/capture-onpolicy-v1/train/shards/train/*.mtps.json" --overlap overlap.json \
      --evalb-prompts ../../../eval-b-n22/prompts-evalb-xhigh.jsonl --out SPLIT.json
  # cloud GPU, after the PUSH-DONE markers (exit 1 = a disjointness assertion failed -> no arm may start):
  python s2_sets.py check --split SPLIT.json --eval-sidecars "/data/val-onpolicy/*.mtps.json" \
      --train-sidecars "/data/train-onpolicy/*.mtps.json" "/data/train/*.mtps.json" --evalb-sidecars "/data/val-onpolicy-b/*.mtps.json" \
      --out /data/runs/s2/SPLIT-CHECK.json
  python s2_sets.py build-sel --check /data/runs/s2/SPLIT-CHECK.json --out /data/sel-onpolicy [--dry]
  python s2_sets.py mix --train-sidecars "/data/train-onpolicy/*.mtps.json" --out-root /data --report /data/runs/s2/MIX.json [--build]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBSET = HERE.parent / "ablations" / "gmtp_subset.py"
SALT_SPLIT = "gufo-s2-split-v1|"
SALT_MIX = "gufo-s2-mix-v1|"
KIND_RE = re.compile(r"(?:^|-)(code|prose|agent)(?:-|$)")
MIXES = {"c": {"code": 0.5, "agent": 0.3, "prose": 0.2},            # code-heavy (R4 sec. 4 / R5 sec. 5: N18's prompt mix)
         "b": {"code": 1 / 3, "agent": 1 / 3, "prose": 1 / 3}}       # balanced
CHUNK = 12          # <= 15 records per shard -> stage2_data.load_eval_set's stride len//8 is 1 -> every record is used


def arm_of(rid: str) -> str:
    return rid.rsplit("-", 1)[-1]


THINK_ARM = re.compile(r"^(?:xhigh|a2x|n18x)")   # thinking arms: 042 xhigh512; queue 071 cloud-gen a2x1024 / n18x1024 (gpu1 2026-10-27)


def is_think(rid: str) -> bool:
    return bool(THINK_ARM.match(arm_of(rid)))


def kind_of(rid: str) -> str:
    m = KIND_RE.search(rid.split("onpolicy-", 1)[-1])
    return m.group(1) if m else "?"


def key_of(rid: str) -> str:                     # prompt key: the id without the arm suffix (xhigh/nothink twins share it)
    return rid.rsplit("-", 1)[0]


def h(salt: str, s: str) -> str:
    return hashlib.sha256((salt + s).encode()).hexdigest()


def sidecars(globs: list[str]) -> list[tuple[str, dict]]:
    out = []
    for g in globs:
        for p in sorted(glob.glob(g)):
            if p.endswith(".mtps.json"):
                out.append((p.replace("\\", "/"), json.load(open(p, encoding="utf-8"))))
    return out


def ordinal(path: str) -> int:
    m = re.search(r"shard-(\d+)-", Path(path).name)
    return int(m.group(1)) if m else 0


def a1_selected(evals: list[tuple[str, dict]]) -> list[str]:
    """The records A1/a1x's in-loop selection used: /data/val-onpolicy/*.mtps@response, --eval-set-records 24,
    stage2_data.load_eval_set = every shard (sorted), records[::max(1, len // 8)] -> 9 + 8 = 17 (R5 sec. 6.5)."""
    out = []
    for _, s in sorted(evals, key=lambda x: x[0]):
        recs = s.get("records", [])
        out += [r["id"] for r in recs[::max(1, len(recs) // 8)]]
    return out[:24]


# ------------------------------------------------------------------------------------------------------------- split
def cmd_split(a) -> int:
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (pre-registered split; never overwritten)")
    evals = sidecars(a.eval_sidecars)
    recs = [(r, p) for p, s in evals for r in s.get("records", [])]
    if len(recs) != 100:
        raise SystemExit(f"expected the eval 100, got {len(recs)} records")
    train_src = {r.get("source_id") for _, s in sidecars(a.train_sidecars) for r in s.get("records", [])} - {None}
    ov = json.load(open(a.overlap, encoding="utf-8"))
    clean = set(ov["eval100_clean"])
    evb = set()
    for f in a.evalb_prompts:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                evb.add(json.loads(line)["id"])
    excluded, elig = {}, {}
    for r, p in recs:
        why = []
        if r["id"] not in clean:
            why.append("content-overlap>=5%-with-042 (overlap.json)")
        if r.get("source_id") in train_src:
            why.append(f"source-key-in-train ({r.get('source_id')})")
        if why:
            excluded[r["id"]] = why
        else:
            elig[r["id"]] = r
    forced = set(a1_selected(evals))
    keys = {}
    for rid in elig:
        keys.setdefault(kind_of(rid), set()).add(key_of(rid))
    sel_keys, judge_keys, plan = set(), set(), {}
    for kind, ks in sorted(keys.items()):
        f_ = {k for k in ks if any(key_of(x) == k for x in forced)}
        rest = sorted(ks - f_, key=lambda k: h(SALT_SPLIT, k))
        want = max(len(f_), round(a.frac * len(ks)))
        take = rest[:max(0, want - len(f_))]
        sel_keys |= f_ | set(take); judge_keys |= set(rest[len(take):])
        plan[kind] = {"keys": len(ks), "forced_a1_keys": len(f_), "selection_keys": len(f_) + len(take), "judge_keys": len(rest) - len(take)}
    sel = sorted(i for i in elig if key_of(i) in sel_keys)
    jud = sorted(i for i in elig if key_of(i) in judge_keys)
    assert not set(sel) & set(jud) and not ({key_of(i) for i in sel} & {key_of(i) for i in jud})
    cnt = lambda ids: {"records": len(ids), "thinking": sum(arm_of(i).startswith("xhigh") for i in ids),
                       "nothink": sum(arm_of(i).startswith("nothink") for i in ids),
                       "by_kind_thinking": {k: sum(kind_of(i) == k and arm_of(i).startswith("xhigh") for i in ids) for k in ("code", "prose", "agent")}}
    doc = {"format": "gufo.s2-split.v1", "rule": (
        "eval 100 -> drop records not in overlap.json eval100_clean or whose source_id is in any train sidecar; split the rest BY PROMPT "
        "KEY per kind: keys holding A1's 17 in-loop selection records go to selection, the others ordered by sha256(salt+key), the first "
        f"round({a.frac} x keys) of each kind to selection. selection = best.pt only; judge_rest = judge only; never swapped, only shrunk"),
        "salt": SALT_SPLIT, "frac": a.frac, "sources": {"eval": [p for p, _ in evals], "train": a.train_sidecars, "overlap": a.overlap},
        "a1_selection_records": sorted(forced), "selection": sel, "judge_rest": jud, "excluded": excluded,
        "evalb_prompt_ids": sorted(evb), "counts": {"selection": cnt(sel), "judge_rest": cnt(jud), "excluded": len(excluded)}, "per_kind": plan}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["counts"]))
    return 0


# ------------------------------------------------------------------------------------------------------------- check
def cmd_check(a) -> int:
    sp = json.load(open(a.split, encoding="utf-8"))
    evals = sidecars(a.eval_sidecars)
    where = {r["id"]: (p, r) for p, s in evals for r in s.get("records", [])}
    train = sidecars(a.train_sidecars)
    t_ids = {r["id"] for _, s in train for r in s.get("records", [])}
    t_src = {r.get("source_id") for _, s in train for r in s.get("records", [])} - {None}
    b_ids = {r["id"] for _, s in sidecars(a.evalb_sidecars) for r in s.get("records", [])} if a.evalb_sidecars else set()
    b_keys = {key_of(i) for i in b_ids} | set(sp.get("evalb_prompt_ids", []))
    dropped, keep = {}, {"selection": [], "judge_rest": []}
    for side in keep:
        for i in sp[side]:
            if i not in where:
                dropped[i] = "not on the box"
            elif where[i][1].get("source_id") in t_src:
                dropped[i] = f"source-key-in-train ({where[i][1].get('source_id')})"
            else:
                keep[side].append(i)
    S, J = set(keep["selection"]), set(keep["judge_rest"])
    errs = []
    if S & J:
        errs.append(f"selection and judge_rest share {sorted(S & J)[:3]}")
    if {key_of(i) for i in S} & {key_of(i) for i in J}:
        errs.append("a prompt key sits on both sides")
    if (S | J) & t_ids:
        errs.append(f"a split record is in the training data {sorted((S | J) & t_ids)[:3]}")
    if {key_of(i) for i in S} & b_keys or S & b_ids:
        errs.append("an eval-B record/prompt is in the selection set")
    if b_ids & t_ids:
        errs.append("an eval-B record is in the training data")
    if not S:
        errs.append("empty selection set")
    doc = {"format": "gufo.s2-split-check.v1", "split": a.split, "ok": not errs, "errors": errs,
           "selection": keep["selection"], "judge_rest": keep["judge_rest"], "dropped_here": dropped,
           "evalb_records": len(b_ids), "train_records": len(t_ids),
           "shard_of": {i: where[i][0][:-len(".json")] for i in keep["selection"] + keep["judge_rest"]},
           "counts": {k: {"records": len(v), "thinking": sum(arm_of(i).startswith("xhigh") for i in v)} for k, v in keep.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"split check {'OK' if doc['ok'] else 'FAILED'}: selection {doc['counts']['selection']} judge_rest {doc['counts']['judge_rest']} "
          f"eval-B {len(b_ids)} dropped_here {len(dropped)} {'; '.join(errs)}")
    return 0 if doc["ok"] else 1


def subset(src: str, ids: list[str], out_dir: Path, tag: str, dry: bool) -> int:
    if list(out_dir.glob("*.mtps")):
        return 0                                    # built before (gmtp_subset refuses to overwrite anyway)
    idf = out_dir.parent / f"{out_dir.name}.ids.txt"
    cmd = [sys.executable, str(SUBSET), "--src", src, "--out-dir", str(out_dir), "--ids", str(idf), "--tag", tag]
    if dry:
        print("DRY " + " ".join(cmd) + f"   # {len(ids)} records")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    idf.write_text("\n".join(ids) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"gmtp_subset FAILED {out_dir}: {r.stderr[-400:]}", file=sys.stderr)
    return r.returncode


# --------------------------------------------------------------------------------------------------------- build-sel
def cmd_build_sel(a) -> int:
    ck = json.load(open(a.check, encoding="utf-8"))
    if not ck.get("ok"):
        raise SystemExit("SPLIT-CHECK not ok - refusing to build a selection set")
    out = Path(a.out)
    by = {}
    for i in ck["selection"]:
        by.setdefault(ck["shard_of"][i], []).append(i)
    chunks, rc = [], 0
    for src, ids in sorted(by.items()):
        for j in range(0, len(ids), CHUNK):
            n = len(chunks)
            d = out / f"c{n:02d}"
            rc |= subset(src, ids[j:j + CHUNK], d, f"s2sel{n:02d}", a.dry)
            chunks.append({"dir": str(d), "src": src, "ids": ids[j:j + CHUNK]})
    if not a.dry and rc == 0:
        (out / "SEL-DONE.json").write_text(json.dumps({"records": len(ck["selection"]), "chunks": chunks}, indent=1) + "\n")
    print(f"selection set: {len(ck['selection'])} records in {len(chunks)} shard(s) -> {out} {'(dry)' if a.dry else ('rc=%d' % rc)}")
    return rc


# --------------------------------------------------------------------------------------------------------------- mix
def cmd_mix(a) -> int:
    tr = sidecars(a.train_sidecars)
    think, other = {}, 0
    for p, s in tr:
        for r in s.get("records", []):
            if is_think(r["id"]):
                k = kind_of(r["id"])
                if k == "?":
                    other += 1
                    continue
                think.setdefault(k, []).append((h(SALT_MIX, r["id"]), r["id"], int(r["tokens"]), p))
    avail = {k: sum(x[2] for x in v) for k, v in think.items()}
    T = sum(avail.values())
    if not T or set(avail) != {"code", "prose", "agent"}:
        raise SystemExit(f"need thinking records of all three kinds, have {avail}")
    t_mix = a.t_mix or int(0.98 * min(avail[k] / sh for m in MIXES.values() for k, sh in m.items()))
    rep = {"format": "gufo.s2-mix.v1", "T_all_thinking": T, "natural_share": {k: round(v / T, 3) for k, v in avail.items()},
           "available": avail, "unknown_kind_thinking_records_excluded": other, "T_mix": t_mix, "salt": SALT_MIX, "mixes": {}}
    for name, shares in MIXES.items():
        chosen = set()
        got = {}
        for k, sh in shares.items():
            tgt, cum = sh * t_mix, 0
            for _, rid, tok, _ in sorted(think[k]):                # same hash order for both mixes -> maximal overlap
                if cum >= tgt or (cum + tok - tgt) > tok / 2:
                    break
                chosen.add(rid); cum += tok
            got[k] = cum
        rep["mixes"][name] = {"target_share": {k: round(v, 3) for k, v in shares.items()}, "tokens": got, "T": sum(got.values()),
                              "share": {k: round(v / max(1, sum(got.values())), 3) for k, v in got.items()}, "records": len(chosen)}
        rc = 0
        for p, s in tr:
            ids = [r["id"] for r in s.get("records", []) if not is_think(r["id"]) or r["id"] in chosen]
            if ids:
                d = Path(a.out_root) / f"mix-{name}" / f"s{ordinal(p):06d}-{hashlib.sha256(p.encode()).hexdigest()[:8]}"   # unique even if two captures reuse an ordinal
                rc |= subset(p[:-len(".json")], ids, d, f"mix{name}", not a.build)
        if a.build and rc == 0:
            (Path(a.out_root) / f"mix-{name}" / "MIX-DONE.json").write_text(json.dumps(rep["mixes"][name], indent=1) + "\n")
        if rc:
            return rc
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("T_all_thinking", "natural_share", "T_mix")}),
          json.dumps({n: (m["T"], m["share"]) for n, m in rep["mixes"].items()}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("split")
    p.add_argument("--eval-sidecars", nargs="+", required=True)
    p.add_argument("--train-sidecars", nargs="+", required=True)
    p.add_argument("--overlap", required=True)
    p.add_argument("--evalb-prompts", nargs="+", default=[])
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--split", required=True)
    p.add_argument("--eval-sidecars", nargs="+", required=True)
    p.add_argument("--train-sidecars", nargs="+", required=True)
    p.add_argument("--evalb-sidecars", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p = sub.add_parser("build-sel")
    p.add_argument("--check", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("mix")
    p.add_argument("--train-sidecars", nargs="+", required=True)
    p.add_argument("--out-root", default="/data")
    p.add_argument("--t-mix", type=int, default=0, help="thinking record tokens per mix (default: largest feasible x 0.98)")
    p.add_argument("--report", default=None)
    p.add_argument("--build", action="store_true", help="write the subset shards (default: plan only)")
    a = ap.parse_args(argv)
    return {"split": cmd_split, "check": cmd_check, "build-sel": cmd_build_sel, "mix": cmd_mix}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
