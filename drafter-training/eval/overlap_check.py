"""Content-overlap check between the stage-2 judge sets and the train-eligible on-policy data (host, read-only, 1 thread).

Id/source disjointness (inventory_z.py) is necessary but not sufficient: 042's prose prompts and the eval 100 share the
source key `log:server.stderr.log` (different server logs with the same base name; repetitive content). This measures
what actually overlaps:

  eval-100   token 32-grams of each eval record's RESPONSE (the judged anchors) found anywhere (prompt or response) in the
             train capture jobs (042 + every other train jobs-*.json found on Z:). -> per-record share.
  eval-B     not captured yet (queue 056): 120-char text shingles of each eval-B prompt found in the train prompt texts
             (042 prompts-train*.jsonl, 035 sessions-a2-*/n18 prompts). Re-run after 056 for the token form.

Writes overlap.json with a `clean` id list per judge set (share < --max-share, default 0.05) for the sensitivity score
(stage2_score.sh drops the others).

  python overlap_check.py --out overlap.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

W = Path(__file__).resolve().parents[3]
Z = "data"
N, K = 32, 120


def jobs_records(jobs_json: str):
    for j in json.load(open(jobs_json, encoding="utf-8"))["jobs"]:
        with open(j["input"], encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    segs = (r.get("metadata") or {}).get("segments") or [{"response_start": 0}]
                    yield r["id"], r["tokens"], int(segs[0]["response_start"])


def grams(t):
    return {tuple(t[i:i + N]) for i in range(max(0, len(t) - N + 1))}


def shingles(t, step=1):
    t = re.sub(r"\s+", " ", t)
    return {t[i:i + K] for i in range(0, max(0, len(t) - K + 1), step)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).with_name("overlap.json")))
    ap.add_argument("--max-share", type=float, default=0.05)
    a = ap.parse_args(argv)
    train_jobs = [p for d in ("capture-onpolicy-v1/train", "capture-onpolicy-v1/train-b", "capture-onpolicy-s2/*")
                  for p in glob.glob(f"{Z}/{d}/jobs-*.json")]
    tg = set()
    for jj in train_jobs:
        for _, t, _ in jobs_records(jj):
            tg |= grams(t)
    ev = []
    for jj in glob.glob(f"{Z}/capture-onpolicy-v1/eval/jobs-*.json"):
        for rid, t, ps in jobs_records(jj):
            g = grams(t[ps:])
            ev.append({"id": rid, "resp_share": round(len(g & tg) / max(1, len(g)), 4)})
    tx = set()
    for f in ("onpolicy/prompts-train.jsonl", "onpolicy/prompts-train-nothink.jsonl",
              "stage2-data-035/n18-prompts/prompts-train.jsonl"):
        p = W / f
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    tx |= shingles(json.dumps(json.loads(line).get("messages"), ensure_ascii=False))
    for f in ("stage2-data-035/sessions-a2-xhigh.json", "stage2-data-035/sessions-a2-pilot.json", "stage2-data-035/sessions-n18-xhigh.json"):
        p = W / f
        if p.is_file():
            tx |= shingles(p.read_text(encoding="utf-8"))
    eb = []
    for f in ("prompts-evalb-xhigh.jsonl", "prompts-evalb-nothink.jsonl"):
        for line in (W / "eval-b-n22" / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                s = shingles(json.dumps(r.get("messages"), ensure_ascii=False), 20)
                eb.append({"id": r["id"], "arm": f.split("-")[-1].split(".")[0], "prompt_text_share": round(len(s & tx) / max(1, len(s)), 4)})
    res = {"format": "gufo.stage2-overlap.v1", "train_jobs": train_jobs, "max_share": a.max_share,
           "eval100": sorted(ev, key=lambda x: -x["resp_share"]),
           "eval100_clean": sorted(x["id"] for x in ev if x["resp_share"] < a.max_share),
           "evalB_prompt": sorted(eb, key=lambda x: -x["prompt_text_share"]),
           "evalB_flag_ids": sorted({x["id"] for x in eb if x["prompt_text_share"] >= 0.10})}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"eval100_records": len(ev), "eval100_clean": len(res["eval100_clean"]),
                      "eval100_top": res["eval100"][:5], "evalB_top": res["evalB_prompt"][:4],
                      "evalB_flag_ids": res["evalB_flag_ids"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
