"""flashmap_generate responses (exact prompt/response token ids) -> w3mtp_capture jobs, split=train.

Like dflash-replay/make_replay_jobs.py `diverse`, but:
  * split "train" (the prompts come from mtp-split-v1 train sources only);
  * several response files (arms) in one jobs file, with record ids suffixed by arm;
  * records grouped into jobs of <= --tokens-per-shard tokens (default 100K, about
    2.1 GB per shard with features=mtp), with ordinals from --first-ordinal.

  python make_train_jobs.py --responses responses-train-xhigh1024.jsonl:xhigh1024 \
      responses-train-nothink512.jsonl:nothink512 --out jobs-train
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", nargs="+", required=True, help="FILE:ARM")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens-per-shard", type=int, default=100_000)
    ap.add_argument("--first-ordinal", type=int, default=1000)
    a = ap.parse_args()
    records = []
    for spec in a.responses:
        path, _, arm = spec.rpartition(":")
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            prompt, resp = list(r["prompt_tokens"]), list(r["response_tokens"])
            um = (r.get("metadata") or {}).get("user_metadata") or {}
            if isinstance(um, str):
                um = {}
            rec = {"id": f"{r['id']}-{arm}", "source_id": r.get("source_id", r["id"]), "split": "train",
                   "tokens": prompt + resp, "supervise_from": 0,
                   "metadata": {"kind": "target-responses",
                                "segments": [{"source_id": r.get("source_id", r["id"]), "response_id": r["id"],
                                              "start": 0, "response_start": len(prompt),
                                              "end": len(prompt) + len(resp)}],
                                "prompt_tokens": str(len(prompt)), "response_tokens": str(len(resp)),
                                "category": um.get("category") or "unknown", "task": um.get("task"), "arm": arm,
                                "termination": (r.get("metadata") or {}).get("termination")}}
            if r.get("target_sha256"):
                rec["target_sha256"] = r["target_sha256"]
            records.append(rec)
    a.out.mkdir(parents=True, exist_ok=True)
    jobs, cur, n = [], [], 0

    def flush():
        nonlocal cur, n
        if not cur:
            return
        k = len(jobs)
        path = a.out / f"job-train-{k:03d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for rec in cur:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        jobs.append({"input": str(path.resolve()), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "shard_ordinal": a.first_ordinal + k, "split": "train", "records": len(cur), "tokens": n})
        cur, n = [], 0

    for rec in records:
        if n + len(rec["tokens"]) > a.tokens_per_shard:
            flush()
        cur.append(rec)
        n += len(rec["tokens"])
    flush()
    (a.out / "jobs.json").write_text(json.dumps({"format": "gufo.mtp-capture.jobs.v1", "jobs": jobs}, indent=1) + "\n",
                                     encoding="utf-8")
    print(json.dumps({"records": len(records), "jobs": len(jobs), "tokens": sum(j["tokens"] for j in jobs),
                      "response_tokens": sum(int(r["metadata"]["response_tokens"]) for r in records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
