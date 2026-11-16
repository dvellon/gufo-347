"""End-to-end CPU test of train_mtp.py on the tiny synthetic MTP (< 1 GB, ~1 min, 2 threads).

Checks: training steps run with finite loss; checkpoints are written; --resume auto
continues from the latest checkpoint (step count, data cursor); expert-mode full
(Adafactor) and lora both run; --benchmark-steps reports a projection; in-loop
eval produces acceptance numbers.
  python selftest_train.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import mtp_net as N
import tiny
import train_mtp


def log(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "train-log.jsonl").read_text().splitlines()]


def main() -> int:
    N.lower_priority()
    tmp = Path(tempfile.mkdtemp(prefix="mtp-tiny-"))
    tr, va = tiny.write_shards(tmp / "data", n_records=8, n_tokens=48)
    common = ["--device", "cpu", "--tiny-synthetic", "--threads", "2", "--shards", str(tr), "--val-shards", str(va),
              "--depth", "4", "--grad-accum", "2", "--ckpt-every", "2", "--eval-every", "2", "--eval-records", "2",
              "--log-every", "1", "--warmup", "1"]
    res = {}
    run = tmp / "lora"
    train_mtp.main(common + ["--out", str(run), "--max-steps", "3", "--expert-mode", "lora", "--resume", "none"])
    l1 = log(run)
    train_mtp.main(common + ["--out", str(run), "--max-steps", "4", "--expert-mode", "lora", "--resume", "auto"])
    l2 = log(run)[len(l1):]
    steps = [r for r in log(run) if r.get("event") == "step"]
    res["lora_finite"] = all(r["loss"] == r["loss"] and abs(r["loss"]) < 1e6 for r in steps)
    res["resumed"] = any(r.get("event") == "resumed" for r in l2) and \
        [r for r in l2 if r.get("event") == "end"][0]["step"] == 4
    res["checkpoints"] = sorted(p.name for p in (run / "ckpt").glob("*.pt"))
    res["eval_km_step0"] = [r for r in l1 if r.get("event") == "eval"][0]["km"]
    run2 = tmp / "full"
    train_mtp.main(common + ["--out", str(run2), "--max-steps", "2", "--expert-mode", "full", "--dense-mode", "lora",
                             "--resume", "none", "--qat"])
    res["full_qat_end"] = [r for r in log(run2) if r.get("event") == "end"][0]["step"] == 2
    run3 = tmp / "bench"
    train_mtp.main(common + ["--out", str(run3), "--benchmark-steps", "2", "--resume", "none"])
    b = [r for r in log(run3) if r.get("event") == "benchmark"]
    res["benchmark"] = b[0] if b else None
    # growing shard set: start with one shard, add a second while/after training (new block, cursor kept)
    import shutil as _sh
    gdir = tmp / "grow"
    gdir.mkdir()
    _sh.copy(tr, gdir / tr.name)
    tr2 = tiny.write_shards(tmp / "data2", n_records=6, n_tokens=48, seed=5)[0]
    tr2_named = gdir / tr2.name.replace("shard-000007", "shard-000017")
    run4 = tmp / "grow-run"
    g = ["--device", "cpu", "--tiny-synthetic", "--threads", "2", "--shards", str(gdir / "*.mtps"),
         "--depth", "3", "--grad-accum", "2", "--ckpt-every", "1", "--log-every", "1", "--warmup", "1",
         "--refresh-shards-steps", "1", "--out", str(run4)]
    train_mtp.main(g + ["--max-steps", "2", "--resume", "none"])
    _sh.copy(tr2, tr2_named)                     # a shard "arrives" (complete file)
    train_mtp.main(g + ["--max-steps", "7", "--resume", "auto"])
    lg = log(run4)
    starts = [r for r in lg if r.get("event") == "start"]
    res["grow_blocks"] = starts[-1]["train_records"] if starts else None
    ends = [r for r in lg if r.get("event") == "end"]
    res["grow_ok"] = bool(ends) and ends[-1]["step"] == 7 and ends[-1]["tokens"] == 14 * 48 and any(r.get("event") == "resumed" for r in lg)         and starts[-1]["train_records"] == 14
    ok = res["lora_finite"] and res["resumed"] and res["full_qat_end"] and bool(b) and res["checkpoints"] and res["grow_ok"]
    res["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(res, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
