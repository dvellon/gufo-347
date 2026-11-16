"""On-policy TRAINING prompts for the MTP drafter (host-only; train split only).

This mirrors dflash-replay/make_onpolicy_prompts.py, which builds the held-out
test-split eval prompts, but draws every source from the mtp-split-v1 *train*
split. Train-split eval leakage is impossible by construction:
  code   train-split gufo source file + a task whose answer is code
  prose  train-split doc / build log / commit diff + summarise/explain task
  agent  train-split coding-agent transcript (<data-dir>/corpus/train-*.jsonl,
         redacted), cut before an assistant turn, + "continue as the assistant"
The target then answers greedily with qwen38_flash_next_flashmap_generate,
which records exact prompt and response token ids. Those responses are on-policy
for the current target. w3mtp_capture captures them; see README.md here.

  python make_train_prompts.py --out . --per-category 100 --transcripts <data-dir>/corpus/train-00003.jsonl
Memory: one transcript shard (~100 MB JSON, < 1.5 GB). Single thread.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DFR = HERE.parents[2] / "dflash-replay"
sys.path.insert(0, str(DFR))
import make_onpolicy_prompts as M  # noqa: E402  (puts capture/ and diverse-suite on sys.path)

bc = M.bc
SPLIT = "train"


def repo_files() -> dict[str, list[str]]:
    out = defaultdict(list)
    for rel in bc.git_files(M.REPO) or []:
        kind = bc.wanted(rel)
        if kind not in ("code", "doc") or bc.split_of(rel, set()) != SPLIT:
            continue
        if kind == "code" and Path(rel).suffix.lower() not in M.CODE_EXT:
            continue
        out[kind].append(rel)
    return out


def commits(n_scan: int) -> list[str]:
    import subprocess
    shas = subprocess.run(["git", "-C", str(M.REPO), "log", "--no-merges", "-n", str(n_scan), "--format=%H"],
                          capture_output=True, check=True, text=True).stdout.split()
    return [s for s in shas if bc.split_of(f"commit:{s}", set()) == SPLIT]


def logs() -> list[Path]:
    return [p for p in sorted((M.REPO / "build").glob("**/*.log"))
            if bc.split_of(f"log:{p.as_posix()}", set()) == SPLIT and p.stat().st_size > 2000]


def main() -> int:
    import subprocess
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=100)
    ap.add_argument("--prompt-tokens", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--transcripts", type=Path, required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tok, red = M.HostTokenizer(), M.Redactor()
    budget = a.prompt_tokens
    prompts, stats = [], Counter()

    def emit(category, task, source, text):
        n = tok.count_user_prompt(text)
        pid = f"onpolicy-train-{category}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"
        prompts.append({"id": pid, "messages": [{"role": "user", "content": text}],
                        "metadata": {"kind": "mtp-onpolicy-train", "category": category, "task": task,
                                     "prompt_tokens": str(n)},
                        "source_id": source})
        stats[category] += 1
        stats[f"{category}_tokens"] += n

    files = repo_files()
    pool = files["code"][:]
    rng.shuffle(pool)
    for rel in pool:
        if stats["code"] >= a.per_category:
            break
        text = bc.read_text(M.REPO / rel, 256 << 10)
        if not text or len(text) < 1500:
            continue
        task = M.CODE_TASKS[stats["code"] % len(M.CODE_TASKS)]
        lang = bc.LANG.get(Path(rel).suffix.lower(), "")
        lines = M.clean(text, tok, red).split("\n")
        body = M.trim_to_tokens(tok, lambda s: f"{task}\n\n[file] {rel}\n```{lang}\n{s}\n```", lines, budget)
        if body and tok.count_user_prompt(body) >= budget // 2:
            emit("code", task, f"gufo:{rel}", body)
    sources = defaultdict(list)
    docs = files["doc"][:]
    rng.shuffle(docs)
    sources["doc"] = [(f"gufo:{r}", lambda r=r: bc.read_text(M.REPO / r, 256 << 10)) for r in docs]
    lg = logs()
    rng.shuffle(lg)
    sources["log"] = [(f"log:{p.name}", lambda p=p: bc.read_text(p, 64 << 20)) for p in lg]
    cm = commits(1500)
    rng.shuffle(cm)
    sources["diff"] = [(f"commit:{s[:12]}", lambda s=s: subprocess.run(
        ["git", "-C", str(M.REPO), "show", "--stat", "--patch", "--no-color", s],
        capture_output=True).stdout.decode("utf-8", "replace").replace("\r\n", "\n")) for s in cm]
    order = ["doc", "log", "diff"]
    idx = Counter()
    while stats["prose"] < a.per_category and any(idx[k] < len(sources[k]) for k in order):
        want = order[stats["prose"] % 3]
        kind = want if idx[want] < len(sources[want]) else next(k for k in order if idx[k] < len(sources[k]))
        src, read = sources[kind][idx[kind]]
        idx[kind] += 1
        text = read()
        if not text or len(text) < 1500:
            continue
        if kind == "log" and len(text) > 40_000:
            text = text[:24_000] + "\n[... log truncated ...]\n" + text[-16_000:]
        task = M.PROSE_TASKS[kind]
        lines = M.clean(text, tok, red).split("\n")
        body = M.trim_to_tokens(tok, lambda s: f"{task}\n\n[{kind}] {src}\n```\n{s}\n```", lines, budget)
        if body and tok.count_user_prompt(body) >= budget // 2:
            emit("prose", f"{kind}: {task}", src, body)
    if a.transcripts.exists():
        sessions = M.transcript_sessions(a.transcripts)
        keys = sorted(sessions)
        rng.shuffle(keys)
        for key in keys:
            if stats["agent"] >= a.per_category:
                break
            segs = sessions[key]
            if any(int(str(d.get("session_hash", key))[:8], 16) % 100 >= 90 for d in segs[:1]):
                continue                                    # train split only (session_hash rule)
            cuts = [i for i, d in enumerate(segs) if d["kind"] in ("assistant_text", "tool_call") and i >= 4]
            if not cuts:
                continue
            cut = cuts[rng.randrange(len(cuts))]
            turns = [t for t in (M.render_turn(d) for d in segs[:cut]) if t]
            if len(turns) < 3:
                continue
            turns = [M.clean(t, tok, red) for t in turns]
            head, body = turns[0], None
            for start in range(1, len(turns)):
                text = head + "\n\n" + "\n\n".join(turns[start:]) + "\n\n" + M.AGENT_TASK
                try:
                    if tok.count_user_prompt(text) <= budget:
                        body = text
                        break
                except ValueError:
                    break
            if body and tok.count_user_prompt(body) >= budget // 3:
                emit("agent", "next-step", f"transcript:{key[:16]}", body)
    else:
        print(f"warning: transcripts not found: {a.transcripts}", file=sys.stderr)
    rng.shuffle(prompts)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "prompts-train.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    summary = {"prompts": len(prompts), "by_category": dict(stats), "seed": a.seed, "split": SPLIT,
               "prompt_tokens_target": budget, "transcripts": str(a.transcripts),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (a.out / "prompts-train-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
