"""Host-only corpus builder for the MTP-drafter capture (w3mtp_capture jobs).

Turns local coding-agent text into chat-framed, tokenized, packed records and
cuts them into capture jobs (one JSONL file per shard) plus ``jobs.json``.
Nothing is downloaded or launched; the tokenizer is gufo's own BPE ported in
``../../diverse-suite/host_tokenizer.py`` (validated there against server
counts and HF goldens) reading only the GGUF header.

Sources (all optional, repeatable):
  --repo NAME=PATH        tracked files (git ls-files) or a walked tree
  --git-log NAME=PATH[@N] the last N (default 400) non-merge commit patches
  --logs GLOB             build/test logs (head+tail capped)
  --responses JSONL       target-generated responses from flashmap_generate
                          (exact prompt+response token ids; target-bound)

Text documents are redacted with the diverse-suite Redactor (user-profile
paths, account/host names, e-mails, IPs, secret-like assignments), deduped by
content hash and assigned a split by a hash of the source's *relative path*
(``mtp-split-v1``: 90% train / 5% val / 5% test), so copies of a file in
several local checkouts always land in the same split. Files used by the
diverse pp2048 suite are forced into ``test`` so live acceptance A/B on that
suite measures held-out content. Records pack several turns from the same
split into one multi-turn agent transcript of at most --record-tokens:

  read : user turn "[tool call] read_file PATH / [tool result]" + numbered lines
  write: user asks for PATH, assistant turn writes it in a fenced block
  tool : "[tool call] git show SHA" / "[tool call] run build" + output

Every row is supervised by the target's own distribution at capture time, so
the rendered text only needs to look like real agent context. Metadata keeps
per-turn token spans (kind/template/source) so the trainer can weight
assistant-side and target-generated positions.

    python build_corpus.py --out corpus-v1 --repo gufo=../../../.. --git-log gufo=../../../..@600 \\
        --logs "C:/.../build/**/*.log" --responses .../flashmap-qwen38-responses-v2.jsonl
    python build_corpus.py ... --dry-run          # counts only
    python build_corpus.py ... --emit-prompts 2000 --out prompts-v1   # flashmap_generate input
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[1] / "diverse-suite"
sys.path.insert(0, str(SUITE))
from host_tokenizer import DEFAULT_MODEL, IM_END, IM_START, XHIGH, HostTokenizer  # noqa: E402
from prepare import Redactor  # noqa: E402

SPLIT_SALT = "mtp-split-v1"
CODE_EXT = {".cpp", ".cc", ".c", ".h", ".hpp", ".hip", ".cu", ".cuh", ".py", ".ps1", ".sh", ".cmake",
            ".rs", ".go", ".js", ".ts", ".tsx", ".java", ".kt", ".cs", ".swift", ".m", ".mm", ".zig",
            ".lua", ".rb", ".nix", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".bat", ".cmd", ".mk",
            ".proto", ".sql", ".glsl", ".hlsl", ".metal"}
DOC_EXT = {".md", ".rst", ".txt", ".adoc"}
JSON_EXT = {".json", ".jsonl"}
NAMED_FILES = {"CMakeLists.txt", "Makefile", "Dockerfile", "meson.build", "BUILD", "WORKSPACE"}
SKIP_DIRS = {".git", "build", "node_modules", "__pycache__", ".venv", "venv", "dist", "out", ".cache",
             "third_party_downloads", ".pytest_cache", ".mypy_cache"}
LANG = {".cpp": "cpp", ".cc": "cpp", ".c": "c", ".h": "cpp", ".hpp": "cpp", ".hip": "cpp", ".cu": "cuda",
        ".py": "python", ".ps1": "powershell", ".sh": "bash", ".cmake": "cmake", ".rs": "rust",
        ".go": "go", ".js": "javascript", ".ts": "typescript", ".md": "markdown", ".json": "json",
        ".yml": "yaml", ".yaml": "yaml", ".toml": "toml", ".nix": "nix"}
TASKS_READ = ["Read the file and tell me where the main logic lives.",
              "I need to change this module; first show me its current contents.",
              "Review this file for correctness issues.",
              "Find the function that handles this and explain it.",
              "Check how this is implemented before we edit it."]


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def split_of(source_key: str, forced_test: set[str]) -> str:
    if source_key in forced_test:
        return "test"
    bucket = int(sha256(f"{SPLIT_SALT}:{source_key}")[:8], 16) % 100
    return "train" if bucket < 90 else ("val" if bucket < 95 else "test")


# --------------------------------------------------------------- discovery

def git_files(root: Path) -> list[str] | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]


def walk_files(root: Path) -> list[str]:
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            result.append(Path(dirpath, name).relative_to(root).as_posix())
    return result


def wanted(rel: str) -> str | None:
    name = rel.rsplit("/", 1)[-1]
    ext = Path(name).suffix.lower()
    if name in NAMED_FILES or ext in CODE_EXT:
        return "code"
    if ext in DOC_EXT:
        return "doc"
    if ext in JSON_EXT:
        return "json"
    return None


def read_text(path: Path, max_bytes: int) -> str | None:
    try:
        size = path.stat().st_size
        if size == 0 or size > max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = text.splitlines()
    if not lines or max(len(l) for l in lines) > 4000 or len(text) / len(lines) > 300:
        return None  # minified, generated or data-like
    return text.replace("\r\n", "\n")


def discover(args) -> list[dict]:
    docs = []
    for spec in args.repo:
        name, _, path = spec.partition("=")
        root = Path(path).resolve()
        files = git_files(root) or walk_files(root)
        for rel in sorted(files):
            kind = wanted(rel)
            if kind is None or any(fnmatch.fnmatch(rel, pat) for pat in args.exclude):
                continue
            limit = args.max_file_bytes if kind != "json" else min(args.max_file_bytes, 32 << 10)
            text = read_text(root / rel, limit)
            if text is None:
                continue
            docs.append({"kind": "doc" if kind == "doc" else "code", "key": rel, "repo": name,
                         "path": rel, "text": text})
    for spec in args.git_log:
        name, _, rest = spec.partition("=")
        path, _, count = rest.partition("@")
        root = Path(path).resolve()
        try:
            shas = subprocess.run(["git", "-C", str(root), "log", "--no-merges", "-n", count or "400",
                                   "--format=%H"], capture_output=True, check=True, text=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            shas = []
        for sha in shas:
            out = subprocess.run(["git", "-C", str(root), "show", "--stat", "--patch", "--no-color", sha],
                                 capture_output=True).stdout.decode("utf-8", "replace").replace("\r\n", "\n")
            if len(out) > 48_000:
                out = out[:48_000] + "\n[... diff truncated ...]\n"
            docs.append({"kind": "diff", "key": f"commit:{sha}", "repo": name, "path": sha[:12], "text": out})
    seen_logs = set()
    for pattern in args.logs:
        for path in sorted(glob.glob(pattern, recursive=True)):
            p = Path(path)
            text = read_text(p, 64 << 20) if p.is_file() else None
            if not text:
                continue
            if len(text) > 40_000:
                text = text[:24_000] + "\n[... log truncated ...]\n" + text[-16_000:]
            digest = sha256(text)
            if digest in seen_logs:
                continue
            seen_logs.add(digest)
            docs.append({"kind": "log", "key": f"log:{p.as_posix()}", "repo": "logs",
                         "path": p.name, "text": text})
    return docs


# --------------------------------------------------------------- rendering

_TOK: HostTokenizer | None = None
_RED: Redactor | None = None


def _init(model: str) -> None:
    global _TOK, _RED
    _TOK = HostTokenizer(Path(model))
    _RED = Redactor()


def _frame(text: str) -> list[int]:
    return _TOK.encode_with_specials(text)


def _plain(text: str) -> list[int]:
    return _TOK.encode_plain(text)


def render_doc(job: tuple[dict, str, int]) -> dict:
    """Redacts, windows and renders one document into turns (worker)."""
    doc, split, max_doc_tokens = job
    before = dict(_RED.counts)
    text = _RED(doc["text"])
    counts = {k: _RED.counts[k] - before[k] for k in before}
    rng = random.Random(sha256(doc["key"]))
    lines = text.split("\n")
    turns = []
    start = 0
    window_lines = max(20, int(max_doc_tokens * 3.0 / max(8.0, len(text) / max(1, len(lines)))))
    while start < len(lines):
        count = min(window_lines, len(lines) - start)
        while True:
            chunk = "\n".join(lines[start:start + count])
            content = _plain(chunk)
            if len(content) <= max_doc_tokens or count <= 8:
                break
            count = max(8, count * 2 // 3)
        if len(content) > max_doc_tokens:
            content = content[:max_doc_tokens]  # pathological long lines
        first, last = start + 1, start + count
        ext = Path(doc["path"]).suffix.lower()
        kind = doc["kind"]
        template = ("write" if kind in ("code", "doc") and rng.random() < (0.35 if kind == "code" else 0.4)
                    else "read" if kind in ("code", "doc") else "tool")
        if template == "read":
            numbered = "\n".join(f"{i:>5}| {l}" for i, l in enumerate(lines[start:start + count], first))
            content = _plain(numbered)[:max_doc_tokens + 400]
            head = _frame(f"{IM_START}user\n{rng.choice(TASKS_READ)}\n\n[tool call] read_file "
                          f"{doc['path']} lines {first}-{last}\n[tool result]\n")
            tail = _frame(f"{IM_END}\n")
        elif template == "write":
            lang = LANG.get(ext, "")
            head = _frame(f"{IM_START}user\nWrite `{doc['path']}`"
                          + (f" (lines {first}-{last})" if start or count < len(lines) else "")
                          + f".{IM_END}\n{IM_START}assistant\n<think>\nI will write {doc['path']}.\n</think>\n\n"
                          + f"```{lang}\n")
            tail = _frame(f"\n```{IM_END}\n")
        else:
            call = (f"git show {doc['path']}" if kind == "diff" else f"run build and tests ({doc['path']})")
            head = _frame(f"{IM_START}user\n[tool call] {call}\n[tool result]\n")
            tail = _frame(f"{IM_END}\n")
        turns.append({"tokens": head + content + tail, "head": len(head), "body": len(content),
                      "kind": kind, "template": template, "source_id": f"{doc['repo']}:{doc['key']}",
                      "lines": [first, last]})
        start += count
    return {"key": doc["key"], "split": split, "turns": turns, "redactions": counts,
            "text_sha256": sha256(text)}


def fill_bins(items: list, size, capacity: int, lookahead: int = 256) -> list[list]:
    """Order-preserving first-fit with bounded lookahead (dense records)."""
    pending = [it for it in items if size(it) <= capacity]
    groups = []
    while pending:
        group, used, i = [], 0, 0
        while i < len(pending) and i < lookahead and capacity - used >= 32:
            if used + size(pending[i]) <= capacity:
                it = pending.pop(i)
                group.append(it)
                used += size(it)
            else:
                i += 1
        groups.append(group)
    return groups


def pack(turns: list[dict], system: list[int], record_tokens: int, split: str, prefix: str) -> list[dict]:
    records = []
    for group in fill_bins(turns, lambda t: len(t["tokens"]), record_tokens - len(system)):
        cur, spans = list(system), []
        for t in group:
            s = len(cur)
            cur.extend(t["tokens"])
            spans.append({"source_id": t["source_id"], "kind": t["kind"], "template": t["template"],
                          "start": s, "body_start": s + t["head"], "body_end": s + t["head"] + t["body"],
                          "end": len(cur), "lines": t.get("lines")})
        records.append({"id": f"{prefix}-{split}-{len(records):06d}",
                        "source_id": f"mtp-corpus:{group[0]['source_id']}", "split": split,
                        "tokens": cur, "supervise_from": 0,
                        "metadata": {"kind": "agent-transcript", "turns": spans}})
    return records


def response_records(paths: list[str], record_tokens: int, forced_test: set[str]) -> tuple[list[dict], Counter]:
    by_split: dict[tuple[str, str], list[dict]] = defaultdict(list)
    stats = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                toks = list(r["prompt_tokens"]) + list(r["response_tokens"])
                # Held-out by source prompt (the upstream dataset's own split is ignored).
                split = split_of(r["source_id"], forced_test)
                by_split[(split, r.get("target_sha256", ""))].append(
                    {"tokens": toks, "prompt": len(r["prompt_tokens"]), "source_id": r["source_id"], "id": r["id"]})
                stats["responses"] += 1
                stats["response_tokens"] += len(r["response_tokens"])
                stats["prompt_tokens"] += len(r["prompt_tokens"])
    records = []
    for (split, target), convs in sorted(by_split.items()):
        for n, group in enumerate(fill_bins(convs, lambda c: len(c["tokens"]), record_tokens)):
            cur, segs = [], []
            for c in group:
                s = len(cur)
                cur.extend(c["tokens"])
                segs.append({"source_id": c["source_id"], "response_id": c["id"], "start": s,
                             "response_start": s + c["prompt"], "end": len(cur)})
            records.append(_resp_record(cur, segs, split, target, n))
    return records, stats


def _resp_record(tokens, segs, split, target, n):
    rec = {"id": f"resp-{split}-{target[:8]}-{n:06d}", "source_id": f"mtp-responses:{segs[0]['source_id']}",
           "split": split, "tokens": tokens, "supervise_from": 0,
           "metadata": {"kind": "target-responses", "segments": segs}}
    if target:
        rec["target_sha256"] = target
    return rec


def emit_prompts(rendered: list[dict], docs_by_key: dict, n: int, out: Path) -> None:
    """flashmap_generate input: agent-style requests over real excerpts."""
    tasks = {"code": ["Review this code and list concrete bugs or risks, with fixes.",
                      "Write unit tests for the code below.",
                      "Refactor the code below for clarity without changing behaviour; show the diff.",
                      "Explain what this code does and how it could fail."],
             "doc": ["Answer: what are the key decisions in this document, and what is left open?",
                     "Summarize this document for a new contributor."],
             "diff": ["Explain this change for a reviewer and point out anything risky."],
             "log": ["Summarize this log: what failed, why, and what to do next."]}
    rng = random.Random(1234)
    pool = [r for r in rendered if r["turns"]]
    rng.shuffle(pool)
    with (out / "prompts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in pool[:n]:
            doc = docs_by_key[r["key"]]
            text = _RED_MAIN(doc["text"])[:6000]
            task = rng.choice(tasks[doc["kind"]])
            content = f"{task}\n\n[file] {doc['path']}\n```\n{text}\n```"
            f.write(json.dumps({"id": f"mtp-prompt-{sha256(r['key'])[:16]}",
                                "messages": [{"role": "user", "content": content}],
                                "metadata": {"split": r["split"], "kind": doc["kind"], "source_key": r["key"]},
                                "source_id": f"{doc['repo']}:{r['key']}"}, ensure_ascii=False) + "\n")


_RED_MAIN = None


def main(argv=None) -> int:
    global _RED_MAIN
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", action="append", default=[])
    p.add_argument("--git-log", action="append", default=[])
    p.add_argument("--logs", action="append", default=[])
    p.add_argument("--responses", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[], help="fnmatch on repo-relative paths")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--record-tokens", type=int, default=2048)
    p.add_argument("--max-doc-tokens", type=int, default=1200)
    p.add_argument("--shard-tokens", type=int, default=100_000)
    p.add_argument("--max-file-bytes", type=int, default=256 << 10)
    p.add_argument("--heldout-manifest", type=Path, default=SUITE / "manifest.json")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--limit-docs", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--emit-prompts", type=int, default=0)
    p.add_argument("--strip-response-target", action="store_true",
                   help="drop target_sha256 from response records (use when the capture model differs)")
    args = p.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        p.error("--out must be new or empty")
    forced_test = set()
    if args.heldout_manifest and args.heldout_manifest.exists():
        m = json.loads(args.heldout_manifest.read_text(encoding="utf-8"))
        forced_test = {s["path"] for pr in m.get("prompts", []) for s in pr.get("sources", [])}

    docs = discover(args)
    if args.limit_docs:
        docs = docs[: args.limit_docs]
    stats = Counter()
    unique, seen = [], {}
    for d in docs:
        digest = sha256(d["text"])
        if digest in seen:
            stats["dedup_exact"] += 1
            continue
        seen[digest] = d["key"]
        unique.append(d)
    jobs = [(d, split_of(d["key"], forced_test), args.max_doc_tokens) for d in unique]
    print(f"documents: {len(docs)} discovered, {len(unique)} unique; tokenizing with {args.workers} workers",
          flush=True)
    rendered, redactions = [], Counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.model,)) as pool:
        for i, r in enumerate(pool.map(render_doc, jobs, chunksize=8)):
            rendered.append(r)
            redactions.update(r["redactions"])
            if (i + 1) % 500 == 0:
                print(f"  rendered {i + 1}/{len(jobs)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.emit_prompts:
        _RED_MAIN = Redactor()
        emit_prompts(rendered, {d["key"]: d for d in unique}, args.emit_prompts, args.out)
        print(f"wrote {args.out / 'prompts.jsonl'}")
        return 0

    _init(args.model)
    system = _frame(f"{IM_START}system\n{XHIGH}{IM_END}\n")
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rendered:
        by_split[r["split"]].extend(r["turns"])
    records_by_split: dict[str, list[dict]] = defaultdict(list)
    for split, turns in by_split.items():
        rng = random.Random(f"pack:{split}")
        rng.shuffle(turns)  # mix kinds and sources inside each split
        records_by_split[split] += pack(turns, system, args.record_tokens, split, "text")
    resp_records, resp_stats = response_records(args.responses, args.record_tokens, forced_test)
    for r in resp_records:
        if args.strip_response_target:
            r.pop("target_sha256", None)
            r["metadata"]["kind"] = "foreign-target-responses"
        records_by_split[r["split"]].append(r)

    summary = {"documents": len(docs), "unique_documents": len(unique), "dedup_exact": stats["dedup_exact"],
               "redactions": dict(redactions), "responses": dict(resp_stats), "splits": {},
               "kinds": dict(Counter(d["kind"] for d in unique))}
    jobs_out, ordinal = [], 0
    for split in ("test", "val", "train"):
        recs = records_by_split.get(split, [])
        random.Random(f"order:{split}").shuffle(recs)
        tokens = sum(len(r["tokens"]) for r in recs)
        summary["splits"][split] = {"records": len(recs), "tokens": tokens}
        kinds = Counter()
        for r in recs:
            for t in r["metadata"].get("turns", []):
                kinds[t["kind"] + "/" + t["template"]] += t["end"] - t["start"]
            for s in r["metadata"].get("segments", []):
                kinds["target-response"] += s["end"] - s["response_start"]
                kinds["target-prompt"] += s["response_start"] - s["start"]
        summary["splits"][split]["turn_tokens_by_kind"] = dict(kinds)
        if args.dry_run:
            continue
        d = args.out / "jobs" / split
        d.mkdir(parents=True, exist_ok=True)
        batches, batch, batch_tokens = [], [], 0
        for r in recs:
            if batch and batch_tokens + len(r["tokens"]) > args.shard_tokens:
                batches.append((batch, batch_tokens))
                batch, batch_tokens = [], 0
            batch.append(r)
            batch_tokens += len(r["tokens"])
        if batch:
            batches.append((batch, batch_tokens))
        for batch, batch_tokens in batches:
            path = d / f"job-{ordinal:06d}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as f:
                for rec in batch:
                    f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
            jobs_out.append({"input": str(path.resolve()), "input_sha256": sha256(path.read_bytes()),
                             "shard_ordinal": ordinal, "split": split, "records": len(batch),
                             "tokens": batch_tokens})
            ordinal += 1
    total = sum(s["tokens"] for s in summary["splits"].values())
    summary["total_tokens"] = total
    summary["estimated_gb_at_20700_bytes_per_token"] = round(total * 20_700 / 1e9, 1)
    summary["generator_sha256"] = sha256(Path(__file__).read_bytes())
    (args.out / "corpus-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if not args.dry_run:
        (args.out / "jobs.json").write_text(json.dumps({"format": "gufo.mtp-capture.jobs.v1",
                                                        "jobs": jobs_out}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
