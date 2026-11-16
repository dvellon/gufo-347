"""corpus_counts_public.py: per-token counts over an agent-transcript corpus (public-bench DESIGN.md section 4.2, step 2).

Port of dell-sims-026-oov/sim/corpus_counts.py (sha256 dbd3fee1...) with every path an argument: the counting loop is
the same statement for statement (same arrays, same "miss" rule, same file order), so on the same inputs it writes the
same arrays (checked on the Dell against the original on the public corpus; PERM-PUBLIC.md).

Reads gufo.agent-transcript-corpus.v1 jsonl shards ({session_hash, agent, role, kind, text}), tokenizes every segment
with the host tokenizer port (host_tokenizer.py from --tokenizer-dir, HostTokenizer.encode_plain; tokens/merges from
--vocab vocab.json, or from the GGUF itself with --gguf), and per session (session_hash, agent) in file order counts,
for every token id:
  all_<kind>   occurrences in assistant segments of that kind (tool_call / assistant_text / reasoning)
  miss_<kind>  occurrences in assistant segments that are NOT already in the session's earlier text (prompt-side and
               earlier assistant segments) nor earlier in the same segment -> what the per-request table cannot catch
  all_prompt   occurrences in non-assistant segments (user / tool_result / system / ...)
Rows with role == 'assistant' and a kind outside the three count as prompt (as the original).
Output: one .npz of int64 arrays [248320] (no pickle) + OUT.npz.json summary (adds input sha256s to the original's).

  python3 -I -B corpus_counts_public.py --tokenizer-dir DIR (--vocab vocab.json | --gguf MODEL.gguf) --out OUT.npz
          [--max-tokens N] SHARD.jsonl [SHARD.jsonl ...]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

V = 248320
DV = 49152
SPEC = set(range(248044, 248077))
ASST = ("tool_call", "assistant_text", "reasoning")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load_tokenizer(tok_dir: str, vocab: str = None, gguf: str = None):
    sys.path.insert(0, str(Path(tok_dir).resolve()))
    import host_tokenizer as ht  # our own repo port (read-only import)
    if vocab:
        def _arrays_from_vocab(_path):
            with open(vocab, encoding="utf-8") as f:
                v = json.load(f)
            return v["tokens"], v["merges"]
        ht._read_tokenizer_arrays = _arrays_from_vocab
        return ht, ht.HostTokenizer("unused")
    return ht, ht.HostTokenizer(Path(gguf))


def count(tok, shards, cap=0):
    import numpy as np
    cnt = {f"all_{k}": np.zeros(V, np.int64) for k in ASST}
    cnt.update({f"miss_{k}": np.zeros(V, np.int64) for k in ASST})
    cnt["all_prompt"] = np.zeros(V, np.int64)
    seen = {}
    t0 = time.time()
    ntok = nseg = 0
    by_kind = {}
    for sh in shards:
        for line in open(sh, encoding="utf-8"):
            d = json.loads(line)
            key = (d["session_hash"], d.get("agent"))
            s = seen.setdefault(key, set())
            ids = tok.encode_plain(d["text"] or "")
            kind = d["kind"] if d["role"] == "assistant" else "prompt"
            if kind not in ASST:
                kind = "prompt"
            a = np.asarray(ids, np.int64)
            if kind == "prompt":
                np.add.at(cnt["all_prompt"], a, 1)
                s.update(t for t in ids if t >= DV)
            else:
                np.add.at(cnt[f"all_{kind}"], a, 1)
                m = cnt[f"miss_{kind}"]
                for t in ids:
                    if t >= DV and t not in SPEC and t not in s:
                        m[t] += 1
                        s.add(t)
            bk = by_kind.setdefault(kind, [0, 0])
            bk[0] += 1
            bk[1] += len(ids)
            ntok += len(ids)
            nseg += 1
            if nseg % 20000 == 0:
                print(f"{sh} seg {nseg} tok {ntok} {time.time() - t0:.0f}s", flush=True)
            if cap and ntok >= cap:
                break
        if cap and ntok >= cap:
            break
    summ = {"shards": [str(x) for x in shards], "segments": nseg, "tokens": ntok, "sessions": len(seen),
            "seconds": time.time() - t0, "by_kind": by_kind,
            "rare_share": {k: float(cnt[f"all_{k}"][DV:].sum() / max(cnt[f"all_{k}"].sum(), 1)) for k in ASST},
            "miss_share": {k: float(cnt[f"miss_{k}"].sum() / max(cnt[f"all_{k}"].sum(), 1)) for k in ASST}}
    return cnt, summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", required=True, help="directory holding host_tokenizer.py")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vocab", help="vocab.json {tokens, merges} extracted from the public GGUF")
    g.add_argument("--gguf", help="read tokens / merges from the GGUF with host_tokenizer's own reader")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=0, help="cap the stream (a test); 0 = all")
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args(argv)
    import numpy as np
    _, tok = load_tokenizer(a.tokenizer_dir, a.vocab, a.gguf)
    cnt, summ = count(tok, a.shards, a.max_tokens)
    np.savez(a.out, **cnt)
    summ["inputs_sha256"] = {
        "shards": [sha256_file(x) for x in a.shards],
        "host_tokenizer.py": sha256_file(Path(a.tokenizer_dir) / "host_tokenizer.py"),
        ("vocab.json" if a.vocab else "gguf"): sha256_file(a.vocab or a.gguf)}
    out_npz = a.out if a.out.endswith(".npz") else a.out + ".npz"
    summ["npz_sha256"] = sha256_file(out_npz)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
