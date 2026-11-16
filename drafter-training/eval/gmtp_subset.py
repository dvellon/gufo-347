"""Write a GMTP v1 shard holding a subset of another shard's records (byte copy, no decode). CPU only, numpy-free.

Fable shallow review R2 N13: train on 30 of the 60 thinking records of the on-policy eval shard (10 per kind, in shard
order) and judge on the other 70. The file header is copied verbatim (same target/ABI/producer hashes, split and
ordinal), records are copied as byte ranges with their ordinal renumbered, the record sha256 (trailer[:32]) and the
trailer ordinal re-computed, and the file is named shard-<ordinal>-<tag>-sha256-<file sha256>.mtps with a sidecar
.mtps.json in the capture's format (read_gmtp.verify checks both).

  python gmtp_subset.py --src /data/val-onpolicy/shard-000910-*.mtps --out-dir /data/n13/train30 \
      --select code=10,prose=10,agent=10 --tag n13train30
  python gmtp_subset.py --src ... --out-dir ... --ids ids.txt --tag ...      # explicit record ids, one per line

Prints one JSON line: {file, sha256, records, tokens, ids}. Writes SELECTION.json next to the shard (selected ids
and the held-out ids of the source shard). Refuses to overwrite an existing output directory with .mtps files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for cand in (HERE.parents[2] / "capture", HERE.parents[1] / "capture", Path("capture")):
    if (cand / "read_gmtp.py").exists():
        sys.path.insert(0, str(cand))
        break
import read_gmtp as G  # noqa: E402

KIND_RE = re.compile(r"onpolicy-(code|prose|agent)-[0-9a-f]+-(xhigh512|nothink512)")


def kind_of(rid: str) -> str:
    m = KIND_RE.search(rid)
    return m.group(1) if m else "?"


def select_ids(shard: G.Shard, select: str | None, ids_file: Path | None) -> list[str]:
    all_ids = [r.metadata["id"] for r in shard.records]
    if ids_file:
        want = [l.strip() for l in ids_file.read_text().splitlines() if l.strip()]
        missing = [w for w in want if w not in all_ids]
        if missing:
            raise SystemExit(f"ids not in shard: {missing[:3]}")
        return want
    out = []
    for part in filter(None, (select or "").split(",")):
        kind, n = part.split("=")
        n = int(n)
        picked = [i for i in all_ids if kind_of(i) == kind][:n]
        if len(picked) < n:
            raise SystemExit(f"only {len(picked)} records of kind {kind} (wanted {n})")
        out += picked
    if not out:
        raise SystemExit("nothing selected (--select or --ids)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--select", default=None, help="kind=N,... (first N records of each kind in shard order)")
    ap.add_argument("--ids", default=None, type=Path, help="file with one record id per line")
    ap.add_argument("--tag", default="subset", help="name tag: shard-<ordinal>-<tag>-sha256-<sha>.mtps")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)

    src = a.src
    if not src.exists():
        hits = sorted(src.parent.glob(src.name))
        if len(hits) != 1:
            raise SystemExit(f"--src must name exactly one shard, got {hits}")
        src = hits[0]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if list(a.out_dir.glob("*.mtps")):
        raise SystemExit(f"{a.out_dir} already holds a .mtps; refusing to overwrite")

    shard = G.Shard(src)
    by_id = {r.metadata["id"]: r for r in shard.records}
    ids = select_ids(shard, a.select, a.ids)
    held = [r.metadata["id"] for r in shard.records if r.metadata["id"] not in set(ids)]

    tmp = a.out_dir / "subset.part"
    fsha = hashlib.sha256()
    n_tok = 0
    sup_rows = 0
    entries = []
    with src.open("rb") as f, tmp.open("wb") as out:
        header = f.read(G.FILE_HEADER)
        G.parse_header(header)                   # digest check
        out.write(header); fsha.update(header)
        pos = G.FILE_HEADER
        for k, rid in enumerate(ids):
            r = by_id[rid]
            f.seek(r.offset)
            buf = bytearray(f.read(r.total_bytes))
            tr = r.trailer_offset - r.offset
            struct.pack_into("<Q", buf, 24, k)                           # record header ordinal
            buf[tr:tr + 32] = hashlib.sha256(bytes(buf[:tr])).digest()   # record sha256 (header..trailer)
            struct.pack_into("<Q", buf, tr + 40, k)                      # trailer ordinal
            out.write(buf); fsha.update(buf)
            entries.append({"id": rid, "source_id": r.metadata["source_id"], "offset": pos, "bytes": r.total_bytes,
                            "tokens": r.n_tokens, "supervise_from": r.supervise_from})
            n_tok += r.n_tokens
            sup_rows += r.n_tokens - r.supervise_from
            pos += r.total_bytes
    digest = fsha.hexdigest()
    h = shard.header
    name = f"shard-{h.shard_ordinal:06d}-{a.tag}-sha256-{digest}.mtps"
    dst = a.out_dir / name
    tmp.rename(dst)

    side_src = src.with_name(src.name + ".json")
    meta = json.loads(side_src.read_text()) if side_src.exists() else {}
    src_entries = {e["id"]: e for e in meta.get("records", [])}
    for e in entries:
        extra = src_entries.get(e["id"], {})
        for key in ("hidden_absmax",):
            if key in extra:
                e[key] = extra[key]
    meta.update({"file": name, "sha256": digest, "bytes": dst.stat().st_size, "records_count": len(entries),
                 "tokens": n_tok, "supervised_rows": sup_rows if h.topk else 0, "records": entries,
                 "subset_of": {"file": src.name, "sha256": meta.get("sha256"), "records": len(shard.records)},
                 "subset_note": "gmtp_subset.py (Fable R2 N13): byte copy of selected records, ordinals renumbered"})
    meta.setdefault("format", "gufo.mtp-capture.shard.v1")
    for key, val in (("shard_ordinal", h.shard_ordinal), ("split", h.split), ("target_sha256", h.target_sha256),
                     ("abi_sha256", h.abi_sha256), ("producer_sha256", h.producer_sha256), ("topk", h.topk),
                     ("hidden", h.hidden), ("hc_count", h.hc_count), ("vocab", h.vocab),
                     ("features", "both" if h.mtp_hidden and h.dflash_taps else ("dflash" if h.dflash_taps else "mtp"))):
        meta[key] = val
    dst.with_name(name + ".json").write_text(json.dumps(meta, indent=1) + "\n")
    (a.out_dir / "SELECTION.json").write_text(json.dumps(
        {"source": src.name, "selected": ids, "held_out": held, "shard": name, "sha256": digest}, indent=1) + "\n")

    if not a.no_verify:
        G.verify(dst)                                                     # structure, hashes, name, sidecar
        chk = G.Shard(dst)
        for k, rid in enumerate(ids):                                     # payload identical to the source
            r0, r1 = by_id[rid], chk.records[k]
            if not (bytes(shard.tokens(r0)) == bytes(chk.tokens(r1)) and bytes(shard.sup(r0)) == bytes(chk.sup(r1))
                    and bytes(shard.hidden_raw(r0)) == bytes(chk.hidden_raw(r1)) and r0.metadata == r1.metadata):
                raise SystemExit(f"payload mismatch for {rid}")
    print(json.dumps({"file": str(dst), "sha256": digest, "records": len(ids), "tokens": n_tok,
                      "held_out": len(held), "ids": ids}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
