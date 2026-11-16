"""Reader and validator for GMTP v1 MTP-drafter capture shards.

Library use (trainer side; numpy required for arrays)::

    from read_gmtp import Shard
    shard = Shard("shard-000012-train-sha256-<hex>.mtps")
    for rec in shard.records:
        tokens = shard.tokens(rec)            # uint32 [n]
        hidden = shard.hidden(rec)            # float32 [n, hc_count, hidden] (from BF16)
        sup = shard.sup(rec)                  # structured [n - supervise_from]

DFlash2 shards (``--features dflash|both``) add ``shard.dflash(rec)``: float32
[n, 5, hidden], the plain mean of the 4 hyper-connection streams after
completed layers 5, 15, 25, 35, 45 (llama.cpp ``hc_stream_mean``), i.e. the
DFlash fc input row is ``dflash(rec)[i].reshape(-1)`` (12800 wide).
``shard.header.mtp_hidden`` is False for dflash-only shards (no ``hidden``).

``sup`` fields: ids u32[K] (logit-desc, ties id-asc; ids[0] = target greedy
token), logprob f16[K], lse f32, entropy f32 (nats), next_logprob f32 (NaN on
the last row), next_rank u32 (0xFFFFFFFF on the last row). Row r of ``sup``
describes position ``supervise_from + r``: the target's distribution of the
token at position + 1. MTP row i (hidden row i + embedding of token i+1)
predicts token i+2, supervised by sup row of position i+1.

CLI::

    python read_gmtp.py verify SHARD.mtps [--sidecar SHARD.mtps.json]
    python read_gmtp.py summary SHARD.mtps
    python read_gmtp.py selftest-check DIR      # output of w3mtp_capture --self-test-output DIR
    python read_gmtp.py selftest-write DIR      # Python reference writer (reader self-test)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

FILE_MAGIC = b"GMTPCAP1"
REC_MAGIC = b"GMTPREC1"
END_MAGIC = b"GMTPEND1"
FILE_HEADER = 512
HEADER_DIGEST = 480
REC_HEADER = 192
TRAILER = 64
ALIGN = 64
NO_NEXT = 0xFFFFFFFF
SPLITS = {0: "train", 1: "val", 2: "test", 3: "other"}


DFLASH_TAPS = (5, 15, 25, 35, 45)


def abi_text(hidden: int, hc_count: int, vocab: int, topk: int, hidden_dtype: int = 1,
             mtp_hidden: bool = True, dflash: bool = False) -> str:
    if not mtp_hidden:
        storage = "hidden-storage=none\n"
    else:
        storage = "hidden-storage=f32-le\n" if hidden_dtype == 3 else "hidden-storage=bf16-rne-le\n"
    return ("gufo.mtp-capture.abi.v1\n"
            "architecture=qwen4exp\n"
            "hidden-source=trunk-final-wide-residual-pre-hc-head\n"
            "hidden-layout=token,hc-stream,hidden\n"
            + storage
            + "alignment=row-i-after-input-token-i\n"
            "mtp-pairing=hidden-row-i+embed(token-i+1)->token-i+2\n"
            "supervision=target-head(hc_head,output)-row-i->token-i+1\n"
            "topk-order=logit-desc,id-asc\n"
            "topk-ids=u32-le\n"
            "topk-logprob=f16-rne-le\n"
            "row-scalars=lse-f32,entropy-nats-f32,next-logprob-f32,next-rank-u32\n"
            f"hidden={hidden}\nhc-count={hc_count}\nvocab={vocab}\ntopk={topk}\n"
            + ("dflash-taps=5,15,25,35,45\n"
               "dflash-tap-point=raw-residual-after-completed-layer-before-next-ple\n"
               "dflash-reduction=plain-mean-of-hc-streams:((s0+s1)+s2)+s3)*(1/hc)\n"
               "dflash-storage=bf16-rne-le\n"
               "dflash-layout=token,tap,hidden\n" if dflash else ""))


def sup_row_bytes(topk: int) -> int:
    return 0 if topk == 0 else (topk * 6 + 16 + 15) // 16 * 16


def align(value: int) -> int:
    return (value + ALIGN - 1) & ~(ALIGN - 1)


class FormatError(RuntimeError):
    pass


@dataclass
class Header:
    hidden: int
    hc_count: int
    hc_dim: int
    vocab: int
    topk: int
    sup_row_bytes: int
    max_context: int
    hidden_dtype: int
    mtp_hidden: bool
    dflash_taps: tuple
    target_sha256: str
    abi_sha256: str
    producer_sha256: str
    producer_git: str
    split: str
    shard_ordinal: int
    created_ns: int
    uuid: str


def parse_header(data: bytes) -> Header:
    if len(data) < FILE_HEADER or data[:8] != FILE_MAGIC:
        raise FormatError("not a GMTP v1 shard")
    version, _, header_bytes = struct.unpack_from("<HHI", data, 8)
    if version != 1 or header_bytes != FILE_HEADER:
        raise FormatError(f"unsupported GMTP version {version}")
    if hashlib.sha256(data[:HEADER_DIGEST]).digest() != data[HEADER_DIGEST:FILE_HEADER]:
        raise FormatError("file header digest mismatch")
    (hidden, hc_count, hc_dim, vocab, topk, row_bytes, hidden_dtype, token_dtype,
     logprob_dtype, max_context, alignment, features) = struct.unpack_from("<12I", data, 16)
    if features & ~3:
        raise FormatError(f"unknown feature mask {features}")
    mtp_hidden, dflash = not (features & 2), bool(features & 1)
    n_taps, = struct.unpack_from("<I", data, 232)
    taps = struct.unpack_from(f"<{n_taps}I", data, 236) if n_taps else ()
    if (taps != DFLASH_TAPS) if dflash else n_taps:
        raise FormatError(f"unexpected DFlash taps {taps}")
    if hidden_dtype not in (1, 3) or (token_dtype, logprob_dtype, alignment) != (1, 2, ALIGN):
        raise FormatError("unexpected storage types")
    if hc_dim != hidden * hc_count or row_bytes != sup_row_bytes(topk):
        raise FormatError("inconsistent shape fields")
    if data[64:80].rstrip(b"\0") != b"gguf-sha256-v1":
        raise FormatError("unknown identity scheme")
    abi = data[112:144].hex()
    if abi != hashlib.sha256(abi_text(hidden, hc_count, vocab, topk, hidden_dtype,
                                      mtp_hidden, dflash).encode()).hexdigest():
        raise FormatError("ABI hash differs from this reader's ABI text")
    split_code, = struct.unpack_from("<I", data, 196)
    ordinal, created = struct.unpack_from("<QQ", data, 200)
    return Header(hidden, hc_count, hc_dim, vocab, topk, row_bytes, max_context, hidden_dtype,
                  mtp_hidden, tuple(taps),
                  data[80:112].hex(), abi, data[144:176].hex(), data[176:196].hex(),
                  SPLITS.get(split_code, f"code{split_code}"), ordinal, created,
                  str(uuid.UUID(bytes=data[216:232])))


@dataclass
class Record:
    ordinal: int
    offset: int            # absolute file offset of the record header
    total_bytes: int
    n_tokens: int
    supervise_from: int
    flags: int
    metadata: dict
    token_offset: int      # absolute
    hidden_offset: int     # absolute
    sup_offset: int        # absolute
    trailer_offset: int    # absolute
    hidden_bytes: int
    dflash_offset: int     # absolute
    dflash_bytes: int
    id_sha256: str
    source_sha256: str
    token_sha256: str
    extra: dict = field(default_factory=dict)


def parse_record_header(data: bytes, offset: int, header: Header) -> Record:
    if len(data) < REC_HEADER or data[:8] != REC_MAGIC:
        raise FormatError(f"bad record magic at {offset}")
    version, _, header_bytes = struct.unpack_from("<HHI", data, 8)
    if version != 1 or header_bytes != REC_HEADER:
        raise FormatError("unsupported record version")
    total, ordinal = struct.unpack_from("<QQ", data, 16)
    n, sf, meta_bytes, flags = struct.unpack_from("<4I", data, 32)
    m_off, t_off, h_off, s_off, tr_off, h_bytes = struct.unpack_from("<6Q", data, 48)
    sup_rows = 0 if header.topk == 0 else n - sf
    want_h = n * header.hc_dim * (4 if header.hidden_dtype == 3 else 2) if header.mtp_hidden else 0
    d_off = align(h_off + h_bytes)
    d_bytes = n * len(header.dflash_taps) * header.hidden * 2
    if (m_off != REC_HEADER or t_off != align(m_off + meta_bytes)
            or h_off != align(t_off + 4 * n)
            or h_bytes != want_h
            or s_off != align(d_off + d_bytes)
            or tr_off != align(s_off + sup_rows * header.sup_row_bytes)
            or total != tr_off + TRAILER or sf > n or n == 0):
        raise FormatError(f"inconsistent record layout at {offset}")
    return Record(ordinal, offset, total, n, sf, flags, {}, offset + t_off, offset + h_off,
                  offset + s_off, offset + tr_off, h_bytes, offset + d_off, d_bytes, data[96:128].hex(),
                  data[128:160].hex(), data[160:192].hex())


class Shard:
    """Random access to one shard. Arrays are numpy memmaps (read-only)."""

    def __init__(self, path, verify: bool = False):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        with self.path.open("rb") as f:
            self.header = parse_header(f.read(FILE_HEADER))
            self.records: list[Record] = []
            pos = FILE_HEADER
            while pos < self.size:
                f.seek(pos)
                raw = f.read(REC_HEADER)
                rec = parse_record_header(raw, pos, self.header)
                if rec.ordinal != len(self.records) or pos + rec.total_bytes > self.size:
                    raise FormatError(f"record {len(self.records)} is out of order or truncated")
                meta_len = struct.unpack_from("<I", raw, 40)[0]
                rec.metadata = json.loads(f.read(meta_len).decode("utf-8"))
                f.seek(rec.trailer_offset)
                trailer = f.read(TRAILER)
                if trailer[56:64] != END_MAGIC:
                    raise FormatError(f"record {rec.ordinal} lacks its commit trailer")
                t_total, t_ord = struct.unpack_from("<QQ", trailer, 32)
                t_n, t_flags = struct.unpack_from("<II", trailer, 48)
                if (t_total, t_ord, t_n, t_flags) != (rec.total_bytes, rec.ordinal, rec.n_tokens, rec.flags):
                    raise FormatError(f"record {rec.ordinal} trailer differs from its header")
                rec.extra["record_sha256"] = trailer[:32].hex()
                if verify:
                    f.seek(pos)
                    h = hashlib.sha256()
                    remaining = rec.trailer_offset - pos
                    while remaining:
                        chunk = f.read(min(remaining, 8 << 20))
                        h.update(chunk)
                        remaining -= len(chunk)
                    if h.hexdigest() != rec.extra["record_sha256"]:
                        raise FormatError(f"record {rec.ordinal} hash mismatch")
                    f.seek(rec.token_offset)
                    if hashlib.sha256(f.read(4 * rec.n_tokens)).hexdigest() != rec.token_sha256:
                        raise FormatError(f"record {rec.ordinal} token hash mismatch")
                    if hashlib.sha256(rec.metadata["id"].encode()).hexdigest() != rec.id_sha256:
                        raise FormatError(f"record {rec.ordinal} id hash mismatch")
                    if hashlib.sha256(rec.metadata["source_id"].encode()).hexdigest() != rec.source_sha256:
                        raise FormatError(f"record {rec.ordinal} source hash mismatch")
                self.records.append(rec)
                pos += rec.total_bytes
            if pos != self.size:
                raise FormatError("trailing bytes after the last record")
        self._mm = None

    # ---- arrays (numpy) ----
    def _map(self):
        if self._mm is None:
            import numpy as np
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._mm

    def tokens(self, rec: Record):
        mm = self._map()
        return mm[rec.token_offset:rec.token_offset + 4 * rec.n_tokens].view("<u4")

    def hidden_raw(self, rec: Record):
        """BF16 bit patterns (uint16) or F32 values, shaped [n, hc_count, hidden]."""
        h = self.header
        mm = self._map()
        raw = mm[rec.hidden_offset:rec.hidden_offset + rec.hidden_bytes]
        raw = raw.view("<f4" if h.hidden_dtype == 3 else "<u2")
        return raw.reshape(rec.n_tokens, h.hc_count, h.hidden)

    def hidden(self, rec: Record):
        """Final wide residual as float32 [n, hc_count, hidden]."""
        import numpy as np
        if not self.header.mtp_hidden:
            raise FormatError("this shard was captured without the MTP hidden (features=dflash)")
        raw = self.hidden_raw(rec)
        if self.header.hidden_dtype == 3:
            return np.asarray(raw, dtype=np.float32)
        return (raw.astype(np.uint32) << 16).view(np.float32)

    def dflash_bf16(self, rec: Record):
        """DFlash2 taps as BF16 bit patterns [n, taps, hidden]."""
        h = self.header
        if not h.dflash_taps:
            raise FormatError("this shard has no DFlash features (features=mtp)")
        mm = self._map()
        raw = mm[rec.dflash_offset:rec.dflash_offset + rec.dflash_bytes].view("<u2")
        return raw.reshape(rec.n_tokens, len(h.dflash_taps), h.hidden)

    def dflash(self, rec: Record):
        """DFlash2 taps as float32 [n, taps, hidden] (stream means)."""
        import numpy as np
        return (self.dflash_bf16(rec).astype(np.uint32) << 16).view(np.float32)

    def sup_dtype(self):
        import numpy as np
        k = self.header.topk
        return np.dtype({"names": ["ids", "logprob", "lse", "entropy", "next_logprob", "next_rank"],
                         "formats": [("<u4", (k,)), ("<f2", (k,)), "<f4", "<f4", "<f4", "<u4"],
                         "offsets": [0, 4 * k, 6 * k, 6 * k + 4, 6 * k + 8, 6 * k + 12],
                         "itemsize": self.header.sup_row_bytes})

    def sup(self, rec: Record):
        rows = 0 if self.header.topk == 0 else rec.n_tokens - rec.supervise_from
        mm = self._map()
        size = rows * self.header.sup_row_bytes
        return mm[rec.sup_offset:rec.sup_offset + size].view(self.sup_dtype())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, sidecar: Path | None = None) -> dict:
    shard = Shard(path, verify=True)
    digest = file_sha256(path)
    name = path.name
    if "-sha256-" in name and name.split("-sha256-")[1].split(".")[0] != digest:
        raise FormatError("file name hash differs from content")
    if sidecar is None:
        candidate = path.with_name(path.name + ".json")
        sidecar = candidate if candidate.exists() else None
    if sidecar is not None:
        meta = json.loads(sidecar.read_text())
        h = shard.header
        expect = {"sha256": digest, "bytes": shard.size, "shard_ordinal": h.shard_ordinal,
                  "split": h.split, "target_sha256": h.target_sha256, "abi_sha256": h.abi_sha256,
                  "producer_sha256": h.producer_sha256, "topk": h.topk, "hidden": h.hidden,
                  "hc_count": h.hc_count, "vocab": h.vocab,
                  "features": "both" if h.mtp_hidden and h.dflash_taps else (
                      "dflash" if h.dflash_taps else "mtp"),
                  "records_count": len(shard.records),
                  "tokens": sum(r.n_tokens for r in shard.records)}
        for key, value in expect.items():
            if meta.get(key) != value:
                raise FormatError(f"sidecar {key}={meta.get(key)!r} differs from shard {value!r}")
        for rec, entry in zip(shard.records, meta["records"], strict=True):
            if (entry["offset"], entry["bytes"], entry["tokens"], entry["supervise_from"], entry["id"]) != (
                    rec.offset, rec.total_bytes, rec.n_tokens, rec.supervise_from, rec.metadata["id"]):
                raise FormatError(f"sidecar index differs at record {rec.ordinal}")
    return {"file": str(path), "sha256": digest, "records": len(shard.records),
            "tokens": sum(r.n_tokens for r in shard.records),
            "supervised_rows": sum(r.n_tokens - r.supervise_from for r in shard.records) if shard.header.topk else 0,
            "bytes": shard.size, "split": shard.header.split, "target_sha256": shard.header.target_sha256,
            "mtp_hidden": shard.header.mtp_hidden, "dflash_taps": list(shard.header.dflash_taps),
            "bytes_per_token": round(shard.size / max(1, sum(r.n_tokens for r in shard.records)), 1)}


# --------------------------------------------------------------------------
# Self-test: synthetic data formulas shared with mtp_capture.cpp SelfTest().
# --------------------------------------------------------------------------
ST_SHAPE = dict(hidden=16, hc_count=4, vocab=1000, topk=32, max_context=64)
ST_RECORDS = [
    {"id": "synthetic-0", "source_id": "synthetic/a.cpp", "tokens": [1, 2, 3, 4, 5, 6, 7],
     "supervise_from": 0, "metadata": {"kind": "selftest"}},
    {"id": "synthetic-1", "source_id": "synthetic/b.py", "tokens": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 999],
     "supervise_from": 3, "metadata": None},
]


def st_hidden(t: int, j: int) -> float:
    return ((t * 31 + j) % 61 - 30) / 8.0


def st_tap(k: int, t: int, s: int, j: int) -> float:
    return ((t * 7 + s * 13 + j * 3 + k * 11) % 29 - 14) / 4.0


def st_logit(row: int, v: int) -> float:
    return ((v * 2654435761 + row * 40503) % 1000003 % 2048) / 64.0 - 16.0


def st_expected_row(row: int, next_token: int | None, vocab: int, topk: int):
    logits = [st_logit(row, v) for v in range(vocab)]
    m = max(logits)
    s = sum(math.exp(x - m) for x in logits)
    lse = m + math.log(s)
    entropy = math.log(s) - sum(math.exp(x - m) * (x - m) for x in logits) / s
    order = sorted(range(vocab), key=lambda v: (-logits[v], v))[:topk]
    nxt = (logits[next_token] - lse, sum(1 for x in logits if x > logits[next_token])) if next_token is not None else None
    return order, [logits[v] - lse for v in order], lse, entropy, nxt


def check_synthetic(shard: Shard) -> None:
    import numpy as np
    h = shard.header
    assert (h.hidden, h.hc_count, h.vocab, h.topk) == (16, 4, 1000, 32), "self-test shape"
    assert bool(h.dflash_taps) == (h.shard_ordinal == 8) and h.mtp_hidden, "self-test features"
    assert len(shard.records) == 2
    for rec, spec in zip(shard.records, ST_RECORDS):
        assert rec.metadata["id"] == spec["id"] and rec.metadata["source_id"] == spec["source_id"]
        assert rec.supervise_from == spec["supervise_from"]
        assert shard.tokens(rec).tolist() == spec["tokens"]
        hid = shard.hidden(rec).reshape(rec.n_tokens, -1)
        want = np.array([[st_hidden(t, j) for j in range(h.hc_dim)] for t in range(rec.n_tokens)], np.float32)
        assert np.array_equal(hid, want), "hidden rows differ"
        if h.dflash_taps:
            got = shard.dflash(rec)
            for k in range(len(h.dflash_taps)):
                for t in range(rec.n_tokens):
                    for j in range(h.hidden):
                        acc = np.float32(st_tap(k, t, 0, j))
                        for st in range(1, h.hc_count):
                            acc = np.float32(acc + np.float32(st_tap(k, t, st, j)))
                        assert got[t, k, j] == np.float32(acc * np.float32(1.0 / h.hc_count)), "dflash tap"
        sup = shard.sup(rec)
        assert len(sup) == rec.n_tokens - rec.supervise_from
        for r, row in enumerate(sup):
            pos = rec.supervise_from + r
            nxt = spec["tokens"][pos + 1] if pos + 1 < rec.n_tokens else None
            ids, lps, lse, ent, nx = st_expected_row(pos, nxt, h.vocab, h.topk)
            assert row["ids"].tolist() == ids, f"top-k ids differ at {spec['id']} row {pos}"
            assert np.allclose(row["logprob"].astype(np.float64), lps, rtol=2e-3, atol=2e-3), "logprob"
            assert abs(float(row["lse"]) - lse) < 2e-4, ("lse", float(row["lse"]), lse)
            assert abs(float(row["entropy"]) - ent) < 2e-4, ("entropy", float(row["entropy"]), ent)
            if nx is None:
                assert math.isnan(float(row["next_logprob"])) and int(row["next_rank"]) == NO_NEXT
            else:
                assert abs(float(row["next_logprob"]) - nx[0]) < 2e-4 and int(row["next_rank"]) == nx[1]


def selftest_check(directory: Path) -> dict:
    staging = directory / "staging"
    shards = sorted(staging.glob("*.mtps"))
    if len(shards) != 2 or [Shard(x).header.shard_ordinal for x in shards] != [7, 8]:
        raise FormatError(f"expected self-test shards 7 (mtp) and 8 (both) in {staging}")
    results = []
    for shard in shards:
        result = verify(shard)
        check_synthetic(Shard(shard))
        results.append({k: result[k] for k in ("file", "bytes_per_token", "mtp_hidden", "dflash_taps")})
    return {"selftest": "PASS", "shards": results}


def selftest_write(directory: Path) -> list[Path]:
    """Reference writer for the same synthetic shards as the C++ self-test
    (ordinal 7: features=mtp, ordinal 8: features=both); exercises this reader."""
    staging = directory / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    return [_write_shard(staging, 7, False), _write_shard(staging, 8, True)]


def _bf16_bytes(values) -> bytes:
    import numpy as np
    bits = np.asarray(values, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()


def _write_shard(staging: Path, shard_ordinal: int, dflash: bool) -> Path:
    import numpy as np
    s = ST_SHAPE
    hc_dim, k, row_bytes = s["hidden"] * s["hc_count"], s["topk"], sup_row_bytes(s["topk"])
    taps = DFLASH_TAPS if dflash else ()
    abi = hashlib.sha256(abi_text(s["hidden"], s["hc_count"], s["vocab"], k, 1, True, dflash).encode()).digest()
    header = bytearray(FILE_HEADER)
    header[:8] = FILE_MAGIC
    struct.pack_into("<HHI", header, 8, 1, 0, FILE_HEADER)
    struct.pack_into("<12I", header, 16, s["hidden"], s["hc_count"], hc_dim, s["vocab"], k, row_bytes,
                     1, 1, 2, s["max_context"], ALIGN, 1 if dflash else 0)
    header[64:78] = b"gguf-sha256-v1"
    header[80:112] = bytes.fromhex("1" * 64)
    header[112:144] = abi
    header[144:176] = hashlib.sha256(b"python-selftest").digest()
    header[176:196] = b"\x22" * 20
    struct.pack_into("<IQQ", header, 196, 1, shard_ordinal, 0)
    header[216:232] = uuid.uuid4().bytes
    struct.pack_into(f"<I{len(taps)}I", header, 232, len(taps), *taps)
    header[HEADER_DIGEST:] = hashlib.sha256(bytes(header[:HEADER_DIGEST])).digest()
    out = bytearray(header)
    for ordinal, spec in enumerate(ST_RECORDS):
        n, sf = len(spec["tokens"]), spec["supervise_from"]
        meta = {"id": spec["id"], "source_id": spec["source_id"], "split": "val", "supervise_from": sf}
        if spec["metadata"] is not None:
            meta["metadata"] = spec["metadata"]
        meta_b = json.dumps(meta, separators=(",", ":")).encode()
        tok_b = struct.pack(f"<{n}I", *spec["tokens"])
        hid_b = _bf16_bytes([[st_hidden(t, j) for j in range(hc_dim)] for t in range(n)])
        dfl = np.zeros((n, len(taps), s["hidden"]), np.float32)
        for kk in range(len(taps)):
            for t in range(n):
                for j in range(s["hidden"]):
                    acc = np.float32(st_tap(kk, t, 0, j))
                    for st in range(1, s["hc_count"]):
                        acc = np.float32(acc + np.float32(st_tap(kk, t, st, j)))
                    dfl[t, kk, j] = np.float32(acc * np.float32(1.0 / s["hc_count"]))
        dfl_b = _bf16_bytes(dfl) if taps else b""
        sup_b = bytearray()
        for pos in range(sf, n):
            nxt = spec["tokens"][pos + 1] if pos + 1 < n else None
            ids, lps, lse, ent, nx = st_expected_row(pos, nxt, s["vocab"], k)
            row = bytearray(row_bytes)
            struct.pack_into(f"<{k}I", row, 0, *ids)
            row[4 * k:6 * k] = np.array(lps, np.float16).astype("<f2").tobytes()
            struct.pack_into("<fff", row, 6 * k, lse, ent, nx[0] if nx else float("nan"))
            struct.pack_into("<I", row, 6 * k + 12, nx[1] if nx else NO_NEXT)
            sup_b += row
        t_off = align(REC_HEADER + len(meta_b))
        h_off = align(t_off + len(tok_b))
        d_off = align(h_off + len(hid_b))
        s_off = align(d_off + len(dfl_b))
        tr_off = align(s_off + len(sup_b))
        total = tr_off + TRAILER
        rec = bytearray(tr_off)
        rec[:8] = REC_MAGIC
        struct.pack_into("<HHIQQ4I6Q", rec, 8, 1, 0, REC_HEADER, total, ordinal, n, sf, len(meta_b), 0,
                         REC_HEADER, t_off, h_off, s_off, tr_off, len(hid_b))
        rec[96:128] = hashlib.sha256(spec["id"].encode()).digest()
        rec[128:160] = hashlib.sha256(spec["source_id"].encode()).digest()
        rec[160:192] = hashlib.sha256(tok_b).digest()
        rec[REC_HEADER:REC_HEADER + len(meta_b)] = meta_b
        rec[t_off:t_off + len(tok_b)] = tok_b
        rec[h_off:h_off + len(hid_b)] = hid_b
        rec[d_off:d_off + len(dfl_b)] = dfl_b
        rec[s_off:s_off + len(sup_b)] = sup_b
        trailer = bytearray(TRAILER)
        trailer[:32] = hashlib.sha256(bytes(rec)).digest()
        struct.pack_into("<QQII", trailer, 32, total, ordinal, n, 0)
        trailer[56:64] = END_MAGIC
        out += rec + trailer
    digest = hashlib.sha256(bytes(out)).hexdigest()
    path = staging / f"shard-{shard_ordinal:06d}-val-sha256-{digest}.mtps"
    path.write_bytes(bytes(out))
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("shard", type=Path, nargs="+")
    p.add_argument("--sidecar", type=Path)
    p = sub.add_parser("summary")
    p.add_argument("shard", type=Path)
    p = sub.add_parser("selftest-check")
    p.add_argument("dir", type=Path)
    p = sub.add_parser("selftest-write")
    p.add_argument("dir", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "verify":
        for shard in args.shard:
            print(json.dumps(verify(shard, args.sidecar if len(args.shard) == 1 else None)))
    elif args.cmd == "summary":
        shard = Shard(args.shard)
        print(json.dumps({"header": shard.header.__dict__,
                          "records": [{"ordinal": r.ordinal, "tokens": r.n_tokens,
                                       "supervise_from": r.supervise_from, "id": r.metadata.get("id"),
                                       "source_id": r.metadata.get("source_id")} for r in shard.records]},
                         indent=1))
    elif args.cmd == "selftest-check":
        print(json.dumps(selftest_check(args.dir)))
    elif args.cmd == "selftest-write":
        for path in selftest_write(args.dir):
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
