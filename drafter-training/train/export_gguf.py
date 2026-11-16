"""Write a fine-tuned MTP head back into the sidecar GGUF the engine loads.

Method: byte copy of the original `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
(header, metadata, tensor table, alignment and padding untouched), then patch
each *trained* tensor's data region in place with the same name, shape and
type. Q8_0 uses quant_sim.q8_0_bytes, a bit-exact port of ggml
quantize_row_q8_0_ref; F32 tensors are written as raw f32. Every other tensor,
including the BF16 indexer and anything not in the checkpoint, keeps its
original bytes. LoRA adapters are merged (W + alpha/r * B A) before
quantization. Memory stays small: one expert or tensor at a time.

Checks (all written to <out>.export.json):
  * noop  : `--noop` re-quantizes the original dequantized values of every
            tensor and must reproduce the original bytes bit for bit (Q8_0 is
            idempotent for grids produced by ggml). With --noop and --out the
            written file must have the same sha256 as the original.
  * write : after writing, each patched region is re-read and its sha256
            compared with the in-memory bytes; every untouched region is
            compared with the original file's region.
  * error : per trained tensor, relative RMS of (dequant(Q8_0(W_ft)) - W_ft) and
            of the fine-tune delta itself, to show whether quantization erases the
            update (delta << quant step).
  * forward (`--forward-check`, cloud): MtpNet from the checkpoint
            (unquantized f32) vs MtpNet loaded from the exported GGUF, and
            mtp_ref.MtpDrafter on the exported GGUF. Reports per-depth logits rel
            RMS and argmax agreement on a val record.

Host (no model load; <1 GB; a sample of tensors/experts only):
  python export_gguf.py --noop --max-experts 4 --threads 2
Cloud:
  python export_gguf.py --noop                                   # full no-op, all 1,536 expert slices
  python export_gguf.py --checkpoint runs/s1a/ckpt/step-001200.pt --out runs/s1a/export/mtp-ft-Q8_0.gguf \
      --forward-check --shard /data/val/<shard>.mtps --frozen-bundle /data/model/frozen-target.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import mtp_net as N
import quant_sim as Q
R = N.R

Q8_0, F32, BF16 = 8, 0, 30


def region(info: dict, t: dict) -> tuple[int, int]:
    from tools.gufo.gguf import GGML_TYPE_BLOCK  # via mtp_ref's sys.path
    block, size = GGML_TYPE_BLOCK[t["type"]]
    n = int(np.prod(t["shape"]))
    return info["data_offset"] + t["offset"], n // block * size


def read_bytes(path: Path, off: int, n: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


def region_sha(path: Path, off: int, n: int, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        while n:
            b = f.read(min(chunk, n))
            h.update(b)
            n -= len(b)
    return h.hexdigest()


def attr_of(name: str) -> str | None:
    for attr, gname in N.ATTR_TO_GGUF.items():
        if gname == name:
            return attr
    return None


class NewWeights:
    """Fine-tuned values per GGUF tensor from a train_mtp checkpoint (or none)."""

    def __init__(self, ckpt_path: str | None):
        self.state, self.scale = {}, {}
        self.opts = None
        if ckpt_path:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.state = ck["trainable"]
            self.opts = ck["net_options"]
            r, re, alpha = self.opts["lora_rank"], self.opts["expert_lora_rank"], self.opts["lora_alpha"]
            for attr in N.ATTR_TO_GGUF:
                if f"lora.{attr}.A" in self.state:
                    self.scale[attr] = alpha / (re if attr in N.EXPERTS else r)

    def changed(self, attr: str) -> bool:
        return attr in self.state or f"lora.{attr}.A" in self.state

    def value(self, attr: str, base: torch.Tensor, expert: int | None = None) -> torch.Tensor:
        """base: original dequantized f32 (whole tensor, or one expert slice)."""
        w = base
        if attr in self.state:
            full = self.state[attr].float()
            w = full if expert is None else full[expert]
        if f"lora.{attr}.A" in self.state:
            A, B = self.state[f"lora.{attr}.A"].float(), self.state[f"lora.{attr}.B"].float()
            if expert is not None:
                A, B = A[expert], B[expert]
            w = w + (B @ A) * self.scale[attr]
        return w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtp-gguf", default=str(R.DEFAULT_MTP))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None, help="output GGUF (omit for an in-memory check only)")
    ap.add_argument("--noop", action="store_true", help="requantize the original weights; expect bit identity")
    ap.add_argument("--max-experts", type=int, default=0, help="noop sample: experts per expert tensor (0 = all)")
    ap.add_argument("--tensors", nargs="*", default=None, help="restrict to these GGUF tensor names")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--forward-check", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--frozen-bundle", default=None)
    ap.add_argument("--trunk-dir", default=str(R.DEFAULT_TRUNK_DIR))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rss-cap-gb", type=float, default=0.0, help="exit if resident memory exceeds (host: 2.8)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    print("priority", N.lower_priority(), flush=True)
    if a.rss_cap_gb:
        N.start_rss_watchdog(a.rss_cap_gb)
    if a.out and a.noop and (a.max_experts or a.tensors):
        raise SystemExit("--out with a sampled --noop would write a partial file; drop --out or the sampling")
    if not a.noop and not a.checkpoint:
        raise SystemExit("need --checkpoint (or --noop)")
    t0 = time.time()
    src = Path(a.mtp_gguf)
    g = R.GgufFile(src)
    info = g.info
    g._mm.close()
    g._fh.close()
    nw = NewWeights(None if a.noop else a.checkpoint)
    out = Path(a.out) if a.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".partial")
        shutil.copyfile(src, tmp)
        fo = open(tmp, "r+b")
    report = {"source": str(src), "checkpoint": a.checkpoint, "noop": a.noop, "tensors": {}}
    for t in info["tensors"]:
        name = t["name"]
        if a.tensors and name not in a.tensors:
            continue
        attr = attr_of(name)
        off, nbytes = region(info, t)
        typ = t["type"]
        entry = {"type": t["type_name"], "shape": t["shape"], "bytes": nbytes}
        train = attr is not None and attr not in N.NO_TRAIN and (a.noop or nw.changed(attr))
        if not train or typ not in (Q8_0, F32):
            entry["action"] = "copied"
            report["tensors"][name] = entry
            continue
        cols = t["shape"][-1]
        lead = t["shape"][0] if len(t["shape"]) == 3 else None
        slices = range(lead) if lead else [None]
        if lead and a.noop and a.max_experts:
            slices = range(min(lead, a.max_experts))
        per_slice = nbytes // (lead or 1)
        h_new, mism_blocks, tot_blocks = hashlib.sha256(), 0, 0
        se = sq = sd = sw = 0.0
        for e in slices:
            soff = off + (e or 0) * per_slice
            orig = read_bytes(src, soff, per_slice)
            if typ == Q8_0:
                base = Q.q8_0_from_bytes(orig, cols)
            else:
                base = torch.from_numpy(np.frombuffer(orig, np.float32).copy()).reshape(-1, cols)
            w = base if a.noop else nw.value(attr, base.reshape(base.shape) if lead else
                                             base.reshape(t["shape"]), e).reshape(-1, cols).float()
            if typ == Q8_0:
                new = Q.q8_0_bytes(w)
                deq = Q.q8_0_from_bytes(new, cols)
                se += float(((deq - w) ** 2).sum())
                sq += float((w ** 2).sum())
                sd += float(((w - base) ** 2).sum())
                sw += float((base ** 2).sum())
                a_ = np.frombuffer(new, np.uint8).reshape(-1, 34)
                b_ = np.frombuffer(orig, np.uint8).reshape(-1, 34)
                mism_blocks += int((a_ != b_).any(1).sum())
                tot_blocks += a_.shape[0]
            else:
                new = w.numpy().astype("<f4").tobytes()
                b_ = np.frombuffer(orig, np.uint8)
                mism_blocks += int((np.frombuffer(new, np.uint8) != b_).any())
                tot_blocks += 1
                sd += float(((w - base) ** 2).sum())
                sw += float((base ** 2).sum())
            h_new.update(new)
            if out:
                fo.seek(soff)
                fo.write(new)
        entry.update(action="noop-requantized" if a.noop else "patched", slices=len(slices),
                     sha256_new=h_new.hexdigest(), differing_blocks=mism_blocks, blocks=tot_blocks,
                     quant_rel_rms=(se / sq) ** 0.5 if sq else 0.0,
                     delta_rel_rms=(sd / sw) ** 0.5 if sw else 0.0)
        report["tensors"][name] = entry
        print(f"{name}: {entry['action']} differing {mism_blocks}/{tot_blocks} "
              f"quant {entry['quant_rel_rms']:.2e} delta {entry['delta_rel_rms']:.2e}", flush=True)
    if out:
        fo.close()
        # write check: patched regions re-read; untouched regions equal to the source
        bad = []
        for t in info["tensors"]:
            e = report["tensors"].get(t["name"])
            if e is None:
                continue
            off, nbytes = region(info, t)
            h = region_sha(tmp, off, nbytes)
            if e["action"] == "copied":
                if h != region_sha(src, off, nbytes):
                    bad.append(t["name"])
            elif h != e["sha256_new"]:
                bad.append(t["name"])
        report["write_check"] = "PASS" if not bad else {"FAIL": bad}
        tmp.replace(out)
        report["out"] = str(out)
        report["out_sha256"] = N.file_sha256(out)
        report["source_sha256"] = N.file_sha256(src)
        if a.noop:
            report["noop_file_identical"] = report["out_sha256"] == report["source_sha256"]
    if a.noop:
        diff = sum(e.get("differing_blocks", 0) for e in report["tensors"].values())
        report["noop_verdict"] = "PASS (bit-identical)" if diff == 0 else f"DIFF ({diff} blocks)"
    if a.forward_check and out:
        report["forward_check"] = forward_check(a, out)
    report["seconds"] = round(time.time() - t0, 1)
    dest = Path(str(out) + ".export.json") if out else Path(f"export-check-{int(time.time())}.json")
    dest.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "tensors"}, indent=1))
    return 0 if report.get("write_check", "PASS") == "PASS" and "DIFF" not in report.get("noop_verdict", "") else 1


def forward_check(a, exported: Path, rows: int = 48, depth: int = 3) -> dict:
    """Cloud only (loads the full head twice): trained f32 net vs the exported Q8_0 file."""
    dev = torch.device(a.device)
    target = (N.FrozenTarget.from_bundle(Path(a.frozen_bundle)) if a.frozen_bundle
              else N.FrozenTarget.from_trunk(Path(a.trunk_dir)))
    shard = N.read_gmtp.Shard(a.shard)
    rd = N.truncate(N.read_record(shard, shard.records[0]), rows)
    hidden = rd.hidden_tensor(dev, torch.float32)
    tokens = torch.from_numpy(rd.tokens).to(dev)

    def run(net):
        got = {}

        def on_depth(j, res):
            lg = N.full_logits(net, net.head_mixed(res))
            got[j] = lg
            return lg.argmax(-1)
        with torch.no_grad():
            net.unrolled(hidden, tokens, depth, "greedy", on_depth)
        return got

    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        o = N.opts_from_dict(ck["net_options"])
        o.decode, o.qat, o.param_dtype, o.expert_dtype, o.expert_storage = False, False, torch.float32, torch.float32, "dense"
        ft = N.build_net(Path(a.mtp_gguf), target, o, dev)
        ft.load_trainable(ck["trainable"])
        ref_logits = run(ft)
        del ft
    else:
        ref_logits = None
    o2 = N.NetOptions(decode=False, dense_mode="frozen", expert_mode="frozen",
                      expert_storage="dense", expert_dtype=torch.float32)
    ex = N.build_net(exported, target, o2, dev)
    ex_logits = run(ex)
    res = {"record": rd.rid, "rows": rows, "depth": depth, "depths": []}
    for j in sorted(ex_logits):
        d = {"depth": j}
        if ref_logits is not None:
            r_, e_ = ref_logits[j].double(), ex_logits[j].double()
            d["rel_rms_vs_trained_f32"] = float((e_ - r_).norm() / r_.norm())
            d["argmax_agree_vs_trained_f32"] = float((e_.argmax(-1) == r_.argmax(-1)).float().mean())
        res["depths"].append(d)
    # loader check: mtp_ref itself reads the exported file (depth-1 logits on the same rows)
    shared = N._StreamShared(target)
    mref = R.MtpDrafter(exported, shared=shared, decode=False)
    T = rows - depth
    with torch.no_grad():
        lr, _ = mref.block_unrolled(hidden[:T].cpu(), tokens[1:T + 1].cpu(), 1, feed="greedy")
    e1 = ex_logits[1][:T].double().cpu()
    res["mtp_ref_loader_rel_rms"] = float((e1 - lr[0].double()).norm() / lr[0].double().norm())
    return res


if __name__ == "__main__":
    raise SystemExit(main())
