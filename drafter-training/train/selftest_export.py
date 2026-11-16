"""Export path test on the real sidecar with a synthetic checkpoint (no training needed).

The checkpoint holds a dense full-FT change (eh_proj x 1.002, hc_head_norm + 0.01)
plus an expert LoRA on gate_exps (rank 2). export_gguf writes a new GGUF. The
test then checks:
  * write_check PASS: patched regions re-read; untouched regions equal the source;
  * the exported file parses, and only the three trained tensors differ in bytes;
  * eh_proj bytes == q8_0_bytes(base * 1.002); hc_head_norm f32 bytes exact;
  * gate_exps expert e dequantizes to base_e + B_e A_e within one Q8_0 step;
  * mtp_ref's own loader reads the exported tensors (streamed, low memory).
Memory: one expert slice at a time (< 2 GB). Disk: writes one 2.79 GB GGUF to --out.

  python selftest_export.py --mtp-gguf model/mtp-...gguf --out runs/export-test/mtp-test.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import export_gguf as X
import mtp_net as N
import quant_sim as Q
R = N.R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtp-gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    N.lower_priority()
    src = Path(a.mtp_gguf)
    g = R.GgufFile(src)
    info = g.info
    eh = R._dense(g, "blk.48.nextn.eh_proj.weight")
    hn = R._dense(g, "blk.48.nextn.hc_head_norm.weight")
    g._mm.close()
    g._fh.close()
    gen = torch.Generator().manual_seed(7)
    A = torch.randn(512, 2, 2560, generator=gen) * 0.02
    B = torch.randn(512, 640, 2, generator=gen) * 0.02
    opts = N.opts_to_dict(N.NetOptions(dense_mode="full", expert_mode="lora", expert_lora_rank=2, lora_alpha=2.0))
    ck = Path(a.out).with_suffix(".ckpt.pt")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net_options": opts, "trainable": {"eh_proj": eh * 1.002, "hc_head_norm": hn + 0.01,
                                                   "lora.gate_exps.A": A, "lora.gate_exps.B": B}}, ck)
    rc = X.main(["--mtp-gguf", str(src), "--checkpoint", str(ck), "--out", a.out, "--threads", str(a.threads)])
    rep = json.loads(Path(a.out + ".export.json").read_text())
    res = {"export_rc": rc, "write_check": rep.get("write_check")}
    patched = sorted(n for n, e in rep["tensors"].items() if e["action"] == "patched")
    res["patched"] = patched
    # byte-level diff of every tensor region vs the source
    differ = []
    for t in info["tensors"]:
        off, nb = X.region(info, t)
        if X.region_sha(src, off, nb) != X.region_sha(Path(a.out), off, nb):
            differ.append(t["name"])
    res["regions_differing"] = sorted(differ)
    res["only_trained_differ"] = sorted(differ) == patched == sorted(
        ["blk.48.ffn_gate_exps.weight", "blk.48.nextn.eh_proj.weight", "blk.48.nextn.hc_head_norm.weight"])
    t_eh = [t for t in info["tensors"] if t["name"] == "blk.48.nextn.eh_proj.weight"][0]
    off, nb = X.region(info, t_eh)
    res["eh_proj_bytes_exact"] = X.read_bytes(Path(a.out), off, nb) == Q.q8_0_bytes(eh * 1.002)
    t_hn = [t for t in info["tensors"] if t["name"] == "blk.48.nextn.hc_head_norm.weight"][0]
    off, nb = X.region(info, t_hn)
    res["hc_head_norm_exact"] = X.read_bytes(Path(a.out), off, nb) == (hn + 0.01).numpy().astype("<f4").tobytes()
    # gate_exps through mtp_ref's GGUF reader (streamed) vs base + B A
    g2 = R.GgufFile(Path(a.out))
    new_q = N._q8file(g2, "blk.48.ffn_gate_exps.weight")
    g0 = R.GgufFile(src)
    old_q = N._q8file(g0, "blk.48.ffn_gate_exps.weight")
    worst = 0.0
    exact = True
    t_ge = [t for t in info["tensors"] if t["name"] == "blk.48.ffn_gate_exps.weight"][0]
    off_ge, nb_ge = X.region(info, t_ge)
    per = nb_ge // 512
    for e in (0, 1, 255, 511):
        want = old_q.dequant(e) + (B[e] @ A[e]) * 1.0
        exact &= X.read_bytes(Path(a.out), off_ge + e * per, per) == Q.q8_0_bytes(want)
        got = new_q.dequant(e)
        step = want.abs().unflatten(-1, (-1, 32)).amax(-1, keepdim=True) / 127
        worst = max(worst, float(((got - want).abs().unflatten(-1, (-1, 32)) / step.clamp(min=1e-12)).max()))
    res["gate_exps_max_err_in_q8_steps"] = round(worst, 4)   # <= 0.5 + 127*2^-11 (f16 scale rounding)
    res["gate_exps_bytes_exact"] = bool(exact)
    res["delta_vs_quant"] = {n: {"delta_rel_rms": rep["tensors"][n]["delta_rel_rms"],
                                 "quant_rel_rms": rep["tensors"][n]["quant_rel_rms"]} for n in patched}
    ok = (rc == 0 and res["write_check"] == "PASS" and res["only_trained_differ"] and res["eh_proj_bytes_exact"]
          and res["hc_head_norm_exact"] and exact and worst <= 0.5 + 127 * 2 ** -11 + 1e-4)
    res["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(res, indent=1))
    Path(a.out + ".selftest.json").write_text(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
