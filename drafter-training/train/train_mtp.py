"""Stage-1 fine-tune of the shared Q8_0 MTP head on captured target distributions.

Model: mtp_net.MtpNet (validated re-implementation of ../reference/mtp_ref.py),
weights from the sidecar GGUF (Q8_0 dequantized; trainable masters f32, compute
bf16), token_embd/output frozen (shared with the trunk; never exported).

Objective (chained drafting exactly as gufo drafts): per record, depths 1..D
(D <= 7 = kMaxMtpDraftTokens) with the model's own carried residual as the next
depth's hidden and draft KV visible only to its own chain ("training-time
test"). Depth j>=2 is fed the corpus token t_{p+j} (teacher feed), so the
target's captured distribution at position p+j is the exact label for that
prefix; rows whose fed tokens are off the target's greedy path (never reached
at inference, where only accepted = greedy drafts are fed) get --offpath-weight.
Loss per row: kl_w * KL(target top-32 + residual-mass bucket || student) +
ce_w * CE(target argmax); depth weights decay (--depth-decay); gradients flow
through the whole chain (BPTT) unless --detach-chain.

Run (cloud GPU; see TRAIN-YOUR-OWN-DRAFTER):
  python train_mtp.py --device cuda --mtp-gguf /data/model/mtp.gguf --frozen-bundle /data/model/frozen-target.pt \
     --shards "/data/train/*.mtps" --val-shards "/data/val/*.mtps" --out /data/runs/s1a \
     --dense-mode full --expert-mode lora --depth 7 --rate-usd-per-hour 1.57 --max-usd 40
  python train_mtp.py ... --benchmark-steps 20          # G1 throughput gate: tokens/s + projected cost, exits
  python train_mtp.py ... --resume auto                 # continue from runs/s1a/ckpt/latest.json
  touch /data/runs/s1a/STOP                             # graceful stop (checkpoint, exit 0)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401

import mtp_net as N
import stage2_data as S2  # [stage2] mixture weights, response windows, eval sets, early stop (all off by default)
R = N.R


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class RecordStream:
    """Deterministic shuffled record order with a resumable cursor, growable by blocks.

    Shards are grouped in arrival *blocks*. Block 0 is the shard list at start; with
    --refresh-shards-steps, shards that appear later (e.g. still being pushed to
    staging) form new blocks appended at the END of the current epoch's order. That
    keeps every earlier cursor position valid. Within a block, shard order is
    permuted per (epoch, block) and records are shuffled within windows of
    --shuffle-window shards (0 = the whole block). Checkpoints store the block
    lists. A resume requires that every saved shard still exists; shards present
    at resume but unknown to the checkpoint form a new block. A background thread
    prefetches decoded records."""

    def __init__(self, patterns: list[str], seed: int, window: int, max_tokens: int, prefetch: int = 4,
                 blocks: list[list[str]] | None = None, shard_weights: str = "", window_long: bool = False):
        self.patterns, self.seed, self.window, self.max_tokens, self.prefetch = patterns, seed, window, max_tokens, prefetch
        # [stage2] mixture weights / response windows (stage2_data.py); both off = stage-1 order exactly
        self.weight_rules = S2.parse_shard_weights(shard_weights)
        self.window_long = window_long
        self.weights: dict[str, float] = {}
        self.windows: dict[str, list] = {}
        self._open: dict = {}
        self.counts: dict[str, int] = {}
        self.paths: dict[str, Path] = {}
        self.blocks: list[list[str]] = []
        self._lock = threading.Lock()
        current = {p.name: p for p in N.resolve_shards(patterns)}
        if blocks:
            missing = [n for b in blocks for n in b if n not in current]
            if missing:
                raise SystemExit(f"resume refused: {len(missing)} checkpoint shards are missing, e.g. {missing[:2]}")
            for b in blocks:
                self._add_block([current[n] for n in b])
        self.refresh()

    def _add_block(self, paths: list[Path]) -> None:
        for p in paths:
            self.paths[p.name] = p
            self.counts[p.name] = len(self.shard(p).records)
            self.weights[p.name] = S2.shard_weight(p, self.weight_rules)                 # [stage2]
            if self.window_long:                                                        # [stage2]
                self.windows[p.name] = [S2.record_windows(r.metadata, r.n_tokens, self.max_tokens)
                                        for r in self.shard(p).records]
        with self._lock:
            self.blocks.append([p.name for p in paths])

    def refresh(self) -> int:
        """Adds shards that appeared since the last call as one new block; returns how many."""
        known = set(self.paths)
        new = [p for p in N.resolve_shards(self.patterns) if p.name not in known]
        if new:
            self._add_block(sorted(new, key=lambda q: q.name))
        return len(new)

    @property
    def total_records(self) -> int:
        return sum(self.counts.values())

    @property
    def stage2(self) -> bool:
        return bool(self.weight_rules) or self.window_long

    @property
    def entries_per_epoch(self) -> int:
        """[stage2] training items per epoch (records x windows x mixture repeats); = total_records in stage 1."""
        if not self.stage2:
            return self.total_records
        return sum(len(self.block_order(0, b)) for b in range(len(self.blocks)))

    def config(self) -> dict:
        return {"shard_weights": self.weight_rules, "window_long": self.window_long,
                "max_tokens": self.max_tokens if self.window_long else None}

    @property
    def fingerprint(self) -> str:
        return N.hashlib.sha256("\n".join("|".join(b) for b in self.blocks).encode()).hexdigest()

    def shard(self, p):
        s = self._open.get(p)
        if s is None:
            s = self._open[p] = N.read_gmtp.Shard(p)
        return s

    def block_order(self, epoch: int, b: int) -> list[tuple[str, int]]:
        names = self.blocks[b]
        rng = np.random.default_rng(self.seed + 1000003 * epoch + 7919 * b)
        order = [names[i] for i in rng.permutation(len(names))]
        win = self.window or len(order)
        out = []
        for w0 in range(0, len(order), win):
            blk = [(n, r) for n in order[w0:w0 + win] for r in range(self.counts[n])]
            if self.window_long:                                                        # [stage2]
                blk = [(n, r, w) for n, r in blk for w in range(len(self.windows[n][r]))]
            if self.weight_rules:                                                       # [stage2]
                blk = S2.repeat_entries(blk, [self.weights[e[0]] for e in blk], rng)
            out += [blk[i] for i in rng.permutation(len(blk))]
        return out

    def iterate(self, epoch: int, start: int, epochs: int):
        """Yields (epoch, global_index, RecordData) from the cursor onwards. Blocks added
        by refresh() while an epoch is running are appended to that epoch."""
        q: queue.Queue = queue.Queue(self.prefetch)
        stop = threading.Event()

        def work():
            try:
                for ep in range(epoch, epochs):
                    base, b = 0, 0
                    while True:
                        with self._lock:
                            if b >= len(self.blocks):
                                break
                        order = self.block_order(ep, b)
                        for j, item in enumerate(order):
                            name, ri = item[0], item[1]
                            i = base + j
                            if ep == epoch and i < start:
                                continue
                            if stop.is_set():
                                return
                            s = self.shard(self.paths[name])
                            rd = N.read_record(s, s.records[ri])
                            if len(item) == 3:                                          # [stage2] response window
                                rd = S2.slice_record(rd, *self.windows[name][ri][item[2]])
                            elif self.max_tokens:
                                rd = N.truncate(rd, self.max_tokens)
                            q.put((ep, i, rd))
                        base += len(order)
                        b += 1
                q.put(None)
            except Exception as exc:  # surface in the main thread
                q.put(exc)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()


def parse_kind_weights(spec: str) -> np.ndarray:
    w = np.ones(max(N.KIND_CODES.values()) + 1, np.float32)
    for part in filter(None, (spec or "").split(",")):
        k, v = part.split("=")
        w[N.KIND_CODES[k.strip()]] = float(v)
    return w


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def chunk_loss(x, W, ids, lp, w, kl_w: float, ce_w: float):
    """x [r, H] head input (bf16), W [V, H] frozen head. ids/lp [r, K], w [r].
    Returns (weighted loss sum, kl sum, ce sum, top1 hits, weight sum) - the last
    four detached."""
    z = F.linear(x, W).float()
    lse = torch.logsumexp(z, -1, keepdim=True)
    idc = ids.clamp(min=0)
    logq = z.gather(-1, idc) - lse                                  # [r, K]
    p = lp.exp()
    r_mass = (1.0 - p.sum(-1)).clamp(min=0.0)
    q_top = logq.exp().sum(-1)
    logq_other = torch.log((1.0 - q_top).clamp(min=1e-6))
    kl = (p * (lp - logq)).sum(-1) + torch.where(
        r_mass > 1e-6, r_mass * (torch.log(r_mass.clamp(min=1e-12)) - logq_other), torch.zeros_like(r_mass))
    ce = -logq[:, 0]
    per = kl_w * kl + ce_w * ce
    with torch.no_grad():
        hit = (z.argmax(-1) == idc[:, 0]).float()
    return ((per * w).sum(), (kl.detach() * w).sum(), (ce.detach() * w).sum(), (hit * w).sum(), w.sum())


def chunk_loss_ext(x, W, ids, lp, w, kl_w: float, ce_w: float, margin_w: float = 0.0, margin: float = 2.0):
    """[ablation] chunk_loss + per-row extras: student argmax ids (no grad) and log q(target greedy)
    (with grad; the expected-accepted-length term). The 5 sums are computed exactly as chunk_loss."""
    z = F.linear(x, W).float()
    lse = torch.logsumexp(z, -1, keepdim=True)
    idc = ids.clamp(min=0)
    logq = z.gather(-1, idc) - lse
    p = lp.exp()
    r_mass = (1.0 - p.sum(-1)).clamp(min=0.0)
    q_top = logq.exp().sum(-1)
    logq_other = torch.log((1.0 - q_top).clamp(min=1e-6))
    kl = (p * (lp - logq)).sum(-1) + torch.where(
        r_mass > 1e-6, r_mass * (torch.log(r_mass.clamp(min=1e-12)) - logq_other), torch.zeros_like(r_mass))
    ce = -logq[:, 0]
    per = kl_w * kl + ce_w * ce
    if margin_w > 0:                                  # [ablation B3] hinge on the target-argmax logit margin
        zg = z.gather(-1, idc[:, :1]).squeeze(-1)
        z_other = z.scatter(-1, idc[:, :1], float("-inf")).max(-1).values
        per = per + margin_w * torch.relu(margin - (zg - z_other))
    with torch.no_grad():
        am = z.argmax(-1)
        hit = (am == idc[:, 0]).float()
    return ((per * w).sum(), (kl.detach() * w).sum(), (ce.detach() * w).sum(), (hit * w).sum(), w.sum(),
            am, logq[:, 0])


def expected_len_loss(lq: list, valid: list, w1, depth: int):
    """[ablation] -E[accepted drafts]/depth with E[p] = sum_k prod_{j<=k} q_j(g) over the anchors p of a
    record (lq[j-1]: [R_j] log q of the target greedy token at depth j, row p = anchor p; valid[j-1]:
    [R_j] bool, label observable and supervised). The product stops at the first invalid depth."""
    R1 = lq[0].shape[0]
    cum = torch.zeros(R1, device=lq[0].device)
    alive = torch.ones(R1, dtype=torch.bool, device=lq[0].device)
    E = torch.zeros(R1, device=lq[0].device)
    for l, v in zip(lq, valid):
        Rj = l.shape[0]
        cum = cum[:Rj] + l
        alive = alive[:Rj] & v
        E = E + F.pad(torch.where(alive, cum.exp(), torch.zeros_like(cum)), (0, R1 - Rj))
    return -(E * w1).sum() / (w1.sum().clamp(min=1e-6) * depth)


class DepthLoss:
    def __init__(self, net, W, args, kind_w):
        self.net, self.W, self.a, self.kind_w = net, W, args, kind_w
        # [ablation] accepted-chain weighting / expected-accepted-length loss (all off by default)
        self.chain_mode = getattr(args, "chain_weighting", "none")
        self.offchain_w = getattr(args, "offchain_weight", 1.0)
        self.elen_w = getattr(args, "elen_weight", 0.0)
        self.margin_w = getattr(args, "margin_weight", 0.0)
        self.margin = getattr(args, "margin", 2.0)
        self.surv_floor = getattr(args, "survival_floor", 0.2)
        self.p1_gamma = getattr(args, "p1_gamma", 0.0)
        self.ext = self.chain_mode != "none" or self.elen_w > 0 or self.margin_w > 0

    def record(self, rd: N.RecordData, device):
        n = rd.n
        t, g = rd.tokens, rd.greedy
        tg = np.zeros(n, bool)
        tg[:-1] = (t[1:] == g[:-1]) & (g[:-1] >= 0)
        self.rd, self.tg, self.dev = rd, tg, device
        self.ids_all = torch.from_numpy(rd.sup_ids).to(device)
        self.lp_all = torch.from_numpy(rd.sup_lp).to(device)
        self.stats = []
        self.loss = 0.0
        self.wsum = 0.0
        self.chain_ok = None                          # [ablation] student drafts 1..j-1 all == target greedy
        self.surv = None                              # [ablation B1] prod_{i<j} q_i(target greedy), detached
        self.lq, self.valid, self.w1 = [], [], None

    def extra(self):
        """[ablation] the expected-accepted-length term (0.0 when off; added after the KL/CE mean)."""
        if self.elen_w <= 0 or not self.lq:
            return 0.0
        return self.elen_w * expected_len_loss(self.lq, self.valid, self.w1, len(self.lq))

    def weights(self, j: int, Rj: int) -> torch.Tensor:
        rd = self.rd
        p = np.arange(Rj)
        lab = p + j
        w = rd.supervised[lab].astype(np.float32) * self.kind_w[rd.kinds[lab]]
        on = np.ones(Rj, bool)
        for i in range(1, j):
            on &= self.tg[p + i]
        w *= np.where(on, 1.0, self.a.offpath_weight).astype(np.float32)
        if self.p1_gamma:                             # [ablation A5] confident-target weighting p1^gamma
            w *= np.exp(rd.sup_lp[lab, 0].astype(np.float64) * self.p1_gamma).astype(np.float32)
        return torch.from_numpy(w).to(self.dev), int(on.sum())

    def __call__(self, j: int, res: torch.Tensor):
        a = self.a
        Rj = res.shape[0]
        w, on = self.weights(j, Rj)
        ids, lp = self.ids_all[j:j + Rj], self.lp_all[j:j + Rj]
        if self.ext:                                  # [ablation]
            w_base = w
            if self.chain_mode == "accepted" and self.chain_ok is not None:
                w = w * torch.where(self.chain_ok[:Rj], 1.0, self.offchain_w)
            if self.chain_mode == "survival" and self.surv is not None:   # [ablation B1] P(chain alive), floored
                w = w * self.surv[:Rj].clamp(min=self.surv_floor)
        mixed = self.net.head_mixed(res)
        x = self.net.head_input(mixed)
        tot = [torch.zeros((), device=res.device) for _ in range(5)]
        ams, lqs = [], []
        fn = chunk_loss_ext if self.ext else chunk_loss
        for r0 in range(0, Rj, a.head_chunk):
            r1 = min(Rj, r0 + a.head_chunk)
            args_ = (x[r0:r1], self.W, ids[r0:r1], lp[r0:r1], w[r0:r1], a.kl_weight, a.ce_weight)
            if self.ext:
                args_ = args_ + (self.margin_w, self.margin)
            if getattr(a, "no_head_checkpoint", False):   # [stage2] keep f32 logits for backward: 2 head GEMMs, not 3
                out = fn(*args_)
            else:
                out = torch.utils.checkpoint.checkpoint(fn, *args_, use_reentrant=False)
            tot = [u + v for u, v in zip(tot, out[:5])]
            if self.ext:
                ams.append(out[5])
                lqs.append(out[6])
        if self.ext:                                  # [ablation] chain bookkeeping for the next depth / E-len term
            am = torch.cat(ams)
            okj = (am == ids[:, 0]) & (ids[:, 0] >= 0)
            self.chain_ok = okj if self.chain_ok is None else (self.chain_ok[:Rj] & okj)
            if self.chain_mode == "survival":
                qg = torch.cat(lqs).detach().exp().masked_fill(ids[:, 0] < 0, 0.0)
                self.surv = qg if self.surv is None else self.surv[:Rj] * qg
                tot[4] = w_base.sum()                 # normalise by the unweighted rows: survival sets the depth weight
            if self.elen_w > 0:
                on_t = torch.from_numpy(self._on_mask(j, Rj)).to(self.dev)
                self.lq.append(torch.cat(lqs))
                self.valid.append(on_t & (w_base > 0))
                if j == 1:
                    self.w1 = w_base
        dw = a.depth_decay ** (j - 1)
        wsum = tot[4].clamp(min=1e-6)
        self.loss = self.loss + dw * tot[0] / wsum
        self.wsum += dw
        # [stage2] per-depth stats stay on the device (no host sync per depth); floats at log time
        self.stats.append({"depth": j, "kl": (tot[1] / wsum).detach(), "ce": (tot[2] / wsum).detach(),
                           "top1": (tot[3] / wsum).detach(), "rows": Rj, "onpath": on})
        return None

    def _on_mask(self, j: int, Rj: int) -> np.ndarray:
        p = np.arange(Rj)
        on = np.ones(Rj, bool)
        for i in range(1, j):
            on &= self.tg[p + i]
        return on

    def stats_floats(self) -> list:
        return [{k: (float(v) if torch.is_tensor(v) else v) for k, v in s.items()} for s in self.stats]


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_ckpt(out: Path, net, opt, sched, state: dict, args, keep: int) -> Path:
    ck_dir = out / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    path = ck_dir / f"step-{state['step']:06d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"format": "gufo-mtp-stage1-ckpt-v1", "net_options": N.opts_to_dict(net.opt), "args": vars(args),
                "trainable": net.trainable_state(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "state": state,
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                        "python": random.getstate(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}, tmp)
    os.replace(tmp, path)
    (ck_dir / "latest.json.tmp").write_text(json.dumps({"path": path.name, "step": state["step"],
                                                        "time": time.time()}))
    os.replace(ck_dir / "latest.json.tmp", ck_dir / "latest.json")
    cks = sorted(ck_dir.glob("step-*.pt"))
    for old in cks[:-keep] if keep > 0 else []:
        old.unlink()
    return path


def merge_expert_lora(net, state: dict, ck_opts: dict) -> list:
    """[ablation] Initialising a full-expert run (--expert-mode full) from an expert-LoRA checkpoint:
    W_full = W_base (this run's --mtp-gguf, i.e. the sidecar the LoRA run started from) + (B A) * alpha/r,
    the same merge export_gguf applies. Rewrites `state` in place; returns the merged tensor names."""
    merged = []
    for attr in N.EXPERTS:
        ka, kb = f"lora.{attr}.A", f"lora.{attr}.B"
        if ka in state and kb in state and attr not in state and getattr(net, attr).requires_grad:
            r = state[ka].shape[-2]
            scale = float(ck_opts.get("lora_alpha", 16.0)) / r
            w = getattr(net, attr).detach().float().cpu()
            w = w + torch.bmm(state[kb].float(), state[ka].float()) * scale
            state[attr] = w
            merged.append(attr)
    own = dict(net.named_parameters())
    for k in [k for k in state if k not in own]:      # e.g. expert LoRA keys a full-expert net does not have
        del state[k]
    return merged


def save_best(out: Path, net, state: dict, args, info: dict) -> Path:
    """[stage2] best held-out checkpoint (trainable + options; export_gguf/eval_offline --checkpoint
    read only these). Never rotated; overwritten atomically on each improvement."""
    ck_dir = out / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    path = ck_dir / "best.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"format": "gufo-mtp-stage1-ckpt-v1", "net_options": N.opts_to_dict(net.opt), "args": vars(args),
                "trainable": net.trainable_state(), "state": dict(state), "best": info}, tmp)
    os.replace(tmp, path)
    (ck_dir / "best.json.tmp").write_text(json.dumps(dict(info, path=path.name)))
    os.replace(ck_dir / "best.json.tmp", ck_dir / "best.json")
    return path


def find_resume(out: Path, spec: str):
    if spec in ("", "none"):
        return None
    if spec == "auto":
        lj = out / "ckpt" / "latest.json"
        return (out / "ckpt" / json.loads(lj.read_text())["path"]) if lj.exists() else None
    return Path(spec)


# ---------------------------------------------------------------------------
# in-loop evaluation (float numerics, bf16; final gate uses eval_offline decode)
# ---------------------------------------------------------------------------
def quick_eval(net, records, depth, device, head="small") -> dict:
    net.eval()
    stats = N.AcceptanceStats(depth)
    with torch.no_grad():
        for rd in records:
            d, g, f = N.draft_record(net, rd, depth, head, device, torch.bfloat16)
            stats.add(rd, d, g, f)
    net.train()
    rep = stats.report()
    s = rep["stop"][f"k{depth}"]
    return {"km": s["km"], "lower": s["lower"], "path": s["path"], "r": s["r"],
            "nostop_km": rep["nostop"][f"k{depth}"]["km"], "anchors": rep["anchors"]}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--val-shards", nargs="*", default=[])
    ap.add_argument("--mtp-gguf", default=str(R.DEFAULT_MTP))
    ap.add_argument("--frozen-bundle", default=None)
    ap.add_argument("--trunk-dir", default=str(R.DEFAULT_TRUNK_DIR))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    # what trains
    ap.add_argument("--dense-mode", choices=["full", "lora", "frozen"], default="full")
    ap.add_argument("--expert-mode", choices=["full", "lora", "frozen"], default="lora")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--expert-lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--qat", action="store_true", help="Q8_0 fake-quant (STE) of Q8 weights in forward")
    ap.add_argument("--moe-impl", choices=["loop", "grouped", "bmm", "auto"], default="loop",
                    help="[stage2] routed-expert GEMMs: loop = stage-1 per-expert loop; see moe_grouped.py")
    # objective
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--depth-decay", type=float, default=0.85)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=0.5)
    ap.add_argument("--offpath-weight", type=float, default=0.5)
    ap.add_argument("--kind-weights", default="", help="e.g. response=2,code=1,log=0.5")
    ap.add_argument("--detach-chain", action="store_true")
    ap.add_argument("--head-chunk", type=int, default=512)
    ap.add_argument("--no-head-checkpoint", action="store_true",
                    help="[stage2] do not recompute the head-loss chunks in backward (+~0.5-1 GB per 512-row chunk "
                         "held until backward, i.e. ~13-26 GB per 2K-token record; saves 1/3 of the head GEMMs)")
    # optimization
    ap.add_argument("--lr", type=float, default=3e-5, help="dense full-FT params")
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-experts", type=float, default=1e-5)
    ap.add_argument("--expert-optim", choices=["adafactor", "adamw"], default="adafactor")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=8, help="records per optimizer step")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048, help="truncate records")
    ap.add_argument("--shuffle-window", type=int, default=0)
    ap.add_argument("--refresh-shards-steps", type=int, default=0,
                    help="every N optimizer steps, add newly arrived shards (complete, sha-named .mtps) as a new block")
    # [stage2] data mixture / long records / held-out sets / early stopping (stage2_data.py; all off by default)
    ap.add_argument("--shard-weights", default="", help='[stage2] "GLOB=W,..." per-epoch sampling weight of matching shards')
    ap.add_argument("--window-long-records", action="store_true",
                    help="[stage2] records > --max-tokens with response segments -> response windows (no truncation)")
    ap.add_argument("--eval-set", action="append", default=[],
                    help="[stage2] NAME=GLOB[@kind+kind] extra held-out quick-eval set (repeatable)")
    ap.add_argument("--eval-set-records", type=int, default=32)
    ap.add_argument("--early-stop", default="", help="[stage2] eval-set NAME whose km drives best-checkpoint/early stop")
    ap.add_argument("--early-stop-patience", type=int, default=0, help="[stage2] evals without improvement (0 = off)")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.02)
    # [ablation] post-s1c slate (train/ablations/SLATE.md); all off by default
    ap.add_argument("--margin-weight", type=float, default=0.0,
                    help="[ablation B3] + weight * relu(margin - (z_target_argmax - max other logit)) per row")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--survival-floor", type=float, default=0.2, help="[ablation B1] floor of the survival weight")
    ap.add_argument("--p1-gamma", type=float, default=0.0,
                    help="[ablation A5] row weight x p1^gamma (p1 = target top-1 probability of the label row)")
    ap.add_argument("--chain-weighting", choices=["none", "accepted", "survival"], default="none",
                    help="[ablation A] weight depth-j rows by whether the student's own greedy drafts 1..j-1 "
                         "matched the target (the rows serving verifies); others get --offchain-weight")
    ap.add_argument("--offchain-weight", type=float, default=0.1)
    ap.add_argument("--elen-weight", type=float, default=0.0,
                    help="[ablation C] + weight * (-E[accepted drafts]/depth), E = sum_k prod_j q_j(target greedy)")
    ap.add_argument("--l2sp", type=float, default=0.0,
                    help="[ablation D] decoupled decay of dense weights toward the run's start weights (lr * l2sp)")
    ap.add_argument("--depth-bias", action="store_true",
                    help="[ablation C3] zero-init learned bias per depth on the fused residual (export cannot carry "
                         "it: judge with eval_offline --checkpoint)")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="[ablation D] residual dropout on the attention and MoE outputs (training only)")
    ap.add_argument("--draft-head-quant", choices=["q8_0", "q4_0", "q5_0"], default="q8_0",
                    help="[stage2/R4-28] fake-requantize the small draft head rows [0, DRAFT_VOCAB) used by the loss "
                         "and quick-eval (head-aware fine-tune for a Q4_0 draft-head copy); score with eval_offline "
                         "--head-quant")
    ap.add_argument("--eval-sampled-temps", type=float, nargs="*", default=[],
                    help="[stage2/R3-14] also score sampled-path acceptance sum min(p_T, q_T) at these T (<= 1)")
    ap.add_argument("--early-stop-metric", default="km",
                    help="[stage2/R3-14] km (greedy tokens/call) or sampled@T (e.g. sampled@1) of the --early-stop set")
    ap.add_argument("--init-trainable", default="",
                    help="[stage2] start from a checkpoint's trainable weights (fresh optimizer/schedule/data cursor); "
                         "ignored when resuming")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--deterministic", action="store_true")
    # bookkeeping
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--keep-ckpts", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-records", type=int, default=24)
    ap.add_argument("--eval-max-tokens", type=int, default=2048)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--benchmark-steps", type=int, default=0)
    # cost guard (wall clock of this process + --usd-offset of earlier runs)
    ap.add_argument("--rate-usd-per-hour", type=float, default=1.57)
    ap.add_argument("--usd-offset", type=float, default=0.0)
    ap.add_argument("--max-usd", type=float, default=0.0)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--gate-hours", type=float, default=0.0, help="G2: check val gain after this many hours")
    ap.add_argument("--gate-min-gain", type=float, default=0.15)
    ap.add_argument("--tiny-synthetic", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = (out / "train-log.jsonl").open("a", encoding="utf-8")

    def log(rec: dict):
        rec = dict(rec, t=round(time.time(), 1))
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps(rec), flush=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    t_start = time.time()

    # ---- model
    if a.tiny_synthetic:                     # selftest_train.py: same code paths, random tiny weights
        import tiny
        tiny_ref, target = tiny.make_tiny(a.seed)
    else:
        target = (N.FrozenTarget.from_bundle(Path(a.frozen_bundle)) if a.frozen_bundle
                  else N.FrozenTarget.from_trunk(Path(a.trunk_dir)))
    target.to_dense(dev, torch.bfloat16)
    opts = N.NetOptions(decode=False, dense_mode=a.dense_mode, expert_mode=a.expert_mode,
                        lora_rank=a.lora_rank, expert_lora_rank=a.expert_lora_rank, lora_alpha=a.lora_alpha,
                        qat=a.qat, param_dtype=torch.float32,
                        expert_dtype=torch.float32 if a.expert_mode == "full" else torch.bfloat16,
                        moe_impl=a.moe_impl, dropout=a.dropout, depth_bias=a.depth_bias)
    net = N.MtpNet(tiny_ref, target, opts, dev) if a.tiny_synthetic else N.build_net(Path(a.mtp_gguf), target, opts, dev)
    net.train()
    if a.draft_head_quant != "q8_0":     # [stage2/R4-28] train against the requantized small draft head rows
        import quant_sim as Q
        hd = target.head_dense
        with torch.no_grad():
            for r0 in range(0, min(N.DRAFT_VOCAB, hd.shape[0]), 16384):
                r1 = min(r0 + 16384, N.DRAFT_VOCAB, hd.shape[0])
                hd[r0:r1] = Q.fake(hd[r0:r1].float(), a.draft_head_quant).to(hd.dtype)
    named = net.trainable_named()
    groups = {"dense": [], "lora": [], "experts": []}
    for n_, p_ in named:
        groups["lora" if n_.startswith("lora.") else "experts" if n_ in N.EXPERTS else "dense"].append(p_)
    n_params = {k: sum(p.numel() for p in v) for k, v in groups.items()}
    adam_groups = [g for g in ({"params": groups["dense"], "lr": a.lr, "name": "dense"},
                               {"params": groups["lora"], "lr": a.lr_lora, "name": "lora"}) if g["params"]]
    opt = torch.optim.AdamW(adam_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.weight_decay)
    opt_e = None
    if groups["experts"]:
        if a.expert_optim == "adafactor" and hasattr(torch.optim, "Adafactor"):
            opt_e = torch.optim.Adafactor(groups["experts"], lr=a.lr_experts, weight_decay=a.weight_decay)
        else:
            opt_e = torch.optim.AdamW(groups["experts"], lr=a.lr_experts, betas=(0.9, 0.95),
                                      weight_decay=a.weight_decay)

    class Both:  # one object for checkpointing both optimizers
        def state_dict(self):
            return {"main": opt.state_dict(), "experts": opt_e.state_dict() if opt_e else None}

        def load_state_dict(self, d):
            opt.load_state_dict(d["main"])
            if opt_e and d.get("experts"):
                opt_e.load_state_dict(d["experts"])

    both = Both()

    # ---- data
    train_shards = N.resolve_shards(a.shards)
    if not train_shards:
        raise SystemExit("no training shards")
    ck_path = find_resume(out, a.resume)
    saved_blocks = None
    if ck_path and ck_path.exists():
        saved_blocks = torch.load(ck_path, map_location="cpu", weights_only=False)["state"].get("blocks")
    stream = RecordStream(a.shards, a.seed, a.shuffle_window, a.max_tokens, blocks=saved_blocks,
                          shard_weights=a.shard_weights, window_long=a.window_long_records)
    steps_per_epoch = stream.entries_per_epoch // a.grad_accum
    total_steps = a.max_steps or steps_per_epoch * a.epochs
    base_lrs = [g["lr"] for g in opt.param_groups] + ([g["lr"] for g in opt_e.param_groups] if opt_e else [])

    def lr_factor(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        prog = min(1.0, (step - a.warmup) / max(1, total_steps - a.warmup))
        return a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    val_records = []
    if a.val_shards:
        for p in N.resolve_shards(a.val_shards):
            s = N.read_gmtp.Shard(p)
            for r in s.records[::max(1, len(s.records) // 8)]:
                val_records.append(N.truncate(N.read_record(s, r), a.eval_max_tokens))
                if len(val_records) >= a.eval_records:
                    break
            if len(val_records) >= a.eval_records:
                break

    state = {"step": 0, "epoch": 0, "cursor": 0, "tokens": 0, "records": 0, "usd_runs": a.usd_offset,
             "fingerprint": stream.fingerprint, "blocks": stream.blocks}
    if stream.stage2:
        state["stream_config"] = stream.config()                                        # [stage2]
    if ck_path and ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        saved = ck["state"].get("blocks")
        if saved is None or stream.blocks[:len(saved)] != saved:
            raise SystemExit("resume refused: checkpoint shard blocks are not a prefix of the current blocks")
        if json.dumps(ck["state"].get("stream_config")) != json.dumps(state.get("stream_config")):  # [stage2]
            raise SystemExit("resume refused: --shard-weights/--window-long-records/--max-tokens differ from the checkpoint")
        net.load_trainable(ck["trainable"])
        both.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        state = ck["state"]
        state["blocks"], state["fingerprint"] = stream.blocks, stream.fingerprint
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        if ck["rng"].get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        log({"event": "resumed", "from": str(ck_path), "step": state["step"]})
    elif a.init_trainable:                                                                  # [stage2] warm start
        ck_i = torch.load(a.init_trainable, map_location="cpu", weights_only=False)
        st_i = dict(ck_i["trainable"])
        merged = merge_expert_lora(net, st_i, ck_i.get("net_options", {}))   # [ablation] LoRA ckpt -> full experts
        net.load_trainable(st_i)
        if merged:
            log({"event": "init-merged-expert-lora", "tensors": merged})
        state["init_trainable"] = a.init_trainable
        log({"event": "init-trainable", "from": a.init_trainable})
    l2sp_ref = []                                     # [ablation D] reference = the run's start weights
    if a.l2sp > 0:
        if a.init_trainable:                          # also on resume: the init checkpoint is the reference
            ref_st = dict(torch.load(a.init_trainable, map_location="cpu", weights_only=False)["trainable"])
            merge_expert_lora(net, ref_st, {})
        else:
            ref_st = {}
        for n_, p_ in named:
            if not n_.startswith("lora.") and n_ not in N.EXPERTS:
                ref = ref_st.get(n_)
                l2sp_ref.append((p_, (ref if ref is not None else p_.detach()).to(p_.device, p_.dtype).clone()))
    log({"event": "start", "params": n_params, "train_shards": len(train_shards),
         "train_records": stream.total_records, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
         "val_records": len(val_records), "net_options": N.opts_to_dict(opts), "args": vars(a),
         "setup_s": round(time.time() - t_start, 1)})
    if val_records and state["step"] == 0 and not a.benchmark_steps:
        ev0 = quick_eval(net, val_records, a.depth, dev)
        state["km0"] = ev0["km"]
        log({"event": "eval", "step": 0, **ev0})
    gate_checked = state.get("g2_checked", False)

    # [stage2] extra held-out eval sets (anchor-kind filtered), best checkpoint + early stopping
    eval_sets = []
    for spec in a.eval_set:
        es_ = S2.parse_eval_set(spec)
        es_["records"] = S2.load_eval_set(es_, a.eval_set_records, a.eval_max_tokens, a.window_long_records)
        eval_sets.append(es_)
        log({"event": "eval-set-loaded", "set": es_["name"], "records": len(es_["records"]), "kinds": es_["kinds"]})
    if a.early_stop and a.early_stop not in [e["name"] for e in eval_sets]:
        raise SystemExit(f"--early-stop {a.early_stop}: no --eval-set of that name")
    early = S2.EarlyStop(a.early_stop_patience, a.early_stop_min_delta, state.get("early")) if a.early_stop else None
    if a.early_stop_metric != "km" and a.early_stop_metric not in [f"sampled@{t:g}" for t in a.eval_sampled_temps]:
        raise SystemExit(f"--early-stop-metric {a.early_stop_metric}: needs the matching --eval-sampled-temps")

    def run_eval_sets(step):
        res = {}
        for es_ in eval_sets:
            r = S2.quick_eval_set(net, es_["records"], a.depth, dev, es_["kinds"], torch.bfloat16,
                                  tuple(a.eval_sampled_temps))
            res[es_["name"]] = r
            log({"event": "eval-set", "set": es_["name"], "step": step, **r})
        if "trainprobe" in res and a.early_stop in res:      # overfitting monitor: seen-data vs held-out acceptance
            log({"event": "overfit-gap", "step": step, "gap": round(res["trainprobe"]["km"] - res[a.early_stop]["km"], 4)})
        if early is not None and a.early_stop in res:
            if early.update(step, res[a.early_stop][a.early_stop_metric]):
                info = {"step": step, "set": a.early_stop, "metric": a.early_stop_metric,
                        "value": res[a.early_stop][a.early_stop_metric], "km": res[a.early_stop]["km"],
                        "sets": {k: {m: v[m] for m in v if m == "km" or m.startswith("sampled@")} for k, v in res.items()}}
                save_best(out, net, state, a, info)
                log({"event": "best", **info})
            state["early"] = early.state()
        return res

    if eval_sets and state["step"] == 0 and not a.benchmark_steps:
        run_eval_sets(0)

    run_t0 = time.time()

    def usd_now():
        return state["usd_runs"] + (time.time() - t_start) / 3600 * a.rate_usd_per_hour

    loss_fn = DepthLoss(net, target.head_dense, a, parse_kind_weights(a.kind_weights))
    trainable = [p for _, p in named]
    accum, tok_window, t_window = 0, 0, time.time()
    stop_reason = None
    bench_t0, bench_tok = None, 0
    for ep, idx, rd in stream.iterate(state["epoch"], state["cursor"], a.epochs):
        if (out / "STOP").exists():
            stop_reason = "STOP file"
            break
        if a.max_usd and usd_now() >= a.max_usd:
            stop_reason = f"budget {usd_now():.2f} >= {a.max_usd} USD"
            break
        if a.max_hours and (time.time() - t_start) / 3600 >= a.max_hours:
            stop_reason = "max hours"
            break
        hidden = rd.hidden_tensor(dev, torch.bfloat16)
        tokens = torch.from_numpy(rd.tokens).to(dev)
        loss_fn.record(rd, dev)
        net.unrolled(hidden, tokens, a.depth, "teacher", loss_fn, checkpoint=True, detach_chain=a.detach_chain)
        loss = loss_fn.loss / max(loss_fn.wsum, 1e-6) / a.grad_accum
        if loss_fn.ext:                                                                     # [ablation] E-len term
            loss = loss + loss_fn.extra() / a.grad_accum
        loss.backward()
        accum += 1
        state["tokens"] += rd.n
        state["records"] += 1
        tok_window += rd.n
        state["epoch"], state["cursor"] = ep, idx + 1
        if accum < a.grad_accum:
            continue
        accum = 0
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.clip) if a.clip else torch.tensor(0.0)
        f = lr_factor(state["step"])
        if opt_e:
            for g in opt_e.param_groups:
                g["lr"] = a.lr_experts * f
        opt.step()
        if l2sp_ref:                                  # [ablation] decoupled decay toward the start weights (L2-SP)
            with torch.no_grad():
                lr_now = opt.param_groups[0]["lr"]
                for p_, p0 in l2sp_ref:
                    p_.add_(p_ - p0.to(p_.device), alpha=-lr_now * a.l2sp)
        if opt_e:
            opt_e.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if opt_e:
            opt_e.zero_grad(set_to_none=True)
        state["step"] += 1
        step = state["step"]
        if a.refresh_shards_steps and step % a.refresh_shards_steps == 0:
            added = stream.refresh()
            if added:
                state["blocks"], state["fingerprint"] = stream.blocks, stream.fingerprint
                if not a.max_steps:
                    steps_per_epoch = stream.entries_per_epoch // a.grad_accum
                    total_steps = steps_per_epoch * a.epochs
                log({"event": "shards-added", "step": step, "added": added, "blocks": len(stream.blocks),
                     "train_records": stream.total_records, "total_steps": total_steps})
        if a.benchmark_steps:
            if step == 2:
                torch.cuda.synchronize() if dev.type == "cuda" else None
                bench_t0, bench_tok = time.time(), state["tokens"]
            if step >= a.benchmark_steps + 2:
                torch.cuda.synchronize() if dev.type == "cuda" else None
                tps = (state["tokens"] - bench_tok) / (time.time() - bench_t0)
                epoch_tok = stream.entries_per_epoch * (state["tokens"] / state["records"])
                hours = epoch_tok / tps / 3600
                log({"event": "benchmark", "tokens_per_s": round(tps, 1), "depth": a.depth,
                     "epoch_tokens_est": int(epoch_tok), "epoch_hours_est": round(hours, 2),
                     "epoch_usd_est": round(hours * a.rate_usd_per_hour, 2),
                     "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if dev.type == "cuda" else None})
                return 0
        if step % a.log_every == 0:
            dt = time.time() - t_window
            log({"event": "step", "step": step, "epoch": ep, "loss": float(loss) * a.grad_accum,
                 "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"], "tok_s": round(tok_window / dt, 1),
                 "tokens": state["tokens"], "usd": round(usd_now(), 2),
                 "depths": loss_fn.stats_floats(),
                 "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if dev.type == "cuda" else None})
            tok_window, t_window = 0, time.time()
        if a.ckpt_every and step % a.ckpt_every == 0:
            p = save_ckpt(out, net, both, sched, state, a, a.keep_ckpts)
            log({"event": "checkpoint", "step": step, "path": str(p)})
        if val_records and a.eval_every and step % a.eval_every == 0:
            ev = quick_eval(net, val_records, a.depth, dev)
            log({"event": "eval", "step": step, **ev})
            # G2 early-signal gate: after --gate-hours of this run, val km must have gained >= --gate-min-gain
            if (a.gate_hours and not gate_checked and "km0" in state
                    and (time.time() - t_start) / 3600 >= a.gate_hours):
                gate_checked = state["g2_checked"] = True
                gain = ev["km"] - state["km0"]
                log({"event": "gate-G2", "gain": round(gain, 4), "need": a.gate_min_gain,
                     "pass": gain >= a.gate_min_gain})
                if gain < a.gate_min_gain:
                    stop_reason = f"G2 failed: gain {gain:.3f} < {a.gate_min_gain}"
                    break
        if eval_sets and a.eval_every and step % a.eval_every == 0:                      # [stage2]
            run_eval_sets(step)
            if early is not None and early.stop:
                stop_reason = (f"early stop: {a.early_stop} km gained <= {a.early_stop_min_delta} in "
                               f"{a.early_stop_patience} evals (best {a.early_stop_metric} {early.best:.3f} @ step "
                               f"{early.best_step})")
                break
        if total_steps and step >= total_steps:
            stop_reason = "total steps"
            break
    else:
        stop_reason = stop_reason or "data exhausted"
    p = save_ckpt(out, net, both, sched, state, a, a.keep_ckpts)
    final = {"event": "end", "reason": stop_reason, "step": state["step"], "tokens": state["tokens"],
             "usd": round(usd_now(), 2), "hours": round((time.time() - run_t0) / 3600, 3), "checkpoint": str(p)}
    if val_records:
        final["eval"] = quick_eval(net, val_records, a.depth, dev)
    if eval_sets:                                                                          # [stage2]
        final["eval_sets"] = {k: v["km"] for k, v in run_eval_sets(state["step"]).items()}
        if early is not None:
            final["best"] = early.state()
    log(final)
    (out / "DONE.json").write_text(json.dumps(final, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
