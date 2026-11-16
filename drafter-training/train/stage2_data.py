"""Stage 2 data helpers: source mixture weights, response windows for long records,
held-out eval sets with anchor-kind filters, early stopping. Used by train_mtp.py
(--shard-weights, --window-long-records, --eval-set, --early-stop*) and
eval_offline.py (--window-long-records). All off by default: stage-1 behaviour,
record order and checkpoints are unchanged unless a stage-2 flag is given.

Mixture weights (--shard-weights "GLOB=W,..."): every record of a shard whose path
matches GLOB appears floor(W) times per epoch, plus once more with probability
frac(W) (deterministic per seed/epoch/block). W < 1 subsamples, W = 0 drops. The
first matching GLOB wins; unmatched shards have W = 1. DepthLoss normalises each
record by its own weight sum, so --kind-weights only moves emphasis *inside* a
record; cross-source emphasis needs this sampling weight.

Long records (--window-long-records): records longer than --max-tokens that carry
target-response segments (on-policy, MG3 tool-path chains up to 32K tokens) are
cut into windows that END at response ends, each <= max_tokens rows, with as much
preceding context as fits. Rows before `train_from` (already covered by the
previous window) stay as attention context but are unsupervised, so no row is
trained or scored twice. The trunk hidden states are exact (captured at full
context); only the MTP layer's own attention loses keys older than the window
(RoPE is relative, so positions need no shift). gufo itself switches to QSA
(top-512 blocks) beyond 2,052 keys, so the MTP layer never sees the full history
densely in production either. Records without segments keep stage-1 truncation.
"""
from __future__ import annotations

import fnmatch
import math
from pathlib import Path
from typing import Optional

import numpy as np

import mtp_net as N

MIN_CONTEXT = 256          # rows of context kept before a response piece when the response is longer than a window


# ---------------------------------------------------------------------------
# mixture weights
# ---------------------------------------------------------------------------
def parse_shard_weights(spec: str) -> list[tuple[str, float]]:
    out = []
    for part in filter(None, (spec or "").split(",")):
        g, w = part.rsplit("=", 1)
        out.append((g.strip(), float(w)))
    return out


def shard_weight(path: Path, rules: list[tuple[str, float]]) -> float:
    s = str(path).replace("\\", "/")
    for g, w in rules:
        g = g.replace("\\", "/")
        if fnmatch.fnmatch(s, g) or fnmatch.fnmatch(Path(s).name, g):
            return w
    return 1.0


def repeat_entries(entries: list, weights: list[float], rng: np.random.Generator) -> list:
    """entries[i] repeated floor(w) times + 1 with prob frac(w); input order kept (caller shuffles)."""
    out = []
    u = rng.random(len(entries))
    for e, w, x in zip(entries, weights, u):
        k = int(math.floor(w)) + (1 if x < (w - math.floor(w)) else 0)
        out += [e] * k
    return out


# ---------------------------------------------------------------------------
# response windows for long records
# ---------------------------------------------------------------------------
def response_spans(meta: Optional[dict], n: int) -> list[tuple[int, int]]:
    md = (meta or {}).get("metadata") or {}
    if md.get("kind") != "target-responses":
        return []
    spans = []
    for seg in md.get("segments", []) or []:
        r0, e0 = int(seg.get("response_start", 0)), int(seg.get("end", n))
        r0, e0 = max(0, r0), min(n, e0)
        if e0 > r0:
            spans.append((r0, e0))
    return sorted(spans)


def record_windows(meta: Optional[dict], n: int, max_tokens: int, min_context: int = MIN_CONTEXT) -> list[tuple[int, int, int]]:
    """[(start, end, train_from)] covering every response row of a record once.
    n <= max_tokens or no response segments -> [(0, min(n, max_tokens), 0)] (stage-1 truncation)."""
    if not max_tokens or n <= max_tokens:
        return [(0, n, 0)]
    spans = response_spans(meta, n)
    if not spans:
        return [(0, max_tokens, 0)]
    ctx = min(min_context, max_tokens // 2)
    pieces = []                                       # response pieces, each short enough for one window
    for r0, e0 in spans:
        p = r0
        while p < e0:
            q = min(e0, p + max_tokens - min(ctx, p))
            pieces.append((p, q))
            p = q
    wins: list[list[int]] = []
    for p, q in pieces:
        if wins and q - wins[-1][0] <= max_tokens:    # extend the current window (shared context)
            wins[-1][1] = q
        else:
            wins.append([max(0, q - max_tokens), q])
    out, covered = [], 0
    for s, e in wins:
        out.append((s, e, max(s, covered)))
        covered = e
    return out


def slice_record(rd: "N.RecordData", s: int, e: int, train_from: int) -> "N.RecordData":
    sup = rd.supervised[s:e].copy()
    sup[:max(0, train_from - s)] = False
    return N.RecordData(f"{rd.rid}@{s}-{e}", rd.tokens[s:e], rd.hidden_bf16[s:e], rd.sup_ids[s:e], rd.sup_lp[s:e],
                        sup, rd.kinds[s:e], rd.meta)


def windowed_records(rd: "N.RecordData", max_tokens: int, window: bool) -> list["N.RecordData"]:
    if not window:
        return [N.truncate(rd, max_tokens) if max_tokens else rd]
    return [slice_record(rd, s, e, t) for s, e, t in record_windows(rd.meta, rd.n, max_tokens)]


# ---------------------------------------------------------------------------
# held-out eval sets (in-loop quick eval)
# ---------------------------------------------------------------------------
def parse_eval_set(spec: str) -> dict:
    """NAME=GLOB[|GLOB...][@kind+kind], e.g. serve=/data/val-onpolicy/*.mtps|/data/val-mg3/*.mtps@response"""
    name, rest = spec.split("=", 1)
    globs, _, kinds = rest.partition("@")
    return {"name": name, "globs": [g for g in globs.split("|") if g], "kinds": [k for k in kinds.split("+") if k] or None}


def load_eval_set(spec: dict, records: int, max_tokens: int, window: bool) -> list:
    """Up to `records` (windows count as records), drawn round-robin across the globs (each glob:
    every shard, records strided by len/8 as train_mtp's val sample), so a multi-source set stays mixed."""
    per_glob = []
    for g in spec["globs"]:
        items = []
        for p in N.resolve_shards([g]):
            s = N.read_gmtp.Shard(p)
            items += [(s, r) for r in s.records[::max(1, len(s.records) // 8)]]
        per_glob.append(items)
    out, k = [], 0
    while len(out) < records and any(k < len(it) for it in per_glob):
        for it in per_glob:
            if k < len(it) and len(out) < records:
                s, r = it[k]
                out += windowed_records(N.read_record(s, r), max_tokens, window)
        k += 1
    return out[:records]


def quick_eval_set(net, records: list, depth: int, device, kinds: Optional[list[str]], dtype,
                   sampled_temps=()) -> dict:
    """Greedy km (production small head + confidence stop) and, with sampled_temps (R3-14), the
    sampled-path E_lo/E_hi per temperature (all anchors of the set's kinds, and per kind)."""
    import torch
    stats = N.AcceptanceStats(depth)
    sst = SampledStats(depth, sampled_temps) if sampled_temps else None
    codes = [N.KIND_CODES[k] for k in kinds] if kinds else None
    net.eval()
    with torch.no_grad():
        for rd in records:
            d, g, f = N.draft_record(net, rd, depth, "small", device, dtype)
            keep = None
            if codes is not None:
                n = rd.n
                keep = np.isin(rd.kinds[np.minimum(np.arange(n - 1) + 1, n - 1)], codes)
                d = d.copy()
                d[~keep] = -1
                f = f & keep
            stats.add(rd, d, g, f, kind_split=False)
            if sst is not None:
                sst.add(rd, sampled_record(net, rd, depth, device, dtype, sampled_temps), keep)
    net.train()
    rep = stats.report()
    s = rep["stop"][f"k{depth}"]
    out = {"km": s["km"], "lower": s["lower"], "r": s["r"], "anchors_scored": int(s["cohort"][0]),
           "records": len(records)}
    if sst is not None:
        out["sampled"] = sst.report()
        for T in sampled_temps:                       # flat keys for early stopping / logs
            out[f"sampled@{T:g}"] = out["sampled"].get(f"T{T:g}", {}).get("all", {}).get("E_lo", 1.0)
    return out


class EarlyStop:
    """Best held-out km with patience. improved() -> True when `km` beats best by min_delta."""

    def __init__(self, patience: int, min_delta: float, state: Optional[dict] = None):
        self.patience, self.min_delta = patience, min_delta
        st = state or {}
        self.best = st.get("best_km")
        self.best_step = st.get("best_step")
        self.bad = st.get("bad_evals", 0)

    def update(self, step: int, km: float) -> bool:
        if self.best is None or km > self.best + self.min_delta:
            self.best, self.best_step, self.bad = km, step, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self) -> bool:
        return bool(self.patience) and self.bad >= self.patience

    def state(self) -> dict:
        return {"best_km": self.best, "best_step": self.best_step, "bad_evals": self.bad}


# ---------------------------------------------------------------------------
# R3-14: sampled-path acceptance  sum_v min(p_T(v), q_T(v))  per depth and per kind
# ---------------------------------------------------------------------------
def alpha_rows(z_ids, lse, lp, ids_valid, T: float):
    """Per-row bounds on the speculative-sampling acceptance of one draft at temperature T <= 1.

    z_ids [r,K] drafter logits at the target's top-K ids, lse [r] logsumexp(z/T) over the full
    vocabulary, lp [r,K] target top-K log-probs (T=1). The target is known only on its top-K
    (+ residual mass rho); residual tokens each have p <= p_K, so at T < 1 their tempered mass
    is <= rho * p_K^(1/T-1). Returns (lower, upper) tensors [r]; T = 1: exact on top-K,
    upper adds min(rho, 1 - q_topK). Presence penalties (no-think chat) are not modelled."""
    import torch
    p = lp.exp().masked_fill(~ids_valid, 0.0)
    rho = (1.0 - p.sum(-1)).clamp(min=0.0)
    q = torch.exp(z_ids / T - lse[:, None]).masked_fill(~ids_valid, 0.0)
    q_out = (1.0 - q.sum(-1)).clamp(min=0.0)
    if T == 1.0:
        lower = torch.minimum(p, q).sum(-1)
        return lower, lower + torch.minimum(rho, q_out)
    w = p.pow(1.0 / T)
    zt = w.sum(-1).clamp(min=1e-30)
    pk = p.masked_fill(~ids_valid, 2.0).min(-1).values.clamp(max=1.0)
    zr = rho * pk.pow(1.0 / T - 1.0)
    lower = torch.minimum(w / (zt + zr)[:, None], q).sum(-1)
    upper = torch.minimum(w / zt[:, None], q).sum(-1) + torch.minimum(zr / (zt + zr), q_out)
    return lower, upper


def sampled_record(net, rd: "N.RecordData", depth: int, device, dtype, temps=(1.0, 0.7)) -> dict:
    """Teacher-fed chain (the record's own tokens = the sampled path for sampled on-policy data):
    per depth j, per row p (label row p+j) the bounds of alpha_T. Returns {T: (lo, hi)} with
    arrays [n-1, depth] (nan = no row / unsupervised label)."""
    import torch
    import torch.nn.functional as F
    n = rd.n
    out = {T: (np.full((n - 1, depth), np.nan, np.float32), np.full((n - 1, depth), np.nan, np.float32))
           for T in temps}
    ids_all = torch.from_numpy(rd.sup_ids).to(device)
    lp_all = torch.from_numpy(rd.sup_lp).to(device)

    def on_depth(j, res):
        Rj = res.shape[0]
        x = net.head_input(net.head_mixed(res))
        ids, lp = ids_all[j:j + Rj], lp_all[j:j + Rj]
        valid = ids >= 0
        idc = ids.clamp(min=0)
        z_ids = torch.zeros(ids.shape, dtype=torch.float32, device=device)
        lse = {T: torch.full((Rj,), float("-inf"), device=device) for T in temps}
        for v0, W in N.head_chunks(net.target, device):
            zc = F.linear(x.to(W.dtype), W).float()
            v1 = v0 + W.shape[0]
            inr = (idc >= v0) & (idc < v1)
            g = zc.gather(-1, (idc - v0).clamp(0, W.shape[0] - 1))
            z_ids = torch.where(inr, g, z_ids)
            for T in temps:
                lse[T] = torch.logaddexp(lse[T], torch.logsumexp(zc / T, -1))
        sup = torch.from_numpy(rd.supervised[j:j + Rj]).to(device)
        for T in temps:
            lo, hi = alpha_rows(z_ids, lse[T], lp, valid, T)
            out[T][0][:Rj, j - 1] = lo.masked_fill(~sup, float("nan")).cpu().numpy()
            out[T][1][:Rj, j - 1] = hi.masked_fill(~sup, float("nan")).cpu().numpy()
        return None

    hidden = rd.hidden_tensor(device, dtype)
    tokens = torch.from_numpy(rd.tokens).to(device)
    with torch.no_grad():
        net.unrolled(hidden, tokens, depth, "teacher", on_depth)
    return out


class SampledStats:
    """Mean alpha_T per depth (all rows with a label) overall and per anchor kind; expected tokens
    per verify under sampling, depths treated as independent: E = 1 + sum_k prod_{j<=k} mean(alpha_j)."""

    def __init__(self, depth: int, temps):
        self.D, self.temps = depth, tuple(temps)
        self.acc: dict = {}                             # (T, kind|-1, bound) -> [sum[D], cnt[D]]

    def add(self, rd: "N.RecordData", arrays: dict, keep: Optional[np.ndarray] = None) -> None:
        n = rd.n
        kinds = rd.kinds[np.minimum(np.arange(n - 1) + 1, n - 1)]
        for T, (lo, hi) in arrays.items():
            for b, arr in (("lo", lo), ("hi", hi)):
                ok = ~np.isnan(arr)
                if keep is not None:
                    ok &= keep[:, None]
                for kc in [-1] + sorted(int(k) for k in np.unique(kinds)):
                    m = ok if kc == -1 else ok & (kinds == kc)[:, None]
                    s, c = self.acc.setdefault((T, kc, b), [np.zeros(self.D), np.zeros(self.D)])
                    s += np.where(m, arr, 0.0).sum(0)
                    c += m.sum(0)

    def report(self) -> dict:
        inv = {v: k for k, v in N.KIND_CODES.items()}
        out: dict = {}
        for (T, kc, b), (s, c) in sorted(self.acc.items()):
            if kc != -1 and c[0] == 0:
                continue
            a = np.where(c > 0, s / np.maximum(c, 1), 0.0)
            d = out.setdefault(f"T{T:g}", {"all": {}, "by_kind": {}})
            tgt = d["all"] if kc == -1 else d["by_kind"].setdefault(inv.get(kc, str(kc)), {})
            tgt[f"E_{b}"] = round(float(1 + np.cumprod(a).sum()), 4)
            tgt[f"alpha_{b}"] = [round(float(x), 4) for x in a]
            tgt["rows_depth1"] = int(c[0])
        return out
