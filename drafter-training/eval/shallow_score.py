"""Shallow-depth scoring of MTP heads on eval_offline outputs (owner direction 2026-10-27: shallow depth first).

Inputs: eval_offline output dirs (summary.json + records/*.npz), normally the on-policy set scored with
--anchor-kinds response (score_onpolicy.sh / run_slate.sh judge_export). Numpy only (no torch).

Per head: per-depth conditional acceptance r_j (stop, k=7, product-limit km estimator, exactly as
mtp_net.AcceptanceStats), km = E = 1 + sum_k prod_{i<=k} r_i.

Value-weighted gain vs a reference head:
    dE/dr_j = sum_{k>=j} prod_{i<=k, i!=j} r_i = P_{j-1} * T_j,
    P_{j-1} = prod_{i<j} r_i  (chain alive into depth j),   T_j = 1 + r_{j+1} T_{j+1},  T_D = 1  (tail value)
    V = sum_j (dE/dr_j at the reference's r) * (r_j(head) - r_j(ref))     [tokens/call, first order]
dE = km(head) - km(ref) is printed next to it (V - dE = higher-order terms). Note dE/dr_1 = (E-1)/r_1 and
dE/dr_D = P_{D-1}.

CI: paired cluster bootstrap over records (npz files present in BOTH dirs, matched by file name; a windowed
record's -wNNN parts form one cluster), B resamples of per-record cohort/success counts -> r -> V, d1, d2, dE.

Decision rule (SHALLOW-VERDICT.md): PROMOTE if V >= +0.05 AND d1 = dr_1 >= +0.5 pp; KILL if V < +0.02;
otherwise inconclusive (both on the on-policy set, response anchors).

--fit-decay prints the --depth-decay that best matches dE/dr_j at the reference (log-LS fit through
depth 1): for the stage-1 objective (per-depth weight ~ P_{j-1} T_j) and for --chain-weighting survival
(B1 already supplies ~P_{j-1}, so the decay must supply the tail T_j).

  python shallow_score.py --ref base base=/data/runs/onpolicy-x0/base s1a=/data/runs/onpolicy-x0/s1a ...
  python shallow_score.py --ref ctrl --ref ctrl0=<INIT onpolicy dir> ctrl=/data/runs/abl/ctrl/eval-export/onpolicy ...
  python shallow_score.py --selftest
Heads: NAME=DIR or DIR (name = parent dir name if DIR ends in eval-export/onpolicy, else basename).
--ref NAME (a listed head) or NAME=DIR (extra reference, not otherwise listed); repeatable.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

D = 7
CONF_FLOOR = 0.5            # mtp_net.CONF_FLOOR = reference GREEDY_DRAFT_CONFIDENCE_FLOOR
KIND_CODES = {"other": 0, "code": 1, "doc": 2, "diff": 3, "log": 4, "response": 5, "prompt": 6}
PROMOTE_V, PROMOTE_D1, KILL_V = 0.05, 0.005, 0.02


# ----------------------------------------------------------------------------------------------- math
def km_of(r) -> float:
    return float(1.0 + np.cumprod(np.asarray(r, float)).sum())


def partials(r) -> np.ndarray:
    """dE/dr_j for E = 1 + sum_k prod_{i<=k} r_i. r: [..., D] -> [..., D]."""
    r = np.asarray(r, float)
    Dn = r.shape[-1]
    P = np.concatenate([np.ones(r.shape[:-1] + (1,)), np.cumprod(r, -1)[..., :-1]], -1)   # P_{j-1}
    T = np.ones_like(r)
    for j in range(Dn - 2, -1, -1):
        T[..., j] = 1.0 + r[..., j + 1] * T[..., j + 1]
    return P * T


def tail(r) -> np.ndarray:
    r = np.asarray(r, float)
    T = np.ones_like(r)
    for j in range(len(r) - 2, -1, -1):
        T[j] = 1.0 + r[j + 1] * T[j + 1]
    return T


def fit_decay(w) -> float:
    """decay^(j-1) ~ w_j / w_1, least squares on logs through depth 1."""
    w = np.asarray(w, float)
    x = np.arange(len(w))
    y = np.log(w / w[0])
    return float(math.exp((x * y).sum() / (x * x).sum()))


def r_from_counts(coh, succ) -> np.ndarray:
    return np.where(coh > 0, succ / np.maximum(coh, 1), 0.0)


# ----------------------------------------------------------------------------------- record counting
def record_counts(z, depth: int = D, codes=None):
    """Stop-mode cohort/success counts of one eval_offline record npz = mtp_net.AcceptanceStats.add
    (s == "stop"), with eval_offline.summarize's anchor-kind filter."""
    t = z["tokens"].astype(np.int64)
    n = len(t)
    sup = z["supervised"].astype(bool)
    g = np.where(sup, z["sup_ids0"].astype(np.int64), -1)
    drafts = z["drafts"][:, :depth].astype(np.int64).copy()
    gaps = z["gaps"][:, :depth].astype(np.float32)
    if codes is not None:
        keep = np.isin(z["kinds"][np.minimum(np.arange(n - 1) + 1, n - 1)], codes)
        drafts[~keep] = -1
    P = n - 1
    p = np.arange(P)
    j = np.arange(1, depth + 1)
    lab = p[:, None] + j[None, :]
    avail = (lab <= n - 1) & (drafts >= 0)
    labc = np.minimum(lab, n - 1)
    avail &= sup[labc] & sup[np.minimum(p + 1, n - 1)][:, None]
    acc = avail & (drafts == g[labc])
    tg = np.zeros(n, bool)
    tg[:-1] = (t[1:] == g[:-1]) & (g[:-1] >= 0)
    obs = np.ones((P, depth), bool)
    for jj in range(1, depth):
        q = np.minimum(p + jj, n - 1)
        obs[:, jj] = obs[:, jj - 1] & tg[q] & (p + jj <= n - 1)
    low_gap = np.nan_to_num(gaps, nan=np.inf) <= CONF_FLOOR
    first = np.where(low_gap.any(1), low_gap.argmax(1) + 1, depth)
    ok = acc & (j[None, :] <= np.minimum(depth, first)[:, None])
    coh = np.zeros(depth, np.int64)
    succ = np.zeros(depth, np.int64)
    prev = np.ones(P, bool)
    for jj in range(depth):
        c = prev & avail[:, jj] & obs[:, jj]
        coh[jj] = c.sum()
        succ[jj] = (c & ok[:, jj]).sum()
        prev &= ok[:, jj]
    return coh, succ


def cluster_key(name: str) -> str:
    return re.sub(r"-w\d{3}\.npz$", ".npz", name)


class Head:
    def __init__(self, name: str, d: Path, codes, subset: str | None):
        self.name, self.dir = name, d
        self.summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        self.anchor_kinds = None
        if (d / "run.json").exists():
            try:
                self.anchor_kinds = json.loads((d / "run.json").read_text())["args"].get("anchor_kinds")
            except Exception:
                pass
        files = sorted((d / "records").glob("*.npz")) if (d / "records").is_dir() else []
        if subset:
            files = [f for f in files if Path(f.name).match(subset)]
        self.counts: dict[str, tuple] = {}
        for f in files:
            if f.name.endswith(".tmp.npz"):
                continue
            with np.load(f, allow_pickle=False) as z:
                c, s = record_counts(z, D, codes)
            k = cluster_key(f.name)
            if k in self.counts:
                c0, s0 = self.counts[k]
                c, s = c0 + c, s0 + s
            self.counts[k] = (c, s)
        self.subset = subset
        self.rec_r = self.rec_km = None
        if self.counts:
            C = np.sum([v[0] for v in self.counts.values()], 0)
            S = np.sum([v[1] for v in self.counts.values()], 0)
            self.rec_r = r_from_counts(C, S)
            self.rec_km = km_of(self.rec_r)
        self.sum_r = self.sum_km = None
        if self.summary is not None:
            s = self.summary["stop"][f"k{D}"]
            self.sum_r, self.sum_km = np.array(s["r"], float), float(s["km"])
        # point estimate: the summary unless a subset is requested (then records)
        if subset or self.sum_r is None:
            self.r, self.km = self.rec_r, self.rec_km
        else:
            self.r, self.km = self.sum_r, self.sum_km
        self.match = None
        if self.rec_r is not None and self.sum_r is not None and not subset:
            self.match = bool(np.allclose(self.rec_r, self.sum_r, atol=6e-5)
                              and len(self.counts) == self.summary.get("records", len(self.counts)))

    def sampled(self, T: str):
        if self.subset or not self.summary:
            return None
        s = (self.summary.get("sampled") or {}).get(T)
        return np.array(s["all"]["alpha_lo"], float) if s and "alpha_lo" in s["all"] else None


def compare(h: Head, ref: Head, B: int, seed: int) -> dict:
    g = partials(ref.r)
    dr = h.r - ref.r
    out = {"head": h.name, "ref": ref.name, "km": h.km, "km_ref": ref.km, "dE": h.km - ref.km,
           "V": float((g * dr).sum()), "V_d12": float((g[:2] * dr[:2]).sum()), "d1": float(dr[0]),
           "d2": float(dr[1]), "dr": dr.tolist()}
    for T in ("T1", "T0.7"):
        a, a0 = h.sampled(T), ref.sampled(T)
        if a is not None and a0 is not None:
            out[f"Vs_{T}"] = float((partials(a0) * (a - a0)).sum())
            out[f"dEs_{T}"] = km_of(a) - km_of(a0)
            out[f"ds1_{T}"] = float(a[0] - a0[0])
    keys = sorted(set(h.counts) & set(ref.counts))
    out["paired_records"] = len(keys)
    if B and len(keys) >= 5:
        Ch = np.array([h.counts[k][0] for k in keys], float)
        Sh = np.array([h.counts[k][1] for k in keys], float)
        Cr = np.array([ref.counts[k][0] for k in keys], float)
        Sr = np.array([ref.counts[k][1] for k in keys], float)
        rng = np.random.default_rng(seed)
        W = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)), size=B).astype(float)
        rh = r_from_counts(W @ Ch, W @ Sh)
        rr = r_from_counts(W @ Cr, W @ Sr)
        gb = partials(rr)
        stats = {"V": (gb * (rh - rr)).sum(1), "d1": rh[:, 0] - rr[:, 0], "d2": rh[:, 1] - rr[:, 1],
                 "dE": (1 + np.cumprod(rh, 1).sum(1)) - (1 + np.cumprod(rr, 1).sum(1))}
        out["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}
        out["p_V_ge_promote"] = float((stats["V"] >= PROMOTE_V).mean())
    out["verdict"] = verdict(out["V"], out["d1"])
    return out


def verdict(V: float, d1: float) -> str:
    if V < KILL_V:
        return "KILL"
    if V >= PROMOTE_V and d1 >= PROMOTE_D1:
        return "PROMOTE"
    return "inconclusive"


def head_name(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        n, d = spec.split("=", 1)
        return n, Path(d)
    d = Path(spec)
    if d.name == "onpolicy" and d.parent.name == "eval-export":
        return d.parent.parent.name, d
    return d.name, d


# ------------------------------------------------------------------------------------------- report
def pp(x):
    return f"{100 * x:+.1f}"


def ci(c, k, scale=1.0, fmt="{:+.3f}"):
    if not c or k not in c:
        return "-"
    lo, hi = c[k]
    return "[" + fmt.format(lo * scale) + "," + fmt.format(hi * scale) + "]"


def report(heads: dict, refs: list[str], B: int, seed: int, fit: bool) -> dict:
    res = {"heads": {}, "comparisons": []}
    print(f"per-depth r (stop, k7){' subset ' + next(iter(heads.values())).subset if next(iter(heads.values())).subset else ''}:")
    print(f"{'head':10s} {'km':>6s}  " + " ".join(f"r{j + 1:d}".rjust(6) for j in range(D)) + "   recs  rec==summary  anchors")
    for n, h in heads.items():
        if h.r is None:
            print(f"{n:10s} (no summary.json and no records)")
            continue
        print(f"{n:10s} {h.km:6.3f}  " + " ".join(f"{x:6.3f}" for x in h.r) +
              f"   {len(h.counts):4d}  {str(h.match):12s} {h.anchor_kinds}")
        res["heads"][n] = {"km": h.km, "r": list(map(float, h.r)), "records": len(h.counts), "records_match_summary": h.match,
                           "anchor_kinds": h.anchor_kinds, "dir": str(h.dir), "dE_dr": partials(h.r).tolist()}
    for rn in refs:
        ref = heads[rn]
        if ref.r is None:
            continue
        g = partials(ref.r)
        print(f"\nreference {rn}: km {ref.km:.3f}  dE/dr_j = " + " ".join(f"{x:.2f}" for x in g) +
              f"  (d1/d7 = {g[0] / g[-1]:.1f}x)")
        if fit:
            print(f"  value-matched --depth-decay: stage-1 objective {fit_decay(g):.3f}; "
                  f"with --chain-weighting survival (tail T_j only) {fit_decay(tail(ref.r)):.3f}")
            res.setdefault("fit_decay", {})[rn] = {"stage1": fit_decay(g), "survival": fit_decay(tail(ref.r))}
        print(f"  {'head':10s} {'V':>7s} {'CI95':>17s} {'V_d12':>6s} {'dE':>7s} {'d1pp':>6s} {'CI95':>13s} "
              f"{'d2pp':>6s} {'CI95':>13s} {'Vs@T1':>6s} {'Vs@.7':>6s}  verdict")
        for n, h in heads.items():
            if n == rn or h.r is None:
                continue
            c = compare(h, ref, B, seed)
            res["comparisons"].append(c)
            cc = c.get("ci95")
            vs1 = f"{c['Vs_T1']:+.3f}" if "Vs_T1" in c else "-"
            vs7 = f"{c['Vs_T0.7']:+.3f}" if "Vs_T0.7" in c else "-"
            print(f"  {n:10s} {c['V']:+7.3f} {ci(cc, 'V'):>17s} {c['V_d12']:+6.3f} {c['dE']:+7.3f} {pp(c['d1']):>6s} "
                  f"{ci(cc, 'd1', 100, '{:+.1f}'):>13s} {pp(c['d2']):>6s} {ci(cc, 'd2', 100, '{:+.1f}'):>13s} "
                  f"{vs1:>6s} {vs7:>6s}  {c['verdict']}  (paired {c['paired_records']})")
    return res


# ------------------------------------------------------------------------------------------ selftest
def _synth_record(rng, n, p_acc, p_greedy_text=0.9, p_prompt=0.15):
    """A synthetic eval_offline npz dict: greedy-matching drafts with per-depth acceptance p_acc[j]."""
    g = rng.integers(0, 50, n)
    t = np.empty(n, np.int64)
    t[0] = rng.integers(0, 50)
    nz = rng.random(n - 1) < p_greedy_text
    t[1:] = np.where(nz, g[:-1], (g[:-1] + 1 + rng.integers(0, 3, n - 1)) % 50)
    sup = rng.random(n) > 0.03
    kinds = np.where(rng.random(n) < p_prompt, 6, 5).astype(np.int8)
    P = n - 1
    lab = np.minimum(np.arange(P)[:, None] + np.arange(1, D + 1)[None, :], n - 1)
    hit = rng.random((P, D)) < np.asarray(p_acc)[None, :]
    drafts = np.where(hit, g[lab], (g[lab] + 7) % 50)
    drafts[rng.random(P) < 0.02] = -1
    gaps = rng.exponential(3.0, (P, D)).astype(np.float32)
    return dict(rid=np.array("synth"), tokens=t, sup_ids0=np.where(sup, g, -1), supervised=sup, kinds=kinds,
                drafts=drafts.astype(np.int64), gaps=gaps, fits=np.ones(P, bool), ctx_miss=np.array([0, 0]))


def selftest() -> int:
    import tempfile
    ok = {}
    rng = np.random.default_rng(7)
    r0 = np.array([.795, .755, .781, .799, .813, .812, .816])
    # 1. partials vs central finite differences, and vs the closed forms (E-1)/r1 and P_6
    g = partials(r0)
    fd = np.array([(km_of(r0 + 1e-6 * np.eye(D)[j]) - km_of(r0 - 1e-6 * np.eye(D)[j])) / 2e-6 for j in range(D)])
    ok["partials_fd"] = bool(np.allclose(g, fd, rtol=1e-6))
    ok["partials_closed"] = bool(abs(g[0] - (km_of(r0) - 1) / r0[0]) < 1e-9 and abs(g[-1] - np.prod(r0[:-1])) < 1e-9)
    # 2. V ~ dE for small perturbations; second-order residual tiny
    dr = np.array([.01, .002, 0, 0, 0, 0, .001])
    ok["V_first_order"] = bool(abs((g * dr).sum() - (km_of(r0 + dr) - km_of(r0))) < 2e-3)
    # 3. decay fit recovers an exact geometric profile
    ok["fit_decay_exact"] = bool(abs(fit_decay(0.7 ** np.arange(D)) - 0.7) < 1e-12)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        heads = {"ref": [.80, .76, .78, .80, .81, .81, .82], "same": [.80, .76, .78, .80, .81, .81, .82],
                 "d1up": [.85, .76, .78, .80, .81, .81, .82]}
        seeds = rng.integers(0, 2**31, 40)
        for hn, pa in heads.items():
            (td / hn / "records").mkdir(parents=True)
            for i, s in enumerate(seeds):
                # same seed per record for every head -> same text and same uniforms; "same" = identical drafts,
                # "d1up" = a superset of depth-1 hits only (higher threshold on the same uniforms)
                rec = _synth_record(np.random.default_rng(int(s)), 300 + 20 * (i % 5), pa)
                np.savez_compressed(td / hn / "records" / f"{i:05d}-synth-{i}.npz", **rec)
        # 4. cross-check counting vs the real eval_offline.summarize when importable (torch + mtp_net)
        try:
            for cand in ("train",
                         str(Path(__file__).resolve().parents[1])):   # train dir next to ablations/, or gpu1 mirror
                if (Path(cand) / "eval_offline.py").exists():
                    sys.path.insert(0, cand)
            import eval_offline as EO                       # noqa: E402
            rep = EO.summarize(td / "ref", D, ["response"])
            h = Head("ref", td / "ref", [KIND_CODES["response"]], None)
            ok["matches_eval_offline_summarize"] = bool(np.allclose(h.rec_r, rep["stop"]["k7"]["r"], atol=6e-5)
                                                       and abs(h.rec_km - rep["stop"]["k7"]["km"]) < 1e-9)
            for hn in heads:
                (td / hn / "summary.json").write_text(json.dumps(EO.summarize(td / hn, D, ["response"])))
        except ImportError as e:
            print("note: eval_offline not importable here (", e, ") - summarize cross-check skipped")
            for hn in heads:
                hh = Head(hn, td / hn, [KIND_CODES["response"]], None)
                (td / hn / "summary.json").write_text(json.dumps(
                    {"records": len(hh.counts), "stop": {"k7": {"r": hh.rec_r.tolist(), "km": hh.rec_km}}}))
        H = {hn: Head(hn, td / hn, [KIND_CODES["response"]], None) for hn in heads}
        ok["records_match_summary"] = all(h.match for h in H.values())
        c_same = compare(H["same"], H["ref"], 500, 1)
        ok["identical_V0_ci_has_0"] = bool(abs(c_same["V"]) < 1e-12 and c_same["ci95"]["V"][0] <= 0 <= c_same["ci95"]["V"][1])
        # d1up re-draws all hits with a higher depth-1 rate (same seeds -> same text): r1 up ~5 pp
        c_up = compare(H["d1up"], H["ref"], 1000, 1)
        ok["d1up_detected"] = bool(0.03 < c_up["d1"] < 0.07 and c_up["ci95"]["d1"][0] > 0 and c_up["V"] > 0.1
                                   and c_up["ci95"]["V"][0] > 0)
        ok["d1up_V_tracks_dE"] = bool(abs(c_up["V"] - c_up["dE"]) < 0.25 * abs(c_up["dE"]) + 0.02)
        # 5. anchor filter: no filter counts more depth-1 cohort than response-only
        hn_all = Head("x", td / "ref", None, None)
        ok["anchor_filter"] = bool(sum(v[0][0] for v in hn_all.counts.values()) > sum(v[0][0] for v in H["ref"].counts.values()))
        # 6. window cluster: a -w001 part merges into its record's cluster
        f0 = sorted((td / "ref" / "records").glob("*.npz"))[0]
        (td / "ref" / "records" / (f0.name[:-4] + "-w001.npz")).write_bytes(f0.read_bytes())
        hw = Head("w", td / "ref", [KIND_CODES["response"]], None)
        ok["window_cluster"] = bool(len(hw.counts) == len(H["ref"].counts)
                                    and (hw.counts[f0.name][0] == 2 * H["ref"].counts[f0.name][0]).all())
        # 7. verdict rule
        ok["verdict_rule"] = (verdict(0.06, 0.006), verdict(0.06, 0.004), verdict(0.019, 0.02), verdict(0.03, 0.01)) == \
            ("PROMOTE", "inconclusive", "KILL", "inconclusive")
        print(f"synthetic: d1up vs ref V {c_up['V']:+.3f} CI {c_up['ci95']['V']} dE {c_up['dE']:+.3f} d1 {pp(c_up['d1'])} pp")
    for k, v in ok.items():
        print(f"  {k:34s} {'PASS' if v else 'FAIL'}")
    allok = all(ok.values())
    print("SELFTEST", "PASS" if allok else "FAIL")
    return 0 if allok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("heads", nargs="*")
    ap.add_argument("--ref", action="append", default=[], help="reference: NAME of a listed head or NAME=DIR")
    ap.add_argument("--anchor-kinds", nargs="*", default=["response"],
                    help="anchor filter for records (default response; pass with no value for all anchors)")
    ap.add_argument("--subset", default=None, help="record file-name glob, e.g. '*-xhigh512.npz' (records-only)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--fit-decay", action="store_true")
    ap.add_argument("--json", default=None, help="write the full result here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    codes = [KIND_CODES[k] for k in a.anchor_kinds] if a.anchor_kinds else None
    heads: dict[str, Head] = {}
    refs = []
    for spec in a.ref:
        if "=" in spec:
            n, d = head_name(spec)
            heads[n] = Head(n, d, codes, a.subset)
            refs.append(n)
        else:
            refs.append(spec)
    for spec in a.heads:
        n, d = head_name(spec)
        if n not in heads:
            heads[n] = Head(n, d, codes, a.subset)
    refs = refs or [next(iter(heads))]
    for rn in refs:
        if rn not in heads:
            raise SystemExit(f"--ref {rn}: not a listed head")
    for n, h in heads.items():
        if h.anchor_kinds is not None and codes is not None and sorted(h.anchor_kinds or []) != sorted(a.anchor_kinds):
            print(f"WARNING {n}: summary.json was made with --anchor-kinds {h.anchor_kinds}; records are filtered with "
                  f"{a.anchor_kinds} (point estimate = summary; use --subset '*' to force records)")
        if h.match is False:
            print(f"WARNING {n}: records-derived r != summary.json r (or record count differs) - CI may not match")
    res = report(heads, refs, a.boot, a.seed, a.fit_decay)
    res["rule"] = {"PROMOTE": f"V >= {PROMOTE_V} and d1 >= {PROMOTE_D1}", "KILL": f"V < {KILL_V}"}
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
