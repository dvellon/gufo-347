"""receipt_lib.py: shared code for signed, re-runnable gufo benchmark receipts (make_receipts.py, verify_receipt.py).

Meant to be publishable with verify_receipt.py: it holds no paths, no host names and no private asset data. The metric and
statistic functions are byte-for-byte the definitions of the benchmark runner (run_pb.py), so a third party recomputes the
same numbers from a receipt; the selftest checks that against the runner itself.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
from pathlib import Path

SCHEMA_LAUNCH = 'gufo.bench-receipt.launch.v1'
SCHEMA_ROW = 'gufo.bench-receipt.row.v1'
SCHEMA_INDEX = 'gufo.bench-receipt.index.v1'
NAMESPACE = 'gufo-bench-receipt'          # ssh-keygen -Y sign / verify namespace
PRINCIPAL = 'gufo-bench-receipts'         # identity in allowed_signers
OUTPUT_FIELDS = ('reasoning_sha256', 'content_sha256', 'tool_calls_sha256', 'completion_tokens', 'finish_reason')
C8_MIN_FULL_S, C8_MIN_FULL_SHARE = 30.0, 0.25
BOOT_N, BOOT_SEED = 4000, 260
BAND_FLOOR, BAND_K = 0.03, 2.0            # speed band: max(3 %, 2 x the arm's launch-to-launch CV in that stage)
STAGE_CONC = {'canon': 1, 'warm': 1, 'c1': 1, 'c8': 8, 'chat': 1}
STAGE_STAT_NAME = {'c1': 'pooled client decode tok/s (sum completion tokens / sum client decode s)',
                   'chat': 'pooled client decode tok/s (sum completion tokens / sum client decode s)',
                   'c8': 'mean full-window tok/s (>= 8 in flight, valid launches only)',
                   'canon': 'mean server tg tok/s (prose samples)'}
HEADLINE_NAME = {'c1': 'client_decode_tps', 'chat': 'client_decode_tps', 'c8': 'full_window_tps',
                 'canon': 'prose.tg_server_tps', 'warm': None}


# ---- bytes, hashes, json --------------------------------------------------------------------------------------------------------
def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class HashCache:
    """sha256 of large files, cached by (path, size, mtime_ns) in a local json (never published)."""

    def __init__(self, path=None):
        self.path, self.d = (Path(path) if path else None), {}
        if self.path and self.path.exists():
            try:
                self.d = json.loads(self.path.read_text(encoding='utf-8'))
            except ValueError:
                self.d = {}

    def sha(self, p) -> str:
        p = Path(p)
        st = p.stat()
        k = f'{p.resolve()}|{st.st_size}|{st.st_mtime_ns}'
        if k not in self.d:
            self.d[k] = sha_file(p)
            if self.path:
                self.path.write_text(json.dumps(self.d, indent=0), encoding='utf-8')
        return self.d[k]


def canon_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def receipt_bytes(obj) -> bytes:
    """the exact bytes written to disk and signed (stable: sorted keys, ASCII, LF)."""
    return (json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default


# ---- time ---------------------------------------------------------------------------------------------------------------------
def _et_offset_manual(utc: dt.datetime) -> tuple:
    """US Eastern: DST from the 2nd Sunday of March 07:00 UTC to the 1st Sunday of November 06:00 UTC."""
    def nth_sunday(month, n):
        d = dt.datetime(utc.year, month, 1, tzinfo=dt.timezone.utc)
        return d + dt.timedelta(days=(6 - d.weekday()) % 7 + 7 * (n - 1))
    start = nth_sunday(3, 2).replace(hour=7)
    end = nth_sunday(11, 1).replace(hour=6)
    return (-4, 'EDT') if start <= utc < end else (-5, 'EST')


def stamps(epoch) -> dict:
    """{utc, et} for an epoch; ET via zoneinfo when present, else the fixed US rule (Windows Python may lack tzdata)."""
    if epoch is None:
        return {'utc': None, 'et': None}
    utc = dt.datetime.fromtimestamp(float(epoch), dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        loc = utc.astimezone(ZoneInfo('America/New_York'))
        et = loc.strftime('%Y-%m-%d %H:%M:%S ') + loc.tzname()
    except Exception:  # noqa: BLE001  (ZoneInfoNotFoundError, ImportError)
        off, name = _et_offset_manual(utc)
        et = (utc + dt.timedelta(hours=off)).strftime('%Y-%m-%d %H:%M:%S ') + name
    return {'utc': utc.strftime('%Y-%m-%dT%H:%M:%SZ'), 'et': et}


def stamps_manual(epoch) -> dict:
    utc = dt.datetime.fromtimestamp(float(epoch), dt.timezone.utc)
    off, name = _et_offset_manual(utc)
    return {'utc': utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'et': (utc + dt.timedelta(hours=off)).strftime('%Y-%m-%d %H:%M:%S ') + name}


# ---- metrics (identical to run_pb.py) ----------------------------------------------------------------------------------------
def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def segments(rows: list) -> list:
    ev = sorted([(r['t_start_s'], 1) for r in rows] + [(r['t_end_s'], -1) for r in rows], key=lambda x: (x[0], x[1]))
    segs, act, prev = [], 0, None
    for t, dlt in ev:
        if prev is not None and t > prev:
            segs.append((prev, t, act))
        act += dlt
        prev = t
    return segs


def c8_window(rows: list, n: int) -> dict:
    ok = [r for r in rows if not r.get('error')]
    if not rows:
        return {'n': n, 'full_s': 0.0, 'tok_s': None, 'span_s': None, 'share': None, 'max_in_flight': 0}
    segs = segments(rows)
    full = [(a, b) for a, b, k in segs if k >= n]
    fs = sum(b - a for a, b in full)
    span = max(r['t_end_s'] for r in rows) - min(r['t_start_s'] for r in rows)
    tok = 0.0
    for r in ok:
        if r.get('t_first_output_ms') is None or not r.get('completion_tokens'):
            continue
        d0, d1 = r['t_start_s'] + r['t_first_output_ms'] / 1000.0, r['t_end_s']
        if d1 <= d0:
            continue
        tok += r['completion_tokens'] * sum(max(0.0, min(d1, b) - max(d0, a)) for a, b in full) / (d1 - d0)
    return {'n': n, 'full_s': round(fs, 3), 'tok_s': (tok / fs) if fs > 0 else None, 'span_s': round(span, 3),
            'share': round(fs / span, 4) if span else None, 'max_in_flight': max((k for _, _, k in segs), default=0)}


def replay_metrics(rows: list, workers: int) -> dict:
    ok = [r for r in rows if not r.get('error')]
    comp = sum(r.get('completion_tokens') or 0 for r in ok)
    dec = sum((r['wall_ms'] - r['t_first_output_ms']) / 1000.0 for r in ok if r.get('t_first_output_ms') is not None)
    sdec = sum(((r.get('usage') or {}).get('gufo') or {}).get('decode_ms') or 0 for r in ok) / 1000.0
    drafted = sum((r.get('usage') or {}).get('draft_tokens') or 0 for r in ok)
    acc = sum((r.get('usage') or {}).get('draft_tokens_accepted') or 0 for r in ok)
    rounds = sum(((r.get('usage') or {}).get('gufo') or {}).get('draft_rounds') or 0 for r in ok)
    ttft_cold = [r['t_first_output_ms'] for r in ok if r['turn'] == 0]
    ttft_warm = [r['t_first_output_ms'] for r in ok if r['turn'] > 0]
    span = (max(r['t_end_s'] for r in ok) - min(r['t_start_s'] for r in ok)) if ok else None
    m = {'requests': len(rows), 'errors': len(rows) - len(ok), 'completion_tokens': comp, 'decode_s': dec,
         'client_decode_tps': comp / dec if dec > 0 else None, 'server_decode_tps': comp / sdec if sdec > 0 else None,
         'ttft_cold_p50_ms': _pct(ttft_cold, 0.5), 'ttft_warm_p50_ms': _pct(ttft_warm, 0.5),
         'ttft_warm_p90_ms': _pct(ttft_warm, 0.9), 'acceptance': acc / drafted if drafted else None,
         'accepted_per_round': acc / rounds if rounds else None,
         'tokens_per_target_pass': comp / (comp - acc) if comp > acc and drafted else None,
         'makespan_tps': comp / span if span else None, 'span_s': span,
         'gap_max_ms': max((((r.get('usage') or {}).get('gufo') or {}).get('max_inter_token_ms') or 0) for r in ok) if ok else None}
    if workers > 1:
        w = c8_window(rows, workers)
        w7 = c8_window(rows, workers - 1)
        valid = bool(w['tok_s']) and w['full_s'] >= C8_MIN_FULL_S and (w['share'] or 0) >= C8_MIN_FULL_SHARE
        m.update(full_window_tps=w['tok_s'], full_window_s=w['full_s'], full_window_share=w['share'],
                 max_in_flight=w['max_in_flight'], c8_valid=valid, c8_flag=None if valid else 'C8-THIN',
                 ge7_window_tps=w7['tok_s'], ge7_window_s=w7['full_s'])
    return m


def canon_metrics(doc: dict, output_tokens: int = 128) -> dict:
    rows = (doc or {}).get('rows') or []
    out = {}
    for task in ('prose', 'repetition'):
        rr = [r for r in rows if r['task'] == task]
        if not rr:
            continue
        drafted = sum(r['draft_tokens'] or 0 for r in rr)
        acc = sum(r['draft_accepted_tokens'] or 0 for r in rr)
        comp = sum(r['completion_tokens'] for r in rr)
        out[task] = {'n': len(rr), 'tg_server_tps': statistics.fmean([r['decode_tokens_per_second'] for r in rr
                                                                    if r['decode_tokens_per_second']] or [0]) or None,
                     'tg_client_tps': statistics.fmean([r['client_decode_tps'] for r in rr if r['client_decode_tps']] or [0]) or None,
                     'pp_server_tps': statistics.fmean([r['prefill_tokens_per_second'] for r in rr
                                                        if r['prefill_tokens_per_second']] or [0]) or None,
                     'ttft_p50_ms': _pct([r['client_ttft_ms'] for r in rr], 0.5),
                     'acceptance': acc / drafted if drafted else None,
                     'tokens_per_target_pass': comp / (comp - acc) if comp > acc and drafted else None,
                     'all_in_tolerance': all(r['in_tolerance'] for r in rr),
                     'all_full_length': all(r['completion_tokens'] == output_tokens for r in rr)}
    return out


def headline(stage: str, metrics: dict):
    """(value, counts) for one launch, as run_pb.headline_metrics decides whether a launch counts."""
    if stage == 'canon':
        v = ((metrics or {}).get('prose') or {}).get('tg_server_tps')
        return v, bool(v)
    if stage == 'c8':
        return metrics.get('full_window_tps'), bool(metrics.get('c8_valid'))
    return metrics.get('client_decode_tps'), bool(metrics.get('decode_s'))


# ---- transcript (outputs) and timing (for metrics) ----------------------------------------------------------------------------
def slim_replay_row(r: dict) -> dict:
    """the fields replay_metrics / lineage read, nothing else."""
    u = r.get('usage') or {}
    g = u.get('gufo') or {}
    return {'session': r.get('session'), 'turn': r.get('turn'), 'attempt': r.get('attempt', 0), 'worker': r.get('worker'),
            'error': bool(r.get('error')), 'completion_tokens': r.get('completion_tokens'), 'wall_ms': r.get('wall_ms'),
            't_first_output_ms': r.get('t_first_output_ms'), 't_start_s': r.get('t_start_s'), 't_end_s': r.get('t_end_s'),
            'usage': {'draft_tokens': u.get('draft_tokens'), 'draft_tokens_accepted': u.get('draft_tokens_accepted'),
                      'gufo': {'decode_ms': g.get('decode_ms'), 'draft_rounds': g.get('draft_rounds'),
                               'max_inter_token_ms': g.get('max_inter_token_ms')}}}


def replay_outputs(rows: list) -> list:
    """per request, in client order: the request body sha and the content hashes of the output (the strict transcript)."""
    out = []
    for r in rows:
        o = {'session': r.get('session'), 'turn': r.get('turn'), 'attempt': r.get('attempt', 0),
             'request_sha256': r.get('request_sha256'), 'error': bool(r.get('error'))}
        o.update({k: r.get(k) for k in OUTPUT_FIELDS})
        out.append(o)
    return sorted(out, key=lambda x: (str(x['session']), x['turn'] or 0, x['attempt'] or 0))


CANON_OUT = ('task', 'repetition', 'attempt', 'prompt_sha256', 'completion_sha256', 'completion_tokens')


def canon_outputs(doc: dict) -> list:
    return [{k: r.get(k) for k in CANON_OUT} for r in (doc or {}).get('rows') or []]


def transcript_sha(outputs: list) -> str:
    return sha_bytes(canon_json(outputs))


# ---- exactness (identical to run_pb.py, on transcript rows) -------------------------------------------------------------------
def lineage(spec_rows: list, ref_rows: list) -> dict:
    ref = {(r['session'], r['turn']): r for r in ref_rows if r.get('attempt', 0) == 0 and not r.get('error')}
    by = {}
    for r in spec_rows:
        if r.get('attempt', 0) == 0:
            by.setdefault(r['session'], []).append(r)
    same = differ = notc = 0
    first = []
    for s, rr in sorted(by.items()):
        broken = False
        for r in sorted(rr, key=lambda x: x['turn']):
            q = ref.get((s, r['turn']))
            if broken or q is None or r.get('error'):
                notc += 1
                continue
            if all(r.get(k) == q.get(k) for k in OUTPUT_FIELDS):
                same += 1
            else:
                differ += 1
                broken = True
                first.append(f'{s}:{r["turn"]}')
    return {'identical': same, 'differ': differ, 'not_compared': notc, 'first_diffs': first[:8]}


def canon_exact(spec_rows: list, ref_rows: list) -> dict:
    ref = {(r['task'], r['repetition'], r['prompt_sha256']): r['completion_sha256'] for r in ref_rows or []}
    same = differ = notc = 0
    for r in spec_rows or []:
        k = (r['task'], r['repetition'], r['prompt_sha256'])
        if k not in ref:
            notc += 1
        elif ref[k] == r['completion_sha256']:
            same += 1
        else:
            differ += 1
    return {'identical': same, 'differ': differ, 'not_compared': notc, 'first_diffs': []}


def pool(parts: list, gate: bool) -> dict:
    s = {k: sum(p[k] for p in parts) for k in ('identical', 'differ', 'not_compared')}
    tot = sum(s.values())
    s['first_diffs'] = [x for p in parts for x in p.get('first_diffs') or []][:8]
    s['launches'] = len(parts)
    if s['differ']:
        v = 'FAIL-EXACT' if gate else 'DIFFERS (report only)'
    elif tot and s['identical'] >= 0.5 * tot:
        v = 'PASS' if gate else 'IDENTICAL (report only)'
    else:
        v = 'NO-DATA'
    s['verdict'] = v
    return s


# ---- statistics -----------------------------------------------------------------------------------------------------------------
def stat_pooled_rate(ms: list):
    t, d = sum(m['completion_tokens'] for m in ms), sum(m['decode_s'] for m in ms)
    return t / d if d else None


def stat_mean(key):
    def f(ms):
        v = [m[key] for m in ms if m.get(key)]
        return statistics.fmean(v) if v else None
    return f


STAGE_STAT = {'c1': stat_pooled_rate, 'chat': stat_pooled_rate, 'c8': stat_mean('full_window_tps'),
              'canon': stat_mean('tg_server_tps')}


def boot_ratio(rounds: list, num: str, den: str, stat) -> dict:
    """run_pb's statistic: paired percentile bootstrap over ROUNDS, BOOT_N resamples, seed BOOT_SEED."""
    rr = [x for x in rounds if x.get(num) is not None and x.get(den) is not None]
    if len(rr) < 2:
        return {'point': None, 'ci': None, 'rounds': len(rr)}

    def ratio(sel):
        a, b = stat([x[num] for x in sel]), stat([x[den] for x in sel])
        return a / b if a and b else None
    pt = ratio(rr)
    rng = random.Random(BOOT_SEED)
    bs = sorted(v for v in (ratio([rng.choice(rr) for _ in rr]) for _ in range(BOOT_N)) if v is not None)
    return {'point': pt, 'ci': [bs[int(0.025 * len(bs))], bs[min(len(bs) - 1, int(0.975 * len(bs)))]] if bs else None,
            'rounds': len(rr)}


def units_of(stage: str, launch_rec: dict) -> dict:
    """second-level units of one launch: key -> (numerator, denominator) whose pooled ratio is the launch statistic.
    c1 / chat: per session (completion tokens, client decode s); canon: per prose sample (server tg tok/s, 1);
    c8: the launch itself (full-window tok/s, 1): the full window is not separable by session."""
    m = launch_rec['metrics']
    if stage == 'c8':
        return {'launch': (m['full_window_tps'], 1.0)}
    if stage == 'canon':
        return {f'{r["task"]}:{r["repetition"]}': (r['decode_tokens_per_second'], 1.0)
                for r in launch_rec['timing'] if r['task'] == 'prose' and r.get('decode_tokens_per_second')}
    u = {}
    for r in launch_rec['timing']:
        if r.get('error') or r.get('t_first_output_ms') is None:
            continue
        a, b = u.get(r['session'], (0.0, 0.0))
        u[r['session']] = (a + (r.get('completion_tokens') or 0), b + (r['wall_ms'] - r['t_first_output_ms']) / 1000.0)
    return u


def boot_two_level(rounds_units: list, num: str, den: str, n=BOOT_N, seed=BOOT_SEED) -> dict:
    """two-level paired percentile bootstrap: resample rounds, then (paired across arms) the units inside each drawn round.
    rounds_units: [{arm: {key: (x, y)}}]; statistic per arm = sum x / sum y over the drawn units."""
    rr = [x for x in rounds_units if x.get(num) and x.get(den)]
    if len(rr) < 2:
        return {'point': None, 'ci': None, 'rounds': len(rr)}

    def arm_stat(sel, arm):
        sx = sy = 0.0
        for rnd, keys in sel:
            u = rnd[arm]
            for k in keys:
                if k in u:
                    sx += u[k][0]
                    sy += u[k][1]
        return sx / sy if sy else None

    def ratio(sel):
        a, b = arm_stat(sel, num), arm_stat(sel, den)
        return a / b if a and b else None
    full = [(x, sorted(set(x[num]) | set(x[den]))) for x in rr]
    pt = ratio(full)
    rng = random.Random(seed)
    bs = []
    for _ in range(n):
        sel = []
        for _ in rr:
            x = rng.choice(rr)
            keys = sorted(set(x[num]) | set(x[den]))
            sel.append((x, [rng.choice(keys) for _ in keys]))
        v = ratio(sel)
        if v is not None:
            bs.append(v)
    bs.sort()
    return {'point': pt, 'ci': [bs[int(0.025 * len(bs))], bs[min(len(bs) - 1, int(0.975 * len(bs)))]] if bs else None,
            'rounds': len(rr), 'resamples': n, 'seed': seed}


def speed_band(values: list) -> dict:
    """tolerance band for a re-run's headline, from this run's own launch-to-launch spread of the same arm and stage."""
    v = [x for x in values if x]
    if len(v) < 2:
        return {'rel': None, 'n': len(v), 'rule': 'fewer than 2 launches: no band, speed not judged'}
    mean, sd = statistics.fmean(v), statistics.stdev(v)
    cv = sd / mean if mean else None
    rel = max(BAND_FLOOR, BAND_K * cv) if cv is not None else None
    return {'rel': rel, 'n': len(v), 'mean': mean, 'stdev': sd, 'cv': cv, 'min': min(v), 'max': max(v),
            'rule': f'band = max({BAND_FLOOR:.0%}, {BAND_K:g} x CV of the launch headline over this arm\'s timed launches '
                    'in this stage, same run)'}


def within_band(recorded, rerun, band: dict):
    if not recorded or not rerun or not band or band.get('rel') is None:
        return None
    return abs(rerun / recorded - 1.0) <= band['rel']


# ---- signing (OpenSSH ssh-keygen -Y; no extra packages) -----------------------------------------------------------------------
def find_keygen(override=None) -> str:
    k = override or os.environ.get('GUFO_SSH_KEYGEN') or shutil.which('ssh-keygen')
    if not k:
        raise SystemExit('ssh-keygen not found (OpenSSH >= 8.1 is needed; pass --ssh-keygen)')
    return k


def sign_file(path: Path, key: Path, keygen: str) -> Path:
    sig = Path(str(path) + '.sig')
    if sig.exists():
        sig.unlink()
    r = subprocess.run([keygen, '-Y', 'sign', '-f', str(key), '-n', NAMESPACE, str(path)], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=120)
    if r.returncode != 0 or not sig.exists():
        raise RuntimeError(f'ssh-keygen -Y sign failed rc {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}')
    return sig


def verify_file(path: Path, sig: Path, allowed_signers: Path, keygen: str, principal: str = PRINCIPAL) -> tuple:
    if not Path(sig).exists():
        return False, 'signature file missing'
    with open(path, 'rb') as f:
        r = subprocess.run([keygen, '-Y', 'verify', '-f', str(allowed_signers), '-I', principal, '-n', NAMESPACE, '-s',
                            str(sig)], stdin=f, capture_output=True, text=True, timeout=120)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def key_fingerprint(pub: Path, keygen: str):
    r = subprocess.run([keygen, '-l', '-E', 'sha256', '-f', str(pub)], capture_output=True, text=True, timeout=60)
    parts = r.stdout.split()
    return parts[1] if r.returncode == 0 and len(parts) > 1 else None


# ---- placeholders (receipts carry roles, never paths) -------------------------------------------------------------------------
def placeholder(role: str, name: str) -> str:
    return f'<{role}:{name}>'


def parse_placeholder(v: str):
    if isinstance(v, str) and v.startswith('<') and v.endswith('>') and ':' in v:
        role, name = v[1:-1].split(':', 1)
        return role, name
    return None


# ---- result rows (computed only from launch receipts, so a third party recomputes them) ---------------------------------------
def headline_metrics_of(rec: dict):
    """the per-launch metrics dict a stage statistic uses (run_pb.headline_metrics), or None when it does not count."""
    if rec['launch']['state'] != 'DONE':
        return None
    st, m = rec['launch']['stage'], rec['metrics']
    if st == 'canon':
        p = m.get('prose') or {}
        return {'tg_server_tps': p.get('tg_server_tps')} if p.get('tg_server_tps') else None
    if st == 'c8':
        return m if m.get('c8_valid') else None
    return m if m.get('decode_s') else None


def compute_row(spec: dict, recs: dict) -> dict:
    """spec: {kind: ratio|arm|exactness, stage, num, den | arm | pairs [[spec_lid, ref_lid]], gate}; recs: lid -> receipt."""
    st = spec['stage']
    if spec['kind'] == 'exactness' and spec.get('pairs_vs_pooled_ref'):
        ref = [o for lid in spec['pairs_vs_pooled_ref'] for o in recs[lid]['outputs']['transcript']]
        return {'exactness': pool([lineage(recs[lid]['outputs']['transcript'], ref) for lid in spec['spec_launches']],
                                  spec['gate'])}
    if spec['kind'] == 'exactness':
        parts = []
        for a, b in spec['pairs']:
            ra, rb = recs[a]['outputs']['transcript'], recs[b]['outputs']['transcript']
            parts.append(canon_exact(ra, rb) if st == 'canon' else lineage(ra, rb))
        return {'exactness': pool(parts, spec['gate'])}
    rounds, units = {}, {}
    for lid in spec['launches']:
        r = recs[lid]
        hm = headline_metrics_of(r)
        if hm is None or r['launch']['round'] is None:
            continue
        rounds.setdefault(r['launch']['round'], {})[r['launch']['arm']] = hm
        units.setdefault(r['launch']['round'], {})[r['launch']['arm']] = units_of(st, r)
    rl = [rounds[k] for k in sorted(rounds)]
    ul = [units[k] for k in sorted(units)]
    if spec['kind'] == 'arm':
        ms = [x[spec['arm']] for x in rl if spec['arm'] in x]
        return {'value': STAGE_STAT[st](ms) if ms else None, 'launches': len(ms)}
    return {'run_statistic': boot_ratio(rl, spec['num'], spec['den'], STAGE_STAT[st]),
            'two_level': boot_two_level(ul, spec['num'], spec['den'])}


def same_number(a, b, rel=1e-9) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same_number(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_number(a[k], b[k], rel) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= rel * max(1.0, abs(a), abs(b))
    return a == b
