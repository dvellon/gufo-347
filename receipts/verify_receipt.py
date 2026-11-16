"""verify_receipt.py: check a signed gufo benchmark receipt and, on the same hardware class, re-run it.

  python verify_receipt.py RECEIPT.json --allowed-signers FILE                       signature + internal consistency
  python verify_receipt.py RECEIPT.json --allowed-signers FILE --assets A.json --dry-run    + every input hash, prints the plan
  python verify_receipt.py RECEIPT.json --allowed-signers FILE --assets A.json --rerun [--work DIR]   + re-runs the launch

What it checks, in order (it stops at the first failure):
  1. SIGNATURE   `ssh-keygen -Y verify` (OpenSSH >= 8.1) against allowed_signers, principal gufo-bench-receipts, namespace
                 gufo-bench-receipt. Any changed byte, a different key or a missing .sig fails.
  2. CONSISTENCY the transcript sha and the metrics are recomputed from the receipt's own rows; a row receipt's launch
                 receipts must be present, signed and unchanged (sha), and the row's statistic is recomputed from them.
  3. INPUTS      (launch receipts, --dry-run / --rerun) every input you supply in A.json must hash to the receipt's value:
                 server exe, every model shard, MTP head, perm file, workload, client. It refuses on any difference.
  4. RE-RUN      (--rerun; needs the GPU) starts the server with the recorded args + env, replays the recorded workload with
                 the recorded client call, then compares
                   strict: every request's content hashes byte-identical (gate for strict / AR launches at one request in
                           flight; reported only for C8 or the relaxed tier)
                   speed:  the headline within the receipt's band (derived from the run's own launch-to-launch spread).
A.json maps roles to YOUR local paths: {"server_exe": "...", "model": "<first shard>", "mtp_head": "...", "perm": "...",
"workload": "...", "client": "run_agent_replay.py", "upstream_tools": "<stock repo>/tools", "rocm_sdk": "<dir>"}.
Exit: 0 verified / reproduced, 1 not reproduced, 2 signature or consistency failure, 3 refused (input missing or different).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import receipt_lib as L  # noqa: E402


class Stop(Exception):
    def __init__(self, code, verdict, detail=None):
        super().__init__(verdict)
        self.code, self.verdict, self.detail = code, verdict, detail


# ---- 1 + 2 --------------------------------------------------------------------------------------------------------------------
def check_signature(path: Path, a, keygen) -> None:
    ok, msg = L.verify_file(path, Path(str(path) + '.sig'), Path(a.allowed_signers), keygen)
    if not ok:
        raise Stop(2, 'NOT VERIFIED: the signature check failed. The receipt was changed after it was signed, its .sig is '
                      'missing, or it was signed by a key that is not in your allowed_signers file.', msg[-300:])


def check_launch_consistency(r: dict) -> None:
    out = r['outputs']
    if L.transcript_sha(out['transcript']) != out['transcript_sha256']:
        raise Stop(2, 'INCONSISTENT: the transcript does not hash to the recorded transcript sha.')
    st = r['launch']['stage']
    m = L.canon_metrics({'rows': r['timing']}) if st in ('canon', 'warm') else L.replay_metrics(r['timing'], L.STAGE_CONC[st])
    if not L.same_number(m, r['metrics']):
        raise Stop(2, 'INCONSISTENT: the metrics do not follow from the recorded per-request timings.')
    if not L.same_number(L.headline(st, m)[0], r['headline']['value']):
        raise Stop(2, 'INCONSISTENT: the headline value does not match the metrics.')


def check_row(path: Path, r: dict, a, keygen) -> dict:
    recs = {}
    for ln in r['launches']:
        p = path.parent / ln['receipt']
        if not p.exists():
            raise Stop(3, f'REFUSED: the row needs launch receipt {ln["receipt"]}, which is not next to it.')
        if L.sha_bytes(p.read_bytes()) != ln['receipt_sha256']:
            raise Stop(2, f'NOT VERIFIED: launch receipt {ln["receipt"]} is not the one this row was computed from (sha differs).')
        check_signature(p, a, keygen)
        lr = json.loads(p.read_bytes())
        check_launch_consistency(lr)
        recs[ln['lid']] = lr
    spec = {**r['row'], 'launches': [ln['lid'] for ln in r['launches']]}
    got = L.compute_row(spec, recs)
    if not L.same_number(got, r['result']):
        raise Stop(2, 'INCONSISTENT: recomputing the row from its launch receipts gives a different result.',
                   json.dumps({'recomputed': got, 'recorded': r['result']}, default=str)[:600])
    return got


# ---- 3 inputs ------------------------------------------------------------------------------------------------------------------
def _need(assets: dict, role: str) -> Path:
    v = assets.get(role)
    if not v:
        raise Stop(3, f'REFUSED: input "{role}" is not in your assets file, so the launch cannot be reproduced.')
    p = Path(v)
    if not p.exists():
        raise Stop(3, f'REFUSED: input "{role}" was not found at the path you gave.')
    return p


def _same(role: str, p: Path, want: str) -> None:
    got = L.sha_file(p)
    if got != want:
        raise Stop(3, f'REFUSED: your "{role}" ({p.name}) differs from the receipt\'s (sha256 {got[:12]} != {want[:12]}). '
                      'A re-run would not test the same thing.')


def check_inputs(r: dict, assets: dict) -> dict:
    inp, paths = r['inputs'], {}
    paths['server_exe'] = _need(assets, 'server_exe')
    _same('server_exe', paths['server_exe'], inp['server_exe']['sha256'])
    first = _need(assets, 'model')
    for sh in inp['model']['shards']:
        p = first.with_name(sh['name'])
        if not p.exists():
            raise Stop(3, f'REFUSED: model shard {sh["name"]} is not next to the model file you gave.')
        _same(f'model shard {sh["name"]}', p, sh['sha256'])
    paths['model'] = first.with_name(inp['model']['first_shard'])
    for role in ('mtp_head', 'perm'):
        if inp.get(role):
            paths[role] = _need(assets, role)
            _same(role, paths[role], inp[role]['sha256'])
    if inp.get('rocm_sdk'):
        paths['rocm_sdk'] = _need(assets, 'rocm_sdk')
    st = r['launch']['stage']
    if st in ('canon', 'warm'):
        up = _need(assets, 'upstream_tools')
        for rel, k in (('gufo/serving_bench.py', 'serving_bench_sha256'), ('gufo/model_bench/llm.py', 'llm_sha256')):
            _same(f'upstream {rel}', up / rel, inp['client'][k])
        paths['upstream_tools'] = up
    else:
        paths['workload'] = _need(assets, 'workload')
        _same('workload', paths['workload'], inp['workload']['sha256'])
        paths['client'] = _need(assets, 'client')
        _same('client', paths['client'], inp['client']['sha256'])
    return paths


def rehydrate(r: dict, paths: dict, port: int) -> tuple:
    def fill(v):
        if v == '<port>':
            return str(port)
        ph = L.parse_placeholder(v)
        if ph:
            if ph[0] not in paths:
                raise Stop(3, f'REFUSED: the server needs input "{ph[0]}", which you did not supply.')
            return str(paths[ph[0]])
        return v
    argv = [str(paths['server_exe'])] + [fill(v) for v in r['inputs']['server_args']]
    env = {k: fill(v) for k, v in r['inputs']['server_env'].items()}
    return argv, env


# ---- 4 re-run --------------------------------------------------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as x:
            return x.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_canon_again(r: dict, url: str, up: Path) -> list:
    sys.path.insert(0, str(up))
    import gufo.serving_bench as sb          # noqa: E402
    import gufo.model_bench.llm as llm       # noqa: E402
    rc = r['inputs']['workload']['recipe']

    def req(prompt, max_tokens, idx):
        return sb.run_request(base_url=url, model='flash-next', prompt=prompt, max_tokens=max_tokens, temperature=0.0,
                              timeout_seconds=900.0, client_id='public-bench-full', concurrency=1, repetition=1,
                              request_index=idx, endpoint_profile='gufo', cache_prompt=None, messages=None)
    overhead = req('Hi', 1, 0).prompt_tokens - 1
    ratio = (req(llm.synthetic_text(7777, 3000), 1, 0).prompt_tokens - overhead) / 3000
    req(llm.synthetic_text(8888, max(1, round(rc['prompt_tokens'] / ratio))), 16, 0)
    rows = []
    for task, rep in rc['samples']:
        for attempt in range(4):
            prompt = llm.turn_prompt(rc['prompt_tokens'] - overhead, ratio, task=task, depth=0, repetition=rep, attempt=attempt)
            o = req(prompt, rc['output_tokens'], attempt)
            ok = abs(o.prefill_tokens - rc['prompt_tokens']) <= max(32, int(rc['prompt_tokens'] * rc['tolerance']))
            d = {k: getattr(o, k) for k in ('prompt_tokens', 'cached_prompt_tokens', 'completion_tokens', 'prefill_tokens',
                                            'wall_ms', 'client_ttft_ms', 'decode_ms', 'prefill_ms', 'server_ttft_ms',
                                            'draft_tokens', 'draft_accepted_tokens', 'completion_sha256',
                                            'decode_tokens_per_second', 'prefill_tokens_per_second')}
            dec = (o.wall_ms - o.client_ttft_ms) / 1000.0 if o.client_ttft_ms is not None else None
            d.update(task=task, repetition=rep, attempt=attempt, in_tolerance=ok,
                     client_decode_tps=(o.completion_tokens / dec) if dec and dec > 0 else None,
                     prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest())
            if ok or attempt == 3:
                rows.append(d)
                break
            ratio = (o.prefill_tokens + o.cached_prompt_tokens - overhead) / len(prompt.split())
    return rows


def rerun(r: dict, paths: dict, work: Path) -> dict:
    port = free_port()
    argv, env_add = rehydrate(r, paths, port)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GUFO_', 'QFN_'))}
    env.update(env_add)
    work.mkdir(parents=True, exist_ok=False)
    err, out = open(work / 'server.stderr.log', 'wb'), open(work / 'server.stdout.log', 'wb')
    kw = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
    proc = subprocess.Popen(argv, stdout=out, stderr=err, env=env, **kw)
    base = f'http://127.0.0.1:{port}'
    try:
        t0 = time.time()
        while time.time() - t0 < 600 and proc.poll() is None:
            if http_ok(base + '/ready') or http_ok(base + '/v1/models'):
                break
            time.sleep(2)
        else:
            raise Stop(1, 'NOT REPRODUCED: the server did not become ready (see server.stderr.log in the work dir).')
        st = r['launch']['stage']
        if st in ('canon', 'warm'):
            rows = run_canon_again(r, base, paths['upstream_tools'])
            return {'outputs': L.canon_outputs({'rows': rows}), 'metrics': L.canon_metrics({'rows': rows}),
                    'inputs_seen': [x['prompt_sha256'] for x in rows]}
        cl = load_module('receipt_replay_client', paths['client'])
        call = dict(r['inputs']['client']['call'])
        sig = inspect.signature(cl.run_replay).parameters
        kwargs = {k: v for k, v in call.items() if k in sig and k not in ('base_url',)}
        cl.run_replay(base, work / 'replay', paths['workload'], **kwargs)
        rows = [json.loads(x) for x in (work / 'replay' / 'requests.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        outs = L.replay_outputs(rows)
        return {'outputs': outs, 'metrics': L.replay_metrics([L.slim_replay_row(x) for x in rows], L.STAGE_CONC[st]),
                'inputs_seen': [o['request_sha256'] for o in outs if o['turn'] == 0]}
    finally:
        if proc.poll() is None:                           # only the server this script started
            if os.name == 'nt':
                subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        err.close()
        out.close()


def compare(r: dict, got: dict) -> tuple:
    """(code, verdict lines) for a re-run against its receipt."""
    lines, bad = [], False
    st = r['launch']['stage']
    want_in = (r['inputs']['workload'].get('prompt_sha256') if st in ('canon', 'warm') else
               [o['request_sha256'] for o in r['outputs']['transcript'] if o['turn'] == 0])
    if want_in != got['inputs_seen']:
        return 3, ['REFUSED: the re-run sent different first-turn request bodies than the receipt records, so the client or '
                   'workload is not the same.']
    same = L.transcript_sha(got['outputs']) == r['outputs']['transcript_sha256']
    if st in ('canon', 'warm'):
        diff = L.canon_exact(got['outputs'], r['outputs']['transcript'])
    else:
        diff = L.lineage(got['outputs'], r['outputs']['transcript'])
    gate = r['outputs']['strict_compare'] == 'gate'
    if same:
        lines.append('Outputs: byte-identical to the receipt (every request\'s content hashes match).')
    elif gate:
        bad = True
        lines.append(f'Outputs: DIFFERENT from the receipt ({diff["differ"]} request(s) differ, first at '
                     f'{", ".join(diff["first_diffs"][:3]) or "-"}). This launch is strict, so outputs must be identical.')
    else:
        lines.append(f'Outputs: differ from the receipt in {diff["differ"]} request(s); for this launch that is reported, '
                     'not required (' + r['outputs']['strict_compare_why'] + ').')
    hv, _ = L.headline(st, got['metrics'])
    rec, band = r['headline']['value'], r.get('speed_band') or {}
    w = L.within_band(rec, hv, band)
    if w is None:
        lines.append(f'Speed: {r["headline"]["name"]} {hv if hv is None else round(hv, 2)} (receipt {rec}); no band to judge.')
    else:
        lines.append(f'Speed: {r["headline"]["name"]} {hv:.2f} vs {rec:.2f} recorded ({(hv / rec - 1) * 100:+.1f} %), band '
                     f'+-{band["rel"] * 100:.1f} % from the run\'s own launch-to-launch spread: '
                     + ('WITHIN the band.' if w else 'OUTSIDE the band.'))
        bad = bad or not w
    return (1 if bad else 0), (['NOT REPRODUCED.'] if bad else ['REPRODUCED.']) + lines


# ---- main --------------------------------------------------------------------------------------------------------------------
def verify(a) -> tuple:
    path = Path(a.receipt)
    keygen = L.find_keygen(a.ssh_keygen)
    check_signature(path, a, keygen)
    r = json.loads(path.read_bytes())
    sch = r.get('schema')
    if sch == L.SCHEMA_ROW:
        check_row(path, r, a, keygen)
        return 0, [f'VERIFIED: row "{r["row"]["name"]}" is signed, unchanged, and its result recomputes exactly from '
                   f'{len(r["launches"])} signed launch receipt(s).'] + (
                   ['(--dry-run / --rerun apply to launch receipts; run them per launch.)'] if (a.dry_run or a.rerun) else [])
    if sch == L.SCHEMA_INDEX:
        for f in r['receipts']:
            p = path.parent / f['file']
            if not p.exists() or L.sha_bytes(p.read_bytes()) != f['sha256']:
                raise Stop(2, f'NOT VERIFIED: {f["file"]} is missing or changed since the index was signed.')
        return 0, [f'VERIFIED: index is signed and all {len(r["receipts"])} receipts it lists are present and unchanged '
                   '(verify each one for its own signature).']
    if sch != L.SCHEMA_LAUNCH:
        raise Stop(2, f'NOT VERIFIED: unknown receipt schema {sch!r}.')
    check_launch_consistency(r)
    msg = [f'VERIFIED: launch {r["launch"]["lid"]} ({r["launch"]["arm"]}, {r["launch"]["stage"]}, {r["launch"]["tier"]}) is '
           f'signed and self-consistent; headline {r["headline"]["name"]} = {r["headline"]["value"]}.']
    if not (a.dry_run or a.rerun):
        return 0, msg
    if not a.assets:
        raise Stop(3, 'REFUSED: --dry-run / --rerun need --assets (your local paths for the receipt\'s inputs).')
    assets = L.read_json(a.assets)
    if assets is None:
        raise Stop(3, 'REFUSED: the assets file is unreadable.')
    paths = check_inputs(r, assets)
    msg.append('Inputs: every input you supplied hashes to the receipt\'s value.')
    if a.dry_run:
        argv, env = rehydrate(r, paths, 0)
        plan = {'server_argv': argv, 'server_env': env, 'client_call': r['inputs']['client'].get('call'),
                'compare': {'strict': r['outputs']['strict_compare'], 'speed_band': (r.get('speed_band') or {}).get('rel')}}
        msg += ['DRY RUN: nothing was started. A re-run would do:', json.dumps(plan, indent=1)]
        return 0, msg
    work = Path(a.work or f'receipt-rerun-{r["launch"]["lid"]}-{time.strftime("%Y%m%d-%H%M%S")}')
    got = rerun(r, paths, work)
    code, lines = compare(r, got)
    return code, msg + lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='verify (and optionally re-run) a signed gufo benchmark receipt')
    ap.add_argument('receipt')
    ap.add_argument('--allowed-signers', required=True)
    ap.add_argument('--assets')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--rerun', action='store_true')
    ap.add_argument('--work')
    ap.add_argument('--ssh-keygen')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    try:
        code, lines = verify(a)
    except Stop as e:
        code, lines = e.code, [e.verdict] + ([f'  detail: {e.detail}'] if e.detail else [])
    if a.json:
        print(json.dumps({'code': code, 'verdict': lines}))
    else:
        print('\n'.join(lines))
    return code


if __name__ == '__main__':
    sys.exit(main())
