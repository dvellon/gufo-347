"""perm_public.py: rebuild the permuted draft-vocabulary prefix from a PUBLIC corpus (public-bench, DESIGN.md section 4).

Input: a counts .npz written by corpus_counts.py (int64 arrays [248320]: all_tool_call, all_assistant_text,
all_reasoning, miss_tool_call, miss_assistant_text, miss_reasoning, all_prompt) over a PUBLIC corpus. Output: a pairs file
`<drop> <src>` per line (the format the perm knob parses) and a manifest json.

The derivation is the SAME rule the private table used, re-implemented here so that nothing private is read:
  * rank_miss  = rare tokens (id >= 49,152, not special) with assistant-miss count > 0, ordered by (-miss, -all)
                 (rank(asst_miss, asst_all));
  * pre_order  = the 49,152 prefix rows ordered by (assistant + prompt) uses ascending, ties by id descending
                 (np.lexsort((-arange(DV), pre_tot)));
  * K          = min(#prefix rows with 0 uses, len(rank_miss), 16,384 (the knob's pair cap));
  * pairs      = zip(pre_order[:K] minus specials, rank_miss[:len(drop)]).
K is whatever the public corpus gives (the private table had 13,063); the harness checks `pairs=<K>` from the manifest.

  python -I -B perm_public.py build --counts COUNTS.npz --out perm-public.pairs [--corpus-note TEXT] [--source-sha256 HEX ...]
  python -I -B perm_public.py check --pairs perm-public.pairs            (parse + invariants, prints K / fnv)
  python -I -B perm_public.py --selftest                                  (synthetic counts only; no data needed)
No pickle (np.load allow_pickle=False). Writes only the two output files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True

V = 248320            # vocabulary rows (GGUF output.weight)
DV = 49152            # draft prefix rows
SPEC = tuple(range(248044, 248077))   # 33 special ids
PAIR_CAP = 16384      # the knob refuses more pairs
KEYS_ASST = ('tool_call', 'assistant_text', 'reasoning')
SCHEMA = 'public-bench.perm-public.v1'


def _np():
    import numpy as np  # local import: --help works without numpy
    return np


def derive(C) -> dict:
    """C: mapping of the counts arrays. Returns {'pairs', 'K', 'zero_any', 'ranked', ...}."""
    np = _np()
    for k in [f'all_{x}' for x in KEYS_ASST] + [f'miss_{x}' for x in KEYS_ASST] + ['all_prompt']:
        if k not in C:
            raise ValueError(f'counts: missing array {k}')
        if C[k].shape != (V,):
            raise ValueError(f'counts: {k} shape {C[k].shape} != ({V},)')
    asst_all = sum(np.asarray(C[f'all_{x}'], np.int64) for x in KEYS_ASST)
    asst_miss = sum(np.asarray(C[f'miss_{x}'], np.int64) for x in KEYS_ASST)
    rare = np.zeros(V, bool)
    rare[DV:] = True
    rare[list(SPEC)] = False
    sc = np.where(rare, asst_miss, -1)
    order = np.lexsort((-asst_all, -sc))
    rank_miss = [int(t) for t in order if rare[t] and sc[t] > 0]
    pre_tot = asst_all[:DV] + np.asarray(C['all_prompt'], np.int64)[:DV]
    pre_order = np.lexsort((-np.arange(DV), pre_tot))
    zero_any = int((pre_tot == 0).sum())
    k = min(zero_any, len(rank_miss), PAIR_CAP)
    drop = [int(t) for t in pre_order[:k] if int(t) not in SPEC]
    src = rank_miss[:len(drop)]
    pairs = list(zip(drop, src))
    return {'pairs': pairs, 'K': len(pairs), 'zero_any': zero_any, 'ranked': len(rank_miss),
            'asst_tokens': int(asst_all.sum()), 'prompt_tokens': int(np.asarray(C['all_prompt']).sum())}


def fnv64(pairs) -> str:
    """Same fold as the knob's load line (fnv1a64 over little-endian uint32 drop, src)."""
    h = 1469598103934665603
    for a, b in pairs:
        for v in (a, b):
            for byte in struct.pack('<I', v & 0xffffffff):
                h ^= byte
                h = (h * 1099511628211) & 0xffffffffffffffff
    return f'{h:016x}'


def pairs_text(pairs, note: str) -> bytes:
    head = ('# public-bench permuted draft prefix (perm_public.py). Derived from a PUBLIC corpus only.\n'
            f'# {note}\n'
            f'# K={len(pairs)} pairs: <dropped prefix id (0 uses in the corpus)> <rare id (ranked by assistant misses)>\n')
    return (head + ''.join(f'{a} {b}\n' for a, b in pairs)).encode('ascii')


def parse_pairs(b: bytes):
    nums = []
    for line in b.decode('ascii').splitlines():
        nums += [int(x) for x in line.split('#', 1)[0].split()]
    if not nums or len(nums) % 2:
        raise ValueError('empty / odd number of ids')
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def problems(pairs) -> list:
    """The knob's own refusal rules (parse side) + map injectivity."""
    p = []
    drop, src = [d for d, _ in pairs], [s for _, s in pairs]
    if not pairs:
        p.append('no pairs')
    if len(pairs) > PAIR_CAP:
        p.append(f'{len(pairs)} pairs > cap {PAIR_CAP}')
    if len(set(drop)) != len(drop):
        p.append('duplicate drop')
    if len(set(src)) != len(src):
        p.append('duplicate src')
    if any(d < 0 or d >= DV for d in drop):
        p.append('a drop outside [0, 49152)')
    if any(s < DV or s >= V for s in src):
        p.append('a src outside [49152, 248320)')
    if set(SPEC) & (set(src) | set(drop)):
        p.append('a special id in the pairs')
    m = list(range(DV))
    for d, s in pairs:
        if 0 <= d < DV:
            m[d] = s
    if len(set(m)) != DV:
        p.append('index -> id map not injective')
    return p


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cmd_build(a) -> int:
    np = _np()
    raw = Path(a.counts).read_bytes()
    with np.load(a.counts, allow_pickle=False) as z:
        C = {k: z[k] for k in z.files}
    d = derive(C)
    pr = problems(d['pairs'])
    if pr:
        print('REFUSED: ' + '; '.join(pr))
        return 2
    note = a.corpus_note or 'corpus: (not stated)'
    body = pairs_text(d['pairs'], note)
    out = Path(a.out)
    out.write_bytes(body)
    man = {'schema': SCHEMA, 'pairs_file': out.name, 'pairs_sha256': sha256(body), 'K': d['K'], 'fnv64': fnv64(d['pairs']),
           'expected_knob_line': f"event=draft_perm pairs={d['K']} prefix={DV} specials={len(SPEC)} fnv={fnv64(d['pairs'])}",
           'zero_use_prefix_rows': d['zero_any'], 'ranked_rare_tokens': d['ranked'], 'assistant_tokens': d['asst_tokens'],
           'prompt_tokens': d['prompt_tokens'], 'counts_sha256': sha256(raw), 'corpus_note': note,
           'corpus_source_sha256': list(a.source_sha256 or []), 'rule': __doc__.split('\n\n')[2].strip()}
    Path(str(out) + '.manifest.json').write_text(json.dumps(man, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: man[k] for k in ('K', 'fnv64', 'pairs_sha256', 'zero_use_prefix_rows', 'ranked_rare_tokens')}))
    return 0


def cmd_check(a) -> int:
    b = Path(a.pairs).read_bytes()
    pairs = parse_pairs(b)
    pr = problems(pairs)
    print(json.dumps({'K': len(pairs), 'fnv64': fnv64(pairs), 'sha256': sha256(b), 'problems': pr}))
    return 0 if not pr else 2


def selftest() -> int:
    np = _np()
    ok = 0
    fails = []

    def t(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fails.append(name)

    rng = np.random.default_rng(1992)
    C = {f'all_{x}': np.zeros(V, np.int64) for x in KEYS_ASST}
    C.update({f'miss_{x}': np.zeros(V, np.int64) for x in KEYS_ASST})
    C['all_prompt'] = np.zeros(V, np.int64)
    # prefix: every row used except 300 rows (ids 1000..1299) -> zero_any 300
    C['all_prompt'][:DV] = rng.integers(1, 50, DV)
    C['all_prompt'][1000:1300] = 0
    # rare tokens: 500 with misses; give id 60000 the top miss, 60001 second, a tie broken by 'all'
    ids = rng.choice(np.arange(DV, 248000), 500, replace=False)
    C['miss_reasoning'][ids] = rng.integers(1, 20, 500)
    C['miss_tool_call'][60000] = 1000
    C['miss_tool_call'][60001] = 900
    C['miss_assistant_text'][60002] = 900
    C['all_assistant_text'][60002] = 5          # tie on miss (900) -> higher 'all' ranks first
    C['miss_tool_call'][248050] = 10 ** 6        # a special id: never ranked
    d = derive(C)
    t('zero_any', d['zero_any'] == 300)
    t('K = zero rows', d['K'] == 300)
    t('top src', d['pairs'][0][1] == 60000)
    t('tie by all', [s for _, s in d['pairs'][1:3]] == [60002, 60001])
    t('special not ranked', all(s not in SPEC for _, s in d['pairs']))
    t('drops are the zero rows', set(x for x, _ in d['pairs']) == set(range(1000, 1300)))
    t('drop order: ties by id descending', d['pairs'][0][0] == 1299)
    t('no problems', problems(d['pairs']) == [])
    # K limited by ranked tokens
    C2 = {k: v.copy() for k, v in C.items()}
    C2['all_prompt'][2000:42000] = 0
    d2 = derive(C2)
    t('K = min(zero, ranked)', d2['K'] == min(d2['zero_any'], d2['ranked']) and d2['K'] <= PAIR_CAP)
    # cap
    C3 = {k: v.copy() for k, v in C.items()}
    C3['all_prompt'][:DV] = 0
    C3['miss_reasoning'][DV:DV + 20000] = 1
    d3 = derive(C3)
    t('cap 16384', d3['K'] == PAIR_CAP)
    # text round trip + fnv stable
    b = pairs_text(d['pairs'], 'selftest')
    t('round trip', parse_pairs(b) == d['pairs'])
    t('fnv deterministic', fnv64(d['pairs']) == fnv64(parse_pairs(b)))
    # mutants refused
    t('dup drop refused', problems(d['pairs'] + [(d['pairs'][0][0], 70000)]) != [])
    t('src in prefix refused', problems([(5, 10)]) != [])
    t('drop outside prefix refused', problems([(50000, 60000)]) != [])
    t('special refused', problems([(5, 248044)]) != [])
    t('odd parse refused', _raises(lambda: parse_pairs(b'1 2 3\n')))
    t('bad shape refused', _raises(lambda: derive({**C, 'all_prompt': np.zeros(10, np.int64)})))
    print(f'perm_public selftest: {ok} PASS, {len(fails)} FAIL' + (f' {fails}' if fails else ''))
    return 0 if not fails else 1


def _raises(f) -> bool:
    try:
        f()
    except (ValueError, KeyError):
        return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    sub = ap.add_subparsers(dest='cmd')
    b = sub.add_parser('build')
    b.add_argument('--counts', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--corpus-note')
    b.add_argument('--source-sha256', nargs='*')
    c = sub.add_parser('check')
    c.add_argument('--pairs', required=True)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == 'build':
        return cmd_build(a)
    if a.cmd == 'check':
        return cmd_check(a)
    ap.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
