"""build_sessions_public.py: PUBLIC workloads -> replay sessions (public-bench DESIGN.md section 2). STUB, host-checked.

Readers (each fail-closed on an unexpected format; nothing is fetched, inputs are local files the owner approved):
  mtbench   FastChat mt_bench/question.jsonl            {question_id, category, turns[2]}           -> chat sessions
  tau       tau-bench historical_trajectories/*.json    [{task_id, traj:[messages]}]                -> tool-agent sessions
  sweagent  SWE-smith-trajectories parquet (tool split) rows {traj_id, instance_id, messages(json)} -> coding-agent sessions
  openhands nebius SWE-rebench parquet                  rows {trajectory_id, instance_id, trajectory, tools}
  corpus    any agent reader -> gufo.agent-transcript-corpus.v1 jsonl (the perm corpus, DESIGN 4.2; --exclude-instances)
  tau-tools tau-bench envs/<domain>/tools/*.py -> OpenAI tool schemas, by STATIC parsing (ast.literal_eval of the
            get_info() return literal); the downloaded code is never imported or run.

Output schema = gufo.agent-replay-sessions.v1 (what our replay client runs): {schema, params, rendering, harness_tools,
sessions:[{id, source, harness, system:[msg], turns:[{inputs:[msg], input_meta:[...]}]}]}. Assistant messages of the
transcript are DROPPED: the replay sends the server's own replies back (performance workload, not an accuracy benchmark).
Selection is deterministic: sessions ordered by sha256(seed + ':' + id), first N that pass the caps. Manifest: sha256 of
every input file and of the output, params, selected ids; `--verify` regenerates and compares bytes.

STUB LIMITS (to finish in S1, after the downloads): token caps use a 4-chars-per-token estimate unless --tokenizer-dir
points at agent-replay/agent_template.py's directory (then exact host counts); the SWE-smith message layout is checked,
not assumed (B6 in DESIGN.md).

  python -I -B build_sessions_public.py mtbench  --in question.jsonl --out chat.json [--n 24 --per-category 3]
  python -I -B build_sessions_public.py tau      --in gpt-4o-retail.json --tools retail-tools.json --out tau.json [--n 12]
  python -I -B build_sessions_public.py sweagent --in tool-00004.parquet --out code.json [--n 12] [--tools tools.json]
  python -I -B build_sessions_public.py corpus   --in tool-00002.parquet --reader sweagent --out corpus.jsonl --exclude-instances code.json
  python -I -B build_sessions_public.py tau-tools --in <tau_bench/envs/retail/tools dir> --out retail-tools.json
  python -I -B build_sessions_public.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SCHEMA = 'gufo.agent-replay-sessions.v1'
CORPUS_SCHEMA = 'gufo.agent-transcript-corpus.v1'
SEED = 'public-bench-v1'
CAPS = {'tool': 6144, 'user': 8192, 'system': 8192}
CHARS_PER_TOKEN = 4.0
HISTORY_REPLY = 256


class FormatError(ValueError):
    pass


def sha256b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def order_key(i: str, seed: str = SEED) -> str:
    return sha256b(f'{seed}:{i}'.encode('utf-8'))


def text_of(content) -> str:
    """OpenAI content: str, or a list of {type: text, text} parts (images refused)."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict) or part.get('type') not in ('text', 'input_text', 'output_text'):
                raise FormatError(f'non-text content part: {str(part)[:60]}')
            out.append(part.get('text', ''))
        return ''.join(out)
    raise FormatError(f'content type {type(content).__name__}')


def est_tokens(s: str) -> int:
    return int(len(s) / CHARS_PER_TOKEN + 0.999)


def cap_text(s: str, cap_tokens: int) -> tuple[str, bool, int]:
    n = est_tokens(s)
    if n <= cap_tokens:
        return s, False, n
    keep = int(cap_tokens * CHARS_PER_TOKEN) // 2
    cut = n - cap_tokens
    return s[:keep] + f'\n[... {cut} tokens truncated ...]\n' + s[-keep:], True, n


def split_turns(messages: list) -> tuple[list, list]:
    """messages (OpenAI roles) -> (system msgs, turns). A turn = the non-assistant inputs before an assistant run; the
    assistant run itself is dropped (the replay generates it). Trailing inputs after the last assistant run are dropped."""
    system, turns, cur, seen_non_system = [], [], [], False
    roles_ok = {'system', 'developer', 'user', 'assistant', 'tool', 'function', 'observation'}
    prev_assistant = False
    for m in messages:
        if not isinstance(m, dict) or 'role' not in m:
            raise FormatError('message without role')
        r = m['role']
        if r not in roles_ok:
            raise FormatError(f'role {r!r}')
        if r in ('system', 'developer') and not seen_non_system:
            system.append({'role': 'system', 'content': text_of(m.get('content'))})
            continue
        seen_non_system = True
        if r == 'assistant':
            if cur:
                turns.append(cur)
                cur = []
            prev_assistant = True
            continue
        if r in ('system', 'developer'):
            continue          # mid-session system text: dropped (gufo hoists it), counted by the caller
        role = 'tool' if r in ('tool', 'function', 'observation') else 'user'
        cur.append({'role': role, 'content': text_of(m.get('content'))})
        prev_assistant = False
    del prev_assistant
    return system, turns


def capped_session(sid: str, source: str, harness: str, system: list, turns: list, min_turns: int, max_turns: int,
                   context: int, reply_cap: int):
    """Apply caps, trim the turn count until the worst case fits; None if it cannot."""
    sysm, meta_s = [], []
    for m in system:
        c, tr, n = cap_text(m['content'], CAPS['system'])
        sysm.append({'role': 'system', 'content': c})
        meta_s.append({'kind': 'system', 'original_tokens': n, 'truncated': tr})
    out = []
    for inputs in turns:
        ins, meta = [], []
        for m in inputs:
            c, tr, n = cap_text(m['content'], CAPS['tool' if m['role'] == 'tool' else 'user'])
            ins.append({'role': m['role'], 'content': c})
            meta.append({'kind': 'tool_result' if m['role'] == 'tool' else 'user', 'original_tokens': n, 'truncated': tr})
        out.append({'inputs': ins, 'input_meta': meta})
    k = min(len(out), max_turns)
    while k >= min_turns:
        worst = sum(est_tokens(m['content']) for m in sysm)
        # earlier replies priced at HISTORY_REPLY tokens each (agent-replay v1 rule: min(reply, 256)); the last
        # request must still leave room for a full reply_cap generation
        worst += sum(est_tokens(m['content']) for t in out[:k] for m in t['inputs']) + HISTORY_REPLY * (k - 1)
        if worst <= context - reply_cap - HISTORY_REPLY:
            return {'id': sid, 'source': source, 'harness': harness, 'system': sysm, 'system_meta': meta_s,
                    'turns': out[:k], 'prompt_tokens_worst_est': worst}
        k -= 1
    return None


# ---- readers ---------------------------------------------------------------------------------------------------------
def read_mtbench(path: Path) -> list:
    rows = []
    for ln, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if not {'question_id', 'category', 'turns'} <= set(d) or not isinstance(d['turns'], list) or len(d['turns']) < 2:
            raise FormatError(f'mtbench line {ln}: want question_id / category / turns[>=2]')
        rows.append(d)
    return rows


def build_mtbench(rows: list, n: int, per_category: int) -> list:
    by = {}
    for d in sorted(rows, key=lambda d: order_key(str(d['question_id']))):
        by.setdefault(d['category'], []).append(d)
    pick = []
    for cat in sorted(by):
        pick += by[cat][:per_category]
    pick = sorted(pick, key=lambda d: order_key(str(d['question_id'])))[:n]
    return [{'id': f"mtbench-{d['question_id']}", 'source': 'mt_bench', 'harness': 'chat', 'category': d['category'],
             'system': [], 'system_meta': [],
             'turns': [{'inputs': [{'role': 'user', 'content': t}], 'input_meta': [{'kind': 'user'}]} for t in d['turns']]}
            for d in pick]


def read_tau(path: Path) -> list:
    d = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(d, list) or not d:
        raise FormatError('tau: want a non-empty list')
    for i, r in enumerate(d):
        if not isinstance(r, dict) or 'traj' not in r or not isinstance(r['traj'], list):
            raise FormatError(f'tau entry {i}: no traj list')
    return d


def read_parquet_rows(path: Path, columns: list) -> list:
    import pyarrow.parquet as pq   # local: only the parquet readers need it
    t = pq.read_table(path, columns=columns)
    return t.to_pylist()


def messages_of(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError as e:
            raise FormatError(f'messages not json: {e}') from None
    if not isinstance(v, list):
        raise FormatError('messages not a list')
    return v


def agent_sessions(items: list, source: str, harness: str, n: int, min_turns: int, max_turns: int, context: int,
                   reply_cap: int) -> list:
    """items: [(id, instance_id, messages)] -> first n sessions (deterministic order) that pass the caps."""
    out = []
    for sid, inst, msgs in sorted(items, key=lambda x: order_key(str(x[0]))):
        system, turns = split_turns(msgs)
        s = capped_session(f'{harness}-{order_key(str(sid))[:10]}', source, harness, system, turns, min_turns, max_turns,
                           context, reply_cap)
        if s is not None:
            s['instance_id'] = inst
            out.append(s)
        if len(out) == n:
            break
    if len(out) < n:
        raise FormatError(f'only {len(out)} sessions pass the caps (want {n})')
    return out


def tau_tools_from_source(tools_dir: Path) -> list:
    """Static: find `def get_info(...)` whose body is `return {...literal...}` in each .py; never import."""
    specs = []
    for f in sorted(tools_dir.glob('*.py')):
        tree = ast.parse(f.read_text(encoding='utf-8'), filename=f.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'get_info':
                for st in node.body:
                    if isinstance(st, ast.Return) and st.value is not None:
                        try:
                            specs.append(ast.literal_eval(st.value))
                        except ValueError:
                            raise FormatError(f'{f.name}: get_info() is not a literal') from None
    if not specs:
        raise FormatError('no tool schemas found')
    return specs


# ---- corpus (perm corpus, DESIGN 4.2) ---------------------------------------------------------------------------------
# One jsonl row per segment: {session_hash, agent, role, kind, text}. Kinds as the private corpus (corpus_counts.py
# reads role == 'assistant' rows as tool_call / assistant_text / reasoning, everything else as prompt):
#   assistant prose                         -> assistant / assistant_text
#   assistant tool call                     -> assistant / tool_call, rendered `<function=NAME>\n<parameter=K>\nV\n
#                                              </parameter>\n...</function>` (the private corpus' layout) from OpenAI
#                                              tool_calls; SWE-agent text blocks (`<function=...` to the end of the
#                                              message) are kept verbatim
#   (no reasoning kind: SWE-smith has no separate thinking; `thought` == content, checked)
#   system / developer                      -> system / system
#   role tool, or role user after the first assistant message (SWE-agent observations, incl. the
#   'Your command ran successfully...' / 'Observation:' variants without the 'OBSERVATION:' prefix) -> tool / tool_result
#   user before the first assistant message (the task) -> user / user_text
# session_hash = sha256('swe-smith:' + traj_id); sessions written in session_hash order; segments in message order;
# empty segments skipped; a segment is cut to MAX_SEGMENT_CHARS (the private corpus' max_segment_chars); a segment
# holding a whitespace-free run longer than MAX_RUN_CHARS is dropped (counted): the pure-Python host BPE is quadratic
# in the piece length (shard 00002 has one tool_result with a 98,460-char run of 'X': hours for one segment).
MAX_SEGMENT_CHARS = 200000
MAX_RUN_CHARS = 20000
FUNC_RE = None


def _func_re():
    global FUNC_RE
    if FUNC_RE is None:
        import re
        FUNC_RE = re.compile(r'<function=[^>\n]*>')
    return FUNC_RE


def render_tool_call(tc) -> str:
    f = tc.get('function') if isinstance(tc, dict) else None
    if not isinstance(f, dict) or not isinstance(f.get('name'), str) or not f['name']:
        raise FormatError('tool_call without function.name')
    args = f.get('arguments')
    if args is None or (isinstance(args, str) and not args.strip()):
        args = {}
    elif isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            raise FormatError('tool_call arguments not json') from None
    if not isinstance(args, dict):
        raise FormatError('tool_call arguments not an object')
    lines = [f"<function={f['name']}>"]
    for k, v in args.items():
        lines.append(f'<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>')
    return '\n'.join(lines + ['</function>'])


def longest_run(t: str) -> int:
    best = cur = 0
    for ch in t:
        cur = 0 if ch.isspace() else cur + 1
        best = cur if cur > best else best
    return best


def corpus_segments(msgs: list) -> list:
    """OpenAI-role messages (SWE-agent text blocks or OpenAI tool_calls) -> [(agent, role, kind, text)]."""
    out, after_assistant = [], False
    for m in msgs:
        if not isinstance(m, dict) or 'role' not in m:
            raise FormatError('message without role')
        r, agent = m['role'], str(m.get('agent') or 'main')
        txt = text_of(m.get('content'))
        if r in ('system', 'developer'):
            segs = [('system', 'system', txt)]
        elif r in ('tool', 'function', 'observation'):
            segs = [('tool', 'tool_result', txt)]
        elif r == 'user':
            segs = [('tool', 'tool_result', txt) if after_assistant else ('user', 'user_text', txt)]
        elif r == 'assistant':
            after_assistant = True
            mt = _func_re().search(txt)
            segs = [('assistant', 'assistant_text', txt[:mt.start()] if mt else txt)]
            if mt:
                segs.append(('assistant', 'tool_call', txt[mt.start():]))
            tcs = m.get('tool_calls') or []
            if not isinstance(tcs, list):
                raise FormatError('tool_calls not a list')
            segs += [('assistant', 'tool_call', render_tool_call(tc)) for tc in tcs]
        else:
            raise FormatError(f'role {r!r}')
        out += [(agent, role, kind, t[:MAX_SEGMENT_CHARS]) for role, kind, t in segs if t.strip()]
    return out


def exclude_ids(sessions_json: Path) -> set:
    d = json.loads(sessions_json.read_text(encoding='utf-8'))
    ss = d.get('sessions') if isinstance(d, dict) else None
    if not isinstance(ss, list) or not ss or any(not s.get('instance_id') for s in ss):
        raise FormatError(f'{sessions_json.name}: want sessions[] that all carry instance_id')
    return {str(s['instance_id']) for s in ss}


def write_corpus(items: list, out, exclude: set) -> dict:
    """items: [(traj_id, instance_id, messages)]; out: a binary file object. Returns the stats (deterministic)."""
    h = hashlib.sha256()
    st = {'trajectories_in': len(items), 'trajectories_excluded': 0, 'sessions': 0, 'segments': 0, 'bytes': 0,
          'segments_cut': 0, 'segments_dropped_long_run': [], 'by_kind': {}, 'formats': {'openai_tool_calls': 0, 'swe_agent_text_blocks': 0},
          'excluded_instances_matched': sorted({str(i) for _, i, _ in items} & exclude)}
    keyed = sorted(((sha256b(f'swe-smith:{tid}'.encode('utf-8')), tid, inst, msgs) for tid, inst, msgs in items))
    if len({k for k, *_ in keyed}) != len(keyed):
        raise FormatError('duplicate traj_id')
    for sh, _, inst, msgs in keyed:
        if str(inst) in exclude:
            st['trajectories_excluded'] += 1
            continue
        segs = corpus_segments(msgs)
        st['formats']['openai_tool_calls' if any(m.get('tool_calls') for m in msgs) else 'swe_agent_text_blocks'] += 1
        st['sessions'] += 1
        for agent, role, kind, text in segs:
            if len(text) > MAX_RUN_CHARS and longest_run(text) > MAX_RUN_CHARS:
                st['segments_dropped_long_run'].append({'session_hash': sh, 'role': role, 'kind': kind,
                                                        'chars': len(text), 'run': longest_run(text)})
                continue
            b = (json.dumps({'session_hash': sh, 'agent': agent, 'role': role, 'kind': kind, 'text': text},
                            ensure_ascii=False) + '\n').encode('utf-8')
            out.write(b)
            h.update(b)
            st['bytes'] += len(b)
            st['segments'] += 1
            st['segments_cut'] += len(text) == MAX_SEGMENT_CHARS
            bk = st['by_kind'].setdefault(f'{role}/{kind}', [0, 0])
            bk[0] += 1
            bk[1] += len(text)
    st['sha256'] = h.hexdigest()
    return st


def doc(sessions: list, params: dict, tools: dict) -> dict:
    return {'schema': SCHEMA, 'params': params,
            'rendering': {'server_defaults': '--think on --reasoning-effort xhigh', 'token_counts': 'estimate (stub)'},
            'harness_tools': tools, 'sessions': sessions}


def dump(d: dict) -> bytes:
    return (json.dumps(d, ensure_ascii=False, indent=1) + '\n').encode('utf-8')


def write_out(out: Path, d: dict, inputs: list) -> None:
    b = dump(d)
    out.write_bytes(b)
    man = {'schema': SCHEMA + '.manifest', 'sessions_file': out.name, 'sha256': sha256b(b),
           'inputs': [{'file': Path(p).name, 'sha256': sha256b(Path(p).read_bytes())} for p in inputs],
           'params': d['params'], 'selected': [s['id'] for s in d['sessions']]}
    Path(str(out) + '.manifest.json').write_text(json.dumps(man, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'sessions': len(d['sessions']), 'requests': sum(len(s['turns']) for s in d['sessions']),
                      'sha256': man['sha256']}))


# ---- selftest (synthetic fixtures only) -------------------------------------------------------------------------------
def selftest() -> int:
    ok, fails = 0, []

    def t(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fails.append(name)

    def raises(f):
        try:
            f()
        except FormatError:
            return True
        return False

    # mtbench
    rows = [{'question_id': i, 'category': c, 'turns': [f'q{i} a', f'q{i} b']}
            for i, c in enumerate(['writing', 'math', 'coding', 'reasoning'] * 6, 81)]
    s1 = build_mtbench(rows, 8, 2)
    s2 = build_mtbench(list(reversed(rows)), 8, 2)
    t('mtbench deterministic', [s['id'] for s in s1] == [s['id'] for s in s2])
    t('mtbench per-category', all(sum(1 for s in s1 if s['category'] == c) == 2 for c in ('writing', 'math')))
    t('mtbench 2 turns', all(len(s['turns']) == 2 for s in s1))
    t('mtbench bad line refused', raises(lambda: read_mtbench_text('{"question_id": 1}')))
    # split_turns
    msgs = [{'role': 'system', 'content': 'policy'}, {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c1'}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'result 1'},
            {'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'more'}]},
            {'role': 'assistant', 'content': 'done'}, {'role': 'user', 'content': 'trailing'}]
    sy, tu = split_turns(msgs)
    t('system hoisted', sy == [{'role': 'system', 'content': 'policy'}])
    t('turns = inputs before each assistant run', [[m['content'] for m in x] for x in tu] == [['hi'], ['result 1'], ['more']])
    t('tool role kept', tu[1][0]['role'] == 'tool')
    t('assistant text dropped', all(m['content'] not in ('ok', 'done') for x in tu for m in x))
    t('bad role refused', raises(lambda: split_turns([{'role': 'robot', 'content': 'x'}])))
    t('image part refused', raises(lambda: text_of([{'type': 'image_url'}])))
    # caps + selection
    big = 'x' * 40000
    items = [(f't{i}', f'inst{i}', [{'role': 'system', 'content': 'sys'}] +
              sum(([{'role': 'user' if j == 0 else 'tool', 'content': big if j == 3 else f'out {j}'},
                    {'role': 'assistant', 'content': 'a'}] for j in range(10)), [])) for i in range(20)]
    ss = agent_sessions(items, 'fixture', 'swe-agent', 12, 6, 20, 32768, 4096)
    t('12 sessions', len(ss) == 12)
    t('tool cap applied', any(m['truncated'] for s in ss for tt in s['turns'] for m in tt['input_meta']))
    t('worst fits', all(s['prompt_tokens_worst_est'] <= 32768 - 4096 - HISTORY_REPLY for s in ss))
    t('selection deterministic', [s['id'] for s in ss] ==
      [s['id'] for s in agent_sessions(list(reversed(items)), 'fixture', 'swe-agent', 12, 6, 20, 32768, 4096)])
    t('too few refused', raises(lambda: agent_sessions(items[:3], 'f', 'h', 12, 6, 20, 32768, 4096)))
    t('messages json string ok', messages_of('[{"role": "user", "content": "x"}]')[0]['role'] == 'user')
    t('messages non-list refused', raises(lambda: messages_of('{"a": 1}')))
    # tau tools by static parsing (no import)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        Path(td, 'get_order.py').write_text(
            'import os\nclass GetOrder:\n    @staticmethod\n    def invoke(data, order_id):\n        os.system("never")\n'
            '    @staticmethod\n    def get_info():\n        return {"type": "function", "function": {"name": "get_order",'
            ' "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}}\n', encoding='utf-8')
        sp = tau_tools_from_source(Path(td))
        t('tau tool parsed statically', sp and sp[0]['function']['name'] == 'get_order')
        Path(td, 'bad.py').write_text('def get_info():\n    return make()\n', encoding='utf-8')
        t('non-literal refused', raises(lambda: tau_tools_from_source(Path(td))))
    # output doc + byte determinism
    d1 = dump(doc(ss, {'n': 12}, {'swe-agent': []}))
    d2 = dump(doc(ss, {'n': 12}, {'swe-agent': []}))
    t('dump deterministic', d1 == d2 and json.loads(d1)['schema'] == SCHEMA)
    # corpus: SWE-agent text blocks + OpenAI tool_calls, exclusion, byte determinism
    txt_fmt = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'task'},
               {'role': 'assistant', 'content': 'look\n\n<function=bash>\n<parameter=command>ls</parameter>\n</function>'},
               {'role': 'user', 'content': 'OBSERVATION:\na.py'}, {'role': 'assistant', 'content': 'done'},
               {'role': 'user', 'content': 'Your command ran successfully and did not produce any output.'}]
    oa_fmt = [{'role': 'system', 'content': 'sys', 'agent': 'main'},
              {'role': 'user', 'content': [{'type': 'text', 'text': 'task'}], 'message_type': 'observation'},
              {'role': 'assistant', 'content': 'view it', 'thought': 'view it', 'tool_calls': [
                  {'id': 'c1', 'type': 'function', 'function': {'name': 'str_replace_editor',
                                                                'arguments': '{"command": "view", "path": "/t", "view_range": [1, 5]}'}}]},
              {'role': 'tool', 'content': [{'type': 'text', 'text': 'OBSERVATION:\nfile'}], 'tool_call_ids': ['c1']},
              {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'submit', 'arguments': '{}'}}]}]
    sg = corpus_segments(txt_fmt)
    t('corpus text kinds', [(r, k) for _, r, k, _ in sg] == [('system', 'system'), ('user', 'user_text'),
      ('assistant', 'assistant_text'), ('assistant', 'tool_call'), ('tool', 'tool_result'), ('assistant', 'assistant_text'),
      ('tool', 'tool_result')])
    t('corpus text block verbatim', sg[3][3] == '<function=bash>\n<parameter=command>ls</parameter>\n</function>')
    sg = corpus_segments(oa_fmt)
    t('corpus openai kinds', [(r, k) for _, r, k, _ in sg] == [('system', 'system'), ('user', 'user_text'),
      ('assistant', 'assistant_text'), ('assistant', 'tool_call'), ('tool', 'tool_result'), ('assistant', 'tool_call')])
    t('corpus openai render', sg[3][3] == '<function=str_replace_editor>\n<parameter=command>\nview\n</parameter>\n'
      '<parameter=path>\n/t\n</parameter>\n<parameter=view_range>\n[1, 5]\n</parameter>\n</function>'
      and sg[5][3] == '<function=submit>\n</function>')
    t('corpus bad args refused', raises(lambda: render_tool_call({'function': {'name': 'x', 'arguments': '[1]'}})))
    import io
    its = [('tA', 'i1', txt_fmt), ('tB', 'i2', oa_fmt), ('tC', 'i3', txt_fmt)]
    b1, b2 = io.BytesIO(), io.BytesIO()
    s1c = write_corpus(its, b1, {'i3'})
    s2c = write_corpus(list(reversed(its)), b2, {'i3'})
    t('corpus deterministic', b1.getvalue() == b2.getvalue() and s1c['sha256'] == sha256b(b1.getvalue()))
    b3 = io.BytesIO()
    s3c = write_corpus([('tD', 'i4', [{'role': 'user', 'content': 'task'},
                                      {'role': 'assistant', 'content': 'x'}, {'role': 'tool', 'content': 'a ' + 'X' * (MAX_RUN_CHARS + 1)}])],
                       b3, set())
    t('corpus long run dropped', s3c['segments'] == 2 and len(s3c['segments_dropped_long_run']) == 1)
    t('corpus exclusion', s1c['sessions'] == 2 and s1c['trajectories_excluded'] == 1
      and s1c['excluded_instances_matched'] == ['i3'])
    rows_c = [json.loads(x) for x in b1.getvalue().decode('utf-8').splitlines()]
    t('corpus row schema', all(set(r) == {'session_hash', 'agent', 'role', 'kind', 'text'} for r in rows_c)
      and s1c['formats'] == {'openai_tool_calls': 1, 'swe_agent_text_blocks': 1})
    print(f'build_sessions_public selftest: {ok} PASS, {len(fails)} FAIL' + (f' {fails}' if fails else ''))
    return 0 if not fails else 1


def read_mtbench_text(text: str) -> list:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td, 'q.jsonl')
        p.write_text(text, encoding='utf-8')
        return read_mtbench(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('cmd', nargs='?', choices=['mtbench', 'tau', 'sweagent', 'openhands', 'corpus', 'tau-tools'])
    ap.add_argument('--in', dest='inp')
    ap.add_argument('--out')
    ap.add_argument('--tools')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--per-category', type=int, default=3)
    ap.add_argument('--min-turns', type=int, default=6)
    ap.add_argument('--max-turns', type=int, default=20)
    ap.add_argument('--context', type=int, default=32768)
    ap.add_argument('--reply-cap', type=int, default=4096)
    # quick-look (2026-11-08): conservative token estimate + full-reply history pricing, so the worst case of a
    # session (every earlier reply at the reply cap) still fits the context; recorded in params
    ap.add_argument('--chars-per-token', type=float, default=CHARS_PER_TOKEN)
    ap.add_argument('--history-reply', type=int, default=HISTORY_REPLY)
    # corpus (perm corpus, DESIGN 4.2)
    ap.add_argument('--reader', default='sweagent')
    ap.add_argument('--exclude-instances', help='a sessions json: every instance_id in it is left out of the corpus')
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    globals()['CHARS_PER_TOKEN'] = a.chars_per_token
    globals()['HISTORY_REPLY'] = a.history_reply
    sizing = {'max_tokens': a.reply_cap, 'context': a.context, 'chars_per_token': a.chars_per_token,
              'history_reply': a.history_reply, 'min_turns': a.min_turns, 'max_turns': a.max_turns}
    if not a.cmd or not a.inp or not a.out:
        ap.print_help()
        return 2
    src, out = Path(a.inp), Path(a.out)
    try:
        if a.cmd == 'mtbench':
            ss = build_mtbench(read_mtbench(src), a.n if a.n != 12 else 24, a.per_category)
            write_out(out, doc(ss, {'source': 'mt_bench', 'n': len(ss), 'reply_cap': a.reply_cap, **sizing}, {'chat': []}),
                      [src])
        elif a.cmd == 'tau':
            tools = json.loads(Path(a.tools).read_text(encoding='utf-8')) if a.tools else []
            items = [(str(r.get('task_id', i)), str(r.get('task_id', i)), r['traj']) for i, r in enumerate(read_tau(src))]
            ss = agent_sessions(items, 'tau-bench', 'tau', a.n, 3, a.max_turns, a.context, a.reply_cap)
            write_out(out, doc(ss, {'source': 'tau-bench', 'n': a.n, 'reply_cap': a.reply_cap}, {'tau': tools}),
                      [src] + ([Path(a.tools)] if a.tools else []))
        elif a.cmd == 'sweagent':
            rows = read_parquet_rows(src, ['traj_id', 'instance_id', 'messages'])
            items = [(r['traj_id'], r['instance_id'], messages_of(r['messages'])) for r in rows]
            tools = json.loads(Path(a.tools).read_text(encoding='utf-8')) if a.tools else []
            ss = agent_sessions(items, 'swe-smith', 'swe-agent', a.n, a.min_turns, a.max_turns, a.context, a.reply_cap)
            write_out(out, doc(ss, {'source': 'swe-smith', 'n': a.n, 'reply_cap': a.reply_cap, **sizing,
                                    'tool_format': 'swe-agent text function blocks in the system prompt; observations '
                                                   'as user turns; no OpenAI tools declared' if not tools else 'openai'},
                               {'swe-agent': tools}),
                      [src] + ([Path(a.tools)] if a.tools else []))
        elif a.cmd == 'corpus':
            if a.reader != 'sweagent':
                print(f'corpus reader {a.reader!r}: only sweagent is implemented')
                return 2
            excl = exclude_ids(Path(a.exclude_instances)) if a.exclude_instances else set()
            rows = read_parquet_rows(src, ['traj_id', 'instance_id', 'messages'])
            items = [(r['traj_id'], r['instance_id'], messages_of(r['messages'])) for r in rows]
            del rows
            tmp = Path(str(out) + '.partial')
            with open(tmp, 'wb') as f:
                stc = write_corpus(items, f, excl)
            if sha256b(tmp.read_bytes()) != stc['sha256']:
                raise FormatError('corpus write: sha mismatch')
            tmp.replace(out)
            inputs = [src] + ([Path(a.exclude_instances)] if a.exclude_instances else [])
            man = {'schema': CORPUS_SCHEMA + '.manifest', 'corpus_file': out.name, 'sha256': stc.pop('sha256'),
                   'inputs': [{'file': Path(p).name, 'sha256': sha256b(Path(p).read_bytes())} for p in inputs],
                   'reader': 'sweagent', 'source': 'SWE-smith-trajectories (tool split)',
                   'exclude_instances_requested': len(excl), 'max_segment_chars': MAX_SEGMENT_CHARS, 'max_run_chars': MAX_RUN_CHARS,
                   'session_hash_rule': "sha256('swe-smith:' + traj_id)",
                   'kind_map': 'assistant prose -> assistant_text; assistant tool call -> tool_call (OpenAI tool_calls '
                               'rendered <function=NAME>/<parameter=K> one per line, SWE-agent text blocks verbatim); '
                               'system -> system; role tool, or role user after the first assistant message -> tool_result; '
                               'user before it -> user_text; no reasoning kind', **stc}
            Path(str(out) + '.manifest.json').write_text(json.dumps(man, indent=1) + '\n', encoding='utf-8')
            print(json.dumps({k: man[k] for k in ('sha256', 'sessions', 'segments', 'trajectories_in',
                                                  'trajectories_excluded', 'exclude_instances_requested')}))
        elif a.cmd == 'tau-tools':
            Path(a.out).write_text(json.dumps(tau_tools_from_source(src), indent=1) + '\n', encoding='utf-8')
        else:
            print(f'{a.cmd}: not implemented in the stub (S1 work item)')
            return 2
    except FormatError as e:
        print(f'FORMAT REFUSED: {e}')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
