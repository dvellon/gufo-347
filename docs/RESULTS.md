# Results: benchmark run 287



Run 287 is the final public benchmark: one build of ours (upstream v0.34.0 plus our changes behind runtime switches), measured in its strict tier and in its relaxed (non-standard, near-tie) tier in the same session, against stock v0.34.0, with our previous build as a reference row. Every ratio is paired within this run. Strict exactness is lineage-judged against the build's own no-speculation references.

Written 2026-11-15 23:56:05 ET. Windows 11, gfx1151.

## Setup

| arm | what |
|---|---|
| **A** stock | upstream gufo v0.34.0 (commit `2845939b`), built with TheRock 10.0.0 (upstream BUILD-WINDOWS.md, cmake --preset windows-release); the public shared Q8_0 MTP draft head that upstream's model guide uses (`5ff54097`) |
| **B** ours, strict, public assets | our build `fe7b7048` = upstream v0.34.0 + our patches, built with TheRock 10.0.0, with the switches listed below; the **same public head**, and a public draft-vocabulary table (K = 6,931, `caade56a`) built with the tools in this repo |
| **P** previous build (reference) | our pre-rebase production build, built with ROCm 10.1 (`edb3fe51`, profiles `a1f1493d`), same assets; shown for reference only |
| **BX** ours, relaxed, public assets | the same binary and assets, with the relaxed (non-standard, near-tie) profile |
| **C / CX** ours, private drafter | strict / relaxed with our privately trained head and vocabulary table: **numbers only** |
| **R / AR** references | our binary / stock with speculation off: the exactness references |

Every arm uses the same machine, driver, model files (Qwen3.8-Flash-Next, Unsloth UD-Q4_K_XL GGUF (revision 38bb39ee)) and client. Decoding is greedy, and thinking is on
(xhigh) for the agent and chat stages. Each launch gets a fresh server, `--context 32768` and
`--cache-ram-bytes 9e9`. Within a round, every arm runs the same session subset, and the subsets rotate between
rounds. Ratios are given as a point estimate and a 95 % paired bootstrap CI over rounds. 

| stage | workload | headline metric |
|---|---|---|
| canonical C1 | upstream's model-bench recipe: pp2048 / tg128 prose, thinking off | server tg tok/s |
| agent C1 | SWE-smith agent trajectories (tool split, public shard), all turns replayed with the server's own replies fed back, one session at a time | client decode tok/s |
| agent C8 | the same workload, 8 sessions in flight | full-window tok/s (while >= 8 requests are in flight) |
| chat C1 | MT-Bench questions, 2 turns each, thinking on | client decode tok/s |

## Headline

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| **Stock** upstream gufo v0.34.0 (tok/s) | 41.03 | 62.98 | 57.43 | 49.82 |
| **Ours, strict** (lossless; public draft head and public vocab table) | 40.41 (-1.5 % [-1.9 %, -0.8 %]) | 71.27 (+13.2 % [+6.5 %, +19.5 %]) | 61.16 (+6.5 % [+3.0 %, +10.1 %]) | 53.98 (+8.3 % [+5.5 %, +11.0 %]) |
| **Ours, relaxed** (non-standard: near-tie acceptance; public assets) | 40.58 (-1.1 % [-1.3 %, -0.8 %]) | 71.93 (+14.2 % [+8.4 %, +20.1 %]) | 62.40 (+8.7 % [+3.6 %, +14.2 %]) | 54.08 (+8.6 % [+6.3 %, +10.3 %]) |
| **Ours, strict + privately trained drafter** (not reproducible: head not published; run 289, see the note below) | 47.81 (+17.0 % [+16.5 %, +17.6 %]; vs public drafter +19.4 % [+18.3 %, +20.6 %]) | 74.88 (+17.0 % [+4.6 %, +30.7 %]; vs public drafter +3.4 % [+3.0 %, +3.6 %]) | 63.54 (+7.5 % [+4.8 %, +11.8 %]; vs public drafter +1.0 % [-1.5 %, +3.4 %]) | 56.03 (+11.6 % [+9.4 %, +15.3 %]; vs public drafter +5.1 % [+4.7 %, +5.4 %]) |
| **Ours, relaxed + privately trained drafter** (non-standard; not reproducible: head not published; run 289, see the note below) | 47.44 (+16.1 % [+16.0 %, +16.2 %]) | 76.79 (+20.0 % [+8.5 %, +34.0 %]) | 62.67 (+6.0 % [+0.8 %, +14.8 %]) | 56.16 (+11.9 % [+10.3 %, +14.5 %]) |

The private-drafter rows were measured in run 289 (2026-11-16), a separate paired session on the same build that includes its own stock arm (stock there, tok/s: canonical C1 40.85; agent C1 63.99; agent C8 59.13; chat C1 50.20). Their changes vs stock are that run's own paired ratios, and "vs public drafter" compares the private and the public draft head within that same run; no ratio is taken across runs. The private head is not published, so these rows cannot be reproduced. Run 289's signed receipts are published for its stock and public-drafter launches; its private-drafter launches have none, because a receipt records the hash of every input and those inputs are private.

Where we are not clearly ahead of stock (95 % CI not above zero, or no CI): canonical C1 strict -1.5 % [-1.9 %, -0.8 %]; canonical C1 relaxed -1.1 % [-1.3 %, -0.8 %].

Relaxed vs strict (wins / losses only where the 95 % CI excludes zero): canonical C1 +0.4 % [-0.0 %, +0.7 %]; agent C1 +0.9 % [-0.2 %, +2.2 %]; agent C8 +2.0 % [+0.3 %, +4.2 %]; chat C1 +0.2 % [-0.7 %, +0.9 %].







Against our previous build (pre-rebase, `edb3fe51`), strict: canonical C1 -3.4 % [-3.8 %, -2.8 %]; agent C1 +2.0 % [-2.6 %, +7.6 %]; agent C8 +16.5 % [+14.4 %, +18.1 %]; chat C1 -0.5 % [-2.8 %, +0.8 %].

## Switches

Build `fe7b7048` (R7R); changes against the run's base profile: +GUFO_C82, +GUFO_U882, +GUFO_U882_BCONF, +GUFO_XM_STRICT_ALIGN, -GUFO_AS_EARLY, -GUFO_B5_FLOOR, -GUFO_B5_HIDE.

| switch | canonical + agent C1 | agent C8 | chat | what it does |
|---|---|---|---|---|
| `GUFO_A1_ARM` | `actual` | `actual` | unset | where the exact LM-head screen (A1) checks a draft row: on the row actually verified (`actual`) |
| `GUFO_A1_DRAFT2` | `1` | `1` | `1` | screen the top-2 (confidence-stop) draft heads with A1 |
| `GUFO_A1_MAX_ROWS` | `4` | `4` | `4` | A1 screens head passes of at most N rows (wider passes use the normal head); unset = A1 off |
| `GUFO_A1_TOOL` | `1` | `1` | unset | tool-call frontier: take the A1-screened argmax as the draft anchor when it is provably the argmax |
| `GUFO_B5_FLOOR` | unset | unset | `1.5` | draft-length rule: stop drafting when the draft's top-2 logit margin is below this floor |
| `GUFO_B5_HIDE` | unset | unset | `inf` | draft-length rule: margin below which a draft is not proposed at all (inf = off) |
| `GUFO_C82` | `1` | `1` | `1` | 8-session batched prefill across requests inside one pass (bitwise-checked against solo prefill) |
| `GUFO_CACHE_SEG_SPLIT` | `1` | `1` | `1` | query-free fast path for segmented KV-cache copies (upstream already guarantees correctness) |
| `GUFO_CAP_TAIL` | `1` | `1` | unset | keep a cache checkpoint at the end of a request cut by max_tokens |
| `GUFO_DRAFT_PERM` | public draft-vocabulary table (K = 6,931) | public draft-vocabulary table (K = 6,931) | public draft-vocabulary table (K = 6,931) | draft-vocabulary table: the draft head scores only the K most useful token rows, in a fixed order |
| `GUFO_GDN_DEFER` | `lazy` | `lazy` | unset | lazy rollback of the linear-attention (GDN) state after a rejected draft |
| `GUFO_H1_MAX_SESSIONS` | `1` | `1` | `1` | draft-head policy applies only while at most N sessions are active |
| `GUFO_HC_DOWN_SMALL` | `1` | `1` | `1` | narrow-matrix kernel route for the hyper-connection down projection |
| `GUFO_MTP_CTL` | `c1` | `c1` | `c1` | our draft-length controller (`c1`) instead of upstream's |
| `GUFO_Q2_SCREEN` | `w1` | unset | `w1` | exact screen for the small draft head (`w1` = width 1) |
| `GUFO_TOOL_STOP` | `1` | `1` | `1` | stop drafting at tool-call boundaries |
| `GUFO_U882` | `drop` | `drop` | `drop` | mode of upstream's draft-confidence gate under our stop rules (`drop` = no early gate, drop only the last low-confidence draft) |
| `GUFO_U882_BCONF` | `1` | `1` | `1` | fix: batched draft heads compute the same confidence as the single-session path |
| `GUFO_VF_BATCH_WAIT` | `1` | `1` | `1` | batch verify passes across sessions that are ready in the same scheduler tick |
| `GUFO_XM_STRICT_ALIGN` | `1` | `1` | `1` | strict requests also use upstream's live-frontier cache alignment |



## Private drafter (numbers only)

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| Private vs public drafter, strict (same session, run 289) | +19.4 % [+18.3 %, +20.6 %] | +3.4 % [+3.0 %, +3.6 %] | +1.0 % [-1.5 %, +3.4 %] | +5.1 % [+4.7 %, +5.4 %] |

The same private-drafter configurations, with tok/s and the change vs stock, are rows of the headline table above (run 289).

## Every arm

| stage | arm | tok/s | launches | acceptance (mean) | tokens / target pass (mean) | warm TTFT p50 ms (mean) |
|---|---|---:|---:|---:|---:|---:|
| canonical C1 | stock | 41.03 | 4 | 0.564 | 2.03 | 1167 |
| canonical C1 | ours strict, public | 40.41 | 4 | 0.401 | 2.12 | 1155 |
| canonical C1 | ours relaxed, public | 40.58 | 4 | 0.407 | 2.13 | 1149 |
| canonical C1 | ours, previous build (pre-rebase; reference) | 41.83 | 4 | 0.467 | 2.11 | 1146 |
| agent C1 | stock | 62.98 | 8 | 0.808 | 3.96 | 368 |
| agent C1 | ours strict, public | 71.27 | 8 | 0.776 | 4.35 | 368 |
| agent C1 | ours relaxed, public | 71.93 | 8 | 0.776 | 4.45 | 365 |
| agent C1 | ours, previous build (pre-rebase; reference) | 69.90 | 8 | 0.764 | 4.14 | 446 |
| agent C8 | stock | 57.43 | 6 | 0.912 | 2.92 | 2320 |
| agent C8 | ours strict, public | 61.16 | 6 | 0.875 | 3.11 | 1902 |
| agent C8 | ours relaxed, public | 62.40 | 6 | 0.881 | 3.07 | 1987 |
| agent C8 | ours, previous build (pre-rebase; reference) | 52.51 | 6 | 0.832 | 3.27 | 1593 |
| chat C1 | stock | 49.82 | 4 | 0.711 | 2.55 | 172 |
| chat C1 | ours strict, public | 53.98 | 4 | 0.649 | 2.81 | 162 |
| chat C1 | ours relaxed, public | 54.08 | 4 | 0.647 | 2.82 | 161 |
| chat C1 | ours, previous build (pre-rebase; reference) | 54.27 | 4 | 0.649 | 2.84 | 164 |

Acceptance, tokens per target pass and TTFT are plain means over the launches (unpaired).

## Every ratio

| stage | ratio | point | 95 % CI | rounds | source |
|---|---|---:|---|---:|---|
| canonical C1 | B / A | 0.985 | [0.981, 0.992] | 4 | booked |
| canonical C1 | BX / A | 0.989 | [0.987, 0.992] | 4 | booked |
| canonical C1 | BX / B | 1.004 | [1.000, 1.007] | 4 | booked |
| canonical C1 | B / P | 0.966 | [0.962, 0.972] | 4 | booked |
| canonical C1 | P / A | 1.020 | [1.018, 1.020] | 4 | booked |
| agent C1 | B / A | 1.132 | [1.065, 1.195] | 8 | booked |
| agent C1 | BX / A | 1.142 | [1.084, 1.201] | 8 | booked |
| agent C1 | BX / B | 1.009 | [0.998, 1.022] | 8 | booked |
| agent C1 | B / P | 1.020 | [0.974, 1.076] | 8 | booked |
| agent C1 | P / A | 1.110 | [1.051, 1.157] | 8 | booked |
| agent C8 | B / A | 1.065 | [1.030, 1.101] | 6 | booked |
| agent C8 | BX / A | 1.087 | [1.036, 1.142] | 6 | booked |
| agent C8 | BX / B | 1.020 | [1.003, 1.042] | 6 | booked |
| agent C8 | B / P | 1.165 | [1.144, 1.181] | 6 | booked |
| agent C8 | P / A | 0.914 | [0.875, 0.958] | 6 | booked |
| chat C1 | B / A | 1.083 | [1.055, 1.110] | 4 | booked |
| chat C1 | BX / A | 1.086 | [1.063, 1.103] | 4 | booked |
| chat C1 | BX / B | 1.002 | [0.993, 1.009] | 4 | booked |
| chat C1 | B / P | 0.995 | [0.972, 1.008] | 4 | booked |
| chat C1 | P / A | 1.089 | [1.077, 1.105] | 4 | booked |

## Relaxed tier: quality

| stage | relaxed outputs identical to no-speculation | strict outputs identical (for scale) | median first difference (est. tokens into the reply) | task success |
|---|---:|---:|---:|---|
| canonical C1 | 80.0 % (16 / 20) | 100.0 % (20 / 20) | n/a | none (synthetic prose, fixed 128 tokens) |
| agent C1 | 73.5 % (136 / 185) | 100.0 % (185 / 185) | 99 | none (replay of recorded SWE-smith turns: observations are fixed, no test run) |
| agent C8 | 81.2 % (234 / 288) | 94.1 % (271 / 288) | 27 | none (replay of recorded SWE-smith turns: observations are fixed, no test run) |
| chat C1 | 89.6 % (43 / 48) | 89.6 % (43 / 48) | 61 | none (MT-Bench needs an LLM judge; not run) |

## Exactness

| check (lineage judge, speculation flag explicit) | paired | compared | differing on the same cache path | unattributed breaks | excluded by cause | verdict |
|---|---:|---:|---:|---:|---|---|
| E1 c1 ours vs ours' no-speculation reference (own references, lineage judge) | 185 | 185 | 0 | 0 | K1 0, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 c1 previous build vs R (268 R, cross-check) | 185 | 183 | 0 | 0 | K1 2, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 chat ours vs ours' no-speculation reference (own references, lineage judge) | 48 | 43 | 0 | 0 | K1 5, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 chat previous build vs R (268 R, cross-check) | 48 | 42 | 0 | 0 | K1 6, K2 0, K3 0, UNATTRIBUTED 0 | PASS |

Our build against its own no-speculation references (lineage judge, speculation flag explicit): agent C1: 185 of 185 requests on the same cache path as the reference, 0 differ (PASS); chat: 43 of 48 requests on the same cache path as the reference, 0 differ (PASS); canonical prompts: 20 of 20 completions identical (PASS). Requests whose cache path differs from the reference are excluded by cause in the table above.

Plain identity checks (session lineage over content hashes; canonical rows compare the completion text):

| check (plain identity) | identical | differ | not compared | verdict |
|---|---:|---:|---:|---|
| E1 c1 ours vs ours' no-speculation reference (own references) | 185 | 0 | 0 | PASS |
| E1 c1 previous build vs R (268 R) | 183 | 1 | 1 | FAIL-EXACT |
| E1 chat ours vs ours' no-speculation reference (own references) | 43 | 5 | 0 | FAIL-EXACT |
| E1 chat previous build vs R (268 R) | 42 | 6 | 0 | FAIL-EXACT |
| E1 canon ours vs canon-ours' no-speculation reference (completion sha per prompt) | 20 | 0 | 0 | PASS |
| E1 canon previous build vs canon-R (268 R, completion sha per prompt) | 20 | 0 | 0 | PASS |
| canon ours' no-speculation reference vs R(268) (no speculation, completion sha per prompt; report only) | 2 | 3 | 0 | DIFFERS (report only) |
| canon ours' no-speculation reference vs canon-RR7 (run 272; no speculation, completion sha per prompt; report only) | 2 | 3 | 0 | DIFFERS (report only) |

"C8 ... (report only)" rows compare 8-session outputs with single-session references. Batching legitimately changes
which cache boundaries exist, so these rows are informational. "A/A" rows run the same subset twice. See
[EXACTNESS.md](EXACTNESS.md).

## Limitations

- Windows 11 only. Upstream treats Windows as a development target, so Linux numbers may differ.
- Both our build and stock are built with the same toolchain, TheRock 10.0.0, upstream's documented Windows recipe, so the ratios compare code, not toolchains.
- Every arm uses the previous Unsloth revision (38bb39ee); upstream now recommends a newer one.
- The coding workload uses SWE-agent text function calling. Observations are replayed as recorded, so these are
  performance workloads, not accuracy benchmarks.
- Each launch is labelled QUIET or LOUD by a host-load monitor with pre-registered thresholds (background CPU or GPU use, or memory paging). LOUD launches are kept and reported, not re-run. Run 287: 21 of 88 timed launches LOUD (agent C1 3/32, agent C8 7/24, canonical C1 1/16, chat C1 10/16).

---
Numbers filled by fill_release.py from benchmark run 287 (written 2026-11-15 23:56:05 ET).
