---
layout: page
title: What's in production
permalink: /production/
---



## The release build

**Benchmarked in run 287: build `fe7b7048` (R7R) = upstream gufo v0.34.0
(`2845939b`) plus our changes, built with TheRock 10.0.0.** Every change sits behind a runtime switch;
with no switch set the build behaves as upstream. The changes are published as patches in the repository's
[patches/](https://github.com/dvellon/gufo-347/tree/main/patches), and the benchmarked launch settings for each task as ready-to-run files in
[profiles/](https://github.com/dvellon/gufo-347/tree/main/profiles).

Both our build and stock are built with the same toolchain, TheRock 10.0.0, upstream's documented Windows recipe, so the ratios compare code, not toolchains.



### Launch files (the benchmarked settings for each task)

| launch file | use this for | sessions | context | exactness |
|---|---|---:|---:|---|
| [`profiles/agent-1-session.ps1`](https://github.com/dvellon/gufo-347/blob/main/profiles/agent-1-session.ps1) | one coding agent at a time (tool-calling agent loops); output = the model's own greedy output | 8 | 32768 | strict |
| [`profiles/agent-8-sessions.ps1`](https://github.com/dvellon/gufo-347/blob/main/profiles/agent-8-sessions.ps1) | up to 8 coding agents in parallel against one server; output = greedy | 8 | 32768 | strict |
| [`profiles/chat.ps1`](https://github.com/dvellon/gufo-347/blob/main/profiles/chat.ps1) | interactive multi-turn chat; output = greedy | 1 | 32768 | strict |
| [`profiles/canonical-bench.ps1`](https://github.com/dvellon/gufo-347/blob/main/profiles/canonical-bench.ps1) | reproducing the canonical prose row (upstream's model-bench recipe, thinking off) | 1 | 32768 | strict |
| [`profiles/relaxed-agent-1-session.ps1`](https://github.com/dvellon/gufo-347/blob/main/profiles/relaxed-agent-1-session.ps1) | one coding agent at a time when you accept near-tie differences from greedy for speed; the relaxed cell timed in the benchmark | 8 | 32768 | relaxed |
| [`profiles/relaxed-agent-8-sessions.ps1`](https://github.com/dvellon/gufo-347/blob/main/profiles/relaxed-agent-8-sessions.ps1) | up to 8 coding agents in parallel with near-tie acceptance; the relaxed cell timed in the benchmark | 8 | 32768 | relaxed |
| [`profiles/relaxed-chat.ps1`](https://github.com/dvellon/gufo-347/blob/main/profiles/relaxed-chat.ps1) | interactive multi-turn chat with near-tie acceptance; the relaxed cell timed in the benchmark | 1 | 32768 | relaxed |

Each file sets the switches below and starts `gufo.exe serve` with the public draft head and the public
draft-vocabulary table. Strict files give the model's own greedy output; relaxed files are non-standard (near-tie
acceptance) and are labelled. The agent and chat files drop the benchmarked cell's `--max-tokens 128`, the server's
default reply cap for clients that do not send `max_tokens` (the relaxed agent files also add
`--preserve-thinking auto`); the benchmark clients sent `max_tokens` 2048, so the numbers are unaffected (an internal
like-for-like check, run 288, relaxed agent at 1 session, no published receipts: 1.003 [0.998, 1.009]).

### Switches set by the strict profiles

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

- The exact LM-head screen (A1) scales its bound with a per-channel table. The benchmarked build and the released patches both use a flat, data-free calibration table (every channel 1.0). The screen is exact for any table; the table only changes how many candidate rows are re-read.
- All draft-side changes are **strict-safe by construction**: they change what is proposed or how fast it is checked,
  never the verified token. Run 287's exactness tables are on the [Results]({{ site.baseurl }}/results/#exactness)
  page.

## Before the rebase: our previous build (still in production)

Our production server still runs the pre-rebase build, a fork of an older upstream base: package `c5`, `gufo.exe` `edb3fe51`, adopted
server profiles `a1f1493d`. It is the "previous build" reference row in run 287. The history below is how it was
built up; each change was ported to the rebased build behind the same switch.

### Promotion history of the previous build

| # | when (ET) | package | binary | what changed | evidence (gate) |
|---|---|---|---|---|---|
| 1 | 2026-10-30 20:18 | **promo-3cg** | `38ff7eeb` -> `135c823d` | first gated composite: the strict-tier exactness fixes, the relaxed drafting stack (H1 small-head drafting + B5 margin stop) as the relaxed default, and a fixed prefill grid that makes relaxed output judgeable | promotion gates incl. a power re-run of the agent stage |
| 2 | 2026-11-01 | **consolidated build** (`w3cons`) | -> `9a9fe3a6` | every GO item folded into one build, gated once: the KTc draft controller + tool-call stop, H1/B5 admitted to the strict tier, a small-batch `hc_down` kernel, an out-of-vocabulary static draft list, the B5 floor raised to 1.5, and a one-session limit for H1 drafting in the strict-agent profile | members: KTc +11.5 % agent C1 strict; `hc_down` +9.2 % [8.7, 9.6]; OOV list +0.97 % [0.65, 1.28] (later corrected to +0.65 %); gate 178 PASS, then the 179 re-profile |
| 3 | 2026-11-06 19:34 | **r3** | -> `f0dcd4d4` | fix for a rare `device_lost` fault in the prompt-cache copy path (cache segments split per host registration), on in every profile | no-harm gate: relaxed agent C8 +0.47 % [-0.63, 2.21]; exactness PASS; promoted on a REVIEW-HOST verdict by owner decision |
| 4 | 2026-11-06 20:08 | **strict chat, 1 session** | unchanged | a new strict-chat profile: chat served with `--sessions 1` so the drafter plans as for a single stream | strict chat decode +7.39 % (paging-adjusted +7.28 % [5.15, 9.20]); exactness PASS |
| 5 | 2026-11-07 ~15:15 | **combo-c1** | -> `56052e5e` (profiles `b225cfe2`) | three draft-side changes, each behind its own knob: a frequency-permuted draft vocabulary, a 2-bit screen of the small draft head's top-2 candidates with exact rescoring, and a smaller-weight draft head (routed experts at Q4_K gate/up + Q5_1 down) | agent C1 strict +2.74 % [1.62, 3.84] (paging-adjusted +2.03 % [0.90, 3.12]), lineage exactness PASS; relaxed agent C8 +3.33 % [1.63, 5.45], no harm; owner accepted a REVIEW-HOST flag and 1.5 % A/A drift |
| 6 | 2026-11-08 ~12:50 | **c3** | -> `4d787f38` (profiles `0ea5e314`) | adds `option_logprobs` (score a fixed set of answer options from the prompt pass, no generation), and gives the strict-chat profile the strict-agent draft settings | normal requests byte-identical to `56052e5e`; option scores equal the decode-path values; strict-chat greedy canonical +2.29 % [2.05, 2.48] exact; sampled chat +0.99 % [-0.72, 2.57] (not significant) |
| 7 | 2026-11-10 ~15:00 | **c4** + relaxed-mixed settings | -> `8d71f7a3` (profiles `61c0bd95`, then `71eee716`) | a prompt-cache checkpoint at the start of a short final user turn (fast re-scoring on the same context); the round-1 settings for the relaxed-mixed profile | tail latency p50 1,256 -> 187 ms at 8K, exact; relaxed-mixed agent C1 +4.6 %, C8 +1.5 % (REVIEW-HOST, owner decision) |
| 8 | 2026-11-10 ~20:35 | **c5** | -> `edb3fe51` (profiles `ccae2401`) | deferred state rollback after partial acceptance, and a draft-width rule on the strict-agent profile | combined +1.85 % [1.19, 2.52], exact (members REVIEW-HOST, owner decision) |
| 9 | 2026-11-12 ~00:25 | **settings: B5 hide + floor** | unchanged (`edb3fe51`; profiles `ccae2401` -> `a1f1493d`) | two draft-length settings, no binary change: the strict-agent profile stops proposing a draft whose top-2 margin is below 1.5 (`GUFO_B5_HIDE` inf -> 1.5), and the relaxed-mixed profile raises its margin floor to 2 (`GUFO_B5_FLOOR`) | strict-agent: canonical +1.01 % [0.73, 1.24], agent C1 -0.22 %, agent C8 +0.09 % (no harm), canonical exact and agent C1 lineage PASS; relaxed-mixed: canonical +1.64 % [1.53, 1.82], reproducible; start-up smoke on all four profiles (strict replies identical to the c5 reference) |

Notes:

- All draft-side changes are **strict-safe by construction**: they change what is proposed or how fast it is checked,
  never the verified token. The exactness gate confirms that on every package.
- Gains are per gate stage and **do not add up**: later changes act on a faster base, and several gates were later
  corrected for host paging (see [Methods]({{ site.baseurl }}/methods/#the-paging-audit)).

### What each piece does, in short

| piece | in plain words | tier |
|---|---|---|
| MTP speculative decoding | the model's own multi-token-prediction head drafts several tokens; one target pass verifies them | strict |
| fine-tuned draft head | the head was fine-tuned on the model's own outputs (on-policy): +4.2 % tokens per round over the stock head; this is the one weight change that paid (private, not published) | strict |
| KTc controller + tool stop | plans how many tokens to draft per round from measured costs, and stops drafting at tool-call boundaries | strict |
| H1 / B5 | H1 drafts tool-constrained greedy requests with a small head and a confidence stop; B5 stops a draft chain when the draft's margin is low (floor 1.5) | strict (admitted) and relaxed |
| small `hc_down` kernel | a faster kernel for a small-batch projection in the verify pass | strict |
| OOV static list, permuted vocab | the draft head reads a 49,152-row prefix of the vocabulary; these tables put the tokens it most often needs into that prefix | strict |
| 2-bit top-2 screen | a 2-bit copy of the small head ranks candidates first; the exact head rescoring keeps the result identical | strict |
| Q4_K/Q5_1 draft experts | smaller draft-head weights: fewer bytes per draft step, acceptance unchanged in offline replays | strict |
| fixed prefill grid (`scope=all`) | prompts are cut at fixed grid points instead of arrival-timed cuts, which makes relaxed output reproducible and saves prefill passes | relaxed |
| cache segment split (r3) | removes a rare joined-span copy across two host memory registrations that could fault the device | all |
| `option_logprobs` | a scoring endpoint feature for multiple-choice decisions without generating text | all |

### Server profiles of the previous build

Four launch specifications are adopted together (the "ADOPTED" profile set, hash-pinned with the binary):

| profile | use | key settings |
|---|---|---|
| **strict-agent** | coding agents, replay-exact | `--exactness strict`; 8 sessions x 32K context; KTc + tool stop; H1 drafts only while one session decodes; B5 floor 1.5; draft-side tables and screen on |
| **strict-chat** | people chatting, replay-exact | strict-agent's draft settings, `--sessions 1` |
| **relaxed-agent** | agent-only box, daily use | `--exactness relaxed`; fixed prefill grid, chunk 4096; 8 x 32K; temperature 0 |
| **relaxed-mixed** | agents and chat on one box | relaxed; grid chunk 2048 (a 4096-row prefill pass stalls other streams ~1 s longer: gap p90 1.6 s -> 2.7 s); 4 x 64K |

Shared: `--cache-ram-bytes 9e9` (prompt-cache RAM pin), thinking on with effort `xhigh`, MTP speculation on.
Relaxed requests can still ask for `strict` per request; a conversation keeps one level.

A small supervisor restarts the server **only** on the `device_lost` exit code (75), at most three times per rolling
hour, after the host is ready again; clients retry a `503 device_lost` once. The rate before the r3 fix was about 1
in 17,000 agent requests.

**Deliberately not used on the previous build**, with the reason:

- batched prefill across 8 sessions (`--prefill-batch 8`): KILL-SPEED, about -17 % on 8 x 32K agent traffic;
- the disk prompt-cache tier: rejected as a default because of SSD write volume;
- any approximate tier (lossy weights): none approved; the best candidate (a Q6_K LM head) had excellent quality but
  no live speed gain (KILL at C8);
- a 4096 prefill chunk on a box that serves chat (see above).


<small>Numbers filled by fill_release.py from benchmark run 287 (written 2026-11-15 23:56:05 ET).</small>
