# gufo-347: gufo on Strix Halo, 347 experiments later



The number in the name is the number of experiments we actually ran for this first release: GPU experiment runs plus
offline CPU simulations. Work that was designed but never run, engineering-only fixes and pure research notes are not
counted.

This repository holds the results, methods and experiment log of a performance study of
[gufo](https://github.com/gufo-org/gufo), an open-source local inference engine, on one AMD Strix Halo machine. It
also holds the tooling to **train your own MTP draft head** on your own traffic.

- **Model:** Qwen3.8-Flash-Next, Unsloth UD-Q4_K_XL GGUF (revision 38bb39ee), with its MTP (multi-token prediction) draft head for speculative decoding.
- **Machine:** AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GiB unified memory, Windows 11 Pro.
- **Engine:** our gufo fork, rebased on upstream v0.34.0 (2845939b); every change sits behind a runtime switch (off = upstream behaviour). Each change was kept only after a pre-registered A/B test passed; the switches
  used in the benchmark are listed [below](#switches-used-in-the-benchmark).
- **Goal:** more throughput for a machine that serves coding agents around the clock, **without changing the model's
  output**.

This release contains results, methods, the experiment log, the drafter-training tooling, and our engine changes as patches against upstream gufo v0.34.0 (patches/), each behind a runtime switch. It does not contain binaries, model files, or our private drafter.

## Headline results

Benchmark run 287, 2026-11-15. Same machine, model files, workloads and client for every arm. Each cell
shows tok/s, then the change vs stock with its 95 % CI. Ratios are paired bootstrap over rotated rounds, and each
launch uses a fresh server. 

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| **Stock** upstream gufo v0.34.0 (tok/s) | 41.03 | 62.98 | 57.43 | 49.82 |
| **Ours, strict** (lossless; public draft head and public vocab table) | 40.41 (-1.5 % [-1.9 %, -0.8 %]) | 71.27 (+13.2 % [+6.5 %, +19.5 %]) | 61.16 (+6.5 % [+3.0 %, +10.1 %]) | 53.98 (+8.3 % [+5.5 %, +11.0 %]) |
| **Ours, relaxed** (non-standard: near-tie acceptance; public assets) | 40.58 (-1.1 % [-1.3 %, -0.8 %]) | 71.93 (+14.2 % [+8.4 %, +20.1 %]) | 62.40 (+8.7 % [+3.6 %, +14.2 %]) | 54.08 (+8.6 % [+6.3 %, +10.3 %]) |
| **Ours, strict + privately trained drafter** (not reproducible: head not published; run 289, see the note below) | 47.81 (+17.0 % [+16.5 %, +17.6 %]; vs public drafter +19.4 % [+18.3 %, +20.6 %]) | 74.88 (+17.0 % [+4.6 %, +30.7 %]; vs public drafter +3.4 % [+3.0 %, +3.6 %]) | 63.54 (+7.5 % [+4.8 %, +11.8 %]; vs public drafter +1.0 % [-1.5 %, +3.4 %]) | 56.03 (+11.6 % [+9.4 %, +15.3 %]; vs public drafter +5.1 % [+4.7 %, +5.4 %]) |
| **Ours, relaxed + privately trained drafter** (non-standard; not reproducible: head not published; run 289, see the note below) | 47.44 (+16.1 % [+16.0 %, +16.2 %]) | 76.79 (+20.0 % [+8.5 %, +34.0 %]) | 62.67 (+6.0 % [+0.8 %, +14.8 %]) | 56.16 (+11.9 % [+10.3 %, +14.5 %]) |

The private-drafter rows were measured in run 289 (2026-11-16), a separate paired session on the same build that includes its own stock arm (stock there, tok/s: canonical C1 40.85; agent C1 63.99; agent C8 59.13; chat C1 50.20). Their changes vs stock are that run's own paired ratios, and "vs public drafter" compares the private and the public draft head within that same run; no ratio is taken across runs. The private head is not published, so these rows cannot be reproduced. Run 289's signed receipts are published for its stock and public-drafter launches; its private-drafter launches have none, because a receipt records the hash of every input and those inputs are private.

- **Strict** is standard lossless greedy speculative decoding: output is the model's own greedy output (see
  [Exactness](docs/EXACTNESS.md)).

- **Relaxed** is non-standard. Speculative acceptance also takes a drafted token whose logit is within 0.05 of the top token (a near-tie; the switch refuses values above 0.10), so output can
  differ from greedy. It is labelled wherever it appears.
- Stock is upstream gufo **v0.34.0** (commit `2845939b`), built with upstream's own Windows recipe,
  with the public shared Q8_0 MTP draft head that upstream's model guide uses. Our arms use the **same public head**, plus a draft-vocabulary table rebuilt from a public
  corpus with the tools in this repo (K = 6,931).

Where we are not clearly ahead of stock (95 % CI not above zero, or no CI): canonical C1 strict -1.5 % [-1.9 %, -0.8 %]; canonical C1 relaxed -1.1 % [-1.3 %, -0.8 %].

Relaxed vs strict (wins / losses only where the 95 % CI excludes zero): canonical C1 +0.4 % [-0.0 %, +0.7 %]; agent C1 +0.9 % [-0.2 %, +2.2 %]; agent C8 +2.0 % [+0.3 %, +4.2 %]; chat C1 +0.2 % [-0.7 %, +0.9 %].







Against our previous build (pre-rebase, `edb3fe51`), strict: canonical C1 -3.4 % [-3.8 %, -2.8 %]; agent C1 +2.0 % [-2.6 %, +7.6 %]; agent C8 +16.5 % [+14.4 %, +18.1 %]; chat C1 -0.5 % [-2.8 %, +0.8 %].

### Launch files: the benchmarked settings for each task

Each file starts the server as it was benchmarked in run 287 (arguments and switches), with the public
draft head and the public draft-vocabulary table in [perm/](perm/), except that the agent and chat files drop
`--max-tokens 128`, the server's default reply cap for clients that do not send `max_tokens` (the relaxed agent files
also add `--preserve-thinking auto`). The benchmark clients sent `max_tokens` 2048, so the numbers are unaffected
(an internal like-for-like check, run 288, relaxed agent at 1 session, no published receipts: 1.003 [0.998, 1.009]);
`canonical-bench.ps1` keeps the cap (upstream's tg128 recipe).
Run it from a PowerShell prompt, e.g.
`.\profiles\agent-1-session.ps1 -Gufo <path>\gufo.exe -ModelDir <path>\Qwen3.8-Flash-Next-GGUF`.

| launch file | use this for | sessions | context | exactness |
|---|---|---:|---:|---|
| [`profiles/agent-1-session.ps1`](profiles/agent-1-session.ps1) | one coding agent at a time (tool-calling agent loops); output = the model's own greedy output | 8 | 32768 | strict |
| [`profiles/agent-8-sessions.ps1`](profiles/agent-8-sessions.ps1) | up to 8 coding agents in parallel against one server; output = greedy | 8 | 32768 | strict |
| [`profiles/chat.ps1`](profiles/chat.ps1) | interactive multi-turn chat; output = greedy | 1 | 32768 | strict |
| [`profiles/canonical-bench.ps1`](profiles/canonical-bench.ps1) | reproducing the canonical prose row (upstream's model-bench recipe, thinking off) | 1 | 32768 | strict |
| [`profiles/relaxed-agent-1-session.ps1`](profiles/relaxed-agent-1-session.ps1) | one coding agent at a time when you accept near-tie differences from greedy for speed; the relaxed cell timed in the benchmark | 8 | 32768 | relaxed |
| [`profiles/relaxed-agent-8-sessions.ps1`](profiles/relaxed-agent-8-sessions.ps1) | up to 8 coding agents in parallel with near-tie acceptance; the relaxed cell timed in the benchmark | 8 | 32768 | relaxed |
| [`profiles/relaxed-chat.ps1`](profiles/relaxed-chat.ps1) | interactive multi-turn chat with near-tie acceptance; the relaxed cell timed in the benchmark | 1 | 32768 | relaxed |

### Switches used in the benchmark

Our build is upstream v0.34.0 plus the patches in [patches/](patches/), built with TheRock 10.0.0. Every
change is off unless its switch is set; with no switch set the build behaves as upstream. The strict profiles set
20 switches:

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

### What a drafter trained on your own traffic adds (numbers only)

The rows below use a draft head and vocabulary table that we trained on our **own** private traffic. They are not
reproducible from public data, and the head is not published. We show them so you can see what training on your own
traffic can buy. The [training guide](docs/TRAIN-YOUR-OWN-DRAFTER.md) and the [tooling](drafter-training/) are
published.

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| Private vs public drafter, strict (same session, run 289) | +19.4 % [+18.3 %, +20.6 %] | +3.4 % [+3.0 %, +3.6 %] | +1.0 % [-1.5 %, +3.4 %] | +5.1 % [+4.7 %, +5.4 %] |

The same private-drafter configurations, with tok/s and the change vs stock, are rows of the headline table above (run 289).

## Read more

| document | what it holds |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | every arm, every ratio with its CI, acceptance and tokens per target pass, TTFT, the exactness tables, the limitations |
| [docs/SPEC-COST.md](docs/SPEC-COST.md) | what speculation costs: drafted / accepted / discarded tokens per round, tokens per target pass, TTFT |
| [profiles/](profiles/) | ready-to-run launch files, one per task, as benchmarked (agent and chat files without the 128-token default reply cap) |
| [patches/](patches/) | our engine changes as patches against upstream v0.34.0 |
| [docs/EXACTNESS.md](docs/EXACTNESS.md) | what "strict" means and how it was checked, the one known cache-cut class and its fix, the open near-tie flip, and how stock's speculation compares |
| [docs/BUILD-AND-RUN.md](docs/BUILD-AND-RUN.md) | building and running gufo on Strix Halo under Windows, with the flags we benchmarked |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | pre-registration, interleaved A/B launches, quiet-host gates, bootstrap CIs, receipts |
| [docs/EXPERIMENT-LOG.md](docs/EXPERIMENT-LOG.md) | summary of the experiment log (555 rows, detailed through run 268, including the many that failed); 347 experiments run in all (2026-10-26 to 2026-11-16) |
| [docs/TRAIN-YOUR-OWN-DRAFTER.md](docs/TRAIN-YOUR-OWN-DRAFTER.md) | data, steps, compute and cost, the expected gain by workload, and the pitfalls |
| [drafter-training/](drafter-training/) | the drafter-training and draft-vocabulary tooling (Python) |
| [tools/](tools/) | public workload builders and the public draft-vocabulary pipeline |
| site | the same material as browsable pages, with the per-experiment log (through run 268): https://dvellon.github.io/gufo-347/ |

## What is not in this release

- Binaries. Build from upstream v0.34.0 plus [patches/](patches/) (see [BUILD-AND-RUN](docs/BUILD-AND-RUN.md)).
  We plan to send the clean, self-contained changes upstream as separate pull requests.
- The exact LM-head screen (A1) scales its bound with a per-channel table. The benchmarked build and the released patches both use a flat, data-free calibration table (every channel 1.0). The screen is exact for any table; the table only changes how many candidate rows are re-read.
- Our privately trained draft head, our private vocabulary table, and any private workload or traffic.
- The benchmark runner itself is not in this release, because it pins private files. The public workload builders and the draft-vocabulary tools are in tools/.

## Limitations

- Windows 11 only. Upstream treats Windows as a development target, and its production packages are built on Linux.
  Linux numbers may differ.
- Both our build and stock are built with the same toolchain, TheRock 10.0.0, upstream's documented Windows recipe, so the ratios compare code, not toolchains.
- One machine, one model and one quantisation. Every arm uses Unsloth revision 38bb39ee, while upstream now
  recommends a newer revision.
- The workloads measure performance, not accuracy: SWE-smith agent trajectories replayed with the server's own
  replies fed back, MT-Bench questions, and upstream's pp2048/tg128 recipe.

## Licence

MIT (same as upstream gufo). The model, the GGUF files, the datasets and upstream gufo are linked here, not redistributed. Each
keeps its own licence.

---
Numbers filled by fill_release.py from benchmark run 287 (written 2026-11-15 23:56:05 ET).
