# gufo-347: gufo on Strix Halo, 347 experiments later



The number in the name is the number of experiments we actually ran for this first release: GPU experiment runs plus
offline CPU simulations. Work that was designed but never run, engineering-only fixes and pure research notes are not
counted.

This repository holds the results, methods and experiment log of a performance study of
[gufo](https://github.com/gufo-org/gufo), an open-source local inference engine, on one AMD Strix Halo machine. It
also holds the tooling to **train your own MTP draft head** on your own traffic.

- **Model:** Qwen3.8-Flash-Next, Unsloth UD-Q4_K_XL GGUF (revision 38bb39ee), with its MTP (multi-token prediction) draft head for speculative decoding.
- **Machine:** AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GiB unified memory, Windows 11 Pro.
- **Engine:** our gufo fork **R8**, rebased on upstream **v0.35.0** (0260e116); every change sits behind a runtime switch (off = upstream behaviour). Each change was kept only after a pre-registered A/B test passed; the switches
  used in the benchmark are listed [below](#switches-used-in-the-benchmark).
- **Goal:** more throughput for a machine that serves coding agents around the clock, **without changing the model's
  output**.

This release contains results, methods, the experiment log, the drafter-training tooling, and our engine changes as patches against upstream gufo v0.35.0 (patches/), each behind a runtime switch. The repository itself contains no binaries, model files, or our private drafter; a prebuilt Windows package is attached to the GitHub release (see [Install on Windows](#install-on-windows)).

## Headline results

Benchmark run 294, 2026-11-17. Same machine, model files, workloads and client for every arm. Each cell
shows tok/s, then the change vs stock with its 95 % CI. Ratios are paired bootstrap over rotated rounds, and each
launch uses a fresh server. Rounds: agent C1 4, agent C8 4, chat C1 2, canonical C1 2.

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| **Stock** upstream gufo v0.35.0 (tok/s) | 38.76 | 64.80 | 57.74 | 49.62 |
| **Ours, strict** (R8; lossless; public draft head and public vocab table) | 39.39 (+1.6 % [+1.2 %, +2.1 %]) | 71.56 (+10.4 % [+9.0 %, +12.9 %]) | 61.54 (+6.6 % [+2.2 %, +10.5 %]) | 52.35 (+5.5 % [+5.5 %, +5.5 %]) |
| **Ours, relaxed** (R8; non-standard: near-tie acceptance; public assets) | 39.38 (+1.6 % [+0.8 %, +2.4 %]) | 72.63 (+12.1 % [+10.6 %, +13.6 %]) | 63.48 (+9.9 % [+7.9 %, +12.4 %]) | 52.51 (+5.8 % [+5.7 %, +6.0 %]) |
| Our previous build, strict (R7R on v0.34.0; reference row in the same run) | 39.21 (+1.2 % [+1.0 %, +1.3 %]) | 71.53 (+10.4 % [+9.1 %, +12.7 %]) | 62.33 (+8.0 % [+6.2 %, +9.8 %]) | 51.98 (+4.8 % [+4.6 %, +5.0 %]) |

- **Strict** is standard lossless greedy speculative decoding: output is the model's own greedy output (see
  [Exactness](docs/EXACTNESS.md)). In run 294 it passed every pre-registered exactness gate: agent C1 93 of 93
  requests identical to the no-speculation reference, chat 0 differences on the same cache path (23 compared, 1
  excluded by the known cache-cut class), canonical 10 of 10 completions identical.

- **Relaxed** is non-standard. Speculative acceptance also takes a drafted token whose logit is within 0.05 of the top token (a near-tie; the switch refuses values above 0.10), so output can
  differ from greedy. It is labelled wherever it appears.
- Stock is upstream gufo **v0.35.0** (commit `0260e116`), built with upstream's own Windows recipe (TheRock 10.0.0),
  with the public shared Q8_0 MTP draft head that upstream's model guide uses. Our arms use the **same public head**, plus a draft-vocabulary table rebuilt from a public
  corpus with the tools in this repo (K = 6,931).
- Chat and canonical rows have 2 rounds each. With 2 rounds the bootstrap CI only spans the two round ratios, so it
  understates the uncertainty there; the agent rows (4 rounds each) carry the more reliable CIs.

Where we are not clearly ahead of stock (95 % CI not above zero, or no CI): none in run 294; every strict and relaxed cell's CI is above zero.

Relaxed vs strict (wins / losses only where the 95 % CI excludes zero): canonical C1 -0.0 % [-0.4 %, +0.3 %]; agent C1 +1.5 % [-0.7 %, +3.8 %]; agent C8 +3.2 % [+1.7 %, +6.2 %]; chat C1 +0.3 % [+0.2 %, +0.5 %].

Against our previous build (R7R on v0.34.0, `fe7b7048`, same run), R8 strict: canonical C1 +0.5 % [-0.1 %, +1.0 %]; agent C1 +0.0 % [-0.4 %, +0.6 %]; agent C8 -1.3 % [-4.2 %, +1.5 %]; chat C1 +0.7 % [+0.5 %, +0.9 %]. R8 is about as fast as the previous build; what it adds is upstream v0.35.0 and its fixes (see [patches/](patches/)).

Every launch of run 294 has a signed receipt in [receipts/runs/294/](receipts/runs/294/); every number in the table
above can be recomputed from those receipts.

### Launch files: the benchmarked settings for each task

Each file starts the server as it was benchmarked in run 294 (arguments and switches; the same cells as run 287), with the public
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

Our build is upstream v0.35.0 plus the patches in [patches/](patches/), built with TheRock 10.0.0. Every
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

The relaxed profiles add `GUFO_NT_ACCEPT_EPS=0.05` (near-tie acceptance) and `GUFO_XM_RELAXED_ALIGN=1` and start the
server with `--exactness relaxed`. `GUFO_DRAFT_Q4K` is left unset everywhere, so upstream v0.35.0's Q4_K copy of the
draft-head prefix (#1209, ported to our permuted head) is active, as in every benchmarked R8 launch.

### What a drafter trained on your own traffic adds (numbers only)

The rows below use a draft head and vocabulary table that we trained on our **own** private traffic. They are not
reproducible from public data, and the head is not published. We show them so you can see what training on your own
traffic can buy. The [training guide](docs/TRAIN-YOUR-OWN-DRAFTER.md) and the [tooling](drafter-training/) are
published. They were measured on the **previous build** (R7R on upstream v0.34.0, run 289) and were not re-measured
on R8.

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| Private vs public drafter, strict (same session, run 289, previous build) | +19.4 % [+18.3 %, +20.6 %] | +3.4 % [+3.0 %, +3.6 %] | +1.0 % [-1.5 %, +3.4 %] | +5.1 % [+4.7 %, +5.4 %] |

The same private-drafter configurations, with tok/s and the change vs stock v0.34.0, are in the
[previous release](#previous-release-r7r-on-upstream-v0340-run-287) section below.

## Previous release: R7R on upstream v0.34.0 (run 287)

The first release of this repository was our build R7R (`fe7b7048`) on upstream v0.34.0 (`2845939b`), benchmarked in
run 287 (2026-11-15) against stock v0.34.0. Those numbers stay valid for that pair; they are not comparable with run
294's ratios, because stock changed between v0.34.0 and v0.35.0 and no ratio is taken across runs. The patches for
that build (`0001-r7.patch` ... `0006-r7r.patch` against v0.34.0) are in this repository's git history, and the
receipts of runs 287 and 289 stay in [receipts/runs/](receipts/runs/).

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| **Stock** upstream gufo v0.34.0 (tok/s) | 41.03 | 62.98 | 57.43 | 49.82 |
| **Ours, strict** (R7R; lossless; public draft head and public vocab table) | 40.41 (-1.5 % [-1.9 %, -0.8 %]) | 71.27 (+13.2 % [+6.5 %, +19.5 %]) | 61.16 (+6.5 % [+3.0 %, +10.1 %]) | 53.98 (+8.3 % [+5.5 %, +11.0 %]) |
| **Ours, relaxed** (R7R; non-standard: near-tie acceptance; public assets) | 40.58 (-1.1 % [-1.3 %, -0.8 %]) | 71.93 (+14.2 % [+8.4 %, +20.1 %]) | 62.40 (+8.7 % [+3.6 %, +14.2 %]) | 54.08 (+8.6 % [+6.3 %, +10.3 %]) |
| **Ours, strict + privately trained drafter** (not reproducible: head not published; run 289, see the note below) | 47.81 (+17.0 % [+16.5 %, +17.6 %]; vs public drafter +19.4 % [+18.3 %, +20.6 %]) | 74.88 (+17.0 % [+4.6 %, +30.7 %]; vs public drafter +3.4 % [+3.0 %, +3.6 %]) | 63.54 (+7.5 % [+4.8 %, +11.8 %]; vs public drafter +1.0 % [-1.5 %, +3.4 %]) | 56.03 (+11.6 % [+9.4 %, +15.3 %]; vs public drafter +5.1 % [+4.7 %, +5.4 %]) |
| **Ours, relaxed + privately trained drafter** (non-standard; not reproducible: head not published; run 289, see the note below) | 47.44 (+16.1 % [+16.0 %, +16.2 %]) | 76.79 (+20.0 % [+8.5 %, +34.0 %]) | 62.67 (+6.0 % [+0.8 %, +14.8 %]) | 56.16 (+11.9 % [+10.3 %, +14.5 %]) |

The private-drafter rows were measured in run 289 (2026-11-16), a separate paired session on the same build that includes its own stock arm (stock there, tok/s: canonical C1 40.85; agent C1 63.99; agent C8 59.13; chat C1 50.20). Their changes vs stock are that run's own paired ratios, and "vs public drafter" compares the private and the public draft head within that same run; no ratio is taken across runs. The private head is not published, so these rows cannot be reproduced. Run 289's signed receipts are published for its stock and public-drafter launches; its private-drafter launches have none, because a receipt records the hash of every input and those inputs are private.

In run 287, R7R was not clearly ahead of stock v0.34.0 on canonical C1 (strict -1.5 % [-1.9 %, -0.8 %]; relaxed
-1.1 % [-1.3 %, -0.8 %]). Full run-287 detail: [docs/RESULTS.md](docs/RESULTS.md#previous-release-run-287-r7r-on-upstream-v0340).

## Install on Windows

A prebuilt package of R8 (upstream v0.35.0 plus [patches/](patches/), built with TheRock 10.0.0) for Windows 11 x64
on the Radeon 8060S / 8050S (gfx1151) is attached to the
[GitHub release](https://github.com/dvellon/gufo-347/releases/tag/v0.35.0-r8):
`gufo-347-r8-windows-x64-gfx1151.zip` (sha256 `793bc1b8aa59b97a1e0a458a46f289f437cfefe2b17890933e78896802474e07`),
the installer `install-gufo-347.ps1`, and `SHA256SUMS`. The zip holds gufo.exe with its runtime DLLs, the launch
files above and the public draft-vocabulary table; it contains no model files and no private draft head or
vocabulary table.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-gufo-347.ps1 -ModelDir D:\models\Qwen3.8-Flash-Next-GGUF
gufo-347 agent-1-session     # in a new terminal; the server listens on 127.0.0.1:8080
gufo-347 test                # from a second terminal: one short chat request
```

The installer checks the machine (GPU, driver, memory carve-out, pagefile, Visual C++ runtime), downloads the zip
from this repository's release and installs it only if its sha256 matches `SHA256SUMS`, installs per user to
`%LOCALAPPDATA%\gufo-347` (no administrator rights), and adds a `gufo-347` command, Start-menu shortcuts and an
uninstaller. Prerequisites, the model download, a manual install and troubleshooting:
[docs/INSTALL.md](docs/INSTALL.md).

## Read more

| document | what it holds |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | every arm, every ratio with its CI, acceptance and tokens per target pass, TTFT, the exactness tables, the limitations (run 294, then the previous release's run 287) |
| [docs/SPEC-COST.md](docs/SPEC-COST.md) | what speculation costs: drafted / accepted / discarded tokens per round, tokens per target pass, TTFT |
| [profiles/](profiles/) | ready-to-run launch files, one per task, as benchmarked (agent and chat files without the 128-token default reply cap) |
| [patches/](patches/) | our engine changes as patches against upstream v0.35.0 |
| [receipts/](receipts/) | signed per-launch and per-row receipts (runs 294, 287, 289) and the verifier |
| [docs/EXACTNESS.md](docs/EXACTNESS.md) | what "strict" means and how it was checked, the one known cache-cut class and its fix, the open near-tie flip, and how stock's speculation compares |
| [docs/BUILD-AND-RUN.md](docs/BUILD-AND-RUN.md) | building and running gufo on Strix Halo under Windows, with the flags we benchmarked |
| [docs/INSTALL.md](docs/INSTALL.md) | installing the prebuilt Windows package: prerequisites, model download, memory settings, troubleshooting |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | pre-registration, interleaved A/B launches, quiet-host gates, bootstrap CIs, receipts |
| [docs/EXPERIMENT-LOG.md](docs/EXPERIMENT-LOG.md) | summary of the experiment log (555 rows, detailed through run 268, including the many that failed); 347 experiments run in all (2026-10-26 to 2026-11-16) |
| [docs/TRAIN-YOUR-OWN-DRAFTER.md](docs/TRAIN-YOUR-OWN-DRAFTER.md) | data, steps, compute and cost, the expected gain by workload, and the pitfalls |
| [drafter-training/](drafter-training/) | the drafter-training and draft-vocabulary tooling (Python) |
| [tools/](tools/) | public workload builders and the public draft-vocabulary pipeline |
| site | the same material as browsable pages, with the per-experiment log (through run 268): https://dvellon.github.io/gufo-347/ |

## What is not in this release

- Binaries in the git tree. The prebuilt Windows package is a release asset (see
  [Install on Windows](#install-on-windows)); to build it yourself, use upstream v0.35.0 plus [patches/](patches/)
  (see [BUILD-AND-RUN](docs/BUILD-AND-RUN.md)).
  We plan to send the clean, self-contained changes upstream as separate pull requests.
- The exact LM-head screen (A1) scales its bound with a per-channel table. The benchmarked build and the released patches both use a flat, data-free calibration table (every channel 1.0). The screen is exact for any table; the table only changes how many candidate rows are re-read.
- Our privately trained draft head, our private vocabulary table, and any private workload or traffic.
- The benchmark runner itself is not in this release, because it pins private files. The public workload builders and the draft-vocabulary tools are in tools/.

## Limitations

- Windows 11 only. Upstream treats Windows as a development target, and its production packages are built on Linux.
  Linux numbers may differ.
- Both our build and stock are built with the same toolchain, TheRock 10.0.0, which upstream v0.35.0's Windows guide names as the only SDK validated at full speed on gfx1151, so the ratios compare code, not toolchains.
- One machine, one model and one quantisation. Every arm uses Unsloth revision 38bb39ee, while upstream now
  recommends a newer revision.
- The workloads measure performance, not accuracy: SWE-smith agent trajectories replayed with the server's own
  replies fed back, MT-Bench questions, and upstream's pp2048/tg128 recipe.
- Run 294 is shorter than run 287 (48 timed launches instead of 88), and 33 of its 48 timed launches were labelled
  LOUD by the host-load monitor (kept and reported, as pre-registered; see [docs/RESULTS.md](docs/RESULTS.md)).

## Licence

MIT (same as upstream gufo). The model, the GGUF files, the datasets and upstream gufo are linked here, not redistributed. Each
keeps its own licence.

---
Numbers from benchmark run 294 (verdict written 2026-11-17 18:31:16 ET), recomputed from its signed receipts; previous-release numbers from runs 287 and 289.
