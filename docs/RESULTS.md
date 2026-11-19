# Results: benchmark run 294 (R8 on upstream v0.35.0)

Run 294 is the benchmark of this release: our build R8 (upstream v0.35.0 plus our changes behind runtime switches), measured in its strict tier and in its relaxed (non-standard, near-tie) tier in the same session, against stock v0.35.0, with our previous build (R7R on v0.34.0, the build of the previous release) as a reference row. Every ratio is paired within this run. It was pre-registered as the gate for moving our build to v0.35.0: R8 strict had to beat stock (95 % CI lower bound above 1.00) at agent C8 and again at agent C1, and pass every strict exactness check. It passed both checkpoints and every exactness check.

**Benchmark run 294, written 2026-11-17 18:31:16 ET.** AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GiB unified memory, Windows 11 Pro; Qwen3.8-Flash-Next, Unsloth UD-Q4_K_XL GGUF (revision 38bb39ee). The workloads, client and launch cells (arguments and switches) are run 287's, re-pointed to the new binaries. Each launch gets a fresh server, and every arm runs the same session subset in each round (4-arm Williams design). Ratios are paired percentile bootstrap 95 % CIs over rounds (4000 resamples, seed 260). Rounds: agent C8 4, agent C1 4, chat C1 2, canonical C1 2.

| arm | what | name in the receipts |
|---|---|---|
| **stock** | upstream gufo v0.35.0 (commit `0260e116`), built with TheRock 10.0.0 (upstream BUILD-WINDOWS.md, `cmake --preset windows-release`); the public shared Q8_0 MTP draft head that upstream's model guide uses (`5ff54097`) | A |
| **ours, strict** | our build R8 `8735d242` = upstream v0.35.0 + our patches, built with TheRock 10.0.0, with the switches listed in the [README](../README.md#switches-used-in-the-benchmark); the **same public head**, and a public draft-vocabulary table (K = 6,931, `caade56a`) built with the tools in this repo | R7Q |
| **ours, relaxed** | the same binary and assets, with the relaxed (non-standard, near-tie) profile | R7QX |
| **previous build** (reference) | our previous release build R7R `fe7b7048` = upstream v0.34.0 + the previous patch series, TheRock 10.0.0, the same strict cells and assets | B |
| **references** | R8 with speculation off: the exactness references | RR7Q |

The receipt names are the benchmark harness's cell names, kept from run 287 so that the cells stay byte-identical.

## Headline

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| **Stock** upstream gufo v0.35.0 (tok/s) | 38.76 | 64.80 | 57.74 | 49.62 |
| **Ours, strict** (R8; lossless; public draft head and public vocab table) | 39.39 (+1.6 % [+1.2 %, +2.1 %]) | 71.56 (+10.4 % [+9.0 %, +12.9 %]) | 61.54 (+6.6 % [+2.2 %, +10.5 %]) | 52.35 (+5.5 % [+5.5 %, +5.5 %]) |
| **Ours, relaxed** (R8; non-standard: near-tie acceptance; public assets) | 39.38 (+1.6 % [+0.8 %, +2.4 %]) | 72.63 (+12.1 % [+10.6 %, +13.6 %]) | 63.48 (+9.9 % [+7.9 %, +12.4 %]) | 52.51 (+5.8 % [+5.7 %, +6.0 %]) |
| Our previous build, strict (R7R on v0.34.0; reference row) | 39.21 (+1.2 % [+1.0 %, +1.3 %]) | 71.53 (+10.4 % [+9.1 %, +12.7 %]) | 62.33 (+8.0 % [+6.2 %, +9.8 %]) | 51.98 (+4.8 % [+4.6 %, +5.0 %]) |

- **Strict** = standard lossless greedy speculative decoding. **Relaxed** = non-standard near-tie acceptance; its
  output can differ from greedy (see [Exactness](#exactness)).
- Where we are not clearly ahead of stock (95 % CI not above zero, or no CI): none in run 294.
- Chat and canonical rows have 2 rounds each. With 2 rounds the bootstrap CI only spans the two round ratios, so it
  understates the uncertainty there; the agent rows (4 rounds each) carry the more reliable CIs.

Relaxed vs strict (wins / losses only where the 95 % CI excludes zero): canonical C1 -0.0 % [-0.4 %, +0.3 %]; agent C1 +1.5 % [-0.7 %, +3.8 %]; agent C8 +3.2 % [+1.7 %, +6.2 %]; chat C1 +0.3 % [+0.2 %, +0.5 %].

Against our previous build (R7R on v0.34.0, `fe7b7048`, same run), R8 strict: canonical C1 +0.5 % [-0.1 %, +1.0 %]; agent C1 +0.0 % [-0.4 %, +0.6 %]; agent C8 -1.3 % [-4.2 %, +1.5 %]; chat C1 +0.7 % [+0.5 %, +0.9 %]. R8 relaxed vs the previous build (strict): canonical C1 +0.4 % [-0.5 %, +1.4 %]; agent C1 +1.5 % [-0.6 %, +3.3 %]; agent C8 +1.8 % [+0.4 %, +3.6 %]; chat C1 +1.0 % [+1.0 %, +1.1 %].

**Pre-registered checkpoints.** After the last agent C8 launch: R8 strict / stock 1.066 [1.022, 1.105], PROCEED. After the last agent C1 launch: 1.104 [1.090, 1.129], PROCEED. The relaxed tier also cleared both checkpoints (R8 relaxed / stock CI lower bound above 1.00).

**Why the canonical row changed sign since the previous release.** Mostly because stock moved, not because our build got faster: our previous build R7R measures +1.2 % [+1.0 %, +1.3 %] against stock v0.35.0 in this run, where it measured -1.5 % [-1.9 %, -0.8 %] against stock v0.34.0 in run 287. Between those releases upstream changed the draft-head path (#1209, a Q4_K copy of the draft-head prefix) and the device-code math flags for this model (#1215); stock's canonical draft acceptance is 0.534 here against 0.564 in run 287. We have not isolated which change moved stock. Cross-session tok/s are not paired and are not used for any ratio.

## Every arm

| stage | arm | tok/s | launches | acceptance (mean) | tokens / target pass (mean) | warm TTFT p50 ms (mean) |
|---|---|---:|---:|---:|---:|---:|
| canonical C1 | stock v0.35.0 | 38.76 | 2 | 0.534 | 1.83 | 1154 |
| canonical C1 | ours strict (R8) | 39.39 | 2 | 0.401 | 2.12 | 1147 |
| canonical C1 | ours relaxed (R8) | 39.38 | 2 | 0.407 | 2.13 | 1152 |
| canonical C1 | previous build (R7R on v0.34.0; reference) | 39.21 | 2 | 0.401 | 2.12 | 1145 |
| agent C1 | stock v0.35.0 | 64.80 | 4 | 0.809 | 3.92 | 368 |
| agent C1 | ours strict (R8) | 71.56 | 4 | 0.791 | 4.43 | 372 |
| agent C1 | ours relaxed (R8) | 72.63 | 4 | 0.793 | 4.58 | 374 |
| agent C1 | previous build (R7R on v0.34.0; reference) | 71.53 | 4 | 0.791 | 4.43 | 382 |
| agent C8 | stock v0.35.0 | 57.74 | 4 | 0.916 | 2.87 | 1985 |
| agent C8 | ours strict (R8) | 61.54 | 4 | 0.872 | 3.09 | 1804 |
| agent C8 | ours relaxed (R8) | 63.48 | 4 | 0.872 | 3.13 | 1615 |
| agent C8 | previous build (R7R on v0.34.0; reference) | 62.33 | 4 | 0.875 | 3.13 | 1791 |
| chat C1 | stock v0.35.0 | 49.62 | 2 | 0.710 | 2.50 | 170 |
| chat C1 | ours strict (R8) | 52.35 | 2 | 0.647 | 2.77 | 166 |
| chat C1 | ours relaxed (R8) | 52.51 | 2 | 0.645 | 2.78 | 167 |
| chat C1 | previous build (R7R on v0.34.0; reference) | 51.98 | 2 | 0.647 | 2.77 | 172 |

Acceptance, tokens per target pass and TTFT are plain means over the launches (unpaired). Canonical TTFT is the time to first token of the pp2048 prose prompts.

## Every ratio

| stage | ratio | point | 95 % CI | rounds |
|---|---|---:|---|---:|
| canonical C1 | R8 strict / stock | 1.016 | [1.012, 1.021] | 2 |
| canonical C1 | R8 relaxed / stock | 1.016 | [1.008, 1.024] | 2 |
| canonical C1 | R8 relaxed / R8 strict | 1.000 | [0.996, 1.003] | 2 |
| canonical C1 | R8 strict / previous build | 1.005 | [0.999, 1.010] | 2 |
| canonical C1 | R8 relaxed / previous build | 1.004 | [0.995, 1.014] | 2 |
| canonical C1 | previous build / stock | 1.012 | [1.010, 1.013] | 2 |
| agent C1 | R8 strict / stock | 1.104 | [1.090, 1.129] | 4 |
| agent C1 | R8 relaxed / stock | 1.121 | [1.106, 1.136] | 4 |
| agent C1 | R8 relaxed / R8 strict | 1.015 | [0.993, 1.038] | 4 |
| agent C1 | R8 strict / previous build | 1.000 | [0.996, 1.006] | 4 |
| agent C1 | R8 relaxed / previous build | 1.015 | [0.994, 1.033] | 4 |
| agent C1 | previous build / stock | 1.104 | [1.091, 1.127] | 4 |
| agent C8 | R8 strict / stock | 1.066 | [1.022, 1.105] | 4 |
| agent C8 | R8 relaxed / stock | 1.099 | [1.079, 1.124] | 4 |
| agent C8 | R8 relaxed / R8 strict | 1.032 | [1.017, 1.062] | 4 |
| agent C8 | R8 strict / previous build | 0.987 | [0.958, 1.015] | 4 |
| agent C8 | R8 relaxed / previous build | 1.018 | [1.004, 1.036] | 4 |
| agent C8 | previous build / stock | 1.080 | [1.062, 1.098] | 4 |
| chat C1 | R8 strict / stock | 1.055 | [1.055, 1.055] | 2 |
| chat C1 | R8 relaxed / stock | 1.058 | [1.057, 1.060] | 2 |
| chat C1 | R8 relaxed / R8 strict | 1.003 | [1.002, 1.005] | 2 |
| chat C1 | R8 strict / previous build | 1.007 | [1.005, 1.009] | 2 |
| chat C1 | R8 relaxed / previous build | 1.010 | [1.010, 1.011] | 2 |
| chat C1 | previous build / stock | 1.048 | [1.046, 1.050] | 2 |

## Relaxed tier: quality

| stage | relaxed outputs identical to no-speculation | strict outputs identical (for scale) | median first difference (est. tokens into the reply) | task success |
|---|---:|---:|---:|---|
| canonical C1 | 80.0 % (8 / 10) | 100.0 % (10 / 10) | n/a | none (synthetic prose, fixed 128 tokens) |
| agent C1 | 76.3 % (71 / 93) | 100.0 % (93 / 93) | 99 | none (replay of recorded SWE-smith turns: observations are fixed, no test run) |
| agent C8 (vs pooled single-session references; proxy) | 83.8 % (83 / 99) | 96.0 % (95 / 99) | 182.5 | none (replay of recorded SWE-smith turns: observations are fixed, no test run) |
| chat C1 | 95.8 % (23 / 24) | 95.8 % (23 / 24) | 54 | none (MT-Bench needs an LLM judge; not run) |

Quality proxies (share of replies cut by the length cap / share of well-formed tool calls): agent C1 strict 0.000 / 0.914, relaxed 0.000 / 0.914, previous build 0.000 / 0.914; agent C8 strict 0.000 / 0.922, relaxed 0.000 / 0.917, previous build 0.000 / 0.922; chat C1 length-capped share strict 0.250, relaxed 0.250, previous build 0.250 (6 of 24 chat replies reach the 2,048-token cap in every arm).

## Exactness

| check (lineage judge, speculation flag explicit) | paired | compared | differing on the same cache path | unattributed breaks | excluded by cause | verdict |
|---|---:|---:|---:|---:|---|---|
| E1 c1 ours strict (R8) vs the majority no-speculation reference (REF-EQUIV held) | 93 | 93 | 0 | 0 | K1 0, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 c1 previous build (control) vs the majority no-speculation reference (REF-EQUIV held) | 93 | 93 | 0 | 0 | K1 0, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 chat ours strict (R8) vs R8's no-speculation reference | 24 | 23 | 0 | 0 | K1 1, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 chat previous build (control) vs R8's no-speculation reference | 24 | 23 | 0 | 0 | K1 1, K2 0, K3 0, UNATTRIBUTED 0 | PASS |

- E1 canonical, ours strict (R8): 10 of 10 completions identical to R8's no-speculation reference (PASS).
- E1 canonical, previous build (control): 10 of 10 completions identical to R8's no-speculation reference (PASS).
- The agent C1 reference pools R8's own no-speculation launches with four earlier no-speculation runs of the same cells on our v0.34.0 builds (runs 287, 275 and 284). The pre-registered REF-EQUIV check allowed that pooling only if R8's own references matched the earlier majority; they did (0 differing turns).

Plain identity against the no-speculation references (report only beyond the gates above):

| arm | agent C1 identical / paired | chat C1 identical / paired | agent C8 vs pooled C1 references (proxy) | canonical identical / paired |
|---|---:|---:|---:|---:|
| ours strict (R8) | 93 / 93 | 23 / 24 | 95 / 99 | 10 / 10 |
| ours relaxed (R8; non-standard) | 71 / 93 | 23 / 24 | 83 / 99 | 8 / 10 |
| previous build | 93 / 93 | 23 / 24 | 93 / 99 | 10 / 10 |

The strict chat request that is not identical is the one excluded by cause above (a turn after a reply cut by `max_tokens`, class K1). Agent C8 rows compare 8-session outputs with single-session references; batching legitimately changes which cache boundaries exist, so they are informational. See [EXACTNESS](EXACTNESS.md) for the K1 class, the open near-tie flip and how stock's speculation compares.

## Launches and receipts

- Every timed launch finished (none missing), with 0 request errors and 0 HIP / heap / memory error lines in total; no launch was flagged for an anomaly re-run.
- Each launch is labelled QUIET or LOUD by a host-load monitor with pre-registered thresholds (background CPU or GPU use, or memory paging). LOUD launches are kept and reported, not re-run. Run 294: 33 of 48 timed launches LOUD (agent C1 10/16, agent C8 7/16, canonical C1 8/8, chat C1 8/8). LOUD launches occur in every arm (see the per-stage counts in the verdict).
- Signed receipts: 55 launch receipts (every timed and reference launch; warm-up launches excluded) and 15 row receipts in [receipts/runs/294](../receipts/runs/294/). Every ratio and per-arm value in this section was recomputed from the launch receipts with the receipts' own statistic and matches the run's verdict exactly.

## Limitations

- Windows 11 only. Upstream's production packages are built on Linux, so Linux numbers may differ.
- Both our build and stock are built with the same toolchain, TheRock 10.0.0, which upstream v0.35.0's Windows guide names as the only SDK validated at full speed on gfx1151, so the ratios compare code, not toolchains.
- Every arm uses the previous Unsloth revision (38bb39ee).
- These are performance workloads, not accuracy benchmarks.
- Run 294 is shorter than run 287 (48 timed launches instead of 88; chat and canonical have 2 rounds each).
- Our privately trained drafter was not measured on R8; its numbers below are from the previous build.


## Previous release: run 287 (R7R on upstream v0.34.0)

The first release of this repository: our build R7R (`fe7b7048`) on upstream v0.34.0 (`2845939b`), benchmarked in run 287 (2026-11-15) against stock v0.34.0. These numbers are kept unchanged. They are not comparable with run 294's ratios, because stock changed between v0.34.0 and v0.35.0, and no ratio is taken across runs.



Run 287 was the previous release's benchmark: one build of ours (upstream v0.34.0 plus our changes behind runtime switches), measured in its strict tier and in its relaxed (non-standard, near-tie) tier in the same session, against stock v0.34.0, with our previous build as a reference row. Every ratio is paired within this run. Strict exactness is lineage-judged against the build's own no-speculation references.

Written 2026-11-15 23:56:05 ET. Windows 11, gfx1151.

### Setup

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

### Headline

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

### Switches

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



### Private drafter (numbers only)

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| Private vs public drafter, strict (same session, run 289) | +19.4 % [+18.3 %, +20.6 %] | +3.4 % [+3.0 %, +3.6 %] | +1.0 % [-1.5 %, +3.4 %] | +5.1 % [+4.7 %, +5.4 %] |

The same private-drafter configurations, with tok/s and the change vs stock, are rows of the headline table above (run 289).

### Every arm

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

### Every ratio

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

### Relaxed tier: quality

| stage | relaxed outputs identical to no-speculation | strict outputs identical (for scale) | median first difference (est. tokens into the reply) | task success |
|---|---:|---:|---:|---|
| canonical C1 | 80.0 % (16 / 20) | 100.0 % (20 / 20) | n/a | none (synthetic prose, fixed 128 tokens) |
| agent C1 | 73.5 % (136 / 185) | 100.0 % (185 / 185) | 99 | none (replay of recorded SWE-smith turns: observations are fixed, no test run) |
| agent C8 | 81.2 % (234 / 288) | 94.1 % (271 / 288) | 27 | none (replay of recorded SWE-smith turns: observations are fixed, no test run) |
| chat C1 | 89.6 % (43 / 48) | 89.6 % (43 / 48) | 61 | none (MT-Bench needs an LLM judge; not run) |

### Exactness

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

### Limitations

- Windows 11 only. Upstream treats Windows as a development target, so Linux numbers may differ.
- Both our build and stock are built with the same toolchain, TheRock 10.0.0, upstream's documented Windows recipe, so the ratios compare code, not toolchains.
- Every arm uses the previous Unsloth revision (38bb39ee); upstream now recommends a newer one.
- The coding workload uses SWE-agent text function calling. Observations are replayed as recorded, so these are
  performance workloads, not accuracy benchmarks.
- Each launch is labelled QUIET or LOUD by a host-load monitor with pre-registered thresholds (background CPU or GPU use, or memory paging). LOUD launches are kept and reported, not re-run. Run 287: 21 of 88 timed launches LOUD (agent C1 3/32, agent C8 7/24, canonical C1 1/16, chat C1 10/16).

---
Numbers from benchmark run 294 (verdict written 2026-11-17 18:31:16 ET), recomputed from its signed receipts; previous-release numbers filled by fill_release.py from benchmark run 287 (written 2026-11-15 23:56:05 ET) and run 289.
