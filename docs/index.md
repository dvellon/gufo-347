---
layout: home
title: Overview
---



## What this is

[gufo](https://github.com/gufo-org/gufo) is an open-source local inference engine for AMD Strix Halo. This site
documents **347 experiments** (2026-10-26 to 2026-11-16) of performance work on **one model on one machine**:

| | |
|---|---|
| model | Qwen3.8-Flash-Next, Unsloth UD-Q4_K_XL GGUF (revision 38bb39ee), with its MTP draft head for speculative decoding |
| machine | AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GiB unified memory, Windows 11 Pro |
| stack | our build on TheRock 10.0.0; stock on upstream's own Windows recipe |
| engine | our gufo fork R8, rebased on upstream v0.35.0 (0260e116); every change sits behind a runtime switch (off = upstream behaviour) |
| workloads | coding-agent sessions (one and eight at a time), multi-turn chat with thinking, and upstream's canonical pp2048/tg128 benchmark |

The goal was throughput for a machine that serves coding agents around the clock, **without changing the model's
output**. Every production change is checked to give the model's own greedy output (the *strict* tier). The only
exception is an explicitly labelled *relaxed* tier. See [Exactness]({{ site.baseurl }}/results/#exactness).

## Headline results

**Release benchmark, run 294 (2026-11-17), stock upstream v0.35.0 vs ours (R8)**
([details]({{ site.baseurl }}/results/)):

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| **Stock** upstream gufo v0.35.0 (tok/s) | 38.76 | 64.80 | 57.74 | 49.62 |
| **Ours, strict** (R8; lossless; public draft head and public vocab table) | 39.39 (+1.6 % [+1.2 %, +2.1 %]) | 71.56 (+10.4 % [+9.0 %, +12.9 %]) | 61.54 (+6.6 % [+2.2 %, +10.5 %]) | 52.35 (+5.5 % [+5.5 %, +5.5 %]) |
| **Ours, relaxed** (R8; non-standard: near-tie acceptance; public assets) | 39.38 (+1.6 % [+0.8 %, +2.4 %]) | 72.63 (+12.1 % [+10.6 %, +13.6 %]) | 63.48 (+9.9 % [+7.9 %, +12.4 %]) | 52.51 (+5.8 % [+5.7 %, +6.0 %]) |
| **Ours, strict + privately trained drafter** (R8; lossless; head and vocab table not published, so not reproducible; run 295, own stock arm) | 47.84 (+20.0 % [+19.6 %, +20.4 %]) | 73.58 (+14.0 % [+11.7 %, +17.4 %]) | 62.44 (+9.8 % [+8.3 %, +11.4 %]) | 56.53 (+10.9 % [+10.3 %, +11.3 %]) |
| **Ours, relaxed + privately trained drafter** (R8; non-standard; not reproducible; run 295, own stock arm) | 47.98 (+20.4 % [+19.6 %, +21.1 %]) | 75.27 (+16.6 % [+14.6 %, +18.6 %]) | 61.94 (+8.9 % [+7.7 %, +10.2 %]) | 56.50 (+10.8 % [+10.4 %, +11.1 %]) |
| Our previous build, strict (R7R on v0.34.0; reference row) | 39.21 (+1.2 % [+1.0 %, +1.3 %]) | 71.53 (+10.4 % [+9.1 %, +12.7 %]) | 62.33 (+8.0 % [+6.2 %, +9.8 %]) | 51.98 (+4.8 % [+4.6 %, +5.0 %]) |

Privately trained drafter rows: our own draft head and vocabulary table, trained on our private traffic and not published, so these rows cannot be reproduced and run 295 has no published receipts. Run 295 (2026-11-17) is a separate paired session on the same R8 build with its own stock v0.35.0 arm (stock there, tok/s: canonical C1 39.87; agent C1 64.56; agent C8 56.87; chat C1 50.97); changes are that run's own paired ratios. The strict row passed the same exactness gates as run 294.

Strict passed every pre-registered exactness gate in run 294. Chat and canonical rows have 2 rounds each, so their
CIs understate the uncertainty; the agent rows have 4. Where we are not clearly ahead of stock (95 % CI not above
zero): none in run 294.

The previous release (R7R on upstream v0.34.0, run 287 against stock v0.34.0, with the privately trained drafter rows
of run 289) is kept on [Results]({{ site.baseurl }}/results/#previous-release-run-287-r7r-on-upstream-v0340). Ratios
are never taken across runs.

**Train your own drafter.** The [guide]({{ site.baseurl }}/drafter/) gives the data, steps, compute
(about $21 of H100 time for the path that worked) and pitfalls, and the tooling is in the repository.

**What shipped:** a series of gated production promotions, each with its own evidence and rollback. See
[Production]({{ site.baseurl }}/production/).

**What did not:** the [experiment log]({{ site.baseurl }}/experiments/) lists every queue and packet through run 268, each re-checked
in a gains audit. There are far more KILLs than GOs, and every INVALID run is listed with its cause.

## How to read this site

| page | what it holds |
|---|---|
| [Results]({{ site.baseurl }}/results/) | the release benchmark (run 294) and the previous release (run 287): every arm, ratio, exactness check and limitation |
| [Train your own drafter]({{ site.baseurl }}/drafter/) | data, steps, compute, expected gain by workload, pitfalls |
| [Benchmarks]({{ site.baseurl }}/benchmarks/) | earlier runs: the quick look, the matched benchmark, the production re-profile history |
| [Production]({{ site.baseurl }}/production/) | what is running, the promotion history, the server profiles |
| [Experiment log]({{ site.baseurl }}/experiments/) | one row per queue or packet: what was tested, its true verdict, the key number with its CI, what happened next |
| [Open items]({{ site.baseurl }}/open-items/) | the audit's ranked list of possible gains not yet in production |
| [Methods]({{ site.baseurl }}/methods/) | how we measure, and the lessons learned the hard way |
| [Reproduce]({{ site.baseurl }}/reproduce/) | hardware notes and what can be reproduced |

Terms: **C1 / C8** = 1 or 8 concurrent sessions; **canonical** = upstream's pp2048/tg128 recipe; **strict / relaxed**
= the exactness tiers; **GO / KILL / REVIEW / INVALID** = verdicts against a line written down before the run.
Times are New York time (ET).

<small>Numbers from benchmark run 294 (verdict written 2026-11-17 18:31:16 ET); previous release: runs 287 and 289.</small>
