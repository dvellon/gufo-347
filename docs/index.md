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
| engine | our gufo fork, rebased on upstream v0.34.0 (2845939b); every change sits behind a runtime switch (off = upstream behaviour) |
| workloads | coding-agent sessions (one and eight at a time), multi-turn chat with thinking, and upstream's canonical pp2048/tg128 benchmark |

The goal was throughput for a machine that serves coding agents around the clock, **without changing the model's
output**. Every production change is checked to give the model's own greedy output (the *strict* tier). The only
exception is an explicitly labelled *relaxed* tier. See [Exactness]({{ site.baseurl }}/results/#exactness).

## Headline results

**Final public benchmark, run 287 (2026-11-15), stock upstream v0.34.0 vs ours**
([details]({{ site.baseurl }}/results/)):

| configuration | canonical C1 | agent C1 | agent C8 | chat C1 |
|---|---:|---:|---:|---:|
| **Stock** upstream gufo v0.34.0 (tok/s) | 41.03 | 62.98 | 57.43 | 49.82 |
| **Ours, strict** (lossless; public draft head and public vocab table) | 40.41 (-1.5 % [-1.9 %, -0.8 %]) | 71.27 (+13.2 % [+6.5 %, +19.5 %]) | 61.16 (+6.5 % [+3.0 %, +10.1 %]) | 53.98 (+8.3 % [+5.5 %, +11.0 %]) |
| **Ours, relaxed** (non-standard: near-tie acceptance; public assets) | 40.58 (-1.1 % [-1.3 %, -0.8 %]) | 71.93 (+14.2 % [+8.4 %, +20.1 %]) | 62.40 (+8.7 % [+3.6 %, +14.2 %]) | 54.08 (+8.6 % [+6.3 %, +10.3 %]) |
| **Ours, strict + privately trained drafter** (not reproducible: head not published; run 289, see the note below) | 47.81 (+17.0 % [+16.5 %, +17.6 %]; vs public drafter +19.4 % [+18.3 %, +20.6 %]) | 74.88 (+17.0 % [+4.6 %, +30.7 %]; vs public drafter +3.4 % [+3.0 %, +3.6 %]) | 63.54 (+7.5 % [+4.8 %, +11.8 %]; vs public drafter +1.0 % [-1.5 %, +3.4 %]) | 56.03 (+11.6 % [+9.4 %, +15.3 %]; vs public drafter +5.1 % [+4.7 %, +5.4 %]) |
| **Ours, relaxed + privately trained drafter** (non-standard; not reproducible: head not published; run 289, see the note below) | 47.44 (+16.1 % [+16.0 %, +16.2 %]) | 76.79 (+20.0 % [+8.5 %, +34.0 %]) | 62.67 (+6.0 % [+0.8 %, +14.8 %]) | 56.16 (+11.9 % [+10.3 %, +14.5 %]) |

The private-drafter rows were measured in run 289 (2026-11-16), a separate paired session on the same build that includes its own stock arm (stock there, tok/s: canonical C1 40.85; agent C1 63.99; agent C8 59.13; chat C1 50.20). Their changes vs stock are that run's own paired ratios, and "vs public drafter" compares the private and the public draft head within that same run; no ratio is taken across runs. The private head is not published, so these rows cannot be reproduced. Run 289's signed receipts are published for its stock and public-drafter launches; its private-drafter launches have none, because a receipt records the hash of every input and those inputs are private.

Where we are not clearly ahead of stock (95 % CI not above zero, or no CI): canonical C1 strict -1.5 % [-1.9 %, -0.8 %]; canonical C1 relaxed -1.1 % [-1.3 %, -0.8 %].

**Train your own drafter.** The [guide]({{ site.baseurl }}/drafter/) gives the data, steps, compute
(about $21 of H100 time for the path that worked) and pitfalls, and the tooling is in the repository.

**What shipped:** a series of gated production promotions, each with its own evidence and rollback. See
[Production]({{ site.baseurl }}/production/).

**What did not:** the [experiment log]({{ site.baseurl }}/experiments/) lists every queue and packet through run 268, each re-checked
in a gains audit. There are far more KILLs than GOs, and every INVALID run is listed with its cause.

## How to read this site

| page | what it holds |
|---|---|
| [Results]({{ site.baseurl }}/results/) | the final public benchmark: every arm, ratio, exactness check and limitation |
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

<small>Numbers filled by fill_release.py from benchmark run 287 (written 2026-11-15 23:56:05 ET).</small>
