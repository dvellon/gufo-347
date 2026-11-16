---
layout: page
title: Benchmarks
permalink: /benchmarks/
---

Every table on this page names its machine, OS, date, binaries and caveats. Ratios are only quoted between arms that
ran **interleaved in the same run**; numbers from different runs are never divided by each other.

Common to all rows: AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GiB unified memory, Windows 11 Pro,
Qwen3.8-Flash-Next UD-Q4_K_XL (Unsloth revision `38bb39ee`), greedy decoding unless stated, `--context 32768`.

- [Public quick look: stock upstream vs ours](#public-quick-look)
- [Matched benchmark against a published Strix Halo engine study](#matched-benchmark)
- [Production re-profiles over time](#production-re-profiles)
- [Upstream's own published numbers](#upstreams-published-numbers)

## Public quick look

**Run 2026-11-08, 13:11-14:24 ET.** A first look before a full public benchmark, not the benchmark itself.

| arm | binary | MTP draft head | draft-vocabulary table | role |
|---|---|---|---|---|
| **A** stock | upstream main **`17d6fa25`** (v0.31.0-36), built with upstream's Windows recipe, TheRock 10.0.0 | public shared Q8_0 head (Unsloth) | none (upstream default) | reference |
| **B** ours, public assets | production `gufo.exe` **`4d787f38`** (profiles `0ea5e314`) | the same public head | rebuilt from a public corpus (SWE-smith tool shard 00002, K = 6,931) | the reproducible claim |
| **C** ours, production | the same binary as adopted | private fine-tuned head | private table | numbers only |
| **R** reference | production with speculation off | - | - | exactness reference for B and C |

![Quick look tok/s]({{ site.baseurl }}/assets/quicklook.svg)

| stage | A stock | B ours-public | C production | B/A | C/A | C/B |
|---|---:|---:|---:|---:|---:|---:|
| canonical pp2048/tg128 C1 (upstream model-bench recipe, prose), server tg tok/s | 36.42 | 40.45 | 44.91 | 1.111 [1.102, 1.119] | 1.233 [1.225, 1.240] | 1.110 [1.097, 1.123] |
| agent C1 (SWE-smith shard 00004 replay), client decode tok/s | 58.35 | 74.84 | 79.01 | 1.282 [1.273, 1.292] | 1.354 [1.343, 1.365] | 1.056 [1.048, 1.064] |
| agent C8 full-window tok/s | 55.42 | - | - | - | - | - |
| MT-Bench chat C1, client decode tok/s | 46.82 | 57.89 | 60.58 | 1.236 [1.227, 1.246] | 1.294 [1.285, 1.303] | 1.046 [1.036, 1.057] |

Brackets are the **cross-launch range** (two launches per arm), not a confidence interval.

Mechanism (per launch, mean): draft acceptance on agent C1 was 0.77 (stock), 0.81 (B), 0.86 (C); tokens per target
pass 4.31 / 4.69 / 4.87. TTFT p50 was the same across arms on C1 and chat (155-609 ms); on C8 our arms had lower TTFT
p50 (1.26-1.80 s vs 2.18-2.59 s for stock). The speculation-off reference ran at about 29.5-30.0 tok/s on every stage.

**Strict exactness.** B and C against the speculation-off reference R: canonical 5/5 identical, agent C1 15/15, chat
8/8 on every launch; C8 compared 9 of 24 requests (report only) and all 9 were identical. Stock A is not checked
(upstream's contract is that greedy MTP equals AR).

**Caveats.**

- **n = 2 launches per arm**, ~15 / 24 / 8 requests per stage: no confidence intervals, no GO / KILL.
- Host paging: 10 of the 30 launches were flagged LOUD by our host-load monitor (6 stock, 2 ours, 2 untimed
  warm-ups; see [the paging finding]({{ site.baseurl }}/methods/#the-paging-audit)). On the canonical stage the two
  stock launches agree within 0.1 tok/s whether LOUD or quiet.
- **C8 is missing for B and C** (the full-window metric did not compute for our arms); it must be fixed before a full run.
- B's binary is production's: it contains a small calibration table derived from private data, so B is reproducible
  only after that table is regenerated from public data (see [Reproduce]({{ site.baseurl }}/reproduce/)).
- Upstream treats Windows as a development target; its production packages come from Linux. Stock here is upstream's
  **Windows** path.
- The SWE-smith shard uses SWE-agent text function calling (no OpenAI tool schema), so the agent stage does not take
  the constrained tool-grammar path. These are performance workloads, not accuracy benchmarks.
- All arms use the same local GGUF (revision `38bb39ee`). Upstream now recommends Unsloth's "UD 3.1" revision; a
  sampled byte comparison found its UD-Q4_K_XL tensors identical to `38bb39ee` (re-split shards), so no separate row.

## Matched benchmark

**Run 2026-11-08 ~05:30 ET, production `56052e5e` (the build before `c3`; same decode path).** A published Strix
Halo engine study  reported MTP decode rates 1.3-1.6x what our internal agent
numbers showed. We rebuilt their three prompt classes and ran ours on them. **Report only.**

| tier | prompt | prompt tokens | decode tok/s, median [range] (n = 3) | tokens per round | acceptance | their stock / distilled / best | parity line |
|---|---|---:|---:|---:|---:|---|---:|
| strict | code-1k | 1,054 | **95.07** [95.07, 95.23] | 6.38 | 0.931 | 53.7 / 57.3 / 62.44 | 48.3 |
| strict | prose-1k | 1,000 | **48.72** [48.65, 48.83] | 2.35 | 0.569 | 39.7 / 45.1 / 51.56 | 35.7 |
| strict | code-8k | 7,938 | **89.79** [89.70, 90.01] | 6.07 | 0.926 | - / - / 54.0 | 48.6 |
| strict | their public kit-code prompt | 42 | 99.12 [98.93, 99.16] | 6.89 | 0.936 | - | - |
| strict | their public kit-prose prompt | 48 | 47.76 [47.64, 47.80] | 2.30 | 0.537 | - | - |
| relaxed | code-1k | 1,054 | 92.36 [92.22, 92.37] | 6.22 | 0.895 | 53.7 / 57.3 / 62.44 | 48.3 |
| relaxed | prose-1k | 1,000 | 48.22 [48.16, 48.26] | 2.39 | 0.527 | 39.7 / 45.1 / 51.56 | 35.7 |
| relaxed | code-8k | 7,938 | 88.38 [88.27, 88.46] | 6.07 | 0.902 | - / - / 54.0 | 48.6 |

- Pre-registered reading: PARITY if our median is at or above the line; **PARITY on every row**, both tiers.
- Speculation-off decode on the same prompts: ~29.5-30.4 tok/s. Strict output identical to it on every prompt;
  relaxed completions identical to strict on all five prompts.
- decode tok/s = completion tokens / server-reported decode time; tokens per round = (completion - 1) /
  (completion - 1 - accepted).
- **Caveats:** their 1K / 8K prompts are unpublished, so ours are reconstructions; their numbers are from their
  hardware and OS, not re-run here. Read it as "no engine gap", not as a speed ratio. The finding: the earlier gap
  was workload and draft head (code drafts at ~6.4 tokens per round), not the engine round itself.

## Production re-profiles

Two full re-profiles of the production stack, on the same harness: **queue 179** (2026-11-01, consolidated build
`9a9fe3a6`) and **queue 237** (2026-11-07, combined build `56052e5e`). Report only; headline rows have 3-4 launches
with a launch-level bootstrap CI.

![Re-profile history]({{ site.baseurl }}/assets/history.svg)

| row | 11-01 (`9a9fe3a6`) | 11-07 (`56052e5e`) [95 % CI] | change | speculation off (same binary) | host state 11-07 |
|---|---:|---:|---:|---:|---|
| canonical C1 strict | 37.27 | **37.93** [37.86, 37.98] | +1.8 % | 23.35 | quiet |
| canonical C4 strict | 50.33 | 51.30 [50.79, 51.82] | +1.9 % | 43.18 | paging on 3/5 arms |
| canonical C8 strict | 56.35 | 56.24 [55.79, 56.80] | -0.2 % | 51.80 | quiet |
| canonical C1 relaxed | 36.38 | 36.99 [36.80, 37.11] | +1.7 % | 22.94 | quiet |
| canonical C8 relaxed | 56.33 | 56.12 [55.65, 56.59] | -0.4 % | 49.93 | quiet |
| agent C1 strict (internal agent replay, not published) | 42.97 | 43.83 [41.52, 45.64] | +2.0 % | - | paging on 3/3 arms |
| agent C8 relaxed (internal agent replay, not published; per-stream) | 11.09 | 11.33 [9.71, 12.74] | +2.2 % | - | paging on 1/3 arms |
| chat C1 strict (internal chat replay, not published) | 42.33 | 45.79 | +8.2 % | - | paging; profile changed (see below) |
| chat C1 relaxed (internal chat replay, not published) | 44.15 | 47.41 | +7.4 % | - | paging |

- The canonical rows here use our internal pp2048 harness (fixed prompts, server-reported decode), not upstream's
  model-bench recipe, so they are **not comparable** with the quick-look canonical row.
- The internal agent and chat replays are recorded sessions from our own use; they are not published and only
  aggregate numbers are shown. Agent C8 is per stream (8 streams in flight).
- "chat C1 strict" on 11-07 uses the strict-chat profile (one session, adopted 11-07), on 11-01 the strict-agent
  profile; the +8.2 % is mostly that profile change.
- 11-01 ran under host-load monitor v1 (no paging gate), 11-07 under v2 (paging-aware). The paging columns matter
  when reading a 2 % change.
- Long context (11-01, report only): at 128K tokens of context, strict C1 decode was ~28.9 tok/s without speculation
  and ~56 with relaxed speculation; cold prefill fell from ~1,770 tok/s at 32K to ~1,520 tok/s at 240K.
- Also from the 11-01 profile, relaxed-tier serving defaults (fixed prefill grid, `scope=all`) measured **+7.1 %**
  agent C8 full-window throughput over production cuts; see [Production]({{ site.baseurl }}/production/#server-profiles).

## Upstream's published numbers

Upstream's README lists, for this model, up to 116.93 tok/s single-user and 245.81 aggregate tok/s at 8 users with
MTP. Those are upstream's own Linux measurements on their prompts; we did not reproduce them and do not compare our
Windows numbers to them. A 2026-11-05 attempt to build then-current upstream main with its Windows recipe failed to
build; the 11-08 quick look is the first stock Windows comparison.
