---
layout: page
title: Open items
permalink: /open-items/
---

<p class="caveat"><b>Status (2026-11-08):</b> tests on these items are in progress. Results will be added to the
<a href="{{ site.baseurl }}/experiments/">experiment log</a> as they land, and this list will be updated.</p>

After two weeks and about 500 logged queues and packets, we audited every row of the
[experiment log]({{ site.baseurl }}/experiments/) with one question: did we keep every performance gain we measured?

**No large measured win was left behind.** Every clean GO is in production, was displaced by something that is in
production, or was killed by a later valid test. What is left is small to medium and mostly untested or unproven:

- features proven on one server profile but never applied to another;
- features compiled into the production build but switched off;
- one booked win that turned out to be inert in the current build (B1 below);
- INVALID runs that were never re-run.

If everything below pans out, the realistic total is roughly **+2-4 % agent C1** and **+3-6 % agent C8 (relaxed
tier)**, plus some task-throughput levers in the client proxy. Estimates are ours and unmeasured unless the evidence
column says otherwise. Queue numbers link to rows in the experiment log.

## A. Profile and setting changes (no new build; cheapest)

| # | item | evidence | est. gain | tier | needs |
|---|---|---|---|---|---|
| A1 | The mixed chat + agent profile has none of the combined build's three drafting changes (vocabulary permutation, 2-bit screen, 4-bit drafter experts) | the same set gave the relaxed agent profile +3.33 % at C8 (233c); on the mixed profile only chat was measured (+0.14 %, 241) | +1..3 % on agent traffic served by the mixed profile | relaxed | an agent C1 / C8 A/B on the mixed profile |
| A2 | Strict agent profile: prefill chunk grid 4096 (the relaxed profile already uses 4096) | 138: +1.99 % at C8; 184 P1: +4.13 % (one pair); 180: +7.98 % service time; turns per minute unproven | +2..4 % per token, strict C8 | strict, starts a new reference lineage | a policy decision, then a powered C8 re-run on the current build |
| A3 | Relaxed tier: suffix drafter, suffix batching and a narrower draft-vocabulary window (compiled in, off) | 006r +2.4 % (unclean run); CPU sim +1.35 points; window sim +0.66 % on chat | +0.5..1.4 % C1, +0.5..1.1 % C8 | relaxed | one 4-arm A/B |
| A4 | The verify-batch wait setting is missing from the relaxed profiles | 218: C8 +1.29 % [-0.69, 3.84]; +0.54 % at matched paging | ~+0.5 % relaxed C8 | relaxed | a single-setting A/B |
| A5 | The K1 controller prior on the mixed profile | 194r: chat +1.2 % (at the A/A noise level) | ~+1 % relaxed chat | relaxed | a re-run |
| A6 | Attention select-skip (compiled in, off) | 095b: GO-STRICT on its own; the bundle it shipped in measured -1.16 % at agent C1 (178b) | 0..+0.4 % | strict | a single-setting powered pair |

## B. Re-base onto the current build and run (new builds)

| # | item | evidence | est. gain | tier | needs |
|---|---|---|---|---|---|
| B1 | **The A1 draft screen never fires in the current build.** Its settings are on, but under the KTc controller the planned width is always too wide for it to arm (no A1 summary line in 118 logs). Queue 191 (A1 on the actual chain width) fixes this | three audits agree; arithmetic | +0.9..1.4 % agent C1 wall time | strict | re-base 191, about 105 GPU-minutes |
| B2 | 206 kernel pair (AF#5 + AF#6, plus a sampled-chat variant) | INVALID only because the canary compared 87 requests, under the 90 its rule needs; 0 differences | +0.35..0.9 % C1 / C8; sampled-chat variant +0.7..2.2 % | strict | re-base and re-run |
| B3 | 219b: three upstream pulls (#742 selector unroll, #881, #903); production still shows register spills in the selector | parked for a re-base | C1 up to +0.5..1 %, C8 up to +0.8..1 %, cold 32K prefill +1.3 % | strict | re-base and gate |
| B4 | 092 tokenizer prefix cache | token ids identical in both orders; time to first token -1.8..-2.3 %; prep 22 -> 4.5 ms at 12K | +0.2..0.3 % throughput, TTFT -2 % | strict | re-base a 2-file change and a short ABBA |
| B5 | 224 kernel fusions | +0.36 % [0.18, 0.61], identity PASS, but the two halves disagree | ~+0.3 % | strict | re-base and re-run |
| B6 | 235 trunk LM head at Q6_K (sim-071) | offline KL 0.0003, projected +1.86 %; live run INVALID (noisy host) | +0.5..1.1 % relaxed C1 | approximate | a ~2 GPU-hour re-run on a quiet host |
| B7 | Batch the BF16 indexer k / q projections across sessions in a C8 round | never run | +0.5..1 % C8 | strict | check it is not already done, then build |
| B8 | 044: move the draft-stop decision onto the device | 238: the host part is 4.6 % of a C1 round | +0.6..1.0 % C1 | strict | about 4-6 days of build work |
| B9 | 187: expert-affinity cohorting of sessions | route overlap 0.91 (the 0.93 kill line was not hit) | +1..3 % C8 | strict | a scheduler change on the backlog |

## C. Workload and client levers (task throughput, not tok/s)

- **C1. Effort budgets** (211c, sim-064). At low reasoning effort, docs tasks could go from 121.7 to 401.9 passing
  tasks per hour if a judge pass approves the outputs; structured and research tasks +20-30 % passing tasks per hour.
  Needs a judge pass on stored outputs ($0) and a change to the proxy's effort table.
- **C2. Copy-following prompt lookup.** An outside engine reports +20..40 % on file-write turns. Needs a $0 replay on
  our file-write turns.
- **C3. Server warm-up request at start, and pre-reading the n-gram memory table when a request arrives.** Exact and
  cheap.
- **C4. Proxy loop guard** (staged, not applied): insurance for long unattended runs.
- **C5. Two client-side items** (a minimal tool list, split scripts): their quality arms never ran.

## Plan

1. **Round 1, profile A/Bs** (no builds, about 5-6 GPU-hours): A1, A3 + A4 together, A5. A2 only after the policy
   decision.
2. **Round 2, one combined candidate build:** B1 + B2 + B3 + B4 + B5, each behind its own setting, gated together
   the way the current production build was (one combined gate plus a per-member A/B).
3. **$0 checks now:** C1 judge pass, C2 replay, B7 source check, C3 design.

Items blocked by policy rather than by a speed result (a disk cache tier, which costs too much SSD wear; one change
shipped in the relaxed tier only; a drafter model swap whose quality was never checked) are not listed.
