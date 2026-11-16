---
layout: page
title: Methods
permalink: /methods/
---

Most ideas we tested are worth 0.3-3 %. On a shared desktop, run-to-run noise is of the same size, so the measurement
discipline matters more than any single kernel. This page describes the rules every speed verdict follows, the
exactness tiers, and the mistakes that taught us the rules.

## Measurement discipline

**One question per queue, written down first.** Each GPU queue is a script plus a *pre-registration* file, pinned by
hash before any GPU data exists. It fixes the arms, the stages, the metric, the GO and KILL lines, the exactness
rule and a prediction with probabilities for each verdict. The judge is code: it reads the run folder and prints the
verdict. Later changes to the rule need a new, dated amendment and a re-pin; the old verdict stays on record.

**Interleaved launches.** Arms of one stage run interleaved in palindromes: `A B B A` for two arms, `A B C C B A` for
three, `A B C C B A B C A` when three launches per arm are needed. Each launch is a fresh server process. Ratios are
only taken between arms of the same run.

**Untimed warm-ups.** The first timed arm of a stage used to run with a cold page cache (see the paging audit below).
Every stage now starts with an untimed warm-up launch per binary.

**Quiet host before every launch (`wait_quiet_v2`).** Wait (up to 300 s) for one 60-second window in which CPU load
from other processes is low, pages-in per second have a median below 100, no scanner process (search indexer, antivirus,
`find` / `grep`) bursts, and at least 12,000 MB of memory is available. A launch also needs a commit-headroom margin
(108.8 GiB for 8 x 32K sessions + an 8 GiB reserve) and a "timed arm active" marker that tells every other job on the
machine to stay off it.

**Host-load audit after every launch.** A sampler records CPU, GPU, paging and memory once a second. Each launch is
labelled QUIET or LOUD (pages-in p50 > 100/s, or mean > 500/s, or > 10 % of samples > 500/s, or a scanner burst); a
pair of arms with different states, or a > 5x paging ratio, is `REVIEW-HOST`. The verdict is printed with the label
rather than silently re-run.

**Statistics.**

- Headline metric per stage: decode tok/s for one session (sum of completion tokens / sum of decode time), and
  **full-window tok/s** for eight sessions (output tokens per second while all eight streams are in flight). The
  older "output over makespan" metric was dropped because a change that alters reply length biases it.
- A **two-level bootstrap** (resample launches within an arm, then sessions within a launch) gives the 95 % CI of
  every ratio; per-launch t and the A/A spread are reported beside it.
- A/A arms (the same binary twice) measure the noise floor in the same run. Typical A/A spread in the re-profiles: 0.5-1.2 % on
  canonical C1, 1.4-4.4 % on canonical C4 / C8, ~1.4 % on agent C1.
- **Three or more launches per arm when a result lands near its line** (within 0.4 pp, the measured launch-to-launch
  sigma), and a `PRED>2x` flag when a result exceeds twice its pre-registered prediction: such a result goes to
  review, not straight to promotion.

**Knob checks.** Every arm's server log must print the configuration line of each knob the arm sets (and none it
does not). A run that cannot show its knob was active is stopped as INERT or INVALID rather than judged.

**Offline first.** Ideas are first priced with $0 checks on a separate CPU box: bandwidth arithmetic, replays of
recorded draft traces, quantisation-error census. Only ideas that clear an offline kill line get GPU time. Offline
gains shrink 2-5x live, so the live line is set from the offline number, not equal to it.

## Exactness tiers

| tier | contract | how it is checked |
|---|---|---|
| **strict** | greedy speculative output is byte-identical to the same binary's output with speculation off; draft-side changes may only change what is proposed or how fast it is verified | lineage judge on every strict arm (below) |
| **relaxed** | speculation may accept a draft token when the target's top-2 logits are a near-tie (gap <= 0.10), so output can differ from greedy only at near-ties; relaxed prefill uses a fixed grid so outputs are reproducible | relaxed lineage gate: every divergence must start at a near-tie |
| **approximate** | lossy weights or heads (quantisation, pruning), allowed only within a quality budget (KL excess <= 0.05 over the measured floor) and only if >= 3 % faster | offline KL / top-1 agreement / tool-call validity, then live speed; **no approximate profile is approved** |

**The lineage judge.** In multi-turn sessions the server's reply to turn *k* becomes part of the prompt for turn
*k+1*, so one early difference makes every later prompt different. A request is therefore compared only if every
earlier output in its session was identical; the judge reports identical / differ / not compared, and any
lineage-compared difference on a strict arm is `FAIL-EXACT`. At 8 sessions the judge uses two independent
speculation-off references and excludes requests on which the two references disagree with each other. One known,
documented class of difference (the depth at which a prompt-cache checkpoint is restored, which changes with memory
layout) is attributed from server traces rather than counted as a failure.

Upstream gufo's contract is the same as our strict tier: verified speculation, independent of how work is split.

## The paging audit

**2026-11-07.** Queue 192 (a draft-vocabulary change) had booked +1.59 % on agent C1. Re-tested on a newer base, it
showed +0.09 %. The draft-side effect was bit-identical in both runs; what changed was the timing around it. An
audit of **49 comparisons from 33 runs**, in both directions (GO, KILL, NO-GO and REVIEW alike):

- **The first timed arm of a stage paged in 91 % of stages (59 / 65), against 45 % of later arms.** The first arm is
  almost always the base arm (the A of ABBA), so cold-start paging inflated candidate gains across the whole ledger.
  In 18 of 20 paging-imbalanced comparisons the bias favoured the candidate.
- **No KILL flipped.** Most paging-imbalanced kills moved further from their line.
- Promoted or approved wins that were smaller than booked: the draft-vocabulary change +1.59 % -> +0.97 % [0.79, 1.19];
  the OOV static list +0.97 % -> +0.65 % [0.33, 0.96] (still a GO); a C8 batching wait booked +3.1 % became
  indeterminate (its baseline paged at 25,000 pages/s; a clean re-test gave +0.54 %).
- Confirmed clean: the KTc controller (+11.5 %), the H1 gate (+6.7 %), strict chat with one session (+7.3 % at matched
  paging) and the interleaved microbenchmarks (which cannot be biased between arms).
- Paging became the normal state of the machine from 11-02 (LOUD arms 17-38 % of timed arms on 10-31 / 11-01, 57-94 %
  on 11-02 .. 11-06) as available memory fell from ~15 GB to ~11 GB.

Adjustment method, in order of preference: compare only requests where both arms were quiet; drop LOUD arms; a
fixed-effects model with a paging slope (never decisive on its own). The shift is added to the verdict's own number
and CI. Rules adopted: the paging-aware host-load rules, warm-up arms, >= 3 launches near a line, `PRED>2x` review.

## Lessons learned

| lesson | what happened | rule now |
|---|---|---|
| cold-start paging biases ABBA toward the candidate | above | warm-up arm, paging in the host label, audit every speed verdict |
| a launcher can break a run that looks fine | two queues (240, 244) launched the server through Windows PowerShell 5.1 from a PowerShell 7 parent; the mixed module path hid a built-in hash cmdlet and the runs went INVALID | launch only through PowerShell 7; every re-queued old packet is checked for the 5.1 path |
| "exact by construction" needs a device check | a knob-search harness compared GPU kernels with a CPU mirror compiled with `-ffast-math`; one mirror rounded one operation differently and the negative-control bar was miscalibrated (queue 239: INVALID-HARNESS; the valid re-run found no faster exact variant). An earlier audit found that under `-ffast-math` the "round-to-nearest" intrinsics are plain operators | any exactness claim is checked bitwise on the device, not by a CPU mirror |
| offline gains shrink live | every pre-registered live prediction for a new draft head landed low, by 2-5x | live lines are set from the shrunk estimate |
| a metric can reward longer replies | the output/makespan metric rose when a change made replies longer | full-window tok/s for C8; decode tok/s for C1 |
| an inert knob looks like "no effect" | several runs showed 0 % because a knob never armed (wrong path, wrong binary) | knob-check lines in the server log are required per arm; INERT stops the queue |
| a log line printed lazily can stall a smoke test | a smoke test waited for a policy line that the server prints only at the first drafting request | smoke tests send one warm request before checking log lines |
| a quiet-request subset is not a free fix | it can move an estimate for reasons other than paging when the effect depends on the request | quiet-subset numbers are reported as estimates, with the caveat |
| gains on canonical prose do not transfer to agents | a geometry stop rule was +4.5 % on canonical traffic and +0.5 % [-1.2, 2.4] on agent traffic | agent replay is the deciding stage for agent features |

## Where the time goes (why the remaining gains are small)

A strict C1 decode round on this machine moves about 10.6 GB of weights in ~57 ms, about 78 % of the measured
238 GB/s memory roofline, and most of a verify row's cost is routed-expert weights. Kernel fusion was priced from
measured launch stamps (a GPU-side launch boundary costs ~1 microsecond), so fusing launches cannot pay. The levers
that remained were the drafter (how many tokens each round accepts), the verify-row cost, and serving policy, which
is what the experiment log is mostly about.
