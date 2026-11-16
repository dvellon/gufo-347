# Experiment log: summary



The detailed log has one row per queue or packet, through run 268: what was tested, its kind, its true verdict, the key number with its
CI, whether it is in production, what happened next, and the date. It is on the site (https://dvellon.github.io/gufo-347/experiments/), and
it is generated from a gains audit that re-checked every row. This page summarises it.

The detailed log lists every queue and packet through run 268. The 24 runs after it (269 to 289, 2692, 275R, 275R2) are counted in the 347 but are not yet in the detailed log; runs 287 (the final benchmark) and 289 (the private-drafter rows) are reported on the results pages.

## Scale

**347 experiments actually run** (2026-10-26 to 2026-11-16). That count is GPU experiment runs plus offline CPU
simulations, which is where the repository name comes from. The log has 555 rows in all, including
engineering, research and work that was never run. The counts below come from the gains audit of 2026-11-08
(516 rows):

| kind | rows | GO | KILL | REVIEW | INVALID | other |
|---|---:|---:|---:|---:|---:|---:|
| experiment (GPU or cloud run, measured) | 204 | 38 | 64 | 32 | 49 | 21 |
| offline check (CPU replay, simulation, arithmetic) | 100 | 10 | 71 | 8 | 0 | 11 |
| research / design | 12 | 6 | 1 | 2 | 0 | 3 |
| engineering (fixes, tools, promotion packages) | 117 | 0 | 3 | 0 | 10 | 104 |
| built or designed, never measured | 83 | 0 | 0 | 0 | 3 | 80 |

There are many more KILLs than GOs, and every INVALID run is listed with its cause: harness 28, setup 22, knob
check 7, other 5.

## What shipped (promotion history)

| when | package | what changed | gate evidence |
|---|---|---|---|
| 10-30 | first gated composite | strict-tier exactness fixes; relaxed drafting stack; fixed prefill grid for the relaxed tier | promotion gates incl. a power re-run |
| 11-01 | consolidated build | draft-length controller + tool-call stop; small-head drafting with a margin stop admitted to strict; a small-batch projection kernel; an out-of-vocabulary draft list | agent C1 strict, controller alone +11.5 % |
| 11-06 | cache copy fix | removes a rare device fault in the prompt-cache copy path | no-harm gate, exact |
| 11-06 | strict chat, 1 session | chat served as a single stream so the drafter plans for one stream | strict chat decode +7.3 % (paging-adjusted), exact |
| 11-07 | draft-side trio | frequency-permuted draft vocabulary; 2-bit top-2 screen with exact rescoring; smaller draft-head expert weights | agent C1 strict +2.0 % (paging-adjusted), exact |
| 11-08 | scoring + chat settings | a scoring path for multiple-choice options; strict-chat gets the agent draft settings | canonical strict-chat +2.3 %, exact |
| 11-10 | c4 | a checkpoint for short final user turns (fast re-scoring on the same context) | tail latency 1,256 -> 187 ms at 8K, exact |
| 11-10 | c5 | a draft-width rule + deferred state rollback after partial acceptance | combined +1.85 % [1.19, 2.52], exact |

Gains are measured per stage and **do not add up**, because each later change acts on a faster base.

## Notable results since the audit (2026-11-08 .. 11-12)

- **Public benchmark vs stock v0.33.0** (run 2603): ahead on single-session agent, chat and canonical prose, but
  **behind at 8 concurrent agent sessions** on the public workload, which is prefill-bound. That led to the work
  below.
- **Larger exact prefill split:** +5.4 % at 8 sessions, but not self-reproducible under batching, and -2 % at one
  session. KILLED as strict and kept for relaxed only.
- **Capped-turn cache fix:** removes the known one-token cache-cut class and is speed-neutral. One rare near-tie flip
  is still open (see [EXACTNESS.md](EXACTNESS.md)).
- **Settings sweep on the public workload:** a draft-stop setting for strict agents and a higher draft floor for the
  relaxed profile. Both GO; the strict one also passed exactness.
- **Narrow verify kernel v2:** KILLED. It was slower at widths 5-8.
- **Fan-out:** with 8 branches sharing a prefix, aggregate output is ~2.0x one stream, because the shared prefix is
  prefilled once.

## Lessons that shaped the rules

| lesson | rule now |
|---|---|
| Cold-start paging biased A/B toward the second arm | an untimed warm-up per binary, plus a paging audit on every verdict |
| Offline drafter gains shrank 2-5x once measured live | each live line is set from the shrunk estimate |
| Tokens per makespan rewarded longer replies | full-window tok/s at C8; decode tok/s at C1 |
| An inert switch looks like "no effect" | each switch's configuration line must appear in each arm's log |
| Gains on canonical prose did not transfer to agents | agent replay is the deciding stage for agent features |
| "Exact by construction" still needs a device check | exactness is judged on GPU outputs, never on CPU mirrors alone |

## Where the time goes

A strict single-session decode round moves about 10.6 GB of weights in ~57 ms, about 78 % of the measured
238 GB/s memory roofline. Fusing kernel launches cannot pay much: a launch boundary costs about a microsecond. The
levers that were left are the drafter (how many tokens each round accepts), the cost of verify rows, and serving
policy. Most of the log is about those three.

---
Numbers filled by fill_release.py from benchmark run 287 (written 2026-11-15 23:56:05 ET).
