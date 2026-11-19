## The cost of speculation

Tokens per second alone hides what speculation costs. Per round, the drafter proposes a few tokens and the target model checks them all in one pass; rejected drafts are discarded work.

### Release build R8 (run 294, public assets)

| workload | engine | launches | drafted tokens / round | accepted / round | acceptance | discarded drafts / round | tokens per target pass | TTFT warm p50 / p90 ms | TTFT cold p50 ms | run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| canonical pp2048/tg128, 1 stream | stock v0.35.0 | 2 | 1.55 | 0.83 | 0.534 | 0.72 | 1.83 | 1154 / - | - | 294 |
| canonical pp2048/tg128, 1 stream | ours strict (R8) | 2 | 2.80 | 1.12 | 0.401 | 1.67 | 2.12 | 1147 / - | - | 294 |
| canonical pp2048/tg128, 1 stream | our previous build (R7R; reference) | 2 | 2.80 | 1.12 | 0.401 | 1.67 | 2.12 | 1145 / - | - | 294 |
| coding agent, 1 at a time | stock v0.35.0 | 4 | 3.74 | 3.03 | 0.809 | 0.71 | 3.92 | 368 / 2008 | 963 | 294 |
| coding agent, 1 at a time | ours strict (R8) | 4 | 4.33 | 3.43 | 0.791 | 0.91 | 4.43 | 372 / 2015 | 966 | 294 |
| coding agent, 1 at a time | our previous build (R7R; reference) | 4 | 4.33 | 3.43 | 0.791 | 0.91 | 4.43 | 382 / 1998 | 959 | 294 |
| coding agent, 8 at once | stock v0.35.0 | 4 | 2.46 | 2.25 | 0.916 | 0.21 | 2.87 | 1985 / 6916 | 2439 | 294 |
| coding agent, 8 at once | ours strict (R8) | 4 | 2.58 | 2.25 | 0.872 | 0.33 | 3.09 | 1804 / 5823 | 2130 | 294 |
| coding agent, 8 at once | our previous build (R7R; reference) | 4 | 2.61 | 2.29 | 0.875 | 0.33 | 3.13 | 1791 / 5677 | 2222 | 294 |
| MT-Bench chat, 1 stream | stock v0.35.0 | 2 | 2.28 | 1.62 | 0.710 | 0.66 | 2.50 | 170 / 188 | 253 | 294 |
| MT-Bench chat, 1 stream | ours strict (R8) | 2 | 2.73 | 1.77 | 0.647 | 0.96 | 2.77 | 166 / 179 | 260 | 294 |
| MT-Bench chat, 1 stream | our previous build (R7R; reference) | 2 | 2.73 | 1.77 | 0.647 | 0.96 | 2.77 | 172 / 185 | 255 | 294 |

Run-294 values are plain means over the launches, computed from the signed launch receipts. The cost model below
was fitted in run 287 on the previous build and was not refitted for R8.

## Previous release: run 287 (R7R on upstream v0.34.0)

Every number below comes from benchmark run 287.

### Public assets

| workload | engine | launches | drafted tokens / round | accepted / round | acceptance | discarded drafts / round | tokens per target pass | TTFT warm p50 / p90 ms | TTFT cold p50 ms | run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| canonical pp2048/tg128, 1 stream | stock v0.34.0 | 4 | 1.83 | 1.03 | 0.564 | 0.80 | 2.03 | 1167 / - | - | 287 |
| canonical pp2048/tg128, 1 stream | ours strict | 4 | 2.80 | 1.12 | 0.401 | 1.67 | 2.12 | 1156 / - | - | 287 |
| canonical pp2048/tg128, 1 stream | our previous build (reference) | 4 | 2.38 | 1.11 | 0.467 | 1.27 | 2.11 | 1146 / - | - | 287 |
| coding agent, 1 at a time | stock v0.34.0 | 8 | 3.74 | 3.02 | 0.808 | 0.72 | 3.96 | 387 / 1824 | 998 | 287 |
| coding agent, 1 at a time | ours strict | 8 | 4.32 | 3.35 | 0.776 | 0.97 | 4.35 | 385 / 1787 | 983 | 287 |
| coding agent, 1 at a time | our previous build (reference) | 8 | 4.11 | 3.14 | 0.764 | 0.97 | 4.14 | 467 / 2110 | 1018 | 287 |
| coding agent, 8 at once | stock v0.34.0 | 6 | 2.49 | 2.27 | 0.912 | 0.22 | 2.92 | 2038 / 6891 | 2614 | 287 |
| coding agent, 8 at once | ours strict | 6 | 2.59 | 2.27 | 0.875 | 0.32 | 3.11 | 1814 / 6088 | 2278 | 287 |
| coding agent, 8 at once | our previous build (reference) | 6 | 2.88 | 2.40 | 0.832 | 0.48 | 3.27 | 1539 / 6161 | 1308 | 287 |
| MT-Bench chat, 1 stream | stock v0.34.0 | 4 | 2.35 | 1.67 | 0.711 | 0.68 | 2.55 | 174 / 191 | 256 | 287 |
| MT-Bench chat, 1 stream | ours strict | 4 | 2.79 | 1.81 | 0.649 | 0.98 | 2.81 | 161 / 177 | 256 | 287 |
| MT-Bench chat, 1 stream | our previous build (reference) | 4 | 2.83 | 1.84 | 0.649 | 0.99 | 2.84 | 163 / 184 | 249 | 287 |

- drafted / round = accepted per round / acceptance; discarded = drafted - accepted.
- tokens per target pass = accepted drafts + the one token every verify pass yields anyway.

### When does another draft row pay? (1 agent, this machine)

- Cost model fitted on this build from run 287 (185 agent C1 requests; per request, round time = decode time / draft rounds): a round costs 35.9 ms plus 5.49 ms per drafted row (95 % CI 5.16 - 5.85), at the median prompt of 6.1K tokens (+0.04 ms per extra 1K of prompt).
- Strict, 1 agent: about 4.3 drafted rows per round, ~60 ms per round, 4.35 tokens per round = ~73 tokens/s predicted (measured client decode: 72 tokens/s).
- One more draft row pays only if its acceptance chance is above ~40%.

### Not measured

- Watts per useful token (no power meter on this run).
- TTFT at 32K / 128K prompts: no long-prompt sweep was run.

