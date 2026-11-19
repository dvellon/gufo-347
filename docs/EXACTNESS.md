# Exactness: what "strict" means, and how far it holds



## The claim, in one sentence

In the strict tier, our speculative output is **identical to the same binary's output with speculation off, whenever
the two requests run on the same cache path**. Two known exceptions sit outside that condition. One is a
one-token cache cut after a reply truncated by `max_tokens` (a fix is in test). The other is one rare near-tie flip
that is still under investigation. Both are described below.

## Tiers

| tier | standard? | what it guarantees |
|---|---|---|
| **strict** | yes: lossless greedy speculative decoding | The verified output is the model's greedy output. Draft-side features change only what is proposed and how fast. They never change what is accepted: the vocabulary table, draft-length control and stopping rules, the faster kernels. |
| **relaxed** | **no** | Speculative acceptance also takes a drafted token whose logit is within 0.05 of the top token (a near-tie; the switch refuses values above 0.10), and prefill uses a coarser fixed grid. Output can differ from greedy. Relaxed rows are always labelled and are reported with a quality check. |

## How it is checked

- **Canonical prompts:** byte identity of every completion against the no-speculation reference (R).
- **Agent and chat sessions:** *session lineage*. A turn is compared only while every earlier turn of its session
  matched, so both arms saw the same inputs. After the first difference the session is not comparable.
- **Lineage judge:** for every difference, the judge also asks whether both arms took the same **cache path**. On a
  multi-turn session the server reuses cached prefix state, so it matters whether each arm reused the same amount and
  re-prefilled the same tokens. A difference on the same path is a real break and fails the run. A difference that
  follows a different cache history is attributed to that history and listed by cause. The speculation flag of each
  arm is passed explicitly.

## Results in run 294 (R8 on upstream v0.35.0)

Run 294 gated R8's strict tier before any release claim (pre-registered: R8 is a candidate only if every check below
passes). It passed every one.

- Canonical prose: 10 of 10 completions identical to the no-speculation reference (completion sha per prompt).
- Agent C1: 93 of 93 requests on the same cache path as the reference, 0 differ (PASS). The reference is the majority
  no-speculation output of R8's own reference launches and four earlier no-speculation runs of the same cells on our
  v0.34.0 builds (runs 287, 275 and 284). That pooling is allowed only if R8's own references agree with the earlier
  majority (pre-registered), and they did: 0 differing turns, so R8's no-speculation output equals our v0.34.0 builds'
  on this workload.
- Chat: 23 of 24 requests on the same cache path as R8's own no-speculation reference, 0 differ (PASS); the one
  remaining request follows a reply cut by `max_tokens` and is excluded by cause (K1, below).
- Eight concurrent sessions (report only, see below): 95 of 99 comparable requests identical to the pooled
  single-session references.

| check (lineage judge, speculation flag explicit) | paired | compared | differing on the same cache path | unattributed breaks | excluded by cause | verdict |
|---|---:|---:|---:|---:|---|---|
| E1 c1 R8 strict vs the majority no-speculation reference | 93 | 93 | 0 | 0 | K1 0, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 c1 previous build (R7R) vs the same reference (control) | 93 | 93 | 0 | 0 | K1 0, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 chat R8 strict vs R8's own no-speculation reference | 24 | 23 | 0 | 0 | K1 1, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 chat previous build (R7R) vs R8's no-speculation reference (control) | 24 | 23 | 0 | 0 | K1 1, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 canon R8 strict vs R8's no-speculation reference (completion sha) | 10 | 10 | 0 | - | - | PASS |
| E1 canon previous build (R7R) vs the same reference (control) | 10 | 10 | 0 | - | - | PASS |

## Previous release: results in run 287 (R7R on upstream v0.34.0)

- Canonical prose: 20 of 20 completions identical to the no-speculation
  reference.
- Agent and chat (single session): Our build against its own no-speculation references (lineage judge, speculation flag explicit): agent C1: 185 of 185 requests on the same cache path as the reference, 0 differ (PASS); chat: 43 of 48 requests on the same cache path as the reference, 0 differ (PASS); canonical prompts: 20 of 20 completions identical (PASS). Requests whose cache path differs from the reference are excluded by cause in the table above.

| check (lineage judge, speculation flag explicit) | paired | compared | differing on the same cache path | unattributed breaks | excluded by cause | verdict |
|---|---:|---:|---:|---:|---|---|
| E1 c1 ours vs ours' no-speculation reference (own references, lineage judge) | 185 | 185 | 0 | 0 | K1 0, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 c1 previous build vs R (268 R, cross-check) | 185 | 183 | 0 | 0 | K1 2, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 chat ours vs ours' no-speculation reference (own references, lineage judge) | 48 | 43 | 0 | 0 | K1 5, K2 0, K3 0, UNATTRIBUTED 0 | PASS |
| E1 chat previous build vs R (268 R, cross-check) | 48 | 42 | 0 | 0 | K1 6, K2 0, K3 0, UNATTRIBUTED 0 | PASS |

## The known cache-cut class (K1)

When a reply hits `max_tokens`, the speculative path has already verified the last generated token at width 1. It
reuses the cache up to that token on the next turn. The no-speculation path discards that last pending token, so its
cache ends one token earlier and the next turn prefills that token instead. The two paths then hold the same text, but
one token's state was written by decode in one arm and by prefill in the other. Later turns can differ slightly as a
result. This is a cache-history difference, not a verify error, but it means plain byte identity does **not** hold
after a truncated turn.

**Fix status:** A fix is built and tested. When a reply is cut by max_tokens, the fix feeds the final pending token with a one-row decode, so the speculative and non-speculative paths keep the same cache. In a re-test it removed 6 of the 7 differences and was speed-neutral (+0.6 % / -0.6 %, both inside noise). It is not in the binaries benchmarked here (neither R7R in run 287 nor R8 in run 294), and it is not in patches/.

## The open near-tie flip

The re-test also showed one difference on the same cache path: one chat request picked "list" instead of "string" at a near-tie, about 590 characters into its reasoning. It did not happen again in 6 targeted re-runs (0 / 6), or in the 5 earlier runs of the same request. We think it is rare run-to-run GPU nondeterminism in the speculative verify path. It is still under investigation, and we do not claim bit-exactness until it is explained.

## Eight concurrent sessions

With 8 sessions in flight, outputs are compared with single-session references and reported only. Under batching, a
request can restore a shared prefix that another session cached. The restore depth depends on timing (for example
1,017 vs 1,029 tokens), and prefill is not exactly invariant to where it is cut. So a later turn can differ, even
between two runs of the same binary. A fix that cuts and restores prefill only at canonical positions (never at a boundary set by another session) is built and in test. It trades a little cross-session prefix reuse for reproducibility.

## Stock's speculation

Stock's speculative decoding is not exact on this machine. On the same cache path, its output differs from its own --speculative off output, because its verify numerics differ from its non-speculative path. That is a legitimate design choice, but it means stock's speculative speed-up and ours are not the same kind of claim. Runs 294 and 287 had no stock `--speculative off` arm, so they do not repeat this
comparison for stock v0.35.0 or v0.34.0. In run 2603, an earlier full benchmark against stock v0.33.0, 12 of 20 canonical prompts differed from
stock's own `--speculative off` output, and 12 of 31 lineage-compared agent C1 requests differed (19 identical; 62
further requests were not comparable after an earlier difference in their session).

## What we do and do not claim

- We claim strict-tier speed-ups **at identical output on the same cache path**, checked on every benchmark run.
- We do not claim bit-identical output across different cache histories, across concurrency levels, or across
  different binaries. Our no-speculation output also differs from stock's no-speculation output in places, because
  the kernels and accumulation orders differ.
- Relaxed numbers are never presented as lossless.

---
Numbers from benchmark run 294 (verdict written 2026-11-17 18:31:16 ET); previous-release numbers from run 287 (written 2026-11-15 23:56:05 ET).
