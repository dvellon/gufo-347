# Methodology



Most of the ideas we tested were worth 0.3-3 %. On a shared desktop, run-to-run noise is about the same size, so how
we measured mattered more than any single kernel. Every speed verdict in this repository follows the rules below.

## 1. Pre-registration

- Each test is a script plus a **pre-registration file**, pinned by hash before any GPU data exists.
- The file fixes the arms, the stages, the headline metric, the GO and KILL lines, and the exactness rule. It also
  gives a prediction with a probability for each verdict.
- **The judge is code.** It reads the run folder and prints the verdict. Changing a rule after the run needs a dated
  amendment and a re-pin, and the old verdict stays on record.
- A result more than twice its prediction is flagged and goes to review, not straight to promotion.

## 2. Interleaved A/B with fresh servers

- The arms of one stage run interleaved (palindromes such as `A B C C B A`, or rotated orders for four arms). Each
  launch is a fresh server process.
- Every stage starts with an **untimed warm-up** launch per binary, because the first timed arm used to run on a cold
  page cache.
- In the public benchmark, each round gives every arm the **same session subset**, and the subsets rotate between
  rounds. Most of the noise comes from launch to launch, so many short, paired launches beat a few long ones.
- Ratios are only taken between arms of the **same run**. We never divide numbers from different runs.

## 3. Quiet host

- **Before** every launch, the runner waits for a 60-second quiet window. Other processes must use little CPU,
  pages-in must stay low, no scanner can burst, and enough memory must be free.
- **During** every launch, a sampler records CPU, GPU, paging and memory once a second. The launch is then labelled
  QUIET or LOUD.
- A comparison whose arms ran under different host states is labelled `REVIEW-HOST`. It is reported with that label,
  never silently re-run.
- A paging audit of 49 earlier comparisons showed that host paging alone could move a result by up to ~1.5 pp. That
  audit is the reason for these rules.

## 4. Statistics

- **Headline metrics:**
  - one session: decode tok/s, i.e. the sum of completion tokens over the sum of decode time;
  - eight sessions: **full-window tok/s**, i.e. output tokens per second while all eight streams are in flight.
- **CIs:** each ratio gets a 95 % paired bootstrap CI over rounds; our internal tests use a two-level bootstrap over
  launches, then sessions.
- **Noise floor:** A/A arms (the same binary twice) measure the noise floor within the same run.
- **Verdicts:** GO needs the point estimate at or above the line, the CI-low above zero, and exactness PASS.

## 5. Exactness on every run

Every strict arm is compared with the same binary run with speculation off. Canonical prompts use byte identity.
Agent and chat sessions use session lineage and the cache-path judge. See [EXACTNESS.md](EXACTNESS.md).

## 6. Knob checks

Every arm's server log must print the configuration line of each runtime switch the arm sets, and none that it does
not set. A run that cannot show its switch was active is stopped as INERT or INVALID, not judged.

## 7. Receipts

The final benchmark writes a **signed receipt** per launch and per result row:

- input hashes: binary, model shards, draft head, vocabulary table, workload, every request body and the client;
- hardware and driver;
- per-request output hashes and timings;
- the metrics recomputed from those timings.

The signatures use `ssh-keygen -Y sign`. To check a receipt, run `receipts/verify_receipt.py` against the public key
in `receipts/keys/`. It checks the signature, the internal consistency and the input hashes. If you have the same
inputs, it can also re-run the launch and tell you whether your speed falls inside the stated band.

Run 287's receipts (101 launches, 15 result rows, a signed index) are in [receipts/runs/287/](../receipts/runs/287/).

Run 289's receipts (31 launches, 9 result rows, a signed index) are in [receipts/runs/289/](../receipts/runs/289/). They cover that run's stock and public-drafter launches. Its private-drafter launches (the private rows of the headline table) have no receipts: a receipt records the hash of every input, and those inputs are private.

## 8. Offline first

Ideas were first priced with $0 checks on a separate CPU machine: bandwidth arithmetic, replays of recorded draft
traces and quantisation-error counts. Only ideas that cleared an offline kill line got GPU time. Offline gains
shrank 2-5x once measured live, so each live line was set from the offline number, not equal to it.

---
Numbers filled by fill_release.py from benchmark run 287 (written 2026-11-15 23:56:05 ET).
