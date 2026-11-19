# Patches against upstream gufo v0.35.0 (build R8)

Base: upstream tag `v0.35.0` (commit `0260e116`). Apply in order:

```
git clone https://github.com/gufo-org/gufo && cd gufo && git checkout v0.35.0
git apply --whitespace=nowarn ../patches/0*.patch
```

Afterwards `git add -A && git write-tree` prints `9510077d10e03a6bc88eca1b90ff89a0bdc546da`. That tree equals the
benchmarked build's source (`v0.35.0-6-gaa982395`, tree `062d2a1e`), except for 143 comment lines that name internal
folders and are rewritten here; no code line differs.

Every change is behind a runtime switch (environment variable); with no switch set the build behaves as upstream
v0.35.0. The switches used in the benchmark are listed in the top-level README.

| patch | what | switches added | files |
|---|---|---|---:|
| `0001-r8-rebase.patch` | rebase: our production changes (the R7R stack of the previous release) on upstream v0.35.0 | every switch of the previous series, including `GUFO_U882_BCONF`, `GUFO_U882=drop`, `GUFO_XM_STRICT_ALIGN`, `GUFO_W3_DECODE_GRAPHS`, `GUFO_W3_FB_FIX`, `GUFO_C82`, `GUFO_NT_ACCEPT_EPS`, `GUFO_XM_RELAXED_ALIGN` | 90 |
| `0002-r8-q4k-draft-head.patch` | port upstream #1209 (Q4_K copy of the MTP draft-head prefix) to our permuted draft head | `GUFO_DRAFT_Q4K=0` (kill switch: drafting as on v0.34.0) | 3 |
| `0003-r8-xa-retained-test.patch` | test: relaxed live-frontier alignment stays within its exactness level for #1183's retained frontiers | - | 1 |
| `0004-r8-c82-resume.patch` | cross-request batched prefill: resume a ready capture before every peer of the grouped step (#1181) | - | 1 |
| `0005-r8-layered-prefill.patch` | `GUFO_LAYERED_PREFILL=0` becomes a one-layer-group profile (#1175 makes a scheduling profile mandatory) | - | 1 |
| `0006-r8-identity.patch` | the startup identity line names the upstream base (v0.35.0) | - | 1 |

The series is one patch per commit of our rebase branch. The rebase itself (0001) is a single patch because the
previous layers (R7, R7Q, R7H, R7I, R7J, R7R) conflict with upstream v0.35.0 in the same files. The previous release's
per-layer series against v0.34.0 (`0001-r7.patch` ... `0006-r7r.patch`) is in this repository's git history.

`GUFO_DRAFT_Q4K` is unset in every benchmarked launch and in every profile, so upstream's #1209 Q4_K draft-head copy is
active. `GUFO_DRAFT_Q4K=0` restores the previous drafting path; that setting was not benchmarked.

The exact LM-head screen (A1) ships with a flat, data-free scaling table (all 1.0). The screen is exact for any table;
the table only changes how many candidate rows are re-read.

`patches.json` lists, for each patch, its source commit, title, files, line counts, sha256 and the number of
comment lines rewritten.
