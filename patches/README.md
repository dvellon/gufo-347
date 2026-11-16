# Patches against upstream gufo v0.34.0

Base: upstream tag `v0.34.0` (commit `2845939b`). Apply in order:

```
git clone https://github.com/gufo-org/gufo && cd gufo && git checkout v0.34.0
git apply --whitespace=nowarn ../patches/0*.patch
```

Every change is behind a runtime switch (environment variable); with no switch set the build behaves as upstream v0.34.0. The switches used in the benchmark are listed in the top-level README.

| patch | what | switches added | files |
|---|---|---|---:|
| `0001-r7.patch` | rebase: our production changes re-implemented on upstream v0.34.0 | - | 85 |
| `0002-r7q.patch` | ask for upstream #882 draft confidence only when a gate reads it | - | 3 |
| `0003-r7h.patch` | batched draft confidence fix, #882 drop mode, strict live-frontier alignment | `GUFO_U882_BCONF`, `GUFO_U882=drop`, `GUFO_XM_STRICT_ALIGN` | 11 |
| `0004-r7i.patch` | decode HIP graphs on Windows, frontier readback fix | `GUFO_W3_DECODE_GRAPHS`, `GUFO_W3_FB_FIX` | 3 |
| `0005-r7j.patch` | cross-request batched prefill at 8 sessions | `GUFO_C82` | 4 |
| `0006-r7r.patch` | relaxed tier: near-tie acceptance and relaxed live-frontier alignment (off unless set) | `GUFO_NT_ACCEPT_EPS`, `GUFO_XM_RELAXED_ALIGN` | 10 |

The exact LM-head screen (A1) ships with a flat, data-free scaling table (all 1.0). The screen is exact for any table; the table only changes how many candidate rows are re-read.
