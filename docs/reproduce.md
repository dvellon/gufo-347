---
layout: page
title: Hardware notes and how to reproduce
permalink: /reproduce/
---

## Hardware and OS

| item | value |
|---|---|
| APU | AMD Ryzen AI Max+ 395, Radeon 8060S integrated GPU (`gfx1151`, RDNA 3.5) |
| memory | 128 GiB LPDDR5X unified; measured streaming roofline ~238 GB/s |
| OS | Windows 11 Pro, WDDM driver |
| our stack | gufo fork built against ROCm 10.1 (HIP SDK) |
| stock stack | upstream's Windows recipe: TheRock 10.0.0 runtime, Rust >= 1.88, vcpkg ports |

Practical limits we hit:

- **Device memory.** With 8 sessions x 32K context and a 9 GB prompt-cache RAM pin, peak dedicated GPU memory is
  ~93.5 GiB of ~110 GiB visible to the device; 8 x 64K does not fit, which is why the long-context profile is 4 x 64K.
- **Windows commit charge** is the binding host-side constraint: a launch needs ~100.8 GiB of commit headroom plus
  an 8 GiB reserve. Every launch checks it first.
- **Paging** from other processes (antivirus scans, file indexing, search tools) is enough to move a result by 1-3 %;
  see [Methods]({{ site.baseurl }}/methods/#the-paging-audit).
- **Upstream on Windows.** Upstream builds Windows as a development target; its release packages come from Linux, and
  its published benchmark numbers are Linux numbers.

## What can be reproduced today

| piece | public? | how |
|---|---|---|
| stock upstream arm | yes | build [gufo](https://github.com/gufo-org/gufo) main at `17d6fa25` with upstream's Windows build instructions (TheRock 10.0.0) |
| model weights | yes | [Unsloth Qwen3.8-Flash-Next GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL), revision `38bb39ee`, `UD-Q4_K_XL` |
| public MTP draft head | yes | [`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf), same revision |
| coding-agent workload | yes (builder pending publication) | [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories), split `tool`, shard `00004`; 8 sessions picked deterministically; assistant turns are generated live and fed back |
| chat workload | yes (builder pending publication) | [MT-Bench questions](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data/mt_bench), first 4 questions x 2 turns |
| canonical workload | yes | upstream's own model-bench pp2048 / tg128 recipe at the stock commit |
| public draft-vocabulary table | yes (tool pending publication) | rebuilt from SWE-smith `tool` shard `00002` (3,010 sessions) by our permutation tool: K = 6,931 pairs, fnv64 `7c191787b213acee`, sha256 `caade56a...`. The same tool, run on our private counts, reproduces the private production table byte for byte, so the method is verified even though the private input is not published |
| our engine changes | **not yet** | planned as a standalone public repository plus upstream pull requests |
| the benchmark harness | **not yet** | being scrubbed of internal paths and private workload references |
| production binary as is | **no** | it compiles in a small calibration table derived from private capture data (2,560 aggregated floats); a public build will regenerate the table from public text or ship a neutral one, then pass a short identity check against production |
| private draft head and tables | **no** | trained / derived from private agent sessions; shown as numbers only ("ours, production") |

## Reproducing the quick look (once the harness is public)

1. Build the stock arm at `17d6fa25` with upstream's Windows recipe; build ours from the public repository.
2. Download the weights and the public head at revision `38bb39ee`; check the sha256 values against the Hugging
   Face LFS ids.
3. Build the two workloads and the public draft-vocabulary table with the published builders; each builder checks
   its output against a published manifest (input URL, revision and sha256; output sha256) and refuses on mismatch.
4. Run each stage as `A B C C B A` with fresh servers, untimed warm-ups and a quiet-host wait before each launch
   (see [Methods]({{ site.baseurl }}/methods/)), plus a speculation-off reference for the exactness check.
5. Report per stage: client decode tok/s (C1), full-window tok/s (C8), TTFT, acceptance, tokens per target pass,
   host state per launch, and the strict exactness table.

A full public benchmark (more workloads, 3 launches per arm, bootstrap CIs, C8) is designed but not yet run: about
7-8 GPU-hours for the headline rows.

## Licences

Licence review is **pending**; links only:
[gufo](https://github.com/gufo-org/gufo),
[Qwen3.8-Flash-Next model card](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
[SWE-smith trajectories dataset card](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories),
[FastChat / MT-Bench](https://github.com/lm-sys/FastChat).
