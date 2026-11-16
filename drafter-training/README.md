# Drafter-training tooling

This folder holds the tooling we used to fine-tune Qwen3.8-Flash-Next's MTP draft head on our own traffic. Read
[`docs/TRAIN-YOUR-OWN-DRAFTER.md`](../docs/TRAIN-YOUR-OWN-DRAFTER.md) first. It covers what data to collect, what it
costs and what to expect.

Nothing here contains our data, our trained head or our vocabulary table. You bring your own prompts.

## Layout

| folder | files | what it does |
|---|---|---|
| `onpolicy/` | `make_train_prompts.py`, `make_train_jobs.py` | Pick prompts from your traffic (code, prose, agent next-steps), split them by source key, and turn the target model's greedy answers into capture jobs |
| `capture/` | `mtp_capture.cpp`, `make_overlay.py`, `read_gmtp.py`, `build_corpus.py` | The capture overlay: a teacher-forced pass of the target that writes the trunk's final hidden state and the target's top-32 distribution per position, plus the shard reader / verifier and a teacher-forced corpus builder (your repos, patches, logs) |
| `reference/` | `mtp_ref.py`, `validate_vs_oracle.py` | A CPU reference of the MTP head, used to validate the trainable re-implementation |
| `train/` | `mtp_net.py`, `train_mtp.py`, `stage2_data.py`, `moe_grouped.py`, `quant_sim.py`, `export_gguf.py`, `eval_offline.py`, `tiny.py`, `selftest_*.py` | The trainable head, the trainer (stage 1 + stage 2 mixtures, early stop, warm restarts), grouped-MoE kernels, a bit-exact Q8_0 quantizer, GGUF export, offline evaluation, selftests |
| `eval/` | `s2_sets.py`, `overlap_check.py`, `shallow_score.py`, `gmtp_subset.py` | Pre-registered selection / judge splits, the leakage check (32-gram overlap + source-key collisions), depth-1 / depth-2 acceptance scoring, shard subsets |
| `../tools/` | `build_sessions_public.py`, `corpus_counts_public.py`, `perm_public.py` | Token counts from a corpus, and the draft-vocabulary table (permutation) built from them |

## Requirements

- **Training:** Python 3.10+, PyTorch with CUDA, NumPy, and one GPU with 80 GB (we used one H100 80 GB). The CPU
  paths (`--device cpu`, `tiny.py`, the selftests) run anywhere, slowly.
- **Reference / export:** `reference/mtp_ref.py` reads GGUF files through gufo's Python GGUF reader
  (`tools/gufo/gguf.py`), so put a gufo checkout on `PYTHONPATH`.
- **Capture:** a gufo build with the capture overlay (see below) and the model's GGUF files. Plan about 20.7 KB of
  disk per captured token.
- **Vocabulary table:** Python stdlib plus the model's tokenizer files.

## Pipeline

```text
1. prompts       python onpolicy/make_train_prompts.py --out work --per-category 100 --transcripts <your-agent-transcripts.jsonl>
2. answers       generate greedy answers with the target model (any engine; keep exact token ids), one file per arm
                 (e.g. thinking on / off)
3. jobs          python onpolicy/make_train_jobs.py --responses answers-think.jsonl:think answers-nothink.jsonl:nothink --out jobs-train
4. capture       run the capture overlay over the jobs -> sharded .mtps files + .mtps.json sidecars;
                 python capture/read_gmtp.py verify SHARD.mtps
5. sets          python eval/overlap_check.py ...; python eval/s2_sets.py split ... (selection half / judge half by prompt key)
6. stage 1       python train/train_mtp.py --device cuda --mtp-gguf mtp.gguf --frozen-bundle frozen-target.pt \
                     --shards "train/*.mtps" --val-shards "val/*.mtps" --out runs/s1 \
                     --dense-mode full --expert-mode lora --depth 7
7. stage 2       python train/train_mtp.py ... --init-trainable runs/s1/<best> --dense-mode full --expert-lora-rank 16 \
                     --depth 7 --epochs 2 --grad-accum 8 --lr 1.5e-5 --lr-lora 1e-4 --warmup 10 --min-lr-frac 0.1 \
                     --kind-weights response=2,prompt=0.5 --offpath-weight 0 --window-long-records --early-stop <selection-set>
                 then a continuation: the same command from the best stage-2 checkpoint (fresh optimizer, warm-up 10)
8. export        python train/export_gguf.py --checkpoint runs/s2/ckpt/<best>.pt --out mtp-ft-Q8_0.gguf --forward-check ...
9. score         python train/eval_offline.py ...; python eval/shallow_score.py ...   (judge half only)
10. live A/B     serve with --mtp-model mtp-ft-Q8_0.gguf vs the public head, interleaved launches, nothing else changed
11. optional     python tools/corpus_counts_public.py ... ; python tools/perm_public.py build ...  (draft-vocabulary table;
                 needs our fork's draft-permutation support to use)
```

Run each script with `--help` for its options. The docstrings say what each check refuses on.

## Notes

- **Capture format.** The shard format is "GMTP v1": a header, per-position records and a trailer, with a JSON
  sidecar. `capture/read_gmtp.py` documents it and verifies it. In this public tooling the shard file extension is
  `.mtps`.
- **The capture overlay targets our fork.** `make_overlay.py` applies anchored edits to four engine source files and
  refuses if any anchor does not match exactly once. It was written against our fork's working tree. Against stock
  gufo, the anchors will need porting. We plan to offer a capture hook upstream.
- **Generation and capture can use different engines.** Answers generated elsewhere are fine as text. About 4 % of
  positions disagreed with the serving engine's argmax, and re-capturing with the serving engine fixes the labels.
- **Pin threads for golden tests.** A near-tie can flip with a different reduction order.
- **Use `allow_abbrev=False`.** Every script uses it. One of our early bugs was `--lr` silently matching another
  flag.

## Not included

- Our captured data, prompts, transcripts and token counts.
- Our trained head, its re-quantised copy, and our vocabulary table.
- The cloud provisioning and cost-guard scripts. They are specific to our account. Use any single-GPU 80 GB instance,
  and pull your checkpoints off it before you delete it.
