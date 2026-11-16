# Public workload and draft-vocabulary tools

Python 3.10+, stdlib only (`corpus_counts_public.py` also needs the model's tokenizer files via `--tokenizer-dir`).

| file | what it does |
|---|---|
| `build_sessions_public.py` | Builds the public benchmark workloads from their published sources (SWE-smith trajectories, `tool` split; MT-Bench questions). It also builds the corpus that the vocabulary counts are taken from. Every output is checked against a manifest (input revision and sha256, output sha256). |
| `corpus_counts_public.py` | Per-token use counts over a corpus (all tokens, assistant tokens, and assistant "misses": rare tokens not seen earlier in the session) |
| `perm_public.py` | Builds the draft-vocabulary table. The draft head reads only a 49,152-row prefix of the vocabulary. The table swaps prefix rows that are never used for the rare tokens your traffic misses most. K = min(unused prefix rows, ranked rare tokens, 16,384). `build` writes the pairs and a manifest; `check` re-derives and compares. |

The public table in our benchmark came from SWE-smith `tool` shard 00002 (3,010 sessions, 76.9 M tokens): K = 6,931.
Run on our private counts, the same rule reproduces our private table byte for byte. So the method is verified even
though our private input is not published. Using the table needs our fork's draft-permutation support; we plan to
send that upstream.

The gain from the table depends on the workload, so rebuild it from counts of **your** traffic.
