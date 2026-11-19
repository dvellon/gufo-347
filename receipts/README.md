# Benchmark receipts

Each launch of the final benchmark has a signed JSON receipt, and so does each result row. A receipt records:

- input hashes: server binary, model shards, draft head, vocabulary table, workload, every request body, client;
- hardware and driver, with no host names or serials;
- per-request output hashes and timings;
- the metrics recomputed from those timings, and a speed band.

The signatures are `ssh-keygen -Y sign` signatures, with principal `gufo-bench-receipts` and namespace
`gufo-bench-receipt`.

```text
python verify_receipt.py RECEIPT.json --allowed-signers keys/allowed_signers                       # signature + internal consistency
python verify_receipt.py RECEIPT.json --allowed-signers keys/allowed_signers --assets A.json --dry-run    # + every input hash
python verify_receipt.py RECEIPT.json --allowed-signers keys/allowed_signers --assets A.json --rerun      # + re-runs the launch
```

`keys/receipt_ed25519.pub` and `keys/allowed_signers` are the public key. `receipt_lib.py` holds the shared hashing,
canonical-JSON, metric and statistics code, with the same definitions the benchmark runner uses. The verifier reports
in plain language what it checked, and what it could not check.

## Runs

| folder | run | arms (receipt names) |
|---|---|---|
| `runs/294/` | release benchmark of R8 (upstream v0.35.0 + patches/), 2026-11-17 | A = stock v0.35.0, R7Q = R8 strict, R7QX = R8 relaxed, B = previous build R7R (v0.34.0), RR7Q = R8 with speculation off |
| `runs/287/` | previous release benchmark of R7R (upstream v0.34.0), 2026-11-15 | A = stock v0.34.0, R7Q = R7R strict, R7QX = R7R relaxed, B = our pre-rebase build `edb3fe51` (reference), RR7Q = R7R with speculation off |
| `runs/289/` | previous release, private-drafter session, 2026-11-16 | A = stock v0.34.0, R7Q = R7R strict with the public drafter, RR7Q = references; the private-drafter launches have no receipts |

The arm names are the benchmark harness's cell names (kept from run 287 so the cells stay byte-identical across runs).
Row receipts cover the harness's standard rows (`B/A`, per-arm values); every other ratio of a run can be recomputed
from its launch receipts with `receipt_lib.py`.
