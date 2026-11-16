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
