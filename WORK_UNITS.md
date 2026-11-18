# Local work units

## Archived arithmetic

Behavior: three public releases plus an offline checker with historical rounding, failure/fallback rules and paired bootstrap.

Verification: `python3 -m unittest discover -s tests -v` passed 5 tests; `python3 -m panda_bench check stage-2`, `stage-3`, `stage-4` matched all 32 rows.

Runtime boundary: archived files only; no provider credentials or calls. A fresh subprocess asserts no urllib/http/requests/httpx/openai/anthropic/socket modules on the check path.

Rollback: remove releases, offline analysis/storage/CLI, packaging and their arithmetic tests together. No external state exists.

## Pinned inputs and archived regrading

Behavior: hash-verified dataset fetch; exact historical listwise judge plans; budget refusal before submission; synthetic batch collection; an installed script that avoids runtime packaging/network imports. Missing judge labels preserve the original omission policy.

Verification: `python3 -m unittest discover -s tests -p test_judging.py -v` passed 6 tests. `panda-bench fetch` reconstructed 298 records from both cached pinned parquet files, all hashes matching. The installed CLI import guard passed after reinstalling with `pip install -e .`. Stage 4 plan has 150 requests and 1,493 candidates, with $47.227696 conservative exposure; $15 refuses before fetch or SDK initialization.

Runtime boundary: live batch creation is untested and was never invoked. A fake batch exercises serialization, parsing, usage and privacy-safe state. The dataset reader was exercised on byte-identical public input files. Package installation was confined to the local virtual environment.

Rollback: remove datasets.py, judging.py, their tests and the fetch/regrade/grade CLI branches; revert the associated README and direct-script packaging changes. Archived releases and arithmetic remain independently usable.

