# Local work units

## Archived arithmetic

Behavior: three public releases plus an offline checker with historical rounding, failure/fallback rules and paired bootstrap.
Verification: `python3 -m unittest discover -s tests -v` passed5 tests; `python3 -m panda_bench check stage-2`, `stage-3`, `stage-4` matched all32 rows.
Runtime boundary: archived files only; no provider credentials/calls. A fresh subprocess asserts no urllib/http/requests/httpx/openai/anthropic/socket modules on the check path.
Rollback: remove releases, offline analysis/storage/CLI, packaging and their arithmetic tests together. No external state exists.
