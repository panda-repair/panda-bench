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

## Current Panda experiments

Behavior: three-arm plans, at most two concurrent calls, content-bound per-model idempotency, settled-charge thresholds, partial/failure recording and conservative resume. Unknown responses block instead of causing a new paid attempt.

Verification: `python3 -m unittest discover -s tests -v` passed 22 tests in 4.418 seconds. `python3 -m compileall -q panda_bench tests` passed. Installed stage 2/3/4 checks matched 13/9/10 rows in 2.532/1.790/1.916 seconds. Keyless plans covered 450 generation requests and the new-run grading path; the $10 grading example refused conservatively.

Runtime boundary: fake transports covered parallel scheduling, budget overshoot, safe resume, missing charges, partial answers, charged failures, replay responses, changed content and secret-bearing exceptions. No Panda or model-provider call was made. Current API deployment and billing remain unverified.

Rollback: remove runner.py, run CLI dispatch and run/plan tests with their docs. Keep offline checks and archived regrading. Ignored local run folders hold all run state; no remote state was created during implementation.
