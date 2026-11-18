# Panda Bench

Recompute the published stage 2–4 arithmetic, regrade the archived answers. The releases contain 4,774 answer records and 32 displayed result rows. They contain no dataset prompt bodies.

## Start with the free check

```sh
git clone https://github.com/panda-repair/panda-bench
cd panda-bench
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
panda-bench check stage-4
```

This repository is prepared locally; the clone URL is the intended publication location, not evidence that it has been published. Use the source checkout and editable install. Python 3.9 or newer is supported; the verified environment used Python 3.9.6 and pip 21.2.4.

| Tier | What it proves | Needs | Cost and time |
| --- | --- | --- | --- |
| **Check our arithmetic** | Archived grades, failure/fallback rules, paired intervals, recorded costs and waits reproduce every displayed value | Release files; no keys, dataset fetch or model calls | Free; about 2–3 seconds per stage in the verified environment |
| **Grade our answers yourself** | A new judge sample over the complete original candidate bundle | Anthropic key and fetched WildBench prompts/checklists | Original stage 4 judging used about $10.07 at recorded batch rates; the conservative preflight is $47.227696. Batch completion time was not verified; allow up to 24 hours per provider guidance |

`check` imports no network or provider modules, including through the installed console script. It recomputes 4,000 paired bootstrap resamples with seed 11 and sorted indexes 100 and 3899. It exits nonzero on any difference from the expected displayed precision or an artifact checksum mismatch. A new run has no historical expected file; its check prints new results.

```sh
panda-bench check stage-2
panda-bench check stage-3
panda-bench check stage-4
```

Stage 2 has 148 prompts (97 from the hard configuration and 51 regular). Stage 3 selected 150 fresh prompts and scored 149 after one judge refusal; stage 4 scored all 150. Compare options within each stage. Stage 4 reuses the stage 3 prompts and drafts while selecting configurations, so it is not fresh confirmation. The one judge is not a substitute for human evaluation; new judgments and answers can move scores.

## Fetch the public dataset

```sh
panda-bench fetch
```

This fetches two hash-verified files, about 4.1 MB total, from the pinned WildBench revision. It selects 298 distinct prompt/checklist records by public session ID and source configuration. The cache stays under ignored `data/`. Each reconstructed record must match its release hash. The pinned revision matches the historical downloaded bytes; the originally requested revision was not recorded. See [DATA_SOURCES.md](DATA_SOURCES.md) for attribution and licensing.

## Grade the archived answers

Set `ANTHROPIC_API_KEY` in your environment, then inspect the plan before authorizing spend:

```sh
panda-bench regrade stage-4 --max-cost-usd 15 --plan
panda-bench regrade stage-4 --max-cost-usd 15
```

**The second command refuses before submitting anything.** Preserving the original 16,000-output-token allowance creates $24 of output-only exposure for 150 requests at the recorded batch rate. The complete conservative estimate is $47.227696: a UTF-8 byte-based input allowance plus request framing and the full output allowance. The observed $10.07 is retrospective usage, not a sufficient cap for this preflight. Rates are dated historical rates, not a current provider quotation or a billing guarantee; verify them before choosing a higher cap.

The plan lists public prompt IDs, shuffled candidate IDs, request count, missing answers, token allowances and both cost figures. It needs no key and makes no network calls. On execution, the kit reconstructs the original listwise context: 150 requests with 1,493 candidates, including lists of 10, 9 and 5. It preserves the original system prompt, schema, deterministic shuffle, Claude Opus 5.5, medium effort and 16,000-token output bound. Grading only the three landing-page rows would change that context.

A successful batch writes `runs/<timestamp>-regrade/` and updates `runs/latest`. `--judge MODEL` is accepted, but an unpriced model is refused: to use a different judge, make an explicit copy of the release, update its judge model and verified rate table, and pass that directory. The kit does not invent a rate for an unknown model.

The collector keeps historical score clamping, missing-grade handling, failed-generation scores and fallback inheritance. A judge refusal excludes that whole prompt; missing answer labels leave the affected option out of analysis. Raw provider identifiers and errors are never saved. If batch creation or collection has an uncertain outcome, it leaves a blocked state and does not retry. Automatic batch recovery is unavailable without retaining the provider batch identifier; inspect the provider account before deliberately starting another grading transaction.

