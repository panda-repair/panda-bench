# Panda Bench

Recompute the published stage 2, 3 and 4 arithmetic from archived numeric grades, costs and timings. This local kit does not independently establish that the judge or experimental design is correct.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
panda-bench check stage-2
panda-bench check stage-3
panda-bench check stage-4
```

`check` is offline, requires no keys and imports no networking or provider library. It checks every displayed score, paired difference and 95% interval (4,000 resamples, seed 11), cost, cost share, median wait and denominator against the release's expected table. It also verifies artifact checksums. A mismatch exits nonzero.

Stage 2 includes 148 prompts (97 hard and 51 regular). Stage 3 selected 150 fresh prompts and scored 149 after one judge refusal. Stage 4 scored all 150, reusing stage 3 prompts and drafts to select configurations. Scores compare only within a stage. One AI judge and reused-prompt selection do not establish general superiority or human agreement. Costs are historical provider list-price calculations, not customer bills.

The Python code is MIT licensed. Dataset and answer provenance have separate terms in DATA_SOURCES.md.
