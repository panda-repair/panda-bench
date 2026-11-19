# Frontier comparison, November 2026

Panda's single models, Panels and routers on five public benchmarks, run on 18–19 November 2026 and set next to the scores frontier labs' models have on the same benchmarks. Everything needed to check the arithmetic is in [`releases/frontier-2026-11/`](../../releases/frontier-2026-11/). The datasets' text is not in it.

## In brief

These are our runs set next to the labs' published rows, not entries on those boards. The judges, reasoning settings and number of runs differ (see [How to read these numbers](#how-to-read-these-numbers)), so treat gaps of a few points as noise.

**The best Panda setups sit in the middle of the frontier pack, not at the top.**

| Benchmark | Best Panda setup | Published rows for reference |
|---|---|---|
| HealthBench Professional | `auto:quality` 56.0, `panel:panda` 55.7. The judge reads about 3 points high here | Artificial Analysis: Claude Fable 5.2 68.1, GPT-6 Astra 66.0, Grok 4.7 55.8, Gemini 4 Argon 55.1 |
| GPQA Diamond | `auto:quality` and `gemini-4-argon` 93.9 (no judge) | Epoch, mean of 16 runs at maximum effort: GPT-6.2 Astra 96.8, Claude Fable 5.2 93.9, Claude Opus 5.5 90.6 |
| SimpleQA Verified | `gemini-4-argon` 73.2, `auto:quality` 72.9 | Kaggle: Gemini 3.1 Pro Preview 77.5, GPT-6 Astra 75.8, GPT-6.1 Sol 73.8 |
| MultiChallenge | `gemini-4-argon` 73.6, `auto:quality` 70.2 | Scale: GPT-6.1 Sol 81.95, Gemini 4 Argon 78.70 |
| RouterArena `sub_10` | `auto` 74.3% at $0.20 per 1,000 queries (RouterArena's prices) | On the same 809 queries, from their published predictions: Azure Model Router 75.8%, Divyam 83.5%. Board routers cost $0.66–$4.10 per 1,000 on the full split |

**`auto:quality` is the best all-rounder.** It sends facts, science and most other requests to Gemini 4 Argon, and writing, role-play and reasoning to `panel:panda-quality`. It came first, or within 3.4 points of first, on all four benchmarks, at 1.0–3.5¢ per answer.

**Panels beat their own members on open-ended health answers, not on facts, multiple choice or multi-turn instructions.**

- On HealthBench Professional, `panel:panda` (DeepSeek V4.1 Flash and Gemini 3.8 Flash, 1.8¢) scored 55.7, against 49.0 and 38.3 for its members alone. That is above Artificial Analysis's rows for both models at their highest listed effort (Gemini 3.8 Flash high 51.7, DeepSeek V4.1 Flash max 50.7), even after the judge's 3 points.
- The self-merge control explains part of that gain: merging two DeepSeek drafts gained 6.8 points, and making the second draft Gemini 3.8 Flash's gained 10.6 more.
- On SimpleQA Verified, GPQA Diamond and MultiChallenge, every Panel scored below Gemini 3.8 Flash alone. A likely cause, not yet verified, is that the combiner sometimes keeps the weaker draft's fact or choice.

**DeepSeek V4.1 Flash and GLM-5.3 score far below their published rows; Gemini 4 Argon stays close.**

- DeepSeek V4.1 Flash scored 79.3 on GPQA Diamond, against Epoch's 93.2. On HealthBench Professional it scored 38.3, against Artificial Analysis's 50.7.
- GLM-5.3 scored 72.2 on GPQA Diamond, against Epoch's 90.9.
- Gemini 4 Argon came within 1.5–5.2 points of its published rows on all four benchmarks.
- The published rows use maximum reasoning effort, while Panda serves these models on Workers AI at its own settings. This run does not measure which of the two explains the gap.

**`auto` (balanced) sends most requests to DeepSeek V4.1 Flash** and scores close to it, at about 1¢ per answer.

**The 1¢ minimum dominates cheap answers.** At list prices, `auto` costs $0.27 per 1,000 RouterArena queries; at Panda's price, with at least 1¢ per answer, it costs $10.


## What was compared

| Setup | What it is | Benchmarks |
|---|---|---|
| `deepseek-v4.1-flash` | DeepSeek V4.1 Flash, as Panda serves it (Workers AI) | all four |
| `glm-5.3` | GLM-5.3 (Workers AI) | all four |
| `glm-5.3-flash` | GLM-5.3 Flash (Workers AI). Its product output bound is 4,096 tokens, and Workers AI retired it on 2026-11-20 | all four |
| `gemini-3.8-flash` | Gemini 3.8 Flash (Vertex AI) | all four |
| `gemini-4-argon` | Gemini 4 Argon (Vertex AI) | all four |
| `panel:panda` | Panda's Panel: DeepSeek V4.1 Flash and Gemini 3.8 Flash answer, and DeepSeek V4.1 Flash combines the two drafts | all four |
| `panel:panda-quality` | The same two drafts, combined by GLM-5.3 | all four |
| `panel:trio-glm` | A custom Panel: GLM-5.3 Flash, DeepSeek V4.1 Flash and Gemini 3.8 Flash, combined by GLM-5.3 | all four |
| `auto` | Panda's router, balanced mode: it classifies each request and picks one model | all four, and RouterArena |
| `auto:quality` | The router in quality mode, which sends some task types to `panel:panda-quality` | all four |
| `control:deepseek-self-merge` | Two DeepSeek V4.1 Flash drafts combined by DeepSeek V4.1 Flash. It separates a Panel's gain from simply sampling twice. Research only: production refuses a Panel with a repeated member | HealthBench Professional and MultiChallenge |

"All four" means HealthBench Professional (525 conversations), MultiChallenge (266 conversations), SimpleQA Verified (1,000 questions) and GPQA Diamond (198 questions). RouterArena's `sub_10` split (809 queries) ran `auto` only, since RouterArena ranks routers that pick one model per query.

Each setup answered each item once, at Panda's model settings, with an output limit of 16,384 tokens (Panda's API maximum). The per-benchmark pages give the data pins, prompts and graders: [HealthBench Professional](healthbench_professional.md), [MultiChallenge](multichallenge.md), [SimpleQA Verified](simpleqa_verified.md), [GPQA Diamond](gpqa_diamond.md) and [RouterArena](routerarena.md).

## How it was run

**A replay of Panda's product, not its public API.** Panda's API runs two requests at a time per workspace, so the roughly 21,000 answers in this comparison would have taken days through it. The harness instead imports Panda's own product code, from revision `3cedbb0d` of Panda's private platform repository, which carries the same Panel and router code as production on these dates. With it, the harness:

- builds every provider request with the product's request builder;
- reads every answer with the product's parser and classifies failures with the product's own verdicts;
- runs the product's Panel draft order, combiner prompt and bounds, stand-in rule and router.

Only the transport differs: the harness calls Workers AI and Vertex AI directly.

**A live check through the public API.** The same 8 items per benchmark went through `https://app.panda.repair/v1` for `panel:panda`, `panel:panda-quality`, `auto` and `auto:quality`: 128 paid requests, each with an idempotency key. The judge graded the live answers the same way, and the release pairs each live score with the replay's score on the same items.

**The judge.** Claude Sonnet 5 with thinking off, through Anthropic's Message Batches API, stands in for each benchmark's official grader: GPT-5.4 for HealthBench Professional, GPT-4o (paper) or Gemini 2.5 Pro (Scale's board) for MultiChallenge, and GPT-4.1 for SimpleQA Verified. GPQA Diamond and RouterArena grade by rules, with no judge. Each judge request's id binds the judge settings and a digest of the exact answer, so a replaced answer is never graded by an older reply.

**How far the stand-in judge sits from the official ones.** Before the main run, the stand-in judge graded answers whose official scores are published:

| Answers graded | Official score | Stand-in judge | Difference |
|---|---:|---:|---:|
| MultiChallenge, o1-preview (paper's stored responses, GPT-4o judge) | 35.73 | 36.44 | +0.7 |
| MultiChallenge, Mixtral-8x7B | 11.08 | 14.02 | +2.9 |
| MultiChallenge, Llama-3.3-70B | 23.19 | 29.50 | +6.3 |
| HealthBench Professional, the dataset's 525 physician answers (GPT-5.4 grader) | 43.7 | 46.77 | +3.1 |

The stand-in judge kept the rank order and was equal or up to 6 points more generous. Read every judged score here with that in mind. `grader-check.json` in the release has the full numbers.

**Price and time.**

- **Price** is Panda's price for the answer: the providers' list cost of every call it made, times 3/2, rounded once to the cent, at least 1¢ for an answer.
- **Model time** is the time spent in model calls: a single model's call; a Panel's slowest drafting member, then its combiner; and `auto`'s classification, then its pick. It counts each call's last attempt and leaves out the harness's own queueing and retries.
- First attempts that failed under our load (rate limits, timeouts) were retried. They are reported per setup as `first_attempt_trouble`.

**Failures count.** A model that refused, ran out of room or returned nothing scores as each board scores a missing answer: wrong, or not attempted. A Panel that lost a member answered from the members it had (`partial`).

There was one exception. On 2026-11-19, about 05:40–07:30 UTC, GLM-5.3 Flash returned server errors on many calls, the day before its retirement. The answers that lost a call that way were sent again after the main pass:

- `auto` answers that failed outright;
- trio Panels that combined two drafts instead of three.

Boards also retry API errors, and a setup's score should measure its design rather than one provider's outage. The release counts every redone answer (`rows_written`, `redone`).

## Results

### HealthBench Professional

Length-adjusted score (0–100), Claude Sonnet 5 judge. One answer per item; ± is the standard error where the benchmark defines one.

| Setup | Score | Price per answer | Model time, median / p90 | Failed | Partial |
|---|---:|---:|---:|---:|---:|
| `auto:quality` | 56.0 | 3.5¢ | 32.5 s / 109 s | 2 | 2 |
| `panel:panda` | 55.7 | 1.8¢ | 18.7 s / 49 s | 0 | 3 |
| `panel:trio-glm` | 55.1 | 3.0¢ | 22.3 s / 59 s | 0 | 4 |
| `panel:panda-quality` | 54.6 | 2.7¢ | 20.2 s / 52 s | 0 | 4 |
| `gemini-4-argon` | 53.6 | 3.8¢ | 30.8 s / 135 s | 3 | 0 |
| `gemini-3.8-flash` | 49.0 | 1.3¢ | 11.1 s / 36 s | 4 | 0 |
| `glm-5.3` | 47.8 | 1.1¢ | 6.9 s / 14 s | 0 | 0 |
| `control:deepseek-self-merge` | 45.1 | 1.1¢ | 15.2 s / 28 s | 0 | 0 |
| `auto` | 44.6 | 1.0¢ | 9.6 s / 20 s | 0 | 0 |
| `glm-5.3-flash` | 44.2 | 1.0¢ | 10.4 s / 23 s | 1 | 0 |
| `deepseek-v4.1-flash` | 38.3 | 1.0¢ | 7.8 s / 15 s | 0 | 0 |

### MultiChallenge

Pass rate, mean of the 4 axes (0–100), Claude Sonnet 5 judge. One answer per item; ± is the standard error where the benchmark defines one.

| Setup | Score | Price per answer | Model time, median / p90 | Failed | Partial |
|---|---:|---:|---:|---:|---:|
| `gemini-4-argon` | 73.6 | 3.2¢ | 21.1 s / 80 s | 1 | 0 |
| `auto:quality` | 70.2 | 2.9¢ | 21.1 s / 84 s | 0 | 0 |
| `gemini-3.8-flash` | 69.7 | 1.2¢ | 8.3 s / 25 s | 0 | 0 |
| `panel:trio-glm` | 67.4 | 2.6¢ | 16.1 s / 36 s | 0 | 1 |
| `panel:panda-quality` | 66.7 | 2.4¢ | 14.7 s / 48 s | 0 | 0 |
| `panel:panda` | 62.5 | 1.6¢ | 14.4 s / 44 s | 0 | 0 |
| `control:deepseek-self-merge` | 61.2 | 1.0¢ | 11.5 s / 18 s | 0 | 0 |
| `auto` | 59.6 | 1.0¢ | 7.2 s / 12 s | 0 | 0 |
| `deepseek-v4.1-flash` | 57.8 | 1.0¢ | 5.8 s / 10 s | 0 | 0 |
| `glm-5.3` | 57.6 | 1.1¢ | 4.6 s / 8 s | 0 | 0 |
| `glm-5.3-flash` | 56.5 | 1.0¢ | 6.7 s / 13 s | 0 | 0 |

### SimpleQA Verified

F1 (0–100), Claude Sonnet 5 grader. One answer per item; ± is the standard error where the benchmark defines one.

| Setup | Score | Price per answer | Model time, median / p90 | Failed | Partial |
|---|---:|---:|---:|---:|---:|
| `gemini-4-argon` | 73.1 | 1.0¢ | 6.5 s / 77 s | 0 | 0 |
| `auto:quality` | 72.9 | 1.0¢ | 10.8 s / 98 s | 0 | 0 |
| `gemini-3.8-flash` | 64.5 | 1.0¢ | 2.1 s / 23 s | 0 | 0 |
| `panel:panda-quality` | 60.0 | 1.1¢ | 5.8 s / 35 s | 0 | 0 |
| `panel:trio-glm` | 56.4 | 1.1¢ | 7.6 s / 43 s | 0 | 1 |
| `panel:panda` | 53.6 | 1.0¢ | 4.0 s / 23 s | 0 | 0 |
| `glm-5.3` | 35.0 | 1.0¢ | 2.1 s / 4 s | 0 | 0 |
| `deepseek-v4.1-flash` | 34.4 | 1.0¢ | 1.3 s / 3 s | 0 | 0 |
| `auto` | 33.5 | 1.0¢ | 2.5 s / 5 s | 0 | 0 |
| `glm-5.3-flash` | 30.6 | 1.0¢ | 2.9 s / 6 s | 2 | 0 |

### GPQA Diamond

Accuracy (%), rule-graded, no judge. One answer per item; ± is the standard error where the benchmark defines one.

| Setup | Score | ± | Price per answer | Model time, median / p90 | Failed | Partial |
|---|---:|---:|---:|---:|---:|---:|
| `auto:quality` | 93.9 | 1.7 | 2.6¢ | 23.1 s / 121 s | 0 | 1 |
| `gemini-4-argon` | 93.9 | 1.7 | 2.5¢ | 17.6 s / 104 s | 0 | 0 |
| `gemini-3.8-flash` | 91.9 | 1.9 | 1.5¢ | 9.3 s / 37 s | 0 | 0 |
| `panel:panda-quality` | 91.4 | 2.0 | 2.5¢ | 14.3 s / 51 s | 0 | 3 |
| `panel:trio-glm` | 90.4 | 2.1 | 2.5¢ | 17.5 s / 69 s | 0 | 27 |
| `panel:panda` | 88.4 | 2.3 | 1.8¢ | 14.1 s / 63 s | 0 | 0 |
| `deepseek-v4.1-flash` | 79.3 | 2.9 | 1.0¢ | 3.9 s / 9 s | 0 | 0 |
| `auto` | 76.8 | 3.0 | 1.0¢ | 5.5 s / 11 s | 0 | 0 |
| `glm-5.3` | 72.2 | 3.2 | 1.0¢ | 3.2 s / 12 s | 7 | 0 |
| `glm-5.3-flash` | 68.7 | 3.3 | 0.8¢ | 4.4 s / 42 s | 32 | 0 |

### RouterArena sub_10

Accuracy (%), rule-graded, no judge. One answer per item; ± is the standard error where the benchmark defines one.

| Setup | Score | Price per answer | Model time, median / p90 | Failed | Partial |
|---|---:|---:|---:|---:|---:|
| `auto` | 74.3 | 1.0¢ | 2.6 s / 6 s | 0 | 0 |

### RouterArena's own metrics

`auto` routed 712 of the 809 queries to DeepSeek V4.1 Flash, 60 to GLM-5.3 Flash and 37 to GLM-5.3.

| Prices used | Cost per 1,000 queries | Acc-Cost Arena |
|---|---:|---:|
| RouterArena's pinned prices | $0.20 | 73.26 |
| Panda's list prices | $0.27 | 72.92 |
| Panda's price (at least 1¢ per answer) | $10.00 | 64.55 |

The Arena number is RouterArena's formula on `sub_10`, while its board is on the full split. RouterArena's scores are lower on the full split: Azure Model Router scores 75.8% on `sub_10` and 72.9% on the full split. So this is a placement, not a board rank.

All 5 SuperGLUE cloze queries scored 0. RouterArena's prompt asks for "the correct letter choice in \\boxed{X}, where X is the contetn of the correct option", but its board scorer compares the reply with the option's text. Our models answered with the letter. The prompts are byte-identical to RouterArena's, so these are lost points, not a port error.

### The live check

128 requests through Panda's public API, 8 items per benchmark, cost $2.14 at Panda's prices.

- **Scores.** On these 8-item samples, live and replay scores differed by −12 to +25 points per setup and benchmark, with no consistent direction. Samples this small cannot detect a difference of a few points, so they confirm the setup, not the scores.
- **Router.** Production's router classified 53 of 64 requests (83%). The other 11 missed the 1-second classification deadline and went to the mode's default model.
- **Failures.** Panels completed 62 of 64 requests, and 2 were partial. `auto:quality` failed 3 of 32 with no answer (`outcome_unknown`). Timeouts on Gemini 4 Argon, whose 90th-percentile model time here was 77–135 s, are a likely cause, not verified.


## How to read these numbers

- **Different judges and repeats.** Published rows come from official graders (GPT-4.1, GPT-5.4 low, GPT-4o or Gemini 2.5 Pro) and often average several answers per item. Artificial Analysis uses 4 on HealthBench Professional, and the paper uses 8. Here each setup answered once, and a stand-in judge graded. Small gaps are within that noise; the judge offset above is the main known bias.
- **The replay's router always classifies.** Production's router gets one try within 1 second to classify a request. The replay gave it time to finish, so `auto` here is the router as designed. In the live check, production classified 53 of 64 router requests (83%). The other 11 went to the mode's default model, a behavior the replay does not show.
- **Panda's settings, not the labs' highest ones.** Each model ran at Panda's product settings. Published frontier rows often use the highest reasoning effort a lab offers.
- **The models are served by Cloudflare and Google.** Scores can move when a provider changes how it serves a model.

## Check it yourself

The free check needs no key, dataset or network. It verifies every release file against the manifest's SHA-256 and recomputes every setup's score from the published grades:

```sh
pip install -e .
python -m panda_bench.frontier check-release releases/frontier-2026-11
```

The release has no answer text, so regrading needs answers you generate. The harness can generate them through Panda's public API (`python -m panda_bench.frontier live`, with your own Panda key) and grade them with the same judge (`judge` and `score`, with an Anthropic key). The replay backend needs Panda's private product code, so outside readers cannot run it.

### Run GPQA Diamond yourself

GPQA Diamond needs no judge, since a rule grades each answer. So you can rerun it end to end with only a Panda API key:

```sh
git clone https://github.com/panda-repair/panda-bench
cd panda-bench
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
export PANDA_API_KEY=...   # a Panda API key that allows the model you test
python -m panda_bench.frontier live --run runs/my-gpqa --benchmarks gpqa-diamond --setups auto:quality --n 198 --max-charge-usd 7
python -m panda_bench.frontier score --run runs/my-gpqa --benchmarks gpqa-diamond
```

- **Data.** The first command downloads the pinned GPQA zip from the authors' GitHub repository and checks its SHA-256 before use. The zip is password-protected to keep the questions away from web crawlers; the authors publish the password. The cache stays under the ignored `data/`. The authors ask that examples not be posted online.
- **What it sends.** All 198 questions, two at a time (Panda's per-workspace limit), with an idempotency key on each request. A rerun skips the questions already answered. The `--max-charge-usd` cap stops it from sending more once your recorded charges reach that amount.
- **What it costs.** At our run's prices, `auto:quality` answers cost about 2.6¢ each, so all 198 come to about $5 and take roughly 40 minutes. `--setups` also takes `gemini-4-argon`, `gemini-3.8-flash`, `deepseek-v4.1-flash`, `glm-5.3`, `panel:panda`, `panel:panda-quality` and `auto`.
- **The result.** `score` prints the accuracy, the number of questions answered, and your actual charge per answer, and writes `results.json` and per-question grades to the run folder. Expect a few points' spread between runs: one run on 198 questions has a standard error of about 1.7 points. Ours scored 93.9.

## The release

`releases/frontier-2026-11/` holds:

- `manifest.json`: the frozen design (setups, models, output limit, dataset pins, judge and its settings), every start of the run with its harness and product revisions, counts and the SHA-256 of every file;
- per benchmark:
  - `items.jsonl`: item ids, the public fields and prompt digests;
  - `answers.jsonl`: status, route or Panel steps, price, model time, and the answer's SHA-256 and length;
  - `usage.jsonl`: every model call, with model, role, outcome, attempts, tokens, list cost and seconds;
  - `judge.jsonl`: every judge reply's result and tokens, without its text;
  - `grades.jsonl` and `results.json`;
- `live/`: the live check's answers, grades and paired scores;
- `grader-check.json`: the stand-in judge's scores on the reference answers above.

No file has a prompt, rubric, reference answer, answer option, model answer or judge reply. HealthBench Professional and GPQA ask that their examples not be posted online, and RouterArena is evaluation-only. The release writer refuses any long string, credential or local path, and its tests check that planted markers never reach a release. See [DATA_SOURCES.md](../../DATA_SOURCES.md) for licenses.

## What we learned running it

[learnings.md](learnings.md) is the log kept during the run: why each benchmark was chosen or dropped, the judge checks, the harness faults and their fixes, the provider limits and outages, and each pace change.
