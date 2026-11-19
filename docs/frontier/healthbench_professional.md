# HealthBench Professional

`healthbench-professional` measures an assistant’s final reply to 525 clinician conversations. Its headline is the mean length-adjusted rubric score on a 0–100 scale. Panda uses Claude Sonnet 5 as the rubric judge. Artificial Analysis (AA) and OpenAI use GPT-5.4 at low reasoning effort, so the published rows are comparison points with a different judge. [AA methodology, accessed 2026-11-18](https://artificialanalysis.ai/methodology/intelligence-benchmarking) · [OpenAI paper, §4–5](https://arxiv.org/html/2604.27470v1)

## Data and candidate prompt

The loader reads `healthbench_professional_eval.jsonl` in source order through `core.fetch_pinned` and caches it under the runner’s `cache/healthbench-professional/` directory. Hugging Face revision `349962fd46dd02343a0d8a606491baf59154ea1a` pins the release. The 2,755,136-byte file has SHA-256 `d44b08e6e952e04c945e2c406f02533d9e7a989a84e35820ee7efdff20c9e4e2`. Its MIT-licensed dataset card asks users to keep examples out of public text and images. [Pinned data](https://huggingface.co/datasets/openai/healthbench-professional/blob/349962fd46dd02343a0d8a606491baf59154ea1a/healthbench_professional_eval.jsonl) · [Pinned dataset card and terms](https://huggingface.co/datasets/openai/healthbench-professional/blob/349962fd46dd02343a0d8a606491baf59154ea1a/README.md)

The exact candidate prompt is the row’s `conversation.messages`, with each role, content string, whitespace and turn order preserved. The release has only user and assistant turns and ends each conversation with a user turn. The runner requests one assistant reply, with no added instructions, tools or decoding override and a maximum output of **16,384 tokens**. A complete pass includes 410 single-turn and 115 multi-turn conversations. [Dataset card](https://huggingface.co/datasets/openai/healthbench-professional/blob/349962fd46dd02343a0d8a606491baf59154ea1a/README.md) · [AA protocol](https://artificialanalysis.ai/methodology/intelligence-benchmarking)

The public item ID is `hbp-` plus the first 16 lowercase hex digits of SHA-256 over the UTF-8 source ID. The public metadata holds the source ID, use case and `core.messages_digest` of the conversation. Rubric text and signed points stay in the private item key alongside use case, type, difficulty and specialty. Physician reference text is available separately for calibration.

## Grading one answer

The module embeds OpenAI’s MIT-licensed `GRADER_TEMPLATE` verbatim from `simple-evals` commit `652c89d0ca9df547706735883097e9537d40dc47`. The pinned `healthbench_eval.py` has SHA-256 `b763d8c1f53ecd165f0d170a32bc948216b3dfeadafe238b2083328e1af4b2df`. Its effective 3,181-character template has SHA-256 `2adffd51fd259554ebcd036ad1072d4aa2b7ce3aec2bbffe36271f911632ed3c` in UTF-8. OpenAI describes this public grader as a reference implementation of its internal evaluation. [Exact template and rendering](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py#L40) · [Upstream license](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/LICENSE) · [OpenAI dataset card](https://huggingface.co/datasets/openai/healthbench-professional/blob/349962fd46dd02343a0d8a606491baf59154ea1a/README.md)

For every criterion, starting at index zero, one judge request contains a single user message. Build the two template substitutions as follows.

```python
turns = (*item.messages, ("assistant", visible_answer))
conversation = "\n\n".join(f"{role}: {content}" for role, content in turns)
rubric_item = f"[{criterion['points']}] {criterion['criterion_text']}"
```

Replace the template’s `<<conversation>>` and `<<rubric_item>>` slots with those strings, preserving literal characters in both substitutions. The request name is the decimal criterion index. The runner sends up to **1,024 judge output tokens** per request. Parse the last complete JSON object in the reply, including fenced objects. Its `criteria_met` field must be a JSON boolean. Only that verdict survives into the grade. Explanations, answers and rubric wording stay private. [Reference code](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py#L397)

## Score and failures

For an answer with complete judging, let `p[j]` be each criterion’s signed integer points, `met[j]` its boolean verdict and `L` the number of Unicode code points in the visible final answer, including whitespace. Thinking tokens and earlier turns do not contribute to `L`. Each task has a positive denominator. [Paper, §4.1](https://arxiv.org/html/2604.27470v1) · [AA protocol, accessed 2026-11-18](https://artificialanalysis.ai/methodology/intelligence-benchmarking)

```text
raw      = sum(p[j] for met criteria) / sum(p[j] for positive criteria)
adjusted = raw - 0.0000294 × (L - 2000)

raw_score = 100 × clip(mean(raw), 0, 1)
score     = 100 × clip(mean(adjusted), 0, 1)
```

Meeting a negative criterion subtracts its points’ magnitude. A 1,500-character reply gains 1.47 percentage points and a 2,500-character reply loses 1.47 points. Individual raw values can be negative and adjusted values can exceed one. Clip **after** averaging. Every task and repeat carries equal weight. AA reports raw scores for use case, type and difficulty. This module clips each subgroup mean by the same rule and returns `by_use_case`, `by_type`, `by_difficulty`, `raw_score` and `mean_answer_length` alongside the headline. [AA board](https://artificialanalysis.ai/evaluations/healthbench)

A missing, empty or whitespace-only target reply produces zero in every score, with no judge requests. A substantive reply with any missing, malformed or nonboolean verdict also receives zero on the entire task after the runner’s retries. This local rule prevents a missing penalty verdict from improving the score. AA specifies zero for a task that ultimately fails, but does not publish a separate partial-judge rule. Both failure cases stay in score denominators and are excluded from mean answer length. The result reports `n`, `failed_answers` and `failed_judge_calls` separately. A returned refusal with visible text follows ordinary rubric grading. On an empty cohort, `n` and scores are zero. With no successfully graded answers, mean answer length is JSON null. [AA failure policy](https://artificialanalysis.ai/methodology/intelligence-benchmarking)

## Published comparison points

The table shows AA’s Professional results inspected **2026-11-18**. AA ran four independent answers per task with GPT-5.4 low as judge. The paper used eight samples per task for its main model comparisons, each model’s default system settings and its highest available reasoning effort. The two use the same length coefficient, center and aggregate clipping. Panda’s runner can start with a single pass and fixed product reasoning settings. It uses the Claude Sonnet 5 Batch API because this runner has no OpenAI judge credential. The different judge, sample count and model settings limit any direct ranking claim. [AA board](https://artificialanalysis.ai/evaluations/healthbench) · [AA methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking) · [Paper, §4.2](https://arxiv.org/html/2604.27470v1)

| System and reasoning effort | Length-adjusted | Raw | Mean characters |
|---|---:|---:|---:|
| Claude Fable 5.2, xhigh | 68.1 | — | — |
| Claude Fable 5.2, max | 67.7 | 71.1 | 3,170 |
| Claude Fable 5.2, high | 67.6 | — | — |
| GPT-6 Astra, max | 66.0 | 69.7 | 3,256 |
| GPT-6.1 Sol, max | 65.1 | 68.3 | 3,106 |
| Claude Sonnet 5.5, max | 63.0 | 70.8 | 4,663 |
| GPT-6 Sol, max | 62.3 | 61.2 | 1,619 |
| GPT-6 Luna, max | 61.9 | 62.7 | 2,291 |
| Claude Opus 5.5, max | 59.9 | 71.4 | 5,931 |
| Claude Haiku 5.5, max | 58.4 | 64.8 | 4,182 |
| Claude Fable 5.1, max | 57.8 | 69.2 | 5,901 |
| Grok 4.7, xhigh | 55.8 | 57.4 | 2,553 |
| Gemini 4 Argon, high | 55.1 | 64.2 | 5,133 |
| Kimi K3, max | 53.2 | 56.9 | 3,250 |
| GLM-5.3, max | 52.8 | 57.9 | 3,748 |
| Gemini 3.8 Flash, high | 51.7 | 59.0 | 4,494 |
| DeepSeek V4.1 Flash, max | 50.7 | 52.9 | 2,760 |

A dash means the November 18 comparison extract recorded only the headline. OpenAI separately reported 66.3 for GPT-6.2 Astra and 65.9 for GPT-6.2 Sol on **2026-11-17** using its own evaluation. [GPT-6.2 system card](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/model-safety-training-and-evaluation)

## Verification report

- **Files written.** [Benchmark implementation](../../panda_bench/frontier/benchmarks/healthbench_professional.py), [offline unit tests](../../tests/frontier/test_healthbench_professional.py) and this protocol.
- **Pinned inputs.** Verified the dataset, reference file and effective template against the revisions and SHA-256 values above. A read-only load of the pinned dataset found 525 unique item IDs and 1,135 criteria, including 213 negative criteria. In a network-disabled check, all 1,135 unsent judge prompts for a synthetic answer matched the pinned template filled against the corresponding source row.
- **Offline tests.** From the repository root, `PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -m unittest discover -s tests/frontier -t . -v` reported `Ran 9 tests` and `OK` on 2026-11-18. All nine HealthBench tests use synthetic data.
- **Runner and calibration.** Each complete pass requires 525 target answers and 1,135 judge requests, averaging 2.16 criteria per answer. Four repeats require 2,100 answers and 4,540 judgements. `reference_answers(cache_dir)` returns all 525 physician texts under the same item IDs. Comparing their stand-in score with the paper’s **43.7** requires 1,135 judge requests and zero target generations. Calibration is possible and remains unrun. No model API calls were made for this implementation. [Physician baseline, paper §5.1](https://arxiv.org/html/2604.27470v1)
