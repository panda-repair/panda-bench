# MultiChallenge

`multichallenge` measures a setup's final reply in realistic multi-turn conversations. It covers four axes: instruction retention, inference memory, reliable versioned editing and self-coherence. Each conversation has a YES/NO rubric question about the reply. The headline is Scale's: the pass rate of each axis, then the unweighted mean of the four, times 100.

The official judge is GPT-4o in the paper and the repository, and Gemini 2.5 Pro on Scale's board. Neither is available to us, so the judge is **Claude Sonnet 5** with the repository's judge prompt verbatim. Published rows are therefore comparison points with a different judge. Module: `panda_bench/frontier/benchmarks/multichallenge.py`.

All URLs below were accessed on 18 Nov 2026.

## Data

### The current release: `BENCHMARK` (`multichallenge`)

Scale's board links its "Open-Source Dataset" button and its "open source benchmark and auto–eval tool" to the Hugging Face dataset [`ScaleAI/MultiChallenge`](https://huggingface.co/datasets/ScaleAI/MultiChallenge). The GitHub repository named in the paper has not moved, but this dataset is the newer official release, so the module scores it.

| | |
|---|---|
| Revision | `670fd81f270314d0f8f82b34fcfe2187fd28f699`, uploaded 2026-03-31; still the latest commit on 18 Nov 2026 |
| File | `data/test-00000-of-00001.parquet`, 975,319 bytes |
| SHA-256 | `2711646aba61f814823297714aaf757e3a1b71c66c501402231de48490650515` (equals Hugging Face's own LFS oid) |
| Dataset card | `README.md` at the same revision, SHA-256 `5a7c331376ced07501a6fa36d5cdc89102901206955d7a6c7d9bba0abeabee66`; license CC BY 4.0 |
| Rows | 266: INSTRUCTION_RETENTION 69, INFERENCE_MEMORY 109, RELIABLE_VERSION_EDITING 40, SELF_COHERENCE 48 |

- Fields: `question_id`, `axis`, `conversation` (parallel `role` and `content` lists), `target_question`, `pass_criteria` and `num_turns`.
- `num_turns` counts messages; `load` checks that it does.
- Every `pass_criteria` is `YES`.
- No row has a system message. Every conversation alternates from a user turn to a final user turn: 2–10 user turns, 3–19 messages.
- The largest conversation is 45,447 UTF-8 bytes.

**How it differs from the 2025 release, and why it is very likely the board's set.** Compared row by row with the GitHub file below:

- 266 question ids are shared, and all 266 conversations are byte-identical.
- 47 rubric questions (`target_question`) were rewritten: SELF_COHERENCE 16, INFERENCE_MEMORY 15, RELIABLE_VERSION_EDITING 10, INSTRUCTION_RETENTION 6.
- 7 conversations were removed:
  - INFERENCE_MEMORY: `6765f4400e0c1f56ddde8290`, `6765f440b0ca0453b824ac33`, `6765f441d19c1caaffcdba35`, `67819bbb6e1295dafab2cd03`
  - SELF_COHERENCE: `674568562274da0299d9b121`, `6781aa53113b7dd6a81d27cb`
  - RELIABLE_VERSION_EDITING: `6765fc5ec794f0de7ce718ad`
- No axis label or pass criterion changed.

The board's note says: "Update February 3, 2026: … Evaluation questions updated for clearer axis alignment and improved judge accuracy on 54 tasks". Scale's [blog of 23 Mar 2026](https://labs.scale.com/blog/multichallenge-update) describes the revision as refined rubric questions plus removed edge cases. Here 47 + 7 = 54.

The blog's per-axis figures also fit this split and not the 2025 one:
- The top RELIABLE_VERSION_EDITING score, 57.50, is 23/40 (or 69/120 over three runs). No count of 41 items, over one to five runs, gives 57.50.
- Gemini 3 Pro's INFERENCE_MEMORY 81.96 is 268/327, that is 109 items over three runs. No count of 113 items, over one to five runs, gives 81.96.

Scale does not say that this file is the board's exact set, so treat that as a strong inference.

### The 2025 release: `REFERENCE_BENCHMARK` (`multichallenge-2025`), used for the grader check only

| | |
|---|---|
| Repository | [`github.com/ekwinox117/multi-challenge`](https://github.com/ekwinox117/multi-challenge), linked from the paper |
| Commit | `5ccefcca6a39020d66c1383c4e6a809cb07afa33` (2025-02-05, "fixed axis names, minor bugs"); still `main` on 18 Nov 2026; GitHub reports no license |
| File | `data/benchmark_questions.jsonl`, 2,270,010 bytes, SHA-256 `06c72d1ffa387481910d100b146295f28ab68c4ba0032743eac5c9195edb596e` |
| Rows | 273: INSTRUCTION_RETENTION 69, INFERENCE_MEMORY 113, RELIABLE_VERSION_EDITING 41, SELF_COHERENCE 50 (the paper's Table 1) |

- Fields: `QUESTION_ID`, `AXIS`, `CONVERSATION`, `TARGET_QUESTION`, `PASS_CRITERIA`.
- The repository's only other data commit, its first, used the axis names RECOLLECTION, COHERENCE and REFINEMENT.
- The pinned commit renamed them and changed no conversation or rubric question.

### Pinned files and cache

`load` and `reference_responses` fetch through `core.fetch_pinned`. Each file is refused unless its SHA-256 matches. Files are cached under the runner's cache directory, `cache/multichallenge/`:

- `huggingface/data/test-00000-of-00001.parquet` from `https://huggingface.co/datasets/ScaleAI/MultiChallenge/resolve/<revision>/data/test-00000-of-00001.parquet`.
- `github/data/benchmark_questions.jsonl` and the twelve `github/data/final_model_responses/*.jsonl` files from `https://raw.githubusercontent.com/ekwinox117/multi-challenge/<commit>/…`. Their digests are in the grader-check table below.

## Items and the prompt the setup answers

- **Item id:** `mc-` plus the first 16 hex digits of SHA-256 over the UTF-8 `question_id`. The 2025 release uses `mc25-` the same way. Its ids are distinct because 47 of its rubric questions differ from the current release's, so a grade on one release cannot be mistaken for the other.
- **Messages:** the conversation exactly as given, with every role, content string, whitespace and turn order preserved. It ends with the final user turn.
  - Nothing is added: no system message, instruction, tool or example.
  - The setup writes the final assistant reply. The runner asks for one reply with `max_output_tokens` = **16,384** and one sample per item.
  - A leading system message would be kept first, but neither release has one. The official harness accepted only user and assistant roles.
- **Private key** (never published): `source_id`, `axis`, `target_question`, `pass_criteria`.
- **Public fields:** `source_id` (the dataset's `question_id`), `category` (the axis), `messages_sha256` (`core.messages_digest`) and `rubric_sha256` (SHA-256 of the rubric question). They hold no question, conversation or answer text. The rubric digest shows which revision of a rubric was used.

## Judge

**Prompt.** It is `JUDGE_PROMPT` from [`src/evaluator.py`](https://github.com/ekwinox117/multi-challenge/blob/5ccefcca6a39020d66c1383c4e6a809cb07afa33/src/evaluator.py) at the pinned commit, character for character.

- The file has SHA-256 `17f53109b2c7c0d13dafce7a181414467ac0af00209d48b32b9e4bb209d0997b`.
- The prompt is 383 characters with SHA-256 `7519dc6e1af487f8a9850239b334f16238032994b763a8b5108ad67bd0d5be7f`. The tests pin that digest.
- To re-derive it, read the assignment with Python's `ast.literal_eval`, without running the file.

As in the official evaluator, the prompt is filled with `JUDGE_PROMPT.format(answer, target_question)`. `str.format` inserts both texts literally, so braces inside them stay as they are.

```text
You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".
```

The judge sees only the final answer and the rubric question, never the conversation. This is the paper's design (§3.2): the rubric question "only requires the final model response as context to answer". The paper measured 93.95% agreement between GPT-4o with these rubrics and human raters (Table 4).

**Request.** There is one `JudgeRequest` per answer and one attempt:
- name `verdict`;
- the filled prompt as the user message;
- `max_tokens` 4,096, the value the official `Evaluator` passes to its judge;
- this system message, verbatim:

```text
Respond with a single JSON object, with no other text and no code fence, that conforms to this JSON schema (named JudgeResponse):
{"properties": {"reasoning": {"title": "Reasoning", "type": "string"}, "verdict": {"enum": ["YES", "NO"], "title": "Verdict", "type": "string"}}, "required": ["reasoning", "verdict"], "title": "JudgeResponse", "type": "object", "additionalProperties": false}
```

The official judge answered through OpenAI structured outputs with the repository's pydantic model `JudgeResponse(reasoning: str, verdict: Literal["YES", "NO"])`, sent with no system message. `JudgeRequest` has no response-format field. The system message therefore carries the same schema that OpenAI's SDK derives from that model in strict mode, and it is the only text we add.

**Parsing.** The repository parses the judge's reply with `JudgeResponse.model_validate_json`. For such a reply, `parse_verdict` accepts exactly what that accepts:
- one JSON object with a string `reasoning` and a `verdict` of exactly `"YES"` or `"NO"`;
- other keys are ignored, and a repeated key keeps its last value;
- lowercase `"yes"`, padding inside the string, a boolean or a missing `reasoning` is refused.

Python's strict `json.loads` matched pydantic 2.14 on every case in the tests. A single Markdown code fence around the whole object is tolerated. The official judge's constrained decoding could not produce one, and stripping it cannot change a verdict.

The stand-in judge has no constrained decoding, so it does not always return the object alone. Three fallbacks read the verdict it clearly gave, in this order:

1. **The last JSON object with a valid verdict**, after prose. In grader check 1 the judge wrote its reasoning first and the object last in 278 of 819 replies.
2. **The last "Verdict: YES" or "Verdict: NO"**, any case, Markdown emphasis allowed. This matches the official prompt's own words, "Print your reasoning followed by your verdict". 28 of the 819 replies ended this way.
3. **A last line that is only "YES" or "NO"**, after at least one line of reasoning. 75 of the main run's 2,925 replies ended this way; before this fallback existed, they were re-sent as unusable.

Anything else is a failed judge call, for example a bare verdict with no reasoning, or "the answer is YES" inside a sentence. A failed call is re-sent, up to three attempts, and a setup's score is complete only when every answer has a usable reply.

**Judge model.** It is Claude Sonnet 5 (`claude-sonnet-5`), called by the runner through the Message Batches API.
- The paper and the repository use `gpt-4o-2024-08-06` at temperature 0. That needs an OpenAI key we do not have, and OpenAI shuts the model down on 14 Jan 2027 ([deprecations](https://developers.openai.com/api/docs/deprecations)).
- Scale's board has used **Gemini 2.5 Pro** since its 3 Feb 2026 update; the blog says it raised judge–human agreement by more than 5 points.
  - `gemini-2.5-pro` was **retired on Vertex AI on 20 Oct 2026**, with `gemini-3.8-flash` or `gemini-3.5-flash` as replacements ([Vertex model versions](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions)). Our Vertex project returns 404 for it.
  - On the Gemini API it is limited to existing users, and from 1 Dec 2026 requests to it route to `gemini-3.8-flash` ([Gemini API deprecations](https://ai.google.dev/gemini-api/docs/deprecations)).
- The paper reports that it tried Claude as the judge and reached "the same conclusions" (§5.1), but it gives no Claude-judge numbers.
- Sonnet 5 rejects `temperature`, so the judge samples at its default and is not deterministic.
- The module does not set the judge's thinking or effort. The runner should keep one setting for every run, the grader check included.

## Grading one answer

`grade` returns only `axis`, `verdict` (`"YES"`, `"NO"` or null), `passed`, `failed_answer` and `failed_judge_calls` (0 or 1).
- **Answered and judged:** `passed` is `verdict == pass_criteria`, the official rule.
- **Missing answer** (`None`), or an empty or whitespace-only one: no judge call is made, the item does not pass, and `failed_answer` is true. The official harness grades a generation error as a fail; it judges a placeholder text ending "FAIL THIS QUESTION".
- **Failed judge call:** the reply is `None`, absent or unparseable. The item does not pass, `verdict` is null and `failed_judge_calls` is 1. It is never counted as a pass.

## Score

```text
rate[axis] = passed[axis] / n[axis]              over the four axes
score      = 100 × mean(rate[axis] for axes with n > 0)
ci95       = 100 × 1.96 × sqrt(Σ rate·(1 − rate) / n[axis]) / (number of axes with n > 0)
```

**Verification of the formula:**
- **Official code.** `src/result_parser.py` at the pinned commit (SHA-256 `35b807f65c0cccf0c3d7e0aa3b5b81bac808773c19891a399dde6538ba448e95`) computes each axis's pass percentage, then `sum(axis_scores.values()) / len(axis_scores)`. That commit replaced an earlier per-conversation percentage. The README's "percentage of conversations" sentence still describes the old formula.
- **Paper** ([arXiv:2501.17399v2](https://arxiv.org/abs/2501.17399v2), 6 Mar 2025). Every "Average" in Tables 3 and 5 for the twelve stored models is the mean of its four axis percentages to the printed precision. For example, GPT-4o in Table 3 is (12.86 + 5.93 + 14.63 + 18.18) / 4 = 12.90. Llama-3.3-70B in Table 5 is (23/69 + 17/113 + 10/41 + 10/50) / 4 × 100 = 23.19.
- **Board.** It does not state its formula and publishes no per-axis numbers for its rows, so it cannot be recomputed. Its blog reports per-axis accuracies, which fit axis rates on the 266-item split (see Data). The board's ranks use 95% confidence intervals: "1 + the number of models whose lower CI bound exceeds this model's upper CI bound".

**Parts** of the result:
- `axes`, which holds, for each axis, `n`, `passed`, `failed_answers`, `failed_judge_calls` and `score`;
- `n`, `passed`, `failed_answers` and `failed_judge_calls` in total;
- `ci95`.

`ci95` is our normal-approximation half-width. It treats each axis as binomial and the axes as independent, and it ignores judge noise. Scale does not say how it computes its ± values. At a pass rate of 0.787 on every axis, `ci95` is ±5.29 on the current split, next to the board's ±5.30 for Gemini 4 Argon.

**Failures** stay in their own axis as not passed and are reported in the counts. The official harness instead books a judge exception under a fifth axis named "NA" at 0%, which drags the macro-average down sharply. When `failed_judge_calls` > 0, the headline is a lower bound. Re-counting those items as passes gives the upper bound.

**Partial runs.** The mean covers the axes present, as in the official code, and `score` is null for no grades.

## Grader check: measuring the judge offset

The paper's twelve stored response files (2024 models, one response per conversation, no verdicts) are at the pinned commit. `reference_responses(cache_dir)` returns `{model key: {item id: response text}}`, keyed by the ids of `REFERENCE_BENCHMARK.load(cache_dir)`.
- The runner judges them with `REFERENCE_BENCHMARK.judge_requests`, `grade` and `score`, which use the 2025 rubrics the paper scored, not the revised current ones.
- It compares the result with `REFERENCE_SCORES[model]` on the field named by its `compare_on`.
- The texts are for private calibration and must not be published.

`REFERENCE_SCORES` holds the paper's GPT-4o-judged numbers from [arXiv:2501.17399v2](https://arxiv.org/abs/2501.17399v2): Table 3 for the six models that built the set, and Table 5 for open models. Pass counts are derived from the percentages; each percentage matches exactly one integer count.

| Model key (file) | Paper row | Table | IR | IM | RVE | SC | Average | Passes / 273 | Compare on |
|---|---|---|---|---|---|---|---|---|---|
| `gpt-4o-2024-08-06` | GPT-4o (August 2024) | 3 | 12.86 | 5.93 | 14.63 | 18.18 | 12.9 | 30 | passes |
| `llama3-1-405b-instruct-v1` | Llama 3.1 405B Instruct | 3 | 14.29 | 16.95 | 9.76 | 25.0 | 16.5 | 45 | passes |
| `mistral-large-latest` | Mistral Large | 3 | 18.57 | 6.78 | 9.76 | 18.18 | 13.32 | 33 | passes |
| `claude-3-5-sonnet-20241022` | Claude 3.5 Sonnet (June 2024) | 3 | 61.43 | 37.29 | 26.83 | 45.45 | 42.75 | 118 | passes |
| `gemini-1.5-pro-002` | Gemini 1.5 Pro (August 27 2024) | 3 | 31.43 | 16.1 | 19.51 | 18.18 | 21.3 | 57 | passes |
| `o1-preview` | o1-preview | 3 | 28.57 | 38.98 | 39.02 | 36.36 | 35.73 | 98 | passes |
| `llama-3-2-3b-instruct` | Llama-3.2-3B-Instruct | 5 | 15.94 | 8.85 | 36.59 | 6.0 | 16.85 | 39 | score |
| `llama-3-3-70b-instruct` | Llama-3.3-70B-Instruct | 5 | 33.33 | 15.04 | 24.39 | 20.0 | 23.19 | 60 | score |
| `qwen2_72b` | Qwen2-72B-Instruct | 5 | 27.54 | 7.96 | 26.83 | 20.0 | 20.58 | 49 | score |
| `qwen2-5_14b` | Qwen2.5-14B-Instruct | 5 | 15.94 | 15.93 | 24.39 | 12.0 | 17.07 | 45 | score |
| `qwen2-5_72b` | Qwen2.5-72B-Instruct | 5 | 21.74 | 17.70 | 12.20 | 16.0 | 16.91 | 48 | score |
| `mixtral-8x7b-instruct` | Mixtral-8x7B-Instruct-v0.1 | 5 | 13.04 | 7.08 | 12.20 | 12.0 | 11.08 | 28 | score |

- **Table 5** percentages are exact fractions of the released split 69/113/41/50, so compare the macro score.
- **Table 3** percentages are exact fractions of 70/118/41/44. That split is in neither release, so compare total passes over the same 273 conversations. Whether Table 3's rubric wording equals the released wording is not documented.
- **Two stored files may not be the model in their paper row.** The Claude file is `20241022` while Table 3 says June 2024; the board's page says October 2024. The Gemini file is `gemini-1.5-pro-002` while Table 3 says August 27 2024.
- The paper's GPT-4o row was judged by GPT-4o itself.
- The paper's prose quotes human-evaluation figures in places (for example the abstract's 41.4%, which is Table 2), so cite the tables.

| Stored file | SHA-256 |
|---|---|
| `claude-3-5-sonnet-20241022_responses.jsonl` | `d83582385768fdc9006e0882dc9f11cb6ac8cdec06eae3a25a7e2ea88a4761fb` |
| `gemini-1.5-pro-002_responses.jsonl` | `3be040a7bdd20672eba209de31ca27c738de7df9c5af361872db707000b3c59f` |
| `gpt-4o-2024-08-06_responses.jsonl` | `12579bda05fb4dcc7330b093d93cd3eec5d89005f3d1be559fe7457048a31b96` |
| `llama-3-2-3b-instruct.jsonl` | `6750f070f7afc1b8c6ccee255a70e0e14020ce78e665e09247fcb04cc7b07b62` |
| `llama-3-3-70b-instruct.jsonl` | `9c65e6227ac2cf7315be55f72534ff678942f6a0b60d8cc097d74571f0134728` |
| `llama3-1-405b-instruct-v1_responses.jsonl` | `342d6fb0007e84d5c4370ece2994207b34cdd477e1c6c68f053e10df61e3f55d` |
| `mistral-large-latest_responses.jsonl` | `2db7c28bbc171f27fd3ad1f3ce24966436ed9e38b2d35500a00217249706c6c7` |
| `mixtral-8x7b-instruct.jsonl` | `1b30398599d3c7091c1edbde518e8f8e54552daef1896eb73efed78864a526d0` |
| `o1-preview_responses.jsonl` | `94398716d955e961db3b0d1ca676d8209f3246597dcf1ddd9930b3ad5adef255` |
| `qwen2-5_14b.jsonl` | `5e24d7891c9a3b9b1ceac84e1731c115a851d3c08bd146cf650a1263802b6104` |
| `qwen2-5_72b.jsonl` | `7607a7cf69912343016c06aa36bd2b4d43b63866d595826ed8d08e15a94fde21` |
| `qwen2_72b.jsonl` | `5a9c9275db9899a32bb70fe6f46ac7ec19d9ba427a2601a53dbc60048872ff70` |

**Recommendation (`REFERENCE_RECOMMENDED`):** re-judge three models.
- `mixtral-8x7b-instruct` (11.08, 28 passes) for the low end.
- `llama-3-3-70b-instruct` (23.19, 60 passes) for the middle.
- `o1-preview` (35.73, 98 passes) for the high end.

They span 10–36% of conversations, and their model versions match the paper's labels. They avoid the Claude file, which shares the judge's family and has the version doubt, and the GPT-4o file. If only two fit, use Mixtral-8x7B and o1-preview. Each model costs 273 judge requests.

This check measures Sonnet 5 against GPT-4o on the 2025 rubrics, at pass rates of 10–36%. It does not measure Sonnet 5 against Gemini 2.5 Pro, which can no longer be called, or offsets near the board's 67–82 range. Those come from anchors: setups that are also on the board, such as Gemini 4 Argon and DeepSeek V4.1 Flash.

## Deviations from the official protocol

1. **Data.** The setups are scored on the current official release (Hugging Face, 266 conversations, the 2026 revision) rather than the repository's 2025 file of 273. It is very likely the board's set (see Data), which the paper's set is not. The 2025 set is used only for the grader check, because the paper's numbers were scored on it.
2. **Judge model.** Claude Sonnet 5 replaces `gpt-4o-2024-08-06` (paper and repository) and Gemini 2.5 Pro (board). Gemini 2.5 Pro is retired on Vertex AI; GPT-4o needs an OpenAI key we do not have.
3. **Judge sampling.** The judge uses its default temperature, because Sonnet 5 rejects `temperature`; the official judge used 0. Re-judging a sample measures the noise.
4. **Response format.** A system message carries the `JudgeResponse` JSON schema, in place of OpenAI structured outputs. One surrounding code fence is tolerated, and three fallbacks read a verdict given in prose (see Parsing).
5. **Judge failures.** As published, the official harness cannot run: `Evaluator.__init__` passes `max_tokens` to `OpenAIModel`, which raises `TypeError`. With that fixed, a judge exception is booked under a fifth "NA" axis at 0%. Here an unusable reply is re-sent, up to three attempts. A failure that remains is marked in the grade, counted as not passed in its own axis, and reported in `score()`, and the setup's result is marked incomplete.
6. **Attempts.** There is one answer and one judge call per conversation. The README's `--attempts 3` example counts a conversation as passed if any attempt passes. The board does not state its setting; its blog's per-axis figures suggest older rows average three runs.
7. **Blank answers** fail without a judge call. The official harness would send the text to the judge.
8. **Confidence interval.** `ci95` is our construction; the board's method is unpublished.
9. **Setup settings.** Panda's setups run at their product thinking settings and 16,384 output tokens. The board does not show its reasoning settings or output limits, so a gap to a published row mixes the judge, the item revision, the setting and sampling noise.

## License and terms

- **The scored data** (Hugging Face) is under **CC BY 4.0**, according to the dataset card at the pinned revision. Attribution: MultiChallenge, Scale AI (Sirdeshmukh et al., 2025), https://huggingface.co/datasets/ScaleAI/MultiChallenge.
- **The GitHub repository** has no license file, and GitHub reports none. We use its question file, judge prompt and stored responses only locally, for the grader check. We redistribute none of them.
- **Releases** carry item ids, the public fields above and grades. They never carry conversations, rubric questions, answers or judge reasoning.
- **The paper** is under arXiv's non-exclusive license.
- The MIT license of this repository covers our code only.

## Published rows this score sits next to

The rows below come from [Scale's MultiChallenge board](https://labs.scale.com/leaderboard/multichallenge), accessed 18 Nov 2026.
- The judge is Gemini 2.5 Pro, and the items reflect the 54-task revision.
- The board shows no reasoning setting or run count per row.
- "Entry date" is the `createdAt` field in the page's data. Many older rows share 2026-10-07, so it may be a migration date rather than the evaluation date.

| Model | Board label (version id) | Score | ± (95% CI) | Entry date |
|---|---|---|---|---|
| GPT-6.1 Sol | gpt 6.1 Sol (`gpt-6.1-sol`) | 81.95 | 4.16 | 2026-10-07 |
| Gemini 4 Argon | Gemini 4 Argon (`gemini/gemini-4-argon`) | 78.70 | 5.30 | 2026-10-21 |
| Muse Spark | Muse Spark (none shown) | 75.52 | 4.05 | 2026-10-07 |
| DeepSeek V4.1 Flash (1017) | Deepseek V4p1 Flash 1017 (`fireworks_ai/deepseek-v4p1-flash-1017`) | 74.90 | 5.30 | 2026-10-26 |
| GPT 6.1 Terra | GPT 6.1 Terra (`openai/gpt-6.1-terra`) | 74.20 | 5.40 | 2026-10-27 |
| Claude Haiku 5.5 | Claude Haiku 5.5 (`claude-haiku-5-5`) | 67.67 | 5.34 | 2026-10-08 |

Scale revised 54 tasks in 2026: the 47 rewritten rubric questions and 7 removed conversations described under Data. The paper's set and the GitHub file predate that revision and differ from the board's. The current Hugging Face release, which this module scores, matches the revision's size and per-axis figures.

## Sources

- Board: https://labs.scale.com/leaderboard/multichallenge (update note dated 3 Feb 2026)
- Blog: https://labs.scale.com/blog/multichallenge-update (23 Mar 2026)
- Current release: https://huggingface.co/datasets/ScaleAI/MultiChallenge/tree/670fd81f270314d0f8f82b34fcfe2187fd28f699
- 2025 release and harness: https://github.com/ekwinox117/multi-challenge/tree/5ccefcca6a39020d66c1383c4e6a809cb07afa33
- Paper: https://arxiv.org/abs/2501.17399v2 (Tables 1–5, §3.2, §5.1)
- Judge availability:
  - https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
  - https://ai.google.dev/gemini-api/docs/deprecations
  - https://developers.openai.com/api/docs/deprecations
