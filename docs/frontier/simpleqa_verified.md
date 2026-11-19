# SimpleQA Verified

`simpleqa-verified` asks the 1,000 SimpleQA Verified short factual questions. Its headline is the Kaggle board's score, an F1 on a 0–100 scale. Each question goes to the model alone, with no tools. One grader call per answer returns A, B or C, meaning CORRECT, INCORRECT or NOT_ATTEMPTED. The F1 is the harmonic mean of accuracy and accuracy given attempted. The board and the official starter notebook grade with GPT-4.1. This project's runner sends the identical grader prompt to Claude Sonnet 5 instead, so the published rows are same-scale comparison points with a different grader.

Sources, all accessed 2026-11-18:

- [Kaggle board](https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified)
- [Starter notebook, Version 9](https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code)
- [Paper, arXiv:2509.07968 v2](https://arxiv.org/abs/2509.07968)
- [Dataset card](https://huggingface.co/datasets/google/simpleqa-verified/blob/0dc97e0d28d8233463e005cdc4475cc2a13ba2dc/README.md)

## Data

| | |
|---|---|
| Source | Hugging Face `google/simpleqa-verified`, not gated |
| Pinned revision | `0dc97e0d28d8233463e005cdc4475cc2a13ba2dc` (repository head, 2026-03-09) |
| Download | `https://huggingface.co/datasets/google/simpleqa-verified/resolve/0dc97e0d28d8233463e005cdc4475cc2a13ba2dc/simpleqa_verified.csv`, fetched with `core.fetch_pinned` into the runner's `cache/simpleqa-verified/` |
| `simpleqa_verified.csv` | 349,184 bytes, SHA-256 `b5db21155444763543fe31b67e7cf28ce2bb225742a5b889421c5f182e2f92f5`, git blob `89f2a3fdf6395038ec45721a090ff59cd37ed375` |
| Rows | 1,000, unique `original_index`, in ascending order |
| Topics | Politics 176, Science and technology 160, Art 145, Sports 117, Geography 111, Music 102, Other 102, History 52, TV shows 20, Video games 15 |
| Answer types | Other 249, Date 222, Person 198, Number 185, Place 146 |

The board describes the benchmark as this 1,000-prompt set. The notebook loads the same file name from Kaggle's copy, `kagglehub` handle `deepmind/simpleqa-verified`. The pinned revision's commit, "Edit to fix an error in dataset", changed one word in one question, `original_index` 55. Otherwise the file equals the first upload of 2025-09-22, commit `e021cfed345ed3b6135dc501d7a91cab056558fa`, SHA-256 `fb990cca0fae8515ae2f51b794d5fa791d095c1f919ecdbe22d0c1642e9ab966`. The fix is in the open copy. Kaggle's copy was not checked, because this project uses no Kaggle account and accepts no Kaggle terms. So at most this one item may differ from the board's data.

The loader reads the file with Python's `csv` module and keeps values as stored. The notebook reads it with pandas `read_csv`. With pandas 2.2.3, the notebook's version, both readers give identical `problem` and `answer` strings for all 1,000 rows. No value is one of pandas' default missing-value strings. Two gold answers carry surrounding whitespace and one contains a line break; both readers keep them.

## Items

- **ID:** `sqav-` followed by the first 16 hex digits of SHA-256 over `original_index` exactly as written in the file, for example `5`.
- **Public fields:** `source_id` (the `original_index`), `category` (`topic`), `answer_type` and `messages_sha256` (`core.messages_digest` of the conversation). They hold no question or answer text.
- **Private key:** `question` (the `problem` text) and `target` (the gold `answer`).

## Prompt to the model

The model receives the question text alone as its only user message. There is no system prompt, no instruction and no example. The notebook sends exactly that:

```python
question = data_row.get("problem", "")                                       # cell In[20]
predicted_answer_text = generate(prompt=question, model=model_name_to_test)  # cell In[20]
response = google_client.models.generate_content(model=TEST_MODEL, contents=prompt)  # cell In[14]
```

The paper says: "All models are evaluated with their standard API parameters and without tools." The dataset card adds that the benchmark "is meant to be be used without any tools (i.e. search or retrieval tools)."

## Grader

**Prompt.** The module's `GRADER_TEMPLATE` is the notebook's `GRADER_TEMPLATE` from cell In[16], Version 9, character for character. It includes the notebook's two-space indentation: the string's lines are indented in the notebook source, and `.strip()` trims only the ends. The text was taken from the notebook's rendered code, whose `<pre>` blocks keep whitespace.

| | |
|---|---|
| Length | 7,125 characters in 84 lines |
| SHA-256 (UTF-8) | `84c004ec4fcf8f0703bb0d734544036a72e847bfa7116429e5aee5e53ccc8cf3` |
| Placeholders | `{question}`, `{target}` and `{predicted_answer}`, and nothing else |

The text matches the "SimpleQA Verified Evaluation Prompt" in the paper's Appendix A word for word. The paper prints it without the indentation. It differs from OpenAI's original SimpleQA grader as the paper describes. Hedged answers must commit to a single candidate, numeric targets carry an acceptable range, and only the direct answer is graded. The prompt ends:

```text
  Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
  ```
  Question: {question}
  Gold target: {target}
  Predicted answer: {predicted_answer}
  ```

  Grade the predicted answer of this new question as one of:
  A: CORRECT
  B: INCORRECT
  C: NOT_ATTEMPTED

  Just return the letters "A", "B", or "C", with no text around it.
```

**Request.** Each non-empty answer gets one `JudgeRequest`, named `grade`. It is `GRADER_TEMPLATE.format(question=problem, target=answer, predicted_answer=reply)`, sent as the grader's only user message with no system prompt. The fill matches the notebook's `format_grader_prompt` in cell In[17], and braces inside the values are not re-read. The notebook calls the grader with `messages=[{"role": "user", "content": prompt}]` and default sampling. The request keeps the interface's default limit of 2,048 output tokens, although the expected reply is one letter.

**Reading the reply.** `parse_grade()` reproduces this excerpt of `grade_answer_with_llm` from cell In[18], except for its last line:

```python
match = re.search(r"(A|B|C)", grading_response_text)
if match: return match.group(0)
if "CORRECT" in grading_response_text.upper(): return "A"
if "INCORRECT" in grading_response_text.upper(): return "B"
if "NOT_ATTEMPTED" in grading_response_text.upper(): return "C"
return DEFAULT_GRADE_IF_UNPARSEABLE   # the notebook's "C"; parse_grade() returns None
```

Its rules are kept as they are, quirks included:

- The first capital A, B or C anywhere in the reply wins. A reply of "CORRECT" therefore reads as C, and "NOT_ATTEMPTED" reads as A.
- Lowercase "incorrect" reads as A, because the `CORRECT` test comes first.
- A grader that follows the prompt and answers with a bare letter never meets these quirks.

Each grade records `bare_judge_reply`, which is true when the stripped reply is exactly A, B or C. `score()` counts `nonbare_judge_replies`, so a run in which the grader strayed is visible. That count should be zero; otherwise, inspect those items.

## Grades

A grade holds `grade_letter`, `verdict` (CORRECT, INCORRECT, NOT_ATTEMPTED or null), `correct`, `attempted`, `failed_answer`, `failed_judge_calls` (0 or 1) and `bare_judge_reply`. It holds no question, gold answer or reply text.

- **Missing or empty answer.** `None` from a failed request, or an empty or whitespace-only reply, makes no grader call. It is scored NOT_ATTEMPTED with `failed_answer` true. The grader template defines NOT_ATTEMPTED as an answer that lacks the gold information and contradicts nothing, which describes an empty reply. The notebook has no failure path, because a failed call raises.
- **Failed grader call.** When the grader call failed (`None`), its reply is missing, or the reply is unreadable by the rules above, the grade has `verdict` null and `failed_judge_calls` 1. It is never counted as correct or attempted, and it stays in `n`. That scores it like the notebook's default C, but the failure is marked and counted rather than hidden. The runner should retry failed grader calls, so a release reports zero.

## Score

```text
n          = number of items (failures included)
correct    = items graded CORRECT
attempted  = items graded CORRECT or INCORRECT
accuracy   = correct / n
given      = correct / attempted              (0 when attempted = 0)
F1         = 2 × accuracy × given / (accuracy + given)   (0 when both are 0)
score      = 100 × F1
```

This is the notebook's `get_f1_score` and `get_accuracy_given_attempted` from cell In[22]. The paper describes the same metric: "the harmonic mean (F1-Score) of overall correct … and correct given attempted". As a check, the paper's Table 7 gives Gemini 2.5 Pro accuracy 55.3 and given-attempted 55.9, which produce F1 55.6, as printed. The unit tests rebuild that row from 553 correct, 436 incorrect and 11 not attempted.

`score()` also returns `accuracy`, `correct_given_attempted` and `attempted_rate`, all ×100. It returns the counts `n`, `correct`, `incorrect`, `not_attempted`, `failed_answers`, `failed_judge_calls` and `nonbare_judge_replies`. The counts partition `n` as correct + incorrect + not attempted + failed grader calls. The paper's Table 7 also shows a "Hedged" column, which equals 100 minus the attempted rate. The board shows a ± interval beside each score but does not publish its method. Its values are close to a binomial 95% interval over 1,000 items but not equal to one: across the 57 scored rows they differ from it by −0.1 to +0.8 points. This module does not compute one.

## Deviations from the official protocol

| | Board and notebook | This module and runner | Why |
|---|---|---|---|
| Grader model | `gpt-4.1-2025-04-14` | Claude Sonnet 5 (`claude-sonnet-5`) through the Batch API, with the same prompt | Grading with GPT-4.1 needs an OpenAI key this project does not use for grading, and OpenAI shuts GPT-4.1 down on 2027-01-14. None of the Panda setups scored here uses a Claude model, so the grader does not grade its own model family. GPT-4.1 is not a reasoning model, so the runner should call the grader without extended thinking. Claude Sonnet 5 rejects `temperature`, and the notebook sets none. [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations) |
| Unreadable grader reply | Defaults to C and prints a warning | Failed grader call: scored like C, marked and counted | Failures are never hidden |
| Missing or empty answer | No failure path | NOT_ATTEMPTED without a grader call | Nothing to grade |
| Output limit | Each provider's default | 16,384 tokens | Panda's API maximum |
| Reasoning settings | Each listed model's own, not published per row | The runner's Panda product settings | Panda's product fixes them per model |
| Dataset copy | Kaggle's copy | Hugging Face revision `0dc97e0d` | Open and revision-pinned. It may differ from the board's copy in one word of one question. |
| Interval | ± shown, method unpublished | None computed | Method unknown |

## License and terms

- **Dataset:** MIT, from the dataset card (`license: mit`). It was built from OpenAI's SimpleQA, which is also MIT. No request restricts publishing examples. This project still releases no question or answer text, since releases never carry prompt bodies.
- **Grader template:** The notebook page says it "has been released under the Apache 2.0 open source license". The module reproduces the template unchanged, with attribution and a link to the license, <https://www.apache.org/licenses/LICENSE-2.0>.
- **Citation:** Haas, Yona, D'Antonio, Goldshtein and Das, "SimpleQA Verified: A Reliable Factuality Benchmark to Measure Parametric Knowledge", [arXiv:2509.07968](https://arxiv.org/abs/2509.07968).

## Published rows

These rows come from the Kaggle board, [kaggle.com/benchmarks/deepmind/simpleqa-verified](https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified). The board was marked "Updated October 20, 2026" and was read on 2026-11-18, with deprecated models shown. It lists 58 models, 57 of them with a score, all graded by GPT-4.1, with the F1 score and the board's ± interval. The board does not show reasoning settings or run dates.

| Rank | Model | F1 | ± |
|---:|---|---:|---:|
| 1 | Gemini 3.1 Pro Preview | 77.5 | 2.6 |
| 2 | GPT-6 Astra | 75.8 | 2.6 |
| 3 | Gemini 4 Argon | 75.6 | 2.6 |
| 4 | Gemini 3.8 Flash | 74.6 | 2.7 |
| 5 | GPT-6.1 Sol | 73.8 | 2.8 |
| 6 | Gemini 3.7 Flash | 73.2 | 3.0 |
| 7 | Gemini 3 Pro Preview | 72.1 | 2.9 |
| 8 | Gemini 3.5 Flash | 70.4 | 2.9 |
| 9 | GPT-5.6 Sol | 69.2 | 2.9 |
| 10 | GPT-6 Sol | 64.4 | 3.0 |
| 11 | Claude Opus 5 | 63.3 | 3.2 |
| 12 | Grok 4.7 | 59.4 | 3.2 |
| 21 | DeepSeek V4.1 Flash | 42.3 | 3.2 |
| 28 | Claude Sonnet 5 | 32.1 | 3.3 |

GLM-5.3 is not on the board. Gemini 4 Argon, Gemini 3.8 Flash and DeepSeek V4.1 Flash are on it, so a run of those models through this module measures the combined offset of the grader swap and Panda's settings against the board.

## Verification

- `tests/frontier/test_simpleqa_verified.py` runs offline on synthetic rows. It checks the pins and the template's hash, length and placeholders. It checks that the bare question is the only message and that one request is built with the template filled once. It covers the notebook's parse rules and their quirks, every grade path including missing answers and failed or unreadable grader replies, a hand-computed F1, the paper's Table 7 row, and the absence of question and answer text from public fields, grades and scores.
- During implementation, the module was compared with the notebook's own cells In[16] to In[18] and In[22], executed with a stub in place of the model call. The comparison checked the template, three filled prompts, 20 curated replies and 200,000 generated replies, and F1 on 2,000 random grade sets using pandas 2.2.3. They matched, with this module's None standing where the notebook defaults to C.
- A load of the pinned file produced 1,000 items with unique IDs and one user message each, identically under Python 3.9.6 and 3.13. Each filled grader prompt is 7,089 characters of template plus the question, gold answer and reply.
