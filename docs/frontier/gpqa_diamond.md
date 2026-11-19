# GPQA Diamond

`gpqa-diamond` asks the 198 GPQA Diamond multiple-choice questions in biology, chemistry and physics and reports accuracy on a 0–100 scale. It follows Epoch AI's Benchmarking Hub protocol, so its score can sit beside Epoch's rows. Grading is exact. The reply's last `ANSWER: <letter>` line must name the correct option, and no judge model is involved.

Epoch runs the benchmark in [Inspect](https://inspect.aisi.org.uk/) with "an empty system prompt for all models", zero-shot chain-of-thought prompting and each model's API-default temperature. It runs most models "16 times on GPQA Diamond". Its GPQA page states: "If the model's answer is not of the form "ANSWER: LETTER", it receives no points for that question." [Epoch FAQ](https://epoch.ai/benchmarks/about) · [Epoch GPQA Diamond page](https://epoch.ai/benchmarks/gpqa-diamond) (both accessed 2026-11-18)

## Data

| | |
|---|---|
| Source | GitHub `idavidrein/gpqa`, the GPQA authors' repository |
| Pinned commit | `56686c06f5e19865c153de0fdb11be3890014df7` (repository head, 2024-09-30). `dataset.zip` last changed in `d46dc8d5`, 2023-11-20. |
| Download | `https://raw.githubusercontent.com/idavidrein/gpqa/56686c06f5e19865c153de0fdb11be3890014df7/dataset.zip`, fetched with `core.fetch_pinned` into the runner's `cache/gpqa-diamond/` |
| `dataset.zip` | 2,348,038 bytes, SHA-256 `461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc` |
| Member read | `dataset/gpqa_diamond.csv`, decrypted in memory and never written to disk |
| `gpqa_diamond.csv` | 1,373,492 bytes, SHA-256 `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`, 198 rows, 78 columns |
| Domains | Chemistry 93, Physics 86, Biology 19 |

The zip uses traditional ZipCrypto encryption to keep the questions away from web crawlers. The repository's README publishes the password, `deserted-untie-orchid`, and Python's `zipfile` opens it with `pwd`. [README, line 7](https://github.com/idavidrein/gpqa/blob/56686c06f5e19865c153de0fdb11be3890014df7/README.md)

The CSV is byte-identical to simple-evals' public copy at `https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv`. inspect_evals pins that copy with the same SHA-256, and Artificial Analysis cites it. Epoch's published task loads the gated Hugging Face copy, `Idavidrein/gpqa`, config `gpqa_diamond`. That copy has the same file name and size. Its hash cannot be checked without accepting the gate's terms, which this project does not do. [inspect_evals GPQA task](https://github.com/UKGovernmentBEIS/inspect_evals/blob/81edb2765c0bce51ff6fcd20a951d22e20bcb6e6/src/inspect_evals/gpqa/gpqa.py)

The loader reads `Record ID`, `Question`, `Correct Answer`, `Incorrect Answer 1` to `3`, `High-level domain` and `Subdomain`, in file order. It uses the values as stored, without trimming, as Epoch's and inspect_evals' loaders do. Twenty-five questions end in whitespace. Two items, `recZSGUkn56v9kEp1` and `recnGEpF1srQpaqWq`, show two identical wrong options. The correct option is distinct in every row, so the answer key is unambiguous.

## Items

- **ID:** `gpqa-` followed by the first 16 hex digits of SHA-256 over the UTF-8 `Record ID`.
- **Public fields:** `source_id` (the `Record ID`), `category` (`High-level domain`), `subcategory` (`Subdomain`) and `messages_sha256` (`core.messages_digest` of the conversation). They hold no question or option text.
- **Private key:** `correct_letter`, the letter under which the correct option appears.

### Choice order

GPQA's CSV always lists the correct answer first, so the options must be shuffled. Here the order is fixed per item and derived from the public item ID:

```python
options = [Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3]  # indexes 0-3
order = sorted(range(4), key=lambda k: sha256(f"{item_id}:{k}".encode()).hexdigest())
# option order[0] is shown as A, order[1] as B, order[2] as C, order[3] as D
```

On the pinned data the correct option lands on A 47 times, B 46, C 53 and D 52. SHA-256 is used instead of `random.Random(seed).shuffle` because Python guarantees only `random()` across versions, not `shuffle`. Anyone can recompute the order from the public ID in any language.

**How Epoch differs.** Epoch's published task shuffles with Python's global, unseeded `random.shuffle` while loading the dataset. Each run therefore gets its own order, and its 16 samples per question share that order. inspect_evals, which Epoch says it builds on, switched to a fixed seed of 42 on 2026-09-10 (commit `700904ddee2ce14b069f05c941c8e3a6242782dd`). Here every run and every setup sees the identical prompt, so setups can be compared item by item. Like each Epoch run, a single run uses one order per question, so a model's preference for a letter is not averaged out. [Epoch's task gist](https://gist.github.com/tadamcz/a61515465e34a3c66f3a78673502bc3f)

## Prompt

The setup receives one user message and no system message:

```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of A,B,C,D. Think step by step before answering.

{question}

A) {option shown as A}
B) {option shown as B}
C) {option shown as C}
D) {option shown as D}
```

In code, `PROMPT_TEMPLATE.format(letters="A,B,C,D", question=Question, choices=...)` builds the message. `choices` is `"\n".join(f"{letter}) {option}")` over the four options in display order, exactly as Inspect's `answer_options()` renders them. The template string with its `{letters}`, `{question}` and `{choices}` placeholders is 240 characters, SHA-256 `252d1efff06346836376902f3125e8f98f95b148b99161c285b9230840c1cf66`. The questions and options are never published.

Where the text comes from:

1. **Inspect's `SINGLE_ANSWER_TEMPLATE_COT`**, commit `7a168f5ff95496bcd8bb237247f56cbf9d366437` (2026-11-08), reads `'ANSWER: $LETTER'`. It has been unchanged since at least 2025-05-07. [Source, line 25](https://github.com/UKGovernmentBEIS/inspect_ai/blob/7a168f5ff95496bcd8bb237247f56cbf9d366437/src/inspect_ai/solver/_multiple_choice.py#L25)
2. **Epoch's published task** (gist revision `69ce610eab3ffc263ade42676e29c478087dd658`, 2025-02-03) passes the same text as `SIMPLE_COT_TEMPLATE` to `multiple_choice()`. [Gist](https://gist.github.com/tadamcz/a61515465e34a3c66f3a78673502bc3f)
3. **Epoch's changelog for 2026-02-20** states: "We updated the GPQA answer format from `ANSWER: $LETTER` to `ANSWER: LETTER` to match the format parsed by our evaluation pipeline. With this minor change, the benchmark version was bumped to 1.0.6." Every Epoch row below was started after that date. [Epoch GPQA Diamond page, Changelog](https://epoch.ai/benchmarks/gpqa-diamond)

The module therefore uses Inspect's text with the one `$` removed. Epoch has not published its version 1.0.6 code, so this single edit is an inference from the changelog. Epoch's page displays the template with typographic quotes and "one of ABCD". It is a paraphrase, not the sent text. An archived copy from 2026-01-01 shows the page then quoted simple-evals' `'Answer: $LETTER'`, while Epoch's published task sent Inspect's `'ANSWER: $LETTER' … one of A,B,C,D`. The module keeps Inspect's straight quotes and `A,B,C,D`. [Archive, 2026-01-01](http://web.archive.org/web/20260101234522/https://epoch.ai/benchmarks/gpqa-diamond) · [Archive, 2026-02-23](http://web.archive.org/web/20260223114357/https://epoch.ai/benchmarks/gpqa-diamond)

## Grading

`extract_letter()` ports the single-answer branch of Inspect's `parse_answers()` at the same commit. `grade()` applies the rule of Inspect's `choice()` scorer. [parse_answers, line 82](https://github.com/UKGovernmentBEIS/inspect_ai/blob/7a168f5ff95496bcd8bb237247f56cbf9d366437/src/inspect_ai/solver/_multiple_choice.py#L82) · [choice scorer](https://github.com/UKGovernmentBEIS/inspect_ai/blob/7a168f5ff95496bcd8bb237247f56cbf9d366437/src/inspect_ai/scorer/_choice.py#L62)

1. **Unwrap.** Three substitutions run in order. `\x24\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\x24` turns `$X$` into `X`. `\*\*\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\*\*` turns `**X**` into `X`. `(?i)(^|[:,])(\s*)\(([A-Za-z\d])\)(?=\s*(?:,|\.|\n|\Z))` turns `(X)` after a colon or comma into `X`.
2. **Strict pattern.** Apply `(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)` with `re.MULTILINE` and keep the **last** match.
3. **Fallback.** Only when the strict pattern finds nothing, apply `(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)` without `MULTILINE` and keep the last match. The fallback is not simply the strict pattern without `^`. It also drops the `\s*` before the terminator and accepts any non-word character as one.
4. **One letter.** Strip the match, drop trailing periods and uppercase it. Split on commas and drop empty pieces. The answer counts only if exactly one piece remains and it is A, B, C or D.
5. **Correct** when that letter equals the key's `correct_letter`.

As a result, hedges and multiple letters score wrong: `ANSWER: B or C`, `ANSWER: B, C`, `ANSWER: BC` and `ANSWER: B and C`. So do replies without a usable answer line, such as `**ANSWER:** B`, `\boxed{B}` and "the answer is (D)". `ANSWER: b`, `ANSWER: $B$`, `**ANSWER: B**`, `ANSWER: (C)` and `ANSWER: A,` all score as their letter. If a reply revises itself, its last answer line counts.

**A missing answer scores wrong.** The runner passes `None` for a failed request. That case, and an empty or whitespace-only reply, gets no letter and stays in the denominator. Inspect scores an empty completion as no answer, which its `value_to_float` maps to 0. Epoch's Grok 4 note says requests that failed on network errors went straight to scoring, where they were generally marked incorrect.

**What a grade publishes:** `letter` (the extracted letter or null), `correct_letter`, `correct`, `failed_answer`, `extraction_failed` (a non-empty reply without a usable letter) and `failed_judge_calls` (always 0). The correct letter refers to this module's fixed option order and reveals nothing without the prompt.

**Parser history.** Inspect's two patterns have been unchanged since at least 2025-08-05, but the code around them changed in 2026:

| Commit | Date | Change |
|---|---|---|
| `846da97f9f6a56e21ffbc864f6235f7caaa6a7d3` | 2026-05-12 | The last match counts instead of the first, and lowercase letters are accepted |
| `127da5b93b5bd4d8ee88b81934325fa58816eab0` | 2026-08-16 | A stray trailing comma, `ANSWER: A,`, is tolerated |
| `ff06b985ed6c4d0ce59d6127b89d0f6b6b27daca` | 2026-09-09 | The `$X$`, `**X**` and `(X)` unwrapping pass |

Epoch does not publish the Inspect version of each run. Its FAQ says it plans to pin runs to exact revisions. Rows started before 2026-09-09 may have used one of the older parsers in the table, which score slightly fewer replies.

## Score

```text
score  = 100 × correct / n                     (n = all items, failures included)
stderr = 100 × sqrt(p × (1 − p) / (n − 1))     (p = correct / n; 0 when n < 2)
```

`stderr` equals Inspect's `stderr()` metric, the sample standard deviation with `ddof=1` divided by √n, for one sample per question. `score()` also returns `n`, `correct`, `extraction_failures`, `failed_answers` and `failed_judge_calls`, which is always 0. Epoch's ± is one standard error after averaging 16 samples per question, so it is smaller than this module's for the same accuracy. At 90% accuracy over 198 questions, one sample gives about ±2.1. Do not compare the error bars directly. [Inspect stderr](https://github.com/UKGovernmentBEIS/inspect_ai/blob/7a168f5ff95496bcd8bb237247f56cbf9d366437/src/inspect_ai/scorer/_metrics/std.py#L122)

## Deviations from Epoch's protocol

| | Epoch | This module | Why |
|---|---|---|---|
| Samples per question | 16 for most models | 1 | Budget. The runner may repeat runs; the order stays fixed. |
| Output limit | No maximum by default | 16,384 tokens | Panda's API maximum. Long reasoning replies that hit it lose their answer line and score wrong. |
| Reasoning effort and temperature | Each row's own effort, mostly high or max; API-default temperature | Set by the runner, not by this module. Panda setups run at their fixed product settings. | Panda's product fixes them per model. |
| Option order | Unseeded random order per run | Fixed SHA-256 order per item | Reproducible, identical across setups, and checkable from public IDs |
| Prompt text | Version 1.0.6 code unpublished | Inspect's text with `$` removed, per Epoch's changelog | Closest verifiable text |
| Parser version | Not published per run | Inspect `7a168f5f` | Newest; history above |
| Data copy | Gated Hugging Face copy | Ungated GitHub zip, same CSV as simple-evals | No terms to accept |

## License and terms

- **GPQA data:** CC BY 4.0. The zip's `license.txt` states "GPQA (c) by Irving David Rein" and "GPQA is licensed under a Creative Commons Attribution 4.0 International License." The repository's code is MIT. Cite Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark", [arXiv:2311.12022](https://arxiv.org/abs/2311.12022).
- **Request not to publish examples:** The authors ask that examples not be revealed online. The gated Hugging Face copy makes users agree "to NOT reveal examples from this dataset in plain text or images online, to reduce the risk of leakage into foundation model training corpora." The GitHub zip has no click-through, but this module follows the request anyway. Releases carry IDs, categories, digests and letters, never questions, options or transcripts. [Dataset card](https://huggingface.co/datasets/Idavidrein/gpqa)
- **Inspect code:** The template and parser come from `inspect_ai`, MIT License, Copyright (c) 2024 UK AI Security Institute. The full notice is in the module.
- **Epoch rows:** Epoch's data is "free to use, distribute, and reproduce provided the source and authors are credited" under CC BY. Source: Epoch AI, "Capabilities & benchmarking".

## Published rows

These are Epoch AI's Benchmarking Hub rows, from [epoch.ai/benchmarks/gpqa-diamond](https://epoch.ai/benchmarks/gpqa-diamond). The data file is [benchmark_data.zip](https://epoch.ai/data/benchmark_data.zip), `gpqa_diamond.csv`, 340 rows, with zip SHA-256 `4e3e52330e276cc63f4b304dc05f6575ede8da8e5e468c74e544eaaf0b71f379`. The data was accessed on 2026-11-18. Scores are Epoch's `mean_score × 100` ± `stderr × 100`. The date is Epoch's "Started at" date for the run. The suffix is the reasoning effort Epoch used.

| Epoch row | Score | ± | Run started |
|---|---:|---:|---|
| gpt-6.2-astra_max | 96.84 | 1.23 | 2026-10-27 |
| gemini-4-carbon_high | 96.53 | 1.21 | 2026-11-02 |
| gemini-4-argon_high | 96.40 | 1.19 | 2026-10-13 |
| claude-fable-5-2-fallback_max | 96.02 | 1.31 | 2026-11-06 |
| gpt-6-astra_max | 95.77 | 1.37 | 2026-08-30 |
| claude-sonnet-5-5_max | 95.58 | 1.37 | 2026-09-29 |
| gpt-6.1-sol_max | 95.39 | 1.38 | 2026-09-29 |
| gemini-3.8-flash_high | 95.39 | 1.40 | 2026-09-02 |
| gemini-3.1-pro-preview_high | 94.44 | 1.63 | 2026-08-06 |
| gpt-6.1-terra_max | 94.32 | 1.49 | 2026-10-26 |
| gpt-6-sol_max | 94.32 | 1.49 | 2026-09-22 |
| claude-opus-5-5-fallback_max | 94.07 | 1.52 | 2026-09-22 |
| claude-fable-5-2_max | 93.88 | 1.63 | 2026-11-06 |
| deepseek-v4.1-flash-1017_max | 93.18 | 1.57 | 2026-10-19 |
| glm-5.3_max | 90.91 | 1.59 | 2026-08-24 |
| claude-opus-5-5_max | 90.59 | 1.94 | 2026-09-22 |
| gpt-6-luna_max | 90.47 | 1.85 | 2026-09-22 |
| deepseek-v4.1-flash_max | 89.84 | 1.73 | 2026-10-09 |
| gemma-4-26b-a4b-it_minimal | 73.23 | 3.15 | 2026-08-06 |

Epoch lists Claude rows with and without a fallback model; quote the matching one consistently. Epoch has no low-effort or thinking-off rows for Gemini 4 Argon, Gemini 3.8 Flash, GLM-5.3 or DeepSeek V4.1 Flash. A setup running one of these models at a lower effort than Epoch's is therefore compared with a stronger configuration of the same model. The board is near its ceiling: its top eight rows lie within 1.5 points, about one standard error.

## Verification

- `tests/frontier/test_gpqa_diamond.py` runs offline on synthetic rows. It packs them in an in-memory ZipCrypto archive that `zipfile` must decrypt with the pinned password. The tests check the exact rendered prompt, the template against Inspect's text, the option order, 23 parser cases, every grade path, a hand-computed score and the absence of question and option text from public fields, grades and scores.
- During implementation, the parser and prompt rendering were compared with Inspect's own `parse_answers()`, `prompt()` and `answer_options()` source at the pinned commit. The comparison covered 33 curated replies and 600,000 generated ones, 73,011 of which contained a usable letter, and found no disagreements.
- A load of the pinned zip produced 198 items with unique IDs, one user message each, identically under Python 3.9.6 and 3.13.
