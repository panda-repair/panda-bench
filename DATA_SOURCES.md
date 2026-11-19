# Data sources and artifact scope

The MIT license covers this kit's code and documentation. Upstream dataset rights remain separate. The archived model outputs are provided for benchmark inspection; this does not assert ownership of any underlying material quoted by a model.

## WildBench

Source: [allenai/WildBench](https://huggingface.co/datasets/allenai/WildBench). The [pinned dataset card](https://huggingface.co/datasets/allenai/WildBench/raw/26c49eb39d7d5ce2099b0bbafed5a88dcce954ec/README.md) specifies CC BY, links [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and describes research/educational use under Ai2 Responsible Use Guidelines. Retain attribution to the WildBench authors and follow upstream use terms. The separate upstream code license does not replace dataset terms.

The pinned revision is `26c49eb39d7d5ce2099b0bbafed5a88dcce954ec`. Both files at this revision match the original downloaded file hashes. The originally requested historical revision was not recorded; byte identity is the evidence for this pin.

| Config / split | Parquet SHA-256 | Bytes |
| --- | --- | ---: |
| v2 / test | e1fbdc5d46deca34e77f92991c23b0476860e26adf2bf28379a3bb3bf00d3ed6 | 3695374 |
| v2-hard / test | 44e2a0d20f791c51d68083c8acc8916698d4bb8d460d1dbc393b54030dbea40a | 402410 |

Reconstruction uses `session_id` (prefix `wb-`), single-turn `conversation_input[0].content`, and `list(checklist or [])`, selecting the exact source config for each released prompt. The manifests include prompt/checklist hashes but no prompt bodies. Downloaded text stays in ignored `data/`.

## Archived releases

`answers.jsonl` contains generated outputs, without source prompt fields, credentials, service IDs or raw error bodies. A whole-prompt normalized substring check and conservative private-path/credential-pattern scan were applied. Any withheld output is identified in `omissions.json`; checks use numeric grades and remain possible even when text is withheld. This screening is a limited heuristic, not a claim that outputs cannot contain quotations or personal data. The local owner requested inclusion for transparency; publication still requires their review.

`usage.jsonl` records request-level costs and the calls that contribute to them. Archived call costs were rounded to six decimal USD before aggregation; recomputing from retained token counts and manifest rates therefore agrees within half a microdollar, rather than bit-for-bit. Cached member calls appear in multiple configurations; summing all rows is not actual experiment spend. Some combiner input token counts were not retained and are null rather than invented. Error outcomes can have unreported provider charges. Per-attempt retries and invoices are not reconstructed.

Stage 1 data, competition problems and code tests are not part of these three releases.

## Frontier release (November 2026)

`releases/frontier-2026-11/` carries item ids, each benchmark's public fields (source id, category and similar labels), prompt digests, grades, statuses, costs and timings. It carries no prompt, rubric, reference answer, answer option, model answer, Panel draft or judge reply. Answers appear only as SHA-256 digests and lengths. The release writer refuses long strings, credential shapes and local paths, and a scan of the final files found no 40-character run of any question or reference. The data is fetched from these pins, hash-checked, and kept under the ignored cache:

| Benchmark | Source and pin | License | Terms we follow |
| --- | --- | --- | --- |
| HealthBench Professional | [openai/healthbench-professional](https://huggingface.co/datasets/openai/healthbench-professional) at `349962fd46dd02343a0d8a606491baf59154ea1a` | MIT (dataset card) | The card asks that examples not be revealed in plain text or images online; none are. Grader template from `openai/simple-evals` at `652c89d0`, MIT |
| MultiChallenge | [ScaleAI/MultiChallenge](https://huggingface.co/datasets/ScaleAI/MultiChallenge) at `670fd81f270314d0f8f82b34fcfe2187fd28f699` | CC BY 4.0 | Attribution: MultiChallenge, Scale AI (Sirdeshmukh et al., 2025). The 2025 GitHub set has no license; it is used locally for the grader check only and not redistributed |
| SimpleQA Verified | [google/simpleqa-verified](https://huggingface.co/datasets/google/simpleqa-verified) at `0dc97e0d28d8233463e005cdc4475cc2a13ba2dc` | MIT | Grader template from the Kaggle notebook, Apache-2.0. Cite Haas et al., arXiv:2509.07968 |
| GPQA Diamond | [idavidrein/gpqa](https://github.com/idavidrein/gpqa) at `56686c06f5e19865c153de0fdb11be3890014df7` (`dataset.zip`) | CC BY 4.0 (Rein et al.) | The authors ask that examples not be revealed online; none are. Prompt and parser from Inspect (`inspect_ai`), MIT |
| RouterArena `sub_10` | [RouteWorks/RouterArena](https://huggingface.co/datasets/RouteWorks/RouterArena) at `a4a062ce3313b56bb09c042e1bc37b61d34e3bd8`; evaluator at `00cb25a470165b846a24bf3f2112d22dfc5758ce` | Code Apache-2.0. The dataset card declares no license; its queries come from about 40 public datasets under their own terms | Evaluation-only, as RouterArena asks: ids, labels and digests only. LiveCodeBench tests from `lighteval/code_generation_lite` at `89e5fc5c` (card: cc) |

The judge's replies and the models' answers are kept privately with the run. Their digests in the release let anyone check a later disclosure against what was graded.
