# Experiment log

This file is the living scientific record for `paper-vqa`. It separates experimental
evidence and research decisions from the installation and usage documentation in the
README. Every future experiment must be added here, including negative results and invalid
runs that revealed an implementation problem.

## How to maintain this log

For each experiment, record:

1. the hypothesis and the single factor being changed;
2. the exact development protocol and artifact paths;
3. the metrics needed to test the hypothesis;
4. the conclusion and resulting decision; and
5. whether the result is an engineering check, a pilot, or paper evidence.

The evidence labels used below are:

- **Invalid**: an implementation defect makes the numbers scientifically unusable.
- **Engineering check**: verifies that the pipeline works, but is not a model comparison.
- **Development pilot**: suitable for choosing what to investigate next, but not for the paper's
  final result table.
- **Paper candidate**: the protocol has been frozen and evaluated with the required seeds.
- **Final**: evaluated on the held-out VizWiz test split after all choices were frozen.

No run in this document is currently a paper candidate or final result.

> **Scorer provenance note.** Runs through EXP-008 used the correct leave-one-annotator-out
> consensus formula but a simplified answer normaliser. The current evaluator now mirrors the
> official VQA punctuation, number, article, contraction, whitespace, and conditional-normalisation
> rules. Historical AP, AUROC, calibration, coverage, and acceptance decisions are unaffected;
> historical VQA-derived values are provisional and the selected comparisons must be regenerated
> before entering a paper table.

## Dataset and evaluation protocol

### Frozen VizWiz sources

| Split | Records | Answerable | Purpose | Frozen revision |
|---|---:|---:|---|---|
| Train | 20,523 | 73.04% | Optimisation | `0398414ffdf6ca445b9e90204ce5d0def5d73da2` |
| Validation | 4,319 | 67.93% | Development, checkpoint selection, threshold calibration | `54b3cc446d5138b72ca1b3539bc64fd1848cff70` |
| Test | 8,000 | 73.16% | One frozen final evaluation | `6de3ff2103743051521ac6cc001025a9b399a1f1` |

The Hugging Face test source provides images and questions but not answers. The official
`VQA_test.json` annotations were joined by image identifier. The adapter verifies missing,
extra, duplicated, and question-mismatched records before constructing examples. The audit
confirmed all 8,000 test joins, but test has not been used to select any model or threshold.

The complete audit artifact is
[`outputs/debug/2026-09-03_19-55-49/seed_42/data_audit.json`](outputs/debug/2026-09-03_19-55-49/seed_42/data_audit.json).

### Shared pilot protocol

Unless an entry says otherwise, development pilots use:

- seed `42`;
- 2,048 deterministically sampled VizWiz training examples;
- the same 512 deterministically sampled VizWiz validation examples;
- three epochs, batch size four, and no replay;
- `Salesforce/blip-vqa-base` at revision
  `787b3d35d57e49572baabd22884b3d5a05acf072`;
- LoRA rank 8, alpha 16, dropout 0.05, targeting `query` and `value`;
- beam search with five beams and at most ten new tokens;
- an MLP head with masked-mean pooling unless explicitly ablated; and
- validation threshold calibration constrained to at least 90% answerable recall.

The 512-example development subset contains 335 answerable and 177 unanswerable records. At
the calibrated operating point, the experiments below normally accept 302 of the 335
answerable examples (`90.15%` recall). This makes the number of accepted unanswerable examples
directly comparable across configurations.

All values are reported in `[0, 1]` in JSON artifacts and as percentages in this document.
`threshold=0` means every example is accepted; it isolates decoder behaviour and does not
represent a deployable selective policy.

## Metric interpretation

- **VQA Accuracy** is the official-style leave-one-annotator-out consensus score using all
  available references.
- **Answerability AP** is the primary threshold-independent head metric.
- **Unsafe answer rate** is the fraction of genuinely unanswerable questions accepted by the
  head.
- **Specific unsafe answer rate** is the fraction of genuinely unanswerable questions that are
  accepted and receive text other than `"unanswerable"`. This is the closest current diagnostic
  to a specific hallucination.
- **Answerable VQA Accuracy** evaluates VQA only on genuinely answerable questions; an
  abstention receives zero.
- **Accepted VQA Accuracy** evaluates answer quality among examples that pass the head.

Raw VQA Accuracy alone can reward the literal benchmark answer `"unanswerable"`. It must not be
interpreted as a complete safety metric.

## EXP-001 — Frozen BLIP-VQA baseline

**Status:** Development pilot.

**Hypothesis.** Establish how well the frozen pretrained BLIP-VQA model performs before LoRA
adaptation or answerability training.

| Evaluation set | Records | VQA Accuracy | Answerable VQA Accuracy |
|---|---:|---:|---:|
| Full validation | 4,319 | 22.01% | Not recorded in the original artifact |
| Shared pilot validation subset | 512 | 20.23% | 25.46% |

Artifacts:

- [full-validation baseline](outputs/baseline/2026-09-03_20-15-47/seed_42/baseline/metrics.json);
- [512-example comparable baseline](outputs/blip_baseline_pilot/2026-09-04_00-24-33/seed_42/baseline/metrics.json).

The frozen baseline never generated `"unanswerable"` on the 512-example subset, so every one
of its responses to an unanswerable question was specific. This establishes the need for
domain adaptation and explicit abstention.

**Decision.** Retain the frozen model as the zero-shot baseline. Do not compare its full-
validation score directly with 512-example pilot scores.

## EXP-002 — LoRA-only VQA and causal-loss correction

### EXP-002A — Invalid same-position decoder objective

**Status:** Invalid.

The first LoRA-only training run produced `0.0` VQA Accuracy and degenerate repetitions such
as `##cacacaca...` and `white white white...`. Inspection showed that decoder logits and target
tokens were compared at the same position. BLIP's decoder is causal: logits at position `t`
predict the token at `t + 1`.

Invalid artifact:
[initial LoRA evaluation](outputs/lora_vqa_pilot_evaluation/2026-09-03_22-03-19/seed_42/development_evaluation/metrics.json).

This number must never be used in a result table.

### EXP-002B — Correct next-token objective

The objective was corrected to compare `logits[:, :-1]` with `labels[:, 1:]`, ignoring padded
labels. The same alignment is now covered by unit tests.

| Model | VQA Accuracy | Answerable VQA Accuracy | Specific unsafe rate at threshold 0 |
|---|---:|---:|---:|
| Frozen BLIP | 20.23% | 25.46% | 100.00% |
| LoRA VQA only | 57.23% | 41.16% | 15.82% |

Artifact:
[corrected LoRA-only evaluation](outputs/lora_vqa_pilot_fixed_evaluation/2026-09-03_22-30-47/seed_42/development_evaluation/metrics.json).

**Conclusion.** The corrected LoRA training substantially adapts BLIP to VizWiz. The very large
global VQA gain partly includes learning the benchmark's `"unanswerable"` response, so
answerable-only quality and selective metrics remain necessary.

**Decision.** Use the corrected causal objective in every subsequent experiment. LoRA-only is
the required no-head fine-tuning baseline.

## EXP-003 — Decoder supervision policy

**Status:** Development pilot.

**Hypothesis.** Determine whether the decoder should learn from explicitly unanswerable VizWiz
examples, or whether those examples should supervise only the answerability head.

Both configurations used `lambda=0.25`, modal decoder targets, and the same MLP head.

### Calibrated selective system

| VQA policy | AP | AUROC | Coverage | Accepted VQA | Answerable VQA | Unsafe | Specific unsafe |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_examples` | **93.46%** | **87.81%** | 70.90% | **44.27%** | **33.97%** | **34.46%** | **10.73%** (19/177) |
| `answerable_only` | 93.00% | 87.03% | 71.29% | 39.67% | 32.69% | 35.59% | 20.34% (36/177) |

### Decoder diagnostic (`threshold=0`)

| VQA policy | VQA Accuracy | Answerable VQA | Specific unsafe | Answerable examples producing `"unanswerable"` |
|---|---:|---:|---:|---:|
| `all_examples` | **56.86%** | **40.21%** | **15.25%** (27/177) | 101/335 |
| `answerable_only` | 49.24% | 38.27% | 37.29% (66/177) | **80/335** |

Artifacts:

- [`all_examples`, calibrated](outputs/lora_answerability_pilot_all_evaluation/2026-09-04_00-52-09/seed_42/development_evaluation/metrics.json);
- [`all_examples`, threshold zero](outputs/lora_answerability_pilot_all_threshold0/2026-09-04_00-59-04/seed_42/development_evaluation/metrics.json);
- [`answerable_only`, calibrated](outputs/lora_answerability_pilot_answerable_only_evaluation/2026-09-04_01-22-32/seed_42/development_evaluation/metrics.json);
- [`answerable_only`, threshold zero](outputs/lora_answerability_pilot_answerable_only_threshold0/2026-09-04_01-26-33/seed_42/development_evaluation/metrics.json).

**Conclusion.** Masking unanswerable records from decoder training reduces some textual false
abstentions on answerable questions, but it more than doubles the calibrated specific unsafe
rate. A head false positive reaches a decoder that was never trained to provide a safe fallback
on those examples.

**Decision.** Keep `all_examples`. Retain `answerable_only` as a negative ablation required to
separate the head's contribution from decoder fallback learning.

## EXP-004 — Answerability-loss weight sweep

**Status:** Development pilot.

**Hypothesis.** Measure the trade-off introduced by `lambda` in
`L_total = L_vqa + lambda * L_answerability` while keeping modal targets, `beta=1`, and
`all_examples` fixed.

### Calibrated results

| `lambda` | AP | AUROC | VQA Accuracy | Accepted VQA | Answerable VQA | Unsafe | Specific unsafe |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 93.08% | 87.13% | **32.81%** | **45.41%** | **34.66%** | 38.42% | 11.86% (21/177) |
| 0.10 | 93.33% | 87.53% | 32.01% | 45.03% | 34.63% | 35.03% | **10.73%** (19/177) |
| 0.25 | **93.46%** | 87.81% | 31.39% | 44.27% | 33.97% | **34.46%** | **10.73%** (19/177) |
| 0.50 | 93.42% | **87.83%** | 31.58% | 43.94% | 33.88% | 37.29% | 12.99% (23/177) |

The corresponding threshold-zero VQA scores were 57.25%, 57.23%, 56.86%, and 56.58%.
The decoder therefore remains broadly stable across this small sweep; most of the operational
trade-off comes from answerability ranking and threshold behaviour.

Artifacts are stored under:

- `outputs/lambda_005_*`;
- `outputs/lambda_010_*`;
- `outputs/lambda_025_*`; and
- `outputs/lambda_050_*`.

**Conclusion.** `lambda=0.10` and `lambda=0.25` form a small Pareto region. `0.10` retains
slightly more answer quality, whereas `0.25` obtains the strongest AP and lowest unsafe rate.
Differences are too small to claim significance from one training seed.

**Decision.** Use `lambda=0.25` as the safety-oriented primary configuration and retain
`lambda=0.10` as the closest alternative if later multi-seed results reveal a consistent VQA
advantage.

## EXP-005 — Metric refinement after qualitative error analysis

**Status:** Engineering and analysis improvement.

The initial selective metrics counted every head false positive as unsafe, including cases in
which the decoder itself returned `"unanswerable"`. Prediction inspection showed that this hid
an important distinction between a textual fallback and a specific invented answer.

Three diagnostics were introduced:

1. `answerable_vqa_accuracy`;
2. `accepted_unanswerable_answer_rate`; and
3. `specific_unsafe_answer_rate`.

This refinement changed the interpretation, not the stored model weights. Historical
predictions were re-analysed to populate the tables in this document. Older `metrics.json`
files do not acquire new keys retroactively.

**Decision.** Every future evaluation must report the specific unsafe rate. Raw VQA Accuracy
or aggregate textual `"unanswerable"` frequency is insufficient for model selection.

## EXP-006 — Target policy and negative decoder weight factorial

**Status:** Development pilot.

**Motivation.** The `answerable_only` result suggested trying an intermediate objective:

```text
L_total = L_vqa_answerable + beta * L_vqa_unanswerable
          + lambda * L_answerability
```

At the same time, `label_consistent_modal` was introduced to prevent an officially answerable
record from using `"unanswerable"` as its decoder target. Because changing both factors at once
would be confounded, a two-by-two experiment was completed with `lambda=0.25`.

### Calibrated factorial

| Target policy | `beta` | AP | Accepted VQA | Answerable VQA | Unsafe | Specific unsafe |
|---|---:|---:|---:|---:|---:|---:|
| Modal | 1.00 | **93.46%** | **44.27%** | **33.97%** | **34.46%** | **10.73%** (19/177) |
| Modal | 0.25 | 93.28% | 42.37% | 33.22% | 36.72% | 15.25% (27/177) |
| Label-consistent | 1.00 | 93.12% | 40.70% | 32.42% | 38.42% | 18.64% (33/177) |
| Label-consistent | 0.25 | 92.93% | 35.75% | 30.33% | 37.85% | 25.42% (45/177) |

All four calibrated configurations accepted 302 of 335 answerable examples. Their numbers of
accepted unanswerable examples were respectively 61, 65, 68, and 67.

### Decoder factorial (`threshold=0`)

| Target policy | `beta` | VQA Accuracy | Answerable VQA | Specific unsafe | Answerable examples producing `"unanswerable"` |
|---|---:|---:|---:|---:|---:|
| Modal | 1.00 | **56.86%** | **40.21%** | **15.25%** (27/177) | 101/335 |
| Modal | 0.25 | 54.57% | 39.25% | 21.47% (38/177) | 94/335 |
| Label-consistent | 1.00 | 53.26% | 38.18% | 23.16% (41/177) | 66/335 |
| Label-consistent | 0.25 | 47.42% | 36.00% | 38.42% (68/177) | **45/335** |

Artifacts:

- modal, `beta=1`: `outputs/lambda_025_new_metrics_*`;
- [modal, `beta=0.25`, calibrated](outputs/modal_beta025_calibrated/2026-09-16_14-08-00/seed_42/development_evaluation/metrics.json);
- [modal, `beta=0.25`, threshold zero](outputs/modal_beta025_threshold0/2026-09-16_14-10-59/seed_42/development_evaluation/metrics.json);
- [label-consistent, `beta=1`, calibrated](outputs/label_consistent_beta100_calibrated/2026-09-16_13-55-01/seed_42/development_evaluation/metrics.json);
- [label-consistent, `beta=1`, threshold zero](outputs/label_consistent_beta100_threshold0/2026-09-16_13-57-06/seed_42/development_evaluation/metrics.json);
- [label-consistent, `beta=0.25`, calibrated](outputs/label_consistent_beta025_calibrated/2026-09-15_21-52-14/seed_42/development_evaluation/metrics.json);
- [label-consistent, `beta=0.25`, threshold zero](outputs/label_consistent_beta025_threshold0/2026-09-15_21-54-08/seed_42/development_evaluation/metrics.json).

### Interpretation

Increasing `beta` helps under both target policies. Full decoder supervision on negative
examples is beneficial both for VQA and for safety.

`label_consistent_modal` successfully reduces `"unanswerable"` outputs on genuinely answerable
questions, but those removed abstentions do not become more correct answers. Answerable VQA
falls and specific unsafe answers rise. On the evaluated subset, the label-consistent policy
changes 59 of 335 answerable targets. In many such cases the selected non-`"unanswerable"`
answer has support from only one or two annotators, injecting noisy sequence targets.

**Decision.** Freeze the current primary pilot objective as:

```yaml
answerability_weight: 0.25       # lambda
vqa_loss_policy: all_examples
vqa_target_policy: modal
unanswerable_vqa_weight: 1.0     # beta
```

Keep the other three cells as negative ablations. Do not continue sweeping `beta` unless new
multi-seed evidence contradicts this result.

## Current decision register

| Decision | Current choice | Evidence | Revisit when |
|---|---|---|---|
| Causal decoder alignment | Next-token shift | EXP-002 | Never, unless model architecture changes |
| Decoder supervision policy | `all_examples` | EXP-003 | A different dataset defines abstention differently |
| Answerability weight | `lambda=0.25` primary, `0.10` alternative | EXP-004 | Multi-seed full-data comparison |
| Target policy | `modal` | EXP-006 | A principled multi-reference loss is implemented |
| Negative decoder weight | `beta=1.0` | EXP-006 | Multi-seed evidence contradicts the pilot |
| Head pooling | `masked_mean` selected; `cls` retained as ablation | EXP-007 full validation | Multi-seed evidence |
| Replay source and ratio | None provisional | Not yet ablated | Replay experiments |

## EXP-007 — Answerability-head pooling

**Status:** Development pilot complete; decision deferred to full-validation evaluation.

**Hypothesis.** Compare `masked_mean` against `cls` while fixing modal targets, `beta=1`,
`lambda=0.25`, no replay, the shared pilot subset, the seed, and checkpoint selection by
validation AP. Both variants were retrained with the same protocol.

### Calibrated results

| Pooling | AP | AUROC | Brier | ECE | Coverage | Accepted VQA | Answerable VQA | Unsafe | Specific unsafe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `masked_mean` | 93.46% | 87.81% | 13.89% | 6.18% | 70.90% | 44.27% | 33.97% | **34.46%** | **10.73%** |
| `cls` | **93.55%** | **88.02%** | **13.66%** | **5.93%** | **71.88%** | **45.30%** | **34.27%** | 37.29% | **10.73%** |

Both models accept 302 of 335 answerable examples. `masked_mean` accepts 61 unanswerable
examples: 42 decode to `"unanswerable"` and 19 receive a specific answer. `cls` accepts 66:
47 decode to `"unanswerable"` and the same total of 19 receive a specific answer. The specific
error sets are not identical: 18 examples overlap, and each model has one unique specific
unsafe error.

### Decoder diagnostic (`threshold=0`)

| Pooling | VQA Accuracy | Answerable VQA | Specific unsafe | Literal `"unanswerable"` outputs |
|---|---:|---:|---:|---:|
| `masked_mean` | 56.86% | **40.21%** | 15.25% (27/177) | 251/512 |
| `cls` | **57.03%** | 40.18% | **14.69%** (26/177) | 252/512 |

Artifacts:

- [`masked_mean`, calibrated](outputs/pooling_masked_mean_calibrated/2026-09-16_17-21-12/seed_42/development_evaluation/metrics.json);
- [`masked_mean`, threshold zero](outputs/pooling_masked_mean_threshold0/2026-09-16_17-24-02/seed_42/development_evaluation/metrics.json);
- [`cls`, calibrated](outputs/pooling_cls_calibrated/2026-09-16_17-32-05/seed_42/development_evaluation/metrics.json); and
- [`cls`, threshold zero](outputs/pooling_cls_threshold0/2026-09-16_17-34-31/seed_42/development_evaluation/metrics.json).

### Interpretation and decision

The `cls` point estimate is slightly better for AP, AUROC, Brier, ECE, coverage, accepted VQA,
and total calibrated VQA. It nevertheless produces five additional head false positives at
the fixed recall. All five additional outputs are the decoder fallback `"unanswerable"`, so
the specific unsafe rate is unchanged.

A paired example bootstrap does not resolve the comparison. For `cls - masked_mean`, its 95%
intervals are approximately `[-0.15, +0.35]` percentage points for AP,
`[-0.04, +2.44]` for calibrated VQA Accuracy, and `[-1.67, +1.69]` for the specific unsafe
rate. These intervals measure validation-example uncertainty for the two fixed checkpoints;
they do not include training-seed uncertainty.

**Pilot decision.** Do not declare either pooling superior from 512 examples and one seed. `cls` is
the provisional performance candidate, while `masked_mean` is the more conservative head-only
candidate at the current operating point. Evaluate both existing checkpoints over all 4,319
validation examples before paying the much larger cost of full-data training. Test remains
closed.

### Full-validation extension

The same two fixed checkpoints were then evaluated on all 4,319 validation records. This
increases evaluation coverage without changing training data, weights, seed, or test status.

| Pooling | AP | AUROC | Brier | ECE | Coverage | Accepted VQA | Answerable VQA | Unsafe | Specific unsafe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `masked_mean` | **93.77%** | **87.94%** | 13.11% | 5.02% | 72.05% | 45.44% | 37.14% | **34.01%** | **12.92%** |
| `cls` | 93.70% | 87.90% | **13.09%** | **4.87%** | **72.29%** | **45.70%** | **37.22%** | 34.73% | **12.92%** |

Both models accept 2,641 of 2,934 answerable records (`90.01%`). `masked_mean` accepts 471 of
1,385 unanswerable records: 292 decode to `"unanswerable"` and 179 receive a specific answer.
`cls` accepts 481: 302 decode to `"unanswerable"` and the same total of 179 receive a specific
answer. Thus the ten additional `cls` false positives do not increase the aggregate specific
unsafe rate.

The specific unsafe sets still differ: 172 records are shared and each model has seven unique
specific errors. The accepted-answer sets also differ rather than being nested: 3,063 records
are shared, 49 are unique to `masked_mean`, and 59 are unique to `cls`.

The threshold-zero diagnostic remains nearly tied:

| Pooling | VQA Accuracy | Answerable VQA | Specific unsafe | Literal `"unanswerable"` outputs |
|---|---:|---:|---:|---:|
| `masked_mean` | 57.36% | 43.43% | 16.75% (232/1,385) | 2,038/4,319 |
| `cls` | **57.45%** | **43.49%** | **16.61%** (230/1,385) | 2,058/4,319 |

Full-validation artifacts:

- [`masked_mean`, calibrated](outputs/pooling_masked_mean_full_validation_calibrated/2026-09-16_17-45-58/seed_42/development_evaluation/metrics.json);
- [`masked_mean`, threshold zero](outputs/pooling_masked_mean_full_validation_threshold0/2026-09-16_18-01-49/seed_42/development_evaluation/metrics.json);
- [`cls`, calibrated](outputs/pooling_cls_full_validation_calibrated/2026-09-16_18-16-07/seed_42/development_evaluation/metrics.json); and
- [`cls`, threshold zero](outputs/pooling_cls_full_validation_threshold0/2026-09-17_13-00-05/seed_42/development_evaluation/metrics.json).

A paired example bootstrap again includes zero for every central difference. For
`cls - masked_mean`, the approximate 95% intervals are `[-0.20, +0.06]` percentage points for
AP, `[-0.12, +0.69]` for calibrated VQA Accuracy, and `[-0.52, +0.51]` for the specific unsafe
rate. Training-seed uncertainty remains unmeasured.

**Development decision.** Select `masked_mean` for the next phase because the predeclared
primary head metric, AP, is higher on complete validation; AUROC is also higher, and it accepts
ten fewer unanswerable records at the fixed recall. `cls` has slightly better calibration and
VQA point estimates, but the differences are very small and unsupported by the paired
intervals. This is a protocol-driven selection, not a claim that `masked_mean` is statistically
superior. Keep `cls` in the paper as the required representation ablation.

## EXP-008 — Does the head add value beyond decoder abstention?

**Status:** Preliminary post-hoc development analysis; formal baseline still required.

**Question.** A decoder trained on unanswerable examples can already emit the literal
`"unanswerable"`. The central paper claim therefore requires showing that the auxiliary head
adds value beyond treating that decoder output as an abstention.

Two policies were reconstructed from predictions of the same selected multitask checkpoint:

1. **Decoder-only abstention:** disable the head at inference, generate for every example, and
   emit an answer only when the decoded text is not `"unanswerable"`.
2. **Head + decoder abstention:** first apply the calibrated head; among accepted examples,
   treat a decoded `"unanswerable"` as a textual abstention rather than a specific answer.

Using the same checkpoint isolates the value of its head decision at inference from changes in
decoder weights.

### Full-validation comparison

| Policy | Coverage | Answerable emission recall | Specific unsafe | Emission answerability precision | Accepted specific VQA |
|---|---:|---:|---:|---:|---:|
| Decoder only | 52.81% | **69.84%** | 16.75% (232/1,385) | 89.83% | **37.68%** |
| Head + decoder | 50.57% | 68.34% | **12.92%** (179/1,385) | **91.80%** | 37.62% |

The head prevents 53 specific responses on unanswerable questions. This is a 3.83 percentage-
point absolute and approximately 22.8% relative reduction in the specific unsafe rate. The
cost is 2.25 points of overall coverage and 1.50 points of answerable emission recall. Quality
among emitted specific answers is effectively unchanged.

The 512-example pilot shows the same direction: specific unsafe rate falls from 15.25% to
10.73%, while answerable emission recall falls from 69.85% to 68.36%.

**Interpretation.** This is meaningful preliminary evidence that the head contributes beyond
the decoder's literal fallback: it removes disproportionately more unsafe specific answers than
useful answerable emissions. It is not yet a publication-grade demonstration because the
policy was reconstructed post hoc, uses one training seed, and compares only one operating
point.

**Required confirmation.** Make decoder-only, head-only, and hybrid policies first-class
evaluation modes; compare their risk-coverage curves at matched coverage or answerable recall;
and add a decoder-confidence baseline based on generation likelihood. Repeat the frozen
comparison on full-data multi-seed checkpoints.

## Planned experiments

### Experiment tracking hardening (completed before further GPU runs)

W&B support is now safe to enable for convergence and hyperparameter-search runs. Each run uses
the stable name `<experiment.name>-seed_<trainer.seed>`, stores its resolved Hydra configuration,
and logs every available training or evaluation scalar. Epoch history is also written locally to
`history.json` after every completed epoch, independently of W&B, so a terminal transcript or
network connection is not required to reconstruct learning curves.

Cloud artefacts follow an explicit allow-list. Manifests are enabled by default; checkpoints and
per-example prediction directories are disabled by default because checkpoints are currently
about 1.4 GB and predictions contain dataset-derived content. If checkpoint upload is explicitly
enabled, only the final validation-selected checkpoint is considered once after training rather
than once per improving epoch. The integration is covered by unit and smoke tests.

The first real offline attempt confirmed that W&B created its run in the Hydra output directory
and admitted only the manifest artefact. It then exposed a server-resource interaction before
the first batch: Hugging Face Tokenizers could not initialise its Rayon thread pool after W&B
started its local service (`Resource temporarily unavailable`). The reproducibility initialiser
now enforces `TOKENIZERS_PARALLELISM=false`. This removes unnecessary tokenizer worker threads,
reduces oversubscription.

The repeated offline smoke completed successfully on 16 training and 8 validation examples. It
recorded four optimiser steps, selected epoch 1, and produced a local `history.json` matching the
terminal and W&B summaries (`val/loss=2.8865`, `val/answerability_ap=0.6193`). The W&B run stored
the resolved configuration, dependency list, scalar metrics, and manifests. Its run file was
approximately 13 KB; the 1.45 GB checkpoint remained local because
`logging.upload_checkpoints=false`. These numerical metrics are infrastructure diagnostics only
and must not be used as scientific evidence. This validated the offline integration before the
final cloud setup check.

The successful offline run was subsequently synchronised to the institutional W&B entity
`models-universidad-complutense-de-madrid`, project `paper-vqa`, with run ID `cr6m12q7`. W&B
preserved the intended run name `wandb_offline_smoke-seed_42`. Authentication, cloud upload,
project routing, and offline-to-online synchronisation are therefore validated. Future
scientific runs can use `logging.mode=online` directly while retaining the conservative
artefact policy.

### Evaluation semantics and policy reporting (implemented)

The VQA scorer now follows the official evaluator's text processing and held-out-annotator
consensus semantics, with parity fixtures for the cases that differed from the earlier
simplified normaliser. Evaluation unconditionally generates each answer once, records the
auxiliary-head probability and the decoder's length-normalised geometric mean token probability,
and compares six first-class policies: always answer, literal decoder abstention,
decoder-confidence selection, decoder-confidence plus literal abstention, head-only selection,
and the head-plus-literal hybrid.

Head and decoder-confidence thresholds are calibrated independently on validation under the
same answerable-recall constraint. Frozen-test code calibrates both on validation before any test
policy is scored. Each evaluation now writes fixed policy metrics, long-form risk--coverage CSV,
and a vector SVG. A real cached BLIP + LoRA model successfully produced both confidence signals
on CPU, and the implementation passes unit tests covering policy decisions, calibration, curve
generation, and report serialization.

The first real CLI policy smoke exposed a second server-resource constraint while loading BLIP:
`libgomp` could not create its default OpenMP worker pool. W&B had not started yet, so this was
independent of cloud tracking. Reproducibility setup now applies configurable
`trainer.cpu_threads` limits to OpenMP, MKL, OpenBLAS, NumExpr, and PyTorch intra/inter-op pools;
the server-safe default is one.

The repeated eight-example online smoke then completed model loading and scored generation, and
wrote every local evaluation report. It exposed a separate issue only during final W&B artefact
upload: `wandb.Artifact.add_dir` attempted to create its own thread pool and failed under the
server's thread limit. Directory artefacts are now traversed deterministically and added one file
at a time with their relative paths preserved. A real two-example offline W&B run subsequently
logged all 62 evaluation scalars, added the manifest artefact, and finished normally. Together,
these checks validate the formal policy evaluation path and its W&B integration under constrained
server resources. Both sample sizes are infrastructure smokes only; their metric values are not
scientific evidence. A final eight-example online repetition (`zig5wc2k`) also completed without
errors, synchronized all 62 scalar metrics and two artefact files, and produced the complete set
of five policy summaries plus the risk--coverage reports. The evaluation and online-tracking smoke
is therefore closed.

The remaining reporting work before final multi-seed results is a standalone aggregation CLI,
paired bootstrap comparisons, PR/ROC and reliability plots, and confusion matrices. Those do not
block the single-run convergence study and will be completed before hyperparameter finalists are
compared across seeds.

### Training-length convergence pilot

Every three-epoch pilot selected its final available epoch. Before spending roughly ten times
more compute on full VizWiz train, run the selected `masked_mean` configuration with a larger
epoch ceiling and validation early stopping. This checks whether three epochs truncated an
improving model and fixes the epoch/early-stopping protocol for full-data training.

**Completed (`convergence_masked_mean_seed42`).** The deliberately reduced development run used
2,048/20,523 training examples (9.98%) and 512/4,319 validation examples (11.85%), with a ten-epoch
ceiling, 5% proportional warmup, and patience three on validation answerability AP. Early stopping
ended training after epoch 5 and restored epoch 2, whose validation AP was `0.93352` and validation
loss was `1.52595`. After epoch 2, training loss continued falling from `1.40705` to `0.89872` and
training AP rose from `0.95240` to `0.99205`, while validation loss worsened to `1.64482` and
validation AP fell to `0.91808` by epoch 5. This is clear subset-level overfitting and demonstrates
that checkpoint selection and early stopping are necessary.

This run does not determine the optimal epoch for full-data training: a full epoch contains about
ten times as many distinct examples and optimisation steps, and the generalisation dynamics can
change with dataset scale. Its valid conclusions are limited to closing the ten-epoch pilot,
retaining validation-AP checkpoint selection, and avoiding an unconditional fixed epoch count.
The selected checkpoint still requires formal policy evaluation; the scalar training history
alone does not measure selective VQA behaviour.

#### Full-validation policy evaluation

The restored epoch-2 checkpoint was evaluated on all 4,319 validation examples with one shared
unconditional generation per example and the first-class policy evaluator. The head obtained
AP `93.67%`, AUROC `87.78%`, Brier `13.07%`, and ECE `4.10%`. Validation calibration selected a
head threshold of `0.48539`, producing `90.01%` answerable recall and `65.92%` specificity. The
unconditional official VQA Accuracy was `57.65%`; answerable-only VQA Accuracy was `43.35%`.

The operational comparison that treats a generated literal `"unanswerable"` as abstention is:

| Policy | Coverage | Answerable emission recall | Specific unsafe | Emission answerability precision | Accepted specific VQA |
|---|---:|---:|---:|---:|---:|
| Decoder literal | 51.17% | **68.03%** | 15.45% (214/1,385) | 90.32% | **37.79%** |
| Head + literal | 49.29% | 66.70% | **12.42% (172/1,385)** | **91.92%** | 37.67% |

The head therefore prevents 42 specific responses on unanswerable questions, a `3.03`-point
absolute and `19.6%` relative reduction, while losing 39 answerable emissions (`1.33` recall
points) and `1.88` overall coverage points. Accepted specific VQA changes by only `-0.12` points.
This formally reproduces the direction of EXP-008 with the new evaluator and a separately trained
checkpoint: the head removes proportionally more unsafe outputs than useful emissions.

Decoder token confidence requires a symmetric interpretation. At its independently calibrated
pre-literal 90% answerable-recall point, it accepts `96.53%` of ground-truth unanswerable examples
because it is often highly confident when generating the literal `"unanswerable"`; consequently,
raw confidence coverage is not directly comparable with the head-plus-literal deployment policy.
A post-hoc matched-emission diagnostic applies the literal gate to both confidence signals and
sets each threshold to the same `66.70%` answerable emission recall:

| Policy at matched recall | Coverage | Specific unsafe | Emission answerability precision | Accepted specific VQA |
|---|---:|---:|---:|---:|
| Decoder confidence + literal | 50.01% | 14.66% (203/1,385) | 90.60% | **38.62%** |
| Head + literal | 49.25% | **12.27% (170/1,385)** | **92.01%** | 37.66% |

At this matched point, the head prevents 33 unsafe responses and improves emission answerability
precision by `1.41` points, but accepted specific VQA is `0.96` points lower. The full curves show
the same safety direction at 60% and 65% answerable emission recall. This is evidence of a genuine
safety--utility trade-off, not uniform dominance. The matched diagnostic was derived after viewing
validation and must not be treated as confirmatory evidence.

The symmetric `decoder_confidence_plus_literal` policy is now first-class. Future runs predeclare
matched answerable-emission recall targets of 50%, 60%, and 65% in Hydra; validation calibrates a
separate threshold for each confidence signal, and frozen-test evaluation transports those
thresholds unchanged. JSON, CSV, risk--coverage SVG, and W&B reporting include the new policy and
matched table. A validation-only reporting command can rebuild these outputs from saved generations
without loading the model or decoding again.

Regenerating the present checkpoint's report verifies the implementation, but remains post-hoc for
this already inspected run. At 50%, 60%, and 65% answerable-emission recall respectively, the head
produces `53/110/147` specific unsafe outputs versus `149/174/198` for decoder confidence. Its
emission answerability precision is higher by `5.73/3.11/2.25` points, while its accepted VQA is
lower by `8.59/4.54/1.78` points. The gap narrows as recall rises. These results sharpen the paper
hypothesis: the answerability head selects safer emissions, whereas token likelihood preferentially
selects easier-to-answer examples and therefore obtains higher accepted VQA. The targets are frozen
prospectively for subsequent hyperparameter and multi-seed experiments.

Compared with the earlier three-epoch `masked_mean` pilot checkpoint, the convergence checkpoint
has slightly lower AP (`93.67%` versus `93.77%`) and answerable VQA (`43.35%` versus `43.43%`),
but better ECE (`4.10%` versus `5.02%`) and fewer specific unsafe outputs at the calibrated head
point (`172` versus `179`). These small, mixed one-seed differences do not identify a superior
training length. They reinforce using a Pareto decision over answerability, safety, and VQA rather
than selecting solely by the number of epochs or validation loss.

### Bounded hyperparameter search

Run a validation-only Bayesian or successive-halving search with no replay and with the
scientific choices already supported by ablations held fixed: `masked_mean`, modal targets,
`all_examples`, and `beta=1`. Search optimisation and capacity parameters rather than reopening
rejected hypotheses:

- learning rate and weight decay;
- LoRA rank, alpha-to-rank ratio, and dropout;
- head hidden width and dropout;
- warmup ratio; and
- `lambda` restricted to the existing `0.10`/`0.25` Pareto candidates.

Use one seed for screening, evaluate the best candidates with the complete validation split,
and confirm the top three over three pilot seeds before freezing one configuration. Select from
the Pareto set using AP first, specific unsafe rate at matched recall second, and answerable VQA
third. Do not optimise the safety threshold itself; report risk-coverage curves and keep the
minimum-recall policy explicit.

**Infrastructure prepared; no screening results yet.** Two versioned W&B configurations now
separate a one-run integration smoke from the Bayesian screen. The screen uses 4,096 train and
1,024 validation examples, seed 42, a six-epoch ceiling, patience two, and no replay. It searches
learning rate, weight decay, LoRA dropout, head width/dropout, warmup ratio, and
`lambda ∈ {0.10, 0.25}`. Six model-group choices encode ranks 4/8/16 with alpha-to-rank ratios
one or two, avoiding invalid independent rank/alpha combinations.

The sweep objective is `selection/answerability_ap`. This scalar is logged only when validation
selects a new checkpoint, so its final W&B summary represents the restored best epoch rather than
the last epoch. AP is only a screening objective: finalists still require unconditional generation
and Pareto comparison on safety, answerable VQA, and calibration.

To make 24 local checkpoints practical, checkpoint format version 2 stores only trainable LoRA and
head tensors while reconstructing frozen BLIP from its recorded immutable revision. A real
BLIP+LoRA+MLP save/load check restored all 200 trainable tensors and produced a 5,608,272-byte
checkpoint, compared with approximately 1.4 GB previously. Loading legacy full-model checkpoints
remains supported. Sweep checkpoint and prediction uploads stay disabled.

**W&B sweep integration smoke completed (`i6q6ouh8`, run `vmtgvtkc`).** The versioned
`hpo_smoke.yaml` configuration was created in the university W&B entity and its single grid run
completed online with exit code zero. W&B received the resolved Hydra sweep parameters, epoch
metrics, and the checkpoint-selection summary, including
`selection/answerability_ap = 0.61929`. The run also synchronized the configured manifest
artefact without attempting to upload the checkpoint or predictions.

The corresponding local run retained both deterministic data manifests, the resolved Hydra
configuration, `history.json`, and checkpoint metadata. The selected checkpoint declares
`format_version: 2` and `weights_scope: trainable`; its `model.safetensors` file is 5.4 MiB.
This closes the Hydra--training--checkpoint--W&B integration gate. Its AP and losses are not
scientific results: the smoke deliberately used only 16 training examples, eight validation
examples, and one epoch.

**Three-run screening gate completed.** The first three Bayesian trials used byte-identical
4,096-example training and 1,024-example validation manifests. All runs completed normally,
logged the best-checkpoint objective to W&B, wrote compact trainable-only checkpoints, and obeyed
the validation-AP early-stopping rule.

| W&B run | LoRA | LR | WD | LoRA drop. | Head (width/drop.) | Warmup | Lambda | Selected epoch / run epochs | Selected val AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `u8k27fag` | r16 / a32 | 2.038e-4 | 0.00 | 0.05 | 512 / 0.30 | 0.05 | 0.25 | 1 / 3 | 0.94455 |
| `7r40o05v` | r4 / a4 | 1.614e-4 | 0.05 | 0.10 | 512 / 0.50 | 0.03 | 0.25 | 4 / 6 | **0.94536** |
| `zdj02o8t` | r16 / a32 | 2.908e-4 | 0.00 | 0.05 | 256 / 0.30 | 0.03 | 0.25 | 1 / 3 | 0.94492 |

The two rank-16 trials selected epoch one and stopped after two subsequent AP decreases. Their
training AP continued rising to approximately 0.984 while validation AP fell to 0.937--0.939,
which is consistent with rapid overfitting at this screening scale. The more strongly regularised
rank-4 trial improved until epoch four and is the provisional leader, but the complete spread
between all three selected AP values is only 0.00081. Three Bayesian proposals, all of which
happened to sample `lambda=0.25`, cannot establish a capacity, regularisation, or lambda winner.
The appropriate decision is therefore to continue the predefined sweep rather than narrow its
search space from this small initial sample.

**Screening batch completed: 22 valid trials, two startup failures.** The requested agent budget
started 24 W&B runs. Twenty-two trained and saved a selected checkpoint; every attempted run had
the same train-manifest SHA-256 (`921c7dd8...f755b3`) and validation-manifest SHA-256
(`0642ff9c...6dce892`). Two attempts (`pa0n7neu` and `8x0sgyme`) failed before the first epoch
and before creating a checkpoint. Their local W&B logs show a reset connection in the W&B service
startup path, not a model error, CUDA OOM, dataset failure, or invalid sampled hyperparameter.
They are excluded from analysis and must be replaced with two new agent trials to preserve the
predeclared target of 24 completed screening configurations.

Across the 22 completed trials, selected validation AP ranged from `0.94333` to `0.94847`
(mean `0.94616`, sample SD `0.00148`). The top screening rows were:

| Rank | W&B run | LoRA | LR | WD | LoRA drop. | Head | Warmup | Lambda | Selected epoch | AP |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | `934f1ayj` | r16 / a16 | 9.893e-5 | 0.05 | 0.10 | 512 / 0.50 | 0.10 | 0.25 | 4 | **0.94847** |
| 2 | `9fl0n4qw` | r16 / a32 | 7.231e-5 | 0.05 | 0.10 | 512 / 0.50 | 0.10 | 0.25 | 4 | 0.94835 |
| 3 | `tq3fu0v6` | r16 / a32 | 8.668e-5 | 0.05 | 0.05 | 512 / 0.50 | 0.10 | 0.25 | 4 | 0.94832 |
| 4 | `l0j4m2ta` | r8 / a8 | 1.683e-4 | 0.05 | 0.10 | 512 / 0.50 | 0.05 | 0.10 | 4 | 0.94775 |
| 5 | `z6vhsabp` | r8 / a8 | 1.621e-4 | 0.01 | 0.10 | 512 / 0.50 | 0.10 | 0.10 | 4 | 0.94770 |

The leading three candidates form a coherent high-regularisation family: rank 16, head width 512,
head dropout 0.50, weight decay 0.05, warmup 10%, lambda 0.25, and lower learning rates
(roughly `0.7e-4` to `1.0e-4`). This is a useful hypothesis, not a causal conclusion: Bayesian
proposals are correlated and this study has one training seed. The best `lambda=0.10` alternative
is the rank-8 run `l0j4m2ta`, which should remain in the later safety/VQA comparison rather than
being silently discarded solely because it is 0.00072 AP below the current leader.

Validation-AP early stopping behaved as intended: selected epochs were 1 (2 runs), 2 (6 runs),
4 (12 runs), and 6 (2 runs). Most useful candidates selected epoch four; this supports retaining
the six-epoch ceiling and AP-based checkpoint selection for screening. It does not establish the
correct epoch budget for full-data training.

### After hyperparameters are frozen

1. Run the selected core configurations once on full VizWiz train and validation.
2. Freeze TextVQA and VQAv2 revisions before replay experiments.
3. Run replay ablations: none, TextVQA, VQAv2, and both.
4. Freeze architecture, objective, replay, checkpoint rule, and threshold policy.
5. Train the final candidates with five seeds and report mean, sample standard deviation, and
   bootstrap confidence intervals.
6. Evaluate VizWiz test only after all preceding decisions are frozen.

## Limitations of the current evidence

- All fine-tuning comparisons use one training seed.
- Most evaluations use a deterministic 512-example validation subset rather than all 4,319
  validation records.
- Validation is used for both development comparison and threshold calibration; test remains
  untouched for final reporting.
- The current sequence objective uses one deterministic modal target even though ten human
  references are retained for evaluation.
- Replay and full-data multi-seed experiments are incomplete. Pooling has been selected by a
  predefined development rule, but its small difference has not been established across seeds.
- Differences between close configurations must not be described as statistically significant
  until training-seed variability is measured.
