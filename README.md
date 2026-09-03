# paper-vqa

`paper-vqa` is a reproducible research codebase for **selective Visual Question Answering (VQA)** on photographs taken by blind people. It studies whether a BLIP-VQA model adapted with LoRA can answer a visual question *and* estimate whether the image-question pair is answerable. When the answerability score is below a validation-calibrated threshold, the system abstains and does not generate an answer.

This repository is deliberately independent from the original TFG application. It does not import code from, or modify, `mi-tfg`. The original project is used only as historical context for the research refactor described below.

## Research objective

Standard VQA systems always generate an answer. This is unsafe for the target use case: a blurry, occluded, wrongly framed, or otherwise unusable photo can lead to a plausible but incorrect response. The research question is therefore:

> Can a LoRA-adapted BLIP-VQA model retain answer quality on VizWiz while an auxiliary answerability head reliably abstains on questions that cannot be answered from the image?

The main benchmark is [VizWiz-VQA](https://vizwiz.org/tasks-and-datasets/vqa/), whose images were collected by blind people and whose annotations include both multiple VQA answers and an answerability label.

## What is implemented

```text
image + question
       │
       ▼
BLIP vision encoder + multimodal text encoder
       │
       ├──► answerability MLP ──► P(answerable) ──► calibrated threshold
       │                                               │
       │                                     abstain ◄─┴─► generate answer
       │                                                        │
       └──────────────────────── BLIP text decoder ◄──────────┘
```

The package provides:

- A typed, modular Python package under `src/paper_vqa`.
- BLIP-VQA at a fixed Hugging Face revision, optional LoRA adapters, and an optional answerability head.
- A configurable multitask objective with two VQA-loss policies.
- VizWiz adapters, including a hybrid official-test adapter, plus configurable TextVQA and VQAv2 replay sources.
- Deterministic sampling, training, checkpoint selection, provenance manifests, leakage detection, and data audits.
- Validation-only development evaluation, validation calibration, selective inference, and a frozen-test evaluation entry point.
- Official-style VQA scoring, answerability, calibration, and selective-risk metrics.
- Hydra configuration, optional Weights & Biases logging, safe `safetensors` checkpoints, `uv`, Ruff, mypy, and pytest.

## Repository layout

```text
paper-vqa/
├── configs/                         # Hydra configuration groups
│   ├── data/                        # VizWiz sources and fixed revisions
│   ├── replay/                      # none, TextVQA, VQAv2, both
│   ├── model/, head/, loss/          # model and objective ablations
│   ├── trainer/, evaluation/         # optimisation and evaluation settings
│   ├── logging/                     # W&B settings
│   └── experiment/                  # named experiment metadata
├── src/paper_vqa/
│   ├── cli/                         # command-line entry points
│   ├── data/                        # records, adapters, loaders, audit
│   ├── models/                      # BLIP factory, head, selective wrapper
│   ├── training/                    # objective, trainer, tracker, checkpoints
│   ├── evaluation/                  # metrics, inference, prediction reports
│   └── utils/                       # reproducibility and manifests
├── tests/                           # unit and smoke tests
├── data/                            # local benchmark annotations; ignored by Git
├── outputs/                         # Hydra run artefacts; ignored by Git
├── pyproject.toml
└── uv.lock
```

`main.py` is a compatibility entry point for local development. Prefer the explicit `paper-vqa-*` commands below.

## Installation

The project requires Python 3.12 and uses `uv`.

```bash
uv sync --extra dev
```

PyTorch is pinned to `2.6.0` and Torchvision to `0.21.0`. Both are resolved from the official CUDA 12.4 PyTorch index through `pyproject.toml`; this avoids accidentally installing an incompatible CUDA wheel from PyPI.

Check that the environment can see the GPU:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Hugging Face downloads work anonymously, but setting `HF_TOKEN` is recommended to obtain higher rate limits:

```bash
export HF_TOKEN=your_hugging_face_token
```

## Data, provenance, and leakage prevention

### VizWiz protocol

The intended experimental protocol is strict:

| Partition | Allowed purpose |
|---|---|
| VizWiz `train` | Model optimisation and replay mixing |
| VizWiz `validation` | Checkpoint selection, hyperparameter development, and threshold calibration |
| VizWiz `test` | One final evaluation per configuration frozen beforehand |

Training commands load only train and validation by default. They cannot accidentally evaluate the frozen test set. `paper-vqa-evaluate-development` is restricted to `data.validation`. Only `paper-vqa-evaluate` loads test, after loading a previously selected checkpoint.

The current VizWiz Hugging Face revisions are fixed in [configs/data/vizwiz.yaml](configs/data/vizwiz.yaml):

| Source | Dataset revision |
|---|---|
| `Multimodal-Fatima/VizWiz_train` | `0398414ffdf6ca445b9e90204ce5d0def5d73da2` |
| `Multimodal-Fatima/VizWiz_validation` | `54b3cc446d5138b72ca1b3539bc64fd1848cff70` |
| `Multimodal-Fatima/VizWiz_test` | `6de3ff2103743051521ac6cc001025a9b399a1f1` |

The hosted VizWiz test split contains the images and questions but not the answer annotations. Download the official annotation file once:

```bash
curl --fail --location --create-dirs \
  --output data/vizwiz/VQA_test.json \
  https://vizwiz.cs.colorado.edu/VizWiz_all_answers/VQA_test.json
```

The `huggingface_with_annotations` adapter then joins hosted test images/questions to the official JSON by filename. It fails if an identifier is duplicated, missing, extra, or associated with a different question. This makes test evaluation possible without silently using unlabelled `null` answers.

### Typed records and annotations

Every source is converted into an immutable `VQAExample` containing:

- a source-specific `sample_id`, dataset name, and split;
- the image or a lazy Hugging Face image reference;
- the question;
- all available human reference answers; and
- `answerable` in `{0, 1, None}` plus `has_answerable`.

VizWiz normally has ten references. They are retained for evaluation. A deterministic modal answer is used only as the single sequence target required by the decoder loss; it does not replace the references stored for scoring.

Hosted datasets are initially read without decoding all images. Images are loaded lazily only by the DataLoader or evaluator.

### Manifests and replay

Every run writes JSON manifests containing dataset name, split, sample IDs, revision, and seed. `assert_disjoint_manifests` aborts if a sample from the same dataset appears in more than one train, validation, test, or replay manifest.

Replay is selected with `replay=none`, `replay=textvqa`, `replay=vqav2`, or `replay=both`. TextVQA and VQAv2 samples have no answerability labels in the current adapters, so they contribute to the VQA loss but are excluded from the answerability loss. `sampling=proportional` concatenates sources; `sampling=source_weighted` uses a seeded weighted sampler. Source size, weight, split, revision, and sampling budget are Hydra values rather than Python constants.

VizWiz revisions are frozen now. Replay configuration currently leaves the TextVQA and VQAv2 revisions as `null`; freeze those revisions before any replay result is treated as publishable.

### Prepare and audit data

Validate the configured train and validation sources and write manifests:

```bash
uv run paper-vqa-prepare-data
```

Validate train, validation, and test explicitly:

```bash
uv run paper-vqa-prepare-data data.include_test=true
```

Create a reproducible descriptive audit. It records record counts, IDs, question lengths, reference-answer counts, answerability prevalence, frequent modal answers, previews, and a deterministic image-decoding sample.

```bash
uv run paper-vqa-audit-data
```

Avoid opening test images while developing:

```bash
uv run paper-vqa-audit-data audit.include_test=false
```

Hydra writes the audit JSON and manifests under the run directory in `outputs/`.

## Model and learning method

### BLIP, LoRA, and the answerability head

The base model is `Salesforce/blip-vqa-base` at revision `787b3d35d57e49572baabd22884b3d5a05acf072`.

- **BLIP-VQA** encodes the image, encodes the question while cross-attending to the image representation, and autoregressively decodes an answer.
- **LoRA** is optionally injected into the `query` and `value` projections. The default rank is 8, alpha is 16, and LoRA dropout is 0.05.
- **AnswerabilityHead** is optional. It receives the multimodal text-encoder states and predicts two logits ordered as `[unanswerable, answerable]`.

The default head is an MLP with the following layers:

```text
pooled BLIP states → Linear(768, 256) → LayerNorm → GELU → Dropout
                   → Linear(256, 64)  → GELU      → Dropout → Linear(64, 2)
```

All new linear layers use Xavier-uniform weights and zero biases. Its pooling is configurable:

- `masked_mean` is the default. It averages only question tokens whose `attention_mask` is one, so padding cannot alter the representation.
- `cls` uses the first sequence token and is available as an architectural ablation.

At inference, the head runs first. For examples below the threshold, `SelectiveVQAModel.predict` returns `prediction: null` and never invokes the text decoder. For accepted examples, it generates with the configured BLIP decoding parameters (`max_new_tokens=10`, `num_beams=5` by default). A VQA-only model has no head and is considered accepted with score 1.0.

### Multitask objective

The objective is:

```text
L_total = L_vqa + λ × L_answerability
```

`λ` is `loss.answerability_weight` and defaults to 0.25. `L_answerability` is two-class cross-entropy over the head logits, but only examples with `has_answerable=True` contribute. Optional class weights are supported through `loss.answerability_class_weights`.

BLIP's answer decoder is causal: the logits at position `t` predict the target token at `t + 1`. The implementation explicitly computes VQA cross-entropy as:

```text
cross_entropy(logits[:, :-1], labels[:, 1:])
```

with `-100` padding ignored. This next-token alignment is tested. It is essential: comparing each logit to the token at the same position produces an invalid language-modelling objective and can cause degenerate generations.

The VQA-loss policy is itself an ablation:

| Policy | Decoder loss uses | Motivation |
|---|---|---|
| `all_examples` | answerable and unanswerable VizWiz examples | Lets the decoder learn the benchmark's literal `"unanswerable"` targets. |
| `answerable_only` | only examples labelled answerable | Isolates whether any gain comes from decoder supervision on unanswerable examples rather than the head. |

The answerability head is supervised by labelled samples in both policies.

## Metrics and calibration

### VQA answer quality

`official_vqa_accuracy` preserves all references and applies VQA-style normalisation: lowercasing, article removal, punctuation removal, and whitespace normalisation. For a prediction with `m` matches among `N` references, it averages the leave-one-annotator-out score over all possible held-out annotators. With ten VizWiz references this is the usual VQA consensus principle, rather than a single-reference exact match.

### Answerability and safety

For a head-enabled model, the primary classification metric is **average precision (AP)**. The evaluator also reports:

- AUROC;
- F1, precision, recall, specificity, and balanced accuracy at the selected threshold;
- Brier score and equal-width expected calibration error (ECE, 15 bins by default);
- coverage, accepted-answer VQA accuracy, selective risk, and unsafe answer rate.

`unsafe_answer_rate` is the fraction of genuinely unanswerable questions for which the system emits an answer. Lower is better. `coverage` is the fraction of all questions for which it emits an answer. `selective_risk = 1 - accepted_vqa_accuracy`.

When no threshold is provided for a head-enabled development evaluation, the threshold is selected **only on validation**. Among all thresholds satisfying `evaluation.minimum_answerable_recall` (0.90 by default), the selector chooses the one with the smallest unsafe answer rate and, on ties, the largest threshold. The selected numerical threshold is model- and seed-specific; do not compare its absolute value across different models.

`bootstrap_confidence_interval` and `aggregate_seed_metrics` are available as library utilities for paper reporting. They are not yet exposed as a standalone aggregation or plotting CLI, and PR/ROC figures and confusion-matrix figures are not automatically written by the current evaluator.

## Configuration and reproducibility

All experiment-facing values live in Hydra YAML groups:

```text
data, replay, model, head, loss, trainer, evaluation, logging, experiment
```

The resolved configuration is saved in checkpoint metadata and Hydra's `.hydra/` directory. Override a value from the terminal with `group=value` or `key=value`:

```bash
uv run paper-vqa-train head=disabled trainer.epochs=1
uv run paper-vqa-train replay=both replay.sources.0.max_samples=3000
uv run paper-vqa-train head.pooling=cls
```

The run layout is:

```text
outputs/<experiment-name>/<timestamp>/seed_<seed>/
├── .hydra/                    # resolved Hydra configuration and overrides
├── manifests/                 # exact source records used
├── checkpoint/
│   ├── model.safetensors       # best model weights; no pickle
│   └── metadata.json           # selected epoch, monitor, config
└── development_evaluation/ or evaluation/
    ├── metrics.json
    └── predictions.json        # score, abstention, references, label
```

`seed_everything` seeds Python, NumPy, PyTorch, CUDA, DataLoader generators, and worker initialisers. It disables cuDNN benchmarking, requests deterministic algorithms with warnings for unsupported kernels, and sets `CUBLAS_WORKSPACE_CONFIG`. Exact bitwise reproducibility can still depend on hardware, CUDA, and third-party kernels; record the generated configuration and dependency lockfile with every reported run.

## Commands

All commands below are run from the repository root.

### Baseline

The zero-shot baseline forbids LoRA, the auxiliary head, and the test split:

```bash
uv run paper-vqa-baseline \
  model=blip_vqa_zero_shot \
  head=disabled \
  experiment=baseline \
  baseline.max_samples=512
```

### Train a VQA-only LoRA pilot

When the head is disabled, select checkpoints with `val/loss`, because `val/answerability_ap` is unavailable.

```bash
uv run paper-vqa-train \
  head=disabled \
  experiment.name=lora_vqa_pilot \
  data.train.max_samples=2048 \
  data.validation.max_samples=512 \
  trainer.epochs=3 \
  trainer.batch_size=4 \
  trainer.num_workers=0 \
  trainer.checkpoint_monitor=val/loss \
  trainer.checkpoint_mode=min \
  trainer.progress.leave=true \
  logging.enabled=false
```

### Train the selective multitask pilot

This is the current preferred starting point from the small development ablation. It enables the head, uses `all_examples`, and selects by validation AP by default.

```bash
uv run paper-vqa-train \
  head=mlp \
  loss.answerability_weight=0.25 \
  loss.vqa_loss_policy=all_examples \
  experiment.name=lora_answerability_pilot_all \
  data.train.max_samples=2048 \
  data.validation.max_samples=512 \
  trainer.epochs=3 \
  trainer.batch_size=4 \
  trainer.num_workers=0 \
  trainer.progress.leave=true \
  logging.enabled=false
```

To run the policy ablation, change only the policy and experiment name:

```bash
uv run paper-vqa-train \
  head=mlp \
  loss.answerability_weight=0.25 \
  loss.vqa_loss_policy=answerable_only \
  experiment.name=lora_answerability_pilot_answerable_only \
  data.train.max_samples=2048 \
  data.validation.max_samples=512 \
  trainer.epochs=3 \
  trainer.batch_size=4 \
  trainer.num_workers=0 \
  trainer.progress.leave=true \
  logging.enabled=false
```

The terminal shows separate training and validation progress bars, running loss, generation loss, and an epoch summary. The `val/answerability_ap` metric is logged during head-enabled training and can be used for checkpoint selection. Set `trainer.progress.enabled=false` to hide bars.

### Development evaluation on validation

Evaluate a VQA-only checkpoint without opening test:

```bash
uv run paper-vqa-evaluate-development \
  head=disabled \
  experiment.name=lora_vqa_development \
  evaluation.checkpoint_path=outputs/<training-run>/seed_42/checkpoint \
  development.max_samples=512 \
  trainer.batch_size=4 \
  trainer.num_workers=0 \
  logging.enabled=false
```

Evaluate a head-enabled checkpoint. Omitting `development.threshold` triggers validation-only calibration under the configured recall constraint:

```bash
uv run paper-vqa-evaluate-development \
  head=mlp \
  experiment.name=selective_development \
  evaluation.checkpoint_path=outputs/<training-run>/seed_42/checkpoint \
  development.max_samples=512 \
  trainer.batch_size=4 \
  trainer.num_workers=0 \
  trainer.progress.leave=true \
  logging.enabled=false
```

For a diagnostic that accepts every sample and therefore measures the multitask decoder without abstention, set a zero threshold:

```bash
uv run paper-vqa-evaluate-development \
  head=mlp \
  experiment.name=selective_threshold_zero \
  evaluation.checkpoint_path=outputs/<training-run>/seed_42/checkpoint \
  development.max_samples=512 \
  development.threshold=0.0 \
  trainer.batch_size=4 \
  trainer.num_workers=0 \
  logging.enabled=false
```

This diagnostic has recall 1.0 and unsafe answer rate 1.0 by construction. Its thresholded F1, specificity, and safety metrics are not meaningful; AP, AUROC, Brier, ECE, and raw VQA accuracy remain useful.

### Frozen final test evaluation

Do this only after the architecture, hyperparameters, replay policy, selection metric, and threshold rule have been frozen through validation. The command calibrates on validation if necessary, then evaluates test:

```bash
uv run paper-vqa-evaluate \
  experiment=final \
  evaluation.checkpoint_path=outputs/<selected-training-run>/seed_42/checkpoint
```

Do not repeatedly run this command while choosing hyperparameters. That would turn test into a development set and invalidate the final result.

### Five-seed multirun

One seed is for debugging and pilots only. After the protocol is fixed, launch independent training runs with Hydra:

```bash
uv run paper-vqa-train -m \
  experiment=final \
  trainer.seed=41,42,43,44,45
```

Each run writes separate checkpoints and manifests under `multirun/`. Evaluate selected checkpoints according to the frozen test protocol, then aggregate the resulting per-seed metrics with mean and sample standard deviation for the paper. The repository contains the aggregation utility but not yet a dedicated aggregation CLI.

### Weights & Biases

W&B is disabled by default. Enable it for a remote or local experiment record:

```bash
uv run paper-vqa-train \
  logging.enabled=true \
  logging.mode=online \
  logging.project=paper-vqa \
  logging.entity=<your-wandb-entity> \
  logging.group=pilot
```

Use `logging.mode=offline` if the machine has no network connection. The tracker stores the resolved configuration in the W&B run and logs epoch scalar metrics. It uploads manifests, selected checkpoints, and evaluation directories as W&B artifacts when enabled. Hyperparameter sweeps can invoke the same Hydra overrides, for example over `loss.answerability_weight`, `head.pooling`, replay weights, or replay sample budgets.

## Current pilot evidence

The following figures are useful pipeline checks, not paper results. They all use one seed, 2,048 VizWiz training records, and the same deterministic 512-example VizWiz validation subset.

| Configuration | VQA Accuracy | Notes |
|---|---:|---|
| Frozen BLIP-VQA baseline | 0.2023 | No LoRA, no head |
| LoRA VQA only | 0.5723 | No abstention |
| LoRA + head, `all_examples`, threshold 0 | 0.5686 | Decoder quality control; no abstention |
| LoRA + head, `answerable_only`, threshold 0 | 0.4924 | Decoder policy ablation; no abstention |
| LoRA + head, `all_examples`, calibrated | 0.3139 | Coverage 0.7090, AP 0.9346, AUROC 0.8781, unsafe rate 0.3446 |
| LoRA + head, `answerable_only`, calibrated | 0.2828 | Coverage 0.7129, AP 0.9300, AUROC 0.8703, unsafe rate 0.3559 |

This pilot suggests that `all_examples` is the stronger starting policy under the current settings. It does **not** establish a publishable improvement: no replay ablation, loss-weight sweep, pooling ablation, confidence interval, or five-seed analysis has yet been completed.

## What changed relative to the TFG

The TFG implementation was a complete accessibility application with a React frontend, a REST backend, image description, and a VQA component. This repository deliberately narrows the scope to the VQA research question. The central changes are methodological as well as structural.

### Research protocol and data logic

| TFG implementation | `paper-vqa` |
|---|---|
| Notebook-driven experimentation with manually chosen dataset compositions and splits. | A Python package with declarative Hydra configuration, typed dataset records, and run-local resolved configuration. |
| VizWiz data was manually split internally for development; replay data also used ad-hoc partitions. | Official VizWiz train/validation/test roles are separated. Test labels are joined safely from the official release only for one frozen evaluation. |
| Dataset versions and exact sampled IDs were not recorded as experiment artefacts. | Fixed VizWiz revisions, deterministic sub-sampling, immutable manifests, and overlap checks are written for every run. |
| External replay was mixed by notebook logic. | Replay is a separate configuration group with source, split, revision, seed, weight, and sampler policy. Unlabelled replay cannot train the answerability head. |
| Evaluation and qualitative inspection were intertwined with development. | Development evaluation is validation-only; a separate final command is explicitly responsible for calibration followed by test evaluation. |

### Model and learning logic

| TFG implementation | `paper-vqa` |
|---|---|
| BLIP-VQA + LoRA rank 8 on `query` and `value`, plus a manually defined respondibility MLP. | The same research idea is retained, but built through typed factories and ablated explicitly as BLIP-only, LoRA-only, and LoRA + head. |
| Mean pooling averaged every sequence position, including question padding. | Default `masked_mean` pooling excludes padded question tokens; `cls` is configurable for a controlled ablation. |
| The head produced a score around a manually fixed 0.5 decision boundary and the decoder still generated an answer. | The score is calibrated on validation under a configurable answerable-recall constraint. Rejected inputs cause real abstention: no answer is generated. |
| A single VQA loss was consumed directly from the model. | The explicit causal next-token loss exposes logits safely and supports the controlled `all_examples` versus `answerable_only` policy. |
| The original implementation did not isolate whether decoder supervision on unanswerable cases or the head caused any observed effect. | The loss policy ablation separates those mechanisms while preserving the same answerability supervision. |
| Randomness was seeded in the notebook. | Python, NumPy, PyTorch, CUDA, DataLoader generators, and workers are seeded; deterministic PyTorch settings are requested and recorded. |

The explicit next-token implementation also corrects an important training issue discovered during the refactor. BLIP decoder logits must be aligned with the *following* target token. The tested formulation uses `logits[:, :-1]` against `labels[:, 1:]`; a same-position comparison is not an autoregressive VQA objective and produced degenerate answers in the early pilot.

### Metrics and reporting logic

| TFG implementation | `paper-vqa` |
|---|---|
| Development focused on loss, response-classification accuracy, qualitative examples, and simplified VQA comparisons. | VQA references are preserved and scored with official-style leave-one-annotator-out consensus. |
| A single response-classification accuracy can conceal imbalance, ranking quality, calibration, and safety trade-offs. | AP is primary; AUROC, F1, precision, recall, specificity, balanced accuracy, Brier, and ECE are reported. |
| A score alone did not define a safe selective policy. | Coverage, accepted VQA accuracy, selective risk, and unsafe answer rate quantify the cost and benefit of abstention. |
| Checkpoint choices and artefacts were managed manually. | Validation-only checkpoint monitoring, early stopping, `safetensors` weights, JSON metadata, predictions with references, manifests, and optional W&B artifacts are automatic. |

One subtle but important point: VizWiz can reward a generated literal answer such as `"unanswerable"` through the VQA references, while a deployed selective system should often abstain instead. For this reason, raw VQA accuracy, calibrated selective VQA metrics, and answerability metrics must all be reported together. None alone describes the safety of the system.

## Quality checks

Run the full local quality suite:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

The tests cover answerability masking and pooling, causal next-token VQA loss alignment, reference preservation, official-style metrics, calibration, leakage detection, hybrid test joins, deterministic sampling, baseline restrictions, and a training smoke test.

## Data use and limitations

VizWiz is distributed under CC BY 4.0; consult the [official VizWiz documentation](https://vizwiz.org/tasks-and-datasets/vqa/) for data terms, downloads, annotation format, and citation requirements. Local data, model caches, W&B runs, Hydra outputs, and experiment results are ignored by Git.

This code is a research implementation, not a safety-certified assistive product. Before deployment, evaluate on held-out data, inspect harmful failure cases with affected users and accessibility experts, consider abstention wording and user interaction design, and conduct an appropriate privacy and security review for uploaded images.
