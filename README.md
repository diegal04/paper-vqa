# paper-vqa

`paper-vqa` is a reproducible research codebase for **selective Visual Question Answering (VQA)** on photographs taken by blind people. It studies whether a BLIP-VQA model adapted with LoRA can answer a visual question *and* estimate whether the image-question pair is answerable. When the answerability score is below a validation-calibrated threshold, the system abstains and does not generate an answer.

This repository is deliberately independent from the original TFG application. It does not import code from, or modify, `mi-tfg`. The original project is used only as historical context for the research refactor described below.

Experimental hypotheses, run artifacts, results, negative findings, and research decisions are
tracked separately in [EXPERIMENTS.md](EXPERIMENTS.md). The README describes the current system
and how to reproduce it; the experiment log explains why the current choices were made.

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
├── sweeps/                           # versioned W&B smoke and Bayesian searches
├── tests/                           # unit and smoke tests
├── data/                            # local benchmark annotations; ignored by Git
├── outputs/                         # Hydra run artefacts; ignored by Git
├── EXPERIMENTS.md                   # living experiment log and decision register
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

The original multitask objective is:

```text
L_total = L_vqa + λ × L_answerability
```

`λ` is `loss.answerability_weight` and defaults to 0.25. `L_answerability` is two-class cross-entropy over the head logits, but only examples with `has_answerable=True` contribute. Optional class weights are supported through `loss.answerability_class_weights`.

The weighted formulation additionally separates positive and negative decoder supervision:

```text
L_total = L_vqa_answerable + β × L_vqa_unanswerable + λ × L_answerability
```

`β` is `loss.unanswerable_vqa_weight`. It controls how strongly the decoder learns the safe fallback on explicitly unanswerable records: `β=1` recovers `all_examples`, `β=0` recovers the decoder masking of `answerable_only`, and intermediate values retain fallback supervision without giving negative examples full decoder weight. Unlabelled replay examples always retain decoder weight 1 and never contribute to the answerability loss.

BLIP's answer decoder is causal: the logits at position `t` predict the target token at `t + 1`. The implementation explicitly computes VQA cross-entropy as:

```text
cross_entropy(logits[:, :-1], labels[:, 1:])
```

with `-100` padding ignored. This next-token alignment is tested. It is essential: comparing each logit to the token at the same position produces an invalid language-modelling objective and can cause degenerate generations.

The VQA-loss policy is itself an ablation:

| Policy | Decoder loss uses | Motivation |
|---|---|---|
| `all_examples` | answerable and unanswerable VizWiz examples | Lets the decoder learn the benchmark's literal `"unanswerable"` targets. |
| `answerable_only` | labelled answerable examples and unlabelled replay | Isolates whether any gain comes from decoder supervision on explicitly unanswerable examples rather than the head. |

The answerability head is supervised by labelled samples in both policies.

Decoder targets are also configurable through `loss.vqa_target_policy`:

| Target policy | Behaviour |
|---|---|
| `modal` | Uses the first most-frequent answer among all references; this preserves the original experimental behaviour. |
| `label_consistent_modal` | Uses the most-frequent non-`"unanswerable"` reference for an answerable record, forces `"unanswerable"` for an unanswerable record, and preserves the modal target for unlabelled replay. |

The ready-made `loss=label_consistent_weighted` group combines `label_consistent_modal`, `β=0.25`, `λ=0.25`, and the `all_examples` policy. It is an experimental candidate and does not replace the historical pilot configurations silently.

## Metrics and calibration

### VQA answer quality

`official_vqa_accuracy` preserves all references and follows the [official VQA evaluator](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py)'s conditional normalisation: whitespace cleanup, punctuation handling, number-word mapping, article removal, lowercasing, and contraction mapping. As in the reference implementation, full normalisation is applied when the cleaned human references are not unanimous. For a prediction with `m` matches among `N` references, it averages the leave-one-annotator-out score over all possible held-out annotators. With ten VizWiz references this is the official consensus principle rather than a single-reference exact match. Parity cases cover punctuation, decimal commas, number words, contractions, unanimous references, and held-out consensus.

### Answerability and safety

For a head-enabled model, the primary classification metric is **average precision (AP)**. The evaluator also reports:

- AUROC;
- F1, precision, recall, specificity, and balanced accuracy at the selected threshold;
- Brier score and equal-width expected calibration error (ECE, 15 bins by default);
- coverage, accepted-answer VQA accuracy, selective risk, and unsafe answer rate;
- the specific unsafe-answer rate: genuinely unanswerable questions that receive an
  emitted answer other than the literal `"unanswerable"`;
- VQA accuracy restricted to genuinely answerable questions, with abstentions scored as zero; and
- the fraction of emitted answers that literally say `"unanswerable"` after VQA normalisation.

`unsafe_answer_rate` is the fraction of genuinely unanswerable questions accepted by the
answerability head. `specific_unsafe_answer_rate` uses the same denominator but excludes
accepted outputs whose normalised text is `"unanswerable"`; it therefore isolates the cases
in which a head error becomes a specific, potentially hallucinated response. Lower is better.
`coverage` is the fraction of all questions for which the head emits an answer.
`selective_risk = 1 - accepted_vqa_accuracy`. These diagnostics separate useful answer
quality from the benchmark reward obtained by generating the literal token `"unanswerable"`.

When no threshold is provided for a head-enabled development evaluation, the threshold is selected **only on validation**. Among all thresholds satisfying `evaluation.minimum_answerable_recall` (0.90 by default), the selector chooses the one with the smallest unsafe answer rate and, on ties, the largest threshold. The selected numerical threshold is model- and seed-specific; do not compare its absolute value across different models.

### Formal abstention policies

Scientific evaluation generates each answer exactly once and then applies every selection policy to that same output. This avoids attributing decoding differences to the selector:

| Policy | Emits an answer when |
|---|---|
| `always_answer` | Always; this is the non-selective VQA reference. |
| `decoder_literal` | The decoded answer is not the normalised literal `"unanswerable"`. |
| `decoder_confidence` | The geometric mean generated-token probability exceeds its validation-calibrated threshold. |
| `decoder_confidence_plus_literal` | Decoder confidence accepts and the decoded answer is not literal `"unanswerable"`. |
| `head_only` | The auxiliary head score exceeds its validation-calibrated threshold. |
| `hybrid` | The head accepts and the decoder does not output literal `"unanswerable"`. |

The head and decoder-confidence thresholds are calibrated independently on validation under the same answerable-recall constraint. Frozen-test evaluation first generates the validation partition to fix both thresholds, then applies those unchanged thresholds to test. It never calibrates decoder confidence on test.

For a fair safety comparison after textual abstention, Hydra also declares fixed answerable-emission recall targets in `evaluation.matched_answerable_recall_targets`. Decoder-confidence-plus-literal and hybrid thresholds are calibrated independently on validation to reach each target. `matched_answerable_recall.csv` then compares their achieved recall, coverage, specific unsafe rate, emission answerability precision, accepted VQA, and exact counts. Frozen-test evaluation transports these validation thresholds unchanged.

`policy_metrics.json` records every operating point and the matched-recall table. `risk_coverage.csv` and `risk_coverage.svg` contain confidence sweeps for decoder confidence, decoder-confidence-plus-literal, head-only selection, and the hybrid system. The SVG is generated without an additional plotting dependency and the CSV is the source for later paper figures. `evaluation.risk_coverage_points` controls the maximum number of deterministic curve points.

`bootstrap_confidence_interval` and `aggregate_seed_metrics` remain available as library utilities. A standalone multi-run aggregation command, PR/ROC figures, reliability diagrams, and confusion-matrix figures remain future reporting work.

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

When `trainer.warmup_ratio` is not `null`, it takes precedence over `trainer.warmup_steps` and scales the warmup with the planned number of optimizer updates. This is the preferred setting for convergence, full-data, and hyperparameter-search runs because an absolute step count changes meaning with dataset size and epoch ceiling. The end-of-epoch learning rate is stored in both `history.json` and W&B.

The run layout is:

```text
outputs/<experiment-name>/<timestamp>/seed_<seed>/
├── .hydra/                    # resolved Hydra configuration and overrides
├── manifests/                 # exact source records used
├── history.json               # local train/validation metrics after every epoch
├── checkpoint/
│   ├── model.safetensors       # best trainable LoRA + head weights; no pickle
│   └── metadata.json           # selected epoch, monitor, config
└── development_evaluation/ or evaluation/
    ├── metrics.json
    ├── predictions.json        # raw generation, both scores, decision, references
    ├── policy_metrics.json     # every fixed abstention-policy operating point
    ├── matched_answerable_recall.csv # policy comparison at fixed useful recall
    ├── risk_coverage.csv       # long-form curve data
    └── risk_coverage.svg       # vector figure
```

New checkpoints store only trainable parameters. The frozen BLIP base is reconstructed from the
model identifier and immutable revision in `metadata.json`; the LoRA adapters and answerability
head are then restored from `model.safetensors`. For the current rank-8 model this reduces a
checkpoint from roughly 1.4 GB to 5.6 MB. The loader remains compatible with earlier full-model
checkpoints (`format_version=1`).

Policy reports can be rebuilt from saved validation predictions without loading BLIP or decoding
the images again. The command is deliberately restricted to validation so it cannot recalibrate
thresholds on frozen test predictions:

```bash
uv run paper-vqa-report-development-policies \
  policy_report.predictions_path=outputs/<run>/seed_42/development_evaluation/predictions.json \
  policy_report.output_dir=outputs/<run>/seed_42/development_evaluation/policy_report_v2 \
  logging.enabled=false
```

`seed_everything` seeds Python, NumPy, PyTorch, CUDA, DataLoader generators, and worker initialisers. It disables cuDNN benchmarking, requests deterministic algorithms with warnings for unsupported kernels, sets `CUBLAS_WORKSPACE_CONFIG`, and disables Hugging Face Tokenizers' internal Rayon pool through `TOKENIZERS_PARALLELISM=false`. It also applies `trainer.cpu_threads` to OpenMP, MKL, OpenBLAS, NumExpr, and PyTorch thread pools; the default of one avoids oversubscription on the university server while GPU computation remains unaffected. Exact bitwise reproducibility can still depend on hardware, CUDA, and third-party kernels; record the generated configuration and dependency lockfile with every reported run.

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

The terminal shows separate training and validation progress bars, running loss, generation loss, and an epoch summary. The `val/answerability_ap` metric is logged during head-enabled training and can be used for checkpoint selection. Set `trainer.progress.enabled=false` to hide bars. Every completed epoch is also written atomically to `history.json`, including the global step, all train/validation losses, answerability AP, and whether that epoch selected the checkpoint. This local record is always created, even when W&B is disabled.

### Train the label-consistent weighted candidate

This candidate prevents an officially answerable record from using `"unanswerable"` as its decoder target, while retaining a down-weighted safe fallback for explicitly unanswerable records:

```bash
uv run paper-vqa-train \
  head=mlp \
  loss=label_consistent_weighted \
  experiment.name=label_consistent_beta025_pilot \
  data.train.max_samples=2048 \
  data.validation.max_samples=512 \
  trainer.epochs=3 \
  trainer.batch_size=4 \
  trainer.num_workers=0 \
  trainer.checkpoint_monitor=val/answerability_ap \
  trainer.checkpoint_mode=max \
  trainer.progress.leave=true \
  logging.enabled=false
```

Override `loss.unanswerable_vqa_weight` to ablate β without changing target selection or λ.

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

Evaluate a head-enabled checkpoint. Omitting `development.threshold` triggers validation-only calibration under the configured recall constraint. The command generates once per example and writes the six-policy comparison, matched-recall table, and risk--coverage outputs:

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

W&B is disabled by default. First validate the integration in offline mode, which does not send the run to the service:

```bash
uv run paper-vqa-train \
  experiment.name=wandb_offline_smoke \
  data.train.max_samples=16 \
  data.validation.max_samples=8 \
  trainer.epochs=1 \
  trainer.batch_size=4 \
  trainer.num_workers=0 \
  trainer.warmup_steps=0 \
  trainer.checkpoint_monitor=val/answerability_ap \
  trainer.checkpoint_mode=max \
  logging.enabled=true \
  logging.mode=offline \
  logging.group=setup
```

The W&B offline files are kept inside that Hydra run directory. To use the online dashboard and Sweeps, create a W&B account and log in locally. Keep the API key outside the repository:

```bash
uv run wandb login
```

After inspecting the offline smoke run and logging in, repeat it online:

```bash
uv run paper-vqa-train \
  experiment.name=wandb_online_smoke \
  data.train.max_samples=16 \
  data.validation.max_samples=8 \
  trainer.epochs=1 \
  trainer.batch_size=4 \
  trainer.num_workers=0 \
  trainer.warmup_steps=0 \
  trainer.checkpoint_monitor=val/answerability_ap \
  trainer.checkpoint_mode=max \
  logging.enabled=true \
  logging.mode=online \
  logging.project=paper-vqa \
  logging.entity=YOUR_WANDB_ENTITY \
  logging.group=setup
```

Runs have the stable display name `<experiment.name>-seed_<trainer.seed>`. Use `logging.group` to collect related runs, for example `convergence`, `hpo`, `full-data`, `replay`, or `final-5seeds`. Training logs all available epoch scalars and development/final evaluation commands log every available VQA, answerability, calibration, and selective metric rather than only VQA accuracy. Whenever validation selects a new checkpoint, training also logs `selection/*`; consequently `selection/answerability_ap` remains the AP of the final selected checkpoint rather than the AP of the last, potentially overfitted epoch.

Artifact upload is deliberately conservative:

| Setting | Default | Uploaded content |
|---|---:|---|
| `logging.upload_manifests` | `true` | Exact dataset manifests and provenance |
| `logging.upload_checkpoints` | `false` | The final validation-selected checkpoint, considered only once after training |
| `logging.upload_predictions` | `false` | Evaluation directories, including per-example predictions |

Checkpoint upload remains disabled during searches even though new adapter-plus-head checkpoints are compact (approximately 5.6 MB for rank 8). Prediction files contain dataset-derived questions, references, and model outputs and must not be uploaded casually. Metrics, Hydra configuration, local checkpoints, local predictions, and `history.json` remain available regardless of those upload switches. An offline run can later be uploaded with `uv run wandb sync <offline-run-directory>`.

### Bounded W&B hyperparameter search

The versioned sweep keeps the already selected scientific choices fixed: no replay,
`masked_mean`, modal targets, decoder supervision on all examples, and full weight for
unanswerable decoder examples. It searches learning rate, weight decay, LoRA rank and
alpha-to-rank ratio, LoRA dropout, head width and dropout, proportional warmup, and
`lambda ∈ {0.10, 0.25}`. Screening uses seed 42, 4,096 train examples, 1,024 validation examples,
a six-epoch ceiling, and validation-AP early stopping. W&B optimises
`selection/answerability_ap`, the AP attached to the saved best checkpoint.

First create and execute the one-run integration smoke:

```bash
uv run wandb sweep \
  --entity YOUR_WANDB_ENTITY \
  --project paper-vqa \
  sweeps/hpo_smoke.yaml

uv run wandb agent --count 1 YOUR_WANDB_ENTITY/paper-vqa/SMOKE_SWEEP_ID
```

After verifying that the smoke finishes, that `selection/answerability_ap` appears in W&B, and
that its local checkpoint metadata declares `weights_scope: trainable`, create the Bayesian
screening sweep:

```bash
uv run wandb sweep \
  --entity YOUR_WANDB_ENTITY \
  --project paper-vqa \
  sweeps/hpo_screening.yaml

# Run three trials first and inspect them before committing the remaining budget.
uv run wandb agent --count 3 YOUR_WANDB_ENTITY/paper-vqa/SCREENING_SWEEP_ID

# Continue the same sweep after the three-run gate passes (24 total trials).
uv run wandb agent --count 21 YOUR_WANDB_ENTITY/paper-vqa/SCREENING_SWEEP_ID
```

Random and Bayesian W&B agents do not stop by themselves, so `--count` is mandatory. The sweep
optimises AP only as a cheap screening signal; it does not declare the winner. The leading
checkpoints must subsequently be generated over validation and selected from a Pareto comparison
of AP, specific unsafe rate at matched answerable recall, answerable VQA, and calibration.

## Experiment history

The README intentionally does not duplicate changing result tables. See
[EXPERIMENTS.md](EXPERIMENTS.md) for the chronological experiment record, comparable pilot
tables, invalid-run diagnoses, negative results, current decisions, and planned studies.

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
| A score alone did not define a safe selective policy. | Coverage, accepted VQA accuracy, selective risk, unsafe answer rate, and specific unsafe-answer rate quantify the cost and benefit of abstention while separating textual abstentions from specific responses. |
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
