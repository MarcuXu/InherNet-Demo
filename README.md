# HeteroInherNet

Official research implementation for **HeteroInherNet: Data-Aware
Heterogeneous Neural Network Inheritance**.

This repository extends the InherNet paper, **Beyond Student: An Asymmetric
Network for Neural Network Inheritance**
([arXiv:2602.09509](https://arxiv.org/abs/2602.09509)). InherNet remains
available as a baseline in this codebase, but the main method exposed by this
repository is `hetero`: a data-aware heterogeneous extension that replaces
uniform low-rank inheritance with layer-wise rank allocation guided by
activation-weighted spectra.

The maintained entry point is `demo_code.py`, normally launched through
`run.sh`. `demo_code_org.py` is preserved as the legacy InherNet reference
script. Some internal names and environment variables still use `inhernet` for
backward compatibility with earlier experiments.

## Method Overview

HeteroInherNet starts from a trained dense source network and constructs a
compact inheriting network by factorizing source layers. The source can be a
trained teacher or, for compatibility experiments, a trained compact baseline.
The important principle is **trained-source inheritance**: the decomposed
weights should already contain task knowledge.

For a dense layer with weight `W` and input covariance

```text
Sigma_x = C C^T,
```

HeteroInherNet analyzes the data-weighted operator

```text
W_tilde = W C.
```

Truncated SVD of `W_tilde` is the optimal rank-r approximation under the
data-weighted reconstruction norm

```text
||W - W_hat||^2_{Sigma_x}
  = tr((W - W_hat) Sigma_x (W - W_hat)^T)
  = ||(W - W_hat) C||_F^2.
```

For convolutional layers, the implementation uses a practical channel-covariance
approximation rather than a full im2col covariance. This keeps calibration
tractable while still making the spectral analysis data-aware.

## What Hetero Adds

| Component | InherNet baseline | HeteroInherNet |
|---|---|---|
| Decomposition target | Raw source weight `W` | Data-weighted operator `W C` |
| Rank policy | Uniform rank preset | Entropy-budgeted layer-wise ranks |
| Gate input | Compressed feature gate | Compressed or uncompressed gate based on rank |
| Expert initialization | Identical SVD up heads | Zero-mean expert perturbations |
| Collapse control | Implicit through training | Explicit load-balance regularization |
| Default targets | Conv and linear layers | Conv layers by default, linear optional |

The layer score is the spectral entropy of the data-weighted singular spectrum:

```text
p_{l,i} = sigma_{l,i}^2 / sum_j sigma_{l,j}^2
H_l = - sum_i p_{l,i} log p_{l,i}.
```

Given a total rank budget, HeteroInherNet allocates more rank to layers whose
spectral energy is more diffuse and less rank to layers whose spectrum is
concentrated. The temperature parameter smooths this allocation so that the
budget does not collapse onto a small number of layers.

## Implementation Notes

- `GenericInherNet` implements the InherNet-style one-down-many-ups module with
  SVD initialization. The maintained implementation uses balanced factors
  `U sqrt(S)` and `sqrt(S) V^T`; with softmax gates that sum to one, identical
  expert heads exactly reconstruct the rank-r approximation at initialization.
- `GenericHeteroNet` keeps the same differentiable topology, but computes
  activation covariances, decomposes the data-weighted operator, allocates ranks
  by spectral entropy, and applies rank-dependent routing.
- Hetero expert noise is zero-mean across heads, so the average expert remains
  centered on the SVD reconstruction while early expert symmetry is broken.
- The load-balance loss is
  `H * sum_h mean_batch(g_h)^2`; it is minimized by uniform expert usage and is
  used as a training stabilizer, not as the central theory claim.
- Distillation uses PyTorch
  `kl_div(log_softmax(student / T), softmax(teacher / T))`, which corresponds to
  `KL(teacher || student)` with the usual `T^2` scaling.
- Decomposed models are trained end-to-end after initialization. Inherited
  layers are not frozen.

## Installation

Install the dependencies in your experiment environment:

```bash
pip install -r requirements.txt
```

The expanded vision/text experiments require the Hugging Face `datasets` and
`transformers` packages listed in `requirements.txt`. GLUE/SST-2 files and
tokenizers are cached under `data/huggingface/`.

In this workspace, use the existing conda environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate inherdemo
```

## Quick Start

Run a fast smoke test:

```bash
./run.sh --suite all --dataset cifar100 --pair resnet56_to_resnet20 -- \
  --smoke-test --plot-mode none --device cuda --svd-backend device
```

Run the Oxford-IIIT Pet vision target:

```bash
./run.sh --suite comparison --dataset oxford_pets \
  --pair resnet34_to_resnet18 -- \
  --download --device cuda --svd-backend device
```

Run one GLUE language target:

```bash
./run.sh --dataset glue --glue-task sst2 --pair bert4_to_bert2 \
  --method hetero --device cuda --svd-backend device --plot-mode none
```

Run the default HeteroInherNet method on CIFAR-100:

```bash
./run.sh --dataset cifar100 --pair resnet56_to_resnet20 --method hetero \
  --download --device cuda
```

Run a comparison suite containing teacher, KD student, InherNet small/large, and
HeteroInherNet:

```bash
./run.sh --background --suite comparison --dataset cifar100 \
  --pair resnet56_to_resnet20 -- --download --device cuda
```

Run all supported methods for one pair:

```bash
./run.sh --background --suite all --dataset cifar100 \
  --pair resnet56_to_resnet20 -- --download --device cuda
```

Suite-mode arguments after `--` are passed to `demo_code.py`. For example, use
`-- --epochs 10 --plot-mode none` for a short training check.

## Hetero Options

The main HeteroInherNet options are:

```bash
--budget-ratio 0.35
--min-rank 8
--hetero-temperature 1.4
--compress-threshold 12
--max-calib-batches 16
--hetero-expert-noise-scale 0.01
--aux-loss-weight 0.01
--hetero-compress-linear
```

The default CIFAR configuration compresses convolutional layers only. Add
`--hetero-compress-linear` to include linear layers in the heterogeneous
decomposition.

For GLUE datasets, linear-layer compression is enabled by the pair default because
transformer feed-forward and classifier blocks are linear-heavy. The CLI flag is
still available as an explicit override for other pairs.

## Supported Datasets

The repository now covers small image classification, fine-grained natural-image
classification, and the GLUE language tasks used by the InherNet paper without
moving into ImageNet-scale training.

| Dataset key | Task | Pair | Default model scale | Why it is included |
|---|---|---|---|---|
| `cifar10` | 10-class image classification | `resnet50_to_resnet18_org` | legacy torchvision ResNets | Backward-compatible InherNet reference workflow. |
| `cifar100` | 100-class image classification | `resnet56_to_resnet20`, `vgg13_to_vgg8`, `wrn40_2_to_wrn16_2`, ... | CIFAR-native CNNs | Paper-style teacher inheritance and current local result logs. |
| `oxford_pets` | 37-class fine-grained pet breed classification | `resnet34_to_resnet18` | ResNet-34 teacher, ResNet-18 student, 128x128 images | Small natural-image target with more realistic scale, pose, and lighting variation than CIFAR while remaining practical on one NVIDIA A6000. |
| `glue_mrpc`, `glue_qqp`, `glue_sst2`, `glue_mnli`, `glue_rte`, `glue_qnli`, `glue_cola`, `glue_stsb` | GLUE sentence classification/regression | `bert4_to_bert2` | Google BERT-Mini teacher, BERT-Tiny student | Text benchmark coverage matching Table 2 of the InherNet paper while remaining tractable on one NVIDIA A6000. |

The added datasets were selected for practical single-GPU research:

- Oxford-IIIT Pet has 37 categories with roughly 200 images per class and is
  available through `torchvision.datasets.OxfordIIITPet`.
- The GLUE tasks are run with compact BERT checkpoints. Larger tasks such as
  QQP, MNLI, and QNLI are larger than SST-2 but still practical with the
  registered small-BERT pair; CoLA, MRPC, RTE, and STS-B are small diagnostic
  tasks.
- ImageNet is intentionally not included. Its dataset size and common teacher
  models would make iteration slow and expensive for this repository's current
  single-A6000 target.

## Metrics And Logs

Use the following metrics for paper tables:

| Dataset key | Primary comparison metric | Evaluation split | Recommended secondary values |
|---|---|---|---|
| `cifar10` / `cifar100` | Top-1 accuracy (%) | test | Cross-entropy loss, parameter count, compression ratio. |
| `oxford_pets` | Top-1 accuracy (%) | test | Macro-F1, balanced accuracy, cross-entropy loss, parameter count, compression ratio. Top-1 accuracy remains the standard headline metric; Macro-F1 and balanced accuracy are logged as class-balance diagnostics. |
| `glue_mrpc` / `glue_qqp` | Accuracy (%) | validation | F1, cross-entropy loss, parameter count, compression ratio. |
| `glue_sst2` / `glue_mnli` / `glue_rte` / `glue_qnli` | Accuracy (%) | validation, except MNLI uses `validation_matched` | Cross-entropy loss, parameter count, compression ratio. |
| `glue_cola` | Matthews correlation coefficient (%) | validation | Accuracy, cross-entropy loss, parameter count, compression ratio. |
| `glue_stsb` | Pearson correlation coefficient (%) | validation | Spearman correlation coefficient, MSE loss, parameter count, compression ratio. |

Plots are optional. The logs are designed to be sufficient for table extraction
when running with `--plot-mode none`.

Each run log contains:

- `RUN_METADATA`: one JSON object with dataset, pair, method, task type,
  training settings, teacher/student names, parameter count, evaluation split,
  primary metric, all requested metric names, and method-specific compression
  metadata.
- `RUN_METRICS`: one JSON object per epoch with train objective, train loss,
  task-specific train metrics, evaluation loss, task-specific evaluation
  metrics, epoch time, and average batch time. `test_loss` is kept as an
  evaluation-loss alias. Accuracy tasks also keep `test_accuracy`; non-accuracy
  tasks use explicit keys such as `validation_matthews_correlation` or
  `validation_pearson`.
- A human-readable epoch line, for example
  `validation_loss=... | validation_accuracy=...` for accuracy-based GLUE
  tasks, `validation_matthews_correlation=...` for CoLA, and
  `test_macro_f1=...` / `test_balanced_accuracy=...` inside the structured
  metrics for Oxford-IIIT Pet.
- `RUN_SUMMARY`: one JSON object at the end of training with best evaluation
  primary metric, best epoch, final evaluation primary metric, final evaluation
  loss, and the primary metric display name.

## Supported Workflows

### CIFAR-100 Paper-Style Inheritance

CIFAR-100 pairs follow the teacher-inheritance workflow:

- Teacher/student pairs include `resnet56_to_resnet20`, `vgg13_to_vgg8`,
  `wrn40_2_to_wrn16_2`, and related CIFAR-native architectures.
- The compressed source defaults to the trained teacher.
- Compressed models train with knowledge distillation by default.
- Training defaults: SGD, LR `0.05`, momentum `0.9`, weight decay `5e-4`, batch
  size `64`, `240` epochs, milestones at `150`, `180`, and `210`.
- KD defaults: temperature `2`, KD weight `9`, CE weight `0.1`.

### CIFAR-10 Legacy Compatibility

`resnet50_to_resnet18_org` preserves the original-style
`demo_code_org.py` workflow:

- Torchvision ResNet-50 teacher and ResNet-18 student with the ImageNet stem.
- Adam, LR `0.001`, batch size `256`, `100` epochs.
- KD temperature `7`, KD weight `0.7`, CE weight `0.3`.
- The compressed source defaults to the trained student source.
- Compressed models train supervised by default.

### Oxford-IIIT Pet Fine-Grained Vision

`oxford_pets` adds a small natural-image benchmark:

- Teacher/student pair: `resnet34_to_resnet18`.
- Input resolution: `128x128`.
- Dataset split: `trainval` for training, `test` for evaluation.
- The compressed source defaults to the trained teacher.
- Compressed models train with knowledge distillation by default.
- Training defaults: SGD, LR `0.01`, momentum `0.9`, weight decay `1e-4`, batch
  size `64`, `80` epochs, milestones at `50` and `70`.
- KD defaults: temperature `2`, KD weight `1`, CE weight `1`.

### GLUE Language Inheritance

The registered GLUE tasks are `mrpc`, `qqp`, `sst2`, `mnli`, `rte`, `qnli`,
`cola`, and `stsb`. You can run them directly as `glue_<task>` or through the
launcher alias:

```bash
./run.sh --dataset glue --glue-task mrpc --pair bert4_to_bert2 \
  --method hetero --device cuda --plot-mode none
```

- Teacher/student pair: `bert4_to_bert2`.
- Teacher checkpoint: `google/bert_uncased_L-4_H-256_A-4`.
- Student checkpoint: `google/bert_uncased_L-2_H-128_A-2`.
- Tokenizer: `bert-base-uncased`.
- Maximum sequence length: `128`.
- The compressed source defaults to the trained teacher.
- Compressed models train with knowledge distillation by default.
- Training defaults: Adam, LR `2e-5`, weight decay `0.01`, batch size `32`,
  `3` epochs.
- Hetero compresses linear layers by default for this pair.
- STS-B is handled as regression with MSE task loss and regression KD loss.
  The other GLUE tasks are handled as classification with CE plus logit KD.

## Example Local Result

The repository currently includes one completed CIFAR-100
`resnet56_to_resnet20` suite under:

```text
logs/cifar100/resnet56_to_resnet20/all/20260502_221731/
```

These are single-seed local log values, not final paper numbers.

| Method | Params | Best top-1 in log | Final top-1 |
|---|---:|---:|---:|
| Teacher ResNet56 | 861,620 | 73.19 | 71.80 |
| Student ResNet20 | 278,324 | 69.28 | 68.77 |
| Student KD | 278,324 | 70.70 | 70.42 |
| InherNet small, rank 8 | 202,402 | 60.57 | 60.24 |
| InherNet large, rank 16 | 383,182 | 70.09 | 69.63 |
| HeteroInherNet | 335,747 | 72.31 | 72.10 |

In this run, HeteroInherNet used ranks from `8` to `15` with average rank
`12.77`, a `0.35` budget ratio, three heads, 16 calibration batches, and
zero-mean expert noise scale `0.01`.

## Outputs

Training logs are written under:

```text
logs/<dataset>/<pair>/<suite>/<timestamp>/
```

Plots are written under:

```text
results/<dataset>/<pair>/
```

`comparison/overview.png` is refreshed after each completed suite method when
comparison plotting is enabled.

## Project Structure

- `demo_code.py`: maintained CLI runner for single methods and suites.
- `run.sh`: shell launcher with background execution and suite log management.
- `experiment_registry.py`: dataset specs, model-pair lookup, defaults, suites,
  and run tags.
- `cifar10_models.py`: CIFAR-10 torchvision ResNet factories and pair specs.
- `cifar100_models.py`: CIFAR-100 ResNet, VGG, and WideResNet factories.
- `pet_models.py`: Oxford-IIIT Pet ResNet-34/ResNet-18 factories and pair spec.
- `glue_models.py`: GLUE small-BERT pair spec and model loader.
- `glue_data.py`: Hugging Face GLUE tokenization, collation, and dataloaders.
- `model_wrappers.py`: generic InherNet and HeteroInherNet SVD wrappers.
- `training_utils.py`: supervised and KD loops, metrics, structured logging,
  and finite-value checks.
- `plotting_utils.py`: log parsing and plotting.
- `tests/`: unit and smoke-oriented behavior checks.
- `ideas/`: paper-writing notes and methodology drafts for HeteroInherNet.
- `demo_code_org.py`: original reference script kept for historical comparison.

## Validation

Use the project conda environment for Python checks:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate inherdemo
python -m py_compile demo_code.py experiment_registry.py training_utils.py \
  plotting_utils.py cifar10_models.py cifar100_models.py pet_models.py \
  glue_models.py glue_data.py model_wrappers.py tests/test_registry_and_wrappers.py
python -m unittest discover -s tests -t .
./run.sh --suite all --dataset cifar10 --pair resnet50_to_resnet18_org -- \
  --smoke-test --plot-mode none --device cuda --svd-backend device
./run.sh --suite comparison --dataset oxford_pets --pair resnet34_to_resnet18 -- \
  --smoke-test --plot-mode none --device cuda --svd-backend device
./run.sh --dataset glue --glue-task sst2 --pair bert4_to_bert2 --method hetero \
  --smoke-test --plot-mode none --device cuda --svd-backend device
./run.sh --dataset glue --glue-task stsb --pair bert4_to_bert2 --method teacher \
  --smoke-test --plot-mode none --device cuda --svd-backend device
```

## Paper Notes

The strongest paper positioning is:

```text
HeteroInherNet turns neural network inheritance from a uniform compression rule
into a data-aware budget allocation problem.
```

The current `ideas/` drafts intentionally avoid claiming universal
information-bottleneck optimality. The defensible claims are data-weighted
low-rank approximation, entropy-budget rank allocation, convergence
compatibility under standard smoothness and boundedness assumptions, and a
linear-Gaussian rate-distortion interpretation.

## Citation

The HeteroInherNet citation will be added when the new paper is released. For
the inherited baseline, cite InherNet:

```bibtex
@misc{zhou2026studentasymmetricnetworkneural,
      title={Beyond Student: An Asymmetric Network for Neural Network Inheritance},
      author={Yiyun Zhou and Jingwei Shi and Mingjing Xu and Zhonghua Jiang and Jingyuan Chen},
      year={2026},
      eprint={2602.09509},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.09509},
}
```

## License

This repository keeps the original project license in `LICENSE`.
