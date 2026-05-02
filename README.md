# InherNet Demo

This repository is a registry-driven research demo for the paper
`Beyond Student: An Asymmetric Network for Neural Network Inheritance`
([arXiv:2602.09509](https://arxiv.org/abs/2602.09509)).

The implemented methods are:

- `teacher`: train the teacher architecture from scratch.
- `student`: train the student architecture from scratch.
- `student_kd`: train the student with teacher distillation.
- `inhernet`: decompose a dense source model with the InherNet SVD wrapper.
- `hetero`: an experimental extension that allocates layer ranks by input-covariance spectral entropy.

`demo_code_org.py` is preserved as the legacy reference script. The maintained entry point is
`demo_code.py`, normally launched through `run.sh`.

## Environment

Install the Python dependencies in the environment you use for experiments:

```bash
pip install -r requirements.txt
```

In this workspace, use the existing conda environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate inherdemo
```

## Quick Start

Run the original-style CIFAR-10 suite. The pair defaults reproduce the important
`demo_code_org.py` workflow choices, so no long compatibility flag list is needed.

```bash
./run.sh --background --suite all --dataset cifar10 --pair resnet50_to_resnet18_org -- --download --device cuda
```

Run a paper-style CIFAR-100 suite:

```bash
./run.sh --background --suite all --dataset cifar100 --pair resnet56_to_resnet20 -- --download --device cuda
```

Run a fast smoke test without training:

```bash
./run.sh --suite all --dataset cifar10 --pair resnet50_to_resnet18_org -- --smoke-test --plot-mode none --device cuda
```

Run one method directly:

```bash
./run.sh --dataset cifar100 --pair vgg13_to_vgg8 --method hetero --download --device cuda
```

Suite-mode arguments after `--` are passed to `demo_code.py`. For example, use
`-- --epochs 10 --plot-mode none` for a short training check.

## Default Workflows

Pair and dataset registries define the default experimental protocol. CLI flags can still override
these defaults.

### CIFAR-10

`resnet50_to_resnet18_org` is the compatibility pair for `demo_code_org.py`:

- torchvision ResNet-50 teacher and ResNet-18 student with the original ImageNet stem.
- Adam optimizer, LR `0.001`, batch size `256`, `100` epochs.
- No LR scheduler and no weight decay.
- KD temperature `7`, KD weight `0.7`, CE weight `0.3`.
- Legacy train/eval behavior enabled for direct comparison with `demo_code_org.py`.
- InherNet/Hetero decompose the trained student source and train supervised by default.

`resnet50_to_resnet18` and `resnet50_to_resnet18_cifar_stem` use CIFAR-style ResNet stems and the
dataset-level CIFAR-10 training defaults.

### CIFAR-100

CIFAR-100 pairs follow the paper-style inheritance workflow:

- CIFAR-native teacher/student architectures such as `resnet56_to_resnet20`, `vgg13_to_vgg8`,
  and `wrn40_2_to_wrn16_2`.
- SGD, LR `0.05`, momentum `0.9`, weight decay `5e-4`, batch size `64`, `240` epochs.
- LR milestones at `150`, `180`, and `210`.
- KD temperature `2`, KD weight `9`, CE weight `0.1`.
- InherNet/Hetero decompose the trained teacher source and train with distillation by default.

For CIFAR-100 plots, accuracy is labeled as top-1 accuracy. For CIFAR-10, standard accuracy and
top-1 accuracy are the same metric, so plots use the shorter `Accuracy (%)` label.

## Available Suites

- `baseline`: `teacher`, `student`, `student_kd`.
- `comparison`: `teacher`, `student_kd`, small/large InherNet, `hetero`.
- `all`: every supported method for the selected dataset/pair.

Examples:

```bash
./run.sh --suite baseline --dataset cifar100 --pair resnet56_to_resnet20 -- --download
./run.sh --suite comparison --dataset cifar10 --pair resnet50_to_resnet18_org -- --download --device cuda
```

## Outputs

Training logs are written under:

```text
logs/<dataset>/<pair>/<suite>/<timestamp>/
```

Plots are written under:

```text
results/<dataset>/<pair>/
```

`comparison/overview.png` is refreshed after each completed suite method when comparison plotting is
enabled.

## Project Structure

- `demo_code.py`: maintained CLI runner for single methods and suites.
- `run.sh`: shell launcher with background execution and suite log directory management.
- `experiment_registry.py`: dataset specs, pair lookup, train defaults, suite specs, and tag helpers.
- `cifar10_models.py`: CIFAR-10 torchvision ResNet factories and pair specs.
- `cifar100_models.py`: CIFAR-100 ResNet/VGG/WideResNet factories and pair specs.
- `model_wrappers.py`: generic InherNet and Hetero SVD wrappers.
- `training_utils.py`: supervised/KD loops, metrics, structured logging, and finite-value checks.
- `plotting_utils.py`: log parsing and single-run/suite plotting.
- `tests/`: unit and smoke-oriented behavior checks.
- `demo_code_org.py`: original reference script kept for comparison.

## Extending the Code

To add a new vision dataset:

1. Add model factories and pair specs in a dataset-specific module.
2. Add a `DatasetSpec` in `experiment_registry.py`.
3. Define pair-level defaults only when they differ from the dataset defaults.
4. Add or update tests that verify model construction, pair defaults, and smoke-test execution.

For non-vision tasks such as GLUE, keep the same registry idea but add a task-specific data/model
adapter instead of forcing NLP inputs through the CIFAR image pipeline.

## Validation

Run these checks after changing code:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate inherdemo
python -m py_compile demo_code.py experiment_registry.py training_utils.py plotting_utils.py cifar10_models.py cifar100_models.py model_wrappers.py tests/test_registry_and_wrappers.py
python -m unittest discover -s tests -t .
./run.sh --suite all --dataset cifar10 --pair resnet50_to_resnet18_org -- --smoke-test --plot-mode none --device cuda --svd-backend device
```

## Citation

```text
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
