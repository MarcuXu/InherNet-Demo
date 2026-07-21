# Experiment command reference

Run launchers from the repository root. They automatically select the
`inherdemo` interpreter at `$HOME/miniconda3/envs/inherdemo/bin/python` (or the
active `inherdemo` environment):

```bash
cd /root/nas/mingjing/InherNet-Demo
```

Conda activation is still recommended before running standalone `python`
commands that do not go through a launcher.

Dataset and pair names are defined in the root [README](../README.md#dataset-and-model-registry).
Use `scripts/run.sh --help` for the complete Python-runner option list. The
grouped launchers below intentionally reserve arguments that define their
experiment matrices, so they cannot be accidentally changed for only part of
a comparison.

## Common runtime arguments

Arguments after a launcher's positional arguments are forwarded to
`demo_code.py` when the launcher permits them. The most useful are:

| Argument | Meaning |
|---|---|
| `--download` | Permit torchvision to download a missing CIFAR or Oxford Pets dataset. Pinned GLUE data and models use the Hugging Face cache/Hub independently, so this flag does not control GLUE downloads. |
| `--data-root PATH` | Dataset/cache root; defaults to `data/`. Hugging Face assets are cached below `PATH/huggingface/`. |
| `--num-workers N` | Worker processes per PyTorch data loader. `0` loads in the training process; increase only as host CPU and memory permit. |
| `--device cuda`, `cuda:N`, or `cpu` | Execution device for direct `run.sh` commands. Grouped launchers default to `cuda` and use the optional `DEVICE` environment override; `CUDA_VISIBLE_DEVICES` can restrict which physical GPUs are visible. |
| `--seed N` | Seed for one direct `run.sh` command. Grouped launchers use the seed environment variables documented below. |
| `--teacher-checkpoint PATH` | Artifact saved by `--method teacher`, or strictly loaded by `student_kd`, `inhernet`, and `hetero`. |
| `--size small\|large` | Registered fixed-rank capacity. For `hetero`, `large` is publicly named Hetero and `small` is Hetero-Lite. If omitted in a direct command, Hetero defaults to `large` and InherNet to `small`; grouped launchers are always explicit. |
| `--rank N` | Manual InherNet baseline diagnostic only. Hetero rejects custom ranks so its name always denotes a registered, exactly matched capacity. |
| `--no-final-test` | Do not evaluate the held-out final test set after validation selection. Searches and ablations set this automatically. |
| `--plot-mode none\|single\|compare\|both` | Plot policy for a direct run. Grouped launchers use `none`; paper figures should be generated from completed structured logs. |
| `--hetero-allocation-scale weighted_uniform` | Hetero decomposition policy. `weighted_uniform` is the only formal/HPO setting; `unweighted_uniform` is the activation-weighting ablation, and `research_*` values are pre-study diagnostics only. |
| `--inheritance-diagnostics` | Log teacher-relative initialization fidelity before training. |
| `--inheritance-diagnostics-only` | Run that initialization audit and stop before optimizer construction. |

`PYTHON_BIN=/path/to/python` explicitly overrides the interpreter. Otherwise
the launchers use the active `inherdemo` environment or the project environment
under `$HOME/miniconda3/envs/inherdemo`; they fail clearly instead of silently
using a different Conda environment.

## `run.sh`: one run

```text
scripts/run.sh <demo_code.py arguments...>
```

This is the transparent, foreground entry point for one method. It changes to
the repository root and forwards every argument to `demo_code.py` without
constructing an experiment matrix.

`--method` selects exactly one role:

| Value | Action |
|---|---|
| `teacher` | Train the dense teacher and save its selected state to the teacher artifact. |
| `student` | Train the registered compact baseline without a teacher. |
| `student_kd` | Train the compact baseline with logits from a frozen teacher artifact. |
| `inhernet` | Construct the selected fixed-rank inherited model from a frozen teacher artifact, then train it. |
| `hetero` | Calibrate and construct activation-weighted Hetero or Hetero-Lite at the selected registered InherNet rank, then train it. |

```bash
scripts/run.sh --dataset cifar100 --pair resnet56_to_resnet20 \
  --method hetero --size small --seed 42 --device cuda --download \
  --teacher-checkpoint checkpoints/cifar100/resnet56_to_resnet20/teacher_seed_42.pt \
  --plot-mode none
```

This example runs Hetero-Lite. The internal `small|large` values remain stable
for compatibility with InherNet, checkpoints, and structured logs.

By default it creates a unique structured log at
`logs/run_<UTC timestamp>.log`. Set `INHERNET_RUN_LOG=/path/to/run.log` to
choose the log explicitly. `run.sh` itself does not implement background or
matrix execution.

## `formal.sh`: multi-seed comparison

```text
scripts/formal.sh DATASET PAIR [permitted runtime arguments...]
```

For every seed, this runs/reuses the teacher, then runs the student, student KD,
small/large supervised InherNet, and the headline Hetero method. If the selected
Hetero recipe uses distillation, the matrix also includes InherNet-Large with
the identical distillation objective and Hetero with supervised training. These
controls separate initialization from objective effects. Hetero-Lite is
reserved for the ablation launcher. CIFAR runs use the fixed
training holdout for selection and evaluate the official test split only after
selection.

Controls:

- `SEEDS=7,17,...` selects distinct comma-separated final-reporting seeds;
  default: `7,17,27,37`, disjoint from search and confirmation.
- `DEVICE=...` selects the device; default: `cuda`.
- `OVERWRITE_TEACHER=1` intentionally replaces matching teacher artifacts.
  Without it, existing checkpoints are reused.
- Formal matrices start as one detached `nohup` job by default.
- `FOREGROUND=1` runs the matrix in the current terminal instead.
- `DRY_RUN=1` prints the generated `run.sh` commands without executing them.

```bash
scripts/formal.sh cifar100 resnet56_to_resnet20 --download --num-workers 4
```

Teacher artifacts go to
`checkpoints/<dataset>/<pair>/teacher_seed_<seed>.pt`. Each constituent run has
its deterministic structured log below
`logs/formal/<dataset>/<pair>/seed_<seed>/`; complete cells resume safely. The
headline and derived objective controls consume exactly one reviewed row from
`configs/hetero_selected_recipes.csv`. A detached launch additionally writes
`logs/jobs/formal_<timestamp>/{job.pid,stdout.log}`.

## `prestudy.sh`: initialization diagnostics

```text
scripts/prestudy.sh [oxford_pets|cifar100|all] [permitted runtime arguments...]
```

This is an initialization-only pre-study, not HPO or formal evaluation. For
each selected target, the default maintained scope runs registered-rank
InherNet plus three Hetero cells: weight-only decomposition, the
activation-aware base, and the base with its zero-mean conditional lift. The
three failed rank-reallocation controls are available only through the
separate research scope. Every run stops before optimizer construction and
never evaluates the held-out final test. Full fidelity metrics use the complete
validation split. The local-operator probe replays dense-teacher inputs through
matching inherited modules for the first four deterministic validation
minibatches and aggregates a ratio of summed squared errors. The router probe
uses validation minibatch zero. Both probes record their split and batch
provenance; construction metadata records expert-mean preservation per layer.

Controls:

- `PRESTUDY_SEED=N` selects the teacher and diagnostic seed; default: `42`.
- `PRESTUDY_SCOPE=maintained|research|all` selects the matrix; default:
  `maintained`. Research rank-allocation controls never run implicitly.
- `TEACHER_CHECKPOINT=PATH` overrides the target's default teacher path. Use it
  only for a single target.
- `DEVICE` defaults to `cuda`; `DRY_RUN=1` prints the matrix.
- The launcher is detached by default; `FOREGROUND=1` keeps it attached.
- `RESUME=1` validates and skips complete diagnostic logs.

Logs produced before the local-operator and mean-preservation diagnostics are
rejected as stale. Move those logs elsewhere before rerunning; the launcher
does not delete or overwrite them.

```bash
scripts/prestudy.sh all --num-workers 4
PRESTUDY_SCOPE=research scripts/prestudy.sh all --num-workers 4
PRESTUDY_SEED=123 scripts/prestudy.sh oxford_pets --num-workers 4
```

The default teachers are read from
`checkpoints/search/<dataset>/<pair>/teacher_seed_<seed>.pt`. Structured logs
are written below `logs/prestudy/<dataset>/<pair>/seed_<seed>/`. The camera-ready
figures do not depend on those logs or a CSV: each plotting module contains the
exact raw values for one 300-DPI PNG.

```bash
python scripts/plot_prestudy_progression.py
python scripts/plot_prestudy_local_operator.py
python scripts/plot_prestudy_router_activity.py
python scripts/plot_prestudy_allocation.py
```

The progression and local-operator figures isolate the behavioral and
layerwise effects of activation weighting; the router figure tests the
conditional lift. The allocation figure is intended for motivation or an
appendix. These fixed-checkpoint, single-seed diagnostics use exact points and
layerwise dots, not seed error bars. Multi-seed formal and ablation figures
report the observed seed points and their aggregate separately.

The `research_*` policies are rejected failure/sensitivity probes; they never
enter formal, ablation, or HPO recipes.

## `ablation.sh`: mechanism ablations

```text
scripts/ablation.sh DATASET PAIR [permitted runtime arguments...]
```

This runs both registered InherNet capacities, Hetero-Lite as a capacity
reference, and Hetero component ablations: full Hetero, no activation
weighting, no expert perturbation, no load-balance loss, neither perturbation
nor balance, fixed uniform routers, and 4- and 8-batch calibration budgets
against the 16-batch reference. It uses validation only
(`--no-final-test`) and requires a trained seed-matched selection teacher.
Rank-allocation probes belong to `prestudy.sh`, not this component ablation.

Controls:

- `SEED=N` selects one seed; default: `42`.
- `TEACHER_CHECKPOINT=PATH` explicitly selects the teacher artifact; otherwise
  `checkpoints/search/selection/...` is used.
- `DEVICE` defaults to `cuda`; `DRY_RUN=1` prints the matrix.
- Ablation stays in the foreground unless `BACKGROUND=1` is supplied.
- `RESUME=1` skips a variant whose deterministic log already contains a
  complete `RUN_SUMMARY`.

```bash
SEED=42 TEACHER_CHECKPOINT=checkpoints/cifar100/resnet56_to_resnet20/teacher_seed_42.pt \
  scripts/ablation.sh cifar100 resnet56_to_resnet20 --num-workers 4
```

The launcher never trains a missing teacher and never evaluates the held-out
final test set. Variant logs are written to
`logs/ablation/<dataset>/<pair>/seed_<seed>/recipe_<recipe_id>/size_<size>/<variant>.log`;
after a completed foreground matrix, `summary.csv` is generated in that recipe
directory.

## `smoke.sh`: construction and forward checks

```text
scripts/smoke.sh DATASET PAIR [permitted runtime arguments...]
```

This invokes smoke mode for teacher, student, student KD, and both capacities
of InherNet and Hetero. Smoke mode uses synthetic inputs to check construction
and forward execution; it does not train or provide experimental evidence.

```bash
DEVICE=cuda scripts/smoke.sh glue_sst2 bert4_to_bert2 --svd-backend device
```

`DEVICE` defaults to `cuda`; `DRY_RUN=1` prints the matrix. This launcher is
foreground-only and does not require a teacher checkpoint or dataset download.

## `search.sh`: teachers and full-epoch search

```text
scripts/search.sh PHASE [DATASET PAIR] [permitted runtime arguments...]
scripts/search.sh                         # shorthand for an option-free `all`
```

`DATASET` and `PAIR` must be supplied together. With a target, only that target
is run. Without one, the phase uses its five prespecified development targets.
Every inherited search evaluates only Hetero and uses the normal training
epoch count. Write `all`
explicitly when passing runtime arguments; only the completely argument-free
command defaults to `all`.

Phases:

| Phase | Work performed |
|---|---|
| `teachers` | Train or reuse the pair-bound search teacher checkpoint(s). |
| `mechanism` | Evaluate nine Hetero auxiliary-loss, second-moment-shrinkage, expert-noise (through `0.02`), and one sparse joint candidate; all use `weighted_uniform`. |
| `optimization` | Evaluate registered LR scales while explicitly fixing the complete reference mechanism. |
| `distillation` | Evaluate registered/common KD mixtures, temperature, KD fraction, and supervised/no-KD candidates while explicitly fixing the complete reference mechanism; inapplicable duplicates/temperatures and supervised CIFAR-100 are skipped. |
| `confirmation` | Evaluate manually committed complete Hetero finalist recipes; it is never populated or launched automatically. |
| `all` | Prepare five seed-matched development teachers, then run all three independent screens. |

The mechanism screen explicitly fixes `lr_scale=1.0`; the mechanism and
learning-rate screens use the objective registered for each dataset profile
(supervised or distillation).  The distillation screen also explicitly fixes
`lr_scale=1.0`.  These references come from the committed confirmation
registry, not from selected recipes that may be updated after HPO, and every
generated Hetero command contains each controlled argument exactly once.

Controls:

- `SEARCH_SEEDS=42,123,...` selects distinct search seeds; default:
  `42,123,2026`.
- `SEARCH_CANDIDATES=id1,id2` limits one explicit non-`all` search phase to
  named rows in its committed candidate CSV. Unknown names fail.
- Completed candidate logs containing `RUN_SUMMARY` are skipped by default; an
  incomplete/existing log is not overwritten.
- `OVERWRITE_TEACHER=1` intentionally replaces search-teacher artifacts.
- Search defaults to `DEVICE=cuda` and detached execution. `FOREGROUND=1`
  disables detachment; `RESUME=0` makes any existing candidate log an error;
  `DRY_RUN=1` prints the matrix without detaching.

Examples:

```bash
# One target's teacher only
scripts/search.sh teachers glue_sst2 bert4_to_bert2 --num-workers 4

# Two mechanism candidates on all mechanism-development targets
SEARCH_CANDIDATES=reference,noise_0005 \
  scripts/search.sh mechanism --download --num-workers 4

# One target's LR screen
scripts/search.sh optimization oxford_pets resnet34_to_resnet18 \
  --download --num-workers 4

# Complete sequential search in one detached process
scripts/search.sh all --download --num-workers 4
```

The complete three-seed screen contains 135 mechanism, 45 optimization, and 69
distillation runs (249 full-epoch Hetero runs). It covers CIFAR-10,
CIFAR-100 ResNet-56, Oxford Pets, SST-2, and STS-B. The stages keep non-searched
settings at their registered references, keep `weighted_uniform` fixed, and do
not automatically select or propagate winners. Rank and inspect the complete
logs after the search. Runtime depends on modality, cache state, and missing
teachers; the earlier larger-search projection does not apply to this matrix.

After manual analysis, add complete recipes to
`configs/hetero_confirmation_candidates.csv` beside the committed
`weighted_uniform` reference.
Each candidate ID needs all five transfer-profile rows. Then
prepare held-out-seed teachers, and run joint confirmation:

```bash
SEARCH_SEEDS=3407 scripts/search.sh teachers --download --num-workers 4
SEARCH_SEEDS=3407 scripts/search.sh confirmation --download --num-workers 4
python scripts/rank_search.py logs/search/selection --stage confirmation --seed 3407
```

`all` deliberately excludes confirmation, so no screen winner is automatically
selected or propagated.

After confirmation, manually copy the chosen five rows into
`configs/hetero_selected_recipes.csv`. Formal Hetero and the Hetero/Hetero-Lite
ablation rows resolve this file; Hetero-Lite is never tuned separately.

Search teachers are saved below
`checkpoints/search/selection/<dataset>/<pair>/teacher_seed_<seed>.pt`. Structured logs
are placed below `logs/search/selection/<dataset>/<pair>/seed_<seed>/`, and each target
is summarized after its phase. A detached launch additionally creates
`logs/jobs/search_<timestamp>/{job.pid,stdout.log}`.

CIFAR search uses a fixed stratified training holdout. GLUE search likewise
uses a fixed training holdout, leaving the official validation split untouched
for formal reporting. KD-fraction candidates preserve each dataset's registered
total KD-plus-label loss weight. InherNet is never hyperparameter-searched;
its registered settings remain the formal baseline.
Because the auxiliary-loss coefficient is absolute and objective scales differ,
compare those candidates within each transfer profile and confirm one complete
recipe per profile instead of selecting a universal coefficient by global mean.

Each inherited run records every epoch and the validation-selected best epoch
in its structured log. Search does not save inherited `.pt` files; only teacher
checkpoints are persisted. Rerun a manually selected configuration explicitly
through `run.sh`, or first commit it to the reviewed selected-recipe file. An
inherited-checkpoint output option would still need to be added if the selected
model state itself must be persisted.

## `train_teachers.sh`: grouped search teachers

```text
scripts/train_teachers.sh [all|glue|vision] [permitted runtime arguments...]
```

This convenience command is equivalent to the global `search.sh teachers`
phase with `--download` enabled. The optional group defaults to `all`; `glue`
selects the eight text tasks and `vision` selects the ten vision pairs. It
processes the selected pair-bound teachers sequentially on one GPU and is
detached with `nohup` by default. This registry-maintenance launcher defaults
to seed 42; it is separate from the seed-matched selection teachers under
`checkpoints/search/selection/`.

Controls:

- `FOREGROUND=1` disables the default detached launch.
- `NUM_WORKERS=N` sets `--num-workers`; default: `4`.
- `DEVICE` defaults to `cuda`.
- `OVERWRITE_TEACHER=1` and `DRY_RUN=1` retain their search meanings.
- `HF_HUB_DISABLE_PROGRESS_BARS` and `HF_DATASETS_DISABLE_PROGRESS_BARS`
  default to `1` for readable job logs. `HF_TOKEN`, if exported or stored by
  Hugging Face login, is consumed by the Hugging Face libraries; the script
  never embeds or prints it.

```bash
scripts/train_teachers.sh

# Atomically replace only the eight GLUE teachers after a protocol change.
OVERWRITE_TEACHER=1 scripts/train_teachers.sh glue
```

The command prints the detached PID and paths under
`logs/jobs/teachers_<timestamp>/`. Existing compatible artifacts are reused.

## Background jobs and logs

`formal.sh`, `prestudy.sh`, `search.sh`, and `train_teachers.sh` enable detached
execution by default; use `FOREGROUND=1` to keep them attached. `ablation.sh` uses
`BACKGROUND=1` explicitly. One detached parent process runs the matrix
sequentially, which avoids contending jobs on a single GPU. `stdout.log`
is the launcher's console stream; epoch metrics and machine-readable metadata
live in the constituent structured run logs.

To inspect a job:

```bash
job_dir="$(ls -dt logs/jobs/* | head -n 1)"
cat "$job_dir/job.pid"
tail -f "$job_dir/stdout.log"
```

`common.sh` is an internal sourced helper for argument validation, detached
launching, and command execution. It is not a user-facing command.
