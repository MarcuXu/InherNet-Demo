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
| `--data-root PATH` | Dataset/cache root; defaults to `data/`. GLUE data, tokenizer files, and pretrained model weights are cached below `PATH/huggingface/`. |
| `--num-workers N` | Worker processes for the stochastic training loader. Deterministic evaluation and calibration stay in the training process to avoid repeated post-CUDA forks. |
| `--device cuda`, `cuda:N`, or `cpu` | Execution device for direct `run.sh` commands. Grouped launchers default to `cuda` and use the optional `DEVICE` environment override; `CUDA_VISIBLE_DEVICES` can restrict which physical GPUs are visible. |
| `--seed N` | Seed for one direct `run.sh` command. Grouped launchers use the seed environment variables documented below. |
| `--teacher-checkpoint PATH` | Artifact saved by `--method teacher`, or strictly loaded by every teacher-dependent distillation baseline, `inhernet`, and `inheract`. |
| `--size small\|large` | Registered fixed-rank capacity. For `inheract`, `large` is publicly named InherAct and `small` is InherAct-Lite. If omitted in a direct command, InherAct defaults to `large` and InherNet to `small`; grouped launchers are always explicit. |
| `--rank N` | Manual InherNet baseline diagnostic only. InherAct rejects custom ranks so its name always denotes a registered, exactly matched capacity. |
| `--no-final-test` | Do not evaluate the held-out final test set after validation selection. Searches and ablations set this automatically. |
| `--plot-mode none\|single\|compare\|both` | Plot policy for a direct run. Grouped launchers use `none`; paper figures should be generated from completed structured logs. |
| `--inheract-allocation-scale weighted_uniform` | InherAct decomposition policy. `weighted_uniform` is the only formal/HPO setting; `unweighted_uniform` is the activation-weighting ablation, and `research_*` values are pre-study diagnostics only. |
| `--inheritance-diagnostics` | Log teacher-relative initialization fidelity before training. |
| `--inheritance-diagnostics-only` | Run that initialization audit and stop before optimizer construction. |

`PYTHON_BIN=/path/to/python` explicitly overrides the interpreter. Otherwise
the launchers use the active `inherdemo` environment or the project environment
under `$HOME/miniconda3/envs/inherdemo`; they fail clearly instead of silently
using a different Conda environment.

## Teacher reuse policy

Teacher-maintenance and HPO launchers first validate their canonical destination
checkpoint. If it is missing, a launcher that owns teacher training may
atomically snapshot an already trained artifact only when dataset, pair,
architecture, seed, registered teacher settings, model/data profiles, selection
policy, and training/selection split protocol all match. Different seeds are
never substituted. In particular, same-seed registry teachers are reusable for
CIFAR and Oxford HPO, but registry GLUE teachers use official validation and
therefore cannot replace GLUE HPO teachers selected on a training holdout.

The `teachers` phase of `search.sh` applies this policy before training from
scratch; `train_teachers.sh` delegates to that search phase. In contrast, a
new formal run trains its own teacher checkpoints in its own run namespace,
then reuses those checkpoints only within that same formal run. `prestudy.sh`
and `ablation.sh` consume an existing checkpoint and never train one;
`smoke.sh` uses synthetic models and no checkpoint. Direct `run.sh` commands
remain explicit: `--method teacher` trains to the requested path, while
dependent methods load exactly the requested path.

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
| `student_kd_logit_standardized` | Train standalone KD with the released Logit Standardization plug-in; seven standard CIFAR-100 pairs only, and distinct from MLKD + Logit Standardization. |
| `student_dkd` | Train DKD with six released CIFAR-100 pair recipes or the explicitly labeled CIFAR-10 ResNet repository adaptation; it is not used for regression. |
| `student_ctkd` | Train curriculum-temperature KD on six released CIFAR-100 pairs or the labeled CIFAR-10 ResNet adaptation. |
| `student_catkd` | Train the source-configured CAT-KD objective adaptation on six CIFAR-100 pairs. |
| `student_simkd` | Train SimKD with its retained projector and frozen teacher classifier on seven CIFAR-100 pairs. |
| `student_reviewkd` | Train ReviewKD with source pre-activation teacher targets and train-only ABFs on five released CIFAR-100 pairs. |
| `student_crd` | Train CRD with train-only projection heads and memory banks on seven CIFAR-100 pairs. |
| `inhernet` | Construct the selected fixed-rank inherited model from a frozen teacher artifact, then train it. |
| `inheract` | Calibrate and construct activation-weighted InherAct or InherAct-Lite at the selected registered InherNet rank, then train it. |

```bash
scripts/run.sh --dataset cifar100 --pair resnet56_to_resnet20 \
  --method inheract --size small --seed 42 --device cuda --download \
  --teacher-checkpoint checkpoints/cifar100/resnet56_to_resnet20/teacher_seed_42.pt \
  --plot-mode none
```

This example runs InherAct-Lite. The internal `small|large` values remain stable
for compatibility with InherNet, checkpoints, and structured logs.

By default it creates a unique structured log at
`logs/run_<UTC timestamp>.log`. Set `INHERNET_RUN_LOG=/path/to/run.log` to
choose the log explicitly. `run.sh` itself does not implement background or
matrix execution.

## `formal.sh`: multi-seed comparison

```text
scripts/formal.sh DATASET PAIR [permitted runtime arguments...]
```

For every seed, this trains a run-scoped teacher, then runs the compact
student, student KD, paper-configured InherNet-Small + KD, supervised
InherNet-Large, and headline InherAct. If the selected InherAct recipe uses
distillation, the matrix also includes InherNet-Large with the same
distillation objective and learning-rate scale, and InherAct with supervised
training; these controls separate initialization from objective effects under
matched optimization. InherAct-Lite and one-head
Direct-SVD inheritance are reserved for the ablation launcher.

The CIFAR-100 matrix adds baselines only where released configurations cover
the architecture pair: standalone KD with Logit Standardization and SimKD on
seven standard pairs; CTKD, DKD, and CAT-KD on six; ReviewKD on five; and CRD
on seven. The standalone Logit-Standardized KD row is not the paper's
480-epoch MLKD + Logit Standardization row. CIFAR-10 includes explicitly
labeled CTKD and DKD repository adaptations derived from released ImageNet
ResNet recipes; neither is presented as a published CIFAR-10 result. No
unreviewed classification-only method is used for Oxford Pets, GLUE, or
STS-B. Training-only feature/contrastive auxiliaries and SimKD's retained
inference projector are reported separately in structured metadata.

CIFAR runs use the fixed training holdout for selection and evaluate the
official test split only after selection. GLUE runs likewise select on a
deterministic training holdout, then restore that epoch and report the public
validation split once.

Controls:

- `SEEDS=7,17,...` selects distinct comma-separated final-reporting seeds;
  default: `7,17,27,37`, disjoint from HPO selection.
- `DEVICE=...` selects the device; default: `cuda`.
- Formal matrices start as one detached `nohup` job by default.
- `FOREGROUND=1` runs the matrix in the current terminal instead.
- `DRY_RUN=1` prints the generated `run.sh` commands without executing them or
  creating result directories.
- By default a fresh `FORMAL_RUN_ID` is generated. To revisit an existing
  namespace, set both `RESUME=1` and `FORMAL_RUN_ID=<existing-id>`; this
  validates and skips complete cells. A partial log is not an epoch checkpoint
  and must be moved aside before that cell can be rerun.
- A detached launch prints its `FORMAL_RUN_ID` alongside the PID; retain it for
  paired ablations or an explicit resume.

```bash
scripts/formal.sh cifar100 resnet56_to_resnet20 --download --num-workers 4
```

Teacher artifacts go to
`checkpoints/formal/<run-id>/<dataset>/<pair>/teacher_seed_<seed>.pt`. Each
constituent run writes a structured log below
`logs/formal/<run-id>/<dataset>/<pair>/seed_<seed>/`. A vision run that selects
on a validation split is considered complete only after its held-out final-test
record has been written. The
headline and derived objective controls consume exactly one reviewed row from
`configs/inheract_selected_recipes.csv`. A detached launch additionally writes
`logs/jobs/formal_<timestamp>/{job.pid,stdout.log}`.

After each seed, `summary.csv` is regenerated in that seed directory. It keeps
the resolved training objective, learning-rate scale and rate, KD configuration,
teacher checkpoint/selected epoch, validation-selected epoch and metrics, plus
the held-out final-test split, selected epoch, primary metric, and all
final-test metrics when final evaluation is enabled. The individual structured
logs remain the complete epoch-level record.

## `formal_all.sh`: complete main suite

```text
scripts/formal_all.sh [all|vision|cifar100|glue] [permitted runtime arguments...]
```

This is the safe entry point for the complete paper matrix. One detached parent
runs each selected target's `formal.sh` matrix in the foreground, sequentially,
so multiple jobs never contend for one GPU. A normal invocation creates one
fresh formal-run namespace for the entire suite.

```bash
scripts/formal_all.sh all --download --num-workers 4
scripts/formal_all.sh glue --num-workers 4

# Validate/skip complete cells only in this exact run namespace.
RESUME=1 FORMAL_RUN_ID=formal_20260813_120000_000000000 \
  scripts/formal_all.sh all --download --num-workers 4
```

`all` covers 18 registered targets: one CIFAR-10 pair, eight CIFAR-100 pairs,
Oxford Pets, and eight compact-BERT GLUE tasks. `--download` controls
torchvision datasets; Hugging Face assets use their cache independently. The
fresh suite contains 158 one-seed cells and 632 runs over four seeds: 8 for
CIFAR-10, 92 for CIFAR-100, 8 for Oxford Pets, and 50 for GLUE per seed. The
CIFAR-100 total spans eight teacher/student architecture pairs; it is not 92
baselines on one pair. These
are the full planned counts, not a count remaining after a prior matrix. The
first command never imports, merges, or silently skips historical formal
results. `RESUME=1` is deliberately explicit because it applies only to the
named run and never merges historical artifacts.

## `prestudy.sh`: initialization diagnostics

```text
scripts/prestudy.sh [oxford_pets|cifar100|all] [permitted runtime arguments...]
```

This is an initialization-only pre-study, not HPO or formal evaluation. For
each selected target, the default maintained scope runs registered-rank
InherNet plus three InherAct cells: weight-only decomposition, the
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

This runs both registered InherNet capacities, a one-head Direct-SVD mechanism
control, InherAct-Lite as a capacity reference, and InherAct component
ablations: full InherAct, no activation
weighting, no expert perturbation, no load-balance loss, neither perturbation
nor balance, fixed uniform routers, and 4- and 8-batch calibration budgets
against the 16-batch reference. The InherNet references retain the formal
objectives: Small uses KD and Large uses supervised task loss. It uses validation only
(`--no-final-test`) and uses the matching formal teacher artifact for every
seed. Rank-allocation probes belong to `prestudy.sh`, not this component
ablation.

Controls:

- `ABLATION_SEEDS=7,17,...` selects distinct paired seeds; default: `7,17,27`.
  Every default cell uses the seed-matched teacher in
  `checkpoints/formal/<run-id>/<dataset>/<pair>/teacher_seed_<seed>.pt`,
  produced by the named formal matrix.
- `FORMAL_RUN_ID=<run-id>` is required so the ablation uses the paired formal
  teachers and writes its logs under the same experimental namespace.
- `TEACHER_CHECKPOINT=PATH` is only for a one-seed diagnostic
  (`ABLATION_SEEDS=N`); the default paired matrix never substitutes one teacher
  across seeds.
- `DEVICE` defaults to `cuda`; `DRY_RUN=1` prints the matrix.
- Ablation stays in the foreground unless `BACKGROUND=1` is supplied.
- `RESUME=1` skips a variant whose deterministic log already contains a
  complete `RUN_SUMMARY`.

```bash
scripts/formal.sh cifar100 resnet56_to_resnet20 --download --num-workers 4
FORMAL_RUN_ID=formal_20260813_120000_000000000 \
  scripts/ablation.sh cifar100 resnet56_to_resnet20 --num-workers 4
```

The launcher never trains a missing teacher and never evaluates the held-out
final test set. HPO uses `42,123,2026`, formal results use `7,17,27,37`, and
the three-seed ablation uses the paired subset `7,17,27`; their roles are
different, while all methods within each matrix share the same seed/teacher
pair. Variant logs are written to
`logs/ablation/<run-id>/<dataset>/<pair>/seed_<seed>/recipe_<recipe_id>/size_<size>/<variant>.log`;
after a completed foreground matrix, `summary.csv` is generated in that recipe
directory.

## `smoke.sh`: construction and forward checks

```text
scripts/smoke.sh DATASET PAIR [permitted runtime arguments...]
```

This invokes smoke mode for teacher, student, student KD, every registered
CTKD, DKD, Logit-Standardized KD, CAT-KD, SimKD, ReviewKD, and CRD cell for the
selected pair, both capacities of InherNet, the one-head Direct-SVD ablation
control, and both capacities of InherAct. Smoke mode uses synthetic inputs to
check construction and forward execution; it does not train or provide
experimental evidence.

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
Every inherited search evaluates only InherAct and uses the normal training
epoch count. Write `all`
explicitly when passing runtime arguments; only the completely argument-free
command defaults to `all`.

Phases:

| Phase | Work performed |
|---|---|
| `teachers` | Train or reuse the pair-bound search teacher checkpoint(s). |
| `mechanism` | Evaluate nine InherAct auxiliary-loss, second-moment-shrinkage, expert-noise (through `0.02`), and one sparse joint candidate; all use `weighted_uniform`. |
| `optimization` | Evaluate registered LR scales while explicitly fixing the complete reference mechanism. |
| `distillation` | Evaluate registered/common KD mixtures, temperature, KD fraction, and supervised/no-KD candidates while explicitly fixing the complete reference mechanism; inapplicable duplicates/temperatures and supervised CIFAR-100 are skipped. |
| `all` | Prepare five seed-matched development teachers, then run all three independent screens. |

The mechanism screen explicitly fixes `lr_scale=1.0`; the mechanism and
learning-rate screens use the objective registered for each dataset profile
(supervised or distillation).  The distillation screen also explicitly fixes
`lr_scale=1.0`.  These references come from the committed
`configs/inheract_reference_recipes.csv` registry, not from screen-selected
recipes, and every
generated InherAct command contains each controlled argument exactly once.

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
distillation runs (249 full-epoch InherAct runs). It covers CIFAR-10,
CIFAR-100 ResNet-56, Oxford Pets, SST-2, and STS-B. The stages keep non-searched
settings at their registered references, keep `weighted_uniform` fixed, and
support manual selection of compact profile-specific defaults. This lightweight
protocol is not a joint factorial search; formal multi-seed experiments validate
the screen-selected defaults. The launcher never propagates winners into the
selected-recipe file. Runtime depends on modality, cache state, and missing
teachers; the earlier larger-search projection does not apply to this matrix.
The completed matrix remains valid after this name-only migration and does not
need to be rerun; `scripts/search.sh all --download --num-workers 4` remains
the standard command for a fresh checkout or an incomplete screen.

The reviewed completed screen is manually committed as `screen_selected` in
`configs/inheract_selected_recipes.csv`: all profiles use auxiliary weight
`0.01`, shrinkage `0.01`, expert noise `0.005`, and `weighted_uniform`.
CIFAR-10 and CIFAR-100 use LR `0.5` with supervision; Oxford Pets uses LR
`1.0` with the temperature-2, 50/50 distillation objective; GLUE classification
uses LR `2.0` with supervision; and GLUE regression uses LR `2.0`, temperature
2, and KD fraction `0.25`. The five fixed controls remain in
`configs/inheract_reference_recipes.csv`, allowing every screen to retain its
prespecified non-searched fields. Formal InherAct and the InherAct/InherAct-Lite
ablation rows resolve the selected file; InherAct-Lite is never tuned separately.

Search teachers are saved below
`checkpoints/search/selection/<dataset>/<pair>/teacher_seed_<seed>.pt`. Structured logs
are placed below `logs/search/selection/<dataset>/<pair>/seed_<seed>/`, and each target
is summarized after its phase. A detached launch additionally creates
`logs/jobs/search_<timestamp>/{job.pid,stdout.log}`.

For `search.sh all`, the teacher phase first reuses a valid checkpoint already
in that selection path. If the path is missing, it checks same-seed registry
and formal artifacts and snapshots the first strictly compatible one into the
selection namespace. Only then, if none matches, does it train a new teacher.
With the default seeds, a seed-42 CIFAR/Oxford registry artifact can be reused;
seed 123/2026 artifacts normally require training, and GLUE selection teachers
require training unless another same-seed training-holdout artifact exists.
An absent selection checkpoint is never reconstructed after dependent search
logs exist, because that could mix teacher states under one experiment cell.

CIFAR search uses a fixed stratified training holdout. GLUE search likewise
uses a fixed training holdout, leaving the official validation split untouched
for formal reporting. KD-fraction candidates preserve each dataset's registered
total KD-plus-label loss weight. InherNet is never hyperparameter-searched;
its registered settings remain the formal baseline.
Because the auxiliary-loss coefficient is absolute and objective scales differ,
compare those candidates within each transfer profile and select a
profile-specific recipe instead of selecting a universal coefficient by global
mean.

Each inherited run records every epoch and the validation-selected best epoch
in its structured log. Search does not save inherited `.pt` files; only teacher
checkpoints are persisted. Rerun a manually selected configuration explicitly
through `run.sh`; the reviewed selected-recipe file records the default. An
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
