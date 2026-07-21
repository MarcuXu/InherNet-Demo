# Hetero

Research implementation of activation-aware conditional-expert neural-network inheritance.
The repository extends **Beyond Student: An Asymmetric Network for Neural
Network Inheritance** ([arXiv:2602.09509](https://arxiv.org/abs/2602.09509)).
`inhernet` is the fixed-rank reference method. `hetero` keeps that registered
rank and exact parameter contract, but makes the decomposition activation-aware
and initializes conditionable experts without changing their mean reconstruction.

The maintained entry point is `demo_code.py`, normally invoked through the
shell launchers. `demo_code_org.py` is a frozen historical reference and is not
the implementation used for current experiments.

## Method

Both inherited methods start from a **trained dense source**, normally the
teacher. The registered student architecture is a conventional compact-model
baseline; it is not the architecture into which the teacher is converted.
Consequently, an Oxford inherited model starts from the trained ResNet-34, and
a GLUE inherited model starts from the trained four-layer BERT. The ResNet-18
and two-layer BERT are student and student-distillation baselines.

InherNet replaces each eligible source layer with one down-projection and
multiple gated up-projections. It uses one fixed rank across eligible layers.
The implementation initializes balanced factors, `U sqrt(S)` and
`sqrt(S) V^T`, so identical softmax-weighted experts reconstruct the truncated
SVD at initialization.

This is a deliberate resolution of an internal paper/demo ambiguity. The prose
specifies one shared down-projection and a normalized softmax router, while the
released initialization divides each identical up expert by the head count;
those operations together would initialize to only one head-count fraction of
the truncated weight. The maintained code keeps the stated shared-down design
and exact normalized-mixture reconstruction. Therefore it follows the paper's
architectural intent but should not be described as byte-for-byte demo
reproduction. For the same reason, paper tables should use parameter counts
measured from each constructed model; some published pair/rank counts are not
consistent with the shared-down formula.

Hetero estimates the uncentered input second moment

```text
M_x = E[x x^T] = C C^T
```

and decomposes the data-weighted operator `W C`. This follows from

```text
E ||(W - W_hat) x||^2
  = tr((W - W_hat) M_x (W - W_hat)^T)
  = ||(W - W_hat) C||_F^2.
```

Thus, at the registered InherNet rank, truncated SVD of `W C` minimizes a local
empirical output-reconstruction objective. Hetero then writes each up expert as
the inherited mean factor plus a deviation whose sum across experts is zero.
Under the initially uniform router, those deviations preserve the inherited
rank-truncated reconstruction while exposing conditional first-order tangent
directions. This is one constrained **preserve-then-adapt initialization**:
select the best local mean operator under the empirical activation metric, then
make its conditional degrees of freedom learnable without changing that mean.

Formal Hetero uses this `weighted_uniform` policy for every eligible layer: it
does not search for or allocate layer-specific ranks. The resulting
construction has the same rank and parameter count as the corresponding
InherNet model, isolating the initialization principle from a capacity change.

Heterogeneous-rank policies were useful during method development but were less
stable than the registered-rank construction. They remain available only under
explicit `research_*` names for the validation-only pre-study. They are not
formal Hetero variants, HPO candidates, or alternative solutions presented by
the method.

For convolutions with patch dimension at most 256, calibration uses exact
`unfold` patches. Wider convolutions use per-location channel moments as a
scalable block-diagonal approximation. Expert perturbations are zero-mean
across heads. Because the router starts uniform, their mean is exactly the
activation-weighted rank-truncated reconstruction at initialization; training
can subsequently learn input-conditional expert behavior. The load-balance
term enters the training objective with reference weight `0.01`; HPO compares
weights `0`, `0.01`, and `0.03`, and the component ablation removes it.
Inherited layers remain trainable after initialization.

Both inherited methods use the same `--size small|large` interface. The size
selects a registered uniform rank: paper-printed for the seven covered
CIFAR-100 pairs and repository-defined for extensions. Hetero applies exactly
that rank to the same eligible layers as InherNet. Controlled ablations isolate
activation weighting, expert perturbation, learned routing, the auxiliary
objective, and calibration budget without introducing a rank-allocation
confound.

## Oxford Pets Is Vision Transfer, Not GLUE

Oxford-IIIT Pet and GLUE use unrelated model families:

- Oxford uses torchvision ResNet-34 as teacher and ResNet-18 as the compact
  baseline. Their dense runs start from the official
  `IMAGENET1K_V1` weights and replace the classifier with a 37-class head.
  ImageNet is used only as weight initialization; the ImageNet dataset is not
  trained or evaluated by this repository.
- GLUE uses pretrained compact BERT checkpoints and tokenized text. A BERT
  model cannot process Oxford images, and no GLUE model is used in the Oxford
  pipeline.

Transfer initialization is appropriate for Oxford because it is a relatively
small natural-image dataset. It supplies general visual features before
task-specific fine-tuning and makes the ResNet comparison substantially more
stable than training these ImageNet-scale stems from random initialization.

Oxford inheritance compresses convolutional layers by default and leaves the
final linear classifier dense. The ResNet-34 classifier has only
`512 * 37 + 37 = 18,981` parameters, a very small fraction of the network.
Factorizing it would save little, while directly restricting the class decision
map can unnecessarily reduce task adaptation and generalization. In contrast,
transformers are dominated by eligible linear projections, so GLUE inheritance
compresses linear layers by default.

## Dataset and Model Registry

### Vision

| Dataset | Pair key | Teacher | Student baseline | InherNet ranks | Source / target objective |
|---|---|---|---|---|---|
| CIFAR-10 | `resnet50_to_resnet18` | torchvision ResNet-50 with CIFAR stem | torchvision ResNet-18 with CIFAR stem | 32 / 64 | teacher / distillation |
| CIFAR-100 | `resnet32_to_resnet8` | CIFAR ResNet-32 | CIFAR ResNet-8 | 4 / 8 | teacher / supervised |
| CIFAR-100 | `resnet32x4_to_resnet8x4` | CIFAR ResNet-32x4 | CIFAR ResNet-8x4 | 4 / 8 | teacher / supervised |
| CIFAR-100 | `vgg13_to_vgg8` | CIFAR VGG-13 | CIFAR VGG-8 | 128 / 256 | teacher / supervised |
| CIFAR-100 | `wrn40_2_to_wrn40_1` | WideResNet-40-2 | WideResNet-40-1 | 16 / 32 | teacher / supervised |
| CIFAR-100 | `wrn40_2_to_wrn16_2` | WideResNet-40-2 | WideResNet-16-2 | 16 / 32 | teacher / supervised |
| CIFAR-100 | `resnet56_to_resnet20` | CIFAR ResNet-56 | CIFAR ResNet-20 | 8 / 16 | teacher / supervised |
| CIFAR-100 | `resnet110_to_resnet32` | CIFAR ResNet-110 | CIFAR ResNet-32 | 8 / 32 | teacher / supervised |
| CIFAR-100 | `resnet110_to_resnet20` | CIFAR ResNet-110 | CIFAR ResNet-20 | 4 / 8 | teacher / supervised |
| Oxford-IIIT Pet | `resnet34_to_resnet18` | ImageNet-initialized ResNet-34 | ImageNet-initialized ResNet-18 | 32 / 64 | teacher / distillation |

`demo_code_org.py` remains an unregistered historical reference. The maintained
CIFAR-10 experiment uses only `resnet50_to_resnet18` with the CIFAR stem.

The seven non-extension CIFAR-100 rows reproduce the ranks printed in the
paper, not ranks fitted by this repository. The paper's parameter-count table
is internally inconsistent for ResNet-32x4 (printed ranks 4/8) and
ResNet-110-to-ResNet-32 Large (printed rank 32): those counts do not follow
from its stated shared-down, three-head construction. The code therefore keeps
the printed ranks and records measured model sizes instead of silently
substituting ranks inferred from the conflicting counts. CIFAR-10, Oxford, and
compact-BERT GLUE are explicitly repository-defined extensions.

### Text

All text datasets use pair `bert4_to_bert2`: teacher
`google/bert_uncased_L-4_H-256_A-4`, student
`google/bert_uncased_L-2_H-128_A-2`, tokenizer `bert-base-uncased`, and maximum
sequence length 128. The inherited source is the fine-tuned teacher, the target
objective is distillation, and the InherNet ranks are 32 / 64.

| Dataset key | GLUE task | Problem | Primary validation metric |
|---|---|---|---|
| `glue_mrpc` | MRPC | binary classification | accuracy; F1 is also logged |
| `glue_qqp` | QQP | binary classification | accuracy; F1 is also logged |
| `glue_sst2` | SST-2 | binary classification | accuracy |
| `glue_mnli` | MNLI matched | three-class classification | accuracy |
| `glue_rte` | RTE | binary classification | accuracy |
| `glue_qnli` | QNLI | binary classification | accuracy |
| `glue_cola` | CoLA | binary acceptability classification | Matthews correlation |
| `glue_stsb` | STS-B | regression | Pearson correlation; Spearman is also logged |

These compact-BERT GLUE experiments are repository extensions. They are not a
replication of the paper's T5 protocol.

### Default Training Settings

The CIFAR-100 InherNet baseline is frozen to the paper-listed hyperparameters: SGD with
learning rate 0.05, momentum 0.9, weight decay `5e-4`, batch size 64, 240
epochs, learning-rate drops at 150/180/210, three heads, and the printed
pair-specific ranks. Formal InherNet uses supervised task loss as the clean
InherNet-only baseline and is not part of Hetero HPO; the paper does not state
one unambiguous objective for every Small/Large table cell, so this objective
choice is disclosed rather than called an exact reproduction. The public repository describes its script as a basic runtime
demo and supplies only a didactic CIFAR-10 Adam/rank-32 example, so it is not
used to override the paper's CIFAR-100 protocol. Settings for CIFAR-10,
Oxford, and compact-BERT GLUE are labeled adaptations rather than attributed
to the paper.

| Profile | Optimizer | LR | Batch | Epochs | Schedule | Weight decay |
|---|---:|---:|---:|---:|---|---:|
| CIFAR-10 main | SGD, momentum 0.9 | 0.05 | 128 | 200 | milestones 100, 150, 180 | `5e-4` |
| CIFAR-10 compatibility | Adam | 0.001 | 256 | 100 | none | 0 |
| CIFAR-100 | SGD, momentum 0.9 | 0.05 | 64 | 240 | milestones 150, 180, 210 | `5e-4` |
| Oxford Pets | SGD, momentum 0.9 | 0.001 | 32 | 30 | milestones 15, 25 | `1e-4` |
| GLUE compact BERT | AdamW | `5e-5` | 32 | 4 | 10% linear warmup, then linear decay | 0.01* |

`*` GLUE excludes bias and LayerNorm parameters from weight decay and clips the
global gradient norm at 1.0. This is the architecture-matched compact-BERT
protocol; it is not the paper's T5 recipe.

Oxford uses random resized 224-pixel crops and horizontal flips for training,
then resize-to-256 and center-crop-to-224 for evaluation. Its `trainval` split
is divided by a fixed class-stratified 80/20 split with split seed 2026.
Validation balanced accuracy selects the state; the official test split is
evaluated only after selection. Calibration uses training indices with
deterministic evaluation transforms supplied when the torchvision dataset is
constructed. Oxford teacher artifacts created under the older stochastic
validation/calibration profile are deliberately rejected by the updated data
profile and must be regenerated before search or formal reporting.

## Registered-Rank Capacity Semantics

`small` and `large` are the two registered capacity settings for every model
pair. InherNet applies the selected fixed rank uniformly to every
eligible layer. Hetero applies the same rank to the same layers, including the
same decision to leave a layer dense when the registered rank is not a
truncation. Consequently, the two constructions have the same measured
parameter count. This definition is identical for vision and text and does not
require a task-dependent ratio denominator.

For a fair table, report the size, registered rank, measured parameter count,
and each method's accuracy and efficiency. Dense-model and eligible-layer
ratios remain useful diagnostics in structured logs, but they are outputs
rather than user-selected hyperparameters.

The public method names are **Hetero** for internal size `large` and
**Hetero-Lite** for internal size `small`. Hetero is the headline method for
formal baseline comparisons. Hetero-Lite is retained as a capacity/efficiency
ablation and, after selection is complete, must use the Hetero-selected
hyperparameters without a second search. The stable CLI and structured-artifact
values remain `--method hetero --size large|small` for compatibility. A direct
Hetero command with no `--size` now resolves to headline `large`; direct
InherNet retains the historical `small` default. Hetero rejects custom
`--rank` values so its public name cannot be attached to an unregistered
capacity.

The main Hetero options are:

```bash
--size large                        # headline Hetero; small is Hetero-Lite
--max-calib-batches 16
--hetero-max-features-per-batch 4096
--hetero-second-moment-shrinkage 0.01
--hetero-allocation-scale weighted_uniform
--hetero-expert-noise-scale 0.01
--aux-loss-weight 0.01
```

`weighted_uniform` is the only formal and HPO policy. `unweighted_uniform`
removes activation weighting for a component ablation. Policies prefixed with
`research_` are restricted to the pre-study diagnostics and must not be used as
formal methods or search candidates.

`--inheritance-diagnostics` logs teacher/inherited task metrics, per-example
summed output squared error,
cosine similarity, prediction agreement, and KL before the first optimizer
step. For Hetero it also logs a four-batch held-out local-operator probe using
dense-teacher inputs and a ratio of summed squared errors. Construction
metadata records the per-layer conditional-expert mean shift and diversity;
router diagnostics record their evaluation split and batch index.
`--inheritance-diagnostics-only` stops after that audit. These flags do not
alter training state and are intended for mechanism analysis, not HPO.

## Teacher-Checkpoint Pipeline

Training is deliberately split into two process stages:

1. Train the dense teacher on the task and atomically save a `.pt` artifact.
   The artifact records its schema, dataset, pair, architecture, seed,
   training settings, split metadata, selection policy, metrics, and
   state dictionary.
2. Start each dependent method in a fresh Python process and strictly load the
   matching artifact. The teacher is frozen and placed in evaluation mode. KD
   keeps it in memory only to produce logits under `no_grad`; it is never in the
   optimizer. Supervised inheritance releases the dense source after
   decomposition.

Teacher and inherited parameters are therefore not optimized simultaneously.
Missing or incompatible artifacts fail rather than silently retraining a
teacher. The default path is

```text
checkpoints/<dataset>/<pair>/teacher_seed_<seed>.pt
```

Checkpoint compatibility is validated against the teacher-training settings
stored inside the artifact. Target-only optimizer overrides (for example a
Hetero learning-rate search candidate) do not alter teacher provenance or
make the frozen checkpoint unloadable. Architecture, dataset, pair, seed,
model/data profiles, recorded split, and the checkpoint's own training
integrity metadata remain strict.

Registry-maintenance teachers use `checkpoints/search/...`; HPO selection
teachers use `checkpoints/search/selection/...` because their training-holdout
protocol and seeds differ. Checkpoints
and logs are runtime artifacts and are ignored by Git: the current teachers
occupy hundreds of MiB and should live in durable experiment/object storage,
not ordinary source history. `teacher_checkpoints.json` is the small tracked
manifest of their paths, selected metrics, and semantic provenance. Internal
integrity fields are used only by checkpoint save/load validation; they are not
an experimental result or a reporting key. Run

```bash
python scripts/audit_teachers.py --json > /tmp/teacher_checkpoints.json
```

to strictly reconstruct and validate every registered teacher. Copy the output
into the tracked manifest only after intentionally changing the teacher set.
An Oxford checkpoint created under the former stochastic calibration-transform
profile is deliberately rejected. The registry teacher has been regenerated
under the deterministic profile; prepare the separate seed-matched selection
teachers before Oxford HPO with:

```bash
scripts/search.sh teachers oxford_pets resnet34_to_resnet18 --download --num-workers 4
```

The compact-BERT and GLUE dataset revisions are pinned. Checkpoints record
semantic split provenance. The loader explicitly transfers the BERT
pretraining encoder into a newly initialized task-classification head and
tokenizes only the required splits. Hetero calibration uses a fixed seed-2027
subset of at most 512 training examples, stratified for classification, rather
than whichever examples happen to occupy the first batches. Public downloads work without
authentication; `hf auth login` removes the Hub rate-limit advisory on a fresh
machine without placing a token in this repository.

### GLUE teacher horizon audit

The current compact-BERT teachers each completed four epochs. Their artifacts
contain the validation-selected state rather than necessarily the last state:
MRPC, QQP, MNLI, CoLA, and STS-B selected epoch 4; SST-2 and RTE selected epoch
3; QNLI selected epoch 2. All eight pass the strict checkpoint audit.

The original repository artifacts used three epochs at constant `2e-5`, and
six tasks reached their best observed metric at the horizon. The canonical row
counts and strict checkpoint audit ruled out download corruption, but the
optimization horizon was weakly justified. The maintained compact-BERT
protocol now follows Google's architecture-matched setup more closely: four
epochs, initial LR `5e-5`, 10% step-wise linear warmup followed by linear
decay, gradient clipping at 1.0, and no AdamW decay on bias or LayerNorm
parameters. The eight teachers have been regenerated under this protocol;
dependent GLUE runs must use these regenerated checkpoints rather than
historical artifacts.

The paper's GLUE numbers are not targets for this compact track. The paper uses
T5-Base/T5-Small (222M/60M), AdamW LR `3e-4` for teacher training, weight decay
0.1, ten epochs, batch 32, KD LR `5e-4`, coefficient 0.3, temperature 4, and
InherNet rank 128 with two heads. The released demo contains no T5/GLUE code,
and the paper omits the text-to-text formulation, revisions, scheduler, seeds,
and selection rule. A literal T5 reproduction therefore belongs in a separate
track and must not be approximated silently with `T5ForSequenceClassification`.

Train one teacher and then one inherited model as follows:

```bash
scripts/run.sh --dataset cifar100 --pair resnet56_to_resnet20 --method teacher \
  --teacher-checkpoint checkpoints/cifar100/resnet56_to_resnet20/teacher_seed_42.pt \
  --seed 42 --download --device cuda --plot-mode none --search-validation

scripts/run.sh --dataset cifar100 --pair resnet56_to_resnet20 --method hetero \
  --size large \
  --teacher-checkpoint checkpoints/cifar100/resnet56_to_resnet20/teacher_seed_42.pt \
  --seed 42 --download --device cuda --plot-mode none --search-validation
```

## Experiment Launchers

See [scripts/README.md](scripts/README.md) for the concise command reference,
including positional phases, forwarded runtime flags, environment controls,
artifact paths, background behavior, and examples. The summary below explains
the intended experiment matrices.

- `scripts/run.sh` runs one foreground `demo_code.py` command. If no log path is set,
  it creates a unique `logs/run_<UTC timestamp>.log`.
- `scripts/formal.sh DATASET PAIR` runs teacher, student, student KD, both
  registered supervised InherNet capacities, and the headline Hetero method.
  When the selected Hetero recipe uses distillation, it also runs an
  objective-matched InherNet-Large and a supervised Hetero control. It defaults to seeds
  `7,17,27,37`, which are disjoint from search and confirmation, CUDA, and a
  detached `nohup` job. CIFAR formal runs use the fixed training holdout
  for model selection and touch the official test set only after selection.
  Existing matching teacher artifacts are reused.
- `scripts/prestudy.sh [oxford_pets|cifar100|all]` performs initialization-only
  diagnostics for the fixed-rank methods and the explicitly named research
  allocators. It never trains or evaluates the held-out final test.
- `scripts/ablation.sh DATASET PAIR` compares the two InherNet capacities,
  Hetero-Lite, full Hetero, and Hetero without activation weighting, expert
  perturbation, balance loss, both perturbation and balance, or learned routing,
  plus 4- and 8-batch calibration-budget controls against the 16-batch reference.
  It disables held-out final-test evaluation and writes variant-tagged logs
  below `logs/ablation/` for paired analysis.
- `scripts/smoke.sh DATASET PAIR` performs construction and forward checks. It does
  not train on the dataset and is not empirical evidence.
- `scripts/search.sh PHASE [DATASET PAIR]` is the single hyperparameter-search
  entry point. Supplying a dataset and pair runs one target; omitting both runs
  the prespecified development representatives. `PHASE` is `teachers`,
  `mechanism`, `optimization`, `distillation`, `confirmation`, or `all`. A global `all` run
  prepares the required development teachers before the search stages.
  Search defaults to distinct seeds `42,123,2026`, CUDA, detached execution, and skipping completed
  candidate logs.
- `scripts/train_teachers.sh [all|glue|vision]` trains or reuses the selected
  registered, pair-bound search-teacher checkpoints. Pair-bound artifacts are necessary because the
  current checkpoint compatibility contract includes the pair identity. The
  launcher defaults to CUDA, downloads missing datasets, uses four data
  workers, and starts with `nohup` in the background. Set `FOREGROUND=1` only
  when foreground execution is intentionally required.

All grouped launchers are sequential, which is safe for one GPU. Set
`DRY_RUN=1` to print their matrices. Formal experiments, the pre-study,
hyperparameter search, and teacher training start detached by default; set
`FOREGROUND=1` for intentional foreground execution. Ablations retain the
explicit `BACKGROUND=1` control.
The detached helper starts one
`nohup` job, records its PID under `logs/jobs/`, and redirects console output to
the adjacent `stdout.log`. The individual structured run logs remain under
`logs/` or `logs/search/`.

`scripts/formal.sh` intentionally rejects optimizer, epoch, rank, size, and method
hyperparameter overrides: a common override would otherwise change teacher and
target semantics inconsistently when a teacher artifact is reused. Its Hetero
run resolves the dataset's reviewed row from
`configs/hetero_selected_recipes.csv`; Hetero-Lite ablations resolve that same
row. After manual HPO analysis, replace the reviewed profile rows with the
confirmed recipes. No launcher writes or selects this file automatically.
Formal evaluation runs exactly one headline Hetero recipe per seed; any
supervised Hetero control is derived from that same recipe and differs only in
the training objective.

Examples:

```bash
scripts/formal.sh cifar100 resnet56_to_resnet20 --download --num-workers 4
scripts/prestudy.sh all --num-workers 4
scripts/ablation.sh cifar100 resnet56_to_resnet20
scripts/smoke.sh oxford_pets resnet34_to_resnet18 --svd-backend device
FOREGROUND=1 scripts/formal.sh oxford_pets resnet34_to_resnet18 --download
```

`OVERWRITE_TEACHER=1` permits intentional replacement of a formal/search
teacher artifact. Without it, formal teacher training refuses to overwrite an
existing artifact; search reuses its dedicated existing teacher. Search skips
candidate logs that already contain a complete `RUN_SUMMARY` by default. An
incomplete log is never overwritten and must be inspected before retrying.

## Initialization Pre-Study

The pre-study is a mechanism diagnostic, not hyperparameter search. It uses an
existing seed-matched teacher and stops before optimizer construction. Its
default maintained scope compares registered-rank InherNet with three Hetero
cells on Oxford Pets and CIFAR-100: weight-only decomposition, the
activation-aware registered-rank base, and the base plus its fixed zero-mean
conditional lift. The failed `research_*` rank-allocation controls run only
when explicitly requested with `PRESTUDY_SCOPE=research` or `all`.
Fidelity and task metrics use the complete validation split. A local-operator
probe uses the first four deterministic validation minibatches and a ratio of
summed squared errors; the router-gradient probe uses minibatch zero. The logs
record split/batch provenance and per-layer expert-mean preservation.

```bash
scripts/prestudy.sh all --num-workers 4
```

The launcher defaults to seed 42, CUDA, detached execution, resume mode, and
the maintained scope. Use `PRESTUDY_SCOPE=research|all`, `PRESTUDY_SEED=N`,
`DEVICE=...`, or `FOREGROUND=1` when needed. It reads
teachers from `checkpoints/search/<dataset>/<pair>/teacher_seed_<seed>.pt`,
writes structured logs below `logs/prestudy/`. It never touches the held-out
final test. Pre-causal-diagnostic logs are rejected as stale and must be moved
before rerunning; no log is overwritten automatically. Four focused figures are reproduced independently from raw values
embedded in their corresponding plotting modules:

```bash
python scripts/plot_prestudy_progression.py
python scripts/plot_prestudy_local_operator.py
python scripts/plot_prestudy_router_activity.py
python scripts/plot_prestudy_allocation.py
```

Each command writes one 300-DPI PNG under `figures/`; no CSV or retained log is
needed. The progression and local-operator figures establish the behavioral
and layerwise effects of activation weighting, while the router figure tests
the conditional-lift mechanism. The allocation figure records the fixed-rank
finding for the motivation or appendix. Because these are single-seed,
zero-step diagnostics, the plots show exact observations and layerwise dots
rather than statistical error bars.
The `research_*` allocators are rejected controls, not HPO candidates or
competing Hetero solutions.

## Full-Epoch Hyperparameter Search

Search runs use the same epoch count as normal training; `scripts/search.sh`
rejects epoch overrides. This is slower than low-fidelity screening but avoids
changing the learning-rate schedule or selecting candidates from a different
training horizon.

Search is exclusively for Hetero (`hetero --size large`). InherNet retains its
registered/paper settings as a formal baseline; Hetero-Lite receives the final
Hetero-selected recipe without separate tuning and is reported as a capacity
ablation. Search sets `--no-final-test` and selects only on validation:

- CIFAR automatically uses a fixed, class-stratified 90/10 holdout from the
  training set. Its official test set remains untouched during search.
- Oxford uses its fixed stratified validation split and does not evaluate the
  official test split.
- GLUE uses a fixed 90/10 split of its training data (stratified for
  classification); the official validation split remains untouched for formal
  reporting because public GLUE test labels are unavailable.

The search is deliberately shared rather than pair-specific:

- `mechanism` evaluates nine focused regularization configurations over routing
  auxiliary weight, second-moment shrinkage, expert noise through an upper-sided
  `0.02` value, and one sparse joint check on CIFAR-10, the canonical CIFAR-100
  ResNet-56 pair, Oxford Pets, SST-2, and
  STS-B. Every row uses `weighted_uniform`, explicitly fixes `lr_scale=1.0`,
  and uses the objective registered for its dataset profile; rank policy is not
  searched.
- `optimization` evaluates LR scales `0.5, 1, 2` for Hetero on the same
  five targets, with the complete non-searched mechanism passed explicitly from
  the registered reference and the profile-registered objective passed
  explicitly.
- `distillation` evaluates a common 0.5 KD mixture, temperatures 1 and 4,
  fractions 0.25 and 0.75, and supervised/no-KD training. CIFAR-10 additionally
  retains its registered 9.0/0.1 objective as a baseline. Duplicate registered
  0.5 mixtures and inapplicable STS-B temperatures are skipped; supervised-default
  CIFAR-100 is excluded. Fractions preserve each dataset's registered
  KD-plus-label coefficient sum, reducing a trivial coefficient-scaling confound.
  The complete registered mechanism and reference `lr_scale=1.0` are likewise
  passed explicitly to every distillation candidate; only the objective fields
  change. Static reference fields are read from the committed confirmation
  registry rather than post-HPO selected recipes, and each controlled argument
  occurs exactly once in every generated Hetero command.

Across three distinct seeds this produces 135 mechanism, 45 optimization, and
69 distillation runs: **249 full-epoch Hetero runs**. The five targets
cover scratch vision, supervised inheritance, transfer vision, text
classification, and regression; the remaining architectures/tasks are held
out for transfer tests.
The stages are independent sensitivity screens; the launcher does not select,
propagate, or combine winners. Analyze the complete logs after all runs finish,
then test manually assembled complete recipes jointly before freezing the
formal configuration. InherNet is not assigned Hetero-selected settings.
Auxiliary-loss weights are absolute coefficients, while the scale of the base
objective differs across transfer profiles. Interpret that screen within each
profile and confirm complete recipes per profile; a globally averaged auxiliary
weight is not evidence of a universally optimal trade-off.
The `all` phase deliberately excludes `confirmation` because its finalist rows
must be written manually after inspecting the three screens.

Search outputs are stored under

```text
logs/search/selection/<dataset>/<pair>/seed_<seed>/
```

Search teachers are kept separately under
`checkpoints/search/selection/<dataset>/<pair>/teacher_seed_<seed>.pt`, so tuning cannot
silently replace a formal teacher. `scripts/summarize_search.py` writes
`summary.csv` with candidate identity, size, reference InherNet rank and
parameter count, primary metric, all registered metrics at the primary-selected
epoch, epoch count, achieved diagnostic ratios, and actual parameter count.
`scripts/rank_search.py` aggregates candidates by
within-target validation rank so accuracy, MCC, and correlation are never
averaged on incompatible raw scales.

Every search run logs all epochs and the primary-selected best epoch, including
all registered metrics at that epoch. Only teacher models are saved as `.pt`
artifacts. Inherited search states are restored to the selected epoch in memory
for consistent evaluation but are not persisted; saving 249 temporary models
would add substantial storage without helping post-hoc hyperparameter analysis.

Before target search, train the matching selection teacher. It is not technically
necessary to finish teachers for unrelated datasets before starting the first
target search. `scripts/train_teachers.sh` maintains all 18 registered seed-42
teacher artifacts separately. The `all` search command
prepares five development teachers for every search seed, then performs the independent screens. Explicit phases support partial
runs and resuming:

```bash
scripts/search.sh teachers cifar100 resnet56_to_resnet20 --download
scripts/search.sh mechanism oxford_pets resnet34_to_resnet18 --download
scripts/search.sh optimization glue_sst2 bert4_to_bert2 --download
scripts/search.sh distillation oxford_pets resnet34_to_resnet18 --download
```

After all runs finish, rank each complete screen for manual analysis:

```bash
python scripts/rank_search.py logs/search/selection --stage mechanism
python scripts/rank_search.py logs/search/selection --stage optimization \
  --dataset oxford_pets
python scripts/rank_search.py logs/search/selection --stage distillation \
  --dataset oxford_pets
```

The following starts the complete development search sequentially on the first
visible GPU. The launcher selects the repository's `inherdemo` Python,
trains/reuses the required seed-matched teachers, then runs the mechanism,
optimization, and distillation families for Hetero. The built-in detached launcher leaves
a console log and PID under `logs/jobs/`:

```bash
cd /root/nas/mingjing/InherNet-Demo
scripts/search.sh all --download --num-workers 4
```

`all` is the complete reproducible screen. Candidate selection is deliberately
post-hoc and manual; the launcher never rewrites later commands from an earlier
result. Here `all` means all 249 runs in the prespecified shared-development
screen, not separate tuning of all 18 registered teacher/model pairs.
`--download` permits missing torchvision datasets to be downloaded,
and `--num-workers 4` uses four data-loader worker processes. Optional overrides
include `SEARCH_SEEDS=42`, `CUDA_VISIBLE_DEVICES=1`, `DEVICE=cpu`, and
`FOREGROUND=1`; none is required for the normal run.

On the current A6000, one-epoch Hetero pilots took 53.5 seconds for
CIFAR-10 and 34.1 seconds for CIFAR-100; extrapolating their full 200/240-epoch
runs gives approximately 3.0 and 2.3 hours before setup. Runtime for the
complete 249-run screen depends strongly on modality, cache state, and missing
teachers; measure completed target phases rather than reusing the obsolete
larger-search projection. Use
`SEARCH_SEEDS=42` for a deliberate one-seed diagnostic screen.

For architecture-generalization evidence, freeze the selected settings and
evaluate the remaining registered pairs; independently tuning each pair or
each size would make comparisons optimistic. Full epochs improve search
fidelity, but do not replace independent seeds or held-out final evaluation.

Repeating one deterministic seed is not an independent trial. Seeds 42, 123,
and 2026 form the multi-seed selection set. After analysis, add complete
finalist recipes beside the committed `weighted_uniform` reference in
`configs/hetero_confirmation_candidates.csv`. Each candidate must provide one
complete row for `cifar10`, `cifar100`, `oxford_pets`,
`glue_classification`, and `glue_regression`. Evaluate those profiles jointly
on confirmation seed 3407. The launcher never writes this file or chooses
finalists:

```bash
SEARCH_SEEDS=3407 scripts/search.sh teachers --download
SEARCH_SEEDS=3407 scripts/search.sh confirmation --download

python scripts/rank_search.py logs/search/selection --stage mechanism \
  --seed 42 --seed 123 --seed 2026
python scripts/rank_search.py logs/search/selection --stage confirmation --seed 3407
```

After choosing a finalist manually, copy its five reviewed profile rows to
`configs/hetero_selected_recipes.csv`. Formal evaluation then uses the disjoint
seeds `7,17,27,37`. The selection and confirmation logs are evidence for model
choice, not part of the headline final-seed mean.

### Search and ablation figures

`scripts/plot_experiments.py` creates 300-DPI PNGs directly from
completed structured logs. Search plots show every within-cell normalized rank
and its mean: the ranking cell is dataset, pair, method, size, and seed. Thus
accuracy, Matthews correlation, and Pearson correlation are never averaged as
raw values. The script requires the complete prespecified search matrix and
labels each candidate's coverage. Ablation plots pair each Hetero
component variant with full Hetero for the same dataset, pair, size, and seed;
they show validation-metric point changes separately by target. Hetero-Lite and
the two InherNet capacities are capacity references rather than paired
component removals. Thin ranges show observed minima and maxima, not confidence
intervals.

Mechanism settings are shared, so its figure spans all five development targets.
Optimization and distillation settings are selected by training family, so the
plotter requires an explicit `--dataset` filter for those stages; it will not
silently pool unrelated decision groups. Oxford, GLUE classification, and
STS-B regression should be plotted and selected separately.

After the relevant runs finish:

```bash
python scripts/plot_experiments.py search logs/search/selection \
  --stage mechanism \
  --output results/paper/search_mechanism.png

python scripts/plot_experiments.py search logs/search/selection \
  --stage optimization --dataset oxford_pets \
  --output results/paper/search_optimization_oxford.png

SEED=42 scripts/ablation.sh cifar100 resnet56_to_resnet20
python scripts/plot_experiments.py ablation logs/ablation \
  --dataset cifar100 --seed 42 \
  --output results/paper/ablation_cifar100.png
```

The recommended paper set uses the activation-geometry and router-tangent
pre-study figures in the main narrative, with the rank-surrogate figure in the
motivation or appendix, one search-rank figure per hyperparameter decision
group, one component-ablation
figure spanning the prespecified representative tasks, and a compact
main-results parameter/performance plot generated only after formal multi-seed
results exist. Do not turn incomplete screening logs into a figure and do not
use pre-study or search validation plots as final-test evidence.

To prepare every registered pair-bound search teacher with the default CUDA/background
behavior, activate the environment and run one command:

```bash
scripts/train_teachers.sh
```

Use `scripts/train_teachers.sh glue` or `scripts/train_teachers.sh vision` to
restrict a maintenance retrain to one modality. With `OVERWRITE_TEACHER=1`,
each old checkpoint remains available until the replacement has completed and
is then atomically replaced; there is no unsafe delete-before-train window.

The launcher prints the PID file and master `stdout.log` under a new
`logs/jobs/teachers_<timestamp>/` directory. Epoch records are deliberately
written to the active structured teacher log, not repeated in the master
console log. To verify the first three completed epochs without stopping the
job:

```bash
job_dir="$(ls -dt logs/jobs/teachers_* | head -n 1)"
cat "$job_dir/job.pid"
tail -n 20 "$job_dir/stdout.log"

run_log="$(sed -n 's/^Log file: //p' "$job_dir/stdout.log" | tail -n 1)"
rg '^RUN_METRICS ' "$run_log" | tail -n 3
```

Wait until that final command shows epochs 1, 2, and 3 for the active teacher.
The background process then continues through the remaining teachers. When a
checkpoint already exists, the launcher reuses it and advances to the next
target.

## Metrics and Logs

Each structured run log contains:

- `RUN_METADATA`: dataset/pair/method, model and data profiles, exact training
  settings, seed, environment, parameter count, split identity, checkpoint
  lineage, selected size, reference InherNet cost, and method-specific
  compression data.
- `RUN_METRICS`: training objective/loss and task metrics plus evaluation
  metrics and timing for every epoch. Distillation runs also record KD and
  router auxiliary-loss components; supervised Hetero records its auxiliary
  component when enabled.
- `INHERITANCE_DIAGNOSTICS`: optional teacher-relative measurements made before
  the first optimizer update.
- `RUN_SUMMARY`: the primary-selected epoch, all registered metrics at that
  epoch, and all registered metrics from the final epoch. Checkpoint selection
  continues to use only the predeclared primary metric.
- `RUN_FINAL_TEST`: only when a held-out final test is enabled after validation
  selection.
- `TEACHER_CHECKPOINT`: artifact path, schema, semantic provenance, and selected
  epoch for teacher runs. Integrity metadata is internal to checkpoint I/O and
  is not used to compare or group experimental results.

Primary metrics are top-1 accuracy for CIFAR, balanced accuracy for Oxford,
accuracy for most GLUE tasks, Matthews correlation for CoLA, and Pearson
correlation for STS-B. Report secondary metrics listed in the registry, total
parameters, the matched InherNet registered rank, diagnostic achieved ratios,
FLOPs/latency, peak memory, and training/decomposition cost where applicable.

## Installation and Validation

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate inherdemo
pip install -r requirements.txt

python -m py_compile demo_code.py experiment_registry.py training_utils.py \
  checkpointing.py plotting_utils.py cifar10_models.py cifar100_models.py \
  pet_models.py glue_models.py glue_data.py model_wrappers.py \
  scripts/audit_teachers.py scripts/summarize_search.py \
  scripts/rank_search.py scripts/plot_experiments.py scripts/summarize_prestudy.py \
  scripts/plot_prestudy_progression.py scripts/plot_prestudy_router_activity.py \
  scripts/plot_prestudy_allocation.py scripts/plot_prestudy_local_operator.py \
  tests/test_registry_and_wrappers.py tests/test_plot_experiments.py
python -m unittest discover -s tests -t .
bash -n scripts/run.sh scripts/formal.sh scripts/prestudy.sh scripts/ablation.sh scripts/smoke.sh \
  scripts/search.sh scripts/train_teachers.sh scripts/common.sh
scripts/smoke.sh cifar100 resnet56_to_resnet20 --svd-backend device
scripts/smoke.sh oxford_pets resnet34_to_resnet18 --svd-backend device
scripts/smoke.sh glue_sst2 bert4_to_bert2 --svd-backend device
```

## Project Structure

- `demo_code.py`: maintained CLI and experiment orchestration.
- `checkpointing.py`: atomic, validated teacher artifact persistence.
- `experiment_registry.py`: datasets, pairs, splits, defaults, and run tags.
- `model_wrappers.py`: InherNet and Hetero factorization/routing modules.
- `training_utils.py`: supervised/KD loops, metrics, logging, and checks.
- `cifar10_models.py`, `cifar100_models.py`, `pet_models.py`,
  `glue_models.py`: model registries.
- `glue_data.py`: GLUE tokenization and dataloaders.
- `scripts/run.sh`, `scripts/formal.sh`, `scripts/prestudy.sh`, `scripts/ablation.sh`,
  `scripts/smoke.sh`, `scripts/search.sh`, `scripts/train_teachers.sh`: launchers
  grouped by purpose.
- `configs/`: committed search candidate tables.
- `scripts/common.sh`: launcher validation, foreground execution, and detached
  job helper.
- `scripts/summarize_search.py`: deterministic search-summary export.
- `scripts/summarize_prestudy.py`: optional structured-log extraction to standard output.
- `scripts/plot_prestudy_progression.py`: fixed-capacity progression figure and its raw data.
- `scripts/plot_prestudy_local_operator.py`: held-out per-layer operator-fidelity figure and its raw data.
- `scripts/plot_prestudy_router_activity.py`: per-router control figure and its raw data.
- `scripts/plot_prestudy_allocation.py`: allocation trade-off figure and its raw data.
- `scripts/rank_search.py`: metric-agnostic within-target rank aggregation.
- `scripts/plot_experiments.py`: validated 300-DPI PNG search and ablation
  figures.
- `scripts/audit_teachers.py`: strict validation and manifest export for all
  registered teacher artifacts.
- `plotting_utils.py`: structured-log plotting.
- `tests/`: unit and smoke-oriented checks.
- `ideas/`: synchronized method and experimental design notes.
- `demo_code_org.py`: frozen historical reference.

## Evidence Required for a Top-Tier Paper

The repository provides an experimental framework, not sufficient evidence by
itself. A strong submission still requires:

- multi-seed means, standard deviations or confidence intervals, and a
  prespecified model-selection protocol;
- teacher, compact student, student KD, fixed-rank InherNet, and
  competitive contemporary compression/inheritance baselines;
- results across multiple CIFAR-100 architecture families plus the Oxford and
  GLUE extensions, without treating alias pairs as independent evidence;
- parameter, FLOP, measured latency, throughput, peak-memory, decomposition,
  and training-cost comparisons on declared hardware;
- ablations for activation weighting, conditional routing, load balancing,
  calibration size, shrinkage, expert perturbation, and objective;
- sensitivity and failure analysis, including the diagnostic heterogeneous-rank
  policies and the gap between local reconstruction error and downstream quality;
- strictly validation-only tuning and untouched final-test evaluation.

The current formal CIFAR launcher selects with the fixed training holdout and
evaluates the official test once after selection. An optional retraining rule
on all training data would need to be prespecified rather than chosen after
seeing test results. Likewise, the compact-BERT GLUE suite and Oxford transfer
experiment broaden evaluation but do not replace replication of the original
paper's full benchmark protocol.

The defensible mechanism claim is deliberately narrow: at the same registered
rank as InherNet, activation moments make each weighted low-rank approximation
minimize a local empirical output-MSE objective, while zero-mean expert
perturbations preserve that approximation under the initially uniform router.
The convolution channel-moment mode is an approximation, local reconstruction
is not a downstream-loss or Hessian objective, and lower local error does not
guarantee better generalization or expert specialization. The `research_*`
rank allocators are reported only as pre-study diagnostics and are not part of
the claimed method.

## Citation

The Hetero citation will be added if a paper is released. Cite the
InherNet baseline as:

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

This repository uses the original project license in `LICENSE`.
