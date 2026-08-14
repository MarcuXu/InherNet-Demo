# InherAct: Activation-Aware Conditional Experts for Fixed-Capacity Model Inheritance

InherAct turns fixed-capacity neural-network inheritance into a
**preserve-then-adapt** initialization. At the exact InherNet rank and parameter
count, it minimizes activation-weighted local reconstruction error for the
inherited expert mean, then introduces zero-sum expert deviations that preserve
this reconstruction while exposing conditional router directions.

## 1. Motivation and Contributions

Weight-space SVD preserves dominant parameters, but not necessarily behavior on
task-relevant activations. Moreover, identical inherited experts make the
router locally insensitive at initialization. InherAct addresses both limitations
through one constrained construction:

1. the expert-mean operator is selected under the empirical activation metric;
2. expert deviations are introduced in its zero-sum subspace, preserving the
   inherited mean while making conditional routing locally trainable.

These are sequential consequences of the inheritance objective, not
independently accumulated enhancements. The first constraint selects a
fixed-rank mean operator for functional preservation. Once that operator is
fixed, the second is restricted to the nullspace of the uniform expert average:
it changes the router's local tangent geometry, but not the inherited function.
Accordingly, full InherAct must reproduce the activation-aware base's zero-step
map up to numerical error; its additional effect appears in the conditional
tangent.

This construction provides exact capacity matching, a closed-form local optimum
under the empirical activation metric, and reconstruction-preserving conditional
directions. Unlike activation-aware decomposition or symmetry-preserving model
transformation in isolation, InherAct studies their joint role in inheriting a
trained conditional-expert network at the exact InherNet capacity.

[SVD-LLM (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3104e1ab39875cf54fe1eb4473e7c5a1-Abstract-Conference.html)
and [CorDA (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/83f95bb0ac5046338ea2afe3390e9f4b-Abstract-Conference.html)
establish activation-oriented decomposition, while
[LEMON (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0e705ac30e573d1526f81a0fd071a151-Abstract-Conference.html)
provides a precedent for symmetry-preserving transformation. InherAct's
contribution is the inheritance-specific preserve-then-adapt constraint that
connects these principles at fixed conditional-expert capacity.

## 2. InherNet Baseline and Trained Sources

For a trained source weight

$$
W\in\mathbb{R}^{m\times n},\qquad W=U\Sigma V^\top,
$$

the rank-$r$ InherNet initialization uses balanced factors

$$
W^{\mathrm{down}}=\Sigma_r^{1/2}V_r^\top,\qquad
W_h^{\mathrm{up}}=U_r\Sigma_r^{1/2}.
$$

The gate is a softmax over experts. Because its weights sum to one, identical
up-projection experts reconstruct $U_r\Sigma_rV_r^\top$ at initialization.
All methods inherit from the same task-trained dense checkpoint. The source
remains frozen when used for distillation, ensuring that differences arise from
the inherited parameterization and training objective rather than teacher
drift.

## 3. InherAct

### 3.1 Uncentered Activation Second Moments

For layer input feature $x$, InherAct estimates

$$
\widehat M=\frac{1}{N}\sum_{i=1}^{N}x_ix_i^\top.
$$

The uncentered second moment follows directly from the local reconstruction
objective

$$
\mathbb{E}\| (W-\widehat W)x\|_2^2
=\operatorname{tr}\!\left((W-\widehat W)M
(W-\widehat W)^\top\right).
$$

The empirical estimate is stabilized as

$$
\mu=\frac{\operatorname{tr}(\widehat M)}{d},\qquad
M_\lambda=(1-\lambda)\widehat M
+\lambda\mu I+\epsilon\mu I,
$$

with an analogous element-wise form for diagonal moments. The reference
configuration uses shrinkage $\lambda=0.01$, 16 calibration batches, and at
most 4096 sampled features per layer per batch. Matrix-valued modes add
positive diagonal jitter before Cholesky factorization. We denote the
stabilized matrix actually factored by $\widetilde M_\lambda$.

Calibration scales with layer shape:

- convolution patch dimension at most 256: exact unfolded patch moments;
- wider convolutions: a stride-, dilation-, padding-, and
  kernel-position-aware channel-block approximation;
- linear input dimension at most 512: a full second moment;
- wider linear layers: a diagonal second moment; and
- transformer inputs: padding tokens are excluded using the attention mask.

The full-moment construction instantiates the theorem in Section 4 exactly.
Channel-block and diagonal statistics define memory-bounded surrogate metrics.

### 3.2 Data-Weighted Preserve-Then-Adapt Decomposition

Let

$$
\widetilde M_\lambda=CC^\top,\qquad A=WC.
$$

If $[A]_r=U_r\Sigma_rV_r^\top$, InherAct initializes

$$
W^{\mathrm{down}}
=\Sigma_r^{1/2}V_r^\top C^{-1},\qquad
W_h^{\mathrm{up}}
=U_r\Sigma_r^{1/2}+E_h,\qquad
\sum_{h=1}^{H}E_h=0.
$$

The first constraint preserves the source where calibration activations place
mass. The second breaks expert symmetry without changing the average
reconstruction. A zero-initialized softmax router consumes the compressed
feature, so the two constraints form one preserve-then-adapt initialization
rather than a collection of independent modules.

Setting every $E_h=0$ gives the **activation-aware base** used only as an
analytical intermediate and experimental control. Full InherAct uses nonzero
zero-sum $E_h$. The two constructions intentionally implement the same
zero-step input--output map under uniform routing, up to numerical precision;
they differ in whether that map has trainable conditional router directions.

### 3.3 Registered-Rank Capacity Matching

InherAct uses the same registered layer rank, eligible layers, expert count,
router dimensions, biases, and dense-layer decisions as its matched InherNet.
InherAct-Lite matches InherNet-Small, while InherAct matches InherNet-Large.
Consequently, each comparison has exact total-model parameter equality by
construction, including fixed vision parameters and fixed BERT embeddings.

The CIFAR-100 pairs use the ranks printed in the InherNet paper. Measured
construction counts are reported for the two pairs whose printed ranks and
reported counts disagree. InherAct fixes these registered ranks by design. The
pre-study evaluates layer-varying allocation under the same parameter cap only as a
diagnostic of whether additional allocation complexity improves behavioral
preservation.

### 3.4 Routing and Training Objective

For mean routing probabilities $\bar g_h$, define

$$
\mathcal L_{\mathrm{bal}}
=H\sum_{h=1}^{H}\bar g_h^2-1\ge 0.
$$

The current reference fine-tuning objective uses
$\lambda_{\mathrm{bal}}=0.01$:

$$
\mathcal L
=w_{\mathrm{CE}}\mathcal L_{\mathrm{CE}}
+w_{\mathrm{KD}}T^2
\operatorname{KL}(p_{\mathrm{teacher}}^T\|p_{\mathrm{student}}^T)
+\lambda_{\mathrm{bal}}\mathcal L_{\mathrm{bal}}.
$$

For supervised configurations, the task term replaces the CE--KD mixture.
HPO compares $\lambda_{\mathrm{bal}}\in\{0,0.01,0.03\}$, and the zero-weight
component ablation isolates its contribution. All inherited factors and routers
are optimized end to end.

## 4. Theoretical Analysis

### 4.1 Data-Weighted Eckart--Young Theorem

Let $M\succ0$, $M=CC^\top$, and $A=WC$. Among all matrices $Q$ with rank at
most $r$,

$$
Q_r^*=[A]_rC^{-1}
$$

minimizes

$$
\operatorname{tr}\!\left((W-Q)M(W-Q)^\top\right)
=\|(W-Q)C\|_F^2.
$$

The proof changes variables to $B=QC$ and applies the ordinary
Eckart--Young--Mirsky theorem. The result is exact for the stabilized empirical
full-moment metric.

### 4.2 Initialization Preservation and Conditional Accessibility

For one inherited layer, write

$$
u_h(x)=(B+E_h)z(x)+b,
\qquad
y(x)=\sum_{h=1}^H g_h(x)u_h(x),
$$

where $g=\operatorname{softmax}(a)$ and
$a_h=q_h^\top v+c_h$. InherAct initializes $q_h=c_h=0$, so $g_h=1/H$.
The zero-sum constraint $\sum_hE_h=0$ then gives

$$
y_0(x)=\frac{1}{H}\sum_{h=1}^H\big((B+E_h)z(x)+b\big)
=Bz(x)+b.
$$

Layerwise, this preserves the activation-weighted rank-$r$ inherited network.
Preservation does not force conditional learning to vanish, because the router
derivatives are

$$
\frac{\partial y}{\partial a_j}
=g_j(u_j-y)=\frac{1}{H}E_jz,
\qquad
\frac{\partial y}{\partial q_j}
=\frac{1}{H}(E_jz)v^\top,
\qquad
\frac{\partial y}{\partial c_j}=\frac{1}{H}E_jz.
$$

Thus the zero-sum lift preserves the inherited function while making routing
immediately observable in its parameter-space tangent. With duplicated experts
($E_h=0$), the router derivative is zero for every routing distribution.
Original InherNet nevertheless uses randomly initialized gates, so its generally
unequal gate probabilities weight the copied expert-factor gradients
differently. The copies can separate after the first factor update, after which
the gate becomes trainable. InherAct replaces this delayed conditional
adaptation with an accessible conditional tangent at the preserved inherited
function.

### 4.3 Positive-Semidefinite Tangent Augmentation

Stack the model outputs on a fixed minibatch into $F$. Let $J_0$ collect the
initialized Jacobian blocks shared by the activation-aware base and InherAct,
and let $J_{\mathrm{gate}}$ collect the router blocks. At their common initial
function,

$$
J_{\mathrm{base}}=[J_0,0],
\qquad
J_{\mathrm{InherAct}}=[J_0,J_{\mathrm{gate}}].
$$

Their empirical tangent kernels therefore satisfy

$$
K_{\mathrm{InherAct}}
=J_{\mathrm{InherAct}}J_{\mathrm{InherAct}}^\top
=K_{\mathrm{base}}+K_{\mathrm{gate}},
\qquad
K_{\mathrm{gate}}=J_{\mathrm{gate}}J_{\mathrm{gate}}^\top\succeq0.
$$

For one linear inherited layer and one input, the router-weight contribution has

$$
\operatorname{tr}(K_{\mathrm{gate}}^{(q)})
=\frac{\|v\|_2^2}{H^2}\sum_{h=1}^H\|E_hz\|_2^2,
$$

while router biases add
$H^{-2}\sum_h\|E_hz\|_2^2$. In a deep network the new direction is propagated
through the downstream Jacobian $D_l(x)$:

$$
\frac{\partial F(x)}{\partial q_{l,h}}
=D_l(x)\left[\frac{E_{l,h}z_l(x)}{H}\right]v_l(x)^\top.
$$

The lift therefore adds accessible first-order directions without removing the
directions of the matched base. Their optimization value is determined by how
strongly these new directions align with the task residual.

### 4.4 Conditional Descent and Local Rate

For a teacher-matching squared loss
$\mathcal L=\tfrac12\|F-T\|_2^2$ with residual $r=F-T$,

$$
\nabla_q\mathcal L=J_{\mathrm{gate}}^\top r,
\qquad
\|\nabla_q\mathcal L\|_2^2=r^\top K_{\mathrm{gate}}r.
$$

For another differentiable training objective, the same identity holds with
$r$ replaced by its output-space gradient $s=\nabla_F\mathcal L$.

If $\mathcal L$ is locally $\beta_q$-smooth in the router block, a router step
satisfies

$$
\mathcal L(q-\eta\nabla_q\mathcal L)
\leq \mathcal L(q)
-\eta\left(1-\frac{\beta_q\eta}{2}\right)
\|\nabla_q\mathcal L\|_2^2.
$$

For $\eta\leq1/\beta_q$, residual-aligned lift directions
($r^\top K_{\mathrm{gate}}r>0$) therefore supply a strictly positive initial
conditional descent term. If the training trajectory remains in a locally
tangent-stable region satisfying the Polyak--Lojasiewicz condition

$$
\frac12\|\nabla\mathcal L(\theta)\|_2^2
\geq\mu\big(\mathcal L(\theta)-\mathcal L^*\big),
$$

then gradient descent with $\eta\leq1/\beta$ obeys

$$
\mathcal L_{k+1}-\mathcal L^*
\leq(1-\eta\mu)(\mathcal L_k-\mathcal L^*).
$$

This connects the construction to a testable rate prediction: when the added
kernel energy is residual aligned and raises the effective local curvature,
InherAct should reduce the matched objective faster than the function-equivalent
base ([Jacot et al., 2018](https://papers.nips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html);
[Karimi et al., 2016](https://mlanthology.org/ecmlpkdd/2016/karimi2016ecmlpkdd-linear/)).

### 4.5 Evidence and Matched Convergence Study

The current evidence supports this progression at three levels:

1. **Structural pre-study.** On CIFAR-100, the activation-aware base and
   InherAct preserve the same zero-step accuracy (7.42%) and nearly identical
   teacher-output relative SSE (0.8368906 versus 0.8368925), while the
   router-gradient RMS rises from $3.20\times10^{-9}$ to
   $3.63\times10^{-4}$ and all 37 measured routers become active. Oxford Pets
   shows the same separation: equal zero-step balanced accuracy (77.468%),
   teacher-output relative SSE 0.2837889 versus 0.2838478, and router-gradient RMS
   $1.54\times10^{-9}$ versus $2.27\times10^{-4}$.
2. **Development mechanism screen.** At fixed CIFAR-100 rank, capacity,
   optimizer, and 240-epoch horizon, the three-seed validation results for lift
   scales $0$, $0.005$, $0.010$, and $0.020$ are respectively
   $69.213\pm0.248$, $71.300\pm0.745$, $70.873\pm0.601$, and
   $70.647\pm0.705$ percent. This screen identifies a small nonzero lift as the
   useful regime and selects $0.005$ for the formal recipe.
3. **Formal adaptation traces.** Across four CIFAR-100 seeds, the selected
   InherAct recipe begins at 42.34% mean epoch-1 validation accuracy and reaches
   50% in 4.25 epochs, versus 31.54% and 26.5 epochs for InherNet-Large. These
   traces motivate the matched rate study because the independently selected
   learning rates differ (0.025 and 0.05).

The causal study isolates the conditional lift with a paired comparison between
the activation-aware base ($E_h=0$) and InherAct ($E_h$ scale $0.005$).
InherNet and weight-only SVD remain external inheritance controls. The paired
cells use the same frozen teacher checkpoint, rank, head count, calibration
examples and statistics, seed, post-initialization training RNG reset, data
order, objective, optimizer, momentum, weight decay, learning rate, schedule,
and 240-epoch budget. Across four paired seeds, the study records the actual
training objective by update, validation accuracy, router-gradient RMS,
$\operatorname{tr}(K_{\mathrm{gate}})$ or a documented trace estimate, and
$r^\top K_{\mathrm{gate}}r$ on fixed probe batches through the first 10--20
epochs. Equal zero-step outputs, zero base router energy, positive lifted router
energy, and faster residual reduction constitute the predicted causal chain.

The data-weighted theorem is exact for the stabilized full moment, while
channel-block and diagonal statistics provide scalable surrogate metrics.
Sections 6 and 7 connect these initialization properties to downstream
accuracy and learned specialization.

## 5. Dataset and Model Protocols

| Dataset family | Teacher $\rightarrow$ student | Initialization | Inherited source/objective | Targets |
|---|---|---|---|---|
| CIFAR-10 | ResNet-50 $\rightarrow$ ResNet-18 with CIFAR stem | random | trained teacher / KD | convolution |
| CIFAR-100 | eight CIFAR-native ResNet, VGG, and WRN pairs | random | trained teacher / supervised | convolution |
| Oxford-IIIT Pet | ResNet-34 $\rightarrow$ ResNet-18 | ImageNet pretrained, new 37-class head | fine-tuned teacher / KD | convolution |
| GLUE | BERT-Mini $\rightarrow$ BERT-Tiny | pretrained compact BERT | fine-tuned teacher / KD | linear |

| Dataset family | Optimizer | Batch | Epochs | Learning rate | Weight decay | Schedule |
|---|---|---:|---:|---:|---:|---|
| CIFAR-10 | SGD, momentum 0.9 | 128 | 200 | 0.05 | $5\times10^{-4}$ | decay at 100, 150, 180 |
| CIFAR-100 | SGD, momentum 0.9 | 64 | 240 | 0.05 | $5\times10^{-4}$ | decay at 150, 180, 210 |
| Oxford-IIIT Pet | SGD, momentum 0.9 | 32 | 30 | 0.001 | $10^{-4}$ | decay at 15, 25 |
| GLUE compact BERT | AdamW | 32 | 4 | $5\times10^{-5}$ | 0.01* | 10% linear warmup, linear decay |

`*` Bias and LayerNorm parameters are excluded from weight decay, and the
global gradient norm is clipped at 1.0.

The benchmark suite spans scratch-trained vision on CIFAR,
ImageNet-initialized fine-grained transfer on Oxford Pets, and compact
transformer transfer on GLUE. Convolutions are inherited for vision, while
attention and feed-forward projections are inherited for GLUE. The small
Oxford classifier, transformer embeddings, and normalization layers remain
dense. The compact-BERT GLUE track is a resource-efficient cross-modal
extension that complements the original T5 benchmark.

The CIFAR-100 pairs are ResNet-32/8, ResNet-32x4/8x4, VGG-13/8,
WRN-40-2/40-1, WRN-40-2/16-2, ResNet-56/20, ResNet-110/32, and
ResNet-110/20. GLUE covers MRPC, QQP, SST-2, MNLI, RTE, QNLI, CoLA, and
STS-B. MRPC and QQP report accuracy and F1; SST-2, MNLI, RTE, and QNLI report
accuracy; CoLA reports Matthews correlation; and STS-B reports Pearson and
Spearman correlations.

## 6. Experiment and Search Protocol

Each teacher is trained for the complete dataset schedule, selected using its
validation metric, and reused unchanged by student KD, InherNet, InherAct, and
ablations within that experiment. CIFAR uses a fixed stratified 10% training
holdout, Oxford uses its fixed stratified validation split, and GLUE reserves a
deterministic 10% training holdout for selection. CIFAR/Oxford official test
data and the public GLUE validation split are evaluated only after the selected
state has been restored.

Every HPO candidate uses the same epoch count as formal training. Mechanism and
learning-rate screening cover CIFAR-10, CIFAR-100 ResNet-56-to-ResNet-20,
Oxford Pets, SST-2, and STS-B. HPO tunes InherAct at the InherNet-Large capacity;
InherNet retains its registered settings, and InherAct-Lite inherits the selected
InherAct recipe as a capacity ablation.

The mechanism screen varies routing regularization, moment shrinkage, and
zero-sum lift scale; learning rate and distillation settings are screened
separately to avoid confounding optimizer and mechanism effects. The completed
three-seed screen selects the profile-specific default recipes; final estimates
use disjoint seeds 7, 17, 27, and 37.

For reproducibility, mechanism candidates fix the learning-rate multiplier at
1.0, and mechanism and learning-rate candidates use the objective registered
for each dataset profile. Distillation candidates likewise fix the multiplier
at 1.0. These references are read from the committed
`configs/inheract_reference_recipes.csv` registry, not from screen-selected
recipes; each generated command contains every
controlled argument exactly once.

The main comparison includes the teacher, compact student, student KD,
paper-configured InherNet-Small + KD, capacity-matched supervised
InherNet-Large, and InherAct. Registered CIFAR-100 pairs additionally use
standalone KD with Logit Standardization, CTKD, DKD, SimKD, ReviewKD, and CRD,
plus a CAT-KD objective/configuration adaptation that retains the repository's
native classifier head; unsupported architecture pairs are omitted rather than
assigned an invented coefficient. The standalone Logit Standardization row is not the
paper's 480-epoch MLKD + Logit Standardization method. CIFAR-10 adds explicitly
labeled CTKD and DKD repository adaptations, while Oxford and GLUE retain the
domain-appropriate common core. A supervised InherAct control accompanies
every distilled headline recipe. Component ablations isolate activation
weighting, the zero-sum conditional lift, learned versus fixed-uniform routing,
routing regularization, calibration budget, and one-head Direct-SVD
inheritance. InherAct-Lite tests the accuracy--efficiency trade-off.

The fresh four-seed formal suite contains 158 one-seed cells and 632 runs:
8 CIFAR-10 cells, 92 CIFAR-100 cells, 8 Oxford Pets cells, and 50 GLUE cells
per seed. CIFAR-100 has eight architecture pairs, so its total is not the
number of baselines on one pair. These are planned totals for a new run
namespace, not a remaining count after prior experiments.

## 7. Problem-Driven Pre-Study

The theorem and router-Jacobian analysis specify two hypotheses before any
measurement:

1. activation weighting should improve teacher-behavior preservation at the
   same rank and parameter count; and
2. the zero-sum lift should activate router gradients without materially
   changing the inherited predictions.

The pre-study tests these hypotheses with a nested, capacity-matched contrast.
Matched InherNet checks the registered baseline, and the weight-only
conditional-expert construction supplies the ordinary-SVD reference in the
same parameterization. The activation-aware base sets $E_h=0$ and isolates
preservation under the empirical activation metric. Full InherAct then changes
only $E_h$, subject to $\sum_hE_h=0$, and isolates the conditional tangent.
Thus InherAct is included to verify both functional preservation and router
activation; its success criterion is output equivalence to the activation-aware
base, not a further zero-step gain. Near identity between those two rows is a
required invariant, whereas nonzero router gradients are the new effect. The
hypotheses are falsifiable: failure of the base to improve held-out local
preservation rejects the preservation hypothesis, while a materially changed
inherited map or dormant router rejects the conditional-lift hypothesis.

Oxford Pets provides a small, fine-grained transfer setting, while CIFAR-100
ResNet-56-to-ResNet-20 tests whether the effect transfers to scratch-trained
vision. Fidelity and task metrics use the complete validation split. To test
the decomposition claim without confounding it with upstream approximation
errors, a held-out local-operator probe feeds dense-teacher activations into
each matching inherited operator for four deterministic validation batches and
reports the ratio of summed output errors to summed dense-output energy.
Construction also records each layer's relative expert-mean shift, directly
checking the zero-mean lift invariant. Router gradients are measured on the
first deterministic evaluation minibatch—32
Oxford examples or 64 CIFAR-100 examples—using teacher KL without an optimizer
step. Each router dot reports
$\max(\|\partial\mathrm{KL}/\partial W_{\mathrm{gate}}\|_2,
\|\partial\mathrm{KL}/\partial b_{\mathrm{gate}}\|_2)$ for one inherited
layer.

### 7.1 The Activation-Aware Base Isolates Preservation

Comparing the weight-only reference with the activation-aware base, at fixed
capacity, isolates the first constraint. Activation weighting reduces
teacher-output relative SSE by
66.52% on Oxford and 17.59% on CIFAR-100. Teacher KL falls by 82.44% and 14.10%,
respectively; agreement rises by 61.41 and 5.28 percentage points. These
behavioral improvements coincide with gains of 59.175 points in Oxford
balanced accuracy and 4.88 points in CIFAR-100 accuracy. Output cosine
similarity rises by 0.419 and 0.282.

![The fixed-capacity contrast isolates activation weighting as the source of improved output fidelity and zero-step task behavior.](../figures/prestudy_inheritance_progression.png)

### 7.2 Activation Weighting Improves Every Probed Local Operator

The held-out local probe isolates each factorized layer by replaying the same
dense-teacher inputs through its dense and inherited operators. Activation
weighting reduces aggregate local relative SSE from 0.1255 to 0.0598 on Oxford
(52.3%) and from 0.2869 to 0.1849 on CIFAR-100 (35.6%). The improvement holds
for all 65 probed operators across both architectures. This layerwise result
connects the activation geometry used during decomposition to the end-to-end
behavioral gains above.

![Activation-aware decomposition lowers held-out local operator error for every probed layer on both architectures.](../figures/prestudy_local_operator.png)

### 7.3 The Conditional Lift Exposes Router Tangents

As predicted, full InherAct preserves the activation-aware base's zero-step
function. The zero-sum lift changes relative SSE by only
$5.89\times10^{-5}$ on Oxford and $1.90\times10^{-6}$ on CIFAR-100, while
leaving task performance and teacher agreement unchanged. This near identity
verifies mean preservation rather than indicating a redundant component.
Normalized routing entropy remains exactly 1.0,
while mean relative expert diversity rises from numerical zero to 0.008173 on
Oxford and 0.008152 on CIFAR-100: the lift creates distinct experts without
disturbing uniform mean routing. It raises router-gradient RMS from
$1.54\times10^{-9}$ to $2.27\times10^{-4}$ on Oxford and from
$3.20\times10^{-9}$ to $3.63\times10^{-4}$ on CIFAR-100—approximately
$1.47\times10^5$-fold and $1.13\times10^5$-fold gains. All 28 Oxford and all
37 CIFAR-100 router layers cross the $10^{-7}$ activity tolerance after the
lift, whereas none cross it with identical experts.

![Only the zero-sum conditional lift activates every measured router across the four inheritance controls.](../figures/prestudy_router_activity.png)

### 7.4 Fixed Rank Avoids a Surrogate--Task Mismatch

Fixed registered rank follows from the controlled-comparison requirement:
preservation quality must be identifiable at exactly the InherNet capacity.
The separate allocation diagnostic asks whether relaxing that design is
nevertheless justified; it is not a component used to define InherAct. On
CIFAR-100, relative allocation lowers relative SSE from 0.8369 to
0.8108 but also lowers zero-step accuracy from 7.42% to 6.80%. On Oxford, both
relative and nested allocation underperform the fixed-rank activation-aware
construction in reconstruction and balanced accuracy. Optimizing a layerwise
allocation surrogate therefore adds complexity without reliably improving
teacher behavior or the downstream task metric. The allocation evidence is
therefore a sensitivity analysis supporting the simpler design constraint, not
a source of additional InherAct components.

![Budget-matched rank allocation can improve output and KL surrogates without improving task retention.](../figures/prestudy_allocation_tradeoff.png)

### 7.5 Completed Zero-Step Diagnostics

| Dataset | Construction | Parameters | Rank range | Relative SSE | Teacher KL / agreement | Initial metric | Router-grad RMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Oxford | weight-only | 5,699,257 | 64 | 0.847742 | 2.766147 / 19.57% | 18.293 BAcc | $1.63\times10^{-9}$ |
| Oxford | activation-aware base ($E_h=0$) | 5,699,257 | 64 | 0.283789 | 0.485746 / 80.98% | 77.468 BAcc | $1.54\times10^{-9}$ |
| Oxford | InherAct ($\sum_hE_h=0$) | 5,699,257 | 64 | 0.283848 | 0.485838 / 80.98% | 77.468 BAcc | $2.27\times10^{-4}$ |
| CIFAR-100 | weight-only | 383,507 | 16 | 1.015533 | 4.350336 / 2.54% | 2.54 Acc | $2.36\times10^{-9}$ |
| CIFAR-100 | activation-aware base ($E_h=0$) | 383,507 | 16 | 0.836891 | 3.736992 / 7.82% | 7.42 Acc | $3.20\times10^{-9}$ |
| CIFAR-100 | InherAct ($\sum_hE_h=0$) | 383,507 | 16 | 0.836893 | 3.736822 / 7.82% | 7.42 Acc | $3.63\times10^{-4}$ |

This seed-42 initialization study verifies the two theory-specified invariants:
the base improves preservation, and full InherAct retains that function while
opening conditional tangent directions. The completed four-seed experiment
below tests whether those effects improve the final accuracy--efficiency
frontier.

## 8. Completed Four-Seed CIFAR-100 Reference Result

The ResNet-56-to-ResNet-20 formal matrix selects epochs on a fixed stratified
training holdout and evaluates the official test split once after restoration.
Results are mean $\pm$ sample standard deviation for seeds 7, 17, 27, and 37.
The fresh suite retains these core recipes and adds the finalized
dataset-specific baseline matrix in a new run namespace; it will provide the
unified reportable table.

| Method | Parameters | Test accuracy (\%) |
|---|---:|---:|
| Teacher | 861,620 | $71.56 \pm 0.44$ |
| Student | 278,324 | $68.49 \pm 0.20$ |
| Student + KD | 278,324 | $69.92 \pm 0.58$ |
| Student + DKD | 278,324 | $70.19 \pm 0.15$ |
| InherNet-Small + KD | 205,663 | $47.90 \pm 14.50$ |
| InherNet-Large | 383,507 | $70.74 \pm 0.34$ |
| **InherAct** | **383,507** | **$71.87 \pm 0.51$** |

At the exactly matched 383,507-parameter budget, InherAct improves over
InherNet-Large in all four paired seeds by 1.13 points on average. It improves
over DKD by 1.68 points and reaches a slightly higher four-seed mean than the
teacher while using 44.5\% as many parameters. The paired teacher differences
are $-0.26,+0.63,+0.25,+0.62$ points, supporting a competitive and occasionally
teacher-surpassing inheritor rather than a universal superiority claim.

The completed formal traces motivate a targeted convergence study: the
HPO-selected InherAct recipe used learning rate 0.025, while paper-configured
InherNet used 0.05, so the existing trajectories are not a matched optimizer
comparison. Mean epoch-1 validation accuracy was 42.34\% for InherAct and
31.54\% for InherNet-Large, and the first epoch reaching 50\% averaged 4.25 and
26.5. The zero-step router-tangent experiment already isolates the relevant
mechanism; the protocol in Section 4.5 tests whether it causes the predicted
early optimization gain.

The InherNet-Small KD row is complete but unstable: its four test accuracies are
50.47, 54.21, 59.95, and 26.95. An exact factorization audit rules out a
shared-down or head-normalization implementation error (maximum checked
rank-truncated reconstruction error: $1.27\times10^{-7}$). Rank 8 factorizes
all 57 convolutional layers, whereas rank 16 factorizes 37, and captures much
less deeper-layer spectral energy. Combined with its `0.1 CE + 36 KL`
objective, this makes Small an aggressive-compression stress case rather than
the headline InherNet baseline. It is retained exactly as the paper-configured
baseline; the current result is a reproducibility gap and a low-rank-sensitivity
observation, not evidence of an implementation defect.

## 9. Conclusion

InherAct recasts inheritance as a constrained initialization problem: preserve
the teacher where task activations place mass, then expose conditional degrees
of freedom without additional capacity or initial reconstruction error. The
pre-study verifies both predicted initialization effects and supports carrying
the registered-rank preserve-then-adapt mechanism into formal
evaluation.
The activation-aware base is the preservation control, not a competing method;
full InherAct is the function-preserving conditional parameterization trained in
subsequent experiments.
