# HeteroInherNet-IB: Paper Framework and Revision Notes

This document is a writing guide for a possible Hetero paper built on
InherNet.  It intentionally separates the paper narrative from engineering
details.  The paper does not need to describe every implementation branch, but
the theoretical claims must be stated with assumptions that we can defend.

Working title:

**HeteroInherNet-IB: Data-Aware Heterogeneous Neural Network Inheritance**

The `IB` suffix should be used only if the final theory section contains a
clear information-bottleneck or rate-distortion derivation.  Otherwise, a safer
title is **HeteroInherNet: Data-Aware Heterogeneous Neural Network
Inheritance**.

---

## 1. What InherNet Actually Contributes

InherNet starts from a pretrained dense source network and constructs an
inheriting network by low-rank factorizing full weights.  This is different
from LoRA-style adaptation, where the pretrained weight is frozen and low-rank
matrices parameterize a task-specific weight update.

For a weight matrix
$$
W \in \mathbb{R}^{m \times n}, \qquad W = U\Sigma V^\top,
$$
the rank-$r$ approximation is
$$
W_r = U_r \Sigma_r V_r^\top .
$$
For a layer acting as $y = Wx$, the conventional two-factor initialization is
$$
W^{\mathrm{down}} = \Sigma_r^{1/2}V_r^\top,\qquad
W^{\mathrm{up}} = U_r\Sigma_r^{1/2}.
$$
Some papers or codebases transpose this notation depending on whether features
are represented as row or column vectors.  The paper should therefore emphasize
the factorization principle rather than over-commit to a notation that creates
dimension confusion.

For convolution kernels, the channel-decomposition view reshapes a kernel
$K\in \mathbb{R}^{c_{\mathrm{out}}\times c_{\mathrm{in}}\times k_h\times k_w}$
into a matrix
$\widehat K\in\mathbb{R}^{c_{\mathrm{out}}\times (c_{\mathrm{in}}k_hk_w)}$
before SVD.

InherNet's structure inheritance uses a one-down-many-ups module:
$$
Y = \sum_{h=1}^{H} G_h(X)\, W_h^{\mathrm{up}}
        \bigl(W^{\mathrm{down}}(X)\bigr),
\qquad
G(X)=\mathrm{softmax}(W_g\phi(X)).
$$
Here $\phi(X)$ is a pooled or flattened routing feature.  The main message is:

- SVD initialization transfers principal spectral knowledge from the dense
  source.
- Rank $r$ is the main knob controlling inherited knowledge.
- Multiple heads can improve specialization, but the benefit is secondary and
  empirically/theoretically diminishing.
- The original convergence discussion should be treated as conditional on
  standard nonconvex SGD assumptions, not as a universal guarantee.

---

## 2. Corrections to the Previous Drafts

The earlier Markdown and LaTeX drafts contained several claims that should be
softened or corrected before becoming a paper.

1. **Do not claim that Hetero preserves InherNet's guarantees without
   assumptions.**
   It is safer to say that Hetero keeps the same differentiable
   one-down-many-ups parameterization, so an analogous stationarity analysis can
   be obtained under the same smoothness/variance assumptions plus bounded
   routing and bounded auxiliary gradients.

2. **Do not claim information-bottleneck optimality.**
   We can claim IB consistency or a rate-distortion motivation.  A full IB
   optimality theorem requires a specific probabilistic model, e.g.,
   linear-Gaussian activations and task-relevance assumptions.

3. **Do not present copy-based inheritance as the only valid source.**
   The key paper principle is **trained-source inheritance**.  The source may be
   a trained teacher, a trained compact baseline, or a copied trained base.  In
   the current code, CIFAR-100 paper-style pairs decompose the trained teacher,
   while the CIFAR-10 `demo_code_org.py` compatibility pair decomposes a trained
   student source.

4. **Do not put engineering cache reuse in the core contribution list.**
   Spectral cache reuse is useful and should appear in implementation details or
   appendix, but it is not the paper's theoretical novelty.

5. **Fix the KD direction.**
   The distillation loss used by PyTorch `kl_div(log_softmax(student),
   softmax(teacher))` corresponds to
   $\mathrm{KL}(p_{\mathrm{teacher}}\|p_{\mathrm{student}})$ up to the usual
   temperature scaling.

6. **Align practical defaults with the current code.**
   The current Hetero defaults are budget ratio `0.35`, min rank `8`, entropy
   temperature `1.4`, routing threshold/compression threshold `12`, calibration
   batches `16`, expert noise scale `0.01`, and balance weight `0.01`.

---

## 3. Proposed Method: HeteroInherNet-IB

HeteroInherNet keeps InherNet's asymmetric inheritance backbone but changes
rank selection from a uniform hyperparameter to a data-aware layer-wise budget.

### 3.1 Trained-Source Inheritance

The decomposed weights must already encode task knowledge:
$$
\theta_{\mathrm{source}}\in
\{\theta_{\mathrm{teacher}},\theta_{\mathrm{trained\ compact}},
\theta_{\mathrm{copied\ base}}\}.
$$
The paper should describe this as a design principle:

> HeteroInherNet decomposes a trained source network, not random weights.

The exact source can be dataset/protocol dependent.  This avoids forcing the
CIFAR-10 legacy workflow and the CIFAR-100 paper-style workflow into one
incorrect statement.

### 3.2 Data-Weighted Spectral Analysis

Uniform SVD minimizes Frobenius error in weight space.  Hetero instead estimates
an activation covariance from a calibration set and analyzes the data-weighted
operator.

For a linear layer, or for a convolutional layer under a channel-covariance
approximation:
$$
\Sigma_x = \mathbb{E}[\phi(X)\phi(X)^\top] + \epsilon I,\qquad
\Sigma_x = CC^\top,
$$
and
$$
\widetilde W = W C.
$$
The SVD of $\widetilde W$ minimizes the data-weighted reconstruction error
$$
\|W-\widehat W\|_{\Sigma_x}^2
= \mathrm{tr}\left((W-\widehat W)\Sigma_x(W-\widehat W)^\top\right).
$$
For CNNs, the implementation uses a tractable channel-wise covariance rather
than the full im2col covariance.  The paper can present the full linear theorem
and state the channel version as the practical convolutional approximation.

### 3.3 Spectral-Entropy Rank Allocation

For layer $l$, let $\{\tilde\sigma_{l,i}\}$ be singular values of the
data-weighted operator.  Define
$$
p_{l,i}=\frac{\tilde\sigma_{l,i}^2}
{\sum_j \tilde\sigma_{l,j}^2},
\qquad
H_l=-\sum_i p_{l,i}\log p_{l,i}.
$$
This is a von-Neumann-style spectral entropy of the normalized Gram spectrum.
It is better to call it a **spectral entropy proxy for effective rank**, not a
direct measure of mutual information.

Given total rank budget $B$, floor $r_{\min}$, and temperature $\tau$:
$$
r_l = r_{\min}
 + \operatorname{round}\left(
\frac{H_l^{1/\tau}}{\sum_j H_j^{1/\tau}}
\bigl(B-Lr_{\min}\bigr)
\right),
$$
followed by clipping and integer budget correction.

The defensible theory angle is:

- high entropy means the layer has diffuse spectral energy and needs more rank;
- low entropy means spectral energy is concentrated and can tolerate stronger
  compression;
- temperature smoothing prevents a brittle winner-take-most allocation.

### 3.4 Decoupled Routing

If a layer receives a small rank, routing on the compressed representation may
be too weak.  Hetero therefore uses a routing threshold:

- use compressed features when $r_l\ge r_{\mathrm{gate}}$;
- use uncompressed pooled features when $r_l<r_{\mathrm{gate}}$.

In the current code this threshold is `compress_threshold`, default `12`.
The paper should call this a stability device, not the central theoretical
contribution.

### 3.5 Expert De-Symmetrization

Identical experts can create symmetric gradients for the router.  Hetero uses
zero-mean expert perturbations:
$$
W_h^{\mathrm{up}}
= U_r\Sigma_r^{1/2}+\epsilon_h,\qquad
\sum_{h=1}^{H}\epsilon_h=0.
$$
Zero-mean noise preserves the average reconstruction while breaking early
expert symmetry.  Avoid a fixed $1/H$ factor unless the aggregation formula is
also changed accordingly; with softmax gates that sum to one, identical full
experts already reconstruct the rank-$r$ map before perturbation.

### 3.6 Load-Balance Regularization

For mean routing probabilities $\bar G_h$ in a mini-batch:
$$
\mathcal{L}_{\mathrm{bal}}
= H\sum_{h=1}^{H}\bar G_h^2.
$$
This is minimized by uniform expert usage and discourages early expert collapse.
It should be positioned as an MoE-inspired training stabilizer.

---

## 4. Method Comparison

| Component | InherNet | HeteroInherNet-IB |
|---|---|---|
| Source | Trained teacher/source weights | Trained source weights; teacher or trained compact base depending on protocol |
| Decomposition target | Raw weight $W$ | Data-weighted operator $\widetilde W = WC$ |
| Rank policy | Uniform rank | Layer-wise entropy-budgeted rank |
| Entropy | Not used | Spectral entropy / effective-rank proxy |
| Routing | Standard compressed-feature gate | Decoupled routing when rank is too small |
| Experts | Asymmetric one-down-many-ups | Same topology + zero-mean de-symmetrization |
| Collapse control | Mostly implicit | Explicit load-balance penalty |
| Theory emphasis | SVD inheritance and rank/head effects | Data-weighted approximation, budget allocation, routing stability |

---

## 5. Theory Plan for a Strong Paper

The paper should avoid presenting all theory as already proven unless the
appendix contains complete derivations.  A strong theory section can be built
around four defensible claims.

### 5.1 Data-Weighted Eckart-Young-Mirsky Theorem

For $\Sigma_x\succ0$, truncated SVD of $\widetilde W=WC$ is optimal for
$$
\min_{\mathrm{rank}(\widehat W)\le r}
\mathrm{tr}\left((W-\widehat W)\Sigma_x(W-\widehat W)^\top\right).
$$
This is a clean theorem and should be central.

### 5.2 Entropy-Budget Allocation Lemma

A safer derivation than the earlier draft is to define an explicit concave
utility:
$$
\max_{r_l\ge r_{\min}}
\sum_l H_l^{1/\tau}\log(r_l-r_{\min}+\epsilon)
\quad
\text{s.t.}\quad
\sum_l r_l=B.
$$
The KKT solution allocates the extra budget in proportion to
$H_l^{1/\tau}$, matching the algorithm before integer correction.  This is
defensible because it says Hetero is optimal for a stated entropy-weighted
utility, not for an unsupported approximation-error model.

### 5.3 Convergence Compatibility Lemma

State that if:

- the original InherNet smoothness/variance assumptions hold;
- routing features are bounded;
- expert noise has bounded variance;
- the load-balance gradient is bounded;

then the same stationarity rate order follows with modified constants.  Do not
state that Hetero strictly improves the asymptotic rate.

### 5.4 Linear-Gaussian IB Interpretation

Under a linear-Gaussian approximation, the data-weighted spectrum controls how
much input variance passes through the bottleneck.  Heterogeneous ranks then
implement a rate-distortion-style allocation: more rank where predictive
variance is diffuse, less rank where predictive variance is concentrated.

This is a much more credible IB story than claiming exact IB optimality for
deep nonlinear networks.

---

## 6. Narrative and Storyline for a Top-Tier AI Paper

Top-tier AI introductions usually work because they make one problem feel
inevitable, then make the proposed method feel like the simplest principled
answer.  For this paper, the story should not be "we added many tricks to
InherNet."  The story should be:

> Neural network inheritance is a promising alternative to student design and
> distillation, but current inheritance uses a uniform bottleneck across layers.
> This is misaligned with how information is distributed in real networks.
> HeteroInherNet makes the bottleneck data-aware.

Recommended introduction arc:

1. **Broad motivation.**
   Efficient model compression is still dominated by KD and PEFT.  KD transfers
   behavior but not structure; PEFT adapts a frozen model but does not produce a
   standalone compact inheritor.

2. **InherNet as the starting point.**
   InherNet reframes compression as inheritance: directly factorize a trained
   source network and retain its spectral structure.  This is a stronger
   starting point than training a small student from scratch.

3. **Core gap.**
   Uniform rank assumes every layer should pass through the same bottleneck.
   Empirically and spectrally this is unlikely: different layers have different
   data-conditioned spectra, different effective ranks, and different
   sensitivity to compression.

4. **Central idea.**
   Replace a uniform architectural bottleneck with an information-aware
   heterogeneous bottleneck.  The method estimates data-weighted spectra,
   measures spectral entropy, and allocates rank under a global budget.

5. **Why this is principled.**
   The method is supported by: data-weighted low-rank approximation theory,
   entropy-budget optimization, convergence compatibility with InherNet, and a
   linear-Gaussian IB/rate-distortion interpretation.

6. **Contributions.**
   Keep the contribution list short and defensible:
   - a data-aware heterogeneous inheritance framework;
   - an entropy-budget rank allocation algorithm;
   - routing and expert-utilization stabilizers for low-rank MoE inheritance;
   - theory connecting the method to data-weighted SVD and rate-distortion;
   - experiments showing better accuracy/parameter tradeoffs and diagnostics
     explaining when heterogeneous ranks help.

Suggested opening sentence:

> Neural network inheritance offers a direct route to compact models by
> factorizing trained weights, but existing inheritance methods impose the same
> low-rank bottleneck on every layer, ignoring the fact that task-relevant
> information is distributed unevenly across a network.

Suggested one-sentence method pitch:

> HeteroInherNet turns inheritance from a uniform compression rule into a
> data-aware budget allocation problem.

Suggested reviewer-facing caution:

> We do not claim a universal IB optimum for deep networks; instead, we provide
> a linear/data-weighted analysis that explains the allocation rule and matches
> the observed layer-wise behavior.

This framing matches common successful ML-paper practice: state the problem,
show why it matters, identify a narrow gap, present a simple principle, then
support it with theory and experiments.

---

## 7. Practical Configuration to Report

Method defaults:

- Budget ratio: `0.35`
- Minimum rank floor: `8`
- Entropy temperature: `1.4`
- Routing/compression threshold: `12`
- Calibration batches: `16`
- Expert heads: default `3`; ablate `1,2,3`
- Expert noise scale: `0.01`
- Balance loss weight: `0.01`
- Linear-layer compression: off by default in current CIFAR experiments

Training protocol should be reported separately by dataset:

- CIFAR-10 original-compatible workflow: Adam, LR `1e-3`, batch `256`,
  100 epochs, trained student source, supervised compressed-model training.
- CIFAR-100 paper-style workflow: SGD, LR `0.05`, momentum `0.9`, weight decay
  `5e-4`, batch `64`, 240 epochs, trained teacher source, KD training.

---

## 8. Final Positioning

The cleanest positioning is:

> HeteroInherNet extends InherNet from uniform low-rank inheritance to
> data-aware heterogeneous inheritance.  Its central claim is not that every
> engineering detail is theoretically optimal, but that the main design choice
> - allocating rank according to data-weighted spectral complexity - is a
> principled correction to uniform inheritance.

This is the strongest narrative for a top-tier submission because it gives
reviewers a single conceptual hook and a clear theory/experiment checklist.
