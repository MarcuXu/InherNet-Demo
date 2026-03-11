# InherNet vs. Revised Hetero-InherNet (Latest Summary)

> Date: 2026-03-11  
> Context: Based on `revision_0311_0843.md`, `summary.md`, and the latest implemented code path where **Issue 1 uses `copy`**.

---

## 1) Original Method: InherNet (Uniform NNI)

Original InherNet is an inheritance-style compression and transfer method built on fixed low-rank decomposition and multi-expert reconstruction.

### Core mechanism

1. **Weight-only truncated SVD**  
   For each target layer weight matrix $W$, apply truncated SVD:
   $$W \approx U_r \Sigma_r V_r^\top$$

2. **Uniform rank and uniform expert count across layers**  
   All layers share the same manually chosen rank $r$ and same number of experts $H$.

3. **One-down-many-ups factorized layer**  
   Replace original layer with a bottleneck branch and multiple expert branches:
   - down projection (shared)
   - up projection (expert-specific, but commonly symmetric at init)

4. **Coupled gating**  
   Gate is computed from compressed features:
   $$G(X) = \mathrm{softmax}(W^g(W^{down}(X)))$$

### Practical implication

- Strong baseline and effective knowledge inheritance when decomposition is applied to trained weights.
- But rank allocation is static and layer-agnostic, so it cannot adapt to heterogeneous layer complexity.

---

## 2) Latest Revised Method: Hetero-InherNet (Revised)

The revised hetero method keeps the inheritance principle, but replaces heuristic uniform decomposition by data-aware, layer-adaptive, and engineering-stable mechanisms.

## 2.1 Inheritance source (Issue 1, chosen mode = `copy`)

You selected **`copy`** for Issue 1. This means:

- First train the baseline ResNet-18 path (`model`) as before.
- Then initialize hetero base model by copying trained weights:
  $$\theta_{\text{hetero-base}} \leftarrow \theta_{\text{trained-baseline}}$$

This preserves the inheritance assumption and avoids decomposing random weights.

## 2.2 Activation whitening before SVD

Instead of decomposing raw $W$, revised hetero estimates activation covariance from a calibration set:
$$\Sigma_x = \frac{1}{N}XX^\top,\quad \Sigma_x = CC^\top$$
then performs SVD on whitened weight:
$$\tilde{W}=WC$$

This makes spectral analysis and rank decision more aligned with actual data distribution.

## 2.3 Corrected spectral entropy for rank allocation

Entropy is computed from squared singular values (von Neumann style):
$$p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2},\quad
\mathcal{H}_l=-\sum_i p_i\ln p_i$$

(Previously, using $\sigma_i$ directly made layer entropy contrast too weak.)

## 2.4 Budgeted heterogeneous rank allocation with safety floor + temperature

Ranks are allocated by entropy under global budget, with:
- **minimum rank floor** (`min_rank`, default 8) to avoid rank-1 collapse
- **temperature smoothing** (`temperature`, default 1.5) to avoid extreme allocations

This provides a robust compromise between adaptivity and stability.

## 2.5 Decoupled gating with routing floor

For each layer rank $r_l$:
- if $r_l \ge r_{min}$, gate uses compressed feature
- if $r_l < r_{min}$, gate uses uncompressed feature

This avoids routing collapse in aggressively compressed layers.

## 2.6 Improved factor initialization (asymmetric + de-symmetrized experts)

Revised hetero applies $\Sigma^{1/2}$ split and expert perturbation:
- down path uses $\Sigma^{1/2}$-scaled basis (with whitening inverse mapping)
- each expert up projection gets small noise around base initialization

Result: experts are no longer perfectly symmetric at start, so specialization can emerge earlier.

## 2.7 Engineering upgrades

- **SVD cache reuse**: avoids repeated Cholesky/SVD in decomposition pipeline.
- **MoE load-balance auxiliary loss** (lightweight):
  $$\mathcal{L}=\mathcal{L}_{task}+\lambda\,\mathcal{L}_{balance}$$
  with default $\lambda=0.01$ for hetero training/distillation.

---

## 3) What changed from old to new (concise comparison)

| Aspect | Original InherNet | Revised Hetero-InherNet |
|---|---|---|
| Inheritance base | Trained model decomposition | **Trained model decomposition (via `copy`)** |
| Decomposition space | Raw weight $W$ | Whitened weight $\tilde{W}=WC$ |
| Rank policy | Uniform fixed $r$ | Layer-wise adaptive $r_l$ (entropy + budget) |
| Entropy definition | Not used / heuristic | Corrected $\sigma^2$ entropy |
| Rank safety | None | `min_rank` + temperature smoothing |
| Gating | Coupled compressed gating | Decoupled gating with `r_min` fallback |
| Expert init | Often symmetric | $\Sigma^{1/2}$ split + noise de-symmetrization |
| Training stabilization | Standard CE / KD | + load-balance auxiliary term |
| Efficiency | Repeated SVD possible | Cached SVD/Cholesky reuse |

---

## 4) Why the new method is reasonable

1. **Theoretical consistency with inheritance**  
   Using `copy` ensures decomposition starts from meaningful learned weights, not random matrices. This is the most critical correction for InherNet-style transfer.

2. **Information-aware compression**  
   Whitening + entropy-based rank budgeting approximates per-layer information complexity, so capacity is assigned where it matters most.

3. **Optimization stability under heterogeneity**  
   `min_rank`, temperature smoothing, and decoupled gating reduce pathological collapse (too low rank or routing degeneration).

4. **MoE specialization dynamics are improved**  
   Breaking expert symmetry and adding load-balance pressure encourage real expert differentiation rather than redundant heads.

5. **Engineering correctness and scalability**  
   Caching expensive linear algebra and keeping configurable knobs (`copy`/`pretrain`, `min_rank`, `temperature`, `aux_loss_weight`) make the method reproducible and extensible.

---

## 5) Final takeaway

Compared with original InherNet, the revised hetero method is no longer just “uniform low-rank with different labels”; it is now a **data-aware, budget-constrained, routing-stable inheritance framework**.  
Given your current selection (**Issue 1 = `copy`**), the revised pipeline is both theoretically aligned and practically reasonable, which explains the observed accuracy improvement.
