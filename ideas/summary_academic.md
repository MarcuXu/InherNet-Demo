# Methods Draft (Ready-to-Paste): From InherNet to **HeteroInherNet-IB**

> This section summarizes the original InherNet in *Beyond Student: An Asymmetric Network for Neural Network Inheritance* (arXiv:2602.09509), and presents our latest revised heterogeneous variant.  
> We adopt the naming style common in top AI venues and denote the revised method as **HeteroInherNet-IB** (Information-Bottleneck-Driven Heterogeneous Neural Network Inheritance).

---

## 1. Preliminaries and Notation

Let a pretrained teacher layer weight be $W \in \mathbb{R}^{m \times n}$ with SVD
$$
W = U\Sigma V^\top.
$$
For truncated rank $r$,
$$
W_r = U_r\Sigma_r V_r^\top,
$$
and by Eckart–Young–Mirsky,
$$
\|W - W_r\|_F^2 = \sum_{i=r+1}^{\min(m,n)} \sigma_i^2(W).
$$

For convolution kernels $K \in \mathbb{R}^{N \times c \times k_w \times k_h}$, channel decomposition reshapes $K$ into $\hat K \in \mathbb{R}^{N \times (c k_w k_h)}$ before SVD.

---

## 2. Original InherNet (Uniform NNI)

### 2.1 Knowledge inheritance via low-rank factorization

InherNet approximates full pretrained teacher weights (not LoRA-style updates), and initializes low-rank factors from top singular components:
$$
W \approx U_r\Sigma_rV_r^\top.
$$
The factorized module uses two projections with asymmetric expert reconstruction.

### 2.2 Structure inheritance via one-down-many-ups

Given input $X$, InherNet defines
$$
Y = \sum_{h=1}^{H} G_h(X)\, W_h^{\text{up}}\big(W^{\text{down}}(X)\big),
$$
where $H$ is expert count, and gating is
$$
G(X) = \text{softmax}(W_g(X)).
$$

The paper initializes (asymmetric design):
$$
W^{\text{down}} \leftarrow U_r\Sigma_r^{1/2},
\qquad
W_h^{\text{up}} \leftarrow \frac{1}{H}\Sigma_r^{1/2}V_r^\top.
$$

### 2.3 Core theoretical insights from the original paper

- **Convergence**: under standard non-convex SGD assumptions (Lipschitz gradient, bounded variance), InherNet achieves $\mathcal{O}(1/T)$ stationarity rate.
- **Conditioning benefit**: orthonormal SVD initialization improves optimization conditioning (effective smoother landscape).
- **Efficiency–expressivity tradeoff**: parameter reduction bounds are established for low-rank + multi-head parameterization; representational power is argued to be preserved with proper rank/head selection.
- **Empirical insight in the paper**: rank is primary for inheritance quality; multiple heads help but with diminishing returns.

---

## 3. Revised Method: **HeteroInherNet-IB**

We retain the InherNet asymmetric inheritance backbone but replace static, heuristic rank design with data-aware heterogeneous allocation and optimization-stable training.

### 3.1 Inheritance source correction (Issue 1)

A critical correction is to avoid decomposing random weights. In our final setup we choose **copy-based inheritance source**:
- train baseline inherited model first,
- copy trained backbone weights to the heterogeneous base,
- then perform heterogeneous decomposition.

Formally:
$$
\theta_{\text{hetero-base}} \leftarrow \theta_{\text{trained baseline}}.
$$

This restores the fundamental inheritance assumption: decomposition should act on meaningful pretrained spectra.

### 3.2 Activation whitening before decomposition

Instead of decomposing $W$ directly, compute activation covariance from a calibration set:
$$
\Sigma_x = \frac{1}{N}XX^\top = CC^\top,
$$
and whitened weights
$$
\tilde W = W C.
$$
SVD is applied to $\tilde W$, improving data alignment of spectral structure.

### 3.3 Spectral-entropy-driven heterogeneous rank allocation

For each layer $l$ with singular values of $\tilde W_l$, we use squared-spectrum distribution:
$$
p_{l,i} = \frac{\sigma_{l,i}^2}{\sum_j \sigma_{l,j}^2},
\qquad
\mathcal{H}_l = -\sum_i p_{l,i}\log p_{l,i}.
$$

Given global budget $\mathcal{B}$, minimum floor $r_{\min\_floor}$, and temperature $\tau$:
$$
r_l = r_{\min\_floor} +
\frac{\mathcal{H}_l^{1/\tau}}{\sum_j \mathcal{H}_j^{1/\tau}}
\big(\mathcal{B} - L\,r_{\min\_floor}\big),
$$
followed by integer/budget correction and clipping by layer-wise max rank.

This resolves two weaknesses of uniform NNI: (i) non-adaptive rank usage, (ii) rank-collapse risk.

### 3.4 Decoupled gating with routing floor

For each layer with assigned rank $r_l$, gating input is selected as:
- compressed representation if $r_l \ge r_{\text{route}}$,
- uncompressed representation if $r_l < r_{\text{route}}$.

This decoupled policy prevents routing degeneration in aggressively compressed layers.

### 3.5 Asymmetric de-symmetrized expert initialization

We follow a balanced split with $\Sigma^{1/2}$ and inject small perturbation to break expert symmetry:
$$
W^{\text{down}} \sim \Sigma_r^{1/2}V_r^\top C^{-1},
\qquad
W_h^{\text{up}} \sim \frac{1}{H}U_r\Sigma_r^{1/2} + \epsilon_h,
$$
where $\epsilon_h$ is small zero-mean noise.

This improves early expert specialization and avoids gate-gradient ambiguity under identical experts.

### 3.6 Load-balancing auxiliary objective

To reduce expert collapse, we add a lightweight balancing regularizer:
$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda\,\mathcal{L}_{\text{balance}},
$$
with $\lambda=0.01$ in our implementation.

### 3.7 Engineering refinement: spectral cache reuse

Cholesky and SVD are cached in the entropy stage and reused in decomposition construction, avoiding repeated high-cost factorizations and improving decomposition efficiency.

---

## 4. Method Comparison

| Component | Original InherNet (Uniform NNI) | Revised HeteroInherNet-IB |
|---|---|---|
| Inheritance source | Pretrained teacher/baseline decomposition | **Copy-based trained-source decomposition** (chosen) |
| Decomposition target | Raw weight $W$ | Whitened weight $\tilde W = WC$ |
| Rank policy | Uniform fixed $r$ across layers | Layer-wise adaptive $r_l$ via spectral entropy + budget |
| Entropy definition | Not central / static-rank design | Correct von Neumann-style ($\sigma^2$) entropy |
| Rank safety | No explicit global floor in classic setup | Min-rank floor + temperature smoothing |
| Gating input | Coupled to compressed branch | Decoupled with routing floor |
| Expert initialization | Asymmetric base, often near-symmetric experts | Asymmetric + deliberate de-symmetrization noise |
| Expert utilization control | Implicit | Explicit load-balance auxiliary term |
| Decomposition efficiency | Potential repeated SVD/Cholesky | Cached reuse of spectral factors |

---

## 5. Why the Revised Method is Theoretically Reasonable

### 5.1 Alignment with original InherNet theory

The original paper emphasizes that rank controls inheritance quality and SVD initialization stabilizes optimization. HeteroInherNet-IB does not contradict this; instead, it **strengthens** it by:
1. ensuring decomposition acts on trained weights (`copy` path),
2. allocating rank where spectral information demands it,
3. preserving asymmetric one-down-many-ups design validated in the original paper and appendices.

### 5.2 Information-bottleneck consistency

The original appendix argues one-down-many-ups is superior via IB and credit-assignment analysis. Our revised method preserves this topology and adds data-dependent rank control + routing safeguards, which is consistent with the same IB motivation: retain sufficient predictive information while controlling compression.

### 5.3 Optimization and variance considerations

The original analysis highlights conditioning and gradient-routing effects. The revised method further reduces optimization pathologies via:
- non-collapsing rank floor,
- expert de-symmetrization,
- auxiliary load balancing.

Hence, improvements are not heuristic-only; they are coherent with the paper’s convergence and specialization arguments.

---

## 6. Practical Configuration Used in the Latest Revision

- Inheritance source mode: **`copy`** (selected)
- Rank budget ratio: `0.35`
- Minimum rank floor: `8`
- Entropy temperature: `1.5`
- Routing floor `r_route`: `4`
- Expert heads tested: `H \in {1,2,3}`
- Auxiliary load-balance weight: `0.01`

---

## 7. Suggested Paper-Style Naming and Positioning

We recommend naming the revised method:

**HeteroInherNet-IB: Information-Bottleneck-Driven Heterogeneous Neural Network Inheritance**

This name is concise, method-informative, and stylistically consistent with top conference naming conventions (core mechanism + theoretical lens).

---

## 8. Concluding Statement for Methods Section

In summary, HeteroInherNet-IB extends InherNet from *uniform* low-rank inheritance to *data-aware heterogeneous* inheritance. By combining activation-whitened spectral analysis, entropy-budgeted rank allocation, decoupled routing, and stability-oriented training refinements, the method preserves InherNet’s asymmetric inheritance principle while improving robustness, efficiency, and empirical competitiveness under comparable parameter budgets.
