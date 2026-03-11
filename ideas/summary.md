# Summary of the Methods

## Old Method: Uniform NNI (from the ICLR 2026 Conference Paper)

The original "InherNet" framework introduced a paradigm shift from Knowledge Distillation (implicit learning) to Neural Network Inheritance (explicit capacity transfer). The key mechanics are:

1. **Weight-Only SVD:** It takes the pretrained teacher's weight matrix $W$ and performs truncated SVD directly: $W \approx U_r \Sigma_r V_r^\top$.
2. **Uniform Rank & Experts:** It relies on a manually tuned, static rank ($r$) and a fixed number of expert heads ($H$) applied uniformly across all layers of the network.
3. **Asymmetric MoE Initialization:** It replaces the original layer with a "One-Down-Many-Ups" structure. The shared down-projection $W^{down}$ is initialized as $U_r \Sigma_r^{1/2}$, and the $H$ expert up-projections $W_h^{up}$ are uniformly initialized as $\frac{1}{H}\Sigma_r^{1/2}V_r^\top$.
4. **Coupled Gating:** The adaptive gating network routes tokens based on the *compressed* bottleneck representation: $G(X) = \text{softmax}(W^g(W^{down}(X)))$.

## New Method: Heterogeneous Inheritance via Spectral Entropy (Journal Extension)

The upgraded method replaces the static, heuristic decomposition with a globally optimized, data-aware framework grounded in the Information Bottleneck principle. The key mechanics are:

1. **Activation Whitening:** Instead of decomposing $W$, it computes the input covariance $\Sigma_x = \frac{1}{N} X X^\top$ using a small calibration set. It takes the Cholesky decomposition $\Sigma_x = C C^\top$ and performs SVD on the whitened weights $\tilde{W} = W C$.
2. **Spectral Entropy Rank Allocation:** It dynamically allocates rank per layer. It computes the normalized singular value distribution $\hat{\sigma}_{l,i}$ of $\tilde{W}_l$, calculates the von Neumann spectral entropy $\mathcal{H}_l = - \sum_i \hat{\sigma}_{l,i} \ln(\hat{\sigma}_{l,i})$, and assigns a specific rank $r_l$ proportional to $\mathcal{H}_l$ under a global parameter budget $\mathcal{B}$.
3. **Decoupled Gating (Rank Floor $r_{min}$):** To prevent routing collapse in highly redundant layers assigned very low ranks (e.g., $r < 4$), it introduces a routing rank floor $r_{min}$. If $r_l < r_{min}$, the gating network calculates probabilities using the *uncompressed* input $X$ directly: $G(X) = \text{softmax}(W^g(X))$.
