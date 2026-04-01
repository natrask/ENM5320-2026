# Lecture 18: Attention Mechanisms and Physics-Inspired Architectures

**Reference:** Lecture 18 - Transformers, Attention, and Hamiltonian Neural Networks

**Topics Covered:**
- Conditional neural fields and cross-attention
- Multi-head attention mechanisms
- Dropout regularization
- Graph attention networks (GATs)
- Physics-inspired architectures (GRAND)
- Hamiltonian neural networks
- Double-bracket dynamics

---

## 0. Overview

This lecture bridges classical attention mechanisms from natural language processing with physics-inspired neural network architectures. Having developed graph calculus and the Hodge decomposition in previous lectures, we now see how these mathematical structures naturally arise in modern deep learning architectures—particularly in **transformers** and **graph neural networks**.

The key insight is that attention can be viewed as a **soft dictionary lookup** or, equivalently, as a data-driven basis function similar to finite element methods. When we introduced finite elements, we saw that functions could be represented as $f(x) = \sum_i \alpha_i(x, x_i) y_i$ where $\alpha_i$ are piecewise linear basis functions. Attention generalizes this by learning the weights $\alpha_i(q, \vec{k})$ from data rather than fixing them geometrically.

We begin with **conditional neural fields**, where a latent code $z$ modulates input-output relationships—crucial for operator learning in PDEs. We then dissect the **transformer architecture**: queries, keys, values, multi-head attention, and dropout. Moving to graphs, we examine **Graph Attention Networks (GATs)**, which suffer from over-squashing and over-smoothing problems. Physics-inspired solutions emerge from recognizing that GATs implicitly define graph Laplacians.

Finally, we introduce **Hamiltonian neural networks** and **double-bracket dynamics**, showing how energy-preserving and energy-dissipating systems can be built into neural architectures. These connections reveal deep relationships between optimization, dynamical systems, and modern machine learning.

## 1. Conditional Neural Fields

### 1.1 Problem Setup

Consider the **conditional neural field**:

$$
y = f(x \mid z; \theta)
$$

**Idea:** The latent code $z$ **modulates** the $x \to y$ input-output relationship.

**Examples:**
- $y$ = finite difference stencil
- $x$ = grid function
- $z$ = material properties / microscopic structure

### 1.2 Architecture Options

Many options for implementing this:
1. **DeepONet** (Lu et al.)
2. **PCA/FNO** (Stewart, Li et al.)
3. **Cross-attention** (⭐ focus of this lecture)

**Pedagogical note:** Cross-attention provides the most flexible framework, allowing the model to dynamically select relevant features from the conditioning variable $z$ based on the query location $x$.

## 2. Attention as Soft Dictionary Lookup

### 2.1 Key-Value Pairs

**Reference:** Murphy §15.4.1

Consider $(k_i, v_i)_{i=1}^N$ as **key-value pairs** describing input-output labels.

Consider a **query** $q$ representing where the model is evaluated.

**Definition:** The **attention function** is:

$$
\text{Attn}(q, \{k_i, v_i\}) = \sum_{i=1}^N \alpha_i(q, \vec{k}) v_i
$$

where **attention weights** satisfy:

$$
\begin{aligned}
0 &\leq \alpha_i \leq 1 \\
\sum_i \alpha_i &= 1
\end{aligned}
$$

### 2.2 Attention Scores and Softmax

Introduce an **attention score** $a(q, k_i)$:

$$
\alpha_i = \text{softmax}(a(q, \vec{k})) = \frac{\exp[a(q, k_i)]}{\sum_j \exp[a(q, k_j)]}
$$

### 2.3 Connection to Finite Elements

**Remark:** This is a **data-driven basis**!

For example, taking $\alpha_i(q, \vec{k}) = \phi_i(q)$ (CPWL basis functions) satisfies the same properties:

$$
f(x) = \sum_i \alpha_i(x, x_n) y_i
$$

where:
- $q \to x$ (query location)
- $k \to x_n$ (nodal positions)
- $v \to y_i$ (nodal values)

**Critical observation:** Attention generalizes finite element interpolation by learning the basis functions rather than fixing them geometrically!

## 3. Multi-Head Attention

### 3.1 Architecture

Introduce **dense MLP blocks** for $i = 1, \ldots, M$ (number of heads):

$$
\begin{aligned}
\hat{q}_i &= Q_i(q) \\
\hat{k}_i &= K_i(k) \\
\hat{v}_i &= V_i(v)
\end{aligned}
$$

Then:

$$
h_i = \text{Attn}(\hat{q}_i, \{\hat{k}_i, \hat{v}_i\})
$$

**Final aggregation:**

$$
h = \text{MHA}(q, \{k, v\}) = \text{Concat}(h_1, \ldots, h_M) W^O
$$

**Interpretation:** Each head learns to attend to different aspects of the input (e.g., position vs. semantic meaning), analogous to using multiple finite element spaces for different solution components.

## 4. Dropout Regularization

### 4.1 Mechanism

**With probability $p$**, at each forward pass the output of a given neuron will be **dropped** (set to zero).

**Implementation:** Replace weights:

$$
\begin{aligned}
\theta_{lji} &= \omega_{lji} \varepsilon_{li} \\
\varepsilon_{li} &\sim \text{Bernoulli}(1-p)
\end{aligned}
$$

**Interpretation:** Dropout forces the network to learn robust features that don't rely on any single neuron, preventing co-adaptation and improving generalization.

## 5. Graph Attention Networks

### 5.1 GAT Algorithm

**Steps:**

**Step 1:** Input features. Each node $n_i \in N$ has feature vector $h_i^n \in \mathbb{R}^F$.

**Step 2:** Linear transform:

$$
h_i' = W h_i^n, \quad W \in \mathbb{R}^{F \times F}
$$

**Step 3:** Pre-attention coefficients:

$$
e_{ij} = \text{LeakyReLU}(\vec{a}^\top [W h_i \| W h_j])
$$

where $\|$ denotes concatenation.

**Step 4:** Attention mechanism (softmax normalization):

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \sim i} \exp(e_{ik})}
$$

**Step 5:** Aggregate:

$$
h^{n+1} = \sigma\left( \sum_{j \sim i} \alpha_{ij} W h_j^n \right)
$$

### 5.2 Problems with GATs

**Important Notes:**
- GATs are **notoriously hard to train deep** due to:
  - **Over-squashing:** Information bottleneck in long-range propagation
  - **Over-smoothing:** Node features become indistinguishable after many layers

**Critical observation:** With physics ideas we'll show how to keep them stable!

## 6. Physics-Inspired Architectures: GRAND

### 6.1 Graph Neural Diffusion (GRAND)

**Key idea:** Rewrite GAT update as a graph diffusion equation:

$$
h_i^{n+1} = h_i^n + \sigma\left( \sum_{j \sim i} \alpha_{ij} (h_j^n - h_i^n) \right)
$$

or in **graph calculus**:

$$
h^{n+1} = h^n + \sigma(\delta^* \alpha \delta h^n)
$$

where:

$$
\langle f, \delta^* g \rangle_\alpha = \langle \delta f, g \rangle_\alpha
$$

**Interpretation:** View attention as inducing a **weighted inner product**.

### 6.2 Formulation as Dynamical System

Writing $h^{n+1} = Q^n h^n$, given graph data $D = \{x_i, y_i\}$:

$$
\text{encoder} \to h^0 \to Q h^0 \to Q h^1 \to \cdots \to h^N \to \text{decoder}
$$

**Critical observation:** This is a discrete dynamical system where the graph Laplacian provides inherent stability through diffusion.

## 7. Hamiltonian Neural Networks

### 7.1 Review: Hamiltonian Mechanics

Recall the **Hamiltonian formulation**:

$$
\frac{d}{dt} \begin{pmatrix} q \\ p \end{pmatrix} = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix} \begin{pmatrix} \partial_q E \\ \partial_p E \end{pmatrix}
$$

where $E = \text{NN}(p, q)$ is the energy (learned by a neural network).

**Key property:** Energy conservation:

$$
\dot{E} = 0
$$

### 7.2 Connection to GRAND

**Same structure as GRAND!**

For graph dynamics:

$$
\frac{d}{dt} \begin{pmatrix} q \\ p \end{pmatrix} = \begin{pmatrix} 0 & -\delta_0^* \\ \delta_0 & 0 \end{pmatrix} \begin{pmatrix} \partial_q E \\ \partial_p E \end{pmatrix}
$$

**Interpretation:** The graph coboundary operators $\delta_0$ and $\delta_0^*$ play the role of symplectic structure, ensuring energy conservation exactly at the discrete level.

### 7.3 Architecture Implications

The **encoder-decoder pipeline**:

$$
x \to h^0 \to Q h^0 \to Q h^1 \to \cdots \to Q h^N \to y
$$

can be viewed as evolving a Hamiltonian system in latent space, where energy conservation provides implicit regularization.

## 8. Double-Bracket Dynamics

### 8.1 Formulation

Consider the dynamics:

$$
\frac{dx}{dt} = J \nabla E - J^\top J \nabla E
$$

### 8.2 Energy Dissipation

Compute the energy rate of change:

$$
\begin{aligned}
\frac{dE}{dt} &= \frac{\partial E}{\partial x} \cdot \frac{dx}{dt} \\
&= \frac{\partial E}{\partial x}^\top J \nabla E - \frac{\partial E}{\partial x}^\top J^\top J \nabla E \\
&= 0 - \|\nabla E\|_{J^\top J}^2 \\
&\leq 0
\end{aligned}
$$

**Key property:** The first term (Hamiltonian part) conserves energy, while the second term (double-bracket part) **dissipates energy**.

### 8.3 Hybrid Architecture

**Interpretation:** By combining Hamiltonian and double-bracket terms, we can design neural networks that:
1. Preserve important conservation laws (via Hamiltonian part)
2. Dissipate to stable equilibria (via double-bracket part)

This provides a principled way to build **stable deep architectures** with physics-inspired inductive biases.

### 8.4 Network Pipeline

$$
x \to \text{encoder} \to h^0 \to Q^0 h^0 \to \cdots \to h^N \to \text{decoder} \to y
$$

where each $Q^k$ implements either:
- Hamiltonian dynamics (energy-preserving)
- Double-bracket dynamics (energy-dissipating)
- A learned combination of both

---

## Summary

This lecture covered:

1. **Conditional neural fields** for operator learning
2. **Attention mechanisms** as soft dictionary lookups
3. **Multi-head attention** for learning diverse feature representations
4. **Dropout** regularization for preventing overfitting
5. **Graph Attention Networks (GATs)** and their limitations
6. **GRAND architecture** using graph diffusion
7. **Hamiltonian neural networks** with energy conservation
8. **Double-bracket dynamics** for controlled energy dissipation

**Key Takeaway:** Attention mechanisms can be viewed as learned finite element basis functions, generalizing interpolation from geometry to data. Graph attention networks implicitly define weighted graph Laplacians, connecting to diffusion equations and Hamiltonian mechanics. By recognizing these connections, we can design physics-inspired architectures that combine the expressiveness of modern deep learning with the stability and interpretability of classical numerical methods. The double-bracket formulation $\dot{x} = J \nabla E - J^\top J \nabla E$ provides a unified framework for energy-preserving and energy-dissipating dynamics, enabling neural networks that respect physical conservation laws while achieving stable training and generalization.
