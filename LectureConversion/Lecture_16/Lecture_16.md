# Lecture 16: Graph Calculus and Spectral Graph Theory

**Reference:** Lecture 16 - Graph Laplacian and Graph Exterior Calculus

**Topics Covered:**
- Graphs as discrete manifolds: chains and cochains
- Graph gradient, divergence, and Laplacian
- Stability analysis via Rayleigh quotients
- Spectral graph theory and the Fiedler eigenvalue
- Graph exterior calculus and higher-order operators
- Applications to circuits, GATs, and recommender systems

---

## 0. Overview

This lecture bridges the continuous world of PDEs and the discrete world of graphs, revealing that many concepts from differential geometry have natural analogs on networks. Just as we studied finite element methods on meshes, we now develop **graph calculus**—a framework where nodes, edges, and higher-dimensional simplices replace points, curves, and surfaces.

The key insight is that graphs provide a natural discretization of manifolds, and we can define discrete versions of fundamental operators: **gradient** (measuring differences along edges), **divergence** (measuring net flow at nodes), and **curl** (measuring circulation around triangles). These operators satisfy the same algebraic identities as their continuous counterparts: $\text{curl} \circ \text{grad} = 0$ and $\text{div} \circ \text{curl} = 0$.

We begin by defining graphs as collections of **chains** (geometric objects) and **cochains** (functions on those objects), paralleling the continuous distinction between manifolds and differential forms. The **graph Laplacian** emerges as the composition $-\delta^* \delta$, where $\delta$ is the coboundary operator and $\delta^*$ is its adjoint. This operator plays the same role as the continuous Laplacian in PDE theory.

Using **spectral graph theory**, we analyze the eigenvalues of the Laplacian to understand stability and coercivity. The smallest nonzero eigenvalue—the **Fiedler eigenvalue** $\lambda_1$—controls the coercivity constant in Lax-Milgram, and we derive diameter-dependent bounds that explain how graph geometry affects numerical stability. Applications to circuit analysis, graph attention networks, and recommender systems demonstrate the practical relevance of this theory.

## 1. Graphs as Discrete Manifolds

### 1.1 Basic Definitions

Define $G(N, E)$ as a **graph** with nodes and directed edges:
- $n_i \in N$ is a node
- $e_{ij} \in E$ is an edge pointing from $j$ to $i$

We refer to $N, E$ as **chains**:
- $N$ → **0-chains** (vertices)
- $E$ → **1-chains** (oriented edges)

**Definition:** A **$k$-chain** is an oriented set of $k+1$ vertices.

### 1.2 Boundary Operators

The **boundary operator** $\partial_k : C^{k+1} \to C^k$ maps chains to their boundaries.

**Example:** For an edge $e_{ij}$:

$$
\partial_0 e_{ij} = n_j - n_i
$$

**Interpretation:** The boundary of an edge is the difference between its endpoint and startpoint.

This forms a chain complex:

$$
C^0 \xleftarrow{\partial_0} C^1
$$

## 2. Cochains and Graph Functions

### 2.1 Cochains

We can assign real numbers to chains to develop **graph functions** (think of grid functions in FDM as functions on 0-chains).

**Definition:** $f \in C_k = \mathbb{R}^{\dim(C^k)}$ is a **cochain** (note subscript).

**Note:** In mathematics, "co-" often means a real number associated with an object.

For edge functions: $f_{ij} \in C_1$ with the property $f_{ij} = -f_{ji}$ (anti-symmetry).

### 2.2 Coboundary Operator (Graph Gradient)

**Definition:** The **coboundary operator** is a mapping:

$$
d_k : C_k \to C_{k+1}
$$

**Example:** Let $\phi \in C_0$ (a node function). The **graph gradient** is:

$$
d_0 \phi_{ij} = \phi_j - \phi_i
$$

**Why "gradient"?** In traditional calculus, by the fundamental theorem of calculus:

$$
\int_{e_{ij}} \nabla u \cdot dl = u_j - u_i
$$

The graph gradient $d_0$ is the discrete analog!

## 3. Codifferential (Graph Divergence)

### 3.1 Definition

**Definition:** The **codifferential** is the adjoint of the coboundary operator.

Note: $\delta_k : \mathbb{R}^{\dim(C_k)} \to \mathbb{R}^{\dim(C_{k+1})}$

$$
\Rightarrow \delta_k \in \mathbb{R}^{\dim(C_{k+1}) \times \dim(C_k)}
$$

For $v \in C_k$, $u \in C_{k+1}$:

$$
\langle v, \delta_k^* u \rangle = \langle \delta_k v, u \rangle
$$

### 3.2 Explicit Computation

**Step 1:** Expand the right-hand side:

$$
\begin{aligned}
\langle \delta_k v, u \rangle &= \sum_{i,j,k} \delta_{ij,k} v_k u_{ij} \\
&= \sum_k v_k \left( \sum_{i,j} \delta_{ij,k} u_{ij} \right)
\end{aligned}
$$

**Step 2:** Identify:

$$
(\delta^* u)_k = \sum_{i,j} \delta_{ij,k} u_{ij}
$$

**Step 3:** Therefore:

$$
\langle \delta_k v, u \rangle = \sum_k v_k \delta^* u_k = \langle v, \delta^* u \rangle
$$

### 3.3 Weighted Inner Product

Alternatively, we can define $\delta^*$ with respect to a **weighted inner product**:

$$
\langle v, \delta^* u \rangle_{w_k} = \langle \delta v, u \rangle_{w_{k+1}}
$$

$$
\Rightarrow \delta^* u = w_k^{-1} \delta^\top w_{k+1} u
$$

## 4. Applications of Graph Calculus

### 4.1 Circuit Analysis

**Ohm's Law:**

$$
I_{ij} = R_{ij}^{-1} (V_j - V_i)
$$

**Kirchhoff's Law:** At each node, the net current is zero:

$$
\sum_{j \sim i} I_{ij} = 0
$$

**Recast in graph language:**

$$
\begin{aligned}
I &= R^{-1} \delta_0 V \\
\delta_0^* I &= 0
\end{aligned}
$$

**Step 1:** Substitute $I$:

$$
\delta_0^* R^{-1} \delta_0 V = 0
$$

**This is the graph Laplacian!**

### 4.2 Graph Attention Networks

For a **GAT layer**:

$$
\begin{aligned}
m_{ij}^n &= F(x_i^n, x_j^n \mid \theta_m) \\
x^{n+1} &= x^n + G\left( \sum_{j \sim i} \alpha_{ij}(x_i^n, x_j^n) m_{ij}^n \right)
\end{aligned}
$$

can be rewritten:

$$
x^{n+1} = x^n + G(\delta_0^* m)
$$

where the **attention** induces the codifferential:

$$
\delta_m^* = \delta_0^\top \alpha
$$

**Important Notes:**
- GATs are notoriously hard to train deep (over-squashing/over-smoothing problems)
- With physics ideas we'll show how to keep them stable

### 4.3 Recommender Systems

Given $n$ ratings of other movies, predict what someone will like:
- **0-cochain** → user scores
- **1-cochain** → movie preferences

Similar graph analytics apply to social networks and epidemiology.

## 5. The Weighted Graph Laplacian

### 5.1 Definition

Let $W = \text{diag}(w_1, \ldots, w_{N_{\text{edges}}}) > 0$ be edge weights.

**Definition:** The **weighted graph Laplacian** is:

$$
L_W[u]_i = \sum_{j \sim i} w_{ij}(u_j - u_i)
$$

**Example:** $w_{ij} = R_{ij}^{-1}$ for circuit analysis.

Consider the problem:

$$
L_W[u]_i = f_i
$$

### 5.2 Variational Formulation

We can mimic the Galerkin procedure, using the $\ell^2$-inner product:

$$
\langle f, g \rangle = \sum_i f_i g_i
$$

instead of integration.

**Step 1:** For all $v \in C_0$:

$$
\langle v, L_W[u]_i \rangle = \langle v, f \rangle
$$

**Step 2:** Expand the left-hand side:

$$
\text{LHS} = \sum_{i,j} v_i w_{ij}(u_j - u_i)
$$

**Step 3:** Symmetrize:

$$
= \frac{1}{2} \sum_{i,j} v_i w_{ij}(u_j - u_i) + v_j w_{ji}(u_i - u_j)
$$

**Step 4:** Note $w_{ij} = w_{ji}$ (symmetric weights):

$$
= -\frac{1}{2} \sum_{i,j} (v_j - v_i) w_{ij}(u_j - u_i)
$$

**Step 5:** In matrix form:

$$
= -\frac{1}{2}(\delta v)^\top W \delta u = a(u, v)
$$

### 5.3 Bilinear Form

Like before, we have identified a **variational problem**:

$$
a(u, v) = \langle f, v \rangle
$$

with **energy norm**:

$$
\|u\|_E = \sqrt{a(u, u)} = \|u\|_{\delta^\top W \delta}
$$

**Critical observation:** We have all of our FEM machinery! To invoke Lax-Milgram, we need to prove coercivity of the energy norm:

$$
\|u\|_E^2 \geq \alpha \langle u, u \rangle
$$

## 6. Spectral Graph Theory

### 6.1 Rayleigh Quotients

Let $M = M^\top$ with orthogonal eigenpairs $(\lambda_i, v_i)$:

$$
v_i^\top v_j = \delta_{ij}
$$

So $M v_i = \lambda_i v_i$ for the $i$-th eigenpair:

$$
v_i^\top M v_i = \lambda_i v_i^\top v_i = \lambda_i
$$

**Definition:** The **Rayleigh quotient** is:

$$
R(M, x) = \frac{x^\top M x}{x^\top x}
$$

**Fact:** Symmetric matrices have real, non-negative eigenvalues (when $M \geq 0$).

### 6.2 Eigen-Expansion Analysis

Expanding $x$ in the eigenbasis:

$$
\vec{x} = \sum_i \hat{y}_i \vec{v}_i \quad \text{where} \quad \hat{y}_i = \langle v_i, x \rangle
$$

**Interpretation:** The basis expansion coefficients come from projecting $x$ onto the $v_i$'s.

Then:

$$
R(M, x) = \frac{\sum_i \lambda_i y_i^2}{\sum_j y_j^2}
$$

We can see that **max/min of $R$ gives max/min eigenvalues**:

$$
\lambda_{\min} \leq \lambda_{\min} \frac{\sum y_i^2}{\sum y_j^2} \leq \frac{\sum_i \lambda_i y_i^2}{\sum_j y_j^2} \leq \lambda_{\max} \frac{\sum y_j^2}{\sum y_j^2} = \lambda_{\max}
$$

### 6.3 Degree-Scaled Laplacian

Let $d_v$ denote the **degree** of vertex $v$ (number of edges).

Following **Spectral Graph Theory** (Fan Chung), denote:

$$
T = \text{diag}(d_v)_{v=1}^{N_{\text{nodes}}}
$$

**Definition:** The **degree-scaled Laplacian** is:

$$
\mathcal{L}_{ij} = \begin{cases}
1 & i = j, \, d_v \neq 0 \\
-\frac{1}{\sqrt{d_i d_j}} & i \neq j, \, j \sim i \\
0 & \text{otherwise}
\end{cases}
$$

$$
\mathcal{L} = T^{-1/2} L T^{-1/2}
$$

### 6.4 Eigenvalue Scaling

Consider the Rayleigh quotient:

$$
\frac{\langle g, \mathcal{L} g \rangle}{\langle g, g \rangle} = \frac{\langle g, T^{-1/2} L T^{-1/2} g \rangle}{\langle g, g \rangle}
$$

Let $f = T^{-1/2} g$:

$$
= \frac{\langle f, L f \rangle}{\langle T^{1/2} f, T^{1/2} f \rangle} = \frac{\sum_{i \sim j} (f_i - f_j)^2}{\sum_k f_k^2 d_k}
$$

**Eigenvalues of $\mathcal{L}$ match those of $L$ scaled by degree!**

### 6.5 Properties of the Spectrum

As a Rayleigh quotient, we can note:

1. **Non-negativity:** $\lambda_i$ are all non-negative

2. **Zero eigenvalue:** $\min(\lambda) = \lambda_0 = 0$ with eigenvector $\vec{v} = \vec{1}$ (constant function)

3. **Connectivity:** For a non-simply connected graph with $k$ connected components:

$$
\lambda_0, \ldots, \lambda_{k-1} = 0
$$

4. **Coercivity constant:** For a simply connected graph, the constant needed for Lax-Milgram is:

$$
\alpha = \lambda_1 = \min_{f \perp \vec{1}} \frac{\sum_{i \sim j} (f_i - f_j)^2}{\sum_k f_k^2 d_k}
$$

**Important Notes:**
- $\lambda_1$ is called the **Fiedler eigenvalue** and is used in spectral graph theory, graph cut algorithms, diffusion problems, and spectral clustering
- This $\alpha$ only holds for $f \perp N_0$ (functions orthogonal to constants), so when we solve the variational problem we need to choose good functions orthogonal to the constant vector

## 7. Estimating the Fiedler Eigenvalue

### 7.1 Computational Options

**Option 1:** For a fixed graph, can always solve the eigenvalue problem
- Arnoldi iteration gives a scalable estimate for large graphs

**Option 2:** Estimate how scaling depends on nodes/edges geometrically

### 7.2 Diameter-Based Bound

Some definitions:

- **Volume:** $\text{vol}(G) = \sum_i d_i$
- **Distance:** $d(v_i, v_j) = $ shortest path connecting $v_i, v_j$
- **Diameter:** $D = \max_{i,j \in V} d(v_i, v_j)$

**Theorem:** For a simply connected graph $G$ with diameter $D$:

$$
\lambda_1 \geq \frac{1}{D \cdot \text{vol}(G)}
$$

**Proof:**

**Step 1:** For the smallest eigenvalue, $f_0 = \vec{1}$.

**Step 2:** By orthogonality of eigenvectors, $\langle f_0, f_1 \rangle = 0$:

$$
\sum_i f_1(v_i) = 0
$$

**Step 3:** Choose $v_0$ as a vertex satisfying:

$$
|f(v_0)| = \max_v |f(v)|
$$

**Step 4:** Since $\sum_i f(v_i) = 0$, we can find a vertex $u_0$ such that $f(u_0) f(v_0) < 0$ (opposite signs).

**Step 5:** Pick path $P$ joining $u_0, v_0$:

$$
\lambda_1 = \frac{\sum_{j \sim i} (f_i - f_j)^2}{\sum_k f_k^2 d_k} \geq \frac{\sum_{j,k \in P} (f_j - f_k)^2}{\text{vol}(G) \cdot f^2(v_0)}
$$

**Step 6:** By the path of length $\leq D$:

$$
\geq \frac{\frac{1}{D}(f(v_0) - f(u_0))^2}{\text{vol}(G) \cdot f^2(v_0)} \geq \frac{1}{D \cdot \text{vol}(G)}
$$

**Interpretation:** The coercivity constant degrades with diameter (long thin graphs are ill-conditioned) and scales inversely with volume.

## 8. Graph Exterior Calculus

### 8.1 Higher-Order Chains

So far we only introduced graph gradient and divergence. For **higher-order chains**, extend the coboundary:

$$
C^k = [v_1, \ldots, v_k]_{v_i \in V}
$$

**Note:** Anti-symmetric with respect to vertex swap.

**Definition:** The boundary operator is:

$$
\partial_{k-1} [v_1, \ldots, v_k] = \sum_{i=1}^k (-1)^{i-1} [v_1, \ldots, \hat{v}_i, \ldots, v_k]
$$

where $\hat{v}_i$ means "leave this vertex out."

### 8.2 Example: Edge Boundary

Given edge $[v_1, v_2] = -[v_2, v_1]$:

$$
\begin{aligned}
\partial_0[v_1, v_2] &= (-1)^0 v_2 + (-1)^1 v_1 \\
&= v_2 - v_1
\end{aligned}
$$

### 8.3 Fundamental Identity

**Theorem:** $\partial_{k-1} \circ \partial_k = 0$

**Proof:**

$$
\begin{aligned}
\partial_{k-1} \partial_k &= \sum_{i=1}^k (-1)^{i-1} \partial_{k-1} [v_1, \ldots, \hat{v}_i, \ldots, v_k] \\
&= \sum_{i,j} (-1)^{i-1} (-1)^{j-1} [v_1, \ldots, \hat{v}_i, \ldots, \hat{v}_j, \ldots, v_k] \\
&= \sum_{i,j} \alpha_{ij}
\end{aligned}
$$

**Note:** Anti-symmetric: $\alpha_{ij} = -\alpha_{ji}$, so the sum vanishes:

$$
= 0
$$

### 8.4 Coboundary Operators

A similar algebraic definition of coboundary can be obtained:

$$
\langle f, \partial_k c \rangle = \langle d_k f, c \rangle
$$

**Examples:**

$$
\begin{aligned}
\delta_0 \phi_{ij} &= \phi_j - \phi_i \quad \text{(gradient)} \\
\delta_1 \phi_{ijk} &= \phi_{ij} + \phi_{jk} + \phi_{ki} \quad \text{(curl)}
\end{aligned}
$$

Similarly, $\delta_k \circ \delta_{k-1} = 0$ (discrete curl-grad identity).

**Example:**

$$
\begin{aligned}
\delta_1 \delta_0 \phi &= \delta_0 \phi_{ij} + \delta_0 \phi_{jk} + \delta_0 \phi_{ki} \\
&= (\phi_j - \phi_i) + (\phi_k - \phi_j) + (\phi_i - \phi_k) \\
&= 0
\end{aligned}
$$

---

## Summary

This lecture covered:

1. **Graphs as discrete manifolds** with chains and cochains
2. **Boundary and coboundary operators** (gradient, divergence, curl)
3. **Graph Laplacian** and its variational formulation
4. **Rayleigh quotients** and spectral analysis
5. **Fiedler eigenvalue** and coercivity bounds
6. **Degree-scaled Laplacian** and diameter-dependent stability estimates
7. **Graph exterior calculus** with higher-order operators
8. **Applications** to circuits, GATs, and recommender systems

**Key Takeaway:** The graph Laplacian $L = -\delta^* \delta$ provides discrete analogs of continuous PDEs, with spectral properties controlled by graph geometry. The Fiedler eigenvalue $\lambda_1$ determines the coercivity constant for Lax-Milgram, with the bound $\lambda_1 \geq 1/(D \cdot \text{vol}(G))$ showing that long thin graphs (large diameter) lead to ill-conditioned systems. Graph exterior calculus extends these ideas to higher-order operators, ensuring discrete versions of identities like $\partial \circ \partial = 0$ hold exactly—critical for structure-preserving algorithms in machine learning and scientific computing.
