# Lecture 17: Helmholtz-Hodge Decomposition and Graph Calculus

**Reference:** Lecture 17 - Hodge Decomposition and Applications

**Topics Covered:**
- Vector calculus review: Levi-Civita tensor and exact sequences
- Helmholtz-Hodge decomposition theorem
- Projections onto curl-free and divergence-free spaces
- Graph Hodge decomposition
- Applications to recommender systems and causal inference

---

## 0. Overview

This lecture brings together several threads from the course: vector calculus identities, mixed finite element methods, and graph theory. The unifying concept is the **Helmholtz-Hodge decomposition**, which states that any vector field can be uniquely decomposed into a gradient (curl-free) part and a curl (divergence-free) part.

This decomposition has profound implications across physics and machine learning. In fluid mechanics, it provides a way to enforce incompressibility constraints without solving large saddle-point problems—leading to **projection methods** that are computationally efficient. In electromagnetics, it separates electric and magnetic contributions. In machine learning on graphs, it enables structured models for recommendation systems and causal discovery.

We begin with a review of vector calculus using the **Levi-Civita tensor**, providing compact notation for curl operations and proving the fundamental exact sequence property: $\nabla \cdot \nabla \times = 0$ and $\nabla \times \nabla = 0$. We then prove the Helmholtz-Hodge decomposition by solving two Poisson problems, carefully handling the null spaces that arise. The continuous theory translates directly to graphs, where we construct a **graph Hodge decomposition** using coboundary operators.

Finally, we explore applications: the **Chorin projection method** for incompressible Navier-Stokes, recommender systems that learn preference functions on graphs, and **causal inference** where fluxes define parent-child relationships in probabilistic graphical models.

## 1. Vector Calculus Review

### 1.1 The Levi-Civita Tensor

**Reference:** "Div, Grad, Curl, and All That" by H.M. Schey

Recall the **Levi-Civita tensor** $\varepsilon_{ijk}$ satisfying:

$$
\varepsilon_{ijk} = \begin{cases}
+1 & \text{even permutations of } (1,2,3) \\
-1 & \text{odd permutations} \\
0 & \text{equal indices}
\end{cases}
$$

This provides a **compact notation for curl**:

$$
(\vec{F} \times \vec{G})_i = \varepsilon_{ijk} F_j G_k
$$

**Pedagogical note:** The Levi-Civita tensor converts cross products into index manipulations, making proofs mechanical and eliminating the need for component-wise calculations.

### 1.2 Exact Sequence Properties

Some identities—analogous to the exact sequence property:

$$
\nabla \cdot \nabla \times u = 0 \quad \text{and} \quad \nabla \times \nabla \phi = 0
$$

**Proof (Divergence of Curl):**

Adopting **Einstein notation** (repeated indices are summed):

**Step 1:** Write curl in index notation:

$$
(\nabla \times u)_i = \varepsilon_{ijk} \partial_{x_j} u_k
$$

**Step 2:** Take divergence:

$$
\nabla \cdot \nabla \times u = \varepsilon_{ijk} \partial_{x_i} \partial_{x_j} u_k
$$

**Step 3:** Symmetrize with respect to $i,j$ (partial derivatives commute):

$$
= \frac{1}{2} \varepsilon_{ijk} \partial_{x_i} \partial_{x_j} u_k + \frac{1}{2} \varepsilon_{jik} \partial_{x_j} \partial_{x_i} u_k
$$

**Step 4:** Swap dummy indices in second term:

$$
= \frac{1}{2}(\varepsilon_{ijk} + \varepsilon_{jik}) \partial_{x_i} \partial_{x_j} u_k
$$

**Step 5:** Note that $\varepsilon_{jik} = -\varepsilon_{ijk}$ (odd permutation), so:

$$
= 0
$$

**Proof (Curl of Gradient):** Similarly,

$$
\begin{aligned}
(\nabla \times \nabla \phi)_i &= \varepsilon_{ijk} \partial_{x_j} \partial_{x_k} \phi \\
&= \frac{1}{2}(\varepsilon_{ijk} + \varepsilon_{ikj}) \partial_{x_j} \partial_{x_k} \phi \\
&= 0
\end{aligned}
$$

since $\varepsilon_{ikj} = -\varepsilon_{ijk}$.

## 2. Helmholtz-Hodge Decomposition

### 2.1 Statement of Theorem

**Theorem (Helmholtz-Hodge Decomposition):** Given a smooth vector field $F \in \mathbb{R}^d$ on a simply connected subset of $\mathbb{R}^d$, there exist unique scalar potential $\phi$ and vector potential $A$ such that:

$$
F = -\nabla \phi + \nabla \times A
$$

where:
- $-\nabla \phi$ is **curl-free** (irrotational)
- $\nabla \times A$ is **divergence-free** (solenoidal)

### 2.2 Proof: Finding the Scalar Potential

**Step 1:** Take divergence of both sides:

$$
\nabla \cdot F = -\nabla \cdot \nabla \phi + \nabla \cdot (\nabla \times A)
$$

**Step 2:** Use $\nabla \cdot (\nabla \times A) = 0$:

$$
\nabla \cdot F = -\nabla^2 \phi
$$

**Step 3:** This gives a **pure Neumann Poisson problem**:

$$
\begin{cases}
-\nabla^2 \phi = \nabla \cdot F \\
\left. \partial_n \phi \right|_{\partial \Omega} = -F \cdot \hat{n}
\end{cases}
$$

**Important Note:** This problem has a constant vector in its **null space**. For any constant $c$:

$$
\begin{aligned}
\nabla^2(\phi + c) &= \nabla^2 \phi \\
\partial_n(\phi + c) &= \partial_n \phi
\end{aligned}
$$

**Step 4:** To maintain coercivity in Lax-Milgram, need to work in a space orthogonal to the kernel:

$$
V_h = \{\text{CPWL functions } u \mid u \cdot \vec{1} = 0\}
$$

**Alternative:** Enforce with a Lagrange multiplier:

$$
\min_\phi \int_\Omega \frac{1}{2}|\nabla \phi|^2 - \phi \nabla \cdot F \, dx + \lambda(\phi \cdot \vec{1}) + \int_{\partial \Omega} F \cdot \phi \, dA
$$

### 2.3 Proof: Finding the Vector Potential

**Step 1:** Take curl of both sides:

$$
\nabla \times F = -\nabla \times \nabla \phi + \nabla \times \nabla \times A
$$

**Step 2:** Use $\nabla \times \nabla \phi = 0$:

$$
\nabla \times F = \nabla \times \nabla \times A
$$

**Step 3:** Now we have a more challenging null space. For any $q$:

$$
\nabla \times \nabla \times (A + \nabla q) = \nabla \times \nabla \times A
$$

So we'd like solutions orthogonal to $\nabla q$ for any $q$:

$$
(A, \nabla q) = 0 \quad \forall q
$$

**Step 4:** Applying Galerkin:

$$
(\nabla \times \nabla \times A, v) = (\nabla \times F, v)
$$

**Step 5:** Integrate by parts:

$$
(\nabla \times A, \nabla \times v) + \int_{\partial \Omega} (\nabla \times A \times \hat{n}) \cdot v \, dA = (\nabla \times F, v)
$$

**Natural BC:**

$$
\nabla \times A \times \hat{n} = \nabla \times F \times \hat{n}
$$

**Step 6:** Enforce orthogonality with **Lagrange multiplier**:

$$
\begin{aligned}
(\nabla \times \nabla \times A, v) + (\nabla \lambda, v) &= (\nabla \times F, v) - \langle \nabla \times F \times \hat{n}, v \rangle \\
(A, \nabla q) &= 0
\end{aligned}
$$

**Critical observation:** This is a saddle-point problem that we showed how to approach last week.

## 3. Applications of Hodge Decomposition

### 3.1 Projection Operators

The Hodge decomposition provides useful tools for **projecting onto divergence-free or curl-free spaces**:

$$
\begin{aligned}
\pi_{\text{div-free}} u &= \nabla \times A \\
\pi_{\text{curl-free}} u &= -\nabla \phi
\end{aligned}
$$

### 3.2 Potential Flow (Fluid Mechanics)

**Potential flow** was an early model of fluids (with large, restrictive assumptions) used to design airfoils in the 1920s-1940s.

**Assumptions:**

$$
\begin{cases}
u = -\nabla \phi \\
\nabla \cdot u = 0
\end{cases}
$$

**Result:** Given boundary conditions, solve:

$$
\begin{cases}
-\nabla^2 \phi = 0 \\
\left. \partial_n \phi \right|_{\partial \Omega} = u \cdot \hat{n}
\end{cases}
$$

**Note:** No FEM back then → Laplace transforms, etc. You'll still see this in graduate fluid mechanics courses!

### 3.3 Chorin Projection Method (1968)

When solving **Navier-Stokes with incompressibility**:

$$
\begin{aligned}
\partial_t u + u \cdot \nabla u &= -\nabla p + \nu \nabla^2 u \\
\nabla \cdot u &= 0
\end{aligned}
$$

The incompressibility constraint is annoying:
- Makes system bigger
- How to choose inf-sup compatible $u, p$?
- Couples all points in domain

**A popular scheme** (Alexandre Chorin, 1968), called a **splitting scheme**:

**Step 1:** Advection-diffusion step (ignoring pressure):

$$
\frac{u^* - u^n}{k} = -u^n \cdot \nabla u^n + \nu \nabla^2 u^*
$$

**Step 2a:** Pressure-projection step:

$$
\begin{cases}
\frac{u^{n+1} - u^*}{k} = -\nabla p^{n+1} \\
\nabla \cdot u^{n+1} = 0
\end{cases}
$$

**Step 2b:** Taking divergence of Step 2a:

$$
\frac{\nabla \cdot u^{n+1} - \nabla \cdot u^*}{k} = -\nabla^2 p^{n+1}
$$

or:

$$
u^{n+1} = \pi_{\text{div-free}} u^* = u^* - \nabla p^{n+1}
$$

**This is the pressure-projection method!**

## 4. Graph Hodge Decomposition

### 4.1 Review of Graph Operators

We introduced grad/div as a coboundary/codifferential pair:

$$
\langle v, \delta_0 u \rangle = \langle \delta_0^* v, u \rangle
$$

Now for the formal definitions so we can talk about curl.

**Definition:** A **$k$-chain** $C^k = [v_1, \ldots, v_k]_{v_i \in V}$ is anti-symmetric with respect to vertex swaps.

**Definition:** The **boundary operator** $\partial_{k-1} : C^k \to C^{k-1}$ is:

$$
\partial_{k-1}[v_1, \ldots, v_k] = \sum_{i=1}^k (-1)^{i-1} [v_1, \ldots, \hat{v}_i, \ldots, v_k]
$$

where $\hat{v}_i$ means "leave this vertex out."

### 4.2 Example: Edge Boundary

For edge $[v_1, v_2] = -[v_2, v_1]$:

$$
\begin{aligned}
\partial_0[v_1, v_2] &= (-1)^0 [v_2] + (-1)^1 [v_1] \\
&= v_2 - v_1
\end{aligned}
$$

### 4.3 Exact Sequence Property

**Theorem:** $\partial_{k-1} \circ \partial_k = 0$

**Proof:** Given $c \in C^{k+1}$:

$$
\begin{aligned}
\partial_{k-1} \partial_k c &= \sum_{i=1}^k (-1)^{i-1} \partial_k [v_1, \ldots, \hat{v}_i, \ldots, v_{k+1}] \\
&= \sum_{i,j} (-1)^{i-1} (-1)^{j-1} [v_1, \ldots, \hat{v}_i, \ldots, \hat{v}_j, \ldots, v_k] \\
&= \sum_{i,j} (-1)^{i+j-2} \alpha_{ij}
\end{aligned}
$$

**Note:** $\alpha_{ij} = -\alpha_{ji}$ (anti-symmetric if we exchange $i,j$), so the sum vanishes:

$$
= 0
$$

### 4.4 Coboundary Operators

To define the coboundary (divergence/curl), just swap chains with cochains.

For $f \in C_k$:

$$
\delta_{k-1} f = \sum (-1)^i f(v_1, \ldots, \hat{v}_i, \ldots, v_k)
$$

**Examples:**

$$
\begin{aligned}
\delta_0 \phi_{ij} &= \phi_j - \phi_i \quad \text{(graph gradient)} \\
\delta_1 A_{ijk} &= A_{ij} + A_{jk} + A_{ki} \quad \text{(graph curl)}
\end{aligned}
$$

**Verification:** Curl of gradient is zero:

$$
\begin{aligned}
\delta_1 \delta_0 \phi &= \delta_0 \phi_{ij} + \delta_0 \phi_{jk} + \delta_0 \phi_{ki} \\
&= (\phi_j - \phi_i) + (\phi_k - \phi_j) + (\phi_i - \phi_k) \\
&= 0
\end{aligned}
$$

### 4.5 Divergence of Curl is Zero

If $\delta_1 \delta_0 = 0$, then $\delta_0^* \delta_1^* = 0$ (divergence of curl is zero):

$$
\begin{aligned}
\langle f, \delta_0^* \delta_1^* g \rangle &= \langle \delta_0 f, \delta_1^* g \rangle \\
&= \langle \delta_1 \delta_0 f, g \rangle \\
&= 0
\end{aligned}
$$

Recall the operator identities:

$$
\begin{array}{ll}
\delta_0 = \text{grad} & \delta_0^* = \text{div} \\
\delta_1 = \text{curl} & \delta_1^* = \text{curl}
\end{array}
$$

### 4.6 Graph Hodge Decomposition Theorem

**Theorem:** Given $f \in C_k$:

$$
f = \delta_0 \phi + \delta_1^* A
$$

**Proof:** Apply operators:

$$
\begin{aligned}
\delta_0^* f &= \delta_0^* \delta_0 \phi + \delta_0^* \delta_1^* A \\
\delta_1 f &= \delta_1 \delta_0 \phi + \delta_1 \delta_1^* A
\end{aligned}
$$

**Critical observation:** These are graph Laplacian equations that can be solved separately!

## 5. Applications to Machine Learning

### 5.1 Recommender Systems

Given observations on some preferences $Y_{ij}^{\text{data}}$:

$$
\min_s \sum_{i,j} w_{ij} (Y_{ij}^{\text{data}} - \delta_0 s_{ij})^2
$$

**Take the variation** (HW exercise):

$$
\delta_0^* W \delta_0 s = \delta_0^* W Y^{\text{data}}
$$

**This is a weighted graph Laplacian!**

### 5.2 Causality

Given a set of events $X_i$, a **causal structural model** is a decomposition:

$$
P(x_1, \ldots, x_N) = \prod_{i=1}^N P(x_i \mid \text{Pa}(x_i))
$$

where $\text{Pa}(x_i)$ are the parents of $x_i$.

**Graph-based approach:** Define a score $s_i$ and compute:

$$
\text{flux } F = \delta_0 s
$$

Define parents as:

$$
\text{Pa}(x_i) = \{x_j \text{ s.t. } F_{ij} \to 0\}
$$

or a differentiable version:

$$
\text{Pa}(x_i) = \left\{ \text{ReLU}\left( \tanh\left( \frac{1}{\beta} F_{ij} \right) \right) \geq \epsilon \right\}
$$

**Interpretation:** Causal relationships are identified by significant flux values, naturally incorporating graph structure into causal discovery algorithms.

---

## Summary

This lecture covered:

1. **Vector calculus review** using Levi-Civita tensor notation
2. **Exact sequence properties**: $\nabla \cdot \nabla \times = 0$ and $\nabla \times \nabla = 0$
3. **Helmholtz-Hodge decomposition** theorem and proof
4. **Null space handling** via Lagrange multipliers
5. **Projection methods** for incompressible flow (Chorin scheme)
6. **Graph Hodge decomposition** and exact sequences on graphs
7. **Applications** to recommender systems and causal inference

**Key Takeaway:** The Helmholtz-Hodge decomposition $F = -\nabla \phi + \nabla \times A$ provides a fundamental splitting of vector fields into curl-free and divergence-free components. This decomposition has computational advantages (splitting large problems into smaller ones), physical interpretations (separating irrotational and rotational flow), and extends naturally to graphs via coboundary operators. The projection method for Navier-Stokes exemplifies how Hodge decomposition enables efficient algorithms by decoupling advection-diffusion from pressure-projection steps. On graphs, the discrete Hodge decomposition provides a principled framework for structured learning in recommender systems and causal inference.
