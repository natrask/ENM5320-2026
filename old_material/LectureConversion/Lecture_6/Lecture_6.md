# Constrained Optimization and Polynomial Reproduction for Learning Finite Difference Stencils

**Date:** February 5, 2025  
**Topics:** Constrained optimization, polynomial reproduction, moving least squares, Schur complement method

---

## 0. Overview

This lecture establishes the mathematical foundations for learning finite difference (FD) stencils from data while enforcing physical and mathematical constraints. We introduce **constrained optimization** as a central tool for ensuring that learned numerical methods satisfy desired properties such as conservation laws, consistency, and stability.

The lecture connects three critical ideas: (1) **equality-constrained quadratic programming** using Lagrange multipliers and the Schur complement, (2) strategies for training FD stencils from noisy data (penalty methods, reduced-space optimization, and Lagrange multiplier methods), and (3) **polynomial reproduction** as a systematic framework for constructing high-quality stencils that exactly replicate polynomial solutions. The polynomial reproduction framework builds on moving least squares (MLS) and provides a dual optimization perspective: we seek minimal-norm stencil weights that satisfy reproduction constraints.

This material bridges classical numerical analysis (consistency, stability, convergence) with modern machine learning (data-driven stencil discovery, constrained neural networks). Understanding these techniques is essential for designing **physics-informed neural networks** and learning numerical schemes that respect fundamental physical principles like energy conservation and translation invariance.

---

## 1. Constrained Optimization Framework

### 1.1 General Constrained Optimization Problem

Our main tool will be **constrained optimization**:

$$
\begin{aligned}
\min_{\theta} \quad & F(x; \theta) \\
\text{s.t.} \quad & G(x; \theta) = 0
\end{aligned}
$$

**Typical use cases:**

- **Physics-constrained optimization (PDE-constrained optimization):** $F$ is a reconstruction loss, $G$ encodes physics constraints (e.g., satisfying a PDE)
- **Feasibility problems:** $F$ is arbitrary, $G$ represents properties we actually care about (e.g., conservation laws, stability conditions)

### 1.2 Equality Constrained Quadratic Programming

Consider the canonical problem:

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^\top A x + b^\top x + c \\
\text{s.t.} \quad & Bx = d
\end{aligned}
$$

**Note:** For nonlinear problems, replace $A$ with the Hessian and $B$ with the Jacobian.

**Solution via Lagrange multipliers:**

To solve, introduce Lagrange multiplier $\lambda$ and form the Lagrangian:

$$
\mathcal{L}(x, \lambda) = \frac{1}{2} x^\top A x + b^\top x + c + \lambda^\top (Bx - d)
$$

The **Karush-Kuhn-Tucker (KKT) conditions** state that the minimizer satisfies:

$$
\begin{aligned}
\nabla_x \mathcal{L} &= 0 \\
\nabla_\lambda \mathcal{L} &= 0
\end{aligned}
$$

Explicitly:

$$
\begin{aligned}
\nabla_\lambda \mathcal{L} &= Bx - d = 0 \\
\nabla_x \mathcal{L} &= Ax + b + B^\top \lambda = 0
\end{aligned}
$$

This yields the **saddle point problem:**

$$
\begin{pmatrix}
A & B^\top \\
B & 0
\end{pmatrix}
\begin{pmatrix}
x \\
\lambda
\end{pmatrix}
=
\begin{pmatrix}
-b \\
d
\end{pmatrix}
\tag{KKT}
$$

### 1.3 Schur Complement Solution

**Step 1:** Multiply the top row of (KKT) by $BA^{-1}$:

$$
BA^{-1}Ax + BA^{-1}B^\top \lambda = -BA^{-1}b
$$

This simplifies to:

$$
Bx + BA^{-1}B^\top \lambda = -BA^{-1}b
$$

**Step 2:** Extract the constraint $Bx = d$ from the second row of (KKT) and substitute:

$$
BA^{-1}B^\top \lambda = -d - BA^{-1}b
$$

**Step 3:** Define the **Schur complement** $S = BA^{-1}B^\top$ and solve:

$$
\begin{aligned}
\lambda &= -S^{-1}(d + BA^{-1}b) \\
x &= A^{-1}(-B^\top \lambda - b)
\end{aligned}
$$

**Important:** We will use this Schur complement formula repeatedly as a tool to enforce properties we desire as equality constraints in stencil learning.

---

## 2. Training Strategies for Data-Driven Stencil Learning

### 2.1 A Note on Force Matching vs. Solution Fitting

**Force matching (derivative matching):**

In homework problems, we often match derivatives (also called **force matching** in molecular dynamics):

$$
\mathcal{L} = \left\| \frac{\hat{u}_j^{n+1} - \hat{u}_j^n}{k} + \frac{1}{h^{|\alpha|}} \sum_{k=-\alpha}^{\beta} S_{j+k} u_{j+k}^n \right\|^2
$$

to approximate a differential operator $D^\alpha$.

**Problem:** This approach is prone to overfitting for noisy realistic data.

**Mitigation strategies:**

- Explicitly model noise as maximum likelihood estimation (MLE)
- Tikhonov regularization / weight decay (we'll discuss these later)
- Filtering / preprocessing (later topic)
- **Better approach:** Fit to the solution itself rather than its derivatives

### 2.2 Three Approaches to Constrained Fitting

Define the **reconstruction loss** and **physics residual:**

$$
\begin{aligned}
\mathcal{L}_R &= \frac{1}{2} \sum_{n,j} |u_j^n - \hat{u}_j^n|^2 \\
r_{n,j} &= u_j^{n+1} - u_j^n - k D_\theta^\alpha u_j^n
\end{aligned}
$$

where $D_\theta^\alpha$ is a **parameterized stencil**.

We pose the constrained fit to the solution:

$$
\begin{aligned}
\min_{u_j^n, \theta} \quad & \mathcal{L}_R \\
\text{s.t.} \quad & r_{n,j} = 0 \quad \forall n, j
\end{aligned}
$$

#### Method 1: Penalty Method (Worst!)

$$
\min_{u_j^n, \theta} \mathcal{L}_R + \lambda \sum_{n,j} r_{n,j}^2
$$

**Drawbacks:**

- Take larger penalty parameter $\lambda$; for large $\lambda$, $r_{n,j}$ is small
- No way to pick $\lambda$ a priori (too big → unstable, too small → large error)
- Small residuals don't guarantee small solution errors (recall the $O(k^2)$ errors we saw in explicit/Euler)

#### Method 2: Reduced Space Constrained Optimization

Substitute the residual constraint back into $\mathcal{L}_R$ to optimize directly over the space of FD solutions.

If $u_j^n = Q(u_j^{n-1})$ is the (potentially nonlinear) update operator:

$$
\mathcal{L} = \frac{1}{2} \sum_{n,j} \left| Q(u_j^{n-1}) \cdots Q(u_j^0) u_j^0 - \hat{u}_j^n \right|^2
$$

**Challenge:** For nonlinear/implicit schemes, need to backpropagate through a linear solve or Newton/iterative nonlinear solve.

#### Method 3: Lagrange Multipliers (Recommended)

$$
\min_{u^n, r^n, \theta} \frac{1}{2} \sum_{n,j} |u_j^n - \hat{u}_j^n|^2 + \sum_n \lambda_n(r_{n,j})
$$

**Process:** For iterations $1, \ldots, \#$:

1. $\nabla_u \mathcal{L} = 0 \Rightarrow$ Solve forward problem
2. $\nabla_\lambda \mathcal{L} = 0 \Rightarrow$ Solve adjoint problem:
   $$
   (J_{ru}(r))^\top \lambda = -(u_j^n - \hat{u}_j^n)
   $$
3. Update $\theta$ with gradient optimization using current guess for $u, \lambda$

**Advantage:** Backpropagation doesn't need to go through forward + adjoint solve; best for nonlinear/implicit schemes.

**Note:** For linear problems, methods (2) and (3) are equivalent.

### 2.3 Mini-Batching

**Important considerations:**

- These are large gradient calculations
- Typically our dataset consists of many simulations
- Use **stochastic gradient descent (SGD)**
- Pick random solution $\hat{u}_{j,d}^n$ (where $d$ indexes different simulations)
- Pick subset of time series: $u_{j,d}^n, \ldots, u_{j,d}^{n+m}$
- Train using $\hat{u}_{j,d}$ as initial condition

---

## 3. Polynomial Reproduction and Least Squares

### 3.1 Scattered Data and Fill Distance

Consider points $\bar{X} = \{x_1, \ldots, x_N\} \subseteq \Omega \subseteq \mathbb{R}^d$.

**Fill distance** (characterizes scattered data):

$$
h_{\bar{X}, \Omega} = \sup_{x \in \Omega} \min_{j \leq N} \|x - x_j\|_2
$$

Interpretation: Radius of the biggest ball in $\Omega$ without any data points in it.

**Separation distance:**

$$
q_{\bar{X}} = \frac{1}{2} \min_{i \neq j} \|x_i - x_j\|_2
$$

**Definition:** Data sites are **quasi-uniform** (with constant $c_{qu}$) if:

$$
q_{\bar{X}} \leq h_{\bar{X}, \Omega} \leq c_{qu} q_{\bar{X}}
$$

**For our FD stencils so far:**

$$
h_{\bar{X}, \Omega} = h, \quad q_{\bar{X}} = h \quad \Rightarrow c_{qu} = 1
$$

(uniform grid)

### 3.2 Moving Least Squares (MLS)

**Idea:** Introduce moving least squares to generalize FDM for scattered data.

For $x \in \Omega$, define $S_{f,\bar{X}}(x) = p^*(x)$ where:

$$
p^* = \arg\min_{p \in \Pi_m(\mathbb{R}^d)} \sum_{i=1}^N [f(x_i) - p(x_i)]^2 \omega(x, x_i)
$$

where $\Pi_m(\mathbb{R}^d)$ denotes polynomials of degree $\leq m$ in $\mathbb{R}^d$.

We'll assume $\omega(x,y) = \Phi_\delta(x-y)$ (radial basis function kernel).

#### Simple Example: $m = 0$ (Constant Approximation)

$$
\min_{c(x)} \sum_{i=1}^N \frac{1}{2}[f(x_i) - c(x)]^2 \Phi_\delta(x - x_i)
$$

**Step 1:** Take derivative with respect to $c(x)$ and set to zero:

$$
0 = \sum_{i=1}^N (f(x_i) - c(x)) \Phi_\delta(x - x_i)
$$

**Step 2:** Solve for $c(x)$:

$$
c(x) = \frac{\sum_i \Phi_\delta(x - x_i) f(x_i)}{\sum_i \Phi_\delta(x - x_i)}
$$

This recovers **kernel density estimation**, also known as the **Shepard interpolant**.

---

## 4. Primal and Dual Formulations of MLS

**Theorem:** Solving the moving least squares problem is equivalent to solving:

$$
\begin{aligned}
S_{f,\bar{X}}(x) &= \sum_i a_i^*(x) f(x_i) \\
a_i^* &= \arg\min \frac{1}{2} \sum_i \frac{a_i(x)^2}{\Phi_\delta(x - x_i)}
\end{aligned}
$$

subject to polynomial reproduction constraints.

**Remark:** We can thus search for a **minimal norm stencil** that has reproduction properties.

### 4.1 Proof: Setup and Notation

**Notation (linear algebra heavy):**

- $a \in \mathbb{R}^N$ — stencil weights
- $W = \text{diag}(\Phi_\delta(x - x_i)) \in \mathbb{R}^{N \times N}$ — weight matrix
- $P(x) \in \mathbb{R}^M$ — polynomial basis evaluated at $x$
- $P \in \mathbb{R}^{M \times N}$ — polynomial basis evaluated at all data points: $P_{ij} = p_j(x_i)$
- $u = u(x_i) \in \mathbb{R}^N$ — function values at data points
- $c \in \mathbb{R}^M$ — polynomial coefficients

**Dimensions:**
- $N$ = number of nodes
- $M = \dim(\Pi_m(\mathbb{R}^d))$ = number of polynomial basis functions

### 4.2 Primal Problem

$$
\min_{c \in \mathbb{R}^M} \frac{1}{2} \underbrace{(u - Pc)^\top W (u - Pc)}_{\mathcal{L}}
$$

**Step 1:** Expand the objective:

$$
\mathcal{L} = \frac{1}{2} c^\top P W P^\top c - c^\top P W u + \text{(terms independent of } c\text{)}
$$

**Step 2:** Recall calculus identities:

$$
\frac{\partial}{\partial x} \frac{1}{2} x^\top A x = \frac{1}{2}(A + A^\top)x, \quad \frac{\partial}{\partial x} y^\top x = y
$$

**Step 3:** Take gradient and set to zero:

$$
\frac{\partial \mathcal{L}}{\partial c} = PWP^\top c - PWu = 0
$$

$$
\Rightarrow c = (PWP^\top)^{-1} PWu
$$

**Step 4:** The MLS approximation is:

$$
\begin{aligned}
S_{f,\bar{X}}(x) &= c \cdot P(x) \\
&= P(x)^\top (PWP^\top)^{-1} PWu
\end{aligned}
$$

### 4.3 Dual Problem

$$
\begin{aligned}
\min_{a \in \mathbb{R}^N} \quad & \frac{1}{2} a(x)^\top W^{-1} a(x) \\
\text{s.t.} \quad & a(x)^\top P = P(x)^\top
\end{aligned}
$$

**Interpretation:** Find minimal weighted norm stencil weights $a(x)$ that reproduce all polynomials up to degree $m$.

**Applying the Schur complement formula:**

The KKT system is:

$$
\begin{pmatrix}
W^{-1} & P^\top \\
P & 0
\end{pmatrix}
\begin{pmatrix}
a(x) \\
\lambda
\end{pmatrix}
=
\begin{pmatrix}
0 \\
P(x)
\end{pmatrix}
$$

**Step 1:** Identify Schur complement:

$$
S = P W P^\top
$$

**Step 2:** Solve for $\lambda$:

$$
\lambda = -S^{-1} P(x) = -(PWP^\top)^{-1} P(x)
$$

**Step 3:** Solve for $a(x)$:

$$
\begin{aligned}
a(x) &= -W P^\top \lambda \\
&= W P^\top S^{-1} P(x)
\end{aligned}
$$

**Step 4:** The MLS approximation is:

$$
\begin{aligned}
S_{f,\bar{X}}(x) &= u^\top a(x) \\
&= u^\top W P^\top S^{-1} P(x) \\
&= P(x)^\top (PWP^\top)^{-1} PWu
\end{aligned}
$$

This matches the primal formulation! ✓

---

## 5. Extension to Differential Operators

**Corollary:** Replace the constraint $a(x)^\top P = P(x)^\top$ with:

$$
a(x)^\top P = D^\alpha P(x)
$$

to obtain **differential operator stencils** that reproduce differentiated polynomials.

**Question:** Is the polynomial reproduction set non-empty? (i.e., do solutions always exist?)

---

## 6. Existence Theorem for Polynomial Reproduction

**Theorem (Wendland, "Scattered Data Approximation," Theorem 4.7):**

Suppose $\Omega \subseteq \mathbb{R}^d$ is compact and satisfies an **interior cone condition** with angle $\theta \in (0, \pi/2)$ and radius $r$. Then there exists stencil weights $a_j^*(x)$ for any $x$ such that:

1. **Polynomial reproduction:** 
   $$
   \sum_i a_j^*(x) p(x_j) = p(x) \quad \forall p \in \Pi_m(\mathbb{R}^d)
   $$

2. **Bounded stability:** 
   $$
   \sum_j |a_j^*(x)| \leq \tilde{c}_1
   $$

3. **Compact support (locality):** 
   $$
   a_j^*(x) = 0 \quad \text{if} \quad \|x - x_j\|_2 > \tilde{c}_2 h_{\bar{X}, \Omega}
   $$

where $\tilde{c}_1$ and $\tilde{c}_2$ are independent of $h_{\bar{X}, \Omega}$ and can be explicitly derived.

**Implications:**

- Stencils with polynomial reproduction always exist under mild geometric conditions
- Stencil weights remain bounded (stability)
- Stencils have compact support (locality, efficient computation)

---

## Summary

This lecture established three key frameworks for learning constrained numerical methods:

1. **Constrained optimization via Lagrange multipliers and Schur complement** — provides a systematic approach to enforce equality constraints (conservation laws, consistency conditions) in learned stencils

2. **Three training strategies** for data-driven stencil learning:
   - Penalty methods (avoid due to difficulty choosing $\lambda$)
   - Reduced-space optimization (efficient but requires differentiating through solvers)
   - Lagrange multipliers (most flexible, decouples optimization from PDE solve)

3. **Polynomial reproduction framework** — connects moving least squares, primal-dual optimization, and stencil design; guarantees existence of stable, local stencils that exactly reproduce polynomials

4. **Wendland's theorem** — ensures polynomial reproduction stencils exist, are stable, and have compact support under mild geometric conditions

**Key Takeaway:** Constrained optimization provides the mathematical machinery to design physics-informed learning algorithms that respect fundamental properties like conservation laws and polynomial consistency. The primal-dual formulation reveals that seeking minimal-norm stencils with polynomial reproduction is equivalent to moving least squares approximation, unifying classical numerical analysis with modern data-driven methods.

**Next steps:** Apply these principles to construct energy-conserving time integrators using Hamiltonian mechanics and symplectic structure (Lecture 7).
