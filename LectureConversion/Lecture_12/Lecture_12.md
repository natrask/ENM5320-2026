# Lecture 12: Introduction to Finite Element Methods

**Date:** March 3

**Topics Covered:**
- Transition from Finite Difference Methods (FDM) to Finite Element Methods (FEM)
- Weak formulation and Galerkin methods
- Function spaces: $L^2$, $H^1$, $H_0^1$
- Basis functions and shape functions
- Assembly of stiffness matrices
- Gaussian quadrature
- Connection to energy minimization
- Galerkin orthogonality and optimal error estimates
- Handling Neumann boundary conditions

---

## 0. Overview

This lecture marks a pivotal transition in our course: we move from **finite difference methods (FDM)** to **finite element methods (FEM)**. While FDM has served us well for periodic problems on regular grids, its limitations become apparent when dealing with complex geometries, non-periodic boundary conditions, and mesh refinement. FEM provides the flexibility needed for real-world engineering problems while maintaining the mathematical rigor we've developed.

Before making this transition, we pause to reflect on what FDM has given us: proofs of stability and conservation properties, convergence guarantees via the Lax equivalence theorem and polynomial reproduction, stable integration of Hamiltonian systems, and a framework for physics learning as nonlinear perturbations of well-understood problems. In engineering, the **verification and validation (V&V)** paradigm is fundamental: *verification* ensures code correctly implements the mathematical theory (math ⇒ good code), while *validation* ensures the code predicts experimental data (good code ⇔ good model).

The **Galerkin approach** at the heart of FEM takes a fundamentally different perspective. Instead of requiring solutions to be twice differentiable (as FDM does), we formulate problems in a **weak sense** that only requires first derivatives. This isn't a limitation—"weak" doesn't mean "bad"—but rather enables a broader class of solutions and more flexible discretizations. We'll see that FEM naturally connects to energy minimization (the Ritz method) and provides **optimal error estimates** through the beautiful property of Galerkin orthogonality.

---

## 1. Reflection on Finite Difference Methods

### 1.1 What We Can Do with FDM for ML

**(1) Prove stability, energy conservation, momentum conservation**

**(2) Use Lax equivalence theorem + polynomial reproduction to guarantee convergence**

**(3) Integrate Hamiltonian systems with discrete energy conservation, or use RK4+ to get stable, convergent solutions**

**(4) Pose physics learning as nonlinear perturbation of "nice" problems**

### 1.2 Verification and Validation in Engineering

In engineering, **"V&V"** (Verification & Validation) is the bedrock of trust in simulations:

- **Verification:** Is the code correctly implemented? Use unit tests to verify reproduction of the claims theory provides  
  → **Math ⇒ Good Code**

- **Validation:** Is the code predictive of experimental data?  
  → **Good Code ⇔ Good Model**

### 1.3 What We Cannot Do with FDM

**(1) Handle complex geometries**

**(2) Handle non-periodic boundary conditions naturally**

**(3) Refine meshes adaptively**

**⇒ We need FEM!**

---

## 2. Historical Context

### 2.1 FEM History: Perspective Beyond the ML Hype

**1909:** Walter Ritz establishes the **Rayleigh-Ritz method** for variational mechanics

**1915:** Boris Galerkin develops what we now call the **Galerkin method**

**1930s:** Richard Courant solves PDEs using **piecewise polynomial functions**  
→ Early theory developed in the 1940s  
→ **No adoption in industry** due to computational limitations

**1960:** Ray Clough coins the term **"Finite Element Method"**  
→ Shift from "spring-mass elements" to the mathematical framework we'll learn today

**1960s:** Mathematical formalism developed rigorously

**1980s:** Computers catch up; early commercial software (Abaqus, etc.)

### 2.2 ML History for Comparison

**1987:** First Multi-Layer Perceptron (MLP) with backpropagation (Rumelhart, Hinton, Williams)

**1998:** First "PINN" paper (Lagaris et al.)

**Note:** FEM has 60+ years of mathematical rigor and engineering validation—don't fall for the hype that ML invented scientific computing!

---

## 3. The Poisson Problem and Weak Formulation

### 3.1 Strong Form (Classical Formulation)

Consider the **Poisson problem** with **Dirichlet boundary conditions**:

$$(P) \quad \begin{cases}
-u'' = f & \text{on } (0,1) \\
u(0) = u(1) = 0
\end{cases}$$

**Example:** For $f = 1$, the exact solution is:
$$
u(x) = \frac{x(x-1)}{2}
$$

**Problem with finite differences:** Assumes too much regularity in the solution:
$$
u \in C^2([0,1])
$$

(i.e., $u$ must be twice continuously differentiable)

**Weak form:** Poses the problem without requiring this regularity.

**Note:** "Weak" ≠ "bad" or "not strong"—actually enables a **more general class of solutions**.

### 3.2 The "Galerkin Cookbook" (Error Measurement Approach)

**Step 1:** Take an arbitrary test function $v$ with $v(0) = v(1) = 0$

**Step 2:** Multiply $(P)$ by $v$ and integrate:
$$
-\int_0^1 u'' v \, dx = \int_0^1 f v \, dx
$$

**Step 3:** Integrate by parts to get the **least restrictive derivatives** on $u$ and $v$ possible:
$$
\int_0^1 u' v' \, dx - \underbrace{(u'(1) v(1) - u'(0) v(0))}_{= 0 \text{ by Dirichlet BC}} = \int_0^1 f v \, dx
$$

This defines the **Galerkin (weak) form** of $(P)$:

### 3.3 Weak Formulation

**(G) Find $u \in V$ such that for any $v \in V$:**

$$
\boxed{\int_0^1 u' v' \, dx = \int_0^1 f v \, dx}
$$

**Remarks:**
- There is a **symmetry** similar to what we've seen with $D_+, D_-$ in FDM
- **How do we choose the function space $V$?**

---

## 4. Function Spaces for FEM

### 4.1 Choosing the Right Function Space

The weak formulation must be well-defined. If we pick $v = u$:

$$
\int_0^1 (u')^2 \, dx = \int_0^1 f u \, dx
$$

Both sides need to be finite!

### 4.2 Useful Function Spaces

**$L^2$ space** (square-integrable functions):
$$
L^2([0,1]) = \left\{ f : \int_0^1 f^2 \, dx < \infty \right\}
$$

**$H^1$ space** (Sobolev space, first derivative in $L^2$):
$$
H^1([0,1]) = \left\{ f \in L^2 : f' \in L^2 \right\}
$$

**$H_0^1$ space** (functions in $H^1$ with zero boundary values):
$$
H_0^1([0,1]) = \left\{ f \in H^1 : f(0) = f(1) = 0 \right\}
$$

### 4.3 Well-Posedness Analysis

**LHS of (G):** Finite if $u \in H^1$

**RHS of (G):** By Cauchy-Schwarz,
$$
\left| \int f u \, dx \right| \leq \left( \int |f|^2 \, dx \right)^{1/2} \left( \int |u|^2 \, dx \right)^{1/2} < \infty
$$
if $f \in L^2$ and $u \in L^2$.

**Boundary terms:** $H_0^1 \Rightarrow$ Don't worry about boundary terms during integration by parts.

**Therefore:** $\boxed{V = H_0^1}$

**Important Note:** This choice is specific to Dirichlet BCs. Different PDEs require different function spaces.

---

## 5. Making It "Finite": The Galerkin Approximation

### 5.1 Finite-Dimensional Subspace

Choose $V_h \subseteq V$ with $\dim(V_h) = N$ (i.e., pick a basis with $N$ shape functions).

Then:
$$
u \in V_h \quad \Rightarrow \quad u(x) = \sum_{i=1}^N \hat{u}_i \phi_i(x)
$$

### 5.2 Many Choices for Basis Functions

| Type | Example | Method Name |
|------|---------|-------------|
| **Discontinuous** | Piecewise constant | Finite Volume |
| **Continuous polynomial** | Piecewise linear/quadratic | (Bubnov-)Galerkin |
| **Trigonometric / Orthogonal** | Fourier, Chebyshev, Legendre | Spectral Element |
| **Non-polynomial** | Gaussians, RBFs | Meshfree |

### 5.3 Piecewise Linear Basis Functions

**For this lecture, take** $V_h = \{\text{piecewise linear functions}\}$

**Define mesh:**
$$
\begin{aligned}
X_h &= \{ih : i = 0, 1, \ldots, N_{el}\} \\
e_{i+1} &= [ih, (i+1)h]
\end{aligned}
$$

**Define "hat functions":**
$$
\phi_i(x) = \begin{cases}
\frac{x - x_{i-1}}{x_i - x_{i-1}} & x \in e_i \\
1 - \frac{x - x_i}{x_{i+1} - x_i} & x \in e_{i+1} \\
0 & \text{otherwise}
\end{cases}
$$

**Key property:** $\phi_i(x_j) = \delta_{ij}$ (Kronecker delta)

This means the coefficients $\hat{u}_i$ are simply the **nodal values**: $\hat{u}_i = u(x_i)$.

---

## 6. Assembly of the Linear System

### 6.1 Discrete Weak Form

Rewrite **(G)**: Find $u \in V_h$ such that for any $v \in V_h$:
$$
\int_0^1 u' v' \, dx = \int_0^1 f v \, dx
$$

**Equivalently:** Find $\hat{u} \in \mathbb{R}^{N_{el}}$ such that $\forall \hat{v} \in \mathbb{R}^{N_{el}}$:

$$
\sum_j \hat{v}_i \left[ \underbrace{\int_0^1 \phi_i' \phi_j' \, dx}_{S_{ij}} \hat{u}_j - \int_0^1 \phi_i f \, dx \right] = 0
$$

**This is true if:**
$$
\boxed{S \hat{u} = b}
$$

where:
- $S_{ij} = \int_0^1 \phi_i' \phi_j' \, dx$ is the **stiffness matrix**
- $b_i = \int_0^1 \phi_i f \, dx$ is the **load vector**

---

## 7. Uniqueness and Properties of the Stiffness Matrix

### 7.1 Theorem: FEM Discretization of (G) is Unique

**Proof:** Suppose $u_1, u_2$ both solve $(G)$:

$$
\begin{aligned}
\int_0^1 u_1' v' \, dx &= \int_0^1 f v \, dx \\
\int_0^1 u_2' v' \, dx &= \int_0^1 f v \, dx
\end{aligned}
$$

**Subtract:**
$$
\int_0^1 (u_1 - u_2)' v' \, dx = 0 \quad \text{for any } v \in V_h
$$

**Pick** $v = u_1 - u_2$:
$$
\int_0^1 [(u_1 - u_2)']^2 \, dx = 0
$$

$$
\Rightarrow (u_1 - u_2)' = 0 \quad \text{pointwise}
$$

$$
\Rightarrow u_1 - u_2 = \text{constant}
$$

**Since** $V_h \subseteq H_0^1$, we have $u_1 - u_2 = 0$ on the boundary.

$$
\Rightarrow u_1 - u_2 = 0 \text{ everywhere} \quad \checkmark
$$

### 7.2 Theorem: $S$ is Symmetric Positive Definite

**Proof:**

$$
\begin{aligned}
\hat{y}^\top S \hat{y} &= \sum_{i,j} \hat{y}_i \int_0^1 \phi_i' \phi_j' \, dx \, \hat{y}_j \\
&= \int_0^1 \left( \sum_i \hat{y}_i \phi_i' \right)^2 dx \\
&= \int_0^1 (y')^2 \, dx \geq 0
\end{aligned}
$$

with equality **only if** $y = 0$.

**Therefore:** $S$ is SPD (symmetric positive definite), which guarantees unique solutions. $\quad \checkmark$

---

## 8. Sparse Matrix Structure

### 8.1 Sparsity Pattern

**Key observation:**
$$
S_{ij} = \int_0^1 \phi_i' \phi_j' \, dx
$$

is **only non-zero if** $\operatorname{supp}(\phi_i) \cap \operatorname{supp}(\phi_j) \neq \emptyset$

Since $\phi_i$ has support only on $[x_{i-1}, x_{i+1}]$, most entries of $S$ are zero!

**Result:** $S$ is **sparse** (tridiagonal for 1D problems).

For piecewise linear elements on uniform mesh:

$$
S = \frac{1}{h} \begin{bmatrix}
2 & -1 & & & \\
-1 & 2 & -1 & & \\
& -1 & 2 & -1 & \\
& & \ddots & \ddots & \ddots
\end{bmatrix}
$$

**This looks familiar!** Compare to FDM discretization of $-u''$.

---

## 9. Gaussian Quadrature

### 9.1 Quadrature Rules

To actually compute $S_{ij}$ and $b_i$, we need numerical integration.

**Definition:** A **quadrature rule** is a set of $N_q$ points and weights:
- $w_i$ = quadrature weights
- $x_i$ = quadrature points

such that:
$$
\int_{-1}^1 f(x) \, dx = \sum_{i=1}^{N_q} w_i f(x_i)
$$

that is **exact for some class of functions**.

### 9.2 Gauss-Legendre Quadrature

**Exact for polynomial** $f$ of degree $\leq 2(N_q - 1)$.

- $x_i$ are **zeros of Legendre polynomials** $P_{N_q}$
- $w_i = \frac{2}{(1 - x_i^2)[P_{N_q}'(x_i)]^2}$

| $N_q$ | $x_i$ | $w_i$ |
|-------|-------|-------|
| 1 | $0$ | $2$ |
| 2 | $\pm 1/\sqrt{3}$ | $1, 1$ |
| 3 | $0, \pm\sqrt{3/5}$ | $8/9, 5/9, 5/9$ |

### 9.3 Change of Variables

To map the domain of integration from $[-1, 1]$ to $[a, b]$:

**$u$-substitution:**
$$
u = 2\left(\frac{x - a}{b - a}\right) - 1
$$

$$
\begin{aligned}
\int_a^b f(x) \, dx &= \int_{-1}^1 f\left(\frac{b-a}{2}u + \frac{a+b}{2}\right) \frac{b-a}{2} \, du \\
&= \frac{b-a}{2} \sum_{i=1}^{N_q} w_i f\left(\frac{b-a}{2}x_i + \frac{a+b}{2}\right)
\end{aligned}
$$

---

## 10. Connection to Energy Minimization

### 10.1 The Ritz Method

From Klaus Johnson:

Consider the **energy functional**:
$$
F(v) = \frac{1}{2}(v', v') - (f, v) = \frac{1}{2}\int_0^1 (v')^2 \, dx - \int_0^1 f v \, dx
$$

**Ritz method (R):** 
$$
\min_{v \in V} F(v)
$$

### 10.2 Equivalence to Galerkin Method

To solve, use the first variation:

**Step 1:** Compute variation:
$$
\begin{aligned}
(\delta_v F(v), \delta v) &= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon}[F(v + \varepsilon \delta v) - F(v)] \\
&= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon}\left[\frac{1}{2}(v' + \varepsilon \delta v', v' + \varepsilon \delta v') - \frac{1}{2}(v', v')\right. \\
&\quad\quad\quad\quad\quad\quad \left. - (f, v + \varepsilon \delta v) + (f, v)\right]
\end{aligned}
$$

**Step 2:** Simplify:
$$
= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon}\left[\varepsilon(v', \delta v') + \frac{\varepsilon^2}{2}(\delta v', \delta v') - \varepsilon(f, \delta v)\right]
$$

**Step 3:** Take limit:
$$
= (v', \delta v') - (f, \delta v)
$$

**Step 4:** Set to zero for any $\delta v$:
$$
\boxed{(v', \delta v') = (f, \delta v) \quad \text{for any } \delta v}
$$

**This is exactly the Galerkin formulation (G)!**

**Remark:** Not all PDEs can be written as an energy minimization. But when they can, Galerkin = Ritz.

---

## 11. Galerkin Orthogonality and Optimal Error

### 11.1 Galerkin Orthogonality

If the exact solution is $u \in V$ and the FEM solution is $u_h \in V_h$:

$$
\begin{aligned}
(u', v') &= (f, v) \quad \text{for any } v \in V \\
(u_h', v') &= (f, v) \quad \text{for any } v \in V_h
\end{aligned}
$$

**Subtract:**
$$
\boxed{(u' - u_h', v') = 0 \quad \text{for any } v \in V_h}
$$

This is **Galerkin orthogonality**: the error is orthogonal to the approximation space!

### 11.2 Theorem: Optimal Error in $H^1$ Norm

**For any $v \in V_h$:**
$$
\boxed{\|(u - u_h)'\| \leq \|(u - v)'\|}
$$

**Interpretation:** The FEM solution is the **best possible approximation** in your function space!

### 11.3 Proof

Let $v \in V_h$ and $w = u_h - v \in V_h$.

**Step 1:** Expand norm:
$$
\|(u - u_h)'\|^2 = ((u - u_h)', (u - u_h)')
$$

**Step 2:** Add and subtract $(u - v)'$:
$$
= ((u - u_h)', (u - v)') + \underbrace{((u - u_h)', w')}_{= 0 \text{ by Galerkin orthogonality}}
$$

**Step 3:** Simplify:
$$
= ((u - u_h)', (u - v)')
$$

**Step 4:** Apply Cauchy-Schwarz:
$$
\leq \|(u - u_h)'\| \cdot \|(u - v)'\|
$$

**Step 5:** Divide both sides by $\|(u - u_h)'\|$:
$$
\|(u - u_h)'\| \leq \|(u - v)'\| \quad \checkmark
$$

**This holds for ANY $v \in V_h$, so the FEM solution minimizes the error!**

---

## 12. Handling Neumann Boundary Conditions

### 12.1 Problem Statement

Consider:
$$
\begin{cases}
-u'' = 1 & \text{on } (0,1) \\
u(0) = 0 & \text{(Dirichlet)} \\
u'(1) = 0 & \text{(Neumann)}
\end{cases}
$$

**Exact solution:**
$$
u(x) = x\left(1 - \frac{x}{2}\right)
$$

### 12.2 Weak Form with Neumann BC

Return to the original weak form **before** imposing boundary conditions:

$$
-\int_0^1 u'' v \, dx = \int_0^1 f v \, dx
$$

**Integrate by parts:**
$$
\int_0^1 u' v' \, dx - (u'(1) v(1) - u'(0) v(0)) = \int_0^1 f v \, dx
$$

**If $u'(1) = g$ (Neumann condition), substitute it in:**

$$
\boxed{\int_0^1 u' v' \, dx = \int_0^1 f v \, dx + g \cdot v(1)}
$$

**Key observation:** Neumann BCs appear **naturally** in the weak formulation as boundary terms. We don't need to enforce them explicitly—they emerge from integration by parts!

**Modified linear system:**
$$
S \hat{u} = b + g \cdot e_N
$$

where $e_N$ is the vector with 1 in the last position (corresponding to $x = 1$) and 0 elsewhere.

---

## 13. Summary

This lecture covered:

1. **Transition from FDM to FEM**: motivation and historical context
2. **Weak (Galerkin) formulation**: reducing regularity requirements
3. **Function spaces** $L^2$, $H^1$, $H_0^1$ for well-posed weak problems
4. **Finite-dimensional approximation** with basis functions (shape functions)
5. **Stiffness matrix** $S$ and **load vector** $b$ assembly
6. **Uniqueness** and **SPD property** of the stiffness matrix
7. **Sparsity** structure for efficient computation
8. **Gaussian quadrature** for numerical integration
9. **Equivalence** between Galerkin method and energy minimization (Ritz method)
10. **Galerkin orthogonality** and **optimal error estimates**
11. **Natural handling** of Neumann boundary conditions

**Key Takeaway:** The finite element method provides a principled, flexible framework for solving PDEs on complex domains with various boundary conditions. By working in weak form and choosing appropriate function spaces, we obtain optimal approximations with provable error bounds. The Galerkin orthogonality property ensures that our discrete solution is the best possible projection of the true solution onto our finite-dimensional space. Moreover, the natural emergence of Neumann boundary conditions from the weak formulation, combined with the connection to energy minimization, reveals the deep geometric structure underlying FEM—a structure we'll exploit when building learnable PDE solvers.

