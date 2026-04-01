# Spatial Discretization via the Discrete Action Principle

**Date:** February 19, 2025  
**Topics:** Discrete Lagrangian mechanics, discrete integration by parts, Noether constraints for stencils, polynomial reproduction via symmetry

---

## 0. Overview

This lecture completes the bridge between classical mechanics and numerical methods by applying the **discrete action principle** to spatial discretization of PDEs. We previously developed the Lagrangian formulation in continuous space-time; now we discretize the spatial domain while keeping time continuous, deriving finite difference stencils from variational principles.

The key innovation is recognizing that **discrete stencils must respect discrete analogs of continuous symmetries**. Just as Noether's theorem connects continuous symmetries to conserved quantities, discrete symmetries impose algebraic constraints on stencil coefficients. For the wave equation, **translation invariance** requires stencils to sum to zero, while **polynomial reproduction** (the discrete analog of Taylor consistency) imposes additional constraints on higher-order terms.

We introduce **discrete integration by parts** using shift operators, which elegantly mirrors the continuous integration by parts used to derive Euler-Lagrange equations. This leads to a systematic framework: discretize the Lagrangian, apply discrete variational calculus, and obtain stencil weights from polynomial reproduction constraints. The resulting stencils automatically inherit symmetry properties from the continuous Lagrangian.

This approach unifies our previous ad-hoc stencil constructions (centered differences, compact schemes) under a single variational principle. It also reveals a deep connection: searching for minimal-norm stencils with polynomial reproduction (Lecture 6) is equivalent to applying the discrete action principle with appropriate symmetry constraints.

**Pedagogical significance:** This material demonstrates how mathematical physics principles (symmetries, variational principles) translate directly into numerical algorithms. Understanding this connection is essential for designing structure-preserving numerical methods and physics-informed neural networks that respect conservation laws at the discrete level.

---

## 1. Discretizing the Wave Equation Lagrangian

### 1.1 Continuous Action

Recall the action for the 1D wave equation:

$$
S = \int_0^T \int_0^L \left[\frac{1}{2}(\partial_t u)^2 - \frac{c^2}{2}(\partial_x u)^2\right] dx\, dt
$$

where:
- $\frac{1}{2}(\partial_t u)^2$ is kinetic energy density
- $\frac{c^2}{2}(\partial_x u)^2$ is potential energy density (elastic strain)

### 1.2 Piecewise Constant Basis Functions

Define **piecewise constant extension** of a grid function $u_i(t)$ on nodes $x_i = ih$ for $i = 1, \ldots, N$:

$$
\phi_i(x) = \mathbb{I}_{(x_i - h/2, x_i + h/2)}(x) = \begin{cases}
1 & \text{if } x \in (x_i - h/2, x_i + h/2) \\
0 & \text{else}
\end{cases}
$$

**Discrete representation:**

$$
u(x, t) = \sum_{i=1}^N u_i(t) \phi_i(x)
$$

**Discrete spatial derivative:**

$$
\partial_x u(x, t) = \sum_{i=1}^N \underbrace{\frac{1}{h}(\alpha u_{i-1}(t) + \beta u_i(t) + \gamma u_{i+1}(t))}_{D_h u_i} \phi_i(x)
$$

**Key observation:**

$$
\int_0^L u(x, t)\, dx = \sum_{i=1}^N h u_i(t)
$$

Each basis function $\phi_i$ contributes area $h$ (trapezoid rule).

### 1.3 Discrete Lagrangian

Selecting $q_i = u_i$ as generalized coordinates:

$$
L = \sum_{i=1}^N \left[\frac{1}{2}\dot{q}_i^2 h - \frac{c^2}{2}(D_h q_i)^2 h\right]
$$

where $D_h q_i$ is the spatial finite difference stencil.

---

## 2. Discrete Noether's Theorem

### 2.1 Symmetries in the Discrete Lagrangian

**Important:** In light of Noether's theorem, we analyze symmetries:

**(1) Time translation invariance:**

Time only enters via $\dot{q}_i$, so the Lagrangian is time-translation invariant.

**Conserved quantity (Hamiltonian):**

$$
H = \partial_{\dot{q}} L \cdot \dot{q} - L
$$

where the **conjugate momentum** is:

$$
p_i = \frac{\partial L}{\partial \dot{q}_i} = \dot{q}_i h
$$

**(2) Spatial translation invariance:**

$q \to q + \delta q$ (shift all grid values by a constant) is a symmetry **only if** $D_h \delta q = 0$ for constant $\delta q$.

**Conserved quantity (total momentum):**

$$
\text{const.} = \sum_{i=1}^N p_i = \sum_{i=1}^N h\dot{q}_i
$$

This is the total discrete momentum (integral of velocity).

**Critical constraint:** Translation invariance requires $D_h(\text{const.}) = 0$.

---

## 3. Discrete Integration by Parts

### 3.1 Shift Operator Formalism

**Lemma:** Consider two periodic grid functions $u_i, v_i$ satisfying $u_0 = u_N, v_0 = v_N$.

**Step 1:** Change index in summation:

$$
\begin{aligned}
\sum_{i=1}^N x_i y_{i+\alpha} &= \sum_{j=1+\alpha}^{N+\alpha} x_{j-\alpha} y_j \quad (j = i + \alpha) \\
&= \sum_{j=1}^N x_{j-\alpha} y_j \quad \text{(periodicity, shift limits)}
\end{aligned}
$$

**Step 2:** Relabel $j \to i$:

$$
\sum_{i=1}^N x_i y_{i+\alpha} = \sum_{i=1}^N x_{i-\alpha} y_i
$$

**Compact notation using shift operators:**

Define $E^\alpha u_i = u_{i+\alpha}$ (forward shift by $\alpha$). Then:

$$
\langle x, E^\alpha y \rangle = \langle E^{-\alpha} x, y \rangle
$$

where $\langle \cdot, \cdot \rangle$ denotes the discrete inner product $\sum_i x_i y_i$.

### 3.2 Adjoint Operator

Consider a stencil operator:

$$
D_h u_i = \sum_{k=-m}^m c_k E^k u_i
$$

**Definition:** The **adjoint operator** is:

$$
D_h^* u_i = \sum_{k=-m}^m c_{-k} E^k u_i
$$

(flip stencil coefficients: $c_k \to c_{-k}$).

**Theorem:**

$$
\langle D_h^* u, \delta u \rangle = \langle u, D_h \delta u \rangle
$$

**Proof:**

$$
\begin{aligned}
\langle x, D_h u \rangle &= \left\langle x, \sum_{k=-m}^m c_k E^k u \right\rangle \\
&= \sum_{k=-m}^m c_k \langle x, E^k u \rangle \\
&= \sum_{k=-m}^m c_k \langle E^{-k} x, u \rangle \quad \text{(shift property)} \\
&= \left\langle \sum_{k=-m}^m c_k E^{-k} x, u \right\rangle \\
&= \left\langle \sum_{k=-m}^m c_{-k} E^k x, u \right\rangle \quad \text{(reindex } k \to -k\text{)} \\
&= \langle D_h^* x, u \rangle
\end{aligned}
$$

This is the **discrete analog of integration by parts**!

---

## 4. Discrete Variational Calculus

### 4.1 Continuous Laplacian Example

**Continuous:**

$$
F[u] = \frac{1}{2}\int |\nabla u|^2\, dx
$$

**Step 1:** Take variation:

$$
\begin{aligned}
(\delta_u F, \delta u) &= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon}(F[u + \varepsilon\delta u] - F[u]) \\
&= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} \int \left[\frac{1}{2}|\nabla u|^2 + \varepsilon \nabla u \cdot \nabla \delta u + \frac{\varepsilon^2}{2}|\nabla\delta u|^2 - \frac{1}{2}|\nabla u|^2\right] dx \\
&= \int \nabla u \cdot \nabla \delta u\, dx
\end{aligned}
$$

**Step 2:** Integration by parts:

$$
(\delta_u F, \delta u) = -\int (\nabla^2 u) \delta u\, dx = -(\nabla^2 u, \delta u)
$$

**Result:**

$$
\delta_u F = -\nabla^2 u
$$

### 4.2 Discrete Laplacian Example

**Discrete functional:**

$$
F_h[u] = \frac{1}{2}\int \left(\sum_i D_0 u_i \phi_i(x)\right)^2 dx = \sum_i \frac{h}{2}(D_0 u_i)^2
$$

where $D_0 u_i = \frac{u_{i+1} - u_{i-1}}{2h}$ (centered difference).

**Step 1:** Take variation:

$$
\begin{aligned}
\langle \delta_{u_i} F_h[u], \delta u_i \rangle &= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} \sum_i \left[\frac{h}{2}(D_0 u_i + \varepsilon D_0 \delta u_i)^2 - \frac{h}{2}(D_0 u_i)^2\right] \\
&= \sum_i h \cdot D_0 u_i \cdot D_0 \delta u_i \\
&= \langle D_0 u, D_0 \delta u \rangle
\end{aligned}
$$

**Step 2:** Apply discrete integration by parts:

$$
\langle D_0 u, D_0 \delta u \rangle = \langle D_0^* D_0 u, \delta u \rangle
$$

**Result:**

$$
\delta_{u_i} F_h = D_0^* D_0 u
$$

This is the discrete Laplacian!

---

## 5. Deriving the Discrete Wave Equation

### 5.1 Split the Lagrangian

$$
L = \underbrace{\sum_i \frac{h}{2}\dot{q}_i^2}_{S_1} - \underbrace{\sum_i \frac{c^2 h}{2}(D_h q_i)^2}_{S_2}
$$

**Variation of $S_1$ (kinetic term):**

$$
\begin{aligned}
(\delta_{q_i} S_1, \delta q_i) &= \int_0^T \sum_i h\dot{q}_i \cdot \delta q_i\, dt \\
&= \int_0^T -\sum_i h\ddot{q}_i \delta q_i\, dt \quad \text{(integration by parts in time)} \\
\Rightarrow \delta_{q_i} S_1 &= -h\ddot{q}_i
\end{aligned}
$$

**Variation of $S_2$ (potential term):**

From the discrete Laplacian example:

$$
\delta_{q_i} S_2 = -c^2 h \cdot D_h^* D_h q_i
$$

**Euler-Lagrange equation:**

$$
\delta_{q_i} S = \delta_{q_i} S_1 + \delta_{q_i} S_2 = 0
$$

$$
-h\ddot{q}_i - c^2 h \cdot D_h^* D_h q_i = 0
$$

$$
\ddot{q}_i = -c^2 D_h^* D_h q_i
$$

**Remark:** While tedious, no thought was needed to reach this point — just careful application of calculus!

---

## 6. Constraints for a Good Stencil

### 6.1 Summary of Requirements

We now have **two constraints** for a good stencil:

**(1) Noether constraint (translation invariance):**

$$
D_h(\text{const.}) = 0 \quad \Rightarrow \quad \sum_k c_k = 0
$$

**Derivation:**

$$
D_h \delta q = \sum_k c_k E^k \delta q = \delta q \sum_k c_k
$$

If $\delta q = \text{const.}$, then $D_h \delta q = 0$ requires $\sum_k c_k = 0$.

**(2) Polynomial reproduction:**

Does $D_h^* D_h q$ give a stable prediction of $\nabla^2 u$?

Recall from earlier lectures: this is true if $D_h^* D_h p = \nabla^2 p$ for any quadratic polynomial $p$.

Let's translate this into algebraic constraints.

---

## 7. Polynomial Reproduction Constraints

### 7.1 Stencil Arithmetic

**General stencil:**

$$
D_h u_i = \frac{1}{h}\sum_j c_j u_{i+j}
$$

**Adjoint:**

$$
D_h^* v_j = \frac{1}{h}\sum_k c_{-k} v_{j+k}
$$

**Composed operator:**

$$
D_h^* D_h u_i = \frac{1}{h^2} \sum_{j,k} c_j c_{-k} u_{i+j+k}
$$

### 7.2 Three-Point Stencil

Set:

$$
c_{-1} = \alpha, \quad c_0 = \beta, \quad c_1 = \gamma
$$

**Build table of products $c_j c_{-k}$ for indices $i+j+k$:**

| $k \backslash j$ | -1 | 0 | 1 |
|:---:|:---:|:---:|:---:|
| **-1** | $\alpha^2$ (i-2) | $\alpha\beta$ (i-1) | $\alpha\gamma$ (i) |
| **0** | $\beta\alpha$ (i-1) | $\beta^2$ (i) | $\beta\gamma$ (i+1) |
| **1** | $\gamma\alpha$ (i) | $\gamma\beta$ (i+1) | $\gamma^2$ (i+2) |

**Sum along diagonals:**

$$
D_h^* D_h u_i = \frac{1}{h^2}\left[\alpha\gamma u_{i-2} + (\alpha\beta + \gamma\beta) u_{i-1} + (\alpha^2 + \beta^2 + \gamma^2) u_i + (\beta\alpha + \beta\gamma) u_{i+1} + \alpha\gamma u_{i+2}\right]
$$

Simplifying:

$$
D_h^* D_h u_i = \frac{1}{h^2}\left[\alpha\gamma u_{i-2} + \beta(\alpha + \gamma) u_{i-1} + (\alpha^2 + \beta^2 + \gamma^2) u_i + \beta(\alpha + \gamma) u_{i+1} + \alpha\gamma u_{i+2}\right]
$$

### 7.3 Four Polynomial Reproduction Constraints

**(1) Noether constraint (constant reproduction):**

$$
\alpha + \beta + \gamma = 0
$$

**(2) Constant reproduction (apply to $u_i = 1$):**

$$
2\alpha\gamma + 2\beta(\alpha + \gamma) + \alpha^2 + \beta^2 + \gamma^2 = 0
$$

**(3) Linear reproduction (apply to $u_i = ih$):**

Take $u_{i-1} = (i-1)h, u_i = ih, u_{i+1} = (i+1)h$, etc.:

$$
-2\alpha\gamma h - \beta(\alpha + \gamma)h + 0 + \beta(\alpha + \gamma)h + 2\alpha\gamma h = 0
$$

This is **automatically satisfied** (terms cancel).

**(4) Quadratic reproduction (apply to $u_i = (ih)^2$, want $\nabla^2 u = 2$):**

$$
\alpha\gamma(2h)^2 + \beta(\alpha + \gamma)h^2 + 0 + \beta(\alpha + \gamma)h^2 + \alpha\gamma(2h)^2 = -2h^2
$$

Simplifying:

$$
8\alpha\gamma + 2\beta(\alpha + \gamma) = -2
$$

### 7.4 Solutions

Solving (1)–(4) is **non-unique**. Let's try some simplifications:

#### (A) Asymmetric Stencil

Set $\gamma = 0$:

From (1): $\alpha = -\beta$

From (4): $-2\beta^2 = -2 \Rightarrow \beta = \pm 1$

Taking $\beta = 1$:

$$
\alpha = -1, \quad \beta = 2, \quad \gamma = -1
$$

$$
D_h u = \frac{1}{h^2}(-u_{i-1} + 2u_i - u_{i+1})
$$

We recover the **3-point Laplacian** $D_+ D_- = D_- D_+$! ✓

#### (B) Symmetric Stencil

Set $\alpha = \gamma$:

From (1): $\alpha = -\beta/2$

From (4): $8\alpha^2 + 2\beta \cdot 2\alpha = -2$

Substitute $\beta = -2\alpha$:

$$
8\alpha^2 - 4\alpha \cdot 2\alpha = -2
$$

$$
8\alpha^2 - 8\alpha^2 = -2 \quad \Rightarrow \quad 0 = -2
$$

**No real-valued solutions!** Symmetric 3-point stencils cannot satisfy polynomial reproduction up to degree 2.

#### (C) Skew-Symmetric Stencil

Set $\alpha = -\gamma$:

From (1): $\beta = 0$

From (4): $-8\alpha^2 = -2 \Rightarrow \alpha^2 = 1/4 \Rightarrow \alpha = \pm 1/2$

Taking $\alpha = 1/2$:

$$
\alpha = \frac{1}{2}, \quad \beta = 0, \quad \gamma = -\frac{1}{2}
$$

$$
D_h u = \frac{1}{h^2}\left[-\frac{1}{4}u_{i-2} + \frac{1}{2}u_i - \frac{1}{4}u_{i+2}\right]
$$

This is a **5-point compact stencil** using only nodes $i-2, i, i+2$.

**Verification of constraint (2):**

$$
2\alpha\gamma + \beta^2 = 2 \cdot \frac{1}{2} \cdot \left(-\frac{1}{2}\right) + 0^2 = -\frac{1}{2} + \frac{1}{4} + \frac{1}{4} = 0 \quad \checkmark
$$

Wait, let me recalculate:

$$
2\alpha\gamma + 2\beta(\alpha + \gamma) + \alpha^2 + \beta^2 + \gamma^2
$$

$$
= 2 \cdot \frac{1}{2} \cdot \left(-\frac{1}{2}\right) + 0 + \left(\frac{1}{2}\right)^2 + 0 + \left(-\frac{1}{2}\right)^2
$$

$$
= -\frac{1}{2} + \frac{1}{4} + \frac{1}{4} = 0 \quad \checkmark
$$

---

## Summary

This lecture unified variational principles with numerical analysis to derive finite difference stencils from symmetry and polynomial reproduction:

1. **Discrete action principle** extends the Lagrangian formulation to spatial discretization; choose piecewise constant basis functions $\phi_i(x)$ and derive discrete Lagrangian $L = \sum_i [T_i - V_i]h$

2. **Discrete Noether's theorem** translates continuous symmetries to algebraic constraints on stencils:
   - Time translation → Energy conservation
   - Spatial translation → $\sum_k c_k = 0$ (stencil sums to zero)

3. **Discrete integration by parts** using shift operators $E^\alpha$ provides the discrete analog of integration by parts: $\langle D_h^* u, v \rangle = \langle u, D_h v \rangle$

4. **Polynomial reproduction constraints** ensure discrete consistency with continuous operators:
   - Constant reproduction (order 0)
   - Linear reproduction (order 1, automatically satisfied)
   - Quadratic reproduction (order 2) → accuracy constraint

5. **Classification of 3-point stencils:**
   - Asymmetric: Standard 3-point Laplacian $(-1, 2, -1)/h^2$
   - Symmetric: No real solutions satisfying polynomial reproduction
   - Skew-symmetric: 5-point compact stencil $(-1/4, 0, 1/2, 0, -1/4)/h^2$ on nodes $\{i-2, i, i+2\}$

**Key Takeaway:** Finite difference stencils are not arbitrary — they emerge naturally from applying the discrete action principle with appropriate symmetry constraints. The variational framework guarantees that discrete schemes inherit conservation laws from continuous physics. Noether's theorem provides the algebraic constraints (e.g., $\sum c_k = 0$), while polynomial reproduction ensures accuracy. This unifies our previous ad-hoc constructions under a single principle and provides a blueprint for designing novel structure-preserving schemes.

**Connection to machine learning:** When learning stencils from data, we should encode these symmetry constraints (translation invariance, polynomial reproduction) as architectural inductive biases. This reduces the hypothesis space, improves sample efficiency, and guarantees physically meaningful solutions.

**Next steps:** Apply these principles to more complex PDEs (nonlinear equations, systems, higher dimensions) and extend to space-time discretization where both spatial and temporal symmetries interact.
