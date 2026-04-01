# Lagrangian Mechanics, Functional Derivatives, and Noether's Theorem

**Date:** February 17, 2025  
**Topics:** Functional derivatives, Euler-Lagrange equations, principle of least action, Legendre transform, Noether's theorem

---

## 0. Overview

This lecture introduces **Lagrangian mechanics**, the third pillar of classical mechanics alongside Newtonian and Hamiltonian formulations. While all three describe the same physics, they emphasize different aspects and offer complementary tools for analysis and computation.

The Lagrangian approach centers on **optimization**: physical systems evolve along paths that extremize the **action functional** $S[u] = \int_0^T L\, dt$, where $L = T - V$ is the difference between kinetic and potential energy. This **principle of least action** transforms mechanics into a variational problem, making it natural to enforce constraints (via Lagrange multipliers) and identify conserved quantities (via Noether's theorem).

To work with action functionals, we need calculus in infinite-dimensional function spaces. We introduce two approaches: the **Fréchet derivative** (abstract, general) and the **Gâteaux derivative** (practical, directional). These lead to the **Euler-Lagrange equations**, which systematically convert variational principles into differential equations. We then connect Lagrangian and Hamiltonian mechanics via the **Legendre transform**, showing they are dual perspectives on the same physics.

Finally, **Noether's theorem** reveals the profound connection between symmetries and conservation laws: every continuous symmetry of the Lagrangian corresponds to a conserved quantity. This explains *why* momentum, energy, and angular momentum are conserved and guides the design of symmetry-respecting neural network architectures.

**Pedagogical context:** Understanding the optimization-centric Lagrangian viewpoint is essential for modern physics-informed machine learning. Many recent methods (Physics-Informed Neural Networks, Neural ODEs, Hamiltonian Neural Networks) implicitly use variational principles and functional derivatives to encode physical constraints.

---

## 1. Three Perspectives on Mechanics

### 1.1 Comparison of Formalisms

**Newtonian Mechanics:** $F = m\ddot{x}$

- **Strengths:** Good for force balance, intuitive
- **Weaknesses:** Hard to associate with invariants, coordinate-dependent

**Hamiltonian Mechanics:** $\dot{x} = J\nabla H$

- **Strengths:** Energy-centric, good for designing schemes, natural for phase space analysis
- **Weaknesses:** Requires doubling dimension ($q, p$), Legendre transform can be non-trivial

**Lagrangian Mechanics:** $x = \arg\min_{\tilde{x}} S(\tilde{x})$

- **Strengths:** Optimization-centric, easy constraint treatment, natural symmetry analysis
- **Weaknesses:** Requires functional calculus, indirect (no explicit time evolution)

---

## 2. Lagrangian Mechanics: Basic Setup

### 2.1 Definitions

**Lagrangian density:**

$$
L = \underbrace{T}_{\text{kinetic}} - \underbrace{V}_{\text{potential}}
$$

For continuous systems (fields):

$$
L = \int_\Omega \mathcal{L}[u]\, dx
$$

where $\mathcal{L}$ is the **Lagrangian density**.

**Action:**

$$
S = \int_0^T L\, dt = \int_0^T \int_\Omega \mathcal{L}[u]\, dx\, dt
$$

### 2.2 Principle of Least Action

Informally, the path that a system will evolve through for fixed initial and final states will be the $u(t)$ which minimizes the action:

$$
u = \arg\min_{\hat{u}} \int_0^T \int_\Omega \mathcal{L}[u]\, dx\, dt
$$

**Challenge:** To make sense of this, we need to minimize expressions like $S$. In standard calculus, we take a derivative of a function $f(x)$, set it to zero to find extremal points. Here though:
- **Input:** $x$ is a coordinate, $f$ is a function
- **Variational calculus:** $u$ is a function, $S$ is a **functional**

A **functional** maps function spaces to real numbers:

$$
S: V \to \mathbb{R}, \quad V \text{ a space of functions}
$$

i.e., the input space is now **infinite-dimensional**.

---

## 3. Functional Derivatives

### 3.1 Fréchet Derivative

Let $V$ be a function space, $F: V \to \mathbb{R}$. The **differential** $\delta F[u]$ is defined as follows: for $\delta u \in V$,

$$
F[u + \delta u] - F[u] = \delta F[u] + \varepsilon \|\delta u\|
$$

such that $\lim_{\varepsilon \to 0} \varepsilon \|\delta u\| = 0$.

**Interpretation:** This is analogous to the total derivative in finite dimensions, but the norm $\|\cdot\|$ is now in a function space (e.g., $L^2$ norm).

**Note:** This is a challenging definition to work with in practice.

### 3.2 Gâteaux Derivative

**Idea:** Introduce a scalar parameter and use the vector calculus definition of a directional derivative.

$$
\begin{aligned}
\delta F[u, \delta u] &= \lim_{\varepsilon \to 0} \frac{F[u + \varepsilon \delta u] - F[u]}{\varepsilon} \\
&= \left.\frac{d}{d\varepsilon} F[u + \varepsilon \delta u]\right|_{\varepsilon = 0}
\end{aligned}
$$

This is analogous to the **Gâteaux derivative** or **directional derivative** in infinite dimensions.

**Practical computation:**

Given $F[u] = \int_\Omega f(x, u, D^\alpha u)\, dx$, the Gâteaux derivative is defined via: for all $\delta u$ vanishing on the boundary,

$$
(\delta F, \delta u) = \lim_{\varepsilon \to 0} \frac{F[u + \varepsilon \delta u] - F[u]}{\varepsilon}
$$

### 3.3 Example: Kinetic Energy

Let $K = \frac{1}{2}\int_\Omega \rho u^2\, dx$. Take variation with respect to $u$:

$$
\begin{aligned}
(\delta_u K, \delta u) &= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} \int_\Omega \left[\frac{\rho(u + \varepsilon \delta u)^2}{2} - \frac{\rho u^2}{2}\right] dx \\
&= \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} \int_\Omega \rho\left[\frac{u^2}{2} + \varepsilon u \delta u + \frac{\varepsilon^2}{2}(\delta u)^2 - \frac{u^2}{2}\right] dx \\
&= \lim_{\varepsilon \to 0} \int_\Omega \left[\rho u \delta u + \frac{\varepsilon}{2}\rho (\delta u)^2\right] dx \\
&= \int_\Omega \rho u \delta u\, dx \\
&= (\rho u, \delta u)
\end{aligned}
$$

And so we identify:

$$
\delta_u K = \rho u \quad \text{(the variation)}
$$

**Be careful:** Physicists will often write $\delta u F = \rho u$ (the "variation"). Only makes sense for continuous functionals.

### 3.4 Further Properties

Denote $\frac{\delta}{\delta u} F[u] = \delta_u F[u]$.

**Linearity:**

$$
\frac{\delta}{\delta u}(\lambda F[u] + \mu G[u]) = \lambda \delta_u F[u] + \mu \delta_u G[u]
$$

**Product rule:**

$$
\frac{\delta}{\delta u}(F[u] G[u]) = \delta_u F \cdot G + F \cdot \delta_u G
$$

**Chain rule:**

$$
\frac{\delta}{\delta u}(F[g(u)]) = \frac{\delta F[g(u)]}{\delta g(u)} \cdot \frac{dg(u)}{du}
$$

---

## 4. The Euler-Lagrange Equations

### 4.1 General Derivation

Assume a generic functional density:

$$
F[u] = \int_\Omega f(x, u, \nabla u)\, dx
$$

**Step 1:** Take Gâteaux derivative:

$$
\begin{aligned}
(\delta_u F, \delta u) &= \left[\frac{d}{d\varepsilon} \int_\Omega f(x, u + \varepsilon \delta u, \nabla u + \varepsilon \nabla \delta u)\, dx\right]_{\varepsilon = 0} \\
&= \int_\Omega \left[\frac{\partial f}{\partial u} \delta u + \frac{\partial f}{\partial \nabla u} \cdot \nabla \delta u\right] dx
\end{aligned}
$$

**Step 2:** Apply integration by parts to the second term:

$$
\int_\Omega \frac{\partial f}{\partial \nabla u} \cdot \nabla \delta u\, dx = \int_{\partial\Omega} \frac{\partial f}{\partial \nabla u} \cdot \delta u\, dA - \int_\Omega \nabla \cdot \frac{\partial f}{\partial \nabla u} \cdot \delta u\, dx
$$

**Step 3:** Assume vanishing boundary conditions ($\delta u = 0$ on $\partial\Omega$):

$$
(\delta_u F, \delta u) = \int_\Omega \left[\frac{\partial f}{\partial u} - \nabla \cdot \frac{\partial f}{\partial \nabla u}\right] \delta u\, dx
$$

**Step 4:** Since this must hold for all $\delta u$, we identify:

$$
\delta_u F = \frac{\partial f}{\partial u} - \sum_i \frac{\partial}{\partial x_i} \frac{\partial f}{\partial(\partial_i u)}
$$

**Euler-Lagrange equation:**

$$
\delta_u F = 0 \quad \Rightarrow \quad \frac{\partial f}{\partial u} - \nabla \cdot \frac{\partial f}{\partial \nabla u} = 0
$$

**Practical note:** We can just apply this as a formula and skip all the limits.

### 4.2 Higher-Order Derivatives

For different forms of $f$, we get different equations. For example:

$$
F[u] = \int_0^1 f(x, u, \partial_x u, \partial_{xx} u)\, dx
$$

Then:

$$
\delta_u F = \frac{\partial f}{\partial u} - \frac{d}{dx}\frac{\partial f}{\partial(\partial_x u)} + \frac{d^2}{dx^2}\frac{\partial f}{\partial(\partial_{xx} u)}
$$

### 4.3 Application: Principle of Least Action

For the action functional:

$$
S[u] = \int_a^b \int_\Omega \mathcal{L}(x, u, \dot{u})\, dx\, dt
$$

The path which admits an extremal value of $S$ satisfies:

$$
\delta_u S[u] = 0
$$

Applying the Euler-Lagrange equations:

$$
\frac{\partial \mathcal{L}}{\partial u} - \frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{u}} = 0
$$

### 4.4 Example: Linear Pendulum

**Kinetic energy:**

$$
T = \frac{1}{2}m v^2 = \frac{1}{2}m L^2 \dot{\theta}^2
$$

**Potential energy:**

$$
\begin{aligned}
U &= mgh \\
h &= L(1 - \cos\theta) \approx \frac{1}{2}L\theta^2 \quad \text{(small angle)}
\end{aligned}
$$

**Lagrangian:**

$$
L = \frac{1}{2}m L^2 \dot{\theta}^2 - \frac{1}{2}mgL\theta^2
$$

**Compute partial derivatives:**

$$
\begin{aligned}
\partial_\theta L &= -mgL\theta \\
\partial_{\dot{\theta}} L &= mL^2 \dot{\theta}
\end{aligned}
$$

**Euler-Lagrange equation:**

$$
\frac{d}{dt}(mL^2\dot{\theta}) = -mgL\theta
$$

$$
mL^2\ddot{\theta} = -mgL\theta \quad \Rightarrow \quad \ddot{\theta} = -\frac{g}{L}\theta
$$

Recovering the harmonic oscillator equation. ✓

**Note:** Depending on $\mathcal{L}$, we get different versions of the equations of motion.

---

## 5. Legendre Transform

### 5.1 Definition

A technique to switch between the Lagrangian and Hamiltonian descriptions.

Given a Lagrangian:

$$
L(q_1, \ldots, q_N, \dot{q}_1, \ldots, \dot{q}_N)
$$

The **Legendre transform** induces the **conjugate momenta** $p_1, \ldots, p_N$:

$$
\begin{cases}
H(q_1, \ldots, q_N, p_1, \ldots, p_N) = \sum_i p_i \dot{q}_i - L(q_1, \ldots, q_N, \dot{q}_1, \ldots, \dot{q}_N) \\
p_i = \frac{\partial L}{\partial \dot{q}_i}
\end{cases}
$$

### 5.2 Example: Pendulum

**Lagrangian:** $L = \frac{1}{2}mL^2\dot{q}^2 - \frac{1}{2}mgLq^2$ (where $q = \theta$)

**Step 1:** Compute conjugate momentum:

$$
p = \frac{\partial L}{\partial \dot{q}} = mL^2\dot{q} \quad \Rightarrow \quad \dot{q} = \frac{p}{mL^2}
$$

**Step 2:** Substitute into Lagrangian:

$$
L = \frac{1}{2}mL^2\left(\frac{p}{mL^2}\right)^2 - \frac{1}{2}mgLq^2 = \frac{p^2}{2mL^2} - \frac{mgLq^2}{2}
$$

**Step 3:** Apply Legendre transform:

$$
\begin{aligned}
H &= p\dot{q} - L \\
&= p \cdot \frac{p}{mL^2} - \left(\frac{p^2}{2mL^2} - \frac{mgLq^2}{2}\right) \\
&= \frac{p^2}{mL^2} - \frac{p^2}{2mL^2} + \frac{mgLq^2}{2} \\
&= \frac{p^2}{2mL^2} + \frac{mgLq^2}{2}
\end{aligned}
$$

This recovers the Hamiltonian we derived earlier! ✓

---

## 6. Constrained Lagrangians

### 6.1 Holonomic Constraints

**Definition:** If a constraint is an equality and a function of coordinates only (no $\dot{q}$'s), it is **holonomic**:

$$
f_i(q, t) = 0
$$

We can augment the Lagrangian with Lagrange multipliers:

$$
\hat{L}(q, \dot{q}, \lambda) = L(q, \dot{q}) - \sum_i \lambda_i f_i(q, t)
$$

where $\lambda$ is a **field** (in contrast to the KKT multipliers we considered previously, which were scalar).

### 6.2 Equations of Motion

Applying Euler-Lagrange to the augmented Lagrangian:

$$
\begin{cases}
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_j}\right) = \frac{\partial L}{\partial q_j} + \lambda(t) \frac{\partial f_i}{\partial q_j} \\
f_i(q, t) = 0
\end{cases}
$$

These are the **constrained equations of motion**.

---

## 7. Noether's Theorem

### 7.1 Statement

**Informal statement:** Symmetries which leave the Lagrangian unchanged each have a corresponding **conserved quantity**.

Consider the transformations:

$$
\begin{aligned}
t &\to t' = t + \delta t \\
q &\to q' = q + \delta q
\end{aligned}
$$

Assume $N$ such maps parameterized by small parameters $\varepsilon_r$:

$$
\begin{aligned}
\delta t &= \sum_r \varepsilon_r T_r \\
\delta q &= \sum_r \varepsilon_r Q_r
\end{aligned}
$$

**Noether's Theorem:** If the Lagrangian is invariant under these transformations, then:

$$
\left(\frac{\partial L}{\partial \dot{q}} \cdot \dot{q} - L\right) T_r - \frac{\partial L}{\partial \dot{q}} \cdot Q_r = \text{const.}
$$

are **conserved quantities**.

### 7.2 Example 1: Time Invariance

**Transformation:**

$$
\begin{aligned}
t &\to t + \delta t \\
N &= 1, \quad T = 1, \quad Q = 0
\end{aligned}
$$

**Conserved quantity:**

$$
\frac{\partial L}{\partial \dot{q}} \cdot \dot{q} - L = \text{const.}
$$

This is the **Hamiltonian** (total energy)! Energy conservation follows from time-translation symmetry.

### 7.3 Example 2: Translation Invariance

**Transformation:**

$$
\begin{aligned}
q &\to q + \delta q \\
N &= 1, \quad T = 0, \quad Q = 1
\end{aligned}
$$

**Conserved quantity:**

$$
\frac{\partial L}{\partial \dot{q}} = \text{const.}
$$

This is the **conjugate momentum**! Momentum conservation follows from spatial translation symmetry.

### 7.4 Example 3: Rotational Invariance

**Transformation:** Rotation of $\delta\theta$ about a given axis $n$:

$$
r \to r + \delta\theta\, n \times r
$$

$$
N = 1, \quad T = 0, \quad Q = n \times r
$$

**Conserved quantity:**

$$
\begin{aligned}
\text{const.} &= \frac{\partial L}{\partial \dot{q}} \cdot Q \\
&= p \cdot (n \times r) \\
&= n \cdot (r \times p) \\
&= n \cdot L_{\text{ang}}
\end{aligned}
$$

where $L_{\text{ang}} = r \times p$ is **angular momentum**. Angular momentum conservation follows from rotational symmetry.

---

## 8. Implications for Machine Learning

**Design principle:** When machine learning dynamics, we will design architectures which respect this set of symmetries.

**Examples:**

- **Translation equivariance** in CNNs
- **Rotation equivariance** in graph neural networks (e.g., E(3)-equivariant networks)
- **Gauge symmetries** in physics-informed networks

By building symmetries into the architecture, we:

1. Reduce the hypothesis space (fewer parameters to learn)
2. Guarantee conservation laws by construction
3. Improve generalization to out-of-distribution data

---

## Summary

This lecture established the Lagrangian perspective on mechanics and connected it to machine learning:

1. **Functional derivatives** (Fréchet and Gâteaux) extend calculus to infinite-dimensional function spaces, enabling variational principles

2. **Euler-Lagrange equations** systematically convert optimization principles ($\delta S = 0$) into differential equations; formula: $\frac{\partial \mathcal{L}}{\partial u} - \frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{u}} = 0$

3. **Legendre transform** connects Lagrangian and Hamiltonian mechanics via duality: $H = \sum_i p_i\dot{q}_i - L$ where $p_i = \frac{\partial L}{\partial \dot{q}_i}$

4. **Constrained Lagrangians** naturally incorporate holonomic constraints via Lagrange multipliers, extending to PDE-constrained optimization

5. **Noether's theorem** reveals deep connection between symmetries and conservation laws:
   - Time invariance → Energy conservation
   - Translation invariance → Momentum conservation
   - Rotation invariance → Angular momentum conservation

**Key Takeaway:** The Lagrangian formulation transforms mechanics into an optimization problem, making it natural to enforce constraints and identify conserved quantities. Noether's theorem explains *why* conservation laws exist and provides a blueprint for designing neural network architectures that respect fundamental symmetries. By building symmetries into model architectures, we guarantee conservation laws by construction and dramatically improve generalization.

**Next lecture:** We'll apply these principles to spatial discretization of the wave equation, deriving finite difference schemes from the discrete action principle and enforcing translation invariance via Noether's theorem. This connects variational principles to the stencil design problems we've been studying.
