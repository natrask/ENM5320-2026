# Hamiltonian Dynamics and Energy-Conserving Time Integrators

**Date:** February 12, 2025  
**Topics:** Hamiltonian mechanics, symplectic structure, machine learning Hamiltonians, discrete gradient method

---

## 0. Overview

This lecture transitions from learning spatial discretizations (finite difference stencils) to learning **time integrators** that preserve fundamental physical invariants. We address a critical question: In Monday's lecture, we built generic nonlinear FD stencil fitters — they worked most of the time, but *why*? Today we'll talk about building in **energy conservation exactly**.

The key insight is that many physical systems are naturally described as **Hamiltonian dynamical systems**, where energy (the Hamiltonian $H$) is automatically conserved. Standard numerical integration methods (forward Euler, Runge-Kutta) violate this conservation, leading to unphysical energy drift or damping over long time horizons. By designing time integrators that respect the **symplectic structure** of Hamiltonian systems, we can guarantee discrete energy conservation.

This lecture introduces three perspectives on mechanics — **Newtonian** (forces), **Hamiltonian** (energy-centric), and **Lagrangian** (optimization-centric) — and shows how they connect through the Legendre transform. We then develop the **discrete gradient method**, a geometric integration technique that constructs time integrators preserving energy exactly at the discrete level. This provides the foundation for machine learning continuous Hamiltonians from data while ensuring learned models conserve energy.

**Pedagogical note:** While the machinery may seem abstract, the payoff is substantial: learned models that automatically respect conservation laws exhibit dramatically improved long-time accuracy and physical realism compared to black-box neural networks.

---

## 1. Preliminaries: Skew-Symmetry and Energy Conservation

**Lemma:** If $A = -A^\top$ (i.e., $A$ is skew-symmetric), then $x^\top A x = 0$ for all $x$.

**Proof:**

$$
\begin{aligned}
x^\top A x &= \frac{1}{2}(x^\top A x + x^\top A x) \\
&= \frac{1}{2}(x^\top A x + x^\top A^\top x) \quad \text{(transpose is scalar)} \\
&= \frac{1}{2} x^\top (A + A^\top) x \\
&= \frac{1}{2} x^\top (A - A) x \quad \text{(since } A = -A^\top\text{)} \\
&= 0
\end{aligned}
$$

This simple observation is the foundation for energy conservation in Hamiltonian systems.

---

## 2. Hamiltonian Mechanics

### 2.1 Canonical Hamiltonian Systems

Let:
- $q \in \mathbb{R}^N$ be the vector of **generalized positions**
- $p \in \mathbb{R}^N$ be the vector of **generalized momenta**
- $x = \binom{q}{p} \in \mathbb{R}^{2N}$ be the **state** of the system

The system is **canonically Hamiltonian** if:

$$
\begin{aligned}
\frac{dp}{dt} &= -\frac{\partial H}{\partial q} \\
\frac{dq}{dt} &= \frac{\partial H}{\partial p}
\end{aligned}
\quad \text{for some } H(q, p) \in \mathbb{R}
$$

where $H$ is the **Hamiltonian** (typically total energy).

**Theorem:** A canonically Hamiltonian system conserves $H$, i.e., $\frac{dH}{dt} = 0$.

**Proof:**

$$
\begin{aligned}
\frac{dH}{dt} &= \frac{\partial H}{\partial q} \frac{dq}{dt} + \frac{\partial H}{\partial p} \frac{dp}{dt} \\
&= \frac{\partial H}{\partial q} \cdot \frac{\partial H}{\partial p} + \frac{\partial H}{\partial p} \cdot \left(-\frac{\partial H}{\partial q}\right) \\
&= 0
\end{aligned}
$$

### 2.2 Example: Nonlinear Pendulum

**Newton's formulation:**

$$
\begin{aligned}
m\ddot{x} &= -mg \sin\theta \\
\text{s.t.} \quad \|x\| &= L
\end{aligned}
$$

**Enforce constraint** by switching to polar coordinates: $x = L\theta(t)$

$$
\begin{aligned}
mL\ddot{\theta} &= -mg\sin\theta \\
\ddot{\theta} &= -\lambda^2 \sin\theta, \quad \lambda = \sqrt{\frac{g}{L}}
\end{aligned}
$$

**Small angle limit:** $\sin\theta \approx \theta$ gives $\ddot{\theta} = -\lambda^2 \theta$ (harmonic oscillator).

**Hamiltonian formulation:**

Let $q = \theta$ and $p = \dot{\theta}$ (angular velocity). Then:

$$
H = \frac{1}{2}p^2 + \frac{\lambda^2}{2}q^2
$$

Verify:

$$
\begin{aligned}
\partial_p H &= p, \quad \partial_q H = \lambda^2 q \\
\frac{dp}{dt} &= -\partial_q H = -\lambda^2 q \\
\frac{dq}{dt} &= \partial_p H = p
\end{aligned}
$$

Recovering $\ddot{\theta} = \dot{p} = -\lambda^2 q = -\lambda^2 \theta$. ✓

**Identifying Hamiltonians for nonlinear systems:**

For dynamics of the form $\ddot{\theta} + F(\theta) = 0$ where $F$ has a simple antiderivative, we can use the trick:

$$
\frac{d}{dt}\left[\frac{1}{2}\dot{\theta}^2 + \int^{\theta(t)} F(\phi)\, d\phi\right] = [\ddot{\theta} + F(\theta)]\dot{\theta}
$$

**For the nonlinear pendulum:** Multiply $\ddot{\theta} + \lambda^2 \sin\theta = 0$ by $\dot{\theta}$:

$$
\ddot{\theta}\dot{\theta} + \lambda^2 \dot{\theta}\sin\theta = 0
$$

$$
\frac{d}{dt}\left(\frac{1}{2}\dot{\theta}^2 - \lambda^2\cos\theta\right) = 0
$$

$$
\Rightarrow \frac{1}{2}\dot{\theta}^2 - \lambda^2\cos\theta = \text{const.}
$$

Taking:

$$
\begin{aligned}
H &= \frac{1}{2}p^2 - \lambda^2\cos q \\
\frac{dp}{dt} &= -\partial_q H = -\lambda^2\sin q \\
\frac{dq}{dt} &= \partial_p H = p
\end{aligned}
$$

**Important observations:**

- $H$ is often called an **integral** or **invariant** of the dynamics
- The whole game is to reverse-engineer an $H$ which, for a good choice of $p, q$, gives our dynamics (typically energy)
- This is hard in general and becomes involved for PDEs
- This is **not hard** if we **machine learn** $H$ and use it to steer the system dynamics

**Phase diagram of the nonlinear pendulum:**

The phase portrait $(q, p)$ reveals qualitative features:

- **Separatrix** denotes switch between small rocking oscillations and pendulum rotating forever
- **Inner loops** are closed, so solutions repeat forever (periodic orbits)
- **Outer trajectories** wind around, corresponding to continuous rotation

We'd like to be able to recover this structure in an ML model.

---

## 3. Symplectic Structure

### 3.1 Compact Notation

We can compactly write Hamiltonian dynamics as:

$$
\begin{aligned}
\dot{x} &= S(x) \nabla_x H \\
S(x) &= \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}, \quad \nabla_x H = \begin{pmatrix} \partial_q H \\ \partial_p H \end{pmatrix}
\end{aligned}
$$

**Note:** Some references use $S = \begin{pmatrix} 0 & -I \\ I & 0 \end{pmatrix}$; the sign convention depends on whether you write $\dot{q} = \partial_p H$ or $\dot{p} = \partial_q H$ first.

### 3.2 Area Preservation (Liouville's Theorem)

Recall the **Gauss divergence theorem:**

$$
\int_\Omega \nabla \cdot F = \int_{\partial\Omega} F \cdot dA
$$

Taking $F = x$ where $x \in \mathbb{R}^d$:

$$
\int_{\partial\Omega} x \cdot dA = \int_\Omega \nabla \cdot x\, dx = d|\Omega|
$$

So $\int_{\partial\Omega} x \cdot dA$ corresponds to the area (volume) of $\Omega$ times dimension.

**Time evolution of phase space volume:**

$$
\begin{aligned}
\frac{d}{dt} \int_{\partial\Omega} x \cdot dA &= \int_{\partial\Omega} \dot{x} \cdot dA \\
&= \int_{\partial\Omega} S\nabla H \cdot dA \\
&= \int_\Omega \nabla \cdot (S\nabla H)\, dx \\
&= \int_\Omega \sum_{i,j} \left(\partial_{q_i} \partial_{p_j} H - \partial_{p_i} \partial_{q_j} H\right) dx \\
&= 0 \quad \text{(mixed partials commute)}
\end{aligned}
$$

**Conclusion:** Area (volume) in phase space is conserved. This is exactly the property violated when we over-damp our system (e.g., with forward Euler or excessive numerical dissipation).

---

## 4. Non-Canonical Hamiltonians

In general, we can consider arbitrary skew-symmetric matrices $S(x)$:

$$
\begin{aligned}
\dot{x} &= S(x) \nabla_x H \\
S(x) &= -S(x)^\top
\end{aligned}
$$

$S$ is often referred to as a **Poisson matrix** or **Poisson structure**.

**Theorem:** A non-canonical Hamiltonian system preserves $H$.

**Proof:**

$$
\begin{aligned}
\frac{dH}{dt} &= \nabla_x H^\top \dot{x} \\
&= \nabla_x H^\top S(x) \nabla_x H \\
&= \frac{1}{2}\nabla_x H^\top S(x) \nabla_x H + \frac{1}{2}\nabla_x H^\top S(x) \nabla_x H \\
&= \frac{1}{2}\nabla_x H^\top S(x) \nabla_x H - \frac{1}{2}\nabla_x H^\top S(x)^\top \nabla_x H \\
&= \frac{1}{2}\nabla_x H^\top S(x) \nabla_x H - \frac{1}{2}(\nabla_x H^\top S(x) \nabla_x H)^\top \quad \text{(scalar)} \\
&= 0
\end{aligned}
$$

---

## 5. Machine Learning Hamiltonian Dynamics

In the continuous setting, we can easily learn a Hamiltonian system using neural networks:

$$
\begin{aligned}
Q &= NN_1(x), \quad NN_1: \mathbb{R}^N \to \mathbb{R}^{N \times N} \\
S(x) &= Q - Q^\top \\
H &= NN_2(x), \quad NN_2: \mathbb{R}^N \to \mathbb{R} \\
\Rightarrow \dot{x} &= (NN_1(x) - NN_1(x)^\top) \nabla_x NN_2(x)
\end{aligned}
$$

This architecture **guarantees** $S(x) = -S(x)^\top$, so energy is conserved in continuous time.

**Problem:** To fit to data, we need to finite difference $\dot{x} \approx \frac{x^{n+1} - x^n}{k}$, which will either:
- Grow area in phase space (explicit Euler)
- Shrink area in phase space (implicit Euler)

**Solution:** To discretely conserve $H$, we turn to **geometric integration theory**.

**References:**
- Hairer, Lubich, Wanner, *Geometric Numerical Integration*
- "On the construction of energy-preserving schemes..."

---

## 6. Discrete Gradient Method

### 6.1 Framework

Consider $\dot{x} = S(x) \nabla H(x)$ (dropping explicit $x$-dependence).

**Discrete scheme:**

$$
\frac{x^{n+1} - x^n}{k} = \tilde{S}(x^{n+1}, x^n) \bar{\nabla} H(x^{n+1}, x^n)
$$

where:

$$
\begin{aligned}
\lim_{x^{n+1} \to x^n} \tilde{S}(x^{n+1}, x^n) &= S(x^n) \quad \text{(consistency)} \\
\lim_{x^{n+1} \to x^n} \bar{\nabla} H(x^{n+1}, x^n) &= \nabla H(x^n) \quad \text{(consistency)} \\
(x^{n+1} - x^n) \cdot \bar{\nabla} H(x^{n+1}, x^n) &= H(x^{n+1}) - H(x^n) \quad \text{(discrete gradient property)}
\end{aligned}
$$

**Key requirement:**

$$
\bar{\nabla} H(x^n, x^n) = \nabla H(x^n)
$$

**Theorem:** Under these conditions, $H(x^{n+1}) - H(x^n) = 0$ (exact discrete energy conservation).

**Proof:**

$$
\begin{aligned}
H(x^{n+1}) - H(x^n) &= \bar{\nabla} H^\top (x^{n+1} - x^n) \\
&= \bar{\nabla} H^\top k \tilde{S}(x^{n+1}, x^n) \bar{\nabla} H \\
&= 0 \quad \text{(skew-symmetry of } \tilde{S}\text{)}
\end{aligned}
$$

This gives a **wish list** for how to build $\tilde{S}$ and $\bar{\nabla} H$.

### 6.2 Example 1: Harten, Lax, van Leer (1983)

**Notation:**

$$
\nabla H = (H_{x_1}, \ldots, H_{x_n}), \quad H_{x_i} = \frac{\partial H}{\partial x_i}
$$

**Define the discrete gradient:**

$$
\begin{aligned}
\bar{\nabla} H(x, y) &= (\bar{H}_{x_1}, \ldots, \bar{H}_{x_n}) \\
\bar{H}_{x_i} &= \int_0^1 H_{x_i}((1-\xi)x + \xi y)\, d\xi
\end{aligned}
$$

**Verification:** Show $(y - x) \cdot \bar{\nabla} H(x, y) = H(y) - H(x)$.

**Proof:**

$$
\begin{aligned}
\bar{\nabla} H(x, y) \cdot (y - x) &= \sum_{i=1}^n \bar{H}_{x_i}(x, y)(y_i - x_i) \\
&= \sum_{i=1}^n \int_0^1 H_{x_i}[(1-\xi)x + \xi y] (y_i - x_i)\, d\xi \\
&= \int_0^1 \sum_{i=1}^n H_{x_i}[(1-\xi)x + \xi y] (y_i - x_i)\, d\xi \\
&= \int_0^1 \frac{d}{d\xi} H[(1-\xi)x + \xi y]\, d\xi \quad \text{(chain rule)} \\
&= H(y) - H(x)
\end{aligned}
$$

where in the second-to-last step, we used:

$$
\frac{d}{d\xi} H[(1-\xi)x + \xi y] = \nabla H \cdot \frac{d}{d\xi}[(1-\xi)x + \xi y] = \nabla H \cdot (y - x)
$$

### 6.3 Other Discrete Gradient Formulations

- **Example 2:** Itoh & Abe (1988)
- **Example 3:** Gonzalez (1996)

Different formulations have different computational costs and accuracy properties, but all satisfy the discrete gradient property.

---

## Summary

This lecture established the mathematical framework for learning time integrators that preserve energy:

1. **Hamiltonian mechanics** provides an energy-centric description of dynamical systems where conservation of $H$ follows automatically from the equations of motion

2. **Symplectic structure** ($S = -S^\top$) is the geometric property ensuring energy conservation and phase space volume preservation

3. **Machine learning Hamiltonians** in continuous time is straightforward using neural networks with built-in skew-symmetry constraints

4. **Discrete gradient method** extends energy conservation to discrete time by carefully constructing approximations $\bar{\nabla}H(x^{n+1}, x^n)$ satisfying:
   - Consistency: $\bar{\nabla}H(x, x) = \nabla H(x)$
   - Discrete gradient property: $(x^{n+1} - x^n) \cdot \bar{\nabla}H = H(x^{n+1}) - H(x^{n})$

5. **Harten-Lax-van Leer discrete gradient** provides an explicit formula using path integrals in state space

**Key Takeaway:** Standard time integrators (Euler, RK4) violate conservation laws at the discrete level, leading to spurious energy drift. Geometric integrators constructed via the discrete gradient method **exactly** preserve energy at every timestep, dramatically improving long-time accuracy for Hamiltonian systems. This principle extends to learning time integrators from data: by parameterizing both $S(x)$ and $H(x)$ with neural networks and using discrete gradients, we can discover novel integration schemes that respect physics.

**Next lecture:** We'll shift from Hamiltonian to Lagrangian mechanics, introducing functional derivatives and the principle of least action. This provides an optimization-centric view of dynamics and enables elegant treatment of constraints and symmetries via Noether's theorem.
