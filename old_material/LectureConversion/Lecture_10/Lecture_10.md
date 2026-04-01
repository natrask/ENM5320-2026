# Lecture 10: Multi-Stage Time Integration and Symplectic Methods

**Date:** February 24

**Topics Covered:**
- Multi-stage integration methods (Runge-Kutta schemes)
- Linear stability analysis
- Störmer-Verlet symplectic integrator

---

## 0. Overview

This lecture marks a critical transition in our course: we now have all the tools needed to build sophisticated physics-informed neural architectures. We've learned to derive energy-preserving ODEs from least action principles, compute Hamiltonians via Legendre transforms, integrate using discrete gradients, solve for nonlinear stencils with PyTorch, and impose conservation constraints through polynomial reproduction and Noether's theorem. With these foundational techniques in place, we're ready to tackle our ultimate objective: **constructing a nonlinear wave equation solver**.

The focus today shifts to the crucial question of *how* to integrate these systems in time. We begin by examining multi-stage integration schemes, particularly the Runge-Kutta (RK) family of methods. These explicit methods are especially important for machine learning applications where gradient evaluations drive the optimization process. We'll see how different RK schemes achieve various orders of accuracy and, critically, how to analyze their **stability regions** for systems with complex eigenvalues—a key consideration when solving hyperbolic PDEs like the wave equation.

Finally, we introduce the **Störmer-Verlet (leapfrog) integrator**, a symplectic method that exactly preserves the Hamiltonian structure of our dynamics. This conservation property makes it ideal for long-time integration of energy-conserving systems derived from variational principles. The connection between our discrete gradient methods and symplectic integration provides a beautiful unification of the variational and Hamiltonian perspectives.

---

## 1. Synthesis: The Nonlinear Wave Equation Problem

### 1.1 Review of Available Tools

Let's recall what we now know how to do:

- Use **least action** to derive an energy/momentum-preserving ODE
- Use **Legendre transform** to compute a Hamiltonian
- Use **discrete gradient** to integrate (example to come)
- Use **PyTorch** to solve for nonlinear stencils
- Use **polynomial reproduction** and **Noether's theorem** to put constraints on stencils

**With these in hand, we are ready to assume our final form: a nonlinear wave equation solver!**

### 1.2 1D Wave Equation with Periodic Boundary Conditions

Consider the wave equation:

$$
\begin{cases}
\partial_{tt} u = c^2 \partial_{xx} u \\
u(x, t=0) = f(x)
\end{cases}
$$

The general solution is given by **d'Alembert's formula**:

$$
u(x,t) = f(x + ct) + f(x - ct)
$$

**To confirm this is indeed a solution:**

**Step 1:** Compute spatial derivatives:
$$
\partial_{xx} u = f''(x + ct) + f''(x - ct)
$$

**Step 2:** Compute first time derivative:
$$
\partial_t u = c f'(x + ct) - c f'(x - ct)
$$

**Step 3:** Compute second time derivative:
$$
\partial_{tt} u = c^2 f''(x + ct) + c^2 f''(x - ct)
$$

**Step 4:** Verify the wave equation:
$$
\partial_{tt} u = c^2 \partial_{xx} u \quad \checkmark
$$

**Goal:** Derive a neural architecture that can recover this simple case.

---

## 2. Variational Approach to the Wave Equation

### 2.1 Motivation from Discrete Laplacian

**Important Note:** In a previous class, we showed that in the absence of the right-most node $(D_+ u_N^2 = D_+^2 u_N)$, the Lagrangian gives the approximation:

$$
\nabla_h^2 \approx D_+^{-1} D_+ h = D_+ D_- h
$$

This approximation motivates our approach. We want to construct a **learnable stencil** that naturally encodes spatial derivatives.

### 2.2 Discrete Action with Learnable Nonlinear Stencil

Our goal is to be **as expressive as possible while satisfying conservation constraints**.

We hypothesize the discrete action:

$$
S_h[\theta] = \sum_i \frac{1}{2} \dot{q}_i^2 h - \frac{1}{2} N(D_- q_i; \theta)^2 h
$$

where:
- $N(\cdot; \theta)$ is a learnable nonlinear function (e.g., a neural network)
- $D_- q_i = (q_i - q_{i-1})/h$ is the backward difference operator
- $\theta$ represents learnable parameters

**Key Properties:**
- This action is **shift invariant** for both $q$ and $t$
- Alternative formulations are possible: $D_- N(q; \theta)$ or $D_- q \Rightarrow \alpha q_{i-1} - \alpha q_i$ for any $\alpha$

### 2.3 Deriving the Equations of Motion

**Variation of the first term:**

$$
\delta_q S_1 = -\ddot{q}_i h
$$

**Variation of the second term:**

**Step 1:** Define the variation:
$$
(\delta_q S_2, \delta_q) = \lim_{\varepsilon \to 0} \int_{t_0}^t \frac{1}{2\varepsilon} \left[ N(D_-(q + \varepsilon \delta_q); \theta)^2 - N(D_- q; \theta)^2 \right] dt
$$

**Step 2:** Apply the definition of directional derivative:
$$
= \lim_{\varepsilon \to 0} \int_{t_0}^t \frac{1}{\varepsilon} \nabla N(D_-(q + \varepsilon \delta_q); \theta) \cdot D_-(\varepsilon \delta_q) \, dt
$$

**Step 3:** Take the limit:
$$
= \int_{t_0}^t \nabla N(D_- q) \cdot D_- \delta q \, dt
$$

**Step 4:** Integration by parts (summation by parts in discrete case):
$$
= \int_{t_0}^t D_+ \nabla N(D_- q) \cdot \delta_q \, dt
$$

Therefore:
$$
\frac{\delta S_2}{\delta q} = D_+ \nabla N(D_- q; \theta)
$$

**Euler-Lagrange equation:**

$$
\boxed{\ddot{q}_i = D_+ \nabla N(D_- q; \theta)}
$$

This is our **learnable wave equation** in discrete form.

---

## 3. Hamiltonian Formulation

### 3.1 Generalized Momentum

We can identify the generalized momentum as:

$$
p_i(t) = \frac{\partial L}{\partial \dot{q}_i} = \dot{q}_i h
$$

### 3.2 Legendre Transform

Applying the Legendre transform:

**Step 1:** Start with definition:
$$
H = \sum_i p_i \dot{q}_i - L
$$

**Step 2:** Substitute $L = \frac{1}{2}\sum_i \dot{q}_i^2 h - \frac{1}{2}\sum_i N(D_- q; \theta)^2 h$:
$$
H = \sum_i \dot{q}_i^2 h - \frac{1}{2} \dot{q}_i^2 h - \frac{1}{2} N(D_- q; \theta)^2 h
$$

**Step 3:** Simplify kinetic term:
$$
H = \sum_i \frac{1}{2} \dot{q}_i^2 h + \frac{1}{2} N(D_- q; \theta)^2 h
$$

**Step 4:** Express in terms of momentum $p_i = \dot{q}_i h$:
$$
\boxed{H = \underbrace{\sum_i \frac{1}{2} p_i^2 h^{-1}}_{T_\theta(p) \text{ (kinetic)}} + \underbrace{\frac{1}{2} \sum_i N(D_- q; \theta)^2 h}_{V_\theta(q) \text{ (potential)}}}
$$

### 3.3 Hamilton's Equations

We want to solve:

$$
\begin{aligned}
\frac{dp}{dt} &= -\partial_q V_\theta(q) \\
\frac{dq}{dt} &= \partial_p T_\theta(p)
\end{aligned}
$$

These are the canonical Hamiltonian equations for our learnable wave system.

---

## 4. Multi-Stage Time Integration Methods

### 4.1 Motivation: Linear Stability Analysis

**Some final remarks on time integration:**

To choose an integration scheme (so far we've only looked at explicit Euler), we need to understand **stability**.

Consider a linear system of ODEs:

$$
\dot{y} = Ay
$$

**Critical observation:** Depending on the eigenvalues of $A$, the ODE will have distinct character:
- **Hyperbolic PDEs** correspond to **purely imaginary eigenvalues**
- We'll next cover some options that can handle purely imaginary eigenvalues

**Important Note:** Why is explicit integration important for ML applications?

In machine learning contexts, we need to evaluate gradients with respect to both state variables and parameters $\theta$. Implicit methods require solving nonlinear systems at each timestep, which is computationally expensive and complicates backpropagation. Explicit methods allow straightforward gradient flow through time integration.

### 4.2 Two Classes of Explicit Methods

Consider now the nonlinear system: $\ddot{x} = F(x; \theta)$

There are two broad classes of explicit methods:

#### Multi-Stage Schemes (Runge-Kutta)

$$
\begin{aligned}
k_i &= F\left(t_n + c_i h, x^n + h \sum_{j=1}^{i-1} a_{ij} k_j\right) \\
x^{n+1} &= x^n + h \sum_{i=1}^s b_i k_i
\end{aligned}
$$

**Idea:** At each stage, you can make an additional gradient evaluation using points generated in previous stages.

#### Multi-Step Schemes

$$
\sum_{j=0}^{S-1} a_j x^{n+1-j} = h \sum_{j=0}^{S-1} b_j f(t_{n-j}, x_{n-j})
$$

where $a_0 = 1$. If $a_j = 0$ for $j > 0$, the scheme is explicit.

**Idea:** Use information about the derivative from previous timesteps to predict the next state.

### 4.3 Comparison

**Multi-step methods:**
- Only need 1 function evaluation per step ⇒ generally faster
- Complicated for first $s$ steps (need to "start up" with multi-stage methods)

**Multi-stage methods:**
- Self-starting (no history required)
- Easier to implement with variable timesteps
- Natural for backpropagation through time

**For simplicity, we'll just use multi-stage methods.**

---

## 5. Runge-Kutta Schemes

### 5.1 Example: RK1 (Explicit Euler)

$$
\begin{aligned}
k_1 &= F(t_n, x_n) \\
x_{n+1} &= x_n + h k_1
\end{aligned}
$$

This is our familiar explicit Euler method.

### 5.2 Second-Order Schemes

After explicit Euler (EE) for stage 1, make a second stage:

$$
\begin{aligned}
k_2 &= F(t_n + \alpha h, x_n + \beta h k_1) \\
x_{n+1} &= x_n + h(a k_1 + b k_2)
\end{aligned}
$$

To choose coefficients, expand in Taylor series:

**Step 1:** Expand $k_2$ (note that $k_1 = F(t_n, x_n)$):
$$
F(t_n + \alpha h, x_n + \beta h k_1) = F(t_n, x_n) + \partial_t F(t_n, x_n) \cdot \alpha h + \partial_x F(t_n, x_n) \cdot \beta h k_1
$$

**Step 2:** Denote $F_n = F(t_n, x_n)$ and substitute:
$$
\begin{aligned}
x_{n+1} &= x_n + h(a k_1 + b k_2) \\
&= x_n + ha F_n + hb F_n + \alpha b h^2 \partial_t F_n + \beta b h^2 \partial_x F_n \cdot F_n \\
&= x_n + h(a+b) F_n + \frac{1}{2} h^2 (2\alpha b) \partial_t F_n + \frac{1}{2} h^2 (2\beta b) \partial_x F_n \cdot F_n
\end{aligned}
$$

**Step 3:** Compare to Taylor series expansion of exact solution to $\mathcal{O}(h^2)$:

$$
x(t_n + h) = x_n + h \dot{x}_n + \frac{h^2}{2} \ddot{x}_n + \mathcal{O}(h^3)
$$

where $\dot{x} = F$ and $\ddot{x} = \partial_t F + \partial_x F \cdot F$.

**Step 4:** Match coefficients:
$$
\begin{aligned}
a + b &= 1 \\
\alpha b &= 1/2 \\
\beta b &= 1/2
\end{aligned}
$$

**Result:** Non-unique! Multiple second-order RK schemes exist.

**Example: RK2** Take $a = b = 1/2$, then $\alpha = \beta = 1$:
$$
\begin{aligned}
k_1 &= F(t_n, x_n) \\
k_2 &= F(t_n + h, x_n + h k_1) \\
x_{n+1} &= x_n + \frac{h}{2}(k_1 + k_2)
\end{aligned}
$$

### 5.3 Butcher Tableaux

For the general $s$-stage scheme:

$$
\begin{aligned}
x_{n+1} &= x_n + h \sum_{i=1}^s b_i k_i \\
k_i &= F\left(t_n + h c_i, x_n + h \sum_{j=1}^{i-1} a_{ij} k_j\right)
\end{aligned}
$$

Coefficients are written compactly as a **Butcher tableau**:

$$
\begin{array}{c|cccc}
c_1 & a_{11} & a_{12} & \cdots & a_{1s} \\
c_2 & a_{21} & a_{22} & \cdots & a_{2s} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
c_s & a_{s1} & a_{s2} & \cdots & a_{ss} \\
\hline
& b_1 & b_2 & \cdots & b_s
\end{array}
$$

For explicit methods, $a_{ij} = 0$ for $j \geq i$ (lower triangular with zero diagonal).

### 5.4 First-Order Methods

$$
\begin{array}{c|c}
0 & 0 \\
\hline
& 1
\end{array}
$$

Only one choice → our friend explicit Euler.

### 5.5 Second-Order Methods

General 2-stage, 2nd-order RK scheme for any $\alpha$:

$$
\begin{array}{c|cc}
0 & 0 & 0 \\
\alpha & \alpha & 0 \\
\hline
& 1 - \frac{1}{2\alpha} & \frac{1}{2\alpha}
\end{array}
$$

**Notable examples:**
- $\alpha = 1/2$: **Explicit midpoint method**
- $\alpha = 1$: **Heun's method**
- $\alpha = 2/3$: **Ralston's method**

### 5.6 Third-Order Methods

General 3-stage, 3rd-order RK scheme:

$$
\begin{array}{c|ccc}
0 & 0 & 0 & 0 \\
\alpha & \alpha & 0 & 0 \\
\beta & \frac{\beta}{\alpha} \frac{\beta - 3\alpha(1-\alpha)}{3\alpha - 2} & -\frac{\beta}{\alpha} \frac{\beta - \alpha}{3\alpha - 2} & 0 \\
\hline
& 1 - \frac{3\alpha + 3\beta - 2}{6\alpha\beta} & \frac{3\beta - 2}{6\alpha(\beta - \alpha)} & \frac{3\alpha - 2}{6\beta(\beta - \alpha)}
\end{array}
$$

**Note:** At this point we should get excited when we see parameterized schemes! These free parameters can potentially be optimized for specific problem classes.

### 5.7 Fourth-Order Method: Standard RK4

$$
\begin{array}{c|cccc}
0 & 0 & 0 & 0 & 0 \\
1/2 & 1/2 & 0 & 0 & 0 \\
1/2 & 0 & 1/2 & 0 & 0 \\
1 & 0 & 0 & 1 & 0 \\
\hline
& 1/6 & 1/3 & 1/3 & 1/6
\end{array}
$$

**To understand why this is the default**, we need to analyze the stability. We'll see that it works for purely imaginary problems (hyperbolic PDEs).

---

## 6. Stability of Multi-Stage Schemes

### 6.1 Linear Stability Analysis

Consider again $\dot{y} = Ay$. Let's analyze RK2:

**Step 1:** Write out the RK2 scheme:
$$
\begin{aligned}
y_{n+1} &= y_n + \frac{h}{2}(k_1 + k_2) \\
k_1 &= A y_n \\
k_2 &= A(y_n + h k_1) = A y_n + h A^2 y_n
\end{aligned}
$$

**Step 2:** Substitute:
$$
y_{n+1} = y_n + \frac{h}{2}(A y_n + A y_n + h A^2 y_n) = \left(I + hA + \frac{h^2}{2} A^2\right) y_n
$$

**Step 3:** Define amplification matrix:
$$
y_{n+1} = Q y_n \quad \text{where} \quad Q = I + hA + \frac{h^2}{2} A^2
$$

**Note:** What drives the choices of coefficients in RK is that $Q \approx \exp(hA)$ (Taylor expansion of matrix exponential).

### 6.2 Stability Condition

Consider an arbitrary scheme where $Q$ is diagonalizable:

$$
y_n = Q^n y_0
$$

If $Q = S \Lambda S^{-1}$ where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)$, then:

**Step 1:** Expand the iteration:
$$
\begin{aligned}
y_n &= (S \Lambda S^{-1})(S \Lambda S^{-1}) \cdots (S \Lambda S^{-1}) y_0 \\
&= S \Lambda^n S^{-1} y_0
\end{aligned}
$$

**Step 2:** Define $w_n = S^{-1} y_n$:
$$
w_n = \Lambda^n w_0
$$

**Step 3:** Component-wise:
$$
w_{n,i} = \lambda_i^n w_{0,i}
$$

**Stability condition:** The solution is bounded if:
$$
\max_i |\lambda_i| \leq 1
$$

where $\lambda_i$ is the $i$-th eigenvalue of $Q$.

### 6.3 Spectral Radius and Stability Region

**Recall definition:** The **spectral radius** is:
$$
\rho(A) = \max\{|\lambda_i| : \lambda_i \text{ is an eigenvalue of } A\}
$$

We want:
$$
\rho(Q) = \left| 1 + h\rho(A) + \frac{h^2}{2} \rho(A)^2 \right| \leq 1
$$

Let $z = h \rho(A) \in \mathbb{C}$ (complex number). Solve for $z$ such that:

$$
|g(z)| = \left| 1 + z + \frac{1}{2} z^2 \right| \leq 1
$$

The set of all such $z$ forms the **stability region** in the complex plane.

**Important Note:** In general, this is tricky to do by hand but simple to visualize numerically in matplotlib.

### 6.4 Stability Regions for RK Methods

**RK1 (Explicit Euler):**

Stability region: Disk of radius 1 centered at $(-1, 0)$ in the complex plane. We already saw this only works with dissipation/stabilization.

**RK2:**

Stretches the stability region by approximately 3/2 in the imaginary direction, allowing larger timesteps for oscillatory problems.

**RK3:**

Further extends the stability region along the imaginary axis.

**RK4:**

The standard RK4 has the largest stability region along the imaginary axis among explicit methods of its order, which is why it's the default choice for Hamiltonian/wave-like systems.

**Critical observation:** For the wave equation (purely imaginary eigenvalues), we need stability regions that extend along the imaginary axis. RK4 provides the best balance of accuracy and stability for such problems.

---

## 7. Symplectic Integrators

### 7.1 Hamiltonian Systems

Recall that for a canonical Hamiltonian system:

$$
\begin{aligned}
\dot{p} &= -\partial_q H \\
\dot{q} &= \partial_p H
\end{aligned}
$$

with $\frac{dH}{dt} = 0$ (energy conservation).

**(We now know how to identify stencils that make Hamiltonians!)**

Assuming the decomposition:

$$
H = \underbrace{T(p)}_{\text{kinetic}} + \underbrace{V(q)}_{\text{potential}}
$$

Then we want to solve:

$$
\begin{aligned}
\dot{p} &= -\partial_q V \\
\dot{q} &= \partial_p T
\end{aligned}
$$

### 7.2 Störmer-Verlet / Leapfrog Integrator

**Only works for separable Hamiltonians, no dissipation.**

The **Störmer-Verlet** (also called **leapfrog**) integrator is a second-order symplectic method that exactly preserves the symplectic structure of Hamiltonian mechanics.

**Algorithm:**

**Step 1:** Half-step momentum update:
$$
p_{n+1/2} = p_n - \frac{h}{2} \partial_q V(q_n)
$$

**Step 2:** Full-step position update:
$$
q_{n+1} = q_n + h \partial_p T(p_{n+1/2})
$$

**Step 3:** Half-step momentum update:
$$
p_{n+1} = p_{n+1/2} - \frac{h}{2} \partial_q V(q_{n+1})
$$

**Key Properties:**
- **Symplectic:** Exactly preserves phase space volume
- **Time-reversible:** Running backward gives exact trajectory
- **Energy-conserving:** Bounded energy error over long times (no drift)
- **Second-order accurate:** $\mathcal{O}(h^2)$ local truncation error

**For kinetic energy** $T(p) = \frac{1}{2m} p^2$, we have $\partial_p T = p/m$, so the position update simplifies to:
$$
q_{n+1} = q_n + \frac{h}{m} p_{n+1/2}
$$

**Important Note:** The leapfrog structure (alternating half-steps) is what gives the method its symplectic property. This makes it ideal for long-time integration of conservative systems like our wave equation.

---

## 8. Summary

This lecture covered:

1. **Synthesis of tools** for building physics-informed neural architectures
2. **Variational derivation** of the learnable wave equation using discrete action principles
3. **Hamiltonian formulation** via Legendre transform
4. **Multi-stage integration** methods (Runge-Kutta family)
5. **Butcher tableaux** for compact representation of RK schemes
6. **Linear stability analysis** and stability regions in the complex plane
7. **Störmer-Verlet integrator** for symplectic time-stepping of Hamiltonian systems

**Key Takeaway:** For energy-conserving systems derived from variational principles, symplectic integrators like Störmer-Verlet provide exact conservation of phase space structure and bounded long-time energy behavior. Combined with learnable nonlinear stencils, these methods enable principled construction of physics-informed neural architectures for PDEs with guaranteed geometric properties. The choice of time integrator—whether high-order RK for accuracy or symplectic methods for conservation—depends critically on the eigenvalue structure and conservation requirements of the target system.
