# Lecture 13: Quasi-Optimality, Error Estimation, and Lax-Milgram Theory

**Date:** March 19

**Topics Covered:**
- Quasi-optimality in energy norm
- Error estimation in $L^2$ norm via duality argument
- Interpolation theory and approximation bounds
- Lax-Milgram theory for general elliptic problems

**References:**
- Johnson
- Brenner & Scott

---

## 0. Overview

In the previous lecture, we introduced the **Galerkin discretization** of the Poisson problem and established Galerkin orthogonality. Today, we take a major step toward understanding **why finite element methods work so well**: we'll prove rigorous error estimates that show how quickly FEM solutions converge to the true solution as we refine our mesh. These aren't just numerical observations—they're mathematical guarantees.

We begin by generalizing our Galerkin formulation to **abstract bilinear forms**, which will allow us to handle a much broader class of PDEs. The key concept is the **energy norm** induced by the bilinear form, which naturally connects the Galerkin method to the Rayleigh-Ritz energy minimization principle we saw before. We'll prove **quasi-optimality**: the FEM solution is the best approximation in the chosen function space, measured in the energy norm.

But what about the $L^2$ norm, the most natural measure of error? Through an elegant **duality argument** (sometimes called Aubin-Nitsche trick), we'll show that $L^2$ error converges **twice as fast** as energy error—a phenomenon called **superconvergence**. The proof hinges on constructing an auxiliary problem and leveraging interpolation theory. Finally, we'll abstract these ideas through **Lax-Milgram theory**, which provides a unified framework for proving existence, uniqueness, and stability for any elliptic bilinear form. This theory is the foundation for learning both function spaces $V_h$ and operators $a(u,v)$ in physics-informed machine learning.

---

## 1. Bilinear Forms and Energy Norms

### 1.1 Generalization to Abstract Bilinear Forms

Last time we introduced the Galerkin discretization of the Poisson problem:

$$(G) \quad (\nabla u, \nabla v) = (f, v) \quad \text{for all } v \in V_h$$

We will now consider this as an example of a **general bilinear form** $a(\cdot, \cdot)$:

$$
a(u, v) = (f, v)
$$

**Definition:** A bilinear form $a: V \times V \to \mathbb{R}$ satisfies:

$$
\begin{aligned}
a(\alpha u_1 + \beta u_2, \gamma v_1 + \delta v_2) &= \alpha \gamma a(u_1, v_1) + \alpha \delta a(u_1, v_2) \\
&\quad + \beta \gamma a(u_2, v_1) + \beta \delta a(u_2, v_2)
\end{aligned}
$$

(linear in both arguments)

### 1.2 Energy Norm and Cauchy-Schwarz

For bilinear forms, we're interested in those that **generate an energy norm** with a Cauchy-Schwarz inequality:

$$
\boxed{\|v\|_E = \sqrt{a(v,v)}}
$$

$$
\boxed{a(v,w) \leq \|v\|_E \|w\|_E}
$$

**Recall from last class:** This is precisely the connection between Galerkin and Rayleigh-Ritz:

$$
\text{Solving } a(u,v) = (f,v) \quad \Leftrightarrow \quad \min_{v \in V_h} \|v\|_E^2
$$

### 1.3 Galerkin Orthogonality (Recap)

Last class we showed from **Galerkin orthogonality**:

$$
\|u - u_h\|_E \leq \|u - v\|_E \quad \forall v \in V_h
$$

or taking the infimum over $V_h$:

$$
\boxed{\|u - u_h\|_E \leq \inf_{v \in V_h} \|u - v\|_E}
$$

**This is quasi-optimality:** The FEM error is bounded by the best approximation error in $V_h$.

---

## 2. Error in $L^2$ Norm: The Duality Argument

### 2.1 From Energy Norm to $L^2$ Norm

We know that $(G)$ naturally minimizes error in the **energy norm**. What about the **$L^2$ norm**?

$$
\|u - u_h\|^2 = \int_\Omega (u - u_h)^2 \, dx
$$

**We'll show:** $L^2$ error is smaller than energy error, following a **duality argument** (Aubin-Nitsche trick).

### 2.2 The Auxiliary Problem

Define a new problem:

$$
\begin{cases}
-w'' = u - u_h \\
w(0) = w(1) = 0
\end{cases}
$$

**Key idea:** Use $w$ to relate $L^2$ error to energy error.

### 2.3 The Duality Argument

**Step 1:** Start with $L^2$ norm squared:
$$
\begin{aligned}
\|u - u_h\|^2 &= (u - u_h, u - u_h) \\
&= (u - u_h, -w'')
\end{aligned}
$$

**Step 2:** Integrate by parts:
$$
\begin{aligned}
&= (u' - u_h', w') + \underbrace{[w'(0)(u - u_h)(0) - w'(1)(u - u_h)(1)]}_{= 0 \text{ by BCs}} \\
&= a(u - u_h, w)
\end{aligned}
$$

**Step 3:** Apply Galerkin orthogonality (for any $v \in V_h$):
$$
= a(u - u_h, w - v)
$$

**Step 4:** Apply Cauchy-Schwarz for the bilinear form:
$$
\leq \|u - u_h\|_E \|w - v\|_E
$$

**Step 5:** Divide both sides by $\|u - u_h\|$ and use $-w'' = u - u_h$:
$$
\|u - u_h\| \leq \frac{\|u - u_h\|_E \|w - v\|_E}{\|u - u_h\|} = \frac{\|u - u_h\|_E \|w - v\|_E}{\|-w''\|}
$$

**Step 6:** Take infimum over $v \in V_h$:
$$
\boxed{\|u - u_h\| \leq \|u - u_h\|_E \inf_{v \in V_h} \frac{\|w - v\|_E}{\|w''\|}}
$$

### 2.4 The Key Approximation Property

**If we can find $v \in V_h$ such that:**

$$
\|w - v\|_E \leq \varepsilon \|w''\| \tag{A}
$$

**Then:**
$$
\|u - u_h\| \leq \varepsilon \|u - u_h\|_E
$$

**Applying quasi-optimality again:**
$$
\boxed{\|u - u_h\| \leq \varepsilon^2 \|u''\| = \varepsilon^2 \|f\|}
$$

**Everything comes down to showing that $V_h$ can approximate $w$ well enough!**

For piecewise linear choice of $V_h$, we'll show what $\varepsilon$ is.

---

## 3. Interpolation Theory

### 3.1 Setup

Define nodes: $0 = x_0 < x_1 < \ldots < x_n = 1$

$$
V_h = \{v \in C^0[0,1] : v|_{[x_{i-1}, x_i]} \text{ is a linear polynomial}, v(0) = 0\}
$$

Define **nodal basis functions** $\phi_i(x)$, $i = 1, \ldots, n$ satisfying:

$$
\phi_i(x_j) = \delta_{ij} = \begin{cases}
1 & i = j \\
0 & i \neq j
\end{cases}
$$

**(Kronecker-$\delta$ property)**

### 3.2 The Interpolant

Define the **interpolant** $\pi u \in V_h$ satisfying:

$$
\pi u(x_i) = u(x_i) \quad \forall \text{ nodes } x_i
$$

**This can be computed directly:**

$$
\begin{aligned}
\pi u(x_i) &= \sum_j \widehat{\pi u}_j \phi_j(x_i) = u(x_i) \\
&= \sum_j \widehat{\pi u}_j \delta_{ij} \\
&= \widehat{\pi u}_i
\end{aligned}
$$

**Therefore:**
$$
\boxed{\pi u(x) = \sum_j u(x_j) \phi_j(x)}
$$

### 3.3 The Interpolation Error Theorem

We'll show that $\pi u$ is the $v$ we needed in $(A)$.

**Theorem:** For all $u \in V$ with $u'' \in L^2$:
$$
\boxed{\|u - \pi u\|_E \leq Ch \|u''\|}
$$

where $C$ is independent of $h$ and $u$.

**This is the key approximation result!**

---

## 4. Proof of Interpolation Error Bound

### 4.1 Element-by-Element Analysis

**Proof strategy:** Work on 1 element, then sum up.

We need to show:
$$
\int_{x_{j-1}}^{x_j} (u - \pi u)'^2 \, dx \leq C(x_j - x_{j-1})^2 \int_{x_{j-1}}^{x_j} u''^2 \, dx
$$

**Let** $e = u - \pi u$ be the error. Note that $u'' = e''$ since $\pi u$ is piecewise linear ($(\pi u)'' = 0$ on each element).

### 4.2 Change of Variables

By change of variables:
$$
x = x_{j-1} + \tilde{x}(x_j - x_{j-1})
$$

We rewrite the problem on the reference element $[0,1]$:
$$
\int_{x_{j-1}}^{x_j} e'^2 \, dx \leq C(x_j - x_{j-1})^2 \int_{x_{j-1}}^{x_j} e''^2 \, dx
$$

**This is equivalent to:**
$$
\int_0^1 e'^2 \, d\tilde{x} \leq C \int_0^1 e''^2 \, d\tilde{x}
$$

### 4.3 Using Rolle's Theorem

**Rolle's Theorem:** If $e$ is continuous on $[a,b]$ and $e(a) = e(b)$, then there exists at least one point $c \in [a,b]$ such that $e'(c) = 0$.

Since $e(0) = e(1) = 0$ (interpolant matches at endpoints), there exists $\xi \in [0,1]$ such that:
$$
e'(\xi) = 0
$$

### 4.4 The Key Estimate

By the **Fundamental Theorem of Calculus**:
$$
e'(y) - e'(\xi) = \int_\xi^y e'' \, dx
$$

Since $e'(\xi) = 0$:
$$
|e'(y)|^2 = \left|\int_\xi^y e'' \, dx\right|^2
$$

**Apply Cauchy-Schwarz:**
$$
\begin{aligned}
|e'(y)|^2 &= \left|\int_\xi^y 1 \cdot e'' \, dx\right|^2 \\
&\leq \left(\int_\xi^y 1^2 \, dx\right) \left(\int_\xi^y (e'')^2 \, dx\right) \\
&= |y - \xi| \int_\xi^y (e'')^2 \, dx \\
&\leq |y - \xi| \int_0^1 (e'')^2 \, dx
\end{aligned}
$$

### 4.5 Integration Over the Element

Integrating over $y \in [0,1]$:
$$
\int_0^1 (e'(y))^2 \, dy \leq \int_0^1 |y - \xi| \, dy \cdot \int_0^1 (e'')^2 \, dx
$$

**Taking max over $\xi$** gives worst-case scenario:
$$
\max_{\xi \in [0,1]} \int_0^1 |y - \xi| \, dy = \frac{1}{2}
$$

(achieved when $\xi = 1/2$)

**Therefore:**
$$
\int_0^1 (e')^2 \, dy \leq \frac{1}{2} \int_0^1 (e'')^2 \, dx
$$

**And we're done!** Taking $C = 1/2$ and summing over all elements completes the proof. $\quad \checkmark$

---

## 5. Convergence Rates Summary

### 5.1 Main Results

We just showed $(A)$ holds with $\varepsilon = h$.

**Therefore:**

**(1) Energy norm error (quasi-optimality):**
$$
\|u - u_h\|_E \leq \|u - v\|_E \quad \forall v \in V_h
$$

**Picking $v = \pi u \in V_h$:**
$$
\boxed{\|u - u_h\|_E \leq C_1 h \|u''\| = C_1 h \|f\|}
$$

**(2) $L^2$ norm error (superconvergence via duality):**
$$
\boxed{\|u - u_h\| \leq C_2 h^2 \|u''\| = C_2 h^2 \|f\|}
$$

**Key observation:** $L^2$ error converges **twice as fast** (order $h^2$) compared to energy error (order $h$). This is **superconvergence**.

### 5.2 What's Up Next

Now that we understand classical FEM error analysis, we can extend to machine learning:

- **Machine learning $V_h$** (learning the function space itself)
- **Machine learning $a(u,v)$** (learning the bilinear form)

First, an abstraction of what we learned today, so it's clear this isn't just something special about Poisson...

---

## 6. Lax-Milgram Theory

### 6.1 General Framework

**There isn't anything special about Poisson**—the same process holds for any elliptic bilinear form.

Let $V$ be a **Hilbert space** with norm $\|\cdot\|_V$.

Suppose $a(u,v)$ satisfies:

**(1) Symmetric:**
$$
a(u,v) = a(v,u)
$$

**(2) Continuous** (have a Cauchy-Schwarz):

There exists $\gamma > 0$ such that:
$$
|a(v,w)| \leq \gamma \|v\|_V \|w\|_V \quad \forall v, w \in V
$$

**(3) Elliptic** (aka $V$-elliptic, coercive):

There exists $\alpha > 0$ such that:
$$
a(v,v) \geq \alpha \|v\|_V^2 \quad \forall v \in V
$$

### 6.2 Lax-Milgram Theorem

**Theorem:** If $L: V \to \mathbb{R}$ is a bounded linear functional with:
$$
|L(v)| \leq \Lambda \|v\|_V
$$

Then:

**(1) Existence & Uniqueness:** The problem
$$
a(u,v) = L(v) \quad \forall v \in V
$$
has a **unique solution** $u \in V$.

**(2) Stability estimate:**
$$
\boxed{\|u\|_V \leq \frac{\Lambda}{\alpha}}
$$

**(3) Equivalence to energy minimization:**
$$
u = \arg\min_{v \in V} F(v)
$$
where $F(v) = \frac{1}{2}a(v,v) - L(v)$.

### 6.3 Proof Sketch

**Min $F(v) \Rightarrow a(u,v) = L(v)$:** Immediate from $\delta_v F(v) = 0$.

**Stability estimate:** Take $v = u$:
$$
a(u,u) = L(u)
$$

By coercivity and boundedness:
$$
\alpha \|u\|_V^2 \leq a(u,u) = L(u) \leq \Lambda \|u\|_V
$$

Therefore:
$$
\|u\|_V \leq \frac{\Lambda}{\alpha} \quad \checkmark
$$

**Uniqueness:** Assume two solutions $u_1, u_2$:
$$
a(u_1 - u_2, v) = 0 \quad \forall v \in V
$$

Take $v = u_1 - u_2$:
$$
a(u_1 - u_2, u_1 - u_2) = 0
$$

By coercivity:
$$
\alpha \|u_1 - u_2\|_V^2 \leq 0
$$

Therefore:
$$
\|u_1 - u_2\|_V = 0 \quad \Rightarrow \quad u_1 = u_2 \quad \checkmark
$$

---

## 7. The General FEM Playbook

For **any elliptic operator**, we have the playbook:

**(1) Prove Lax-Milgram conditions** (continuity, coercivity)  
→ Guarantees existence, uniqueness, stability

**(2) Get quasi-optimality** for best fit in $\|\cdot\|_V$:
$$
\|u - u_h\|_V \leq C \inf_{v \in V_h} \|u - v\|_V
$$

**(3) Construct interpolant** as upper bound:
$$
\inf_{v \in V_h} \|u - v\|_V \leq \|u - \pi u\|_V
$$

**(4) Prove interpolation error bounds**:
$$
\|u - \pi u\|_V \leq C h^k \|u^{(k+1)}\|
$$

where $k$ depends on the polynomial degree of $V_h$.

**(5) Apply duality argument** to relate $\|\cdot\|_V$ to $\|\cdot\|_{L^2}$ and understand convergence rates.

**This framework extends to:**
- Higher dimensions
- Vector-valued problems (elasticity, fluid dynamics)
- Mixed formulations
- Nonlinear problems
- **Learning both $V_h$ and $a(u,v)$ in ML contexts**

---

## 8. Summary

This lecture covered:

1. **Bilinear forms** and **energy norms** as abstractions of Galerkin methods
2. **Quasi-optimality** in energy norm: FEM gives best approximation in $V_h$
3. **Duality argument** (Aubin-Nitsche trick) for $L^2$ error estimates
4. **Interpolation theory**: constructing $\pi u$ and bounding approximation error
5. **Convergence rates**: $\mathcal{O}(h)$ in energy norm, $\mathcal{O}(h^2)$ in $L^2$ norm
6. **Lax-Milgram theory**: general framework for elliptic problems
7. **FEM playbook**: systematic approach to error analysis for any elliptic operator

**Key Takeaway:** The power of finite element methods lies not just in their flexibility for complex geometries, but in their **provable optimal approximation properties**. The Lax-Milgram framework shows that for any elliptic bilinear form satisfying continuity and coercivity, we obtain existence, uniqueness, stability, and convergence with quantifiable error bounds. The duality argument reveals superconvergence in $L^2$ norm—convergence twice as fast as in the energy norm. This theoretical foundation is essential for physics-informed machine learning: when we learn function spaces $V_h$ or operators $a(u,v)$, we need to ensure these learned components preserve the mathematical structure (symmetry, coercivity, continuity) that guarantees convergence.

