# Lecture 15: Conservation Laws and De Rham Complexes

**Reference:** Lecture 15 - Mixed Finite Element Methods for Conservation Laws

**Topics Covered:**
- Saddle-point problems and inf-sup stability review
- Conservation laws: divergence and curl forms
- Discrete de Rham complexes (Whitney forms)
- Local and global conservation properties
- Monotonicity conditions for nonlinear problems

---

## 0. Overview

This lecture marks an important transition in our course: from studying abstract saddle-point problems to understanding their application in **conservation laws**. Having established the theory of mixed finite element methods for Stokes flow in the previous lecture, we now leverage that framework to design numerical schemes that preserve fundamental physical conservation principles.

The key insight is recognizing that many important PDEs can be written in the form $\nabla \cdot F = f$ (mass/energy conservation) or $\nabla \times F = f$ (circulation/vorticity conservation), where the flux $F$ may involve both linear and nonlinear terms. For these problems, simply choosing stable finite element spaces is not enough—we need spaces that form a **de Rham complex**, ensuring that discrete versions of identities like $\nabla \times \nabla = 0$ and $\nabla \cdot (\nabla \times) = 0$ hold exactly.

We begin by revisiting the inf-sup stability proof from the previous lecture, then show how to construct compatible finite element spaces for conservation laws in 1D. The construction generalizes to higher dimensions through Whitney forms, building the classical sequence $W^0 \xrightarrow{\text{grad}} W^1 \xrightarrow{\text{curl}} W^2$. Along the way, we prove that these spaces satisfy both **global conservation** (total flux through boundaries equals total sources) and **local conservation** (element-wise balance of fluxes).

Finally, we introduce **monotonicity conditions** for nonlinear terms, which guarantee stability even when $\varepsilon N(u)$ is added to linear operators. This sets the stage for studying nonlinear conservation laws in physics and machine learning applications.

## 1. Review: Saddle-Point Problems and Inf-Sup Stability

Last time we introduced **mixed FEM spaces** for Stokes flow:

$$
\begin{aligned}
\nabla^{2} u - \nabla p &= f \\
\nabla \cdot u &= 0
\end{aligned}
$$

This is an example of the general class of **saddle-point problems**:

$$
\begin{aligned}
a(u, v) + b(p, v) &= L_1(v) \quad \forall (v, q) \in V_h \otimes M_h \\
b(q, u) &= L_2(q)
\end{aligned}
$$

### 1.1 Uniqueness Proof Review

**Important Note:** Assuming $a$ satisfies the **Lax-Milgram conditions**, let's repeat the uniqueness proof from last lecture.

**Step 1:** Set $L_1 = L_2 = 0$, $q = p$, $v = u$:

$$
\begin{gathered}
a(u, u) + b(p, u) = 0 \\
b(p, u) = 0
\end{gathered}
$$

**Step 2:** Apply coercivity of $a$:

$$
\alpha \|u\|_V \leq a(u, u) = 0 \quad \Rightarrow \quad u = 0
$$

**Step 3:** Equation 1 reduces to:

$$
b(p, v) = 0
$$

**Step 4:** Apply the **inf-sup compatibility condition**. If $V_h, M_h$ satisfy: for all $p \in M_h$, there exists $\tilde{v} \in V_h$ such that

$$
\frac{b(p, \tilde{v})}{\|\tilde{v}\|_V} \geq \beta \|p\|_M
$$

Then:

$$
\|p\|_M \leq 0 \quad \Rightarrow \quad p = 0
$$

**Critical observation:** This proof works whenever we can find compatible spaces $V_h$ and $M_h$ that satisfy the inf-sup condition. The rest of this lecture develops such spaces for conservation laws.

## 2. From Saddle-Point Problems to Conservation Laws

### 2.1 Motivation

So far we've learned models like:

$$
A u + \varepsilon N(u) = f
$$

Now we have the equipment to consider **conservation laws**:

$$
\begin{aligned}
\nabla \cdot F &= f \\
F &= L(u) + \varepsilon N(u)
\end{aligned}
$$

These satisfy, for $f = 0$:

$$
0 = \int_{\Omega} f \, dx = \int_{\Omega} \nabla \cdot F \, dx = \int_{\partial \Omega} F \cdot dA
$$

**Interpretation:** The total flux through the boundary equals the total sources/sinks in the domain.

Similarly, one can define **circulation conservation**:

$$
\begin{aligned}
\nabla \times F &= f \\
F &= L(u) + \varepsilon N(u)
\end{aligned}
$$

For conservation of circulations (e.g., magnetic fields, vorticity):

$$
0 = \int f \, dA = \int \nabla \times F \cdot dA = \oint F \cdot dl
$$

### 2.2 Design Philosophy

**Pedagogical note:** The key challenge is designing finite element spaces that preserve these conservation properties discretely. We'll start with the simplest case (1D, linear) to build intuition, then generalize.

## 3. One-Dimensional Conservation Laws

### 3.1 Saddle-Point Formulation

Let's start by designing a good space in the linear setting ($\varepsilon = 0$) in 1D:

$$
\begin{aligned}
F' &= f \\
F &= L(u)
\end{aligned}
$$

**Step 1:** The first equation gives:

$$
\begin{aligned}
(q, F') &= (q, f) - (q', F) = (f, q)
\end{aligned}
$$

This suggests the form:

$$
b(q, F) = L_2(q)
$$

**Step 2:** To obtain a saddle-point problem (and automatic stability), choose:

$$
L(u) = u'
$$

$$
\begin{aligned}
(F, v) + (u', v) &= 0 \\
a(F, v) + b(u, v) &= 0
\end{aligned}
$$

with $a(u, v) = (u, v)$.

**Important Notes:**
- Can easily check $a$ satisfies Lax-Milgram (just the $L^2$ inner product)
- Just need to choose an inf-sup stable pair $V_h, M_h$

### 3.2 Constructing Inf-Sup Stable Spaces

Let:
- $M_h = \{f \in L^2(\Omega) \mid f|_e = \text{const}\}$ (piecewise constants)
- $V_h = $ piecewise linears $\subseteq H_0^1$

**Construction:** Given a $p \in M_h$, we can build $\tilde{v}$ such that $\tilde{v}' = p$.

**Step 1:** Set $\tilde{v}_0 = 0$

$$
\tilde{v}_j = \sum_{k=0}^{j-1} p_k h
$$

**Step 2:** Then:

$$
\int_{e_j} \nabla \tilde{v} \, dx = \tilde{v}_{j+1} - \tilde{v}_j = p_j h
$$

### 3.3 Verification of Inf-Sup Condition

Note that:

$$
\begin{aligned}
b(p, \tilde{v}) &= \int p \tilde{v}' \\
&= \sum_j \hat{p}_j \int_{[x_j, x_{j+1}]} \tilde{v}' \\
&= \sum_j \hat{p}_j (\tilde{v}_{j+1} - \tilde{v}_j) \\
&= \sum_j \hat{p}_j^2 h \\
&= \|p\|_{L^2}^2
\end{aligned}
$$

And:

$$
\begin{aligned}
|v|_{H^1}^2 &= \sum_{e_j} \int_{x_j}^{x_{j+1}} |\nabla v|^2 \, dx \\
&= \sum_{e_j} \int_{x_j}^{x_{j+1}} p^2 \, dx \\
&= \|p\|_{L^2}^2
\end{aligned}
$$

So:

$$
\frac{b(p, \tilde{v})}{|v|_{H^1}} = \frac{\|p\|_{L^2}^2}{\|p\|_{L^2}} = \|p\|_{L^2}
$$

**This proves inf-sup stability with $\beta = 1$.**

### 3.4 Key Observations

Now that we've worked out a pair of inf-sup stable spaces, notice:

1. For any bigger choice of $V_h \subseteq V_h^{\text{big}}$, $\tilde{v} \in V_h^{\text{big}}$, and so $M_h$ and $V_h^{\text{big}}$ are also inf-sup compatible

2. **The key property** was that $V_h$ is chosen large enough that $\tilde{v}' = p$, or the derivative operator is onto (surjective):

$$
\frac{d}{dx} V_h \supseteq M_h
$$

**Critical observation:** This containment condition is the key to designing stable mixed methods!

## 4. Higher-Dimensional Conservation Laws

### 4.1 General Principle

In higher dimensions, for $D = \text{div/grad/curl}$:

$$
(D^* f, g) = (f, D g) \quad \text{where } D^* = \text{grad/div/curl}
$$

The **saddle-point problem**:

$$
\begin{aligned}
(F, v) + (Du, v) &= 0 \\
(q, D^* F) &= (f, q)
\end{aligned}
$$

is **inf-sup stable** if:

$$
D V_h \supseteq M_h
$$

**This generalizes our 1D observation to arbitrary differential operators!**

### 4.2 Discrete Conservation Properties

Now we have FEM spaces nailed down. What about **discrete conservation properties**?

Consider higher dimensions:

$$
-(\nabla q, F) + \langle q, F \rangle_{\text{BC}} = (f, q)
$$

#### Global Conservation

Let $q = 1$:

$$
\begin{aligned}
-(\nabla q, F) + \langle q, F \rangle &= (f, q) \\
\int_{\partial \Omega} F \cdot dA &= \int_{\Omega} f \, dx
\end{aligned}
$$

**Interpretation:**

$$
\text{flux in/out} = \text{source-sinks}
$$

#### Local Conservation

Let $V_h = \text{span}(\phi_i)$ and choose $q = \phi_i$:

$$
\begin{aligned}
&-\int \nabla \phi_i \cdot F \, dx = (f, \phi_i) \\
&= -\sum_j \int \phi_j \nabla \phi_i \cdot F \, dx \\
&= \sum_j \int \underbrace{(\phi_j \nabla \phi_i - \phi_i \nabla \phi_j)}_{\psi_{ij} = -\psi_{ji}} \cdot F \, dx
\end{aligned}
$$

**Step 1:** Assume $f = 0$ and $F \cdot dA = 0$ on the boundary.

**Step 2:** Note that:

$$
\sum_i \phi_i = 1 \quad \Rightarrow \quad \sum_j \nabla \phi_j = 0
$$

**Step 3:** Therefore:

$$
\sum_j \phi_i \nabla \phi_j = 0
$$

**Step 4:** If $f = 0$:

$$
\sum_j \underbrace{\int \psi_{ij} \cdot F \, dx}_{\substack{\text{equal and opposite} \\ \text{fluxes}}} = 0
$$

**Interpretation:** The anti-symmetry of $\psi_{ij}$ ensures that fluxes between adjacent elements cancel perfectly.

## 5. Whitney Forms and the De Rham Complex

### 5.1 Constructing Whitney 1-Forms

Denote:
- $W^0 = \text{span}(\phi_i)$ (vertex-based functions)
- $W^1 = \text{span}(\psi_{ij})$ (edge-based functions)

where:

$$
\psi_{ij} = \phi_j \nabla \phi_i - \phi_i \nabla \phi_j
$$

**Theorem:** $\text{grad}(W^0) \subseteq W^1$

**Proof:** Let $f \in W^0$:

$$
\begin{aligned}
f &= \sum_i \hat{f}_i \phi_i(x) \\
\nabla f &= \sum_i \hat{f}_i \nabla \phi_i \\
&= \sum_{i,j} \hat{f}_i (\phi_j \nabla \phi_i - \phi_i \nabla \phi_j) \\
&= \sum_{i,j} \hat{f}_i \psi_{ij} \in W^1
\end{aligned}
$$

### 5.2 Constructing Whitney 2-Forms

Motivated by the construction, we'll "play by ear" and make a space for curls.

**Step 1:** Apply integration by parts:

$$
(\psi_{ij}, \nabla \times F) = (\nabla \times \psi_{ij}, F)
$$

**Step 2:** Compute:

$$
\begin{aligned}
\nabla \times \psi_{ij} &= \nabla \times (\phi_j \nabla \phi_i - \phi_i \nabla \phi_j) \\
&= \nabla \phi_j \times \nabla \phi_i
\end{aligned}
$$

(using $\nabla \times (f \vec{G}) = \nabla f \times \vec{G} + f \nabla \times \vec{G}$ and $\nabla \times \nabla = 0$)

**Step 3:** Introduce partition of unity:

$$
\sum_k \phi_k = 1
$$

**Step 4:** Therefore:

$$
\begin{aligned}
\nabla \phi_j \times \nabla \phi_i &= \sum_k \phi_k (\nabla \phi_j \times \nabla \phi_i) \\
&= \sum_k (\phi_k \nabla \phi_j \times \nabla \phi_i + \phi_i \nabla \phi_k \times \nabla \phi_j + \phi_j \nabla \phi_i \times \nabla \phi_k)
\end{aligned}
$$

Define:

$$
\psi_{ijk}^2 = \phi_k \nabla \phi_j \times \nabla \phi_i + \phi_i \nabla \phi_k \times \nabla \phi_j + \phi_j \nabla \phi_i \times \nabla \phi_k
$$

**Note:** Anti-symmetric with respect to index permutations.

Let $W^2 = \text{span}(\psi_{ijk}^2)$ (face-based functions).

**Theorem:** $\text{curl}(W^1) \subseteq W^2$

### 5.3 The Complete De Rham Complex

We can now build a **de Rham complex**:

$$
W^0 \xrightarrow{\text{grad}} W^1 \xrightarrow{\text{curl}} W^2
$$

Preserving exactly $\text{curl} \circ \text{grad} = 0$.

By duality (integrating by parts $(\nabla f, g) = -(f, \nabla \cdot g)$):

$$
W^0 \xleftarrow{-\text{div}} W^1 \xleftarrow{\text{curl}} W^2
$$

This gives $\text{div} \circ \text{curl} = 0$ exactly.

**Pedagogical note:** These spaces, known as **Whitney forms**, are the foundation of finite element exterior calculus. They ensure that topological identities in differential geometry are preserved at the discrete level—crucial for electromagnetics, fluid dynamics, and geometric PDEs.

## 6. Nonlinear Conservation Laws

### 6.1 Monotonicity Conditions

Return to nonlinearity:

$$
Lu + \varepsilon N(u) = f
$$

$$
a(u, v) + \varepsilon (N(u), v) = (f, v)
$$

We introduce a new condition:

**Monotonicity:**

$$
(N(u), u) \geq \alpha_N \|u\|_V^2
$$

### 6.2 Stability Analysis

To prove stability:

$$
\begin{aligned}
\alpha \|u\|_V^2 - \varepsilon \alpha_N \|u\|_V^2 &\leq a(u, u) + \varepsilon (N(u), u) \\
&= (f, u) \\
&\leq C \|f\|_V \|u\|_V
\end{aligned}
$$

Therefore:

$$
\|u\|_V \leq \frac{C}{\alpha - \varepsilon \alpha_N} \|f\|_V
$$

**Critical observation:** Provided $\varepsilon < \alpha / \alpha_N$, the solution is bounded and the problem is well-posed. The monotonicity of $N$ provides a stabilizing effect that prevents blow-up.

---

## Summary

This lecture covered:

1. **Review of inf-sup stability** for saddle-point problems
2. **Conservation laws** in divergence and curl form
3. **One-dimensional construction** of inf-sup stable spaces via derivative surjectivity
4. **Higher-dimensional generalization** using Whitney forms
5. **Global and local conservation** properties at the discrete level
6. **De Rham complex** construction: $W^0 \xrightarrow{\text{grad}} W^1 \xrightarrow{\text{curl}} W^2$
7. **Monotonicity conditions** for nonlinear stability

**Key Takeaway:** The discrete de Rham complex ensures that fundamental identities like $\nabla \times \nabla = 0$ and $\nabla \cdot (\nabla \times) = 0$ hold exactly at the discrete level. This is achieved by choosing finite element spaces that satisfy $D V_h \supseteq M_h$ for the appropriate differential operator $D$. Such structure-preserving discretizations are essential for long-time stability in conservation law simulations and maintain the geometric properties of the continuous problem.
