# Lecture 14: Applications of Lax-Milgram and Introduction to Mixed FEM

**Date:** March 21

**Topics Covered:**
- Application of Lax-Milgram theory to various PDEs
- Biharmonic equation (acoustics, phase field, beam bending)
- Reaction-diffusion equation
- Linear elasticity and near-incompressible limit
- Breakdown of stability and need for mixed formulations
- Introduction to mixed finite elements
- Inf-sup (LBB) condition
- Stokes problem as limiting case

**References:**
- Johnson
- Brenner & Scott

---

## 0. Overview

In the previous lecture, we developed the abstract **Lax-Milgram theory**, which provides existence, uniqueness, and stability for elliptic variational problems. Today, we put this theory into practice by analyzing several important PDEs from physics and engineering. We'll see how checking continuity and coercivity of the bilinear form $a(u,v)$ immediately guarantees well-posedness and convergence—this is the power of abstraction.

We begin with the **biharmonic equation**, which governs phenomena from acoustics to beam bending. Then we tackle **reaction-diffusion equations** that model chemical reactions, population dynamics, and pattern formation. Most importantly, we'll examine **linear elasticity**, where a fascinating challenge emerges: as materials become nearly incompressible (like rubber or hydrogels), the Poisson ratio $\nu \to 1/2$ causes the Lamé parameter $\lambda \to \infty$, and our standard Galerkin formulation loses stability.

This breakdown motivates **mixed finite element methods**, where we introduce additional variables (like pressure or stress) and solve for multiple coupled fields simultaneously. The key mathematical tool is the **inf-sup condition** (also called Ladyzhenskaya-Babuška-Brezzi or LBB condition), which prescribes precise compatibility requirements between the function spaces for different fields. We'll see this framework emerge naturally from the **Stokes equations** for incompressible fluid flow, which is the limiting case of nearly incompressible elasticity. Understanding when and why standard FEM fails, and how mixed methods rescue us, is crucial for building robust physics-informed learning algorithms.

---

## 1. Administrative Notes

**Final Exam:** April 30th

**Final Project:** Plan projects now; presentations May 5-13

---

## 2. Review: Lax-Milgram Framework

Last time we established the abstract framework:

**(G) Find $u \in V_h$ where $V_h \subseteq V$ such that:**
$$
a(u,v) = L(v) \quad \forall v \in V_h
$$

**Lax-Milgram conditions:**

**(1) Continuity:** $a$ is continuous if
$$
|a(v,w)| \leq \gamma \|v\|_V \|w\|_V
$$

**(2) Ellipticity (coercivity):** $a$ is elliptic if
$$
a(v,v) \geq \alpha \|v\|_V^2
$$

**(3) Bounded linear functional:** $L$ is continuous if
$$
|L(v)| \leq \Lambda \|v\|_V
$$

**If (1)-(3) hold, then:**
- Unique solution exists
- Stability: $\|u\|_V \leq \Lambda/\alpha$
- **Céa's Lemma:** Quasi-optimality
$$
\boxed{\|u - u_h\|_V \leq \frac{\gamma}{\alpha} \inf_{v \in V_h} \|u - v\|_V}
$$

---

## 3. Application: Poisson Equation (Review)

### 3.1 Problem Statement

$$
\begin{aligned}
-\nabla^2 u &= f \quad \text{in } \Omega \\
u|_{\partial\Omega} &= 0
\end{aligned}
$$

### 3.2 Weak Form

$$
\begin{aligned}
a(u,v) &= \int_\Omega \nabla u \cdot \nabla v \, dx \\
L(v) &= \int_\Omega f v \, dx
\end{aligned}
$$

### 3.3 Function Spaces

**Remember:**
- $L^2$ norm: $\|f\|_{L^2}^2 = \int f^2 \, dx$
- $H^1$ seminorm: $|f|_{H^1}^2 = \int |\nabla f|^2 \, dx$
- $H^1$ norm: $\|f\|_{H^1}^2 = \|f\|_{L^2}^2 + |f|_{H^1}^2$

**Choose:** $V = H_0^1(\Omega)$ (functions with square-integrable first derivatives, zero on boundary)

**Inner product:**
$$
(u,v)_V = \int_\Omega \nabla u \cdot \nabla v \, dx
$$

### 3.4 Verification of Lax-Milgram Conditions

**(1) Continuity:**
$$
\begin{aligned}
|a(v,w)| &= \left|\int \nabla v \cdot \nabla w \, dx\right| \\
&\leq \|\nabla v\|_{L^2} \|\nabla w\|_{L^2} \quad \text{(Cauchy-Schwarz)} \\
&\leq \|v\|_V \|w\|_V
\end{aligned}
$$

**(2) Ellipticity:**
$$
a(v,v) = \int |\nabla v|^2 \, dx = |v|_{H^1}^2
$$

For $v \in H_0^1$, by **Poincaré inequality**: $\|v\|_{L^2} \leq C |v|_{H^1}$

Therefore: $a(v,v) \geq \alpha \|v\|_{H^1}^2$ for some $\alpha > 0$.

**(3) Bounded functional:**
$$
|L(v)| \leq \|f\|_{L^2} \|v\|_{L^2} \leq \|f\|_{L^2} \|v\|_V
$$

**All conditions satisfied!** $\checkmark$

---

## 4. Application: Biharmonic Equation

### 4.1 Physical Examples

The biharmonic equation arises in:
- **Acoustics**
- **Phase field models**
- **Beam bending**
- **Thin plate theory**

### 4.2 Problem Statement

$$
\begin{aligned}
\nabla^2 \nabla^2 u &= f \quad \text{in } \Omega \\
u|_{\partial\Omega} &= 0 \\
\partial_n u|_{\partial\Omega} &= 0
\end{aligned}
$$

(Both function and normal derivative vanish on boundary—**clamped plate conditions**)

### 4.3 Function Space

$$
V = H^2(\Omega) = \left\{ u : \int (\nabla^2 u)^2 \, dx < \infty \right\}
$$

(Functions with square-integrable second derivatives)

### 4.4 Weak Form

$$
\begin{aligned}
a(u,v) &= \int_\Omega \nabla^2 u \cdot \nabla^2 v \, dx \\
L(v) &= \int_\Omega f v \, dx
\end{aligned}
$$

### 4.5 Verification (Sketch)

**(1) Continuity:** Follows from Cauchy-Schwarz

**(2) Ellipticity:** This is **non-trivial**—requires a result from PDEs called **elliptic regularity**:

**Elliptic regularity theorem:** For convex $\Omega$,
$$
\|v\|_{H^2(\Omega)} \leq C \|\Delta v\|_{L^2(\Omega)} \quad \forall v \in H^2 \cap H_0^1
$$

This ensures $a(v,v) \geq \alpha \|v\|_{H^2}^2$.

**(3) Bounded functional:** 
$$
|L(v)| \leq \|f\|_{L^2} \|v\|_{L^2} \leq \|f\|_{L^2} \|v\|_V
$$

**All conditions satisfied!** $\checkmark$

---

## 5. Application: Reaction-Diffusion Equation

### 5.1 Problem Statement

$$
-\nabla \cdot (A \nabla u) + ru = f
$$

with assumptions:

**(A)** $A$ is positive definite: $x^\top A x > 0$ for all $x \neq 0$

**(B)** $r \geq 0$ (reaction coefficient non-negative)

**(C)** $|A_{ij}| < \infty$, $|r| < \infty$ (bounded coefficients)

### 5.2 Weak Form

$$
\begin{aligned}
a(u,v) &= \int_\Omega \nabla u \cdot A \cdot \nabla v + ruv \, dx \\
L(v) &= \int_\Omega fv \, dx
\end{aligned}
$$

where $v \in H_0^1(\Omega)$.

### 5.3 Verification

**(1) Continuity:**

**Diffusion term:**
$$
\begin{aligned}
\left|\int \nabla v \cdot A \cdot \nabla w \, dx\right| &\leq \max_{i,j} |A_{ij}| \left|\int \nabla v \cdot \nabla w \, dx\right| \\
&\leq \gamma_A |v|_{H^1} |w|_{H^1}
\end{aligned}
$$

**Reaction term:**
$$
\left|\int r v w \, dx\right| \leq \max|r| \int |vw| \, dx \leq \gamma_r \|v\|_{L^2} \|w\|_{L^2}
$$

**Combined:**
$$
|a(v,w)| \leq \gamma_A |v|_{H^1} |w|_{H^1} + \gamma_r \|v\|_{L^2} \|w\|_{L^2} \leq 2\max(\gamma_A, \gamma_r) \|v\|_{H^1} \|w\|_{H^1}
$$

**(2) Ellipticity:**
$$
\begin{aligned}
a(u,u) &= \int \nabla u \cdot A \cdot \nabla u + ru^2 \, dx \\
&\geq \int |\nabla u|^2 + u^2 \, dx \quad \text{(by positive definiteness and } r \geq 0\text{)} \\
&= \|u\|_{H^1}^2
\end{aligned}
$$

**(3) Bounded functional:** Same as Poisson.

**All conditions satisfied!** $\checkmark$

---

## 6. Linear Elasticity: When Things Go Wrong

### 6.1 Setup

**Displacement:** $u \in \mathbb{R}^d$

**Strain tensor:**
$$
\varepsilon(u) = \frac{1}{2}(\nabla u + \nabla u^\top) \in \mathbb{R}^{d \times d}
$$

(symmetric part of displacement gradient)

**Stress tensor:** $\sigma(\varepsilon)$

**Constitutive law (Hooke's law):**
$$
\sigma_{ij} = C_{ijkl} \varepsilon_{kl}
$$

### 6.2 Isotropic Materials

To enforce **rotational invariance** for isotropic materials:

$$
C_{ijkl} = \lambda \delta_{ij} \delta_{kl} + \mu(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})
$$

where:
- $\lambda$ = first Lamé parameter
- $\mu$ = shear modulus (second Lamé parameter)

The stress reduces to:

$$
\boxed{\sigma_{ij} = \lambda \delta_{ij} \varepsilon_{kk} + 2\mu \varepsilon_{ij}}
$$

or in vector notation:

$$
\sigma = 2\mu \varepsilon + \lambda \operatorname{tr}(\varepsilon) I
$$

**Note:** $\operatorname{tr}(\varepsilon) = \nabla \cdot u$ (volumetric strain)

### 6.3 Lagrangian Density

From variational mechanics, the Lagrangian density for elasticity is:

$$
\mathcal{L} = -\frac{1}{2} \varepsilon^\top C \varepsilon + \frac{\rho}{2} (\partial_t u)^2
$$

### 6.4 Governing Equations

$$
\begin{cases}
-\operatorname{div} \sigma = f \\
\sigma = 2\mu \varepsilon + \lambda \operatorname{tr}(\varepsilon) I \\
u|_{\partial\Omega} = 0
\end{cases}
$$

Combining:

$$
\begin{cases}
-\nabla \cdot (2\mu \varepsilon(u) + \lambda \nabla \cdot u \, I) = f \\
u|_{\partial\Omega} = 0
\end{cases}
$$

### 6.5 Weak Form

Work with displacements in vector space:

$$
\vec{u} \in \vec{H}_0^1(\Omega) = [H_0^1(\Omega)]^d
$$

(i.e., each component is in $H_0^1$)

$$
\boxed{a(u,v) = \int_\Omega \mu \varepsilon(u):\varepsilon(v) + \lambda(\nabla \cdot u)(\nabla \cdot v) \, dx}
$$

where $A:B = \sum_{ij} A_{ij} B_{ij}$ is the **Frobenius inner product**.

### 6.6 The Poisson Ratio and Near-Incompressibility

The Lamé parameters relate to engineering constants via:

$$
\lambda = \frac{2\mu \nu}{1 - 2\nu}
$$

where $\nu$ is the **Poisson ratio**:

$$
\nu = -\frac{\varepsilon_{\text{transverse}}}{\varepsilon_{\text{axial}}}
$$

**Problem:** For **near-incompressible materials** (e.g., rubber, hydrogels, clays, saturated porous media):

$$
\nu \to \frac{1}{2} \quad \Rightarrow \quad \lambda \to \infty
$$

**We need to understand the $\lambda \to \infty$ limiting stability!**

### 6.7 Korn's Inequality

To analyze this, we need **Korn's inequality**:

$$
\boxed{\int \varepsilon(u):\varepsilon(u) \, dx \geq C \|u\|_{H^1}^2}
$$

This says the strain energy controls the $H^1$ norm of displacement.

### 6.8 Verification of Lax-Milgram Conditions

**(1) Continuity:**

**Strain energy term:**
$$
\begin{aligned}
\left|\int \mu \varepsilon(w):\varepsilon(v) \, dx\right| &= \left|\frac{\mu}{4} \sum_{i,j} \int (\partial_{x_i} w_j + \partial_{x_j} w_i)(\partial_{x_i} v_j + \partial_{x_j} v_i) \, dx\right| \\
&= \mu \sum_{i,j} \int |\partial_{x_i} w_j|^2 \, dx \\
&\leq \mu \|w\|_{H^1} \|v\|_{H^1}
\end{aligned}
$$

**Volumetric term:**
$$
\left|\int \lambda (\nabla \cdot w)(\nabla \cdot v) \, dx\right| \leq \lambda \|w\|_{H^1} \|v\|_{H^1}
$$

**Combined:**
$$
|a(w,v)| \leq (\mu + \lambda) \|w\|_{H^1} \|v\|_{H^1}
$$

**(2) Ellipticity:**
$$
\begin{aligned}
a(u,u) &= \int \mu \varepsilon(u):\varepsilon(u) + \lambda(\nabla \cdot u)^2 \, dx \\
&\geq \int \mu \varepsilon(u):\varepsilon(u) \, dx \\
&\geq \mu \|u\|_{H^1}^2 \quad \text{(by Korn)}
\end{aligned}
$$

### 6.9 What Goes Wrong?

From Lax-Milgram:

**(1) Stability:** $\|u_h\|_{H^1} \leq \Lambda/\mu$ → **Get a stable solution**

**(2) By Céa's lemma:**
$$
\|u - u_h\|_{H^1} \leq \frac{\mu + \lambda}{\mu} \|u - v\|_{H^1}
$$

**Problem:** As $\lambda \to \infty$, the constant $\frac{\mu + \lambda}{\mu} \to \infty$!

**The error bound blows up, even though a solution exists.**

This is called **locking** in finite element analysis—the discrete solution loses accuracy as the material becomes incompressible.

---

## 7. Mixed Finite Element Methods: The Solution

### 7.1 Reformulation

One solution to locking is **mixed finite element methods**.

Take the original form:
$$
\mu \int \varepsilon(u):\varepsilon(v) \, dx + \lambda \int (\operatorname{div} u)(\operatorname{div} v) \, dx = \int f \cdot v \, dx
$$

**Introduce new variable:** Let $p = \lambda \operatorname{div} u$ (essentially a pressure).

Introduce a second FEM space $q \in M_h$.

### 7.2 Mixed Formulation

$$
\begin{cases}
\mu \int \varepsilon(u):\varepsilon(v) \, dx + \int p \nabla \cdot v \, dx = \int f \cdot v \, dx \\
\int (\nabla \cdot u) q \, dx + \frac{1}{\lambda} \int pq \, dx = 0
\end{cases}
$$

This gives a **new Galerkin equation**, but **how do we choose $M_h$?**

### 7.3 The Stokes Problem as Limiting Case

**Note:** In the limit $\mu = 1$, $\lambda \to \infty$, this reduces to the **stationary Stokes problem**:

$$(S) \quad \begin{cases}
-\Delta u - \nabla p = f \\
\nabla \cdot u = 0
\end{cases}$$

where:
- $u \in H_0^1$ (velocity field)
- $p \in L^2$ (pressure)

**Physical interpretation:** Incompressible viscous fluid flow.

### 7.4 Weak Form of Stokes

By integration by parts:
$$
(\nabla p, v) = -(p, \nabla \cdot v) \leq \|p\|_{L^2} \|v\|_{H^1}
$$

**Galerkin form:**
$$
\begin{cases}
(\nabla u, \nabla v) + (p, \nabla \cdot v) = (f, v) & \forall v \in V_h \\
(\nabla \cdot u, q) = 0 & \forall q \in M_h
\end{cases}
$$

This is a **saddle-point problem** (not a minimization).

---

## 8. Stability of Mixed Methods

### 8.1 Uniqueness (Partial)

We can prove uniqueness **quickly** for the case $f = 0 \Rightarrow p, u = 0$.

**Step 1:** Take $v = u$ in momentum equation:
$$
\|u\|_{H^1}^2 + (p, \nabla \cdot u) = 0
$$

**Step 2:** Take $q = p$ in continuity equation:
$$
(\nabla \cdot u, p) = 0
$$

**Step 3:** Substitute into Step 1:
$$
\|u\|_{H^1}^2 = 0 \quad \Rightarrow \quad u = 0
$$

**Problem:** We'd like to show $u = 0 \Rightarrow p = 0$, but **we can't!**

Nothing in the equations directly gives us information about $p$ when $u = 0$.

**Key idea:** Design a relationship between $V_h$ and $M_h$ so that $u = 0 \Rightarrow p = 0$.

---

## 9. The Inf-Sup Condition

### 9.1 Definition

$V_h$ and $M_h$ are **inf-sup compatible** if:

**For any $q \in M_h$, there exists $v \in V_h$ such that:**

$$
\boxed{\beta \|q\|_{L^2} \leq \frac{(q, \nabla \cdot v)}{\|v\|_{H^1}}}
$$

for some constant $\beta > 0$ independent of $h$.

**This is also called:**
- **Ladyzhenskaya-Babuška-Brezzi (LBB) condition**
- **Babuška-Brezzi condition**
- **Inf-sup condition**

### 9.2 Using Inf-Sup to Prove Uniqueness

Return to Stokes momentum equation with $f = 0$:

$$
(\nabla u, \nabla v) + (p, \nabla \cdot v) = 0
$$

We already showed $u = 0$.

**Take $q = p$ and use inf-sup:**

$$
\|p\|_{L^2} \leq \frac{(p, \nabla \cdot v)}{\beta \|v\|_{H^1}} = \frac{0}{\beta \|v\|_{H^1}} = 0
$$

**Therefore:** $p = 0$ $\quad \checkmark$

### 9.3 Physical Interpretation

The inf-sup condition ensures that **pressure can be uniquely determined from velocity**.

Without it, the pressure is under-determined (non-unique), leading to spurious pressure modes and numerical instabilities.

**Geometrically:** It says that the divergence operator from $V_h$ to $M_h$ has a bounded inverse (in a weak sense).

---

## 10. Summary

This lecture covered:

1. **Application of Lax-Milgram** to Poisson, biharmonic, and reaction-diffusion equations
2. **Linear elasticity** formulation with Lamé parameters
3. **Near-incompressibility** ($\nu \to 1/2$) and breakdown of standard FEM (locking)
4. **Korn's inequality** for ellipticity of elasticity operator
5. **Mixed finite element methods** as solution to locking
6. **Stokes problem** as limiting case of incompressible elasticity
7. **Saddle-point structure** of mixed formulations
8. **Inf-sup (LBB) condition** for stability and uniqueness of pressure

**Key Takeaway:** Not all PDEs can be solved with standard Galerkin FEM. When dealing with near-incompressibility (elasticity, Stokes flow), or more generally when solving for multiple coupled fields with different regularity requirements, we need **mixed finite element methods**. The inf-sup condition provides the mathematical criterion for choosing compatible function spaces $V_h$ and $M_h$—violate it, and you get spurious modes, locking, or non-unique solutions. Understanding this framework is essential for physics-informed machine learning: when learning operators or spaces for coupled multiphysics problems, we must preserve the algebraic structure (saddle-point form) and ensure learned spaces satisfy inf-sup compatibility to guarantee well-posed discrete problems.

**Next time:**
- How to build inf-sup stable spaces in practice
- How to learn physics in the mixed FEM setting

