# ENM 5320 - Take-Home Exam: Finite Difference Methods
**Due Date:** March 16, 2026 - In class, turned in on paper  
**Total Points:** 100

---

## Instructions

This exam covers finite difference methods, stability analysis, and the Lax Equivalence Theorem as discussed in Lectures 2-5 and the hackathon sessions. Answer all questions, showing your work clearly. You may use course notes and the textbook, but all work must be your own.

---

## Problem 1: Implicit Euler Method for the Heat Equation (100 points)

Consider the 1D heat equation on a periodic domain $x \in [0, 1]$:

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

where $\alpha > 0$ is the thermal diffusivity constant.

We will discretize this equation using:
- **Spatial discretization**: Centered finite differences
- **Time discretization**: Implicit Euler (backward Euler)

Let $h = \frac{1}{N}$ be the spatial grid spacing and $k$ be the time step. Denote $v_j^n \approx u(x_j, t_n)$ where $x_j = jh$ and $t_n = nk$.

---

### Relevant Equations

**Centered difference operator for second derivatives:**
$$
D_+D_- v_j = \frac{1}{h^2}(v_{j+1} - 2v_j + v_{j-1})
$$

**Implicit Euler time discretization:**
$$
\frac{v_j^{n+1} - v_j^n}{k} = \text{[spatial operator evaluated at time } n+1]
$$

**Triangle inequality:**
$$
\|u + v\|_h \leq \|u\|_h + \|v\|_h
$$

**Discrete norm:**
$$
\|u\|_h^2 = \sum_{j=0}^{N-1} |u_j|^2 h
$$

**Operator norm:**
$$
\|Q\|_h = \sup_{u \neq 0} \frac{\|Q u\|_h}{\|u\|_h}
$$

**Lax Equivalence Theorem:** For a well-posed linear PDE with finite initial data, a finite difference scheme converges if and only if it is:
1. **Consistent**: The local truncation error approaches zero as $h, k \to 0$
2. **Stable**: The amplification operator $Q$ satisfies $\sup |\hat{Q}^n| \leq K$ for some constant $K$ independent of $h, k$

**Useful trigonometric and exponential identities:**
$$
e^{i\theta} = \cos\theta + i\sin\theta
$$
$$
e^{i\theta} + e^{-i\theta} = 2\cos\theta
$$
$$
e^{i\theta} - e^{-i\theta} = 2i\sin\theta
$$
$$
1 - \cos\theta = 2\sin^2\left(\frac{\theta}{2}\right)
$$
$$
|e^{i\theta}| = 1 \quad \text{for all real } \theta
$$

**Parseval's Theorem (Discrete):** If $v$ can be written as a sum of Fourier modes $v = \sum_{\omega} \tilde{v}(\omega) e^{i\omega x}$, then:
$$
\|v\|_h^2 = \sum_{\omega} |\tilde{v}(\omega)|^2
$$
This means the norm in physical space equals the sum of squared magnitudes in frequency space.

---

### Part A: Derive the Finite Difference Stencil (25 points)

Starting from the heat equation $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$, derive the implicit Euler finite difference scheme.

**Sub-questions:**

a) **(5 points)** Write down the implicit Euler approximation for the time derivative $\frac{\partial u}{\partial t}$ at point $(x_j, t_{n+1})$.

b) **(10 points)** Write down the centered finite difference approximation for $\frac{\partial^2 u}{\partial x^2}$ at point $(x_j, t_{n+1})$. 

c) **(10 points)** Combine your answers from (a) and (b) to write the full discrete scheme. Rearrange it into the form:
$$
v_j^{n+1} + \beta(v_{j-1}^{n+1} - 2v_j^{n+1} + v_{j+1}^{n+1}) = v_j^n
$$
and identify the coefficient $\beta$ in terms of $\alpha$, $k$, and $h$.

---

### Part B: Show Consistency (20 points)

Show that the scheme you derived is consistent with the heat equation.

**Sub-questions:**

a) **(10 points)** Assume $u(x, t)$ is a smooth solution to the heat equation. Using Taylor expansions, show that:
$$
\frac{u(x_j, t_{n+1}) - u(x_j, t_n)}{k} = \frac{\partial u}{\partial t}(x_j, t_{n+1}) + O(k)
$$

b) **(10 points)** Using Taylor expansions, show that:
$$
\frac{u(x_{j+1}, t) - 2u(x_j, t) + u(x_{j-1}, t)}{h^2} = \frac{\partial^2 u}{\partial x^2}(x_j, t) + O(h^2)
$$

c) **(Bonus: 5 points)** Combine parts (a) and (b) to show that the local truncation error is $O(k) + O(h^2)$, which approaches zero as $h, k \to 0$.

---

### Part C: Show the Operator is Bounded (35 points)

We can write the implicit Euler scheme in operator form as:
$$
(I - \mu D_+D_-) v^{n+1} = v^n
$$
where $\mu = \alpha k / h^2$ and $D_+D_-$ is the centered second-difference operator.

This defines the amplification operator $Q$ via $v^{n+1} = Q v^n$ where $Q = (I - \mu D_+D_-)^{-1}$.

**Sub-questions:**

a) **(10 points)** Consider a Fourier mode $v_j = e^{i\xi j}$ where $\xi = \omega h$ is the discrete wavenumber. Show that:
$$
D_+D_- v_j = \frac{-4}{h^2} \sin^2\left(\frac{\xi}{2}\right) v_j
$$ (stated in the Relevant Equations section)

**Hint:** Use $D_+v_j = \frac{1}{h}(e^{i\xi} - 1)v_j$ and $D_-v_j = \frac{1}{h}(1 - e^{-i\xi})v_j$.

b) **(15 points)** Using the result from part (a), show that the Fourier symbol (amplification factor for a single mode) is:
$$
\hat{Q}(\xi) = \frac{1}{1 + 4\mu \sin^2(\xi/2)}
$$

c) **(10 points)** Note that $\hat{Q}(\xi)$ is a real number since $\mu > 0$ and $\sin^2(\xi/2) \geq 0$. Show that:
$$
|\hat{Q}(\xi)| \leq 1
$$
for all $\xi$ when $\mu > 0$. This proves the scheme is unconditionally stable.

**Hint:** For a positive real number of the form $\frac{1}{1+a}$ where $a \geq 0$, what is the maximum value this fraction can take?

---

### Part D: Estimate the Operator Norm (10 points)

**Sub-questions:**

a) **(5 points)** Using the result from Part C(c) and Parseval's theorem, argue that:
$$
\|Q v\|_h \leq \|v\|_h
$$
for any grid function $v$.

b) **(5 points)** What does this tell you about the operator norm $\|Q\|_h$? Give an upper bound.

---

### Part E: Apply the Lax Equivalence Theorem (10 points)

**Sub-questions:**

a) **(5 points)** You have shown in Parts B and C that the implicit Euler scheme for the heat equation is:
   - Consistent (local truncation error $\to 0$ as $h, k \to 0$)
   - Stable ($|\hat{Q}| \leq 1$ for all modes)

State the conclusion that follows from the Lax Equivalence Theorem.

b) **(5 points)** Explain in 1-2 sentences why this result is powerful: what does it guarantee about your numerical solution as you refine the grid?

---

## Submission Guidelines

- Show all steps clearly
- Box or highlight final answers
- Include units and definitions where appropriate
- If you get stuck on one part, continue to the next—partial credit will be awarded
- Take an honest shot at this without using AI, as it will force you to consolidate the information from all of the lectures.

**Good luck!**
