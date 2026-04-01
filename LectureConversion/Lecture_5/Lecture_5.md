# Lecture 5: Numerical Stabilization and Accuracy Analysis via Polynomial Reproduction

**Reference:** [Gustafsson et al., Chapters 1-2](https://find.library.upenn.edu/catalog/9977071082103681?hld_id=53499053800003681), [Mirzaee et al., "On Generalized Moving Least Squares and Diffuse Derivatives"](https://doi.org/10.1093/imanum/drr030)

**Topics:** Artificial Viscosity, Lax-Friedrichs Method, Lax-Wendroff Method, Implicit Schemes, Crank-Nicolson Method, Polynomial Reproduction, Accuracy Theorems for Finite Difference Stencils

---

## 0. Overview

In Lecture 4, we proved the **Lax Equivalence Theorem**: for well-posed problems with finite initial conditions, **consistency + stability ⇒ convergence**. This landmark result tells us that if a finite difference scheme approximates the PDE correctly (consistency) and doesn't amplify errors uncontrollably (stability), then the numerical solution will converge to the true PDE solution as the grid is refined. However, we also discovered a troubling fact: the simple explicit Euler method with centered differences for the transport equation $\partial_t u = \partial_x u$ is **unconditionally unstable**, meaning $|\hat{Q}| > 1$ for all finite ratios $\lambda = k/h$ (where $k$ is the time step and $h$ is the spatial grid spacing). This scheme is consistent but unstable, and therefore it cannot converge—the amplification factor grows without bound, rendering the method useless in practice.

This raises a critical question: **How do we stabilize finite difference schemes?** In this lecture, we explore two fundamental approaches to achieving stability:

1. **Artificial viscosity/dissipation:** Add non-physical diffusion terms to damp high-frequency oscillations and control error growth. This technique leads to the **Lax-Friedrichs** and **Lax-Wendroff** methods, which are our first examples of numerical stabilization through carefully chosen dissipative regularization.

2. **Implicit time stepping:** Evaluate spatial derivatives at the future time level, requiring the solution of a linear system at each time step. This approach yields **unconditional stability** (stable for all $\lambda$), with the **Crank-Nicolson method** as a particularly elegant example that preserves $|\hat{Q}| = 1$ exactly.

Understanding these stabilization strategies is essential not just for classical numerical analysis, but also for modern scientific machine learning. When we later design learned discretizations from data, we'll need to incorporate stabilization mechanisms—either through architectural choices that mimic implicit schemes or through learned regularization that acts like artificial viscosity. The goal is to reverse-engineer models that are not only accurate but also **provably stable** under iteration.

After establishing how to stabilize schemes, we turn to a more general question: **How do we guarantee accuracy for arbitrary finite difference stencils?** The second half of this lecture develops a powerful theorem based on **polynomial reproduction**: if a stencil exactly reproduces derivatives of polynomials up to degree $m$, and if the stencil weights are appropriately bounded, then the stencil approximates the derivative of smooth functions with error $O(h^{m+1-|\alpha|})$, where $|\alpha|$ is the order of the derivative. This result, which connects to generalized moving least squares and meshfree methods, provides the theoretical foundation for designing high-order accurate discretizations.

---

## 1. Review: Lax Equivalence Theorem and the Instability Problem

Let's briefly recall the key results from Lecture 4 that motivate today's discussion.

### 1.1 The Lax Equivalence Theorem

**Lax Equivalence Theorem:** For a well-posed linear PDE with finite initial data, a finite difference scheme converges if and only if it is consistent and stable.

Formally, if:

1. $\|u_0\|^2 < \infty$ (finite initial condition)
2. $\sup |\hat{Q}^n| \leq K$ for all $n$ with $t_n = nk \leq T$ (stability)
3. $\lim_{k,h \to 0} \sup_{0 \leq t_n \leq T} |\hat{Q}^n(\xi) - e^{i\omega t_n}| = 0$ (consistency)

Then:
$$
\lim_{k,h \to 0} \sup_{0 \leq t_n \leq T} \|U(x, t_n) - \text{Int}_N(u(x, t_n))\| = 0
$$

where $\text{Int}_N$ is the trigonometric interpolation operator projecting the continuous solution onto the grid.

**Key insight:** For finite initial conditions, stability plus consistency implies convergence.

### 1.2 The Instability of Explicit Euler + Centered Differences

Consider the transport equation $\partial_t u = \partial_x u$ with the explicit Euler + centered difference scheme:

$$
\frac{v_j^{n+1} - v_j^n}{k} = D_0 v_j^n = \frac{v_{j+1}^n - v_{j-1}^n}{2h}
$$

**Fourier mode analysis:** For a single mode $e^{i\omega x}$, the amplification factor is:

$$
\hat{Q} = 1 + i\lambda \sin(\omega h)
$$

where $\lambda = k/h$ is the **CFL number** (Courant-Friedrichs-Lewy ratio).

**Stability analysis:**

$$
|\hat{Q}| = |1 + i\lambda \sin(\omega h)| = \sqrt{1 + \lambda^2 \sin^2(\omega h)} \geq 1
$$

For any finite $\lambda$, we have $|\hat{Q}| > 1$ for most modes, leading to unbounded growth. This scheme is:
- ✅ **Consistent** (truncation error → 0 as $k, h → 0$)
- ❌ **Unstable** (amplification factor > 1)
- ❌ **Does not converge** (by Lax Equivalence Theorem)

**This brings us to the central question:** Can we modify the scheme to achieve stability while preserving consistency?

---

## 2. Stabilization via Artificial Viscosity

One approach to stabilize explicit schemes is to add a **diffusion term** that damps high-frequency oscillations. This technique, called **artificial viscosity**, deliberately introduces non-physical dissipation to control numerical instability.

### 2.1 Adding Artificial Viscosity

Consider modifying the transport equation by adding a diffusive term:

$$
\partial_t u = \partial_x u + \sigma h \partial_{xx} u
$$

where $\sigma$ is a **stabilization parameter** and $h$ is the grid spacing.

**Key observations:**
- As $h \to 0$, the diffusion term $\sigma h \partial_{xx} u \to 0$, so we recover the true transport equation
- The added term is $O(h)$, which does not destroy consistency
- The diffusion damps high-frequency modes, providing numerical stabilization

The discrete scheme becomes:

$$
\frac{v_j^{n+1} - v_j^n}{k} = D_0 v_j^n + \sigma h D_+ D_- v_j^n
$$

where:
- $D_0 v_j^n = \frac{v_{j+1}^n - v_{j-1}^n}{2h}$ (centered difference for first derivative)
- $D_+ D_- v_j^n = \frac{v_{j+1}^n - 2v_j^n + v_{j-1}^n}{h^2}$ (second difference for $\partial_{xx}$)

### 2.2 Fourier Analysis of the Stabilized Scheme

**Step 1:** Compute the amplification factor.

Recall the action of finite difference operators on Fourier modes:

$$
\begin{aligned}
h D_0 e^{i\omega x} &= i \sin(\omega h) e^{i\omega x} \\
h^2 D_+ D_- e^{i\omega x} &= -4 \sin^2\left(\frac{\omega h}{2}\right) e^{i\omega x}
\end{aligned}
$$

Let $\lambda = k/h$ (CFL number) and $\xi = \omega h$ (scaled wavenumber). The amplification factor is:

$$
\hat{Q} = 1 + i\lambda \sin\xi - 4\sigma\lambda \sin^2\left(\frac{\xi}{2}\right)
$$

**Step 2:** Compute $|\hat{Q}|^2$.

Using $|a + ib|^2 = a^2 + b^2$:

$$
|\hat{Q}|^2 = \left(1 - 4\sigma\lambda \sin^2\frac{\xi}{2}\right)^2 + \lambda^2 \sin^2\xi
$$

**Step 3:** Expand and simplify.

Expanding the squared term:

$$
|\hat{Q}|^2 = 1 - 8\sigma\lambda \sin^2\frac{\xi}{2} + 16\sigma^2\lambda^2 \sin^4\frac{\xi}{2} + \lambda^2 \sin^2\xi
$$

Using the identity $\sin^2\xi = 4\sin^2(\xi/2)\cos^2(\xi/2)$:

$$
\lambda^2 \sin^2\xi = 4\lambda^2 \sin^2\frac{\xi}{2} \cos^2\frac{\xi}{2} = 4\lambda^2 \sin^2\frac{\xi}{2}\left(1 - \sin^2\frac{\xi}{2}\right)
$$

Collecting powers of $\sin^2(\xi/2)$:

$$
|\hat{Q}|^2 = 1 - \left(8\sigma\lambda - 4\lambda^2\right) \sin^2\frac{\xi}{2} + \left(16\sigma^2\lambda^2 - 4\lambda^2\right) \sin^4\frac{\xi}{2}
$$

Factoring the last term:

$$
|\hat{Q}|^2 = 1 - \left(8\sigma\lambda - 4\lambda^2\right) \sin^2\frac{\xi}{2} + 4\lambda^2\left(4\sigma^2 - 1\right) \sin^4\frac{\xi}{2}
$$

### 2.3 Stability Conditions

For stability, we require $|\hat{Q}|^2 \leq 1$ for all $\xi$. This can be achieved in two ways.

#### Strategy 1: Make Both Correction Terms Negative

Require both coefficients to be non-negative:

$$
\begin{aligned}
8\sigma\lambda - 4\lambda^2 &\geq 0 \quad \Rightarrow \quad \sigma \geq \frac{\lambda}{2} \\
4\lambda^2(4\sigma^2 - 1) &\leq 0 \quad \Rightarrow \quad \sigma \leq \frac{1}{2}
\end{aligned}
$$

**Conclusion:** Choose $\frac{\lambda}{2} \leq \sigma \leq \frac{1}{2}$ with $0 < \lambda \leq 1$.

This gives the **Lax-Friedrichs method** at $\sigma = \frac{1}{2\lambda}$.

#### Strategy 2: Balance at Worst-Case Mode

Consider the worst-case scenario where $\sin^2(\xi/2) = 1$ (i.e., $\xi = \pi$, the Nyquist frequency):

$$
|\hat{Q}|^2 = 1 - (8\sigma\lambda - 4\lambda^2) + 4\lambda^2(4\sigma^2 - 1) \leq 1
$$

Simplifying:

$$
0 \leq 8\sigma\lambda - 4\lambda^2 - 16\sigma^2\lambda^2 + 4\lambda^2 = 8\sigma\lambda - 16\sigma^2\lambda^2
$$

Factoring:

$$
8\sigma\lambda(1 - 2\sigma\lambda) \geq 0
$$

Since $\sigma, \lambda > 0$, this requires:

$$
2\sigma\lambda \leq 1 \quad \Rightarrow \quad \sigma \leq \frac{1}{2\lambda}
$$

**Conclusion:** For $\sigma \geq \frac{1}{2}$, stability requires $\lambda \leq \frac{1}{2\sigma}$.

This gives the **Lax-Wendroff method** at $\sigma = \frac{\lambda}{2}$ (satisfying $2\sigma\lambda = \lambda^2 \leq 1$ when $\lambda \leq 1$).

---

## 3. The Lax-Friedrichs and Lax-Wendroff Methods

From the stability analysis above, we obtain two classical stable schemes for the transport equation.

### 3.1 Lax-Friedrichs Method

**Choice:** $\sigma = \frac{h}{2k} = \frac{1}{2\lambda}$

**Scheme:**

$$
v_j^{n+1} = v_j^n + k D_0 v_j^n + \frac{h^2}{2k} D_+ D_- v_j^n
$$

Simplifying:

$$
v_j^{n+1} = v_j^n + \frac{k}{2h}(v_{j+1}^n - v_{j-1}^n) + \frac{h}{2k} \cdot \frac{v_{j+1}^n - 2v_j^n + v_{j-1}^n}{h^2}
$$

$$
v_j^{n+1} = \frac{1}{2}(v_{j+1}^n + v_{j-1}^n) + \frac{k}{2h}(v_{j+1}^n - v_{j-1}^n)
$$

**Alternative form (often given in textbooks):**

$$
v_j^{n+1} = \frac{1}{2}\left(v_{j+1}^n + v_{j-1}^n\right) - \frac{\lambda}{2}\left(v_{j+1}^n - v_{j-1}^n\right)
$$

where $\lambda = k/h$.

**Interpretation:**
- Replaces $v_j^n$ with the average $(v_{j+1}^n + v_{j-1}^n)/2$, introducing strong dissipation
- Stable for $\lambda \leq 1$ (CFL condition)
- First-order accurate in time and second-order accurate in space

### 3.2 Lax-Wendroff Method

**Choice:** $\sigma = \frac{k}{2h} = \frac{\lambda}{2}$

**Scheme:**

$$
v_j^{n+1} = v_j^n + k D_0 v_j^n + \frac{k^2}{2h} D_+ D_- v_j^n
$$

Expanding:

$$
v_j^{n+1} = v_j^n + \frac{k}{2h}(v_{j+1}^n - v_{j-1}^n) + \frac{k^2}{2h^3}(v_{j+1}^n - 2v_j^n + v_{j-1}^n)
$$

**Interpretation:**
- The $k^2$ term naturally arises from a Taylor series expansion in time combined with the PDE
- Represents a second-order accurate approximation in both space and time
- Stable for $\lambda \leq 1$

**Derivation motivation (Taylor series):**

Starting from $u_t = u_x$, we have $u_{tt} = u_{xt} = u_{xx}$. A Taylor expansion in time gives:

$$
u(t + k) = u(t) + k u_t + \frac{k^2}{2} u_{tt} = u(t) + k u_x + \frac{k^2}{2} u_{xx}
$$

The Lax-Wendroff method discretizes this expression exactly!

### 3.3 Summary: Artificial Viscosity for Stability

**Key Takeaways:**
- Artificial viscosity provides our first example of **numerical stabilization** by adding dissipative terms
- The parameter $\sigma$ controls the amount of added diffusion
- Often we can stabilize schemes by adding dissipative terms with a tunable "fudge factor"
- Both Lax-Friedrichs and Lax-Wendroff are conditionally stable ($\lambda \leq 1$)
- The price: added dissipation can smear sharp features (numerical diffusion)

**Important note:** These methods add non-physical terms to achieve stability. While practical, this approach has limitations—excessive dissipation can corrupt solutions. An alternative is to use **implicit schemes**, which achieve stability through algebraic structure rather than added viscosity.

---

## 4. Implicit Schemes and Unconditional Stability

Instead of adding artificial viscosity, we can achieve stability by evaluating spatial derivatives at the **future time level** $n+1$ rather than the current level $n$. This leads to **implicit schemes**, which require solving a linear system at each time step.

### 4.1 Implicit Euler Scheme

**Scheme:**

$$
\frac{v_j^{n+1} - v_j^n}{k} = D_0 v_j^{n+1}
$$

Rearranging:

$$
v_j^{n+1} - k D_0 v_j^{n+1} = v_j^n
$$

or in operator form:

$$
(I - k D_0) v^{n+1} = v^n
$$

**Fourier analysis:**

For a mode $e^{i\omega x}$:

$$
\hat{Q} = [1 - i\lambda \sin\xi]^{-1}
$$

**Stability:**

$$
|\hat{Q}| = \frac{1}{|1 - i\lambda \sin\xi|} = \frac{1}{\sqrt{1 + \lambda^2 \sin^2\xi}} \leq 1
$$

for all $\lambda, \xi$.

**Conclusion:** The implicit Euler scheme is **unconditionally stable** for any CFL number $\lambda = k/h$.

**Tradeoff:**
- ✅ Stable for arbitrarily large time steps
- ❌ Requires solving a linear system $(I - k D_0) v^{n+1} = v^n$ at each step
- ❌ Only first-order accurate in time

### 4.2 Crank-Nicolson Method

A particularly elegant implicit scheme is the **Crank-Nicolson method**, which uses a **trapezoidal rule** in time by averaging the spatial derivative at times $n$ and $n+1$.

**Scheme:**

$$
\frac{v_j^{n+1} - v_j^n}{k} = \frac{1}{2} D_0 v_j^{n+1} + \frac{1}{2} D_0 v_j^n
$$

Rearranging:

$$
\left(I - \frac{k}{2} D_0\right) v_j^{n+1} = \left(I + \frac{k}{2} D_0\right) v_j^n
$$

**Fourier analysis:**

$$
\hat{Q} = \frac{1 + \frac{i\lambda}{2}\sin\xi}{1 - \frac{i\lambda}{2}\sin\xi}
$$

**Stability:**

$$
|\hat{Q}| = \left|\frac{1 + \frac{i\lambda}{2}\sin\xi}{1 - \frac{i\lambda}{2}\sin\xi}\right| = \frac{\sqrt{1 + (\lambda\sin\xi/2)^2}}{\sqrt{1 + (\lambda\sin\xi/2)^2}} = 1
$$

**Remarkable property:** $|\hat{Q}| = 1$ **exactly** for all modes!

**Advantages:**
- ✅ **Unconditionally stable** for all $\lambda$
- ✅ **No numerical dissipation** ($|\hat{Q}| = 1$ means energy is conserved)
- ✅ **Second-order accurate in time** (trapezoidal rule)

**Disadvantages:**
- ❌ Requires solving a linear system at each step
- ❌ Can exhibit oscillations for steep gradients (no dissipation to smooth them)

### 4.3 Comparison: Explicit vs. Implicit Schemes

| **Method** | **Type** | **Stability** | **Accuracy** | **Cost per step** |
|------------|----------|---------------|--------------|-------------------|
| Explicit Euler + Centered | Explicit | Unstable | 1st order time | Low (explicit) |
| Lax-Friedrichs | Explicit | $\lambda \leq 1$ | 1st order time | Low (explicit) |
| Lax-Wendroff | Explicit | $\lambda \leq 1$ | 2nd order time | Low (explicit) |
| Implicit Euler | Implicit | Unconditional | 1st order time | High (linear solve) |
| Crank-Nicolson | Implicit | Unconditional | 2nd order time | High (linear solve) |

**Practical considerations:**
- Explicit methods: cheap per step, but require small time steps for stability
- Implicit methods: expensive per step (linear solve), but allow large time steps
- Crank-Nicolson: "best of both worlds" for smooth problems (2nd order, no dissipation)

---

## 5. Accuracy Analysis via Polynomial Reproduction

Now that we understand how to stabilize finite difference schemes, we turn to a more general question: **How do we guarantee the accuracy of arbitrary finite difference stencils?** The answer lies in a beautiful theorem connecting **polynomial reproduction** to **convergence rates**.

This theory is particularly important for modern methods like **meshfree methods**, **generalized finite differences**, and **learned discretizations**, where stencils may be irregular or adapted to data. The key reference is the paper by Mirzaee et al., ["On Generalized Moving Least Squares and Diffuse Derivatives"](https://doi.org/10.1093/imanum/drr030).

### 5.1 Setting: Generalized Finite Difference Stencils

Consider a **generic finite difference stencil** approximating a differential operator $D^\alpha u$, where $\alpha$ is a **multi-index** denoting the order of derivatives.

**Multi-index notation:**
- $\alpha = (\alpha_1, \alpha_2, \ldots)$ is a tuple of non-negative integers
- $D^\alpha = \partial_x^{\alpha_1} \partial_y^{\alpha_2} \cdots$
- $|\alpha| = \alpha_1 + \alpha_2 + \cdots$ is the **total order** of the derivative

**Examples:**
- $\alpha = (1, 0) \Rightarrow D^\alpha = \partial_x$, $|\alpha| = 1$
- $\alpha = (2, 0) \Rightarrow D^\alpha = \partial_{xx}$, $|\alpha| = 2$
- $\alpha = (1, 2) \Rightarrow D^\alpha = \partial_x \partial_{yy}$, $|\alpha| = 3$

**Generic stencil:** A discrete approximation $D_h^\alpha$ at grid point $x_i$ is given by:

$$
D_h^\alpha u_i = \sum_j S_{ji}^\alpha u_j
$$

where $S_{ji}^\alpha$ are **stencil weights** and the sum is over neighboring grid points $x_j$.

**Key questions:**
1. How do we choose stencil weights $S_{ji}^\alpha$ to guarantee accuracy?
2. What convergence rate can we expect as $h \to 0$?

### 5.2 The Polynomial Reproduction Theorem

The following theorem provides a complete answer, establishing that **polynomial reproduction** is both necessary and sufficient for optimal accuracy.

**Theorem (Polynomial Reproduction for Accuracy):**

Suppose the stencil $D_h^\alpha$ satisfies:

1. **Polynomial reproduction of order $m$:**
   $$
   \sum_j S_{ji}^\alpha p_j = D^\alpha p(x_i) \quad \text{for all polynomials } p \in \mathcal{P}_m
   $$
   where $\mathcal{P}_m$ is the space of polynomials of degree at most $m$.

2. **Bounded stencil weights:**
   $$
   \sum_j |S_{ji}^\alpha| \leq C h^{-|\alpha|}
   $$
   where $C$ is independent of $h$.

Then for smooth functions $u \in C^{m+1}$:

$$
\|D_h^\alpha u - D^\alpha u\|_{L^\infty} \leq C h^{m+1-|\alpha|} |u|_{C^{m+1}}
$$

where $|u|_{C^{m+1}} = \max_{|\beta| = m+1} \|\partial^\beta u\|_{L^\infty}$ is the $(m+1)$-th derivative seminorm.

**Interpretation:**
- If the stencil exactly reproduces derivatives of polynomials up to degree $m$, then it approximates derivatives of smooth functions with error $O(h^{m+1-|\alpha|})$
- Higher-order polynomial reproduction ⇒ higher-order accuracy
- The weight bound condition $\sum |S_{ji}^\alpha| \leq C h^{-|\alpha|}$ ensures stability (prevents stencil weights from blowing up)

### 5.3 Proof of the Polynomial Reproduction Theorem

The proof is a beautiful application of polynomial approximation theory, combining the triangle inequality with Taylor series estimates.

**Step 1:** Use the triangle inequality to separate exact and approximate derivatives.

Let $p \in \mathcal{P}_m$ be an arbitrary polynomial of degree at most $m$. For any function $u$:

$$
|D_h^\alpha u - D^\alpha u| \leq |D^\alpha u - D^\alpha p| + |D^\alpha p - D_h^\alpha p| + |D_h^\alpha p - D_h^\alpha u|
$$

**Step 2:** Use polynomial reproduction to eliminate the middle term.

By assumption (1), the stencil exactly reproduces derivatives of $p$:

$$
D_h^\alpha p(x_i) = \sum_j S_{ji}^\alpha p_j = D^\alpha p(x_i)
$$

Therefore:

$$
|D_h^\alpha u - D^\alpha u| \leq |D^\alpha u - D^\alpha p| + |D_h^\alpha p - D_h^\alpha u|
$$

**Step 3:** Apply the stencil to the difference $u - p$.

$$
|D_h^\alpha u - D_h^\alpha p| = \left|\sum_j S_{ji}^\alpha (u_j - p_j)\right| \leq \sum_j |S_{ji}^\alpha| |u_j - p_j|
$$

Taking the $L^\infty$ norm (maximum over all grid points):

$$
\|D_h^\alpha u - D_h^\alpha p\|_{L^\infty} \leq \|u - p\|_{L^\infty} \sum_j |S_{ji}^\alpha|
$$

**Step 4:** Use the weight bound assumption (2).

$$
\|D_h^\alpha u - D_h^\alpha p\|_{L^\infty} \leq C h^{-|\alpha|} \|u - p\|_{L^\infty}
$$

**Step 5:** Combine terms.

$$
\|D_h^\alpha u - D^\alpha u\|_{L^\infty} \leq \|D^\alpha u - D^\alpha p\|_{L^\infty} + C h^{-|\alpha|} \|u - p\|_{L^\infty}
$$

**Step 6:** Choose $p$ as the Taylor polynomial of $u$ at $x_i$.

By Taylor's theorem, the optimal polynomial approximation of degree $m$ gives:

$$
\begin{aligned}
\|u - p\|_{L^\infty} &\leq C_1 h^{m+1} |u|_{C^{m+1}} \\
\|D^\alpha u - D^\alpha p\|_{L^\infty} &\leq C_2 h^{m+1-|\alpha|} |u|_{C^{m+1}}
\end{aligned}
$$

**Step 7:** Substitute into the bound.

$$
\begin{aligned}
\|D_h^\alpha u - D^\alpha u\|_{L^\infty} &\leq C_2 h^{m+1-|\alpha|} |u|_{C^{m+1}} + C h^{-|\alpha|} \cdot C_1 h^{m+1} |u|_{C^{m+1}} \\
&= (C_2 + C C_1) h^{m+1-|\alpha|} |u|_{C^{m+1}}
\end{aligned}
$$

**Conclusion:**

$$
\boxed{\|D_h^\alpha u - D^\alpha u\|_{L^\infty} \leq C h^{m+1-|\alpha|} |u|_{C^{m+1}}}
$$

This completes the proof. ∎

### 5.4 Implications and Applications

**Critical observations:**

1. **Polynomial reproduction is the key to accuracy:** To achieve $m$-th order accuracy for a $|\alpha|$-th order derivative, the stencil must exactly reproduce derivatives of polynomials up to degree $m$.

2. **Standard finite differences satisfy these conditions:** For example, the centered difference stencil $D_0 u_j = (u_{j+1} - u_{j-1})/(2h)$ reproduces first derivatives of polynomials up to degree 2, giving $O(h^2)$ accuracy for $\partial_x u$.

3. **Generalized to irregular grids:** The theorem applies to **arbitrary stencils**, not just uniform grids. This is crucial for:
   - **Meshfree methods** (moving least squares, RBF methods)
   - **Generalized finite differences** on scattered points
   - **Learned discretizations** with data-adapted stencils

4. **Connection to moving least squares:** The Mirzaee et al. paper referenced above shows that generalized moving least squares (GMLS) methods automatically satisfy polynomial reproduction, making them a powerful tool for constructing high-order accurate stencils on irregular grids.

5. **Design principle for learned discretizations:** When learning stencil weights from data (as we'll do later in this course), we can **enforce polynomial reproduction as a constraint** to guarantee convergence. This is an example of incorporating mathematical structure into machine learning architectures.

**Example:** For the second derivative $\partial_{xx} u$, a standard centered difference stencil is:

$$
D_h^{(2,0)} u_i = \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2}
$$

This stencil reproduces $\partial_{xx} p$ for all polynomials $p$ up to degree 3 (quartic polynomials have constant fourth derivatives, which are captured exactly). The weight bound is:

$$
\sum_j |S_{ji}| = \frac{1 + 2 + 1}{h^2} = \frac{4}{h^2} = C h^{-2}
$$

with $|\alpha| = 2$. Therefore, this stencil achieves $O(h^{3-2}) = O(h)$... wait, that doesn't sound right!

**Correction:** Actually, the centered difference for $\partial_{xx}$ reproduces polynomials up to degree 2, not 3. Cubics and quartics have different second derivatives. The correct accuracy is $O(h^{2+1-2}) = O(h)$ for generic $u$, but for smooth $u$ we actually get $O(h^2)$ due to symmetry (odd-order errors cancel). The theorem gives the worst-case bound; in practice, specific stencils may be more accurate due to special structure.

---

## 6. Summary

This lecture developed two fundamental tools for analyzing and designing finite difference schemes:

1. **Numerical stabilization techniques:**
   - **Artificial viscosity** (Lax-Friedrichs, Lax-Wendroff): Add dissipative terms to control growth
   - **Implicit time stepping** (Implicit Euler, Crank-Nicolson): Evaluate spatial derivatives at future time level

2. **Polynomial reproduction theorem:**
   - Stencils that exactly reproduce derivatives of polynomials up to degree $m$ achieve $O(h^{m+1-|\alpha|})$ accuracy
   - Bounded stencil weights ensure stability
   - Applies to arbitrary stencils, including irregular grids and learned discretizations

### Main Topics Covered:

1. **Review of Lax Equivalence Theorem** and the instability problem for explicit Euler + centered differences
2. **Artificial viscosity:** Adding $\sigma h \partial_{xx} u$ to stabilize transport equation
3. **Lax-Friedrichs method:** $\sigma = \frac{1}{2\lambda}$, stable for $\lambda \leq 1$
4. **Lax-Wendroff method:** $\sigma = \frac{\lambda}{2}$, second-order accurate in space and time
5. **Implicit schemes:** Unconditional stability by evaluating derivatives at $t^{n+1}$
6. **Crank-Nicolson method:** $|\hat{Q}| = 1$ exactly, second-order accurate, no dissipation
7. **Polynomial reproduction theorem:** Accuracy guarantees for generalized stencils
8. **Proof technique:** Triangle inequality + Taylor expansion + polynomial reproduction
9. **Applications:** Meshfree methods, learned discretizations, irregular grids

**Key Takeaway:** Stability can be achieved either by adding dissipation (artificial viscosity) or by implicit formulations (solving linear systems). For learning problems, we must design architectures that either incorporate stabilization mechanisms or learn them from data while preserving convergence guarantees. The polynomial reproduction theorem provides a powerful tool for ensuring accuracy of both conventional and learned discretizations on arbitrary grids.

**Looking ahead:** In the next lectures, we'll explore how to apply these stability and accuracy principles to design physics-informed neural networks and learned discretizations that provably converge to PDE solutions.

---

