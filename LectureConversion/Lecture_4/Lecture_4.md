# Lecture 4: Discrete Norms, Stability Analysis, and the Lax Equivalence Theorem

**Reference:** [Gustafsson et al., Chapters 1.3-1.4](https://find.library.upenn.edu/catalog/9977071082103681?hld_id=53499053800003681)

**Topics:** Discrete Inner Products and Norms, Operator Norms, Discrete Fourier Transform, Trigonometric Interpolation, Discrete Parseval Theorem, Lax Equivalence Theorem

---

## 0. Overview

In Lecture 2, we introduced the basic finite difference operators and explored stability concepts through Fourier analysis of individual modes. We saw how explicit Euler integration of the transport equation leads to unconditional instability, highlighting the critical importance of understanding when and why finite difference schemes succeed or fail. However, our analysis remained somewhat informal, relying on inspecting how amplification factors behave for different wavelengths.

This lecture takes a rigorous turn, developing the mathematical framework needed to formally analyze finite difference schemes. We introduce **discrete inner products and norms**, which allow us to quantify the magnitude of grid functions in a way that connects smoothly to continuous function spaces as the grid spacing $h \to 0$. With these tools in hand, we'll compute **operator norms** for our finite difference operators (forward, backward, and centered differences), revealing exactly how much these discrete derivatives can amplify perturbations.

A key tool emerges from this analysis: the **discrete Fourier transform** and trigonometric interpolation. Just as Fourier analysis decomposes continuous functions into sums of exponentials $e^{i\omega x}$, we'll see that grid functions can be uniquely represented by trigonometric polynomials. This discrete-continuous bridge is formalized through the **discrete Parseval theorem**, which connects grid function norms to their Fourier coefficient magnitudes. These results provide the foundation for analyzing how finite difference operators act on different frequency modes.

The lecture culminates with the **Lax Equivalence Theorem**, one of the most fundamental results in numerical analysis of PDEs. This theorem provides a complete characterization of when finite difference schemes converge to the true solution: **consistency plus stability equals convergence**. Understanding this theorem is essential not just for analyzing conventional numerical methods, but also for designing physics-informed machine learning architectures that inherit convergence guarantees. When we later reverse-engineer learned discretizations from data, we'll want to ensure our learned operators satisfy stability constraints that guarantee reliable predictions.

---

## 1. Discrete Inner Products and Norms

Building on our work with grid functions from Lecture 2, we now formalize how to measure the "size" of discrete data in a way that naturally extends to continuous functions.

### 1.1 Definition of Discrete Inner Product

Previously, we talked about grid functions and interpreted finite differences in terms of how they act on individual modes. Now we formalize this framework rigorously.

**Definition:** For grid functions $u$ and $v$ defined on a uniform grid with spacing $h = \frac{2\pi}{N+1}$ and points $x_j = jh$ for $j = 0, 1, \ldots, N$, we define the **discrete inner product**:

$$
(u, v)_{h} = \sum_{j=0}^{N} \bar{u}_{j} v_{j} h
$$

where $\bar{u}_j$ denotes the complex conjugate of $u_j$. This inner product induces the **discrete norm**:

$$
\|u\|_{h}^{2} = (u, u)_{h} = \sum_{j=0}^{N} |u_{j}|^{2} h
$$

**Important Notes:**
- The factor $h$ in the summation ensures dimensional consistency: if $u$ represents a physical quantity with units, $\|u\|_h$ has the proper units when integrated over the domain
- The discrete inner product is a Riemann sum approximation to the continuous inner product $(u, v) = \int_0^{2\pi} \bar{u}(x) v(x) dx$
- As $h \to 0$, we have $(u, v)_h \to (u, v)$ and $\|u\|_h \to \|u\|$ for smooth functions

### 1.2 Properties of Discrete Inner Products

All the fundamental inequalities from continuous analysis carry over to the discrete setting:

**Cauchy-Schwarz Inequality:**
$$
|(u, v)_{h}| \leq \|u\|_{h} \|v\|_{h}
$$

**Discrete Hölder Inequality:**
$$
|(u, av)_{h}| \leq \|a\|_{\infty} \|u\|_{h} \|v\|_{h}
$$
where $\|a\|_{\infty} = \max_j |a_j|$ is the discrete supremum norm.

**Triangle Inequality:**
$$
\|u + v\|_{h} \leq \|u\|_{h} + \|v\|_{h}
$$

**Reverse Triangle Inequality:**
$$
\left| \|u\|_{h} - \|v\|_{h} \right| \leq \|u - v\|_{h}
$$

These properties ensure that discrete norms behave like proper norms, satisfying all the axioms needed for rigorous analysis.

---

## 2. Stability and Operator Norms

With norms defined, we can now quantify exactly how finite difference operators affect the magnitude of grid functions.

### 2.1 Motivation for Operator Norms

Consider a finite difference scheme for evolving a PDE in time:

$$
v_j^{n+1} = Q v_j^n
$$

where $Q$ is the **amplification operator** that advances the solution from time step $n$ to $n+1$.

**Key Questions:**
- How do we distinguish the **true continuous evolution** from the **discrete finite difference evolution**?
- What happens to **truncation error** as we iterate the scheme?
- How does the amplification operator $Q$ affect solution magnitude?

To answer these questions, we need to understand the operator norm $\|Q\|_h$.

**Interpretation of operator norm:**
- $\|Q\|_h = 1$: Discrete conservation (energy/norm preserved)
- $\|Q\|_h > 1$: Growth (BAD - leads to instability)
- $\|Q\|_h < 1$: Artificial dissipation (possibly bad or good, depending on context)

**Connection to continuous functions:** For $u \in C^k$ (a smooth function), consider its restriction to grid points $u(x_j)$. Then:
$$
\lim_{h \to 0} (u, v)_h = (u, v), \quad \lim_{h \to 0} \|u\|_h^2 = \|u\|^2
$$

This limiting behavior ensures our discrete analysis connects smoothly to the continuous case.

### 2.2 Definition of Operator Norm

**Definition:** The norm of a finite difference operator $Q$ is:

$$
\|Q\|_{h} := \sup_{u \neq 0} \frac{\|Q u\|_{h}}{\|u\|_{h}} = \sup_{\|u\|_h=1} \|Q u\|_{h}
$$

This definition immediately implies the fundamental inequality:

$$
\|Q u\|_{h} \leq \|Q\|_{h} \|u\|_{h}
$$

**Intuition:** The operator norm $\|Q\|_h$ measures the maximum amplification factor that $Q$ can produce on any grid function. If $\|Q\|_h \leq 1$, the operator is **non-amplifying** (stable).

### 2.3 Norms of Shift Operators

The shift operator $E^p$ is fundamental to all finite difference operators. Let's compute its norm.

**Step 1:** Write out the norm of the shifted grid function.

$$
\|E^p u\|_h^2 = \sum_{j=0}^{N} |u_{j+p}|^2 h
$$

**Step 2:** Re-index the summation. Since we're on a periodic domain (or the indices wrap appropriately):

$$
\|E^p u\|_h^2 = \sum_{j=0}^{N} |u_{j}|^2 h = \|u\|_h^2
$$

**Conclusion:**
$$
\|E^p\|_h = 1
$$

**Critical observation:** Shift operators are **isometries** on the discrete inner product space—they preserve norms exactly. This is the discrete analog of how translation preserves $L^2$ norms in continuous spaces.

### 2.4 Upper Bounds on Difference Operator Norms

Now let's bound the norms of our finite difference operators.

**Forward Difference Operator:** $D_+ = \frac{1}{h}(E - E^0) = \frac{1}{h}(E - I)$

**Step 1:** Apply the triangle inequality to bound $\|D_+ u\|_h$.

$$
\|D_+ u\|_h = \frac{1}{h} \|(E - I) u\|_h
$$

**Step 2:** Use triangle inequality:

$$
\|D_+ u\|_h \leq \frac{1}{h} \left( \|E u\|_h + \|I u\|_h \right)
$$

**Step 3:** Since $\|E\|_h = 1$ and $\|I\|_h = 1$:

$$
\|D_+ u\|_h \leq \frac{2}{h} \|u\|_h
$$

**Conclusion:**
$$
\|D_+\|_h \leq \frac{2}{h}
$$

**Backward Difference Operator:** Recall that $D_- = E^{-1} D_+$. Then:

$$
\|D_- u\|_h = \|E^{-1} D_+ u\|_h \leq \|E^{-1}\|_h \|D_+ u\|_h \leq \frac{2}{h} \|u\|_h
$$

Thus:
$$
\|D_-\|_h \leq \frac{2}{h}
$$

**Important Notes:**
- These are **upper bounds**, not necessarily the exact norms
- The factor $2/h$ shows that difference operators amplify by at most a factor proportional to $1/h$
- As the grid is refined ($h \to 0$), the operator norm grows—this is expected since we're approximating derivatives

### 2.5 Lower Bounds and Exact Norms

We can also derive **lower bounds** by choosing specific test functions. Remember the definition of operator norm:

$$
\|Q\|_h = \sup_{u \neq 0} \frac{\|Q u\|_h}{\|u\|_h}
$$

This means for **any** particular non-zero grid function $v$:

$$
\|Q\|_h \geq \frac{\|Q v\|_h}{\|v\|_h}
$$

**Test function for forward difference:** Choose the oscillatory grid function $v_j = (-1)^j$.

**Step 1:** Compute $\|v\|_h^2$.

$$
\|v\|_h^2 = \sum_{j=0}^{N} |(-1)^j|^2 h = \sum_{j=0}^{N} 1 \cdot h = h(N+1)
$$

**Step 2:** Compute $(D_+ v)_j$.

$$
(D_+ v)_j = \frac{1}{h}(v_{j+1} - v_j) = \frac{1}{h}\left((-1)^{j+1} - (-1)^j\right)
$$

Factor out $(-1)^j$:

$$
(D_+ v)_j = \frac{(-1)^j}{h}\left(-1 - 1\right) = \frac{-2(-1)^j}{h}
$$

**Step 3:** Compute $\|D_+ v\|_h^2$.

$$
\|D_+ v\|_h^2 = \sum_{j=0}^{N} \left|\frac{-2(-1)^j}{h}\right|^2 h = \frac{4}{h^2} \sum_{j=0}^{N} |(-1)^j|^2 h = \frac{4}{h^2} \cdot h(N+1) = \frac{4(N+1)}{h}
$$

**Step 4:** Compute the lower bound.

$$
\|D_+\|_h^2 \geq \frac{\|D_+ v\|_h^2}{\|v\|_h^2} = \frac{4(N+1)/h}{h(N+1)} = \frac{4}{h^2}
$$

Therefore:
$$
\|D_+\|_h \geq \frac{2}{h}
$$

**Conclusion:** Combined with our upper bound:

$$
\frac{2}{h} \leq \|D_+\|_h \leq \frac{2}{h}
$$

Thus:
$$
\boxed{\|D_+\|_h = \frac{2}{h}}
$$

Similarly, we can show:
$$
\boxed{\|D_-\|_h = \frac{2}{h}}
$$

**Centered Difference Operator:** Using a similar analysis with test function $v_j = i^j$ (where $i = \sqrt{-1}$), one can show:

$$
\boxed{\|D_0\|_h = \frac{1}{h}}
$$

**Critical observation:** The centered difference operator has **half the norm** of the one-sided operators! This reflects its higher accuracy and better stability properties.

---

## 3. The Discrete Fourier Transform

We now have the analysis tools to discuss **accuracy** rigorously. The key is to understand how finite difference operators act on different frequency components. The discrete Fourier transform provides the bridge between grid functions and their frequency representations.

### 3.1 Trigonometric Interpolation

Just as continuous functions on $[0, 2\pi]$ can be represented by Fourier series, grid functions can be uniquely represented by trigonometric polynomials.

**Theorem (Trigonometric Interpolation):** There exists a unique trigonometric polynomial

$$
\text{Int}_N(u) = \frac{1}{\sqrt{2\pi}} \sum_{\omega = -N/2}^{N/2} \tilde{u}(\omega) e^{i\omega x}
$$

that interpolates the grid function $u$, meaning:

$$
u_j = \text{Int}_N(u)(x_j) \quad \text{for } j = 0, 1, \ldots, N
$$

**Pedagogical note:** This theorem is remarkable because it says we can **exactly** represent any grid function (which has only $N+1$ values) using only $N+1$ Fourier modes. There's no approximation error—the trigonometric polynomial passes through every grid point exactly. This is fundamentally different from truncating a Fourier series for a continuous function.

### 3.2 Discrete Orthogonality

The key to proving the interpolation theorem is a discrete orthogonality result.

**Lemma (Discrete Orthogonality):** The functions $e^{i\nu x}$ for $\nu = 0, \pm 1, \ldots, \pm N/2$ are orthogonal with respect to the discrete inner product:

$$
\left(e^{i\nu x}, e^{i\mu x}\right)_h = \begin{cases}
2\pi & \text{if } \nu = \mu \\
0 & \text{if } \nu \neq \mu
\end{cases}
$$

**Proof (case $\nu = \mu$):**

**Step 1:** Write out the discrete inner product.

$$
\left(e^{i\nu x}, e^{i\nu x}\right)_h = \sum_{j=0}^{N} e^{-i\nu x_j} e^{i\nu x_j} h = \sum_{j=0}^{N} e^{0} h
$$

**Step 2:** Simplify using $e^0 = 1$ and recall $h = \frac{2\pi}{N+1}$.

$$
\left(e^{i\nu x}, e^{i\nu x}\right)_h = \sum_{j=0}^{N} h = (N+1)h = (N+1) \cdot \frac{2\pi}{N+1} = 2\pi
$$

The case $\nu \neq \mu$ requires summing a geometric series with complex exponentials—the details involve showing the sum vanishes due to periodicity. (See Gustafsson §1.3 for the complete proof.)

### 3.3 Fourier Coefficients

With orthogonality established, we can derive the formula for Fourier coefficients.

**Theorem (Discrete Fourier Coefficients):** The interpolation problem has the unique solution:

$$
\tilde{u}(\omega) = \frac{1}{\sqrt{2\pi}} \left(e^{i\omega x}, u\right)_h = \frac{1}{\sqrt{2\pi}} \sum_{j=0}^{N} e^{-i\omega x_j} u_j h, \quad |\omega| \leq N/2
$$

**Proof:**

**Step 1:** Assume the interpolation formula holds:

$$
u_j = \frac{1}{\sqrt{2\pi}} \sum_{\omega = -N/2}^{N/2} \tilde{u}(\omega) e^{i\omega x_j}
$$

**Step 2:** Multiply both sides by $e^{-i\nu x_j} h$ and sum over $j$:

$$
\sum_{j=0}^{N} e^{-i\nu x_j} u_j h = \frac{1}{\sqrt{2\pi}} \sum_{\omega = -N/2}^{N/2} \tilde{u}(\omega) \sum_{j=0}^{N} e^{-i\nu x_j} e^{i\omega x_j} h
$$

**Step 3:** Use the discrete inner product notation:

$$
\left(e^{i\nu x}, u\right)_h = \frac{1}{\sqrt{2\pi}} \sum_{\omega = -N/2}^{N/2} \tilde{u}(\omega) \left(e^{i\nu x}, e^{i\omega x}\right)_h
$$

**Step 4:** Apply discrete orthogonality. The inner product is zero unless $\omega = \nu$:

$$
\left(e^{i\nu x}, u\right)_h = \frac{1}{\sqrt{2\pi}} \sum_{\omega = -N/2}^{N/2} \tilde{u}(\omega) \delta_{\nu\omega} \cdot 2\pi = \sqrt{2\pi} \, \tilde{u}(\nu)
$$

where $\delta_{\nu\omega}$ is the Kronecker delta.

**Step 5:** Solve for $\tilde{u}(\nu)$:

$$
\tilde{u}(\nu) = \frac{1}{\sqrt{2\pi}} \left(e^{i\nu x}, u\right)_h
$$

This establishes both existence and uniqueness of the trigonometric interpolant. $\square$

### 3.4 Discrete Parseval Theorem

The discrete Parseval theorem connects norms in physical space to norms in frequency space.

**Theorem (Discrete Parseval):** Given grid functions $u_1$ and $u_2$ with interpolants $\text{Int}_N u_1$ and $\text{Int}_N u_2$:

$$
(u_1, u_2)_h = \sum_{\omega = -N/2}^{N/2} \overline{\tilde{u}_1(\omega)} \, \tilde{u}_2(\omega)
$$

In particular:
$$
\|u\|_h^2 = \sum_{\omega = -N/2}^{N/2} |\tilde{u}(\omega)|^2
$$

**Critical observation:** This theorem says that the **total energy in physical space equals the sum of energies in each frequency mode**. This is the discrete analog of the continuous Parseval identity, and it's fundamental to stability analysis via Fourier methods.

### 3.5 Connecting Continuous and Discrete Derivatives

We can relate derivatives of the interpolant to finite difference operators on the grid function.

**Theorem (Derivative Bounds):** Let $\text{Int}_N u$ be the trigonometric interpolant of grid function $u$. Then:

$$
\left\|\text{Int}_N u\right\|^2 = \|u\|_h^2
$$

This shows the interpolant has the same $L^2$ norm as the grid function—they represent the same "energy."

Furthermore, we have the inequalities:

$$
\|D_+ u\|_h^2 \leq \left\|\frac{d}{dx} \text{Int}_N u\right\|^2 \leq \frac{\pi^2}{4} \|D_+ u\|_h^2
$$

**Interpretation:** The finite difference operator $D_+$ approximates the continuous derivative, but not perfectly. The right inequality shows that the true derivative of the interpolant can be up to $\pi/2 \approx 1.57$ times larger than the finite difference estimate. This quantifies the **discretization error** inherent in using $D_+$ instead of $\frac{d}{dx}$.

**For those interested:** Gustafsson §1.3-1.4 contains more detailed analysis, but the proofs require extensive trigonometric and complex analysis manipulations. The key takeaway is that discrete and continuous derivatives are related through these bounded inequalities, providing rigorous justification for our finite difference approximations.

---

## 4. The Lax Equivalence Theorem

We're now ready to state and interpret one of the most fundamental theorems in numerical analysis of PDEs.

### 4.1 Abstract Finite Difference Schemes

Consider a general finite difference scheme for a time-dependent PDE:

$$
\begin{aligned}
v_j^{n+1} &= Q v_j^n \\
v_j^0 &= f_j
\end{aligned}
$$

where $Q$ is the **amplification operator**:

$$
Q = \sum_{\nu} A_{\nu}(k, h) E^{\nu}
$$

Here:
- $k$ is the time step
- $h$ is the spatial grid spacing
- $A_{\nu}(k, h)$ are coefficients that may depend on $k$ and $h$
- $E^{\nu}$ are shift operators

**Example:** For the forward-time, forward-space scheme for the transport equation:

$$
v_j^{n+1} = v_j^n - \frac{k}{h}(v_{j+1}^n - v_j^n)
$$

we have:

$$
Q = I - \frac{k}{h}(E - I) = \left(1 + \frac{k}{h}\right)I - \frac{k}{h}E
$$

### 4.2 The Lax Equivalence Theorem

**Theorem (Lax Equivalence Theorem):** Consider a finite difference scheme for a linear PDE. Suppose:

**(1) Finite initial data:**
$$
\|f\|_h^2 < \infty
$$

**(2) Stability:** There exists a constant $K$ independent of $h$ and $k$ such that for all time steps $n$ up to final time $T$:

$$
\sup_{0 \leq t_n \leq T} \left|\hat{Q}^n(\xi)\right| \leq K
$$

where $\hat{Q}(\xi)$ is the Fourier symbol of the amplification operator (how $Q$ acts on mode $e^{i\xi x}$).

**(3) Consistency:** For any wavenumber $\omega$, as $k, h \to 0$ with $t_n = nk$ fixed:

$$
\lim_{k, h \to 0} \sup_{0 \leq t_n < T} \left|\hat{Q}^n(\xi) - e^{i\omega t_n}\right| = 0
$$

**Then:** The finite difference scheme **converges** to the true solution:

$$
\lim_{k, h \to 0} \sup_{0 \leq t_n \leq T} \left\|u(\cdot, t_n) - \text{Int}_N(v_j^n)\right\| = 0
$$

### 4.3 Interpretation and Implications

The Lax theorem provides a **complete characterization of convergence** through three simple conditions:

**Finite initial data:** This is a technical requirement ensuring we start with well-posed data.

**Stability:** This is the critical condition. It says the amplification operator $Q$ cannot grow unboundedly—its Fourier symbol must remain bounded as we iterate the scheme. In practice, we often check whether $|\hat{Q}(\xi)| \leq 1$ for all $\xi$, which ensures:
$$
|Q^n(\xi)| = |\hat{Q}(\xi)|^n \leq 1
$$

This prevents exponential growth of errors.

**Consistency:** This ensures the finite difference scheme is approximating the *correct* PDE. The Fourier symbol $\hat{Q}(\xi)$ should approach the continuous evolution operator $e^{i\omega t}$ as the grid is refined. If this fails, the scheme might converge, but to the wrong equation!

**Key Takeaway:** **Stability + Consistency = Convergence**

This is the fundamental mantra of numerical PDE analysis. To ensure a finite difference scheme works:
1. Check that it's consistent (approximates the right PDE)
2. Check that it's stable (doesn't amplify errors)
3. Then convergence is guaranteed

**Implications for code development:** When implementing numerical methods:
- **Debug first:** Use analytic solutions to verify consistency (order of accuracy tests)
- **Test stability:** Check von Neumann stability conditions or compute $\|\hat{Q}\|$ numerically
- **Trust convergence:** If both hold, the scheme will converge as grid is refined

**Implications for machine learning:** When we later learn discretizations from data, we'll want to:
- **Enforce stability:** Build architectures where learned operators satisfy $\|\hat{Q}\| \leq 1$
- **Guide consistency:** Use physics knowledge to constrain how operators approximate derivatives
- **Guarantee convergence:** By respecting these principles, learned models will generalize reliably

---

## 5. Summary

This lecture developed the rigorous mathematical framework for analyzing finite difference schemes:

1. **Discrete norms and inner products** provide a way to measure grid functions that connects smoothly to continuous $L^2$ norms as $h \to 0$

2. **Operator norms** quantify how much finite difference operators can amplify perturbations:
   - Shift operators are isometries: $\|E^p\|_h = 1$
   - Forward/backward differences: $\|D_+\|_h = \|D_-\|_h = 2/h$
   - Centered difference: $\|D_0\|_h = 1/h$ (better stability)

3. **Discrete Fourier transform** provides exact representation of grid functions as trigonometric polynomials with $N+1$ modes for $N+1$ grid points

4. **Discrete Parseval theorem** connects physical space norms to frequency space: $\|u\|_h^2 = \sum_{\omega} |\tilde{u}(\omega)|^2$

5. **Derivative bounds** quantify the relationship between continuous derivatives and finite difference approximations

6. **Lax Equivalence Theorem** gives the fundamental convergence criterion: **Stability + Consistency = Convergence**

**Key Takeaway:** The Lax theorem provides the blueprint for both analyzing existing numerical methods and designing new ones (including learned discretizations). By ensuring stability (bounded amplification) and consistency (correct limiting behavior), we guarantee convergence to the true solution. This framework will be essential as we move toward learning PDEs from data, where we'll need to enforce these mathematical principles as architectural constraints in our neural network models.
