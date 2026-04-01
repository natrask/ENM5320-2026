# Section 7: SUPG Stabilization for Advection-Dominated Problems

## 7.1 Motivation: Why Galerkin Fails for Advection

Consider the steady advection-diffusion equation:

$$-\epsilon \nabla^2 u + \mathbf{a} \cdot \nabla u = f \quad \text{in } \Omega$$

where $\epsilon > 0$ is the diffusion coefficient and $\mathbf{a}$ is a given velocity field. When $\epsilon$ is small relative to $|\mathbf{a}|h$ (the mesh Péclet number $\text{Pe}_h = \frac{|\mathbf{a}|h}{2\epsilon} \gg 1$), the standard Galerkin method produces **spurious oscillations**. These oscillations are not physical — they arise because the Galerkin method imposes symmetry on an inherently asymmetric (non-self-adjoint) operator.

The core issue: the advection operator $\mathbf{a} \cdot \nabla$ is skew-symmetric in $L^2$. The standard Galerkin weak form inherits no coercivity from this term, so the discrete solution is under-stabilized in the streamline direction.

## 7.2 The Standard Galerkin Weak Form

The standard weak form reads: find $u_h \in V_h$ such that

$$\epsilon(\nabla u_h, \nabla w_h) + (\mathbf{a} \cdot \nabla u_h, w_h) = (f, w_h) \quad \forall\, w_h \in V_h$$

Since $(\mathbf{a} \cdot \nabla w_h, w_h) = 0$ for divergence-free $\mathbf{a}$ with homogeneous boundary conditions, the advection term contributes nothing to the coercivity of the bilinear form. The only control on $u_h$ comes from the diffusion term $\epsilon \|\nabla u_h\|^2$, which vanishes as $\epsilon \to 0$.

## 7.3 The SUPG Formulation

**Streamline Upwind Petrov-Galerkin (SUPG)** modifies the test function by adding a streamline derivative term. Instead of testing with $w_h$, we test with:

$$w_h + \tau_K \, \mathbf{a} \cdot \nabla w_h$$

where $\tau_K$ is an element-wise stabilization parameter. The SUPG weak form becomes:

$$\epsilon(\nabla u_h, \nabla w_h) + (\mathbf{a} \cdot \nabla u_h, w_h) + \sum_{K} \tau_K (\mathbf{a} \cdot \nabla u_h - \epsilon \nabla^2 u_h - f, \, \mathbf{a} \cdot \nabla w_h)_K = (f, w_h)$$

**Key property — Consistency:** The added term is weighted by the **residual** of the PDE. If $u_h$ were the exact solution, the residual would vanish and the extra term would contribute nothing. This means SUPG is a **consistent** stabilization — it does not degrade the order of accuracy.

Rearranging, the SUPG bilinear form is:

$$B_{\text{SUPG}}(u_h, w_h) = \underbrace{\epsilon(\nabla u_h, \nabla w_h) + (\mathbf{a} \cdot \nabla u_h, w_h)}_{\text{Galerkin}} + \underbrace{\sum_K \tau_K (\mathbf{a} \cdot \nabla u_h, \, \mathbf{a} \cdot \nabla w_h)_K}_{\text{streamline diffusion}}$$

with the **modified right-hand side** (load functional):

$$L_{\text{SUPG}}(w_h) = (f, w_h) + \sum_K \tau_K (f, \, \mathbf{a} \cdot \nabla w_h)_K$$

so the full SUPG problem reads: find $u_h \in V_h$ such that $B_{\text{SUPG}}(u_h, w_h) = L_{\text{SUPG}}(w_h)$ for all $w_h \in V_h$.

Here, for linear elements ($\nabla^2 u_h = 0$ element-wise) the Laplacian term drops out. The $(f, \mathbf{a} \cdot \nabla w_h)_K$ term on the right-hand side comes from moving the $-f$ portion of the residual to the load side. The added streamline diffusion term in $B_{\text{SUPG}}$ provides control over $\mathbf{a} \cdot \nabla u_h$ — exactly what Galerkin was missing.

### Coercivity of the SUPG bilinear form

Recall from §7.2 that the standard Galerkin bilinear form fails coercivity because the advection term vanishes when the test and trial functions coincide. Let us check what happens with SUPG. Set $w_h = u_h$:

$$B_{\text{SUPG}}(u_h, u_h) = \epsilon\|\nabla u_h\|^2 + (\mathbf{a} \cdot \nabla u_h, u_h) + \sum_K \tau_K \|\mathbf{a} \cdot \nabla u_h\|_K^2$$

For divergence-free $\mathbf{a}$ with homogeneous boundary conditions, integration by parts gives $(\mathbf{a} \cdot \nabla u_h, u_h) = 0$, so:

$$B_{\text{SUPG}}(u_h, u_h) = \epsilon\|\nabla u_h\|^2 + \sum_K \tau_K \|\mathbf{a} \cdot \nabla u_h\|_K^2$$

Both terms are non-negative, and crucially the **second term does not vanish**. Define the SUPG norm:

$$\|v\|_{\text{SUPG}}^2 = \epsilon\|\nabla v\|^2 + \sum_K \tau_K \|\mathbf{a} \cdot \nabla v\|_K^2$$

Then we have coercivity in this norm:

$$B_{\text{SUPG}}(v_h, v_h) = \|v_h\|_{\text{SUPG}}^2 \geq \|v_h\|_{\text{SUPG}}^2$$

This is the key fix: SUPG restores coercivity by introducing a norm that includes control over the streamline derivative $\mathbf{a} \cdot \nabla u_h$. Standard Galerkin is coercive only in the $H^1$ seminorm with constant $\epsilon$, which degenerates as $\epsilon \to 0$. SUPG is coercive in $\|\cdot\|_{\text{SUPG}}$ with constant $1$, independent of $\epsilon$.

## 7.4 Choosing the Stabilization Parameter

The stabilization parameter $\tau_K$ on each element $K$ with diameter $h_K$ is typically chosen as:

$$\tau_K = \frac{h_K}{2|\mathbf{a}|} \xi(\text{Pe}_K)$$

where the element Péclet number is:

$$\text{Pe}_K = \frac{|\mathbf{a}| h_K}{2\epsilon}$$

and $\xi$ is usually taken as:

$$\xi(\text{Pe}) = \coth(\text{Pe}) - \frac{1}{\text{Pe}}$$

This function smoothly transitions between:
- **Diffusion-dominated regime** ($\text{Pe}_K \ll 1$): $\xi \approx \frac{\text{Pe}}{3} \to 0$, so SUPG adds negligible modification.
- **Advection-dominated regime** ($\text{Pe}_K \gg 1$): $\xi \to 1$, giving $\tau_K \approx \frac{h_K}{2|\mathbf{a}|}$, which corresponds to optimal upwinding.

A simpler alternative is the doubly asymptotic approximation:

$$\tau_K = \frac{h_K}{2|\mathbf{a}|} \min\left(1, \frac{\text{Pe}_K}{3}\right)$$

## 7.5 One-Dimensional Worked Example

Consider the 1D boundary value problem:

$$-\epsilon u'' + a\, u' = 0 \quad \text{on } [0, 1], \qquad u(0) = 0, \quad u(1) = 1$$

with $a > 0$ constant. The exact solution is:

$$u(x) = \frac{e^{ax/\epsilon} - 1}{e^{a/\epsilon} - 1}$$

For $\epsilon \ll a$, this has a sharp **boundary layer** of width $O(\epsilon/a)$ near $x = 1$.

### Standard Galerkin on a uniform mesh

Using linear elements on a uniform mesh with spacing $h$, the Galerkin solution at node $j$ satisfies:

$$\frac{\epsilon}{h}(-u_{j-1} + 2u_j - u_{j+1}) + \frac{a}{2}(-u_{j-1} + u_{j+1}) = 0$$

Rearranging:

$$\left(\frac{\epsilon}{h} + \frac{a}{2}\right) u_{j+1} - \frac{2\epsilon}{h} u_j + \left(\frac{\epsilon}{h} - \frac{a}{2}\right) u_{j-1} = 0$$

When $\text{Pe}_h = \frac{ah}{2\epsilon} > 1$, the coefficient of $u_{j-1}$ changes sign, destroying the M-matrix property and producing **node-to-node oscillations**.

### SUPG on the same mesh

With SUPG using $\tau = \frac{h}{2a}\xi(\text{Pe}_h)$, the modified stencil becomes:

$$\frac{\epsilon + \bar{\epsilon}}{h}(-u_{j-1} + 2u_j - u_{j+1}) + \frac{a}{2}(-u_{j-1} + u_{j+1}) = 0$$

where $\bar{\epsilon} = \tau a^2 = \frac{ah}{2}\xi(\text{Pe}_h)$ is the **artificial streamline diffusion**. With the optimal choice $\xi = \coth(\text{Pe}_h) - 1/\text{Pe}_h$, the SUPG solution is **nodally exact** — it matches the exact solution at every node, regardless of mesh size.

This is a remarkable property: SUPG with the optimal $\tau$ recovers the exact nodal values in 1D. In higher dimensions, SUPG does not achieve nodal exactness, but it eliminates spurious oscillations and maintains optimal convergence rates.

### Comparison table (1D, $a = 1$, $\epsilon = 0.01$, 10 elements)

| Method | Oscillations? | Max overshoot | Convergence order |
|--------|:---:|:---:|:---:|
| Standard Galerkin | Yes | ~30% | Degraded |
| SUPG (optimal $\tau$) | No | 0% | Nodally exact in 1D |

## 7.6 Connection to Conservation Laws

SUPG connects to the broader framework of this lecture in several ways:

1. **Residual-based stabilization preserves conservation.** Because the SUPG term is proportional to the PDE residual, it vanishes for the exact solution and does not introduce artificial sources or sinks. Local conservation is maintained in the sense that the total flux balance on each element is consistent with the original PDE.

2. **Structure of the test space.** SUPG can be viewed as a Petrov-Galerkin method where the trial space $V_h$ uses standard finite element functions, but the test space $W_h$ is enriched with streamline derivatives. This is related to the idea of choosing test and trial spaces carefully to satisfy inf-sup conditions — exactly the design philosophy behind the de Rham compatible spaces discussed earlier.

3. **Nonlinear extension.** For nonlinear conservation laws $\nabla \cdot F(u) = f$, SUPG generalizes naturally: the stabilization term uses the linearized advection velocity $\mathbf{a} = F'(u)$, and the monotonicity conditions from Section 6 ensure well-posedness of the stabilized formulation.

---

## 7.7 Connection to Finite Difference Stabilizations: Lax-Friedrichs and Lax-Wendroff

The SUPG framework provides a unifying lens through which classical finite difference stabilization schemes can be understood. The connections are not merely analogies — they are **algebraic equivalences** at the stencil level.

### Starting point: the SUPG stencil in 1D

Recall from Section 7.5 that the SUPG discretization of $-\epsilon u'' + au' = 0$ on a uniform mesh of spacing $h$ with linear elements produces the stencil:

$$\frac{\epsilon + \bar{\epsilon}}{h}(-u_{j-1} + 2u_j - u_{j+1}) + \frac{a}{2}(-u_{j-1} + u_{j+1}) = 0$$

where $\bar{\epsilon} = \tau a^2$ is the artificial streamline diffusion. The full coefficient form is:

$$\left(-\frac{\epsilon + \bar{\epsilon}}{h} - \frac{a}{2}\right) u_{j-1} + \frac{2(\epsilon + \bar{\epsilon})}{h} u_j + \left(-\frac{\epsilon + \bar{\epsilon}}{h} + \frac{a}{2}\right) u_{j+1} = 0$$

### 7.7.1 Equivalence with First-Order Upwind (Lax-Friedrichs at steady state)

**Claim:** Setting $\tau = h/(2a)$ (i.e., $\xi = 1$) in the SUPG stencil recovers the first-order upwind difference scheme.

**Derivation.** With $\tau = h/(2a)$, the artificial diffusion is:

$$\bar{\epsilon} = \tau a^2 = \frac{h}{2a} \cdot a^2 = \frac{ah}{2}$$

Substitute into the SUPG stencil. For the pure advection case ($\epsilon = 0$):

$$\frac{ah/2}{h}(-u_{j-1} + 2u_j - u_{j+1}) + \frac{a}{2}(-u_{j-1} + u_{j+1}) = 0$$

$$\frac{a}{2}(-u_{j-1} + 2u_j - u_{j+1}) + \frac{a}{2}(-u_{j-1} + u_{j+1}) = 0$$

$$\frac{a}{2}\big[(-u_{j-1} + 2u_j - u_{j+1}) + (-u_{j-1} + u_{j+1})\big] = 0$$

$$\frac{a}{2}\big[-2u_{j-1} + 2u_j\big] = 0$$

$$a \cdot \frac{u_j - u_{j-1}}{h} \cdot h = 0$$

This is exactly the **first-order upwind** (backward difference) approximation:

$$a \cdot \frac{u_j - u_{j-1}}{h} = 0 \qquad \checkmark$$

Now compare with **Lax-Friedrichs** for the time-dependent problem $u_t + au_x = 0$:

$$\frac{u_j^{n+1} - \frac{1}{2}(u_{j+1}^n + u_{j-1}^n)}{\Delta t} + a\frac{u_{j+1}^n - u_{j-1}^n}{2h} = 0$$

Rewrite the averaging as a centered value plus a diffusion term:

$$\frac{u_j^{n+1} - u_j^n}{\Delta t} + \underbrace{\frac{u_j^n - \frac{1}{2}(u_{j+1}^n + u_{j-1}^n)}{\Delta t}}_{\text{artificial viscosity}} + a\frac{u_{j+1}^n - u_{j-1}^n}{2h} = 0$$

The artificial viscosity term is:

$$\frac{u_j^n - \frac{1}{2}(u_{j+1}^n + u_{j-1}^n)}{\Delta t} = -\frac{h^2}{2\Delta t} \cdot \frac{u_{j+1}^n - 2u_j^n + u_{j-1}^n}{h^2} = -\epsilon_{\text{LF}} u_{xx}$$

with $\epsilon_{\text{LF}} = \frac{h^2}{2\Delta t}$. At steady state ($u_j^{n+1} = u_j^n$), the Lax-Friedrichs scheme reduces to:

$$-\frac{h^2}{2\Delta t} \cdot \frac{u_{j-1} - 2u_j + u_{j+1}}{h^2} + a \frac{u_{j+1} - u_{j-1}}{2h} = 0$$

$$\frac{\epsilon_{\text{LF}}}{h}(-u_{j-1} + 2u_j - u_{j+1}) + \frac{a}{2}(-u_{j-1} + u_{j+1}) = 0$$

This is **identical** to the SUPG stencil with $\bar{\epsilon} = \epsilon_{\text{LF}} = h^2/(2\Delta t)$. When the CFL condition $\nu = a\Delta t/h = 1$ is satisfied, $\epsilon_{\text{LF}} = ah/2 = \bar{\epsilon}|_{\xi=1}$, recovering exactly first-order upwind.

**Key insight:** Lax-Friedrichs stabilizes by adding $O(h)$ artificial viscosity — the same mechanism as SUPG with $\xi = 1$. Both are first-order accurate and overly diffusive.

### 7.7.2 Equivalence with Lax-Wendroff via Taylor-Galerkin

**Claim:** SUPG applied to the time-dependent problem $u_t + au_x = 0$ with $\tau = \Delta t/2$ and forward Euler time stepping produces precisely the Lax-Wendroff scheme.

**Derivation — Lax-Wendroff side.** Start from the Taylor expansion:

$$u(x, t + \Delta t) = u + \Delta t\, u_t + \frac{\Delta t^2}{2} u_{tt} + O(\Delta t^3)$$

Use the PDE to replace time derivatives with spatial derivatives:

$$u_t = -au_x, \qquad u_{tt} = -au_{xt} = -a(-au_{xx}) = a^2 u_{xx}$$

Substituting:

$$u^{n+1} = u^n - a\Delta t\, u_x^n + \frac{a^2 \Delta t^2}{2} u_{xx}^n$$

Discretize $u_x$ with centered differences and $u_{xx}$ with the standard three-point stencil:

$$u_j^{n+1} = u_j^n - \frac{a\Delta t}{2h}(u_{j+1}^n - u_{j-1}^n) + \frac{a^2\Delta t^2}{2h^2}(u_{j+1}^n - 2u_j^n + u_{j-1}^n)$$

Rearranging:

$$\frac{u_j^{n+1} - u_j^n}{\Delta t} + a\frac{u_{j+1}^n - u_{j-1}^n}{2h} - \frac{a^2\Delta t}{2} \cdot \frac{u_{j+1}^n - 2u_j^n + u_{j-1}^n}{h^2} = 0$$

The last term is artificial diffusion with coefficient $\epsilon_{\text{LW}} = \frac{a^2\Delta t}{2}$.

**Derivation — SUPG side.** The semi-discrete SUPG weak form for $u_t + au_x = 0$ is: find $u_h(t) \in V_h$ such that

$$(u_{h,t}, w_h) + (au_{h,x}, w_h) + \sum_K \tau_K(u_{h,t} + au_{h,x}, \, a w_{h,x})_K = 0 \quad \forall w_h \in V_h$$

With piecewise linear elements on a uniform mesh, choose a hat function $w_h = \phi_j$ centered at node $j$. The standard Galerkin mass and advection terms give:

$$(u_{h,t}, \phi_j) = \frac{h}{6}(\dot{u}_{j-1} + 4\dot{u}_j + \dot{u}_{j+1})$$

$$(au_{h,x}, \phi_j) = \frac{a}{2}(u_{j+1} - u_{j-1})$$

Now evaluate the SUPG stabilization term. On element $[x_{j-1}, x_j]$, the derivative of the hat function is $\phi_{j,x} = 1/h$, and on $[x_j, x_{j+1}]$ it is $\phi_{j,x} = -1/h$. The SUPG term contributes:

$$\tau \sum_K (u_{h,t} + au_{h,x}, \, a\phi_{j,x})_K$$

The $au_{h,x}$ part of the residual tested against $a\phi_{j,x}$ produces (after integration):

$$\tau a^2 \cdot \frac{-u_{j-1} + 2u_j - u_{j+1}}{h}$$

This is artificial diffusion with $\bar{\epsilon} = \tau a^2$, as seen in the steady case. The $u_{h,t}$ part modifies the mass matrix.

**Now apply forward Euler** in time and **mass lumping** (replace the consistent mass matrix with the diagonal lumped mass $M_L = hI$):

$$h \cdot \frac{u_j^{n+1} - u_j^n}{\Delta t} + \frac{a}{2}(u_{j+1}^n - u_{j-1}^n) + \tau a^2 \frac{-u_{j-1}^n + 2u_j^n - u_{j+1}^n}{h} = 0$$

Divide through by $h$:

$$\frac{u_j^{n+1} - u_j^n}{\Delta t} + a\frac{u_{j+1}^n - u_{j-1}^n}{2h} - \tau a^2 \frac{u_{j+1}^n - 2u_j^n + u_{j-1}^n}{h^2} = 0$$

**Set $\tau = \Delta t/2$:**

$$\frac{u_j^{n+1} - u_j^n}{\Delta t} + a\frac{u_{j+1}^n - u_{j-1}^n}{2h} - \frac{a^2\Delta t}{2} \cdot \frac{u_{j+1}^n - 2u_j^n + u_{j-1}^n}{h^2} = 0$$

This is **term-by-term identical** to the Lax-Wendroff scheme derived above. $\blacksquare$

### 7.7.3 Unified Picture

All three stabilizations correspond to specific choices of $\tau$ within the SUPG framework:

| $\tau$ | $\bar{\epsilon} = \tau a^2$ | Equivalent scheme | Order | Character |
|---|---|---|---|---|
| $0$ | $0$ | Centered differences (Galerkin) | 2nd, unstable | No stabilization |
| $\frac{h}{2a}$ | $\frac{ah}{2}$ | First-order upwind / Lax-Friedrichs (CFL=1) | 1st | Over-diffusive |
| $\frac{\Delta t}{2}$ | $\frac{a^2 \Delta t}{2}$ | **Lax-Wendroff** / Taylor-Galerkin | 2nd | Balanced (dispersive) |
| $\frac{h}{2a}\left(\coth\text{Pe} - \frac{1}{\text{Pe}}\right)$ | optimal | Scharfetter-Gummel | Nodally exact (1D) | Optimal |

The unifying principle: **stabilization = controlled artificial diffusion in the characteristic direction.** The parameter $\tau$ is a single dial that moves continuously between no stabilization (oscillatory) and full upwinding (diffusive). Lax-Friedrichs overshoots, Lax-Wendroff balances, and the optimal SUPG formula adapts to the local Péclet number.

This also reveals *why* Lax-Wendroff is second-order accurate while Lax-Friedrichs is only first: the Lax-Wendroff diffusion $\bar{\epsilon} = O(\Delta t)$ vanishes with refinement (since $\Delta t \sim h$ by the CFL condition, giving $\bar{\epsilon} \sim ah \cdot \nu$ where $\nu < 1$), while the Lax-Friedrichs diffusion $\bar{\epsilon} = ah/2$ contributes an $O(h)$ truncation error that does not improve beyond first order.

---

## Updates to Summary Box

Add the following item to the summary list:

> 8. **SUPG stabilization** for advection-dominated problems: consistent, residual-based stabilization that eliminates spurious oscillations while preserving conservation

Update the Key Takeaway to append:

> For advection-dominated problems where the mesh Péclet number is large, SUPG provides a principled stabilization strategy that adds artificial diffusion only in the streamline direction and only in proportion to the PDE residual, maintaining both accuracy and conservation.
