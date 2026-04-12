# Log-Likelihoods for Implicit SDE Discretizations

## Setup

Consider the Itô SDE

$$dX = f(X)\,dt + g(X)\,dW_t$$

discretized with step size $k$. We seek the transition density $p(x^{n+1}\mid x^n)$ for maximum likelihood estimation.

---

## 1. Explicit Euler–Maruyama (Baseline)

$$x^{n+1} = x^n + k\,f(x^n) + g(x^n)\,\Delta W_n, \qquad \Delta W_n \sim \mathcal{N}(0,kI)$$

The noise maps linearly to the next state, so the transition density is Gaussian:

$$x^{n+1}\mid x^n \sim \mathcal{N}\!\bigl(x^n + k\,f(x^n),\; k\,g(x^n)g(x^n)^\top\bigr)$$

**Log-likelihood:**

$$\ell = \sum_n \left[-\frac{d}{2}\log(2\pi k) - \frac{|\xi_n|^2}{2k} - \log|\det g(x^n)|\right]$$

where $\xi_n = g(x^n)^{-1}(x^{n+1} - x^n - k\,f(x^n))$.

---

## 2. Fully Implicit Scheme

$$x^{n+1} = x^n + k\,f(x^{n+1}) + g(x^{n+1/2})\,\Delta W_n$$

The map $\Delta W \mapsto x^{n+1}$ is now **implicit**. By the implicit function theorem:

$$\frac{\partial x^{n+1}}{\partial(\Delta W)} = \left(I - k\,\nabla_x f(x^{n+1}) - \tfrac{1}{2}\nabla_x g(x^{n+1/2})\,\Delta W_n\right)^{-1} g(x^{n+1/2})$$

Applying the change-of-variables formula with residual $\xi_n = g(x^{n+1/2})^{-1}(x^{n+1} - x^n - k\,f(x^{n+1}))$:

**Log-likelihood:**

$$\ell = \sum_n \left[-\frac{d}{2}\log(2\pi k) - \frac{|\xi_n|^2}{2k} + \log\left|\det\!\left(I - k\,\nabla_x f(x^{n+1}) - \tfrac{1}{2}\nabla_x g(x^{n+1/2})\,\Delta W_n\right)\right| - \log|\det g(x^{n+1/2})|\right]$$

> **Key difficulty:** The Jacobian correction depends on the noise realization $\Delta W_n$.

---

## 3. Semi-Implicit Scheme (Drift-Implicit, Diffusion-Explicit)

$$x^{n+1} = x^n + k\,f(x^{n+1}) + g(x^n)\,\Delta W_n$$

The diffusion is evaluated at the **known** state $x^n$, and only the drift is implicit.

**Jacobian** (via implicit function theorem):

$$\frac{\partial(\Delta W)}{\partial x^{n+1}} = g(x^n)^{-1}\bigl(I - k\,\nabla_x f(x^{n+1})\bigr)$$

With residual $\xi_n = g(x^n)^{-1}(x^{n+1} - x^n - k\,f(x^{n+1}))$:

**Log-likelihood:**

$$\boxed{\ell = \sum_n \left[-\frac{d}{2}\log(2\pi k) - \frac{|\xi_n|^2}{2k} + \log\left|\det\!\left(I - k\,\nabla_x f(x^{n+1})\right)\right| - \log|\det g(x^n)|\right]}$$

> **Key simplification:** The Jacobian correction $\det(I - k\,\nabla_x f)$ is deterministic given the state pair $(x^n, x^{n+1})$ — it does not depend on $\Delta W_n$.

---

## Comparison

| Property | Explicit E-M | Fully Implicit | Semi-Implicit |
|---|---|---|---|
| Transition density | Gaussian | Gaussian + Jacobian | Gaussian + Jacobian |
| Jacobian correction | None | $\det(I - k\nabla f - \tfrac{1}{2}\nabla g\,\Delta W)$ | $\det(I - k\,\nabla f)$ |
| Depends on $\Delta W$? | — | Yes | **No** |
| $g$ evaluated at | $x^n$ (known) | $x^{n+1/2}$ (implicit) | $x^n$ (known) |
| Implicit solve needed | No | Drift + diffusion | Drift only |

---

## Special Case: Additive Noise ($g$ = const)

When $g$ is constant, $\log|\det g|$ and $\nabla_x g$ drop out. For the semi-implicit scheme:

$$\ell = \sum_n \left[-\frac{|\xi_n|^2}{2k} + \log\left|\det\!\left(I - k\,\nabla_x f(x^{n+1})\right)\right|\right] + \text{const.}$$

This is equivalent to a **residual normalizing flow**: each time step is an invertible residual-network layer with an explicit log-det correction — directly connecting implicit SDE integrators to generative modeling (cf. FFJORD, residual flows).
