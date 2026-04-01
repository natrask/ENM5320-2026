# Exercises: Mathematical Fundamentals

**Related Material:** [Analysis.html](Analysis.html)

---

## Exercise 1: $\ell^2$ and $\ell^\infty$ Norm Equivalence

**Problem:** Prove that for any $\mathbf{v} \in \mathbb{R}^n$,
$$\|\mathbf{v}\|_\infty \leq \|\mathbf{v}\|_2 \leq \sqrt{n} \|\mathbf{v}\|_\infty$$

**Solution:**

For the left inequality, $\|\mathbf{v}\|_2^2 = \sum_{i=1}^n v_i^2 \geq \max_i |v_i|^2 = \|\mathbf{v}\|_\infty^2$. Taking square roots gives $\|\mathbf{v}\|_2 \geq \|\mathbf{v}\|_\infty$.

For the right inequality, $|v_i| \leq \|\mathbf{v}\|_\infty$ for all $i$ implies $\|\mathbf{v}\|_2^2 = \sum_{i=1}^n v_i^2 \leq n \|\mathbf{v}\|_\infty^2$. Taking square roots gives $\|\mathbf{v}\|_2 \leq \sqrt{n} \|\mathbf{v}\|_\infty$. $\square$

---

## Exercise 2: Derivative of $\|A\mathbf{x}\|^2$ in Einstein Notation

**Problem:** Let $A \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$. Compute $\frac{\partial}{\partial \mathbf{x}} \|A\mathbf{x}\|^2$ using Einstein notation.

**Solution:**

Write $\|A\mathbf{x}\|^2 = A_{ij} x_j A_{ik} x_k$. Differentiating with respect to $x_\ell$:
$$\frac{\partial}{\partial x_\ell} \left( A_{ij} x_j A_{ik} x_k \right) = A_{i\ell} A_{ik} x_k + A_{ij} x_j A_{i\ell} = 2 A_{i\ell} A_{ik} x_k = 2 (A^T A)_{\ell k} x_k$$

Therefore $\boxed{\frac{\partial}{\partial \mathbf{x}} \|A\mathbf{x}\|^2 = 2 A^T A \mathbf{x}}$. $\square$

---

## Exercise 3: Arithmetic Mean Bounded by RMS

**Problem:** Given $a_1, \ldots, a_n \in \mathbb{R}$, prove that $\left|\frac{1}{n}\sum_{i=1}^n a_i\right| \leq \sqrt{\frac{1}{n}\sum_{i=1}^n a_i^2}$.

**Solution:**

Applying Cauchy-Schwarz to $\mathbf{a} = [a_1, \ldots, a_n]^T$ and $\mathbf{1} = [1, \ldots, 1]^T$:
$$\left|\sum_{i=1}^n a_i\right| \leq \|\mathbf{a}\|_2 \|\mathbf{1}\|_2 = \sqrt{\sum_{i=1}^n a_i^2} \cdot \sqrt{n}$$

Dividing by $n$ gives the result. Equality holds when all $a_i$ are equal. $\square$

---

## Exercise 4: Young's Inequality (Peter-Paul Tradeoff)

**Problem:** Suppose $|g|^2 < \epsilon$ and $|h|^2 < C$ where $C$ is independent of $\epsilon$. Use Young's inequality $ab \leq \frac{a^2}{2\delta} + \frac{\delta b^2}{2}$ to bound $|gh|$ and find optimal $\delta > 0$.

**Solution:**

Applying Young's inequality with $a = |g|$ and $b = |h|$:
$$|gh| \leq \frac{|g|^2}{2\delta} + \frac{\delta |h|^2}{2} < \frac{\epsilon}{2\delta} + \frac{\delta C}{2}$$

Minimizing over $\delta$: set $\frac{d}{d\delta}\left(\frac{\epsilon}{2\delta} + \frac{\delta C}{2}\right) = 0$ to get $\delta_{\text{opt}} = \sqrt{\epsilon/C}$. Substituting:
$$|gh| < \sqrt{\epsilon C}$$

The "Peter-Paul" terminology: small $\delta$ tightens the $C$ term but loosens the $\epsilon$ term, and vice versa. $\square$

---

## Exercise 5: Induced Norms of 3-Point Averaging Filter

**Problem:** Define $A: \mathbb{R}^n \to \mathbb{R}^n$ by $(A\mathbf{x})_i = \frac{1}{3}(x_{i-1} + x_i + x_{i+1})$ with periodic boundary conditions. Compute $\|A\|_2$ and $\|A\|_\infty$.

**Solution:**

The matrix $A$ is circulant with three $1/3$ entries per row/column.

**$\ell^\infty$ norm:** $\|A\|_\infty = \max_i \sum_{j=1}^n |A_{ij}| = \frac{1}{3} + \frac{1}{3} + \frac{1}{3} = 1$.

**$\ell^2$ norm:** Use $\|A\|_2 \leq \sqrt{\|A\|_1 \|A\|_\infty}$. By symmetry, $\|A\|_1 = 1$, so $\|A\|_2 \leq 1$.

For tightness, test $\mathbf{x} = [1, \ldots, 1]^T$: $(A\mathbf{x})_i = \frac{1}{3}(1+1+1) = 1$, so $A\mathbf{x} = \mathbf{x}$ and $\frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = 1$.

Therefore $\boxed{\|A\|_2 = \|A\|_\infty = 1}$. $\square$

---
