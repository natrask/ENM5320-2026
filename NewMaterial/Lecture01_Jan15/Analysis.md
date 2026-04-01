# Math fundamentals

This document provides a summary of mathematical background that will be assumed throughout the course. It is incompete - the following references will be useful if you need to fill in the gaps. 

- [Gene Golub & Charles Van Loan, *Matrix Computations* (4th Edition)](https://find.library.upenn.edu/catalog/9977066647603681?hld_id=53791354980003681) - Available online through Penn Library. The linear algebra bible!
- [The matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) - A pdf collecting essentially all matrix/vector identities in one spot beyond those here.

These tools will provide the machinery we need to talk about stability and prove properties of machine learning architectures. If this is new to you, it's OK to be a little overwhelmed. We will work examples and get plenty of practice using these over and over throughout the semester. If you need help filling in gaps in your math background, be proactive and come to office hours early. Mastering these kinds of manipulations is like learning a foreign language - it takes practice, and if you wait until the day before assignments are due you won't have enough time to get these into your mental muscle memory.

## Vector Norms

Let $\mathbf{x} \in \mathbb{R}^n$. A **norm** $\|\cdot\|$ is a function $\mathbb{R}^n \to \mathbb{R}_+$ satisfying:
1. $\|\mathbf{x}\| \geq 0$ with equality if and only if $\mathbf{x} = \mathbf{0}$ (positive definiteness)
2. $\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|$ for all $\alpha \in \mathbb{R}$ (homogeneity)
3. $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$ (triangle inequality)

### Common Vector Norms

**$\ell^p$ norms:** For $p \geq 1$,
$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$$

Important special cases:
- **$\ell^1$ norm (Manhattan):** $\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$
- **$\ell^2$ norm (Euclidean):** $\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^\top \mathbf{x}}$
- **$\ell^\infty$ norm (max norm):** $\|\mathbf{x}\|_\infty = \max_{i=1,\ldots,n} |x_i|$

### Norm Equivalence

In finite dimensions, all norms are equivalent. Specifically, for any two norms $\|\cdot\|_a$ and $\|\cdot\|_b$ on $\mathbb{R}^n$, there exist constants $c, C > 0$ such that
$$c \|\mathbf{x}\|_a \leq \|\mathbf{x}\|_b \leq C \|\mathbf{x}\|_a \quad \text{for all } \mathbf{x} \in \mathbb{R}^n$$

Concrete examples:
$$\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \sqrt{n} \|\mathbf{x}\|_\infty$$
$$\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq \sqrt{n} \|\mathbf{x}\|_2$$


## Inner Products

An **inner product** on $\mathbb{R}^n$ is a function $\langle \cdot, \cdot \rangle : \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ satisfying:
1. $\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle$ (symmetry)
2. $\langle \alpha \mathbf{x} + \beta \mathbf{y}, \mathbf{z} \rangle = \alpha \langle \mathbf{x}, \mathbf{z} \rangle + \beta \langle \mathbf{y}, \mathbf{z} \rangle$ (linearity in first argument)
3. $\langle \mathbf{x}, \mathbf{x} \rangle \geq 0$ with equality if and only if $\mathbf{x} = \mathbf{0}$ (positive definiteness)

### Standard Inner Product

The **standard inner product** (or **Euclidean inner product**) on $\mathbb{R}^n$ is:
$\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^n x_i y_i = \mathbf{x}^\top \mathbf{y}$

This can also be written in Einstein notation as $\langle \mathbf{x}, \mathbf{y} \rangle = x_i y_i$.

### Weighted Inner Products

More generally, given a symmetric positive definite matrix $M \in \mathbb{R}^{n \times n}$, we can define a weighted inner product:
$\langle \mathbf{x}, \mathbf{y} \rangle_M = \mathbf{x}^\top M \mathbf{y} = \sum_{i,j=1}^n x_i M_{ij} y_j$

The standard inner product corresponds to $M = I$ (the identity matrix).

### Properties

Key properties of inner products:

**Cauchy-Schwarz inequality:**
$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle} \sqrt{\langle \mathbf{y}, \mathbf{y} \rangle}$

**Polarization identity:**
$\langle \mathbf{x}, \mathbf{y} \rangle = \frac{1}{4}\left(\|\mathbf{x} + \mathbf{y}\|^2 - \|\mathbf{x} - \mathbf{y}\|^2\right)$

where $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$ is the induced norm.

**Parallelogram law:**
$\|\mathbf{x} + \mathbf{y}\|^2 + \|\mathbf{x} - \mathbf{y}\|^2 = 2\|\mathbf{x}\|^2 + 2\|\mathbf{y}\|^2$


## Matrix Norms

For $A \in \mathbb{R}^{m \times n}$, an **induced (operator) norm** is defined as
$$\|A\| = \sup_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|}{\|\mathbf{x}\|} = \sup_{\|\mathbf{x}\| = 1} \|A\mathbf{x}\|$$

### Common Matrix Norms

- **Induced $\ell^1$ norm:** $\|A\|_1 = \max_{j=1,\ldots,n} \sum_{i=1}^m |a_{ij}|$ (max column sum)
- **Induced $\ell^2$ norm (spectral norm):** $\|A\|_2 = \sqrt{\lambda_{\max}(A^\top A)} = \sigma_{\max}(A)$ (largest singular value)
- **Induced $\ell^\infty$ norm:** $\|A\|_\infty = \max_{i=1,\ldots,m} \sum_{j=1}^n |a_{ij}|$ (max row sum)
- **Frobenius norm:** $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{tr}(A^\top A)}$ (not induced, but submultiplicative)

### Properties of Matrix Norms

For any induced matrix norm:
1. **Submultiplicativity:** $\|AB\| \leq \|A\| \|B\|$
2. **Consistency:** $\|A\mathbf{x}\| \leq \|A\| \|\mathbf{x}\|$
3. $\|I\| = 1$ where $I$ is the identity matrix

Additional useful properties:
- $\|A\|_2 \leq \sqrt{\|A\|_1 \|A\|_\infty}$
- $\|A\|_2 \leq \|A\|_F \leq \sqrt{\min(m,n)} \|A\|_2$


## Einstein Summation Convention and Tensor Notation

### Einstein Notation

The **Einstein summation convention** automatically sums over repeated indices that appear once as a subscript and once as a superscript (or twice as subscripts in Cartesian coordinates). This eliminates the need for explicit summation symbols.

**Basic rules:**
- Repeated indices are summed over (called **dummy indices**)
- Free indices must match on both sides of an equation
- Each index appears at most twice in any term

**Examples:**

Inner product of vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:
$\mathbf{u} \cdot \mathbf{v} = u_i v_i = \sum_{i=1}^n u_i v_i$

Matrix-vector multiplication $(A\mathbf{x})_i$:
$y_i = A_{ij} x_j = \sum_{j=1}^n A_{ij} x_j$

Matrix-matrix multiplication $(AB)_{ik}$:
$C_{ik} = A_{ij} B_{jk} = \sum_{j=1}^n A_{ij} B_{jk}$

Trace of a matrix:
$\text{tr}(A) = A_{ii} = \sum_{i=1}^n A_{ii}$

Frobenius norm squared:
$\|A\|_F^2 = A_{ij} A_{ij} = \sum_{i,j} A_{ij}^2$

### Kronecker Delta

The **Kronecker delta** $\delta_{ij}$ is defined as:
$\delta_{ij} = \begin{cases} 
1 & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}$

In matrix form, $[\delta_{ij}] = I$ (the identity matrix).

**Properties:**
- $\delta_{ij} v_j = v_i$ (extracts the $i$-th component)
- $\delta_{ii} = n$ (sum over a repeated index gives the dimension)
- $\delta_{ij} \delta_{jk} = \delta_{ik}$ (composition property)
- $A_{ij} \delta_{jk} = A_{ik}$ (relabeling index)

### Levi-Civita Symbol

The **Levi-Civita symbol** (or **permutation tensor**) $\epsilon_{ijk}$ in 3D is defined as:
$\epsilon_{ijk} = \begin{cases} 
+1 & \text{if } (i,j,k) \text{ is an even permutation of } (1,2,3) \\
-1 & \text{if } (i,j,k) \text{ is an odd permutation of } (1,2,3) \\
0 & \text{if any index is repeated}
\end{cases}$

**Even permutations:** $(1,2,3), (2,3,1), (3,1,2)$ have $\epsilon = +1$

**Odd permutations:** $(1,3,2), (3,2,1), (2,1,3)$ have $\epsilon = -1$

**Applications:**

Cross product in $\mathbb{R}^3$:
$(\mathbf{u} \times \mathbf{v})_i = \epsilon_{ijk} u_j v_k$

Explicitly:
$\mathbf{u} \times \mathbf{v} = \begin{pmatrix} u_2 v_3 - u_3 v_2 \\ u_3 v_1 - u_1 v_3 \\ u_1 v_2 - u_2 v_1 \end{pmatrix}$

Determinant of a $3 \times 3$ matrix:
$\det(A) = \epsilon_{ijk} A_{1i} A_{2j} A_{3k}$

**Contraction identity** (very useful):
$\epsilon_{ijk} \epsilon_{lmk} = \delta_{il} \delta_{jm} - \delta_{im} \delta_{jl}$

**Vector identity:**
$(\mathbf{u} \times \mathbf{v}) \cdot (\mathbf{w} \times \mathbf{z}) = (\mathbf{u} \cdot \mathbf{w})(\mathbf{v} \cdot \mathbf{z}) - (\mathbf{u} \cdot \mathbf{z})(\mathbf{v} \cdot \mathbf{w})$

which can be proven using: $\epsilon_{ijk} \epsilon_{imn} = \delta_{jm}\delta_{kn} - \delta_{jn}\delta_{km}$

## Fundamental Inequalities

### Cauchy-Schwarz Inequality

For $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$:
$$|\mathbf{x}^\top \mathbf{y}| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$$

Equality holds if and only if $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent.

**Generalization:** For any inner product space,
$$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \|\mathbf{y}\|$$

### Triangle Inequality

For any norm:
$$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$$

**Reverse triangle inequality:**
$$\big| \|\mathbf{x}\| - \|\mathbf{y}\| \big| \leq \|\mathbf{x} - \mathbf{y}\|$$

### Holder''s Inequality

For $p, q \geq 1$ with $\frac{1}{p} + \frac{1}{q} = 1$ (conjugate exponents):
$$\sum_{i=1}^n |x_i y_i| \leq \|\mathbf{x}\|_p \|\mathbf{y}\|_q$$

Special case ($p = q = 2$): Cauchy-Schwarz inequality.

### Minkowski''s Inequality

For $p \geq 1$:
$$\|\mathbf{x} + \mathbf{y}\|_p \leq \|\mathbf{x}\|_p + \|\mathbf{y}\|_p$$

This is the triangle inequality for $\ell^p$ norms.

### Young''s Inequality

For $a, b \geq 0$ and conjugate exponents $p, q > 1$ with $\frac{1}{p} + \frac{1}{q} = 1$:
$$ab \leq \frac{a^p}{p} + \frac{b^q}{q}$$

Special case ($p = q = 2$):
$$ab \leq \frac{a^2}{2} + \frac{b^2}{2}$$

More generally, for any $\epsilon > 0$:
$$ab \leq \frac{\epsilon a^2}{2} + \frac{b^2}{2\epsilon}$$

This is particularly useful for absorbing terms in energy estimates.

## Taylor Series

### Taylor's Theorem with Remainder

For a function $f: \mathbb{R} \to \mathbb{R}$ that is $(n+1)$-times continuously differentiable on an interval containing $a$ and $x$, Taylor's theorem states:

$$f(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k + R_n(x)$$

where $R_n(x)$ is the **remainder term** (or **truncation error**).

### Forms of the Remainder

**Lagrange form (most common for error analysis):**
$$R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}$$

for some $\xi$ between $a$ and $x$. This follows from the mean value theorem applied repeatedly.

**Integral form:**
$$R_n(x) = \frac{1}{n!}\int_a^x (x-t)^n f^{(n+1)}(t) \, dt$$

**Big-O notation:**
$$R_n(x) = O(|x-a|^{n+1}) \quad \text{as } x \to a$$

### Common Taylor Expansions

Expanding around $a = 0$ (Maclaurin series):

**Exponential:**
$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots + \frac{x^n}{n!} + \frac{e^\xi}{(n+1)!}x^{n+1}$$

**Sine:**
$$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots + R_n(x)$$

**Cosine:**
$$\cos(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots + R_n(x)$$

**Natural logarithm:**
$$\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots + R_n(x), \quad |x| < 1$$

### Application to Finite Differences

**First-order forward difference:**

For a smooth function $u(x)$ with $|u''(x)| \leq M$, expand around $x$:
$$u(x+h) = u(x) + h u'(x) + \frac{h^2}{2} u''(\xi)$$

for some $\xi \in (x, x+h)$. Rearranging:
$$u'(x) = \frac{u(x+h) - u(x)}{h} - \frac{h}{2} u''(\xi)$$

The **truncation error** is:
$$\left|u'(x) - \frac{u(x+h) - u(x)}{h}\right| = \frac{h}{2}|u''(\xi)| \leq \frac{Mh}{2}$$

Thus the forward difference is **first-order accurate**: error $= O(h)$.

**Second-order centered difference:**

For the second derivative, expand both forward and backward:
$$u(x+h) = u(x) + h u'(x) + \frac{h^2}{2} u''(x) + \frac{h^3}{6} u'''(x) + \frac{h^4}{24} u^{(4)}(\xi_+)$$

$$u(x-h) = u(x) - h u'(x) + \frac{h^2}{2} u''(x) - \frac{h^3}{6} u'''(x) + \frac{h^4}{24} u^{(4)}(\xi_-)$$

Adding these equations:
$$u(x+h) + u(x-h) = 2u(x) + h^2 u''(x) + \frac{h^4}{24}\left[u^{(4)}(\xi_+) + u^{(4)}(\xi_-)\right]$$

Solving for $u''(x)$:
$$u''(x) = \frac{u(x-h) - 2u(x) + u(x+h)}{h^2} - \frac{h^2}{24}\left[u^{(4)}(\xi_+) + u^{(4)}(\xi_-)\right]$$

The **truncation error** is:
$$\left|u''(x) - \frac{u(x-h) - 2u(x) + u(x+h)}{h^2}\right| \leq \frac{h^2}{12}\max|u^{(4)}|$$

Thus the centered difference for the second derivative is **second-order accurate**: error $= O(h^2)$.

### Example: Approximating $\sin(x)$ on an Interval

**Problem:** Approximate $f(x) = \sin(x)$ using a linear function on the interval $[x_0, x_0 + h]$.

**Solution:** Use the first-order Taylor expansion:
$$\sin(x_0 + h) = \sin(x_0) + h\cos(x_0) + \frac{h^2}{2}(-\sin(\xi))$$

for some $\xi \in (x_0, x_0 + h)$.

The linear approximation is:
$$p(x) = \sin(x_0) + (x - x_0)\cos(x_0)$$

The maximum error on the interval is:
$$\max_{x \in [x_0, x_0+h]} |f(x) - p(x)| \leq \frac{h^2}{2}\max_{\xi \in [x_0, x_0+h]} |\sin(\xi)| \leq \frac{h^2}{2}$$

**Numerical example:** For $x_0 = 0$ and $h = 0.1$:
- True value: $\sin(0.1) \approx 0.0998334$
- Linear approximation: $p(0.1) = 0 + 0.1 \cdot 1 = 0.1$
- Error: $|0.0998334 - 0.1| = 0.0001666$
- Bound: $\frac{(0.1)^2}{2} = 0.005$

The actual error is much smaller than the bound because $\sin(\xi) \ll 1$ for small $\xi$.

**Key insight:** The truncation error scales as $h^{n+1}$ where $n$ is the degree of the approximating polynomial. This is fundamental to understanding convergence rates in numerical methods.

## Useful Manipulation Techniques

### Completing the Square

For scalar $a, b$ and $\epsilon > 0$:
$$2ab = \epsilon a^2 + \frac{b^2}{\epsilon} - \left(\sqrt{\epsilon} a - \frac{b}{\sqrt{\epsilon}}\right)^2 \leq \epsilon a^2 + \frac{b^2}{\epsilon}$$

For vectors with respect to a positive definite matrix $M$:
$$2\mathbf{x}^\top M \mathbf{y} \leq \epsilon \mathbf{x}^\top M \mathbf{x} + \frac{1}{\epsilon} \mathbf{y}^\top M \mathbf{y}$$

### Gronwall''s Inequality

If $u(t) \geq 0$ satisfies
$$u(t) \leq C + \int_0^t \alpha(s) u(s) \, ds$$
for constants $C \geq 0$ and $\alpha(s) \geq 0$, then
$$u(t) \leq C \exp\left(\int_0^t \alpha(s) \, ds\right)$$

**Discrete version:** If $u_n \geq 0$ satisfies $u_n \leq C + \sum_{k=0}^{n-1} \alpha_k u_k$ with $\alpha_k \geq 0$, then
$$u_n \leq C \prod_{k=0}^{n-1} (1 + \alpha_k)$$

### Matrix Spectral Properties

For a symmetric matrix $A \in \mathbb{R}^{n \times n}$:
- **Rayleigh quotient:** 
$$\lambda_{\min}(A) \leq \frac{\mathbf{x}^\top A \mathbf{x}}{\mathbf{x}^\top \mathbf{x}} \leq \lambda_{\max}(A)$$

- **Weyl''s inequality:** For symmetric $A, B$:
$$\lambda_{\min}(A) + \lambda_{\min}(B) \leq \lambda_{\min}(A + B) \leq \lambda_{\min}(A) + \lambda_{\max}(B)$$

- **Energy inequality:** If $A$ is positive definite with $\lambda_{\min}(A) = \lambda > 0$:
$$\lambda \|\mathbf{x}\|_2^2 \leq \mathbf{x}^\top A \mathbf{x} \leq \lambda_{\max}(A) \|\mathbf{x}\|_2^2$$

## Summary of Common Estimates

Quick reference for frequently used bounds:

1. $|\mathbf{x}^\top \mathbf{y}| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$ (Cauchy-Schwarz)
2. $\|A\mathbf{x}\|_2 \leq \|A\|_2 \|\mathbf{x}\|_2$ (consistency)
3. $\|AB\|_2 \leq \|A\|_2 \|B\|_2$ (submultiplicativity)
4. $ab \leq \frac{\epsilon a^2}{2} + \frac{b^2}{2\epsilon}$ for any $\epsilon > 0$ (Young)
5. $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \sqrt{n} \|\mathbf{x}\|_\infty$ (norm equivalence)
6. $\|A\|_2 \leq \|A\|_F \leq \sqrt{n} \|A\|_2$ (for $A \in \mathbb{R}^{n \times n}$)

These tools form the foundation for deriving stability estimates, convergence rates, and error bounds throughout the course.

