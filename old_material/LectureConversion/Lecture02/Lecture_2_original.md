Ref: Gustafsson et al Ch 1-2

## Today

PyTorch, Linear Model Equations, Analysis essentials

- Consider the domain $\Omega=[0,2 \pi]$ periodic

For PDEs that are $2^{\text{nd}}$ order and linear
$A u_{x x}+2 B u_{x y}+C u_{y y}+D u_{x}+E u_{y}+F=0$
The PDE is parabolic if $B^{2}-A C=0$

- hyperbolic
- elliptic

$$
\begin{aligned}
& B^{2}-A C>0 \\
& B^{2}-A C<0
\end{aligned}
$$

- For $A, B, \ldots F$ varying with space/time, the behavior could be mixed
- For Non-linear PDEs, linearization will show localized behavior

Parabolic - one real characteristic direction

- ex: Heat/diffusion equation

Hyperbolic - Two distinct real characteristics, ex: wave equation
Elliptic - No real characteristics

Different Methods for Each!
ex: Laplace / steady-state diffusion

For each of these we will make use of the following analytic solutions with periodic BC: $u_{k}(0)=u_{k}(2 \pi)$

- Transport Equation

$$
\begin{aligned}
& \partial_{t} u_{k}+\partial_{x} u_{k}=0 \\
& u_{k}=\sin (2 \pi k(x-t))
\end{aligned}
$$

- Unsteady heat equation

$$
\begin{aligned}
& \partial_{t} u_{k}-\partial_{x x} u_{k}=0 \\
& u_{k}=\exp \left(-4 \pi^{2} k^{2} t\right) \sin (2 \pi k x)
\end{aligned}
$$

- Forced Poisson equation

$$
\begin{aligned}
\partial_{x x} u_{k} & =f_{k} \\
u_{k} & =\sin 2 \pi k x \\
f_{k} & =-4 \pi^{2} k^{2} \sin 2 \pi k x
\end{aligned}
$$

**Theorem:** Let $f(x) \in C^{\infty}[0, 2 \pi]$. Then $f$ has the Fourier series representation:

$$
f(x)=\frac{1}{\sqrt{2 \pi}} \sum_{\omega=-\infty}^{\infty} \hat{f}(\omega) e^{i \omega x}
$$

where $\hat{f}(\omega)=\frac{1}{\sqrt{2 \pi}} \cdot \int_{0}^{2 \pi} e^{-i \omega x} f(x) d x$

## Basics of Fourier Analysis

$$
\begin{aligned}
& (f, g)=\int_{0}^{2 \pi} f \bar{g} d x \\
& \|f\|=\sqrt{(f, f)}
\end{aligned}
$$

Inner product

$$
\begin{aligned}
2 i \sin (z) & =\exp (i z)-\exp (-i z) \\
2 \cos (z) & =\exp (i z)+\exp (-i z)
\end{aligned}
$$

**Properties (bilinear):**
$$
\begin{array}{ll}
(f, g)=\overline{(g, f)} & (f+g, h)=(f, h)+(g, h) \\
(f, \lambda g)=\lambda(f, g)
\end{array}
$$

## Useful Inequalities

$$
\begin{aligned}
& |(f, g)| \leq\|f\| \cdot\|g\| \\
& \|f+g\| \leq\|f\|+\|g\| \\
& |\|f\|-\|g\|| \leq\|f-g\|
\end{aligned}
$$

**Lemma:** 

$$
|(f, g)| \leq \delta\|f\|^{2}+\frac{1}{\delta}\|g\|^{2}, \quad \delta>0
$$

**For Matrices/Operators:**

**Spectral Radius:** $\rho(A)=\max |\lambda_{i}|, \quad \rho(A) \leq\|A\|$

$$
\left(\frac{1}{\sqrt{2 \pi}} e^{i n x}, \frac{1}{\sqrt{2 \pi}} e^{i m x}\right)= \begin{cases}1 & n=m \\ 0 & n \neq m\end{cases}
$$

**Theorem (Parseval's Theorem):**
Let $A, B \in L^{2}([0,2 \pi])$

$$
\begin{aligned}
& A(x)=\sum_{n=-\infty}^{\infty} a_{n} \exp (i n x) \\
& \bar{B}(x)=\sum_{m=-\infty}^{\infty} \bar{b}_{m} \exp (-i m x)
\end{aligned}
$$

Then $\sum_{n=-\infty}^{\infty} a_{n} \overline{b_{n}}=\frac{1}{2 \pi} \int_{0}^{2 \pi} A(x) \overline{B(x)} d x$

**Proof:** $\int_{0}^{2 \pi} A \bar{B} d x=\sum_{n, m=-\infty}^{\infty} a_{n} \bar{b}_{m}\left(\exp (inx), \exp (-imx)\right)$

Kronecker delta: $\delta_{i j}= \begin{cases}1 & i=j \\ 0 & i \neq j\end{cases}$

$$
\begin{aligned}
& =\sum_{n, m} a_{n} \overline{b_{m}} 2 \pi \delta_{n m} \\
& =\sum_{n} a_{n} \overline{b_{n}} 2 \pi
\end{aligned}
$$

## First-Order Wave Equation

$$\left\{\begin{array}{l}\partial_{t} u+\partial_{x} u=0 \\ u(x, t=0)=f(x)\end{array}\right.$$

Complete hyperbolic problem

Expand $f$ in Fourier modes:

$$
f(x)=\frac{1}{\sqrt{2 \pi}} \sum_{\omega} \exp (i \omega x) \hat{f}(\omega)
$$

Make ansatz:

$$
u(x,t)=\frac{1}{\sqrt{2 \pi}} \sum_{\omega} \exp (i \omega x) \hat{u}(\omega, t)
$$

Substituting yields the Fourier transform ODE:

$$
\begin{aligned}
& \frac{d \hat{u}}{d t}=-i \omega \hat{u} \\
& \hat{u}(t=0; \omega)=\hat{f}(\omega)
\end{aligned} \Rightarrow \hat{u}(\omega, t)=\exp (-i \omega t) \hat{f}(\omega)
$$

So that, after back substitution and by superposition:

$$
u(x,t)=\frac{1}{\sqrt{2 \pi}} \sum_{\omega=-\infty}^{\infty} \exp [i \omega(x-t)] \hat{f}(\omega)=f(x-t)
$$

⇒ Solutions consist of translating the initial condition

## Our First Energetic Principle

For all $t$:

$$
\begin{aligned}
\|u(\cdot, t)\|^{2}=\int u^{2} d x & =\sum_{\omega=-\infty}^{\infty}|\exp (-i \omega t) \hat{f}(\omega)|^{2} \\
& =\sum_{\omega}|\hat{f}(\omega)|^{2} \\
& =\int f^{2} dx \\
& =\|f\|^{2}
\end{aligned}
$$

We call $\|u\|^{2}$ the energy of $u$. Structure we want to preserve:

- energy conservation
- speed of propagation along $x-t$ characteristics

$$
u(x-t)=\text{const.}
$$

## Periodic Gridfunctions

Let $h=\frac{2 \pi}{N+1}$

$$
x_{j}=j h
$$

**Grid function:**

$$
\begin{aligned}
& u_{j}:=u\left(x_{j}\right)=u\left(x_{j}+2 \pi\right)=u_{j+N+1} \\
& (u v)_{j}=u_{j} v_{j} \quad(u+v)_{j}=u_{j}+v_{j}
\end{aligned}
$$

**Translation operator** - The linear operator satisfying:

$$
\begin{aligned}
(E v)_{j}&=v_{j+1} \\
(E^{-1}v)_{j}&=v_{j-1} \\
(E^{0} v)_{j}&=v_{j}
\end{aligned}
$$

## Finite Difference Operators

$$
\begin{aligned}
& D_{+}=\left(E-E^{0}\right) / h \\
& D_{-}=\left(E^{0}-E^{-1}\right) / h=E^{-1} D_{+} \\
& D_{0}=\frac{E-E^{-1}}{2 h}=\frac{1}{2}\left(D_{+}+D_{-}\right)
\end{aligned}
$$

Consider their action on a Fourier mode $e^{i \omega x}$:

$$
\begin{aligned}
h D_{+} e^{i \omega x} & =e^{i \omega(x+h)}-e^{i \omega x} \\
& =\left(e^{i \omega h}-1\right) e^{i \omega x} \\
& =\left(i \omega h+O\left(\omega^{2} h^{2}\right)\right) e^{i \omega x}
\end{aligned}
$$

Similarly:

$$
\begin{aligned}
h D_{-} e^{i \omega x} & =\left(1-e^{-i \omega h}\right) e^{i \omega x} \\
& =\left(i \omega h+O\left(\omega^{2} h^{2}\right)\right) e^{i \omega x}
\end{aligned}
$$

$$
\begin{aligned}
h D_{0} e^{i \omega x} & =\frac{e^{i\omega h}-e^{-i\omega h}}{2} e^{i \omega x} \\
& =i \sin (\omega h) e^{i \omega x} \\
& =\left(i \omega h+O\left(\omega^{3} h^{3}\right)\right) e^{i \omega x}
\end{aligned}
$$

**Error estimates:**
$$
\begin{aligned}
\left|\left(D_{+}-\partial_{x}\right) e^{i \omega x}\right| & =O\left(\omega^{2} h\right) \\
\left|\left(D_{-}-\partial_{x}\right) e^{i \omega x}\right| & =O\left(\omega^{2} h\right) \\
\left|\left(D_{0}-\partial_{x}\right) e^{i \omega x}\right| & =O\left(\omega^{3} h^{2}\right)
\end{aligned}
$$

**Higher Order Derivatives** follow from products of $D_{+}, D_{-}, D_{0}$:

$$
\text{e.g. }\left(D_{+} D_{-} v\right)_{j}=h^{-2}\left(v_{j+1}-2 v_{j}+v_{j-1}\right)
$$

**Claim:** Difference operators commute

**Proof:**

$$
\begin{aligned}
D_{1} & =\sum_{i} \alpha_{i} E^{i} \\
D_{2} & =\sum_{j} \beta_{j} E^{j} \\
D_{1} D_{2} & =\sum_{i,j} \alpha_{i} \beta_{j} E^{i+j}=D_{2} D_{1}
\end{aligned}
$$

## Finite Differences for Transport Equation

Let

$$
\begin{array}{ll}
h=\frac{2 \pi}{N+1} & k \ll 1 \\
\text{grid size} & \text{timestep}
\end{array}
$$

Build a grid $\left(x_{j}, t_{n}\right)=(j h, n k)$

Discretize transport PDE naively:

$$
\begin{aligned}
v_{j}^{n+1} & =\left(I-k D_{0}\right) v_{j}^{n} \\
& :=Q v_{j}^{n}
\end{aligned}
$$

where $Q$ is the **timestep operator**.

To analyze, again consider a single mode but now on grid:

$$
f_{j}=\frac{1}{\sqrt{2 \pi}} \exp \left(i \omega x_{j}\right) \hat{f}(\omega)
$$

**Ansatz:**
$$
v_{j}^{n}=\frac{1}{\sqrt{2 \pi}} \hat{v}^{n}(\omega) \exp \left(i \omega x_{j}\right)
$$

Substituting into our update formula, with $\lambda=\frac{k}{h}$:

$$
\exp \left(i \omega x_{j}\right) \hat{v}^{n+1}(\omega)=\left[\exp \left(i \omega x_{j}\right)-\frac{\lambda}{2}\left(\exp \left(i \omega x_{j+1}\right)-\exp \left(i \omega x_{j-1}\right)\right)\right]
$$

Define $\xi=\omega h$

$$
\begin{aligned}
& \hat{V}^{n+1}=\hat{Q} \hat{V}^{n} \\
& \hat{Q}=1+i \lambda \sin S
\end{aligned}
$$

⟵ "iymbol" of licuste operator or
"amplification factor"

Taking multiple steps

$$
\hat{v}^{n}(w)=\hat{Q}^{n} v^{0}(w)=\hat{Q}^{n} \hat{f}(w)
$$

$$
\Rightarrow v_{j}^{n}=\frac{1}{\sqrt{2 \pi}}\left(1-i \frac{k}{h} \sin (\omega h)\right)^{n} \exp \left(i \omega x_{j}\right) \hat{f}(\omega)
$$

is the exact expression for the FD solution

## Convergence

How does this behave as $h, k \rightarrow 0$?

$$
\begin{aligned}
&\left(1-i \frac{k}{h} \sin (\omega h)\right)^{n}=\left(1-i \omega k+O\left(k h^{2} \omega^{3}\right)\right)^{n} \\
&=\left(\exp (-i \omega k)+O\left(k^{2} \omega^{2}+k h^{2} \omega^{3}\right)\right)^{n} \\
& \quad=\exp(-i\omega k)^n \left(1+O\left(k \omega^{2}+h^{2} \omega^{3}\right) t_{n}\right) \\
& \quad=e^{-i \omega t_{n}}\left(1+O\left(k \omega^{2}+h^{2} \omega^{3}\right) t_{n}\right)  \\
& \Rightarrow \quad v_{j}^{n}=\frac{1}{\sqrt{2 \pi}}\left(1+O\left(k \omega^{2}+h^{2} \omega^{3}\right) t_{n}\right) \exp \left(i \omega\left(x_{j}-t_{n}\right)\right) \hat{f}(\omega)
\end{aligned}
$$

**Two limits:**

$$
\lim _{k, h \rightarrow 0} v_{j}^{n}=u\left(x_{j}, t_{n}\right)
$$

Now fix $\lambda=\frac{k}{h}>0$

$$
\begin{aligned}
& \hat{Q} \sim\left(1-c i \frac{k}{h}\right)^{n} \\
& \hat{v}_{j}^{n}=(1-c i \lambda)^{t_{n} / k} \hat{f}(\omega) \\
& \lim _{k \rightarrow 0} \hat{v}_{j}^{n}=\infty
\end{aligned}
$$

## Important Notes

