# Lecture 2: Fourier Analysis and Finite Difference Methods for PDEs

**Reference:** [Gustafsson et al., Chapters 1-2](https://find.library.upenn.edu/catalog/9977071082103681?hld_id=53499053800003681)

**Topics:** Fourier Analysis Fundamentals, Finite Difference Operators, Numerical Methods for PDEs

---

## 0. Overview

At the continuum scale, partial differential equations (PDEs) are the lingua franca of describing systems ranging from fluids to solids to electromagnetism. In the old days, we would derive a PDE by performing a force balance on an infinetesimal element of material and pass to the limit to obtain a PDE. In the 1800s, calculus (and the majority of subsequent math at the time) was developed to then solve these PDEs on paper. In the mid 1900s, these PDEs were discretized using techniques like finite differences or finite elements to solve them numerically on a computer. For each type of physics there is a corresponding classification of PDE; for example, *hyperbolic* PDEs describe wave-like behavior like acoustic or elastic waves, while *elliptic* PDEs describe boundary value problems like the steady state distribution of heat in a system. For each, there is a distinct set of mathematical tools and physical structure that need to be maintained.

In this course, we will flip this paradigm around. Rather than derive and then discretize to model a system, we will reverse engineer a discretized PDE whose solution matches data. That data could come from a physical experiment or simulations. In fluids we may measure a velocity field with Particle Image Velocimetry (PIV) and find a model that could predict behavior under different boundary or initial conditions. Alternatively, we could perform expensive molecular dynamics simulations to understand how a small piece of material evolves and then reverse engineer a model that will describe how that material would scale up to complex engineering geometries. Before we can do this, we first need to understand the fundamental aspects of discretizations of PDEs so that we understand how to design an architecture that can guarantee desirable properties of models, like that they are numerically stable, preserve physical principles like energy/momentum conservation, or provide convergence to know limiting behavior.

I stress that the philosophy that we're pursuing in this class is **different from much of the AI4Science** that is going around these days. Many folks aim to treat physics models like a video, treating physical prediction like next frame prediction of a Youtube video or solving for steady state behavior in the same way one uses generative models to make an image of a cat playing basketball. These techniques generally fail to preserve or exploit physical structure that can make models perform well under extrapolation or small data regimes. Those techniques are nonetheless unbelievably powerful, and so the objective of this class is going to be to develop hybrid methods that are as expressive as modern transformer architectures while providing the robustness guarantees of conventional PDE-based models that are necessary for engineering applications.

## 1. Classification of Second-Order Linear PDEs

Consider second-order linear PDEs on a periodic domain $\Omega=[0,2 \pi]$:

$$
A u_{x x}+2 B u_{x y}+C u_{y y}+D u_{x}+E u_{y}+F=0
$$

The classification depends on the discriminant $B^{2}-A C$:

- **Parabolic:** $B^{2}-A C=0$ (one real characteristic direction)
  - Example: Heat/diffusion equation $\partial_t u = \partial_{xx} u$

- **Hyperbolic:** $B^{2}-A C>0$ (two distinct real characteristics)
  - Example: Wave equation $\partial_{tt} u = c^2 \partial_{xx} u$

- **Elliptic:** $B^{2}-A C<0$ (no real characteristics)
  - Example: Laplace equation $\partial_{xx} u + \partial_{yy} u = 0$

**Important Notes:**
- When coefficients $A, B, \ldots, F$ vary with space/time, the PDE behavior can be mixed (locally different types)
- For nonlinear PDEs, linearization reveals localized behavior
- Different numerical methods are required for each type

### 1.1 Model Problems with Analytic Solutions

When we develop techniques for solving these classes of problems, and also for learning models that can recover each of them, it will be useful to have some analytic solutions that describe how a single mode of wavelength $k$ will evolve. We can use these to both verify solvers recover true solutions (no bugs in the code) and to generate training data.

For periodic boundary conditions $u_k(0) = u_k(2\pi)$, we consider:

**Transport Equation (Hyperbolic):**
$$
\begin{aligned}
& \partial_{t} u_{k}+\partial_{x} u_{k}=0 \\
& u_{k}=\sin (2 \pi k(x-t))
\end{aligned}
$$

**Unsteady Heat Equation (Parabolic):**
$$
\begin{aligned}
& \partial_{t} u_{k}-\partial_{x x} u_{k}=0 \\
& u_{k}=\exp \left(-4 \pi^{2} k^{2} t\right) \sin (2 \pi k x)
\end{aligned}
$$

**Forced Poisson Equation (Elliptic):**
$$
\begin{aligned}
\partial_{x x} u_{k} & =f_{k} \\
u_{k} & =\sin 2 \pi k x \\
f_{k} & =-4 \pi^{2} k^{2} \sin 2 \pi k x
\end{aligned}
$$

---

## 2. Fourier Analysis Fundamentals

When working in a periodic domain, Fourier analysis gives us a simple means of analyzing the stability of models that will be sufficient for finite difference discretizations on uniform grids. Later in the class we will cover more sophisticated means of analysis, but this provides a good warm up for us right now.

### 2.1 Fourier Series Representation

**Theorem:** Let $f(x) \in C^{\infty}[0, 2 \pi]$. Then $f$ has the Fourier series representation:

$$
f(x)=\frac{1}{\sqrt{2 \pi}} \sum_{\omega=-\infty}^{\infty} \hat{f}(\omega) e^{i \omega x}
$$

where the Fourier coefficients are:

$$
\hat{f}(\omega)=\frac{1}{\sqrt{2 \pi}} \int_{0}^{2 \pi} e^{-i \omega x} f(x) d x
$$

### 2.2 Inner Products and Norms

**Inner product** on $L^2([0, 2\pi])$:

$$
(f, g)=\int_{0}^{2 \pi} f \bar{g} d x
$$

**Norm:**

$$
\|f\|=\sqrt{(f, f)}
$$

**Euler's Formula for Trigonometric Functions:**

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

### 2.3 Orthonormality of Fourier Modes

The complex exponentials form an orthonormal basis:

$$
\left(\frac{1}{\sqrt{2 \pi}} e^{i n x}, \frac{1}{\sqrt{2 \pi}} e^{i m x}\right)= \begin{cases}1 & n=m \\ 0 & n \neq m\end{cases}
$$

### 2.4 Fundamental Inequalities

We will make extensive use of the following inequalities when we're analyzing finite difference schemes.

**Cauchy-Schwarz Inequality:**
$$
|(f, g)| \leq\|f\| \cdot\|g\|
$$

**Triangle Inequality:**
$$
\|f+g\| \leq\|f\|+\|g\|
$$

**Reverse Triangle Inequality:**
$$
|\|f\|-\|g\|| \leq\|f-g\|
$$

**Lemma (Young's Inequality):**

$$
|(f, g)| \leq \delta\|f\|^{2}+\frac{1}{\delta}\|g\|^{2}, \quad \delta>0
$$

**For Matrices/Operators:**

**Spectral Radius:** $\rho(A)=\max |\lambda_{i}|, \quad \rho(A) \leq\|A\|$

### 2.5 Parseval's Theorem

Parseval's theorem allows us to jump back and forth between descriptions of a function in physical space or Fourier space. Remember that a function $f \in L^2(\Omega)$ if $\int_\Omega f^2 dx < \infty$.

**Theorem (Parseval's Theorem):**
Let $A, B \in L^{2}([0,2 \pi])$ with Fourier expansions:

$$
\begin{aligned}
& A(x)=\sum_{n=-\infty}^{\infty} a_{n} \exp (i n x) \\
& \bar{B}(x)=\sum_{m=-\infty}^{\infty} \bar{b}_{m} \exp (-i m x)
\end{aligned}
$$

Then:

$$
\sum_{n=-\infty}^{\infty} a_{n} \overline{b_{n}}=\frac{1}{2 \pi} \int_{0}^{2 \pi} A(x) \overline{B(x)} d x
$$

**Proof:** Using the inner product and orthonormality:

$$
\int_{0}^{2 \pi} A \bar{B} d x=\sum_{n, m=-\infty}^{\infty} a_{n} \bar{b}_{m}\left(\exp (i n x), \exp (-i m x)\right)
$$

Applying the Kronecker delta $\delta_{i j}= \begin{cases}1 & i=j \\ 0 & i \neq j\end{cases}$:

$$
\begin{aligned}
& =\sum_{n, m} a_{n} \overline{b_{m}} 2 \pi \delta_{n m} \\
& =\sum_{n} a_{n} \overline{b_{n}} 2 \pi
\end{aligned}
$$

---

## 3. Analytic Solution of the Transport Equation

### 3.1 Problem Setup

Consider the first-order wave (transport) equation:

$$
\left\{\begin{array}{l}
\partial_{t} u+\partial_{x} u=0 \\
u(x, t=0)=f(x)
\end{array}\right.
$$

This is a complete hyperbolic problem with initial condition.

### 3.2 Fourier Mode Analysis

**Step 1:** Expand the initial condition in Fourier modes:

$$
f(x)=\frac{1}{\sqrt{2 \pi}} \sum_{\omega} \exp (i \omega x) \hat{f}(\omega)
$$

**Step 2:** Make ansatz for the solution:

$$
u(x,t)=\frac{1}{\sqrt{2 \pi}} \sum_{\omega} \exp (i \omega x) \hat{u}(\omega, t)
$$

**Step 3:** Substitute into the PDE to get the Fourier transform ODE:

$$
\begin{aligned}
& \frac{d \hat{u}}{d t}=-i \omega \hat{u} \\
& \hat{u}(t=0; \omega)=\hat{f}(\omega)
\end{aligned}
$$

**Step 4:** Solve the ODE:

$$
\hat{u}(\omega, t)=\exp (-i \omega t) \hat{f}(\omega)
$$

**Step 5:** Back-substitute and apply superposition:

$$
u(x,t)=\frac{1}{\sqrt{2 \pi}} \sum_{\omega=-\infty}^{\infty} \exp [i \omega(x-t)] \hat{f}(\omega)=f(x-t)
$$

**Physical Interpretation:** Solutions consist of translating the initial condition with constant velocity along the characteristics.

### 3.3 Energy Conservation

Throughout the course we will be interested in developing discrete representations of PDE solutions that preserve analogues of continuum principles. The following is our first example of this: solutions to the transport equation should preserve some notion of the total mass on a periodic domain, which makes sense because material is advected without creating or destroying it. We will see that the $L^2$ norm $\int u^2 \, dx$ is one notion of conservation. Note that the $L^1$ norm (total mass) is also conserved: $\int u \, dx = \int f \, dx$, but we focus on the $L^2$ norm because it naturally falls out of the definition of a norm; sometimes we refer to these types of norms as the *natural norm* or *energy norm* for a given problem.

**Energetic Principle:** For all $t$, the $L^2$ energy is conserved:

$$
\begin{aligned}
\|u(\cdot, t)\|^{2}=\int u^{2} d x & =\sum_{\omega=-\infty}^{\infty}|\exp (-i \omega t) \hat{f}(\omega)|^{2} \\
& =\sum_{\omega}|\hat{f}(\omega)|^{2} \\
& =\int f^{2} dx \\
& =\|f\|^{2}
\end{aligned}
$$

We call $\|u\|^{2}$ the **energy** of $u$. We will see many examples of energies associated with different PDEs. Sometimes energy will be preserved, while for dissipative systems it may be non-increasing. If one is careless, it is very easy to derive an **unstable** discretization which causes the energy to explode (an undesirable feature).

**Key Properties to Preserve in Numerical Methods for Transport Equation:**
- Energy conservation: $\|u(\cdot, t)\| = \|f\|$
- Speed of propagation along $x-t$ characteristics: $u(x-t)=\text{const.}$

---

## 4. Finite Difference Methods

Finite difference methods (FDMs) are the bread and butter of solving PDEs. They were extensively developed during the Manhattan Project (1940s) for numerical simulation of hydrodynamics and neutron transport. Pioneering work by von Neumann, Courant, Friedrichs, and Lewy at Los Alamos established the mathematical foundations we use today. It is worth understanding the history of where these methods come from [Harlow (2004) "Fluid dynamics in Group T-3 Los Alamos"](https://doi.org/10.1016/j.jcp.2003.09.031) or [Goldstine (1977) "A History of Numerical Analysis from the 16th through the 19th Century"](https://doi.org/10.1007/978-1-4684-9472-3). They were designed to run on the earliest computers and helped design aspects of the nuclear weapons program and aerodynamics during the Manhattan project. Contemporary methods are more effective at modeling complex engineering geometries or exploiting GPU acceleration on modern supercomputers, but we will stick to FDMs for now to get some practice with coding simple 1D problems and designing neural architectures that can learn a FDM-like discretization.

The fundamental description of a PDE in FDMs is a *stencil* which uses Taylor series to approximate a PDE residual on a discrete grid of points. Coincidentally, this is precisely how convolutional neural networks (CNNs) act on images.

### 4.1 Periodic Grid Functions

**Grid setup:** Let $h=\frac{2 \pi}{N+1}$ define a grid spacing and define grid points:

$$
x_{j}=j h, \quad j=0,1,\ldots,N
$$

**Grid function:** A periodic function sampled on the grid:

$$
\begin{aligned}
& u_{j}:=u\left(x_{j}\right)=u\left(x_{j}+2 \pi\right)=u_{j+N+1} \\
& (u v)_{j}=u_{j} v_{j} \quad(u+v)_{j}=u_{j}+v_{j}
\end{aligned}
$$

### 4.2 Translation Operators

**Translation operator** $E$ is the linear operator satisfying:

$$
\begin{aligned}
(E v)_{j}&=v_{j+1} \\
(E^{-1}v)_{j}&=v_{j-1} \\
(E^{0} v)_{j}&=v_{j}
\end{aligned}
$$

### 4.3 Finite Difference Operators

**Forward difference:**
$$
D_{+}=\left(E-E^{0}\right) / h
$$

**Backward difference:**
$$
D_{-}=\left(E^{0}-E^{-1}\right) / h=E^{-1} D_{+}
$$

**Centered difference:**
$$
D_{0}=\frac{E-E^{-1}}{2 h}=\frac{1}{2}\left(D_{+}+D_{-}\right)
$$

### 4.4 Fourier Mode Analysis of Difference Operators

Consider the action on a Fourier mode $e^{i \omega x}$:

**Forward difference:**
$$
\begin{aligned}
h D_{+} e^{i \omega x} & =e^{i \omega(x+h)}-e^{i \omega x} \\
& =\left(e^{i \omega h}-1\right) e^{i \omega x} \\
& =\left(i \omega h+O\left(\omega^{2} h^{2}\right)\right) e^{i \omega x}
\end{aligned}
$$

**Backward difference:**
$$
\begin{aligned}
h D_{-} e^{i \omega x} & =\left(1-e^{-i \omega h}\right) e^{i \omega x} \\
& =\left(i \omega h+O\left(\omega^{2} h^{2}\right)\right) e^{i \omega x}
\end{aligned}
$$

**Centered difference:**
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
\left|\left(D_{+}-\partial_{x}\right) e^{i \omega x}\right| & =O\left(\omega^{2} h\right) \quad \text{(first-order)} \\
\left|\left(D_{-}-\partial_{x}\right) e^{i \omega x}\right| & =O\left(\omega^{2} h\right) \quad \text{(first-order)} \\
\left|\left(D_{0}-\partial_{x}\right) e^{i \omega x}\right| & =O\left(\omega^{3} h^{2}\right) \quad \text{(second-order)}
\end{aligned}
$$

**Observation:** The centered difference operator is second-order accurate, while forward/backward differences are only first-order accurate.

### 4.5 Higher-order Derivatives and Commutativity

**Higher-order derivatives** are constructed from products:

$$
\text{e.g. }\left(D_{+} D_{-} v\right)_{j}=h^{-2}\left(v_{j+1}-2 v_{j}+v_{j-1}\right)
$$

**Claim:** Difference operators commute.

**Proof:** Any difference operator can be written as:

$$
\begin{aligned}
D_{1} & =\sum_{i} \alpha_{i} E^{i} \\
D_{2} & =\sum_{j} \beta_{j} E^{j}
\end{aligned}
$$

Therefore:

$$
D_{1} D_{2} =\sum_{i,j} \alpha_{i} \beta_{j} E^{i+j}=D_{2} D_{1}
$$

---

## 5. Finite Difference Discretization of Transport Equation

### 5.1 Space-Time Discretization

**Grid parameters:**
$$
\begin{array}{ll}
h=\frac{2 \pi}{N+1} & k \ll 1 \\
\text{(grid size)} & \text{(timestep)}
\end{array}
$$

**Space-time grid:** $\left(x_{j}, t_{n}\right)=(j h, n k)$

**Discretization:** Using centered differences in space:

$$
\begin{aligned}
v_{j}^{n+1} & =\left(I-k D_{0}\right) v_{j}^{n} \\
& :=Q v_{j}^{n}
\end{aligned}
$$

where $Q$ is the **timestep operator** or **update operator**.

### 5.2 Stability Analysis via Fourier Modes

**Discrete Fourier mode on grid:**

$$
f_{j}=\frac{1}{\sqrt{2 \pi}} \exp \left(i \omega x_{j}\right) \hat{f}(\omega)
$$

**Ansatz for numerical solution:**

$$
v_{j}^{n}=\frac{1}{\sqrt{2 \pi}} \hat{v}^{n}(\omega) \exp \left(i \omega x_{j}\right)
$$

**Substitution:** With CFL parameter $\lambda=\frac{k}{h}$:

$$
\exp \left(i \omega x_{j}\right) \hat{v}^{n+1}(\omega)=\left[\exp \left(i \omega x_{j}\right)-\frac{\lambda}{2}\left(\exp \left(i \omega x_{j+1}\right)-\exp \left(i \omega x_{j-1}\right)\right)\right]
$$

**Amplification factor:** Define $\xi=\omega h$, then:

$$
\begin{aligned}
& \hat{v}^{n+1}=\hat{Q} \hat{v}^{n} \\
& \hat{Q}=1-i \lambda \sin \xi
\end{aligned}
$$

The quantity $\hat{Q}$ is called the **symbol** of the discrete operator or **amplification factor**.

**Multiple timesteps:**

$$
\hat{v}^{n}(\omega)=\hat{Q}^{n} v^{0}(\omega)=\hat{Q}^{n} \hat{f}(\omega)
$$

**Exact FD solution:**

$$
v_{j}^{n}=\frac{1}{\sqrt{2 \pi}}\left(1-i \frac{k}{h} \sin (\omega h)\right)^{n} \exp \left(i \omega x_{j}\right) \hat{f}(\omega)
$$

---

## 6. Convergence Analysis

### 6.1 Limit as Grid Spacing and Timestep Vanish

**Question:** How does the numerical solution behave as $h, k \rightarrow 0$?

**Analysis:** Expanding the amplification factor:

$$
\begin{aligned}
&\left(1-i \frac{k}{h} \sin (\omega h)\right)^{n}=\left(1-i \omega k+O\left(k h^{2} \omega^{3}\right)\right)^{n} \\
&=\left(\exp (-i \omega k)+O\left(k^{2} \omega^{2}+k h^{2} \omega^{3}\right)\right)^{n} \\
& \quad=(\exp(-i\omega k))^n \left(1+O\left(k \omega^{2}+h^{2} \omega^{3}\right) t_{n}\right) \\
& \quad=e^{-i \omega t_{n}}\left(1+O\left(k \omega^{2}+h^{2} \omega^{3}\right) t_{n}\right)
\end{aligned}
$$

**Therefore:**

$$
v_{j}^{n}=\frac{1}{\sqrt{2 \pi}}\left(1+O\left(k \omega^{2}+h^{2} \omega^{3}\right) t_{n}\right) \exp \left(i \omega\left(x_{j}-t_{n}\right)\right) \hat{f}(\omega)
$$

### 6.2 Two Important Limits

**Limit 1 (Both $k, h \rightarrow 0$ independently):**

$$
\lim _{k, h \rightarrow 0} v_{j}^{n}=u\left(x_{j}, t_{n}\right)
$$

The numerical solution converges to the exact solution.

**Limit 2 (Fixed CFL ratio $\lambda = k/h$):**

Now fix $\lambda=\frac{k}{h}>0$ and let $k \rightarrow 0$:

$$
\begin{aligned}
& \hat{Q} \sim\left(1-c i \frac{k}{h}\right)^{n} \\
& \hat{v}_{j}^{n}=(1-c i \lambda)^{t_{n} / k} \hat{f}(\omega)
\end{aligned}
$$

**Critical observation:**

$$
\lim _{k \rightarrow 0} \hat{v}_{j}^{n}=\infty
$$

The solution **diverges** when we fix the ratio $\lambda$ and refine the grid!

**Conclusion:** Simply having $h, k \rightarrow 0$ is not sufficient for convergence. The relationship between spatial and temporal discretization matters critically. This motivates the study of **stability** and the **CFL condition** in the next lectures.

---

## Summary

This lecture covered:

1. **PDE Classification:** Parabolic, hyperbolic, and elliptic PDEs based on characteristic structure
2. **Fourier Analysis:** Inner products, orthonormality, Parseval's theorem
3. **Analytic Solutions:** Fourier mode analysis for the transport equation
4. **Energy Methods:** Conservation principles that numerical methods should preserve
5. **Finite Differences:** Forward, backward, and centered difference operators
6. **Numerical Discretization:** Finite difference scheme for the transport equation
7. **Convergence:** Two different limiting behaviors depending on how $k$ and $h$ approach zero

**Key Takeaway:** Numerical convergence requires careful balance between spatial and temporal discretization, not just refinement of both independently. The CFL condition (to be studied next) provides the necessary constraint.

