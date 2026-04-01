# Lecture 11: Stochastic Differential Equations and Probabilistic Learning

**Date:** February 26

**Topics Covered:**
- Integrating probability into learned models
- Stochastic dynamics
- Maximum likelihood estimation (MLE)
- Markov processes
- Euler-Maruyama integration
- Metriplectic/GENERIC formalism

---

## 0. Overview

Up until now, our focus has been on deterministic systems governed by variational principles and Hamiltonian mechanics. However, real-world physical systems invariably involve **uncertainty**: sensor noise, incomplete physics descriptions, and stochastic forcing all require us to think probabilistically. This lecture marks a crucial expansion of our framework to handle **stochastic differential equations (SDEs)**, which combine deterministic dynamics with random processes.

The motivation for this probabilistic perspective is threefold. First, tracking a single point trajectory is often insufficient—we need to track entire **probability distributions** that evolve under both deterministic forces and random fluctuations. Second, even when our measurements are noisy or our physics model is incomplete, we can still learn meaningful dynamics by fitting probabilistic models to data using **maximum likelihood estimation**. Third, and perhaps most surprisingly, even in the absence of noise, training models in a probabilistic context often provides better regularization and more robust learning.

We'll build our SDE framework systematically, starting with a probability review, then introducing the **Wiener process** (Brownian motion) and showing how to integrate it using the **Euler-Maruyama method**. Using **Markov process** structure, we'll derive a negative log-likelihood (NLL) loss function that enables us to fit learnable SDE models to time series data. Finally, we'll see an elegant example of structure-preserving SDEs through the **metriplectic (GENERIC) formalism**, which simultaneously conserves energy while increasing entropy—a beautiful generalization of Hamiltonian mechanics to irreversible thermodynamic systems.

---

## 1. Motivation for Probabilistic Dynamics

### 1.1 From Point Tracking to Probability Tracking

Instead of tracking a single point trajectory, we often need to track a **"blob" of probability** that evolves over time. This arises naturally in several contexts:

**Aleatoric Uncertainty:**
- **Noisy observations** from sensors with measurement error
- **Noisy physics** due to unmodeled random forcing

**Epistemic Uncertainty:**
- **Incomplete description of physics** (e.g., missing scales or interactions)
- Model uncertainty about functional forms

### 1.2 Probabilistic Regularization

**Even without noise, training a model in a probabilistic context is often easier.** Probabilistic objectives provide natural regularization:

- **Variance terms** penalize overconfident predictions
- **Log-likelihood** naturally balances fit quality against model complexity
- **Bayesian priors** encode structural assumptions

### 1.3 Recommended References for Probability

**For rigorous foundations:**
- *Probability Essentials* by Jacod & Protter  
  → Short paperback, rigorous but quick definitions

- *Probability: Theory and Examples* by Durrett  
  → Measure-theoretic; heavy to get started from

**For machine learning perspective:**
- *Machine Learning: A Probabilistic Perspective* by Kevin Murphy  
  → Accessible, lots of background and follow-up refs  
  → Like Wikipedia for ML (no ODEs/PDEs though)

**For stochastic integration:**
- *Introduction to Stochastic Integration* by Kuo  
  → Where I'll pull references from

**Overall:** I wouldn't recommend following a single text for this course, because there isn't a good one that covers scientific computing + ML together.

---

## 2. Probability Review

### 2.1 Random Variables and Distributions

**Definition:** A **continuous random variable** $\bar{X}$ takes a random value $x \in \mathbb{R}$.

**Definition:** A **cumulative distribution function (CDF)** defines probability over a range of values:

$$
F_{\bar{X}}(x) = \mathbb{P}(\bar{X} \leq x)
$$

**Definition:** If the CDF is differentiable, we define the **probability density function (PDF)**:

$$
f_{\bar{X}}(x) = \frac{d}{dx} F_{\bar{X}}(x)
$$

Thus:
$$
\mathbb{P}(a \leq \bar{X} \leq b) = \int_a^b f(x) \, dx = F(b) - F(a)
$$

**Notation:** We'll drop subscripts unless necessary. Different random variables and events will be distinguished by different notations for their PDFs.

### 2.2 Joint Distributions and Conditioning

**Joint distribution:**
$$
f(x_1, \ldots, x_N) = \mathbb{P}(\bar{X}_1 = x_1, \ldots, \bar{X}_N = x_N)
$$

**Marginalization** (rule of total probability):
$$
f(\bar{X} = x) = \sum_y f(\bar{X} = x, \bar{Y} = y)
$$
or in continuous case: $f(x) = \int f(x, y) \, dy$

**Conditional distribution:**
$$
f(\bar{Y} = y \mid \bar{X} = x) = \frac{f(\bar{X} = x, \bar{Y} = y)}{f(\bar{X} = x)}
$$

**Product rule:**
$$
f(x, y) = f(x \mid y) \cdot f(y)
$$

**Probability chain rule:**
$$
\begin{aligned}
f(x_1, \ldots, x_N) &= f(x_2, \ldots, x_N \mid x_1) f(x_1) \\
&= f(x_3, \ldots, x_N \mid x_1, x_2) f(x_2 \mid x_1) f(x_1) \\
&\vdots \\
&= f(x_N \mid x_1, \ldots, x_{N-1}) \cdots f(x_2 \mid x_1) f(x_1)
\end{aligned}
$$

### 2.3 Independence

**Marginal independence:** $\bar{X} \perp \bar{Y}$ if:
$$
f(x, y) = f(x) f(y)
$$

In general for $N$ variables:
$$
f(x_1, \ldots, x_N) = \prod_{i=1}^N f(x_i)
$$

**Conditional independence:** $\bar{X} \perp \bar{Y} \mid \bar{Z}$ if:
$$
f(x, y \mid z) = f(x \mid z) f(y \mid z)
$$

---

## 3. Fitting Distributions to Data with Maximum Likelihood

### 3.1 Maximum Likelihood Estimation (MLE)

Given a dataset consisting of observations of a random variable $\bar{X}$:

$$
\mathcal{D} = \{x_i\}_{i=1}^{N_{\text{data}}}
$$

**Define the likelihood** by evaluating the parameterized joint distribution:
$$
\mathcal{L}(\theta) = f(x_1, \ldots, x_{N_{\text{data}}} \mid \theta)
$$

**Fit the distribution to data** by solving:
$$
\theta^* = \arg\min_\theta \underbrace{-\log \mathcal{L}(x_1, \ldots, x_{N_{\text{data}}} \mid \theta)}_{\text{NLL (Negative Log-Likelihood)}}
$$

**Why negative log-likelihood?**
- Converts products to sums (easier optimization)
- Avoids numerical underflow from small probabilities
- Equivalent to minimizing KL divergence from data distribution

### 3.2 Example: Fitting a Gaussian Distribution

**Step 1:** Assume independent data:
$$
f(x_1, \ldots, x_N \mid \theta) = \prod_{i=1}^N f(x_i \mid \theta)
$$

**Step 2:** Choose marginal distribution as Gaussian:
$$
f(x; \theta) = \mathcal{N}(x; \mu, \sigma^2) \quad \text{where} \quad \theta = \{\mu, \sigma^2\}
$$

$$
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left[-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right]
$$

**Step 3:** Compute log-likelihood:
$$
\log \mathcal{N} = -C - \frac{1}{2} \log \sigma^2 - \frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2
$$

**Step 4:** Write negative log-likelihood:
$$
\text{NLL} = \sum_{i=1}^N \frac{1}{2} \log \sigma^2 + \frac{1}{2}\left(\frac{x_i - \mu}{\sigma}\right)^2
$$

**Step 5:** Solve for $\mu^*$:
$$
\begin{aligned}
0 &= \nabla_\mu \text{NLL} = \sum_{i=1}^N \nabla_\mu \frac{1}{2}\left(\frac{x_i - \mu}{\sigma}\right)^2 \\
0 &= -\sum_{i=1}^N (x_i - \mu) \\
N\mu &= \sum_{i=1}^N x_i
\end{aligned}
$$

$$
\boxed{\mu^* = \frac{1}{N}\sum_{i=1}^N x_i}
$$

**Similarly for $\sigma^2$:**
$$
\boxed{\sigma^{2*} = \frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}
$$

**Result:** MLE recovers the sample mean and sample variance.

---

## 4. Stochastic Differential Equations

### 4.1 From ODEs to SDEs

To account for random forcing/physics, we'll expand our notion of ODEs.

**Standard ODE notation:**
$$
\frac{dx}{dt} = f(x, t)
$$

**Integral form:**
$$
\int_0^t dx = \int_0^t f(x, \tau) \, d\tau
$$

**Shorthand notation** (omitting limits):
$$
dx_t = f(x_t, t) \, dt \quad \text{where} \quad x_t = x(t)
$$

**Critical observation:** We will see that random forcing leads to solutions which **aren't differentiable**, so it's necessary to interpret them in the integral sense.

### 4.2 SDE Formulation

To account for stochastic terms, we consider an **SDE**:

$$
dx_t = \underbrace{f(x_t, t)}}_{\text{drift}} \, dt + \underbrace{g(x_t, t) \, dW_t}_{\text{diffusion (stochastic process)}}
$$

where:
- $f(x_t, t)$ is the **drift** (deterministic part)
- $g(x_t, t)$ is the **diffusion coefficient**
- $W_t$ is a **Wiener process** (Brownian motion)

**We need to define** $dW_t$ and how to integrate it.

**Important Note:** In general, we could spend a whole course studying stochastic processes—we'll give an accelerated version here.

---

## 5. The Wiener Process

### 5.1 Definition

We'll consider the **Wiener process** $W_t = \int_0^t dW_\tau$ defined via:

**(1) Initial condition:** $W_0 = 0$

**(2) Independent increments:** For $t > 0, u \geq 0$, the increment $W_{t+u} - W_t$ is independent of $W_s$ for any $s \leq t$

**(3) Gaussian increments** with variance equal to time increment:
$$
W_{t+u} - W_t \sim \mathcal{N}(0, u)
$$

**(4) Continuity:** $W_t$ is continuous (almost surely)

### 5.2 Construction

Many different constructions of $W_t$ satisfy these properties.

**Example construction:** Let $\xi_1, \xi_2, \ldots$ be IID random variables with $\mathbb{E}[\xi_i] = 0$ and $\operatorname{Var}[\xi_i] = 1$. Define:

$$
W_n(t) = \frac{1}{\sqrt{n}} \sum_{1 \leq k \leq nt} \xi_k
$$

Then by the **central limit theorem**:
$$
\lim_{n \to \infty} W_n(t) = W_t
$$

### 5.3 Properties

We can prove that with probability 1:

**(1)** $W_t$ is **continuous**

**(2)** $W_t$ is **nowhere differentiable**

**This non-differentiability poses challenges for standard calculus!**

---

## 6. Integrating Against the Wiener Process

### 6.1 Riemann-Like Construction

To integrate against the Wiener process, consider the Riemann-like sum:

$$
\int_0^t g(x_\tau, \tau) \, dW_\tau = \lim_{\Delta t \to 0} \sum_{i=0}^{n-1} g(x_{t_i}, t_i) (W_{t_{i+1}} - W_{t_i})
$$

**Problem:** This leads to many **pathological issues** that violate usual assumptions from standard calculus.

In a probability class, we would get into details about **Itô calculus** vs **Stratonovich calculus**—but this intuition is enough to pose a simple scheme for solving SDEs in our learning problems.

---

## 7. Euler-Maruyama Method

### 7.1 Discretization Scheme

Given the SDE:
$$
dx_t = f(x_t, t) \, dt + g(x_t, t) \, dW_t
$$

**Solve for** $x_{t_n} = x(t = nk)$ for $n = 0, 1, 2, \ldots$:

$$
\boxed{x_{n+1} = x_n + k f(x_{t_n}, t_n) + \xi_n g(x_{t_n}, t_n)}
$$

where:
$$
\xi_n \sim \mathcal{N}(0, k)
$$

**Note:** Some better integrators exist (see **Milstein's method**) but require a deeper dive into stochastic calculus.

---

## 8. Markov Processes and MLE for SDEs

### 8.1 Markov Process Definition

**Definition:** For a $K$-th order **Markov process**, given a discrete time series $\vec{y} = \langle y_0, y_1, \ldots, y_N \rangle$:

$$
P(y_i \mid y_0, \ldots, y_{i-1}) = P(y_i \mid y_{i-1}, \ldots, y_{i-K})
$$

**This means you only need to model the last $K$ timesteps** → this is like a multi-step integrator with $K$ steps.

**Euler-Maruyama is a Markov process with $K = 1$:**

$$
P(x_{n+1} \mid x_n) = \mathcal{N}\left(x_n + k f(x_n, t_n), \, k g(x_n, t_n)^2\right)
$$

### 8.2 Proof of Transition Distribution

**Claim:** If $X \sim \mathcal{N}(\mu, \sigma^2)$, then:
$$
AX + b \sim \mathcal{N}(A\mu + b, \, A^2 \sigma^2)
$$

**For Euler-Maruyama:**
$$
x_{n+1} = x_n + kf_n + g_n \xi_n
$$

where $\xi_n \sim \mathcal{N}(0, k)$.

**Identify:**
- $b = x_n + kf_n$
- $A = g_n$
- $\sigma^2 = k$

Therefore:
$$
x_{n+1} \sim \mathcal{N}(x_n + kf_n, \, k g_n^2) \quad \checkmark
$$

### 8.3 Deriving the NLL for Trainable SDEs

**Step 1:** Apply chain rule using Markov property:
$$
\begin{aligned}
-\log p(y_1, \ldots, y_N) &= -\log \left[ p(y_N \mid y_{N-1}) p(y_{N-1} \mid y_{N-2}) \cdots p(y_1 \mid y_0) \right] \\
&= -\log \prod_{i=1}^N p(y_i \mid y_{i-1}) \\
&= -\sum_{i=1}^N \log p(y_i \mid y_{i-1})
\end{aligned}
$$

**Step 2:** Substitute Gaussian transition probabilities:
$$
= -\sum_{i=1}^N \log \mathcal{N}(x_i + kf_i, \, kg_i^2)
$$

**Step 3:** Expand log-Gaussian:
$$
\text{NLL} = C + \sum_{i=1}^N \frac{1}{2} \log(kg_i^2) + \frac{1}{2} \frac{(x_{i+1} - x_i - kf_i)^2}{kg_i^2}
$$

### 8.4 Learning SDEs from Data

If we replace $f(x, t)$ with $f(x, t; \theta)$ and $g(x, t)$ with $g(x, t; \theta)$ (i.e., **make the SDE trainable**), we can solve:

$$
\boxed{\min_\theta \sum_{i=1}^N \log(kg_{i,\theta}^2) + \frac{(x_{i+1} - x_i - kf_{i,\theta})^2}{kg_{i,\theta}^2}}
$$

**This provides a principled way to learn stochastic dynamics from noisy time series data!**

---

## 9. Example: Structure-Preserving SDEs

### 9.1 Metriplectic / GENERIC Formalism

The **metriplectic** (also called **GENERIC**: General Equation for Non-Equilibrium Reversible-Irreversible Coupling) formalism provides a structure-preserving SDE framework:

$$
\boxed{dx_t = \left(L \partial_x E + M \partial_x S + K_B \partial_x \cdot M\right) dt + \sqrt{2K_B M} \, dW_t}
$$

where:
- $E$ = **energy** (conserved functional)
- $S$ = **entropy** (non-decreasing functional)
- $L = -L^T$ = **Poisson operator** (skew-symmetric)
- $M = M^T \geq 0$ = **friction operator** (symmetric, positive semi-definite)
- $K_B$ = **Boltzmann constant**

**Degeneracy conditions:**
$$
\begin{aligned}
L \partial_x S &= 0 \\
M \partial_x E &= 0
\end{aligned}
$$

**Physical interpretation:**
- When "coarse-graining" a reversible system, the loss of information manifests as **entropy increase** (dissipation)
- To counter this, **stochastic forcing** does work on the system
- The exact **fluctuation-dissipation theorem** guarantees this balance holds rigorously

### 9.2 Deterministic Limit ($K_B \to 0$)

In the deterministic limit, we have:
$$
dx_t = \left(L \partial_x E + M \partial_x S\right) dt
$$

#### Theorem 1: Energy is Conserved

**Proof:**
$$
\begin{aligned}
\frac{dE}{dt} &= \partial_x E^\top \frac{dx}{dt} \\
&= \partial_x E^\top (L \partial_x E + M \partial_x S) \\
&= \underbrace{\partial_x E^\top L \partial_x E}_{= 0 \text{ (skew-symmetry)}} + \underbrace{\partial_x E^\top M \partial_x S}_{= 0 \text{ (degeneracy)}} \\
&= 0
\end{aligned}
$$

Therefore: $\frac{dE}{dt} = 0$ $\quad \checkmark$

#### Theorem 2: Entropy is Non-Decreasing

**Proof:**
$$
\begin{aligned}
\frac{dS}{dt} &= \partial_x S^\top \frac{dx}{dt} \\
&= \partial_x S^\top (L \partial_x E + M \partial_x S) \\
&= \underbrace{\partial_x S^\top L \partial_x E}_{= 0 \text{ (degeneracy)}} + \underbrace{\partial_x S^\top M \partial_x S}_{\geq 0 \text{ (positive semi-definite)}} \\
&\geq 0
\end{aligned}
$$

Therefore: $\frac{dS}{dt} \geq 0$ $\quad \checkmark$

**Remark:** Metriplectic structure is therefore a **generalization of Hamiltonian mechanics**, with the degeneracy conditions preventing cross-terms between reversible ($L$) and irreversible ($M$) parts. This allows simultaneous treatment of conservative and dissipative dynamics in a geometrically consistent framework.

---

## 10. Summary

This lecture covered:

1. **Motivation** for probabilistic dynamics: tracking distributions, handling uncertainty
2. **Probability review**: random variables, distributions, conditioning, independence
3. **Maximum likelihood estimation (MLE)** for fitting distributions to data
4. **Stochastic differential equations (SDEs)** with drift and diffusion
5. **Wiener process** definition and properties (continuous but nowhere differentiable)
6. **Euler-Maruyama method** for discretizing SDEs
7. **Markov processes** and deriving NLL loss for learning SDEs from data
8. **Metriplectic/GENERIC formalism** as structure-preserving SDEs with energy conservation and entropy increase

**Key Takeaway:** Stochastic differential equations provide a principled framework for incorporating uncertainty into physics-informed learning. By combining the Euler-Maruyama discretization with maximum likelihood estimation, we can learn both drift and diffusion terms from noisy time series data. The metriplectic formalism demonstrates that even in the presence of irreversibility and dissipation, we can maintain geometric structure through careful construction of the Poisson and friction operators, ensuring energy conservation and entropy increase simultaneously—a beautiful unification of Hamiltonian mechanics and thermodynamics.

