# Lecture 20: Denoising Diffusion Probabilistic Models

**Reference:** Lecture 20 - Diffusion Models and Physical Priors

**Topics Covered:**
- ELBO review and computational tractability
- Denoising diffusion forward process
- Gaussian noise addition and limiting behavior
- Reverse process as learned denoising
- Justification of Gaussian reverse transitions
- Practical implementation and extensions

---

## 0. Overview

This final lecture brings together probabilistic modeling, variational inference, and physics-inspired design principles to understand **denoising diffusion probabilistic models (DDPMs)**—one of the most successful generative modeling frameworks in modern machine learning. Having developed the ELBO framework in the previous lecture, we now see how a carefully designed **forward diffusion process** leads to a tractable and powerful generative model.

The key insight is deceptively simple: instead of learning a direct mapping from data $x_0$ to a latent code $z$ (as in VAEs), we gradually **add noise** over many timesteps until the data becomes pure Gaussian noise. By learning to **reverse this process**—denoising one step at a time—we obtain a generative model that can sample from the data distribution by starting with Gaussian noise and iteratively removing it.

What makes this approach work mathematically is that (1) the forward process has closed-form Gaussians at each step, (2) the reverse process can be approximated by Gaussians for small noise levels, and (3) the ELBO decomposes into a sum of KL-divergences between these Gaussians, giving tractable training objectives. The result is a model that combines the expressiveness of neural networks with the stability of physical diffusion processes.

We begin by reviewing the ELBO and design criteria, develop the forward diffusion process with proofs of limiting behavior, construct the reverse process, and conclude with practical considerations and research directions. This lecture demonstrates how physical intuition (diffusion equations) can inspire machine learning architectures that are both principled and effective.

## 1. Review: ELBO and Design Goals

### 1.1 ELBO Recap

Recall the **Evidence Lower Bound** from last time:

$$
\mathcal{L} = -\mathbb{E}_{z \sim q}[\log p(x | z)] - \text{KL}(q(z | x) \| p(z))
$$

We discussed Kingma & Welling's **"vanilla" VAE** with:

1. **Single-sample Monte Carlo estimation:**

$$
\mathbb{E}_{z \sim q}[\log p(x | z)] \approx \log p(x_d | z_d)
$$

2. **Standard Gaussian prior:**

$$
p(z) = \mathcal{N}(\mu = 0, \Sigma = I)
$$

This allowed **generative modeling**:
1. Sample $z \sim \mathcal{N}(0, I)$
2. Decode using $p(x | z)$

### 1.2 Design Goals

There is a **craft to designing architectures** around different choices of:
- Embeddings $z$
- Priors $p(z)$

**GOAL: Computationally tractable**

**Two examples:**
1. **Denoising Diffusion** ← ELBO with Gaussians all the way!
2. **Physical Priors** ← Problem-specific structure

**Pedagogical note:** The key is choosing priors and transition distributions that are both expressive and have closed-form ELBO terms.

## 2. Denoising Diffusion

### 2.1 Core Idea

**Reference:** Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"

**Idea:** Don't jump from $x$ to $z$ in one step. Instead, **decode in increments**.

**Vanilla VAE:**

$$
x \to z \to x
$$

**Diffusion model:**

$$
x^0 \to x^1 \to x^2 \to \cdots \to x^T
$$

where $x^T \approx \mathcal{N}(0, I)$ (pure noise).

### 2.2 Forward Process

**Definition:** The **forward diffusion process** adds small Gaussian noise at each step:

$$
q(x^t | x^{t-1}) = \mathcal{N}(x^t; \sqrt{1 - \beta_t} x^{t-1}, \beta_t I)
$$

**Reparameterization:**

$$
x^t = \sqrt{1 - \beta_t} x^{t-1} + \sqrt{\beta_t} \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I)
$$

where $0 < \beta_t \ll 1$ is a small **noise schedule**.

### 2.3 Multi-Step Composition

**Claim:** We can compute $q(x^t | x^0)$ directly without iterating through intermediate steps.

**Step 1:** Consider two steps:

$$
\begin{aligned}
x^t &= \sqrt{1 - \beta_t} x^{t-1} + \sqrt{\beta_t} \varepsilon_t \\
x^{t+1} &= \underbrace{\sqrt{1 - \beta_{t+1}} \sqrt{1 - \beta_t}}_{\text{products}} x^{t-1} + \underbrace{\sqrt{\beta_{t+1} \beta_t} \varepsilon_t + \sqrt{\beta_{t+1}} \varepsilon_{t+1}}_{\text{additive Gaussian noise}}
\end{aligned}
$$

**Step 2:** Define:

$$
\begin{aligned}
\alpha_s &= 1 - \beta_s \\
\bar{\alpha}_t &= \prod_{s=1}^t \alpha_s
\end{aligned}
$$

**Step 3:** By induction:

$$
x^t = \sqrt{\bar{\alpha}_t} x^0 + \sum_{s=1}^t \sqrt{\bar{\alpha}_{s-1} \beta_s} \varepsilon_s
$$

**Step 4:** For Gaussians, **variance of sum is sum of variances**:

$$
x^t \sim \mathcal{N}\left( \sqrt{\bar{\alpha}_t} x^0, \sum_{s=1}^t \bar{\alpha}_{s-1} \beta_s I \right)
$$

### 2.4 Variance Simplification

**Lemma:**

$$
\sum_{s=1}^t \bar{\alpha}_{s-1} \beta_s = 1 - \bar{\alpha}_t
$$

**Proof:**

$$
\begin{aligned}
\bar{\alpha}_s &= \bar{\alpha}_{s-1} \alpha_s = \bar{\alpha}_{s-1}(1 - \beta_s) \\
&\Rightarrow \bar{\alpha}_{s-1} \beta_s = \bar{\alpha}_{s-1} - \bar{\alpha}_s
\end{aligned}
$$

So:

$$
\sum_{s=1}^t \bar{\alpha}_{s-1} \beta_s = \sum_{s=1}^t (\bar{\alpha}_{s-1} - \bar{\alpha}_s) = \bar{\alpha}_0 - \bar{\alpha}_t = 1 - \bar{\alpha}_t
$$

**Result:**

$$
q(x^t | x^0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x^0, (1 - \bar{\alpha}_t) I)
$$

### 2.5 Limiting Behavior

**Claim:** As $t \to \infty$, $x^t \to \mathcal{N}(0, I)$ (pure noise).

**Proof:**

Let $0 < \beta_{\min} \leq \beta_s$ for all $s$.

$$
\begin{aligned}
\lim_{t \to \infty} |\bar{\alpha}_t| &= \lim_{t \to \infty} \left| \prod_{s=1}^t (1 - \beta_s) \right| \\
&\leq \lim_{t \to \infty} \left| \prod_{s=1}^t (1 - \beta_{\min}) \right| \\
&= \lim_{t \to \infty} (1 - \beta_{\min})^t \\
&= 0
\end{aligned}
$$

Therefore:

$$
\lim_{t \to \infty} \mathcal{N}(\sqrt{\bar{\alpha}_t} x^0, (1 - \bar{\alpha}_t) I) = \mathcal{N}(0, I)
$$

**Critical observation:** The forward process **destroys all information** about $x^0$ in the limit, regardless of the initial data distribution!

## 3. Reverse Process

### 3.1 Forward and Reverse Ingredients

**Ingredient 1 (Forward):**

$$
q(x^0, \ldots, x^T) = q(x^0) \prod_{t=1}^T q(x^t | x^{t-1})
$$

where each $q(x^t | x^{t-1})$ is Gaussian.

**Ingredient 2 (Reverse):**

$$
p(x^0, \ldots, x^T) = p(x^T) \prod_{t=1}^T p(x^{t-1} | x^t)
$$

**Note:** We're going **backward in time** now!

**Prior distribution:** $p(x^T) = \mathcal{N}(0, I)$

### 3.2 Gaussian Reverse Transitions

**Claim:** For small $\beta_t$, $p(x^{t-1} | x^t)$ is approximately Gaussian.

**Parameterization:**

$$
p(x^{t-1} | x^t) = \mathcal{N}(\mu = f(x^t; \theta_f), \Sigma = g(x^t; \theta_g))
$$

where $f$ and $g$ are neural networks.

**We will prove this as justification for the parameterization.**

### 3.3 Justification via Bayes' Rule

**Step 1:** By Bayes' theorem:

$$
q(x^{t-1} | x^t, x^0) = \frac{q(x^t | x^{t-1}, x^0) q(x^{t-1} | x^0)}{q(x^t | x^0)}
$$

**Step 2:** From the forward process:

$$
q(x^t | x^{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} x^{t-1}, \beta_t I)
$$

**Step 3:** And we derived:

$$
q(x^{t-1} | x^0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} x^0, (1 - \bar{\alpha}_{t-1}) I)
$$

**Step 4:** **Product of Gaussians is an unnormalized Gaussian**. For some $\mu^*, \Sigma^*, C$:

$$
q(x^t | x^{t-1}) q(x^{t-1} | x^0) \sim C \mathcal{N}(\mu^*, \Sigma^*)
$$

**Step 5:** Skipping algebra details:

$$
q(x^{t-1} | x^t, x^0) = \mathcal{N}(\mu_t^*(x^0, x^t), \beta_t^* I)
$$

where:

$$
\begin{aligned}
\mu_t^* &= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x^0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x^t \\
\beta_t^* &= \frac{(1 - \bar{\alpha}_{t-1}) \beta_t}{1 - \bar{\alpha}_t}
\end{aligned}
$$

### 3.4 Removing Conditioning on $x^0$

**Question:** How do we approximate $q(x^{t-1} | x^t)$ without conditioning on $x^0$?

**Use the laws of total expectation and variance:**

$$
\begin{aligned}
\mathbb{E}[X] &= \mathbb{E}[\mathbb{E}[X | Y]] \\
\text{Var}[X] &= \mathbb{E}[\text{Var}[X | Y]] + \text{Var}[\mathbb{E}[X | Y]]
\end{aligned}
$$

**On the mean:**

$$
\begin{aligned}
\mathbb{E}[x^{t-1} | x^t] &= \mathbb{E}[\mathbb{E}[x^{t-1} | x^t, x^0]] \\
&= \mathbb{E}[\mu_t^*] \sim O(\beta_t^2)
\end{aligned}
$$

**On the variance:**

$$
\begin{aligned}
\text{Var}[x^{t-1} | x^t] &= \mathbb{E}[\text{Var}(x^{t-1} | x^t, x^0)] + \text{Var}[\mathbb{E}[x^{t-1} | x^t, x^0]] \\
&= \underbrace{\mathbb{E}[\beta_t^* I]}_{O(\beta_t^2)} + \underbrace{\text{Var}[\mu_t^*]}_{O(\beta_t)}
\end{aligned}
$$

**Critical observation:** For small $\beta_t$, the conditional mean and variance are approximately constant, justifying the Gaussian approximation!

## 4. The Diffusion ELBO

### 4.1 ELBO Decomposition

**Ingredient 3 (ELBO):**

$$
\begin{aligned}
\mathcal{L} = &\mathbb{E}_{q(x^T | x^0)}[\log p(x^0 | x^T)] \\
&- \sum_{t=2}^T \mathbb{E}_{q(x^t | x^0)}\left[ \text{KL}(q(x^{t-1} | x^t, x^0) \| p(x^{t-1} | x^t)) \right]
\end{aligned}
$$

**Interpretation:**
- **Learn to denoise single diffusion steps** (minimize KL at each timestep)
- **Generate by sampling a Gaussian and denoising** (reverse the forward process)

### 4.2 Key Advantages

**Some takeaways:**

**Designed an encoding which:**
- Required **no learning** (forward process is fixed)
- Took us to a **latent Gaussian** (limiting distribution)
- Has **Gaussian increments** (closed-form distributions)

**Designed a decoding which:**
- Is **indirectly supervised** by encoding (matching forward transitions)
- Gives **Gaussian increments** (tractable reverse steps)

**When combined in ELBO:**
- **Lots of closed-form expressions for KL divergences** (between Gaussians)

## 5. Practical Considerations and Extensions

### 5.1 How to Use in Research

**How much math do you need to track?**

**Level 1 (Off-the-shelf):**
- Just use one of the implementations
- Understand that if you mess with the architecture, you may break the theory
- Maybe leave ELBO loss alone
- Do play with architectures, $\beta$ scheduler
- Display with input/output of denoising steps

**Level 2 (Improve):**
- Many other physical models can give $\lim_{t \to \infty} x^t \sim \mathcal{N}(0, I)$
- **Replace noising with something physical** (e.g., heat equation, transport)
- Think about challenges and requirements

### 5.2 Research Directions

**Challenges:**
- **Expensive to generate** – How can we use shorter $T$?
- What other **SDEs** allow Gaussian reverse processes?
- Can we extend to **SPDEs** (stochastic PDEs)?

**Opportunities:**
- Replace Gaussian noise with **physically motivated** noise (e.g., turbulence, molecular dynamics)
- Use problem-specific structure (e.g., symmetries, conservation laws)
- Combine with **physics-informed neural networks** for constrained generation

**Critical observation:** The diffusion framework is flexible—any forward process that converges to a tractable prior can potentially be inverted via learned reverse transitions.

---

## Summary

This lecture covered:

1. **ELBO review** and design goals for tractability
2. **Forward diffusion process** with Gaussian noise addition
3. **Closed-form marginals** $q(x^t | x^0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x^0, (1 - \bar{\alpha}_t) I)$
4. **Limiting behavior** showing $x^T \to \mathcal{N}(0, I)$
5. **Reverse process** parameterized as learned Gaussian transitions
6. **Justification** via Bayes' rule and product of Gaussians
7. **Diffusion ELBO** as sum of KL-divergences between Gaussians
8. **Practical considerations** and research directions

**Key Takeaway:** Denoising diffusion probabilistic models (DDPMs) provide a principled framework for generative modeling by gradually corrupting data with Gaussian noise (forward process) and learning to reverse this corruption (reverse process). The mathematical foundation rests on three pillars: (1) the forward process $q(x^t | x^{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x^{t-1}, \beta_t I)$ has closed-form marginals $q(x^t | x^0)$, making training efficient; (2) the reverse process $p(x^{t-1} | x^t)$ can be approximated by Gaussians for small $\beta_t$, justified via Bayes' rule; and (3) the ELBO decomposes into tractable KL-divergences between Gaussians. This design enables stable training, high-quality generation, and natural incorporation of physical priors through the choice of forward process. The framework demonstrates how continuous-time diffusion processes can inspire discrete-time machine learning architectures that are both theoretically grounded and practically effective.
