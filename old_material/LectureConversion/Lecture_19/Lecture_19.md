# Lecture 19: Variational Inference and Variational Autoencoders

**Reference:** Lecture 19 - Generative Modeling via Variational Inference

**Topics Covered:**
- Probability basics: PDFs, expectations, Gaussians
- Joint, marginal, and conditional distributions
- Bayes' theorem and KL-divergence
- Jensen's inequality
- Evidence Lower Bound (ELBO)
- Variational Autoencoders (VAEs)
- Mixture of Experts and Product of Experts

---

## 0. Overview

This lecture introduces a fundamentally new type of variational method—one that operates on **probability distributions** rather than mechanical energies. Until now, our variational principles have focused on minimizing functionals like $\int |\nabla u|^2$ or maximizing stability conditions. Now we shift to **variational inference**, where we optimize over spaces of probability distributions to solve inverse problems and perform generative modeling.

The motivating question is: **How do we solve for "latent" physics?** In many applications, observations $x$ are generated from hidden variables $z$ through a process $p(x|z)$. Inverting this relationship—finding $p(z|x)$—is computationally intractable for complex models. Variational inference sidesteps this by introducing an approximate posterior $q(z|x)$ and optimizing it to be as close as possible to the true posterior.

The key mathematical tool is the **Evidence Lower Bound (ELBO)**, derived using Jensen's inequality. The ELBO provides a tractable objective that simultaneously reconstructs observations and regularizes the latent space. This framework underlies **Variational Autoencoders (VAEs)**, which learn to map data to a structured latent space (typically Gaussian) and decode back to the original space.

We begin with probability fundamentals, develop the ELBO through careful manipulations involving KL-divergence, and show how VAEs implement this as a neural architecture. Finally, we explore building blocks like **Mixture of Experts** and **Product of Experts** that extend the basic VAE framework to more complex generative models.

## 1. Probability Fundamentals

### 1.1 Probability Density Functions

For a continuous random variable $X$ in $\mathbb{R}$:

**Definition:** The **probability density function** $p(x)$ satisfies:

$$
\mathbb{P}(a \leq X \leq b) = \int_a^b p(x) \, dx
$$

with properties:

$$
\begin{aligned}
p(x) &\geq 0 \\
\int_{-\infty}^\infty p(x) \, dx &= 1
\end{aligned}
$$

### 1.2 Expectation

**Definition:** The **expectation** of a function $f$ under distribution $p$ is:

$$
\mathbb{E}_p[f(x)] = \int_{-\infty}^\infty f(x) p(x) \, dx
$$

Sometimes we drop the subscript if we are only talking about a specific distribution.

### 1.3 Multivariate Gaussians

**Example (our bread and butter):** Given $\vec{x} \in \mathbb{R}^d$, $\mu \in \mathbb{R}^d$, $\Sigma \in \mathbb{R}^{d \times d}$ invertible and positive semi-definite:

$$
p(x) = (2\pi)^{-d/2} |\Sigma|^{-1/2} \exp\left( -\frac{1}{2}(x - \mu)^\top \Sigma^{-1}(x - \mu) \right)
$$

**Notation:**

$$
X \sim \mathcal{N}(x; \mu, \Sigma)
$$

**Pedagogical note:** Gaussians are like the piecewise polynomials of probability—easy to work with, compose well, and sufficient for most applications.

## 2. Joint, Marginal, and Conditional Distributions

### 2.1 Definitions

**Joint distribution:** $p(x, z)$ is the probability of $x$ AND $z$.

**Marginal distribution:** $p(x) = \int p(x, z) \, dz$ accounts for all possible values of $z$.

**Conditional distribution:** $p(z | x) = \frac{p(x, z)}{p(x)}$ if $p(x) > 0$ (probability of $z$ if $x$ happened).

### 2.2 Bayes' Theorem

Since $p(x, z) = p(x | z) p(z) = p(z | x) p(x)$:

$$
p(x | z) = \frac{p(z | x) p(x)}{p(z)}
$$

**Use:** Flip "input"/"output" relationship!

**Critical observation:** This is the foundation of Bayesian inference—updating beliefs about latent variables $z$ given observations $x$.

## 3. KL-Divergence

### 3.1 Definition

Finally, a "pseudo-metric" on distributions:

**Definition:** The **KL-divergence** (Kullback-Leibler divergence) is:

$$
\text{KL}(q \| p) = \int q(x) \log\left( \frac{q(x)}{p(x)} \right) dx
$$

**Important properties:**
- **Not symmetric:** $\text{KL}(q \| p) \neq \text{KL}(p \| q)$
- **Always non-negative:** $\text{KL}(q \| p) \geq 0$ with equality if and only if $q = p$

### 3.2 KL-Divergence for Gaussians

For Gaussians $q \sim \mathcal{N}(\mu_q, \Sigma_q)$, $p \sim \mathcal{N}(\mu_p, \Sigma_p)$:

$$
\begin{aligned}
\text{KL}(q \| p) = \frac{1}{2} \Bigg[ &\log \frac{|\Sigma_p|}{|\Sigma_q|} - d + \text{tr}(\Sigma_p^{-1} \Sigma_q) \\
&+ (\mu_p - \mu_q)^\top \Sigma_p^{-1} (\mu_p - \mu_q) \Bigg]
\end{aligned}
$$

**Interpretation:** This closed form makes Gaussian VAEs computationally tractable!

## 4. Jensen's Inequality

### 4.1 Statement

Let $\phi$ be a **convex function**:

$$
\forall t \in [0,1], \, x, y: \quad \phi(tx + (1-t)y) \leq t\phi(x) + (1-t)\phi(y)
$$

**Jensen's Inequality:**

$$
\phi(\mathbb{E}[x]) \leq \mathbb{E}[\phi(x)]
$$

**Interpretation:** The function of the average is less than or equal to the average of the function (for convex $\phi$).

## 5. Evidence Lower Bound (ELBO)

### 5.1 Problem Setup

To sample from the data distribution:

$$
p(z | x) = \frac{p(x | z) p(z)}{p(x)} = \frac{p(x | z) p(z)}{\int p(x | z) p(z) \, dz}
$$

The denominator $p(x) = \int p(x | z) p(z) \, dz$ is **computationally intractable**.

Typically, we would do **MLE** (maximum likelihood estimation), i.e., pose a joint distribution and solve:

$$
\min_\theta -\log p_\theta(x, z)
$$

### 5.2 Deriving the ELBO

Instead, build an objective that accounts for any possible distribution on $z$.

**Marginal log likelihood:**

$$
\begin{aligned}
\mathcal{L}_{\text{MLL}} &= -\sum_d \log p(x_d; \theta) \\
&= -\sum_d \log \sum_{z_d} p(x_d, z_d; \theta)
\end{aligned}
$$

**Step 1:** Introduce an arbitrary distribution on $z$: $q(z)$:

$$
= -\sum_d \log \sum_{z_d} q(z_d) \frac{p(x_d, z_d; \theta)}{q(z_d)}
$$

**Step 2:** Rewrite as expectation:

$$
= -\sum_d \log \mathbb{E}_{z \sim q}\left[ \frac{p(x_d, z_d; \theta)}{q(z_d)} \right]
$$

**Step 3:** Apply Jensen's inequality ($\log$ is concave, so flip inequality):

$$
\leq -\sum_d \mathbb{E}_{z \sim q}\left[ \log \frac{p(x_d, z_d; \theta)}{q(z_d)} \right]
$$

**Step 4:** Define the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}(\theta, q) = -\sum_d \mathbb{E}_{z \sim q}[\log p(x_d, z_d; \theta)] + \underbrace{H(q_d)}_{\text{entropy of } q}
$$

where $H(q) = -\mathbb{E}_{z \sim q}[\log q(z)]$ is the entropy.

**Note:** Don't need to plug in actual $z$'s—we work with expectations!

Since $\mathcal{L}_{\text{MLL}} \leq \mathcal{L}(\theta, q)$, we can choose $q$ to make $\mathcal{L}$ as close as possible to $\mathcal{L}_{\text{MLL}}$.

### 5.3 Optimal Choice of $q$

**Rewriting:**

$$
\begin{aligned}
\mathcal{L}(\theta, q) &= \sum_d \mathbb{E}_{z \sim q}\left[ \log \frac{p(z_d | x_d; \theta) p(x_d; \theta)}{q_d} \right] \\
&= \sum_d \mathbb{E}_{z \sim q}\left[ \log \frac{p(z_d | x_d; \theta)}{q_d} \right] + \mathbb{E}_{z \sim q}[\log p(x_d; \theta)] \\
&= \sum_d -\text{KL}(q_d \| p(z_d | x_d; \theta)) + \log p(x_d; \theta)
\end{aligned}
$$

**Tightest bound when $\text{KL} = 0$:**

Choose:

$$
q_d = p(z_d | x_d; \theta)
$$

**Critical observation:** The ELBO becomes exact when the approximate posterior matches the true posterior!

## 6. Variational Autoencoder (VAE)

### 6.1 Architecture

**Reference:** Kingma & Welling, "Auto-Encoding Variational Bayes"

**Encoder:** $q(z | x) = \mathcal{N}(\mu(x), \Sigma(x))$ where $\mu, \Sigma$ are neural networks.

**Reparameterization trick:**

$$
\begin{aligned}
z &= \mu + \sqrt{\Sigma} \varepsilon \\
\varepsilon &\sim \mathcal{N}(0, I) \\
\Rightarrow z &\sim \mathcal{N}(\mu, \Sigma)
\end{aligned}
$$

**Decoder:** $p(x | z) = \mathcal{N}(\mu_{\text{out}}(z), I)$ where $\mu_{\text{out}}$ is a neural network.

### 6.2 Loss Function

$$
\mathcal{L} = \underbrace{\mathbb{E}[\log p(x | z)]}_{\text{reconstruction loss}} - \text{KL}(q(z | x) \| p(z))
$$

$$
\approx C + \|x - \mu_{\text{out}}\|^2 - \text{KL}(q(z|x) \| \mathcal{N}(0, I))
$$

**Two components:**

1. **Reconstruction loss:** Ensures decoded samples match original data
2. **Prior penalty:** Regularizes latent space to be approximately Gaussian

### 6.3 Design Choices

**Architecture choices:**
- Encoder/decoder network architectures (CNNs, transformers, etc.)
- Latent dimension
- Number of samples for Monte Carlo estimation

**Prior choices:**
- Standard Gaussian $p(z) = \mathcal{N}(0, I)$ (most common)
- Learned prior
- Structured prior (e.g., Mixture of Gaussians)

## 7. Building Blocks for Complex Models

### 7.1 Categorical Random Variables

**Definition:** A **categorical random variable** $c \sim \text{Cat}(\pi)$ satisfies:

$$
\begin{aligned}
\pi_i &> 0 \\
\sum_i \pi_i &= 1
\end{aligned}
$$

### 7.2 Mixture of Experts

**Reference:** Jacobs, Jordan, Nowlan, Hinton (1991)

**Idea:** Define:

$$
\pi_i(x) = \text{softmax}(\text{NN}(x))
$$

**Generative model:**

$$
p(y) = \sum_i p(c = i) p(y | c = i)
$$

where $y | c = i \sim \mathcal{N}(\mu_i, \Sigma_i)$.

**Interpretation:** A **means to sparsely increase model parameters without increasing compute time**.

**Reference:** "Switch Transformers: Scaling to Trillion Parameter Models" (Fedus et al., 2022)

### 7.3 Product of Experts

**Lemma:** The **product of Gaussian PDFs is a Gaussian**.

Given:

$$
\begin{aligned}
p_1 &= \mathcal{N}(\mu_1, \sigma_1^2) \\
p_2 &= \mathcal{N}(\mu_2, \sigma_2^2)
\end{aligned}
$$

Then:

$$
p_1 \cdot p_2 = \frac{1}{\sqrt{2\pi \tilde{\sigma}^2}} \exp\left( -\frac{(x - \tilde{\mu})^2}{2\tilde{\sigma}^2} \right)
$$

where:

$$
\begin{aligned}
\frac{\tilde{\mu}}{\tilde{\sigma}^2} &= \frac{\mu_1}{\sigma_1^2} + \frac{\mu_2}{\sigma_2^2} \\
\frac{1}{\tilde{\sigma}^2} &= \frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}
\end{aligned}
$$

**Interpretation:** Product of experts combines multiple Gaussian distributions by **precision-weighted averaging** of means and **adding precisions** (inverse variances). This provides a principled way to fuse information from multiple sources.

---

## Summary

This lecture covered:

1. **Probability fundamentals:** PDFs, expectations, Gaussians
2. **Joint, marginal, conditional distributions** and Bayes' theorem
3. **KL-divergence** as a measure of distributional similarity
4. **Jensen's inequality** for convex functions
5. **Evidence Lower Bound (ELBO)** derivation
6. **Variational Autoencoders (VAEs)** with reparameterization trick
7. **Loss function decomposition:** reconstruction + prior penalty
8. **Mixture of Experts** for scalable capacity
9. **Product of Experts** for information fusion

**Key Takeaway:** Variational inference reformulates intractable posterior inference $p(z|x)$ as an optimization problem over approximate posteriors $q(z|x)$. The Evidence Lower Bound (ELBO) provides a tractable objective: $\mathcal{L} = \mathbb{E}_{q}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$. This balances reconstruction accuracy (likelihood) with regularization (prior matching). VAEs implement this framework by parameterizing $q$ and $p$ as neural networks, using the reparameterization trick to enable gradient-based optimization. The result is a generative model that learns to encode data into a structured latent space (typically Gaussian) and decode back to the data space, enabling both generation (sample $z \sim p(z)$, decode) and representation learning (encode observations to meaningful latent codes).
