Today: Generative Modeling + Variatemal Inference

- How to solve for "latent" physics?
- How to use probability to perform generative modeling
![](https://cdn.mathpix.com/cropped/f89e741b-6f12-42f1-b7f3-12b7fc253fbf-01.jpg?height=462&width=1766&top_left_y=1348&top_left_x=22)

Idea Use an identity map to sample from the data distribution

Toduy VAE → Next Differion models

Variational Inference

A new kind of probabilistic variational method to optimize over proh. d.st. instead of mechnics

$\frac{\text { Some probability basics }}{\text { For cont } R V \text { in } \mathbb{R}}$

- probability derity function

Given curt. RV $\bar{X}$, the prob $\bar{X}$ takes a value $x \in[a, b]$

$$
\begin{aligned}
& \mathbb{P}(a \leq \bar{x} \leq b)=\int_{a}^{b} p \\
& \text { for } \quad p(x) \geq 0 \\
& \int_{-\infty}^{\infty} p(x)=1
\end{aligned}
$$

- expectation

$$
\mathbb{E}_{p}[f(x)]=\int_{-\infty}^{\infty} f(x) \rho(x) d x
$$

Sometimes doep if we are ouly talking alout a specifie dist.

Ex Multivariate Gowian (our breadt hutter)
Given $\vec{x} \in \mathbb{R}^{d}, \quad \mu \in \mathbb{R}^{d}, \quad \Sigma \in \mathbb{R}^{d \times d}$ invetible eRD

$$
\begin{aligned}
& P(x)=(2 \pi)^{-d / 2}|\Sigma|^{-1 / 2} \exp \left(-\frac{1}{2}(x-\mu)^{\top} \Sigma^{-1}(x-\mu)\right) \\
& \bar{X} \sim N(x ; \mu, \Sigma)
\end{aligned}
$$

Causions are life the piocervise polynomials of probability

- easy to work with
def Joint, morginal, conditional diatubutions
Joint $p(x, z)$ - (prob of $x \wedge z)$
Marginn $p(x)=\quad \int p(x, z) d z$
laccount for all prskible values of $Z 1$
conditional $p(z \mid x)=\frac{p(x, z)}{p(x)}$ if $p(x)>0$
(prol of $z$ if $x$ happeed)


## Bayer theorem

$$
\begin{aligned}
& p(x, z)=p(x \mid z) p(z)=p(z \mid x) p(x) \\
& \Rightarrow \quad p(x \mid z)=\frac{p(z \mid x) p(x)}{p(z)}
\end{aligned}
$$

use this to flip "infut" / "output" reludaing?
Finally, a "pado-metric on destribetions
KL-divergence

$$
K L(q \| p)=\int q(x) \log \left(\frac{q(x)}{p(x)}\right) d x
$$

- Nate $K L(q \| \rho) \neq K L\left(p \|_{q}\right)$
- Positive

For Cavaiains $q \sim N\left(\mu_{q}, \varepsilon_{q}\right), p \sim N\left(\mu_{p}, \varepsilon_{p}\right)$

$$
\begin{aligned}
K L(q u p)=\frac{1}{2}\left[\log \frac{\left|\Sigma_{p}\right|}{\left|\Sigma_{q}\right|}-d+\right. & \operatorname{tr}\left(\Sigma_{p}^{-1} \Sigma_{q}\right)+ \\
& \left.\left(\mu_{p}-\mu_{q}\right)^{\top} \Sigma_{p}^{-1}\left(\mu_{p}-\mu_{q}\right)\right]
\end{aligned}
$$

## Jensen's Fuequality

Let $\phi$ be a convex function

$$
\left\{\begin{array}{l}
\forall t \in[0,1], x, y \\
\phi(t x+(1-t) y) \leq t \phi(x)+(1-t) \phi(y)
\end{array}\right.
$$

Jensen $\phi \mathbb{E}[x] \leq \mathbb{E}[\phi(x)]$

To sample foour data distribution

$$
p(z \mid x)=\frac{p(x \mid z) p(z)}{p(x)}=\underbrace{\int p(x \mid z) p(z) d z}_{\substack{\text { Comput troun lly } \\ \text { Intractable }}}
$$

Typieally, we would do MLE i.e. pose a joint dist and solve

$$
\min _{\theta}-\log p_{\theta}(x, z)
$$

Inatend, build an objective that accounts for any possible dist. on $z$

Marginal log likelihood

$$
\begin{aligned}
f_{\text {mal }} & =-\sum_{d} \log p\left(x_{d} ; \theta\right) \quad \text { marginalize } \\
& =-\sum_{d} \log \sum_{z_{d}} p\left(x_{d}, z_{d} ; \theta\right)
\end{aligned}
$$

Infroduce an arliting dist. on $Z, g(Z)$

Interport

$$
\begin{aligned}
& =-\sum_{d} \log \sum_{z_{d}} q\left(z_{d}\right) \frac{p\left(x_{d}, z_{d} ; \theta\right)}{q\left(z_{d}\right)} \\
& =-\sum_{d} \log \mathbb{E}_{z_{\sim} q}\left[\frac{p\left(x_{d}, z_{d} ; \theta\right)}{q\left(z_{d}\right)}\right]
\end{aligned}
$$

Jensan
inquality $\leq-\sum_{d} \mathbb{E}_{z_{\sim q}}\left[\log \frac{p\left(x_{d}, z_{d} ; \theta\right)}{q\left(z_{d}\right)}\right]$

$$
\begin{aligned}
& \varepsilon(\theta, q)=-\sum_{d} \mathbb{E}_{z \sim q}\left[\log p\left(x_{d}, z_{d} ; \theta\right)\right]-\underset{\substack{H\left(q_{d}\right) \\
\uparrow \\
\text { Eviduce }}}{\text { entropy }} \text { of } q
\end{aligned}
$$

Lowes
Bound (ELBO)

Note Don't need to plug $z$ 's in.

Since $\mathcal{L}_{M L L} \leq \varepsilon(\theta, q)$

We can choore $o$ to make $\varepsilon$ as close as posible to $\mathcal{L}_{\text {MLL }}$
Rewriting:

$$
\begin{aligned}
& \varepsilon(\theta q)=\sum_{d} \mathbb{E}_{z \sim \xi}\left[\log \left(\frac{p\left(z_{i} \mid x_{d} ; \theta\right) p\left(x_{d} ; \theta\right)}{q_{d}}\right)\right] \\
& =\sum_{d} \mathbb{E}_{z \sim q}\left[\log \frac{p\left(z_{d} \mid x_{d} ; \theta\right)}{q}\right]+\mathbb{E}_{z \sim q}\left[\log p\left(\mid x_{d} ; \theta\right)\right] \\
& =\sum_{d}-K L\left(p\left(z_{d} \mid x_{d} ; \theta\right) \| q_{d}\right)+
\end{aligned}
$$

Riggent when $K L=0$
Choose $\quad q_{d}=p\left(z_{d} \mid x_{d} ; \theta\right)$

Variatimal Autoencoder (UAE) Kingma. Wollity
![](https://cdn.mathpix.com/cropped/f89e741b-6f12-42f1-b7f3-12b7fc253fbf-09.jpg?height=577&width=950&top_left_y=476&top_left_x=221)

$$
\begin{aligned}
Z & =\mu+\sqrt{\Sigma} \varepsilon \\
\varepsilon & \sim N(0, I) \\
\Rightarrow Z & \sim N(\mu, \Sigma)
\end{aligned}
$$

$z$

$$
\text { decoder } P(x \mid z) \text { Mout } \quad P(x \mid Z)=N\left(M_{\text {out }}, I\right)
$$

$$
\varepsilon=\underbrace{\mathbb{E}[\log p(x \mid z)]}_{2}-K L(z(z \mid x) \| p(z))
$$

$$
C+\left\|X-\mu_{\text {out }}\right\|
$$

## Reconstruction Loss

## Prios penalty

Choices: → Architecture
→ Prior

## Some building blocks

def Categarial RV $\operatorname{crcat}(\pi)$

$$
\begin{aligned}
\pi_{i} & >0 \\
\sum_{i} \pi_{i} & =1
\end{aligned}
$$

Mixture of Experts (Jacobs, Joodan, Nowlan, Hinton 1991)
$\pi_{i}(x)=\operatorname{softmax}(N N(x))$
![](https://cdn.mathpix.com/cropped/f89e741b-6f12-42f1-b7f3-12b7fc253fbf-10.jpg?height=374&width=1005&top_left_y=1122&top_left_x=934)
$P(y)=\sum_{i} P(c=i) P(y \mid c=i)$

$$
y \mid C=N
$$

- A means to sparsely increase model param w/o increasing compute time
(iwitch Transformers: Saling to trillian param models Fedus 2022)

Lemma Product of Gavisinn PDFS is a Gauasian

$$
\begin{aligned}
& P_{1}=N\left(\mu_{1}, \sigma_{1}^{2}\right) \\
& P_{2}=N\left(\mu_{2}, \sigma_{2}^{2}\right) \\
& P_{1} \cdot P_{2}=\frac{1}{\sqrt{2 \pi \tilde{\sigma}^{2}}} \exp \left(-\frac{(x-\tilde{\mu})^{2}}{\tilde{\sigma}^{2}}\right) \\
& \frac{\mu}{\tilde{\sigma}^{2}} \frac{x}{\tilde{\sigma}^{2}}=\frac{\mu_{1}}{\sigma_{1}^{2}}+\frac{\mu_{2}}{\sigma_{2}^{2}} \\
& \tilde{\sigma}^{2}=\frac{1}{\frac{1}{\sigma_{1}^{2}}+\frac{1}{\sigma_{2}^{2}}}
\end{aligned}
$$

Product-of- expets
![](https://cdn.mathpix.com/cropped/f89e741b-6f12-42f1-b7f3-12b7fc253fbf-11.jpg?height=566&width=880&top_left_y=1957&top_left_x=365)

