4/28- Probabilistic plugsics w/ variational inference

- Last day of class → wed 4/28

→ everyone ihould be making progess
→ Last chance for feedbacto
→ final presentations $5 / 9$ 3-5 in DRLB-A2
→ report due by 5/12
Final Course Feedhack

- Please he houest - first time class and feedhack will doastically reshape next iteration
→ More HW
→ Stricter ple-reqs

Recall ELBO from last time

$$
\varepsilon=-\mathbb{E}_{x \sim z}[\log p(x \mid z)]-K L(q(z \mid x) \| p(z))
$$

We discussed Kingma+ Welling "vanilla" VAE
singlersioptle
Monte Carlo
(1) $\mathbb{E}_{x \sim z}[\log P(x \mid z)] \approx \log P\left(x_{d} \mid z_{d}\right)$
(2) $p(z)=N(\mu=0, \Sigma=I)$

Which allowed generative modeling
(1) Sample $Z \sim N(0, I)$
(2) Decode using $P(x \mid z)$

These in a craft to derigning architectures around different choices of embeddings $z$, and prious

## GOAL

Computationally tractalle
Two examples

- Denoising Diffusion ELBO!
- Physical Prioss
${ }^{\text {↑ }}$ Gaussian Ganssian Goussian!

Devoising Daffusian
(Gohl-Dickstein)

Idea Don't jump from $x$ to $z$, instead decode in increments

Vanilla
![](https://cdn.mathpix.com/cropped/24ab066c-c906-4996-baa5-8a751a820200-02.jpg?height=184&width=616&top_left_y=2389&top_left_x=168)
![](https://cdn.mathpix.com/cropped/24ab066c-c906-4996-baa5-8a751a820200-02.jpg?height=664&width=1225&top_left_y=2086&top_left_x=900)

Pf To prove by induction, chect 2 steps

$$
\begin{aligned}
& x_{t}=\sqrt{1-\beta_{t}} x_{t-1}+\sqrt{\beta_{t}} \varepsilon_{t} \\
& x_{t+1}=\underbrace{\sqrt{1-\beta_{t+1}} \sqrt{1-\beta_{t}}}_{\text {products }} x_{t-1}+\underbrace{\sqrt{\beta_{t+1} \beta_{t}} \varepsilon_{t}+\sqrt{\beta_{t+1}} \varepsilon_{t+1}}_{\text {Additive gavioin noisc }} \\
& \text { w/ weights }
\end{aligned}
$$

Let $\quad \alpha_{s}=1-\beta_{s}$

$$
\bar{\alpha}_{t}=\prod_{s=1}^{t} \alpha_{s}
$$

By induction can show

$$
x_{t}=\sqrt{\bar{\alpha}_{t}} x_{0}+\sum_{s=1}^{t} g_{s} \sqrt{\bar{\alpha}_{s-1} \beta_{s}} \varepsilon_{s}
$$

For gaunians, variance of sum is sum of var

$$
X_{t} \sim N\left(\sqrt{\bar{\alpha}_{t}} x_{0}, \sum_{s=1}^{t} \bar{\alpha}_{s-1} \beta_{s} I\right)
$$

Leman

$$
\sum_{s=1}^{t} \bar{\alpha}_{s-1} \beta_{s}=1-\bar{\alpha}_{t}
$$

Pf

$$
\begin{aligned}
\bar{\alpha}_{s} & =\bar{\alpha}_{s-1} \alpha_{s} \\
& =\bar{\alpha}_{s-1}\left(1-\beta_{s}\right)
\end{aligned} \rightarrow \bar{\alpha}_{s-1} \beta_{s}=\bar{\alpha}_{s-1}-\bar{\alpha}_{s}
$$

So that

$$
\sum_{s} \bar{\alpha}_{s-1} \beta_{s}=\sum_{s} \bar{\alpha}_{s-1}-\bar{\alpha}_{s}=1-\bar{\alpha}_{t}
$$

Finally, take limits
Let $0<\beta_{s} \ll 1$

$$
\begin{aligned}
\lim _{t \rightarrow \infty}\left|\bar{\alpha}_{t}\right| & =\lim _{t \rightarrow \infty}\left|\prod_{s=1}^{t}\left(1-\beta_{s}\right)\right| \\
& \leq \lim _{t}\left|\prod_{s=1}^{t}\left(1-\beta_{\min }\right)\right| \\
& \leq \lim _{t}\left(1-\beta_{\min }\right)^{t} \\
& =0
\end{aligned}
$$

$\zeta_{0} \lim _{t \rightarrow \infty} N\left(\sqrt{\bar{\alpha}_{t}} \times 0, \sum_{s=1}^{t} \overline{\alpha_{s-1}} \beta_{s} I\right)$

$$
\begin{aligned}
& =\lim _{t \rightarrow \infty} N\left(\sqrt{\bar{\alpha}_{t}} x_{0}, 1-\overline{\alpha_{t}}\right) \\
& =N(0, I)
\end{aligned}
$$

Back to forward ingredient

$$
q\left(x_{1}^{0} \ldots x^{\top}\right)=q\left(x^{0}\right) \prod_{t=1}^{T} p\left(x^{t} \mid x^{t-1}\right)
$$

$q_{\text {gaussion! }}$

## Ingredient 2

$P\left(x^{\top}\right)=$ prior distibution

$$
P\left(x_{1}^{0} \ldots x^{\top}\right)=P\left(x^{\top}\right) \prod_{t=1}^{T} P\left(x_{1}^{t-1} \mid x^{t}\right)
$$

个 going $\frac{\text { hackward }}{\text { time }}$
in time now
Claim For somall $\beta, p\left(x^{t-1} / x^{t}\right)$ is Gaussion we will prove this as justification for the parameterization
$P\left(x^{t-1} / x^{t}\right)=N\left(M=f\left(x^{t} ; \theta_{f}\right), \quad \Sigma=\frac{g\left(x^{t} ; \theta_{g}\right)}{\lambda}\right)$
![](https://cdn.mathpix.com/cropped/24ab066c-c906-4996-baa5-8a751a820200-06.jpg?height=204&width=647&top_left_y=1433&top_left_x=1167)

Before we prove recap where we're at
![](https://cdn.mathpix.com/cropped/24ab066c-c906-4996-baa5-8a751a820200-06.jpg?height=721&width=1559&top_left_y=1735&top_left_x=494)

Final itep will be to make stepy match w/ KL

Well finally abour
$q\left(x_{t-1} \mid x_{t}\right)=q\left(x_{t-1} \mid x_{t} x_{t}\right)+\sigma\left(\beta_{t}\right)$

Justification of reverse transition heing Gaussion
By Bayen $q\left(x^{t-1} \mid x^{t}, x^{0}\right)=\frac{q\left(x^{t} \mid x^{t-1} x^{0}\right) q\left(x^{t-1} \mid x^{0}\right)}{q\left(x^{t} \mid x^{0}\right)}$
From forward $\quad q\left(x^{t} \mid x^{t-1}\right)=N\left(\sqrt{1-\beta_{t}} x_{t-1}, \beta_{t} I\right)$

An we decived alseady

$$
q\left(x^{t-1} \mid x^{0}\right)=N\left(\sqrt{\overline{\alpha_{t-1}}} x_{0},\left(1-\overline{\alpha_{t-1}}\right) I\right)
$$

product of Gaushiums is an un-noimalized Gacsion for some $\mu^{*}, \Sigma^{*}, C$ that we could derive $\Rightarrow q\left(x^{t} \mid x^{t-1}\right) q\left(x^{t-1} \mid x^{0}\right) \sim C N\left(\mu^{*}, \Sigma^{A}\right)$

Skipping details

$$
\begin{aligned}
& q\left(x_{t-1} \mid x_{t}, x_{0}\right)=N\left(M_{t}^{*}\left(x_{0}, x_{t}\right), \bar{\beta}_{t}^{*} I\right) \\
& \mu^{*}=\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_{t}}{1-\bar{\alpha}_{t}} x_{0}+\frac{\sqrt{\bar{\alpha}_{t}}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_{t}} x_{t} \\
& \beta^{*}=\frac{1-\bar{\alpha}_{t 1}}{1-\bar{\alpha}_{t}}
\end{aligned}
$$

$Q$ - How do
When
approximating $q\left(x_{t-1} \mid x_{t}\right)$ ?

To remove the conditianing on $x_{0}$ well use the laws of total expectation + variance:

$$
\begin{aligned}
& \mathbb{E}[X]=\mathbb{E}[\mathbb{E}[X \mid Y]] \\
& \operatorname{var}[X]=\mathbb{E}[\operatorname{var}[X \mid Y]]+\operatorname{var}[\mathbb{E}[X \mid Y]]
\end{aligned}
$$

On the wean

$$
\begin{aligned}
\mathbb{E}\left[x_{t-1} \mid x_{t}\right] & =\mathbb{E}\left[\mathbb{E}\left[x_{t-1} \mid x_{t}, x_{0}\right]\right] \\
& =\mathbb{E}\left[M^{*}\right] \sim O\left(\beta_{t}^{2}\right)
\end{aligned}
$$

Ov the variance

$$
\begin{aligned}
\operatorname{var}\left[x_{t-1} \mid x_{t}\right] & =\mathbb{E}\left[\operatorname{var}\left(x_{t} \mid x_{h} x_{0}\right)\right]+\underbrace{\operatorname{var}\left[\mathbb{E}\left[x_{t-1} \mid x_{t} x_{0}\right]\right]}_{O\left(\beta_{t}\right)} \\
& =\underbrace{\mathbb{E}\left[\beta^{*} I\right]}_{O\left(\beta_{t}^{2}\right)}+\underbrace{\operatorname{var}\left[\mu^{*}\right]}_{O\left(\mu^{*}\right]}
\end{aligned}
$$

(we're skipping the eatimater on $\beta_{t}$ )

Final Ingredient \#3 - the ELBO

$$
\begin{aligned}
\varepsilon=\mathbb{E}_{q\left(x^{T} \mid x^{0}\right)} & {\left[\log p\left(x_{0} \mid x_{T}\right)\right]-\sum_{t=2}^{T} \mathbb{E}_{q\left(x_{t} \mid x_{0}\right)}[ } \\
& \left.K L\left(q\left(x_{t-1} \mid x_{t}, x_{0}\right) \| p\left(x_{t-1} \mid x_{t}\right)\right)\right]
\end{aligned}
$$

- Learn to denoise single diffusion steps
- Geverate by sampling a gausion + denoising


## Some take aways

- Designed an encoding which
- required no learning
- took us to a latent gaussion
- has gavesian incrementa
- Perigned a decoding which
- is indirectly supervised by encoding
- gives gavision increments
- When camkined in ELBO
- Lots of closed form expressions for KL'S

How can you use this in your reseach?
(How much math do you need to track?)
VI Off-the-shelf

- just gials one of the implementations
- understand that if you mess $w /$ the architecture, you may break the theory
- maybe leave ELBO loss alone - daitplay w/ input loutput of denoising,
- do play $w /$ architecturen, $\beta$ scheduler


## U2 Improve

- Many other physical models can give $\lim _{T \rightarrow \infty} X^{T} \leftarrow N(O, I)$
- Replace noising w/ something physical
- Think about challenges a sequirements
- Expersive to generate - how can we use ahorteT
- What other SDES allow gausian reverse process?
- SPDES?

