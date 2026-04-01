## Today

- Integrating probability into lewned models
- Stochastic djuamics
- Maximum likelihood
- Markov processes
- Euler - Maruyama

Motivation
![](https://cdn.mathpix.com/cropped/5432938a-1b68-4b02-84de-9ceb394bd5e7-01.jpg?height=447&width=570&top_left_y=938&top_left_x=180)

- Tracking a point us tracking a "blob" of prohability
- Sensor noise, may have either $\left.\begin{array}{rl} & \text { noisy observations } \\ \text { or noing physics }\end{array}\right\} \begin{aligned} & \text { aleatoric } \\ & \text { ancertainty }\end{aligned}$
- or an incomplete description of
![](https://cdn.mathpix.com/cropped/5432938a-1b68-4b02-84de-9ceb394bd5e7-01.jpg?height=270&width=611&top_left_y=1529&top_left_x=169)
physics (epointemic uncertainty)
- Even w/o noise, training a model in a probabilistic context is often easier
![](https://cdn.mathpix.com/cropped/5432938a-1b68-4b02-84de-9ceb394bd5e7-01.jpg?height=580&width=1024&top_left_y=1817&top_left_x=59)

→ probabiliatic regularization

References for probability

- 'Problability Essentials' Jacod + Protter

→ Short poperback, rigorous but quick definitions

- 'Probability: theory + examples' Durrett

→ Measure theoretic → gravily to get started from

- Machine learning, a probabilitic persplective' Kerin Murphy

→ Accessible, lots of background and follow-ar refs

- like wikipedin for ML (no ODEs/PDES though)
- 'Introduction to stochastic integration' Kno → where I'll pull refs from

Overall → I wouldn't reommend following a text for thin course, because there inft a good one for scientific computingeme

Probability review
def a continuous randan variable $\bar{X}$ takes a random value $x \in \mathbb{R}$ def a cumulative diatibution function (CDF) defines probability over a range of valves

$$
F_{X}(x)=\mathbb{P}(X \leq x)
$$

def if CDF is differentiable, define probability density function

$$
f_{\bar{X}}(x)=\frac{d}{d x} F_{\bar{X}}(x)
$$

so $\mathbb{P}(a \leq x \leq b)=\int_{a}^{b} f(x) d x=F(b)-F(a)$

Drop ackeringts unless neccury

We'll talk about different RVs + events by different notions of pdfs:

Joint distribution

Marginalization
conditional diat
product rule

$$
f\left(x_{1}, \ldots x_{N}\right)=\mathbb{P}\left(\bar{\nabla}_{1}=x_{1}, \ldots \bar{\nabla}_{N}=x_{N}\right)
$$

$f(\bar{X}=x)=\sum f(\bar{X}=x, \bar{Y}=y) \quad$ Rule of tohil problability

$$
f(\bar{Y}=y \mid \bar{X}=x)=\frac{f(\bar{X}=x, \bar{Y}, \bar{Y}=y)}{f(\bar{X}=x)}
$$

$$
f(x, y)=f(x \mid y) \quad f(y)
$$

pcoh. chain
rule $f\left(x_{1}, \ldots x_{N}\right)=f\left(x_{2}, \ldots x_{N} \mid x_{1}\right) f\left(x_{1}\right)$

$$
\begin{aligned}
& =f\left(x_{3}, \ldots x_{N} \mid x_{1}, x_{2}\right) f\left(x_{2} \mid x_{1}\right) f\left(x_{1}\right) \\
& \vdots \\
& =f\left(x_{N} \mid x_{1}, \ldots x_{N-1}\right) \cdots f\left(x_{2} \mid x_{1}\right) f\left(x_{1}\right)
\end{aligned}
$$

Marginalindep $\quad x \perp y$ if $\quad f(x, y)=f(x) f(y)$
or ingeneral $f\left(x_{1}, \ldots x_{N}\right)=\prod_{i=1}^{N} f\left(x_{i}\right)$
conditional indep $x|z+y| z$ if $f(x, y \mid z)=f(x \mid z) f(y \mid z)$

Fitting distributions to data w/ MLE

Given a dataset consiating of observations of a RV $\overline{\text { r }}$

$$
\mathscr{D}=\left\{x_{i}\right\}_{i=1}^{\text {Ndata }}
$$

Define the likelihood by evaluating the parameterized joint distribution $\mathcal{L}=f\left(x_{1}, \ldots x_{\text {delation }} \mid \theta\right)$
Fit the distribution to data by solving

$$
\theta^{*}=\underbrace{-\log (\text { ikelinod })}_{N L L \quad \log \mathcal{L}\left(x_{1}, \ldots x_{\text {Negatial }} \mid \theta\right)}
$$

## Example

![](https://cdn.mathpix.com/cropped/5432938a-1b68-4b02-84de-9ceb394bd5e7-05.jpg?height=197&width=441&top_left_y=960&top_left_x=1558)

- Assume independent data

$$
f\left(x_{1}, \ldots x_{N} \mid \theta\right)=\prod_{i=1}^{N} f\left(x_{i} \mid \theta\right)
$$

- Choose maginal distribution

$$
\begin{aligned}
& f\left(x ;(\theta)=N\left(x ; \mu, \sigma^{2}\right) \rightarrow \theta=\left\{\mu, \sigma^{2}\right\}\right. \\
& N(x ; \theta)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}\right] \\
& \log N=-C-\frac{1}{2} \log \sigma^{2}-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2} \\
& N L L=\sum_{i} \frac{1}{2} \log \sigma^{2}+\frac{1}{2}\left(\frac{x_{i}-\mu}{\sigma}\right)^{2}
\end{aligned}
$$

- Solve for $\theta^{*}$

$$
\begin{aligned}
0=\nabla_{M} N L L & =\sum_{i} \nabla_{M} \frac{1}{2}\left(\frac{x_{i}-\mu}{\sigma}\right)^{2} \\
0 & =-\sum_{i}\left(x_{i}-\mu\right) \\
\sum_{i} \mu & =\sum_{i} x_{i} \\
N \mu & =\sum_{i} x_{i} \quad \text { similaly } \quad \sigma^{2}=\frac{\sum_{i}\left(x_{i}-\mu\right)^{2}}{N} \\
\mu & =\frac{\sum_{i} x_{i}}{N}
\end{aligned}
$$

Stochastic differential equations
To account for randan forcing/physics we'll expand our notion of ODES

First, write

$$
\frac{d x}{d t}=f(x, t)
$$

Insted as

$$
\int_{0}^{t} d x=\int_{0}^{t} f(x, t) d t
$$

For short hand people omit the limits of integration

$$
d x_{t}=f\left(x_{t}\right) d t \quad, x_{t}=x(t)
$$

![](https://cdn.mathpix.com/cropped/5432938a-1b68-4b02-84de-9ceb394bd5e7-06.jpg?height=215&width=410&top_left_y=2261&top_left_x=1654)

We will see that random forcing leads to solutions which arent differentiable, so its necessary to intepret in integral sense.

To account for stochartic terms, we will consider a SDE

$$
d x_{t}=\underbrace{f\left(x_{t}, t\right)}_{\text {drift }} d t+\underbrace{g(x, t) d W_{t}}_{\text {diffusion }} K_{\substack{\text { stochastic } \\ \text { process }}}
$$

Need to define $d W_{t}$ and how to integrate it.
In general, we could spend a whole course studying stochastic proceshes - well give an accelerated version here.
Well consider the Wiener process $W_{t}=\int_{0}^{t} d W_{t}$ defined via (1) $W_{0}=0$
(2) $W$ has independent increments for $t>0, u \geq 0 W_{t+u}-W_{t}$ are independent of $W_{s}$, for anysut
(3) Whas Gausion increments w/ variance equal to

$$
w_{t+n}-w_{t} \sim N(0, u)
$$

time increment
(4) $W$ is continuous

Many different constructions of $W_{t}$ that satisfy these procensea, e.g. Let $\xi_{1}, \xi_{2}, \ldots$ be IID RVs $w / \mathbb{E}\left[\xi_{i}\right]=0, \operatorname{var}\left[\xi_{i}\right]=1$

$$
W_{n}(t)=\frac{1}{\sqrt{n}} \sum_{1 \leq k \leq \Delta_{n} t} \xi_{k}
$$

Then $\lim _{n \rightarrow \infty} W_{n}(t)=W_{t}$
![](https://cdn.mathpix.com/cropped/5432938a-1b68-4b02-84de-9ceb394bd5e7-07.jpg?height=396&width=637&top_left_y=2179&top_left_x=1218)

To integrate against something like this, consides the Riemann construction
(a)

$$
\int_{0}^{t} g\left(x_{t}, t\right) d w_{t}=\lim _{\Delta t \rightarrow 0} \sum_{i=1}^{n-1} g\left(x_{t_{i}}, t_{i}\right)\left(d w_{t_{i+1}}-d w_{t_{i}}\right)
$$

It turns out that this leads to many pathological issues that violate the usual assumptions from standard calculus.
(1) We can prove that w/ probability $l, W_{t}$ is coctinuous
(2) Also w/ probl, $w_{t}$ is nowhere differentiable

In a probability class we would get into details - lut thin is enough to pose a simple scheme for solving SDES in our learning prollems by discretizing the abme of (A)

Eules-Masuyamn Method
Given SDE

$$
d x_{t}=f\left(x_{t}, t\right) d t+g\left(x_{t}, t\right) d w_{t}
$$

Solve, for $x_{t_{n}}=x(t=n k)$
for $n=13 n 0,1, \ldots$.

$$
\begin{aligned}
& x_{n+1}=x_{n}+k f\left(x_{t_{n}}, t_{n}\right)+\xi_{n} g\left(x_{t_{n}}, t_{n}\right) \\
& \xi_{n} \sim N(0, k)
\end{aligned}
$$

Some better integrators (see Milsteins method) but require a deaper dive

Uning maximum likelihood we can use a martor process 9 to fit an SDE to data
def For a $\mathrm{K}^{\text {th }}$-order Markor process, given a discrete time series $\vec{y}=\left\langle y_{0}, y_{1} \ldots y_{N}\right\rangle$

$$
P\left(y_{i} \mid y_{0}, \ldots y_{i-1}\right)=P\left(y_{i} \mid y_{i-1}, \ldots y_{i-k}\right)
$$

This means you only need to model the last $K$ time steps → this in like a multi-step integrator $w / K$ steps.

Euler-Maryama in Markov w/ $k=1$

$$
P\left(x_{n+1} \mid x_{n}\right)=N\left(x_{n}+K f\left(x_{n}, t_{n}\right), K g\left(x_{n}, t_{n}\right)^{2}\right)
$$

Pf If $x \sim N\left(\mu, \sigma^{2}\right)$

$$
A x+b \sim N\left(A \mu+b, \quad A^{2} \sigma^{2}\right)
$$

Identify

$$
\begin{aligned}
b & =x_{n}+E f_{n} \\
A & =g \\
\sigma^{2} & =k
\end{aligned}
$$

Now we can go back and derive a NLL
smapys w/ts on band.

$$
\begin{aligned}
&-\log p\left(y_{1}, \ldots y_{N}\right)=-\log p\left(y_{N} \mid y_{N-1}\right) p\left(y_{N-1} \mid y_{N-2}\right) \ldots \\
&=-\log \prod_{i=1}^{N} p\left(y_{i} \mid y_{i-1}\right) \\
&=-\sum_{i=1}^{N} \log \rho\left(y_{i} \mid y_{i-1}\right) \\
&=-\sum_{i=1}^{N} \log N\left(x_{i}+k f_{i}, k g_{i}^{2}\right) \\
&=C+\sum_{i=1}^{N} \frac{1}{2} \log k g_{i}^{2}+\frac{1}{2} \frac{\left(x_{i+1}-x_{i}-k f_{i}\right)^{2}}{k g_{i}^{2}}
\end{aligned}
$$

If we replace $f(x, t) w / f(x, t ; \theta)$ i.e. make the

$$
g(x, t) \quad g(x, t ; \theta)
$$

SDE trainable
We can solve

$$
\min _{\theta} \sum_{i=1}^{N} \log k g_{i, \theta}^{2}+\frac{\left(x_{i+1}-x_{i}-k f_{i, \theta}^{2}\right)^{2}}{k g_{i, \theta}^{2}}
$$

## Example of structure preserving SDES

Metriplectic / GENERIC formalism

$$
\begin{aligned}
d x_{t}=\left(L \partial_{x} E+M \partial_{x} S\right. & \left.+K_{B} \partial_{x} \cdot M\right) d t \\
& +\sqrt{2 K_{B} M} d W_{t}
\end{aligned}
$$

where $E \rightarrow$ energy Degeneracy Condition

$$
\begin{aligned}
& S \rightarrow \text { entropy } \\
& L=-L^{T} \quad+\quad \text { Degeneracy Candition } \\
& M=M^{T} S=M \partial_{x} E=0 \\
& K_{B} \rightarrow \text { Bottemann constant }
\end{aligned}
$$

-When "coarse-graining" a reversible system, the loss of information manifests as a entropy increase, or introduction of dissipation

- To counter this, stochastic forcing does work on the system.
- Exact fluctuation dissipation theorem guarantees this holds rigorounly


## Deterministic limit $\left(K_{B} \rightarrow 0\right)$

Thm Energy is conserved
px $\frac{d E}{d t}=\partial_{x} E^{\top} \frac{d x}{d t}$

$$
\begin{aligned}
& =\partial_{x} E^{\top}\left(L \partial_{x} E+M \partial_{x} S\right) \\
& =\partial_{x} E^{\top} L \partial_{x} E+\partial_{x} S^{\top} M \partial_{x} E \\
& \text { By skew-rymmetly } \quad B_{y} \text { degen condition }
\end{aligned}
$$

Remark Menriplectic is thesefore a generalization of Hamiltaion mechanics, with degen condition Killing off coosi terms between rev and irres parts.
Thm Entopy is non-decreasing
PE $\quad \frac{d S}{d t}=\partial_{x} S^{T} \frac{d x}{d t}$

$$
=\frac{\partial_{x} E^{\top} L \partial_{x} S}{=0}+\frac{\partial_{x} S^{\top} M \partial_{x} S}{\geq 0} \geq 0
$$

