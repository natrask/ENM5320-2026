Lecture 4-2/5

- Polynominal Reproduction
- Nonlinear stability
- Constrained Optimization

Our main hammer will be constrained opt

$$
\begin{aligned}
\min _{\theta} & F(x ; \theta) \\
& \operatorname{st} G(x ; \theta)
\end{aligned}
$$

Typical Conses

- $F$ is a reconstuction, $G$ are plysies "Physich - Constrained / PDE-constrained opt"
- $F$ is abitiary, $G$ is property we actually case about/"wish-list"
"Feasibility problem"

Equality constrained quadratic programming

Cancides

$$
\begin{array}{ll}
\min _{x} & \frac{1}{2} x^{\top} A x+b^{\top} x+c \\
\text { s.t } & B x=c
\end{array}
$$

For mon-tinears Replace w/ Hestion 4 Jacolior

To solve, add Layange multiplier $\lambda$

$$
\min _{x, \lambda} \frac{1}{2} x^{\top} A x+b^{\top} x+c+\lambda^{\top}(B x-c)
$$

The Karuih Kehn Tucker (KKT) conditions itate that the monimizer satisfies

$$
\begin{aligned}
& \delta_{x} L=\delta_{\lambda} L=0 \\
& \delta_{\lambda} L=B x-c=0 \\
& \delta_{x} L=A x+b+B^{\top} \lambda=0
\end{aligned}
$$

yielding the suddle point problem

$$
\text { (2) }\left(\begin{array}{ll}
A & B^{\top} \\
B & 0
\end{array}\right)\binom{x}{\lambda}=\binom{-6}{C}
$$

Mult top by $B A^{-1}$

$$
B A^{-1} / A x+B A^{-1} B^{\top} \lambda=-B A^{-1} b
$$

Erptiact from (2)

$$
-B A^{-1} B^{\top} \lambda=c+B A^{-1} b
$$

Let $S=B A^{-1} B^{\top}$ be defined as the schur conplement

$$
\begin{aligned}
& \lambda=-s^{-1}\left(c+B A^{-1} b\right) \\
& x=A^{-1}\left(-B^{\top} \lambda-b\right)
\end{aligned}
$$

We will use this repeatedly as a tool to entorce properties we'd like as equality construints

## A note an training

- In the HW we consider matching derivatives (this is offer called force matching)
$\hat{n} \rightarrow+$ neget $\quad \mathcal{L}=\| \frac{\hat{u}_{j}^{n+1}-\hat{u}_{j}^{n}}{k}+\sum_{k=-\alpha}^{\beta} \frac{S_{j+k} u_{j+k}^{n} \|^{2}}{h^{|\alpha|}}$ to approximate an operator $D^{\alpha}$
- This is prove to overfitting for noisy sealistic data
- Some strategies tomitinate
- expicitly model noise as MLE
![](https://cdn.mathpix.com/cropped/c52942e1-19b7-42c7-8b11-a05d7b7c1b55-04.jpg?height=432&width=887&top_left_y=1418&top_left_x=1041)
- Tithinov/weight decay \& we'll talk about there
- filtering / pre processing \& later
- Better to fit to the solution itself rather than its derivative
(2) Reduced space constrained optimization

Yub res. back into Le to optamize directly sues the space of FD solus.
$u_{j}^{n}=Q\left(u_{j}^{n-1}\right) u_{j}^{n-1}$ is the Lpotentially update noulinear

$$
\mathcal{L}=\frac{1}{2} \sum_{n, j}\left|Q\left(u_{j}^{n-1}\right) \ldots . Q\left(u_{j}^{0}\right) u_{j}^{0}-\hat{u}_{j}^{n}\right|^{2}
$$

- For nonlinear/explicit schemes, need to backprop through a linear solve, or Neucton/iterative nonlinew rolve
(3) Layange Multigliers

$$
\begin{aligned}
& \min _{u^{n}, r^{n}, \theta} \frac{1}{2} \sum_{n_{1}}\left|u_{j}^{n}-\hat{u}_{j}^{n}\right|^{2}+\sum_{n} \lambda_{n}\left(r_{n, j}\right)
\end{aligned}
$$

procesh For $1, \ldots, \#$ iterations

1. $\delta_{x} f=0 \Rightarrow$ Solve forward problem

Define the reconstruction loss and physics residual

$$
\begin{aligned}
& \mathcal{L}_{R}=\frac{1}{2} \sum_{n_{1 j}}\left|u_{j}^{n}-\hat{u}_{j}^{n}\right|^{2} \\
& r_{n_{1 j}}=u_{j}^{n+1}-u_{j}^{n}-k D_{\theta}^{\alpha} u_{j}^{n}
\end{aligned}
$$

$D_{\theta}^{\alpha} \leftarrow \begin{gathered}\text { prameterized } \\ \text { utencil }\end{gathered}$ utencil

Pone the constrained fit to the solution

$$
\begin{array}{ll}
\min _{u_{j, \theta}^{n}} & L_{R} \\
\text { h.t } & r_{n, j}=0 \\
\text { for all } u_{1 j}
\end{array}
$$

3 ways to do this
(1) Penally (wornt!)

$$
\min _{u_{j, \theta}} \mathcal{L}_{R}+\lambda \sum_{n_{i j}} r_{n_{i j}}^{2}
$$

- Take raker penalty pwam $\lambda$. For big $\lambda$, $\Gamma_{n, j}$ is quall.
- No way to piok $\lambda$ a priari
(Too big unstable, too amall big error)
- Small residuals translate to large soln eorors (thank of the $\theta\left(k^{2}\right)$ erross we sam in explectsteled)

2. $S_{u_{r}} L=0$ Tolve adjount problem

$$
\left(J_{r u}(r)\right)^{\top} \lambda=-\left(u_{j}^{n}-\hat{u}_{j}^{n}\right)
$$

3. Update $\theta$ w/ gradient opt. using cullent guess for $u, \lambda$

- Backprop doesut need to go thoough furd + adj solve, hest for noulonear-inglist

For knew problems (2) + (3) are the same

## Min: batching

- These are big gradvent calculations
- Typically our datanet caniets of many aomis
- intochasitic gradient descent
- Pick randoun soln $u_{i, d}^{n}$
- Pick subset of time series
![](https://cdn.mathpix.com/cropped/c52942e1-19b7-42c7-8b11-a05d7b7c1b55-07.jpg?height=300&width=514&top_left_y=2201&top_left_x=1558)

$$
u_{i, d}^{n}, \ldots u_{j, d}^{n+m}
$$

- Train using $\hat{n}_{j, d}$ as IC


## Polynomial Reproduction and least squares

Consider points $\bar{X}=\left\{x_{1} \ldots x_{N}\right\} \subseteq \Omega \subseteq \mathbb{R}^{d}$ For scattered data cluracterize the fill distance

$$
\begin{equation*}
h_{\bar{X}_{1, \Omega}}=\sup \min _{\mathbb{Z}_{\in \Omega}} \quad u_{x-j \leq N} \quad x_{j} u_{2} \tag{N}
\end{equation*}
$$

Biggest ball in $\Omega \mathrm{w} / \mathrm{o}$ data in it

## Sepwation Distance

$$
q_{\underline{X}}=\frac{1}{2} \min _{i \neq j}\left\|x_{i}-x_{j}\right\|_{2}
$$

def Data nites are quasi-uniform (w/constant czu) if

$$
q_{x} \leq h_{x_{1} \Omega} \leq c_{q_{n}} q_{x}
$$

For our FD itencits no far

$$
\begin{aligned}
& h_{\bar{x}, \Omega}=h \\
& q_{\bar{x}}=h
\end{aligned} \quad \Rightarrow c_{q n}=1
$$

![](https://cdn.mathpix.com/cropped/c52942e1-19b7-42c7-8b11-a05d7b7c1b55-08.jpg?height=185&width=930&top_left_y=2445&top_left_x=1012)

Introduce moving least squares, generalized FDM
For $x \in \Omega \quad S_{f, \mathbb{X}}(x)=p^{\star}(x)$

$$
\min _{p} \sum_{i=1}^{N}\left[f\left(x_{i}\right)-p\left(x_{i}\right)\right]^{2} \omega\left(x, x_{i}\right): p \in \pi_{m}\left(\mathbb{R}^{d}\right)
$$

We'll assume $w(x, y)=\bar{\Phi}_{\delta}(x-y)$

As a rimple example where $m=0$

$$
\min _{c(x)} \sum_{i=1}^{N} \frac{1}{2}\left(f\left(x_{i}\right)-c(x)\right)^{2} \Phi_{g}\left(x-x_{i}\right)
$$

Taking derivative and hetting to zero

$$
\begin{aligned}
& 0=\frac{7}{\partial} \sum_{i=1}^{N}\left(f\left(x_{i}\right)-c(x)\right) \Phi_{\delta}\left(x-x_{i}\right) \\
& c(x)=\frac{\sum_{i} \Phi_{g}\left(x-x_{i}\right) f\left(x_{i}\right)}{\sum_{i} \Phi_{g}\left(x-x_{i}\right)}
\end{aligned}
$$

and we recover Kernel density estimation aka the Shepard Interponent

Thum Solving the moving least squares problem is equivalest to solving

$$
\begin{aligned}
& s_{f, \bar{x}}(x)=\sum_{i} a_{i}^{*}(x) f\left(x_{i}\right) \\
& a_{i}^{*}=\operatorname{argmin} \frac{1}{2} \sum_{i} a_{i}(x)^{2} \frac{1}{\Phi_{\delta}\left(x-x_{i}\right)}
\end{aligned}
$$

(DP)
![](https://cdn.mathpix.com/cropped/c52942e1-19b7-42c7-8b11-a05d7b7c1b55-10.jpg?height=189&width=1429&top_left_y=832&top_left_x=164)
Rmk We can thus search for a minimal norm stencil that has reproduction properties
Pf Set some notation → lin alg heavy
$a \in \mathbb{R}^{N}$
$W=\operatorname{diag}\left(\Phi_{g}\left(x-x_{i}\right)\right) \in \mathbb{R}^{N \times N}$
$P(x) \in \mathbb{R}^{m}$
$P\left(x_{i}\right) \in \mathbb{R}^{M \times N} \longleftarrow$ will write as $P$ and distiguish $u:=u\left(x_{i}\right) \in \mathbb{R}^{N}$
$C \in \mathbb{R}^{M}$

N - \# nodes
$M=\operatorname{dim}\left(\pi_{m}\left(\mathbb{R}^{d}\right)\right)$ against $P(x)$

Pf Primal Problem

$$
\min _{c \in \mathbb{R}^{m}} \frac{1}{2} \underbrace{(u-c P)^{\top} W(u-c P)}_{L}
$$

Expunding
$L=\frac{1}{2} c^{\top} P W P^{\top} c-u W P^{\top} c+\begin{gathered}K \text { stutt indep } \\ \text { of } c\end{gathered}$
Recall

$$
\begin{aligned}
& \frac{\partial}{\partial x} \frac{1}{2} x^{\top} A x=\frac{1}{2}\left(A+A^{\top}\right) x \\
& \frac{\partial}{\partial x} y^{\top} x=y \\
& \begin{aligned}
\frac{\partial L}{\partial c}=P W P^{\top} c-W P^{\top} u=0 \\
\Rightarrow c=\left(P W P^{\top}\right)^{-1} W P^{-} u
\end{aligned} \\
& S_{f, E}(x) \\
& =C \cdot P(x) \\
& =P(x)^{\top}\left(P W P^{\top}\right)^{-1} W P^{-} u
\end{aligned}
$$

## Dual Problem

$$
\begin{aligned}
& S(x)=a(x) \cdot u\left(x_{i}\right) \\
& \min _{a \in \mathbb{R}^{N}} \frac{1}{2} a(x)^{\top} W^{-1} a(x) \\
& \text { st } a(x)^{\top} P=P(x)
\end{aligned}
$$

Applying Schur complement formula form before

$$
\begin{aligned}
K K T \rightarrow & \left(\begin{array}{cc}
W^{-1} & P^{\top} \\
P & 0
\end{array}\right)\binom{a(x)}{\lambda}=\binom{0}{P(x)} \\
S & =P W P^{\top} \\
\lambda & =-S^{-1} P(x) \\
a(x) & =-W P^{\top} \lambda \\
& =W P^{\top} S^{-1} P(x) \\
S(x) & =u^{\top} a(x) \\
& =u^{\top} W P^{\top} S^{-1} P(x) \\
& =P^{\top}(x)\left(P W P^{\top}\right)^{-1} P W u
\end{aligned}
$$

Cors Replace $a(x)^{\top} P=P(x)$ constraint with $a(x)^{\top} P=$

$$
=S^{\alpha} P(x) \text { constraint }
$$

to obtain differential operators/utencils
Q? Is polynomial reproduction set non-empty?
Thm (Wendland "Scattered Data Approx" Thm 4.7)
Suppose $\Omega \subseteq \mathbb{R}^{d}$ is compact and satisfies an interior cone condition $w / \theta \in(0, \pi / 2)$ and radius c. Then there exiats $a^{-k}(x)$ for any $x$ such that
(1) $\sum_{i} a_{j}^{\infty}(x) p\left(x_{j}\right)=p(x) \quad \forall p \in \Pi_{m}\left(x^{d}\right)$
(2) $\sum_{j}\left|a_{j}^{*}(x)\right| \leq \tilde{c}_{1}$
(3) $a_{j}^{*}(x)=0$
if $\left\|x-x_{j}\right\|_{2}>\widetilde{c}_{2} h_{x_{1} \Omega}$

For $\tilde{C}_{1}$ and $\tilde{C}_{2}$ indep of $h_{x, \Omega}$ that can be explictly derived

