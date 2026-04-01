Last time
Lax-Milgiam - abstract stability aralysis
(G) Find $n \in V_{n}$ s.t. $V_{h} \subseteq V$

$$
a(u, v)=L(v) \quad \forall v \in V_{h}
$$

- April $30^{\text {th }}$ Last -May 5-13 theals
- Plan projects now

Poisison
ex

$$
\begin{aligned}
-\nabla^{2} u & =f \\
\left.u\right|_{2 \Omega} & =0 \\
a(u, v) & =\int \nabla u \cdot \nabla v d x, L(v)=\int f v d x
\end{aligned}
$$

Binnmmic Eqn (Accostics, phase field, Beam theing)

$$
\begin{aligned}
& \rightarrow \quad \nabla^{2} \nabla^{2} n=f \\
& \quad n\left|\left.\right|_{2 \pi} \partial_{n} n\right|_{\partial \Omega}=0 \\
& V=H^{2}(\Omega)=\left\{\int\left(\nabla^{2} n\right)^{2}<\infty\right\} \\
& a(n, v)=\int \nabla^{2} n \cdot \nabla^{2} v d x \quad L(v)=\int f v d x
\end{aligned}
$$

## Lax-Milytarn

ais cont $0|a(v, w)| \leq \gamma\|v\|_{v} \| w u_{v}$

$$
\text { elliptic (2) } a(v, v) \geq \alpha\|v\|_{v}^{2} \quad \Rightarrow \text { itability }
$$

$L$ is cont (3) $|L(v)| \leq \Lambda U v \|_{v}$

Ceng Lemma

Suppose (1)-(3). Then

$$
n_{u}-u_{h} n_{v} \leq \frac{\gamma}{\alpha} \text { inf }_{v \in v_{h}}\left\|_{u}-v\right\|_{v}
$$

pf $\quad u-u_{n} \in V$

$$
\begin{aligned}
a\left(u-u_{n_{1}}, v\right) & =0 \\
& \forall v \in v_{n_{1}}
\end{aligned}
$$

$$
\begin{aligned}
\alpha u_{n}-u_{n} u_{v}^{2} & \leq a\left(n-u_{n}, n-u_{n}\right) \\
& =a\left(n-u_{n}, n\right)-a\left(n-u_{n}, u_{n}\right) \\
& =a\left(n-u_{n}, n\right)-a\left(n-u_{n}, v\right) \\
& =a\left(n-u_{n}, n-v\right) \\
& \leq \gamma u_{n}-u_{n} u_{v}, u_{n}-v u_{v} \\
u n-u_{n} u_{v} & \leq \frac{\gamma}{\alpha} u_{n}-v \|_{v} \quad \forall v \in V_{n}
\end{aligned}
$$

for any v

Today well get some practice working through thin alsshact theory
we are free to choose the

- PDE
- FEM aubagacc
- Norm
ex

$$
\begin{aligned}
a(u, v) & =\int \nabla u \cdot \nabla v d x \\
L(v) & =\int f v d x
\end{aligned}
$$

Remember

$$
\begin{aligned}
L^{2} \rightarrow\|s\|_{L_{2}} & =\int f^{2} d x \\
H^{\prime} \text {-seminosin } & |f|_{H_{1}}=\int \nabla f^{2} d x
\end{aligned}
$$

$H^{\prime}-$ norm

$$
\|f\|_{H_{1}}^{2}=\|f\|_{L_{2}}^{2}+\|\left. f\right|_{H_{1}} ^{2}
$$

Let $V=H_{0}^{2}(\Omega) \leftarrow$ integrable $2^{\text {nd }}$ der.us
$(u, v)_{v}=\int D^{2} u: D^{2} v+\nabla u \cdot \nabla v+u v$
where

$$
A: B=\sum_{i j} A_{i j} B_{i j} \quad \text { Frobenius inne prod. }
$$

Chack conditions
(1)

$$
\begin{aligned}
|a(v, w)| & =\left|\int \nabla v \cdot \nabla w d x\right| \\
& \leq\|\nabla v\|^{*}\|\nabla w\|^{2} c-s \\
\|v\|_{v}^{2} & =\int \nabla^{2} v: \nabla^{2} v+|\nabla v|^{2}+v^{2} d x \\
\geq & \int|\nabla v|^{2} d x=\|\nabla v\|
\end{aligned}
$$

So $\quad|a(v, w)| \leq\left\|v n_{v} \quad\right\| w \|_{v}$
(2)
$a(\nu, v)=\|\nabla v\| \geq \alpha \int D^{2} \psi^{2}+\mid \nabla u^{2}+v^{2} d x$
Non-Trivial → need a result from PDES called elliptic regularity

For convex $\Omega,\|V\|_{H^{2}(\Omega)} \leq C\|\Delta V\|_{L^{2}(\Omega)}$ for $v \in H^{2} \cap H_{0}^{\prime}$
(3)

$$
\begin{aligned}
|L(v)| & \leq\|f\|_{L^{2}}\|v\|_{L^{2}} \\
& \leq\|f\|_{L^{2}}\|v\|_{V} \quad \Rightarrow \Lambda=u s \|_{L^{2}}
\end{aligned}
$$

## Reaction Diffusion eqn.

$-\nabla \cdot A \nabla n+r n=f$
w/ asqumptions
(A) $A$ is pos def, $x^{\top} A x>0$
(B) $r \geq 0$
(c) $\left|A_{i j}\right|<\infty,|r|<\infty$ hounded

## weak form

$$
\begin{aligned}
& a(u, v)=L(v) \quad v \in H_{0}^{\prime} \\
& a(u, v)=\int_{\Omega} \nabla u \cdot A \cdot \nabla v+r u v d x \\
& L(v)=\int_{\Omega} f v d x
\end{aligned}
$$

(1) Show $|a(v, w)| \leq \gamma$ nunv nwn

$$
\begin{aligned}
\left|\int \nabla v \cdot A \cdot \nabla w d x\right| \leq \max _{i, j}\left|A_{i j}\right|\left|\int \nabla v \cdot \nabla w d x\right| & \leq \gamma_{A}\left\|v H_{H_{1}}\right\| w \|_{H_{1}} \\
\left|\int r v w d x\right| & \leq \gamma_{A}|v|_{H_{1}}|w|_{H_{1}} \\
\leq & \max |r| \quad \int w_{1} w d x \\
\leq & \|v\|_{L^{2}}\|w\|_{L^{2}} \\
a(v, w) \leq \quad \gamma_{A}|v|_{H_{1}}|w|_{H_{1}}+\gamma_{r}\|v\|_{L^{2}}\|w\|_{L^{2}} & \leq 2 \max \left(\gamma_{A_{1}} \gamma_{r}\right)\|v\|_{H_{1}}\|w\|_{H_{1}}
\end{aligned}
$$

(2)

$$
\begin{aligned}
a(u, n) & \geq \alpha\|u\|_{v}^{2} \\
a(u, n) & =\int \nabla u \cdot A \cdot \nabla u \cdot r u v d x \\
& \geq \int \nabla u \cdot \nabla u+u^{2} d x \\
& =\| u U_{H_{1}}^{2}
\end{aligned}
$$

## Elasticity

- First example of how analysis predicts breakolown of stability
def $u \in \mathbb{R}^{d}$-displacement

$$
\begin{aligned}
& \varepsilon(n) \in \mathbb{R}^{d d}-\text { strain } \\
& \varepsilon(n)=\frac{1}{2}\left(\nabla n+\nabla n^{\top}\right) \\
& \sigma(\varepsilon)-\text { stress } \\
& \sigma_{i j}=C_{i j k l} \varepsilon_{k l}
\end{aligned}
$$

To enforce rotational invaiance for isotropic materials

$$
C_{i j k l}=\lambda \delta_{i j} \delta_{k l}+\mu\left(\delta_{i j} \delta_{k l}+\delta_{i l} \delta_{j k}\right)
$$

So the stress reduces to

$$
\sigma_{i j}=\lambda \delta_{i j} \varepsilon_{k k}+\partial \mu \varepsilon_{i j} .
$$

Aside for elasticity, we have the lograngin derisity

$$
\mathcal{L}=-\frac{1}{2} \varepsilon^{\top} C \varepsilon+\frac{e}{2} \partial_{t} u^{2}
$$

Governing Equs

$$
\left\{\begin{array}{l}
-\operatorname{div} \sigma^{\prime}=f \\
\sigma^{\prime}=2 \mu \varepsilon+\lambda \operatorname{tr}(\varepsilon) I \\
\left.u\right|_{2 \pi}=0
\end{array}\right.
$$

Note that $\operatorname{tr}(\varepsilon)=\nabla \cdot u$

$$
\left\{\begin{array}{l}
-\nabla \cdot(\partial \mu \varepsilon(u)+\lambda \nabla \cdot u I)=f \\
\left.u\right|_{2 r}=0
\end{array}\right.
$$

To define varintianal problem, we'll work w/ displacements

$$
\vec{u} \in \vec{H}_{0}^{\prime}(\Omega)=\left[H_{0}^{\prime}(\Omega)\right]^{d} H_{0}^{\prime} \text { aka each camparate (lot }
$$

$0 \Rightarrow \quad a(u, v)=(f, v)$

$$
a(u, v)=\int \mu \varepsilon(u): \varepsilon(v)+\lambda(\nabla \cdot u)(\nabla \cdot v) d x
$$

Again, well cleck continuity and ellifficity of $a$, but write

$$
\begin{array}{r}
\lambda=\frac{\partial \mu \delta}{1-4 \delta}, \quad \begin{array}{r}
\text { where } \delta \text { in the } \\
\text { Poisian ratio } \\
\delta=-\frac{\varepsilon_{\text {tomisure }}}{\varepsilon_{a_{\text {xial }}}}
\end{array}
\end{array}
$$

For near-incompressible materials, as $\delta \rightarrow \# \frac{1}{2}, \lambda \rightarrow \infty$
So we'd like to understand the $\lambda \rightarrow \infty$ limiting stability
To do this we need Korn's inequality

$$
\int \varepsilon(n): \varepsilon(n) d x \geq\|u\|_{H_{1}}
$$

$$
\begin{aligned}
& \text { / ex near-incomp } \\
& \text { Rublers } \\
& \text { Hydroyels }
\end{aligned}
$$

> clays (saturnted
> paras)
(1) $|a(w, v)| \leq \gamma u_{w} u_{H_{1}} n v u_{H_{1}}$

$$
\begin{aligned}
&\left|\int \mu \varepsilon(w): \varepsilon(v) d x\right|=\left|\frac{\mu}{4} \sum_{i, j} \int\left(\partial_{x_{i}} w_{j}+\partial_{x_{j}} w_{i}\right) \cdot\left(\partial_{x_{i}} w_{j}+\partial_{x_{j}} w_{i}\right)\right| \\
&=\mu \sum_{i, j} \int \partial_{x_{i}} w_{j}^{2} \partial_{x_{i}} v_{j} d x \mid \\
&=\mu \| u H_{H_{1}} \\
& \leq \mu \| w H_{H_{1}} u v u_{H_{1}} \\
&\left|\int \lambda(\nabla-w) \nabla \cdot v\right| \quad \leq \lambda \int|\nabla \cdot w| \int \nabla \cdot v \leq \lambda\|w\|_{H_{1}}\|v\|_{H_{1}} \\
&|a(w, v)| \leq(\mu+\lambda)\|w\|_{H_{1}} \| v u_{H_{1}}
\end{aligned}
$$

(2) $a(u, n)=\int \mu \varepsilon(u): \varepsilon(n)+\lambda(\nabla-n)^{2} d x \stackrel{?}{\geq} \alpha U_{n} U_{11}^{2}$

$$
\begin{aligned}
& \geq \int \mu \varepsilon(n): \varepsilon(n) d x \\
& \geq \mu\|n\|_{H_{1}} \quad \text { By Korn }
\end{aligned}
$$

What goes wrong?
(1) $\| u_{n} n_{H_{1}} \leq \frac{\Lambda}{M} \Rightarrow$ get a stable solution
(2) By Cen's lemma $\left\|n-u_{n} n_{H_{1}} \leq \frac{M+\lambda}{M} u_{n}-v\right\|_{H_{1}}$

Ove solution, and our first example of mixed FEM Take
$\mu \int \varepsilon(u): z(v)+\lambda \int \operatorname{div} u \operatorname{divv}=\int f v$
Let $p=\lambda \operatorname{divu}$, and introduce second FEM spice $q \in M_{h}$

$$
\begin{gathered}
\mu \quad \int \varepsilon(n): \varepsilon(v) d x+\int p \nabla \cdot v d x=\int f v d x \\
\int \nabla \cdot u q d x+\frac{1}{\lambda} \int p q d x=0
\end{gathered}
$$

This gives a new galerkin equ but how to $\operatorname{decon} n \quad M_{h} ?$
Note in limit $\mu=1, \lambda \rightarrow \infty$ thin reduces to the stationary stokes problem
(5) $\left\{\begin{aligned}-\Delta n-\nabla p & =f \\ \nabla \cdot n & =0\end{aligned}\right.$
$u \in H_{0}^{\prime}$ for Laplacion to make sente, $M_{n} \subseteq L^{2}$

$$
(\nabla p, v) \underset{\text { IBP }}{\rightarrow}-(p, \nabla \cdot v)<\|\rho\|_{L^{2}}\|v\|_{H_{1}}
$$

Galerkin form

$$
\begin{aligned}
(\nabla u, \nabla v)+(p, \nabla \cdot v) & =(f, v) \\
(\nabla \cdot u, q) & =0
\end{aligned}
$$

Stability
![](https://cdn.mathpix.com/cropped/3a89613e-a1d5-4213-b6cb-1fd6baa1012a-11.jpg?height=174&width=1024&top_left_y=879&top_left_x=169)

This gives our first variational saddle-pt problem Well need same lemmas to tackle it, hut lefs prove uniquess quictly now ( $f=0 \Rightarrow p, n=0$ )

Take $v=u$

$$
\|u\|_{H_{1}}^{2}+\left(p_{1} \nabla \cdot u\right)=0
$$

Taking $q=p$ inte grde eqn, $\left(\rho_{1} \nabla \cdot u\right)=0 \Rightarrow u u u_{1,}^{2}=0$

$$
\Rightarrow u=0
$$

Finally, we'd like to show
$u=0 \Rightarrow p=0$. But we cant. Nothing in equ givesin that
IDEA Deargn relationskip between $V_{h,} M_{h}$ so $n=0 \Rightarrow p=0$

## Inf-up cardition

$V_{h}$ and $M_{h}$ are inf-sup compatible if for any $q \in M_{h}$, there exists a $v \in V_{h}$
s.t

$$
\beta\|q\|_{L^{2}} \leq \frac{(q, \nabla \cdot v)}{\|v\|_{H^{\prime}}}
$$

Returning to itokes mamentum eqn

$$
(\nabla v / \nabla v)+\left(p_{1} \nabla \cdot v\right)=0
$$

Already skown
Take qop

$$
\begin{aligned}
& \| p H_{L}^{2} \leq \frac{(p, \nabla \cdot v)}{\beta\|v\|_{H_{1}}}=0 \\
& \Rightarrow p=0
\end{aligned}
$$

Next time

- How to build inf-sp stable spaces
- How to lean physics in this mixed FEM setting

