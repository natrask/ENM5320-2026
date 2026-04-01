Leature 3

Recall the Lax Equivalence Thesrem
(i) $\|a\|^{2}<\infty$
(2) $\sup \left|\hat{Q}^{n}\right| \leq K_{s}$
(3) $\lim _{K, h \rightarrow 0} \sup _{0 \leq t_{n} \leq T}\left|\hat{Q}^{n}(\xi)-e^{j w t_{n}}\right|=0$

$$
\Rightarrow \quad \lim _{k, \operatorname{Lim}_{0}} \operatorname{up}_{0 \leq t_{A} \leq T}\left\|U\left(0, t_{n}\right)-\operatorname{Int}_{N}\left(i_{1}\right)\right\|=0
$$

For finite IC, stability - consiatercy imply convergence

Today we'll consider the learning problem

$$
\begin{aligned}
& u^{t a g e t}(x, t, x)=\sin (k(x-t)) \\
& \partial_{t} n=\mathcal{L}(n ; \theta) \\
& \min _{\theta}\left\|\partial_{t} n^{t a n g t}-\mathcal{L}(n ; \theta)\right\|^{2}
\end{aligned}
$$

And how to approach (2) + (3)

Recall that the explicit Euler i centered diff. scheme for transport eqn is unconditorually unitable
.e. $\quad \partial_{1} n=\partial_{x} n$
For a single mode $\omega$

$$
\begin{gathered}
\frac{\partial_{i}^{n}-\partial_{x}^{n}}{k}=D_{0} v_{i}^{n} \\
|\hat{Q}| \leq k_{j}^{n}=\frac{1}{\sqrt{2 n}}\left(1+i \frac{k}{h} \sin \right. \\
\left|1+i \frac{k}{h} \sin (w h)\right| \leq 1+\frac{k}{h}|i \sin (w h)| \\
\leq 1+\frac{k}{h}
\end{gathered}
$$

- For any finite $k / h$, it will grow.
- Consistent, but not stable
- Car add artificial viscosity to control growth

$$
\partial_{t} u=\partial_{x} u+\sigma h \partial_{x x} u
$$

as hido recoves true eqne so ok for commisturn
$-\frac{V_{j}^{n+1}-V_{j}^{n}}{k}=D_{0} V_{j}^{n}+\sigma h D_{+} D_{-} V_{j}^{n}$

- Choose $\sigma, K, h$ so that $|\hat{Q}| \leq 1$
- Recall $h D_{0} e^{i \omega x}=i \sin (\omega h) e^{i \omega x}$

$$
h^{2} D_{+} D_{-} e^{i \omega x}=-4 \sin ^{2}\left(\frac{\omega h}{2}\right) e^{i \omega x}
$$

So we could show

$$
\begin{aligned}
& \lambda=k / h \\
& \dot{\xi}=w h
\end{aligned}
$$

$|a+i h|^{2}$
$=a^{2}+b^{2}$

$$
\hat{Q}=1+i \lambda \sin \xi-4 \sigma \lambda \sin ^{2} \frac{\xi}{2}
$$

$\sin \theta=2 \sin \frac{\theta}{2}$

$$
\downarrow \cos \frac{\theta}{2}
$$

Expand power 4

$$
|\hat{Q}|^{2}=\left(1-4 \sigma \lambda \sin ^{2} \frac{\xi}{2}\right)^{2}+\lambda^{2} \sin ^{2} \xi
$$

$$
\begin{array}{rl}
|\hat{Q}|^{2} & =1-8 \sigma \lambda \sin ^{2} \frac{\xi}{2}+16 \sigma^{2} \lambda^{2} \sin ^{4} \frac{\xi}{2}+4 \lambda^{2} \sin ^{2} \frac{\xi}{2} \cos ^{2} \frac{\xi}{2} \\
& =k-1 \\
+4 & 4 \lambda^{2} \sin ^{2} \frac{\xi}{2}\left(1-\sin ^{2} \frac{\xi}{2}\right)
\end{array}
$$

Collect powers of $\sin \frac{9}{2}$

$$
=1-\left(80 \lambda-4 \lambda^{2}\right) \sin ^{2} \frac{5}{2}+\left(16 \sigma^{2}-4\right) \lambda^{2} \sin ^{4} \frac{5}{2}
$$

We can make this $\leq 1$ in tero ways
(1) Make both terms negative

$$
\begin{array}{cc}
8 \sigma \lambda-4 \lambda^{2} \geq 0, & 16 \sigma^{2}-4 \leq 0 \\
\sigma \geq \frac{\lambda}{2} & 0 \leq \frac{1}{2}
\end{array}
$$

So pick $0<\lambda \leq 20^{\circ} \leq 1$
(2) Eor

Take $\sin \frac{\xi}{2}=1$

$$
\begin{gathered}
|\hat{Q}|^{2}=1-\left(8 \sigma \lambda-4 \lambda^{2}\right)+\left(16 \sigma^{2}-4\right) \lambda^{2} \leq 1 \\
0 \leq 8 \sigma \lambda-4 \lambda^{2}-16 \sigma^{2} \lambda^{2}+4 \lambda^{2}
\end{gathered}
$$

So we can have $\sigma \geqslant \frac{1}{2}$ if

$$
2 \sigma \lambda \leq 1
$$

Thin analysis gives two schames:
(i) Lax-Friedrichs Method $\left(\sigma=\frac{h}{2 k}=\frac{1}{2 \lambda}\right)$

$$
V_{j}^{n_{1}=}=\left(I+k D_{0}\right) V_{j}^{n}+\frac{1}{2} h^{2} D_{+} D_{-} V_{j}^{n}
$$

(2) Lax-Wendotf Method $\left(\sigma=\frac{k}{2 h}=\frac{\lambda}{2}\right)$

$$
V_{j}^{n+1}=\left(I+k D_{0}\right) V_{j}^{n}+\frac{1}{2} k^{2} D_{+} D_{-} V_{j}^{n}
$$

- Those give stability by adding non-physical terms, this is our torat example of numerical stablication
- Often we can just add a disaipative teimw/ a fudge factor in front
- Instead of adding terms we can use implicit schemes

$$
\begin{aligned}
& \frac{V_{j}^{n+1}-V_{j}^{n}}{k}=D_{0} V_{i}^{n+1} \\
& \left(I-k D_{0}\right) V_{j}^{n+1}=V_{j}^{n} \\
& \hat{Q}=[1-i \lambda \sin \xi]^{-1} \leq 1
\end{aligned}
$$

for any $\lambda, \sigma^{\prime}$.
The in called unconditional stability, but needs us to solve a linear system to update

One special implicit scheme is a trapezoil rule in time Crank-Wicholson mothod $\frac{V_{j}^{n+1} V_{j}^{n}}{k}=\frac{1}{2} D_{0} V_{j}^{n+1}+\frac{1}{2} D_{0} V_{j}^{n}$ which has $|Q|=1$ excetly

Ref "On genealized maving leanst aquars.
Now that we have a and differe derivatives" Mirzai etal. handle on stability tooks we
can turn to how to build accuracy guarantees Assume we want to approximate a differential operator
$D^{\alpha}$. In multi-index notation $\alpha$ is a tuple denoting anized derivatives (e.g. $\left.\alpha=(1,2) \Rightarrow D^{\alpha}=\partial_{x} \partial_{y y}\right)$ and $|\alpha|$ the ooder of the derivative.
$D_{h}^{\alpha} v_{j}=\sum_{j} s_{j i}^{\alpha} v_{j}$ in a generic etencil
Thm If (1) $\sum_{j} S_{j i} p_{j}=D^{\alpha} p\left(x_{i}\right)$ for a $m^{\text {th }}$ polynamial order polynomial p reproduction
(2) $I\left|s_{i i}\right| \leq C h^{-|\alpha|}$

Then

$$
\left\|D_{h}^{\alpha} n_{i}-D^{\alpha} n_{i}\right\|_{L} \leq 1 ~ C h^{m+1-|\alpha|}
$$

$$
\begin{aligned}
& D_{k h}^{\alpha} u_{i}=\sum_{j} S_{j+\alpha}^{\alpha} u_{j} \\
& \text { (1) } \sum_{i} S_{j i} P_{j}=D_{k} P_{k}\left(x_{i}\right) \\
& \text { (2) } \sum_{j}\left|S_{j i}\right|<C_{i} h^{-1 \alpha 1}
\end{aligned}
$$

pf

- Let $P \in P_{m}$ be $a^{\text {th }}$ arder polynomial

$$
\begin{aligned}
\left|D_{h} n-D_{n}\right| & \leq|D n-D p|+\left|D p-D_{h} n\right| \\
& =\left\|+\sum_{j}\right\| S_{j i}\left(\rho_{j}-\frac{u_{j}}{u_{j}}\right) \\
& \leq\left\|+\sum_{j}\left|S_{j i}\right|\right\| \rho_{j}-u_{j} \mid \\
L_{(n)=\max _{x \in \Omega} n(x)} & \leq\left\|D_{n-D p}\right\|_{L_{\infty}}+\|n-p\|_{L^{\infty}} \Sigma\left|S_{j i}\right| \\
& \leq\|D n-D p\|_{L^{\infty}}+C h^{-1 \alpha \mid}\|n-p\|_{L^{\infty}}
\end{aligned}
$$

Chooking $\rho$ as the taglar exparsion of $u$ at $x_{i}$

$$
\begin{array}{lll}
0 & 0 & 0 \\
< & \| u-p \mu_{L} \infty & \leq C_{2} h^{m+1}|u|_{C^{m+1}} \\
\partial C_{2} h & & \| D^{\alpha} n-D^{\alpha} p \mu_{L} \infty  \tag{7}\\
& & C_{3} h^{m+1-|\alpha|}|u|_{C^{m+1}}
\end{array}
$$

