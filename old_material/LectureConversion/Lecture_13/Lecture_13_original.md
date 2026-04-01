## $\begin{array}{lll}\text { Lecture 13 3/19 } & \text { - Quari-optimality } & \text { Refs } \\ & & \text { - Error equimation } \\ & & \text { - Lax-Milgram theory } \\ & & \text { - Johnson } \\ & & \text { - Brewes } \\ & & \text { + Scott }\end{array}$

Last time we intooduced the Calution diswetization of the poisen problem
(G) $(\nabla u, \nabla v)=(4, v)$ for all $u \in V_{h}$

We will now canider this as an example for a general bilinear form a

$$
a(u, v)=(f, v)
$$

$$
\begin{aligned}
a\left(\alpha u_{1}+\beta u_{2}, \gamma v_{1}+\delta v_{2}\right) & =\alpha a\left(u_{1}, v_{1}\right) \gamma+\alpha a\left(u_{1}, v_{2}\right) \delta \\
& +\beta a\left(u_{1} v_{1}\right) \gamma+\beta a\left(u_{1}, v_{2}\right) \delta
\end{aligned}
$$

For bilinear forms, we're interested in thise that Generate an energy nosm $+C-S$

$$
\|v\|_{E}=\sqrt{a(w, v)}, \quad a(v, w) \leq\|v\|_{E}\|w\|_{E}
$$

Recall from last class that this is precisely the connection between $G$ and Ragle.jh-Ritz

Solving $\sigma \underset{v \in V_{h}}{\Leftrightarrow}\|v\|_{E}^{2}$

Lant class we showed from Galerkin orthogararizy

$$
\left\|n-u_{n}\right\|_{E} \leq\|n-v\|_{E} \quad \forall v \in U_{n}
$$

or taking the inf over $V$

$$
\left\|u-u_{h}\right\|_{E} \leq i n f\|u-v\|_{E} \quad \forall v \in V_{h}
$$

## Enargy norm us $L^{2}$ worm

We know that (G) thes naturally m'nimizes erros in the energy nosm. What about is the $L^{2}$ norm?

$$
\| n-u_{n} u^{2}=\int_{\Omega}\left(n-u_{n}\right)^{2} d x
$$

We'll show $L^{2}$ error is smaller than energy error, following a duality argment
Define a new problem

$$
-w^{\prime \prime}=u-u_{h}, \quad w(0)=w(1)=0
$$

Then

$$
\begin{aligned}
\left\|u-u_{h}\right\|^{2} & =\left(u-u_{h}, u-u_{h}\right) \\
& =\left(u-u_{h},-w^{\prime \prime}\right) \\
& =\left(w^{\prime}, w^{\prime}\right)+\left[w(0)\left(u-u_{n}\right)(0)-w^{\prime}(1)\left(u-u_{n}\right)(1)\right] \\
& \left(u-u_{n}\right)
\end{aligned}
$$

$$
\begin{aligned}
& =a\left(u-u_{n}, w\right) \\
& =a\left(u-u_{n}, w-v\right)
\end{aligned}
$$

for any $v$, by Galevin orthogenality
$\leq\left\|u-u_{n}\right\|_{E}\|w-v\|_{E} \quad$ by cavely-schwartz

$$
\begin{aligned}
\left\|n-u_{h}\right\| & \leq \frac{u n-u_{n}\left\|_{E}\right\| w-v \|_{E}}{\left\|n-u_{n}\right\|} \\
& =\frac{\left\|n-u_{n}\right\|_{E}\|w-v\|_{E}}{\left\|w w^{\prime \prime}\right\|} \\
& \leq\left\|n-u_{n}\right\|_{E} \inf _{\substack{v \in v_{h} \\
\|w-v\|_{E}}}^{\| w n}
\end{aligned}
$$

If we can find a $v \in V_{h}$ such that

$$
\begin{equation*}
u w-v n_{E} \leq \varepsilon\left\|w^{\prime \prime}\right\| \tag{A}
\end{equation*}
$$

Then

$$
\left\|u-u_{n}\right\| \leq \varepsilon\left\|u-u_{n}\right\|_{E}
$$

Applying again, $\left\|u-u_{h}\right\| \leq \varepsilon^{2}\left\|u^{\prime \prime}\right\|=\varepsilon^{2}\|f\|$

So everything comes down to showing that $V_{h}$ can approximate things like $\$$

For the piecewise linear choice of $V_{h}$ we'll show what $\varepsilon$ is.

Define $0=x_{0}<x_{1}<\ldots<x_{n}=1$ a nodes.

$$
\begin{aligned}
& V_{h} \text { satisfying } \\
& \text { (1) }\left.v \in C^{0}[0,1] \quad h\right|_{\left[x_{i-1}, x_{i}\right]} \quad \text { linear polynainal } \mid x_{i+1}-x \\
& \text { (3) } v(0)=0
\end{aligned}
$$

Define functions $\phi_{i}(x), i=1, \ldots n$ as nodal basis
iatisfying $\quad \phi_{i}\left(x_{j}\right)=\delta_{i j}= \begin{cases}1 & i=j \\ 0 & i \neq j\end{cases}$
Komecker-s property
![](https://cdn.mathpix.com/cropped/8ef87879-3c4d-42e2-b4f2-e1b7feed58fe-04.jpg?height=205&width=481&top_left_y=1336&top_left_x=1643)

Define the interpolant $\pi u \in V_{h}$
satisfying $\pi u\left(x_{i}\right)=u\left(x_{i}\right) \quad \forall$ nodeh $x_{i}$
This can be computed directly

$$
\begin{aligned}
\pi u\left(x_{i}\right) & =\sum_{j} \widehat{\pi u_{j}} \phi_{j}\left(x_{i}\right)=u\left(x_{i}\right) \\
& =\sum_{j} \widehat{\pi u_{j}} \delta_{i j} \\
& =\widehat{\pi} u_{i} \\
\varepsilon_{0} \quad \pi u\left(x_{i}\right) & =\sum_{j} u\left(x_{j}\right) \phi_{j}(x)
\end{aligned}
$$

Finally, we'll show that $\pi n$ is the $v$ we neided in (A)

Thm $\|n-\pi n\| \leq c h\left\|n^{\prime \prime}\right\| \forall u \in V$, where $C$ is indep of $h, u$

Pf Work on I element, then sum up
know $\int_{x_{j-1}}^{x_{j}}(\mu-\pi n)^{\prime 2} d x \leq c\left(x_{j}-x_{j-1}\right)^{2} \int_{x_{j-1}}^{x_{j}} u^{\prime \prime 2} d x$
Let $e=u-\pi u$ be the error, note $u^{\prime \prime}=e^{\prime \prime}$ since By change of voiables

$$
x=x_{j-1}+\tilde{x}\left(x_{j}-x_{i}\right)
$$

We rewrite

$$
\begin{aligned}
& \int_{x_{j-1}}^{x_{j}} e^{i^{2}} d x \leq c\left(x_{j}-x_{j-1}\right)^{2} \int_{x_{j-1}}^{x_{j}} e^{\prime \prime 2} d x \\
& \int_{0}^{1} e^{\prime 2} d \tilde{x} \leq \int_{0}^{1} e^{\prime \prime 2} d \tilde{x} \text { equivalent }
\end{aligned}
$$

To prove this is a little bit of calculus

We'll use Rolle's theorem; which in the mean value theorem in the special ease where the endpoints are equal
Then if $e$ is continoun on $[a, b]$ and $f(a)=f(b)$, then there is at least one $p^{t} c \in[a, b]$ s.t $\quad f^{\prime}(c)=0$
![](https://cdn.mathpix.com/cropped/8ef87879-3c4d-42e2-b4f2-e1b7feed58fe-06.jpg?height=274&width=455&top_left_y=716&top_left_x=1477)

So $e^{\prime}(\xi)=0$ for same $\xi \in[0,1]$

$$
\begin{aligned}
e^{\prime}(y)-e^{\prime}(\xi) & =\int_{\xi}^{y} e^{\prime \prime} d x \quad \text { Fund Thm of } \\
& \begin{aligned}
\left|e^{\prime}(y)\right|^{2} & =\left|\int_{\xi}^{y} e^{\prime \prime} d x\right|^{2} \\
& =\left.\left|\int_{\xi}^{y}\right| \cdot e^{\prime \prime} d x\right|^{2} \\
& \leq\left|\int_{\xi}^{y} d x\right|^{y / y}\left|\int_{\xi}^{y} e^{\prime \prime} d x\right|^{y} \\
& =|y-\xi|\left|\int_{\xi}^{y} e^{\prime \prime} d x\right| \\
|e(y)| & \leq|y-\xi|^{1 / 2}\left|\int_{0}^{\prime} e^{\prime \prime} d x\right|^{\prime / 2}
\end{aligned}
\end{aligned}
$$

$$
\int e^{\prime}(y)^{2} d y \leq \int_{0}^{1}|y-\xi| d y \int_{0}^{1} e^{\prime \prime} d x
$$

Taking max over $\xi$ gives work case scenario

$$
\max \int_{0}^{1}|y-\xi| d y=\frac{1}{2}
$$

when $\xi=1 / 2$

And weire done

We just showed (4) holds w/ $\varepsilon=h^{2}$
So: $\quad\left\|u-u_{h}\right\| \leq c h^{2}\|f\|$

## Summarize

(1) $\left\|n-n_{n}\right\|_{E} \leq\|n-v\|_{E} \quad \forall v_{E} v_{h}$

Picking $v=\pi u \in V_{h}=S$

$$
\begin{aligned}
& \left\|u-u_{n}\right\|_{E} \leq c_{1} h\left\|u^{\prime \prime}\right\| \leq c_{1} h u-q \| \\
& \left\|u-u_{n}\right\| \leq c_{2} h^{2}\left\|u^{\prime \prime}\right\| \leq c_{2} h^{2}\|s\|
\end{aligned}
$$

What's up next:

- Machine learning Vh
- Machire learning $a(u, v)$

First, an abatraction of what we lewned today, so that its clear this isn't just something apecial alcout Poision

Lax-Migram theory
There in't anything special about poisian - the same process holds forang elliptic bilinew forms

Let $V$ be a Hilbart space $w /$ norm 11 Ilv inpose $a(u, v)$ satisfies
(1) Symmetric

$$
a(u, v)=a(v, u)
$$

(2) Continuous Have a cardy-saurtz

$$
\begin{aligned}
& \exists \gamma>0 \quad s . t \\
& |a(v, w)| \leq \gamma\|v\|_{v}\|w\|_{v} \quad \forall v, w \in V
\end{aligned}
$$

(3) Ellsptic (aka V-elliptic, coescive)

$$
a(v, v) \geq \alpha\|v\|_{v}^{2} \quad \forall v \in V
$$

Then

$$
\begin{aligned}
& F(n)=\min _{v} F(v) \quad \& \quad a(u, v)=L(v) \\
& F(v)=\frac{1}{2} a(v, v)-L(v) \\
& |L(v)| \leq \Lambda\|v\|_{v} \\
& \text { me } \hat{\text { solution satisfying }}^{\text {unique }} \quad\|u\|_{v} \leq \frac{\Lambda}{\alpha}
\end{aligned}
$$

Pf $\quad \min F(v) \Rightarrow a(u, v)=L(v)$ imnediate from

$$
\delta_{v} F(v)=0
$$

To oltain estimate take $v=u$

$$
a(n, n)=L(n)
$$

$\alpha \| u u_{v}^{2} \leq$
$\leq \Lambda u u u_{v}$

$$
\| u u_{v} \leq \frac{\Lambda}{\alpha} \text { Whatha }
$$

For uniquenesy absume 2 solus $u_{1}, u_{2}$

$$
a\left(u_{1}-u_{2}, v\right)=0
$$

From stability seqult

$$
\left\|n_{1}-n_{2}\right\|_{v} \leq t a \frac{1}{\alpha} \| R+s u=0
$$

For any elliptic operator, we have the plughook

- Get quasioptimality for best fit in Ullv
- Get interpolant as uppes hound
- Relate $\|\cdot\| V$ to $\|\cdot\|$ to understand how quickly error convergen

