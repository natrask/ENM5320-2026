- Nonlinear lenomable stencil


## Lectule of 2/24

- Multi-itage integentorin
- Linew stability analysish
- Stomer-Verlet

Let's recall what we know now how to do

- Use least action to derive an energy/monature presurning ODE
- une legendre to campute a Hamiltonian
- Use discrete gradient to integrate (example to came)
- Use Pytorch to solve for non linear atencils
- Use polynomial reproduction a Noether to pot constrants on stencil

With these in hand, we are ready to assume our final form.... nonlinear wave eqn solver!

ID Wave Egn in Peridic BC
$\left\{\begin{array}{l}\partial_{t t} u=c^{2} \partial_{x x} u \\ u(x, t=0)=f(x)\end{array}\right.$
Soln given by $u(x, t)=f(x+c t)+f(x-c t)$
To confilm

$$
\begin{aligned}
& \partial_{x x} h=f^{\prime \prime}(x+c t)+f(x-c t) \\
& \partial_{t h}=c f^{\prime}(x+c t)-c f(x-c t) \\
& \partial_{i t} h=c^{2} f^{\prime \prime}(x+c t)+c^{2} f(x-c t)
\end{aligned}
$$

We will derive an architecture that can recover thin simple case

- In plevious class, we now that in the absence of the right mont node $\left(D_{n} u^{2}=D_{n}{ }^{2}\right)$ the Layrogrgen gidd the approximation $\left(\nabla_{n}^{2} \approx D_{n}^{-t} D_{n} n=D_{t} D_{-} n\right)$.

Motivated by this, we will hypotherize

Goal as enpessive an ananta as possible while setiaff ing consterits
$\left.S \alpha_{\Delta}=\sum_{i} \frac{1}{\partial} \dot{q}_{i}^{2} h-\frac{1}{2} \| N\left(D_{-} q_{i} ; \theta\right)\right)^{2} h$

- Note that this is shift inumint for $q$ and $t$
- here there were save options - we could also have explased e.g. $D_{-} N(q ; \theta)$, or $D_{-} q \Rightarrow \alpha q_{i-1}-\alpha q_{i}$ for any $\alpha$ variations of the fint term give $\delta_{g} S_{1}=-\ddot{q}_{i} h$
for second term

$$
\begin{aligned}
& \begin{aligned}
&\left(\delta_{q} S_{2}, \delta_{q}\right)=\lim _{\varepsilon=0,0} \int_{t_{0}}^{t} \frac{1}{2} \frac{1}{2}\left[N\left(D_{-}\left(q+\varepsilon \delta_{q}\right) ; \theta\right)-N\left(D_{-} z ; \theta\right)\right] \\
& \text { def of directimal derivative }
\end{aligned} \\
&=\lim _{\varepsilon \rightarrow 0} \int_{t_{0}}^{t} \frac{1}{2} \frac{1}{\varepsilon} \nabla N\left(D_{-}\left(q+\varepsilon \delta_{q}\right) ; \theta\right) \cdot D_{-}\left(\varepsilon \delta_{q}\right) d t \\
&=\int_{t_{0}}^{t} \nabla N\left(D_{-} q\right) \cdot D_{-} \delta q d t \\
&=\int_{t_{0}}^{t} \frac{D_{+} \nabla N\left(D_{-} q\right)}{\delta_{q} S_{2}} d t
\end{aligned}
$$

And we obtain the dynamics

$$
\ddot{q}_{i}=D_{+} \nabla N\left(D_{-} q ; \theta\right)
$$

We can identify the generalized momentum

$$
P_{i}(t)=\partial_{\dot{q}_{i}} L=\dot{q}_{i} h
$$

Applying the Legendre transform

$$
\begin{aligned}
H & =p_{0} \dot{q}-L \\
& =\sum_{i} \dot{q}_{i}^{2} h-\frac{1}{2} \dot{q}_{i}^{2} h-\frac{1}{2} N(D-z ; \theta)^{2} h \\
& =\sum_{i} \frac{1}{2} \dot{q}_{i}^{2} h+\frac{1}{2} N(D-z ; \theta)^{2} h \\
& =\underbrace{\sum_{i} \frac{1}{2} p_{i}^{2} h^{-1}}_{\underbrace{T_{0}}_{\text {kintic }}(p)}+\frac{\frac{1}{2} N(D-q ; \theta)^{2} h}{V_{\theta}(q)} \\
& \underbrace{N}_{\text {potential }}
\end{aligned}
$$

Want to solve

$$
\begin{aligned}
& \frac{d P}{d t}=-\partial q V_{\theta}(q) \\
& \frac{d q}{d t}=\partial p T_{\theta}(p)
\end{aligned}
$$

Some final remarks on time integration

To choose an integration scheme (so far weonly looked at explucit Eder.

Consider a lineas system of ODEs

$$
\dot{y}=A y
$$

- Depending on the eigenvalues of $A_{1}$ the ODE will have distinct character
- Haperkolic PDE collempands to puely imaginary
- Well next coves same optians that can hardle preely imaginny eigenvales
![](https://cdn.mathpix.com/cropped/9131a838-ca2e-44c0-88d5-77aba82993da-04.jpg?height=780&width=751&top_left_y=1071&top_left_x=1344)

Time
Inteyration

Connider now nor-linew $x^{\ddot{x}}=F(x ; \theta)$
Two broad classes for expleit wethods

Multi-stage schemes Why is explicit inpordent for
ML?
$\_\_\_\_$
$k$

X
$k_{i}=F\left(t_{n}+c_{i} h, x^{n}+h \sum_{j=1}^{1} a_{i j} k_{j}\right)$
idea at each stage you can make an additional gradient evaluation using points generated in previous stages

Multi-step schemes
$\sum_{j=0}^{5} a_{j} x^{n+1-j}=h \sum_{j=0}^{S-1} b_{j} f\left(t_{n-j}, x_{n-j}\right)$
$-a_{0}=1$

- if $a_{i}=0$ for $j>0$, scheme is explicit
- idea use information about derivative from previous timestep to prodict
Comprison - Multi-utep only needs I func eual per stap ⇒ generally faster - Complicated for finst s stepts need to stact up u/multistage - For simplicity well just use multistage


## Exmplos

RKI-expleyt Euler

$$
\begin{aligned}
& K_{1}=F\left(t_{n}, x_{n}\right) \\
& x_{n+1}=x_{n}+h K_{1}
\end{aligned}
$$

$\partial^{\text {nd }}$ order solvemes
After E.E. for stage 1, mate arsate

$$
\begin{aligned}
& k_{2}=F\left(t_{n}+\alpha h_{1} x_{n}+\beta k_{1}\right) \\
& x_{n+1}=x_{n}+h\left(a k_{1}+b k_{2}\right)
\end{aligned}
$$

To choose, expand in Taylar neries

$$
\begin{array}{ll}
F\left(t_{n}+\alpha h, x_{n}+\beta k_{1}\right) & \alpha^{\text {becanse }} k_{1}=F\left(t_{n}, x_{n}\right) \\
=F\left(t_{n}, x_{n}\right)+\partial_{t} F\left(t_{n}, x_{n}\right) \alpha h & +\partial_{x} F\left(t_{n}, x_{n}\right)^{\alpha} \beta \Delta t
\end{array}
$$

And so denote $F_{n}=F\left(t_{n}, x_{n}\right)$

$$
\begin{aligned}
x_{n+1} & =x_{n}+h\left(a K_{1}+b K_{2}\right) \\
& =x_{n}+h a F_{n}+h b F_{n}+\alpha b h^{2} \partial_{t} F_{n}+\beta b h^{2} \partial_{x} F_{n} \\
& =x_{n}+h(a+b) F_{n}+\frac{1}{2} h^{2}(\partial \alpha b) \partial_{t} F_{n}+\frac{1}{2} h^{2}(\partial \beta b) \partial_{x} F_{n}
\end{aligned}
$$

Compure to linear terms in T.S exponsion

$$
\begin{aligned}
& a+b=1 \\
& \alpha b=1 / 2 \\
& \beta b=1 / 2
\end{aligned}
$$

Non-unique!
RKZ take $a=b=1 / 2$

$$
\alpha=\beta=1
$$

For the general scheme

$$
\begin{aligned}
& x_{n+1}=x_{n}+h \sum_{i=1}^{5} b_{i} k_{i} \\
& k_{i}=F\left(t_{n}+h c_{i}, x_{n}+h \sum_{j=1}^{i-1} a_{i} ; k_{j}\right)
\end{aligned}
$$

7 AFA How to lead off wikipedia

Coefficients ase witten compactly as a Butcher tableau
![](https://cdn.mathpix.com/cropped/9131a838-ca2e-44c0-88d5-77aba82993da-07.jpg?height=496&width=644&top_left_y=679&top_left_x=476)

## First order mathods

![](https://cdn.mathpix.com/cropped/9131a838-ca2e-44c0-88d5-77aba82993da-07.jpg?height=212&width=263&top_left_y=1373&top_left_x=155)

Only ore choice → our firend EE.

| Jud order | wethods |  |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| $\alpha$ | $\alpha$ | 0 |
|  | $\left(1-\frac{1}{2 \alpha}\right)$ | $\frac{1}{2 \alpha}$ |

Notable examples
for any $\alpha$

$$
\begin{array}{ll}
\alpha=\frac{1}{2} & \text { explicit molpt } \\
\alpha=1 & \text { Heun } \\
\alpha=2 / 3 & \text { Raliton }
\end{array}
$$

| $\frac{3^{\text {sd }} \text { osder }}{}$ |  |  |  |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |
| $\alpha$ | $\alpha$ | 0 | 0 |
| $\beta$ | $\left(\frac{\beta}{\alpha} \frac{\beta-3 \alpha(1-\alpha)}{3 \alpha-2}\right)$ | $\left(-\frac{\beta}{\alpha} \frac{\beta-\alpha}{(3 \alpha-2)}\right)$ | 0 |
|  | $1-\frac{3 \alpha+3 \beta-2}{6 \alpha \beta}$ | $\frac{3 \beta-2}{6 \alpha(\beta-\alpha)}$ | $\frac{\alpha-3 \alpha}{6 \beta(\beta-\alpha)}$ |

Note
At this pt we should get excited when we gee parameterized schove?

## $4^{\text {th }}$ osder

## Atandard RK4

| 0 | 0 | 0 | 0 | 0 |
| :---: | :---: | :---: | :---: | :---: |
| $1 / 2$ | $1 / 2$ | 0 | 0 | 0 |
| $1 / 2$ | 0 | $1 / 2$ | 0 | 0 |
| 1 | 0 | 0 | 1 | 0 |
|  | $1 / 6$ | $1 / 3$ | $1 / 3$ | $1 / 6$ |

To undestand why this
is the default, we noed to analyze the stability. Well see than works for poiely smayinary pioblems
civen
Stability of multi-strige schemes

Consider again $y^{\prime \prime}=A_{y}$

Lets analyze RK2

$$
\begin{aligned}
y_{n+1} & =y_{n}+\frac{h}{2}\left(K_{1}+K_{2}\right) \\
K_{1} & =A y_{n} \\
K_{2} & =F\left(t_{n}+h, y_{n}+h K_{1}\right) \\
& =A\left(y_{n}+h K_{1}\right) \\
& =A y_{n}+h A^{2} y_{n}
\end{aligned}
$$

So $y_{n+1}=\left(I+h A+\frac{h^{2}}{2} A^{2}\right) y_{n}$
![](https://cdn.mathpix.com/cropped/9131a838-ca2e-44c0-88d5-77aba82993da-08.jpg?height=308&width=1020&top_left_y=2208&top_left_x=480)

Note What dives the choices of coeffin in RK in that $Q \approx \exp (A)$

Thun

Consider an arbitrary icheme, where $Q$ is didyonalizable

$$
y_{\text {not }}=Q^{n} y_{0}
$$

Then if $\lambda_{i}$ is the $i^{\text {th }}$ max eigenvalve of $Q$

$$
\max _{i}\left|\lambda_{i}\right| \leq 1 \Rightarrow\left\|y_{n_{1}}\right\| \leq\left\|y_{0}\right\|
$$

ff Nonsingular $\Rightarrow Q=S \Lambda S^{-1}, \Lambda=\operatorname{ding}\left(\lambda_{1}, \ldots \lambda_{N}\right)$

$$
\begin{aligned}
y_{n} & =\left(S \Lambda S^{-1}\right)\left(S \Lambda S^{-1}\right) \cdots\left(4 \Omega S^{-1}\right) y_{0} \\
& =S \Lambda^{n} S^{-1} y_{0} \\
S^{-1} y_{n} & =\Lambda^{n} S^{-1} y_{0}
\end{aligned}
$$

os if $w_{n}=s^{\prime \prime} y_{n}$

$$
w_{n}=\Lambda^{n} w_{0}
$$

Composentwise

$$
w_{n, i}=\lambda_{i}^{n} w_{0, i}
$$

Which is buded if max $\lambda: \leq 1$

- How does this translate to RKQ?
- Recall def spectial radius

$$
e(A)=\max \left\{\left|\lambda_{i}\right|, \lambda_{i} \text { is e.j. of } A\right\}
$$

- Want

$$
l(Q)=1+h e(A)+\frac{h^{2}}{2} e(A)^{2} \quad \text { to be } \leq 1
$$

Let $z=h e(A) \in \mathbb{C} \leftarrow$ complex \#'s

Solve for $z$ s.t

$$
g(z)=1+z+\frac{1}{2} z^{2} \leq 1
$$

- In general guarly to do by hand. Simple to meshyed in matplot lib

RK1
we allieadz zaw thes ally works w/rabilization
![](https://cdn.mathpix.com/cropped/9131a838-ca2e-44c0-88d5-77aba82993da-10.jpg?height=581&width=831&top_left_y=1477&top_left_x=384)

RK2 4tretch $b y 3 / 2$ in impust
![](https://cdn.mathpix.com/cropped/9131a838-ca2e-44c0-88d5-77aba82993da-10.jpg?height=571&width=718&top_left_y=1538&top_left_x=1312)

RK3
![](https://cdn.mathpix.com/cropped/9131a838-ca2e-44c0-88d5-77aba82993da-10.jpg?height=484&width=769&top_left_y=2142&top_left_x=376)

Rk4
![](https://cdn.mathpix.com/cropped/9131a838-ca2e-44c0-88d5-77aba82993da-10.jpg?height=577&width=665&top_left_y=2105&top_left_x=1300)

## Symplectic Inteniators

Recall that for a canarical Hamilturion sy item

$$
\begin{aligned}
d \dot{p} & =-\partial_{g} H \\
\dot{q} & =\partial \rho H, \quad \frac{d H}{d t}=0
\end{aligned}
$$

(we now know how to
ident. If stercils that moke
Hamiltianians) identif sterity
Hamilianians)

Assuming the decomporition

$$
H=\underset{\substack{\text { kinetic } \\ \uparrow}}{\underset{\text { potatial }}{\text { potal }}}
$$

Then we way to solve

$$
\begin{aligned}
& \dot{p}=-\partial_{z} V \\
& \dot{q}=\partial_{p} T
\end{aligned}
$$

Stomer-Verlet / Leapfrog integrator

Ouly works for reparable Hamiltorian, n. dissipation

