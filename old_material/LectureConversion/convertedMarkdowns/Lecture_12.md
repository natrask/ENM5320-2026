Todgs End of FDM Onward to FEM

Before switching to FEM + Graphs in remainder of class, lets pause to dizcuss what we can and can't do w/ FDS for ML
(1) Prove stability, energy cons, mom cons
(2) Use Lax equiv theorem + polynamial reproduction to guarantee convergence
(3) Integrate Hamiltaniar systems w/ discrete energy cons. of use $\mathrm{RKZ}^{+}$to get stable, convergentsolas
(4) Pose physics learning as nonlinew perturbation of "nice" problem In engineering " $V+V$ " is bedoek of trust in sm. verification - is code correctly implemented, using unit
![](https://cdn.mathpix.com/cropped/5102db92-c84c-4fa3-b645-08ae63d540d9-01.jpg?height=322&width=238&top_left_y=1861&top_left_x=3)
test to verify repraduction of the claims theory provides (math $\Rightarrow$ goad code) valitation - is code predictive of experimental data (good code $\Leftrightarrow$ good model)
We are not able to
(1) Handle complex geometries, non-periodic $B C$, refore our approx, etc.

⇒ FEM

Some FEM histary (Get sone perspective away foom MC hype!)
71915-Boris Galerkin develged FEM to make subs
D1909 - Walter Ritz eatabliches Rayleigh-Ritz for variational mechnics
No odoption 30's - Courant solves PDEs using in industry piecewise polynamial functions, enily theory in 40 s
![](https://cdn.mathpix.com/cropped/5102db92-c84c-4fa3-b645-08ae63d540d9-02.jpg?height=178&width=105&top_left_y=901&top_left_x=88)

GO'S - Mathematical formalism
Called FEM by Clough in 1960 ulift from "spring-mash elements" to what we will learn todny
80's - computers catch up, early Abacus

$\frac{\text { ML Histery }}{1457 \text { First MLP (Rumelhat, Hirton, Willims) }}$

Filst "PINN" paper (Lagasis 1998)

## 10 FEM

Consider the poisian problem w/ Diricklet BC
(p) $\left\{\begin{array}{l}-u^{\prime \prime}=f \\ u(0)=u(1)=0\end{array}\right.$
ex 401

$$
\begin{aligned}
& f=1 \\
& u(x)=\frac{x(x-1)}{2}
\end{aligned}
$$

- Finite differences assume too much regularity
in solution $\rightarrow u \in C^{2}([0,1])$
- Weak form poses the problem w/out reguiring regularity

Note weak $\neq$ bad or not strang actually a more general class of solutions
"Galerkin cookhook" "error measurement"
(1) Take an whitury function $V w / v(0)=v(1)=0$
(2) Multiply (P) and integiate

$$
-\int_{0}^{1} u^{\prime \prime} v d x=\int_{0}^{1} f v d x
$$

(3) Integlate by parts until we get the least restrictive derivatives on $u$ and $w$ possible

$$
\int_{0}^{1} u^{\prime} v^{\prime} d x-\left(u^{\prime}(1) v(1)-u^{\prime}(0) v(0)\right)=\int_{0}^{1} f v d x
$$

By Diridet $B C$

Thin defines the Galerkin-form of PI 4 Choose a function space $V$
(G) Find $u \in V$ such that for any $V \in V$

$$
\int_{0}^{1} u^{\prime} v^{\prime} d x=\int_{0}^{1} f v d x
$$

Remaks - there is a symmetiy similar to what weive seen w/ $D_{h}, D_{h}$ in the FDM

- how do we choose the space of functions $V$ ?

It holds for any $v$, pick $v=u$
(E.6) $\int_{0}^{1} u^{2}=\int_{0}^{1} f u \quad$ Needs to be well-defined

Define rome neeful common function spaces

$$
\begin{aligned}
& L^{2}([0,1])=\left\{f, \quad \int_{0}^{1} f^{2} d x<\infty\right\} \\
& H^{\prime}([0,1])=\left\{f \in L^{2}, \quad f^{\prime} \in L^{2}\right\} \\
& H_{0}^{\prime}([0,1])=\left\{f \in H^{\prime}, \quad f(0)=f(1)=0\right\}
\end{aligned}
$$

LHS of EO finite if $u \in H^{\prime}$
RHS $|S f u|^{2} \leq S|f u|^{2} \leq\left.|f|^{2}|u|\right|^{2}<\infty$ if $f \in L^{2} n \in L^{2}$
$H_{0}^{\prime} \Rightarrow$ Don't massy about the boundny tems during IBP

$$
\Rightarrow V=H_{0}^{\prime}
$$

But different for other PDES

What waken EEM "finite"?

Choose $V_{h} \subseteq V, \operatorname{dim}\left(V_{h}\right)=N$
(aka pick a basis w/N shape functions)
$u \in V_{h} \Rightarrow u(x)=\sum_{i} \hat{u}_{i} \phi_{i}(x)$
![](https://cdn.mathpix.com/cropped/5102db92-c84c-4fa3-b645-08ae63d540d9-05.jpg?height=267&width=562&top_left_y=653&top_left_x=849)

Many choices
$\phi_{i}$
discartinuous piecewise constant
piecewice polynonial
Finite Volume

Continuour
polynomial
(Bubnou) Galerkin

Trig/ Porthogowal
polynomial
Gausianans
spectral Element

Meihfree

Take $V_{h}=\{$ piecewise linear functions $\}$

$$
\begin{aligned}
& X_{h}=\left\{i h, i=0,1, \ldots N_{e l}\right\} \\
& e_{i+1}=[i h,(i+1) h]
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/5102db92-c84c-4fa3-b645-08ae63d540d9-06.jpg?height=229&width=776&top_left_y=532&top_left_x=1244)

Let $\phi_{i}=\left\{\begin{array}{cl}\frac{x-x_{i-1}}{x_{i}-x_{i-1}} & x \in e_{i} \\ 1-\frac{x-x_{i}}{x_{i+1}-x_{i}} & x \in e_{i+1} \\ 0 & \text { else }\end{array}\right.$
![](https://cdn.mathpix.com/cropped/5102db92-c84c-4fa3-b645-08ae63d540d9-06.jpg?height=383&width=795&top_left_y=823&top_left_x=1205)

Rewrite (G) Find $u \in V_{h}$ such that for any $v \in V_{h}$

$$
\Rightarrow \quad \int_{0}^{i} u^{\prime} v^{\prime} d x=\int_{0}^{i} f v d x
$$

As $\Rightarrow$ Find $\hat{u} \in \mathbb{R}^{N_{e l}}$ s.t. $\forall \hat{v} \in \mathbb{R}^{N_{e l}}$

Grap $\sum_{i} \hat{v}_{i}[\underbrace{\int_{0}^{\prime} \phi_{i}^{\prime} \phi_{j}^{\prime} d x}_{S_{i j}} \hat{u}_{i}-\int_{0}^{1} \phi_{i} f d x]=0$
True if $\rightarrow S \hat{u}=b, \quad b_{i}=\int_{0}^{1} \phi_{i} \cdot f d x$

Thum FEM discretization of (G) is noique 7

If - Let $u_{1}, u_{2}$ solve $G$

$$
\begin{aligned}
& \int_{1}^{1} u_{1}^{\prime} v^{\prime} d x=\int_{0}^{1} f v d x \\
& \int_{0}^{1} u_{2}^{\prime} v d x=\int_{0}^{1} f v d x \\
& \int_{0}^{1}\left(u_{1}-u_{2}\right)^{\prime} v^{\prime} d x=0
\end{aligned}
$$

for any $v \in V_{h}$

Pick $v=n_{1}-n_{2}$

$$
\int_{0}^{1}\left[\left(u_{1}-u_{2}\right)^{1}\right]^{2}=0
$$

$\Rightarrow\left(u_{1}-u_{2}\right)^{\prime}=0 \quad$ pointwise

$$
u_{1}-u_{2}=\text { constant }
$$

if $V_{h} \subseteq H_{0}^{\prime}, \quad u_{1}-u_{2}=0$ on houndwy
$\Rightarrow u_{1}-u_{2}=0$ everywhere
Thun $S$ is sym. pos. def.
ff $\quad \hat{y}^{\top} S \hat{y}=\sum_{i, j} \hat{y}_{i} \int \phi_{i}^{\prime} \phi_{j}^{\prime} d y_{j} d x \hat{y}_{j}$

$$
=\int y^{\prime 2} d x \geq 0
$$

$w$ / equality anly if $y=0$

So we can solve the matrix proulem, but how to \& Construct it?
$S_{i j}=\int_{0}^{1} \phi_{i}^{\prime} \phi_{j}^{\prime} d x$

- only non-zeso if $\operatorname{supp}\left(\phi_{i}\right) \cap \operatorname{supp}\left(\phi_{j}\right) \neq \phi$
- $\frac{\text { Most of } S_{i j} \text { we } O}{\text { "sparse" }}$
![](https://cdn.mathpix.com/cropped/5102db92-c84c-4fa3-b645-08ae63d540d9-08.jpg?height=559&width=692&top_left_y=723&top_left_x=1063)

Quadrature (aka how to integrate)
def A quad, rule is a set of $N_{q}$
$W_{i}$ - quad weights
$x_{i}$ - quad points
Such that

$$
\int_{-1}^{1} f(x) d x=\sum_{i=1}^{N_{z}} f\left(x_{i}\right) w_{i}
$$

That is exact for some class of functions

## (M) Gavis - Legender QR

Exact for polynamill $f$ of deyree $2\left(N_{q}-1\right)$

- $X_{i}$ are zeros of Legendre polynamials
$-w_{i}=\frac{2}{\left(1-x_{i}^{2}\right)\left[P_{n}^{\prime}\left(x_{i}\right)\right]^{2}}$

$$
\begin{array}{ccc}
\text { Nq } & x_{i} & w_{i} \\
1 & 0 & 2 \\
2 & \pm \sqrt{3} & 1 \\
3 & 0, \pm \sqrt{3 / 5} & 8 / 9,5 / 9,5 / 9
\end{array}
$$

## Change of variables

To map damain of integration from $[-1,1]$ to $[a, b]$

$$
\begin{aligned}
u \text {-substitution } & u=2\left(\frac{x-a}{b-a}\right)-1 \\
\int_{a}^{b} f(x) d x & =\int_{-1}^{1} f\left(\frac{b-a}{2} x+\frac{a+b}{2}\right) \frac{b-a}{2} d x \\
& =\frac{b-a}{2} \sum_{i=1}^{n} w_{i} f\left(\frac{b-a}{2} x_{i}+\frac{a+b}{2}\right)
\end{aligned}
$$

Connection between energy minimization \& Galerkin

From klaes Tohniton)
Consider the functional

$$
F(v)=\frac{1}{2}\left(v^{\prime}, v^{\prime}\right)-(f, v)
$$

Define Ritz method (R)

$$
\min _{v \in V} F(v)
$$

$$
\left.\begin{array}{l}
\text { To solve, use our old frind } \\
\begin{array}{rl}
\left(\delta_{v} F(v), \delta v\right)= & \lim \frac{1}{2 \varepsilon}(F(v+\varepsilon \delta v)-F(v)) \\
= & \lim _{\varepsilon \Delta 0} \frac{1}{\varepsilon}
\end{array}\left[\frac{1}{2}\left(v^{\prime}+\varepsilon \delta v^{\prime}, v^{\prime}+\varepsilon \delta v^{\prime}\right)-\frac{1}{2}\left(v^{\prime}, v^{\prime}\right)\right. \\
\\
-(f, v+\varepsilon \delta v)+(f, v)]
\end{array}\right] ~ \lim _{\varepsilon \Delta 0} \frac{1}{\varepsilon}\left[\varepsilon\left(v^{\prime}, \delta v^{\prime}\right)+\frac{\varepsilon^{2}}{2}\left(\delta v^{\prime}, \delta v^{\prime}\right)-\varepsilon(f, \delta v)\right] ~ \text { For any Swev }
$$

Remark Not oll PDES can be witten as an energy min.

Whats speciel alout Galekin - optimal error
If the exact soln is $u \in V$, $\mathrm{FEM} u_{h} \in V_{h}$

$$
\begin{aligned}
& \left(u_{1}^{\prime}, v^{\prime}\right)=\left(f_{1} v\right) \\
& \left(u_{h}^{\prime}, v^{\prime}\right)=\left(f_{1} v\right) \\
& \left(u^{\prime}-u_{h}^{\prime}, v^{\prime}\right)=0
\end{aligned}
$$

← Galerkin
Orthogonality

Thm For any $v \in V_{h}$

$$
\left\|\left(n-u_{n}\right)^{\prime}\right\| \leq\left\|(n-v)^{\prime}\right\|
$$

Soln is best possible in your space of fuctions

Pf Let $v \in V_{h}$ and $w=u_{h}-v \in V_{h}$

$$
\begin{aligned}
\left\|\left(u-u_{n}\right)^{\prime 2}\right\|^{2} & =\left(\left(u-u_{n}\right)_{1}^{\prime}\left(u-u_{w}\right)^{\prime}\right) \\
& =\left(\left(u-u_{n}\right)_{1}^{\prime}\left(u-u_{w}\right)^{\prime}\right)+\left(\left(u-u_{n}\right)^{\prime}, w^{\prime}\right) \quad \text { galectin } \\
& =\left(\left(u-u_{n}\right)_{1}^{\prime},\left(u-u_{n}+w\right)^{\prime}\right) \\
& =\left(\left(u-u_{n}\right)^{\prime},(u-v)^{\prime}\right) \\
& \leq\left\|\left(u-u_{k}\right)^{\prime}\right\| \quad\left\|(u-v)^{\prime}\right\|
\end{aligned}
$$

Handling Neumann BCs

$$
\begin{aligned}
& u^{\prime \prime}=1 \\
& u(0)=0 \\
& u^{\prime}(1)=0
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/5102db92-c84c-4fa3-b645-08ae63d540d9-12.jpg?height=352&width=433&top_left_y=295&top_left_x=1056)

Soln $u(x)=x\left(1-\frac{x}{2}\right)$
Returning to ariginal form of PDE

$$
\begin{aligned}
& -\int u^{\prime \prime} v d x=\int f v d x \\
& \int u^{\prime} v^{\prime} d x-(\underbrace{\prime}_{r}(1) v(1)-u^{\prime}(0) v(0))=\int f v d x \\
& \text { If } u^{\prime}(1)=g,
\end{aligned}
$$

Substitute it
![](https://cdn.mathpix.com/cropped/5102db92-c84c-4fa3-b645-08ae63d540d9-12.jpg?height=585&width=824&top_left_y=1724&top_left_x=140)
in

