## Today

- Intro to graphs, graph calculus
- The graph laplacian
- Graph Exteríar Calculus
- Applications

Define $G(N, E)$ as graph w/ nodes and directed edyes
$e_{i j} \in E$ points $j \rightarrow i n: \in N$ is node
![](https://cdn.mathpix.com/cropped/3db791f8-426f-4577-abe2-4efa9221e5ee-01.jpg?height=437&width=432&top_left_y=1176&top_left_x=1346)

We refer to $N, E$ as chains ( $N \rightarrow$ o-chains def $K$-chain oriented set of $K+1$ vertices $E \rightarrow 1$-chains)

$$
C^{0} \stackrel{\partial_{0}}{\leftarrow} C^{\prime}
$$

$\partial_{k}$ is a boundary opeator $\partial_{k}: C^{k+1} C^{k}$
ex $\quad \partial_{0} e_{i j}=n_{j}-n_{i}$

We can assign real numbers to chains to develop graph functions (think of the gridfunctions in FDM as a function on O-chains).
Def $f_{\text {烦 }}^{k} \in C_{k}=\mathbb{R}^{\operatorname{dim}\left(c^{k}\right)}$
↑ co-chain (note subscript)
(offen in math, co- means a real number associated w/an object)
Note $f_{i j} \in C_{1}, f_{i j}=-f_{j i}$
Def co-boundary is a mappiny

$$
d_{k}: C_{k} \rightarrow C_{k+1}
$$

ex Let $\phi \in C_{0} \rightarrow \phi=\left\{\phi_{1}, \ldots \phi_{\operatorname{dim}\left(c_{0}\right)}\right\}$
$d_{0} \phi_{i j}=\phi_{j}-\phi_{i}$ is called the graph gradient

Why? In traditional calc, $\int_{e_{i j}} \nabla u \cdot d l=u_{j}-u_{i}$ by FTC

Final ingledient
Def Co-differential is the adjoint of coloundary Note $\quad \delta_{K}: \mathbb{R}^{\operatorname{dim}\left(C_{K}\right)} \rightarrow \mathbb{R}^{\operatorname{dim}\left(C_{K+1}\right)}$

$$
\Rightarrow \delta_{K} \in \mathbb{R}^{\operatorname{dim}\left(c_{k+1}\right) \times \operatorname{dim}\left(c_{k}\right)}
$$

$V \in C_{k}, u \in C_{k+1}$

$$
\left\langle v, \frac{\left.\delta_{k}^{*} u\right\rangle}{\tau_{\text {codiffertial }}}=\left\langle\delta_{k} v, u\right\rangle\right.
$$

or more explicitly

$$
\begin{aligned}
\left\langle\delta_{k} v, n\right\rangle & =\sum_{i, j, k} \delta_{i j, k} v_{k} u_{i j} \\
& =\sum_{k} v_{k} \frac{\left(\sum_{i j} \delta_{i j, k} u_{i j}\right)}{\delta^{*} n} \\
& =\sum_{k} v_{k} \delta^{*} u_{k} \\
& =\left\langle v, \delta^{*} n\right\rangle
\end{aligned}
$$

Alternatively, we can also define $S^{*}$ w.r.t. a weighted inner product

$$
\begin{aligned}
& \left\langle v, \delta^{*} u\right\rangle_{w_{k}}=\langle\delta v, u\rangle_{w_{k+1}} \\
& \Rightarrow \delta^{*} u=w_{k}^{-1} \delta^{\top} w_{k+1} u
\end{aligned}
$$

May seem a little abstract - some examples of common problems cast in this framenork:

- Circuit analysis

Ohm's Law
$I_{i j}=R_{i j}^{-1}\left(V_{j}-V_{i}\right)$
Firchoths Law
![](https://cdn.mathpix.com/cropped/3db791f8-426f-4577-abe2-4efa9221e5ee-05.jpg?height=326&width=791&top_left_y=683&top_left_x=1034)
$\sum_{j \sim i} I_{i j}=0$

Can be recast

$$
\begin{aligned}
& I=R^{-1} \mathbb{A}_{0} V \\
& \nabla_{0}^{A} I=0 \\
\Rightarrow & \delta_{0}^{A} R^{-1} \delta_{0} V=0
\end{aligned}
$$

Graph Attention / Messaye passing networks For a gat layer

$$
\begin{aligned}
& m_{i j}^{n}=F\left(x_{i}^{n}, x_{j}^{n} \mid \theta_{m}\right) \\
& x^{n+1}=x^{n}+G\left(\sum_{j \sim i} \alpha_{i j}\left(x_{i}^{n}, x_{j}^{n}\right) m_{i j}^{n}\right)
\end{aligned}
$$

can be rewitten

$$
x^{n+1}=x^{n}+G\left(\delta_{0}^{*} m\right)
$$

where the attention is inducing the coditerential

$$
\rightarrow \delta_{m}^{*}=\delta_{0}^{\top} \alpha m
$$

![](https://cdn.mathpix.com/cropped/3db791f8-426f-4577-abe2-4efa9221e5ee-06.jpg?height=145&width=296&top_left_y=849&top_left_x=1396)

## Noter

- GATS are noteriously hard to train deep. - over squashing/over moothing
- w/ same physics ideas well show how to keep them lig
Proly 3
Reconmender systems
Given $n^{\text {ratings of }}$ othes movies someme will like

O- cochain → score
1 -cochain → movie proference
Similar graph analyties tor social networks epidemidoyny

## Stability Analysis

Let $W=\operatorname{diag}\left(w_{1,} . . . w_{\text {Neges }}\right)>0$ Define the weighted graph Laplacian
$L_{w}[u]_{i}=\sum_{j \sim i} w_{i j}\left(u_{j}-u_{i}\right)$

$$
e-g, w_{i j}=R_{i j}^{-1}
$$

Consider the problem

$$
L_{W}[n]_{i}=f_{i}
$$

We can minick the Galerkin procelvre, using the $l_{2}$-innes product in

$$
\langle f, g\rangle=\sum_{i} f_{i} g_{i}
$$

instead of integration
$\forall v \in C_{0}$

$$
\left\langle v_{n}, L_{w}[n]_{i}\right\rangle=\langle v, f\rangle
$$

LHS $\quad \sum_{i, j} v_{i} w_{i j}\left(u_{j}-u_{i}\right)$

$$
=\frac{1}{2} \sum_{i, j} v_{i} w_{i j}\left(u_{j}-u_{i}\right)+v_{i} w_{i j}\left(u_{j}-u_{i}\right)
$$

$$
\begin{aligned}
& =\frac{1}{2} \sum_{i, j} v_{i} w_{i j}\left(u_{j}-u_{i}\right)+v_{j} w_{j i}\left(u_{i}-u_{j}\right) \\
& =-\frac{1}{2} \sum_{i, j}\left(v_{j}-v_{i}\right) w_{i j}\left(u_{j}-u_{i}\right) \\
& =-\frac{1}{2}(\delta v)^{\top} w \delta u \\
& =a(u, v)
\end{aligned}
$$

Like before, we have identified a vasiational problem

$$
a(u, v)=\langle f, v\rangle
$$

w/ energy norm

$$
\|u\|_{E}=a\langle u, u\rangle=\langle u, u\rangle_{\delta^{\top} w \delta}
$$

We can therefore see that we have all of our FEM machinery.

To invoke Lax-Milgram, we need to prove coercivity of everyy norm

$$
\left.\|u\|_{E}^{2}>\alpha \quad K u, u\right\rangle
$$

## Rayleigh - Quotients + eigenamalyzis

Let $M=M^{\top} w /$ orthoganal eigenpairs $\left(\lambda_{i}, v_{i}\right)$

$$
\therefore \text { e } \quad V_{i} V_{j}=\delta_{i j}
$$

So $M M V_{i}=\lambda_{i} V_{i}$ for ith eigenpair

$$
v_{i}^{\top} M v_{i}=\lambda_{i} v^{\top} v_{i}=\lambda_{i}
$$

Def The Rayleigh-guotient

$$
R(M, x)=\frac{x^{\top} M x}{x^{\top} x}
$$

## Fact

Synum Matrices have real non-ney eigualwer

Exparding $x$ in eigen basis

$$
\begin{aligned}
& \vec{x}=\sum_{i} \hat{y}_{i} \vec{v}_{i} \\
& \hat{y}_{i}=\left\langle v_{i}, x\right\rangle
\end{aligned}
$$

i.e the basis exponsion coetts projecting $x$ onto $v$ 's

Then

$$
R(m, x)=\frac{\sum_{i} \lambda_{i} y_{i}^{2}}{\sum_{j} y_{j}^{2}}
$$

We can see that maximin of $R$ gives max/min eiganualves

$$
\lambda_{\text {min }} \leq \lambda_{\text {min }} \Sigma y_{i}^{2} \leq \frac{\sum_{i} \lambda_{i} y_{i}^{2}}{\Sigma y_{j}^{2}} \leq \lambda_{j} y_{j}^{2}
$$

Let $d_{v}$ denote the deyree of vertex $V$

## Spectral Graph Theory Fan Chung

Denote $T=\operatorname{diag}\left(d_{v}\right)_{v=1}^{N_{e d j e s}}$

Def degree Scaled Laplacion

$$
\begin{aligned}
& \mathcal{L l}\left(\omega_{1} \quad \mathcal{L}_{i j}=\left\{\begin{array}{c}
1 \\
-\frac{1}{\sqrt{d_{i j} d_{j j}}}
\end{array}\right.\right. \\
& \mathcal{L}=T^{-1 / 2} L T^{-1 / 2}
\end{aligned}
$$

Cassider the Ragleigh quotient

$$
i=j, d_{v} \neq 0
$$

ixts j~i

Eigen values of $L$ match those of $L$ icaled by dysee!

$$
\frac{\langle g, \mathcal{L} g\rangle}{\langle g, g\rangle}=\frac{\left\langle g, T^{-1 / 2}\left\langle T^{-1 / 2} g\right\rangle\right.}{\langle g, g\rangle}
$$

Let $f=T^{-1 / 2} g$

$$
=\frac{\langle f, L f\rangle}{\left\langle T^{1 / 2} f, T^{1 / 2} f\right\rangle}=\frac{\sum_{i \sim i}\left(f_{i}-f_{i}\right)^{2}}{\sum_{k} f_{k}^{2} d_{k}}
$$

As a Rayle.igh quotient, we can note

- $\lambda_{i}$ are all positive.
$-\min (\lambda)=\lambda_{0}=0$ w/ vec v= $\vec{I}$
- For a non-simply connected giaph w/ $k$-connectod compants

$$
\lambda_{0}, \ldots \lambda_{k-1}=0
$$

For a simply connected graph, the constant we need for Lax-Milgram is

$$
\alpha=\lambda_{1}=\min _{f \perp i}^{3} \frac{\sum_{i n i}\left(f_{i}-f_{j}\right)^{2}}{\sum_{k} f_{k}^{2} d_{k}}
$$

Notes - $\lambda_{1}$ is called Fiedler eiganector \& used in spectial graph theory, graph out algaitheris, diffurion problems, upectial cluateing

- this $\alpha$ only holds for $f \perp N_{0}$, so when we solve the voriational problem we need to choose goid functions orthogaral to the constant vector

Estimater for $x_{1}$
Option 1 - For a fixed graph, can always Lolve eigenvalue problem

- Arvoldi iteration gives a acalable extimate for large graphs.

Option 2 The following proof is to estimate how sealing depends on noder edyes

Some definitions:

Volume $\quad \operatorname{vol}(G)=\sum d_{i}$
distance $d\left(v_{i}, v_{j}\right)=$ ihortest puth connecting $v_{i j} v_{i}$
diameter $D=\max _{i, j \in V} d\left(v_{i}, v_{j}\right)$
Thm For simply connected graph $G$ w/ dirmeter $D$

$$
\lambda_{1} \geq \frac{1}{D \operatorname{vol}(G)}
$$

Pf - For smallest eigue $f_{0}=\overrightarrow{1}$

- By orthogmalify of eigenvectors $\left\langle f_{0}, f_{1}\right\rangle=0$

$$
\Rightarrow \sum_{i} f_{1}\left(v_{i}\right)=0
$$

- Choose $v_{0}$ as a vertex ratisfying

$$
\left|f\left(v_{0}\right)\right|=\max _{v}|f(v)|
$$

- Since $\sum_{i} f\left(v_{i}\right)=0$, we can find a vertex $u_{0}$ such that $f\left(u_{0}\right) f\left(v_{0}\right)<0$
- Pick path $P$ joining $u_{01} v_{0}$

$$
\begin{array}{r}
\lambda_{1}=\frac{\sum_{j n_{i}}\left(f_{i}-f_{i}\right)^{2}}{\sum_{k} f_{k}^{2} d_{k}} \geq \frac{\sum_{j, k \in P}\left(f_{j}-f_{i}\right)^{2}}{\operatorname{volG} f^{2}\left(\nu_{0}\right)} \\
>\frac{\frac{1}{D}\left(f\left(\nu_{0}\right)-f\left(u_{0}\right)\right)^{2}}{\operatorname{vol}(G) f^{2}\left(\alpha_{0}\right)}>\frac{1}{D \operatorname{volG}}
\end{array}
$$

Graph div/grad/cull
So far we only introduced graph grad and div
For higher order chains, extend the coboundury

$$
\begin{aligned}
& C^{k}=\left[v_{1}, \ldots v_{k}\right]_{v_{i} \in V} \quad \begin{array}{c}
\text { anti symmetric } \\
\text { wrt vertex } \\
\text { swap }
\end{array} \\
& \partial_{k-1}^{k}\left[v_{1}, \ldots v_{k}\right] \\
& \quad=\sum_{i=1}^{k}(-1)^{i-1}\left[n_{1}, \ldots \hat{n}_{i}, \ldots n_{k}\right] \\
& \quad \tau_{\text {means leave this }}
\end{aligned}
$$

ex Given edge $\left[v_{1}, v_{2}\right]=-\left[v_{2}, v_{1}\right]$ vertex out

$$
\begin{aligned}
\partial_{0}\left[v_{1}, v_{2}\right]= & (-1)^{0} v_{2} \\
& +(-1)^{1} v_{1}=v_{2}-v_{1}
\end{aligned}
$$

Thm $\partial_{k-1} \cdot \partial_{k}=0$

$$
\begin{array}{rlr}
\partial_{k-1} \partial_{k}=\sum_{i=1}^{k}(-1)^{i-1} \partial_{k-1} & {\left[n_{1}, \ldots \hat{n}_{i}, \ldots n_{k}\right]} \\
=\sum_{i=1}(-1)^{i-1}(-1)^{j-1}\left[n_{1}, \ldots, \hat{n}_{i}, \ldots \hat{n}_{i} \ldots n_{k}\right] & \hat{g} \\
=\sum_{i, 1} \alpha_{i j} & \text { antisymmetric } \\
=0 \quad \hat{n}_{i}=-\alpha_{i} &
\end{array}
$$

A simikar alyebraic definition of coboundy can be obtained

$$
\left\langle f, \partial_{k}, c\right\rangle=\left\langle d_{k} f, c\right\rangle
$$

Ex

$$
\begin{aligned}
& \delta_{0} \phi_{i j}=\phi_{j}-\phi_{i} \\
& \delta_{1} \phi_{i j k}=\phi_{i j}+\phi_{j k}+\phi_{k i}
\end{aligned}
$$

grad
curl

Similewly $\delta_{K} \delta_{K-1}=0$

Ex $\quad \delta_{1} \delta_{0} \phi=\delta_{0} \phi_{i j}+\delta_{0} \phi_{j k}+\delta_{0} \phi_{k i}$

$$
\begin{aligned}
& =\phi_{j}-\phi_{i}+\phi_{k}-\phi_{j}+\phi_{i}-\phi_{k} \\
& =0
\end{aligned}
$$

We will use these properties to define flows on graphs

