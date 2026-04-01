To sparially discretize Lagrangian

Lecture 7
2/19

$$
S=\iint \frac{1}{2}\left(\partial_{t} u\right)^{2}-\frac{L^{2}}{2}\left(\partial_{x} u\right)^{2} d x d t
$$

Define piecewise constant extension of a grid function

$$
\begin{align*}
& \phi_{i}(x)=\mathbb{I}_{\left(x-\frac{h}{2}, x+\frac{h}{2}\right)}(x)= \begin{cases}1 & x \in(x-w / 2, x+h / 2) \\
0 & \text { else }\end{cases} \\
& u(x, t)=\sum_{i=1}^{N} u_{i}(t) \phi_{i}(x) \\
& \partial_{x} u(x, t)=\sum_{i=1}^{N} \frac{\frac{1}{h}\left(\alpha u_{i-1}(t)+\beta u_{i}(t)+\gamma u_{i+1}(t)\right)}{D_{h} u_{i}} \phi_{i}(x) \tag{i}
\end{align*}
$$

Note that $\int u d x=\sum_{i} h u_{i}(t)$
and selecting $q_{i}=u_{i}$, we obtain the Lagrangian

$$
L=\sum_{i} \frac{1}{2} \dot{q}_{i}^{2} h-\frac{c^{2}}{2} h^{2} D_{h} q_{i}^{2}
$$

In light of Noetles's theorem
(1) Trivial time invaiance (time only enters is $\dot{q}_{i}$ )

$$
\Rightarrow H=\partial_{\dot{q}} L \cdot \dot{q}-L
$$

Where the conjugate momentum

$$
p_{i}=\partial_{q_{i}} L=\dot{q}_{i} h
$$

(2) Translation invariance $q \rightarrow q+\delta q$
only if $D_{h} \delta q=0$ for a constant
const $=\sum_{i} p_{i}=\sum_{i} h \dot{q}_{i}$ shift vector $S_{q}$

To take variations in this spatially discretized setting we need a discrete version of integration by parts

Lemma Consider two periodic grid functions $u_{i}, v_{i}$ Satisfying $u_{0}=u_{N}, v_{0}=v_{N}$

$$
\begin{array}{rlr}
\sum_{i=1}^{N} x_{i} y_{i+\alpha} & =\sum_{j=1+\alpha}^{(N+\alpha) r_{0} N} x_{j-\alpha} y_{j} & (j=i+\alpha) \\
& =\sum_{j=1}^{N} x_{j-\alpha} y_{j} \quad & \begin{aligned}
\text { (ihift limits/ } & \text { addition is commutic) } \\
& =\sum_{i=1}^{N} x_{i-\alpha} y_{i}
\end{aligned}
\end{array}
$$

Compactly in terms of shift operators

$$
\left\langle x, E^{\alpha} y\right\rangle=\left\langle E^{-\alpha} x, y\right\rangle
$$

Consiler now a given stencil operator $D_{n} n_{i}=\sum_{k=-m}^{m} C_{k} E^{k} U_{i}$
Def Define the adjoint operator $D_{h}^{*} u_{i}=\sum_{k=-m}^{m} C_{-k} E^{k} u_{i}$

Then $\left\langle D_{h}^{*} u, \delta u\right\rangle=\left\langle u, D_{u} \delta u\right\rangle$

PE

$$
\begin{aligned}
D_{h} u & =\sum_{k=-m}^{m} c_{k} E^{k} u_{i} \\
\left\langle x, D_{h} u\right\rangle & =\sum_{k=-m}^{m}\left\langle x, c_{k} E^{k} u_{i}\right\rangle \\
& =\sum_{k=-m}^{m} c_{k}\left\langle x, E^{k} u\right\rangle \\
& =\sum_{k=-m}^{m} c_{k}\left\langle E^{-k} x, u\right\rangle \\
& =\mathbb{R}\left\langle\sum_{k} c_{k} E^{-k} x, u\right\rangle \\
& =\left\langle\sum_{k} c_{-k} E^{k} x, u\right\rangle \\
& =\left\langle D_{h}^{k} x, u\right\rangle
\end{aligned}
$$

Before we attempt to tackle the full Lagrangion, lets see how the vasintions look/mirror the cart. setting

$$
\begin{aligned}
F[n] & =\frac{1}{2} \int_{\nabla n}^{2} \\
\left(\delta_{n} F, \delta_{n}\right) & =\lim _{\varepsilon \Delta 0} \frac{1}{\varepsilon}(F[n+\varepsilon \delta n]-F[n]) \\
& =\lim _{\varepsilon>0} \frac{1}{\varepsilon} \int\left[\frac{1}{2} \nabla n^{2}+\varepsilon \nabla n \cdot \nabla \delta n+\varepsilon^{12} \frac{1}{2} \nabla \delta n^{2}-\frac{1}{2} \nabla n^{2}\right] d x \\
& =\int \nabla n \cdot \nabla \delta n \\
& =(\nabla n, \nabla \delta n) \\
& =-(\nabla \cdot \nabla n, \delta n) \\
\Rightarrow \delta_{n} F & =-\nabla^{2} n
\end{aligned}
$$

$$
\begin{aligned}
\frac{\text { Dincrete }}{F_{h}[u]} & =\frac{1}{2} \int\left(\sum_{i} D_{0} u_{i} \phi_{i}(x)\right)^{2} d x \quad, D_{0} u_{i}=\frac{u_{i+1}-u_{i-1}}{2 h} \\
& =\sum_{i} \frac{h}{2}\left(D_{0} u_{i}\right)^{2} \\
\left\langle\delta_{u_{i}} F_{h}[u], \delta u_{i}\right\rangle & =\lim _{\varepsilon \rightarrow 0} \frac{1}{\varepsilon} \sum_{i}\left[\frac{h}{2}\left(D_{0} u_{i}+\varepsilon D_{0} \delta u_{i}\right)^{2}-\frac{h}{2} D_{0} u_{i}^{2}\right] \\
& =\sum_{i} D_{0} u_{i} \cdot D_{0} \delta u_{i} \\
& =\left\langle D_{0} u, D_{0} \delta u\right\rangle=\left\langle D_{0}^{*} D_{0} u, \delta u\right\rangle
\end{aligned}
$$

Now we return to the discrete wave eqn Lagrangian
Let $D_{h^{n}}=\frac{1}{h} \sum_{k=-m}^{M} C_{k} E^{k} u_{i}$
ex $\quad M=1, C_{-1}=-1, C_{0}=0, C_{1}=1 \Rightarrow D_{h}=D_{0}$
Recall

$$
\begin{aligned}
\left(\delta_{q_{i}} S_{1}, \delta q_{i}\right) & =\int m \sum_{i} \dot{q}_{i} \cdot \delta q_{i} h d t \\
& =\int-\sum_{i} \ddot{q}_{i} h \delta q_{i} d t \\
\Rightarrow \delta_{q_{i}} S_{1} & =-\ddot{q}_{i} h
\end{aligned}
$$

$$
\begin{array}{r}
\delta_{q_{i}} S=\delta_{q_{i}} S_{1}+\delta_{q} S_{2}=0 \\
\ddot{q}_{i}=-c^{2} D_{h}^{*} D_{h} q_{i}
\end{array}
$$

Remark - While tedious, no thought was needed to reach this point - just careful application of calculus

- We now have two caustraints for a good stence ${ }^{-1}$
(1) $q \rightarrow q+\delta q$ inveriance $\Rightarrow \Sigma C_{K}=0$

$$
\text { i.e } \begin{aligned}
D_{h} \delta_{q} & =\sum_{k} C_{k} E^{k} \delta_{q} \\
& =\delta q\left(\sum_{k} C_{k}\right) \quad \text { if } \delta q=\text { const } \\
& =0 \Rightarrow \Sigma C_{k}=0
\end{aligned}
$$

(2) Does-Dhi $D_{h} q$ give a stable prediction of $\nabla_{n}^{2}$ ?

- Recall foom a few lectures back, this is true if
$D_{n}^{*} D_{h} p=\nabla^{2} p$ for any quadratic $p$
-Let's translate that into a system of algebraic constraints.

$$
\begin{aligned}
& D_{h} u_{i}=\frac{1}{h} \sum_{j} C_{j} u_{i+j} \\
& D_{h}^{*} V_{j}=\frac{1}{h} \sum_{k} C_{-k} V_{j+k} \\
& D_{h}^{*} \cdot D_{h} u_{i}=\frac{1}{h^{2}} \sum_{j, k} C_{j} C_{-k} u_{i+j+k}
\end{aligned}
$$

To do arithmetic, set

$$
C_{1}=\alpha \quad C_{0}=\beta \quad C_{1}=\gamma
$$

|  |  | $j=$ |  |  |
| :--- | :--- | :--- | :--- | :--- |
| $k=$ |  | $-1$ | 0 |  |
|  |  | $\alpha^{2}$ | $\alpha \beta$ | $\alpha$ |
|  | 0 | $\beta \alpha$ | $\beta^{2}$ | $\beta \gamma$ |
|  | -1 | $\gamma \alpha$ | $\gamma \beta$ | $\gamma^{2}$ |

Reart of diagarets
![](https://cdn.mathpix.com/cropped/c315478e-75ae-4478-9124-f421ee039341-7.jpg?height=388&width=400&top_left_y=1167&top_left_x=1661)

$$
\begin{aligned}
\Rightarrow D_{h}^{*} \cdot D_{h} u_{i}=-\frac{c^{2}}{h^{2}} & {\left[(\alpha \gamma) u_{i+2}\right.} \\
& +(\alpha \beta+\gamma \beta) u_{i+1} \\
& +\left(\alpha^{2}+\beta^{2}+\gamma^{2}\right) u_{i} \\
& +(\beta \alpha+\beta \gamma) u_{i-1} \\
& \left.+(\alpha \gamma) u_{i-2}\right]
\end{aligned}
$$

(1) Noether constraint

Note to self
Leave stancil of

$$
\alpha+\beta+\gamma=0
$$

on bard
(2) Constant Reproduction

$$
2 \alpha \gamma+2 \beta(\alpha+\gamma)+\alpha^{2}+\beta^{2}+\gamma^{2}=0
$$

(3) Linear Reproduction (Take $u_{i}=0, u_{i+1}=h$, etc) $-2 \alpha \gamma h-\beta(\alpha+\gamma) h+0+\beta(\alpha+\gamma) h+2 \alpha \gamma h=0$ (Actanatically satisfied)
(4) Quadratic Reproduction

$$
\begin{aligned}
& \text { dratic Reproduction } \\
& \alpha \gamma(2 h)^{2}+\beta(\alpha+\gamma) h^{2}+0+\beta(\alpha+\gamma) h^{2}+\alpha \gamma(2 h)^{2}=-2 \\
& 8 \alpha \gamma+2 \beta(\alpha+\gamma)=-2
\end{aligned}
$$

Solving (1-4) is non-unique, hot try some simplifications
(A)

Assyndsic Extenel
$\gamma=0 \quad(1) \Rightarrow \alpha=-\beta$
(4) $\Rightarrow-2 \beta^{2}=-2$

$$
\begin{aligned}
\Rightarrow D_{h} & =-\frac{1}{h^{2}}\left(-q_{i+1}+2 q_{i}-q_{i-1}\right) \\
& =D_{+} D_{-}=D_{-} D_{+}
\end{aligned}
$$

We recover 3 -pt
Iaplacian!
(B) Symmetric

$$
\begin{aligned}
& \alpha=\gamma=-2 \beta \\
& 4 \Rightarrow \quad 8 \alpha^{2}-2 \alpha^{2}=-2 \\
& \quad \alpha=\sqrt{-\frac{1}{3}}
\end{aligned}
$$

No real valued solus!
(c)
$\alpha=-\gamma$
(1) $\Rightarrow \beta=0$
(4) $-8 \alpha^{2}=-2$

$$
\alpha=\frac{1}{2}
$$

(2) $-2 \alpha^{2}+2 \alpha^{2}+\beta^{2}=0$

$$
D_{h} n=-\frac{1}{n^{2}}\left[-\frac{1}{4} q_{i-2}+\frac{1}{2} q_{i}-\frac{1}{4} q_{i+2}\right]
$$

