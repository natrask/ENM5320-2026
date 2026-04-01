Lecture $5-2 / 12$

- Intoo to Hamiltanian dynamies
- Symplectic atructure
- Machine learning (continuous) Hamiltanions
- the didrete gradient method

In Mardogs lecture, we stepped through ooding up a generic FD noulinar stencil fitter

It worked (most of the time) - but why?
Today well talk about building in energy conservation exactly

Lemma If $A=-A^{\top}, x^{+} A x=0 \quad \forall x$

PE $x^{\top} A x=\frac{1}{2} x^{\top}(A+A) x=\frac{1}{2} x^{\top}\left(A-A^{\top}\right) x$

$$
=\frac{1}{2} x^{\top}(A-A) x=0
$$

## Conservation of Energy

Let $q \in \mathbb{R}^{N}$ be vector of generalized position $P \in \mathbb{R}^{N}$ be vectio of generalized momentum. $x=\binom{q}{p} \in \mathbb{R}^{2 N}$ be the state of system

The system is canouically Hamiltorian if

$$
\begin{aligned}
& \frac{d p}{d t}=-\frac{\partial H}{\partial q} \\
& \frac{d q}{d t}=\frac{\partial H}{\partial p}
\end{aligned} \quad \text { for } \quad H(q, p) \in \mathbb{R}
$$

Thum A canarically Hamiltonian system conserves Hg i.e

$$
\frac{d H}{d t}=0
$$

Pf $\quad \frac{d H}{d t}=\frac{\partial H}{\partial p} \frac{d p}{d t}+\frac{\partial H}{\partial q} \frac{d q}{d t}$

$$
\begin{aligned}
& =-\frac{\partial H}{\partial p} \frac{\partial H}{\partial q}+\frac{\partial H}{\partial q} \frac{\partial H}{\partial p} \\
& =0
\end{aligned}
$$

## Ex Nonlinear pendulum

- Newtor

$$
\begin{aligned}
& m \ddot{x}=-m g \sin \theta \\
& \text { st. }\|x\|=L
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/01f579d6-b701-407c-9c04-94d3a58907ee-03.jpg?height=471&width=407&top_left_y=125&top_left_x=1380)

- Can enforce constraint by switching to polar coordinates and assuming $x=L \theta(t)$

$$
\begin{aligned}
m L \ddot{\theta} & =-m g \sin \theta \\
\ddot{\theta} & =-\lambda^{2} \sin \theta, \quad \lambda=\sqrt{\frac{g}{L}}
\end{aligned}
$$

Small $\theta$ limit

$$
\begin{aligned}
& \sin \theta \approx \theta \\
& \ddot{\theta}=-\lambda \theta
\end{aligned}
$$

Letting $q=\theta$.

$$
\begin{aligned}
& p=\dot{\theta} \\
& H=\frac{1}{2} p^{2}+\frac{\lambda^{2}}{2} q^{2}
\end{aligned}
$$

Then

$$
\begin{aligned}
& \partial_{p} H=p, \quad \partial_{q} H=\lambda^{2} q \\
& \frac{d p}{d t}=-\partial_{q} H=-\lambda^{2} q \\
& \frac{d q}{d t}=\partial_{p} H=q
\end{aligned}
$$

For the nonlinear case, there is a trick to identify Hamiltanions of the form of dymmics

$$
\ddot{\theta}+F(\theta)=0
$$

if $F$ is a function w/ simple artideriative

$$
\frac{d}{d t}\left[\frac{1}{2} \dot{\theta}^{2}+\int^{\theta(t)} F(\phi) d \phi\right]=[\ddot{\theta}+F(\theta)] \dot{\theta}
$$

Recall

$$
\frac{d}{d x} \int_{a}^{x} f=f(x)
$$

Motivated by this, multiply our eqn by $\dot{\theta}$

$$
\begin{aligned}
& \ddot{\theta} \dot{\theta}+\lambda^{2} \dot{\theta} \sin \theta=0 \\
= & \frac{d}{d t}\left(\frac{1}{2} \dot{\theta}^{2}-\lambda^{2} \cos \theta\right)=0 \\
\Rightarrow & \frac{1}{2} \dot{\theta}^{2}-\lambda^{2} \cos \theta=\text { const. }
\end{aligned}
$$

Taking

$$
\begin{aligned}
& H=\frac{1}{2} p^{2}-\lambda^{2} \cos q \\
& \frac{d p}{d t}=-2 q H=-\lambda^{2} \sin q \\
& \frac{d q}{d t}=2 p H=p
\end{aligned}
$$

- H is often called an integnl or inveriant of the dynamich
- The whole game is to reverse engineer an $H$ which, for a good choice of P.q. gives our dynamics ( $\begin{gathered}\text { typionly } \\ \text { anereagy, }\end{gathered}$
- Thin is hard in general, and will becare involved for PDE,
- This in not hard if we machine lean $H_{1}$ and use it to steer the uystem dynamics

Phase Diagram of Koln
![](https://cdn.mathpix.com/cropped/01f579d6-b701-407c-9c04-94d3a58907ee-06.jpg?height=786&width=1546&top_left_y=280&top_left_x=169)

Qualitative fentures of solu

- Separatrix denoter switch between small rocking sscillations and pendulum flying asound forever
- Inner loops are closed so that solutions should repeat forever
We'd like to be able to recover this in an ML model


## Symplectic structure

Note that we can compactly wite

$$
\begin{aligned}
& \dot{x}=S(x) \nabla_{x} H \\
& S(x)=\left(\begin{array}{cc}
0 & -I \\
I & 0
\end{array}\right), \quad \nabla_{x} H=\binom{\partial p H}{\partial q H}
\end{aligned}
$$

Recall the Gauss div theorem

$$
\int_{\Omega} \nabla \cdot F=\int_{\partial \Omega} F \cdot d A
$$

Taking $F=x, x \in \mathbb{R}^{d}$

$$
\int_{J_{r}} x \cdot d A=\int_{\Omega} \nabla \cdot x=d|\Omega|
$$

So that $S x-d A$ correnpunds to the area of $\Omega$ times dimension

$$
\begin{aligned}
\frac{d}{d t} \int_{2 \Omega} x \cdot d A & =\int_{2 \Omega} \dot{x} \cdot d A \\
& =\int_{2 \Omega} S \nabla H \cdot d A \\
& =\int_{\Omega} \nabla \cdot S \nabla H d x \\
& =\sum_{i, j} \int \partial_{q_{i}} \partial_{p_{j}} H-\partial_{p_{i}} \partial_{q_{j}} H d x \\
& =0
\end{aligned}
$$

Thun, area in phase space is consesved This exactly the preperty that's violated when we over damp our hystem

## Non-Canonical Hamitonian

In general, we can consider arlitiony $S$.

$$
\begin{aligned}
& \dot{x}=S(x) \nabla_{x} H \\
& S(x)=-S(x)
\end{aligned}
$$

$S$ is often refersed to as a Poisson matrix Thm A non-canarial Hamiltarian preserves H Pt $\frac{d H}{d t}=\nabla_{x} H^{\top} \dot{x}$

$$
\begin{aligned}
& =\nabla_{x} H^{\top} S(x) \nabla_{x} H \\
& =\frac{1}{2} \nabla_{x} H^{\top} S(x) \nabla_{x} H+\frac{1}{2} \nabla_{x} H^{\top} S(x) \nabla_{x} H \\
& =\frac{1}{2} \nabla_{x} H^{\top} S(x) \nabla_{x} H-\frac{1}{2} \nabla_{x} H^{\top} S(x) \nabla_{x} H \\
& =0
\end{aligned}
$$

## Machine learning dynamich

In continum setting we can easily learn a Hamiltanian

$$
\begin{aligned}
& Q=N N_{1}(x), \quad N N_{1}: \mathbb{R}^{N} \rightarrow \mathbb{R}^{N \times N} \\
& S(x)=Q-Q^{\top} \\
& H=N N_{2}(x) \\
& \Rightarrow \dot{x}=\left(N N_{1}(x)-N N_{1}(x)^{\top}\right) \nabla_{x} N N_{2}(x)
\end{aligned}
$$

But to fit to data, we need to finite difference $\dot{x}=\frac{x^{n+1}-x^{n}}{k}$, which will either grow wea in phase space (EE) or shrink

To discretely conserve $H$, we furn to "geametric integration" theory

Ref "Geometric Numerical
Integration"
Hairer, Lubich, Wanner "On the construction of

## Discrete Gradient Method

Congider $\quad \dot{x}=S(x) \nabla H(x) \quad$ (I'll doap $x$-dep)
Discretize $\frac{x^{n+1}-x^{n}}{K}=\widetilde{S}\left(x^{n+1}, x^{n}\right) \bar{\nabla} H\left(x^{n+1}, x^{n}\right)$
Where $\lim _{x^{n+1} \rightarrow x^{n}} \widetilde{S}\left(x^{n+1}, x^{n}\right)=S\left(x^{n}\right)$

$$
\left.\lim _{x^{n+1 \rightarrow x^{n}}} \bar{\nabla} H\left(x^{n+1}, x^{n}\right)=\nabla H\left(x^{n}\right)\right\} \text { consisfuncy }
$$

and $\left(x^{n+1}-x^{n}\right) \cdot \bar{\nabla} H=H\left(x^{n+1}\right)-H\left(x^{n}\right)$

$$
\bar{\nabla} H\left(x^{n}, x^{n}\right)=\nabla H\left(x^{n}\right)
$$

Then $H\left(x^{n+1}\right)-H\left(x^{n}\right)=0$

PE

$$
\begin{aligned}
H\left(x^{n+1}\right)-H\left(x^{n}\right) & =\bar{\nabla} H^{\top}\left(x^{n+1}-x^{n}\right) \\
& =\bar{\nabla} H^{\top} \frac{\widetilde{S}\left(x^{n+1} x\right)}{k} \bar{\nabla} H \\
& =0
\end{aligned}
$$

This given a "wigh lit" for how to build S, F14

Ex 1 Harten, Lax, Van Lees, 1983

Denote

$$
\nabla H=\left(H_{x_{1}}, \ldots H_{x_{n}}\right), \quad H_{x_{i}}=\frac{\partial}{\partial x_{i}} H
$$

Define

$$
\begin{aligned}
& \bar{\nabla} H(x, y)=\left(\bar{H}_{x_{1}}, \ldots \bar{H}_{x_{n}}\right) \\
& \bar{H}_{x_{i}}=\int_{0}^{1} H_{x_{i}}((+\xi) x+\xi y) d \xi
\end{aligned}
$$

Then

$$
\begin{aligned}
& \bar{\nabla} H(x, y) \cdot(y-x) \\
& =\sum_{i=1}^{n} \bar{\nabla} H_{i}(x, y)\left(y_{i}-x_{i}\right)=\sum_{i=1}^{n} \int_{0}^{1} H x_{i}[(1-\xi) x+\xi y]\left(y_{i}-x\right)+\xi \\
& \begin{aligned}
\text { Revers the } & \\
\text { duin } & =\int_{0}^{1} \frac{d}{d \xi} H[(1-\xi) x+\xi y] d \xi \\
= & H(y)-H(x)
\end{aligned}
\end{aligned}
$$

Ex2 Itoh \& Abe, 1988
Ex 3 Gorzalez, 1996

