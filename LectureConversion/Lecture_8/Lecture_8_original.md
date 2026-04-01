Functional Derivatives
Newtorian, Hamiltanian, Layrangion mechanics Principle of least action
Legendo Transform
Noethers Theorem

Lecture 6
2/17

How abould we lean physica?
![](https://cdn.mathpix.com/cropped/efdda193-88f0-4aed-bc9a-aba9d421c76b-01.jpg?height=145&width=558&top_left_y=975&top_left_x=816)

$$
F=m \ddot{x}
$$

- Cood for force balance
-Hod to associnte w/invisonts
- Coordinate dependent
![](https://cdn.mathpix.com/cropped/efdda193-88f0-4aed-bc9a-aba9d421c76b-01.jpg?height=238&width=389&top_left_y=1107&top_left_x=1536)

Hamiltanian Mechanics
$\dot{x}=J \nabla H$

- Energy centria
- Good for designing schemes

Lagrangion Meehanics
$x=\underset{\tilde{x}}{\operatorname{argmin}} S(\hat{x})$
-optimization centric

- eashy constaints, treatment of syoureties

Lagrangian mechanics
Layrangian

$$
L=\underset{\substack{\uparrow \\ \text { knetic }}}{T}-\underset{\uparrow}{\text { potential }}
$$

$$
L=\int_{\Omega} \mathcal{L}[u] d x
$$

$\overbrace{\text { Lagrangion denity }}$
Action

$$
S=\int_{0}^{T} L d t
$$

Principle of least action

$$
u=\underset{\hat{u}}{\operatorname{argmin}} \int_{0}^{T} \int_{\Omega} \mathcal{L}[u] d x d t
$$

- Put informally, the path that a system will evolve through for a fixed intial and final state will be the $u(t)$ which minimizes the action.
- To make sense of this, we need to minimize expression like S. Like standard calculus: take a derivative of a function $f(x)$, set to zero to find extremal pts. Hese though, $x$ is a coordinate, $f$ a function. $n$ is a function, $i$ is a functional
$S: V \rightarrow \mathbb{R}, \overline{V \text { a appace of funtions }}$
i.e. the input space is now infinite divensional
A few definitions/options for how to do that. We'll skow both so you understand what people mean by them, but then get less alestact.


## Freduct derivative

- Let $V$ be funce space, $F: V \rightarrow \mathbb{R}$. The differential $\delta F[u]$ is defined as, for $\delta u \in V$

$$
\begin{aligned}
& F[u+\delta u]-F[u]=\delta F[u]+\varepsilon\|\delta u\| \\
& \text { such that } \lim _{\varepsilon+0} \varepsilon\left\|\delta_{n}\right\|=0
\end{aligned}
$$

- This is a challenging definition to work with


## Gateaux derivative

Iden Introduce a scalar paramets and use the vector calc def of a directional derivative

$$
\begin{aligned}
\delta F[u, \delta u] & =\lim _{\varepsilon d 0} \frac{F[u+\varepsilon \delta n]-F[u]}{\varepsilon} \\
& =\left.\frac{d}{d \varepsilon} F[u+\varepsilon \delta n]\right|_{\varepsilon=0}
\end{aligned}
$$

To compute
Given $F[n]=\int_{\Omega} f\left(x, u, D^{\infty} u\right) d x$
$\delta F$ in defined vin, $\forall \delta u$ vanishing a londry,

$$
\left(\delta F_{1} \delta u\right)=\lim _{\varepsilon \infty} \frac{F[u+\varepsilon \delta u]-F[n]}{\varepsilon}
$$

Example Kinetic energy
$k=\frac{1}{2} \int e u^{2} d x$, take var. wrst. $u$

$$
\begin{aligned}
& \left(\delta_{n} K, \delta_{n}\right)=\lim _{\varepsilon \rightarrow 0} \frac{1}{\varepsilon} \int_{\Omega} \frac{e\left(n+\varepsilon \delta_{n}\right)^{2}}{2}-\frac{e n^{2}}{2} d x \\
& =\lim _{\varepsilon \|_{0}} \frac{1}{\varepsilon} \int e\left[\frac{u^{2}}{2}+\varepsilon u \delta u+\frac{\varepsilon^{2}}{2} \delta u^{2}-\frac{e u^{2}}{2}\right] d x \\
& =\lim _{\varepsilon \rightarrow 0} \int e u \delta n+\frac{\varepsilon}{2} e \delta n^{2} \\
& =\int e u d u \\
& =\left(e u, \delta_{n}\right) \quad \text { And so we identify } \\
& \text { Be caseful physicitts will drive hy setting } \delta_{n} \quad \delta_{n} k=e u \text { (the manerin); } \\
& \text { F to dine detta → only als for cont. Eunotionals } \\
& \delta_{u} k=l u \quad \text { (the mannerin) }
\end{aligned}
$$

Further properties
Denote $\frac{\delta}{\delta_{n}} F[n]=\delta_{n} F[n]$

- Lineasity $\frac{\delta}{\delta n}(\lambda F[n]+\mu G[n])=\lambda \delta_{n} F[n]+\mu \delta_{n} G[n]$
- Productivile $\frac{\delta}{\delta n}(F[n] G[n])=\delta_{n} F G+F \delta_{n} G$
- Chain rule $\frac{\delta}{\delta n}(F[g(n)])=\frac{\delta F[g(n)]}{\delta g(n)} \frac{d g(n)}{d n}$


## The Euler-Lagrange equations

$$
\begin{aligned}
\text { Assume a } & \text { gereric functional denaity } \\
F[n] & =\int_{\Omega} f(x, n, \nabla n) d x \\
\left(\delta_{n} F, \delta_{n}\right) & =\left[\frac{d}{d \varepsilon} \int f(x, n,+\varepsilon \delta n, \nabla n+\varepsilon \nabla \delta n) d x\right]_{\varepsilon=0} \\
& =\int_{\Omega} \frac{\partial f}{\partial n} \delta n+\frac{\partial f}{\partial \nabla n} \cdot \nabla \delta n d x \\
\begin{array}{c}
\text { Chuin } \\
\text { vue }
\end{array} & =\int_{\Omega}\left(\frac{\partial f}{\partial n}-\nabla \cdot \frac{\partial f}{\partial \nabla n}\right) \cdot \delta n+\int_{2 \pi} \frac{\partial f}{\partial f / \operatorname{sut}} \text { Assume vanashing }
\end{aligned}
$$

So

$$
\begin{aligned}
\delta_{n} F & =\frac{\partial n}{\partial n}-\sum_{i} \frac{\partial x_{i}}{\partial x_{i}} \frac{\partial f}{\partial x_{i} n} \\
& =\frac{\partial f}{\partial n}-\sum_{i} \frac{\partial}{\partial x_{i}} \frac{\partial f}{\partial\left(\partial_{i} n\right)}
\end{aligned}
$$

- We can just apply this as a formula and ik.ip all the limits
- For different form $f_{1}$ get a different aqu
ex

$$
\begin{aligned}
& F[u]=\int_{0}^{1} f\left(x, u, \partial_{x} u, \partial_{x x} u\right) d x= \\
& \quad \delta_{n} F=\frac{\partial f}{\partial u}-\frac{d}{d x} \frac{\partial f}{\partial\left(\partial_{x n}\right)}+\frac{d^{2}}{d x^{2}} \frac{\partial f}{\partial\left(\partial_{x, n}\right)}
\end{aligned}
$$

Returning to the least action principle

$$
S[n]=\int_{a}^{b} \int_{\Omega} \mathcal{L}(x, u, \dot{u}) d x d t
$$

Finally we can itate that the path which admits an extremal value of $S$ iatisfies

$$
\delta_{n} S[n]=0
$$

Applying the E-L oques

$$
\frac{\partial \mathscr{L}}{\partial u}-\frac{d}{d t} \frac{\partial f}{\partial \dot{u}}=0
$$

Ex Linear pendulum

$$
\begin{aligned}
& T=\frac{1}{2} m v^{2}=\frac{1}{2} m l^{2} \dot{\theta}^{2} \\
& U=m g h \\
& h=l(1-\cos \theta) \approx \frac{1}{2} l \theta^{2} \\
& L=\frac{1}{2} m l^{2} \dot{\theta}^{2}-\frac{1}{2} m g l \theta^{2} \\
& \partial_{\theta} L=-m g l \theta \\
& \partial_{\dot{\theta}} L=m l^{2} \dot{\theta}
\end{aligned}
$$

ELLon

Again, depending on $\mathcal{L}$ we can get d. fevent versions of this

Legendse transform
A technique to switch between the Lagiangian and Hamiltanian description

Given a Laganyian

$$
L\left(q_{1}, \ldots, q_{N}, \dot{q}_{1}, \ldots \dot{q}_{N}\right)
$$

The legendie transform induces the conjugate momenta $P_{1} \ldots P_{N}$

$$
\left\{\begin{array}{l}
H\left(q_{1}, \ldots q_{N}, p_{1}, \ldots p_{N}\right)=\sum_{i} p_{i} \dot{q}_{i}-L\left(q_{1}, \ldots q_{N}, \dot{q}_{11} \ldots \dot{q}_{N}\right) \\
p_{i}=\partial_{q_{i}} \mathcal{L}
\end{array}\right.
$$

Ex Getting Hamiltonian from Lagrangion for pendulum $q=0, \quad \mathcal{L}=\frac{1}{2} m l^{2} \dot{q}^{2}-\frac{1}{2} m g l q^{2}$
$P_{i}=\partial_{i} \mathcal{L}=m l^{2} \dot{q} \Rightarrow \dot{q}=\frac{1}{m l^{2}} p$

$$
\begin{aligned}
\mathcal{L} & =\frac{1}{2}\left(m l^{2}\right)^{-1} p^{2}-\frac{1}{2} m g l q^{2} \\
H & =p q-\mathcal{L} \\
& =\frac{1}{m l^{2}} p^{2}-\frac{1}{2}\left(m l^{2}\right)^{-1} p^{2}-\frac{1}{2} m g l q^{2} \\
& =\frac{p^{2}}{2 m l^{2}}-\frac{m g l q^{2}}{2}
\end{aligned}
$$

## Coustrained Laycay.ins

def If constraint is equality and afunction of coordinate only (no $\dot{q}^{\prime} s$ ) it in holonomic

$$
f_{i}(q, t)=0
$$

We can augment the Lagrangion w/ a Lagrange multiplies

$$
\hat{L}(q, \dot{q}, \lambda)=L(q, \dot{q})-\sum_{i} \lambda_{i} f_{i}(q, t)
$$

Where $\lambda$ is a field (in contenst to KKT we considesed previouily)

Applying Eules-Lagrange we obtain equations of motion

$$
\left\{\begin{array}{l}
\frac{d}{d t}\left(\frac{\partial L}{\partial \dot{q}_{j}}\right)=\frac{\partial L}{\partial q_{j}}+\lambda(t) \frac{\partial f_{i}}{\partial q_{j}} \\
f_{i}(q, t)=0
\end{array}\right.
$$

Noether's theorem
Informally - symmetries which leave the Lagangion unchanged each have a corresponding cosserved quantity

Consider the maph

$$
\begin{aligned}
& t \rightarrow t^{\prime}=t+\delta t \\
& q \rightarrow q^{\prime}=q+\delta q
\end{aligned}
$$

Assume $N$ of such maps

$$
\begin{align*}
& \delta_{t}=\sum_{r} \varepsilon_{r} T_{r} \\
& \delta q=\sum_{r} \varepsilon_{r} Q_{r} \tag{r}
\end{align*}
$$

Then $\left(\frac{\partial L}{\partial \dot{q}} \cdot \dot{q}-L\right) T_{r}-\frac{\partial L}{\partial \dot{q}}$.

Are conserved quantities.

## Examples

(1) Time Invariance

$$
\begin{aligned}
& t \rightarrow t+\delta t \\
& N=1, T=1, Q=0
\end{aligned}
$$

$\frac{\partial L}{\partial \dot{q}} \cdot \dot{q}-L$
(2) Translation invariance

$$
\begin{aligned}
& q \rightarrow q+\delta q \\
& N=1, T=0, Q=1
\end{aligned}
$$

$\frac{\partial L}{\partial \dot{q}}=$ const (this is the definition of conjigate momentum!)
(3) Rotational invariance
$L=r \times p$ in nugular momentum
Connider $\quad r \rightarrow r+\delta \theta n \times r$
Rotation of $5 \theta$ alout a given axis $n$.
![](https://cdn.mathpix.com/cropped/efdda193-88f0-4aed-bc9a-aba9d421c76b-10.jpg?height=266&width=297&top_left_y=373&top_left_x=1735)
$N=1, T=0 \quad Q=n \times r$

$$
\begin{aligned}
\text { const }=\frac{\partial L}{\partial \dot{q}} \cdot Q & =p \cdot(n \times r) \\
& =n \cdot(r \times p) \\
& =n \cdot L
\end{aligned}
$$

When machine learning dynamies, we will design architectures which respect this set of symmenies

