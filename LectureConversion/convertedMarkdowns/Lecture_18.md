- Atlention
- Code example
- Some more examples of Hamiltonion systems

We point to ENM 531 and Murphy's texthoot for discussion about architectures, but for final projects I want to make sure everyone has access to a undern transformer aschitecture

Connider the conditional neural field
![](https://cdn.mathpix.com/cropped/443f0c17-3ccb-4b55-9eaf-1060288c6364-1.jpg?height=337&width=917&top_left_y=1695&top_left_x=576)
$y=f(x \mid z ; \theta)$
Iden $Z$ modulates the $x, y$ inputloutport relationship
ex $\quad y$-finite difference stencil
$x$ - grid function
$Z$ - material properties/microscopy

Many options: - deeponet (Lucu)

- PCA/FNO (Etewart)
(*) - cross Attention

Attention as soft dictionary lookp (see Muply 15.4.1),

- Consider $\left(K_{i}, V_{i}\right)_{i=1}^{N} \quad$ Key-value pairs descrifing input/output labels
- Cansides a quey of where model is evaluated
def $\operatorname{AHn}\left(q,\left\{k_{i}, v_{i}\right\}\right)=\sum_{i=1}^{N} \alpha_{i}(q, \vec{k}) v_{i}$
for atten weights

$$
\begin{aligned}
& 0 \leq \alpha_{i} \leq 1 \\
& \sum \alpha_{i}=1
\end{aligned}
$$

introducing an attention score $a\left(q, k_{i}\right)$

$$
\alpha_{i}=\operatorname{softmax}(a(q, \vec{k}))=\frac{\exp \left[a\left(q, k_{i}\right)\right]}{\sum_{j} \exp \left[a\left(q, k_{j}\right)\right]}
$$

Remark This in a datadriven hasis
e.g. Taking $\alpha_{i}(q, \vec{k})=\phi_{i}(q)$串: CPWL
satisfien the same propecties

$$
\begin{aligned}
& f(x)=\sum_{i} \alpha_{i}\left(x, x_{n}\right) y_{i} \\
& q \rightarrow x \\
& k \rightarrow x_{n} \quad \text { (nodal values) } \\
& y_{i} \rightarrow \text { nodal basin functo ans } \\
& \text { dintion } q_{1}, v=x
\end{aligned}
$$

Multiliead Attention

- Intooduce Dease MLP Blocks $i=1, \ldots M$

$$
\begin{aligned}
& \hat{q}_{i}=Q_{i}(q) \\
& \hat{k}_{i}=K i(k) \\
& \hat{v}_{i}=V_{i}(v) \\
& h=\operatorname{MHA}\left(q,\left\{k_{1} v\right\}\right)
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/443f0c17-3ccb-4b55-9eaf-1060288c6364-3.jpg?height=876&width=740&top_left_y=1658&top_left_x=1281)

Drapout
With probability $P$, at each forward pars the output of a given neuron will be drepped
![](https://cdn.mathpix.com/cropped/443f0c17-3ccb-4b55-9eaf-1060288c6364-4.jpg?height=673&width=1604&top_left_y=650&top_left_x=480)

Replace weights

$$
\begin{aligned}
& \theta_{l j i}=\omega_{l j i} \varepsilon_{l i} \\
& \varepsilon_{l i} \sim \operatorname{Ber}(1-p)
\end{aligned}
$$

Graph Attention

Steps

1. Input fentures

Each node $n_{i} \in N$ has fenture vector $h_{i}^{n} \in \mathbb{R}^{F}$
2. Linew Transform $\quad h_{i}^{\prime}=W h_{i}^{n}, W \in \mathbb{R}^{F \times F}$
3. Preattention coeffs

$$
e_{i j}=\sigma\left(\underset{\eta}{a} \cdot\left[w h_{i} \cdot{\underset{\tau}{\text { lenky Relh }}}^{\sigma} h_{i}\right]\right)
$$

4. Attention mech.

$$
\alpha_{i j}=\frac{\exp \left(e_{i j}\right)}{\sum_{k i} \exp \left(e_{i k}\right)}
$$

5. Aggregate

$$
h^{n+1}=\sigma\left(\sum_{n_{i}} \alpha_{i j} w h_{j}^{n}\right)
$$

## Phyircs-Inspired Areh.

GRAND

$$
h_{i}^{n+1}=h_{i}^{n}+\sigma\left(\sum_{j \sim i} \alpha_{i j}\left(h_{j}^{n}-h_{i}^{n}\right)\right)
$$

or in graph calculus

$$
h_{i}^{n+1}=h_{i}^{n}+\sigma\left(\delta h^{*} \delta h^{n}\right)
$$

$\left\langle f, \delta^{*} g\right\rangle=\langle\delta f, g\rangle_{\alpha,}$ view attention as leasuring
Writing $h^{n+1}=Q^{n} h^{n}$ inner-product

Aobiting graph data $D=\left\{x_{i}, y_{i}\right\}$
(x) enc. $h$ Q $h$ Q... $h^{N}$ dec (y)

Lets revirit Hamiltanian mech.

$$
\begin{aligned}
& \frac{d}{d t}\binom{q}{p}=\left(\begin{array}{cc}
0 & -\delta_{0}^{*} \\
\delta_{0} & 0
\end{array}\right)\binom{\partial_{q} E}{\partial_{p} E} \\
& E=N N(p, q)
\end{aligned}
$$

Thun $\dot{E}=0$
Same as GRAND
![](https://cdn.mathpix.com/cropped/443f0c17-3ccb-4b55-9eaf-1060288c6364-7.jpg?height=456&width=1962&top_left_y=1316&top_left_x=149)

Double Bracket dynamics

$$
\begin{aligned}
\frac{d x}{d t} & =J \nabla E-J^{\top} J \nabla E \\
\frac{d E}{d t} & =\frac{\partial E}{\partial x} \cdot \frac{d x}{d t} \\
& =\frac{\partial E}{\partial x} f \frac{\partial E}{\partial x}-\frac{\partial E}{\partial x} J^{\top} J \frac{\partial E}{\partial x}
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/443f0c17-3ccb-4b55-9eaf-1060288c6364-8.jpg?height=655&width=1125&top_left_y=1423&top_left_x=314)
x $\pi k^{0} Q^{0}$ h $Q^{1} \ldots h^{N} \square y$

