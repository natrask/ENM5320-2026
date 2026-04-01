Last lime we intooduced mixed FEM spaces for stokes flow

$$
\begin{aligned}
\nabla^{2} n-\nabla p & =f \\
\nabla \cdot n & =0
\end{aligned}
$$

This is an example of the general class of saddle-point problems

$$
\begin{aligned}
a(u, v)+b(p, v) & =L_{1}(v) & \forall(v, q) \\
b(q, u) & =L_{2}(q) & \in V_{h} \otimes M_{h}
\end{aligned}
$$

Assuming a saticfies Lax-Milyram conditions, lets repeat the uniqueness proof from last lecture

Let $L_{1}, L_{2}=0, q=p, v=n$

$$
\begin{gathered}
a(u, u)+b(p, u)=0 \\
b(p, u)=0 \\
\alpha\|u\|_{v} \leq a(u, u)=0 \\
\Rightarrow u=0
\end{gathered}
$$

Eqn 1 reduces to

$$
b(p, v)=0
$$

If $V_{h}, M_{h}$ satisfy influp compatibility i.e for all $p \in M_{n}, \exists \widetilde{V} \in V_{h}$ s.t

$$
\frac{b(p, \tilde{v})}{\|\tilde{v}\|_{v}} \geq \beta\|p\|_{m}
$$

Then

$$
\begin{gathered}
\| \rho n_{M} \leq 0 \\
p=0
\end{gathered}
$$

Back to ML
So far we've learned models like

$$
A n+\varepsilon N(n)=f
$$

Now we have the equipment to consides conservation laws

$$
\begin{aligned}
& \nabla \cdot F=f \\
& F=L(u)+\varepsilon N(u) \\
& \text { liner }
\end{aligned}
$$

These satisfy, for $f=0$

$$
O=\int_{\Omega} f d x=\int_{\Omega} \nabla \cdot F d x=\int_{\partial \Omega} F \cdot d A
$$

Similarly, one can define

$$
\begin{aligned}
& \nabla \times F=f \\
& F=L(n)+\varepsilon N(n)
\end{aligned}
$$

Foor conservation of circulations (e-y magnetic fields vorticity...

$$
0=\int f d A=\int \nabla \times F \cdot d A=\oint F \cdot d l
$$

Lets start by derigning a good space in the linear setting $(\varepsilon=0)$ in ID

$$
\begin{aligned}
& F^{\prime}=f \\
& F+L(u)
\end{aligned}
$$

The first equation gives

$$
\begin{aligned}
\left(q, F^{\prime}\right) & =C(q, q)-\left(q^{\prime}, F\right)=(f, q) \\
b(q, F) & =L_{2}(q)
\end{aligned}
$$

To obtain a saddle point problem (and an automatic, stability, car choose

$$
\begin{aligned}
& L(u)=u^{\prime} \\
& (F, v)+\left(u^{\prime}, v\right)=0 \\
& a(F, v)+b(u, v)=0
\end{aligned}
$$

w/ $a(u, v)=(u, v)$

- Can earily check a satisfier Lax-Mulgam
- Just need to choose an inf-up stable $U_{h}, M_{h}$

Let $M_{h}=\left\{f \in L^{2}(\Omega), \quad f l_{e}=\right.$ const $\}$
$V_{h}=$ piecenise linears $\subseteq H_{0}^{\prime}$
![](https://cdn.mathpix.com/cropped/3526a46e-ac0e-4ce8-9f55-f70554b67dbf-05.jpg?height=354&width=451&top_left_y=145&top_left_x=1598)

Given a $\rho$, we can build $\tilde{v}$ such that $\tilde{v}^{\prime}=p$

Set $\tilde{V}_{0}=0$

$$
\widetilde{V}_{j}=\sum_{k=0}^{j-t} \rho_{k} h
$$

Then $\quad \int_{e_{j}} \nabla \tilde{v} d x=\tilde{v}_{j+1} \tilde{v}_{j+1}=P_{j \text { mi }} h$
Note that

$$
\begin{aligned}
b(p, \tilde{v}) & =\int p \tilde{v}^{\prime} \\
& =\sum_{j} \hat{p}_{j} \int x_{\left[x_{j}, x_{j+1}\right]} \tilde{v}^{\prime} \\
& -\sum_{j} \hat{p}_{j}\left(\tilde{v}_{j+1}-\tilde{v}_{j}\right) \\
& =\sum_{j} \hat{p}_{j}^{2} h \\
& =\|p\|_{L^{2}}^{2}
\end{aligned}
$$

And $|v|_{H_{1}}^{2}=\sum_{e_{j}} \int_{a_{x_{j}}}^{x_{j+1}} \nabla v^{2} d x$

$$
\begin{aligned}
& =\sum_{e_{j}} \int_{x_{j}}^{x_{j+1}} \rho^{2} d x \\
& =\|p\|_{L}^{2}
\end{aligned}
$$

So $\frac{b\left(\rho_{1} \tilde{v}\right)}{|v|_{H_{1}}}=\frac{\|\rho\|_{L^{2}}^{2}}{\|\rho\|_{L^{2}}}=\|\rho\|_{L^{2}}$
Now that we've worked a pair of inf-up stable spaces out, notice
(1) that for any bigger choice of
$V_{h} \subseteq V_{h}^{\text {big }}, \tilde{v} \in V_{h}^{b_{i j}}$, and $s_{0}$
$M_{h}$ and $V_{h}^{i_{k}} y$ are also inf-supcompatible
(2) The key property was that $V_{h}$ is choien lig enough that $\tilde{v}^{\prime}=p$
or the derivative operator is anto (ivijective)

$$
\frac{d}{d x} V_{h} \supseteq M_{h}
$$

Thus
In higher dimensians, for $D=$ div/grad/cuil

$$
\left(D^{\star} f, g\right)=(f, D g) \quad D^{\star}=\text { grad/div/curl }
$$

The raddle point problem

$$
\begin{array}{ll}
(F, v) \div(D u, v) & =0 \\
\left(q, D^{*} F\right) & =(f, q)
\end{array}
$$

is infrup stable if $D V_{h} \supseteq M_{h}$
![](https://cdn.mathpix.com/cropped/3526a46e-ac0e-4ce8-9f55-f70554b67dbf-07.jpg?height=378&width=1188&top_left_y=1375&top_left_x=522)

Now we have FEM spaces nailed down What about discrete conservation properties?

Consider higher-dim now

$$
-(\nabla q, F)+\langle q, F\rangle_{B C}=(f, q)
$$

Globin conservation
Let $q=1$

$$
\begin{aligned}
-(\nabla q / F)+\langle q, F\rangle & =(f, q) \\
\int_{\partial \Omega} F \cdot d A & =\int_{\Omega} f d x \\
\text { flox in/out } & =\text { source-sinks }
\end{aligned}
$$

Local connervation. Let $V_{h}=\operatorname{span}(\phi:)$

Let $q=\phi_{\text {i }}$

$$
\begin{aligned}
& -\int \nabla \phi_{i} \cdot F d x=\left(f, \phi_{i}\right) \\
= & -\sum_{j} \int \phi_{j} \nabla \phi_{i} \cdot F d x \\
= & \sum \int \underbrace{\phi_{j} \nabla \phi_{i}-\phi_{i} \nabla \phi_{j}}_{\psi_{i j}=-\psi_{j i}}) \cdot F d x \\
f=0 &
\end{aligned}
$$

Assume $F \cdot d t=0$

$$
\sum_{i} \phi_{i}=1
$$

$\frac{\pi}{j} \nabla \phi_{j}=0$

$$
\sum_{j} \phi_{i} \nabla \phi_{j}=0
$$

If $f=0$

$$
\sum_{j} \underbrace{\int \psi_{i j}^{\prime} \cdots F}_{\substack{\text { equal and opposite } \\ \text { fluxes }}} d x=0
$$

Denote $W^{0}=\operatorname{span}\left(\phi_{i}\right)$

$$
w^{\prime}=\operatorname{span}\left(\psi_{i j}^{\prime}\right)
$$

Thm $\operatorname{grad}\left(W^{0}\right) \subseteq W^{\prime}$
et Let $f \in W^{\circ}$

$$
\begin{aligned}
f & =\sum \hat{f}_{i} \phi_{i}(x) \\
\nabla f & =\sum_{i} \hat{f}_{i} \nabla \phi_{i} \\
& =\sum_{i, j} \hat{f}_{i}\left(\phi_{j} \nabla \phi_{i}-\phi_{i} \nabla \phi_{j}\right) \\
& =\sum_{i, j} \hat{f}_{i} \psi_{i j} \subseteq W^{\prime}
\end{aligned}
$$

Motivated by the construction, we'll "ploy by en" and make a space for curls

$$
\begin{aligned}
\left(\nabla \psi_{i j}^{\prime}, \nabla \times F\right) & =\left(\nabla \times \psi_{i j}^{\prime}, F\right) \\
& =\int \nabla \times\left(\phi_{j} \nabla \phi_{i}-\phi_{i} \nabla \phi_{j}\right) \cdot F d \times \\
\nabla \times f \vec{G} & =\int \partial \nabla \phi_{j} \times \nabla \phi_{i} \cdot F d x \\
=\nabla \nabla \times G & =\sum_{k} \partial \phi_{k} \nabla \phi_{j}+\nabla \phi_{i} \cdot F d x
\end{aligned}
$$

$$
\begin{aligned}
& =\sum_{k} \partial \underbrace{\phi_{k} \nabla \phi_{j} \times \nabla \phi_{k i}+\phi_{i} \nabla \phi_{k} \nabla \phi_{j}+\phi_{j} \nabla \phi_{i} \times \nabla \phi_{k}}_{\psi_{i j k}^{2}}) \cdot F \\
& \text { Let anti-symmetric wrt } \\
& w^{2}=\operatorname{span}\left(\psi_{i j k}^{2}\right) \quad \text { to index sump. }
\end{aligned}
$$

Thm $\operatorname{curl}\left(W^{\prime}\right) \subseteq W^{2}$

Can now build a dekham complex

$$
W^{0} \xrightarrow{\text { grad }} W^{\prime} \xrightarrow{\text { curl }} W^{2}
$$

Preserving exactly curl.grad $=0$
$W^{0} \sigma_{- \text {div }} W^{1} \leftarrow_{\text {curl }} W^{2}$
integrating by parts $(\nabla f, y)=-(f, \nabla y)$ gives

$$
\operatorname{div} \circ \operatorname{curl}=0
$$

Return to nonlinawity
$L u+\varepsilon \mathbb{N}(u)=f$

$$
a(u, v)+\varepsilon(N(u), v)=(5, v)
$$

We introduce a new condition

Manotaricity $(N(n), n) \geq \alpha_{N}\|u\|_{v}^{2}$

To prove stability
$\alpha\|n\|_{v}^{2}-\varepsilon \alpha_{N}\|n\|_{v}^{2} \leq a(n, n)+\varepsilon(N(n), n)=(f, n) \leq \Omega\|n\|_{v}$

$$
\|n\|_{v} \leq \frac{\alpha}{\alpha-\varepsilon \alpha_{N}}\|f\|_{v}
$$

