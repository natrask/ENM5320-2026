Same vector calc review
"Div/Grad/Curl and all that" Schey

Recall the Levi-Cirita tersor Eijk
satisfing $\quad \varepsilon_{i, k}=\left\{\begin{array}{cl}1 & \text { even permatations of }(i ; k) \\ -1 & \text { odd in } \\ 0 & \text { equal indices }\end{array}\right.$
Provides a compuct notation for curl

$$
\vec{F} \times \vec{G}=\varepsilon_{i j k} F_{j} G_{k}
$$

Some identifies - akn exact sequence property

$$
\nabla \cdot \nabla \times n=\nabla \times \nabla \phi=0
$$

pf

Adppting Einatein notation

$$
\begin{array}{rlr}
\nabla \times u & =\varepsilon_{i ; k} \partial_{x j} u_{k} & A_{i i}=\sum_{i} A_{i i} \\
\nabla \cdot \nabla \times n & =\varepsilon_{i j k} \partial_{x_{i}} \partial_{x_{j}} u_{k} \\
& =\frac{1}{2} \varepsilon_{i j k} \partial_{x_{i}} \partial_{x j} u_{k}+\frac{1}{2} \varepsilon_{j i k} \partial_{x_{i}} \partial_{x_{i}} u_{k} \\
& =\frac{1}{2}\left(\varepsilon_{i j k}+\varepsilon_{j i k}\right) \partial_{x_{i}} \partial_{x j} u_{k} \\
& =0 \quad \text { odd permutation }
\end{array}
$$

iim.iarly,

$$
\begin{aligned}
\nabla \times \nabla \phi & =\varepsilon_{i j k} \partial x_{j} \partial x_{k} \phi \\
& =\frac{1}{2}\left(\varepsilon_{i j k}+\varepsilon_{i k j}\right) \partial x_{i} \partial x_{k} \phi \\
& =0
\end{aligned}
$$

Thm Helmholtz - Holge decamporition
Given a smooth vector field $F \in \mathbb{R}^{d}$ on a simply connected subt of $\mathbb{R}^{d}$

$$
\begin{array}{ll}
\text { exiat unique at, } F= & -\nabla \phi+\nabla \times A \\
\phi, A & \uparrow \\
& \text { scalar potatial } \\
& \text { cule-free }
\end{array} \quad \begin{gathered}
\text { vector potential } \\
\text { div-flee }
\end{gathered}
$$

Pf

$$
\nabla \cdot F=-\nabla \cdot \nabla \phi+\nabla \cdot \Omega \times A
$$

$$
\text { Solve } \quad\left\{\begin{array}{l}
-\nabla^{2} \phi=\nabla \cdot F \\
\left.\partial_{n} \phi\right|_{\partial \Omega}=-F \cdot \hat{n}
\end{array}\right.
$$

⟵ aka pure пештани Poision problem

Note that this problem has a constant vector in its null-space
i.e. forany

$$
\begin{aligned}
& \nabla^{2}(\phi+c)=\nabla^{2} \phi \\
& \partial_{n}(\phi+c)=\partial_{n} \phi
\end{aligned}
$$

To maintain coercivity in Lax-Milyram, need to work in space orthogonal to Kernel

$$
V_{h}=\{\text { CPWL functions }|u| \cdot \vec{i}=0\}
$$

e.g. $\min _{\phi} \int_{\Omega} \frac{1}{2}|\nabla \phi|^{2}-\phi \nabla \cdot F+\lambda(\phi \cdot \vec{I})$

$$
\phi \int_{2 \Omega} F \phi \cdot d A
$$

Would be a way to enforce w/ Laginge Mitiplier

- That gave $\phi_{1}$ now to get $A$

$$
\nabla \times F=-\nabla \times \nabla \hat{\theta} \phi+\nabla \times \nabla \times A
$$

Now we have a more challenging null space
For any $q, \quad \nabla \times \nabla \times(A+\nabla q)=\nabla \times \nabla \times A$
So we'd like solutions orthogonal to $\operatorname{grad} q$, for any $q$

$$
(A, \nabla q)=0 \quad \forall q
$$

Applying Galerkin

$$
\begin{aligned}
&(\nabla \times \nabla \times A, v)=(\nabla \times F, v) \\
&=(\nabla \times A, \nabla \times v)+ \frac{\int(\nabla \times u \times \hat{n}) \cdot v d A}{\partial r}=(\nabla \times F, v) \\
& \text { Natural BC } \\
& \nabla \times u \times \hat{n}=\nabla \times F \times \hat{n}
\end{aligned}
$$

We can enforce $w /$ Laginge mutt. again

$$
\begin{array}{cc}
(\nabla \times \nabla \times A, v)+(\nabla \lambda, v)=(\nabla \times F, v)-\langle\nabla \times F \times n, v\rangle \\
(A, \nabla q)=0
\end{array}
$$

It'd be more work to show, but this is a saddle point problem that we showed how to approach last week.

The Hodye decompoition is a useful tool for projecting onto a div-free or curl-free space.
e.g $\quad \pi_{\text {div-flee }} u=\nabla \times A$
$\pi_{\text {curlifiee }} n=-\nabla \phi$

Ex fluid mecharics

Potential flow was an early model of fluids (w/large, restrictive assumptions) that was used e.g. to derign air foils in 20's.-40's

Assume $\left\{\begin{array}{l}u=-\nabla \phi \\ \nabla \cdot u=0\end{array}\right.$

Given lift, not drag

Then $\left\{\begin{array}{l}-\nabla^{2} \phi=0 \\ \left.\partial_{n} \phi\right|_{2 \Omega}=u \cdot \hat{n}\end{array}\right.$
No FEM back then → Laplace transforms etc. You'll still get this in a gradunte fluidn course

Ex Fluid Mechanics
When solving Navier-stoken w/ incompressibility

$$
\begin{aligned}
& \partial_{t} n+n \cdot \nabla n=-\nabla \rho+v \nabla^{2} n \\
& \nabla \cdot n=0
\end{aligned}
$$

The incompessibility constraint is annoying

- Maker syntem bigger
- How to choose inf-sup u,p?
- Coppler all points in damain

A populas schane (Alexandre Chosin, 1968)
Called a splitting-scheme
(1) $\quad \frac{n^{*}-n^{n}}{k}=-n^{n} \cdot \nabla n^{n}+N \nabla^{2} n^{*}$
(2a) $\quad\left\{\begin{array}{l}\frac{u^{n+1}-u^{t}}{k}=-\nabla p^{n+1} \\ \nabla \cdot u^{n+1}=0\end{array}\right.$
Taking div of $2 a$

$$
\nabla \cdot h^{h+1}-\frac{\nabla \cdot n^{A}}{k}=-\nabla^{2} p^{n+1}
$$

or $\quad u^{n+1 *}=\prod_{\text {divfiee }} u^{*}-u^{*} \quad$ presure-projection
LU

Back to graphs
We introduced grad/div as coboundury, codeffertial pair

$$
\left\langle v, \delta_{0} n\right\rangle=\left\langle\delta_{0}^{*} v, n\right\rangle
$$

Now for the formal def. so we can talk culs.
def $k$-chain $C^{k}=\left[v_{1}, \ldots . v_{k}\right]_{v_{i} \in V}$
anti-symmetric W.r.t. Swaps
def aboundary $\partial_{K-1}: C^{K} \rightarrow C^{K-1}$

$$
\partial_{k-1}\left[v_{1}, \ldots v_{k}\right]=\sum_{i=1}^{k}(-1)^{i-1}\left[n_{1}, \ldots n_{i}, \ldots n_{k}\right]
$$

ex For edge $\left[v_{1}, v_{2}\right]=-\left[v_{2}, v_{1}\right]$ this vertex out

$$
\begin{aligned}
\partial_{0}\left[v_{1}, v_{2}\right]= & (-1)^{0}\left[v_{2}\right] \\
& +(-1)^{1}\left[v_{1}\right] \\
= & v_{2}-v_{1}
\end{aligned}
$$

Thm Exact sequence property

$$
\partial_{k-1} \cdot \partial_{k}=0
$$

DE
Given $C \in C^{k+1}$

$$
\begin{aligned}
& \partial_{k-1} \partial_{k} c=\sum_{i=1}^{k-1}(-1)^{i-1} \partial_{k}\left[n_{1}, \ldots \hat{n}_{i}, \ldots n_{k-1}\right] \\
& \quad=\sum_{i=1} \sum_{j}(-1)^{i-1}(-1)^{j-1}\left[n_{1}, \ldots \hat{n}_{i}, \ldots \hat{n}_{j} \ldots n_{k}\right] \\
& \quad=\sum_{i, j}(-1)^{i+2-2} \alpha_{i, j} \alpha_{\alpha_{i j}=-\alpha_{j i}} \begin{array}{c}
\text { anti-symmetric } \\
\text { if we exchange } i, j
\end{array} \\
& \quad=0
\end{aligned}
$$

Illustrative sketch
$C_{i, j k}=\left[v_{i}, v_{i,} v_{k}\right]$
$\partial c_{i j k}=e_{i i}+e_{i k}+e_{i k} e_{i j}$
![](https://cdn.mathpix.com/cropped/b3cc6841-6944-44b0-892b-3f163958dad5-08.jpg?height=588&width=680&top_left_y=1902&top_left_x=661)

To define the coboundery (divignal curl) just swap chains w/ cochains
For $f \in C_{k}$

$$
\oiint_{k-1} f=\Sigma(-1)^{i} f\left(v_{1}, \ldots \hat{v}_{i}, \ldots v_{k}\right)
$$

ex $\quad \delta_{0} \phi_{i j}=\phi_{j}-\phi_{i} \quad$ graph gradient

$$
\delta_{1} A_{i j k}=A_{i j}+A_{j k}+A_{k i} \quad \text { graph curl }
$$

curl ograd $=0$

$$
\begin{aligned}
\delta_{1} \delta_{0} \phi & =\delta_{0} \phi_{i j}+\delta_{0} \phi_{j k}+\delta_{0} \phi_{k i} \\
& =\phi_{j}-\phi_{i}+\phi_{k}-\phi_{j}+\phi_{i}-\phi_{k} \\
& =0
\end{aligned}
$$

If $\delta_{1} \delta_{0}=0, \delta_{0}^{*} \delta_{1}^{*}=0 \quad$ (div of curl=0)
st

$$
\begin{aligned}
\left\langle f_{1}\right. & \left.\delta_{0}^{*} \delta_{1}^{*} g\right\rangle \\
& =\left\langle\delta_{0} f_{1} \delta_{1}^{*} g\right\rangle \\
& =\left\langle\delta_{1} \delta_{0} f_{1} g\right\rangle \\
& =0
\end{aligned}
$$

Recall
![](https://cdn.mathpix.com/cropped/b3cc6841-6944-44b0-892b-3f163958dad5-10.jpg?height=335&width=476&top_left_y=934&top_left_x=1612)

$$
\begin{array}{ll}
\delta_{0}=\text { grad } & \delta_{0}^{*}=\operatorname{div} \\
\delta_{1}=\text { curl } & \delta_{1}^{*}=\text { curl }
\end{array}
$$

Graph Holye Decompoition
thry Given $f \in C_{k}$

$$
f=\delta_{0} \phi+\delta_{1}^{\star} A
$$

Pf

$$
\begin{aligned}
& \delta_{0}^{*} f=\delta_{0}^{*} \delta_{0} \phi+\delta_{0}^{*} \delta_{1}^{*} A \\
& \delta_{1} f=\delta_{y} \delta_{0} \phi+\delta_{1} \delta_{1}^{*} A
\end{aligned}
$$

Given olpervations on some preferences $Y_{i j}^{\text {data }}$

$$
\min _{s} \sum_{i, j} w_{i j}\left(Y_{i j}^{\text {dutn }}-d s_{i j}\right)^{2}
$$

Take the variation of this (HW exercise)

$$
\delta_{0}^{\star} W \delta_{0} S=\delta_{0}^{\star} W Y^{\text {data }}
$$

## Cauiality

Given a set of events $X_{i}$, a causal structural model is $a$ decomposition
$P\left(x_{1} \ldots x_{N}\right)=\prod_{i=1}^{N} P\left(x_{i} \mid P_{a}\left(x_{i}\right)\right)$ parents

Define a ucore $S_{i}$

$$
\text { flux } F=\delta_{0} S
$$

and parent $P_{a}\left(x_{i}\right)=\left\{x_{j}\right.$ s.t $\left.F_{i} \rightarrow 0\right\}$
<img class="imgSvg" id = "mkbe7bra35asqs93j1q" src="data:image/svg+xml;base64,PHN2ZyBpZD0ic21pbGVzLW1rYmU3YnJhMzVhc3FzOTNqMXEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDE3MCA4NS4wMjk4NDg0OTg5ODQ1OCIgc3R5bGU9IndpZHRoOiAxNzAuMDU5NjAwNDM4MzY1NDRweDsgaGVpZ2h0OiA4NS4wMjk4NDg0OTg5ODQ1OHB4OyBvdmVyZmxvdzogdmlzaWJsZTsiPjxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0ibGluZS1ta2JlN2JyYTM1YXNxczkzajFxLTEiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4MT0iMTAwLjc3OTgxNzg5MDgwNjc4IiB5MT0iNDguMjc5ODE3ODkwODI2NjE0IiB4Mj0iMTI4LjA1OTYwMDQzODM2NTQ0IiB5Mj0iNjQuMDI5ODQ4NDk4OTg0NTgiPjxzdG9wIHN0b3AtY29sb3I9ImN1cnJlbnRDb2xvciIgb2Zmc2V0PSIyMCUiPjwvc3RvcD48c3RvcCBzdG9wLWNvbG9yPSJjdXJyZW50Q29sb3IiIG9mZnNldD0iMTAwJSI+PC9zdG9wPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJsaW5lLW1rYmU3YnJhMzVhc3FzOTNqMXEtMyIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiIHgxPSI2OS4yNzk4MTc4OTA4MjY2IiB5MT0iNDguMjc5NzgyNTQ3NTU4Njk2IiB4Mj0iMTAwLjc3OTgxNzg5MDgwNjc4IiB5Mj0iNDguMjc5ODE3ODkwODI2NjE0Ij48c3RvcCBzdG9wLWNvbG9yPSJjdXJyZW50Q29sb3IiIG9mZnNldD0iMjAlIj48L3N0b3A+PHN0b3Agc3RvcC1jb2xvcj0iY3VycmVudENvbG9yIiBvZmZzZXQ9IjEwMCUiPjwvc3RvcD48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0ibGluZS1ta2JlN2JyYTM1YXNxczkzajFxLTUiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4MT0iODUuMDI5ODQ4NDk4OTg0NTYiIHkxPSIyMSIgeDI9IjEwMC43Nzk4MTc4OTA4MDY3OCIgeTI9IjQ4LjI3OTgxNzg5MDgyNjYxNCI+PHN0b3Agc3RvcC1jb2xvcj0iY3VycmVudENvbG9yIiBvZmZzZXQ9IjIwJSI+PC9zdG9wPjxzdG9wIHN0b3AtY29sb3I9ImN1cnJlbnRDb2xvciIgb2Zmc2V0PSIxMDAlIj48L3N0b3A+PC9saW5lYXJHcmFkaWVudD48bGluZWFyR3JhZGllbnQgaWQ9ImxpbmUtbWtiZTdicmEzNWFzcXM5M2oxcS03IiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgeDE9IjY5LjI3OTgxNzg5MDgyNjYiIHkxPSI0OC4yNzk3ODI1NDc1NTg2OTYiIHgyPSI4NS4wMjk4NDg0OTg5ODQ1NiIgeTI9IjIxIj48c3RvcCBzdG9wLWNvbG9yPSJjdXJyZW50Q29sb3IiIG9mZnNldD0iMjAlIj48L3N0b3A+PHN0b3Agc3RvcC1jb2xvcj0iY3VycmVudENvbG9yIiBvZmZzZXQ9IjEwMCUiPjwvc3RvcD48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0ibGluZS1ta2JlN2JyYTM1YXNxczkzajFxLTkiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4MT0iNDAuNTgyNTAyNzU0NzM2IiB5MT0iNjEuNTc0NTY4MzI5MjA2NTE0IiB4Mj0iNjcuODYyMzIwNjQ1NTYyNiIgeTI9IjQ1LjgyNDU5ODkzNzM4NDI5NSI+PHN0b3Agc3RvcC1jb2xvcj0iY3VycmVudENvbG9yIiBvZmZzZXQ9IjIwJSI+PC9zdG9wPjxzdG9wIHN0b3AtY29sb3I9ImN1cnJlbnRDb2xvciIgb2Zmc2V0PSIxMDAlIj48L3N0b3A+PC9saW5lYXJHcmFkaWVudD48bGluZWFyR3JhZGllbnQgaWQ9ImxpbmUtbWtiZTdicmEzNWFzcXM5M2oxcS0xMSIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiIHgxPSI0My40MTc0OTcyNDUyNjQiIHkxPSI2Ni40ODQ5MzU1NDk1NTUzIiB4Mj0iNzAuNjk3MzE1MTM2MDkwNjEiIHkyPSI1MC43MzQ5NjYxNTc3MzMwOCI+PHN0b3Agc3RvcC1jb2xvcj0iY3VycmVudENvbG9yIiBvZmZzZXQ9IjIwJSI+PC9zdG9wPjxzdG9wIHN0b3AtY29sb3I9ImN1cnJlbnRDb2xvciIgb2Zmc2V0PSIxMDAlIj48L3N0b3A+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PG1hc2sgaWQ9InRleHQtbWFzay1ta2JlN2JyYTM1YXNxczkzajFxIj48cmVjdCB4PSIwIiB5PSIwIiB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ3aGl0ZSI+PC9yZWN0PjxjaXJjbGUgY3g9IjEwMC43Nzk4MTc4OTA4MDY3OCIgY3k9IjQ4LjI3OTgxNzg5MDgyNjYxNCIgcj0iNy44NzUiIGZpbGw9ImJsYWNrIj48L2NpcmNsZT48Y2lyY2xlIGN4PSI4NS4wMjk4NDg0OTg5ODQ1NiIgY3k9IjIxIiByPSI3Ljg3NSIgZmlsbD0iYmxhY2siPjwvY2lyY2xlPjxjaXJjbGUgY3g9IjQyIiBjeT0iNjQuMDI5NzUxOTM5MzgwOTIiIHI9IjcuODc1IiBmaWxsPSJibGFjayI+PC9jaXJjbGU+PC9tYXNrPjxzdHlsZT4KICAgICAgICAgICAgICAgIC5lbGVtZW50LW1rYmU3YnJhMzVhc3FzOTNqMXEgewogICAgICAgICAgICAgICAgICAgIGZvbnQ6IDE0cHggSGVsdmV0aWNhLCBBcmlhbCwgc2Fucy1zZXJpZjsKICAgICAgICAgICAgICAgICAgICBhbGlnbm1lbnQtYmFzZWxpbmU6ICdtaWRkbGUnOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgLnN1Yi1ta2JlN2JyYTM1YXNxczkzajFxIHsKICAgICAgICAgICAgICAgICAgICBmb250OiA4LjRweCBIZWx2ZXRpY2EsIEFyaWFsLCBzYW5zLXNlcmlmOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPjxnIG1hc2s9InVybCgjdGV4dC1tYXNrLW1rYmU3YnJhMzVhc3FzOTNqMXEpIj48bGluZSB4MT0iMTAwLjc3OTgxNzg5MDgwNjc4IiB5MT0iNDguMjc5ODE3ODkwODI2NjE0IiB4Mj0iMTI4LjA1OTYwMDQzODM2NTQ0IiB5Mj0iNjQuMDI5ODQ4NDk4OTg0NTgiIHN0eWxlPSJzdHJva2UtbGluZWNhcDpyb3VuZDtzdHJva2UtZGFzaGFycmF5Om5vbmU7c3Ryb2tlLXdpZHRoOjEuMjYiIHN0cm9rZT0idXJsKCcjbGluZS1ta2JlN2JyYTM1YXNxczkzajFxLTEnKSI+PC9saW5lPjxsaW5lIHgxPSI2OS4yNzk4MTc4OTA4MjY2IiB5MT0iNDguMjc5NzgyNTQ3NTU4Njk2IiB4Mj0iMTAwLjc3OTgxNzg5MDgwNjc4IiB5Mj0iNDguMjc5ODE3ODkwODI2NjE0IiBzdHlsZT0ic3Ryb2tlLWxpbmVjYXA6cm91bmQ7c3Ryb2tlLWRhc2hhcnJheTpub25lO3N0cm9rZS13aWR0aDoxLjI2IiBzdHJva2U9InVybCgnI2xpbmUtbWtiZTdicmEzNWFzcXM5M2oxcS0zJykiPjwvbGluZT48bGluZSB4MT0iODUuMDI5ODQ4NDk4OTg0NTYiIHkxPSIyMSIgeDI9IjEwMC43Nzk4MTc4OTA4MDY3OCIgeTI9IjQ4LjI3OTgxNzg5MDgyNjYxNCIgc3R5bGU9InN0cm9rZS1saW5lY2FwOnJvdW5kO3N0cm9rZS1kYXNoYXJyYXk6bm9uZTtzdHJva2Utd2lkdGg6MS4yNiIgc3Ryb2tlPSJ1cmwoJyNsaW5lLW1rYmU3YnJhMzVhc3FzOTNqMXEtNScpIj48L2xpbmU+PGxpbmUgeDE9IjY5LjI3OTgxNzg5MDgyNjYiIHkxPSI0OC4yNzk3ODI1NDc1NTg2OTYiIHgyPSI4NS4wMjk4NDg0OTg5ODQ1NiIgeTI9IjIxIiBzdHlsZT0ic3Ryb2tlLWxpbmVjYXA6cm91bmQ7c3Ryb2tlLWRhc2hhcnJheTpub25lO3N0cm9rZS13aWR0aDoxLjI2IiBzdHJva2U9InVybCgnI2xpbmUtbWtiZTdicmEzNWFzcXM5M2oxcS03JykiPjwvbGluZT48bGluZSB4MT0iNDAuNTgyNTAyNzU0NzM2IiB5MT0iNjEuNTc0NTY4MzI5MjA2NTE0IiB4Mj0iNjcuODYyMzIwNjQ1NTYyNiIgeTI9IjQ1LjgyNDU5ODkzNzM4NDI5NSIgc3R5bGU9InN0cm9rZS1saW5lY2FwOnJvdW5kO3N0cm9rZS1kYXNoYXJyYXk6bm9uZTtzdHJva2Utd2lkdGg6MS4yNiIgc3Ryb2tlPSJ1cmwoJyNsaW5lLW1rYmU3YnJhMzVhc3FzOTNqMXEtOScpIj48L2xpbmU+PGxpbmUgeDE9IjQzLjQxNzQ5NzI0NTI2NCIgeTE9IjY2LjQ4NDkzNTU0OTU1NTMiIHgyPSI3MC42OTczMTUxMzYwOTA2MSIgeTI9IjUwLjczNDk2NjE1NzczMzA4IiBzdHlsZT0ic3Ryb2tlLWxpbmVjYXA6cm91bmQ7c3Ryb2tlLWRhc2hhcnJheTpub25lO3N0cm9rZS13aWR0aDoxLjI2IiBzdHJva2U9InVybCgnI2xpbmUtbWtiZTdicmEzNWFzcXM5M2oxcS0xMScpIj48L2xpbmU+PC9nPjxnPjx0ZXh0IHg9IjEyOC4wNTk2MDA0MzgzNjU0NCIgeT0iNjQuMDI5ODQ4NDk4OTg0NTgiIGNsYXNzPSJkZWJ1ZyIgZmlsbD0iI2ZmMDAwMCIgc3R5bGU9IgogICAgICAgICAgICAgICAgZm9udDogNXB4IERyb2lkIFNhbnMsIHNhbnMtc2VyaWY7CiAgICAgICAgICAgICI+PC90ZXh0Pjx0ZXh0IHg9Ijk2Ljg0MjMxNzg5MDgwNjc4IiB5PSI1My41Mjk4MTc4OTA4MjY2MTQiIGNsYXNzPSJlbGVtZW50LW1rYmU3YnJhMzVhc3FzOTNqMXEiIGZpbGw9ImN1cnJlbnRDb2xvciIgc3R5bGU9InRleHQtYW5jaG9yOiBzdGFydDsgd3JpdGluZy1tb2RlOiBob3Jpem9udGFsLXRiOyB0ZXh0LW9yaWVudGF0aW9uOiBtaXhlZDsgbGV0dGVyLXNwYWNpbmc6IG5vcm1hbDsgZGlyZWN0aW9uOiBsdHI7Ij48dHNwYW4gc3R5bGU9IgogICAgICAgICAgICAgICAgdW5pY29kZS1iaWRpOiBwbGFpbnRleHQ7CiAgICAgICAgICAgICAgICB3cml0aW5nLW1vZGU6IGxyLXRiOwogICAgICAgICAgICAgICAgbGV0dGVyLXNwYWNpbmc6IG5vcm1hbDsKICAgICAgICAgICAgICAgIHRleHQtYW5jaG9yOiBzdGFydDsKICAgICAgICAgICAgIj5UZTwvdHNwYW4+PC90ZXh0Pjx0ZXh0IHg9IjEwMC43Nzk4MTc4OTA4MDY3OCIgeT0iNDguMjc5ODE3ODkwODI2NjE0IiBjbGFzcz0iZGVidWciIGZpbGw9IiNmZjAwMDAiIHN0eWxlPSIKICAgICAgICAgICAgICAgIGZvbnQ6IDVweCBEcm9pZCBTYW5zLCBzYW5zLXNlcmlmOwogICAgICAgICAgICAiPjwvdGV4dD48dGV4dCB4PSI4NS4wMjk4NDg0OTg5ODQ1NiIgeT0iMjguODc1IiBjbGFzcz0iZWxlbWVudC1ta2JlN2JyYTM1YXNxczkzajFxIiBmaWxsPSJjdXJyZW50Q29sb3IiIHN0eWxlPSJ0ZXh0LWFuY2hvcjogc3RhcnQ7IGdseXBoLW9yaWVudGF0aW9uLXZlcnRpY2FsOiAwOyB3cml0aW5nLW1vZGU6IHZlcnRpY2FsLXJsOyB0ZXh0LW9yaWVudGF0aW9uOiB1cHJpZ2h0OyBsZXR0ZXItc3BhY2luZzogLTFweDsgZGlyZWN0aW9uOiBydGw7IHVuaWNvZGUtYmlkaTogYmlkaS1vdmVycmlkZTsiPjx0c3Bhbj5PPC90c3Bhbj48L3RleHQ+PHRleHQgeD0iODUuMDI5ODQ4NDk4OTg0NTYiIHk9IjIxIiBjbGFzcz0iZGVidWciIGZpbGw9IiNmZjAwMDAiIHN0eWxlPSIKICAgICAgICAgICAgICAgIGZvbnQ6IDVweCBEcm9pZCBTYW5zLCBzYW5zLXNlcmlmOwogICAgICAgICAgICAiPjwvdGV4dD48dGV4dCB4PSI2OS4yNzk4MTc4OTA4MjY2IiB5PSI0OC4yNzk3ODI1NDc1NTg2OTYiIGNsYXNzPSJkZWJ1ZyIgZmlsbD0iI2ZmMDAwMCIgc3R5bGU9IgogICAgICAgICAgICAgICAgZm9udDogNXB4IERyb2lkIFNhbnMsIHNhbnMtc2VyaWY7CiAgICAgICAgICAgICI+PC90ZXh0Pjx0ZXh0IHg9IjQ3LjI1IiB5PSI2OS4yNzk3NTE5MzkzODA5MiIgY2xhc3M9ImVsZW1lbnQtbWtiZTdicmEzNWFzcXM5M2oxcSIgZmlsbD0iY3VycmVudENvbG9yIiBzdHlsZT0idGV4dC1hbmNob3I6IHN0YXJ0OyB3cml0aW5nLW1vZGU6IGhvcml6b250YWwtdGI7IHRleHQtb3JpZW50YXRpb246IG1peGVkOyBsZXR0ZXItc3BhY2luZzogbm9ybWFsOyBkaXJlY3Rpb246IHJ0bDsgdW5pY29kZS1iaWRpOiBiaWRpLW92ZXJyaWRlOyI+PHRzcGFuPk88L3RzcGFuPjwvdGV4dD48dGV4dCB4PSI0MiIgeT0iNjQuMDI5NzUxOTM5MzgwOTIiIGNsYXNzPSJkZWJ1ZyIgZmlsbD0iI2ZmMDAwMCIgc3R5bGU9IgogICAgICAgICAgICAgICAgZm9udDogNXB4IERyb2lkIFNhbnMsIHNhbnMtc2VyaWY7CiAgICAgICAgICAgICI+PC90ZXh0PjwvZz48L3N2Zz4="/>

Cause
or a differentiable version

$$
\operatorname{Pa}\left(x_{i}\right)=\left\{\operatorname{Reln}\left(\tanh \left(\frac{1}{\beta} F_{i ;}\right)\right) \geq 0\right\}
$$

