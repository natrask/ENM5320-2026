Mon 1-27. Zoom lecture

## Last time

- Indroduction to finite differences for transport eqn
- Stability analysis and unconditional instability for explicit Euler
- True dynamics vs FD dynamies

Today.

- Revisit the trigonometric interpolant
- Stable schemes for transport
- Our first learning problem


## STABILITY!

Before we talked alout grid functions and interpeted finite differences in terms of how they act on individual males. Now we will formalize that.
Define the discrete innes product
$(n, v)_{h}=\sum_{j=0}^{N} \bar{u}_{j} v_{j} h$ which indues the norm

$$
\|u\|_{h}^{2}=(n, u)_{h}
$$

All inequalities introduced last time hold analagourly

$$
\begin{aligned}
& \left|(u, v)_{h}\right| \leq\|u\|_{h}\|v\|_{h} \\
& \left|(u, a v)_{h}\right| \leq\|a\|_{\infty}\|u\|_{h}\|v\|_{h} \quad, \quad\|a\|_{\infty}=\max _{\dot{j}}\left|a_{j}\right| \\
& \|u+v\|_{h} \leq\|u\|_{h} \times\|v\|_{h} \\
& \left|\|u\|_{h}-\|v\|_{h}\right| \leq\|n-v\|_{h}
\end{aligned}
$$

## Stability Motivation

-FD demo
$-v_{j}^{n+1}=Q v_{j}^{n}$
τ Amplification operator

- Distinguish true evolution from discrete FD evolution
- What happens to fruncation essor?
- Lives in $a$
- Underitand $\|Q\| \rightarrow\|Q\|=1 \Rightarrow$ Discrete consevation
$\|Q\|>1$ Growth (bod)
$\|Q\|<1$
Artificial dissaptasion (Maybe bad/jasd)

To understand how continuous functions act when evaluated on grid points, take $u \in C^{k}$ and its restriction onto the grid $u\left(x_{j}\right)$

Then $\quad \lim _{h \rightarrow 0}(u, v)_{h}=(u, v), \quad \lim _{h \downarrow 0}\|u\|_{n}^{2}=\|u\|^{2}$
We are now equipped to analyze F.D. operators. Well cane at it in 2 ways

$$
\|Q\|_{h}:=\sup _{u \neq 0} \frac{\|Q u\|_{h}}{\|u\|_{h}}=\sup _{\|u\|=1}\|Q u\|_{h} \Rightarrow\left\|Q u_{h}\right\| \leq\|Q\|_{h}\|a\|_{1}
$$

For a ringle difference operator

$$
\left\|E_{p}^{p} u\right\|_{h}^{2}=\sum_{j=0}^{N}\left|u_{j+p}\right|^{2} h=\sum_{j=0}^{N}\left|u_{j}\right|^{2} h=\|u\|_{h}^{2}
$$

Which implies $\left\|E^{p}\right\|=1$
Also

$$
\begin{aligned}
\left\|D_{+} u\right\|_{h} & =\frac{1}{h}\left\|\left(E-E^{0}\right) u\right\|_{h} \\
& \leq \frac{1}{h}\|E u\|_{h}+\frac{1}{h}\|E u\|_{h} \\
& \leq \frac{2}{h}\|u\|_{h}
\end{aligned}
$$

triangle inequality
non

$$
\begin{array}{rlrl}
\left\|D_{-} u\right\|_{h} & =\left\|E^{-1} D_{r} u\right\|_{h} & \text { Recall } & D_{-} u=E^{-1} D_{r} \\
& \leq\left\|E^{-1}\right\|_{h}\left\|D_{r} u\right\| & \\
& \leq \frac{1}{h}\|u\|_{h} &
\end{array}
$$

Those are actual upper bounds. But we can also bound from below. Rember the def. of the oferator mom
for any $Q_{1} \quad\left\|Q_{h}\right\|=\sup _{u \neq 0} \frac{\left\|Q_{u}\right\|_{h}}{\|u\|_{h}}$
So for any particular $v, \neq 0$

$$
\left\|Q_{h}\right\| \geq \frac{\left\|Q_{v}\right\|_{h}}{\|v\|_{h}}
$$

Pick $V_{j}=(-1)^{j}$

$$
\begin{aligned}
\Rightarrow\|v\|_{h}^{2} & =\sum_{j=0}^{N}(-1)^{2 j} h \\
& =\left(\sum_{j=0}^{N} 1\right) h \\
& =h(N+1) \\
\left\|D_{f} v\right\|_{h}^{2} & =h \sum_{j}\left[\frac{\left(E-E^{0}\right)}{h} V_{j}\right]^{2} \\
& =h^{-1} \sum_{j}\left[(-1)^{j+1}-(-1)^{j}\right]^{2} \quad \text { factor out } \\
& =h^{-1} \sum_{j=0}^{N}(-1)^{2 j}[-1-1]^{2} \\
& =4 h^{-1}(N+1)
\end{aligned}
$$

So $\left\|D_{+}\right\|_{h}^{2} \geq \frac{\left\|D_{+} v\right\|_{h}^{2}}{\|v\|_{h}^{2}}=\frac{4 h^{-1}(N+1)}{h(N+1)}=\frac{4}{h^{2}}$

$$
\left\|D_{+}\right\|_{h} \geq \frac{2}{h}
$$

Now, Since $\frac{\partial}{h} \geq\left\|D_{+}\right\|_{h} \geq \frac{2}{h}$

$$
\left\|D_{r}\right\|_{h}=\frac{2}{h}
$$

Similarly $\left\|D_{-}\right\|_{h}=\frac{2}{h}$
A similar trick for $D_{0}$, picking $v_{j}=i_{\sqrt{-1}}^{a}$ Gives $\left\|D_{0}\right\|_{h}=\frac{1}{h}$

We now have the analysis tools to talk about the first property ⇒ accuracy!

Now we discuss the discrete Fouries transform, which lets un jump from gril functions to modes. We'll skill some proots.
Thm There exists a unigue trignimetrisial polynomial

$$
\operatorname{Int}_{N}(u)=\frac{1}{\sqrt{2 \pi}} \sum_{\omega=-\frac{N}{2}}^{\frac{N}{2}} \tilde{u}(\omega) e^{i \omega x}
$$

interpolating $u$, i.e

$$
u_{j}=\operatorname{Int}_{N}\left(u_{j}\right) \text { for } j=0, \ldots N
$$

Thin needs a discrete orthogonality result

Lemma $e^{i v x}$ for $v=0, \pm 1, \ldots, \pm \frac{N}{2}$ are orthoganal w.r.t. the discrete inner product

$$
\left(e^{i v x}, e^{i \mu x}\right)_{h}= \begin{cases}2 \pi & v=\mu \\ 0 & \text { else }\end{cases}
$$

Pf if $v=\mu$

$$
\begin{align*}
\left(e^{i v x}, e^{i v x}\right)_{h} & =\sum_{j} h \exp \left(2 i v x_{j}\right) \\
\text { Remember } & =\sum_{j} h \\
h=\frac{2 \pi}{N+1} & =(N+1) h  \tag{6}\\
& =2 \pi
\end{align*}
$$

Thun The interpolation problem has the unigue solution

$$
\tilde{u}(w)=\frac{1}{\sqrt{2 \pi}}\left(e^{i w x}, n\right)_{n} \quad|w| \leq N / 2
$$

Pf Assume we can intespolate, so that

$$
u_{j}=\frac{1}{\sqrt{2 \pi}} \sum_{w=-w / 2}^{N / 2} \tilde{u}(w) e^{i w \times j}
$$

Mult both sides by $e^{-i v x} h$ and rum

$$
\begin{aligned}
\sum_{j} e^{i v x} u_{j} h & =\frac{1}{\sqrt{2 \pi}} \sum_{w}\left(e^{i v x}, e^{i w x}\right)_{h} \tilde{u}(w) \\
\left(e^{i v x}, u\right)_{h} & =\frac{1}{\sqrt{2 \pi}} \sum_{w} \delta_{v w} 2 \pi \tilde{u}(w) \\
& =\sqrt{2 \pi} \tilde{u}(v) \\
\Rightarrow \tilde{u}(v) & =\frac{1}{\sqrt{2 \pi}}\left(e^{i v x}, u\right)_{h}
\end{aligned}
$$

## Disurete Parreval

Thm Given

$$
\begin{gathered}
I_{n} t_{N} u_{1}, I_{n} t_{N} u_{2} \\
\left(u_{1}, u_{2}\right)_{h}=\sum_{w=-\frac{w}{2}}^{w / 2} \overline{u_{1}(w)} u_{2}(w)
\end{gathered}
$$

We can relate the derivatives of an interpolant to the finite difference operator on gridfunction

Thm Let Intwn interp. $u$
Them $\|$ Int $_{N} u\left\|^{2}=\right\| u \|_{h}^{2}$

Bridge between continuous in tegrals and discrete and discrets

$$
\begin{aligned}
\left\|D_{+} u\right\|_{h}^{2} & \leq\left\|\frac{d}{d x} I_{n+N} u\right\|^{2} \\
& \leq \frac{\pi^{2}}{4}\left\|D_{+} u\right\|^{2}
\end{aligned}
$$

For those interest, Gustafiem $\& 1.3-1.4$ has more analysis, but proots need some trig/camplex wrangling

## Fundamental Theorem of Finite Differences

(Lax theorem)
Consider the alstract finite difference operator

$$
\begin{aligned}
V_{j}^{n+1}=Q V_{j}^{n} \quad Q & =\sum_{\nu} A_{\nu}\left(k_{i} h\right) E^{\nu} \\
V_{j}^{0} & =f_{j}
\end{aligned}
$$

If
(1) $\|f\|^{2}<\infty \quad$ (finite initial data)
(2) There exists a constant $K$ s indelement of $h, k$ so that

$$
\sup \left|\hat{Q}^{n}\right| \leq K_{s}
$$

(3) For any $w$

$$
\begin{aligned}
\lim _{K, h \rightarrow 0} \sup _{0 \leq t_{n}<T}\left|\hat{Q}^{n}(\xi)-e^{i \omega t_{n}}\right| & =0 \\
& \text { (consitency) }
\end{aligned}
$$

Then the FD scheme converges

$$
\lim _{K_{1} h \rightarrow 0} \sup _{0 \leq t_{n} \leq T}\left\|u\left(\cdot, t_{m}\right)-\operatorname{Int}_{N}\left(v_{j}^{n}\right)\right\|=0
$$

skip proof, ducush implications for code development

