# Coercivity of the DEC Laplacian via Canonical Paths

## 1. Setting

Let $G = (V, E)$ be a connected graph with $N = |V|$ nodes and $|E|$ edges. Equip the graph with:

- **Coboundary operator** $\delta_0 \in \mathbb{R}^{|E| \times N}$: the signed incidence matrix (discrete gradient).
- **Hodge star on 1-forms** $\star_1 \in \mathbb{R}^{|E| \times |E|}$: a **positive** diagonal matrix of edge weights. In DEC on a simplicial mesh, $(\star_1)_{ee} = |e^*|/|e|$. In 2D this reduces to the cotan weights $(\star_1)_e = \tfrac{1}{2}(\cot\alpha_e + \cot\beta_e)$, where $\alpha_e, \beta_e$ are the angles opposite edge $e$.
- **Hodge star on 0-forms** $\star_0 \in \mathbb{R}^{N \times N}$: a **positive** diagonal matrix of node masses. In DEC, $(\star_0)_{ii} = |v_i^*|$ (dual cell volume).

> **Mesh quality requirement.** The hypothesis $\star_1 > 0$ is not automatic. In 2D, $(\star_1)_e > 0$ iff $\alpha_e + \beta_e < \pi$, which is the **Delaunay condition** on edge $e$. In 3D and higher, the analogous condition is **well-centeredness** (circumcenters lie inside their simplices). This is the geometric condition making $\star_1$ a valid inner product on 1-forms. If $\star_1$ has negative entries, the stiffness matrix $K$ loses positive semi-definiteness, the resistance distance becomes ill-defined, and the proof below fails.

Define:

- **Stiffness matrix**: $K = \delta_0^T \star_1 \, \delta_0$
- **Mass matrix**: $M = \star_0$, with entries $m_i = (\star_0)_{ii}$
- **Total mass**: $\operatorname{vol}_M = \sum_{i=1}^N m_i$

## 2. Generalized Eigenvalue Problem and Coercivity

The DEC discretization of $-\Delta u = \lambda u$ is the generalized eigenvalue problem:

$$K u = \lambda \, M u$$

Since $\star_1 > 0$ implies $K$ is symmetric positive semi-definite with $\ker K = \operatorname{span}\{\mathbf{1}\}$ (by connectedness), and $M$ is symmetric positive definite, the eigenvalues are:

$$0 = \lambda_0 < \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_{N-1}$$

The smallest nonzero eigenvalue is given by the Rayleigh quotient:

$$\lambda_1 = \min_{\substack{u \neq 0 \\ \sum_i m_i u_i = 0}} \frac{u^T K u}{u^T M u} = \min_{\substack{u \neq 0 \\ \sum_i m_i u_i = 0}} \frac{\sum_{e=(i,j)} (\star_1)_e \,(u_i - u_j)^2}{\sum_i m_i \, u_i^2}$$

This $\lambda_1$ is the **coercivity constant** for Lax-Milgram: for all $u$ with $\sum_i m_i u_i = 0$,

$$u^T K u \geq \lambda_1 \, u^T M u$$

## 3. Canonical Paths and Congestion

Choose a collection of paths $\Gamma = \{\gamma_{ij}\}$, one path $\gamma_{ij}$ in $G$ for each ordered pair of nodes $(i,j)$.

For each path, define its **resistance length**:

$$\ell_\star(\gamma_{ij}) = \sum_{e \in \gamma_{ij}} \frac{1}{(\star_1)_e}$$

Define the **congestion**:

$$\bar{\rho} = \max_{e \in E} \sum_{\substack{(i,j):\\ e \in \gamma_{ij}}} m_i \, m_j \, \ell_\star(\gamma_{ij})$$

This measures the worst-case load any single edge carries, weighted by the node masses and path resistances of all pairs routed through it.

## 4. Theorem (Discrete Poincaré Inequality)

> For any connected graph $G$ with $\star_0, \star_1 > 0$ and any choice of canonical paths $\Gamma$:
> $$\boxed{\lambda_1 \geq \frac{2\,\operatorname{vol}_M}{\bar{\rho}}}$$

### Proof

**Step 1 (Cauchy-Schwarz along each path).** For each pair $(i,j)$, telescope $u_i - u_j$ along $\gamma_{ij}$ and apply Cauchy-Schwarz with weights $(\star_1)_e$ and $1/(\star_1)_e$:

$$(u_i - u_j)^2 \leq \ell_\star(\gamma_{ij}) \sum_{e \in \gamma_{ij}} (\star_1)_e\,(\Delta u_e)^2 \tag{1}$$

where $\Delta u_e = u_a - u_b$ for edge $e = (a,b)$.

**Step 2 (Weighted variance identity).** For $u$ with $\sum_i m_i u_i = 0$:

$$u^T M u = \frac{1}{2\,\operatorname{vol}_M} \sum_{i,j} m_i \, m_j \, (u_i - u_j)^2 \tag{2}$$

*Proof*: Expand the right-hand side:

$$\frac{1}{2\,\operatorname{vol}_M} \sum_{i,j} m_i m_j (u_i^2 - 2u_i u_j + u_j^2) = \sum_i m_i u_i^2 - \frac{\left(\sum_i m_i u_i\right)^2}{\operatorname{vol}_M} = u^T M u \quad\square$$

**Step 3 (Substitute and exchange summation).** Substitute (1) into (2):

$$u^T M u \leq \frac{1}{2\,\operatorname{vol}_M} \sum_{i,j} m_i m_j \, \ell_\star(\gamma_{ij}) \sum_{e \in \gamma_{ij}} (\star_1)_e(\Delta u_e)^2$$

Exchange the order of summation, grouping by edge $e$:

$$= \frac{1}{2\,\operatorname{vol}_M} \sum_{e \in E} (\star_1)_e(\Delta u_e)^2 \underbrace{\sum_{\substack{(i,j):\\ e \in \gamma_{ij}}} m_i m_j \, \ell_\star(\gamma_{ij})}_{\leq\;\bar{\rho}}$$

$$\leq \frac{\bar{\rho}}{2\,\operatorname{vol}_M} \sum_{e \in E} (\star_1)_e(\Delta u_e)^2 = \frac{\bar{\rho}}{2\,\operatorname{vol}_M} \, u^T K u$$

Rearranging and minimizing over all $u$ with $\sum_i m_i u_i = 0$:

$$\lambda_1 \geq \frac{2\,\operatorname{vol}_M}{\bar{\rho}} \qquad\blacksquare$$

> **Remark (Diameter bound as special case).** If every $\gamma_{ij}$ is a shortest resistance-path and we define $D_\star = \max_{i,j} \ell_\star(\gamma_{ij})$, then $\bar{\rho} \leq D_\star \cdot \operatorname{vol}_M^2$ and the bound reduces to $\lambda_1 \geq 2/(D_\star \cdot \operatorname{vol}_M)$. This simpler bound is sufficient for 1D meshes and general graphs where $D_\star$ is moderate.

## 5. Application: $d$-Dimensional Grid with DEC Hodge Stars

Consider a $d$-dimensional grid graph on $[0, L]^d$ with $n$ nodes per side ($N = n^d$, $h = L/n$), equipped with DEC Hodge stars $(\star_1)_e = h^{d-2}$ and $(\star_0)_i = h^d$.

### Choice of Canonical Paths

For each pair $(\mathbf{i}, \mathbf{j})$ with $\mathbf{i} = (i_1, \ldots, i_d)$ and $\mathbf{j} = (j_1, \ldots, j_d)$, define $\gamma_{\mathbf{i}\mathbf{j}}$ as the **axis-aligned path**: walk in direction 1 from $i_1$ to $j_1$, then direction 2, ..., then direction $d$.

### Path Length

Each path has at most $d \cdot n$ edges, each with resistance $1/(\star_1)_e = h^{2-d}$:

$$\ell_\star(\gamma_{\mathbf{i}\mathbf{j}}) \leq d \cdot n \cdot h^{2-d}$$

### Congestion Computation

Fix an edge $e$ in direction $k$ at grid position $\mathbf{x}$. A canonical path $\gamma_{\mathbf{i}\mathbf{j}}$ passes through $e$ only if:

- $j_l = x_l$ for $l < k$ (already routed to destination in earlier coordinates)
- $i_l = x_l$ for $l > k$ (not yet routed from source in later coordinates)
- $i_k$ on one side of $e$, $j_k$ on the other: at most $n^2$ choices
- $i_l$ for $l < k$: $n$ free choices per coordinate ($k-1$ coordinates)
- $j_l$ for $l > k$: $n$ free choices per coordinate ($d-k$ coordinates)

Total pairs using $e$: at most $n^{k-1} \cdot n^2 \cdot n^{d-k} = n^{d+1}$.

Each contributes $m_i m_j \cdot \ell_\star(\gamma_{\mathbf{i}\mathbf{j}}) \leq h^{2d} \cdot d\,n\,h^{2-d}$. Therefore:

$$\bar{\rho} \leq n^{d+1} \cdot h^{2d} \cdot d\,n\,h^{2-d} = d\,n^{d+2}\,h^{d+2} = d\,L^{d+2}$$

### Result

With $\operatorname{vol}_M = n^d h^d = L^d$:

$$\boxed{\lambda_1 \geq \frac{2\,L^d}{d\,L^{d+2}} = \frac{2}{d\,L^2}}$$

This is **finite, mesh-independent, and valid for all $d$**.

### Comparison with True Eigenvalue

| $d$ | Bound $\lambda_1 \geq$ | True $\lambda_1 \to$ | Ratio |
|---|---|---|---|
| 1 | $2/L^2$ | $\pi^2/L^2$ | $\pi^2/2 \approx 4.9$ |
| 2 | $1/L^2$ | $2\pi^2/L^2$ | $2\pi^2 \approx 19.7$ |
| 3 | $2/(3L^2)$ | $3\pi^2/L^2$ | $9\pi^2/2 \approx 44.4$ |

The bound becomes looser with increasing $d$ (the ratio scales as $d^2\pi^2/2$), but always gives $\lambda_1 = \Omega(1/L^2)$.
