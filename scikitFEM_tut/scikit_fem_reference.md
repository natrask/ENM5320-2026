# scikit-fem Reference Guide for Final Projects

> A curated collection of resources for students in ENM 5320 who want to use
> **scikit-fem** (and related tools) in their final projects.

---

## 1. Official Documentation & Source

| Resource | Link |
|---|---|
| **scikit-fem GitHub** | <https://github.com/kinnala/scikit-fem> |
| **API reference** | <https://scikit-fem.readthedocs.io/en/latest/> |
| **Installation** | `pip install scikit-fem` (pure Python, no compiled dependencies) |
| **Gallery of examples** | <https://github.com/kinnala/scikit-fem/tree/master/docs/examples> |
| **List of supported elements** | <https://scikit-fem.readthedocs.io/en/latest/api.html#module-skfem.element> |

**Why scikit-fem?**  It is a pure-Python library with minimal dependencies
(NumPy + SciPy) and a clean, decorator-based API that closely mirrors the
mathematical weak-form notation.  This makes it ideal for rapid prototyping and
educational use while still being efficient enough for moderate-scale problems.

---

## 2. Core Concepts Cheat Sheet

### Meshes

```python
from skfem import MeshTri, MeshQuad, MeshTet, MeshHex

# Built-in meshes
mesh = MeshTri.init_symmetric()         # symmetric unit square
mesh = MeshQuad().refined(4)            # refined quad mesh
mesh = MeshTri.init_circle()            # unit disk

# Refine an existing mesh
mesh = mesh.refined(n)                  # n uniform refinements

# External meshes (via meshio/pygmsh)
import pygmsh, meshio
# ... see hackathon notebook for pygmsh workflow
```

### Elements

| Element | Description | Order |
|---|---|---|
| `ElementTriP1()` | Piecewise linear on triangles | 1 |
| `ElementTriP2()` | Piecewise quadratic on triangles | 2 |
| `ElementTriP3()` | Cubic on triangles | 3 |
| `ElementQuad1()` | Bilinear on quads | 1 |
| `ElementQuad2()` | Biquadratic on quads | 2 |
| `ElementTetP1()` | Linear on tetrahedra (3D) | 1 |
| `ElementTetP2()` | Quadratic on tetrahedra (3D) | 2 |
| `ElementTriMini()` | MINI element (Stokes) | — |
| `ElementVector(e)` | Vectorize any scalar element | — |

### Basis & Assembly

```python
from skfem import Basis, BilinearForm, LinearForm, Functional
from skfem.helpers import dot, grad, div, sym_grad, ddot

basis = Basis(mesh, ElementTriP1())

@BilinearForm
def a(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm
def L(v, w):
    return f(w.x[0], w.x[1]) * v

K = a.assemble(basis)
b = L.assemble(basis)
```

### Boundary Conditions & Solving

```python
from skfem import condense, solve

# Homogeneous Dirichlet
u_h = solve(*condense(K, b, D=mesh.boundary_nodes()))

# Non-homogeneous Dirichlet
u_D = np.zeros(basis.N)
u_D[bnd] = g(mesh.p[0, bnd], mesh.p[1, bnd])
u_h = solve(*condense(K, b, x=u_D, D=bnd))

# Select specific boundaries
left   = mesh.nodes_satisfying(lambda x: x[0] < 1e-10)
right  = mesh.nodes_satisfying(lambda x: x[0] > 1.0 - 1e-10)
circle = mesh.nodes_satisfying(lambda x: (x[0]-0.5)**2 + (x[1]-0.5)**2 < 0.2**2)
```

---

## 3. Example Gallery by Application

### 3.1 Heat / Diffusion (Poisson-type)

These are closest to what we did in class. Good starting points:

- **Poisson with point source:**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex01.py>
- **Adaptive mesh refinement (Poisson):**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex12.py>
- **Laplace with mixed BCs:**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex03.py>
- **Heat equation (time-dependent):**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex23.py>
- **Convection-diffusion (SUPG stabilization):**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex22.py>

### 3.2 Fluid Mechanics

- **Stokes flow (lid-driven cavity):**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex20.py>
  Uses Taylor–Hood (P2/P1) elements for velocity/pressure.

- **Navier–Stokes (steady, Picard iteration):**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex29.py>
  Nonlinear solve via fixed-point iteration.

- **Navier–Stokes (transient, backward Euler):**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex30.py>

- **Creeping flow around cylinder:**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex05.py>

- **Brinkman flow (porous medium):**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex35.py>

**Key pattern for Stokes/Navier-Stokes:**
```python
from skfem.helpers import dot, grad, div, ddot

e_vel = ElementVector(ElementTriP2())     # P2 velocity
e_pre = ElementTriP1()                     # P1 pressure
basis = Basis(mesh, e_vel * e_pre)         # combined basis

@BilinearForm
def stokes(u, p, v, q, w):
    return ddot(grad(u), grad(v)) - div(u) * q - div(v) * p
```

### 3.3 Solid Mechanics (Linear Elasticity)

- **Linear elasticity (plane stress):**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex02.py>

- **Linear elasticity with body force:**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex10.py>

- **Nearly incompressible elasticity:**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex34.py>

**Key pattern for elasticity:**
```python
from skfem.helpers import sym_grad, ddot, eye, trace

e = ElementVector(ElementTriP1())  # vector-valued P1
basis = Basis(mesh, e)

lam, mu = 1.0, 1.0   # Lamé parameters

@BilinearForm
def linear_elasticity(u, v, w):
    eps_u = sym_grad(u)
    eps_v = sym_grad(v)
    return 2 * mu * ddot(eps_u, eps_v) + lam * trace(eps_u) * trace(eps_v)
```

### 3.4 Eigenvalue Problems

- **Laplacian eigenvalues:**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex07.py>
- **Vibration modes of elastic body:**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex08.py>

```python
from scipy.sparse.linalg import eigsh
# Solve K @ u = lam * M @ u
eigenvalues, eigenvectors = eigsh(K, M=M, k=6, sigma=0.0)
```

### 3.5 Nonlinear Problems

- **Nonlinear minimal surface (Newton):**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex19.py>
- **p-Laplacian:**
  <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex36.py>

**Pattern: Newton's method with scikit-fem**
```python
# At each Newton step, assemble the Jacobian J and residual F,
# then solve J @ du = -F and update u += du
for _ in range(max_iters):
    J = jacobian.assemble(basis, w=basis.interpolate(u))
    F = residual.assemble(basis, w=basis.interpolate(u))
    du = solve(*condense(J, -F, D=boundary_dofs))
    u += du
    if np.linalg.norm(du) < tol:
        break
```

---

## 4. Mesh Generation for Complex Geometries

### pygmsh (used in the hackathon)

```bash
pip install pygmsh meshio
```

```python
import pygmsh

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_max = 0.05
    # Combine primitives with boolean ops
    rect = geom.add_rectangle([0, 0, 0], 1.0, 1.0)
    hole = geom.add_disk([0.5, 0.5], 0.15)
    geom.boolean_difference(rect, hole)
    msh = geom.generate_mesh(dim=2)
```

### dmsh (Distmesh-like, no external dependency)

```bash
pip install dmsh
```

```python
import dmsh
geo = dmsh.Difference(dmsh.Rectangle(0, 1, 0, 1),
                      dmsh.Circle([0.5, 0.5], 0.15))
X, cells = dmsh.generate(geo, 0.05)
mesh = MeshTri(X.T, cells.T)
```

### Built-in scikit-fem meshes

```python
# Unit square, unit circle, L-shape, etc.
mesh = MeshTri.init_symmetric()       # unit square
mesh = MeshTri.init_circle()           # unit disk
mesh = MeshTri.init_lshaped()          # L-shape
mesh = MeshTri.load("path/to/file.msh")  # load from file
```

---

## 5. Post-Processing & Visualization

### scikit-fem built-in plotting

```python
from skfem.visuals.matplotlib import plot, draw
draw(mesh)                     # wireframe
plot(mesh, u_h, shading='gouraud')   # filled contour
```

### Export to ParaView (VTK)

```python
from skfem.visuals.vtk import to_file
to_file(mesh, u_h, "solution.vtk")
# Open in ParaView for 3D interactive visualization
```

### Matplotlib 3D surface

```python
from matplotlib.tri import Triangulation
tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(tri, u_h, cmap='viridis')
```

---

## 6. Tips for Final Projects

1. **Start simple.** Get a working solve on a coarse mesh before refining or
   adding complexity.

2. **Verify with manufactured solutions.** As we did in the hackathon, always
   test convergence rates before trusting your results on problems without
   known solutions.

3. **Use `mesh.refined(n)` for convergence studies** on built-in meshes — it's
   faster than re-generating via pygmsh.

4. **Mixed problems** (Stokes, elasticity) use the product basis syntax:
   `Basis(mesh, e1 * e2)`.

5. **Time stepping** is not built in — use your own time loop with backward
   Euler:
   ```python
   M = mass.assemble(basis)
   for _ in range(nt):
       u = solve(*condense(M + dt * K, M @ u_prev + dt * b, D=bnd))
       u_prev = u.copy()
   ```

6. **Performance:** For large 3D problems, consider using
   `scipy.sparse.linalg.splu` (direct) or iterative solvers from
   `scipy.sparse.linalg` (CG with AMG preconditioner via `pyamg`).

7. **Combine with PyTorch:** You can extract the assembled sparse matrices
   from scikit-fem, convert to `torch.sparse`, and use them in differentiable
   pipelines — relevant for neural-operator or physics-informed approaches.

---

## 7. Related Libraries

| Library | Description | Link |
|---|---|---|
| **FEniCSx** | Full-featured FEM suite (more powerful, heavier) | <https://fenicsproject.org/> |
| **Firedrake** | Automated FEM with composable solvers | <https://www.firedrakeproject.org/> |
| **deal.II** | C++ adaptive FEM library | <https://www.dealii.org/> |
| **NGSolve** | High-performance FEM with Python interface | <https://ngsolve.org/> |
| **PyTorch FEM** | Differentiable FEM in PyTorch | search GitHub for current projects |

---

## 8. Recommended Reading

- **Ern & Guermond**, *Theory and Practice of Finite Elements* (Springer, 2004) — the mathematical foundation behind everything in this course.
- **Braess**, *Finite Elements* (Cambridge, 2007) — concise graduate text, excellent for error analysis.
- **Hughes**, *The Finite Element Method* (Dover, 2000) — engineering perspective, strong on applications.
- **scikit-fem paper**: T. Gustafsson & G. D. McBain, "scikit-fem: A Python package for finite element assembly," *JOSS*, 5(52), 2369, 2020. <https://doi.org/10.21105/joss.02369>
