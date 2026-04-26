"""Stage-1 prototype for the hp learning experiment.

Mirrors DataDriven_Whitney_Forms.ipynb's advection-diffusion training pipeline but at
arbitrary polynomial order r. Verifies that r=1 reproduces the original numerics,
then runs r=1,2,3 to compare trained-model accuracy.

Residual at order r (primal + strong-form NN coupling):
  R_r(u) = mu * K_r @ u + (delta0_r)^T @ N_r(u) - F_r
with
  K_r      = int (phi_a^r)'(phi_b^r)' dx                (primal Galerkin stiffness)
  delta0_r = I_r kron delta0                            (block-diag coboundary)
  N_r      = MLP: R^{r*NPOU} -> R^{r*N1forms}           (flux NN)
  F_r      = fhat + boundary forcing

BC handling: u(0) = u_{0,0} = bc_left;  u(1) = sum_k u_{N-1, k} = bc_right.
Enforced by null-space T_r and a lifting function.
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42); np.random.seed(42)
dtype = torch.float64


# ---------------------------------------------------------------------------
# Fine mesh + P1 basis + quadrature (identical to main tutorial)
# ---------------------------------------------------------------------------

MESHSIZE = 32
NPOU = 4
NQUAD = 8
h = 1.0 / (MESHSIZE - 1)
points = torch.linspace(0.0, 1.0, MESHSIZE, dtype=dtype)
N1FORMS = NPOU * (NPOU - 1) // 2


def evalPhi_i(x):
    return torch.relu(1.0 - torch.abs(x.unsqueeze(0) - points.unsqueeze(1)) / h)

def evalGradPhi_i(x):
    supp = (evalPhi_i(x) > 0).double()
    sP = (points.unsqueeze(1) >  x.unsqueeze(0)).double()
    sN = (points.unsqueeze(1) <= x.unsqueeze(0)).double()
    return supp * (sN - sP) / h

_rp, _rw = np.polynomial.legendre.leggauss(NQUAD)
_rp = torch.tensor(_rp, dtype=dtype); _rw = torch.tensor(_rw, dtype=dtype)
xq = (points[:-1].unsqueeze(-1) + 0.5 * h * (1.0 + _rp.unsqueeze(0))).flatten()
wq = (_rw.unsqueeze(0).expand(MESHSIZE - 1, -1) * (h / 2.0)).flatten()
_idx = torch.argsort(xq); xq = xq[_idx]; wq = wq[_idx]
NQ = xq.shape[0]

nodal_basis = evalPhi_i(xq)
nodal_gradb = evalGradPhi_i(xq)


# ---------------------------------------------------------------------------
# Trainable W + Whitney forms at r=1
# ---------------------------------------------------------------------------

class WijParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.randn(NPOU - 2, MESHSIZE - 2, dtype=dtype))

    def build(self):
        interior = torch.softmax(self.logit, dim=0)
        rL = torch.zeros(1, MESHSIZE, dtype=dtype); rL[0, 0] = 1.0
        rR = torch.zeros(1, MESHSIZE, dtype=dtype); rR[0, -1] = 1.0
        rM = torch.cat([
            torch.zeros(NPOU - 2, 1, dtype=dtype), interior,
            torch.zeros(NPOU - 2, 1, dtype=dtype),
        ], dim=1)
        return torch.cat([rL, rM, rR], dim=0)


def eval_psi1_r1(W, basis, gradb):
    p0 = W @ basis; g0 = W @ gradb
    full = p0.unsqueeze(0) * g0.unsqueeze(1) - p0.unsqueeze(1) * g0.unsqueeze(0)
    mask = torch.triu(torch.ones(NPOU, NPOU, dtype=torch.bool), diagonal=1)
    return full[mask]

def make_delta0():
    D = torch.zeros(N1FORMS, NPOU, dtype=dtype)
    cnt = 0
    for i in range(NPOU):
        for j in range(i + 1, NPOU):
            D[cnt, j] = 1.0; D[cnt, i] = -1.0; cnt += 1
    return D

delta0 = make_delta0()


# ---------------------------------------------------------------------------
# Enriched V^0_r, V^1_r, and operators
# ---------------------------------------------------------------------------

def shifted_legendre(x, r):
    """Evaluate shifted Legendre polynomials P_k(2x-1) for k=0..r-1 on [0,1].
    Returns (r, len(x)) stack.  Bonnet recurrence; orthonormal-in-L2 basis of P_{r-1}."""
    y = 2 * x - 1
    P = [torch.ones_like(x), y.clone()]
    for k in range(1, r - 1):
        P.append(((2 * k + 1) * y * P[k] - k * P[k - 1]) / (k + 1))
    return torch.stack(P[:r], dim=0) if r >= 1 else torch.empty(0, x.shape[0], dtype=x.dtype)

def shifted_legendre_deriv(x, r):
    """Derivative of shifted Legendre polynomials on [0,1].  Returns (r, len(x))."""
    # d/dx P_k(2x-1) = 2 * d/dy P_k(y)|_{y=2x-1}.  Use  (1 - y^2) P_k'(y) = k (P_{k-1}(y) - y P_k(y)).
    y = 2 * x - 1
    P = [torch.ones_like(x), y.clone()]
    for k in range(1, r):
        if len(P) <= k:
            P.append(((2 * k - 1) * y * P[k - 1] - (k - 1) * P[k - 2]) / k)
    dP = [torch.zeros_like(x)]
    for k in range(1, r):
        one_minus_y2 = 1 - y * y
        # Near-endpoint stability: P_k'(y) = (k/(1-y^2)) [P_{k-1}(y) - y P_k(y)] ; at |y|=1 use
        # P_k'(1) = k(k+1)/2,  P_k'(-1) = (-1)^{k+1} k(k+1)/2.
        safe = one_minus_y2.abs() > 1e-12
        val = torch.where(safe,
                          k * (P[k - 1] - y * P[k]) / torch.where(safe, one_minus_y2, torch.ones_like(one_minus_y2)),
                          torch.where(y > 0,
                                      torch.full_like(y, k * (k + 1) / 2.0),
                                      torch.full_like(y, ((-1) ** (k + 1)) * k * (k + 1) / 2.0)))
        dP.append(val)
    return 2.0 * torch.stack(dP[:r], dim=0) if r >= 1 else torch.empty(0, x.shape[0], dtype=x.dtype)


def eval_V0_r(W, r, x=None, basis=None):
    """V^0_r = span{ phi_i(x) * P_k(2x-1) : 0 <= k < r } using shifted Legendre polynomials.
    Span = trained POU tensor polynomials of degree <= r-1; same as monomial tensoring,
    but the Legendre basis is far better conditioned."""
    if x is None: x = xq
    if basis is None:
        basis = nodal_basis if (x.shape == xq.shape and torch.equal(x, xq)) else evalPhi_i(x)
    p0 = W @ basis
    L = shifted_legendre(x, r)                                    # (r, Nq)
    return (L.unsqueeze(1) * p0.unsqueeze(0)).reshape(r * NPOU, -1)

def eval_dV0_r(W, r, x=None, basis=None, gradb=None):
    """d/dx of V^0_r basis. Uses d(phi_i P_k) = phi_i' P_k + phi_i P_k'."""
    if x is None: x = xq
    if basis is None or gradb is None: basis = nodal_basis; gradb = nodal_gradb
    p0 = W @ basis; g0 = W @ gradb
    L = shifted_legendre(x, r)                                    # (r, Nq)
    dL = shifted_legendre_deriv(x, r)                             # (r, Nq)
    t1 = L.unsqueeze(1) * g0.unsqueeze(0)                         # (r, NPOU, Nq): phi_i' P_k
    t2 = dL.unsqueeze(1) * p0.unsqueeze(0)                        # (r, NPOU, Nq): phi_i P_k'
    return (t1 + t2).reshape(r * NPOU, -1)

def eval_V1_r(W, r, x=None, basis=None, gradb=None):
    """V^1_r = span{ P_k(2x-1) * psi_ij } using shifted Legendre."""
    if x is None: x = xq
    if basis is None or gradb is None: basis = nodal_basis; gradb = nodal_gradb
    p1 = eval_psi1_r1(W, basis, gradb)
    L = shifted_legendre(x, r)
    return (L.unsqueeze(1) * p1.unsqueeze(0)).reshape(r * N1FORMS, -1)


def delta0_r(r):
    """Block-diagonal coboundary: (r*N1FORMS, r*NPOU)."""
    return torch.block_diag(*[delta0 for _ in range(r)])


def assemble_r(W, r):
    V0  = eval_V0_r(W, r)
    dV0 = eval_dV0_r(W, r)
    V1  = eval_V1_r(W, r)
    M0 = torch.einsum('iq,jq,q->ij', V0, V0, wq)
    M1 = torch.einsum('eq,fq,q->ef', V1, V1, wq)
    K  = torch.einsum('iq,jq,q->ij', dV0, dV0, wq)
    return V0, dV0, V1, M0, M1, K


def boundary_lifting_and_nullspace(r, bc_left=0.0, bc_right=1.0):
    """Return (u_lift, T) such that u = u_lift + T @ u_red satisfies BCs for any u_red.
    BC constraints on the flat r*NPOU coord vector:
      c1:  u_{0, 0} = bc_left               (flat index 0)
      c2:  sum_k u_{NPOU-1, k} = bc_right   (flat indices (NPOU-1), (NPOU-1)+NPOU, ...)
    """
    n = r * NPOU
    # Lifting
    u_lift = torch.zeros(n, dtype=dtype)
    u_lift[0] = bc_left                                      # u_{0, 0}
    u_lift[NPOU - 1] = bc_right                              # u_{NPOU-1, 0}; higher-k terms 0
    # Constraint matrix C u = 0 for the HOMOGENEOUS residual
    C = torch.zeros(2, n, dtype=dtype)
    C[0, 0] = 1.0
    for k in range(r):
        C[1, k * NPOU + (NPOU - 1)] = 1.0
    _, _, Vh = torch.linalg.svd(C, full_matrices=True)
    T = Vh[2:].T                                             # (n, n-2), null-space basis
    return u_lift, T


# ---------------------------------------------------------------------------
# Flux NN at order r
# ---------------------------------------------------------------------------

class FluxNNr(nn.Module):
    def __init__(self, r, hidden=10):
        super().__init__()
        in_dim = r * NPOU
        out_dim = r * N1FORMS
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.Tanh(),
        )
        self.W_out = nn.Parameter(torch.zeros(out_dim, hidden, dtype=dtype))

    def forward(self, u):
        """u: (B, in_dim) -> (B, out_dim)."""
        h = self.net(u)
        return torch.einsum('ew,bw->be', self.W_out, h)


def compute_N_and_J(flux, u):
    """Return N(u) and dN/du. u: (B, in); returns (B, out), (B, out, in)."""
    u_var = u.detach().requires_grad_(True)
    N = flux(u_var)
    B, out_dim = N.shape
    J = []
    for e in range(out_dim):
        ge = torch.autograd.grad(N[:, e].sum(), u_var, create_graph=True)[0]
        J.append(ge)
    return N, torch.stack(J, dim=1)   # J: (B, out, in)


# ---------------------------------------------------------------------------
# Newton solve at order r
# ---------------------------------------------------------------------------

def newton_solve_r(W, r, flux, log_mu, bc_left, bc_right, n_iter=10,
                   u_init=None, rtol=1e-4, verbose=False):
    """Solve R_r(u) = mu K_r u + (delta0_r)^T N_r(u) - F_r = 0 with Newton.

    F_r = 0 in this problem (body force = 0); BCs enter via a lifting u = u_lift + T u_red.
    The homogeneous residual is R_tilde(u_red) = T^T R(u_lift + T u_red).
    Newton step:  (T^T [mu K + delta0_r^T J_N] T) d_u_red = -T^T R.
    """
    B = bc_left.shape[0]  # batch
    _, _, _, _, M1_r, K_r = assemble_r(W, r)
    D_r = delta0_r(r)

    n = r * NPOU

    # Per-batch lifting: u_lift shape (B, n)
    u_lift = torch.zeros(B, n, dtype=dtype)
    u_lift[:, 0] = bc_left
    u_lift[:, NPOU - 1] = bc_right
    # Nullspace (BC-independent)
    C = torch.zeros(2, n, dtype=dtype)
    C[0, 0] = 1.0
    for k in range(r):
        C[1, k * NPOU + (NPOU - 1)] = 1.0
    _, _, Vh = torch.linalg.svd(C, full_matrices=True)
    T = Vh[2:].T                                             # (n, n-2)
    dim_red = T.shape[1]

    mu = torch.exp(log_mu)

    # Initial guess
    if u_init is None:
        u_red = torch.zeros(B, dim_red, dtype=dtype)
    else:
        u_red = (u_init - u_lift) @ T                        # project to reduced space

    for it in range(n_iter):
        u_full = u_lift + u_red @ T.T                        # (B, n)
        # NN term
        N_u, J_N = compute_N_and_J(flux, u_full)             # (B, n_1form), (B, n_1form, n)
        # Residual:  mu K u + D^T N - 0
        res_full = mu * (K_r @ u_full.T).T + torch.einsum('ef,bf->be', D_r.T, N_u)
        res_red = res_full @ T                               # (B, dim_red)
        # Jacobian:  mu K + D^T J_N
        A_full = mu * K_r.unsqueeze(0).expand(B, -1, -1) \
                 + torch.einsum('ef,bfg->beg', D_r.T, J_N)   # (B, n, n)
        A_red = torch.einsum('ij,bjk,kl->bil', T.T, A_full, T)  # (B, dim_red, dim_red)
        # Pinv solve for Newton step
        step = torch.zeros_like(u_red)
        for b in range(B):
            step[b] = -torch.linalg.pinv(A_red[b], rtol=rtol) @ res_red[b]
        u_red = u_red + step
        if verbose:
            r_norm = res_red.norm().item()
            print(f"  newton {it+1}: ||res||_red = {r_norm:.3e}")
    return u_lift + u_red @ T.T                              # full coords, (B, n)


# ---------------------------------------------------------------------------
# p-multigrid V-cycle as Newton inner linear solver
# ---------------------------------------------------------------------------
# Solves A x = b where A = T^T [mu K_r + delta0_r^T J_N] T arising in Newton.
# Hierarchy: V^0_1 subset V^0_2 subset ... via natural injection P_{r-1 -> r}.
# Coarse operator: Galerkin  A_c = P^T A_f P.
# Smoother: weighted Jacobi with pseudoinverse-diagonal (handles zero diag).

def prolong_natural_full(r_c, r_f):
    """Full (pre-BC) prolongation V^0_{r_c} -> V^0_{r_f} (natural injection of polynomial degrees).
    DOF layout (k, i) -> k*NPOU + i.  First r_c*NPOU indices of r_f match r_c 1-1."""
    assert r_c <= r_f
    P = torch.zeros(r_f * NPOU, r_c * NPOU, dtype=dtype)
    for idx in range(r_c * NPOU):
        P[idx, idx] = 1.0
    return P

def bc_nullspace_T(r):
    n = r * NPOU
    C = torch.zeros(2, n, dtype=dtype)
    C[0, 0] = 1.0
    for k in range(r):
        C[1, k * NPOU + (NPOU - 1)] = 1.0
    _, _, Vh = torch.linalg.svd(C, full_matrices=True)
    return Vh[2:].T

def v_cycle_two_level(A_fine, b_fine, T_f, T_c, r_c, r_f,
                      n_pre=3, n_post=3, n_iters=10, omega=2/3,
                      rtol_coarse=1e-10, verbose=False):
    """Two-level p-multigrid V-cycle for the reduced system A_fine u = b_fine.
    A_fine acts on T_f-reduced coordinates of the r_f mesh.  Returns (u, history)."""
    P_full = prolong_natural_full(r_c, r_f)
    P_red  = T_f.T @ P_full @ T_c               # reduced fine <- reduced coarse

    # Galerkin coarse operator on reduced coords
    A_coarse = P_red.T @ A_fine @ P_red

    # Weighted-Jacobi diagonal (use pinv-style safeguard for zero diag entries)
    D = torch.diag(A_fine)
    D_inv = torch.where(D.abs() > 1e-14, 1.0 / D, torch.zeros_like(D))

    u = torch.zeros(T_f.shape[1], dtype=dtype)
    history = []
    b_norm = b_fine.norm().item() + 1e-30
    for it in range(n_iters):
        # Pre-smooth
        for _ in range(n_pre):
            u = u + omega * D_inv * (b_fine - A_fine @ u)
        # Restrict residual
        r = b_fine - A_fine @ u
        r_c_vec = P_red.T @ r
        # Coarse solve (direct pinv)
        e_c = torch.linalg.pinv(A_coarse, rtol=rtol_coarse) @ r_c_vec
        # Prolong + correct
        u = u + P_red @ e_c
        # Post-smooth
        for _ in range(n_post):
            u = u + omega * D_inv * (b_fine - A_fine @ u)
        res = (b_fine - A_fine @ u).norm().item() / b_norm
        history.append(res)
        if verbose:
            print(f"  V-cycle {it+1:2d}: rel. residual = {res:.4e}")
    return u, history


def assemble_newton_system(W, r, flux, log_mu, u_full, bc_left, bc_right):
    """Assemble the BC-reduced Newton system A u_red = b at a given u_full (linearization point).
    Returns (A, b, T) where A = T^T (mu K + D^T J_N) T  and  b = -T^T R."""
    _, _, _, _, _, K_r = assemble_r(W, r)
    D_r = delta0_r(r)
    T = bc_nullspace_T(r)
    mu = torch.exp(log_mu)

    B = u_full.shape[0]
    N_u, J_N = compute_N_and_J(flux, u_full)                 # (B, n1), (B, n1, n0)

    res_full = mu * (K_r @ u_full.T).T + torch.einsum('ef,bf->be', D_r.T, N_u)
    A_full = mu * K_r.unsqueeze(0).expand(B, -1, -1) \
             + torch.einsum('ef,bfg->beg', D_r.T, J_N)       # (B, n, n)
    A_red = torch.einsum('ij,bjk,kl->bil', T.T, A_full, T)   # (B, dim_red, dim_red)
    b_red = -res_full @ T                                     # (B, dim_red)
    return A_red, b_red, T


# ---------------------------------------------------------------------------
# Advection-diffusion exact solution
# ---------------------------------------------------------------------------

def advdiff_exact(x, eps):
    exp_x = torch.exp(x / eps)
    exp_1 = torch.exp(torch.tensor(1.0 / eps, dtype=dtype))
    u = (1.0 - exp_x) / (1.0 - exp_1)
    return u


# ---------------------------------------------------------------------------
# Training at order r
# ---------------------------------------------------------------------------

def train_at_order_r(r, n_steps=1500, n_newton=8, lr=1e-3, Pe_target=10.0,
                     Nbatch=2, seed=42, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)

    eps_true = 1.0 / (2.0 * Pe_target)
    uscale = torch.linspace(1.0, 3.0, Nbatch, dtype=dtype)
    u_ex_q = advdiff_exact(xq, eps_true)
    udata = u_ex_q.unsqueeze(0) * uscale.unsqueeze(1)         # (B, NQ)

    bc_left  = torch.zeros(Nbatch, dtype=dtype)
    bc_right = uscale.clone()

    wij = WijParam()
    flux = FluxNNr(r).double()
    log_mu = nn.Parameter(torch.tensor(0.0, dtype=dtype))

    params = list(wij.parameters()) + list(flux.parameters()) + [log_mu]
    opt = torch.optim.Adam(params, lr=lr)

    history = []
    u_prev = None
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        W = wij.build()
        u_full = newton_solve_r(W, r, flux, log_mu, bc_left, bc_right,
                                n_iter=n_newton, u_init=u_prev, verbose=False)
        u_prev = u_full.detach()
        # Evaluate solution at quadrature points:  u_h(x_q) = u_full @ V0_r(x_q)
        V0 = eval_V0_r(W, r)                                  # (r*NPOU, NQ)
        u_h = u_full @ V0                                     # (B, NQ)
        loss = ((u_h - udata) ** 2).sum()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if verbose and (step == 1 or step % 250 == 0):
            with torch.no_grad():
                mu_val = torch.exp(log_mu).item()
            print(f"  r={r}  step {step:5d}: loss={loss.item():.4e}  mu={mu_val:.5f}  "
                  f"(eps_true={eps_true:.5f})")
    return {
        'r': r, 'history': history, 'wij': wij, 'flux': flux, 'log_mu': log_mu,
        'u_h_final': u_h.detach(), 'udata': udata, 'eps_true': eps_true,
    }


def main():
    print("=" * 72)
    print("Smoke test: assemble + Newton solve at r=1 with zero NN")
    print("=" * 72)
    wij = WijParam(); flux = FluxNNr(1).double()
    log_mu = torch.tensor(0.0, dtype=dtype)
    bc_l = torch.zeros(2, dtype=dtype); bc_r = torch.tensor([1.0, 3.0], dtype=dtype)
    W = wij.build().detach()
    u = newton_solve_r(W, 1, flux, log_mu, bc_l, bc_r, n_iter=3, verbose=True).detach()
    # With mu=exp(0)=1, no NN, no forcing, solution should be linear: u_h(x) = scale * x
    V0 = eval_V0_r(W, 1)
    u_h = u @ V0
    for b in range(2):
        lin = bc_r[b] * xq
        err = (u_h[b] - lin).abs().max().item()
        print(f"  batch {b}: max |u_h - linear| = {err:.2e}")

    print()
    print("=" * 72)
    print("Training at r = 1, 2, 3  (short run: 500 steps each)")
    print("=" * 72)
    results = {}
    for r in [1, 2, 3]:
        print(f"\n--- order r = {r} ---")
        res = train_at_order_r(r, n_steps=500, n_newton=6, verbose=True)
        results[r] = res

    print()
    print("=" * 72)
    print("Final comparison")
    print("=" * 72)
    for r, res in results.items():
        final = res['history'][-1]
        u_final = res['u_h_final']
        l2_err = torch.sqrt(((u_final - res['udata']) ** 2 * wq.unsqueeze(0)).sum(dim=1))
        print(f"  r={r}: final_loss={final:.4e},  per-batch L2 err = {l2_err.tolist()}")

    return results


if __name__ == "__main__":
    main()
