"""Prototype: hp-enrichment of trained Whitney forms for 1D Poisson.

Tower V^0_1 subset V^0_2 subset ... with V^0_r = span{phi_i * x^k : 0 <= k < r},
where phi_i = (W @ P1_hats)_i.

Main path: primal Galerkin on V^0_r,  K_r u = F_r  with  K_{ab} = int phi_a' phi_b'.
Also shows: the naive tensor-product 1-form space V^1_r = span{x^k psi_ij} does NOT
close under d; a minimal enlargement V^1_r + V^0_{r-1} does.

Tests:
  (1) PoU sanity, de Rham identity at r=1
  (2) Closure:  naive V^1_r fails at r>=2;  enlarged V^1_r holds to machine eps
  (3) p-convergence of primal Galerkin on -u'' = pi^2 sin(pi x), u(0)=u(1)=0
  (4) Two-level p-multigrid V-cycle on the primal system

Run: python _hp_prototype.py
"""
import numpy as np
import torch

torch.manual_seed(0)
dtype = torch.float64


# ---------------------------------------------------------------------------
# Fine mesh and P1 basis (follows DataDriven_Whitney_Forms.ipynb conventions)
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
    signPlus = (points.unsqueeze(1) >  x.unsqueeze(0)).double()
    signNeg  = (points.unsqueeze(1) <= x.unsqueeze(0)).double()
    return supp * (signNeg - signPlus) / h


# Gauss-Legendre quadrature per element
_rp, _rw = np.polynomial.legendre.leggauss(NQUAD)
_rp = torch.tensor(_rp, dtype=dtype)
_rw = torch.tensor(_rw, dtype=dtype)
xq = (points[:-1].unsqueeze(-1) + 0.5 * h * (1.0 + _rp.unsqueeze(0))).flatten()
wq = (_rw.unsqueeze(0).expand(MESHSIZE - 1, -1) * (h / 2.0)).flatten()
_idx = torch.argsort(xq); xq = xq[_idx]; wq = wq[_idx]
NQ = xq.shape[0]

nodal_basis = evalPhi_i(xq)
nodal_gradb = evalGradPhi_i(xq)


# ---------------------------------------------------------------------------
# W and Whitney forms
# ---------------------------------------------------------------------------

def make_W(seed=0):
    """Softmax-parameterized W with boundary pinning: (NPOU, MESHSIZE)."""
    torch.manual_seed(seed)
    logit = torch.randn(NPOU - 2, MESHSIZE - 2, dtype=dtype)
    interior = torch.softmax(logit, dim=0)
    row_L = torch.zeros(1, MESHSIZE, dtype=dtype); row_L[0, 0] = 1.0
    row_R = torch.zeros(1, MESHSIZE, dtype=dtype); row_R[0, -1] = 1.0
    row_M = torch.cat([
        torch.zeros(NPOU - 2, 1, dtype=dtype),
        interior,
        torch.zeros(NPOU - 2, 1, dtype=dtype),
    ], dim=1)
    return torch.cat([row_L, row_M, row_R], dim=0)


def eval_psi0(Wij, basis): return Wij @ basis
def eval_gpsi0(Wij, gradb): return Wij @ gradb

def eval_psi1(Wij, basis, gradb):
    p0 = Wij @ basis; g0 = Wij @ gradb
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
# Enriched spaces V^0_r and the naive V^1_r
# ---------------------------------------------------------------------------

def dof_list(r):
    """Active (i, k) pairs for V^0_r.  k=0 gives all i; k>=1 gives only interior i.
    Returns list of (i, k) tuples.  Total length = NPOU + (r-1)*(NPOU-2).
    """
    out = [(i, 0) for i in range(NPOU)]
    for k in range(1, r):
        for i in range(1, NPOU - 1):
            out.append((i, k))
    return out

def dim_V0(r): return NPOU + (r - 1) * (NPOU - 2)

def eval_V0_r(Wij, r, x=None, basis=None):
    """V^0_r basis at eval points.  Returns shape (dim_V0(r), Nq).
    Boundary POU functions (i=0 and i=NPOU-1) are NOT enriched: their polynomial dilations
    are ill-conditioned on their tiny pinned supports, so we leave them at k=0 only."""
    if x is None: x = xq
    if basis is None:
        basis = nodal_basis if (x.shape == xq.shape and torch.equal(x, xq)) else evalPhi_i(x)
    psi0 = Wij @ basis                                   # (NPOU, Nq)
    dofs = dof_list(r)
    out = torch.zeros(len(dofs), x.shape[0], dtype=dtype)
    for m, (i, k) in enumerate(dofs):
        out[m] = (x**k) * psi0[i]
    return out

def eval_dV0_r(Wij, r, x=None, basis=None, gradb=None):
    """d/dx V^0_r basis. Returns shape (dim_V0(r), Nq).
    d(phi_i x^k)/dx = phi_i' x^k + k phi_i x^{k-1}."""
    if x is None: x = xq
    if basis is None or gradb is None:
        basis = nodal_basis; gradb = nodal_gradb
    psi0  = Wij @ basis
    gpsi0 = Wij @ gradb
    dofs = dof_list(r)
    out = torch.zeros(len(dofs), x.shape[0], dtype=dtype)
    for m, (i, k) in enumerate(dofs):
        x_pow = x**k
        x_lower = x**(k - 1) if k >= 1 else torch.zeros_like(x)
        out[m] = x_pow * gpsi0[i] + k * x_lower * psi0[i]
    return out

def eval_V1_r_naive(Wij, r, x=None, basis=None, gradb=None):
    """Naive tensor-product 1-form basis, shape (r*N1FORMS, Nq). Does NOT close for r>=2."""
    if x is None: x = xq
    if basis is None or gradb is None:
        basis = nodal_basis; gradb = nodal_gradb
    psi1 = eval_psi1(Wij, basis, gradb)
    powers = torch.stack([x**k for k in range(r)], dim=0)
    out = powers.unsqueeze(1) * psi1.unsqueeze(0)
    return out.reshape(r * N1FORMS, -1)

def eval_V1_r_enlarged(Wij, r, x=None, basis=None, gradb=None):
    """Enlarged V^1_r = V^1_r^naive + V^0_{r-1}. Closes under d."""
    V1n = eval_V1_r_naive(Wij, r, x, basis, gradb)
    if r == 1:
        return V1n
    V0_lower = eval_V0_r(Wij, r - 1, x, basis)
    return torch.cat([V1n, V0_lower], dim=0)


# ---------------------------------------------------------------------------
# Primal Galerkin: K u = F  with  K_{ab} = int phi_a' phi_b',  F_a = int f phi_a
# ---------------------------------------------------------------------------

def assemble_primal(Wij, r):
    """Return (M0, K) at order r.
        M0: mass matrix, (r*NPOU, r*NPOU)
        K : stiffness,   (r*NPOU, r*NPOU)
    """
    V0  = eval_V0_r(Wij, r)
    dV0 = eval_dV0_r(Wij, r)
    M0 = torch.einsum('iq,jq,q->ij', V0, V0, wq)
    K  = torch.einsum('iq,jq,q->ij', dV0, dV0, wq)
    return M0, K, V0, dV0

def boundary_nullspace(r):
    """Null-space basis T (n, n-2) for V^0_r under u(0)=u(1)=0.
    With boundary-POU-not-enriched layout, the BCs are:
      u(0) = u_{0,0}      (DOF index 0)
      u(1) = u_{N-1, 0}   (DOF index NPOU-1)
    Both are single-DOF pins, so T is trivial: drop those two rows.
    """
    n = dim_V0(r)
    keep = [m for m in range(n) if m != 0 and m != (NPOU - 1)]
    T = torch.zeros(n, len(keep), dtype=dtype)
    for j, k_idx in enumerate(keep):
        T[k_idx, j] = 1.0
    return T

def rhs_V0_r(Wij, r, f_xq):
    V0 = eval_V0_r(Wij, r)
    return torch.einsum('jq,q,q->j', V0, f_xq, wq)

def solve_primal_direct(Wij, r, f_xq):
    """Solve T^T K T u_red = T^T F; return full u."""
    _, K, _, _ = assemble_primal(Wij, r)
    T = boundary_nullspace(r)
    F = rhs_V0_r(Wij, r, f_xq)
    K_red = T.T @ K @ T
    b_red = T.T @ F
    u_red = torch.linalg.solve(K_red, b_red)
    return T @ u_red, K, T


# ---------------------------------------------------------------------------
# Closure tests
# ---------------------------------------------------------------------------

def closure_residual(Wij, r, V1_eval_fn):
    """L2 projection of dV^0_r onto span of V1_eval_fn(...); return (max, mean) relative residual."""
    V1  = V1_eval_fn(Wij, r)
    dV0 = eval_dV0_r(Wij, r)
    M1  = torch.einsum('eq,fq,q->ef', V1, V1, wq)
    rhs = torch.einsum('eq,jq,q->ej', V1, dV0, wq)
    # Pseudoinverse via SVD handles singular M1
    C = torch.linalg.pinv(M1, rtol=1e-12) @ rhs
    proj = C.T @ V1
    residual = dV0 - proj
    num = torch.einsum('jq,jq,q->j', residual, residual, wq).sqrt()
    den = torch.einsum('jq,jq,q->j', dV0,       dV0,       wq).sqrt() + 1e-30
    return (num / den).max().item(), (num / den).mean().item()


# ---------------------------------------------------------------------------
# Prolongation V^0_{r-1} -> V^0_r  and restriction (L^2 adjoint)
# ---------------------------------------------------------------------------
# DOF layout (k, i) -> k*NPOU + i.  V^0_{r-1} injects into V^0_r as the first
# (r-1)*NPOU coordinates; the new top (r-1)*NPOU..r*NPOU coordinates are zero.

def prolong_matrix(r_coarse, r_fine):
    """Natural injection V^0_{r_coarse} -> V^0_{r_fine}.  Maps (i, k) -> (i, k)."""
    assert r_coarse <= r_fine
    dofs_c = dof_list(r_coarse)
    dofs_f = dof_list(r_fine)
    # Since dof_list is deterministic with enriched-interior ordering, the first
    # dim_V0(r_coarse) entries of dofs_f are precisely dofs_c.
    assert dofs_f[:len(dofs_c)] == dofs_c
    P = torch.zeros(len(dofs_f), len(dofs_c), dtype=dtype)
    for m in range(len(dofs_c)):
        P[m, m] = 1.0
    return P

def restrict_matrix_L2(Wij, r_coarse, r_fine):
    """R = M0_c^{-1} P^T M0_f.  Dual of prolong in L^2."""
    M0_c, _, _, _ = assemble_primal(Wij, r_coarse)
    M0_f, _, _, _ = assemble_primal(Wij, r_fine)
    P = prolong_matrix(r_coarse, r_fine)
    return torch.linalg.solve(M0_c, P.T @ M0_f), P


# ---------------------------------------------------------------------------
# Two-level V-cycle on the REDUCED primal system T^T K T u_red = T^T F
# ---------------------------------------------------------------------------

def weighted_jacobi(A, b, u, omega=2/3):
    return u + omega * (b - A @ u) / torch.diag(A)

def reduced_op(Wij, r):
    """Return (K_red, T, F_builder) for solving primal Galerkin with hom. Dirichlet."""
    _, K, _, _ = assemble_primal(Wij, r)
    T = boundary_nullspace(r)
    K_red = T.T @ K @ T
    return K_red, T

def v_cycle_two_level_primal(Wij, r_fine, r_coarse, f_xq,
                             n_pre=3, n_post=3, n_iters=10, verbose=True):
    K_f, T_f = reduced_op(Wij, r_fine)
    K_c_direct, T_c = reduced_op(Wij, r_coarse)   # for reference
    F_f = rhs_V0_r(Wij, r_fine, f_xq)
    b_f = T_f.T @ F_f

    P_full = prolong_matrix(r_coarse, r_fine)       # (r_f*N, r_c*N)
    # Reduced prolongation: u_red_fine = T_f^T u_full,  u_full = T_f u_red_fine.
    # For the reduced coefficient spaces: P_red = T_f^T P_full T_c.
    P_red = T_f.T @ P_full @ T_c
    # Galerkin coarse operator (guaranteed SPD if fine is)
    K_c = P_red.T @ K_f @ P_red

    u_red = torch.zeros(T_f.shape[1], dtype=dtype)
    history = []
    b_norm = b_f.norm().item() + 1e-30
    for it in range(n_iters):
        for _ in range(n_pre):
            u_red = weighted_jacobi(K_f, b_f, u_red)
        r_f = b_f - K_f @ u_red
        r_c = P_red.T @ r_f
        e_c = torch.linalg.solve(K_c, r_c)
        u_red = u_red + P_red @ e_c
        for _ in range(n_post):
            u_red = weighted_jacobi(K_f, b_f, u_red)
        res = (b_f - K_f @ u_red).norm().item() / b_norm
        history.append(res)
        if verbose:
            print(f"  V-cycle {it+1}: residual = {res:.4e}")
    return T_f @ u_red, history

def jacobi_only(Wij, r, f_xq, n_iters=500):
    K_red, T = reduced_op(Wij, r)
    F = rhs_V0_r(Wij, r, f_xq)
    b = T.T @ F
    u = torch.zeros(T.shape[1], dtype=dtype)
    hist = []
    bn = b.norm().item() + 1e-30
    for _ in range(n_iters):
        u = weighted_jacobi(K_red, b, u)
        hist.append((b - K_red @ u).norm().item() / bn)
    return T @ u, hist


# ---------------------------------------------------------------------------
# Test problem
# ---------------------------------------------------------------------------

def u_exact_fn(x): return torch.sin(torch.pi * x)
def f_exact_fn(x): return (torch.pi ** 2) * torch.sin(torch.pi * x)

def solution_error_L2(Wij, r, u_full, u_ex_q):
    V0 = eval_V0_r(Wij, r)
    u_h = u_full @ V0
    err = u_h - u_ex_q
    return torch.sqrt(torch.einsum('q,q,q->', err, err, wq)).item()


# ===========================================================================
def main():
    Wij = make_W(seed=0).detach()

    pou_err = (Wij @ evalPhi_i(xq)).sum(dim=0).sub(1.0).abs().max().item()
    V1_r1 = eval_psi1(Wij, nodal_basis, nodal_gradb)
    M1_r1 = torch.einsum('eq,fq,q->ef', V1_r1, V1_r1, wq)
    divMat = torch.einsum('eq,jq,q->ej', V1_r1, Wij @ nodal_gradb, wq)
    dR_err = (divMat.T - (-delta0.T @ M1_r1)).norm().item() / divMat.norm().item()

    print(f"PoU error: {pou_err:.2e}")
    print(f"de Rham identity at r=1: {dR_err:.2e}")

    print("\n=== (2) Closure: naive V^1_r = span{x^k psi_ij} ===")
    for r in range(1, 6):
        mx, mn = closure_residual(Wij, r, eval_V1_r_naive)
        print(f"  r={r}: max rel res = {mx:.2e},  mean = {mn:.2e}")

    print("\n=== (2') Closure: enlarged V^1_r = V^1_r^naive + V^0_{r-1} ===")
    for r in range(1, 6):
        mx, mn = closure_residual(Wij, r, eval_V1_r_enlarged)
        print(f"  r={r}: max rel res = {mx:.2e},  mean = {mn:.2e}")

    print("\n=== (3) p-convergence of primal Galerkin ===")
    u_ex_q = u_exact_fn(xq); f_q = f_exact_fn(xq)
    errors = []
    for r in range(1, 8):
        u_full, K, T = solve_primal_direct(Wij, r, f_q)
        err = solution_error_L2(Wij, r, u_full, u_ex_q)
        errors.append(err)
        print(f"  r={r}: DOFs={dim_V0(r):3d} (reduced {T.shape[1]:3d}),  L2 error = {err:.4e}")

    print("\n=== (4) Two-level p-multigrid V-cycle: r_fine=4 -> r_coarse=1 ===")
    u_vc, hist_vc = v_cycle_two_level_primal(Wij, 4, 1, f_q, n_pre=3, n_post=3, n_iters=10)
    u_dir, _, _ = solve_primal_direct(Wij, 4, f_q)
    diff = (u_vc - u_dir).norm().item() / u_dir.norm().item()
    print(f"  V-cycle vs direct relative diff: {diff:.4e}")

    print("\n=== (5) Jacobi alone (same system), 500 iters ===")
    _, hist_j = jacobi_only(Wij, r=4, f_xq=f_q, n_iters=500)
    print(f"  Jacobi residual after  50 iters: {hist_j[ 49]:.4e}")
    print(f"  Jacobi residual after 250 iters: {hist_j[249]:.4e}")
    print(f"  Jacobi residual after 500 iters: {hist_j[-1]:.4e}")
    print(f"  V-cycle residual after 10 cycles: {hist_vc[-1]:.4e}")

    print("\nAll smoke tests done.")


if __name__ == "__main__":
    main()
