import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
np.random.seed(42)
print("Imports OK")

# GIVEN: helper to create standard graphs
def make_graph(graph_type, n=20):
    """Create a named NetworkX graph."""
    if graph_type == 'path':
        return nx.path_graph(n)
    elif graph_type == 'cycle':
        return nx.cycle_graph(n)
    elif graph_type == 'grid':
        side = int(np.sqrt(n))
        return nx.grid_2d_graph(side, side)
    elif graph_type == 'petersen':
        return nx.petersen_graph()
    else:
        raise ValueError(f'Unknown graph type: {graph_type}')

G = make_graph('path', 20)
print(f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

# GIVEN: visualize a small graph
pos = nx.spring_layout(G, seed=0)
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=300, font_size=8)
plt.title('Path graph (n=20)')
plt.show()

# SOLUTION 1
def build_coboundary(G):
    """Build sparse coboundary (gradient) operator delta0."""
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
    edges = sorted(set(edges))  # canonical orientation i < j
    nE, nN = len(edges), len(nodes)
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(edges):
        rows.extend([k, k])
        cols.extend([node_idx[i], node_idx[j]])
        vals.extend([-1.0, 1.0])
    delta0 = sp.csr_matrix((vals, (rows, cols)), shape=(nE, nN))
    return delta0, edges

delta0, edge_list = build_coboundary(G)
print(f'delta0 shape: {delta0.shape}  (|E|={len(edge_list)}, N={G.number_of_nodes()})')

# SOLUTION 2
def build_laplacian(delta0, weights=None):
    """Graph Laplacian L = delta0^T W delta0."""
    if weights is None:
        L = delta0.T @ delta0
    else:
        W = sp.diags(weights)
        L = delta0.T @ W @ delta0
    return L

L = build_laplacian(delta0)
print(f'Laplacian shape: {L.shape}')
print(f'Laplacian is symmetric: {np.allclose(L.toarray(), L.toarray().T)}')

# SOLUTION 3
n = 20
G_path = make_graph('path', n)
d0, _ = build_coboundary(G_path)
L_path = build_laplacian(d0)
eigs = np.linalg.eigvalsh(L_path.toarray())

# Analytic eigenvalues
eigs_analytic = [2 * (1 - np.cos(k * np.pi / n)) for k in range(n)]

print(f'lambda_0 = {eigs[0]:.2e}  (should be ~0)')
print(f'lambda_1 = {eigs[1]:.6f}')
print(f'Analytic lambda_1 = {eigs_analytic[1]:.6f}')

# Diameter bound
D = nx.diameter(G_path)
vol = sum(dict(G_path.degree()).values())
bound = 1.0 / (D * vol)
print(f'Diameter = {D}, vol = {vol}, bound = {bound:.6f}')
print(f'lambda_1 >= bound? {eigs[1] >= bound - 1e-12}')

plt.figure(figsize=(8, 3))
plt.stem(range(n), eigs, markerfmt='o', basefmt=' ')
plt.xlabel('k'); plt.ylabel(r'$\lambda_k$')
plt.title('Spectrum of path graph Laplacian (n=20)')
plt.tight_layout(); plt.show()

# SOLUTION 3b: Compare spectra across topologies
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for ax, gtype in zip(axes.flat, ['path', 'cycle', 'grid', 'petersen']):
    Gi = make_graph(gtype, 20)
    d0i, _ = build_coboundary(Gi)
    Li = build_laplacian(d0i)
    ei = np.linalg.eigvalsh(Li.toarray())
    ax.stem(range(len(ei)), ei, markerfmt='o', basefmt=' ')
    ax.set_title(gtype)
    ax.set_xlabel('k'); ax.set_ylabel(r'$\lambda_k$')
plt.suptitle('Laplacian spectra across graph topologies')
plt.tight_layout(); plt.show()

# GIVEN: Visualize the Fiedler eigenvector on the path graph
L_dense = L_path.toarray()
eigvals, eigvecs = np.linalg.eigh(L_dense)
fiedler = eigvecs[:, 1]  # second eigenvector

pos_path = {i: (i, 0) for i in range(n)}
fig, ax = plt.subplots(figsize=(10, 2))
nx.draw(G_path, pos_path, node_color=fiedler, cmap='RdBu',
        node_size=200, with_labels=True, font_size=7, ax=ax)
sm = plt.cm.ScalarMappable(cmap='RdBu',
        norm=plt.Normalize(fiedler.min(), fiedler.max()))
plt.colorbar(sm, ax=ax, label='Fiedler component')
ax.set_title('Fiedler eigenvector on the path graph')
plt.tight_layout(); plt.show()

# GIVEN: Generate synthetic pairwise comparison data
def generate_ranking_data(n_items=10, n_comparisons=30, noise=0.3, seed=42):
    """Create pairwise comparisons from hidden ground truth scores."""
    rng = np.random.RandomState(seed)
    s_true = rng.randn(n_items)  # ground truth scores
    s_true -= s_true[0]  # pin first item to zero
    
    # Random pairs
    G = nx.gnm_random_graph(n_items, n_comparisons, seed=seed)
    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
    edges = sorted(set(edges))
    
    # Y_ij = s_j - s_i + noise + injected cycles
    Y = np.array([s_true[j] - s_true[i] + noise * rng.randn()
                  for i, j in edges])
    
    # Inject intransitive cycles: corrupt a few comparisons
    n_corrupt = 3
    corrupt_idx = rng.choice(len(edges), n_corrupt, replace=False)
    Y[corrupt_idx] += rng.choice([-3, 3], n_corrupt)  # large flip
    
    return G, edges, Y, s_true

G_rank, rank_edges, Y_obs, s_true = generate_ranking_data()
print(f'Items: {G_rank.number_of_nodes()}, Comparisons: {len(rank_edges)}')
print(f'Ground truth scores: {np.round(s_true, 2)}')

# SOLUTION 6
delta0_rank, _ = build_coboundary(G_rank)

# Normal equations: L s = delta0^T Y
L_rank = build_laplacian(delta0_rank)
rhs = delta0_rank.T @ Y_obs

# Pin s_0 = 0: drop first row/col
L_reduced = L_rank.toarray()[1:, 1:]
rhs_reduced = rhs[1:]
s_reduced = np.linalg.solve(L_reduced, rhs_reduced)
s_recovered = np.concatenate([[0.0], s_reduced])

print('Recovered ranking:', np.round(s_recovered, 2))
print('Ground truth:     ', np.round(s_true, 2))
corr = np.corrcoef(s_recovered, s_true)[0, 1]
print(f'Correlation: {corr:.4f}')

# GIVEN: Compare recovered ranking with ground truth
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Scatter plot
axes[0].scatter(s_true, s_recovered)
mn, mx = min(s_true.min(), s_recovered.min()), max(s_true.max(), s_recovered.max())
axes[0].plot([mn, mx], [mn, mx], 'k--', alpha=0.5)
axes[0].set_xlabel('Ground truth'); axes[0].set_ylabel('Recovered')
axes[0].set_title(f'Ranking recovery (corr = {corr:.3f})')

# Bar chart
order = np.argsort(s_recovered)[::-1]
axes[1].bar(range(len(order)), s_recovered[order], label='Recovered', alpha=0.7)
axes[1].bar(range(len(order)), s_true[order], label='Truth', alpha=0.5)
axes[1].set_xlabel('Rank'); axes[1].set_ylabel('Score')
axes[1].legend(); axes[1].set_title('Scores by rank')

plt.tight_layout(); plt.show()

# GIVEN: Find all triangles in a graph
def find_triangles(G):
    """Return list of sorted triangles (i, j, k) with i < j < k."""
    triangles = []
    for u in G.nodes():
        for v in G.neighbors(u):
            if v <= u:
                continue
            for w in G.neighbors(v):
                if w <= v and G.has_edge(u, w):
                    continue
                if w > v and G.has_edge(u, w):
                    triangles.append(tuple(sorted([u, v, w])))
    return sorted(set(triangles))

triangles = find_triangles(G_rank)
print(f'Found {len(triangles)} triangles in the ranking graph')
for t in triangles[:5]:
    print(f'  {t}')

# SOLUTION 7a: Build the curl operator delta1
def build_curl_operator(G, edges):
    """Build delta1 from triangles. delta1 in R^{|T| x |E|}."""
    edge_idx = {e: k for k, e in enumerate(edges)}
    tris = find_triangles(G)
    nT, nE = len(tris), len(edges)
    if nT == 0:
        return sp.csr_matrix((0, nE)), tris
    rows, cols, vals = [], [], []
    for t_idx, (i, j, k) in enumerate(tris):
        # Boundary of triangle (i,j,k): edges ij, jk, ik
        # with orientation: +ij +jk -ik
        e_ij = (min(i, j), max(i, j))
        e_jk = (min(j, k), max(j, k))
        e_ik = (min(i, k), max(i, k))
        rows.extend([t_idx, t_idx, t_idx])
        cols.extend([edge_idx[e_ij], edge_idx[e_jk], edge_idx[e_ik]])
        vals.extend([1.0, 1.0, -1.0])
    delta1 = sp.csr_matrix((vals, (rows, cols)), shape=(nT, nE))
    return delta1, tris

delta1, tris = build_curl_operator(G_rank, rank_edges)
print(f'delta1 shape: {delta1.shape}')

# Verify: delta1 @ delta0 = 0
check = delta1 @ delta0_rank
print(f'||delta1 @ delta0|| = {np.abs(check.toarray()).max():.2e} (should be ~0)')

# SOLUTION 7b: Hodge decomposition
# Gradient component
Y_grad = delta0_rank @ s_recovered

# Curl component: solve delta1 delta1^T A = delta1 (Y - Y_grad)
residual = Y_obs - Y_grad
if delta1.shape[0] > 0:
    # Solve for curl amplitudes A via least squares
    A, _, _, _ = np.linalg.lstsq(delta1.toarray() @ delta1.T.toarray(),
                                  delta1.toarray() @ residual, rcond=None)
    Y_curl = delta1.T @ A
else:
    Y_curl = np.zeros_like(Y_obs)

# Harmonic component
Y_harm = Y_obs - Y_grad - Y_curl

print(f'||Y||^2        = {np.linalg.norm(Y_obs)**2:.4f}')
print(f'||Y_grad||^2   = {np.linalg.norm(Y_grad)**2:.4f}')
print(f'||Y_curl||^2   = {np.linalg.norm(Y_curl)**2:.4f}')
print(f'||Y_harm||^2   = {np.linalg.norm(Y_harm)**2:.4f}')
print(f'Sum of parts   = {np.linalg.norm(Y_grad)**2 + np.linalg.norm(Y_curl)**2 + np.linalg.norm(Y_harm)**2:.4f}')
print(f'\nPythagorean identity holds: '
      f'{np.allclose(np.linalg.norm(Y_obs)**2, np.linalg.norm(Y_grad)**2 + np.linalg.norm(Y_curl)**2 + np.linalg.norm(Y_harm)**2)}')

# GIVEN: Visualize Hodge decomposition components
pos_rank = nx.spring_layout(G_rank, seed=42)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, Y_comp, title in zip(axes,
    [Y_grad, Y_curl, Y_harm],
    ['Gradient (consistent)', 'Curl (intransitive)', 'Harmonic']):
    nx.draw_networkx_nodes(G_rank, pos_rank, node_size=200,
                           node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G_rank, pos_rank, font_size=8, ax=ax)
    weights = np.abs(Y_comp)
    max_w = max(weights.max(), 1e-10)
    colors = plt.cm.RdBu(Y_comp / (max(np.abs(Y_comp).max(), 1e-10)) * 0.5 + 0.5)
    edge_coll = nx.draw_networkx_edges(
        G_rank, pos_rank, edgelist=rank_edges,
        width=[2 + 3 * w / max_w for w in weights],
        edge_color=[colors[k] for k in range(len(rank_edges))],
        ax=ax)
    ax.set_title(f'{title}\n||.||={np.linalg.norm(Y_comp):.2f}')

plt.suptitle('Hodge Decomposition of Pairwise Preferences')
plt.tight_layout(); plt.show()
