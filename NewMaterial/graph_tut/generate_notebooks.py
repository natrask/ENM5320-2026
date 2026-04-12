"""Generate Graph_Hackathon_SOLUTION.ipynb and Graph_Hackathon.ipynb"""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

# ===================================================================
# Build cells
# ===================================================================
cells_solution = []
cells_student = []

def add(cell, student_cell=None):
    """Add a cell to both notebooks. If student_cell given, use it for student version."""
    cells_solution.append(cell)
    cells_student.append(student_cell if student_cell is not None else cell)

# ------------------------------------------------------------------
# Phase 0: Setup
# ------------------------------------------------------------------
add(md([
    "# Graph Calculus Hackathon\n",
    "\n",
    "In this hackathon you will implement core ideas from **discrete exterior calculus on graphs**,\n",
    "connecting spectral graph theory, graph neural networks, and combinatorial Hodge theory.\n",
    "\n",
    "## Outline\n",
    "| Part | Topic | Key idea |\n",
    "|------|-------|----------|\n",
    "| 1 | Graph Laplacian from Exterior Calculus | $\\delta_0^\\top W \\delta_0$ and its spectrum |\n",
    "| 2 | Graph Attention Networks | Learned codifferential on Cora |\n",
    "| 3 | Ranking via Hodge Decomposition | Preferences as flows, acyclic ranking |\n",
    "\n",
    "**Prerequisites**: Lectures 16\u201317 (graph calculus, Hodge decomposition, GAT).\n",
    "\n",
    "**Libraries**: numpy, scipy, networkx, matplotlib, torch, torch\\_geometric."
]))

add(md(["## 0.  Setup"]))

add(code([
    "# Run this cell on Colab or if packages are not installed\n",
    "# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n",
    "# !pip install torch-geometric\n",
    "# !pip install networkx matplotlib"
]))

add(code([
    "import numpy as np\n",
    "import scipy.sparse as sp\n",
    "import networkx as nx\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch_geometric.nn import GATConv\n",
    "from torch_geometric.datasets import Planetoid\n",
    "\n",
    "np.random.seed(42)\n",
    "torch.manual_seed(42)\n",
    "print('Imports OK')"
]))

# ------------------------------------------------------------------
# Phase 1: Graph Laplacian
# ------------------------------------------------------------------
add(md([
    "---\n",
    "## 1.  Graph Calculus Operators\n",
    "\n",
    "Recall from Lecture 16 the **coboundary operator** $\\delta_0 \\in \\mathbb{R}^{|E|\\times N}$,\n",
    "the discrete analogue of the gradient.  For each directed edge $e = (i \\to j)$:\n",
    "\n",
    "$$\n",
    "(\\delta_0)_{e,i} = -1, \\qquad (\\delta_0)_{e,j} = +1.\n",
    "$$\n",
    "\n",
    "The **graph Laplacian** is $L = \\delta_0^\\top W \\delta_0$ where $W = \\mathrm{diag}(w_e)$ are edge weights.\n",
    "For uniform weights $W = I$ this reduces to the standard combinatorial Laplacian.\n",
    "\n",
    "Key spectral facts:\n",
    "- $L$ is symmetric positive semi-definite.\n",
    "- The smallest eigenvalue is $\\lambda_0 = 0$ with eigenvector $\\mathbf{1}$.\n",
    "- The **Fiedler eigenvalue** $\\lambda_1 > 0$ iff the graph is connected.\n",
    "- Diameter bound: $\\lambda_1 \\ge \\frac{1}{D \\cdot \\mathrm{vol}(G)}$."
]))

# GIVEN: make_graph helper
add(code([
    "# GIVEN: helper to create standard graphs\n",
    "def make_graph(graph_type, n=20):\n",
    "    \"\"\"Create a named NetworkX graph.\"\"\"\n",
    "    if graph_type == 'path':\n",
    "        return nx.path_graph(n)\n",
    "    elif graph_type == 'cycle':\n",
    "        return nx.cycle_graph(n)\n",
    "    elif graph_type == 'grid':\n",
    "        side = int(np.sqrt(n))\n",
    "        return nx.grid_2d_graph(side, side)\n",
    "    elif graph_type == 'petersen':\n",
    "        return nx.petersen_graph()\n",
    "    else:\n",
    "        raise ValueError(f'Unknown graph type: {graph_type}')\n",
    "\n",
    "G = make_graph('path', 20)\n",
    "print(f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges')"
]))

# GIVEN: visualize graph
add(code([
    "# GIVEN: visualize a small graph\n",
    "pos = nx.spring_layout(G, seed=0)\n",
    "nx.draw(G, pos, with_labels=True, node_color='lightblue',\n",
    "        node_size=300, font_size=8)\n",
    "plt.title('Path graph (n=20)')\n",
    "plt.show()"
]))

# TODO 1 explanation
add(md([
    "### TODO 1 \u2014 Build the coboundary operator $\\delta_0$\n",
    "\n",
    "Given a NetworkX graph $G$, construct the signed incidence matrix\n",
    "$\\delta_0 \\in \\mathbb{R}^{|E| \\times N}$ as a sparse CSR matrix.\n",
    "\n",
    "For each edge $e_k = (i, j)$ with the canonical orientation $i < j$:\n",
    "- $(\\delta_0)_{k,i} = -1$\n",
    "- $(\\delta_0)_{k,j} = +1$\n",
    "\n",
    "Return both `delta0` and the ordered `edge_list`."
]))

# TODO 1 code
sol_todo1 = code([
    "# SOLUTION 1\n",
    "def build_coboundary(G):\n",
    "    \"\"\"Build sparse coboundary (gradient) operator delta0.\"\"\"\n",
    "    nodes = sorted(G.nodes())\n",
    "    node_idx = {v: i for i, v in enumerate(nodes)}\n",
    "    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]\n",
    "    edges = sorted(set(edges))  # canonical orientation i < j\n",
    "    nE, nN = len(edges), len(nodes)\n",
    "    rows, cols, vals = [], [], []\n",
    "    for k, (i, j) in enumerate(edges):\n",
    "        rows.extend([k, k])\n",
    "        cols.extend([node_idx[i], node_idx[j]])\n",
    "        vals.extend([-1.0, 1.0])\n",
    "    delta0 = sp.csr_matrix((vals, (rows, cols)), shape=(nE, nN))\n",
    "    return delta0, edges\n",
    "\n",
    "delta0, edge_list = build_coboundary(G)\n",
    "print(f'delta0 shape: {delta0.shape}  (|E|={len(edge_list)}, N={G.number_of_nodes()})')"
])
stu_todo1 = code([
    "# TODO 1: Build the coboundary (gradient) operator delta0\n",
    "def build_coboundary(G):\n",
    "    \"\"\"Build sparse coboundary (gradient) operator delta0.\n",
    "    \n",
    "    Returns\n",
    "    -------\n",
    "    delta0 : scipy.sparse.csr_matrix, shape (|E|, N)\n",
    "    edges  : list of (i, j) tuples with i < j\n",
    "    \"\"\"\n",
    "    nodes = sorted(G.nodes())\n",
    "    node_idx = {v: i for i, v in enumerate(nodes)}\n",
    "    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]\n",
    "    edges = sorted(set(edges))\n",
    "    nE, nN = len(edges), len(nodes)\n",
    "    rows, cols, vals = [], [], []\n",
    "\n",
    "    # --- FILL IN: loop over edges and append to rows, cols, vals ---\n",
    "    ...\n",
    "    # ---------------------------------------------------------------\n",
    "\n",
    "    delta0 = sp.csr_matrix((vals, (rows, cols)), shape=(nE, nN))\n",
    "    return delta0, edges\n",
    "\n",
    "delta0, edge_list = build_coboundary(G)\n",
    "print(f'delta0 shape: {delta0.shape}  (|E|={len(edge_list)}, N={G.number_of_nodes()})')"
])
add(sol_todo1, stu_todo1)

# TODO 2 explanation
add(md([
    "### TODO 2 \u2014 Build the graph Laplacian\n",
    "\n",
    "Compute $L = \\delta_0^\\top W \\delta_0$ where $W = \\mathrm{diag}(w)$.\n",
    "If `weights` is `None`, use uniform weights ($W = I$)."
]))

sol_todo2 = code([
    "# SOLUTION 2\n",
    "def build_laplacian(delta0, weights=None):\n",
    "    \"\"\"Graph Laplacian L = delta0^T W delta0.\"\"\"\n",
    "    if weights is None:\n",
    "        L = delta0.T @ delta0\n",
    "    else:\n",
    "        W = sp.diags(weights)\n",
    "        L = delta0.T @ W @ delta0\n",
    "    return L\n",
    "\n",
    "L = build_laplacian(delta0)\n",
    "print(f'Laplacian shape: {L.shape}')\n",
    "print(f'Laplacian is symmetric: {np.allclose(L.toarray(), L.toarray().T)}')"
])
stu_todo2 = code([
    "# TODO 2: Build the graph Laplacian L = delta0^T W delta0\n",
    "def build_laplacian(delta0, weights=None):\n",
    "    \"\"\"Graph Laplacian L = delta0^T W delta0.\n",
    "    \n",
    "    Parameters\n",
    "    ----------\n",
    "    delta0  : sparse matrix (|E|, N)\n",
    "    weights : 1-D array of length |E| or None (uniform)\n",
    "    \"\"\"\n",
    "    # --- FILL IN ---\n",
    "    L = ...\n",
    "    # ----------------\n",
    "    return L\n",
    "\n",
    "L = build_laplacian(delta0)\n",
    "print(f'Laplacian shape: {L.shape}')\n",
    "print(f'Laplacian is symmetric: {np.allclose(L.toarray(), L.toarray().T)}')"
])
add(sol_todo2, stu_todo2)

# TODO 3 explanation
add(md([
    "### TODO 3 \u2014 Eigenvalue analysis\n",
    "\n",
    "Compute the full eigenvalue spectrum of $L$ for a **path graph** with $n = 20$ nodes.\n",
    "\n",
    "- Verify that $\\lambda_0 \\approx 0$.\n",
    "- Compare the Fiedler eigenvalue $\\lambda_1$ with the diameter bound $\\frac{1}{D \\cdot \\mathrm{vol}(G)}$.\n",
    "- Plot the spectrum.\n",
    "\n",
    "Analytic eigenvalues for the path graph: $\\lambda_k = 2\\bigl(1 - \\cos(k\\pi / N)\\bigr)$."
]))

sol_todo3 = code([
    "# SOLUTION 3\n",
    "n = 20\n",
    "G_path = make_graph('path', n)\n",
    "d0, _ = build_coboundary(G_path)\n",
    "L_path = build_laplacian(d0)\n",
    "eigs = np.linalg.eigvalsh(L_path.toarray())\n",
    "\n",
    "# Analytic eigenvalues\n",
    "eigs_analytic = [2 * (1 - np.cos(k * np.pi / n)) for k in range(n)]\n",
    "\n",
    "print(f'lambda_0 = {eigs[0]:.2e}  (should be ~0)')\n",
    "print(f'lambda_1 = {eigs[1]:.6f}')\n",
    "print(f'Analytic lambda_1 = {eigs_analytic[1]:.6f}')\n",
    "\n",
    "# Diameter bound\n",
    "D = nx.diameter(G_path)\n",
    "vol = sum(dict(G_path.degree()).values())\n",
    "bound = 1.0 / (D * vol)\n",
    "print(f'Diameter = {D}, vol = {vol}, bound = {bound:.6f}')\n",
    "print(f'lambda_1 >= bound? {eigs[1] >= bound - 1e-12}')\n",
    "\n",
    "plt.figure(figsize=(8, 3))\n",
    "plt.stem(range(n), eigs, markerfmt='o', basefmt=' ')\n",
    "plt.xlabel('k'); plt.ylabel(r'$\\lambda_k$')\n",
    "plt.title('Spectrum of path graph Laplacian (n=20)')\n",
    "plt.tight_layout(); plt.show()"
])
stu_todo3 = code([
    "# TODO 3: Eigenvalue analysis for the path graph\n",
    "n = 20\n",
    "G_path = make_graph('path', n)\n",
    "d0, _ = build_coboundary(G_path)\n",
    "L_path = build_laplacian(d0)\n",
    "\n",
    "# --- FILL IN: compute eigenvalues, print lambda_0 and lambda_1, ---\n",
    "# --- compare with analytic formula and diameter bound, then plot ---\n",
    "eigs = ...\n",
    "\n",
    "# ------------------------------------------------------------------"
])
add(sol_todo3, stu_todo3)

# TODO 3b
sol_todo3b = code([
    "# SOLUTION 3b: Compare spectra across topologies\n",
    "fig, axes = plt.subplots(2, 2, figsize=(10, 6))\n",
    "for ax, gtype in zip(axes.flat, ['path', 'cycle', 'grid', 'petersen']):\n",
    "    Gi = make_graph(gtype, 20)\n",
    "    d0i, _ = build_coboundary(Gi)\n",
    "    Li = build_laplacian(d0i)\n",
    "    ei = np.linalg.eigvalsh(Li.toarray())\n",
    "    ax.stem(range(len(ei)), ei, markerfmt='o', basefmt=' ')\n",
    "    ax.set_title(gtype)\n",
    "    ax.set_xlabel('k'); ax.set_ylabel(r'$\\lambda_k$')\n",
    "plt.suptitle('Laplacian spectra across graph topologies')\n",
    "plt.tight_layout(); plt.show()"
])
stu_todo3b = code([
    "# TODO 3b: Repeat eigenvalue analysis for cycle, grid, Petersen graphs\n",
    "# Create a 2x2 subplot comparing the spectra.\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(10, 6))\n",
    "\n",
    "# --- FILL IN: loop over graph types, compute and plot spectra ---\n",
    "...\n",
    "# ---------------------------------------------------------------\n",
    "\n",
    "plt.suptitle('Laplacian spectra across graph topologies')\n",
    "plt.tight_layout(); plt.show()"
])
add(sol_todo3b, stu_todo3b)

# Discussion
add(md([
    "**Discussion**: How does the spectrum change with connectivity? Which graph has\n",
    "the largest spectral gap $\\lambda_1$? How does this relate to the diameter bound?"
]))

# GIVEN: Fiedler eigenvector visualization
add(code([
    "# GIVEN: Visualize the Fiedler eigenvector on the path graph\n",
    "L_dense = L_path.toarray()\n",
    "eigvals, eigvecs = np.linalg.eigh(L_dense)\n",
    "fiedler = eigvecs[:, 1]  # second eigenvector\n",
    "\n",
    "pos_path = {i: (i, 0) for i in range(n)}\n",
    "fig, ax = plt.subplots(figsize=(10, 2))\n",
    "nx.draw(G_path, pos_path, node_color=fiedler, cmap='RdBu',\n",
    "        node_size=200, with_labels=True, font_size=7, ax=ax)\n",
    "sm = plt.cm.ScalarMappable(cmap='RdBu',\n",
    "        norm=plt.Normalize(fiedler.min(), fiedler.max()))\n",
    "plt.colorbar(sm, ax=ax, label='Fiedler component')\n",
    "ax.set_title('Fiedler eigenvector on the path graph')\n",
    "plt.tight_layout(); plt.show()"
]))

# ------------------------------------------------------------------
# Phase 2: GAT
# ------------------------------------------------------------------
add(md([
    "---\n",
    "## 2.  Graph Attention Networks\n",
    "\n",
    "Recall from Lecture 16 that a **graph attention network** performs message passing\n",
    "via a learned codifferential:\n",
    "\n",
    "$$\n",
    "x_i^{(\\ell+1)} = x_i^{(\\ell)} + \\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij}\\, W x_j^{(\\ell)}\n",
    "$$\n",
    "\n",
    "where $\\alpha_{ij}$ are attention coefficients.  We use the **Cora** citation dataset:\n",
    "2708 papers (nodes), 5429 citations (edges), 7 classes, 1433-dim bag-of-words features.\n",
    "\n",
    "Task: **semi-supervised node classification** (only 140 labeled training nodes)."
]))

# GIVEN: load Cora
add(code([
    "# GIVEN: Load the Cora citation dataset\n",
    "dataset = Planetoid(root='/tmp/Cora', name='Cora')\n",
    "data = dataset[0]\n",
    "print(f'Nodes: {data.num_nodes}')\n",
    "print(f'Edges: {data.num_edges}')\n",
    "print(f'Features: {data.num_node_features}')\n",
    "print(f'Classes: {dataset.num_classes}')\n",
    "print(f'Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}')"
]))

# GIVEN: visualize Cora subgraph
add(code([
    "# GIVEN: Visualize a subgraph of Cora\n",
    "subset = list(range(50))\n",
    "sub_edge_index = data.edge_index[:, \n",
    "    (data.edge_index[0] < 50) & (data.edge_index[1] < 50)]\n",
    "Gsub = nx.Graph()\n",
    "Gsub.add_nodes_from(subset)\n",
    "Gsub.add_edges_from(sub_edge_index.T.tolist())\n",
    "pos_sub = nx.spring_layout(Gsub, seed=42)\n",
    "colors = data.y[subset].numpy()\n",
    "nx.draw(Gsub, pos_sub, node_color=colors, cmap='tab10',\n",
    "        node_size=60, width=0.3)\n",
    "plt.title('Cora subgraph (first 50 nodes, colored by class)')\n",
    "plt.show()"
]))

# TODO 4 explanation
add(md([
    "### TODO 4 \u2014 Build a Graph Attention Classifier\n",
    "\n",
    "Implement a 2-layer GAT using `torch_geometric.nn.GATConv`:\n",
    "\n",
    "| Layer | Input dim | Output dim | Heads |\n",
    "|-------|-----------|-----------|-------|\n",
    "| GATConv 1 | 1433 | 8 | 8  (concat \u2192 64) |\n",
    "| GATConv 2 | 64 | 7 | 1  (mean) |\n",
    "\n",
    "Use ELU activation and dropout (p=0.6) between layers.\n",
    "Output: `log_softmax` over classes."
]))

sol_todo4 = code([
    "# SOLUTION 4\n",
    "class GraphAttentionClassifier(nn.Module):\n",
    "    def __init__(self, in_channels, hidden, heads, out_channels, dropout=0.6):\n",
    "        super().__init__()\n",
    "        self.dropout = dropout\n",
    "        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)\n",
    "        self.conv2 = GATConv(hidden * heads, out_channels, heads=1,\n",
    "                             concat=False, dropout=dropout)\n",
    "\n",
    "    def forward(self, x, edge_index):\n",
    "        x = F.dropout(x, p=self.dropout, training=self.training)\n",
    "        x = self.conv1(x, edge_index)\n",
    "        x = F.elu(x)\n",
    "        x = F.dropout(x, p=self.dropout, training=self.training)\n",
    "        x = self.conv2(x, edge_index)\n",
    "        return F.log_softmax(x, dim=1)\n",
    "\n",
    "model = GraphAttentionClassifier(\n",
    "    in_channels=dataset.num_node_features,\n",
    "    hidden=8, heads=8,\n",
    "    out_channels=dataset.num_classes\n",
    ")\n",
    "print(model)"
])
stu_todo4 = code([
    "# TODO 4: Build a 2-layer GAT classifier\n",
    "class GraphAttentionClassifier(nn.Module):\n",
    "    def __init__(self, in_channels, hidden, heads, out_channels, dropout=0.6):\n",
    "        super().__init__()\n",
    "        self.dropout = dropout\n",
    "        # --- FILL IN: define self.conv1 and self.conv2 ---\n",
    "        ...\n",
    "        # --------------------------------------------------\n",
    "\n",
    "    def forward(self, x, edge_index):\n",
    "        # --- FILL IN: dropout -> conv1 -> elu -> dropout -> conv2 -> log_softmax ---\n",
    "        ...\n",
    "        # ---------------------------------------------------------------------------\n",
    "\n",
    "model = GraphAttentionClassifier(\n",
    "    in_channels=dataset.num_node_features,\n",
    "    hidden=8, heads=8,\n",
    "    out_channels=dataset.num_classes\n",
    ")\n",
    "print(model)"
])
add(sol_todo4, stu_todo4)

# TODO 5 explanation
add(md([
    "### TODO 5 \u2014 Train and evaluate the GAT\n",
    "\n",
    "Write a training loop:\n",
    "- Optimizer: `Adam(lr=0.005, weight_decay=5e-4)`\n",
    "- Loss: `F.nll_loss` on `data.train_mask`\n",
    "- 200 epochs, print loss every 20 epochs\n",
    "\n",
    "Then evaluate accuracy on train / val / test masks."
]))

sol_todo5 = code([
    "# SOLUTION 5\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)\n",
    "\n",
    "def train():\n",
    "    model.train()\n",
    "    optimizer.zero_grad()\n",
    "    out = model(data.x, data.edge_index)\n",
    "    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "    return loss.item()\n",
    "\n",
    "def evaluate(mask):\n",
    "    model.eval()\n",
    "    with torch.no_grad():\n",
    "        out = model(data.x, data.edge_index)\n",
    "        pred = out.argmax(dim=1)\n",
    "        return (pred[mask] == data.y[mask]).float().mean().item()\n",
    "\n",
    "for epoch in range(1, 201):\n",
    "    loss = train()\n",
    "    if epoch % 20 == 0:\n",
    "        val_acc = evaluate(data.val_mask)\n",
    "        print(f'Epoch {epoch:3d}  Loss {loss:.4f}  Val acc {val_acc:.3f}')\n",
    "\n",
    "print(f'\\nFinal  Train acc: {evaluate(data.train_mask):.3f}')\n",
    "print(f'       Val acc:   {evaluate(data.val_mask):.3f}')\n",
    "print(f'       Test acc:  {evaluate(data.test_mask):.3f}')"
])
stu_todo5 = code([
    "# TODO 5: Train and evaluate the GAT\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)\n",
    "\n",
    "def train():\n",
    "    model.train()\n",
    "    # --- FILL IN: zero grad, forward, loss on train_mask, backward, step ---\n",
    "    ...\n",
    "    # -----------------------------------------------------------------------\n",
    "\n",
    "# GIVEN: evaluation helper\n",
    "def evaluate(mask):\n",
    "    model.eval()\n",
    "    with torch.no_grad():\n",
    "        out = model(data.x, data.edge_index)\n",
    "        pred = out.argmax(dim=1)\n",
    "        return (pred[mask] == data.y[mask]).float().mean().item()\n",
    "\n",
    "# --- FILL IN: training loop (200 epochs, print every 20) ---\n",
    "...\n",
    "# ------------------------------------------------------------\n",
    "\n",
    "print(f'\\nFinal  Train acc: {evaluate(data.train_mask):.3f}')\n",
    "print(f'       Val acc:   {evaluate(data.val_mask):.3f}')\n",
    "print(f'       Test acc:  {evaluate(data.test_mask):.3f}')"
])
add(sol_todo5, stu_todo5)

# Discussion
add(md([
    "**Discussion**: How does GAT accuracy compare with a simple MLP (no graph\n",
    "structure)? What role does message passing play in semi-supervised classification?"
]))

# GIVEN: attention weight visualization
add(code([
    "# GIVEN: Visualize attention weights on a subgraph\n",
    "model.eval()\n",
    "# Run forward to capture attention weights\n",
    "with torch.no_grad():\n",
    "    x = F.dropout(data.x, p=0.6, training=False)\n",
    "    x, (ei1, alpha1) = model.conv1(x, data.edge_index, return_attention_weights=True)\n",
    "    x = F.elu(x)\n",
    "\n",
    "# Extract for subgraph\n",
    "mask_att = (ei1[0] < 50) & (ei1[1] < 50)\n",
    "edges_att = ei1[:, mask_att].numpy()\n",
    "weights_att = alpha1[mask_att].mean(dim=1).numpy()  # average over heads\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "nx.draw_networkx_nodes(Gsub, pos_sub, node_size=40, node_color='lightblue', ax=ax)\n",
    "edge_coll = nx.draw_networkx_edges(\n",
    "    Gsub, pos_sub, edgelist=list(zip(edges_att[0], edges_att[1])),\n",
    "    width=[w * 5 for w in weights_att], alpha=0.6, ax=ax)\n",
    "ax.set_title('GAT attention weights (layer 1, subgraph)')\n",
    "plt.tight_layout(); plt.show()"
]))

# ------------------------------------------------------------------
# Phase 3: Recommender / Hodge decomposition
# ------------------------------------------------------------------
add(md([
    "---\n",
    "## 3.  Ranking via Hodge Decomposition\n",
    "\n",
    "Given **pairwise comparison data** $Y_{ij}$ (e.g. movie ratings, sports outcomes),\n",
    "we seek a global ranking score $s$ such that $s_j - s_i \\approx Y_{ij}$.\n",
    "\n",
    "From Lecture 17, the edge flow $Y$ decomposes via the **Hodge decomposition**:\n",
    "\n",
    "$$\n",
    "Y = \\underbrace{\\delta_0 s}_{\\text{gradient (consistent)}} +\n",
    "    \\underbrace{\\delta_1^\\top A}_{\\text{curl (intransitive)}} +\n",
    "    \\underbrace{h}_{\\text{harmonic}}\n",
    "$$\n",
    "\n",
    "- The **gradient component** $\\delta_0 s$ recovers the acyclic ranking.\n",
    "- The **curl component** $\\delta_1^\\top A$ captures intransitive cycles\n",
    "  (rock\u2013paper\u2013scissors patterns).\n",
    "- The **harmonic component** $h$ lies in both $\\ker(\\delta_0^\\top)$ and $\\ker(\\delta_1)$.\n",
    "\n",
    "Key identity: $\\delta_1 \\delta_0 = 0$ (curl of gradient is zero)."
]))

# GIVEN: generate synthetic data
add(code([
    "# GIVEN: Generate synthetic pairwise comparison data\n",
    "def generate_ranking_data(n_items=10, n_comparisons=30, noise=0.3, seed=42):\n",
    "    \"\"\"Create pairwise comparisons from hidden ground truth scores.\"\"\"\n",
    "    rng = np.random.RandomState(seed)\n",
    "    s_true = rng.randn(n_items)  # ground truth scores\n",
    "    s_true -= s_true[0]  # pin first item to zero\n",
    "    \n",
    "    # Random pairs\n",
    "    G = nx.gnm_random_graph(n_items, n_comparisons, seed=seed)\n",
    "    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]\n",
    "    edges = sorted(set(edges))\n",
    "    \n",
    "    # Y_ij = s_j - s_i + noise + injected cycles\n",
    "    Y = np.array([s_true[j] - s_true[i] + noise * rng.randn()\n",
    "                  for i, j in edges])\n",
    "    \n",
    "    # Inject intransitive cycles: corrupt a few comparisons\n",
    "    n_corrupt = 3\n",
    "    corrupt_idx = rng.choice(len(edges), n_corrupt, replace=False)\n",
    "    Y[corrupt_idx] += rng.choice([-3, 3], n_corrupt)  # large flip\n",
    "    \n",
    "    return G, edges, Y, s_true\n",
    "\n",
    "G_rank, rank_edges, Y_obs, s_true = generate_ranking_data()\n",
    "print(f'Items: {G_rank.number_of_nodes()}, Comparisons: {len(rank_edges)}')\n",
    "print(f'Ground truth scores: {np.round(s_true, 2)}')"
]))

# TODO 6 explanation
add(md([
    "### TODO 6 \u2014 Solve for the ranking\n",
    "\n",
    "The least-squares ranking minimizes $\\sum_{(i,j)} w_{ij}(Y_{ij} - (\\delta_0 s)_{ij})^2$.\n",
    "\n",
    "The normal equations are:\n",
    "\n",
    "$$\n",
    "\\delta_0^\\top W \\delta_0 \\, s = \\delta_0^\\top W Y\n",
    "$$\n",
    "\n",
    "This is the **graph Laplacian system** from Part 1! Since $L$ has a null space\n",
    "(constant vectors), pin $s_0 = 0$ by dropping the first row/column."
]))

sol_todo6 = code([
    "# SOLUTION 6\n",
    "delta0_rank, _ = build_coboundary(G_rank)\n",
    "\n",
    "# Normal equations: L s = delta0^T Y\n",
    "L_rank = build_laplacian(delta0_rank)\n",
    "rhs = delta0_rank.T @ Y_obs\n",
    "\n",
    "# Pin s_0 = 0: drop first row/col\n",
    "L_reduced = L_rank.toarray()[1:, 1:]\n",
    "rhs_reduced = rhs[1:]\n",
    "s_reduced = np.linalg.solve(L_reduced, rhs_reduced)\n",
    "s_recovered = np.concatenate([[0.0], s_reduced])\n",
    "\n",
    "print('Recovered ranking:', np.round(s_recovered, 2))\n",
    "print('Ground truth:     ', np.round(s_true, 2))\n",
    "corr = np.corrcoef(s_recovered, s_true)[0, 1]\n",
    "print(f'Correlation: {corr:.4f}')"
])
stu_todo6 = code([
    "# TODO 6: Solve for the ranking using the normal equations\n",
    "delta0_rank, _ = build_coboundary(G_rank)\n",
    "\n",
    "# --- FILL IN: Build L, form rhs = delta0^T Y, pin s_0=0, solve ---\n",
    "s_recovered = ...\n",
    "# -----------------------------------------------------------------\n",
    "\n",
    "print('Recovered ranking:', np.round(s_recovered, 2))\n",
    "print('Ground truth:     ', np.round(s_true, 2))\n",
    "corr = np.corrcoef(s_recovered, s_true)[0, 1]\n",
    "print(f'Correlation: {corr:.4f}')"
])
add(sol_todo6, stu_todo6)

# GIVEN: ranking visualization
add(code([
    "# GIVEN: Compare recovered ranking with ground truth\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
    "\n",
    "# Scatter plot\n",
    "axes[0].scatter(s_true, s_recovered)\n",
    "mn, mx = min(s_true.min(), s_recovered.min()), max(s_true.max(), s_recovered.max())\n",
    "axes[0].plot([mn, mx], [mn, mx], 'k--', alpha=0.5)\n",
    "axes[0].set_xlabel('Ground truth'); axes[0].set_ylabel('Recovered')\n",
    "axes[0].set_title(f'Ranking recovery (corr = {corr:.3f})')\n",
    "\n",
    "# Bar chart\n",
    "order = np.argsort(s_recovered)[::-1]\n",
    "axes[1].bar(range(len(order)), s_recovered[order], label='Recovered', alpha=0.7)\n",
    "axes[1].bar(range(len(order)), s_true[order], label='Truth', alpha=0.5)\n",
    "axes[1].set_xlabel('Rank'); axes[1].set_ylabel('Score')\n",
    "axes[1].legend(); axes[1].set_title('Scores by rank')\n",
    "\n",
    "plt.tight_layout(); plt.show()"
]))

# TODO 7 explanation
add(md([
    "### TODO 7 \u2014 Hodge Decomposition\n",
    "\n",
    "Now decompose the observed preferences $Y$ into:\n",
    "1. **Gradient** component: $Y_{\\mathrm{grad}} = \\delta_0 s$ (consistent ranking)\n",
    "2. **Curl** component: $Y_{\\mathrm{curl}} = \\delta_1^\\top A$ (intransitive cycles)\n",
    "3. **Harmonic** component: $h = Y - Y_{\\mathrm{grad}} - Y_{\\mathrm{curl}}$\n",
    "\n",
    "First, build the **curl operator** $\\delta_1$ from triangles in the graph.\n",
    "For triangle $(i, j, k)$ with edges $e_{ij}, e_{jk}, e_{ik}$:\n",
    "\n",
    "$$\n",
    "(\\delta_1)_{t, e_{ij}} = +1,\\quad (\\delta_1)_{t, e_{jk}} = +1,\\quad (\\delta_1)_{t, e_{ik}} = -1\n",
    "$$\n",
    "\n",
    "Verify: $\\delta_1 \\delta_0 = 0$ (curl of gradient is zero)."
]))

# GIVEN: find_triangles helper
add(code([
    "# GIVEN: Find all triangles in a graph\n",
    "def find_triangles(G):\n",
    "    \"\"\"Return list of sorted triangles (i, j, k) with i < j < k.\"\"\"\n",
    "    triangles = []\n",
    "    for u in G.nodes():\n",
    "        for v in G.neighbors(u):\n",
    "            if v <= u:\n",
    "                continue\n",
    "            for w in G.neighbors(v):\n",
    "                if w <= v and G.has_edge(u, w):\n",
    "                    continue\n",
    "                if w > v and G.has_edge(u, w):\n",
    "                    triangles.append(tuple(sorted([u, v, w])))\n",
    "    return sorted(set(triangles))\n",
    "\n",
    "triangles = find_triangles(G_rank)\n",
    "print(f'Found {len(triangles)} triangles in the ranking graph')\n",
    "for t in triangles[:5]:\n",
    "    print(f'  {t}')"
]))

# TODO 7a: build curl operator
sol_todo7a = code([
    "# SOLUTION 7a: Build the curl operator delta1\n",
    "def build_curl_operator(G, edges):\n",
    "    \"\"\"Build delta1 from triangles. delta1 in R^{|T| x |E|}.\"\"\"\n",
    "    edge_idx = {e: k for k, e in enumerate(edges)}\n",
    "    tris = find_triangles(G)\n",
    "    nT, nE = len(tris), len(edges)\n",
    "    if nT == 0:\n",
    "        return sp.csr_matrix((0, nE)), tris\n",
    "    rows, cols, vals = [], [], []\n",
    "    for t_idx, (i, j, k) in enumerate(tris):\n",
    "        # Boundary of triangle (i,j,k): edges ij, jk, ik\n",
    "        # with orientation: +ij +jk -ik\n",
    "        e_ij = (min(i, j), max(i, j))\n",
    "        e_jk = (min(j, k), max(j, k))\n",
    "        e_ik = (min(i, k), max(i, k))\n",
    "        rows.extend([t_idx, t_idx, t_idx])\n",
    "        cols.extend([edge_idx[e_ij], edge_idx[e_jk], edge_idx[e_ik]])\n",
    "        vals.extend([1.0, 1.0, -1.0])\n",
    "    delta1 = sp.csr_matrix((vals, (rows, cols)), shape=(nT, nE))\n",
    "    return delta1, tris\n",
    "\n",
    "delta1, tris = build_curl_operator(G_rank, rank_edges)\n",
    "print(f'delta1 shape: {delta1.shape}')\n",
    "\n",
    "# Verify: delta1 @ delta0 = 0\n",
    "check = delta1 @ delta0_rank\n",
    "print(f'||delta1 @ delta0|| = {np.abs(check.toarray()).max():.2e} (should be ~0)')"
])
stu_todo7a = code([
    "# TODO 7a: Build the curl operator delta1\n",
    "def build_curl_operator(G, edges):\n",
    "    \"\"\"Build delta1 from triangles. delta1 in R^{|T| x |E|}.\n",
    "    \n",
    "    For triangle (i, j, k) with i < j < k:\n",
    "      delta1[t, e_ij] = +1\n",
    "      delta1[t, e_jk] = +1  \n",
    "      delta1[t, e_ik] = -1\n",
    "    \"\"\"\n",
    "    edge_idx = {e: k for k, e in enumerate(edges)}\n",
    "    tris = find_triangles(G)\n",
    "    nT, nE = len(tris), len(edges)\n",
    "    if nT == 0:\n",
    "        return sp.csr_matrix((0, nE)), tris\n",
    "    rows, cols, vals = [], [], []\n",
    "\n",
    "    # --- FILL IN: loop over triangles and build COO entries ---\n",
    "    ...\n",
    "    # -----------------------------------------------------------\n",
    "\n",
    "    delta1 = sp.csr_matrix((vals, (rows, cols)), shape=(nT, nE))\n",
    "    return delta1, tris\n",
    "\n",
    "delta1, tris = build_curl_operator(G_rank, rank_edges)\n",
    "print(f'delta1 shape: {delta1.shape}')\n",
    "\n",
    "# Verify: delta1 @ delta0 = 0\n",
    "check = delta1 @ delta0_rank\n",
    "print(f'||delta1 @ delta0|| = {np.abs(check.toarray()).max():.2e} (should be ~0)')"
])
add(sol_todo7a, stu_todo7a)

# TODO 7b: Hodge decomposition
sol_todo7b = code([
    "# SOLUTION 7b: Hodge decomposition\n",
    "# Gradient component\n",
    "Y_grad = delta0_rank @ s_recovered\n",
    "\n",
    "# Curl component: solve delta1 delta1^T A = delta1 (Y - Y_grad)\n",
    "residual = Y_obs - Y_grad\n",
    "if delta1.shape[0] > 0:\n",
    "    # Solve for curl amplitudes A via least squares\n",
    "    A, _, _, _ = np.linalg.lstsq(delta1.toarray() @ delta1.T.toarray(),\n",
    "                                  delta1.toarray() @ residual, rcond=None)\n",
    "    Y_curl = delta1.T @ A\n",
    "else:\n",
    "    Y_curl = np.zeros_like(Y_obs)\n",
    "\n",
    "# Harmonic component\n",
    "Y_harm = Y_obs - Y_grad - Y_curl\n",
    "\n",
    "print(f'||Y||^2        = {np.linalg.norm(Y_obs)**2:.4f}')\n",
    "print(f'||Y_grad||^2   = {np.linalg.norm(Y_grad)**2:.4f}')\n",
    "print(f'||Y_curl||^2   = {np.linalg.norm(Y_curl)**2:.4f}')\n",
    "print(f'||Y_harm||^2   = {np.linalg.norm(Y_harm)**2:.4f}')\n",
    "print(f'Sum of parts   = {np.linalg.norm(Y_grad)**2 + np.linalg.norm(Y_curl)**2 + np.linalg.norm(Y_harm)**2:.4f}')\n",
    "print(f'\\nPythagorean identity holds: '\n",
    "      f'{np.allclose(np.linalg.norm(Y_obs)**2, np.linalg.norm(Y_grad)**2 + np.linalg.norm(Y_curl)**2 + np.linalg.norm(Y_harm)**2)}')"
])
stu_todo7b = code([
    "# TODO 7b: Compute the Hodge decomposition\n",
    "# 1. Gradient component: Y_grad = delta0 @ s_recovered\n",
    "# 2. Curl component: solve delta1 @ delta1^T @ A = delta1 @ (Y - Y_grad)\n",
    "#    then Y_curl = delta1^T @ A\n",
    "# 3. Harmonic: Y_harm = Y - Y_grad - Y_curl\n",
    "\n",
    "# --- FILL IN ---\n",
    "Y_grad = ...\n",
    "Y_curl = ...\n",
    "Y_harm = ...\n",
    "# ----------------\n",
    "\n",
    "print(f'||Y||^2        = {np.linalg.norm(Y_obs)**2:.4f}')\n",
    "print(f'||Y_grad||^2   = {np.linalg.norm(Y_grad)**2:.4f}')\n",
    "print(f'||Y_curl||^2   = {np.linalg.norm(Y_curl)**2:.4f}')\n",
    "print(f'||Y_harm||^2   = {np.linalg.norm(Y_harm)**2:.4f}')\n",
    "print(f'Sum of parts   = {np.linalg.norm(Y_grad)**2 + np.linalg.norm(Y_curl)**2 + np.linalg.norm(Y_harm)**2:.4f}')"
])
add(sol_todo7b, stu_todo7b)

# Discussion
add(md([
    "**Discussion**: What fraction of the preference data is explained by a consistent\n",
    "ranking ($\\|Y_{\\mathrm{grad}}\\|^2 / \\|Y\\|^2$)? Where do the cyclic components appear?\n",
    "Do they correspond to the injected intransitive cycles?"
]))

# GIVEN: Hodge component visualization
add(code([
    "# GIVEN: Visualize Hodge decomposition components\n",
    "pos_rank = nx.spring_layout(G_rank, seed=42)\n",
    "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
    "\n",
    "for ax, Y_comp, title in zip(axes,\n",
    "    [Y_grad, Y_curl, Y_harm],\n",
    "    ['Gradient (consistent)', 'Curl (intransitive)', 'Harmonic']):\n",
    "    nx.draw_networkx_nodes(G_rank, pos_rank, node_size=200,\n",
    "                           node_color='lightblue', ax=ax)\n",
    "    nx.draw_networkx_labels(G_rank, pos_rank, font_size=8, ax=ax)\n",
    "    weights = np.abs(Y_comp)\n",
    "    max_w = max(weights.max(), 1e-10)\n",
    "    colors = plt.cm.RdBu(Y_comp / (max(np.abs(Y_comp).max(), 1e-10)) * 0.5 + 0.5)\n",
    "    edge_coll = nx.draw_networkx_edges(\n",
    "        G_rank, pos_rank, edgelist=rank_edges,\n",
    "        width=[2 + 3 * w / max_w for w in weights],\n",
    "        edge_color=[colors[k] for k in range(len(rank_edges))],\n",
    "        ax=ax)\n",
    "    ax.set_title(f'{title}\\n||.||={np.linalg.norm(Y_comp):.2f}')\n",
    "\n",
    "plt.suptitle('Hodge Decomposition of Pairwise Preferences')\n",
    "plt.tight_layout(); plt.show()"
]))

# ------------------------------------------------------------------
# Phase 4: Bonus
# ------------------------------------------------------------------
add(md([
    "---\n",
    "## Bonus Exercises\n",
    "\n",
    "**A.** Apply the ranking/Hodge decomposition to a small real dataset (e.g. a\n",
    "sports tournament or MovieLens subset). How much of the data is captured by\n",
    "the gradient (consistent) component?\n",
    "\n",
    "**B.** Compare the learned GAT representations with the Fiedler eigenvector.\n",
    "Does the GAT implicitly learn spectral features? (Hint: project GAT hidden\n",
    "states onto the leading eigenvectors of $L$.)\n",
    "\n",
    "**C.** Add DEC Hodge star weights ($w_e = 1/h$) to Part 1 for a path graph\n",
    "and verify that the smallest nonzero eigenvalue approaches $\\pi^2/L^2$ as\n",
    "$n \\to \\infty$ (connecting to the coercivity / Poincar\u00e9 inequality from the\n",
    "continuous theory)."
]))

add(md([
    "---\n",
    "## References\n",
    "\n",
    "- Lecture 16: Graph Calculus, Spectral Theory, and Graph Attention Networks\n",
    "- Lecture 17: Hodge Decomposition, Recommender Systems, and Causal Inference\n",
    "- Lim (2020). *Hodge Laplacians on Graphs.* SIAM Review.\n",
    "- Veli\u010dkovi\u0107 et al. (2018). *Graph Attention Networks.* ICLR.\n",
    "- Jiang, Lim, Yao, Ye (2011). *Statistical ranking and combinatorial Hodge theory.* Math. Program."
]))

# ===================================================================
# Write notebooks
# ===================================================================
base = os.path.dirname(os.path.abspath(__file__))

sol_path = os.path.join(base, "Graph_Hackathon_SOLUTION.ipynb")
stu_path = os.path.join(base, "Graph_Hackathon.ipynb")

with open(sol_path, "w", encoding="utf-8") as f:
    json.dump(make_nb(cells_solution), f, indent=1, ensure_ascii=False)
print(f"Solution: {len(cells_solution)} cells -> {sol_path}")

with open(stu_path, "w", encoding="utf-8") as f:
    json.dump(make_nb(cells_student), f, indent=1, ensure_ascii=False)
print(f"Student:  {len(cells_student)} cells -> {stu_path}")

# Quick validation
for p in [sol_path, stu_path]:
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"  {os.path.basename(p)}: {len(nb['cells'])} cells, valid JSON ✓")
