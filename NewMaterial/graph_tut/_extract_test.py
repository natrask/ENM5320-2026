"""Test Parts 1 and 3 of the Graph Hackathon Solution (no torch_geometric needed)."""
import json

with open("Graph_Hackathon_SOLUTION.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells total")

skip_keywords = ["Planetoid", "GATConv", "model.", "data.x", "data.edge_index",
                 "model(", "dataset[", "conv1", "conv2", "model.eval", "alpha1",
                 "torch.optim", "F.nll_loss", "F.dropout", "nn.Module"]

combined = []
for i, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    should_skip = any(kw in src for kw in skip_keywords)
    if should_skip:
        print(f"  Skipping cell {i} (GAT/torch): {src[:50].strip()}...")
    else:
        combined.append(src)

# Build test script
header = (
    "import numpy as np\n"
    "import scipy.sparse as sp\n"
    "import networkx as nx\n"
    "import matplotlib\n"
    'matplotlib.use("Agg")\n'
    "import matplotlib.pyplot as plt\n"
    "np.random.seed(42)\n"
    'print("Imports OK")\n'
)

script = header
for c in combined[1:]:  # skip original import cell
    script += "\n" + c + "\n"

with open("_test_parts13.py", "w", encoding="utf-8") as f:
    f.write(script)
print(f"\nTest script: {len(combined)-1} code blocks written to _test_parts13.py")
