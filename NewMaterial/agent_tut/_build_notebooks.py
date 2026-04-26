"""Generator for both versions of the agentic-AI hackathon notebook.

Produces:
  - Agent_Hackathon.ipynb           (student version with TODO blanks)
  - Agent_Hackathon_SOLUTION.ipynb  (complete instructor version)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STU_PATH = os.path.join(HERE, 'Agent_Hackathon.ipynb')
SOL_PATH = os.path.join(HERE, 'Agent_Hackathon_SOLUTION.ipynb')

COLAB_BADGE_STU = ('[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]'
    '(https://colab.research.google.com/github/natrask/ENM5320-2026/blob/main/NewMaterial/agent_tut/Agent_Hackathon.ipynb)')
COLAB_BADGE_SOL = ('[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]'
    '(https://colab.research.google.com/github/natrask/ENM5320-2026/blob/main/NewMaterial/agent_tut/Agent_Hackathon_SOLUTION.ipynb)')

SCIKIT_FEM_TUT_URL = ('https://colab.research.google.com/github/natrask/ENM5320-2026/blob/main/'
                      'scikitFEM_tut/FEM_Hackathon_Poisson2D.ipynb')


def src(text):
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1][:-1]
    return lines


def emit(cells, kind, *srcs):
    cells.append((kind,) + srcs)


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

def build_cells():
    C = []

    # ================================================================
    # Part 0: Intro and setup
    # ================================================================

    emit(C, "md_both",
         "__BADGE_STU__",
         "__BADGE_SOL__")

    emit(C, "md", f"""# Agentic AI for Scientific Computing

This notebook builds a small LLM-based agent from scratch and uses it on two scientific computing tasks:

1. Adaptive mesh refinement for the 2D Poisson equation on an L-shaped domain, using [scikit-fem](https://scikit-fem.readthedocs.io/).
2. Black-box optimization of a projectile simulator, using `scipy.integrate.odeint`.

**Outline.**
- Part 1: build a ReAct agent loop in about 20 lines of Python (no agent framework).
- Part 2: connect the agent to scikit-fem tools and let it drive mesh refinement.
- Part 3: connect the agent to a black-box ODE simulator and let it optimize a launch angle.
- Part 4: pointers for going further.

**Prerequisites.** Python, basic scipy, and the [scikit-fem hackathon]({SCIKIT_FEM_TUT_URL}) from Mar 23 for the Poisson setup used in Part 2.

**LLM backend.** We use Google's Gemini 2.0 Flash via the free API tier. The ReAct loop also works with OpenAI, Anthropic, or Groq if you have a key for any of those.""")

    emit(C, "md", """## 0. Setup

Get a free API key at <https://aistudio.google.com/apikey> and sign in with a Google account.

In Colab, store the key in the *Secrets* panel (key icon in the left sidebar) under the name `GEMINI_API_KEY`. Running locally, set it as an environment variable:

```
export GEMINI_API_KEY=...
```""")

    emit(C, "code", """# Install dependencies (run once).
import sys, subprocess
if 'google.colab' in sys.modules:
    %pip install google-genai scikit-fem scipy matplotlib sympy -q
else:
    pass""")

    emit(C, "code", """import os, json, time, math, random
import numpy as np
import matplotlib.pyplot as plt

from google import genai
from google.genai import types as gtypes

from skfem import (MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm,
                   Functional, condense, solve)
from skfem.helpers import dot, grad

from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize_scalar

plt.rcParams['figure.dpi'] = 90""")

    emit(C, "code", """# Load the API key. In Colab use the Secrets panel; locally use an environment variable.
try:
    from google.colab import userdata
    API_KEY = userdata.get('GEMINI_API_KEY')
except Exception:
    API_KEY = os.environ.get('GEMINI_API_KEY', '')

assert API_KEY, "Set GEMINI_API_KEY in Colab secrets or your environment."

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.0-flash"
print("Gemini client ready.")""")

    emit(C, "code", """# Smoke test: one chat call, no tools.
resp = client.models.generate_content(
    model=MODEL,
    contents="Say hello in one short sentence.",
)
print(resp.text)""")

    # ================================================================
    # Part 1: Build a ReAct agent from scratch
    # ================================================================

    emit(C, "md", """---
## 1. Build a ReAct agent from scratch

The ReAct pattern (Yao et al., 2022) is a loop: the model looks at the conversation, emits either a final answer or a tool call, we run the tool and append the result as an observation, then repeat. It can be written in about 20 lines.

### 1.1. Tool calling

A *tool* is a Python function the model can ask us to run. We hand the model a JSON schema describing the function; when the model wants to use it, its response contains a structured `function_call` with `(name, args)` in place of plain text.""")

    emit(C, "code", """def calculator(op: str, a: float, b: float) -> dict:
    \"\"\"Perform one arithmetic operation.\"\"\"
    ops = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b if b != 0 else float('nan')}
    if op not in ops:
        return {"error": f"unknown op {op!r}; valid: add, sub, mul, div"}
    return {"result": ops[op]}

CALC_SCHEMA = gtypes.FunctionDeclaration(
    name="calculator",
    description="Perform one arithmetic operation.",
    parameters={
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["add", "sub", "mul", "div"],
                   "description": "operation to perform"},
            "a":  {"type": "number"},
            "b":  {"type": "number"},
        },
        "required": ["op", "a", "b"],
    },
)
print(CALC_SCHEMA)""")

    emit(C, "md", """### 1.2. The agent loop

Three moving parts:

1. Call the model with the running conversation and the tool schemas.
2. If the response is a `function_call`, run the tool and append its output as an observation.
3. If the response is plain text, return it as the final answer.

Tool dispatch is wrapped in `try/except` so malformed calls are fed back to the model as observations rather than crashing the loop.""")

    emit(C, "code", """def run_agent(system_prompt: str,
              user_prompt: str,
              tool_schemas: list,
              tool_fns: dict,
              max_steps: int = 12,
              temperature: float = 0.2,
              verbose: bool = True):
    \"\"\"Minimal ReAct loop. Returns (final_text, transcript).\"\"\"
    contents = [gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=user_prompt)])]
    cfg = gtypes.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[gtypes.Tool(function_declarations=tool_schemas)],
        temperature=temperature,
    )
    transcript = []
    for step in range(max_steps):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=cfg)
        parts = resp.candidates[0].content.parts or []
        contents.append(resp.candidates[0].content)

        tool_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
        if not tool_calls:
            text = "".join(getattr(p, "text", "") or "" for p in parts)
            transcript.append(("final", text))
            if verbose: print(f"[step {step}] FINAL: {text[:200]}")
            return text, transcript

        obs_parts = []
        for fc in tool_calls:
            name, args = fc.name, dict(fc.args or {})
            if verbose: print(f"[step {step}] CALL {name}({args})")
            try:
                result = tool_fns[name](**args)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            r_short = json.dumps(result)[:4000]
            transcript.append((name, args, json.loads(r_short) if r_short.startswith(('{','[')) else result))
            if verbose: print(f"[step {step}]   -> {r_short[:200]}")
            obs_parts.append(gtypes.Part.from_function_response(name=name, response=result))
        contents.append(gtypes.Content(role="user", parts=obs_parts))

    transcript.append(("final", "(max_steps reached without final answer)"))
    return "(max_steps reached)", transcript""")

    emit(C, "md", """### 1.3. Run it on the calculator

Ask the agent to compute $13 \\times 47 + 8 = 619$ using the calculator tool. The transcript shows the sequence of tool calls the model made.""")

    emit(C, "code", """answer, transcript = run_agent(
    system_prompt="You are a careful arithmetic assistant. Use the calculator tool for any computation; do not do arithmetic in your head.",
    user_prompt="Compute 13 * 47 + 8. Return just the final number.",
    tool_schemas=[CALC_SCHEMA],
    tool_fns={"calculator": calculator},
    max_steps=8,
)
print("\\n=== Final answer:", answer)""")

    # ================================================================
    # Part 2: Agentic adaptive FEM on the L-shape
    # ================================================================

    emit(C, "md", r"""---
## 2. Agent-driven adaptive FEM on the L-shape

The Poisson problem $-\Delta u = 1$ with $u=0$ on the boundary of an L-shaped domain has a re-entrant corner at the origin. The solution behaves locally like $r^{2/3}\sin(2\theta/3)$, so $u \in H^{5/3-\epsilon}$ but $u \notin H^2$.

Consequences for P1 finite elements:

- Uniform refinement gives the suboptimal asymptotic rate $O(h^{2/3})$ in the $H^1$-seminorm. At the mesh sizes used here, pre-asymptotic rates of 0.7 to 0.95 are typical.
- Adaptive refinement driven by a local error indicator recovers the optimal rate $O(N^{-1/2})$, i.e. rate 1 against $h \sim N^{-1/2}$.

In this part we give an LLM a set of mesh-manipulation tools and let it drive the refinement. Three variants of agent (A1, A2, A3), then a separate verifier agent that runs a manufactured-solution test.

For a refresher on the scikit-fem API used here, see the [Mar 23 scikit-fem hackathon](""" + SCIKIT_FEM_TUT_URL + ").")

    emit(C, "md", """### 2.1. Forms and the solver

`solve_poisson_on(mesh)` solves $-\\Delta u = 1$ with $u=0$ on the boundary and returns the solution vector and basis.""")

    emit(C, "code", """@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm
def unit_load(v, w):
    return 1.0 * v

def solve_poisson_on(mesh):
    basis = Basis(mesh, ElementTriP1())
    K = stiffness.assemble(basis)
    b = unit_load.assemble(basis)
    u = solve(*condense(K, b, D=mesh.boundary_nodes()))
    return u, basis

def energy_norm_sq(mesh, u):
    basis = Basis(mesh, ElementTriP1())
    @Functional
    def grad_sq(w): return w.w.grad[0]**2 + w.w.grad[1]**2
    return grad_sq.assemble(basis, w=basis.interpolate(u))

m0 = MeshTri.init_lshaped()
fig, ax = plt.subplots(figsize=(5, 5))
ax.triplot(m0.p[0], m0.p[1], m0.t.T, lw=0.5)
ax.set_aspect('equal'); ax.set_title(f"Initial L-shape mesh ({m0.nelements} triangles)")
plt.show()""")

    emit(C, "md", r"""### 2.2. Reference energy

The true solution is not closed-form on the L-shape. Galerkin orthogonality gives
$$
\|u - u_h\|_E^2 \;=\; a(u,u) - a(u_h, u_h),
$$
where $a(\cdot,\cdot) = \int \nabla u \cdot \nabla v$. Take $u$ as the solution on a very fine uniform mesh, treat $a(u,u)$ as the reference, and compute energy-error estimates for any coarser solve.""")

    emit(C, "code", """print("Computing reference energy (one-time setup)...", flush=True)
m_ref = MeshTri.init_lshaped()
for _ in range(7):
    m_ref = m_ref.refined()
u_ref, basis_ref = solve_poisson_on(m_ref)
E_REF = energy_norm_sq(m_ref, u_ref)
print(f"Reference: {basis_ref.N} dofs, energy = {E_REF:.6f}")

def h1_error(mesh, u):
    return float(np.sqrt(max(E_REF - energy_norm_sq(mesh, u), 0.0)))""")

    emit(C, "md", r"""### 2.3. Error indicator and Dorfler marking

Gradient-jump indicator on interior edges:
$$
\eta_K^2 \;=\; \tfrac{1}{2}\sum_{e \in \partial K \cap \Omega^\circ} h_e\, |[\![ \nabla u_h \cdot n_e ]\!]|^2.
$$

Dorfler marking picks the smallest set of elements whose squared indicators sum to at least a fraction $\theta$ of the total. We use $\theta = 0.5$.""")

    emit(C, "code", """def gradient_jump_indicator(mesh, u):
    p, t, facets, f2t = mesh.p, mesh.t, mesh.facets, mesh.f2t
    nelem = t.shape[1]
    g = np.zeros((2, nelem))
    for k in range(nelem):
        i0, i1, i2 = t[:, k]
        A = np.column_stack([p[:, i1] - p[:, i0], p[:, i2] - p[:, i0]])
        rhs = np.array([u[i1] - u[i0], u[i2] - u[i0]])
        g[:, k] = np.linalg.solve(A.T, rhs)
    eta_sq = np.zeros(nelem)
    for e in range(facets.shape[1]):
        tp, tm = f2t[0, e], f2t[1, e]
        if tm < 0: continue
        i0, i1 = facets[:, e]
        edge = p[:, i1] - p[:, i0]
        h_e = np.linalg.norm(edge)
        n = np.array([edge[1], -edge[0]]) / h_e
        jump = (g[:, tp] - g[:, tm]) @ n
        contrib = h_e * jump**2
        eta_sq[tp] += 0.5 * contrib
        eta_sq[tm] += 0.5 * contrib
    return np.sqrt(eta_sq)

def dorfler_mark(indicators, theta=0.5):
    eta2 = indicators**2
    idx = np.argsort(-eta2)
    cum = np.cumsum(eta2[idx])
    k = int(np.searchsorted(cum, theta * cum[-1]) + 1)
    return np.sort(idx[:k])""")

    emit(C, "md", """### 2.4. Tools for the FEM agent

The agent can't pass Python mesh objects through JSON, so we keep a mesh registry keyed by integer IDs. Each tool takes a `mesh_id`, performs its work, and returns a dict.

Two guardrails:
- `MAX_DOFS = 20000` caps the final mesh size.
- Every tool is wrapped in `try/except` so the agent sees errors as observations.""")

    emit(C, "code", """MAX_DOFS = 20000
MESH_REGISTRY = {}
NEXT_ID = [0]
HISTORY = []

def _register(mesh):
    mid = NEXT_ID[0]; NEXT_ID[0] += 1
    MESH_REGISTRY[mid] = mesh
    return mid

def reset_fem_state():
    MESH_REGISTRY.clear()
    NEXT_ID[0] = 0
    HISTORY.clear()
    m0 = MeshTri.init_lshaped()
    return _register(m0)

def tool_describe_mesh(mesh_id: int):
    m = MESH_REGISTRY.get(mesh_id)
    if m is None: return {"error": f"no mesh with id {mesh_id}"}
    return {"mesh_id": mesh_id, "vertices": int(m.p.shape[1]), "elements": int(m.nelements)}

def tool_solve_poisson(mesh_id: int):
    m = MESH_REGISTRY.get(mesh_id)
    if m is None: return {"error": f"no mesh with id {mesh_id}"}
    u, basis = solve_poisson_on(m)
    err = h1_error(m, u)
    step = len(HISTORY)
    HISTORY.append({"step": step, "mesh_id": mesh_id, "dofs": int(basis.N), "H1_error": err})
    MESH_REGISTRY[mesh_id] = m
    MESH_REGISTRY[(mesh_id, "u")] = u
    return {"mesh_id": mesh_id, "dofs": int(basis.N), "H1_error": err, "step": step}

def tool_local_indicators(mesh_id: int):
    m = MESH_REGISTRY.get(mesh_id)
    u = MESH_REGISTRY.get((mesh_id, "u"))
    if m is None or u is None:
        return {"error": f"call solve_poisson({mesh_id}) first"}
    eta = gradient_jump_indicator(m, u)
    return {
        "num_elements": int(len(eta)),
        "eta_max":   float(eta.max()),
        "eta_mean":  float(eta.mean()),
        "eta_min":   float(eta.min()),
    }

def tool_uniform_refine(mesh_id: int):
    m = MESH_REGISTRY.get(mesh_id)
    if m is None: return {"error": f"no mesh with id {mesh_id}"}
    m_new = m.refined()
    if m_new.p.shape[1] > MAX_DOFS:
        return {"error": f"refinement would produce {m_new.p.shape[1]} dofs > cap {MAX_DOFS}"}
    new_id = _register(m_new)
    return {"new_mesh_id": new_id, "new_vertices": int(m_new.p.shape[1]),
            "new_elements": int(m_new.nelements)}

def tool_refine_by_threshold(mesh_id: int, theta: float = 0.5):
    \"\"\"Dorfler marking with bulk fraction theta, then refine.\"\"\"
    m = MESH_REGISTRY.get(mesh_id)
    u = MESH_REGISTRY.get((mesh_id, "u"))
    if m is None or u is None:
        return {"error": f"call solve_poisson({mesh_id}) first"}
    if not (0 < theta < 1):
        return {"error": f"theta must be in (0,1), got {theta}"}
    eta = gradient_jump_indicator(m, u)
    marked = dorfler_mark(eta, theta=theta)
    m_new = m.refined(marked)
    if m_new.p.shape[1] > MAX_DOFS:
        return {"error": f"refinement would produce {m_new.p.shape[1]} dofs > cap {MAX_DOFS}"}
    new_id = _register(m_new)
    return {"new_mesh_id": new_id, "marked": int(len(marked)),
            "new_vertices": int(m_new.p.shape[1]),
            "new_elements": int(m_new.nelements)}

def tool_check_rate(mesh_id: int = -1):
    \"\"\"Return empirical log-log slope from the last 3 solves.\"\"\"
    if len(HISTORY) < 3:
        return {"error": "need at least 3 solves first"}
    h = np.array([1.0/np.sqrt(r["dofs"]) for r in HISTORY[-3:]])
    err = np.array([r["H1_error"] for r in HISTORY[-3:]])
    slope = float(np.polyfit(np.log(h), np.log(err), 1)[0])
    interp = ("close to adaptive optimum (1.0)" if slope > 0.9
              else f"rate {slope:.3f} below adaptive optimum (1.0); uniform predicts 2/3")
    return {"empirical_rate": slope, "interpretation": interp,
            "last3_dofs": [int(r["dofs"]) for r in HISTORY[-3:]],
            "last3_errors": [float(r["H1_error"]) for r in HISTORY[-3:]]}

def tool_stop(reason: str = ""):
    return {"stopped": True, "reason": reason}

FEM_TOOL_FNS = {
    "describe_mesh":        tool_describe_mesh,
    "solve_poisson":        tool_solve_poisson,
    "local_indicators":     tool_local_indicators,
    "uniform_refine":       tool_uniform_refine,
    "refine_by_threshold":  tool_refine_by_threshold,
    "check_convergence_rate": tool_check_rate,
    "stop":                 tool_stop,
}

FEM_TOOL_SCHEMAS = [
    gtypes.FunctionDeclaration(name="describe_mesh", description="Report vertex and element counts for a mesh.",
        parameters={"type":"object","properties":{"mesh_id":{"type":"integer"}},"required":["mesh_id"]}),
    gtypes.FunctionDeclaration(name="solve_poisson", description="Solve -Laplace(u)=1, u=0 on boundary. Returns dofs and H1-seminorm error estimate.",
        parameters={"type":"object","properties":{"mesh_id":{"type":"integer"}},"required":["mesh_id"]}),
    gtypes.FunctionDeclaration(name="local_indicators", description="Summary statistics of the per-element gradient-jump indicators on the current mesh.",
        parameters={"type":"object","properties":{"mesh_id":{"type":"integer"}},"required":["mesh_id"]}),
    gtypes.FunctionDeclaration(name="uniform_refine", description="Uniformly refine every element once. Returns new_mesh_id.",
        parameters={"type":"object","properties":{"mesh_id":{"type":"integer"}},"required":["mesh_id"]}),
    gtypes.FunctionDeclaration(name="refine_by_threshold", description="Dorfler-mark the highest-indicator elements (bulk fraction theta in (0,1)) and refine them. Returns new_mesh_id.",
        parameters={"type":"object","properties":{
            "mesh_id":{"type":"integer"},
            "theta":{"type":"number","description":"bulk fraction; 0.5 is standard"}},
            "required":["mesh_id"]}),
    gtypes.FunctionDeclaration(name="check_convergence_rate", description="Empirical log-log slope of H1-error vs h over the last 3 solves.",
        parameters={"type":"object","properties":{"mesh_id":{"type":"integer"}},"required":[]}),
    gtypes.FunctionDeclaration(name="stop", description="Signal that the refinement loop is done.",
        parameters={"type":"object","properties":{"reason":{"type":"string"}},"required":[]}),
]
print(f"Registered {len(FEM_TOOL_FNS)} FEM tools.")""")

    # --- TODO A1: uniform-refinement agent ---
    emit(C, "md", """### 2.5. TODO A1: uniform-refinement agent

Write a system prompt that tells the agent to drive the mesh toward `H1_error < 5e-2` using only `uniform_refine`. The agent should loop `solve_poisson` + `uniform_refine` until the target is met or the DOF cap is hit, then call `stop`.""")

    emit(C, "todo",
         """reset_fem_state()
initial_mesh_id = 0

A1_SYSTEM = \"\"\"You drive mesh refinement for a Poisson solve on an L-shaped domain.

Strategy for this task:
- Use only `uniform_refine` and `solve_poisson` (not refine_by_threshold).
- Each iteration: solve, check H1_error, if above tol then uniform_refine and repeat.
- Stop when H1_error < 5e-2 or the DOF cap prevents further refinement.
- Call `stop` with a short reason when you finish.

Use tool outputs for all numbers; do not fabricate values.\"\"\"

A1_USER = f\"Starting mesh_id={initial_mesh_id}. Target H1_error < 5e-2.\"

answer_A1, transcript_A1 = run_agent(
    system_prompt=A1_SYSTEM,
    user_prompt=A1_USER,
    tool_schemas=FEM_TOOL_SCHEMAS,
    tool_fns=FEM_TOOL_FNS,
    max_steps=14,
    verbose=True,
)
HISTORY_A1 = list(HISTORY)""",
         """reset_fem_state()
initial_mesh_id = 0

# TODO A1: write the system prompt for a uniform-refinement agent.
# It should loop solve_poisson + uniform_refine until H1_error < 5e-2, then call stop.
# Use only uniform_refine (not refine_by_threshold).
A1_SYSTEM = \"\"\"...your prompt here...\"\"\"

A1_USER = f\"Starting mesh_id={initial_mesh_id}. Target H1_error < 5e-2.\"

answer_A1, transcript_A1 = run_agent(
    system_prompt=A1_SYSTEM,
    user_prompt=A1_USER,
    tool_schemas=FEM_TOOL_SCHEMAS,
    tool_fns=FEM_TOOL_FNS,
    max_steps=14,
    verbose=True,
)
HISTORY_A1 = list(HISTORY)""")

    emit(C, "code", """def plot_convergence(histories, labels, theory_slopes=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for hist, lab in zip(histories, labels):
        if not hist: continue
        h = np.array([1.0/np.sqrt(r['dofs']) for r in hist])
        e = np.array([r['H1_error']          for r in hist])
        ax.loglog(h, e, 'o-', label=f"{lab}  (final err={e[-1]:.2e})")
    if theory_slopes:
        h_ref = np.array([0.4, 0.01])
        for slope, lab in theory_slopes:
            ax.loglog(h_ref, 0.5 * (h_ref/0.4)**slope, '--', alpha=0.6, label=f"slope {slope:.2f} ({lab})")
    ax.set_xlabel("h = 1/sqrt(DOFs)"); ax.set_ylabel("H1 error estimate")
    ax.set_title("Convergence"); ax.grid(True, which='both', alpha=0.3); ax.legend()
    ax.invert_xaxis()
    plt.tight_layout(); plt.show()

plot_convergence([HISTORY_A1], ["A1 uniform"],
                 theory_slopes=[(2/3, "theory uniform"), (1.0, "theory adaptive")])""")

    # --- TODO A2: adaptive-refinement agent ---
    emit(C, "md", """### 2.6. TODO A2: adaptive-refinement agent

Same target tolerance, but the agent can use `refine_by_threshold(mesh_id, theta)`. Write a prompt that uses this tool with Dorfler marking (theta = 0.5).""")

    emit(C, "todo",
         """reset_fem_state()
initial_mesh_id = 0

A2_SYSTEM = \"\"\"You drive adaptive mesh refinement for a Poisson solve on an L-shaped domain.

Strategy:
- Loop: solve_poisson, optionally local_indicators, refine_by_threshold(theta=0.5), repeat.
- Target: H1_error < 5e-2.
- Call stop when the target is met or the DOF cap is hit.
- Use refine_by_threshold (bulk fraction 0.5 is standard), not uniform_refine.\"\"\"

A2_USER = f\"Starting mesh_id={initial_mesh_id}. Target H1_error < 5e-2.\"

answer_A2, transcript_A2 = run_agent(
    system_prompt=A2_SYSTEM, user_prompt=A2_USER,
    tool_schemas=FEM_TOOL_SCHEMAS, tool_fns=FEM_TOOL_FNS,
    max_steps=14, verbose=True,
)
HISTORY_A2 = list(HISTORY)""",
         """reset_fem_state()
initial_mesh_id = 0

# TODO A2: write a system prompt for an adaptive-refinement agent.
# It should use refine_by_threshold (Dorfler marking with theta=0.5)
# instead of uniform_refine.
A2_SYSTEM = \"\"\"...your prompt here...\"\"\"

A2_USER = f\"Starting mesh_id={initial_mesh_id}. Target H1_error < 5e-2.\"

answer_A2, transcript_A2 = run_agent(
    system_prompt=A2_SYSTEM, user_prompt=A2_USER,
    tool_schemas=FEM_TOOL_SCHEMAS, tool_fns=FEM_TOOL_FNS,
    max_steps=14, verbose=True,
)
HISTORY_A2 = list(HISTORY)""")

    emit(C, "code", """# Compare A1 and A2 and show the final adaptive mesh.
final_mesh_id_A2 = max(i for i in MESH_REGISTRY if isinstance(i, int))
final_mesh = MESH_REGISTRY[final_mesh_id_A2]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.triplot(final_mesh.p[0], final_mesh.p[1], final_mesh.t.T, lw=0.3, color='k')
ax.set_aspect('equal'); ax.set_title(f"A2 final adaptive mesh ({final_mesh.nelements} elems)")

ax = axes[1]
for hist, lab in [(HISTORY_A1, "A1 uniform"), (HISTORY_A2, "A2 adaptive")]:
    if not hist: continue
    h = np.array([1.0/np.sqrt(r['dofs']) for r in hist])
    e = np.array([r['H1_error']          for r in hist])
    ax.loglog(h, e, 'o-', label=f"{lab}  (final err={e[-1]:.2e})")
h_ref = np.array([0.4, 0.01])
for slope, lab in [(2/3, "theory uniform"), (1.0, "theory adaptive")]:
    ax.loglog(h_ref, 0.5 * (h_ref/0.4)**slope, '--', alpha=0.6, label=f"slope {slope:.2f} ({lab})")
ax.set_xlabel("h = 1/sqrt(DOFs)"); ax.set_ylabel("H1 error estimate")
ax.set_title("Convergence"); ax.grid(True, which='both', alpha=0.3); ax.legend()
ax.invert_xaxis()
plt.tight_layout(); plt.show()""")

    # --- TODO A3: verification-gated agent ---
    emit(C, "md", """### 2.7. TODO A3: rate-checking agent

The agent has the same tools as A2 plus `check_convergence_rate`, which returns the log-log slope of the last three solves plus a short comparison to theory. Write a prompt that calls `check_convergence_rate` each cycle and increases theta (more aggressive marking) if the rate falls below 0.9 after the mesh exceeds about 500 DOFs.""")

    emit(C, "todo",
         """reset_fem_state()
initial_mesh_id = 0

A3_SYSTEM = \"\"\"You drive adaptive refinement with rate checking.

Tools available: solve_poisson, local_indicators, refine_by_threshold, check_convergence_rate, stop.

Policy:
- Each cycle: solve_poisson, then (once at least 3 solves are done) check_convergence_rate, then refine_by_threshold.
- Target: H1_error < 3e-2 or empirical rate >= 0.95, whichever comes first.
- If check_convergence_rate reports rate < 0.9 after the mesh exceeds 500 dofs, use theta=0.7 to mark more aggressively. Otherwise use theta=0.5.
- Call stop when the target is met or the DOF cap is hit.
- Act only on tool outputs.\"\"\"

A3_USER = f\"Starting mesh_id={initial_mesh_id}. Adaptive refinement with rate checking.\"

answer_A3, transcript_A3 = run_agent(
    system_prompt=A3_SYSTEM, user_prompt=A3_USER,
    tool_schemas=FEM_TOOL_SCHEMAS, tool_fns=FEM_TOOL_FNS,
    max_steps=18, verbose=True,
)
HISTORY_A3 = list(HISTORY)""",
         """reset_fem_state()
initial_mesh_id = 0

# TODO A3: write a prompt for a rate-checking agent.
# The agent should call check_convergence_rate every cycle and raise theta
# (to ~0.7) if the reported rate falls below 0.9 after the mesh grows past 500 DOFs.
A3_SYSTEM = \"\"\"...your prompt here...\"\"\"

A3_USER = f\"Starting mesh_id={initial_mesh_id}. Adaptive refinement with rate checking.\"

answer_A3, transcript_A3 = run_agent(
    system_prompt=A3_SYSTEM, user_prompt=A3_USER,
    tool_schemas=FEM_TOOL_SCHEMAS, tool_fns=FEM_TOOL_FNS,
    max_steps=18, verbose=True,
)
HISTORY_A3 = list(HISTORY)""")

    emit(C, "code", """plot_convergence(
    [HISTORY_A1, HISTORY_A2, HISTORY_A3],
    ["A1 uniform", "A2 adaptive", "A3 rate-checked"],
    theory_slopes=[(2/3, "uniform"), (1.0, "adaptive")]
)""")

    # --- Verifier agent ---
    emit(C, "md", r"""### 2.8. Verifier agent: manufactured-solution test

The estimates above use an energy extrapolation that assumes `solve_poisson_on` is correct. As an independent check we run a manufactured-solution test on the final mesh.

Pick a smooth $u_{\text{exact}}(x,y)$, compute $f = -\Delta u_{\text{exact}}$, solve $-\Delta u_h = f$ with $u_h = u_{\text{exact}}$ on the boundary, and measure $\|u_h - u_{\text{exact}}\|$. For smooth $u_{\text{exact}}$ and P1 elements the expected rates are $L^2$ error $O(h^2)$ and $H^1$-seminorm error $O(h)$.

A separate agent chooses $u_{\text{exact}}$ and runs the test.""")

    emit(C, "code", """def tool_manufactured_test(mesh_id: int, u_expr: str = "sin(pi*x)*sin(pi*y)"):
    \"\"\"Run a manufactured-solution test on the given mesh.

    u_expr: Python expression in x, y using numpy functions and pi.
    Returns L2 and H1-seminorm errors vs the exact u_expr.
    \"\"\"
    m = MESH_REGISTRY.get(mesh_id)
    if m is None: return {"error": f"no mesh with id {mesh_id}"}

    import sympy as sp
    xs, ys = sp.symbols('x y')
    u_sym = sp.sympify(u_expr, locals={'pi': sp.pi})
    lap_sym = sp.diff(u_sym, xs, 2) + sp.diff(u_sym, ys, 2)
    f_sym = -lap_sym
    dudx_sym = sp.diff(u_sym, xs)
    dudy_sym = sp.diff(u_sym, ys)
    f_fn    = sp.lambdify((xs, ys), f_sym,    'numpy')
    u_fn    = sp.lambdify((xs, ys), u_sym,    'numpy')
    dudx_fn = sp.lambdify((xs, ys), dudx_sym, 'numpy')
    dudy_fn = sp.lambdify((xs, ys), dudy_sym, 'numpy')

    @LinearForm
    def manufactured_load(v, w):
        return f_fn(w.x[0], w.x[1]) * v

    basis = Basis(m, ElementTriP1())
    K = stiffness.assemble(basis)
    b = manufactured_load.assemble(basis)
    bnd = m.boundary_nodes()
    u_D = np.zeros(basis.N)
    u_D[bnd] = u_fn(m.p[0, bnd], m.p[1, bnd])
    u = solve(*condense(K, b, D=bnd, x=u_D))

    @Functional
    def L2_sq(w):  return (w.w.value - u_fn(w.x[0], w.x[1]))**2
    @Functional
    def H1_sq(w):
        return (w.w.grad[0] - dudx_fn(w.x[0], w.x[1]))**2 + \\
               (w.w.grad[1] - dudy_fn(w.x[0], w.x[1]))**2
    w = basis.interpolate(u)
    L2 = float(np.sqrt(L2_sq.assemble(basis, w=w)))
    H1 = float(np.sqrt(H1_sq.assemble(basis, w=w)))
    return {"u_exact": u_expr, "dofs": int(basis.N),
            "L2_error": L2, "H1_semi_error": H1,
            "expected_rates": "L2: O(h^2), H1: O(h^1) for smooth u and P1 elements"}

VERIFIER_TOOL_FNS = {"manufactured_test": tool_manufactured_test, "stop": tool_stop}
VERIFIER_TOOL_SCHEMAS = [
    gtypes.FunctionDeclaration(name="manufactured_test",
        description="Run a manufactured-solution test: choose u_exact(x,y), compute f = -Laplacian(u_exact), solve, report L2 and H1 errors.",
        parameters={"type":"object","properties":{
            "mesh_id":{"type":"integer"},
            "u_expr":{"type":"string","description":"expression in x, y, e.g. 'sin(pi*x/2)*sin(pi*y/2)'"}},
            "required":["mesh_id"]}),
    gtypes.FunctionDeclaration(name="stop", description="Report the verification result and finish.",
        parameters={"type":"object","properties":{"reason":{"type":"string"}},"required":[]}),
]""")

    emit(C, "code", """candidate_id = max(i for i in MESH_REGISTRY if isinstance(i, int))

VERIFIER_SYSTEM = \"\"\"You are a verification agent. Another procedure produced a Poisson
solve on mesh_id={mid}. Design a manufactured-solution test (pick a smooth u_exact
that is differentiable everywhere and produces a nontrivial forcing), run
manufactured_test, and judge whether the observed L2 and H1 errors are consistent
with the expected rates for P1 elements. Report PASS or FAIL with a one-sentence
justification.\"\"\".format(mid=candidate_id)

VERIFIER_USER = f\"Verify mesh_id={candidate_id}.\"

verifier_answer, verifier_transcript = run_agent(
    system_prompt=VERIFIER_SYSTEM, user_prompt=VERIFIER_USER,
    tool_schemas=VERIFIER_TOOL_SCHEMAS, tool_fns=VERIFIER_TOOL_FNS,
    max_steps=6, verbose=True,
)
print("\\n=== Verifier verdict:")
print(verifier_answer)""")

    # ================================================================
    # Part 3: Black-box QOI optimization
    # ================================================================

    emit(C, "md", r"""---
## 3. Agent-driven optimization of a black-box simulator

The agent now sees a black-box dynamical simulator: it can observe outputs but not inspect the code. Task: find the launch angle $\theta \in (0, 90^\circ)$ that maximizes horizontal range for a projectile with initial speed $v_0 = 50$ m/s under quadratic drag ($c_d = 0.01$ m$^{-1}$) and gravity $g = 9.81$ m/s$^2$. Without drag the optimum is $45^\circ$; with drag it shifts lower.""")

    emit(C, "code", """V0, CD, G = 50.0, 0.01, 9.81

def _rhs_drag(s, t):
    x, y, vx, vy = s
    v = np.sqrt(vx*vx + vy*vy)
    return [vx, vy, -CD*v*vx, -G - CD*v*vy]

def simulate_range(angle_deg: float) -> float:
    a = np.deg2rad(angle_deg)
    s0 = [0.0, 0.0, V0*np.cos(a), V0*np.sin(a)]
    t = np.linspace(0, 20, 4001)
    sol = odeint(_rhs_drag, s0, t)
    y = sol[:, 1]
    hit = np.where((y[:-1] >= 0) & (y[1:] < 0))[0]
    if len(hit) == 0: return float('nan')
    i = hit[0]
    frac = y[i] / (y[i] - y[i+1])
    return float(sol[i, 0] + frac * (sol[i+1, 0] - sol[i, 0]))

def simulate_max_height(angle_deg: float) -> float:
    a = np.deg2rad(angle_deg)
    s0 = [0.0, 0.0, V0*np.cos(a), V0*np.sin(a)]
    t = np.linspace(0, 20, 4001)
    sol = odeint(_rhs_drag, s0, t)
    return float(sol[:, 1].max())

angles = np.linspace(5, 85, 81)
ranges = [simulate_range(a) for a in angles]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(angles, ranges, 'b-')
ax.set_xlabel("launch angle (deg)"); ax.set_ylabel("range (m)")
ax.set_title("Projectile range vs launch angle")
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()""")

    emit(C, "md", """### 3.1. Tools for the optimization agent

Two read-only tools: `evaluate(angle)` runs the simulator, `history()` returns past evaluations. `propose_answer` reports a final guess.""")

    emit(C, "code", """EVAL_LOG = []

def tool_evaluate(angle_deg: float):
    \"\"\"Return the horizontal range for a given launch angle.\"\"\"
    if not (0 < angle_deg < 90):
        return {"error": f"angle must be in (0, 90) degrees; got {angle_deg}"}
    r = simulate_range(angle_deg)
    EVAL_LOG.append({"angle_deg": angle_deg, "range": r})
    return {"angle_deg": angle_deg, "range": r, "evaluations_so_far": len(EVAL_LOG)}

def tool_history():
    return {"evaluations": EVAL_LOG, "count": len(EVAL_LOG)}

def tool_propose_answer(angle_deg: float, rationale: str = ""):
    return {"proposed_optimum_deg": angle_deg, "rationale": rationale}

OPT_TOOL_FNS = {
    "evaluate":        tool_evaluate,
    "history":         tool_history,
    "propose_answer":  tool_propose_answer,
}
OPT_TOOL_SCHEMAS = [
    gtypes.FunctionDeclaration(name="evaluate", description="Run the black-box projectile simulator for a given launch angle and return the horizontal range.",
        parameters={"type":"object","properties":{"angle_deg":{"type":"number"}},"required":["angle_deg"]}),
    gtypes.FunctionDeclaration(name="history", description="Return all previous (angle, range) evaluations.",
        parameters={"type":"object","properties":{},"required":[]}),
    gtypes.FunctionDeclaration(name="propose_answer", description="State your best estimate of the optimal launch angle and stop.",
        parameters={"type":"object","properties":{
            "angle_deg":{"type":"number"},
            "rationale":{"type":"string"}},"required":["angle_deg"]}),
]""")

    # --- TODO B1: naive trial agent ---
    emit(C, "md", """### 3.2. TODO B1: trial agent

Give the agent a small evaluation budget and let it pick its own search strategy.""")

    emit(C, "todo",
         """EVAL_LOG.clear()

B1_SYSTEM = \"\"\"You optimize a black-box function that takes a single number
(angle in degrees, in (0, 90)) and returns a range in meters. Find the angle
that maximizes the range.

You have an evaluation budget of at most 12 `evaluate` calls. Use `history` to
review past results if helpful, then call `propose_answer` with your final guess.\"\"\"

B1_USER = "Find the optimal angle. Budget: 12 evaluations.\"

answer_B1, transcript_B1 = run_agent(
    system_prompt=B1_SYSTEM, user_prompt=B1_USER,
    tool_schemas=OPT_TOOL_SCHEMAS, tool_fns=OPT_TOOL_FNS,
    max_steps=16, verbose=True,
)
EVAL_LOG_B1 = list(EVAL_LOG)""",
         """EVAL_LOG.clear()

# TODO B1: write a prompt for a trial-and-error optimization agent.
# Give it an evaluation budget of ~12 evaluate() calls and let it pick its strategy.
B1_SYSTEM = \"\"\"...your prompt here...\"\"\"

B1_USER = "Find the optimal angle. Budget: 12 evaluations.\"

answer_B1, transcript_B1 = run_agent(
    system_prompt=B1_SYSTEM, user_prompt=B1_USER,
    tool_schemas=OPT_TOOL_SCHEMAS, tool_fns=OPT_TOOL_FNS,
    max_steps=16, verbose=True,
)
EVAL_LOG_B1 = list(EVAL_LOG)""")

    # --- TODO B2: FD-gradient agent ---
    emit(C, "md", r"""### 3.3. TODO B2: finite-difference gradient agent

Instruct the agent to estimate $dR/d\theta$ by centered finite differences,
$$
R'(\theta) \approx \frac{R(\theta+h) - R(\theta-h)}{2h},
$$
and step in the gradient direction. The agent has to pick the step size $h$ and the learning rate.""")

    emit(C, "todo",
         """EVAL_LOG.clear()

B2_SYSTEM = \"\"\"You optimize a 1D black-box function R(theta) using finite-difference gradient ascent.

Procedure:
  1. Pick a starting angle (e.g. 45 degrees) and an FD step h (try h=1 degree).
  2. Call evaluate at (theta - h) and (theta + h); estimate slope = (R(theta+h) - R(theta-h)) / (2h).
  3. Step theta in the direction of the slope with learning rate eta (start eta=2 degrees per unit slope).
  4. Iterate. If the slope changes sign, halve eta. If progress stalls, halve h.
Budget: at most 16 evaluations. Finish with propose_answer.\"\"\"

B2_USER = "Optimize via finite-difference gradient ascent. Budget: 16 evaluations.\"

answer_B2, transcript_B2 = run_agent(
    system_prompt=B2_SYSTEM, user_prompt=B2_USER,
    tool_schemas=OPT_TOOL_SCHEMAS, tool_fns=OPT_TOOL_FNS,
    max_steps=20, verbose=True,
)
EVAL_LOG_B2 = list(EVAL_LOG)""",
         """EVAL_LOG.clear()

# TODO B2: write a prompt that instructs the agent to use finite-difference
# gradient ascent. The agent has to pick the FD step h and the learning rate eta,
# and adapt them if progress stalls or the slope changes sign.
B2_SYSTEM = \"\"\"...your prompt here...\"\"\"

B2_USER = "Optimize via finite-difference gradient ascent. Budget: 16 evaluations.\"

answer_B2, transcript_B2 = run_agent(
    system_prompt=B2_SYSTEM, user_prompt=B2_USER,
    tool_schemas=OPT_TOOL_SCHEMAS, tool_fns=OPT_TOOL_FNS,
    max_steps=20, verbose=True,
)
EVAL_LOG_B2 = list(EVAL_LOG)""")

    # --- TODO B3: constrained variant (bonus) ---
    emit(C, "md", r"""### 3.4. TODO B3 (bonus): constrained variant

Maximize range subject to $\max_t y(t) \le 30$ m. Two tools, objective and constraint, both black-box.""")

    emit(C, "code", """def tool_evaluate_height(angle_deg: float):
    if not (0 < angle_deg < 90):
        return {"error": f"angle must be in (0, 90); got {angle_deg}"}
    return {"angle_deg": angle_deg, "max_height": simulate_max_height(angle_deg)}

OPT_TOOL_FNS_CONSTR = dict(OPT_TOOL_FNS, evaluate_height=tool_evaluate_height)
OPT_TOOL_SCHEMAS_CONSTR = OPT_TOOL_SCHEMAS + [
    gtypes.FunctionDeclaration(name="evaluate_height",
        description="Return the max altitude (meters) reached for a given launch angle.",
        parameters={"type":"object","properties":{"angle_deg":{"type":"number"}},"required":["angle_deg"]}),
]""")

    emit(C, "todo",
         """EVAL_LOG.clear()

B3_SYSTEM = \"\"\"You maximize horizontal range R(theta) for a projectile subject to
max altitude <= 30 meters. Tools:
  evaluate(theta)         -> range
  evaluate_height(theta)  -> max altitude
For any candidate angle, check both the range and the altitude constraint. Only
consider angles whose max altitude <= 30. Budget: up to 16 total tool calls.
Finish with propose_answer giving your best feasible angle.\"\"\"

B3_USER = "Max range subject to max_height <= 30 m. Budget: 16 evaluations.\"

answer_B3, transcript_B3 = run_agent(
    system_prompt=B3_SYSTEM, user_prompt=B3_USER,
    tool_schemas=OPT_TOOL_SCHEMAS_CONSTR, tool_fns=OPT_TOOL_FNS_CONSTR,
    max_steps=20, verbose=True,
)
EVAL_LOG_B3 = list(EVAL_LOG)""",
         """EVAL_LOG.clear()

# TODO B3 (bonus): prompt the agent to maximize range subject to max_height <= 30.
# The agent must call both evaluate and evaluate_height and only propose an angle
# whose max altitude is below the cap.
B3_SYSTEM = \"\"\"...your prompt here...\"\"\"

B3_USER = "Max range subject to max_height <= 30 m. Budget: 16 evaluations.\"

answer_B3, transcript_B3 = run_agent(
    system_prompt=B3_SYSTEM, user_prompt=B3_USER,
    tool_schemas=OPT_TOOL_SCHEMAS_CONSTR, tool_fns=OPT_TOOL_FNS_CONSTR,
    max_steps=20, verbose=True,
)
EVAL_LOG_B3 = list(EVAL_LOG)""")

    # --- Verifier agent for Part 3 ---
    emit(C, "md", """### 3.5. Verifier agent: RK45 oracle

`simulate_range` uses `scipy.integrate.odeint` at default tolerance. A separate agent takes the claimed optimum and re-integrates with `scipy.integrate.solve_ivp(method='RK45', rtol=1e-10)`, then reports the discrepancy between the two integrators.""")

    emit(C, "code", """def tool_oracle_range(angle_deg: float):
    a = np.deg2rad(angle_deg)
    s0 = [0.0, 0.0, V0*np.cos(a), V0*np.sin(a)]
    def rhs(t, s): return _rhs_drag(s, t)
    def hit_ground(t, s): return s[1]
    hit_ground.terminal = True; hit_ground.direction = -1
    sol = solve_ivp(rhs, (1e-6, 20), s0, method='RK45', rtol=1e-10, atol=1e-12, events=hit_ground)
    r_oracle = float(sol.y[0, -1])
    r_default = simulate_range(angle_deg)
    return {"angle_deg": angle_deg, "range_default": r_default,
            "range_RK45": r_oracle,
            "abs_discrepancy": abs(r_oracle - r_default)}

OPT_VERIFIER_TOOL_FNS = {"oracle_range": tool_oracle_range, "stop": tool_stop}
OPT_VERIFIER_TOOL_SCHEMAS = [
    gtypes.FunctionDeclaration(name="oracle_range",
        description="Re-run the projectile simulator with RK45 at tight tolerance and compare to the default solver.",
        parameters={"type":"object","properties":{"angle_deg":{"type":"number"}},"required":["angle_deg"]}),
    gtypes.FunctionDeclaration(name="stop", description="Report the verification result.",
        parameters={"type":"object","properties":{"reason":{"type":"string"}},"required":[]}),
]
print(f"Registered {len(OPT_VERIFIER_TOOL_FNS)} verifier tools.")""")

    emit(C, "code", """candidate_angle = None
for src_ in [transcript_B2, transcript_B1]:
    try:
        for entry in src_[::-1]:
            if entry[0] == 'propose_answer':
                candidate_angle = float(entry[1]['angle_deg']); break
        if candidate_angle is not None: break
    except NameError:
        continue
print(f"Verifying candidate angle: {candidate_angle}")

OPT_V_SYSTEM = f\"\"\"You are a verifier. Another procedure claimed the optimal launch
angle is about {candidate_angle} degrees. Call oracle_range on a few angles near
this claim and check (a) the default simulator agrees with the RK45 oracle to
within 0.01 m, and (b) the claimed angle is the argmax among the nearby angles.
Report PASS or FAIL with one sentence of justification.\"\"\"

OPT_V_USER = f"Verify the claim that {candidate_angle} degrees maximizes range.\"

opt_verifier_answer, _ = run_agent(
    system_prompt=OPT_V_SYSTEM, user_prompt=OPT_V_USER,
    tool_schemas=OPT_VERIFIER_TOOL_SCHEMAS, tool_fns=OPT_VERIFIER_TOOL_FNS,
    max_steps=8, verbose=True,
)
print("\\n=== Verifier verdict:")
print(opt_verifier_answer)""")

    emit(C, "md", """### 3.6. Oracle comparison

For reference, `scipy.optimize.minimize_scalar` solves the same problem directly.""")

    emit(C, "code", """res = minimize_scalar(lambda a: -simulate_range(a), bounds=(1, 89), method='bounded',
                      options={'xatol': 1e-3})
print(f"scipy.minimize_scalar: optimum = {res.x:.4f} deg, range = {-res.fun:.4f} m, evals = {res.nfev}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(angles, ranges, 'b-', alpha=0.6, label='range(theta)')
for eval_log, lab, marker in [(EVAL_LOG_B1, 'B1 (trial)', 'o'),
                              (EVAL_LOG_B2, 'B2 (FD gradient)', 's')]:
    if eval_log:
        xs = [e['angle_deg'] for e in eval_log]; ys = [e['range'] for e in eval_log]
        ax.scatter(xs, ys, marker=marker, s=40, alpha=0.7, label=lab)
ax.axvline(res.x, color='k', ls='--', alpha=0.6, label=f'scipy opt ({res.x:.2f})')
ax.set_xlabel("launch angle (deg)"); ax.set_ylabel("range (m)")
ax.set_title("Agent evaluations vs oracle optimum")
ax.grid(alpha=0.3); ax.legend(); plt.tight_layout(); plt.show()""")

    # ================================================================
    # Part 4: Further reading
    # ================================================================

    emit(C, "md", """---
## 4. Further reading

- [smolagents](https://huggingface.co/docs/smolagents): minimal agent framework.
- [LangGraph](https://langchain-ai.github.io/langgraph/): multi-agent orchestration with explicit state machines.
- [Model Context Protocol](https://modelcontextprotocol.io): emerging standard for how agents talk to tools.
- [Sakana AI Scientist](https://sakana.ai/ai-scientist/): autonomous ML research agent.
- [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) (DeepMind, Nature 2024) and [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (2025): LLM-guided program search.
- [ChemCrow](https://arxiv.org/abs/2304.05376) and [Coscientist](https://www.nature.com/articles/s41586-023-06792-0): agents driving chemistry experiments.""")

    return C


# ---------------------------------------------------------------------------
# Emit the two notebooks
# ---------------------------------------------------------------------------

def write_notebook(path, cells, variant):
    """variant in {'student', 'solution'}."""
    ipynb_cells = []
    for entry in cells:
        kind = entry[0]
        if kind == "md":
            body = entry[1]
            ipynb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src(body)})
        elif kind == "code":
            body = entry[1]
            ipynb_cells.append({"cell_type": "code", "execution_count": None,
                                "metadata": {}, "outputs": [], "source": src(body)})
        elif kind == "todo":
            body = entry[1] if variant == "solution" else entry[2]
            ipynb_cells.append({"cell_type": "code", "execution_count": None,
                                "metadata": {}, "outputs": [], "source": src(body)})
        elif kind == "todo_md":
            body = entry[1] if variant == "solution" else entry[2]
            ipynb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src(body)})
        elif kind == "md_both":
            body = entry[2] if variant == "solution" else entry[1]
            ipynb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src(body)})
        else:
            raise ValueError(f"unknown cell kind {kind}")

    if ipynb_cells and ipynb_cells[0]["cell_type"] == "markdown":
        badge = COLAB_BADGE_SOL if variant == "solution" else COLAB_BADGE_STU
        ipynb_cells[0]["source"] = src(badge)

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "include_colab_link": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": ipynb_cells,
    }
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    return len(ipynb_cells)


def main():
    cells = build_cells()
    n_stu = write_notebook(STU_PATH, cells, variant='student')
    n_sol = write_notebook(SOL_PATH, cells, variant='solution')
    print(f"Wrote {n_stu} cells -> {os.path.basename(STU_PATH)}")
    print(f"Wrote {n_sol} cells -> {os.path.basename(SOL_PATH)}")

    try:
        import nbformat
        for p in (STU_PATH, SOL_PATH):
            nbformat.validate(nbformat.read(p, as_version=4))
            print(f"nbformat validation: PASSED for {os.path.basename(p)}")
    except ImportError:
        print("(nbformat not installed; skipping validation)")
    except Exception as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
