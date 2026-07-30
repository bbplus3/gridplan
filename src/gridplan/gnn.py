"""
A Graph Neural Network that learns to approximate DC power flow.

IMPORTANT FRAMING - read this before the code:

Computing flows for a *given* set of generator/load injections is NOT an
optimization problem - it's just solving the linear system P = B @ theta,
where B is the bus susceptance matrix built from line reactances. That's
what `exact_dc_powerflow()` below does, in a few lines of numpy, exactly
and instantly. dcopf.py is the OPTIMIZATION layer on top of this: it
additionally *chooses* generator dispatch to minimize cost, subject to
that same physics.

So why train a neural net to approximate something you can already solve
exactly and near-instantly? At THIS toy scale (9 buses), you shouldn't -
the exact linear solve is faster than a GNN forward pass and always exact.
The real motivation for a learned power-flow surrogate shows up at scale:

  - Contingency screening: utilities routinely need to check "what if
    line X trips?" across thousands of N-1/N-2 combinations. Refactorizing
    a large admittance matrix (or resolving an LP) for each one is
    expensive; a trained GNN forward pass is a single fast batched matrix
    multiply, and (in principle, if trained on varied topologies) can
    generalize to topology changes it wasn't explicitly solved for.
  - Embedding inside a bigger optimization loop (e.g., a stochastic
    planning model evaluating thousands of scenarios) where re-solving
    an exact power flow inside the loop is the bottleneck.
  - Very large meshed networks where the admittance matrix is big enough
    that repeated exact solves actually do dominate runtime.

This script is an honest, small-scale demonstration of the technique -
including a benchmark at the end that shows the exact solve winning at
this size, so you can see where the crossover would need to happen for a
GNN to actually pay off.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn

from .network import Network

BASE_MVA = 100.0


# ---------------------------------------------------------------------------
# Exact DC power flow via linear algebra (ground truth generator - no ML,
# no optimization solver, just P = B @ theta)
# ---------------------------------------------------------------------------
def build_susceptance_matrix(net):
    """Returns (B, bus_ids, line_ids, line_endpoints, line_susceptance)."""
    bus_ids = sorted(net.buses.keys())
    idx = {b: i for i, b in enumerate(bus_ids)}
    n = len(bus_ids)
    B = np.zeros((n, n))

    lines = net.existing_lines()
    line_ids = list(lines.keys())
    line_endpoints = []
    line_susceptance = []

    for lid in line_ids:
        ln = lines[lid]
        b_l = BASE_MVA / ln.reactance_pu  # susceptance
        i, j = idx[ln.from_bus], idx[ln.to_bus]
        B[i, i] += b_l
        B[j, j] += b_l
        B[i, j] -= b_l
        B[j, i] -= b_l
        line_endpoints.append((i, j))
        line_susceptance.append(b_l)

    return B, bus_ids, idx, line_ids, line_endpoints, np.array(line_susceptance)


def exact_dc_powerflow(B, injections, slack_idx):
    """
    Solve P = B @ theta for theta, given net injections (gen - demand) at
    every bus, with slack bus angle fixed to 0. injections can be a single
    vector (n,) or a batch (n_samples, n).
    """
    n = B.shape[0]
    keep = [i for i in range(n) if i != slack_idx]
    B_red = B[np.ix_(keep, keep)]

    single = injections.ndim == 1
    if single:
        injections = injections[None, :]

    P_red = injections[:, keep]  # (n_samples, n-1)
    theta_red = np.linalg.solve(B_red, P_red.T).T  # (n_samples, n-1)

    theta = np.zeros((injections.shape[0], n))
    theta[:, keep] = theta_red
    # slack column stays 0

    if single:
        theta = theta[0]
    return theta


def flows_from_theta(theta, line_endpoints, line_susceptance):
    """theta: (n_samples, n_buses) or (n_buses,). Returns matching-shaped flows."""
    single = theta.ndim == 1
    if single:
        theta = theta[None, :]
    flows = np.zeros((theta.shape[0], len(line_endpoints)))
    for k, (i, j) in enumerate(line_endpoints):
        flows[:, k] = line_susceptance[k] * (theta[:, i] - theta[:, j])
    if single:
        flows = flows[0]
    return flows


# ---------------------------------------------------------------------------
# Training data: random-but-plausible injection scenarios
# ---------------------------------------------------------------------------
def sample_injections(net, bus_ids, idx, n_samples, seed=0):
    """
    Sample net injection (generation - demand) at every bus, keeping total
    generation = total demand (required for power flow to have a solution).
    Demand buses draw from the historical range in loads.csv; one
    generator (the largest-capacity one) acts as the balancing/slack
    generator to make the sample feasible, mirroring how a utility's
    largest baseload/regulating plant absorbs the residual in practice.
    """
    rng = np.random.default_rng(seed)
    n = len(bus_ids)

    # Historical demand range per bus, from the sample loads.csv
    demand_range = {}
    for b in bus_ids:
        vals = [snap.get(b, 0.0) for snap in net.loads.values()]
        vals = [v for v in vals if v > 0]
        if vals:
            demand_range[b] = (min(vals) * 0.8, max(vals) * 1.15)

    gens = net.generators
    gen_ids = list(gens.keys())
    # balancing generator = largest capacity
    slack_gen = max(gen_ids, key=lambda g: gens[g].capacity_mw)
    other_gens = [g for g in gen_ids if g != slack_gen]

    injections = np.zeros((n_samples, n))
    kept = 0
    attempts = 0
    while kept < n_samples and attempts < n_samples * 20:
        attempts += 1
        demand_by_bus = {b: rng.uniform(*demand_range[b]) for b in demand_range}
        total_demand = sum(demand_by_bus.values())

        other_gen_out = {g: rng.uniform(0, gens[g].capacity_mw) for g in other_gens}
        residual = total_demand - sum(other_gen_out.values())
        if residual < 0 or residual > gens[slack_gen].capacity_mw:
            continue  # infeasible draw, resample

        gen_by_bus = {gens[g].bus_id: gens[g].bus_id for g in []}  # noop init
        inj = np.zeros(n)
        for b, d in demand_by_bus.items():
            inj[idx[b]] -= d
        for g, out in other_gen_out.items():
            inj[idx[gens[g].bus_id]] += out
        inj[idx[gens[slack_gen].bus_id]] += residual

        injections[kept] = inj
        kept += 1

    if kept < n_samples:
        injections = injections[:kept]
    return injections


# ---------------------------------------------------------------------------
# GNN model - hand-rolled message passing (no torch_geometric dependency,
# so it's a plain `pip install torch` on any machine including Windows)
# ---------------------------------------------------------------------------
class SusceptanceGraphConv(nn.Module):
    """
    One message-passing layer. Each bus updates its representation using
    (a) its own features and (b) a weighted sum of neighbor features, where
    the weight is the line susceptance - i.e. the graph structure itself
    (not just "is there an edge") is fed into the aggregation, mirroring
    how real DC power flow weights neighbor influence by 1/reactance.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(self, h, A_weighted):
        # h: (batch, n_buses, in_dim); A_weighted: (n_buses, n_buses) susceptance-weighted adjacency
        agg = torch.matmul(A_weighted, h)  # weighted sum of neighbor features
        return self.self_lin(h) + self.neigh_lin(agg)


class PowerFlowGNN(nn.Module):
    def __init__(self, n_buses, hidden_dim=32, n_layers=3):
        super().__init__()
        self.n_buses = n_buses
        dims = [1] + [hidden_dim] * (n_layers - 1) + [1]
        self.layers = nn.ModuleList([
            SusceptanceGraphConv(dims[i], dims[i + 1]) for i in range(n_layers)
        ])
        self.act = nn.Tanh()

    def forward(self, injections, A_weighted, slack_idx):
        # injections: (batch, n_buses) -> (batch, n_buses, 1)
        h = injections.unsqueeze(-1)
        for i, layer in enumerate(self.layers):
            h = layer(h, A_weighted)
            if i < len(self.layers) - 1:
                h = self.act(h)
        theta_pred = h.squeeze(-1)  # (batch, n_buses)
        # Enforce the reference bus: subtract its predicted angle from all
        # buses, exactly like fixing slack angle = 0 in the exact solve.
        # This is a physics-informed constraint baked into the architecture
        # rather than something the network has to learn.
        theta_pred = theta_pred - theta_pred[:, slack_idx:slack_idx + 1]
        return theta_pred


def normalized_adjacency(B, line_endpoints, line_susceptance, n):
    """Off-diagonal susceptance-weighted adjacency (no self-loops), row-normalized
    by total susceptance at each bus so message scale doesn't blow up with degree."""
    A = np.zeros((n, n))
    for (i, j), b_l in zip(line_endpoints, line_susceptance):
        A[i, j] = b_l
        A[j, i] = b_l
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return A / row_sum


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(net, n_samples=4000, epochs=300, seed=0, out_dir=None):
    B, bus_ids, idx, line_ids, line_endpoints, line_susc = build_susceptance_matrix(net)
    n = len(bus_ids)
    slack_idx = 0  # same convention as dcopf.py: first bus in the dict

    injections = sample_injections(net, bus_ids, idx, n_samples, seed=seed)
    theta_true = exact_dc_powerflow(B, injections, slack_idx)
    flows_true = flows_from_theta(theta_true, line_endpoints, line_susc)

    # Normalize injections to roughly unit scale - raw MW values (tens to
    # hundreds) saturate tanh activations and slow convergence a lot.
    inj_scale = np.abs(injections).std()
    injections_norm = injections / inj_scale

    n_total = injections.shape[0]
    n_train = int(n_total * 0.8)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train = torch.tensor(injections_norm[train_idx], dtype=torch.float32)
    X_test = torch.tensor(injections_norm[test_idx], dtype=torch.float32)
    theta_train = torch.tensor(theta_true[train_idx], dtype=torch.float32)
    theta_test_np = theta_true[test_idx]
    flows_test_np = flows_true[test_idx]

    A_weighted = torch.tensor(normalized_adjacency(B, line_endpoints, line_susc, n),
                               dtype=torch.float32)

    model = PowerFlowGNN(n_buses=n, hidden_dim=32, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    loss_history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        theta_pred = model(X_train, A_weighted, slack_idx)
        loss = loss_fn(theta_pred, theta_train)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  train MSE (angle, rad^2): {loss.item():.6f}")

    # --- Evaluate on held-out test set ---
    model.eval()
    with torch.no_grad():
        theta_pred_test = model(X_test, A_weighted, slack_idx).numpy()
    flows_pred_test = flows_from_theta(theta_pred_test, line_endpoints, line_susc)

    angle_mae = np.mean(np.abs(theta_pred_test - theta_test_np))
    flow_mae = np.mean(np.abs(flows_pred_test - flows_test_np))
    flow_mae_pct_of_range = flow_mae / (np.abs(flows_true).max()) * 100

    print(f"\nHeld-out test set ({len(test_idx)} scenarios):")
    print(f"  angle MAE:  {angle_mae:.5f} rad")
    print(f"  flow MAE:   {flow_mae:.2f} MW  ({flow_mae_pct_of_range:.2f}% of max observed flow)")

    # --- Speed comparison: exact linear solve vs GNN forward pass, batched ---
    n_bench = 2000
    bench_injections = injections[:n_bench] if n_bench <= n_total else \
        sample_injections(net, bus_ids, idx, n_bench, seed=seed + 1)

    t0 = time.perf_counter()
    _ = exact_dc_powerflow(B, bench_injections, slack_idx)
    exact_time = time.perf_counter() - t0

    X_bench = torch.tensor(bench_injections / inj_scale, dtype=torch.float32)
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(X_bench, A_weighted, slack_idx)
        gnn_time = time.perf_counter() - t0

    print(f"\nSpeed, {n_bench} scenarios in one batch:")
    print(f"  exact linear solve (numpy): {exact_time*1000:.2f} ms")
    print(f"  GNN forward pass (torch):   {gnn_time*1000:.2f} ms")
    if exact_time < gnn_time:
        print("  -> exact solve wins at this network size (9 buses) - expected. "
              "The GNN would need a much larger network, or topology variation "
              "the exact solve can't easily reuse work across, to pay off.")
    else:
        print("  -> GNN forward pass wins here.")

    if out_dir:
        _save_diagnostic_plots(loss_history, theta_pred_test, theta_test_np,
                                flows_pred_test, flows_test_np, line_ids, out_dir)

    return {
        "model": model,
        "angle_mae": angle_mae,
        "flow_mae_mw": flow_mae,
        "exact_time_ms": exact_time * 1000,
        "gnn_time_ms": gnn_time * 1000,
    }


def _save_diagnostic_plots(loss_history, theta_pred, theta_true, flows_pred, flows_true,
                            line_ids, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(loss_history)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train MSE (angle, rad^2)")
    axes[0].set_title("Training loss")

    axes[1].scatter(theta_true.flatten(), theta_pred.flatten(), s=6, alpha=0.3)
    lims = [theta_true.min(), theta_true.max()]
    axes[1].plot(lims, lims, "r--", linewidth=1)
    axes[1].set_xlabel("true angle (rad)")
    axes[1].set_ylabel("GNN-predicted angle (rad)")
    axes[1].set_title("Bus angles: predicted vs. true (test set)")

    axes[2].scatter(flows_true.flatten(), flows_pred.flatten(), s=6, alpha=0.3)
    lims = [flows_true.min(), flows_true.max()]
    axes[2].plot(lims, lims, "r--", linewidth=1)
    axes[2].set_xlabel("true flow (MW)")
    axes[2].set_ylabel("GNN-predicted flow (MW)")
    axes[2].set_title("Line flows: predicted vs. true (test set)")

    fig.tight_layout()
    path = os.path.join(out_dir, "gnn_powerflow_diagnostics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved diagnostic plots to {path}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             "examples", "sample_data")
    net = Network.from_csv(data_dir)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")
    train_and_evaluate(net, n_samples=4000, epochs=300, out_dir=out_dir)
