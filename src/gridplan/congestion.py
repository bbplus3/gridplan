"""
DC-OPF congestion analysis. See docs/theory.md (or the top-level README)
for the physics and optimization behind this - in short: minimizes
generation cost subject to power balance and line-flow limits, and the
solve's dual values double as locational marginal prices (LMPs).
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import pyomo.environ as pyo

BASE_MVA = 100.0
DEFAULT_SOLVER = "appsi_highs"  # pure-pip solver (via the `highspy` package) - no separate binary needed
VOLL = 5000  # value of lost load, $/MWh


@dataclass
class LineFlow:
    line_id: str
    flow_mw: float
    capacity_mw: float
    loading_pct: float
    binding: bool


@dataclass
class CongestionResult:
    """Result of solving one snapshot's DC-OPF. Prefer the helper methods
    below over digging through the raw dicts directly."""
    snapshot: str
    status: str
    total_cost: float
    gen_dispatch: dict                  # gen_id -> MW
    line_flows: dict                    # line_id -> LineFlow
    lmps: dict                          # bus_id -> $/MWh
    unserved_load: dict                 # bus_id -> MW (only entries > 0)

    @property
    def is_feasible(self) -> bool:
        """True if the network could fully serve demand without shedding load."""
        return not self.unserved_load

    @property
    def total_unserved_mw(self) -> float:
        return sum(self.unserved_load.values())

    def lines_over(self, pct: float = 90.0):
        """Line IDs loaded above `pct`% of thermal capacity."""
        return [lid for lid, lf in self.line_flows.items() if lf.loading_pct >= pct]

    def binding_lines(self):
        """Line IDs at (or essentially at) their thermal limit."""
        return [lid for lid, lf in self.line_flows.items() if lf.binding]

    def summary(self) -> str:
        lines = [f"[{self.snapshot}] status={self.status}  total_cost=${self.total_cost:,.0f}"]
        if not self.is_feasible:
            lines.append(f"  UNSERVED LOAD: {self.total_unserved_mw:.1f} MW")
        binding = self.binding_lines()
        if binding:
            lines.append(f"  binding lines: {', '.join(binding)}")
        return "\n".join(lines)


def _build_dcopf_model(net, snapshot, slack_bus=None):
    lines = net.existing_lines()
    buses = net.buses
    gens = net.generators
    demand = net.loads.get(snapshot, {})

    if slack_bus is None:
        slack_bus = next(iter(buses))

    m = pyo.ConcreteModel(name=f"DCOPF_{snapshot}")
    m.BUS = pyo.Set(initialize=list(buses.keys()))
    m.LINE = pyo.Set(initialize=list(lines.keys()))
    m.GEN = pyo.Set(initialize=list(gens.keys()))

    m.gen_mw = pyo.Var(m.GEN, within=pyo.NonNegativeReals)
    m.angle = pyo.Var(m.BUS, within=pyo.Reals, bounds=(-math.pi / 2, math.pi / 2))
    m.unserved_mw = pyo.Var(m.BUS, within=pyo.NonNegativeReals)

    def gen_limit_rule(m, g):
        return m.gen_mw[g] <= gens[g].capacity_mw
    m.gen_limit = pyo.Constraint(m.GEN, rule=gen_limit_rule)

    m.slack_angle = pyo.Constraint(expr=m.angle[slack_bus] == 0)

    def flow_expr(m, l):
        ln = lines[l]
        return BASE_MVA * (m.angle[ln.from_bus] - m.angle[ln.to_bus]) / ln.reactance_pu
    m.flow = pyo.Expression(m.LINE, rule=flow_expr)

    def flow_upper_rule(m, l):
        return m.flow[l] <= lines[l].capacity_mw
    m.flow_upper = pyo.Constraint(m.LINE, rule=flow_upper_rule)

    def flow_lower_rule(m, l):
        return m.flow[l] >= -lines[l].capacity_mw
    m.flow_lower = pyo.Constraint(m.LINE, rule=flow_lower_rule)

    def balance_rule(m, b):
        gen_at_bus = sum(m.gen_mw[g] for g in gens if gens[g].bus_id == b)
        inflow = sum(m.flow[l] for l in lines if lines[l].to_bus == b) \
            - sum(m.flow[l] for l in lines if lines[l].from_bus == b)
        d = demand.get(b, 0.0)
        return gen_at_bus + inflow + m.unserved_mw[b] == d
    m.balance = pyo.Constraint(m.BUS, rule=balance_rule)

    def obj_rule(m):
        gen_cost = sum(m.gen_mw[g] * gens[g].marginal_cost_per_mwh for g in gens)
        shed_cost = sum(m.unserved_mw[b] * VOLL for b in buses)
        return gen_cost + shed_cost
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def solve_congestion(network, snapshot, solver_name=DEFAULT_SOLVER) -> CongestionResult:
    """Solve DC-OPF for one snapshot and return a CongestionResult."""
    m = _build_dcopf_model(network, snapshot)
    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(m)
    status = str(result.solver.termination_condition)

    lines = network.existing_lines()
    line_flows = {}
    for l in m.LINE:
        flow = pyo.value(m.flow[l])
        cap = lines[l].capacity_mw
        pct = 100 * abs(flow) / cap if cap else 0
        line_flows[l] = LineFlow(line_id=l, flow_mw=round(flow, 2), capacity_mw=cap,
                                  loading_pct=round(pct, 1), binding=pct >= 99.5)

    lmps = {}
    for b in m.BUS:
        try:
            lmps[b] = round(m.dual[m.balance[b]], 3)
        except (KeyError, ValueError):
            lmps[b] = None

    unserved = {b: round(pyo.value(m.unserved_mw[b]), 2) for b in m.BUS
                if pyo.value(m.unserved_mw[b]) > 1e-3}
    gen_dispatch = {g: round(pyo.value(m.gen_mw[g]), 2) for g in m.GEN}

    return CongestionResult(
        snapshot=snapshot, status=status, total_cost=round(pyo.value(m.obj), 2),
        gen_dispatch=gen_dispatch, line_flows=line_flows, lmps=lmps, unserved_load=unserved,
    )


@dataclass
class CongestionSummary:
    n_snapshots: int
    binding_count_by_line: dict
    max_loading_pct_by_line: dict
    total_unserved_mwh: float
    per_snapshot_results: list = field(default_factory=list)

    def worst_lines(self, top_n: int = 5):
        """Line IDs most often binding, most-congested first."""
        return sorted(self.binding_count_by_line, key=lambda l: -self.binding_count_by_line[l])[:top_n]


def summarize_congestion(network, snapshots=None, solver_name=DEFAULT_SOLVER) -> CongestionSummary:
    """Solve every snapshot (or a given subset) and summarize which lines bind most often."""
    if snapshots is None:
        snapshots = network.snapshots

    binding_counts = {l: 0 for l in network.existing_lines()}
    max_loading = {l: 0.0 for l in network.existing_lines()}
    total_shed = 0.0
    results = []

    for snap in snapshots:
        r = solve_congestion(network, snap, solver_name=solver_name)
        results.append(r)
        for lid, lf in r.line_flows.items():
            if lf.binding:
                binding_counts[lid] += 1
            max_loading[lid] = max(max_loading[lid], lf.loading_pct)
        total_shed += r.total_unserved_mw

    return CongestionSummary(
        n_snapshots=len(snapshots),
        binding_count_by_line=binding_counts,
        max_loading_pct_by_line={l: round(v, 1) for l, v in max_loading.items()},
        total_unserved_mwh=round(total_shed, 2),
        per_snapshot_results=results,
    )
