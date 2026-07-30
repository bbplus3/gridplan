"""
Transmission expansion planning (TEP): a MILP that decides which candidate
lines to build, jointly across all given snapshots, using the standard
disjunctive/big-M formulation for the candidate lines' power-flow equations.
"""
from dataclasses import dataclass, field

import pyomo.environ as pyo

from .congestion import BASE_MVA, VOLL, DEFAULT_SOLVER

BIG_M = 500  # MW-equivalent big-M; should exceed any plausible angle-driven flow


@dataclass
class ExpansionResult:
    status: str
    objective_usd: float
    lines_to_build: list
    build_decision: dict                    # candidate line_id -> bool
    total_capital_cost_million_usd: float
    total_unserved_mwh_sample: float
    annualization_factor: float

    def summary(self) -> str:
        lines = [f"status={self.status}  objective=${self.objective_usd:,.0f}"]
        if self.lines_to_build:
            lines.append(f"  build: {', '.join(self.lines_to_build)} "
                          f"(${self.total_capital_cost_million_usd}M capital)")
        else:
            lines.append("  build: none recommended")
        return "\n".join(lines)


def _build_tep_model(net, snapshots, annualization_factor, slack_bus=None):
    all_lines = net.lines
    existing = net.existing_lines()
    candidates = net.candidate_lines()
    buses = net.buses
    gens = net.generators

    if slack_bus is None:
        slack_bus = next(iter(buses))

    m = pyo.ConcreteModel(name="TEP")
    m.BUS = pyo.Set(initialize=list(buses.keys()))
    m.LINE = pyo.Set(initialize=list(all_lines.keys()))
    m.EXISTING = pyo.Set(initialize=list(existing.keys()))
    m.CANDIDATE = pyo.Set(initialize=list(candidates.keys()))
    m.GEN = pyo.Set(initialize=list(gens.keys()))
    m.SNAP = pyo.Set(initialize=list(snapshots))

    m.build = pyo.Var(m.CANDIDATE, within=pyo.Binary)
    m.gen_mw = pyo.Var(m.GEN, m.SNAP, within=pyo.NonNegativeReals)
    m.angle = pyo.Var(m.BUS, m.SNAP, within=pyo.Reals, bounds=(-1.571, 1.571))
    m.flow = pyo.Var(m.LINE, m.SNAP, within=pyo.Reals)
    m.unserved_mw = pyo.Var(m.BUS, m.SNAP, within=pyo.NonNegativeReals)

    def gen_limit_rule(m, g, s):
        return m.gen_mw[g, s] <= gens[g].capacity_mw
    m.gen_limit = pyo.Constraint(m.GEN, m.SNAP, rule=gen_limit_rule)

    def slack_rule(m, s):
        return m.angle[slack_bus, s] == 0
    m.slack_angle = pyo.Constraint(m.SNAP, rule=slack_rule)

    def exist_flow_eq_rule(m, l, s):
        ln = existing[l]
        return m.flow[l, s] == BASE_MVA * (m.angle[ln.from_bus, s] - m.angle[ln.to_bus, s]) / ln.reactance_pu
    m.exist_flow_eq = pyo.Constraint(m.EXISTING, m.SNAP, rule=exist_flow_eq_rule)

    def exist_upper_rule(m, l, s):
        return m.flow[l, s] <= existing[l].capacity_mw
    m.exist_upper = pyo.Constraint(m.EXISTING, m.SNAP, rule=exist_upper_rule)

    def exist_lower_rule(m, l, s):
        return m.flow[l, s] >= -existing[l].capacity_mw
    m.exist_lower = pyo.Constraint(m.EXISTING, m.SNAP, rule=exist_lower_rule)

    def cand_upper_bigM_rule(m, l, s):
        ln = candidates[l]
        phys = BASE_MVA * (m.angle[ln.from_bus, s] - m.angle[ln.to_bus, s]) / ln.reactance_pu
        return m.flow[l, s] - phys <= BIG_M * (1 - m.build[l])
    m.cand_upper_bigM = pyo.Constraint(m.CANDIDATE, m.SNAP, rule=cand_upper_bigM_rule)

    def cand_lower_bigM_rule(m, l, s):
        ln = candidates[l]
        phys = BASE_MVA * (m.angle[ln.from_bus, s] - m.angle[ln.to_bus, s]) / ln.reactance_pu
        return m.flow[l, s] - phys >= -BIG_M * (1 - m.build[l])
    m.cand_lower_bigM = pyo.Constraint(m.CANDIDATE, m.SNAP, rule=cand_lower_bigM_rule)

    def cand_cap_upper_rule(m, l, s):
        return m.flow[l, s] <= candidates[l].capacity_mw * m.build[l]
    m.cand_cap_upper = pyo.Constraint(m.CANDIDATE, m.SNAP, rule=cand_cap_upper_rule)

    def cand_cap_lower_rule(m, l, s):
        return m.flow[l, s] >= -candidates[l].capacity_mw * m.build[l]
    m.cand_cap_lower = pyo.Constraint(m.CANDIDATE, m.SNAP, rule=cand_cap_lower_rule)

    def balance_rule(m, b, s):
        gen_at_bus = sum(m.gen_mw[g, s] for g in gens if gens[g].bus_id == b)
        inflow = sum(m.flow[l, s] for l in all_lines if all_lines[l].to_bus == b) \
            - sum(m.flow[l, s] for l in all_lines if all_lines[l].from_bus == b)
        d = net.loads.get(s, {}).get(b, 0.0)
        return gen_at_bus + inflow + m.unserved_mw[b, s] == d
    m.balance = pyo.Constraint(m.BUS, m.SNAP, rule=balance_rule)

    def obj_rule(m):
        invest_cost = sum(m.build[l] * candidates[l].cost_million_usd * 1e6 for l in m.CANDIDATE)
        op_cost = annualization_factor * sum(
            m.gen_mw[g, s] * gens[g].marginal_cost_per_mwh for g in m.GEN for s in m.SNAP)
        shed_cost = annualization_factor * sum(
            m.unserved_mw[b, s] * VOLL for b in m.BUS for s in m.SNAP)
        return invest_cost + op_cost + shed_cost
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return m


def solve_expansion(network, snapshots=None, annualization_factor=None,
                     solver_name=DEFAULT_SOLVER) -> ExpansionResult:
    """
    Solve the transmission expansion planning MILP: which candidate lines
    to build to minimize capital + annualized operating/reliability cost.

    annualization_factor: scales operating/shed cost (computed from your
    sample snapshots) up to be comparable with one-time capital costs.
    Defaults to 8760 / len(snapshots), i.e. treats your snapshots as a
    representative sample standing in for a full year.
    """
    if snapshots is None:
        snapshots = network.snapshots
    if annualization_factor is None:
        annualization_factor = 8760 / len(snapshots)

    m = _build_tep_model(network, snapshots, annualization_factor)
    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(m)
    status = str(result.solver.termination_condition)

    built = {l: bool(round(pyo.value(m.build[l]))) for l in m.CANDIDATE}
    total_shed = sum(pyo.value(m.unserved_mw[b, s]) for b in m.BUS for s in m.SNAP)
    invest = sum(pyo.value(m.build[l]) * network.candidate_lines()[l].cost_million_usd
                 for l in m.CANDIDATE)

    return ExpansionResult(
        status=status,
        objective_usd=round(pyo.value(m.obj), 2),
        lines_to_build=[l for l, b in built.items() if b],
        build_decision=built,
        total_capital_cost_million_usd=round(invest, 2),
        total_unserved_mwh_sample=round(total_shed, 2),
        annualization_factor=annualization_factor,
    )
