"""
gridplan: DC-OPF congestion analysis and MILP-based transmission expansion
planning, built on Pyomo.

    from gridplan import Network, solve_congestion, solve_expansion

    net = Network.from_csv("data/")
    result = solve_congestion(net, "summer_peak_h17")
    print(result.summary())
    print("Overloaded lines:", result.lines_over(90))

    plan = solve_expansion(net)
    print(plan.summary())

Optional extras (not imported by default, so `import gridplan` stays
lightweight):
    from gridplan import visualize     # static PNG diagrams (needs matplotlib, networkx)
    from gridplan import interactive   # draggable HTML diagrams (needs pyvis)
    from gridplan import gnn           # GNN power-flow surrogate (needs torch)
"""
from .network import Network, Bus, Line, Generator
from .congestion import solve_congestion, summarize_congestion, CongestionResult, CongestionSummary, LineFlow
from .expansion import solve_expansion, ExpansionResult

__version__ = "0.1.0"

__all__ = [
    "Network", "Bus", "Line", "Generator",
    "solve_congestion", "summarize_congestion", "CongestionResult", "CongestionSummary", "LineFlow",
    "solve_expansion", "ExpansionResult",
]
