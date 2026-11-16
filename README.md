# gridplan

DC-OPF congestion analysis and MILP-based transmission expansion planning,
built on [Pyomo](https://www.pyomo.org/). Answers two questions:

1. **Given today's grid, where's it congested, and what's that costing us?**
2. **Given a list of candidate new lines and their price tags, which ones are
   actually worth building?**

```python
from gridplan import Network, solve_congestion, solve_expansion

net = Network.from_csv("examples/sample_data")

result = solve_congestion(net, "summer_peak_h17")
print(result.summary())
print("Overloaded lines:", result.lines_over(90))
print("LMPs ($/MWh):", result.lmps)

plan = solve_expansion(net)
print(plan.summary())
```

## Install

```bash
pip install gridplan              # core: Network, solve_congestion, solve_expansion
pip install "gridplan[viz]"       # + static/interactive network diagrams
pip install "gridplan[gnn]"       # + GNN power-flow surrogate (needs torch)
pip install "gridplan[app]"       # + Streamlit browser app
pip install "gridplan[all]"       # everything
```

Or install from source to get the examples and app files:

```bash
git clone https://github.com/bbplus3/gridplan.git
cd gridplan
pip install -e ".[all]"
```

No separate solver binary needed — `highspy` (the HiGHS solver, pure pip
install) is a core dependency, used by default for both the LP congestion
model and the MILP expansion planning model.

## Quickstart

From a source checkout (the `examples/` and `app.py` files aren't shipped in
the pip package):

```bash
python examples/quickstart.py
```

or generate a browser diagram / run the full app:

```bash
pip install -e ".[app]"
streamlit run app.py
```

## How the model works

**DC power flow approximation.** Like virtually all transmission planning
tools (PyPSA, PLEXOS, Google Tapestry, etc.), this uses the linearized "DC"
power flow model rather than full AC power flow: flow on a line is
proportional to the voltage-angle difference across it, divided by the
line's reactance. Standard for planning studies — fast, always solves to a
clean global optimum, and captures the loop-flow effects that actually
drive congestion — but not adequate for detailed voltage/operational
studies (use `pandapower` or `PyPSA` with an AC solver for that).

**Congestion model (`gridplan.congestion`).** For a given snapshot, minimizes
generation cost subject to power balance at every bus and flow limits on
every existing line — a linear program, solved to global optimality via
simplex. The dual value on each bus's power-balance constraint is that
bus's locational marginal price (LMP); this falls straight out of the
solve, no separate computation needed. `CongestionResult` wraps the answer
with convenience methods: `.lines_over(pct)`, `.binding_lines()`,
`.is_feasible`, `.summary()`.

**Expansion planning model (`gridplan.expansion`).** Adds a binary
build/don't-build decision per candidate line, solved jointly across every
snapshot you give it (a mixed-integer linear program). Uses the standard
disjunctive/big-M formulation: a candidate line's power-flow equation is
only enforced when its build variable is 1. The objective weighs one-time
capital cost against annualized operating + reliability cost.

**GNN power-flow surrogate (`gridplan.gnn`, optional `[gnn]`).** Simulates
thousands of randomized load scenarios, solves each with the exact DC model
to build a training set, and trains a graph neural network to approximate
power flow from grid topology. This is a demonstration of the technique, not
a production speedup: at the sizes gridplan targets, the exact linear solve is
both faster and exact, and the module is upfront about that. It's here to show
how a learned surrogate is built and trained — and to make honest when it would
and wouldn't pay off on a larger system. See the module docstring for that
discussion.

## Package layout

```
src/gridplan/
    network.py       - Bus, Line, Generator, Network (Network.from_csv(...))
    congestion.py     - solve_congestion(), summarize_congestion(), CongestionResult
    expansion.py      - solve_expansion(), ExpansionResult
    visualize.py       - static PNG network diagrams (needs [viz])
    interactive.py     - draggable HTML network diagrams via pyvis (needs [viz])
    gnn.py              - GNN power-flow surrogate (needs [gnn]; see module docstring
                          for an honest discussion of when this technique actually helps)
examples/
    sample_data/      - a fictional 9-bus, 3-zone utility system
    generate_sample_data.py
    quickstart.py
tests/
    test_gridplan.py  - hand-checkable 2-bus cases + sanity checks on the sample network
app.py                 - Streamlit browser app (needs [app])
```

## Bringing your own network

Replace `examples/sample_data/*.csv` (or point `Network.from_csv()` at your
own directory) with four CSVs in this schema:

| File | Columns | Notes |
|---|---|---|
| `buses.csv` | `bus_id, name, zone, base_kv` | one row per substation/bus |
| `lines.csv` | `line_id, from_bus, to_bus, reactance_pu, capacity_mw, status, length_km, cost_million_usd` | `status` is `existing` or `candidate`; cost only needed for candidates |
| `generators.csv` | `gen_id, bus_id, name, capacity_mw, marginal_cost_per_mwh, fuel_type` | one row per generating unit |
| `loads.csv` | `snapshot, bus_id, hour, season, demand_mw` | one row per (snapshot, bus); a "snapshot" is any label for an hour/condition |

`reactance_pu` needs a consistent MVA base across all lines (default 100
MVA — change `BASE_MVA` in `gridplan/congestion.py` and `gridplan/expansion.py`
if yours differs). For renewable generators, note `capacity_mw` is treated
as a hard cap; if you want per-snapshot availability (e.g. wind/solar
capacity factors), that's a small, contained change to the generator-limit
constraint in both modules.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see `LICENSE`.
