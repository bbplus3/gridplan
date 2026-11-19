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

## Quick start (pip only, no checkout needed)

You don't need the repo to try gridplan — `pip install gridplan` gives you the
full modeling core, and you can build a network in a few lines of Python. Save
this as `gridplan_demo.py` and run `python gridplan_demo.py`:

```python
from gridplan import Network, Bus, Line, Generator, solve_congestion, solve_expansion

net = Network()
for b in (1, 2, 3):
    net.buses[b] = Bus(b, f"Bus {b}", "A", 230.0)

# cheap generation at bus 1, an expensive peaker at bus 3
net.generators["G1"] = Generator("G1", 1, "Cheap",  300, 20, "hydro")
net.generators["G3"] = Generator("G3", 3, "Peaker", 300, 80, "gas")

# a limited direct corridor (1-3), the longer 1-2-3 path, and a candidate reinforcement
net.lines["L13"]  = Line("L13",  1, 3, 0.07, 60,  "existing", 40)
net.lines["L12"]  = Line("L12",  1, 2, 0.10, 100, "existing", 60)
net.lines["L23"]  = Line("L23",  2, 3, 0.10, 100, "existing", 60)
net.lines["L13b"] = Line("L13b", 1, 3, 0.07, 120, "candidate", 40, cost_million_usd=20)

net.snapshots = ["peak"]
net.loads = {"peak": {3: 220.0}}  # 220 MW of demand at bus 3

res = solve_congestion(net, "peak")
print(res.summary())
print("LMPs ($/MWh):", res.lmps)

plan = solve_expansion(net)
print(plan.summary())
```

Output:

```
[peak] status=optimal  total_cost=$12,740
  binding lines: L13
LMPs ($/MWh): {1: 20.0, 2: 50.0, 3: 80.0}
status=optimal  objective=$100,066,400
  build: L13b ($20.0M capital)
```

The direct line `L13` is the binding constraint, so the cheap generator at bus 1
can't fully reach the load — the price separates from $20/MWh at bus 1 to $80/MWh
at bus 3, where the peaker has to run. The expansion model then weighs the cost of
reinforcing that corridor against the congestion it relieves, and recommends
building the $20M line.

Field order is `Line(id, from_bus, to_bus, reactance_pu, capacity_mw, status,
length_km[, cost_million_usd])` and `Generator(id, bus_id, name, capacity_mw,
marginal_cost_per_mwh, fuel_type)`.

## Quickstart from a source checkout

To run the bundled 9-bus sample system or the Streamlit app, use a source
checkout (the `examples/` and `app.py` files aren't shipped in
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
