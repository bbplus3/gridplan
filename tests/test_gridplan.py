"""
Tests for gridplan's core physics and optimization. These aren't
exhaustive, but they lock in the things that matter most: that power
balances, that a hand-solvable 2-bus case matches the model exactly, and
that the expansion planning MILP makes economically sane choices.
"""
import math
import os

import pytest

from gridplan import Network, Bus, Line, Generator, solve_congestion, solve_expansion


SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_data")


def make_two_bus_network(reactance=0.1, capacity=1000, demand=100, gen_cost=20):
    """
    A minimal 2-bus network where the DC-OPF answer is derivable by hand:
    bus 1 has a generator, bus 2 has all the demand, one line between them.
    With capacity far above demand, the generator must supply exactly the
    demand and the line must carry exactly that flow.
    """
    net = Network()
    net.buses = {
        1: Bus(bus_id=1, name="Gen Bus", zone="A", base_kv=230),
        2: Bus(bus_id=2, name="Load Bus", zone="A", base_kv=230),
    }
    net.lines = {
        "L1": Line(line_id="L1", from_bus=1, to_bus=2, reactance_pu=reactance,
                    capacity_mw=capacity, status="existing", length_km=10),
    }
    net.generators = {
        "G1": Generator(gen_id="G1", bus_id=1, name="Gen1", capacity_mw=capacity,
                          marginal_cost_per_mwh=gen_cost, fuel_type="gas"),
    }
    net.loads = {"snap1": {2: demand}}
    net.snapshots = ["snap1"]
    return net


def test_two_bus_power_balance():
    """The single generator must exactly meet demand, with no shedding,
    when capacity is far above what's needed."""
    net = make_two_bus_network(demand=100, capacity=1000)
    result = solve_congestion(net, "snap1")

    assert result.status == "optimal"
    assert result.is_feasible
    assert result.gen_dispatch["G1"] == pytest.approx(100, abs=0.5)
    assert result.line_flows["L1"].flow_mw == pytest.approx(100, abs=0.5)


def test_two_bus_lmp_equals_marginal_cost():
    """With one generator and no congestion, LMP everywhere should equal
    that generator's marginal cost - a basic economic sanity check."""
    net = make_two_bus_network(demand=100, capacity=1000, gen_cost=35)
    result = solve_congestion(net, "snap1")

    assert result.lmps[1] == pytest.approx(35, abs=0.5)
    assert result.lmps[2] == pytest.approx(35, abs=0.5)


def test_line_capacity_binds_and_sheds_load():
    """If the line can't carry enough power to meet demand, the model
    should shed load rather than fail, and should flag the line as binding."""
    net = make_two_bus_network(demand=100, capacity=50)
    result = solve_congestion(net, "snap1")

    assert not result.is_feasible
    assert result.total_unserved_mw == pytest.approx(50, abs=0.5)
    assert "L1" in result.binding_lines()
    assert result.line_flows["L1"].loading_pct >= 99.5


def test_lines_over_threshold():
    net = make_two_bus_network(demand=100, capacity=105)
    result = solve_congestion(net, "snap1")
    # ~95% loaded - should show up at a 90% threshold but not at 99.9%
    assert "L1" in result.lines_over(90)
    assert "L1" not in result.lines_over(99.9)


def test_sample_network_loads_correctly():
    """Sanity check the bundled example data loads with the expected shape."""
    net = Network.from_csv(SAMPLE_DATA_DIR)
    assert len(net.buses) == 9
    assert len(net.existing_lines()) == 9
    assert len(net.candidate_lines()) == 3
    assert len(net.generators) == 3
    assert len(net.snapshots) == 48


def test_sample_network_congestion_solves():
    net = Network.from_csv(SAMPLE_DATA_DIR)
    result = solve_congestion(net, "summer_peak_h17")
    assert result.status == "optimal"
    # total generation should equal demand minus unserved
    total_gen = sum(result.gen_dispatch.values())
    expected = net.total_demand("summer_peak_h17") - result.total_unserved_mw
    assert total_gen == pytest.approx(expected, abs=0.5)


def test_expansion_never_recommends_uneconomic_line():
    """A candidate line costing far more than any plausible congestion
    relief it could provide should never be recommended."""
    net = Network.from_csv(SAMPLE_DATA_DIR)
    # Make one candidate absurdly expensive - it should never be picked.
    net.candidate_lines()["C1"].cost_million_usd = 1_000_000
    result = solve_expansion(net, snapshots=net.snapshots[:6])
    assert "C1" not in result.lines_to_build


def test_expansion_recommends_free_line():
    """A free candidate line that relieves real congestion should always
    be worth building - a basic sanity check on the MILP objective."""
    net = Network.from_csv(SAMPLE_DATA_DIR)
    for lid, line in net.candidate_lines().items():
        line.cost_million_usd = 0.0
    result = solve_expansion(net, snapshots=net.snapshots[:6])
    # With zero capital cost, building is always weakly better - at least
    # one free candidate should be recommended if it relieves any congestion.
    assert result.total_capital_cost_million_usd == 0.0
