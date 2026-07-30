"""Core data model: Bus, Line, Generator, and the Network container."""
import csv
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Bus:
    bus_id: int
    name: str
    zone: str
    base_kv: float


@dataclass
class Line:
    line_id: str
    from_bus: int
    to_bus: int
    reactance_pu: float
    capacity_mw: float
    status: str  # "existing" or "candidate"
    length_km: float
    cost_million_usd: Optional[float] = None


@dataclass
class Generator:
    gen_id: str
    bus_id: int
    name: str
    capacity_mw: float
    marginal_cost_per_mwh: float
    fuel_type: str


@dataclass
class Network:
    """
    A transmission network: buses, lines (existing + candidate), generators,
    and per-snapshot loads. Build one with `Network.from_csv(data_dir)`.
    """
    buses: dict = field(default_factory=dict)       # bus_id -> Bus
    lines: dict = field(default_factory=dict)        # line_id -> Line
    generators: dict = field(default_factory=dict)   # gen_id -> Generator
    loads: dict = field(default_factory=dict)        # snapshot -> {bus_id: demand_mw}
    snapshots: list = field(default_factory=list)

    def existing_lines(self):
        return {lid: l for lid, l in self.lines.items() if l.status == "existing"}

    def candidate_lines(self):
        return {lid: l for lid, l in self.lines.items() if l.status == "candidate"}

    def total_demand(self, snapshot):
        return sum(self.loads.get(snapshot, {}).values())

    def total_capacity(self):
        return sum(g.capacity_mw for g in self.generators.values())

    @classmethod
    def from_csv(cls, data_dir):
        """
        Load a network from a directory containing buses.csv, lines.csv,
        generators.csv, and loads.csv. See the README for the schema.
        """
        net = cls()

        with open(os.path.join(data_dir, "buses.csv")) as f:
            for row in csv.DictReader(f):
                bus_id = int(row["bus_id"])
                net.buses[bus_id] = Bus(
                    bus_id=bus_id,
                    name=row["name"],
                    zone=row["zone"],
                    base_kv=float(row["base_kv"]),
                )

        with open(os.path.join(data_dir, "lines.csv")) as f:
            for row in csv.DictReader(f):
                cost = row.get("cost_million_usd", "")
                net.lines[row["line_id"]] = Line(
                    line_id=row["line_id"],
                    from_bus=int(row["from_bus"]),
                    to_bus=int(row["to_bus"]),
                    reactance_pu=float(row["reactance_pu"]),
                    capacity_mw=float(row["capacity_mw"]),
                    status=row["status"],
                    length_km=float(row["length_km"]),
                    cost_million_usd=float(cost) if cost not in ("", None) else None,
                )

        with open(os.path.join(data_dir, "generators.csv")) as f:
            for row in csv.DictReader(f):
                net.generators[row["gen_id"]] = Generator(
                    gen_id=row["gen_id"],
                    bus_id=int(row["bus_id"]),
                    name=row["name"],
                    capacity_mw=float(row["capacity_mw"]),
                    marginal_cost_per_mwh=float(row["marginal_cost_per_mwh"]),
                    fuel_type=row["fuel_type"],
                )

        with open(os.path.join(data_dir, "loads.csv")) as f:
            for row in csv.DictReader(f):
                snap = row["snapshot"]
                bus_id = int(row["bus_id"])
                demand = float(row["demand_mw"])
                net.loads.setdefault(snap, {})[bus_id] = demand
                if snap not in net.snapshots:
                    net.snapshots.append(snap)

        return net


def load_network(data_dir):
    """Deprecated alias for Network.from_csv - kept for backward compatibility."""
    return Network.from_csv(data_dir)
