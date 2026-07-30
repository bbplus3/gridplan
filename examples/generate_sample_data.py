"""
Generates sample CSV data into examples/sample_data/ - a small fictional
9-bus utility system. Run this if you want to regenerate the bundled
example data (e.g. after tweaking the generation logic below).
"""
import os
import csv
import random

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(DATA_DIR, exist_ok=True)

buses = [
    (1, "North_Substation_A", "North", 230),
    (2, "North_Substation_B", "North", 230),
    (3, "North_Wind_Interconnect", "North", 230),
    (4, "Central_Hub", "Central", 230),
    (5, "Central_Load_A", "Central", 115),
    (6, "Central_Load_B", "Central", 115),
    (7, "South_Substation_A", "South", 230),
    (8, "South_Load_A", "South", 115),
    (9, "South_Peaker_Plant", "South", 230),
]

with open(os.path.join(DATA_DIR, "buses.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["bus_id", "name", "zone", "base_kv"])
    w.writerows(buses)

lines = [
    ("L1", 1, 2, 0.05, 250, "existing", 40, ""),
    ("L2", 1, 4, 0.10, 200, "existing", 90, ""),
    ("L3", 2, 4, 0.09, 200, "existing", 85, ""),
    ("L4", 3, 1, 0.04, 180, "existing", 30, ""),
    ("L5", 4, 5, 0.06, 150, "existing", 25, ""),
    ("L6", 4, 6, 0.07, 150, "existing", 35, ""),
    ("L7", 4, 7, 0.12, 220, "existing", 110, ""),
    ("L8", 7, 8, 0.05, 140, "existing", 20, ""),
    ("L9", 7, 9, 0.04, 160, "existing", 15, ""),
    ("C1", 2, 7, 0.11, 200, "candidate", 105, 42.0),
    ("C2", 3, 4, 0.08, 200, "candidate", 70, 31.5),
    ("C3", 6, 8, 0.09, 150, "candidate", 60, 27.0),
]

with open(os.path.join(DATA_DIR, "lines.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["line_id", "from_bus", "to_bus", "reactance_pu", "capacity_mw",
                "status", "length_km", "cost_million_usd"])
    w.writerows(lines)

generators = [
    ("G1", 1, "North_Baseload_Gas", 400, 28.0, "gas"),
    ("G2", 3, "North_Wind_Farm", 180, 0.0, "wind"),
    ("G3", 9, "South_Peaker", 220, 65.0, "gas_peaker"),
]

with open(os.path.join(DATA_DIR, "generators.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gen_id", "bus_id", "name", "capacity_mw", "marginal_cost_per_mwh", "fuel_type"])
    w.writerows(generators)

load_buses = {2: 90, 5: 130, 6: 110, 8: 150}
hourly_shape = [0.55, 0.52, 0.50, 0.50, 0.52, 0.58, 0.68, 0.78, 0.85, 0.88,
                0.90, 0.91, 0.90, 0.89, 0.90, 0.93, 0.98, 1.00, 0.97, 0.90,
                0.82, 0.72, 0.63, 0.58]

rows = []
for season, scale in [("summer_peak", 1.0), ("shoulder", 0.75)]:
    for hour in range(24):
        shape = hourly_shape[hour] * scale
        for bus_id, peak_mw in load_buses.items():
            noise = 1.0 + random.uniform(-0.02, 0.02)
            demand = round(peak_mw * shape * noise, 1)
            rows.append((f"{season}_h{hour:02d}", bus_id, hour, season, demand))

with open(os.path.join(DATA_DIR, "loads.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["snapshot", "bus_id", "hour", "season", "demand_mw"])
    w.writerows(rows)

print(f"Sample data written to {DATA_DIR}/")
