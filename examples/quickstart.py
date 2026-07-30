"""
Quickstart: the core gridplan API in a dozen lines.

    pip install -e .
    python examples/quickstart.py
"""
import os

from gridplan import Network, solve_congestion, summarize_congestion, solve_expansion

DATA_DIR = os.path.join(os.path.dirname(__file__), "sample_data")


def main():
    net = Network.from_csv(DATA_DIR)
    print(f"Loaded {len(net.buses)} buses, {len(net.existing_lines())} existing lines, "
          f"{len(net.candidate_lines())} candidate lines, {len(net.snapshots)} snapshots\n")

    # --- Congestion analysis for one snapshot ---
    result = solve_congestion(net, "summer_peak_h17")
    print(result.summary())
    print("Lines over 70% loaded:", result.lines_over(70))
    print("LMPs ($/MWh):", result.lmps, "\n")

    # --- Which lines are congested most often across all snapshots? ---
    summary = summarize_congestion(net)
    print("Most-often-binding lines:", summary.worst_lines(top_n=3), "\n")

    # --- Should we build any of the candidate lines? ---
    plan = solve_expansion(net)
    print(plan.summary())


if __name__ == "__main__":
    main()
