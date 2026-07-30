"""
Visualizes the network topology and (optionally) congestion results from
dcopf.py, and the build decision from tep.py. Uses networkx for layout
and matplotlib for rendering - both are general-purpose, not grid-specific,
so this is easy to adapt if you swap in your own network.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from .network import Network

ZONE_COLORS = {"North": "#4C72B0", "Central": "#55A868", "South": "#C44E52"}


def _build_graph(net, include_candidates=False):
    G = nx.Graph()
    for b in net.buses.values():
        G.add_node(b.bus_id, name=b.name, zone=b.zone)
    line_set = net.existing_lines()
    if include_candidates:
        line_set = {**line_set, **net.candidate_lines()}
    for lid, ln in line_set.items():
        G.add_edge(ln.from_bus, ln.to_bus, line_id=lid, status=ln.status,
                   capacity=ln.capacity_mw)
    return G


def plot_topology(net, out_path, include_candidates=True, congestion=None,
                   build_decision=None, title="Transmission Network"):
    """
    congestion: optional dict of line_id -> loading_pct (e.g. built from
        {lid: lf.loading_pct for lid, lf in result.line_flows.items()}
        where result = congestion.solve_congestion(...))
        used to color existing lines by how loaded they are.
    build_decision: optional dict of candidate line_id -> bool, from
        expansion.solve_expansion(...).build_decision, used to highlight
        recommended new lines in green vs. rejected ones as dashed grey.
    """
    G = _build_graph(net, include_candidates=include_candidates)
    pos = nx.spring_layout(G, seed=7, k=1.3)

    fig, ax = plt.subplots(figsize=(10, 7))

    node_colors = [ZONE_COLORS.get(net.buses[n].zone, "#888") for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=650,
                            edgecolors="white", linewidths=1.5, ax=ax)
    labels = {n: net.buses[n].name.replace("_", "\n") for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    existing_edges, candidate_built, candidate_rejected = [], [], []
    for u, v, data in G.edges(data=True):
        if data["status"] == "existing":
            existing_edges.append((u, v))
        elif build_decision and build_decision.get(data["line_id"]):
            candidate_built.append((u, v))
        else:
            candidate_rejected.append((u, v))

    if congestion:
        edge_colors, edge_widths = [], []
        for u, v in existing_edges:
            lid = G[u][v]["line_id"]
            pct = congestion.get(lid, 0)
            edge_widths.append(1.5 + pct / 100 * 4)
            if pct >= 95:
                edge_colors.append("#B22222")
            elif pct >= 70:
                edge_colors.append("#E08214")
            else:
                edge_colors.append("#4C4C4C")
        nx.draw_networkx_edges(G, pos, edgelist=existing_edges, edge_color=edge_colors,
                                width=edge_widths, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, edgelist=existing_edges, edge_color="#4C4C4C",
                                width=2, ax=ax)

    if candidate_built:
        nx.draw_networkx_edges(G, pos, edgelist=candidate_built, edge_color="#2CA02C",
                                width=3, style="solid", ax=ax)
    if candidate_rejected:
        nx.draw_networkx_edges(G, pos, edgelist=candidate_rejected, edge_color="#999999",
                                width=1.5, style="dashed", ax=ax)

    edge_labels = {(u, v): d["line_id"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    legend_items = [mpatches.Patch(color=c, label=z) for z, c in ZONE_COLORS.items()]
    if congestion:
        legend_items += [
            mpatches.Patch(color="#B22222", label=">=95% loaded"),
            mpatches.Patch(color="#E08214", label="70-95% loaded"),
            mpatches.Patch(color="#4C4C4C", label="<70% loaded"),
        ]
    if include_candidates:
        legend_items.append(mpatches.Patch(color="#2CA02C", label="Recommended new line"))
        legend_items.append(mpatches.Patch(color="#999999", label="Candidate (not built)"))
    ax.legend(handles=legend_items, loc="upper left", fontsize=7, framealpha=0.9)

    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import os
    from .congestion import solve_congestion
    from .expansion import solve_expansion

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             "examples", "sample_data")
    net = Network.from_csv(data_dir)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Plain topology
    plot_topology(net, os.path.join(out_dir, "topology.png"),
                  title="Network Topology (existing + candidate lines)")

    # 2) Congestion snapshot
    r = solve_congestion(net, "summer_peak_h17")
    congestion = {lid: lf.loading_pct for lid, lf in r.line_flows.items()}
    plot_topology(net, os.path.join(out_dir, "congestion_peak.png"),
                  congestion=congestion,
                  title="Line Loading at Summer Peak (17:00)")

    # 3) Expansion plan recommendation
    exp_result = solve_expansion(net)
    plot_topology(net, os.path.join(out_dir, "expansion_plan.png"),
                  congestion=congestion,
                  build_decision=exp_result.build_decision,
                  title="Recommended Expansion Plan")

    print(f"Saved figures to {out_dir}/")
