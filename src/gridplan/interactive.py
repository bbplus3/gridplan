"""
Interactive HTML versions of the network diagrams - open the .html output
in any browser and drag nodes around, zoom, pan, and hover over
buses/lines for details. Uses pyvis (a thin wrapper around the vis.js
JavaScript network library), so no server is needed - just double-click
the generated .html file.

This mirrors visualize.py's three diagrams (topology, congestion,
expansion plan) but as draggable HTML instead of static PNGs.
"""
import os
from pyvis.network import Network as VisNetwork

from .network import Network

ZONE_COLORS = {"North": "#4C72B0", "Central": "#55A868", "South": "#C44E52"}


def build_interactive_topology(net, out_path, include_candidates=True,
                                congestion=None, lmps=None, build_decision=None,
                                title="Transmission Network"):
    """
    congestion: optional dict of line_id -> loading_pct, e.g. built from
        {lid: lf.loading_pct for lid, lf in result.line_flows.items()}
        where result = congestion.solve_congestion(...)
    lmps: optional dict of bus_id -> $/MWh, from result.lmps (same result object),
        shown in each bus's hover tooltip if provided
    build_decision: optional dict of candidate line_id -> bool, from
        expansion.solve_expansion(...).build_decision
    """
    net_vis = VisNetwork(height="750px", width="100%", bgcolor="#ffffff",
                       font_color="#222222", directed=False, notebook=False,
                       cdn_resources="in_line")

    # Physics settles the initial layout nicely, then the user can drag
    # freely; "stabilization" just controls the initial settle-down.
    net_vis.set_options("""
    {
      "physics": {
        "stabilization": {"iterations": 200},
        "barnesHut": {"gravitationalConstant": -12000, "springLength": 180, "springConstant": 0.04}
      },
      "interaction": {"hover": true, "tooltipDelay": 100, "dragNodes": true}
    }
    """)

    for bus_id, b in net.buses.items():
        tooltip = f"{b.name}\nZone: {b.zone}\n{b.base_kv} kV"
        if lmps and lmps.get(bus_id) is not None:
            tooltip += f"\nLMP: ${lmps[bus_id]}/MWh"
        net_vis.add_node(
            bus_id,
            label=b.name.replace("_", "\n"),
            title=tooltip,
            color=ZONE_COLORS.get(b.zone, "#888888"),
            shape="dot",
            size=22,
        )

    line_set = dict(net.existing_lines())
    if include_candidates:
        line_set.update(net.candidate_lines())

    for lid, ln in line_set.items():
        edge_kwargs = dict(title=f"{lid}: {ln.from_bus}->{ln.to_bus}\nCapacity: {ln.capacity_mw} MW")

        if ln.status == "existing":
            if congestion and lid in congestion:
                pct = congestion[lid]
                edge_kwargs["title"] += f"\nLoading: {pct}%"
                if pct >= 95:
                    color, width = "#B22222", 5
                elif pct >= 70:
                    color, width = "#E08214", 3.5
                else:
                    color, width = "#4C4C4C", 2
            else:
                color, width = "#4C4C4C", 2
            edge_kwargs.update(color=color, width=width)
        else:  # candidate line
            edge_kwargs["title"] += f"\nCandidate - ${ln.cost_million_usd}M"
            if build_decision and build_decision.get(lid):
                edge_kwargs.update(color="#2CA02C", width=4, dashes=False)
                edge_kwargs["title"] += "\n>> RECOMMENDED TO BUILD"
            else:
                edge_kwargs.update(color="#999999", width=1.5, dashes=True)
                edge_kwargs["title"] += "\n(not recommended)"

        net_vis.add_edge(ln.from_bus, ln.to_bus, **edge_kwargs)

    # Use generate_html() + our own UTF-8 write, rather than pyvis's
    # write_html(), which on Windows can crash with UnicodeEncodeError
    # (it opens the file using the OS default encoding, e.g. cp1252,
    # which can't represent some characters in the embedded JS/CSS).
    html = net_vis.generate_html(notebook=False)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Inject a title heading + simple legend at the top of the page
    with open(out_path, "r", encoding="utf-8") as f:
        html = f.read()
    legend_items = "".join(
        f'<span style="margin-right:16px;"><span style="display:inline-block;width:12px;height:12px;'
        f'background:{c};border-radius:50%;margin-right:4px;"></span>{z}</span>'
        for z, c in ZONE_COLORS.items()
    )
    if congestion:
        legend_items += (
            '<span style="margin-right:16px;"><span style="display:inline-block;width:20px;height:3px;'
            'background:#B22222;margin-right:4px;"></span>&gt;=95% loaded</span>'
            '<span style="margin-right:16px;"><span style="display:inline-block;width:20px;height:3px;'
            'background:#E08214;margin-right:4px;"></span>70-95% loaded</span>'
        )
    if include_candidates:
        legend_items += (
            '<span style="margin-right:16px;"><span style="display:inline-block;width:20px;height:3px;'
            'background:#2CA02C;margin-right:4px;"></span>Recommended new line</span>'
        )
    header = (
        f'<div style="font-family:sans-serif;padding:12px 16px;border-bottom:1px solid #ddd;">'
        f'<h2 style="margin:0 0 8px 0;">{title}</h2>'
        f'<div style="font-size:13px;color:#444;">{legend_items}</div>'
        f'<div style="font-size:12px;color:#888;margin-top:6px;">Drag nodes to rearrange &middot; '
        f'scroll to zoom &middot; hover for details</div></div>'
    )
    html = html.replace("<body>", f"<body>{header}", 1)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


if __name__ == "__main__":
    from .congestion import solve_congestion
    from .expansion import solve_expansion

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             "examples", "sample_data")
    net = Network.from_csv(data_dir)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Plain interactive topology
    build_interactive_topology(
        net, os.path.join(out_dir, "topology_interactive.html"),
        title="Network Topology (drag nodes to explore)")

    # 2) Congestion snapshot, interactive
    r = solve_congestion(net, "summer_peak_h17")
    congestion = {lid: lf.loading_pct for lid, lf in r.line_flows.items()}
    build_interactive_topology(
        net, os.path.join(out_dir, "congestion_peak_interactive.html"),
        congestion=congestion, lmps=r.lmps,
        title="Line Loading at Summer Peak (17:00)")

    # 3) Expansion plan, interactive
    exp_result = solve_expansion(net)
    build_interactive_topology(
        net, os.path.join(out_dir, "expansion_plan_interactive.html"),
        congestion=congestion, build_decision=exp_result.build_decision,
        title="Recommended Expansion Plan")

    print(f"Saved interactive HTML diagrams to {out_dir}/")
    print("Open any .html file in a browser - nodes are draggable, edges show details on hover.")
