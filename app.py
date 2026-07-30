"""
Streamlit front end for gridplan. Run with:
    streamlit run app.py
"""
import os
import tempfile

import streamlit as st
import pandas as pd

from gridplan import Network, solve_congestion, summarize_congestion, solve_expansion
from gridplan.interactive import build_interactive_topology

DATA_DIR = os.path.join(os.path.dirname(__file__), "examples", "sample_data")

st.set_page_config(page_title="Grid Planning Toolkit", layout="wide")


@st.cache_data
def get_network(data_dir):
    return Network.from_csv(data_dir)


def render_topology_html(_net, **kwargs):
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        path = tmp.name
    build_interactive_topology(_net, path, **kwargs)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    os.remove(path)
    return html


st.title("⚡ Transmission Grid Planning Toolkit")

with st.sidebar:
    st.header("Network data")
    st.caption(f"Currently loaded from:\n`{DATA_DIR}`")
    uploaded = st.file_uploader(
        "Optionally replace with your own data (upload all 4: "
        "buses.csv, lines.csv, generators.csv, loads.csv)",
        type="csv", accept_multiple_files=True,
    )
    active_data_dir = DATA_DIR
    if uploaded and len(uploaded) == 4:
        tmp_dir = tempfile.mkdtemp()
        names = {f.name for f in uploaded}
        required = {"buses.csv", "lines.csv", "generators.csv", "loads.csv"}
        if names == required:
            for f in uploaded:
                with open(os.path.join(tmp_dir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            active_data_dir = tmp_dir
            st.success("Using your uploaded network data.")
        else:
            st.error(f"File names must be exactly: {sorted(required)}")
    elif uploaded:
        st.warning("Upload all 4 files (buses, lines, generators, loads) to switch data.")

net = get_network(active_data_dir)

tab_overview, tab_congestion, tab_expansion, tab_gnn = st.tabs(
    ["Network Overview", "Congestion Analysis", "Expansion Planning", "GNN Power Flow"]
)

# ---------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Buses", len(net.buses))
    col2.metric("Existing lines", len(net.existing_lines()))
    col3.metric("Candidate lines", len(net.candidate_lines()))

    st.subheader("Network diagram")
    st.caption("Drag nodes to rearrange · scroll to zoom · hover for details")
    html = render_topology_html(net, title="Network Topology")
    st.components.v1.html(html, height=700, scrolling=True)

    st.subheader("Raw data")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Buses**")
        st.dataframe(pd.DataFrame([vars(b) for b in net.buses.values()]),
                     hide_index=True, use_container_width=True)
    with d2:
        st.markdown("**Lines**")
        st.dataframe(pd.DataFrame([vars(l) for l in net.lines.values()]),
                     hide_index=True, use_container_width=True)
    with d3:
        st.markdown("**Generators**")
        st.dataframe(pd.DataFrame([vars(g) for g in net.generators.values()]),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_congestion:
    st.subheader("Single-snapshot congestion analysis")
    snapshot = st.selectbox("Snapshot", net.snapshots, index=0)

    if st.button("Run DC-OPF for this snapshot", type="primary"):
        with st.spinner("Solving..."):
            result = solve_congestion(net, snapshot)
        st.session_state["last_result"] = result
        st.session_state["last_snapshot"] = snapshot

    if st.session_state.get("last_snapshot") == snapshot and "last_result" in st.session_state:
        result = st.session_state["last_result"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", result.status)
        c2.metric("Total cost", f"${result.total_cost:,.0f}")
        c3.metric("Unserved load", f"{result.total_unserved_mw:.1f} MW")

        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.markdown("**Line loading**")
            rows = [{"line": lid, "flow_mw": lf.flow_mw, "capacity_mw": lf.capacity_mw,
                      "loading_pct": lf.loading_pct, "binding": lf.binding}
                     for lid, lf in result.line_flows.items()]
            df = pd.DataFrame(rows).sort_values("loading_pct", ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
        with cc2:
            st.markdown("**Locational marginal prices ($/MWh)**")
            st.dataframe(pd.DataFrame(
                [{"bus": b, "name": net.buses[b].name, "lmp": v}
                 for b, v in result.lmps.items()]),
                hide_index=True, use_container_width=True)

        st.markdown("**Network diagram (colored by line loading)**")
        congestion = {lid: lf.loading_pct for lid, lf in result.line_flows.items()}
        html = render_topology_html(
            net, congestion=congestion, lmps=result.lmps,
            include_candidates=False,
            title=f"Line Loading — {snapshot}")
        st.components.v1.html(html, height=700, scrolling=True)

    st.divider()
    st.subheader("Congestion summary across all snapshots")
    st.caption("Solves every snapshot to find which lines bind most often — slower, run on demand.")
    if st.button("Run full summary"):
        with st.spinner(f"Solving {len(net.snapshots)} snapshots..."):
            summary = summarize_congestion(net)
        st.metric("Total unserved energy (sample)", f"{summary.total_unserved_mwh} MWh")
        rows = [{"line": l,
                  "binding_count": summary.binding_count_by_line[l],
                  "max_loading_pct": summary.max_loading_pct_by_line[l]}
                 for l in summary.binding_count_by_line]
        st.dataframe(pd.DataFrame(rows).sort_values("binding_count", ascending=False),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_expansion:
    st.subheader("Transmission expansion planning")
    st.caption("Decides which candidate lines are worth building, jointly across all snapshots.")

    if not net.candidate_lines():
        st.info("No candidate lines in this network's lines.csv — nothing to plan.")
    else:
        if st.button("Run expansion planning", type="primary"):
            with st.spinner("Solving MILP..."):
                plan = solve_expansion(net)
            st.session_state["plan"] = plan

        if "plan" in st.session_state:
            plan = st.session_state["plan"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", plan.status)
            c2.metric("Lines recommended", len(plan.lines_to_build) or "None")
            c3.metric("Total capital cost", f"${plan.total_capital_cost_million_usd}M")

            st.markdown("**Per-candidate decision**")
            rows = []
            for l, built in plan.build_decision.items():
                cand = net.candidate_lines()[l]
                rows.append({
                    "line": l, "from": cand.from_bus, "to": cand.to_bus,
                    "cost_million_usd": cand.cost_million_usd,
                    "capacity_mw": cand.capacity_mw,
                    "decision": "BUILD" if built else "do not build",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.markdown("**Network diagram (recommended builds in green)**")
            html = render_topology_html(
                net, build_decision=plan.build_decision,
                title="Recommended Expansion Plan")
            st.components.v1.html(html, height=700, scrolling=True)

# ---------------------------------------------------------------------------
with tab_gnn:
    st.subheader("GNN power flow approximation")
    st.caption(
        "Trains a graph neural network to approximate DC power flow from grid topology. "
        "At this network's small size, the exact linear solve is faster and exact — this "
        "is a demonstration of the technique, not a practical speedup. Requires `torch`."
    )
    try:
        import torch  # noqa: F401
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        st.warning("`torch` isn't installed. Run `pip install gridplan[gnn]` to enable this tab.")
    else:
        n_samples = st.slider("Training scenarios", 500, 8000, 4000, step=500)
        epochs = st.slider("Epochs", 50, 1000, 300, step=50)

        if st.button("Train GNN", type="primary"):
            from gridplan.gnn import train_and_evaluate
            with st.spinner("Generating training data and training..."):
                gnn_result = train_and_evaluate(net, n_samples=n_samples, epochs=epochs,
                                                  out_dir=None)
            st.session_state["gnn_result"] = gnn_result

        if "gnn_result" in st.session_state:
            r = st.session_state["gnn_result"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Angle MAE", f"{r['angle_mae']:.4f} rad")
            c2.metric("Flow MAE", f"{r['flow_mae_mw']:.1f} MW")
            c3.metric("Exact solve vs GNN (2000 scenarios)",
                      f"{r['exact_time_ms']:.1f} ms vs {r['gnn_time_ms']:.1f} ms")
            if r["exact_time_ms"] < r["gnn_time_ms"]:
                st.info("Exact solve wins at this network size, as expected — see the caption above.")
