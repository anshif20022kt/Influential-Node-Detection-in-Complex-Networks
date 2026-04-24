from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from saasp_pipeline import (
    available_local_datasets,
    build_ml_feature_matrix,
    compute_classical_centralities,
    compute_sir_influence_scores,
    create_sir_labels,
    evaluate_model,
    load_graph,
    run_saasp_on_graph,
    summarize_graph,
    train_models,
    validate_rankings,
)


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_NODE_LIMIT = 300
HISTORY_FILE = DATA_DIR / ".analysis_history.pkl"


def show_user_error(message: str, exc: Exception | None = None) -> None:
    st.error(message)
    st.caption("Please adjust the dataset or settings and try again.")
    if exc is not None:
        st.caption(f"Details: {type(exc).__name__}: {exc}")


def build_history_label(dataset_label: str, node_count: int, run_ml_pipeline: bool) -> str:
    timestamp = datetime.now().strftime("%H:%M:%S")
    ml_suffix = " + ML" if run_ml_pipeline else ""
    return f"{timestamp} | {dataset_label} | {node_count} nodes{ml_suffix}"


def load_persisted_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("rb") as handle:
            history = pickle.load(handle)
        return history if isinstance(history, list) else []
    except Exception:
        return []


def save_persisted_history(history: list) -> None:
    with HISTORY_FILE.open("wb") as handle:
        pickle.dump(history, handle)


def load_uploaded_graph(uploaded_file) -> nx.Graph:
    payload = uploaded_file.getvalue().decode("utf-8")
    return nx.read_edgelist(StringIO(payload), nodetype=int)


def draw_degree_histogram(graph: nx.Graph):
    fig, ax = plt.subplots()
    ax.hist([degree for _, degree in graph.degree()], bins=30, color="#2A6F97", edgecolor="white")
    ax.set_title("Degree Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    return fig


def draw_network_preview(graph: nx.Graph, highlight_nodes=None, limit: int = 200):
    sampled_nodes = list(graph.nodes())[:limit]
    preview = graph.subgraph(sampled_nodes).copy()
    if highlight_nodes is None:
        highlight_nodes = set()
    else:
        highlight_nodes = set(highlight_nodes)

    colors = ["#D1495B" if node in highlight_nodes else "#7FB069" for node in preview.nodes()]
    fig, ax = plt.subplots(figsize=(8, 6))
    positions = nx.spring_layout(preview, seed=42)
    nx.draw(preview, positions, node_size=60, node_color=colors, edge_color="#BFC0C0", ax=ax, with_labels=False)
    ax.set_title(f"Network Preview ({len(sampled_nodes)} nodes)")
    ax.axis("off")
    return fig


def build_interactive_network_figure(
    graph: nx.Graph,
    node_metrics: pd.DataFrame,
    selected_node: int | None = None,
    limit: int = 150,
):
    sampled_nodes = list(graph.nodes())[:limit]
    preview = graph.subgraph(sampled_nodes).copy()
    positions = nx.spring_layout(preview, seed=42)
    metrics_indexed = node_metrics.set_index("node")

    edge_x = []
    edge_y = []
    for source, target in preview.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 0.6, "color": "#BFC0C0"},
        hoverinfo="skip",
        showlegend=False,
    )

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    customdata = []

    for node in preview.nodes():
        x, y = positions[node]
        saasp_score = float(metrics_indexed.loc[node, "saasp_score"]) if "saasp_score" in metrics_indexed.columns else 0.0
        degree_score = float(metrics_indexed.loc[node, "degree"]) if "degree" in metrics_indexed.columns else 0.0
        node_x.append(x)
        node_y.append(y)
        node_color.append(saasp_score)
        node_size.append(20 if node == selected_node else 10 + (saasp_score * 16))
        node_text.append(
            "<br>".join(
                [
                    f"Node: {node}",
                    f"SAASP: {saasp_score:.4f}",
                    f"Degree: {degree_score:.4f}",
                ]
            )
        )
        customdata.append([node])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=node_text,
        customdata=customdata,
        hovertemplate="%{text}<extra></extra>",
        marker={
            "size": node_size,
            "color": node_color,
            "colorscale": "YlOrRd",
            "showscale": True,
            "colorbar": {"title": "SAASP"},
            "line": {"width": 1.2, "color": "#1F2937"},
        },
        showlegend=False,
    )

    figure = go.Figure(data=[edge_trace, node_trace])
    figure.update_layout(
        height=560,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title=f"Interactive Network View ({preview.number_of_nodes()} nodes)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return figure


def extract_selected_node(selection_state) -> int | None:
    if not selection_state:
        return None
    selection = selection_state.get("selection", {})
    points = selection.get("points", [])
    if not points:
        return None
    customdata = points[0].get("customdata")
    if customdata:
        return int(customdata[0])
    point_index = points[0].get("point_index")
    if point_index is not None:
        return int(point_index)
    return None


def render_node_detail_panel(
    selected_node: int,
    node_metrics: pd.DataFrame,
    sir_scores: dict | None = None,
    ml_probabilities: pd.Series | None = None,
    sir_labels: dict | None = None,
) -> None:
    node_row = node_metrics.set_index("node").loc[selected_node]
    st.subheader(f"Node {selected_node}")
    col1, col2 = st.columns(2)
    col1.metric("SAASP", f"{node_row['saasp_score']:.4f}")
    col2.metric("Degree", f"{node_row['degree']:.4f}")

    detail_rows = [
        ("Betweenness", node_row["betweenness"]),
        ("Closeness", node_row["closeness"]),
        ("PageRank", node_row["pagerank"]),
        ("Clustering", node_row["clustering"]),
        ("K-shell", node_row["k_shell"]),
        ("Avg Neighbor Degree", node_row["avg_neighbor_degree"]),
    ]

    if sir_scores is not None and selected_node in sir_scores:
        detail_rows.append(("SIR Mean Spread", sir_scores[selected_node]["mean"]))
        detail_rows.append(("SIR Std Dev", sir_scores[selected_node]["std"]))

    if sir_labels is not None and selected_node in sir_labels:
        detail_rows.append(("SIR Label", sir_labels[selected_node]))

    if ml_probabilities is not None and selected_node in ml_probabilities.index:
        detail_rows.append(("ML Influence Probability", ml_probabilities.loc[selected_node]))

    detail_df = pd.DataFrame(detail_rows, columns=["Metric", "Value"])
    detail_df["Value"] = detail_df["Value"].map(lambda value: f"{float(value):.4f}")
    st.dataframe(detail_df, width="stretch", hide_index=True)


def build_analysis_graph(graph: nx.Graph, node_limit: int) -> nx.Graph:
    if graph.number_of_nodes() <= node_limit:
        candidate = graph.copy()
    else:
        sampled_nodes = list(graph.nodes())[:node_limit]
        candidate = graph.subgraph(sampled_nodes).copy()

    if candidate.number_of_nodes() < 2:
        return candidate

    if candidate.is_directed():
        if nx.is_weakly_connected(candidate):
            return candidate
        largest = max(nx.weakly_connected_components(candidate), key=len)
        return candidate.subgraph(largest).copy()

    if nx.is_connected(candidate):
        return candidate

    largest = max(nx.connected_components(candidate), key=len)
    return candidate.subgraph(largest).copy()


def main() -> None:
    st.set_page_config(page_title="SAASP Influence Analyser", layout="wide")
    st.title("SAASP Influence Analyser")
    st.caption("Corrected SAASP pipeline with local subgraph augmentation, SIR-based labels, and ranking validation.")

    local_datasets = available_local_datasets(DATA_DIR)

    st.sidebar.title("Configuration")
    if st.sidebar.button("New scan"):
        st.session_state["selected_node_id"] = None
        st.session_state["active_history_index"] = None
        st.rerun()

    with st.sidebar.form("analysis_form"):
        source_type = st.radio("Dataset Source", ["Local dataset", "Upload file"])
        uploaded_file = None
        selected_path = None

        if source_type == "Local dataset":
            dataset_name = st.selectbox("Choose dataset", list(local_datasets.keys()))
            selected_path = local_datasets[dataset_name]
        else:
            uploaded_file = st.file_uploader("Upload edge list (.txt or .csv)", type=["txt", "csv"])

        hop_depth = st.slider("SAASP hop depth (k)", min_value=1, max_value=3, value=2)
        top_similar_pairs = st.slider("Top similar pairs (M)", min_value=1, max_value=10, value=5)
        sir_runs = st.slider("SIR runs per node", min_value=5, max_value=30, value=5, step=5)
        analysis_node_limit = st.slider("Interactive node limit", min_value=100, max_value=1000, value=300, step=100)
        run_ml_pipeline = st.checkbox("Run ML + SIR pipeline", value=False)
        top_k = st.slider("Top-K for overlap", min_value=5, max_value=25, value=10)
        run_button = st.form_submit_button("Run analysis")

    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = load_persisted_history()

    st.sidebar.divider()
    st.sidebar.subheader("Analysis History")
    history = st.session_state["analysis_history"]
    active_history_index = st.session_state.get("active_history_index")

    if history:
        if active_history_index is None or active_history_index >= len(history):
            active_history_index = len(history) - 1
        history_labels = [entry["label"] for entry in history]
        selected_history_label = st.sidebar.selectbox(
            "Previous runs",
            options=history_labels,
            index=active_history_index,
        )
        active_history_index = history_labels.index(selected_history_label)
        st.session_state["active_history_index"] = active_history_index
        if st.sidebar.button("Clear history"):
            st.session_state["analysis_history"] = []
            st.session_state["active_history_index"] = None
            st.session_state["selected_node_id"] = None
            save_persisted_history([])
            st.rerun()
    else:
        st.sidebar.caption("No saved analysis yet. Run the app once and your results will appear here.")

    if run_button:
        try:
            if uploaded_file is not None:
                graph = load_uploaded_graph(uploaded_file)
                graph.remove_edges_from(nx.selfloop_edges(graph))
                if not nx.is_connected(graph):
                    largest = max(nx.connected_components(graph), key=len)
                    graph = graph.subgraph(largest).copy()
                dataset_label = uploaded_file.name
            elif selected_path is not None:
                graph = load_graph(selected_path)
                dataset_label = Path(selected_path).name
            else:
                st.warning("Please select or upload a dataset first.")
                return
        except Exception as exc:
            show_user_error("The dataset could not be loaded. Please use a valid edge-list file.", exc)
            return

        try:
            full_summary = summarize_graph(graph)
            analysis_graph = build_analysis_graph(graph, analysis_node_limit)
            analysis_summary = summarize_graph(analysis_graph)

            with st.spinner("Computing classical centralities..."):
                centrality_df = compute_classical_centralities(analysis_graph)

            with st.spinner("Running SAASP on the graph..."):
                saasp_scores = run_saasp_on_graph(analysis_graph, k=hop_depth, m=top_similar_pairs)
                centrality_df["saasp_score"] = centrality_df["node"].map(saasp_scores)

            rf_results = None
            threshold = None
            ranking_table = None
            modeling_df = None
            sir_scores = None
            sir_labels = None
            ml_probabilities = None

            if run_ml_pipeline:
                with st.spinner("Running repeated SIR simulations..."):
                    sir_scores = compute_sir_influence_scores(analysis_graph, n_runs=sir_runs)
                    sir_labels, threshold = create_sir_labels(sir_scores, threshold_quantile=0.90)

                with st.spinner("Training ML models..."):
                    x, y, modeling_df = build_ml_feature_matrix(centrality_df, saasp_scores, sir_labels, sir_scores=sir_scores)
                    training_bundle = train_models(x, y)
                    rf_results = evaluate_model(
                        training_bundle["models"]["RandomForest"],
                        training_bundle["x_test"],
                        training_bundle["y_test"],
                    )
                    ml_probabilities = pd.Series(
                        training_bundle["models"]["RandomForest"].predict_proba(x)[:, 1],
                        index=x.index,
                    )
                    ranking_table = validate_rankings(
                        modeling_df,
                        saasp_scores,
                        sir_scores,
                        training_bundle["models"],
                        training_bundle["x_test"],
                        top_k=top_k,
                    )
        except (ImportError, ValueError) as exc:
            show_user_error("The analysis could not be completed with the current dataset.", exc)
            return
        except Exception as exc:
            show_user_error("The graph analysis could not be completed with the current settings.", exc)
            return

        analysis_state = {
            "dataset_label": dataset_label,
            "graph": graph,
            "analysis_graph": analysis_graph,
            "full_summary": full_summary,
            "analysis_summary": analysis_summary,
            "centrality_df": centrality_df,
            "saasp_scores": saasp_scores,
            "run_ml_pipeline": run_ml_pipeline,
            "top_k": top_k,
            "rf_results": rf_results,
            "threshold": threshold,
            "ranking_table": ranking_table,
            "modeling_df": modeling_df,
            "sir_scores": sir_scores,
            "sir_labels": sir_labels,
            "ml_probabilities": ml_probabilities,
        }
        history.append(
            {
                "label": build_history_label(dataset_label, analysis_graph.number_of_nodes(), run_ml_pipeline),
                "state": analysis_state,
            }
        )
        st.session_state["analysis_history"] = history[-5:]
        save_persisted_history(st.session_state["analysis_history"])
        st.session_state["active_history_index"] = len(st.session_state["analysis_history"]) - 1
        st.rerun()

    if not history:
        st.info("Choose a dataset and click Run analysis.")
        return

    active_state = history[active_history_index]["state"]
    dataset_label = active_state["dataset_label"]
    graph = active_state["graph"]
    analysis_graph = active_state["analysis_graph"]
    full_summary = active_state["full_summary"]
    analysis_summary = active_state["analysis_summary"]
    centrality_df = active_state["centrality_df"]
    saasp_scores = active_state["saasp_scores"]
    run_ml_pipeline = active_state["run_ml_pipeline"]
    top_k = active_state["top_k"]
    rf_results = active_state["rf_results"]
    threshold = active_state["threshold"]
    ranking_table = active_state["ranking_table"]
    modeling_df = active_state["modeling_df"]
    sir_scores = active_state["sir_scores"]
    sir_labels = active_state["sir_labels"]
    ml_probabilities = active_state["ml_probabilities"]

    tabs = st.tabs(
        [
            "Network Overview",
            "Centrality Analysis",
            "SAASP Results",
            "ML Predictions",
            "Method Comparison",
        ]
    )

    try:
        with tabs[0]:
            st.subheader(dataset_label)
            st.caption(f"Showing saved results from {history[active_history_index]['label']}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Full nodes", full_summary.nodes)
            col2.metric("Full edges", full_summary.edges)
            col3.metric("Analysis nodes", analysis_summary.nodes)
            col4.metric("Analysis edges", analysis_summary.edges)
            if analysis_graph.number_of_nodes() < graph.number_of_nodes():
                st.warning(
                    f"Interactive mode is analysing the first {analysis_graph.number_of_nodes()} nodes "
                    f"out of {graph.number_of_nodes()} total nodes so results appear quickly."
                )
            st.pyplot(draw_degree_histogram(analysis_graph))
            st.pyplot(draw_network_preview(analysis_graph))

        with tabs[1]:
            st.subheader("Classical centrality scores")
            st.dataframe(centrality_df.sort_values("pagerank", ascending=False).head(top_k), width="stretch")

        with tabs[2]:
            top_saasp = centrality_df.sort_values("saasp_score", ascending=False).head(top_k)
            st.subheader(f"Top {top_k} nodes by SAASP")
            st.dataframe(top_saasp[["node", "saasp_score"]], width="stretch")

            node_options = centrality_df["node"].tolist()
            default_node = int(top_saasp.iloc[0]["node"]) if not top_saasp.empty else int(node_options[0])
            current_selected_node = st.session_state.get("selected_node_id", default_node)
            if current_selected_node not in node_options:
                current_selected_node = default_node

            graph_col, detail_col = st.columns([1.8, 1.0])
            with graph_col:
                st.caption("Click a node in the graph to inspect it, or choose a node from the dropdown.")
                network_chart = build_interactive_network_figure(
                    analysis_graph,
                    centrality_df,
                    selected_node=current_selected_node,
                    limit=min(150, analysis_graph.number_of_nodes()),
                )
                selection_state = st.plotly_chart(
                    network_chart,
                    use_container_width=True,
                    key="network_node_inspector",
                    on_select="rerun",
                    selection_mode="points",
                    config={"displayModeBar": False},
                )
                clicked_node = extract_selected_node(selection_state)
                default_index = node_options.index(clicked_node) if clicked_node in node_options else node_options.index(current_selected_node)
                chosen_node = st.selectbox("Node selector", options=node_options, index=default_index)
                selected_node = clicked_node if clicked_node in node_options else chosen_node
                st.session_state["selected_node_id"] = selected_node

            with detail_col:
                render_node_detail_panel(
                    selected_node=st.session_state["selected_node_id"],
                    node_metrics=centrality_df,
                    sir_scores=sir_scores,
                    ml_probabilities=ml_probabilities,
                    sir_labels=sir_labels,
                )

            fig, ax = plt.subplots()
            ax.hist(list(saasp_scores.values()), bins=30, color="#3B8EA5", edgecolor="white")
            ax.set_title("SAASP score distribution")
            st.pyplot(fig)
    except Exception as exc:
        show_user_error("The graph analysis view could not be displayed with the saved results.", exc)
        return

    try:
        with tabs[3]:
            if rf_results is None:
                st.info("Enable `Run ML + SIR pipeline` in the sidebar to run the slower postgraduate validation stage.")
            else:
                st.subheader("Random Forest performance")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{rf_results['accuracy']:.3f}")
                col2.metric("F1", f"{rf_results['f1_weighted']:.3f}")
                col3.metric("ROC-AUC", "N/A" if rf_results["roc_auc"] is None else f"{rf_results['roc_auc']:.3f}")
                st.caption(f"SIR influence threshold (top 10%): {threshold:.3f}")

                cm_df = pd.DataFrame(
                    rf_results["confusion_matrix"],
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )
                st.dataframe(cm_df, width="stretch")

        with tabs[4]:
            if ranking_table is None or modeling_df is None:
                st.info("Method comparison appears after the optional ML + SIR pipeline finishes.")
            else:
                st.subheader("Ranking comparison against SIR mean spread")
                st.dataframe(ranking_table, width="stretch")

                fig, ax = plt.subplots()
                ax.scatter(modeling_df["saasp_score"], modeling_df["sir_mean_infected"], alpha=0.6, color="#D1495B")
                ax.set_xlabel("SAASP score")
                ax.set_ylabel("SIR mean infected")
                ax.set_title("SAASP vs SIR")
                st.pyplot(fig)
    except Exception as exc:
        with tabs[3]:
            show_user_error("The prediction view could not be displayed right now.", exc)
        with tabs[4]:
            show_user_error("The comparison view could not be displayed right now.", exc)
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        show_user_error("Something unexpected happened while rendering the app.", exc)
