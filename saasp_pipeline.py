from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
    from scipy.stats import kendalltau, spearmanr
except ImportError:  # pragma: no cover - handled at runtime in app/tests
    kendalltau = None
    spearmanr = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - handled at runtime in app/tests
    RandomForestClassifier = None
    LogisticRegression = None
    accuracy_score = None
    confusion_matrix = None
    f1_score = None
    roc_auc_score = None
    StratifiedKFold = None
    cross_val_score = None
    train_test_split = None
    Pipeline = None
    StandardScaler = None


RANDOM_SEED = 42


@dataclass(frozen=True)
class DatasetSummary:
    nodes: int
    edges: int
    density: float
    average_degree: float
    max_degree: int
    average_clustering: float
    average_path_length: float


def _require_sklearn() -> None:
    if RandomForestClassifier is None:
        raise ImportError(
            "scikit-learn is required for the ML pipeline. "
            "Install scikit-learn to train and evaluate the models."
        )


def _require_scipy() -> None:
    if kendalltau is None or spearmanr is None:
        raise ImportError(
            "scipy is required for ranking validation. "
            "Install scipy to compute Kendall tau and Spearman rho."
        )


def load_graph(filepath: str | Path, directed: bool = False) -> nx.Graph:
    graph_type = nx.DiGraph if directed else nx.Graph
    graph = nx.read_edgelist(filepath, nodetype=int, create_using=graph_type())
    graph.remove_edges_from(nx.selfloop_edges(graph))

    if directed:
        if not nx.is_weakly_connected(graph):
            largest = max(nx.weakly_connected_components(graph), key=len)
            graph = graph.subgraph(largest).copy()
    elif not nx.is_connected(graph):
        largest = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest).copy()

    return graph


def summarize_graph(graph: nx.Graph) -> DatasetSummary:
    degrees = [degree for _, degree in graph.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    avg_clustering = float(nx.average_clustering(graph)) if graph.number_of_nodes() else 0.0
    path_graph = graph

    if graph.number_of_nodes() < 2:
        avg_path_length = 0.0
    else:
        if graph.is_directed():
            if not nx.is_weakly_connected(graph):
                largest = max(nx.weakly_connected_components(graph), key=len)
                path_graph = graph.subgraph(largest).copy()
        elif not nx.is_connected(graph):
            largest = max(nx.connected_components(graph), key=len)
            path_graph = graph.subgraph(largest).copy()

        if path_graph.number_of_nodes() < 2:
            avg_path_length = 0.0
        elif path_graph.number_of_nodes() <= 500:
            avg_path_length = float(nx.average_shortest_path_length(path_graph))
        else:
            avg_path_length = estimate_average_path_length(path_graph, sample_size=200, seed=RANDOM_SEED)

    return DatasetSummary(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        density=float(nx.density(graph)),
        average_degree=avg_degree,
        max_degree=max(degrees, default=0),
        average_clustering=avg_clustering,
        average_path_length=avg_path_length,
    )


def estimate_average_path_length(graph: nx.Graph, sample_size: int = 200, seed: int = RANDOM_SEED) -> float:
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return 0.0

    rng = random.Random(seed)
    samples = []
    for _ in range(sample_size):
        source, target = rng.sample(nodes, 2)
        try:
            samples.append(nx.shortest_path_length(graph, source=source, target=target))
        except nx.NetworkXNoPath:
            continue

    return float(np.mean(samples)) if samples else 0.0


def compute_classical_centralities(graph: nx.Graph, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    node_count = graph.number_of_nodes()
    degree_centrality = nx.degree_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    pagerank = nx.pagerank(graph, alpha=0.85)
    clustering = nx.clustering(graph)
    k_shell = nx.core_number(graph)
    avg_neighbor_degree = nx.average_neighbor_degree(graph)

    if node_count > 10_000:
        betweenness = nx.betweenness_centrality(graph, k=200, seed=random_seed)
    else:
        betweenness = nx.betweenness_centrality(graph)

    rows = []
    for node in graph.nodes():
        rows.append(
            {
                "node": node,
                "degree": degree_centrality[node],
                "betweenness": betweenness[node],
                "closeness": closeness[node],
                "pagerank": pagerank[node],
                "clustering": clustering[node],
                "k_shell": k_shell[node],
                "avg_neighbor_degree": avg_neighbor_degree[node],
            }
        )

    return pd.DataFrame(rows)


def get_extended_neighborhood(graph: nx.Graph, node: int, k: int = 2) -> Tuple[nx.Graph, set[int]]:
    visited = {node}
    frontier = {node}

    for _ in range(k):
        next_frontier = set()
        for current in frontier:
            next_frontier.update(graph.neighbors(current))
        frontier = next_frontier - visited
        visited.update(frontier)

    local_subgraph = graph.subgraph(visited).copy()
    return local_subgraph, visited


def max_neighbor_degree(graph: nx.Graph, node: int) -> float:
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return 0.0
    return float(max(graph.degree(neighbor) for neighbor in neighbors))


def build_feature_vector(
    graph: nx.Graph,
    node: int,
    clustering_map: Optional[Mapping[int, float]] = None,
    k_shell_map: Optional[Mapping[int, int]] = None,
    avg_neighbor_degree_map: Optional[Mapping[int, float]] = None,
) -> np.ndarray:
    clustering_map = clustering_map or nx.clustering(graph)
    k_shell_map = k_shell_map or nx.core_number(graph)
    avg_neighbor_degree_map = avg_neighbor_degree_map or nx.average_neighbor_degree(graph)

    return np.array(
        [
            float(graph.degree(node)),
            float(clustering_map[node]),
            float(k_shell_map[node]),
            float(avg_neighbor_degree_map[node]),
            float(max_neighbor_degree(graph, node)),
        ],
        dtype=float,
    )


def _normalize_feature_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return feature_matrix / norms


def build_augmented_graph(local_subgraph: nx.Graph, graph_original: nx.Graph, m: int = 5) -> nx.Graph:
    nodes = list(local_subgraph.nodes())
    if len(nodes) < 2:
        return local_subgraph.copy()

    clustering_map = nx.clustering(graph_original)
    k_shell_map = nx.core_number(graph_original)
    avg_neighbor_degree_map = nx.average_neighbor_degree(graph_original)

    feature_matrix = np.vstack(
        [
            build_feature_vector(
                graph_original,
                node,
                clustering_map=clustering_map,
                k_shell_map=k_shell_map,
                avg_neighbor_degree_map=avg_neighbor_degree_map,
            )
            for node in nodes
        ]
    )
    normalized = _normalize_feature_matrix(feature_matrix)
    similarity_matrix = normalized @ normalized.T

    augmented = local_subgraph.copy()
    similarity_pairs: List[Tuple[float, int, int]] = []

    for index_a, node_a in enumerate(nodes):
        for index_b in range(index_a + 1, len(nodes)):
            node_b = nodes[index_b]
            if augmented.has_edge(node_a, node_b):
                continue
            similarity_pairs.append((float(similarity_matrix[index_a, index_b]), node_a, node_b))

    similarity_pairs.sort(key=lambda item: item[0], reverse=True)
    for similarity, node_a, node_b in similarity_pairs[:m]:
        augmented.add_edge(
            node_a,
            node_b,
            weight=max(1e-6, 1.0 - similarity),
            similarity=similarity,
            augmented=True,
        )

    return augmented


def compute_asp_influence(augmented_graph: nx.Graph, center_node: int) -> float:
    if center_node not in augmented_graph:
        return 0.0

    distances = nx.single_source_dijkstra_path_length(augmented_graph, source=center_node, weight="weight")
    if len(distances) <= 1:
        return 0.0

    avg_distance = float(np.mean(list(distances.values())))
    return 0.0 if avg_distance == 0.0 else 1.0 / avg_distance


def compute_saasp_score(graph: nx.Graph, node: int, k: int = 2, m: int = 5) -> float:
    local_subgraph, neighborhood = get_extended_neighborhood(graph, node, k)
    augmented_graph = build_augmented_graph(local_subgraph, graph, m=m)
    asp_score = compute_asp_influence(augmented_graph, node)
    degree_value = graph.degree(node)
    neighborhood_size = len(neighborhood)
    return asp_score * math.log1p(degree_value) * math.log1p(neighborhood_size)


def run_saasp_on_graph(graph: nx.Graph, k: int = 2, m: int = 5) -> Dict[int, float]:
    raw_scores = {node: compute_saasp_score(graph, node, k=k, m=m) for node in graph.nodes()}
    values = list(raw_scores.values())
    if not values:
        return {}

    min_score = min(values)
    max_score = max(values)
    if math.isclose(min_score, max_score):
        return {node: 0.0 for node in raw_scores}

    return {node: (score - min_score) / (max_score - min_score) for node, score in raw_scores.items()}


def sir_simulation_single_run(
    graph: nx.Graph,
    seed_node: int,
    beta: float,
    gamma: float,
    max_steps: int,
    rng: Optional[random.Random] = None,
) -> int:
    rng = rng or random.Random()
    susceptible = set(graph.nodes()) - {seed_node}
    infected = {seed_node}
    recovered = set()

    for _ in range(max_steps):
        new_infected = set()
        new_recovered = set()

        for node in infected:
            for neighbor in graph.neighbors(node):
                if neighbor in susceptible and rng.random() < beta:
                    new_infected.add(neighbor)
            if rng.random() < gamma:
                new_recovered.add(node)

        infected = (infected | new_infected) - new_recovered
        susceptible -= new_infected
        recovered |= new_recovered

        if not infected:
            break

    return len(recovered | infected)


def compute_sir_influence_scores(
    graph: nx.Graph,
    n_runs: int = 50,
    beta: float = 0.3,
    gamma: float = 0.1,
    max_steps: int = 50,
    seed: int = RANDOM_SEED,
) -> Dict[int, Dict[str, float]]:
    rng = random.Random(seed)
    scores: Dict[int, Dict[str, float]] = {}

    for node in graph.nodes():
        run_results = [
            sir_simulation_single_run(
                graph,
                seed_node=node,
                beta=beta,
                gamma=gamma,
                max_steps=max_steps,
                rng=random.Random(rng.randint(0, 10**9)),
            )
            for _ in range(n_runs)
        ]

        scores[node] = {
            "mean": float(np.mean(run_results)),
            "std": float(np.std(run_results)),
            "min": float(np.min(run_results)),
            "max": float(np.max(run_results)),
        }

    return scores


def create_sir_labels(
    sir_scores: Mapping[int, Mapping[str, float]],
    threshold_quantile: float = 0.90,
) -> Tuple[Dict[int, int], float]:
    mean_scores = np.array([score["mean"] for score in sir_scores.values()], dtype=float)
    threshold = float(np.quantile(mean_scores, threshold_quantile))
    labels = {node: int(score["mean"] >= threshold) for node, score in sir_scores.items()}
    return labels, threshold


def build_ml_feature_matrix(
    centrality_df: pd.DataFrame,
    saasp_scores: Mapping[int, float],
    sir_labels: Mapping[int, int],
    sir_scores: Optional[Mapping[int, Mapping[str, float]]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = centrality_df.copy()
    df["saasp_score"] = df["node"].map(saasp_scores)
    df["sir_label"] = df["node"].map(sir_labels)

    if sir_scores is not None:
        df["sir_mean_infected"] = df["node"].map(lambda node: sir_scores[node]["mean"])
        df["sir_std_infected"] = df["node"].map(lambda node: sir_scores[node]["std"])

    feature_columns = [
        "degree",
        "betweenness",
        "closeness",
        "pagerank",
        "clustering",
        "k_shell",
        "avg_neighbor_degree",
        "saasp_score",
    ]
    x = df.set_index("node")[feature_columns]
    y = df.set_index("node")["sir_label"].astype(int)
    return x, y, df


def check_class_balance(labels: pd.Series) -> Dict[str, float]:
    counts = labels.value_counts().to_dict()
    majority = max(counts.values()) if counts else 0
    minority = min(counts.values()) if counts else 0
    ratio = float(majority / minority) if minority else float("inf")
    return {
        "class_0": float(counts.get(0, 0)),
        "class_1": float(counts.get(1, 0)),
        "imbalance_ratio": ratio,
        "is_imbalanced": ratio > 5.0,
    }


def train_models(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, object]:
    _require_sklearn()
    if y.nunique() < 2:
        raise ValueError("At least two label classes are required to train the ML models.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    balance = check_class_balance(y)
    class_weight = "balanced" if balance["is_imbalanced"] else None

    models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight=class_weight,
                        max_iter=1000,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight=class_weight,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }

    trained_models: Dict[str, object] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model

    return {
        "models": trained_models,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "class_balance": balance,
    }


def evaluate_model(model: object, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    _require_sklearn()

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    roc_auc = None
    if len(set(y_test)) > 1:
        roc_auc = float(roc_auc_score(y_test, y_prob))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def cross_validate_model(model: object, x: pd.DataFrame, y: pd.Series, folds: int = 5) -> Dict[str, float]:
    _require_sklearn()

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, x, y, cv=cv, scoring="roc_auc")
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    return {
        "mean_roc_auc": mean_score,
        "std_roc_auc": std_score,
        "ci_low": mean_score - 2 * std_score,
        "ci_high": mean_score + 2 * std_score,
    }


def _top_k_overlap_from_series(reference: pd.Series, candidate: pd.Series, k: int) -> float:
    reference_top = set(reference.sort_values(ascending=False).head(k).index)
    candidate_top = set(candidate.sort_values(ascending=False).head(k).index)
    return len(reference_top & candidate_top) / float(k)


def validate_rankings(
    df: pd.DataFrame,
    saasp_scores: Mapping[int, float],
    sir_scores: Mapping[int, Mapping[str, float]],
    trained_models: Mapping[str, object],
    x_test: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    _require_scipy()

    ranking_frame = df.set_index("node").copy()
    ranking_frame["saasp_score"] = ranking_frame.index.map(saasp_scores)
    ranking_frame["sir_mean_infected"] = ranking_frame.index.map(lambda node: sir_scores[node]["mean"])

    results = []
    ground_truth = ranking_frame["sir_mean_infected"]

    classical_methods = ["degree", "betweenness", "closeness", "pagerank", "saasp_score"]
    for method in classical_methods:
        candidate = ranking_frame[method]
        kendall_value, kendall_p = kendalltau(ground_truth, candidate)
        spearman_value, spearman_p = spearmanr(ground_truth, candidate)
        results.append(
            {
                "method": "SAASP" if method == "saasp_score" else method.title(),
                "kendall_tau": float(kendall_value),
                "kendall_p_value": float(kendall_p),
                "spearman_rho": float(spearman_value),
                "spearman_p_value": float(spearman_p),
                "top_k_overlap": float(_top_k_overlap_from_series(ground_truth, candidate, top_k)),
            }
        )

    if "RandomForest" in trained_models:
        rf_model = trained_models["RandomForest"]
        test_nodes = x_test.index
        ml_scores = pd.Series(rf_model.predict_proba(x_test)[:, 1], index=test_nodes, name="ml_score")
        ml_truth = ranking_frame.loc[test_nodes, "sir_mean_infected"]
        kendall_value, kendall_p = kendalltau(ml_truth, ml_scores)
        spearman_value, spearman_p = spearmanr(ml_truth, ml_scores)
        results.append(
            {
                "method": "ML (RF)",
                "kendall_tau": float(kendall_value),
                "kendall_p_value": float(kendall_p),
                "spearman_rho": float(spearman_value),
                "spearman_p_value": float(spearman_p),
                "top_k_overlap": float(_top_k_overlap_from_series(ml_truth, ml_scores, min(top_k, len(ml_scores)))),
            }
        )

    return pd.DataFrame(results).sort_values(by="kendall_tau", ascending=False).reset_index(drop=True)


def available_local_datasets(base_path: str | Path) -> Dict[str, str]:
    base = Path(base_path)
    candidates = {}
    for path in sorted(base.glob("*.txt")):
        candidates[path.stem] = str(path)
    for path in sorted(base.glob("*.csv")):
        candidates[path.stem] = str(path)
    return candidates
