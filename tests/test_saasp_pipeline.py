import random
import unittest

import networkx as nx

from saasp_pipeline import (
    build_augmented_graph,
    build_feature_vector,
    compute_asp_influence,
    compute_saasp_score,
    create_sir_labels,
    get_extended_neighborhood,
    run_saasp_on_graph,
    sir_simulation_single_run,
)


class SaaspPipelineTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
            ]
        )

    def test_extended_neighborhood_respects_hop_depth(self):
        local_subgraph, nodes = get_extended_neighborhood(self.graph, 0, k=2)
        self.assertEqual(nodes, {0, 1, 2, 3})
        self.assertEqual(set(local_subgraph.nodes()), {0, 1, 2, 3})

    def test_feature_vector_contains_expected_dimensions(self):
        vector = build_feature_vector(self.graph, 2)
        self.assertEqual(vector.shape, (5,))
        self.assertGreaterEqual(vector[0], 1.0)

    def test_augmented_graph_adds_similarity_edges(self):
        local_subgraph, _ = get_extended_neighborhood(self.graph, 2, k=2)
        augmented = build_augmented_graph(local_subgraph, self.graph, m=2)
        self.assertGreaterEqual(augmented.number_of_edges(), local_subgraph.number_of_edges())
        augmented_edges = [
            (u, v)
            for u, v, data in augmented.edges(data=True)
            if data.get("augmented")
        ]
        self.assertTrue(augmented_edges)

    def test_asp_influence_is_positive_for_connected_subgraph(self):
        local_subgraph, _ = get_extended_neighborhood(self.graph, 2, k=2)
        augmented = build_augmented_graph(local_subgraph, self.graph, m=2)
        score = compute_asp_influence(augmented, 2)
        self.assertGreater(score, 0.0)

    def test_saasp_scores_are_normalized_and_vary(self):
        scores = run_saasp_on_graph(self.graph, k=2, m=2)
        self.assertEqual(set(scores.keys()), set(self.graph.nodes()))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in scores.values()))
        self.assertGreater(len(set(round(value, 6) for value in scores.values())), 1)

    def test_individual_saasp_score_is_non_negative(self):
        score = compute_saasp_score(self.graph, 2, k=2, m=2)
        self.assertGreaterEqual(score, 0.0)

    def test_sir_single_run_covers_entire_path_when_infection_is_certain(self):
        path_graph = nx.path_graph(4)
        infected = sir_simulation_single_run(
            path_graph,
            seed_node=0,
            beta=1.0,
            gamma=0.0,
            max_steps=10,
            rng=random.Random(7),
        )
        self.assertEqual(infected, 4)

    def test_sir_label_threshold_marks_top_quantile(self):
        sir_scores = {
            0: {"mean": 1.0},
            1: {"mean": 2.0},
            2: {"mean": 3.0},
            3: {"mean": 4.0},
            4: {"mean": 5.0},
        }
        labels, threshold = create_sir_labels(sir_scores, threshold_quantile=0.80)
        self.assertGreaterEqual(threshold, 4.0)
        self.assertEqual(labels[4], 1)
        self.assertEqual(labels[0], 0)


if __name__ == "__main__":
    unittest.main()
