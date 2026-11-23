SAASP Influence Analyzer
An Interactive Machine Learning System for Identifying Influential Nodes in Complex Networks
Project Overview
SAASP Influence Analyzer is an interactive research tool that identifies influential nodes in complex social networks by combining graph algorithms, epidemic simulation, machine learning, and interactive visualization.
Unlike traditional centrality measures that rely on a single structural property, this project implements the Shortest-path Augmented Adaptive Subgraph Propagation (SAASP) algorithm, which captures both local topology and structural similarity between nodes.
The complete pipeline is delivered through an interactive Streamlit dashboard that enables researchers and students to analyze social networks without writing code.
Why I Built This
Classical network centrality measures such as Degree Centrality, PageRank, Closeness, and Betweenness often provide different rankings because each captures only one aspect of influence.
The goal of this project was to:
implement the SAASP algorithm from research literature
validate influence using SIR epidemic simulations
train a machine learning model using independent labels
compare multiple ranking algorithms
provide an interactive interface for network exploration
This project was completed as part of my MSc in Data Analytics and Computational Science at Digital University Kerala.
