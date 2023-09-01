#!/usr/bin/env python3

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la

from scipy.spatial.distance import pdist, squareform
from postprocessor.core.processes.catch22 import catch22
from postprocessor.core.processes.standardscaler import standardscaler

from src.utils.utils import graph_prune

data_options = {
    # Experiment ID.
    # Prefix 'is' for islay or 'st' for staffa.
    # ID is 5 digits, add leading zeros as appropriate.
    "experimentID": "st01253",
    # Group (strain) ID for first group
    "group1": "tsa1tsa2morgan",
    # second group
    "group2": "by4742swain",
}

model_options = {
    # Number of nearest neighbours for pruning
    "neighbors": 7,
}

plot_choices = {
    # Pruned graph, coloured by strain/score combined
    # Note: community detection not performed here
    "combined": True,
    # Community detection, modularity, Leiden algorithm
    "leiden": True,
    # Community detection, Constant Potts Model, with resolution
    "cpm": True,
    # Resolution parameter
    "cpm/resolution": 0.01,
    # CPM, range of resolution values
    "cpm_range": True,
    # Resolution parameter: list of numbers to sweep through
    "cpm_range/resolutions": [0.01, 0.02],
}


def prettyfloat(x):
    if x is None:
        # Undefined
        out_str = "Undefd"
    else:
        out_str = f"{x:06.3f}".replace(".", "p")
    return out_str


data_dir = "../data/raw/"

# Define groups/strains
group1_name = data_options["experimentID"] + "_" + data_options["group1"]
group2_name = data_options["experimentID"] + "_" + data_options["group2"]


# Load data
def get_timeseries_labels(group_name):
    """Convenience function to get timeseries and label DataFrames"""
    filepath = data_dir + group_name
    timeseries_filepath = filepath + "_timeseries.csv"
    labels_filepath = filepath + "_labels.csv"
    timeseries_df = pd.read_csv(timeseries_filepath, index_col=[0, 1, 2])
    labels_df = pd.read_csv(labels_filepath, index_col=[0, 1, 2])

    return timeseries_df, labels_df


timeseries1_df, labels1_df = get_timeseries_labels(group1_name)
timeseries2_df, labels2_df = get_timeseries_labels(group2_name)

# Join dataframes
timeseries_df = pd.concat([timeseries1_df, timeseries2_df])
labels_df = pd.concat([labels1_df, labels2_df])

# Featurise
features_df = catch22.as_function(timeseries_df)
# Scale
features_scaled = standardscaler.as_function(features_df.T).T

# Create lists for plot options
# TODO: Merge dictionaries to compact code.  This will also compact the plotting
# code.
## Strain
position_list = features_scaled.index.get_level_values("position").to_list()
strain_list = [position.split("_")[0] for position in position_list]
strain_relabel_lookup = {
    "tsa1tsa2morgan": "tsa1Δ tsa2Δ",
    "by4742swain": "BY4742",
}
strain_list = [strain_relabel_lookup.get(item, item) for item in strain_list]
strain_palette_map = {
    "tsa1Δ tsa2Δ": "C0",
    "BY4742": "C1",
}
## Score
common_idx = features_scaled.index.intersection(labels_df.index)
scores_list = labels_df.loc[common_idx].score.to_list()
scores_relabel_lookup = {
    0: "Oscillatory",
    1: "Non-oscillatory",
}
scores_list = [scores_relabel_lookup.get(item, item) for item in scores_list]
scores_palette_map = {
    "Oscillatory": "k",
    "Non-oscillatory": "lightgrey",
}
## Combined
label_list = []
for strain, score in zip(strain_list, scores_list):
    if score == "Non-oscillatory":
        label_list.append(score)
    elif score == "Oscillatory":
        label_list.append(strain)

label_palette_map = {
    "Non-oscillatory": "#d3d3d3",
    "tsa1Δ tsa2Δ": "#1f77b4",
    "BY4742": "#ff7f0e",
}

# Distance matrix
distances = pdist(features_scaled, metric="euclidean")
distance_matrix = squareform(distances)
# Prune
distance_matrix_pruned = graph_prune(distance_matrix, model_options["neighbors"])
# Create graph
graph = ig.Graph.Weighted_Adjacency(distance_matrix_pruned.tolist(), mode="undirected")

# Draw...
# Uses cairo backend, hence separate PDFs and no collection of figs at end
if plot_choices["combined"]:
    vertex_color = [label_palette_map[label] for label in label_list]
    weights = np.array([weight for weight in graph.es["weight"]])
    # Control width so thickness doesn't overcrowd the drawing
    edge_width = weights / 6
    visual_style = {
        "layout": graph.layout("kk"),
        "vertex_size": 10,
        "vertex_color": vertex_color,
        "edge_width": edge_width,
    }
    filepath_combined = (
        "../reports/graphclust_combined_" + data_options["experimentID"] + ".pdf"
    )
    ig.plot(graph, filepath_combined, **visual_style)
    print(f"Combined drawn\n")

if plot_choices["leiden"]:
    partition_leiden = la.find_partition(graph, la.ModularityVertexPartition)
    print(f"Leiden, number of communities: {partition_leiden._len}")
    filepath_leiden = (
        "../reports/graphclust_leiden_" + data_options["experimentID"] + ".pdf"
    )
    ig.plot(
        partition_leiden,
        target=filepath_leiden,
        layout=graph.layout("kk"),
        vertex_size=10,
    )
    print("\n")

if plot_choices["cpm"]:
    partition_cpm = la.find_partition(
        graph,
        la.CPMVertexPartition,
        resolution_parameter=plot_choices["cpm/resolution"],
    )
    print(f"CPM, resolution: {plot_choices['cpm/resolution']}")
    print(f"CPM, number of communities: {partition_cpm._len}")
    filepath_cpm = (
        "../reports/graphclust_cpm_"
        + prettyfloat(plot_choices["cpm/resolution"])
        + "_"
        + data_options["experimentID"]
        + ".pdf"
    )
    ig.plot(
        partition_cpm,
        target=filepath_cpm,
        layout=graph.layout("kk"),
        vertex_size=10,
    )
    print("\n")

if plot_choices["cpm_range"]:
    for resolution in plot_choices["cpm_range/resolutions"]:
        partition_cpm = la.find_partition(
            graph,
            la.CPMVertexPartition,
            resolution_parameter=resolution,
        )
        print(f"CPM, resolution: {resolution}")
        print(f"CPM, number of communities: {partition_cpm._len}")
        filepath_cpm = (
            "../reports/graphclust_cpm_"
            + prettyfloat(resolution)
            + "_"
            + data_options["experimentID"]
            + ".pdf"
        )
        ig.plot(
            partition_cpm,
            target=filepath_cpm,
            layout=graph.layout("kk"),
            vertex_size=10,
        )
    print("\n")
