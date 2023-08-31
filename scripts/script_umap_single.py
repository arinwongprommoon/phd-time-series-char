#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap

from matplotlib.backends.backend_pdf import PdfPages
from postprocessor.core.processes.catch22 import catch22
from postprocessor.core.processes.standardscaler import standardscaler

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
    "n_neighbors": 10,
    "min_dist": 0.05,
    "n_components": 2,
}

plot_choices = {
    # Colour dots by strain
    "strain": True,
    # Colour dots by score
    "score": True,
    # Combining strains and scores
    "combined": True,
}

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
    "Non-oscillatory": "lightgrey",
    "tsa1Δ tsa2Δ": "C0",
    "BY4742": "C1",
}

# UMAP
reducer = umap.UMAP(
    n_neighbors=model_options["n_neighbors"],
    min_dist=model_options["min_dist"],
    n_components=model_options["n_components"],
)
mapper = reducer.fit(features_scaled)
embedding = mapper.embedding_

# Draw figures
if plot_choices["strain"]:
    fig_strain, ax_strain = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=strain_list,
        palette=strain_palette_map,
        ax=ax_strain,
    )

if plot_choices["score"]:
    fig_score, ax_score = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=scores_list,
        palette=scores_palette_map,
        ax=ax_score,
    )

if plot_choices["combined"]:
    fig_combined, ax_combined = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=label_list,
        palette=label_palette_map,
        ax=ax_combined,
    )

# Save figures
pdf_filename = "../reports/umap_single_" + data_options["experimentID"] + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
