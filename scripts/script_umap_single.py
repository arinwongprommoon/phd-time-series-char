#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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
    "experimentID": "is20016",
    # Group (strain) ID for first group
    "group1": "zwf1egf",
    # second group
    "group2": "by4741",
    # include only first half of the time series
    "only_first_half": False,
}

model_options = {
    # RNG seed
    "random_state": 69,
    # Hyperparameters
    "n_neighbors": 5,
    "min_dist": 0.5,
    "n_components": 2,
}

plot_choices = {
    # Colour dots by strain
    "strain": True,
    # Colour dots by score
    "score": True,
    # Combining strains and scores
    "combined": True,
    # Sample points from bounding boxes
    "samples": True,
    # Number of samples
    "samples/num": 3,
    # Bounding box 1 -- lower left co-ordinate and upper right co-ordinate
    "samples/bbox1": (np.array([-3, 0]), np.array([2, 4])),
    # Bounding box 2
    "samples/bbox2": (np.array([4, -5]), np.array([10, 1])),
    # Bounding box 3
    "samples/bbox3": (np.array([3, 1]), np.array([15, 12])),
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

# Include only first half of time series
# (For thesis corrections -- see if the groupings are because of the properties
# of time series or because of UMAP algorithm.)
if data_options["only_first_half"]:
    num_timepoints = timeseries_df.shape[1]
    timeseries_df = timeseries_df.iloc[:, : (num_timepoints // 2)]

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
    "zwf1egf": "zwf1Δ",
    "by4741": "BY4741",
}
strain_list = [strain_relabel_lookup.get(item, item) for item in strain_list]
strain_palette_map = {
    "zwf1Δ": "C0",
    "BY4741": "C1",
}
## Score
common_idx = features_scaled.index.intersection(labels_df.index)
scores_list = labels_df.loc[common_idx].score.to_list()
scores_relabel_lookup = {
    1: "Oscillatory",
    0: "Non-oscillatory",
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
    "zwf1Δ": "C0",
    "BY4741": "C1",
}

# UMAP
reducer = umap.UMAP(
    random_state=model_options["random_state"],
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

if plot_choices["samples"]:
    # TODO: functionalise and repeat for other bounding boxes
    # Bounding box 1
    for bbox_idx in range(3):
        # https://stackoverflow.com/a/33051576
        bbox_key = f"samples/bbox{bbox_idx + 1}"
        lower_left = plot_choices[bbox_key][0]
        upper_right = plot_choices[bbox_key][1]
        in_bbox_mask = np.all(
            np.logical_and(lower_left <= embedding, embedding <= upper_right), axis=1
        )
        nosc_mask = np.array(scores_list) == "Non-oscillatory"
        # Select only the ones that are non-oscillatory
        in_bbox_nosc_mask = np.logical_and(in_bbox_mask, nosc_mask)
        in_bbox_nosc_idx = np.array(range(len(embedding)))[in_bbox_nosc_mask]
        # Sample, randomly
        sample_idx = in_bbox_nosc_idx[
            np.random.choice(
                len(in_bbox_nosc_idx), plot_choices["samples/num"], replace=False
            )
        ]
        # Convert to cell ID -- this is needed because NaNs were not removed from
        # timeseries_df
        sample_multi_idx = features_df.iloc[sample_idx].index
        sample_timeseries_df = timeseries_df.loc[sample_multi_idx]

        # Draw samples
        fig, axs = plt.subplots(
            nrows=plot_choices["samples/num"], sharex=True, figsize=(6, 6)
        )
        for sample_idx, ax in enumerate(axs):
            ax.plot(sample_timeseries_df.iloc[sample_idx])
            ax.set_title(f"bbox{bbox_idx+1}, sample {sample_idx+1}")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        # Create common axis labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        plt.grid(False)
        plt.xlabel("Time (min)")
        plt.ylabel("Flavin fluorescence, normalised (AU)")

    # Bounding box 2
    # Bounding box 3

# Save figures
pdf_filename = "../reports/umap_single_" + data_options["experimentID"] + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
