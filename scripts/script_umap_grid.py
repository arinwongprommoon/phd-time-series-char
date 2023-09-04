#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from src.umapper.umapper import umap_grid
from src.viz.grid import plot_umap_grid

hyperparam_dict = {
    "n_neighbors": [5, 10, 20, 50, 100, 150],
    "min_dist": [0.0, 0.25, 0.5, 1.0],
}


# TODO: Make this script less reliant on these CSV files
features_scaled = pd.read_csv(
    "../data/processed/features_scaled.csv", index_col=[0, 1, 2]
)
labels_df = pd.read_csv("../data/processed/labels_df.csv", index_col=[0, 1, 2])

position_list = features_scaled.index.get_level_values("position").to_list()
strain_list = [position.split("_")[0] for position in position_list]
strain_relabel_lookup = {
    "zwf1egf": "zwf1Δ",
    "by4741": "BY4741",
}
strain_list = [strain_relabel_lookup.get(item, item) for item in strain_list]

common_idx = features_scaled.index.intersection(labels_df.index)
scores_list = labels_df.loc[common_idx].score.to_list()
scores_relabel_lookup = {
    0: "Oscillatory",
    1: "Non-oscillatory",
}
scores_list = [scores_relabel_lookup.get(item, item) for item in scores_list]

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

embedding_array = umap_grid(hyperparam_dict, features_scaled)
plot_umap_grid(
    hyperparam_dict=hyperparam_dict,
    embedding_array=embedding_array,
    hue=label_list,
    palette=label_palette_map,
)

# Save figures
pdf_filename = "../reports/umap_grid_is20016.pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
