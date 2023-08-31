#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

f = pd.read_csv("../data/processed/features_scaled.csv", index_col=[0, 1, 2])

d = {
    "n_neighbors": [5, 10],
    "min_dist": [0.25, 0.5],
}


def umap_grid(hyperparam_dict, features_scaled):
    n_neighbors_list = hyperparam_dict["n_neighbors"]
    min_dist_list = hyperparam_dict["min_dist"]
    x_dim = len(n_neighbors_list)
    y_dim = len(min_dist_list)
    embedding_array = np.zeros(shape=(x_dim, y_dim), dtype="object")

    for x_index, n_neighbors in enumerate(n_neighbors_list):
        for y_index, min_dist in enumerate(min_dist_list):
            reducer = umap.UMAP(
                random_state=69,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2,
            )
            mapper = reducer.fit(features_scaled)
            embedding = mapper.embedding_
            embedding_array[x_index, y_index] = embedding

    return embedding_array


def plot_umap_grid(
    hyperparam_dict,
    embedding_array,
    xlabel=None,
    ylabel=None,
):
    embedding_array = np.rot90(embedding_array)

    nrows, ncols = embedding_array.shape
    # Have exch_rate values in an extra column & row
    nrows += 1
    ncols += 1
    fig, ax = plt.subplots(nrows, ncols)

    # Define axis labels
    global_xaxislabels = list(hyperparam_dict.values())[0]
    global_yaxislabels = list(hyperparam_dict.values())[1][::-1]
    # Dummy value for corner position -- not used
    # (could be useful for debugging)
    global_xaxislabels = np.append([-1], global_xaxislabels)
    global_yaxislabels = np.append(global_yaxislabels, [-1])

    # Draw scatter plots
    for row_idx, global_yaxislabel in enumerate(global_yaxislabels):
        # Left column reserved for min_dist labels
        ax[row_idx, 0].set_axis_off()
        # Bottom left corner must be blank
        if row_idx == len(global_yaxislabels) - 1:
            pass
        else:
            # Print exch rate label
            ax[row_idx, 0].text(
                x=0.5,
                y=0.5,
                s=f"{global_yaxislabel:.3f}",
                ha="center",
                va="center",
            )
        for col_idx, global_xaxislabel in enumerate(global_xaxislabels):
            # Bottom row reserved for n_neighbors labels
            if row_idx == len(global_yaxislabels) - 1:
                ax[row_idx, col_idx].set_axis_off()
                # Bottom left corner must be blank
                if col_idx == 0:
                    pass
                else:
                    # Print hyperparam label
                    ax[row_idx, col_idx].text(
                        x=0.5,
                        y=0.5,
                        s=f"{global_xaxislabel:.3f}",
                        ha="center",
                        va="center",
                    )
            else:
                # Left column reserved for min_dist labels
                if col_idx == 0:
                    pass
                else:
                    # Get embedding
                    embedding = embedding_array[row_idx, col_idx - 1]
                    # Draw
                    sns.scatterplot(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        # hue=scores_list,
                        # palette=scores_palette_map,
                        ax=ax[row_idx, col_idx],
                    )

    # For global axis labels: create a big subplot and hide everything except
    # for the labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    if xlabel is None:
        xlabel = list(hyperparam_dict.keys())[0]
    if ylabel is None:
        ylabel = list(hyperparam_dict.keys())[1]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Legend: colour = biomass component
    # fig.legend(artists[0], component_list, loc="center right")
    # fig.subplots_adjust(right=0.75)


e = umap_grid(d, f)
plot_umap_grid(d, e)

breakpoint()

print("foo")
