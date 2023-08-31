#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_umap_grid(
    hyperparam_dict,
    embedding_array,
    hue=None,
    palette=None,
    xlabel=None,
    ylabel=None,
):
    embedding_array = np.rot90(embedding_array)

    nrows, ncols = embedding_array.shape
    # Have exch_rate values in an extra column & row
    nrows += 1
    ncols += 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

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
            # Print min_dist label
            ax[row_idx, 0].text(
                x=0.5,
                y=0.5,
                s=f"{global_yaxislabel:.2f}",
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
                    # Print n_neghbors label
                    ax[row_idx, col_idx].text(
                        x=0.5,
                        y=0.5,
                        s=f"{global_xaxislabel:.0f}",
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
                        s=5,
                        hue=hue,
                        palette=palette,
                        legend=False,
                        ax=ax[row_idx, col_idx],
                    )
                    ax[row_idx, col_idx].get_xaxis().set_ticks([])
                    ax[row_idx, col_idx].get_yaxis().set_ticks([])

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
