#!/usr/bin/env python3
#
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
from postprocessor.core.processes.fft import fft
from sklearn.metrics import confusion_matrix

data_options = {
    # Experiment ID.
    # Prefix 'is' for islay or 'st' for staffa.
    # ID is 5 digits, add leading zeros as appropriate.
    "experimentID": "st01253",
    # Group (strain) ID for first group
    "group1": "by4742swain",
}

plot_choices = {
    # Best/Worst ranks of time series/periodograms
    "n_ranks": True,
    # First n ranks
    "n_ranks/n": 5,
    # ROC curve
    "roc": True,
    # False discovery rates
    "roc/fdr_vec": np.power(10, np.linspace(-8, -1, 100)),
}


def fdr_classify(scores, q):
    """Classify time series, with an input false discovery rate"""
    scores_sorted = scores.sort_values(ascending=False)
    n_ts = len(scores_sorted)
    classifications = []
    for k in range(n_ts):
        tmp = -np.log(1 - (1 - ((q * (k + 1)) / n_ts)) ** (1 / M))
        classification = scores_sorted[k] >= tmp
        classifications.append(classification)
    classifications = np.array(classifications)
    classifications_df = pd.DataFrame(
        classifications, scores_sorted.index, ["classification"]
    )
    return classifications_df


# Load data
data_dir = "../data/raw/"
group1_name = data_options["experimentID"] + "_" + data_options["group1"]

filepath1 = data_dir + group1_name
timeseries1_filepath = filepath1 + "_timeseries.csv"
labels1_filepath = filepath1 + "_labels.csv"

timeseries_df = pd.read_csv(timeseries1_filepath, index_col=[0, 1, 2])
labels_df = pd.read_csv(labels1_filepath, index_col=[0, 1, 2])

timeseries_dropna = timeseries_df.dropna()
labels_df = labels_df == 1

# Compute periodograms
freqs, power = fft.as_function(timeseries_df)
freqs = freqs.dropna()
power = power.dropna()

# Compute scores (ranks)
# Scores for each time series is simply the max height of power
scores = power.max(axis=1)
scores_sorted = scores.sort_values(ascending=False)

# Plot best 5/worst 5
if plot_choices["n_ranks"]:
    nrows = plot_choices["n_ranks/n"]

    fig_best_timeseries, ax_best_timeseries = plt.subplots(
        nrows=nrows, figsize=(10, 10), sharex=True
    )
    for row_idx in range(nrows):
        score_idx = row_idx
        timeseries = timeseries_df.loc[scores_sorted.index[score_idx]]
        ax_best_timeseries[row_idx].plot(timeseries)
        ax_best_timeseries[row_idx].xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax_best_timeseries[row_idx].set_title(f"Rank {score_idx+1}")
    fig_best_timeseries.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time (min)")
    plt.ylabel("Flavin fluorescence, normalised (AU)")

    fig_best_pdgram, ax_best_pdgram = plt.subplots(
        nrows=nrows,
        figsize=(5, 10),
        sharex=True,
    )
    for row_idx in range(nrows):
        score_idx = row_idx
        freqs_ax_best_pdgramis = freqs.loc[scores_sorted.index[score_idx]]
        power_ax_best_pdgramis = power.loc[scores_sorted.index[score_idx]]
        ax_best_pdgram[row_idx].plot(freqs_ax_best_pdgramis, power_ax_best_pdgramis)
        ax_best_pdgram[row_idx].set_ylim((0, 50))
        ax_best_pdgram[row_idx].set_title(f"Rank {score_idx+1}")
    fig_best_pdgram.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r"Frequency ($\mathrm{min}^{-1}$)")
    plt.ylabel("Power (dimensionless)")

    fig_worst_timeseries, ax_worst_timeseries = plt.subplots(
        nrows=nrows,
        figsize=(10, 10),
        sharex=True,
    )
    for row_idx in range(nrows):
        score_idx = len(scores_sorted) - 1 - row_idx
        timeseries = timeseries_df.loc[scores_sorted.index[score_idx]]
        ax_worst_timeseries[row_idx].plot(timeseries)
        ax_worst_timeseries[row_idx].xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax_worst_timeseries[row_idx].set_title(f"Rank {score_idx+1}")
    fig_worst_timeseries.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time (min)")
    plt.ylabel("Flavin fluorescence, normalised (AU)")

    fig_worst_pdgram, ax_worst_pdgram = plt.subplots(
        nrows=nrows,
        figsize=(5, 10),
        sharex=True,
    )
    for row_idx in range(nrows):
        score_idx = len(scores_sorted) - 1 - row_idx
        freqs_ax_worst_pdgramis = freqs.loc[scores_sorted.index[score_idx]]
        power_ax_worst_pdgramis = power.loc[scores_sorted.index[score_idx]]
        ax_worst_pdgram[row_idx].plot(freqs_ax_worst_pdgramis, power_ax_worst_pdgramis)
        ax_worst_pdgram[row_idx].set_ylim((0, 50))
        ax_worst_pdgram[row_idx].set_title(f"Rank {score_idx+1}")
    fig_worst_pdgram.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r"Frequency ($\mathrm{min}^{-1}$)")
    plt.ylabel("Power (dimensionless)")

# ROC
if plot_choices["roc"]:
    # Compute number of independent frequencies (M)
    time_axis = timeseries_dropna.columns.to_list()
    time_axis = [float(time) for time in time_axis]
    sampling_pd = time_axis[1] - time_axis[0]
    l_ts = time_axis[-1] - time_axis[0]
    f_lb = 1 / l_ts
    f_ub = 0.5 * (1 / sampling_pd)
    M = f_ub * l_ts
    # False positive rates & True positive rates
    FPR_axis = []
    TPR_axis = []
    vec = plot_choices["roc/fdr_vec"]
    for q in vec:
        classifications_df = fdr_classify(scores, q)
        true_labels = labels_df.loc[classifications_df.index].score.to_list()
        predicted_labels = classifications_df.classification.to_list()
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        (
            true_negative,
            false_positive,
            false_negative,
            true_positive,
        ) = conf_matrix.ravel()
        FPR = false_positive / (false_positive + true_negative)
        TPR = true_positive / (true_positive + false_negative)
        FPR_axis.append(FPR)
        TPR_axis.append(TPR)
    # Draw ROC curve
    fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
    ax_roc.plot(FPR_axis, TPR_axis, marker="o")
    ax_roc.plot([0, 1], [0, 1])
    ax_roc.set_xlim((0, 1))
    ax_roc.set_ylim((0, 1))
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")

# Save figures
pdf_filename = "../reports/glynn_" + data_options["experimentID"] + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
