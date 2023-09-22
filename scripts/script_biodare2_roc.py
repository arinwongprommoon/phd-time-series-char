#!/usr/bin/env python3
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
    "experimentID": "is20016",
    # Group (strain) ID for first group
    "group1": "by4741",
}

plot_choices = {
    # ROC curve
    "roc": True,
    # False discovery rates
    "roc/fdr_vec": np.power(10, np.linspace(-12, 0, 100)),
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


# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def polygon_area(x, y):
    """Compute polygon area based on shoelace formula"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# Load data
data_dir = "../data/raw/"
group1_name = data_options["experimentID"] + "_" + data_options["group1"]

filepath1 = data_dir + group1_name
timeseries1_filepath = filepath1 + "_timeseries.csv"
labels1_filepath = filepath1 + "_labels.csv"

timeseries_df = pd.read_csv(timeseries1_filepath, index_col=[0, 1, 2])
timeseries_dropna = timeseries_df.dropna()

labels_df = pd.read_csv(labels1_filepath, index_col=[0, 1, 2])
labels_df = labels_df == 1
labels_df = labels_df.loc[timeseries_dropna.index]

# Load rhythmicity detection results
rhythm_dir = "../data/interim/"
rhythm_filepath = rhythm_dir + "BioDare_" + group1_name + "_rhythmicity_copy.csv"
rhythm_df_full = pd.read_csv(rhythm_filepath)
rhythm_df = rhythm_df_full[["Data Id", "emp p BH Corrected"]]

# Compute periodograms
freqs, power = fft.as_function(timeseries_df)
freqs = freqs.dropna()
power = power.dropna()

# Compute scores (ranks)
# Scores for each time series is simply the max height of power
scores = power.max(axis=1)
scores_sorted = scores.sort_values(ascending=False)

# ROC
if plot_choices["roc"]:
    true_labels = labels_df.score.to_list()
    # False positive rates & True positive rates
    FPR_axis = []
    TPR_axis = []
    vec = plot_choices["roc/fdr_vec"]
    for q in vec:
        classifications = rhythm_df[["emp p BH Corrected"]].values.ravel() < q
        predicted_labels = classifications.tolist()
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
    # Calculate area under ROC
    x_axis = np.array([0] + FPR_axis + [1])
    y_axis = np.array([0] + TPR_axis + [1])
    auroc = polygon_area(x_axis, y_axis)
    auroc += 0.5
    print(f"Area under ROC = {auroc}")

# Save figures
pdf_filename = (
    "../reports/biodare_roc_"
    + data_options["experimentID"]
    + "_"
    + data_options["group1"]
    + ".pdf"
)
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
