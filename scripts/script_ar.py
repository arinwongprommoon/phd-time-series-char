#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from postprocessor.core.processes.autoreg import autoreg
from scipy.signal import argrelextrema

data_options = {
    # Experiment ID.
    # Prefix 'is' for islay or 'st' for staffa.
    # ID is 5 digits, add leading zeros as appropriate.
    "experimentID": "is20016",
    # Group (strain) ID for first group
    "group1": "zwf1egf",
    # Load periodograms or recompute them
    "load_pdgram": True,
}

plot_choices = {
    # Order vs type of oscillation (Jia & Grima, 2020), split by manual
    # labels of non-oscillatory vs oscillatory categories.
    "order": True,
    # Bar charts of proportions of non-oscillatory/oscillatory time series
    # that have 0, 1, 2, ... local maxima in the analytical power spectra
    "maxima": True,
}


def get_ar_type(power_array):
    local_max_list = argrelextrema(power_array, np.greater)[0]
    local_min_list = argrelextrema(power_array, np.less)[0]
    type = 5
    if len(local_max_list) == 0:
        type = 1
    elif len(local_min_list) == 0:
        type = 4
    elif (power_array[local_max_list] > 1).any():
        type = 3
    elif (power_array[local_max_list] < 1).all():
        type = 2
    else:
        type = 0
    return type


# Load data
data_dir = "../data/raw/"
group1_name = data_options["experimentID"] + "_" + data_options["group1"]

filepath1 = data_dir + group1_name
timeseries1_filepath = filepath1 + "_timeseries.csv"
labels1_filepath = filepath1 + "_labels.csv"

timeseries_df = pd.read_csv(timeseries1_filepath, index_col=[0, 1, 2])
labels_df = pd.read_csv(labels1_filepath, index_col=[0, 1, 2])

timeseries_dropna = timeseries_df.dropna()
# Bodge: For display purposes
labels_df = labels_df.replace(0, "Negative")
labels_df = labels_df.replace(1, "Positive")
# labels_df = labels_df == 1

# Load periodograms
if data_options["load_pdgram"]:
    freqs_df = pd.read_csv("../data/processed/freqs_df.csv", index_col=[0, 1, 2])
    power_df = pd.read_csv("../data/processed/power_df.csv", index_col=[0, 1, 2])
    order_df = pd.read_csv("../data/processed/order_df.csv", index_col=[0, 1, 2])
else:
    freqs_df, power_df, order_df = autoreg.as_function(timeseries_dropna)
    freqs_df.to_csv("../data/processed/freqs_df.csv")
    power_df.to_csv("../data/processed/power_df.csv")
    order_df.to_csv("../data/processed/order_df.csv")

# Get classifications
types = power_df.apply(get_ar_type, axis=1, raw=True)
classifications = types != 1

# Compare: labels, types, order, number of extrema
# Construct dataframe to contain all this information

# Note: this is ugly, inconsistent, and un-Pythonic, but it works.
l = labels_df.loc[classifications.index]
t = types.loc[classifications.index]
local_maxima = []
local_minima = []

for power_spectrum_idx in range(len(power_df)):
    power_array = power_df.iloc[power_spectrum_idx].to_numpy()
    local_max_list = argrelextrema(power_array, np.greater)[0]
    local_min_list = argrelextrema(power_array, np.less)[0]
    local_maxima.append(len(local_max_list))
    local_minima.append(len(local_min_list))

combined_df = pd.concat([l, order_df, t], axis=1)
combined_df.columns = ["Human-defined label", "Order", "Type"]
combined_df["Number of maxima"] = local_maxima

if plot_choices["order"]:
    fig_order, ax_order = plt.subplots()
    sns.boxplot(
        data=combined_df,
        x="Human-defined label",
        y="Order",
        hue="Type",
        ax=ax_order,
    )

if plot_choices["maxima"]:
    fig_maxima, ax_maxima = plt.subplots()
    ct = pd.crosstab(
        combined_df["Human-defined label"],
        combined_df["Number of maxima"],
        normalize="index",
    )
    ct *= 100
    ct.plot(kind="bar", stacked=True, colormap="copper_r", ax=ax_maxima)
    ax_maxima.set_xticklabels(ax_maxima.xaxis.get_majorticklabels(), rotation=45)
    ax_maxima.set_xlabel("Human-defined label")
    ax_maxima.set_ylabel("Percent")

# Save figures
pdf_filename = "../reports/ar_" + data_options["experimentID"] + ".pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
