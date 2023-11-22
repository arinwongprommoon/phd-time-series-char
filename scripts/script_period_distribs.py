#!/usr/bin/env python3

import numpy as np
import pandas as pd

from postprocessor.core.multisignal.crosscorr import crosscorr, crosscorrParameters
from postprocessor.core.processes.findpeaks import findpeaks, findpeaksParameters
from scipy.stats import sem, ttest_ind

data_options = {
    # Format: XXNNNNN_SSSSS
    # - X: server abbrevation ('is' or 'st')
    # - N: experiment ID, 5-digit, with leading zeros
    # - S: strain name
    # Experiment to use as reference for test of difference of means
    "ref_group": "is26643_htb2mCherry",
    # List of experiments to loop through.
    "list_groups": [
        "is26643_htb2mCherry",
        "is27917_htb2mCherry",
        "is19972_htb2mCherry",
        "st00491_by4741",
        "is20212_cenpkkoetter",
        "is31594_htb2mCherry",
        "is31492_htb2mCherry",
        "st00613_htb2mCherry",
        "st00409_zwf1egf",
        "st01649_tsa1tsa2morgan",
        "is20016_zwf1egf",
    ],
}

param_options = {
    # Parameters for findpeaks, used on ACFs
    "prominence": 0.20,
    "width": 4,
}

data_dir = "../data/raw/"


def get_first_interval(x):
    """get interval lengths then get just the first one"""
    list_intervals = np.diff(np.where(np.array(x) > 0))[0]
    # Checks that it is not empty
    if list_intervals.any():
        return list_intervals[0]
    else:
        return np.nan


# Convenience function to functionalise loop
# Note: this breaks several code good practise methods
def estimate_periods(group_name):
    """use acf to estimate periods of timeseries in a dataset"""
    # Load data
    filepath = data_dir + group_name
    timeseries_filepath = filepath + "_flavin_timeseries.csv"
    labels_filepath = filepath + "_labels.csv"

    timeseries_df = pd.read_csv(timeseries_filepath, index_col=[0, 1, 2])
    labels_df = pd.read_csv(labels_filepath, index_col=[0, 1, 2])

    # Select data
    # Drop NaNs
    timeseries_dropna = timeseries_df.dropna()
    num = timeseries_dropna.shape[0]
    labels_dropna = labels_df.loc[timeseries_dropna.index]
    # Select oscillatory time series
    timeseries_osc = timeseries_dropna.loc[
        labels_dropna[labels_dropna.score == 1].index
    ]

    # Estimate period, using ACF
    acfs = crosscorr.as_function(timeseries_osc, normalised=True, only_pos=True)
    acfs_peaks = findpeaks.as_function(
        acfs, prominence=param_options["prominence"], width=param_options["width"]
    )
    periods = acfs_peaks.apply(lambda x: get_first_interval(x), axis=1)
    periods_min = 5 * periods.to_numpy()
    # Drop NaNs
    periods_min = periods_min[~np.isnan(periods_min)]

    return num, periods_min


num_ref, periods_min_ref = estimate_periods(group_name=data_options["ref_group"])

for group_name in data_options["list_groups"]:
    num, periods_min = estimate_periods(group_name=group_name)

    # Compute statistics
    num_osc = len(periods_min)

    mean = np.mean(periods_min)
    std_err_mean = sem(periods_min)

    median = np.median(periods_min)
    q25, q75 = np.percentile(periods_min, [25, 75])

    t, p = ttest_ind(periods_min_ref, periods_min)

    # Print statistics
    print(f"Group: {group_name}")
    print(f"n = {num}; n(osc) = {num_osc} ({100*num_osc/num:.2f}%).")
    print(f"mean = {mean:.2f}; SEM = {std_err_mean:.2f}.")
    print(f"median = {median:.2f}; IQR = {q25:.2f}--{q75:.2f} (diff = {q75-q25:.2f}).")
    print(f"t = {t:.2f}; p = {p}")
    print("\n")
