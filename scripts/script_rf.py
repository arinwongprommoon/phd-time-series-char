#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ml.transformers import Catch22Transformer, FFTTransformer, NullTransformer

from sklearn.ensemble import RandomForestClassifier
from postprocessor.core.processes.catch22 import catch22
from sklearn.inspection import permutation_importance


data_options = {
    # Experiment ID.
    # Prefix 'is' for islay or 'st' for staffa.
    # ID is 5 digits, add leading zeros as appropriate.
    "experimentID": "is20016",
    # Group (strain) ID for first group
    "group1": "zwf1egf",
}

model_options = {
    "tt_split": 0.75,
}


# Load data
data_dir = "../data/raw/"
group1_name = data_options["experimentID"] + "_" + data_options["group1"]

filepath1 = data_dir + group1_name
timeseries1_filepath = filepath1 + "_timeseries.csv"
labels1_filepath = filepath1 + "_labels.csv"

timeseries_df = pd.read_csv(timeseries1_filepath, index_col=[0, 1, 2])
labels_df = pd.read_csv(labels1_filepath, index_col=[0, 1, 2])

timeseries_dropna = timeseries_df.dropna()


# Manipulate data variables to create data and target matrices
features = timeseries_dropna
targets = labels_df.loc[features.index]

# Train-test split
features_train, features_test, targets_train, targets_test = train_test_split(
    features,
    targets,
    train_size=model_options["tt_split"],
    random_state=42,
)

feature_names = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]

# Manually do the transforms because permutation_importance needs features as input

scaler = StandardScaler()
features_test_preprocessed = scaler.fit_transform(catch22.as_function(features_test))
features_train_preprocessed = scaler.fit_transform(catch22.as_function(features_train))
forest = RandomForestClassifier(random_state=69)
forest.fit(features_train_preprocessed, targets_train.to_numpy().ravel())
# Get feature importance using feature permutation
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#feature-importance-based-on-feature-permutation
result = permutation_importance(
    forest,
    features_test_preprocessed,
    targets_test,
    n_repeats=10,
    random_state=69,
    n_jobs=2,
)

forest_importances = pd.Series(result.importances_mean, index=feature_names)

argsorts = forest_importances.argsort().to_numpy()
argsorts = argsorts[::-1]
# Sorted
fig, ax = plt.subplots(figsize=(7, 7))
forest_importances.iloc[argsorts].plot.bar(yerr=result.importances_std[argsorts], ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()

# Save figures
pdf_filename = "../reports/rf.pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
