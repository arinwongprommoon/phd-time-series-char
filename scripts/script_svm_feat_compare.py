#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.ml.transformers import Catch22Transformer, FFTTransformer, NullTransformer
from src.ml.predict import get_predictions, get_predictproba
from src.ml.metrics import StratifiedKFoldHandler

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data_options = {
    # Experiment ID.
    # Prefix 'is' for islay or 'st' for staffa.
    # ID is 5 digits, add leading zeros as appropriate.
    "experimentID": "is20016",
    # Group (strain) ID for first group
    "group1": "zwf1egf",
}

feat_options = {
    # Whether to include a control in which labels are randomised
    "random_control": True,
    # Dict. Keys: name of transformer (for plotting), values: transformer object
    "transformers": {
        "Time series": NullTransformer(),
        "Fourier spectrum": FFTTransformer(),
        "catch22": Catch22Transformer(),
    },
}

model_options = {
    # Hyperparameters for SVC
    "C": 10.0,
    "gamma": "auto",
    # Train-test split
    "tt_split": 0.75,
    # Number of splits for k-fold coss-validation
    "kfold_nsplits": 5,
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
    random_state=69,
)


for transformer in feat_options["transformers"].values():
    # Construct pipeline
    binary_pipeline = Pipeline(
        [
            ("featurise", transformer),
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=model_options["C"],
                    gamma=model_options["gamma"],
                ),
            ),
        ]
    )

    kfold = StratifiedKFoldHandler(
        binary_pipeline, features, targets, model_options["kfold_nsplits"]
    )
    kfold.pretty_print()
