#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import PrecisionRecallDisplay, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ml.predict import get_predictproba
from src.ml.transformers import Catch22Transformer, FFTTransformer, NullTransformer

data_options = {
    # Experiment ID.
    # Prefix 'is' for islay or 'st' for staffa.
    # ID is 5 digits, add leading zeros as appropriate.
    "experimentID": "is20016",
    # Group (strain) ID for first group
    "group1": "zwf1egf",
}

model_options = {
    # Transformer object to choose as featurisation
    "transformer": Catch22Transformer(),
    # Scramble scores, as a control
    "scramble": True,
    # Hyperparameters for SVC
    "C": 10.0,
    "gamma": "auto",
    # Train-test split
    "tt_split": 0.75,
}

plot_options = {
    "precision_recall": True,
    "predict_proba_hist": True,
    "predict_proba_gallery": True,
    "predict_proba_gallery/list_probs": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
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
if model_options["scramble"]:
    # Scramble scores
    targets = targets.sample(frac=1, random_state=69)

# Train-test split
features_train, features_test, targets_train, targets_test = train_test_split(
    features,
    targets,
    train_size=model_options["tt_split"],
    random_state=42,
)

# Construct pipeline
binary_pipeline = Pipeline(
    [
        ("featurise", model_options["transformer"]),
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                C=model_options["C"],
                gamma=model_options["gamma"],
                probability=True,
            ),
        ),
    ]
)

# Predict and get results
binary_pipeline.fit(features_train, targets_train.to_numpy().ravel())
true_targets = targets_test.to_numpy().ravel()
predicted_targets = binary_pipeline.predict(features_test)

# Predict probability
predictproba_df = get_predictproba(binary_pipeline, features_test)


if plot_options["precision_recall"]:
    # Precision-recall curve
    precision = precision_score(true_targets, predicted_targets)
    recall = recall_score(true_targets, predicted_targets)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    y_score = binary_pipeline.decision_function(features_test)
    y_test = true_targets

    fig_pr, ax_pr = plt.subplots()
    display = PrecisionRecallDisplay.from_predictions(
        y_test, y_score, name="LinearSVC", plot_chance_level=True, ax=ax_pr
    )
    ax_pr.set_title("2-class Precision-Recall curve")


if plot_options["predict_proba_hist"]:
    # Histogram
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(
        predictproba_df,
        x=1,
        binwidth=0.05,
        ax=ax_hist,
    )
    ax_hist.set_xlim((-0.05, 1.05))
    ax_hist.set_xlabel("Probability of oscillation")


if plot_options["predict_proba_gallery"]:
    # Gallery
    predictproba_sorted = predictproba_df.sort_values(by=1)
    # Get list of probabilities of oscillation (category '1')
    proba_array = predictproba_sorted.to_numpy()[:, 1].ravel()

    list_probs = plot_options["predict_proba_gallery/list_probs"]

    nrows = len(list_probs)
    fig_gallery, ax_gallery = plt.subplots(
        nrows=nrows,
        figsize=(10, 2 * nrows),
        sharex=True,
    )

    for row_idx, prob in enumerate(list_probs):
        prob_idx = np.searchsorted(proba_array, prob)
        # Deal with edge case
        if prob_idx >= len(proba_array):
            prob_idx = len(proba_array) - 1
        # Print actual probability
        actual_prob = proba_array[prob_idx]
        # Get cell index
        cell_idx = predictproba_sorted.index[prob_idx]
        # Get time series
        timeseries = timeseries_df.loc[cell_idx]
        # Draw
        ax_gallery[row_idx].plot(timeseries)
        ax_gallery[row_idx].xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax_gallery[row_idx].set_title(f"Oscillation probability = {actual_prob:.2f}")

    fig_gallery.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time (min)")
    plt.ylabel("Flavin fluorescence, normalised (AU)")

    # Save figures
pdf_filename = "../reports/svm.pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
