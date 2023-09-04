#!/usr/bin/env python3

"""Routines to evaluate classifiers and visualise results."""
import numpy as np
from sklearn.metrics import auc, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold


class StratifiedKFoldHandler:
    def __init__(self, pipeline, features, targets, n_splits):
        """handler for stratified k-fold

        Parameters
        ----------
        pipeline : scikit-learn pipeline object
            classification pipeline of interest
        features : pandas.DataFrame
            feature matrix
        targets : pandas.DataFrame
            classification targets
        n_splits : int
            number of splits

        Examples
        --------
        FIXME: Add docs.

        """
        self.pipeline = pipeline
        self.features = features
        self.targets = targets
        self.n_splits = n_splits

        kf = StratifiedKFold(n_splits=self.n_splits)
        self.kf_scores = []
        for train_index, test_index in kf.split(features, targets):
            # Split training-testing
            features_train_kf, features_test_kf = (
                features.iloc[train_index],
                features.iloc[test_index],
            )
            targets_train_kf, targets_test_kf = (
                targets.iloc[train_index],
                targets.iloc[test_index],
            )
            # Train & predict
            self.pipeline.fit(features_train_kf, targets_train_kf)
            targets_predicted_kf = self.pipeline.predict(features_test_kf)
            kf_precision = precision_score(
                targets_test_kf, targets_predicted_kf, average="weighted"
            )
            kf_recall = recall_score(
                targets_test_kf, targets_predicted_kf, average="weighted"
            )
            self.kf_scores.append([kf_precision, kf_recall])

    def pretty_print(self):
        """Pretty-prints precision and recall"""
        for kf_score in self.kf_scores:
            print(
                "Precision " + "%.4f" % kf_score[0] + " Recall " + "%.4f" % kf_score[1]
            )

    def barplot(self, ax):
        """Bar plots precision and recall"""
        ax.set_xticks(
            [0 + 1 / (self.n_splits - 1), 1 + 1 / (self.n_splits - 1)],
            ["Precision", "Recall"],
        )
        ax.set_ylabel("Value")
        ax.set_title(str(self.n_splits) + "-fold cross-validation")
        for split in list(range(self.n_splits)):
            x_pos = np.arange(2)
            x_pos = [pos + split / (self.n_splits * 2 - 2) for pos in x_pos]
            ax.bar(x_pos, self.kf_scores[split], width=1 / (self.n_splits * 2 - 2))


class ROCHandler:
    def __init__(self, targets_proba_df, targets_test):
        self.targets_proba_df = targets_proba_df
        self.targets_test = targets_test

        self.scores = targets_proba_df.iloc[:, 1]
        self.false_positive_rate, self.true_positive_rate, _ = roc_curve(
            targets_test, self.scores
        )
        self.auc = auc(self.false_positive_rate, self.true_positive_rate)

    def plot(self, ax):
        ax.set_title("ROC curve")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.plot(self.false_positive_rate, self.true_positive_rate)
