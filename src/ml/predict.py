#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

# Wrappers/Convenience functions to get results from a fitted pipeline
# TODO: Add docs


def get_predictions(pipeline, features_test, target_list):
    """convenience function to get a nice dict of predictions

    Parameters
    ----------
    pipeline : fitted sklearn pipeline object
        classifier pipeline to get predictions from
    features_test : array-like
        test set features
    target_list : list-like
        list of possible targets
    """
    targets_predicted = pipeline.predict(features_test)
    predictions_dict = {}
    for class_label in target_list:
        predictions_dict[class_label] = features_test.iloc[
            targets_predicted == class_label
        ].index.to_numpy()
    return predictions_dict


# convenience function to get df that show probabilities that each is in a
# particular category
def get_predictproba(pipeline, features_test):
    targets_proba = pipeline.predict_proba(features_test)
    return pd.DataFrame(targets_proba, index=features_test.index)


# Plotter
class PredictProbaHistogram:
    def __init__(self, category, predictproba_df, n_bins=40):
        self.n_bins = n_bins
        self.category = category
        self.predictproba_df = predictproba_df

        self.xlabel = "Probability of " + self.category
        self.ylabel = "Frequency"
        self.plot_title = "Histogram of probabilities"

    def plot(self, ax):
        ax.set_ylabel(self.ylabel)
        ax.set_xlabel(self.xlabel)
        ax.set_title(self.plot_title)
        ax.hist(self.predictproba_df.iloc[:, 1], self.n_bins)
