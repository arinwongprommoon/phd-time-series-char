#!/usr/bin/env python3

# Wrap post-processes into objects that scikit-learn can make a pipeline from

from postprocessor.core.processes.catch22 import catch22, catch22Parameters
from sklearn.base import BaseEstimator, TransformerMixin


class Catch22Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        self.x = x
        self.y = y
        return self

    def transform(self, x):
        catch22_runner = catch22(catch22Parameters.default())
        return catch22_runner.run(x)
