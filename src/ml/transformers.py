#!/usr/bin/env python3

# Wrap post-processes into objects that scikit-learn can make a pipeline from

from postprocessor.core.processes.catch22 import catch22, catch22Parameters
from postprocessor.core.processes.fft import fft, fftParameters
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


class FFTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        self.x = x
        self.y = y
        return self

    def transform(self, x):
        fft_runner = fft(fftParameters.default())
        _, power = fft_runner.run(x)
        return power


class NullTransformer(BaseEstimator, TransformerMixin):
    """Use time series as features"""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        self.x = x
        self.y = y
        return self

    def transform(self, x):
        return x
