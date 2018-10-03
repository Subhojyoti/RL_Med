'''
Created on Oct 2, 2018

@author: subhojyotimukherjee
'''

"""
Robust non-linear feature estimation with scikit-learn applied to
the Fourier Transform.
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RANSACRegressor,\
                                 TheilSenRegressor, HuberRegressor

from sklearn.metrics import mean_squared_error


class DFTFeatures(TransformerMixin):
    
    def __init__(self, freqs, *featurizers):
        self.featurizers = featurizers
        self.freqs = freqs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        nfreqs = len(self.freqs)
        """Given a list of original data, return a list of feature vectors."""
        features1 = np.sin(2. * np.pi * self.freqs[None, None, :] * X[:, :,None]).reshape(nsamples, nfeatures * nfreqs)
        features2 = np.cos(2. * np.pi * self.freqs[None, None, 1:] * X[:, :,None]).reshape(nsamples, nfeatures * (nfreqs-1))
        features = np.concatenate([features1, features2], axis=1)
        print(features)
        return features




np.random.seed(42)
X = np.random.uniform(low=-30, high=30, size=400)
x_predict = np.linspace(-25, 25, 1000)
y = np.sin(2 * np.pi * 0.1 * X)
X_test = np.random.uniform(low=-30, high=30, size=200)
y_test = np.sin(2 * np.pi * 0.1 * X_test)

y_errors_large = y.copy()
y_errors_large[::10] = 6

# Make sure that X is 2D
X = X[:, np.newaxis]
X_test = X_test[:, np.newaxis]

freqs = np.fft.rfftfreq(30, d=1.0)
print(freqs)
nfreqs = len(freqs)

estimators = [('Least-Square (DFT)', '-', 'C0',
               LinearRegression(fit_intercept=False)),
              ('Theil-Sen', '>', 'C1', TheilSenRegressor(random_state=42)),
              ('RANSAC', '<', 'C2', RANSACRegressor(random_state=42)),
              ('HuberRegressor', '--', 'C3', HuberRegressor())]

fig, (row1, row2) = plt.subplots(2, 1, figsize=(5, 4))
fig.suptitle('robust Fourier transformations with SKLearn')
row1.plot(X[:, 0], y_errors_large, 'o', ms=5, c='black', label='data points [10% outliers]')


for label, style, color, estimator in estimators:
    model = make_pipeline(DFTFeatures(freqs), estimator)
    model.fit(X, y_errors_large)
    mse = mean_squared_error(model.predict(X_test), y_test)
    y_predicted = model.predict(x_predict[:, None])
    row1.plot(x_predict, y_predicted, style, lw=2, markevery=8, ms=6,
              color=color, label=label + ' E={:2.2g}'.format(mse))

    if hasattr(estimator, 'coef_'):
        spectrum = estimator.coef_[1:nfreqs]**2 + estimator.coef_[nfreqs:]**2
        row2.plot(freqs[1:], spectrum, style, color=color, ms=6, label=label)
row1.legend(loc='upper right', framealpha=0.95)
row1.set(ylim=(-2, 8), xlabel='time [s]', ylabel='amplitude')
row2.set(ylim=(-0.1, 1.5), ylabel='Power', xlabel='frequency [Hz]')
row2.legend(loc='upper right', framealpha=0.95)
plt.show()
