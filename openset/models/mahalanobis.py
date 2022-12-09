#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import mahalanobis

from ..models.base import BaseModel


class Mahalanobis(BaseModel):
    '''Mahalanobis distance.'''

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.means = dict()
        self.icovs = dict()

        for label in self.labels:
            data = self.X[self.y == label]
            cov = np.cov(data.T)

            if not cov.shape:  # hack for 1-dimensional data
                cov.shape = (1, 1)

            self.means[label] = data.mean(axis=0)
            self.icovs[label] = np.linalg.inv(cov)

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        distances = np.array([mahalanobis(vec, self.means[y], self.icovs[y])
                              for vec in X])

        return distances


class MahalanobisSC(BaseModel):
    '''Mahalanobis distance with shared covariance matrix.'''

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.means = dict()
        self.icov = None

        cov = np.cov(self.X.T)

        if not cov.shape:  # hack for 1-dimensional data
            cov.shape = (1, 1)

        self.icov = np.linalg.inv(cov)

        for label in self.labels:
            self.means[label] = self.X[self.y == label].mean(axis=0)

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        distances = np.array([mahalanobis(vec, self.means[y], self.icov)
                              for vec in X])

        return distances
