#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import correlation

from ..models.base import BaseModel


class Correlation(BaseModel):
    '''Correlation distance.'''

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.means = dict()

        for label in self.labels:
            self.means[label] = self.X[self.y == label].mean(axis=0)

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        distances = np.array([correlation(vec, self.means[y])
                              for vec in X])

        return distances
