#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import seuclidean

from ..models.base import BaseModel


class SEuclidean(BaseModel):
    '''Standardized Euclidean distance.'''

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.means = dict()
        self.vars = dict()

        for label in self.labels:
            self.means[label] = self.X[self.y == label].mean(axis=0)
            self.vars[label] = self.X[self.y == label].var(axis=0)

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        distances = np.array([seuclidean(vec, self.means[y], self.vars[y])
                              for vec in X])

        return distances
