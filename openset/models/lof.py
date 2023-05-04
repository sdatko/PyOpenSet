#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors import LocalOutlierFactor as scikit_lof

from ..models.base import BaseModel


class LocalOutlierFactor(BaseModel):
    '''Local outlier factor distance.'''

    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors

    def __repr__(self):
        return f'{self.__class__.__name__}({self.n_neighbors})'

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.classifiers = dict()

        for label in self.labels:
            self.classifiers[label] = scikit_lof(n_neighbors=self.n_neighbors,
                                                 novelty=True)
            self.classifiers[label].fit(self.X[self.y == label])

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        # NOTE(sdatko): This function returns negative numbers, assuming
        #               bigger values (i.e. closer to 0 [zero]) as inliers.
        #               The opposite of it can be used as a distance measure.
        distances = -1 * self.classifiers[y].score_samples(X)

        return distances
