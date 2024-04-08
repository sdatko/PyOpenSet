#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors import KNeighborsTransformer as scikit_kNN

from ..models.base import BaseModel


class KNearestNeighbors(BaseModel):
    '''K nearest neighbors distance.'''

    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors

    def __repr__(self):
        return f'{self.__class__.__name__}({self.n_neighbors})'

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.classifiers = dict()

        for label in self.labels:
            n_neighbors = self.n_neighbors

            samples = len(self.X[self.y == label])
            if samples < n_neighbors:
                n_neighbors = samples

            self.classifiers[label] = scikit_kNN(n_neighbors=n_neighbors)
            self.classifiers[label].fit(self.X[self.y == label])

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        distances, _ = self.classifiers[y].kneighbors(X)

        return distances.mean(axis=1)
