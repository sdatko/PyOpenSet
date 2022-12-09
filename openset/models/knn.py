#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors import KNeighborsTransformer as scikit_kNN

from ..models.base import BaseModel


class KNearestNeighbors(BaseModel):
    '''K nearest neighbors distance.'''

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.classifiers = dict()

        for label in self.labels:
            self.classifiers[label] = scikit_kNN(n_neighbors=5)
            self.classifiers[label].fit(self.X[self.y == label])

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        distances, _ = self.classifiers[y].kneighbors(X)

        return distances.mean(axis=1)
