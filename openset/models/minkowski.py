#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import minkowski

from ..models.base import BaseModel


class Minkowski(BaseModel):
    '''Minkowski distance.'''

    def __init__(self, p=2):
        self.p = p

    def __repr__(self):
        return f'{self.__class__.__name__}({self.p})'

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.means = dict()

        for label in self.labels:
            self.means[label] = self.X[self.y == label].mean(axis=0)

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        distances = np.array([minkowski(vec, self.means[y], self.p)
                              for vec in X])

        return distances
