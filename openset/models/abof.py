#!/usr/bin/env python3

from itertools import combinations

import numpy as np

from ..models.base import BaseModel


class AngleBasedOutlierFactor(BaseModel):
    '''Angle-based outlier factor distance.'''

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.data = dict()

        for label in self.labels:
            self.data[label] = self.X[self.y == label]

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)
        variances = []

        for vec in X:
            values = []

            for vec1, vec2 in combinations(self.data[y], 2):
                vec1 = vec1 - vec
                vec2 = vec2 - vec
                values.append(vec1.dot(vec2) / vec1.dot(vec1) / vec2.dot(vec2))

            variances.append(np.var(values, axis=0))

        return np.array(variances)
