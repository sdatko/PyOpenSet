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
            angles = []
            vectors = self.data[y] - vec

            for vec1, vec2 in combinations(vectors, 2):
                norm1 = vec1.dot(vec1)
                norm2 = vec2.dot(vec2)

                if norm1 == 0 or norm2 == 0:
                    continue

                angles.append(vec1.dot(vec2) / norm1 / norm2)

            variances.append(np.var(angles, axis=0))

        return np.array(variances)
