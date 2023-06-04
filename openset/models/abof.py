#!/usr/bin/env python3

from itertools import combinations

import numpy as np
from sklearn.neighbors import KDTree

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
                norm1_squared = vec1.dot(vec1)
                norm2_squared = vec2.dot(vec2)

                if norm1_squared == 0 or norm2_squared == 0:
                    continue

                angles.append(vec1.dot(vec2) / norm1_squared / norm2_squared)

            variances.append(np.var(angles, axis=0))

        # NOTE(sdatko): For convenience, we want outliers to have higher
        #               numerical values than inliers, but this function
        #               does the opposite, hence we multiply the result
        #               by -1 to just invert the axis
        return -1 * np.array(variances)


class FastAngleBasedOutlierFactor(BaseModel):
    '''Approximated angle-based outlier factor distance.'''

    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors

    def __repr__(self):
        return f'{self.__class__.__name__}({self.n_neighbors})'

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.data = dict()
        self.tree = dict()

        for label in self.labels:
            self.data[label] = self.X[self.y == label]
            self.tree[label] = KDTree(self.X[self.y == label])

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)
        variances = []

        for vec in X:
            neighbors = self.tree[y].query((vec, ),
                                           k=self.n_neighbors,
                                           return_distance=False)

            angles = []
            vectors = self.data[y][*neighbors] - vec

            for vec1, vec2 in combinations(vectors, 2):
                norm1_squared = vec1.dot(vec1)
                norm2_squared = vec2.dot(vec2)

                if norm1_squared == 0 or norm2_squared == 0:
                    continue

                angles.append(vec1.dot(vec2) / norm1_squared / norm2_squared)

            variances.append(np.var(angles, axis=0))

        # NOTE(sdatko): For convenience, we want outliers to have higher
        #               numerical values than inliers, but this function
        #               does the opposite, hence we multiply the result
        #               by -1 to just invert the axis
        return -1 * np.array(variances)
