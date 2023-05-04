#!/usr/bin/env python3

import numpy as np

from ..models.base import BaseModel


class MinMaxWindow(BaseModel):
    '''Min-Max Window distance.'''

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.mins = dict()
        self.maxes = dict()

        for label in self.labels:
            self.mins[label] = self.X[self.y == label].min(axis=0)
            self.maxes[label] = self.X[self.y == label].max(axis=0)

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        in_ranges = (self.mins[y] <= X) & (X <= self.maxes[y])
        distances = np.array([vec.sum() / vec.size
                              for vec in in_ranges])

        # NOTE(sdatko): For convenience, we want outliers to have higher
        #               numerical values than inliers, but this function
        #               does the opposite, hence we invert the axis here
        return 1 - distances
