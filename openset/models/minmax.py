#!/usr/bin/env python3

import numpy as np

from ..models.base import BaseModel


class MinMaxOutFactor(BaseModel):
    '''Min-Max Out Factor distance.

    Returns the number fraction of features that are out of typical values,
    determined as axis-wise minimums and maximums from training set,
    e.g. `0.25` means that the 25% of features are out of min-max ranges.
    '''

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
        distances = (~in_ranges).sum(axis=1) / in_ranges.shape[1]

        return distances


class MinMaxOutScore(BaseModel):
    '''Min-Max Out Score distance.

    Returns the sum of standardized distances from the cluster typical values
    (axis-wise mins and maxes) to a given feature vector elements (zero, if all
    values lie within the min-max box/window).
    '''

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.mins = dict()
        self.maxes = dict()
        self.vars = dict()

        for label in self.labels:
            self.mins[label] = self.X[self.y == label].min(axis=0)
            self.maxes[label] = self.X[self.y == label].max(axis=0)
            self.vars[label] = self.X[self.y == label].var(axis=0)

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        distances = (
            (1 / self.vars[y])
            *
            (
                (self.mins[y] - X) * (X < self.mins[y])
                +
                (X - self.maxes[y]) * (X > self.maxes[y])
            )**2
        ).sum(axis=1)**0.5

        return distances
