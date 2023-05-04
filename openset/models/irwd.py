#!/usr/bin/env python3

import numpy as np

from ..models.base import BaseModel


class IntegratedRankWeightedDepth(BaseModel):
    '''Integrated Rank-Weighted depth distance.

    Based on NIPS 2022 paper by P. Colombo et al.,
    "Beyond Mahalanobis-Based Scores for Textual OOD Detection".

    n_proj = number of random direction vectors,
    U = random directions on hypersphere, shape (self.d x self.n_proj),
    M = dot products between training set vectors and U, shape (n x n_proj).

    Element M[i,j] = <X_i, U_j>, dot product of X_i and U_j, where:
    – X_i = i-th row of X,
    – U_j = j-th column of U.

    Formula D_IRW(x, Sn) taken from page 5 of the NIPS paper.
    '''

    def __init__(self, n_proj=1000):
        self.n_proj = n_proj

    def __repr__(self):
        return f'{self.__class__.__name__}({self.n_proj})'

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        super().fit(X, y)
        self.U = dict()
        self.M = dict()

        for label in self.labels:
            dimension = self.X[self.y == label].shape[1]

            mu = np.zeros(dimension)
            cov = np.identity(dimension)

            rng = np.random.default_rng(42)
            U = rng.multivariate_normal(mu, cov, self.n_proj).T

            self.U[label] = U / np.linalg.norm(U, axis=0)  # normalized
            self.M[label] = np.dot(self.X[self.y == label], self.U[label])

        return True

    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        super().score(X, y)

        def D_IRW(vec):
            v = np.dot(vec, self.U[y])
            M_v = self.M[y] - v

            training_samples = M_v.shape[0]

            D_IRW = sum(min((column <= 0).sum(), (column > 0).sum())
                        for column in M_v.T) / training_samples / self.n_proj

            return D_IRW

        distances = np.array([D_IRW(vec) for vec in X])

        # NOTE(sdatko): For convenience, we want outliers to have higher
        #               numerical values than inliers, but this function
        #               does the opposite, hence we multiply the result
        #               by -1 to just invert the axis
        return -1 * distances
