#!/usr/bin/env python3

from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseModel(ABC):
    '''Abstract class for a distance/similarity model.'''

    def __init__(self):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def train(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        '''Alias for fit() method.'''
        return self.fit(X, y)

    def test(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        '''Alias for score() method.'''
        return self.score(X, y)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> bool:
        '''Abstract method for preparing a model of the known data.

        This method shall construct a model, i.e. register all necessary values
        so the distance/similarity can be calculated with the score() method.

        Parameters
        ----------
        X : np.ndarray
            The data cluster, represented as a collection of feature vectors.
            The array shape (N, D) corresponds to N samples of dimension D.
        y : np.ndarray | None, optional
            The labels for feature vectors, given as an array of shape (N, 1).
            If omitted, all elements of X are assigned with same default label.

        Returns
        -------
        result : bool
            The status of model preparation – True for success.

        Examples
        --------
        N/A
        '''
        self.X = X.copy()
        if y is None:
            self.y = np.full((self.X.shape[0]), None)
        else:
            self.y = y.copy()
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError('X and y must have the same first dimension')

        self.labels = np.unique(y)

        pass  # The model-specific implementation should come here

    @abstractmethod
    def score(self, X: np.ndarray, y: object = None) -> np.ndarray:
        '''Abstract method for calculating the distance/similarity.

        This method shall return estimated distance/similarity of the given
        data with respect to the known (trained) model.

        Parameters
        ----------
        X : np.ndarray
            The data cluster, represented as a collection of feature vectors.
            The array shape (N, D) corresponds to N samples of dimension D.
        y : object, optional
            The label of training data to compare with given feature vectors X.
            If omitted, the default label is assumed.

        Returns
        -------
        distances : np.ndarray
            The calculated distance values of a given data cluster with respect
            to the training data set used to prepare a model.

        Examples
        --------
        N/A
        '''
        if y not in self.labels:
            raise ValueError(f'Given category y={y} is not known')

        pass  # The model-specific implementation should come here
