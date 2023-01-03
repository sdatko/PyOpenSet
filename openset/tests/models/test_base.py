#!/usr/bin/env python3

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from openset.models.base import BaseModel


class TestBaseModel(TestCase):
    # Helper inner class
    class CustomModel(BaseModel):
        def fit(self, X, y=None):
            super().fit(X, y)

        def score(self, X, y=None):
            super().score(X, y)

    def test_base_model_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseModel()

    @patch.object(BaseModel, 'score')
    @patch.object(BaseModel, 'fit')
    @patch.object(BaseModel, '__init__', return_value=None)
    def test_aliases_and_inheritance(self, base_init, base_fit, base_score):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 1])

        model = self.CustomModel()
        model.train(X, y)
        model.test(X, y)

        base_init.assert_called_once()
        base_fit.assert_called_once()
        base_score.assert_called_once()

        self.assertEqual(str(model), 'CustomModel')

    def test_fit_with_labels(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 1])

        model = self.CustomModel()
        model.fit(X, y)  # No exception should raise here

        np.testing.assert_array_equal(model.X, X)
        np.testing.assert_array_equal(model.y, y)
        self.assertEqual(model.labels, y[0])

    def test_fit_without_labels(self):
        X = np.array([[1, 2], [3, 4]])

        model = self.CustomModel()
        model.fit(X)  # No exception should raise here

        np.testing.assert_array_equal(model.X, X)
        np.testing.assert_array_equal(model.y, np.array([None, None]))
        self.assertEqual(model.labels, np.array([None]))

    def test_fit_incorrect_labels(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2, 3])

        model = self.CustomModel()
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_score_with_label(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 1])

        model = self.CustomModel()
        model.fit(X, y)
        model.score(X, 1)  # No exception should raise here

    def test_score_without_label(self):
        X = np.array([[1, 2], [3, 4]])

        model = self.CustomModel()
        model.fit(X)
        model.score(X)  # No exception should raise here

    def test_score_incorrect_label(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 1])

        model = self.CustomModel()
        model.fit(X, y)
        with self.assertRaises(ValueError):
            model.score(X, None)
