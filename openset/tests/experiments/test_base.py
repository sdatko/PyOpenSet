#!/usr/bin/env python3

from unittest import TestCase
from unittest.mock import patch

from openset.experiments.base import BaseExperiment
from openset.utils.cache import MemCache


cache = MemCache()


class TestBaseModel(TestCase):
    # Helper inner class
    class CustomExperiment(BaseExperiment):
        @cache
        def _cache(self, arg1, arg2):
            super()._cache()  # For tests only
            return self._get(arg1, arg2)

        def _get(self, arg1, arg2):
            super()._get()  # For tests only
            return arg1 + arg2

    def test_base_model_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseExperiment()

    def test_abstract_methods_should_not_be_called(self):
        with self.assertRaises(NotImplementedError):
            experiment = self.CustomExperiment(cached=False)
            experiment.get(arg1=7, arg2=24)

        with self.assertRaises(NotImplementedError):
            experiment = self.CustomExperiment(cached=True)
            experiment.get(arg1=7, arg2=24)

    @patch.object(BaseExperiment, '_cache')
    @patch.object(BaseExperiment, '_get')
    def test_get_uncached(self, base_get, base_cache):
        experiment = self.CustomExperiment(cached=False)
        experiment.get(arg1=7, arg2=24)

        actual = experiment.get(arg1=7, arg2=24)
        expected = 31
        self.assertEqual(actual, expected)

        self.assertEqual(base_get.call_count, 2)
        base_cache.assert_not_called()

    @patch.object(BaseExperiment, '_cache')
    @patch.object(BaseExperiment, '_get')
    def test_get_cached(self, base_get, base_cache):
        experiment = self.CustomExperiment(cached=True)
        experiment.get(arg1=7, arg2=24)

        actual = experiment.get(arg1=7, arg2=24)
        expected = 31
        self.assertEqual(actual, expected)

        base_get.assert_called_once()
        base_cache.assert_called_once()

    def test_repr(self):
        experiment = self.CustomExperiment()
        self.assertEqual(str(experiment), 'CustomExperiment')
