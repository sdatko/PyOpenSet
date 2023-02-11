#!/usr/bin/env python3

from abc import ABC
from abc import abstractmethod


class BaseExperiment(ABC):
    '''Abstract class for an experiment.'''

    def __init__(self, cached=False):
        self._cached = cached

    def __repr__(self):
        return f'{self.__class__.__name__}'

    # Hint: simply decorating _cache() with MemCache() and returning here
    #       self._get(*args, **kwargs) may be good enough implementation
    @abstractmethod
    def _cache(self, *args, **kwargs):
        # The experiment-specific implementation should come here
        raise NotImplementedError

    @abstractmethod
    def _get(self, *args, **kwargs):
        # The experiment-specific implementation should come here
        raise NotImplementedError

    def get(self, *args, **kwargs):
        if self._cached:
            return self._cache(*args, **kwargs)
        else:
            return self._get(*args, **kwargs)
