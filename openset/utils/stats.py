#!/usr/bin/env python3

import numpy as np


def deciles(data: np.ndarray) -> np.ndarray:
    return np.percentile(data, tuple(range(0, 101, 10)))


def percentiles(data: np.ndarray) -> np.ndarray:
    return np.percentile(data, tuple(range(101)))


def quartiles(data: np.ndarray) -> np.ndarray:
    return np.percentile(data, (0, 25, 50, 75, 100))
