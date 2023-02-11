#!/usr/bin/env python3

import itertools

from openset.experiments.distributions import Generated
from openset.models import Euclidean
from openset.models import Manhattan
from openset.models import SEuclidean
from openset.utils.runner import Runner


dimensions = (
    range(1, 10),
)

distances = (
    2,
    3,
)

distributions = (
    'gaussian',
    'uniform',
)

training_samples = (
    range(100, 1000, 100),
    (1000, ),
)

models = [
    Euclidean(),
    Manhattan(),
    SEuclidean(),
]

iterations = 2


def main():
    iterator = itertools.product(
        itertools.chain(*dimensions),
        distances,
        distributions,
        models,
        itertools.chain(*training_samples),
        range(iterations),
    )

    runner = Runner()
    experiment = Generated(cached=True)

    runner.run(experiment.get, tuple(iterator), unpack=True)


if __name__ == '__main__':
    main()
