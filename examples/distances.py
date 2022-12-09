#!/usr/bin/env python3

from openset.data.simulated import cluster
from openset.models import Correlation
from openset.models import Cosine
from openset.models import KNearestNeighbors
from openset.models import LocalOutlierFactor
from openset.models import Mahalanobis
from openset.models import Manhattan
from openset.models import Minkowski
from openset.models import SEuclidean


samples = 10
dimension = 3


def main():
    training = cluster(samples=samples, dimension=dimension, location=0)
    typicals = cluster(samples=5, dimension=dimension, location=0)
    outliers = cluster(samples=2, dimension=dimension, location=2)

    models = [
        Correlation,
        Cosine,
        KNearestNeighbors,
        LocalOutlierFactor,
        Mahalanobis,
        Manhattan,
        Minkowski,
        SEuclidean,
    ]

    for model_class in models:
        model = model_class()
        model.fit(training)
        known = model.score(typicals)
        unknown = model.score(outliers)

        print(model, known, unknown)


if __name__ == '__main__':
    main()
