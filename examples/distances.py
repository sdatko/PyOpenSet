#!/usr/bin/env python3

from openset.data.generator import ClusterGenerator
from openset.models import AngleBasedOutlierFactor
from openset.models import Correlation
from openset.models import Cosine
from openset.models import KNearestNeighbors
from openset.models import LocalOutlierFactor
from openset.models import Mahalanobis
from openset.models import MahalanobisSC
from openset.models import Manhattan
from openset.models import Minkowski
from openset.models import MinMaxWindow
from openset.models import SEuclidean


def main():
    generator = ClusterGenerator()
    training = generator.gaussian(samples=10, dimension=3, location=0)
    typicals = generator.gaussian(samples=5, dimension=3, location=0)
    outliers = generator.gaussian(samples=2, dimension=3, location=2)

    models = [
        AngleBasedOutlierFactor,
        Correlation,
        Cosine,
        KNearestNeighbors,
        LocalOutlierFactor,
        Mahalanobis,
        MahalanobisSC,
        Manhattan,
        Minkowski,
        MinMaxWindow,
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
