#!/usr/bin/env python3

from openset.data.generator import ClusterGenerator
from openset.models import AngleBasedOutlierFactor
from openset.models import AngleBasedOutlierFactor2
from openset.models import Correlation
from openset.models import Cosine
from openset.models import Euclidean
from openset.models import FastAngleBasedOutlierFactor
from openset.models import FastAngleBasedOutlierFactor2
from openset.models import IntegratedRankWeightedDepth
from openset.models import KNearestNeighbors
from openset.models import LocalOutlierFactor
from openset.models import Mahalanobis
from openset.models import MahalanobisSC
from openset.models import Manhattan
from openset.models import Minkowski
from openset.models import MinMaxOutFactor
from openset.models import MinMaxOutScore
from openset.models import SEuclidean


def main():
    generator = ClusterGenerator()
    training = generator.gaussian(samples=100, dimension=3, location=0)
    typicals = generator.gaussian(samples=5, dimension=3, location=0)
    outliers = generator.gaussian(samples=2, dimension=3, location=2)

    models = [
        AngleBasedOutlierFactor(),
        AngleBasedOutlierFactor2(),
        Correlation(),
        Cosine(),
        Euclidean(),
        FastAngleBasedOutlierFactor(),
        FastAngleBasedOutlierFactor(20),
        FastAngleBasedOutlierFactor2(),
        FastAngleBasedOutlierFactor2(20),
        IntegratedRankWeightedDepth(),
        KNearestNeighbors(),
        KNearestNeighbors(20),
        LocalOutlierFactor(),
        LocalOutlierFactor(20),
        Mahalanobis(),
        MahalanobisSC(),
        Manhattan(),
        Minkowski(),
        Minkowski(3),
        MinMaxOutFactor(),
        MinMaxOutScore(),
        SEuclidean(),
    ]

    for model in models:
        model.fit(training)
        known = model.score(typicals)
        unknown = model.score(outliers)

        print(model, known, unknown)


if __name__ == '__main__':
    main()
