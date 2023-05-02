#!/usr/bin/env python3

from openset.data.generator import ClusterGenerator


dimension = 3
samples = 5


def main():
    generator = ClusterGenerator()
    generator.reset(42)

    data1 = generator.gaussian(samples, dimension)
    data2 = generator.triangular(samples, dimension)
    data3 = generator.uniform(samples, dimension)
    data4 = generator.mvn(samples, dimension,
                          location=5.0, scale=1.0,
                          n_features=0.75, n_correlated=0.5, covariance=0.25)

    print(data1)
    print(data2)
    print(data3)
    print(data4)


if __name__ == '__main__':
    main()
