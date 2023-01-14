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

    print(data1)
    print(data2)
    print(data3)


if __name__ == '__main__':
    main()
