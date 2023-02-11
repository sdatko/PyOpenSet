#!/usr/bin/env python3

from time import sleep
from time import time

from openset.utils.cache import MemCache
from openset.utils.cache import SQLCache


def main(cache):
    print('Using:', cache)

    @cache
    def fibonacci(n: int) -> int:
        sleep(0.2)  # We artificially extend the calculation time

        if n < 2:
            return n

        return fibonacci(n - 1) + fibonacci(n - 2)

    time1 = time()

    # Here we build the cache for 10 different inputs (0..9)
    print('9th Fibonacci:', fibonacci(9))

    time2 = time()

    # Here we reuse the cache and calculate for only 1 new input
    print('10th Fibonacci:', fibonacci(10))

    time3 = time()

    print('First took', time2 - time1, 'seconds')
    print('Second took', time3 - time2, 'seconds')


if __name__ == '__main__':
    cache = MemCache()
    main(cache)
    cache = SQLCache(':memory:')
    main(cache)
