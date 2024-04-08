#!/usr/bin/env python3

import itertools
from time import sleep

from openset.utils.runner import Runner


def multiply_v1(args):
    sleep(1)
    return args[0], args[1], args[0] * args[1]


def multiply_v2(x, y):
    sleep(1)
    return x, y, x * y


def display(result):
    x, y, x_times_y = result
    print(f'{x} * {y} = {x_times_y}')


def main():
    args1 = (1, 2, 3, 4)
    args2 = (5, 6, 7)
    arguments = itertools.product(args1, args2)

    runner = Runner(4)
    runner.run(multiply_v1, arguments, display, unpack=False)

    arguments = itertools.product(args1, args2)
    total = len(args1) * len(args2)

    runner = Runner(4)
    runner.set_function(multiply_v2)
    runner.set_arguments(arguments)
    runner.set_handler(display)
    runner.set_length(total)
    runner.run(unpack=True)


if __name__ == '__main__':
    main()
