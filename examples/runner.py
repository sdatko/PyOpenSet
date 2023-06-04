#!/usr/bin/env python3

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
    arguments = (
        (2, 6),
        (2, 7),
        (2, 8),
        (3, 9),
        (3, 6),
        (3, 7),
        (4, 8),
        (4, 9),
        (4, 6),
        (5, 7),
        (5, 8),
        (5, 9),
    )

    runner = Runner(4)
    runner.run(multiply_v1, arguments, display, unpack=False)

    runner = Runner(4)
    runner.set_function(multiply_v2)
    runner.set_arguments(arguments)
    runner.set_handler(display)
    runner.run(unpack=True)


if __name__ == '__main__':
    main()
