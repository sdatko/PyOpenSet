#!/usr/bin/env python3

from time import sleep

from openset.utils.runner import Runner


def function1(args):
    sleep(1)
    print(args[0] * args[1])


def function2(arg1, arg2):
    sleep(1)
    print(arg1 * arg2)


def main():
    arguments = tuple((x, x ** 2) for x in range(12))

    runner = Runner(4)
    runner.run(function1, arguments)
    runner.run(function2, arguments, unpack=True)


if __name__ == '__main__':
    main()
