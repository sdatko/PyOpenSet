#!/usr/bin/env python3
#
# NOTE(sdatko): The noqa entries for F401 code (module imported but unused)
#               are here because all the imported package modules are actually
#               really used, just indirectly inside the load_tests() function
#               and flake8 does not recognize that.
#

import doctest
import sys

import openset.data  # noqa: F401
import openset.models  # noqa: F401
import openset.utils  # noqa: F401


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite())

    for module in sys.modules:
        if module.startswith(__package__.split('.', maxsplit=1)[0]):
            tests.addTests(doctest.DocTestSuite(module))

    return tests
