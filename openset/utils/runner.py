#!/usr/bin/env python3

from multiprocessing import Pool

from psutil import cpu_count
from tqdm import tqdm


class Runner(object):
    '''Tool for parallelizing function calls with multiple arguments.'''

    def __init__(self, nproc=None):
        '''Runner initializer.

        Allows to specify the number of concurrent processes to run in a pool
        (equal to number of all physical cores by default).
        '''
        if nproc:
            self.nproc = nproc
        else:
            self.nproc = cpu_count(logical=False)

    def run(self, function, arguments, unpack=False):
        '''Main runner method.

        Launches the given function in a pool of processes. Each function call
        receives a single argument from the given arguments collection.
        Optionally, if arguments is a collection of collections, it is possible
        to unpack the arguments for each call.

        Note that this runner ignores the given function's returned values.
        It is expected that the function will save its result in another way
        (as a file, database entry, using shared memory, etc.).
        '''
        if hasattr(arguments, '__len__'):
            total = len(arguments)
        else:
            total = None

        #
        # Pool creation
        #
        with Pool(processes=self.nproc) as pool:
            if unpack:
                self.func = function
                results = pool.imap_unordered(
                    func=self._starmap,
                    iterable=arguments,
                )
            else:
                results = pool.imap_unordered(
                    func=function,
                    iterable=arguments,
                )

            #
            # NOTE(sdatko): Here we wait for the pool to process all arguments.
            #               We are not interested in any of the returned values
            #               at this point, so we do nothing particular here.
            #               The tqdm is used only for progress tracking.
            #
            for _ in tqdm(results, total=total):
                pass

    def _starmap(self, args):
        '''Helper method to call a defined function with unpacked arguments.'''
        return self.func(*args)
