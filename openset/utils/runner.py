#!/usr/bin/env python3

from multiprocessing import Pool
from numbers import Number

from psutil import cpu_count
from tqdm import tqdm


class Runner(object):
    '''Tool for parallelizing function calls with multiple arguments.

    Launches the given function in a pool of processes. Each function call
    receives a single element from the given function arguments collection.
    '''

    def __init__(self, nproc=None):
        '''Runner initializer.

        Allows to specify the number of concurrent processes to run in a pool
        (equal to number of all physical cores by default).
        '''
        self.function = None
        self.arguments = None
        self.handler = None
        self.nproc = None
        self.length = None

        self.set_nproc(nproc)

    def set_nproc(self, nproc=None):
        '''Specify the number of concurrent processes to run in the pool.

        Equal to the number of all physical CPU cores by default.
        '''
        if nproc:
            self.nproc = nproc
        else:
            self.nproc = cpu_count(logical=False)

    def set_function(self, function):
        '''Specify the main function to run in parallel in the pool.

        The function must take at least a single non-optional argument.
        '''
        self.function = function

    def set_arguments(self, arguments):
        '''Specify the collection of arguments for the function calls.

        The arguments must be an iterable collection or object.
        '''
        try:
            iter(arguments)
        except TypeError:
            raise TypeError('Arguments must be iterable')
        else:  # when no exception occurred
            self.arguments = arguments

    def set_handler(self, handler):
        '''Specify the function used to process the results.

        When the specified function to run in parallel returns any values,
        this handler may be used to process these outcomes one-by-one.
        If unspecified, the main function's returned values are ignored
        and main function is expected to save its result in another way
        (as a file, database entry, using shared memory, etc.).

        NOTE: handler function always takes a single argument.
        '''
        self.handler = handler

    def set_length(self, length):
        '''Specify the number of arguments used as a reference value for tqdm.

        It is useful in case of arguments passed as an iterator or a generator
        that has unknown length until it is eventually traversed (which is not
        effective in case of long collections).
        '''
        self.length = length

    def run(self, function=None, arguments=None,
            handler=None, unpack=False, length=None):
        '''Main runner method – creates a pool of processes.

        Optionally, if arguments is a collection of collections, such as
        a list of tuples, it is possible to unpack the arguments for each
        function call by setting the `unpack` keyword argument to `True`.
        '''
        if function:
            self.set_function(function)
        if arguments:
            self.set_arguments(arguments)
        if handler:
            self.set_handler(handler)
        if length:
            self.set_length(length)

        if not self.function:
            raise ValueError('Function to run must be provided')
        if not self.arguments:
            raise ValueError('Arguments to process must be provided')

        #
        # Pool creation
        #
        with Pool(processes=self.nproc) as pool:
            results = pool.imap_unordered(
                func=(self._starmap if unpack else self.function),
                iterable=self.arguments,
            )

            #
            # Post actions
            #
            # NOTE(sdatko): Here we wait for the pool to process all arguments.
            #
            self._handle(results)

    def _handle(self, results):
        '''The helper function to process the main function calls results.'''
        if hasattr(self.arguments, '__len__'):
            total = len(self.arguments)
        elif isinstance(self.length, Number):
            total = self.length
        else:
            total = None

        if self.handler:
            for result in tqdm(results, total=total):
                self.handler(result)
        else:
            for _ in tqdm(results, total=total):
                pass

    def _starmap(self, packed_arguments):
        '''Helper method to call a defined function with unpacked arguments.'''
        return self.function(*packed_arguments)
