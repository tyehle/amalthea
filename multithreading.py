"""
Contains some convenience methods for multithreading jobs.
"""

__author__ = 'tobin'

import logging
import multiprocessing
import itertools
import fcntl
import gc

logger = logging.getLogger(__name__)


class Worker(object):
    """ A pickleable object to store the function to perform. """
    def __init__(self, do_work, failsafe=False):
        """ Make an object that know how to do the work we want done.
            :param do_work: A function that does the work we want done.
            :param failsafe: If true wraps calls to do_work in a try catch to
            prevent every job from dying if one of them encounters an error.
            Any thread that error out will return false, and log the error.
        """
        self.do_work = do_work
        self.failsafe = failsafe

    def __call__(self, args):
        """ Do the work this worker was build for.
            :param args: A dictionary of keyword arguments for the work function
            :return: The result of the work function
        """
        try:
            if self.failsafe:
                try:
                    return self.do_work(**args)
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    logger.exception(e)
                    return False
            else:
                return self.do_work(**args)
        finally:
            gc.collect()


def combinations(**kwargs):
    """ Creates a list of dictionaries with all combinations of the kwargs.

        :param kwargs: Each argument should be a list of possible values.
        :return: A list of dictionaries. The keys are the argument names, and
        the values store every permutation of the given values.

        Examples
        --------
        >>> names = combinations(first_name=['Tom', 'Jim'],last_name=['Smith', 'Glenn'])
        >>> len(names)
        4
        >>> {'first_name': 'Jim', 'last_name': 'Smith'} in names
        True
    """
    names = kwargs.keys()
    values = itertools.product(*kwargs.values())

    params = [dict(zip(names, v)) for v in values]
    return params


def map_kwargs(func, items, failsafe=False):
    """ Map a function over a list of keyword arguments

        Notes:  Do not pass a lambda in as the function.
                May behave strangely in interactive mode.

        :param func: The function to apply.
        :param items: A list of dictionaries containing keyword arguments to
        the function.
        :return: A list of the results of the function

        Examples
        --------
        >>> def f(first_name, last_name):
        ...   return last_name+', '+first_name
        >>> names = combinations(first_name=['Tom', 'Jim'],last_name=['Smith', 'Glenn'])
        >>> all_names = map_kwargs(f, names)
        >>> len(all_names)
        4
        >>> 'Smith, Tom' in all_names
        True
    """
    pool = multiprocessing.Pool()
    results = pool.map(Worker(func, failsafe), items, chunksize=1)
    pool.terminate()
    return results


def lock_file_handle(handle):
    """ Locks a file for synchronous activities.

        This will block the current thread until the lock can be acquired.

        :param handle: A file handle object pointing to the file to be locked.

        Examples
        --------
        >>> import json
        >>> h = open('test.json', 'r')
        >>> try:
        >>>   lock_file_handle(h)
        >>>   print(json.load(h))
        >>> finally:
        >>>   unlock_file_handle(h)
        >>>   h.close()
    """
    fcntl.flock(handle, fcntl.LOCK_EX)


def unlock_file_handle(handle):
    """ Unlocks a file after finishing synchronous activities.

        This file handle object must be the same one that was previously locked.

        :param handle: The file handle to be unlocked

        Examples
        --------
        >>> import json
        >>> h = open('test.json', 'r')
        >>> try:
        >>>   lock_file_handle(h)
        >>>   print(json.load(h))
        >>> finally:
        >>>   unlock_file_handle(h)
        >>>   h.close()
    """
    fcntl.flock(handle, fcntl.LOCK_UN)
