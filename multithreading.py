"""
Contains some convenience methods for multithreading jobs.
"""

__author__ = 'tobin'

import logging
import multiprocessing
import itertools

logger = logging.getLogger(__name__)


class Worker(object):
    def __init__(self, do_work):
        self.do_work = do_work

    def __call__(self, args):
        return self.do_work(**args)


def combinations(**kwargs):
    names = kwargs.keys()
    values = itertools.product(*kwargs.values())

    params = [dict(zip(names, v)) for v in values]
    return params


def map_kwargs(func, items):
    pool = multiprocessing.Pool()
    return pool.map(Worker(func), items)


def do_all(func, **kwargs):
    params = combinations(**kwargs)
    return zip(params, map_kwargs(func, params))