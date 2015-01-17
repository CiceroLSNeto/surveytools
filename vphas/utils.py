"""Generic utilities used by VphasTools."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
import functools
import numpy as np

from astropy import log


def cached_property(method):
    """A cached version of the @property decorator."""
    def get(self):
        try:
            return self._cache[method.func_name]
        except AttributeError:
            self._cache = {}
            x = self._cache[method.func_name] = method(self)
            return x
        except KeyError:
            x = self._cache[method.func_name] = method(self)
            return x
    return property(get)


def timed(function):
    """Decorator to log the time a function or method takes to execute.

    Inspired by astropy.utils.timefunc
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = function(*args, **kwargs)
        te = time.time()
        tt = (te - ts)
        # Determine the number of decimal places we need to show
        decimals = -int(np.log10(tt)) + 1
        if decimals < 0:
            decimals = 0
        log.info(('{0} took {1:.'+str(decimals)+'f}s.')
                 .format(function.__name__, tt))
        return result
    return wrapper
