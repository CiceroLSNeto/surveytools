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


def angular_separation_degrees(lon1, lat1, lon2, lat2):
    """Angular separation between two points on a sphere, with coordinates in degrees.
    
    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float
        Longitude and latitude of the two points, in DEGREES.

    Returns
    -------
    angular separation : float
        In DEGREES.

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1]_,
    which is slighly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.
    .. [1] http://en.wikipedia.org/wiki/Great-circle_distance

    This function is adapted from 
    `astropy.coordinates.angle_utilities.angular_separation`,
    which works with input and output in radians or Quantity objects.
    """
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)

    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.degrees(np.arctan2(np.sqrt(num1 ** 2 + num2 ** 2), denominator))
