"""Generic utilities used by VphasTools."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
import signal
import functools
import numpy as np

from astropy import table
from astropy import log


class TimeOutException(Exception):
    """
    Raised when a timeout happens,
    see http://stackoverflow.com/questions/8616630/time-out-decorator-on-a-multprocessing-function
    """

def timeout(timeout):
    """
    Return a decorator that raises a TimeOutException exception
    after timeout seconds, if the decorated function did not return.
    """
    def decorate(f):

        def handler(signum, frame):
            raise TimeOutException('{} timed out after {} seconds'.format(f.__name__, timeout))

        def new_f(*args, **kwargs):
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            result = f(*args, **kwargs)
            signal.signal(signal.SIGALRM, old_handler)  # Old signal handler is restored
            signal.alarm(0)  # Alarm removed
            return result

        return new_f

    return decorate


def cached_property(method):
    """A cached version of the @property decorator."""
    def get(self):
        try:
            return self._cache[method.__name__]
        except AttributeError:
            self._cache = {}
            x = self._cache[method.__name__] = method(self)
            return x
        except KeyError:
            x = self._cache[method.__name__] = method(self)
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
    """Angular separation between two points on a sphere (input in degrees).
    
    This function provides a fast alternative to AstroPy's great but somewhat
    sluggish SkyCoord API. The function is adapted from 
    `astropy.coordinates.angle_utilities.angular_separation`, which requires
    the input and output to be in radians or to be Quantity objects.

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


def equ2gal(ra, dec):
    """Converts Equatorial J2000d coordinates to the Galactic frame.

    Note: it is better to use AstroPy's SkyCoord API for this.

    Parameters
    ----------
    ra, dec : float, float [degrees]
        Input J2000 coordinates (Right Ascension and Declination).

    Returns
    -------
    glon, glat: float, float [degrees]
    """
    import math as m
    from math import sin, cos, atan, asin, floor
    OB = m.radians(23.4333334);
    dec = m.radians(dec)
    ra = m.radians(ra)
    a = 27.128251 # The RA of the North Galactic Pole
    d = 192.859481 # The declination of the North Galactic Pole
    l = 32.931918 # The ascending node of the Galactic plane on the equator
    sdec = sin(dec)
    cdec = cos(dec)
    sa = sin(m.radians(a))
    ca = cos(m.radians(a))
    GT = asin(cdec * ca * cos(ra - m.radians(d)) + sdec * sa)
    GL = m.degrees(atan((sdec - sin(GT) * sa) / (cdec * sin(ra - m.radians(d)) * ca)))
    TP = sdec - sin(GT) * sa
    BT = cdec * sin(ra - m.radians(d)) * ca
    if (BT < 0):
        GL += 180
    else:
        if (TP < 0):
            GL += 360  
    GL += l
    if (GL > 360):
        GL -= 360
    LG = floor(GL)
    LM = floor((GL - floor(GL)) * 60)
    LS = ((GL - floor(GL)) * 60 - LM) * 60
    GT = m.degrees(GT)
    D = abs(GT)
    if (GT > 0):
        BG = floor(D)
    else:
        BG = -1*floor(D)
    BM = floor((D - floor(D)) * 60)
    BS = ((D - floor(D)) * 60 - BM) * 60
    if (GT < 0):
        BM = -BM
        BS = -BS
    #if GL > 180:
    #    GL -= 360
    return (GL, GT)


class ColumnMergeError(ValueError):
    pass

def _get_list_of_columns(columns):
    """
    Check that columns is a Column or sequence of Columns.  Returns the
    corresponding list of Columns.
    """
    import collections
    # Make sure we have a list of things
    if not isinstance(columns, collections.Sequence):
        columns = [columns]
    # Make sure each thing is a Column
    if any(not isinstance(x, table.Column) for x in columns) or len(columns) == 0:
        raise TypeError('`columns` arg must be a Column or sequence of Columns')
    return columns


def coalesce(columns):
    """Coalesces masked columns.

    Parameters
    ----------
    columns : iterable of type `MaskedColumn`

    Returns
    -------
    column : coalesced result
    """
    # todo: test if columns have right type
    # todo: test if columns have same size
    columns = _get_list_of_columns(columns)  # validates input
    
    result = columns[0].copy()
    for col in columns[1:]:
        mask_coalesce = result.mask & ~col.mask
        result.data[mask_coalesce] = col.data[mask_coalesce]
        result.mask[mask_coalesce] = False
    return result
