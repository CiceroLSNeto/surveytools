from surveytools.utils import angular_separation_degrees as sep
from surveytools.utils import coalesce

import numpy as np
from astropy.table import MaskedColumn

def test_angular_separation_degrees():
    # Special cases
    np.testing.assert_almost_equal(sep(0, 0, 0, 0), 0)
    np.testing.assert_almost_equal(sep(20, 20, 20, 20), 0)
    np.testing.assert_almost_equal(sep(0, 0, 0, 90), 90)
    np.testing.assert_almost_equal(sep(0, 0, 0, -90), 90)
    # Normal cases
    np.testing.assert_almost_equal(sep(20, 0, 30, 0), 10)
    for dec in np.arange(0, 85, 10):
        np.testing.assert_almost_equal(sep(50, dec, 51, dec),
                                       1 * np.cos(np.radians(dec)), decimal=5)


def test_coalesce_basic():   
    col1 = MaskedColumn(data=[1, 0], mask=[False, True])
    col2 = MaskedColumn(data=[0, 2], mask=[True, False])
    for result in [coalesce((col1, col2)), coalesce((col2, col1))]:
        assert result[0] == 1
        assert result[1] == 2
        assert result.mask.any() == False
