import numpy as np
from surveytools.utils import angular_separation_degrees as sep


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
