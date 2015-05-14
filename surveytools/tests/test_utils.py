from surveytools.utils import angular_separation_degrees as sep
from surveytools.utils import coalesce
from surveytools.utils import timeout, TimeOutException

import time
import numpy as np
from astropy.table import MaskedColumn

import pytest

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


class TestCoalesce():

    def setup_method(self, method):
        self.c1 = MaskedColumn(name='col1', data=[1, 2, 3], mask=[False, False, False])
        self.c2 = MaskedColumn(name='col2', data=[4, 5, 6], mask=[True, False, False])
        self.c3 = MaskedColumn(name='col3', data=[7, 8, 9], mask=[False, True, False])

    def test_basic(self):
        assert np.all(coalesce([self.c1, self.c2, self.c3]) == self.c1)
        assert np.all(coalesce([self.c2, self.c1]) == [1, 5, 6])
        assert np.all(coalesce([self.c2, self.c3]) == [7, 5, 6])

    def test_single_column(self):
        for col in [self.c1, self.c2, self.c3]:
            assert np.all(coalesce(col) == col)

    def test_bad_input_type(self):
        with pytest.raises(TypeError):
            coalesce([])
        with pytest.raises(TypeError):
            coalesce(1)
        with pytest.raises(TypeError):
            coalesce([self.c1, 1])


def test_timeout():
    @timeout(1)
    def somefunction():
        time.sleep(2)
    with pytest.raises(TimeOutException):
        somefunction()

def test_no_timeout():
    @timeout(1)
    def somefunction():
        return
    somefunction()