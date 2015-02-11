"""Tests the surveytools.footprint module."""
import numpy as np

from surveytools.footprint import VphasFootprint, VphasOffset

def test_vphas_offset_coordinates():
    """Test the offset pattern, which is expected to equal
    ra -0, dec +0 arcsec for the "a" pointing;
    ra -588, dec +660 arcsec for the "b" pointing;
    ra -300, dec +350 arcsec for the "c" pointing.
    """
    vf = VphasFootprint()
    np.testing.assert_almost_equal(vf.offsets['0001a']['ra'], 97.2192513369)
    np.testing.assert_almost_equal(vf.offsets['0001a']['dec'], 0)
    np.testing.assert_almost_equal(vf.offsets['0001b']['ra'], 97.2192513369 - 588/3600.)
    np.testing.assert_almost_equal(vf.offsets['0001b']['dec'], 0 + 660/3600.)
    np.testing.assert_almost_equal(vf.offsets['0001c']['ra'], 97.2192513369 - 300/3600.)
    np.testing.assert_almost_equal(vf.offsets['0001c']['dec'], 0 + 350/3600.)


def test_vphas_filenames():
    """Ensure the right filename is returned for a given band/offset."""
    assert VphasOffset('1122a').image_filenames['ha'] == 'o20120330_00032.fit'
    assert VphasOffset('1122b').image_filenames['ha'] == 'o20120330_00034.fit'
    assert VphasOffset('1122c').image_filenames['ha'] == 'o20120330_00033.fit'
    assert VphasOffset('1842a').image_filenames['r'] == 'o20130314_00061.fit'
    assert VphasOffset('1842b').image_filenames['r'] == 'o20130314_00062.fit'
    assert VphasOffset('0765a').image_filenames['g'] == 'o20130413_00024.fit'
    assert VphasOffset('0765b').image_filenames['g'] == 'o20130413_00026.fit'
    assert VphasOffset('0765c').image_filenames['g'] == 'o20130413_00025.fit'


if __name__ == '__main__':
    test_vphas_filenames()