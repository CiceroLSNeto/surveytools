from __future__ import absolute_import

import os

# Use Agg if no DISPLAY is available
DISPLAY = os.environ.get('DISPLAY')
if DISPLAY is None or DISPLAY.startswith('localhost'):
    import matplotlib
    matplotlib.use('Agg')

###########
# CONSTANTS
###########

# Where are VPHAS reduced images and calibration frames?
VPHAS_DATA_PATH = '/home/gb/tmp/vphasdisk'
VPHAS_PIXEL_SCALE = 0.213  # arcsec/px, cf. OmegaCAM manual Sect 2.1
VPHAS_BANDS = ['u', 'g', 'r2', 'ha', 'r', 'i']

# Where is the data that comes with this package?
SURVEYTOOLS_PATH = os.path.abspath(os.path.dirname(__file__))
SURVEYTOOLS_DATA = os.path.join(SURVEYTOOLS_PATH, 'data')
SURVEYTOOLS_CONFIGDIR = os.path.join(SURVEYTOOLS_PATH, 'config')

# Position of the VST/OmegaCAM CCDs.
# left-right = East-West and top-bottom = North-South;
# the numbers refer to the FITS HDU extension number of an OmegaCam image.
OMEGACAM_CCD_ARRANGEMENT = [32, 31, 30, 29, 16, 15, 14, 13,
                            28, 27, 26, 25, 12, 11, 10,  9,
                            24, 23, 22, 21,  8,  7,  6,  5,
                            20, 19, 18, 17,  4,  3,  2,  1]


from .catalogue import VphasFrame, VphasOffsetCatalogue
