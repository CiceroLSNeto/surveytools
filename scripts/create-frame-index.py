"""Creates a table detailing the properties of all VPHAS frames in DR2."""
import os
import numpy as np

from astropy import log
from astropy.table import Table, vstack
from astropy.utils.console import ProgressBar

import surveytools
from surveytools.footprint import VphasExposure

if __name__ == '__main__':
    blue_images = Table.read(os.path.join(surveytools.SURVEYTOOLS_DATA, 'vphas-dr2-blue-images.fits'))['image file']
    red_images = Table.read(os.path.join(surveytools.SURVEYTOOLS_DATA, 'vphas-dr2-red-images.fits'))['image file']
    output_tbl = None
    output_fn = 'vphas-dr2-frames.csv'
    log.info('Writing {0}'.format(output_fn))
    for fn in ProgressBar(np.concatenate((blue_images, red_images))):
        try:
            exp = VphasExposure(fn)
            tbl = exp.frames()
            if output_tbl is None:
                output_tbl = tbl
            else:
                output_tbl = vstack([output_tbl, tbl])
        except Exception as e:
            log.error('{0}: {1}'.format(fn, e))
    output_tbl.write(output_fn, format='ascii.ecsv')
