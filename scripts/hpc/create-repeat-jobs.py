"""Submits a catalogue-creation job for each field using PBS."""
import os
import numpy as np
from astropy.table import Table

from surveytools import SURVEYTOOLS_DATA

# Blacklist based on the qc comments in Janet's image list
BLACKLIST = ['0365b', '1929a', '1905a', '0636a', '0636b']

CATDIR = '/car-data/gb/vphas/psfcat/offsets-20150528'

images = Table.read(os.path.join(SURVEYTOOLS_DATA, 'vphas-dr2-red-images.fits'))
fieldnumbers = [fld[6:] for fld in np.unique(images['Field_1'])]
for field in fieldnumbers:
    for offsetname in ['a', 'b']:
        offset = field + offsetname
        if offset not in BLACKLIST:
            for ext in np.arange(1, 33, 1):
                path = os.path.join(CATDIR, '{}-{}-cat.fits'.format(offset, ext))
                if not os.path.exists(path):
                    cmd = 'qsub -v OFFSET={},EXTENSION={} -N repeat_vphas_{}_{} offset-catalogue2.pbs'.format(offset, ext, offset, ext)
                    print(cmd)
