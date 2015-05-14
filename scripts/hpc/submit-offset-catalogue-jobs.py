"""Submits a catalogue-creation job for each field using PBS."""
import os
import numpy as np
from astropy.table import Table

from surveytools import SURVEYTOOLS_DATA

# Blacklist based on the qc comments in Janet's image list
BLACKLIST = ['0365b', '1929a', '1905a', '0636a', '0636b']

images = Table.read(os.path.join(SURVEYTOOLS_DATA, 'vphas-dr2-red-images.fits'))
fieldnumbers = [fld[6:] for fld in np.unique(images['Field_1'])]
for field in fieldnumbers:
    for offsetname in ['a', 'b']:
        offset = field + offsetname
        if offset not in BLACKLIST:
            cmd = 'qsub -v OFFSET={} -N vphas_{} offset-catalogue.pbs'.format(offset, offset)
            print(cmd)
