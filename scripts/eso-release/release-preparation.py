"""Bring the header of a VPHAS catalogue in line with ESO's requirements.

"""
import os
import re
import json

from datetime import datetime
import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time


def tile_footprint(l, b):
    #crd = SkyCoord('galactic', l=l, b=b, unit='deg')
    fpra, fpde = [], []
    for corner in [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]:
        crd = SkyCoord(frame='galactic', l=l+corner[0], b=b+corner[1], unit='deg')
        fpra.append(crd.icrs.ra.deg)
        fpde.append(crd.icrs.dec.deg)
    return fpra, fpde

def tile_centre(l, b):
    crd = SkyCoord(l=l+0.5, b=b+0.5, frame='galactic', unit='deg')
    return crd.icrs.ra.deg, crd.icrs.dec.deg

def fix_header(input_fn, add_prov_keywords=True):
    """Takes a VPHAS DR2 PSC catalogue tile and makes it compatible with
    the ESO PHASE 3 data products standard."""
    # extract glon and glat
    match = re.findall('vphas_l([0-9]+)_b([+-][0-9]+).fits', input_fn)
    l, b = int(match[0][0]), int(match[0][1])
    tilename = 'VPHASDR2_PSC_L{:.0f}_B{:+.0f}'.format(l, b)
    ra, dec = tile_centre(l, b)
    fpra, fpde = tile_footprint(l, b)

    f = fits.open(input_fn)
    f[0].header.set('NAXIS', 0, 'No data in the primary extension, just keywords')
    f[0].header.set('ORIGIN', 'ESO-PARANAL', 'European Southern Observatory')
    f[0].header.set('DATE', datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'UT date when this file was written')
    f[0].header.set('TELESCOP', 'ESO-VST', 'ESO Telescope designation')
    f[0].header.set('INSTRUME', 'OMEGACAM', 'Instrument name')
    f[0].header.set('OBJECT', tilename, 'Survey tile designation')
    f[0].header.set('RA', ra, 'Survey tile centre (J2000.0)')
    f[0].header.set('DEC', dec, 'Survey tile centre (J2000.0)')
    f[0].header.set('PROG_ID', '177.D-3023', 'ESO programme identification')  
    f[0].header.set('OBSTECH', 'IMAGE,OFFSET', 'Originating science file')
    f[0].header.set('PRODCATG', 'SCIENCE.CATALOGTILE', 'Data product category')
    f[0].header.set('REFERENC', '2014MNRAS.440.2036D', 'Survey paper reference')
    f[0].header.set('SKYSQDEG', 1.0, 'Sky coverage in units of square degrees')   
    for idx, ra in enumerate(fpra):
        f[0].header.set('FPRA{}'.format(idx+1), fpra[idx], 'Footprint (J2000.0)')
        f[0].header.set('FPDE{}'.format(idx+1), fpde[idx], 'Footprint (J2000.0)')

    del f[0].header['COMMENT']

    # Set the provenance pointers
    if add_prov_keywords:
        prov = []
        for band in ['u', 'g', 'r2', 'ha', 'r', 'i']:
            for prefix in np.unique(f[1].data['detectionID_' + band].astype('|S14')):
                if prefix == '':
                    continue
                prov.append('o' + prefix.decode('ascii').replace('-', '_') + '.fits.fz')
        for idx, fn in enumerate(sorted(prov)):
            f[0].header.set('PROV{}'.format(idx+1), fn, 'Originating science file')

    # Sort out the first extension
    f[1].header.set('EXTNAME', 'PHASE3CATALOG', 'FITS Extension name')
    del f[1].header['DATE-HDU']
    del f[1].header['STILCLAS']
    del f[1].header['STILVERS']

    colmeta = json.load(open("/home/gb/dev/surveytools/surveytools/data/vphas-psc-columns.json"))[0]
    for kw in f[1].header['TTYPE*']:
        colname = f[1].header[kw]
        if colname in colmeta:
            for field in ['tdisp', 'tucd', 'tunit', 'tcomm']:
                if colmeta[colname][field] != '':
                    f[1].header.set(kw.replace('TTYPE', field),
                                    colmeta[colname][field],
                                    '',
                                    after=kw.replace('TTYPE', 'TFORM'))

    output_fn = '/home/gb/tmp/eso/' + tilename + '.fits'
    f.writeto(output_fn, clobber=True, checksum=True)


def create_metadata_file(template_fn, output_fn):
    template = fits.open(template_fn)
    mf = fits.HDUList([fits.PrimaryHDU(),
                       fits.BinTableHDU(header=None, data=None)])

    mf[0].header.set('PRODCATG', 'SCIENCE.MCATALOG', 'Data product category')
    mf[0].header.set('DATE', datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                     'UT date when this file was written')

    for kw in ['ORIGIN', 'TELESCOP', 'INSTRUME', 'PROG_ID', 'OBSTECH', 'REFERENC']:
        mf[0].header[kw] = template[0].header[kw]

    # Survey period dates taken from Janet's image metadata file
    DR_BEGIN = '2011-12-28 12:00:00'
    DR_END = '2013-10-01 12:00:00'
    mf[0].header.set('MJD-OBS',
                     Time(DR_BEGIN, format='iso', scale='utc').mjd,
                     'Start of observations (days)')
    mf[0].header.set('MJD-END',
                     Time(DR_END, format='iso', scale='utc').mjd,
                     'End of observations (days)', after='MJD-OBS')    

    mf[0].header.set('FILTER1', 'u_SDSS', 'columns with suffix u')
    mf[0].header.set('FILTER2', 'g_SDSS', 'columns with suffix g')
    mf[0].header.set('FILTER3', 'r_SDSS', 'columns with suffix r and r2')
    mf[0].header.set('FILTER4', 'i_SDSS', 'columns with suffix i')
    mf[0].header.set('FILTER5', 'NB_659', 'columns with suffix ha')
    # 5-sigma Vega limits from Janet
    mf[0].header.set('MAGLIM1', 21.1)
    mf[0].header.set('MAGLIM2', 22.5)
    mf[0].header.set('MAGLIM3', 21.6)
    mf[0].header.set('MAGLIM4', 20.7)
    mf[0].header.set('MAGLIM5', 20.7)
    mf[0].header.set('PHOTSYS', 'VEGA', 'Note: columns with suffix _AB are in AB')
    mf[0].header.set('SKYSQDEG', 629)
    mf[0].header.set('EPS_REG', 'VPHAS')

    # First extension is a data-less header of the tiles
    mf[1].header.set('EXTNAME', 'PHASE3CATALOG', 'FITS Extension name')
    mf[1].header.set('NAXIS', 2)
    mf[1].header.set('NAXIS1', template[1].header['NAXIS1'])
    mf[1].header.set('NAXIS2', 0)

    for kw in template[1].header['T*']:
        mf[1].header.set(kw, template[1].header[kw])

    mf.writeto(output_fn, clobber=True, checksum=True, output_verify='warn')



if __name__ == '__main__':
    TILEDIR = '/home/gb/tmp'
    TARGETDIR = '/home/gb/tmp/eso'
    fix_header('/home/gb/tmp/vphas_l224_b-4.fits', True)
    fix_header('/home/gb/tmp/vphas_l264_b+0.fits', True)
    create_metadata_file(os.path.join(TARGETDIR, 'VPHASDR2_PSC_L264_B+0.fits'),
                         os.path.join(TARGETDIR, 'VPHASDR2_PSC_METADATA.fits'))
