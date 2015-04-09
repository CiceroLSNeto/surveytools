"""Tools to create photometric catalogues from VPHAS data.

Classes
-------
VphasFrame
VphasOffsetCatalogue

Example use
-----------
Create a photometric catalogue of VPHAS pointing 0149a:
```
import vphas
offset = vphas.VphasOffsetCatalogue('0149a')
offset.create_catalogue().write('mycatalogue.fits')
```

Terminology
-----------
This module makes use of the concept of a `field`, a `pointing`, and a `frame`.
defined as follows:
-*field*: a region in the sky covered by 2 or 3 offset pointings of the
          telescope, identified using a 4-character wide, zero-padded number
          string, e.g. '0149'.
-*offset*: a single position in the sky denoting one of the offsets that
           make up a field, denoted using a 5-character wide string,
           e.g. '0149a' (first offset), '0149b' (second offset),
           '0149c' (third offset, for H-alpha and some g-band observations only).
-*exposure*: a single shutter opening, for VPHAS this always cooresponds to
             a single offset-band combination, e.g. '0149a-ha'.
-*frame*: area covered by a single ccd of an exposure, e.g. '0149a-ha-8'.

Each field has 1x2 pointings in u and i, 1x3 pointings in g and H-alpha,
and 2x2 pointings in r. Each pointing consists of 32 ccd frames.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import shutil
import warnings
import tempfile
import itertools
import multiprocessing
import configparser

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.image as mimg

import astropy
from astropy.io import fits
from astropy import log
from astropy import table
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
import astropy.units as u
from astropy.utils.timer import timefunc
from astropy.visualization import ManualInterval, LogStretch

import photutils
import photutils.morphology
from photutils.background import Background

from . import SURVEYTOOLS_CONFIGDIR, OMEGACAM_CCD_ARRANGEMENT, VPHAS_PIXEL_SCALE, VPHAS_BANDS
from .utils import cached_property, timed, coalesce
from . import footprint


DEFAULT_CONFIG = os.path.join(SURVEYTOOLS_CONFIGDIR, 'catalogue.ini')

###########
# CLASSES
###########

class VphasFrame(object):
    """Class representing a single-CCD image obtained by ESO's VST telescope.

    Parameters
    ----------
    filename : str
        Path to the image FITS file.

    cfg : `ConfigParser` object

    extension : int (optional)
        Extension of the image in the FITS file. (default: 0)
    """
    def __init__(self, filename, extension=0, cfg=None, workdir=None):
        if os.path.exists(filename):
            self.orig_filename = filename
        elif os.path.exists(os.path.join(cfg['vphas']['datadir'], filename)):
            self.orig_filename = os.path.join(cfg['vphas']['datadir'], filename)
        else:
            raise IOError('File not found:' + os.path.join(cfg['vphas']['datadir'], filename))
        self.orig_extension = extension
        # Setup configuration
        if cfg is None:
            self.cfg = configparser.ConfigParser()
            self.cfg.read(DEFAULT_CONFIG)
        else:
            self.cfg = cfg
        # Setup the workdir
        if workdir is None:
            workdir_root = cfg['vphas']['workdir']
        else:
            workdir_root = workdir
        self.workdir = tempfile.mkdtemp(prefix='frame-{0}-{1}-'.format(filename, extension),
                                        dir=workdir_root)
        self._cache = {}
        self.filename, self.extension = self._preprocess_image()

    def __del__(self):
        #del self._cache['daophot']
        pass

    def __getstate__(self):
        """Prepare the object before pickling (serialization)."""
        # Pickle does not like serializing `astropy.io.fits.hdu` objects
        for key in ['hdu', 'daophot']:
            try:
                del self._cache[key]
            except KeyError:
                pass
        return self.__dict__

    @timed
    def _preprocess_image(self):
        """Prepare the image for photometry by IRAF.

        IRAF/DAOPHOT does not appears to support RICE-compressed files,
        and does not allow a weight (confidence) map to be provided.
        This method works around these limitations by saving the image
        to an uncompressed FITS file in which low-confidence areas are masked
        out (by assigning a pixel value of -1 ADU). The new file generated
        by this produce is then suitable for analysis by IRAF/DAOPHOT tasks.

        Moreover, this method also subtracts the background to enable the
        source detection to be carried out using the assumption of a smooth
        background. The background is estimated using a high-pass filter.

        Returns
        -------
        (filename, extension): (str, int)
            Path and HDU number of the pre-processed FITS file.
        """ 
        # Open the pipeline-processed image data
        fts = fits.open(self.orig_filename)
        hdu = fts[self.orig_extension]
        fltr = fts[0].header['ESO INS FILT1 NAME']
        # Twilight flats of the VST are known to be affected by scattered light
        # a separate illumination correction map is used to correct for this.
        if self.cfg['catalogue'].getboolean('apply_illumcor', True):
            illumcor_fn = hdu.header['FLATSRC'].replace('flat', 'illum').split('[')[0]
            illumcor_path = os.path.join(self.cfg['vphas']['datadir'], illumcor_fn)
            illumcor_hdu = fits.open(illumcor_path)[self.orig_extension]
            log.debug('{0}: applying illumcor: {1}'.format(fltr, illumcor_fn))
            imgdata = hdu.data * illumcor_hdu.data
            del illumcor_hdu  # Free memory
        else:
            imgdata = hdu.data
        # Open the confidence map and create bad pixel mask
        self.confidence_map_path = os.path.join(self.cfg['vphas']['datadir'],
                                                hdu.header['CIR_CPM'].split('[')[0])
        confmap_hdu = fits.open(self.confidence_map_path)[self.orig_extension]
        minconf = float(self.cfg['catalogue'].get('confidence_threshold', 80.))
        bad_pixel_mask = confmap_hdu.data < minconf
        del confmap_hdu  # Free memory
        # Estimate the background in a mesh of (41, 32) pixels; which is chosen
        # to fit an integer number of times in the image size (4100, 2048).
        # At the pixel scale of 0.21 arcsec/px, this corresponds to ~10 arcsec.
        bg = Background(imgdata, (41, 32), filter_shape=(6, 6),
                        mask=bad_pixel_mask,
                        method='median', sigclip_sigma=3., sigclip_iters=5)
        log.debug('{0} sky estimate = {1:.1f} +/- {2:.1f}'.format(
                  fltr, bg.background_median, bg.background_rms_median))
        self.sky = bg.background_median
        self.sky_sigma = bg.background_rms_median
        # Subtract the background, retaining the median sky level
        if self.cfg['catalogue'].getboolean('subtract_sky', True):
            imgdata = imgdata - (bg.background - bg.background_median)
        # Apply bad pixel mask (will only work if IRAF's DATAMIN > -1)
        if self.cfg['catalogue'].getboolean('mask_bad_pixels', True):
            imgdata[bad_pixel_mask] = -1
        # Write the processed image to an uncompressed FITS file for IRAF
        path = os.path.join(self.workdir, '{0}.fits'.format(fltr))
        log.debug('Writing {0}'.format(os.path.basename(path)))
        newhdu = fits.PrimaryHDU(imgdata, hdu.header)
        newhdu.header.extend(fts[0].header, unique=True)
        del newhdu.header['RADECSYS']  # non-standard keyword; raises errors
        newhdu.writeto(path)
        del newhdu  # Free memory
        # Also write the background frame as a diagnostic
        self.bg_path = os.path.join(self.workdir, '{0}-bg.fits'.format(fltr))
        log.debug('Writing {0}'.format(os.path.basename(self.bg_path)))
        newhdu = fits.PrimaryHDU(bg.background, hdu.header)
        newhdu.writeto(self.bg_path)
        del newhdu  # Free memory
        # Return the pre-processed image filename and extension
        return (path, 0)

    def populate_cache(self):
        """Populate the cache.

        When using parallel computing, call this function before the object
        is serialized and sent off to other nodes, to keep image statistics
        from being re-computed unncessesarily on different nodes.
        """
        self._estimate_psf()

    @property
    def orig_hdu(self):
        """Returns the FITS HDU object corresponding to the original image."""
        return fits.open(self.orig_filename)[self.orig_extension]

    @property
    def bg_hdu(self):
        return fits.open(self.bg_path)[0]

    @cached_property
    def hdu(self):
        """FITS HDU object corresponding to the measured image (after sky subtraction)."""
        return fits.open(self.filename)[self.extension]

    @property
    def data(self):
        """FITS HDU object corresponding to the measured image (after sky subtraction)."""
        return self.hdu.data

    @cached_property
    def header(self):
        """FITS header object."""
        return self.hdu.header

    @cached_property
    def primary_header(self):
        """FITS header object."""
        return fits.open(self.orig_filename)[0].header

    @cached_property
    def object(self):
        """Astronomical target."""
        return self.header['OBJECT']

    @cached_property
    def fieldname(self):
        """VPHAS name of the field, e.g. '0001a'."""
        field_number = self.header['ESO OBS NAME'].split('_')[1]
        expno = self.header['ESO TPL EXPNO']
        if expno == 1:
            offset = 'a'
        elif expno < self.header['ESO TPL NEXP']:
            offset = 'b'
        else:
            offset ='c'
        return '{0}{1}'.format(field_number, offset)

    @cached_property
    def name(self):
        """VPHAS name of the frame, e.g. '0001a-r-8'."""
        return '{0}-{1}-{2}'.format(self.fieldname, self.band, self.orig_extension)

    @cached_property
    def band(self):
        """Returns the colloquial band name.

        VPHAS observations have an OBS NAME of the format "p88vphas_0149_uuna";
        where the first two letters of the third part indicate the band name
        """
        bandnames = {'uu': 'u', 'ug': 'g', 'ur': 'r2',
                     'hh': 'ha', 'hr': 'r', 'hi': 'i'}
        obsname = self.header['ESO OBS NAME']
        return bandnames[obsname.split('_')[2][0:2]]

    @cached_property
    def filtername(self):
        """Filter name."""
        return self.header['ESO INS FILT1 NAME']

    @cached_property
    def exposure_time(self):
        """Exposure time [seconds]."""
        return self.header['EXPTIME']

    @cached_property
    def airmass(self):
        """Airmass."""
        return (self.header['ESO TEL AIRM START']
                + self.header['ESO TEL AIRM END']) / 2.

    @cached_property
    def zeropoint(self):
        """Magnitude zeropoint corrected for airmass."""
        # assuming default extinction
        return self.hdu.header['MAGZPT'] - (self.airmass - 1.) * self.hdu.header['EXTINCT']

    @cached_property
    def gain(self):
        """Detector gain [electrons / adu]."""
        # WARNING: OmegaCam headers contain gain in "ADU per electron",
        # we need to convert this to "electron per ADU" for DAOPHOT.
        return 1. / self.hdu.header['ESO DET OUT1 GAIN']

    @cached_property
    def readnoise(self):
        """Detector read noise in electrons.

        We do not simply return the 'HIERARCH ESO DET OUT1 RON' keyword
        because it appears to contain "0" at all times.
        In reality the noise is documented to be approx 2 ADU,
        i.e. ~5 electrons, cf http://www.eso.org/observing/dfo/quality/OMEGACAM/qc/readnoise_QC1.html
        """
        return 2. * self.gain  # [electrons] i.e. [photons]

    @cached_property
    def datamin(self):
        """Returns the minimum good pixel value. [adu]
        
        We tolerate up to 5-sigma below the average sky level.
        """
        datamin = self.sky - 5 * self.sky_sigma
        # Ensure datamin is positive, because `_preprocess_image` masks out
        # bad pixels by assigning a negative value to them.
        if datamin < 0:
            return 0.
        return datamin

    @cached_property
    def datamax(self):
        """Returns the maximum good (non-saturated) pixel value. [adu]

        The VST/OmegaCAM manual (VST-MAN-OCM-23100-3110-2_7_1) suggests that the
        detector is linear (within ~1%) up to the saturation level.
        The saturation level is not exactly 2^16 = 65536 due to bias subtraction etc,
        so we conservatively ignore pixel values over 55000 ADU.

        It is VERY important to be conservative when choosing PSF template
        stars, because the cores of saturated stars should be avoided at all
        cost. Charge bleeding may cause pixels well below the nominal saturation
        level to give an unrepresentative view of the PSF.
        """
        return 55000

    @cached_property
    def seeing(self):
        """Estimate of the seeing full-width at half-maximum."""
        return self.hdu.header['SEEING']  # pixels

    @property
    def psf_fwhm(self):
        """The Full-Width-Half-Maximum of a Gaussian PSF model fit [pixels]."""
        try:
            return self._cache['psf_fwhm']
        except KeyError:
            self._estimate_psf()
            return self._cache['psf_fwhm'] 

    @property
    def psf_ratio(self):
        try:
            return self._cache['psf_ratio']
        except KeyError:
            self._estimate_psf()
            return self._cache['psf_ratio']

    @property
    def psf_theta(self):
        try:
            return self._cache['psf_theta']
        except KeyError:
            self._estimate_psf()
            return self._cache['psf_theta'] 

    def world2pix(self, ra, dec, origin=1):
        """Shorthand to convert equatorial(ra, dec) into pixel(x, y) coords.

        Use origin=1 if the x/y coordinates are to be used as input
        for IRAF/DAOPHOT, use origin=0 for astropy.
        """
        return astropy.wcs.WCS(self.hdu.header).wcs_world2pix(ra, dec, origin)

    def pix2world(self, x, y, origin=1):
        """Shorthand to convert pixel(x,y) into equatorial(ra,dec) coordinates.

        Use origin=1 if x/y positions were produced by IRAF/DAOPHOT,
        0 if they were produced by astropy."""
        return astropy.wcs.WCS(self.hdu.header).wcs_pix2world(x, y, origin)

    def _estimate_psf(self, threshold=100.):
        """Fits a 2D Gaussian PSF to the stars in the images.

        This will populate self._cache['psf_fwhm'], self._cache['psf_ratio'],
        self._cache['psf_theta']. The estimates are intended to serve as input
        to the DAOFIND routine.

        Parameters
        ----------
        threshold : float (optional)
            Minimum detection significance in units sigma (noise above the
            background) for objects to be considered for PSF fitting.
        """
        sources = photutils.daofind(self.hdu.data - self.sky,
                                    fwhm = self.seeing,
                                    threshold = threshold * self.sky_sigma)
        log.debug("Found {0} sources for Gaussian PSF fitting.".format(len(sources)))
        positions = [[s['xcentroid'], s['ycentroid']] for s in sources]
        prf_discrete = photutils.psf.create_prf(self.hdu.data - self.sky,
                                                positions,
                                                7,
                                                mode='median') #, fluxes=fluxes_catalog, mask=np.logical_not(mask), subsampling=5)
        
        myfit = photutils.morphology.fit_2dgaussian(prf_discrete._prf_array[0][0])
        fwhm = myfit.x_stddev * (2.0 * np.sqrt(2.0 * np.log(2.0)))
        ratio = myfit.y_stddev.value / myfit.x_stddev.value  # Need value to keep it from being an array
        if ratio > 1:  # Daophot will fail if the ratio is larger than 1 (i.e. it wants tthe ratio of minor to major axis)
            ratio = 1. / ratio
        theta = myfit.theta.value
        # pyraf will complain over a negative theta
        if theta < 0:
            theta += 180
        log.debug('{0} PSF FWHM = {1:.1f}px; ratio = {2:.1f}; theta = {3:.1f}'.format(self.band, fwhm, ratio, theta))
        self._cache['psf_fwhm'] = fwhm
        self._cache['psf_ratio'] = ratio
        self._cache['psf_theta'] = theta
        del self.hdu.data  # free memory

    def daophot(self, **kwargs):
        """Returns a Daophot object, pre-configured to work on the image."""
        image_path = '{0}[{1}]'.format(self.filename, self.extension)
        from .daophot import Daophot
        dp = Daophot(image_path, workdir=self.workdir,
                     datamin=self.datamin, datamax=self.datamax,
                     epadu=self.gain, fwhmpsf=self.psf_fwhm,
                     itime=self.exposure_time,
                     ratio=self.psf_ratio, readnoi=self.readnoise,
                     sigma=self.sky_sigma, theta=self.psf_theta,
                     zmag=self.zeropoint,
                     **kwargs)
        self._cache['daophot'] = dp
        return dp

    def compute_source_table(self, threshold=3., **kwargs):
        """Returns a table of sources in the frame, with initial photometry.

        This method will execute the full DAOPHOT pipeline of source detection,
        aperture photometry, PSF-model fitting, and PSF photometry. The reason
        for running the "full monty", rather than daofind alone, is two-fold:
        - the PSF-fitting step will likely refine the coordinates;
        - the CHI score and error code produced by ALLSTAR provides us with
        information to cull likely spurious sources.

        Parameters
        ----------
        threshold : float (optional)
            Daofind's detection threshold (in units background sigma)

        chi_max : float (optional)
            Cull objects with a CHI score larger than chi_max.

        Returns
        -------
        sourcetbl : `astropy.table.Table` object
            Table listing the objects found by DAOFIND, augmented with the
            output from initial (rough) aperture and PSF photometry.
        """
        dp = self.daophot(threshold=threshold, **kwargs)
        sources = dp.do_psf_photometry()
        chi_max = float(self.cfg['sourcelist'].get('chi_max', 5.))
        sharp_max = float(self.cfg['sourcelist'].get('sharp_max', 1.))
        mask_accept = (
                        (sources['SNR'] > threshold)
                        & (np.abs(sources['SHARPNESS']) < sharp_max)
                        & (sources['CHI'] < chi_max)
                        & (sources['PIER_ALLSTAR'] == 0)
                        & (sources['PIER_PHOT'] == 0)
                        )
        sources.meta['band'] = self.band
        tbl = sources[mask_accept]
        log.info('Identified {0} sources in {1} at sigma > {2}'.format(
                     len(tbl), self.band, threshold))
        # Add ra/dec columns
        ra, dec = self.pix2world(tbl['XCENTER_ALLSTAR'],
                                 tbl['YCENTER_ALLSTAR'],
                                 origin=1)
        ra_col = Column(name='ra', data=ra)
        dec_col = Column(name='dec', data=dec)
        tbl.add_columns([ra_col, dec_col])
        return tbl

    def photometry(self, ra, dec, ra_psf, dec_psf, **kwargs):
        """Computes PSF & aperture photometry for a list of sources.

        Parameters
        ----------
        ra, dec : array of float (decimal degrees)
            Positions at which to carry out PSF photometry.

        ra_psf, dec_psf : array of float (decimal degrees)
            Positions of reliable stars for fitting the PSF model.

        Returns
        -------
        tbl : `astropy.table.Table` object
            Table containing the results of the PSF- and aperture photometry.
        """
        # Save the coordinates to a file suitable for daophot
        x, y = self.world2pix(ra, dec)
        col_x = Column(name='XCENTER', data=x)
        col_y = Column(name='YCENTER', data=y)
        coords_tbl = Table([col_x, col_y])
        coords_tbl_filename = os.path.join(self.workdir, 'coords-tbl.txt')
        coords_tbl.write(coords_tbl_filename, format='ascii')

        # Save the coordinates to a file suitable for daophot
        x, y = self.world2pix(ra_psf, dec_psf)
        col_x = Column(name='XCENTER', data=x)
        col_y = Column(name='YCENTER', data=y)
        psf_tbl = Table([col_x, col_y])
        psf_tbl_filename = os.path.join(self.workdir, 'psf-coords-tbl.txt')
        psf_tbl.write(psf_tbl_filename, format='ascii')

        dp = self.daophot(**kwargs)
        # Fit the PSF model
        dp.apphot(coords=psf_tbl_filename)
        dp.pstselect()
        psf_scatter = dp.psf()
        # Carry out the aperture and PSF photometry on all the stars requested
        dp.apphot(coords=coords_tbl_filename)
        dp.allstar()
        # Remember the path of the PSF and the PSF-subtracted image FITS files
        self._cache['daophot_subimage_path'] = dp.subimage_path
        self._cache['daophot_seepsf_path'] = dp.seepsf_path

        # The code below transforms the table into a user-friendly format
        tbl = dp.get_allstar_phot_table()
        tbl.meta['band'] = self.band
        tbl.meta[self.band + 'PsfRms'] = psf_scatter
        # Add celestial coordinates ra/dec and nearest neighbour distance as columns
        ra, dec = self.pix2world(tbl['XCENTER_ALLSTAR'],
                                 tbl['YCENTER_ALLSTAR'],
                                 origin=1)
        tbl['ra_' + self.band] = ra
        tbl['dec_' + self.band] = dec
        crd = SkyCoord(ra*u.deg, dec*u.deg)
        idx, nn_dist, dist3d = crd.match_to_catalog_sky(crd, nthneighbor=2)
        tbl['nndist_' + self.band] = nn_dist.to(u.arcsec)
        #tbl.add_columns([ra_col, dec_col, nndist_col])
        # Add further columns
        # Shift of the source centroid during PSF fitting [arcsec]
        tbl['pixelShift_' + self.band] = np.hypot(tbl['XCENTER_ALLSTAR'] - tbl['XINIT'],
                                          tbl['YCENTER_ALLSTAR'] - tbl['YINIT'])
        id_prefix = (
                     os.path.basename(self.orig_filename)
                        .replace('o', '')
                        .replace('_', '-')
                        .replace('.fit', '')
                    )
        tbl['detectionID_' + self.band] = ['{0}-{1}-{2}'.format(id_prefix, self.orig_extension, idx) for idx in tbl['ID']]
        tbl['mjd_'+self.band] = np.repeat(self.primary_header['MJD-OBS'], len(tbl))
        tbl['psffwhm_'+self.band] = np.repeat(VPHAS_PIXEL_SCALE * self.psf_fwhm, len(tbl))
        tbl['airmass_'+self.band] = np.repeat(self.airmass, len(tbl))
        # Rename existing columns from DAOPHOT to our conventions
        tbl['MAG_ALLSTAR'].name = self.band
        tbl['MERR_ALLSTAR'].name = 'err_' + self.band
        tbl['MSKY_ALLSTAR'].name = 'sky_' + self.band
        tbl['CHI'].name = 'chi_' + self.band
        tbl['SHARPNESS'].name = 'sharpness_' + self.band
        tbl['PIER_ALLSTAR'].name = 'pier_' + self.band
        tbl['PERROR_ALLSTAR'].name = 'perror_' + self.band
        tbl['MAG_PHOT'].name = 'aperMag_' + self.band
        tbl['MERR_PHOT'].name = 'aperMagErr_' + self.band
        tbl['SNR'].name = 'snr_' + self.band
        tbl['LIM3SIG'].name = 'magLim_' + self.band
        tbl['XCENTER_ALLSTAR'].name = 'x_' + self.band
        tbl['YCENTER_ALLSTAR'].name = 'y_' + self.band

        # Write the full table as diagnostic output, before applying masks
        if self.cfg['catalogue'].getboolean('save_diagnostics', True):
            with warnings.catch_warnings():
                # Attribute `keywords` cannot be written to FITS files
                warnings.filterwarnings("ignore",
                                        message='Attribute `keywords`(.*)')
                tbl_fn = os.path.join(self.workdir, 'photometry.fits')
                log.debug('{0}: writing photometry to {1}'.format(self.band, tbl_fn))
                tbl.write(tbl_fn)

        # Mask untrustworthy magnitude estimates at low or negative SNR
        mask_too_faint = (
                             (tbl['snr_' + self.band].filled(-1) < 3)
                             | (tbl[self.band].filled(999) > tbl['magLim_' + self.band])
                          )
        for prefix in ['', 'err_', 'aperMag_', 'aperMagErr_']:
            tbl[prefix + self.band].mask[mask_too_faint] = True

        # Mask PSF magnitudes if not fitted without error, the position should not
        # have shifted, and the CHI score must be decent.
        chi_max = float(self.cfg['photometry'].get('chi_max', 3.))
        shift_max = float(self.cfg['photometry'].get('shift_max', 1.))
        mask_bad_fit = (
                        np.isnan(tbl[self.band].filled(np.nan))
                        | (tbl['pier_' + self.band].filled(0) != 0)
                        | (tbl['chi_' + self.band].filled(999) > chi_max)
                        | (tbl['pixelShift_' + self.band].filled(999) > shift_max)
                    )
        for prefix in ['ra_', 'dec_', '', 'err_']:
            tbl[prefix + self.band].mask[mask_bad_fit] = True 
            
        # The "clean" quality flag helps the user select good sources
        tbl['clean_' + self.band] = (
                                    ~tbl[self.band].mask
                                    & (tbl['err_' + self.band].filled(999) < 0.1)
                                    & (tbl['snr_' + self.band].filled(-999) > 10)
                                    & (tbl['chi_' + self.band].filled(999) < 1.5)
                                 )

        # Finally, specify the columns to keep and their order
        columns = ['detectionID_' + self.band,
                   'x_' + self.band,
                   'y_' + self.band,
                   'ra_' + self.band,
                   'dec_' + self.band,
                   self.band,
                   'err_' + self.band,
                   'chi_' + self.band,
                   'sharpness_' + self.band,
                   'sky_' + self.band,
                   'pier_' + self.band,
                   'perror_' + self.band,
                   'aperMag_' + self.band,
                   'aperMagErr_' + self.band,
                   'snr_' + self.band,
                   'magLim_' + self.band,
                   'psffwhm_' + self.band,
                   'airmass_' + self.band,
                   'mjd_' + self.band,
                   'nndist_' + self.band,
                   'pixelShift_' + self.band,
                   'clean_' + self.band]
        return tbl[columns]

    @timed
    def plot_images(self, image_fn, bg_fn, sampling=1):
        """Plots quicklook bitmaps of the data and the background estimate.

        Parameters
        ----------
        image_fn : str
            Path to save the original ccd frame image.

        bg_fn : str
            Path to save the background estimation image.

        sampling : int (optional)
            Only sample every Nth pixel when plotting the images. (default: 1)
        """
        vmin, vmax = np.percentile(self.orig_hdu.data[::10, ::10], [2, 99])
        transform = LogStretch() + ManualInterval(vmin, vmax)
        imgstyle = {'cmap': pl.cm.gist_heat, 'origin': 'lower',
                    'vmin': 0, 'vmax': 1}
        log.debug('Writing {0}'.format(os.path.basename(image_fn)))
        mimg.imsave(image_fn,
                    transform(self.orig_hdu.data[::sampling, ::sampling]),
                    **imgstyle)
        log.debug('Writing {0}'.format(os.path.basename(bg_fn)))
        mimg.imsave(bg_fn,
                    transform(self.bg_hdu.data[::sampling, ::sampling]),
                    **imgstyle)

    @timed
    def plot_subtracted_images(self, nosky_fn, nostars_fn, psf_fn, sampling=1):
        """Saves quicklook bitmaps of the PSF photometry results.

        Parameters
        ----------
        nosky_fn : str
            Path for saving the sky-subtracted bitmap image.

        nostars_fn : str
            Path for saving the sky- and star-subtracted bitmap image.

        psf_fn : str
            Path for saving a visualisation of the PSF model.

        sampling : int (optional)
           Only sample every Nth pixel when plotting the images. (default: 1)
        """
        # Determine the interval and stretch
        vmin, vmax = np.percentile(self.data[::10, ::10], [2, 99])
        transform = LogStretch() + ManualInterval(vmin, vmax)
        imgstyle = {'cmap': pl.cm.gist_heat, 'origin': 'lower',
                    'vmin': 0, 'vmax': 1}
        # Sky-subtracted image
        log.debug('Writing {0}'.format(os.path.basename(nosky_fn)))
        mimg.imsave(nosky_fn, transform(self.data[::sampling, ::sampling]),
                    **imgstyle)
        # PSF-subtracted image
        if 'daophot_subimage_path' not in self._cache:
            log.warning('Failed to plot the psf-subtracted image, '
                        'you need to call daophot.allstar first.')
        else:
            log.debug('Writing {0}'.format(os.path.basename(nostars_fn)))
            subhdu = fits.open(self._cache['daophot_subimage_path'])[0]
            mimg.imsave(nostars_fn,
                        transform(subhdu.data[::sampling, ::sampling]),
                        **imgstyle)
            # PSF model visualisation
            log.debug('Writing {0}'.format(psf_fn))
            psfhdu = fits.open(self._cache['daophot_seepsf_path'])[0]
            imgstyle['vmin'] = -1
            with np.errstate(divide='ignore', invalid='ignore'):
                imgstyle['vmax'] = np.log10(psfhdu.header['PSFHEIGH'])
                mimg.imsave(psf_fn, np.log10(psfhdu.data), dpi=300, **imgstyle)


class VphasOffsetCatalogue(object):
    """A pointing is a single (ra,dec) position in the sky.

    Parameters
    ----------
    name : str
        5-character wide identifier, composed of the 4-character wide VPHAS
        field number, followed by 'a' (first offset), 'b' (second offset),
        or 'c' (third offset used in the g and H-alpha bands only.)

    config : str
        Filename of the ".ini" file that contains the configuration parameters.
        By default, the catalogue.ini file that comes with the package will be
        used.
    """
    def __init__(self, name, config=DEFAULT_CONFIG, **kwargs):
        # Offset name must be a string of the form "0001a".
        if len(name) != 5 or not name.endswith(('a', 'b', 'c')):
            raise ValueError('"{0}" is an illegal offset name. '
                             'Expected a string of the form "0001a".'
                             .format(name))
        self.name = name
        self.kwargs = kwargs
        # Read the configuration
        self.config = config
        self.cfg = configparser.ConfigParser()
        self.cfg.read(config)
        # Make sure the root working directory exists
        try:
            os.mkdir(self.cfg['vphas']['workdir'])
        except FileExistsError:
            pass
        # Make a subdir in the workingdir for this offset
        self.workdir = tempfile.mkdtemp(prefix='{0}-'.format(name),
                                        dir=self.cfg['vphas']['workdir'])
        # Allow parallel computing
        if self.cfg['catalogue'].getboolean('use_multiprocessing', True):
            # Ask for 6 processes so that all 6 bands can run in parallel,
            # even if the number of cores is less than 6
            self.pool = multiprocessing.Pool(processes=6)
            self.cpumap = self.pool.imap
        else:
            self.cpumap = map #itertools  # Simple sequential processing

    def __del__(self):
        """Destructor."""
        # shutil.rmtree(self.workdir)
        # Make sure to get rid of any multiprocessing-forked processes;
        # they might be eating up a lot of memory!
        try:
            self.pool.terminate()
        except AttributeError:
            pass  # only applies to a multiprocessing.pool.Pool object

    @property
    def save_diagnostics(self):
        # Save diagnostic plots and tables?
        return self.cfg['catalogue'].getboolean('save_diagnostics', True)

    def _get_image_filenames(self, ccdlist=range(1, 33)):
        """Returns a dictionary mapping band names onto the image filenames."""
        offset = footprint.VphasOffset(self.name)
        images = offset.get_red_filenames()
        if self.cfg['catalogue'].getboolean('include_ugr', True):
            try:
                images.update(offset.get_blue_filenames())
            except footprint.NotObservedException as e:
                log.warning(e)  # tolerate a missing blue concat
        log.debug('{0}: filenames found: {1}'.format(self.name, images))
        return images

    @timed
    def create_catalogue(self, ccdlist=range(1, 33)):
        """Main function to create the catalogue.

        Parameters
        ----------
        ccdlist : list of ints (optional)
            Specify the HDU extension numbers to use. (default: all CCDs)

        Returns
        -------
        catalogue : `astropy.table.Table` object
        """
        with log.log_to_file(os.path.join(self.workdir, 'catalogue.log')):
            try:
                images = self._get_image_filenames(ccdlist=ccdlist)
            except footprint.NotObservedException as e:  # No data!
                log.error(e)
                return
            # Compute the catalogue for each ccd
            framecats = []
            for ccd in ccdlist:
                framecats.append(self.create_ccd_catalogue(images=images, 
                                                           ccd=ccd))
            catalogue = table.vstack(framecats, metadata_conflicts='silent')
            if self.save_diagnostics:
                self.plot_diagnostics(catalogue)
                self._plot_psf_overview(bands=images.keys())
            # This is probably unnecessary
            import gc
            log.debug('gc.collect freed {0} bytes'.format(gc.collect()))
            log.info('{0} finished, workdir was {1}'
                     .format(self.name, self.workdir))
            # Returns the catalogue as an astropy table
            return catalogue

    def _plot_psf_overview(self, bands):
        """Saves a pretty plot showing the PSF in each band.

        Parameters
        ----------
        bands : list of str
            Names of the bands to create a plot for.
        """
        from matplotlib._png import read_png
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        import matplotlib.patheffects as path_effects

        for band in bands:
            fig = pl.figure(figsize=(8, 4.5))
            ax = fig.add_subplot(1, 1, 1)
            for idx, ccd in enumerate(OMEGACAM_CCD_ARRANGEMENT):
                psf_fn = os.path.join(self.workdir, 'ccd-{0}'.format(ccd),
                                      '{0}-{1}-{2}-psf.png'.format(self.name,
                                                                   ccd,
                                                                   band))
                try:
                    imagebox = OffsetImage(read_png(psf_fn))
                except IOError:
                    continue
                xy = [idx % 8, int(idx / 8.)]
                ab = AnnotationBbox(imagebox, xy,
                                    xybox=(0., 0.),
                                    xycoords='data',
                                    boxcoords="offset points",
                                    bboxprops={'lw': 0, 'facecolor': 'black'})
                ax.add_artist(ab)
                ax.text(xy[0]-0.45, xy[1]-0.4, ccd, fontsize=8, color='white',
                        ha='left', va='top', zorder=999)

            # Aesthetics
            ax.set_xlim([-.5, 7.5])
            ax.set_ylim([3.5, -.5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            fig.text(0.025, 1.,
                     'PSF for {0}-{1} (log-stretched)'.format(self.name, band),
                     fontsize=10, ha='left', va='top', color='white')
            fig.tight_layout()

            output_fn = os.path.join(self.workdir, 'psf-{0}.png'.format(band))
            log.info('{0}: Writing {1}'
                     .format(self.name, os.path.basename(output_fn)))
            fig.savefig(output_fn, dpi=120, facecolor='black')
            pl.close(fig)

    @timed
    def create_ccd_catalogue(self, images, ccd=1):
        """Create a multi-band catalogue for the area covered by a single ccd.

        Parameters
        ----------
        images : dict
            Dictionary mapping band names onto FITS image filenames.

        ccd : int
            Number of the OmegaCam CCD, corresponding to the extension number
            in the 32-CCD multi-extension FITS images produced by the camera.

        Returns
        -------
        catalogue : `astropy.table.Table` object
            Table containing the band-merged catalogue.
        """
        # Setup the working directory to store temporary files
        ccd_workdir = os.path.join(self.workdir, 'ccd-{0}'.format(ccd))
        os.mkdir(ccd_workdir)
        log.info('Starting to build catalogue for {0} ccd {1}, '
                 'workdir: {2}'.format(self.name, ccd, ccd_workdir))

        jobs = []
        for fn in images.values():
            params = {'filename': fn,
                      'extension': ccd,
                      'workdir': ccd_workdir,
                      'cfg': self.cfg,
                      'kwargs': self.kwargs}
            jobs.append(params)
        frames = {}
        for frame in self.cpumap(frame_initialisation_task, jobs):
            frames[frame.band] = frame
        # Create a merged source table
        sourcetbl, psf_table = self.run_source_detection(frames)
        if self.save_diagnostics:
            with warnings.catch_warnings():
                # Attribute `keywords` cannot be written to FITS files
                warnings.filterwarnings("ignore",
                                        message='Attribute `keywords`(.*)')
                sourcetbl.write(os.path.join(ccd_workdir, 'sourcelist.fits'))
                psf_table.write(os.path.join(ccd_workdir, 'psflist.fits'))
        # Carry out the PSF photometry using the source table created above
        if 'u' in frames:
            bandorder = ['u', 'g', 'r', 'i', 'ha', 'r2']  # sorted by lambda
        else:  # sometimes the blue concat is missing
            bandorder = ['r', 'i', 'ha']
        jobs = []
        for band in bandorder:
            params = {'frame': frames[band],
                      'ra': sourcetbl['ra'],
                      'dec': sourcetbl['dec'],
                      'ra_psf': psf_table['ra'],
                      'dec_psf': psf_table['dec'],
                      'cfg': self.cfg,
                      'workdir': ccd_workdir}
            jobs.append(params)
        tables = [tbl for tbl in self.cpumap(photometry_task, jobs)]

        # Band-merge the tables
        merged = table.hstack(tables, metadata_conflicts='silent')
        # Merge the coordinates into a single reference position
        astrometry_order = ['i', 'r', 'r2', 'g', 'ha', 'u']
        if 'u' not in frames:
            astrometry_order = ['i', 'r', 'ha']
        merged['ra'] = coalesce([merged['ra_' + bnd]
                                 for bnd in astrometry_order])
        merged['dec'] = coalesce([merged['dec_' + bnd]
                                  for bnd in astrometry_order])
        # Add columns containing the ra/dec offsets from the reference position
        for band in frames:
            if band == 'i':
                continue  # always zero in i
            merged['offsetRa_' + band] = (3600. *
                                          (merged['ra_'+band] - merged['ra']) *
                                          np.cos(np.radians(merged['dec'])))
            merged['offsetDec_' + band] = (3600. * (merged['dec_'+band] -
                                           merged['dec']))
            merged.remove_columns(['ra_' + band, 'dec_' + band])
        # One nearest neighbour distance column is sufficient
        merged['nndist'] = merged['nndist_i']
        merged.remove_columns(['nndist_' + bnd for bnd in frames])
        # Add extra fields
        merged['field'] = self.name
        merged['ccd'] = ccd
        merged['r_i'] = merged['r'] - merged['i']
        merged['r_ha'] = merged['r'] - merged['ha']
        if 'u' in frames:
            merged['u_g'] = merged['u'] - merged['g']
            merged['g_r'] = merged['g'] - merged['r']
            merged['clean'] = (merged['clean_u'].filled(False) &
                               merged['clean_g'].filled(False) &
                               merged['clean_r'].filled(False) &
                               merged['clean_i'].filled(False) &
                               merged['clean_ha'].filled(False))
        else:
            merged['clean'] = (merged['clean_r'].filled(False) &
                               merged['clean_i'].filled(False) &
                               merged['clean_ha'].filled(False))
        # As well as returning the catalogue as a `Table`, write it to disk
        if self.save_diagnostics:
            with warnings.catch_warnings():
                # Attribute `keywords` cannot be written to FITS files
                warnings.filterwarnings("ignore",
                                        message='Attribute `keywords`(.*)')
                output_filename = os.path.join(ccd_workdir, 'catalogue.fits')
                merged.write(output_filename, format='fits')
        return merged

    def run_source_detection(self, frames):
        """Creates a list of unique sources for a set of multi-band CCD frames.

        Parameters
        ----------
        frames : dict
            Dictionary mapping band names onto `VphasFrame` objects.

        Returns
        -------
        sourcelist : `astropy.table.Table` object
        """
        jobs = []
        for frame in frames.values():
            threshold = self.cfg['sourcelist']['threshold_'+frame.band]
            params = {'frame': frame,
                      'cfg': self.cfg}
            jobs.append(params)
        sources = {}
        for tbl in self.cpumap(source_detection_task, jobs):
            sources[tbl.meta['band']] = tbl
        # Now merge the single-band lists into a master source table
        master_tbl = sources['i']
        for band in frames.keys():
            if band == 'i':
                continue  # i is the master
            current_crd = SkyCoord(master_tbl['ra']*u.deg,
                                   master_tbl['dec']*u.deg)
            new_crd = SkyCoord(sources[band]['ra']*u.deg,
                               sources[band]['dec']*u.deg)
            idx, sep2d, dist3d = new_crd.match_to_catalog_sky(current_crd)
            mask_extra = sep2d > 2*u.arcsec
            log.info('Found {0} extra sources in {1}.'
                     .format(mask_extra.sum(), band))
            master_tbl = table.vstack([master_tbl, sources[band][mask_extra]],
                                      metadata_conflicts='silent')
        log.info('Found {0} candidate sources for the catalogue.'
                 .format(len(master_tbl)))

        # Determine sources suitable to act as templates for PSF fitting;
        # we use the clean/isolated sources detected in the i-band.
        crd_i = SkyCoord(sources['i']['ra']*u.deg, sources['i']['dec']*u.deg)

        # First, make sure the PSF template stars have no nearby neighbour;
        # the cutoff distance is adaptive (70% percentile).
        idx, nneighbor_dist, dist3d = crd_i.match_to_catalog_sky(crd_i,
                                                                 nthneighbor=2)
        cutoff = np.percentile(nneighbor_dist, 70) * nneighbor_dist.unit
        if cutoff > 5*u.arcsec:
            cutoff = 5.*u.arcsec  # Don't be too strict in sparse fields
        mask_bad_template = nneighbor_dist < cutoff
        log.info('PSF neighbour limit = {0:.2f} (rejects {1}).'
                 .format(cutoff.to(u.arcsec), mask_bad_template.sum()))

        # Second, demand a reliable detection in each band within 0.5 arcsec
        for band in frames.keys():
            # Do not require a detection in u (very sparse)
            if band == 'u':
                continue
            # Sigma-clip on quality indicators. In particular, outlying sky
            # values often flag spurious objects in the wings of bright stars.
            mask_bad = np.repeat(False, len(sources[band]))
            for col in ['CHI', 'SHARPNESS', 'MSKY_PHOT', 'STDEV', 'SSKEW']:
                mask_bad |= sigma_clip(sources[band][col].data,
                                       sig=3.0, iters=None).mask
            if band == 'i':
                mask_bad_template[mask_bad] = True
            else:
                # Demand that templates must be deemed good in this band!
                crd_band = SkyCoord(sources[band]['ra'][~mask_bad]*u.deg,
                                    sources[band]['dec'][~mask_bad]*u.deg)
                idx, sep2d, dist3d = crd_i.match_to_catalog_sky(crd_band)
                mask_bad_template[sep2d > 0.5*u.arcsec] = True
            log.info('PSF sigma-clipping in {0}: now {1} rejected.'
                     .format(band, mask_bad_template.sum()))

        psf_table = sources['i'][~mask_bad_template]
        log.info('Found {0} candidate stars for PSF model fitting (rejected '
                 '{1}).'.format(len(psf_table), mask_bad_template.sum()))
        return master_tbl, psf_table

    def plot_diagnostics(self, tbl):
        fig = pl.figure(figsize=(8.27, 11.7))
        fig.suptitle('CHI vs magnitude', fontsize=24)
        fig.subplots_adjust(top=0.94, bottom=0.07, hspace=0)
        for idx, bnd in enumerate(VPHAS_BANDS):
            ax = fig.add_subplot(6, 1, idx+1)
            ax.plot([10, 25], [1, 1])
            try:
                ax.scatter(tbl[bnd], tbl['chi_' + bnd])
            except KeyError:
                pass  # missing band
            ax.set_xlim([12, 19])
            ax.set_ylim([0, 4])
            ax.set_yticks([0, 1, 2, 3])
            ax.text(0.02, 0.95, bnd, transform=ax.transAxes,
                    fontsize=20, va='top')
            if idx == 5:
                ax.set_xlabel('PSF magnitude')
            else:
                ax.xaxis.set_ticklabels([])
                if idx == 3:
                    ax.set_ylabel('CHI')
                    ax.yaxis.set_label_coords(-0.06, 1)
        fig.savefig(os.path.join(self.workdir, 'diagnostic-chi-vs-mag.jpg'))
        pl.close(fig)

        fig = pl.figure(figsize=(8.27, 11.7))
        fig.suptitle('Sky flux vs magnitude', fontsize=24)
        fig.subplots_adjust(top=0.94, bottom=0.07, hspace=0)
        for idx, bnd in enumerate(VPHAS_BANDS):
            ax = fig.add_subplot(6, 1, idx+1)
            try:
                ax.scatter(tbl[bnd], tbl['sky_' + bnd])
            except KeyError:
                pass  # missing band
            ax.set_xlim([12, 18])
            ax.text(0.02, 0.95, bnd, transform=ax.transAxes,
                    fontsize=20, va='top')
            if idx == 5:
                ax.set_xlabel('PSF magnitude')
            else:
                ax.xaxis.set_ticklabels([])
                if idx == 3:
                    ax.set_ylabel('Sky flux [ADU]')
                    ax.yaxis.set_label_coords(-0.06, 1)
        fig.savefig(os.path.join(self.workdir, 'diagnostic-sky-vs-mag.jpg'))
        pl.close(fig)


############
# FUNCTIONS
############

# Define function for parallel processing
def frame_initialisation_task(par):
    """Returns a `VphasFrame` instance for a given FITS filename/extension.

    This is defined as a separate function with a single argument,
    to allow pickling and hence the use of multiprocessing.map.

    Parameters
    ----------
    par : dict
        Parameters.
    """
    log.debug('Creating VphasFrame instance for {0}[{1}]'.format(
              par['filename'], par['extension']))
    frame = VphasFrame(par['filename'], par['extension'], cfg=par['cfg'],
                       workdir=par['workdir'], **par['kwargs'])
    frame.populate_cache()
    if par['cfg']['catalogue'].getboolean('save_diagnostics', True):
        frame.plot_images(image_fn=os.path.join(par['workdir'],
                                                frame.name+'-data.png'),
                          bg_fn=os.path.join(par['workdir'],
                                             frame.name+'-bg.png'))
    return frame


def source_detection_task(par):
    """Returns a table of sources in a VphasFrame.

    Parameters
    ----------
    par : dict
        par['frame'] : `VphasFrame` instance
        par['cfg'] : `ConfigParser` instance

    Returns
    -------
    tbl : `astropy.table.Table` object
        Sources detected in par['frame']
    """
    # the psfrad and maxiter parameters were carefully chosen to speed
    # up source detection; psfrad_fwhm should be bigger for good photometry
    conf = par['cfg']['sourcelist']
    threshold = float(conf['threshold_' + par['frame'].band])
    tbl = par['frame'].compute_source_table(
                           threshold=threshold,
                           roundlo=float(conf.get('roundlo', -0.75)),
                           roundhi=float(conf.get('roundhi', 0.75)),
                           psfrad_fwhm=float(conf.get('psfrad_fwhm', 3.)),
                           fitrad_fwhm=float(conf.get('fitrad_fwhm', 1.)),
                           maxiter=int(conf.get('maxiter', 20)),
                           maxnpsf=int(conf.get('maxnpsf', 20)),
                           varorder=int(conf.get('varorder', 0)),
                           mergerad_fwhm=float(conf.get('mergerad_fwhm', 0.)),
                           annulus_fwhm=float(conf.get('annulus_fwhm', 4.)),
                           dannulus_fwhm=float(conf.get('dannulus_fwhm', 2.))
                           )
    return tbl


def photometry_task(par):
    """Returns a table with the photometry for a list of sources.

    Parameters
    ----------
    par : dict
        par['frame'] : `VphasFrame` instance
        par['ra'] : RA coordinates of the sources to be measured
        par['dec'] : Dec coordinates "
        par['ra_psf'] : RA coordinates of good stars for PSF-fitting
        par['dec_psf'] : Dec coordinates "
        par['cfg'] : `ConfigParser` instance

    Returns
    -------
    tbl : `astropy.table.Table` object
        Photometry.
    """
    conf = par['cfg']['photometry']
    tbl = par['frame'].photometry(
              par['ra'],
              par['dec'],
              ra_psf=par['ra_psf'],
              dec_psf=par['dec_psf'],
              psfrad_fwhm=float(conf.get('psfrad_fwhm', 4.)),
              fitrad_fwhm=float(conf.get('fitrad_fwhm', 1.)),
              maxiter=int(conf.get('maxiter', 10)),
              maxnpsf=int(conf.get('maxnpsf', 60)),
              varorder=int(conf.get('varorder', 0)),
              mergerad_fwhm=float(conf.get('mergerad_fwhm', 0.)),
              annulus_fwhm=float(conf.get('annulus_fwhm', 4.)),
              dannulus_fwhm=float(conf.get('dannulus_fwhm', 2.)),
              fitsky=conf.get('fitsky', 'no'),
              sannulus_fwhm=float(conf.get('sannulus_fwhm', 1.)),
              wsannulus_fwhm=float(conf.get('wsannulus_fwhm', 3.))
              )
    # Save the sky- and psf-subtracted images as diagnostics
    if par['cfg']['catalogue'].getboolean('save_diagnostics', True):
        fn = [os.path.join(par['workdir'], par['frame'].name+'-'+suffix+'.png')
              for suffix in ['nostars', 'nosky', 'psf']]
        par['frame'].plot_subtracted_images(nostars_fn=fn[0], nosky_fn=fn[1],
                                            psf_fn=fn[2])
    return tbl
