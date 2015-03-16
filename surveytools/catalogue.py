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

from . import SURVEYTOOLS_CONFIGDIR, VPHAS_DATA_PATH, OMEGACAM_CCD_ARRANGEMENT, WORKDIR_PATH
from .utils import cached_property, timed
from . import footprint


###########
# CLASSES
###########

class VphasFrame(object):
    """Class representing a single-CCD image obtained by ESO's VST telescope.

    Parameters
    ----------
    filename : str
        Path to the image FITS file.

    extension : int (optional)
        Extension of the image in the FITS file. (default: 0)

    confidence_threshold : float (optional)
        Pixels with a confidence lower than the threshold will be masked out

    subtract_sky : boolean (optional)
        Use a high-pass filter to subtract the background sky? (default: True)
    """
    def __init__(self, filename, extension=0, confidence_threshold=80.,
                 subtract_sky=True, datadir=VPHAS_DATA_PATH, workdir=WORKDIR_PATH):
        if os.path.exists(filename):
            self.orig_filename = filename
        elif os.path.exists(os.path.join(datadir, filename)):
            self.orig_filename = os.path.join(datadir, filename)
        else:
            raise IOError('File not found:' + os.path.join(datadir, filename))
        self.orig_extension = extension
        self.workdir = tempfile.mkdtemp(prefix='frame-{0}-{1}-'.format(filename, extension),
                                        dir=workdir)
        self._cache = {}
        self.confidence_threshold = confidence_threshold
        self.filename, self.extension = self._preprocess_image(subtract_sky=subtract_sky)

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
    def _preprocess_image(self, subtract_sky=True, mask_bad_pixels=True):
        """Prepare the image for photometry by IRAF.

        IRAF/DAOPHOT does not appears to support RICE-compressed files,
        and does not allow a weight (confidence) map to be provided.
        This method hence saves the image to an uncompressed FITS file in
        which low-confidence areas are masked out, suitable for
        analysis by IRAF tasks.  This method will also subtract the
        background estimated using a high-pass filter.

        Returns
        -------
        (filename, extension): (str, int)
            Path and HDU number of the pre-processed FITS file.
        """ 
        fts = fits.open(self.orig_filename)
        hdu = fts[self.orig_extension]
        fltr = fts[0].header['ESO INS FILT1 NAME']
        # Create bad pixel mask
        self.confidence_map_path = os.path.join(VPHAS_DATA_PATH, hdu.header['CIR_CPM'].split('[')[0])
        confmap_hdu = fits.open(self.confidence_map_path)[self.orig_extension]
        bad_pixel_mask = confmap_hdu.data < self.confidence_threshold
        # Estimate the background in a mesh of (41, 32) pixels; which is chosen
        # to fit an integer number of times in the image size (4100, 2048).
        # At the pixel scale of 0.21 arcsec/px, this corresponds to ~10 arcsec.
        bg = Background(hdu.data, (41, 32), filter_shape=(6, 6),
                        mask=bad_pixel_mask,
                        method='median', sigclip_sigma=3., sigclip_iters=5)
        log.debug('{0} sky estimate = {1:.1f} +/- {2:.1f}'.format(
                  fltr, bg.background_median, bg.background_rms_median))
        self.sky = bg.background_median
        self.sky_sigma = bg.background_rms_median

        # Subtract the background
        if subtract_sky:
            # ensure the median level remains the same
            imgdata = hdu.data - (bg.background - bg.background_median)
        else:
            imgdata = hdu.data
        # Apply bad pixel mask
        if mask_bad_pixels:
            imgdata[bad_pixel_mask] = -1
        # Write the sky-subtracted, bad-pixel-masked image to a new FITS file which IRAF can understand
        path = os.path.join(self.workdir, '{0}.fits'.format(fltr))
        log.debug('Writing background-subtracted image to {0}'.format(path))
        newhdu = fits.PrimaryHDU(imgdata, hdu.header)
        newhdu.header.extend(fts[0].header, unique=True)
        del newhdu.header['RADECSYS']  # non-standard keyword
        newhdu.writeto(path)
        # Also write the background frame
        self.background_path = os.path.join(self.workdir, '{0}-bg.fits'.format(fltr))
        log.debug('Writing background image to {0}'.format(self.background_path))
        newhdu = fits.PrimaryHDU(bg.background, hdu.header)
        newhdu.writeto(self.background_path)
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
    def background_hdu(self):
        return fits.open(self.background_path)[0]

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
        
        In the broad-bands, we tolerate up to 10-sigma below the average sky
        level, which is very permissive because images may contain strong 
        background gradients.  In narrowband H-alpha (NB_659), we fix the
        minimum value at 1 because astrophysical nebulosity may trigger
        exceptionally strong gradients.
        """
        # What is the minimum good pixel value? 
        skymin = self.sky - 10 * self.sky_sigma
        if skymin < 0 or self.filtername == 'NB_659':
            skymin = 0
        return skymin

    @cached_property
    def datamax(self):
        """Returns the maximum good (non-saturated) pixel value. [adu]

        The VST/OmegaCAM manual (VST-MAN-OCM-23100-3110-2_7_1) suggests that the
        detector is linear (within ~1%) up to the saturation level.
        The saturation level is not exactly 2^16 = 65536 due to bias subtraction etc,
        so we conservatively ignore pixel values over 55000 ADU.

        It is VERY important to be conservative, because the cores of saturated
        stars should be avoided during PSF fitting. Experience suggests
        that charge bleeding may cause pixels well below the nominal
        saturation level to give an unrepresentative view of the PSF.
        """
        return 55000

    @cached_property
    def seeing(self):
        """Estimate of the seeing full-width at half-maximum."""
        return self.hdu.header['SEEING']  # pixels

    @property
    def psf_fwhm(self):
        """The Full-Width-Half-Maximum of a 2D Gaussian PSF model fit."""
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
        log.debug('{0}: starting a new daophot session for file {1}'.format(self.band, image_path))
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

    def compute_source_table(self, threshold=3., chi_max=5., **kwargs):
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
        mask = (
                (sources['SNR'] > threshold)
                & (np.abs(sources['SHARPNESS']) < 1)
                & (sources['CHI'] < chi_max)
                & (sources['PIER_ALLSTAR'] == 0)
                & (sources['PIER_PHOT'] == 0)
                )
        sources.meta['band'] = self.band
        tbl = sources[mask]
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
        #dp.daofind()
        dp.apphot(coords=psf_tbl_filename)
        dp.pstselect()
        psf_scatter = dp.psf()
        # Carry out the aperture and PSF photometry
        dp.apphot(coords=coords_tbl_filename)
        dp.allstar()
        # Remember the path of the PSF and the PSF-subtracted image FITS files
        self._cache['daophot_subimage_path'] = dp.subimage_path
        self._cache['daophot_seepsf_path'] = dp.seepsf_path

        # The code below transforms the table into a user-friendly format
        tbl = dp.get_allstar_phot_table()
        tbl.meta['band'] = self.band
        # Add celestial coordinates ra/dec as columns
        ra, dec = self.pix2world(tbl['XCENTER_ALLSTAR'],
                                 tbl['YCENTER_ALLSTAR'],
                                 origin=1)
        ra_col = Column(name=self.band+'Ra', data=ra)
        dec_col = Column(name=self.band+'Dec', data=dec)
        tbl.add_columns([ra_col, dec_col])
        # Rename columns from the DAOPHOT defaults to something sensible
        tbl['MAG_ALLSTAR'].name = self.band
        tbl['MERR_ALLSTAR'].name = self.band + 'Err'
        tbl['CHI'].name = self.band + 'Chi'
        tbl['PIER_ALLSTAR'].name = self.band + 'Pier'
        tbl['PERROR_ALLSTAR'].name = self.band + 'Perror'
        tbl['MAG_PHOT'].name = self.band + 'AperMag'
        tbl['MERR_PHOT'].name = self.band + 'AperMagErr'
        tbl['SNR'].name = self.band + 'SNR'
        tbl['LIM3SIG'].name = self.band + 'MagLim'
        tbl['ID'].name = self.band + 'ID'
        tbl['XCENTER_ALLSTAR'].name = self.band + 'X'
        tbl['YCENTER_ALLSTAR'].name = self.band + 'Y'
        # Add extra columns and tune the value of others
        with np.errstate(invalid='ignore'):
            # Remove the untrustworthy magnitude estimates for undetected sources
            mask_too_faint = (
                                 (tbl[self.band+'SNR'] < 3)
                                 | (tbl[self.band] > tbl[self.band+'MagLim'])
                              )
            tbl[self.band][mask_too_faint] = np.nan
            tbl[self.band+'Err'][mask_too_faint] = np.nan
            tbl[self.band+'AperMag'][mask_too_faint] = np.nan
            tbl[self.band+'AperMagErr'][mask_too_faint] = np.nan
            # Shift of the source centroid during PSF fitting [pixels]
            tbl[self.band+'Shift'] = np.hypot(tbl[self.band+'X'] - tbl['XINIT'],
                                              tbl[self.band+'Y'] - tbl['YINIT'])
            tbl[self.band+'DetectionID'] = ['{0}-{1}'.format(self.name, idx) for idx in tbl[self.band+'ID']]
            tbl[self.band+'10sig'] = (
                                (~np.isnan(tbl[self.band].filled(np.nan)))
                                & (tbl[self.band + 'SNR'] > 10)
                                & (tbl[self.band + 'Pier'] == 0)
                                & (tbl[self.band + 'Chi'] < 1.5)
                                & (tbl[self.band + 'Shift'] < 1)
                                 )
            tbl[self.band+'PsfScatter'] = [psf_scatter] * len(tbl)
        # Finally, specify the columns to keep and their order
        columns = [self.band+'DetectionID', self.band+'ID',
                   self.band+'X', self.band+'Y',
                   self.band+'Ra', self.band+'Dec',
                   self.band, self.band+'Err', self.band+'Chi',
                   self.band+'Pier', self.band+'Perror',
                   self.band+'AperMag', self.band+'AperMagErr',
                   self.band+'SNR', self.band+'MagLim',
                   self.band+'Shift', self.band+'10sig']
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
        log.debug('Writing {0}'.format(image_fn))
        mimg.imsave(image_fn, transform(self.orig_hdu.data[::sampling, ::sampling]), **imgstyle)
        log.debug('Writing {0}'.format(bg_fn))
        mimg.imsave(bg_fn, transform(self.background_hdu.data[::sampling, ::sampling]), **imgstyle)

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
        log.debug('Writing {0}'.format(nosky_fn))
        mimg.imsave(nosky_fn, transform(self.data[::sampling, ::sampling]),
                    **imgstyle)
        # PSF-subtracted image
        if 'daophot_subimage_path' not in self._cache:
            log.warning('Failed to plot the psf-subtracted image, '
                        'you need to call daophot.allstar first.')
        else:
            log.debug('Writing {0}'.format(nostars_fn))
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
    """
    def __init__(self, name, configfile=None, **kwargs):
        # Offset name must be a string of the form "0001a".
        if len(name) != 5 or not name.endswith(('a', 'b', 'c')):
            raise ValueError('"{0}" is an illegal offset name. '
                             'Expected a string of the form "0001a".'.format(name))
        self.name = name
        self.kwargs = kwargs
        # Load the configuration
        if configfile is None:
            configfile = os.path.join(SURVEYTOOLS_CONFIGDIR,
                                      'vphas-offset-catalogue-default.ini')
        self.cfg = configparser.ConfigParser()
        self.cfg.read(configfile)
        # Make sure the root working directory exists
        try:
            os.mkdir(self.cfg['offset_catalogue']['workdir'])
        except FileExistsError:
            pass
        # Make a subdir in the workingdir for this offset
        self.workdir = tempfile.mkdtemp(prefix='{0}-'.format(name),
                                        dir=self.cfg['offset_catalogue']['workdir'])
        # Allow "self.cpufarm.imap(f, param)" to be used for parallel processing
        if self.cfg['offset_catalogue'].getboolean('use_multiprocessing'):
            self.cpufarm = multiprocessing.Pool()
        else:
            self.cpufarm = itertools  # Simple sequential processing

    def __del__(self):
        """Destructor."""
        #shutil.rmtree(self.workdir)
        # Make sure to get rid of any multiprocessing-forked processes;
        # they might be eating up a lot of memory!
        try:
            self.cpufarm.terminate()
        except AttributeError:
            pass  # only applies to a multiprocessing.pool.Pool object

    def _get_image_filenames(self, ccdlist=range(1, 33), include_ugr=True):
        """Returns a dictionary mapping band names onto the image filenames."""
        offset = footprint.VphasOffset(self.name)
        image_fn = offset.get_red_filenames()
        if include_ugr:
            try:
                image_fn.update(offset.get_blue_filenames())
            except NotObservedException as e:
                log.warning(e.message)  # tolerate a missing blue concat
        log.debug('{0}: filenames found: {1}'.format(self.name, image_fn))
        return image_fn

    @timed
    def create_catalogue(self, ccdlist=range(1, 33), include_ugr=True):
        """Main function to create the catalogue.

        Parameters
        ----------
        ccdlist : list of ints (optional)
            Specify the HDU extension numbers to use. (default: all CCDs)

        include_ugr : bool (optional)
            Include ugr (blue concat) data if available? (default: True)

        Returns
        -------
        catalogue : `astropy.table.Table` object
        """
        with log.log_to_file(os.path.join(self.workdir, 'create_catalogue.log')):
            try:
                image_fn = self._get_image_filenames(ccdlist=ccdlist, include_ugr=include_ugr)
            except NotObservedException as e:
                # We do not tolerate red data missing
                log.error(e.message)
                return
            # Having obtained filenames, compute the catalogue for each ccd
            framecats = []
            for ccd in ccdlist:
                framecats.append(self.create_ccd_catalogue(images=image_fn, ccd=ccd))
            catalogue = table.vstack(framecats, metadata_conflicts='silent')
            # Plot evaluation
            self._plot_psf_overview(bands=image_fn.keys())
            # This is probably unnecessary
            import gc
            log.debug('gc.collect freed {0} bytes'.format(gc.collect()))
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
                     'PSFs for {0}-{1} (log-stretched)'.format(self.name, band),
                     fontsize=10, ha='left', va='top', color='white')
            fig.tight_layout()

            output_fn = os.path.join(self.workdir, 'psf-{0}.png'.format(band))
            log.info('{0}: Writing {1}'.format(self.name, output_fn))
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
            Number of the OmegaCam CCD, corresponding to the extension number in
            the 32-CCD multi-extension FITS images produced by the camera.

        Returns
        -------
        catalogue : `astropy.table.Table` object
            Table containing the band-merged catalogue.
        """
        # Setup the working directory to store temporary files
        ccd_workdir = os.path.join(self.workdir, 'ccd-{0}'.format(ccd))
        os.mkdir(ccd_workdir)
        log.info('{0}: started catalogueing ccd {1}, '
                 'workdir: {2}'.format(self.name, ccd, ccd_workdir))

        jobs = []
        for fn in images.values():
            params = {'filename': fn,
                      'extension': ccd,
                      'workdir': ccd_workdir,
                      'kwargs': self.kwargs}
            jobs.append(params)
        frames = {}
        for frame in self.cpufarm.imap(frame_initialisation_task, jobs):
            frames[frame.band] = frame
        # Create a merged source table
        source_table, psf_table = self.run_source_detection(frames)
        with warnings.catch_warnings():
            # Attribute `keywords` cannot be written to FITS files
            warnings.filterwarnings("ignore", message='Attribute `keywords`(.*)')
            source_table.write(os.path.join(ccd_workdir, 'sourcelist.fits'))
            psf_table.write(os.path.join(ccd_workdir, 'psflist.fits'))
        # Carry out the PSF photometry using the source table created above
        jobs = []
        for band in frames:
            params = {'frame': frames[band],
                      'ra': source_table['ra'],
                      'dec': source_table['dec'],
                      'ra_psf': psf_table['ra'],
                      'dec_psf': psf_table['dec'],
                      'cfg': self.cfg,
                      'workdir': ccd_workdir}
            jobs.append(params)
        tables = [tbl for tbl in self.cpufarm.imap(photometry_task, jobs)]

        # Band-merge the tables
        merged = table.hstack(tables, metadata_conflicts='silent')
        merged['field'] = self.name
        merged['ccd'] = ccd
        merged['rmi'] = merged['r'] - merged['i']
        merged['rmha'] = merged['r'] - merged['ha']
        if 'u' in merged.colnames:
            merged['umg'] = merged['u'] - merged['g']
            merged['gmr'] = merged['g'] - merged['r']
            merged['a10'] = (merged['u10sig'].filled(False)
                             & merged['g10sig'].filled(False)
                             & merged['r10sig'].filled(False)
                             & merged['i10sig'].filled(False)
                             & merged['ha10sig'].filled(False))
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
            threshold = self.cfg['source_detection']['threshold_'+frame.band]
            params = {'frame': frame,
                      'cfg': self.cfg}
            jobs.append(params)
        source_tables = {}
        for tbl in self.cpufarm.imap(source_detection_task, jobs):
            source_tables[tbl.meta['band']] = tbl
        # Now merge the single-band lists into a master source table
        master_table = source_tables['i']
        for band in frames.keys():
            if band == 'i':
                continue  # i is the master
            current_coordinates = SkyCoord(master_table['ra']*u.deg, master_table['dec']*u.deg)
            new_coordinates = SkyCoord(source_tables[band]['ra']*u.deg, source_tables[band]['dec']*u.deg)
            idx, sep2d, dist3d = new_coordinates.match_to_catalog_sky(current_coordinates)
            mask_extra = sep2d > 2*u.arcsec
            log.info('Found {0} extra sources in {1}.'.format(mask_extra.sum(), band))
            master_table = table.vstack([master_table, source_tables[band][mask_extra]],
                                        metadata_conflicts='silent')
        log.info('Found {0} candidate sources for the catalogue.'.format(len(master_table)))

        # Determine sources suitable for PSF fitting;
        # we will use the i-band coordinates of sources with a good CHI-score;
        # without a suspicious background value;
        # that have been detected in all bands (bar u)
        crd_i = SkyCoord(source_tables['i']['ra']*u.deg, source_tables['i']['dec']*u.deg)

        # First, make sure the nearest neighbour is 3 arcsec away
        idx, nneighbor_dist, dist3d = crd_i.match_to_catalog_sky(crd_i, nthneighbor=2)
        mask_bad_template = ((nneighbor_dist < 2.*u.arcsec)
                             | (source_tables['i']['CHI'] > 1.5))
        # Then, sigma-clip on various properties
        # in particular, outlying sky values point to suspect templates
        for col in ['CHI', 'SHARPNESS', 'MSKY_PHOT', 'STDEV', 'SSKEW', 'NSREJ']:
            mask_bad_template |= sigma_clip(source_tables['i'][col].data,
                                            sig=3.0, iters=None).mask
        # Finally, demand a detection in each band except u
        for band in frames.keys():
            if band in ['i', 'u']:  # do not require u to have the detection
                continue
            new_coordinates = SkyCoord(source_tables[band]['ra']*u.deg, source_tables[band]['dec']*u.deg)
            #mask_reliable = source_tables[band]['CHI'] < 1.5
            idx, sep2d, dist3d = crd_i.match_to_catalog_sky(new_coordinates)
            # Star must exist in other band and have a good fit
            mask_bad_template[sep2d > 0.5*u.arcsec] = True

        psf_table = source_tables['i'][~mask_bad_template]
        log.info('Found {0} candidate stars for PSF model fitting (rejected '
                 '{1}).'.format(len(psf_table), mask_bad_template.sum()))
        return master_table, psf_table



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
    frame = VphasFrame(par['filename'], par['extension'],
                       workdir=par['workdir'], **par['kwargs'])
    frame.populate_cache()
    frame.plot_images(image_fn=os.path.join(par['workdir'], frame.name+'-data.png'),
                      bg_fn=os.path.join(par['workdir'], frame.name+'-bg.png'))
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
    conf = par['cfg']['source_detection']
    threshold = float(conf['threshold_' + par['frame'].band])
    tbl = par['frame'].compute_source_table(
                           roundlo=float(conf.get('roundlo', -0.75)),
                           roundhi=float(conf.get('roundhi', 0.75)),
                           psfrad_fwhm=float(conf.get('psfrad_fwhm', 3.)),
                           maxiter=int(conf.get('maxiter', 20)),
                           threshold=threshold)
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
    tbl = par['frame'].photometry(par['ra'],
                                  par['dec'],
                                  ra_psf=par['ra_psf'],
                                  dec_psf=par['dec_psf'],
                                  psfrad_fwhm=float(conf.get('psfrad_fwhm', 8.)),
                                  maxiter=int(conf.get('maxiter', 10)),
                                  mergerad_fwhm=float(conf.get('mergerad_fwhm', 0.)))
    # Save the sky- and psf-subtracted images
    fn = [os.path.join(par['workdir'], par['frame'].name+'-'+suffix+'.png')
          for suffix in ['nostars', 'nosky', 'psf']]
    par['frame'].plot_subtracted_images(nostars_fn=fn[0], nosky_fn=fn[1],
                                        psf_fn=fn[2])
    return tbl
