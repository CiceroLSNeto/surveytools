"""Tools to create photometric catalogues from VPHAS data.

Classes
-------
VphasFrame
VphasFrameCatalogue
VphasOffsetPointing

Example use
-----------
Create a photometric catalogue of VPHAS pointing 0149a:
```
import vphas
pointing = vphas.VphasOffsetPointing('0149a')
pointing.create_catalogue().write('mycatalogue.fits')
```

Terminology
-----------
This module makes use of the concept of a `field`, a `pointing`, and a `frame`.
defined as follows:
* `field`: a region in the sky covered by 2 (or 3) offset pointings of the
           telescope, identified using a 4-character wide, zero-padded number
           string, e.g. '0149'.
* `pointing`: a single position in the sky denoting one of the offsets that
              make up a field, e.g. '0149a' (first offset),
              '0149b' (second offset), '0149c' (third offset, for H-alpha and
              some g-band observations only).
* `frame`: area covered by a single ccd of a pointing, e.g. '0149a-8'.

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

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.image as mimg

import astropy
from astropy.io import fits
from astropy import log
from astropy import table
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.timer import timefunc

import photutils
import photutils.morphology
from photutils.background import Background

from . import SURVEYTOOLS_DATA
from .utils import cached_property, timed


###########
# CONSTANTS
###########

WORKDIR_DEFAULT = '/home/gb/tmp/vphas-workdir'  # Where can we store temporary files?
# Directory containing the calibration frames (confmaps and flat fields)
DATAPATH = '/home/gb/tmp/vphasdisk'
CALIBDIR = os.path.join(DATAPATH, 'calib')
DATADIR_DEFAULT = os.path.join(DATAPATH, 'single')

###########
# CLASSES
###########

class NotObservedException(Exception):
    """Raised if a requested field has not been observed yet."""
    pass


class VphasFrame(object):
    """Class representing a single-CCD image obtained by ESO's VST telescope.

    Parameters
    ----------
    filename : str
        Path to the image FITS file.

    extension : int (optional)
        Extension of the image in the FITS file. (default: 0)

    band : str (optional)
        Colloquial name of the filter.

    subtract_sky : boolean (optional)
        TBD

    confidence_threshold : float (optional)
        Pixels with a confidence lower than the threshold will be masked out
    """
    def __init__(self, filename, extension=0, confidence_threshold=80.,
                 subtract_sky=True, datadir=DATADIR_DEFAULT, workdir=WORKDIR_DEFAULT):
        if os.path.exists(filename):
            self.orig_filename = filename
        elif os.path.exists(os.path.join(datadir, filename)):
            self.orig_filename = os.path.join(datadir, filename)
        else:
            raise IOError('File not found:' + os.path.join(datadir, filename))
        self.orig_extension = extension
        self.workdir = tempfile.mkdtemp(prefix='frame-{0}-{1}-'.format(filename, extension), dir=workdir)
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
        """Save the image to FITS file which can be understood by IRAF.

        IRAF/DAOPHOT does not appears to support RICE-compressed files,
        and does not allow a weight (confidence) map to be provided.
        This method saves the image to an uncompressed FITS file in
        which low-confidence areas are masked out, suitable for
        analysis by IRAF tasks.

        In addition, this step allows large-scale structures in the sky
        background to be subtracted.

        Returns
        -------
        (filename, extension)
            Of the newly created pre-processed frame.
        """ 
        fts = fits.open(self.orig_filename)
        hdu = fts[self.orig_extension]
        fltr = fts[0].header['ESO INS FILT1 NAME']
        imgdata = hdu.data
        # Create bad pixel mask
        self.confidence_map_path = os.path.join(CALIBDIR, hdu.header['CIR_CPM'].split('[')[0])
        confmap_hdu = fits.open(self.confidence_map_path)[self.orig_extension]
        bad_pixel_mask = confmap_hdu.data < self.confidence_threshold
        if mask_bad_pixels:
            imgdata[bad_pixel_mask] = -1
        # Estimate the background
        bg = Background(imgdata, (41, 64), filter_shape=(2, 2),
                                     mask=bad_pixel_mask, method='median',
                                     sigclip_sigma=3., sigclip_iters=5)
        log.debug('{0} sky estimate = {1:.1f} +/- {2:.1f}'.format(fltr, bg.background_median, bg.background_rms_median))
        self.sky = bg.background_median
        self.sky_sigma = bg.background_rms_median
        # Subtract the background
        if subtract_sky:
            # ensure the median level remains the same
            imgdata = imgdata - (bg.background - bg.background_median)
        # Write the sky-subtracted, bad-pixel-masked image to a new FITS file which IRAF can understand
        path = os.path.join(self.workdir, '{0}.fits'.format(fltr))
        log.debug('Writing background-subtracted image to {0}'.format(path))
        newhdu = fits.PrimaryHDU(imgdata, hdu.header)
        newhdu.header.extend(fts[0].header, unique=True)
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
        """VPHAS name of the frame, e.g. '0001a-8-r'."""
        return '{0}-{1}-{2}'.format(self.fieldname, self.orig_extension, self.band)

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
        so we conservatively ignore pixel values over 60000 ADU.
        """
        return 60000

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
        return astropy.wcs.WCS(self.hdu.header).wcs_world2pix(ra, dec, 1)

    def pix2world(self, x, y, origin=1):
        """Shorthand to convert pixel(x,y) into equatorial(ra,dec) coordinates.

        Use origin=1 if x/y positions were produced by IRAF/DAOPHOT,
        0 if they were produced by astropy."""
        return astropy.wcs.WCS(self.hdu.header).wcs_pix2world(x, y, 1)

    def _estimate_psf(self, threshold=20.):
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
        theta = myfit.theta.value
        # pyraf will complain over a negative theta
        if theta < 0:
            theta += 180
        log.debug(self.band+' PSF FWHM = {0:.1f}px; ratio = {1:.1f}; theta = {2:.1f}'.format(fwhm, ratio, theta))
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

    def compute_source_table(self, threshold=3., **kwargs):
        # We allow saturated pixels during source detection.  This allows
        # saturated stars to be centroided reasonably well, and reduces
        # the number of spurious sources in the wings.
        #iraf.datapars.datamax = 99999
        dp = self.daophot(threshold=threshold, **kwargs)
        #sources = dp.psf_photometry()
        dp.daofind()
        dp.apphot()
        dp.psf()
        dp.allstar()
        sources = dp.get_allstar_phot_table()
                
        mask = (
                (sources['SNR'] > threshold)
                & (np.abs(sources['SHARPNESS']) < 1)
                & (sources['CHI'] < 5)
                & (sources['PIER_ALLSTAR'] == 0)
                & (sources['PIER_PHOT'] == 0)
                )
        sources.meta['band'] = self.band
        tbl = sources[mask]
        log.info('Identified {0} sources in {1} at sigma > {2}'.format(len(tbl), self.band, threshold))
        # Add ra/dec columns
        ra, dec = self.pix2world(tbl['XCENTER_ALLSTAR'], tbl['YCENTER_ALLSTAR'], origin=1)
        ra_col = Column(name='ra', data=ra)
        dec_col = Column(name='dec', data=dec)
        tbl.add_columns([ra_col, dec_col])
        return tbl

    def compute_photometry(self, ra, dec, **kwargs):
        """Computes photometry (PSF & aperture) for the requested sources.

        Returns a table.
        """
        # Save the coordinates to a file suitable for daophot
        x, y = self.world2pix(ra, dec)
        col_x = Column(name='XCENTER', data=x)
        col_y = Column(name='YCENTER', data=y)
        coords_tbl = Table([col_x, col_y])
        coords_tbl_filename = os.path.join(self.workdir, 'coords-tbl.txt')
        coords_tbl.write(coords_tbl_filename, format='ascii')
        # Now run daophot
        dp = self.daophot(**kwargs)

        # First, create a proper PSF based on the daofind source list
        # Using the master source list to compute the PSF is dangerous,
        # because it might include the odd spurious object
        dp.daofind()
        dp.apphot()
        dp.psf()

        dp.apphot(coords=coords_tbl_filename)
        #dp.psf()
        dp.allstar()

        tbl = dp.get_allstar_phot_table()
        tbl.meta['band'] = self.band
        # Add celestial coordinates ra/dec as columns
        ra, dec = self.pix2world(tbl['XCENTER_ALLSTAR'], tbl['YCENTER_ALLSTAR'], origin=1)
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
            mask_too_faint = (tbl[self.band+'SNR'] < 3) | (tbl[self.band] > tbl[self.band+'MagLim'])
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


class VphasFrameCatalogue(object):
    """Creates a multi-band catalogue for one CCD.

    Parameters
    ----------
    frames : dictionary of (band, `VphasFrame`) pairs

    ccd : int
        Number of the OmegaCam CCD, corresponding to the extension number in
        the 32-CCD multi-extension FITS images produced by the camera.
    """

    def __init__(self, frames, ccd, workdir, cpufarm=None):
        self.frames = frames
        self.fieldname = self.frames['i'].name.split('-')[0]
        self.ccd = ccd
        self.workdir = workdir
        self.name = '{0}-{1}'.format(self.fieldname, self.ccd)
        # Allow "self.cpufarm.imap(f, param)" to be used for parallel processing
        if cpufarm is None:
            self.cpufarm = itertools  # Simple sequential processing
        else:
            self.cpufarm = cpufarm

    def __del__(self):
        """Destructor; cleans up the temporary directory."""
        #shutil.rmtree(self.workdir)
        # Make sure to get rid of any multiprocessing-forked processes;
        # they might be eating up a lot of memory!
        #if type(self.cpufarm) is multiprocessing.pool.Pool:
        #    self.cpufarm.terminate()
        pass

    def create_master_source_table(self):
        """Returns an astropy Table."""
        log.info('Creating the master source list')
        source_tables = {}
        for tbl in self.cpufarm.imap(compute_source_table_task, self.frames.values()):
            source_tables[tbl.meta['band']] = tbl
        # Now merge the single-band lists into a master source table
        master_table = source_tables['i']
        for band in self.frames.keys():
            if band == 'i':
                continue  # i is the master
            current_coordinates = SkyCoord(master_table['ra']*u.deg, master_table['dec']*u.deg)
            new_coordinates = SkyCoord(source_tables[band]['ra']*u.deg, source_tables[band]['dec']*u.deg)
            idx, sep2d, dist3d = new_coordinates.match_to_catalog_sky(current_coordinates)
            mask_extra = sep2d > 2*u.arcsec
            log.info('Found {0} extra sources in {1}'.format(mask_extra.sum(), band))
            master_table = table.vstack([master_table, source_tables[band][mask_extra]],
                                        metadata_conflicts='silent')
        return master_table

    @timed
    def create_catalogue(self):
        """Main function to compute the catalogue, which will take a few minutes.

        Returns
        -------
        catalogue : `astropy.table.Table` object
            Table containing the band-merged catalogue.
        """
        log.info('{0}: started creating a catalogue for ccd {1}'.format(self.fieldname, self.ccd))
        source_table = self.create_master_source_table()
        source_table.write(os.path.join(self.workdir, self.name+'-sourcelist.fits'))

        jobs = []
        for band in self.frames:
            params = {'image': self.frames[band],
                      'ra': source_table['ra'],
                      'dec': source_table['dec'],
                      'workdir': self.workdir}
            jobs.append(params)

        tables = [tbl for tbl in self.cpufarm.imap(compute_photometry_task, jobs)]
        # Band-merge the tables
        merged = table.hstack(tables, metadata_conflicts='silent')
        merged['field'] = self.fieldname
        merged['ccd'] = self.ccd
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
        output_filename =  os.path.join(self.workdir, self.name+'-catalogue.fits')
        merged.write(output_filename, format='fits')
        # Now make diagnostic plots
        self.plot_psf()
        self.plot_images()
        return merged

    @timed
    def plot_psf(self, output_fn=None):
        """Saves a pretty plot showing the PSF in each band."""
        if output_fn is None:
            output_fn = self.name+'-psf.jpg'
        fig = pl.figure()
        cols = 3
        rows = 2
        bandorder = ['u', 'g', 'r2', 'ha', 'r', 'i']
        for idx, band in enumerate(bandorder):
            if band in self.frames:
                hdu = fits.open(os.path.join(self.workdir, self.frames[band].name+'-psf.fits'))[0]
                psf = hdu.data[5:-5, 5:-5]
                ax = fig.add_subplot(rows, cols, idx+1)
                vmin, vmax = 1., hdu.header['PSFHEIGH']
                with np.errstate(divide='ignore', invalid='ignore'):
                    ax.imshow(np.log10(psf),
                              vmin=np.log10(vmin), vmax=np.log10(vmax),
                              cmap=pl.cm.gist_heat, origin='lower',
                              interpolation='nearest')
                ax.set_title(band)
        pl.suptitle('DAOPHOT PSF: '+self.name)
        fig.tight_layout()
        log.debug('Writing {0}'.format(output_fn))
        plot_filename = os.path.join(self.workdir, output_fn)
        fig.savefig(plot_filename, dpi=120)
        pl.close(fig)

    @timed
    def plot_images(self, sampling=3):
        """Plots the origin, sky-, and psf-subtracted images as JPGs in the workdir.

        Parameters
        ----------
        sampling : int (optional)
           Only sample every Nth pixel when plotting the images. (default: 2)
        """
        # Save the images as quicklook jpg
        fig = pl.figure(figsize=(16, 9))
        cols = 6
        rows = 2
        bandorder = ['u', 'g', 'r2', 'ha', 'r', 'i']
        for idx, band in enumerate(bandorder):
            if band in self.frames:  # not all bands may be available
                logvmin, logvmax = np.log10(np.percentile(self.frames[band].hdu.data, [2, 99]))
                imgstyle = {'cmap': pl.cm.gist_heat, 'origin': 'lower', 'vmin': logvmin, 'vmax': logvmax}
                #vmax = vmin + 50

                with np.errstate(divide='ignore', invalid='ignore'):
                    image_data = np.log10(self.frames[band].hdu.data[::sampling, ::sampling])
                    subimg = fits.open(os.path.join(self.workdir, self.frames[band].name+'-sub.fits'))
                    subtracted_data = np.log10(subimg[0].data[::sampling, ::sampling])

                    ax = fig.add_subplot(rows, cols, idx+1)
                    ax.matshow(image_data, **imgstyle)
                    ax.set_title(self.frames[band].name)
                    ax.axis('off')
                    ax = fig.add_subplot(rows, cols, idx+1+cols)
                    ax.matshow(subtracted_data, **imgstyle)
                    ax.axis('off')
                    # Make standalone jpgs while at it
                    # Original image
                    imsave_filename = os.path.join(self.workdir, self.frames[band].name+'-ccd.jpg')
                    orig_data = self.frames[band].orig_hdu.data[::sampling, ::sampling]
                    orig_vmin, orig_vmax = np.percentile(orig_data, [2, 99])
                    mimg.imsave(imsave_filename, np.log10(orig_data),
                                vmin=np.log10(orig_vmin), vmax=np.log10(orig_vmax),
                                cmap=pl.cm.gist_heat, origin='lower')
                    # Background image
                    background_fn = os.path.join(self.workdir, self.frames[band].name+'-ccd-bg.jpg')
                    bg_data = self.frames[band].background_hdu.data[::sampling, ::sampling]
                    mimg.imsave(background_fn, np.log10(bg_data),
                                vmin=np.log10(orig_vmin), vmax=np.log10(orig_vmax),
                                cmap=pl.cm.gist_heat, origin='lower')                 
                    # Sky-subtracted image
                    imsave_filename = os.path.join(self.workdir,
                                                   self.frames[band].name+'-ccd-nosky.jpg')
                    mimg.imsave(imsave_filename, image_data, **imgstyle)
                    # PSF-subtracted image
                    sub_imsave_filename = os.path.join(self.workdir, self.frames[band].name+'-ccd-nostars.jpg')
                    mimg.imsave(sub_imsave_filename, subtracted_data, **imgstyle)      
        fig.tight_layout()
        plot_filename = os.path.join(self.workdir, self.fieldname+'-sub.jpg')
        fig.savefig(plot_filename, dpi=120)
        pl.close(fig)


class VphasOffsetPointing(object):
    """A pointing is a single (ra,dec) position in the sky.

    Parameters
    ----------
    pointing_name : str
        5-character wide identifier, composed of the 4-character wide VPHAS
        field number, followed by 'a' (first offset), 'b' (second offset),
        or 'c' (third offset used in the g and H-alpha bands only.)
    """
    def __init__(self, pointing_name, use_multiprocessing=True, **kwargs):
        if len(pointing_name) != 5 or not pointing_name.endswith(('a', 'b', 'c')):
            raise ValueError('Illegal pointing name. Expected a string of the form "0001a".')
        self.pointing_name = pointing_name
        self.kwargs = kwargs
        # Allow "self.cpufarm.imap(f, param)" to be used for parallel processing
        if use_multiprocessing:
            self.cpufarm = multiprocessing.Pool()
        else:
            self.cpufarm = itertools  # Simple sequential processing

    def __del__(self):
        """Destructor."""
        #shutil.rmtree(self.workdir)
        # Make sure to get rid of any multiprocessing-forked processes;
        # they might be eating up a lot of memory!
        if type(self.cpufarm) is multiprocessing.pool.Pool:
            self.cpufarm.terminate()

    @timed
    def create_catalogue(self, ccdlist=range(1, 33), include_ugr=True, workdir=WORKDIR_DEFAULT):
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
        # We do not tolerate red data missing
        try:
            image_filenames = self.get_red_filenames()
        except NotObservedException as e:
            log.error(e.message)
            return
        if include_ugr:
            try:
                image_filenames.update(self.get_blue_filenames())
            except NotObservedException as e:
                log.warning(e.message)  # We can tolerate the blue data missing
        log.debug('{0}: using the following files: {1}'.format(self.pointing_name, image_filenames))
        # Having obtained the filenames, start computing!
        framecats = []
        for ccd in ccdlist:
            # Setup the working directory to store temporary files
            ccd_workdir = tempfile.mkdtemp(prefix='{0}-{1}-'.format(self.pointing_name, ccd), dir=workdir)
            log.info('{0}-{1}: started preparing the data'.format(self.pointing_name, ccd))
            log.info('{0}-{1}: working directory: {2}'.format(self.pointing_name, ccd, ccd_workdir))

            jobs = []
            for fn in image_filenames.values():
                params = {'filename': fn,
                          'extension': ccd,
                          'workdir': ccd_workdir,
                          'kwargs': self.kwargs}
                jobs.append(params)
            frames = {}
            for frame in self.cpufarm.imap(create_vphasframe_task, jobs):
                frames[frame.band] = frame
            vfc = VphasFrameCatalogue(frames, ccd=ccd, workdir=ccd_workdir, cpufarm=self.cpufarm)
            framecats.append(vfc.create_catalogue())

            import gc
            log.debug('gc.collect freed {0} bytes'.format(gc.collect()))

        catalogue = table.vstack(framecats, metadata_conflicts='silent')
        return catalogue

    def get_red_filenames(self):
        """Returns the H-alpha, r- and i-band FITS filenames of the red concat.

        Parameters
        ----------
        pointing : str
            Identifier of the VPHAS field; must be a 5-character wide string
            composed of a 4-digit zero padded number followed by 'a', 'b', 
            or 'c' to denote the offset, e.g. '0149a' is the first offset
            of field 'vphas_0149'.

        Returns
        -------
        filenames : list of 3 strings
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='(.*)did not parse as fits unit(.*)')
            metadata = Table.read(os.path.join(SURVEYTOOLS_DATA, 'list-hari-image-files.fits'))
        fieldname = 'vphas_' + self.pointing_name[:-1]
        # Has the field been observed?
        if (metadata['Field_1'] == fieldname).sum() == 0:
            raise NotObservedException('{0} has not been observed in the red filters'.format(self.fieldname))
        offset2idx = {'a': 0, 'b': -1, 'c': 1}
        offset = offset2idx[self.pointing_name[-1:]]
        # Define the colloquial band names used in the catalogue
        filter2band = {'NB_659': 'ha', 'r_SDSS': 'r', 'i_SDSS': 'i'}
        result = {}
        for filtername, bandname in filter2band.iteritems():
            mask = ((metadata['Field_1'] == fieldname)
                    & (metadata['filter'] == filtername))
            filenames = metadata['image file'][mask]
            if filtername == 'NB_659':
                assert len(filenames) == 3  # sanity check
            else:
                assert len(filenames) == 2  # sanity check
            filenames.sort()
            result[bandname] = filenames[offset]
        return result

    def get_blue_filenames(self):
        """Returns the u-, g- and r-band FITS filenames of the blue concat.

        Parameters
        ----------
        pointing : str
            Identifier of the VPHAS field; must be a 5-character wide string
            composed of a 4-digit zero padded number followed by 'a', 'b', 
            or 'c' to denote the offset, e.g. '0149a' is the first offset
            of field 'vphas_0149'.

        Returns
        -------
        filenames : list of 3 strings
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='(.*)did not parse as fits unit(.*)')
            metadata = Table.read(os.path.join(SURVEYTOOLS_DATA, 'list-ugr-image-files.fits'))
        fieldname = 'vphas_' + self.pointing_name[:-1]
        # Has the field been observed?
        if (metadata['Field_1'] == fieldname).sum() == 0:
            raise NotObservedException('{0} has not been observed in the blue filters'.format(self.fieldname))
        offset2idx = {'a': 0, 'b': -1, 'c': 1}
        offset = offset2idx[self.pointing_name[-1:]]
        # Define the colloquial band names used in the catalogue
        filter2band = {'u_SDSS': 'u', 'g_SDSS': 'g', 'r_SDSS': 'r2'}
        result = {}
        for filtername, bandname in filter2band.iteritems():
            mask = ((metadata['Field_1'] == fieldname)
                    & (metadata['filter'] == filtername))
            filenames = metadata['image file'][mask]
            if filtername != 'g_SDSS':
                assert len(filenames) == 2  # sanity check
            filenames.sort()
            result[bandname] = filenames[offset]
        return result


############
# FUNCTIONS
############

# Define function for parallel processing
def create_vphasframe_task(params):
    """Returns a `VphasFrame` instance for a given FITS filename/extension.

    This is defined as a separate function to allow pickling for multiprocessing.
    """
    log.debug('Creating a VphasFrame instance for {0}[{1}]'.format(params['filename'], params['extension']))
    frame = VphasFrame(params['filename'], params['extension'], workdir=params['workdir'], **params['kwargs'])
    frame.populate_cache()
    return frame

def compute_source_table_task(image):
    # 4 sigma is recommended by the DAOPHOT manual, but 3-sigma
    # does appear to recover a bunch more genuine sources at SNR > 5.
    thresholds = {'u':5, 'g':5, 'r2':5, 'ha':5, 'r':5, 'i':3}
    # the psfrad and maxiter parameters were carefully chosen to speed
    # up source detection; psfrad_fwhm should be bigger for good photometry
    tbl = image.compute_source_table(psfrad_fwhm=3., maxiter=20,
                                     threshold=thresholds[image.band])
    del image
    return tbl

def compute_photometry_task(params):
    tbl = params['image'].compute_photometry(params['ra'], params['dec'],
                                             psfrad_fwhm=10., maxiter=10,
                                             threshold=5, mergerad_fwhm=0)
    # Save the psf and subtracted images
    dp = params['image']._cache['daophot']
    psf_filename = os.path.join(params['workdir'], params['image'].name+'-psf.fits')
    dp.get_psf().writeto(psf_filename)
    sub_filename = os.path.join(params['workdir'], params['image'].name+'-sub.fits')
    dp.get_subimage().writeto(sub_filename)
    return (tbl, dp)

