"""IRAF/DAOPHOT wrapper to carry out PSF photometry in a user-friendly way."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import re
import shutil
import warnings
import tempfile

from astropy import log
from astropy import table
from astropy.table import Table
from astropy.io import fits

# Importing pyraf -- note that this requires IRAF to be installed
# and the "iraf" and "IRAFARCH" environment variables to be set
with warnings.catch_warnings():
    # Ignore the ImportWarning we often get when importing pyraf, because pyraf
    # likes to create working dirs with its name in the local dir.
    warnings.filterwarnings("ignore", message="(.*)Not importing dir(.*)")
    # Disable pyraf graphics:
    from stsci.tools import capable
    capable.OF_GRAPHICS = False
    from pyraf import iraf
    from iraf import noao, digiphot, daophot
    # Turn cache off to enable multiprocessing to be used;
    # cf. http://www.stsci.edu/institute/software_hardware/pyraf/pyraf_faq#3.5
    iraf.prcacheOff()
    iraf.set(writepars=0)

from .utils import timed


class DaophotError(Exception):
    """Raised if Daophot failed to perform an operation."""
    pass


class Daophot(object):
    """DAOPHOT wrapper class.

    The constructor of this class expects to receive
    ALL the non-default configuration parameters because IRAF is very stateful.

    The speed of psf and allstar tasks are mostly affected by the psfrad
    and psfnmax parameters.
    """
    def __init__(self, image_path, workdir='/tmp', **kwargs):
        self._workdir = tempfile.mkdtemp(prefix='daophot-', dir=workdir)
        self._path_cache = {}
        self._path_cache['image_path'] = image_path
        self._setup_iraf(**kwargs)

    def __del__(self):
        """Destructor; cleans up the temporary directory."""
        pass  # shutil.rmtree(self._workdir)

    def _setup_iraf(self, datamin=0, datamax=60000, epadu=1., fitrad_fwhm=1.,
                    fitsky='yes', function='moffat25', fwhmpsf=3., itime=10.,
                    maxiter=50, mergerad_fwhm=2., nclean=10,
                    psfrad_fwhm=10., ratio=1., theta=0., readnoi=0, sigma=5.,
                    threshold=3., recenter='yes', varorder=1, zmag=20.):
        """Sets the IRAF/DAOPHOT configuration parameters.

        Parameters
        ----------
        fitrad_fwhm : float
            The PSF fitting radius.  Recommended by the DAOPHOT manual to have
            a value close to the FWHM.  Trials suggests that increasing
            the value does not really help saturated stars to be fit.

        fitsky : str
            Recompute sky during fit? One of 'yes' or 'no'.

        function : str
            PSF model function. One of "auto", "gauss", "moffat15", "moffat25",
            "lorentz", "penny1", or "penny2".

        maxiter : int

        mergerad_fwhm : float (optional)
            Use 0 to disable source merging during PSF fitting.

        nclean: int
            Number of passes used to clean the PSF from bad pixels, neighbours,
            etc. The DAOPHOT manual recommends 5 passes. (default: 10)

        psfrad_fwhm : float
             Radius of psf model. Must be somewhat larger than fitrad.
             The wings beyond fitrad will not determine the fit, but a large
             value is necessary to subtract the wings of bright stars properly.
             A large value comes at a computational cost, however.

        threshold: float
            Daofind object detection threshold.

        recenter : str
            One of 'yes' or 'no'.

        varorder : int
            Variation of psf model: 0=constant, 1=linear, 2=cubic
            varorder = -1 (analytic) gives very poor results
            (though it may be more robust in crowded fields, in principle).
            varorder 1 or 2 is possible marginally better than 0,
            though it is not obvious in VPHAS data.
        """
        # Ensure we start from the iraf defaults
        for module in ['datapars', 'findpars', 'centerpars', 'fitskypars',
                       'photpars', 'daopars', 'daofind', 'phot']:
            iraf.unlearn(module)
        # Avoid the "Out of space in image header" exception
        iraf.set(min_lenuserarea=640000)
        # Set data-dependent IRAF/DAOPHOT configuration parameters
        iraf.datapars.fwhmpsf = fwhmpsf   # [pixels]
        iraf.datapars.sigma = sigma       # sigma(background) [ADU]
        iraf.datapars.datamin = datamin   # min good pix value [ADU]
        iraf.datapars.datamax = datamax   # max good pix value [ADU]
        iraf.datapars.readnoi = readnoi   # [electrons]
        iraf.datapars.epadu = epadu       # [electrons per ADU]
        iraf.datapars.itime = itime       # exposure time [seconds]
        iraf.daofind.ratio = ratio        # 2D Gaussian PSF fit ratio
        iraf.daofind.theta = theta        # 2D Gaussian PSF fit angle
        iraf.photpars.aperture = fwhmpsf  # Aperture radius
        iraf.photpars.zmag = zmag         # Magnitude zero point
        # PSF parameters
        iraf.daopars.psfrad = psfrad_fwhm * fwhmpsf  # PSF radius
        iraf.daopars.fitrad = fitrad_fwhm * fwhmpsf  # PSF fitting radius
        iraf.daopars.mergerad = mergerad_fwhm * fwhmpsf
        # Setting a good sky annulus is important; a large annulus will ignore
        # background changes on small scales, and will cause aperture photom
        # in the wings of bright stars to be overestimated.
        # => Inner radius of sky fitting annulus in scale units, default: 0.0
        iraf.daopars.sannulus = 2 * fwhmpsf
        # => Width of sky fitting annulus in scale units, default: 11.0
        iraf.daopars.wsannulus = 2 * fwhmpsf
        # Allstar will re-measure the background in the PSF-subtracted image
        # using this sky annulus -- change this if you see haloes around stars.
        # The annulus should lie outside the psfrad to get the prettiest
        # subtracted image, but putting the annulus far from the star might
        # cause trouble in case of a spatially variable background!
        iraf.fitskypars.annulus = iraf.daopars.psfrad
        iraf.fitskypars.dannulus = 3 * fwhmpsf
        # Non data-dependent parameters
        iraf.daopars.recenter = recenter
        iraf.daopars.nclean = nclean
        iraf.daopars.varorder = varorder
        iraf.daopars.maxnstar = 5e4
        iraf.daopars.fitsky = 'yes'
        iraf.daopars.saturated = 'yes'  # This appears to improve the wings
        iraf.daopars.maxiter = maxiter
        iraf.daopars.function = function
        iraf.findpars.threshold = threshold

    def psf_photometry(self):
        """Runs the daofind, phot, psf, and allstar tasks.

        This method will carry out source detection, aperture photometry,
        psf fitting, and psf photometry. The results are returned as a table.

        Returns
        -------
        table : astropy.table.Table object
            Table containing the results of the PSF photometry;
            this is a merge of the apphot and allstar output tables.
        """
        self.daofind()
        self.apphot()
        self.psf()
        self.allstar()
        return self.get_allstar_phot_table()

    @timed
    def daofind(self, output_fn='output-daofind.txt'):
        """DAOFIND searches the image for local density maxima, with a
        peak amplitude greater than `threshold` * `sky_sigma` above the
        local background.

        Parameters
        ----------
        output_fn : str (optional)
            Where to write the output text file? (defaults to a temporary file)
        """
        self._path_cache['daofind_output'] = os.path.join(self._workdir,
                                                          output_fn)
        daofind_args = dict(image=self._path_cache['image_path'],
                            output=self._path_cache['daofind_output'],
                            verify='no',
                            verbose='no',
                            starmap=os.path.join(self._workdir, 'starmap.'),
                            skymap=os.path.join(self._workdir, 'skymap.'))
        iraf.daophot.daofind(**daofind_args)

    def apphot(self, output_fn='output-apphot.txt', coords=None):
        """Run the DAOPHOT aperture photometry task.

        Parameters
        ----------
        output_fn : str (optional)
            Where to write the output text file? (defaults to a temporary file)

        coords : str
            Path to an output file of DAOFIND.  By default, it will use the
            output from the most recent call to daofind() on this object.
        """
        if coords is None:
            try:
                coords = self._path_cache['daofind_output']
            except KeyError:
                raise DaophotError('You need to run Daophot.daofind '
                                   'before Daophot.apphot can be used.')
        self._path_cache['apphot_output'] = os.path.join(self._workdir,
                                                         output_fn)
        phot_args = dict(image=self._path_cache['image_path'],
                         output=self._path_cache['apphot_output'],
                         coords=coords,
                         verify='no',
                         interactive='no',
                         cache='yes',
                         verbose='no')
        iraf.daophot.phot(**phot_args)

    @timed
    def psf(self, maxnpsf=50, failsafe=True, norm_scatter_limit=0.1):
        """Runs the DAOPHOT PSF model fitting task.

        Parameters
        ----------
        maxnpsf : int
            The maximum number of candidate psf stars to be selected.

        failsafe : bool
            If true and the PSF fitting fails to converge, then re-try the fit
            automatically with varorder = 0 and function = 'auto'.
        """
        if 'apphot_output' not in self._path_cache:
            raise DaophotError('You need to run Daophot.apphot '
                               'before Daophot.psf can be used.')
        # First select the stars to fit the model against
        self._path_cache['pstselect_output'] = os.path.join(self._workdir,
                                                            'output-pstselect.txt')
        pstselect_args = dict(mode='h',
                              image=self._path_cache['image_path'],
                              photfile=self._path_cache['apphot_output'],
                              pstfile=self._path_cache['pstselect_output'],
                              maxnpsf=maxnpsf,
                              verify='no',
                              interactive='no',
                              verbose='yes',
                              Stdout=str(os.path.join(self._workdir,
                                                      'log-pstselect.txt')))
        iraf.daophot.pstselect(**pstselect_args)
        # Then fit the actual model
        self._path_cache['psf_output'] = os.path.join(self._workdir,
                                                      'output-psf')  # daophot will append .fits
        self._path_cache['psg_output'] = os.path.join(self._workdir,
                                                      'output-psg.txt')
        self._path_cache['psf_pst_output'] = os.path.join(self._workdir,
                                                          'output-psf-pst.txt')
        path_psf_log = os.path.join(self._workdir, 'log-psf.txt')
        psf_args = dict(image=self._path_cache['image_path'],
                        photfile=self._path_cache['apphot_output'],
                        psfimage=self._path_cache['psf_output'],
                        pstfile=self._path_cache['pstselect_output'],
                        groupfile=self._path_cache['psg_output'],
                        opstfile=self._path_cache['psf_pst_output'],
                        verify='no',
                        interactive='no',
                        cache='yes',
                        verbose='yes',
                        Stdout=str(path_psf_log))
        iraf.daophot.psf(**psf_args)

        # It is possible for iraf.daophot.psf() to fail to converge.
        # In this case, we re-try with more easy-to-fit settings.
        success, norm_scatter = self._psf_success(path_psf_log, norm_scatter_limit)
        if not success:
            if not failsafe:
                log.error('')
            # It's important to remove the PSF output file before repeating
            # the fit, otherwise daophot will add a 2nd extension to it.
            try:
                os.remove(self._path_cache['psf_output'] + '.fits')
            except OSError:
                pass  # psf output does not exist if the model failed to converge

            log.warning('daophot.psf: failure to fit a good PSF, '
                        'now trying failsafe mode.')
            tmp_varorder = iraf.daopars.varorder
            #tmp_function = iraf.daopars.function
            tmp_nclean = iraf.daopars.nclean
            iraf.daopars.varorder = 0

            iraf.daopars.psfrad = iraf.daopars.psfrad / 2.
            iraf.fitskypars.annulus = iraf.fitskypars.annulus / 2.
            
            #iraf.daopars.function = 'auto'
            iraf.daopars.nclean = tmp_nclean * 3

            # Run pstselect & psf again
            pstselect_args['maxnpsf'] = maxnpsf * 4
            iraf.daophot.pstselect(**pstselect_args)
            iraf.daophot.psf(**psf_args)
            
            # Restore the original configuration
            iraf.daopars.varorder = tmp_varorder
            #iraf.daopars.function = tmp_function
            iraf.daopars.nclean = tmp_nclean

            success, norm_scatter = self._psf_success(path_psf_log, norm_scatter_limit * 10)
            log.warning('daophot.psf: norm_scatter on second attempt = {0}'.format(norm_scatter))
            if not success:
                raise DaophotError('daophot.psf failed on failsafe attempt')

        # Save the resulting PSF into a user-friendly FITS file
        self._path_cache['seepsf_output'] = os.path.join(self._workdir, 'output-seepsf')  # daophot will append .fits
        seepsf_args = dict(psfimage=self._path_cache['psf_output'],
                           image=self._path_cache['seepsf_output'])
        iraf.daophot.seepsf(**seepsf_args)

        return norm_scatter

    def _psf_success(self, path_psf_log, norm_scatter_limit=0.1):
        """Returns True if the daophot.psf log indicates a good PSF fit."""
        logfile = open(path_psf_log, 'r')
        logtxt = logfile.read()
        logfile.close()
        success = True
        if len(re.findall("failed to converge", logtxt)) > 0:
            log.warning('daophot.psf: logfile indicates failure to converge')
            success = False
            norm_scatter_best = 999.
        else:
            # Check for a very poor fit score
            norm_scatter = re.findall("norm scatter[ :=]+([\d\.]+)", logtxt)
            norm_scatter.sort()
            norm_scatter_best = float(norm_scatter[0])
            log.info('daophot.psf: norm scatter = {0} ({1})'.format(norm_scatter_best, self._path_cache['image_path']))
            if norm_scatter_best > norm_scatter_limit:
                log.warning('daophot.psf: norm scatter exceeds limit '
                            '({0} > {1})'.format(norm_scatter_best, norm_scatter_limit))
                success = False
        return (success, norm_scatter_best)

    @timed
    def allstar(self):
        """Run the DAOPHOT allstar task, which extracts PSF photometry."""
        if 'psf_output' not in self._path_cache:
            raise DaophotError('You need to run Daophot.psf '
                               'before Daophot.allstar can be used.')
        self._path_cache['allstar_output'] = os.path.join(self._workdir,
                                                          'output-allstar.txt')
        self._path_cache['subimage_output'] = os.path.join(self._workdir,
                                                           'output-subimage')  # daophot will append .fits
        allstar_args = dict(image=self._path_cache['image_path'],
                            photfile=self._path_cache['apphot_output'],
                            psfimage=self._path_cache['psf_output'] + '.fits[0]',
                            allstarfile=self._path_cache['allstar_output'],
                            rejfile=None,
                            subimage=self._path_cache['subimage_output'],
                            verify='no',
                            verbose='no',
                            cache='yes',
                            Stderr=1)
        iraf.daophot.allstar(**allstar_args)

    def save_fits_catalogue(self):
        if hasattr(self, 'output_daofind'):
            self.catalogue_daofind = os.path.join(self._workdir,
                                                  'output-daofind.fits')
            self.get_daofind_table().write(self.catalogue_daofind,
                                           format='fits', overwrite=True)

        self.catalogue_phot = os.path.join(self._workdir, 'output-phot.fits')
        self.get_phot_table().write(self.catalogue_phot,
                                    format='fits', overwrite=True)

        self.catalogue_allstar = os.path.join(self._workdir,
                                              'output-allstar.fits')
        self.get_allstar_table().write(self.catalogue_allstar,
                                       format='fits', overwrite=True)

    def get_daofind_table(self):
        tbl = Table.read(self._path_cache['daofind_output'], format='daophot')
        # Convert pixel coordinates into ra/dec
        ra, dec = self.image.pix2world(tbl['XCENTER'], tbl['YCENTER'], origin=1)
        ra_col = Column(name='ra', data=ra)
        dec_col = Column(name='dec', data=dec)
        tbl.add_columns([ra_col, dec_col])
        return tbl

    def get_phot_table(self):
        import numpy as np
        tbl = Table.read(self._path_cache['apphot_output'], format='daophot')
        # Compute a signal-to-noise estimate based on the aperture photometry.
        # Note that this SNR estimate is marginally more optimistic than the
        # 'MERR' value produced by DAOPHOT, likely because the latter also
        # folds the flat-fielding error into the estimate.
        # Below stdev is in ADU, gain in photons/ADU, readnoise in photons.
        variance_per_pix = (tbl['STDEV']**2
                            + (iraf.datapars.readnoi / iraf.datapars.epadu)**2)
        variance_signal = tbl['FLUX'] / iraf.datapars.epadu
        with np.errstate(divide='ignore', invalid='ignore'):
            tbl['SNR'] = tbl['FLUX'] / np.sqrt(variance_signal
                                               + tbl['AREA']*variance_per_pix)
            # Add the 3-sigma detection limit; cf. e.g. http://www.ast.cam.ac.uk/~xmmssc/xid-imaging/dqc/wfc_tech/flux2mag.html
            tbl['LIM3SIG'] = iraf.photpars.zmag - 2.5 * np.log10(3 * np.sqrt(tbl['AREA'] * variance_per_pix) / iraf.datapars.itime)
            # Identical to daophot.phot: tbl['APERMAG'] = iraf.photpars.zmag - 2.5*np.log10(tbl['FLUX'] / iraf.datapars.itime)

        # Circumvent a unit-related exception thrown by astropy:
        tbl['NSKY'].unit = None
        tbl['NSREJ'].unit = None
        return tbl

    def get_allstar_table(self):
        return Table.read(self._path_cache['allstar_output'], format='daophot')

    def get_allstar_phot_table(self):
        return table.join(self.get_allstar_table(),
                          self.get_phot_table(),
                          keys=['ID'],
                          table_names=['ALLSTAR', 'PHOT'],
                          metadata_conflicts='silent')

    def get_psf(self):
        return fits.open(self._path_cache['seepsf_output'] + '.fits')

    def get_subimage(self):
        fitsobj = fits.open(self._path_cache['subimage_output'] + '.fits')
        return fitsobj

    @property
    def seepsf_path(self):
        return self._path_cache['seepsf_output'] + '.fits'

    @property
    def subimage_path(self):
        return self._path_cache['subimage_output'] + '.fits'
