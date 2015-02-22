"""IRAF/DAOPHOT wrapper to carry out PSF photometry in a user-friendly way.

Some of the help texts are adapted from the DaoPhot or IRAF manual or
http://iraf.net/irafhelp.php?val=daopars&help=Help+Page
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import re
import shutil
import warnings
import tempfile
import numpy as np

from astropy import log
from astropy import table
from astropy.table import Table
from astropy.io import fits
from astropy import wcs

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
        self.workdir = tempfile.mkdtemp(prefix='daophot-', dir=workdir)
        self._path_cache = {}
        self._path_cache['image_path'] = image_path
        self._setup_iraf(**kwargs)

    def __del__(self):
        """Destructor; cleans up the temporary directory."""
        pass  # shutil.rmtree(self.workdir)

    def _setup_iraf(self, datamin=0, datamax=60000, epadu=1., fitrad_fwhm=1.,
                    fitsky='yes', function='moffat25', fwhmpsf=3., itime=10.,
                    maxiter=50, maxnpsf=60, mergerad_fwhm=2., nclean=10,
                    psfrad_fwhm=10., ratio=1., readnoi=0, recenter='yes',
                    roundlo=-1.0, roundhi=1.0, sannulus_fwhm=2., 
                    saturated='no', sharplo=0.2, sharphi=1.0,
                    sigma=5., theta=0., threshold=3., varorder=1, zmag=20.,
                    wsannulus_fwhm=2.):
        """Sets the IRAF/DAOPHOT configuration parameters.

        Parameters
        ----------
        fitrad_fwhm : float
            The PSF fitting radius in units `fwhmpsf`. Only pixels within the
            fitting radius of the center of a star will contribute to the
            fits computed by the `allstar` task. For most images the fitting
            radius should be approximately equal to the FWHM of the PSF
            (i.e. `fitrad_fwhm = 1`). Under severely crowded conditions a
            somewhat smaller value may be used in order to improve the fit.
            If the PSF is variable, the FWHM is very small, or sky fitting
            is enabled on the other hand, it may be necessary to increase the
            fitting radius to achieve a good fit. (default: 1.)

        fitsky : str (optional)
            Compute new sky values for the stars in the input list (allstar)?
            If fitsky = "no", the `allstar` task compute a group sky value by
            averaging the sky values of the stars in the group. 
            If fitsky = "yes", the `allstar` task computes new sky values for
            each star every third iteration by subtracting off the best
            current fit for the star and and estimating the median of the
            pixels in the annulus defined by sannulus and wsannulus.
            The new group sky value is the average of the new individual
            values. (default: 'yes')

        function : str (optional)
            PSF model function. One of "auto", "gauss", "moffat15", "moffat25",
            "lorentz", "penny1", or "penny2". (default: 'moffat25')

        maxiter : int (optional)
            The maximum number of times that the `allstar` task will iterate
            on the PSF fit before giving up. (default: 50)

        maxnpsf : int (optional)
            The maximum number of candidate psf stars to be selected. (default: 60)

        mergerad_fwhm : float (optional)
            The critical separation in units `psffwhm` between two objects
            for an object merger to be considered by the `allstar` task.
            Objects with separations > mergerad will not be merged; faint
            objects with separations <= mergerad will be considered for merging.
            The default value of mergerad is sqrt (2 *(PAR1**2 + PAR2**2)),
            where PAR1 and PAR2 are the half-width at half-maximum along the
            major and minor axes of the psf model. Merging can be turned off
            altogether by setting mergerad to 0.0. (default: 2.0)

        nclean: int
            The number of additional iterations the `psf` task performs to
            compute the PSF look-up tables. If nclean is > 0, stars which
            contribute deviant residuals to the PSF look-up tables in the
            first iteration, will be down-weighted in succeeding iterations. 
            The DAOPHOT manual recommends 5 passes. (default: 10)

        psfrad_fwhm : float
            The radius of the circle in units `fwhmpsf` within which the PSF
            model is defined (i.e. the PSF model radius in pixels is obtained
            by multiplying `psfrad_fwhm` with `fwhmpsf`). 
            Psfrad_fwhm should be slightly larger than the radius at
            which the intensity of the brightest star of interest fades into
            the noise. Large values are computationally expensive, however.
            Must be larger than `fitrad_fwhm` in any case. (default: 10.)

        ratio : float (optional)
            Ratio of minor to major axis of Gaussian kernel for object
            detection (default: 1.0)

        recenter : str (optional)
            One of 'yes' or 'no'.

        roundlo : float (optional)
            Lower bound on roundness for object detection (default: -1.0)

        roundhi : float (optional)
            Upper bound on roundness for object detection (default: 1.0)

        sannulus_fwhm : float (optional)
            The inner radius of the sky annulus in units `fwhmpsf` used 
            by `allstar` to recompute the sky values. (default: 2.)

        saturated : str (optional)
            Use saturated stars to improve the signal-to-noise in the wings
            of the PSF model computed by the PSF task? This parameter should
            only be set to "yes" where there are too few high signal-to-noise
            unsaturated stars in the image to compute a reasonable model
            for the stellar profile wings. (default: "no")

        sharplo : float (optional)
            Lower bound on sharpness for object detection (default: 0.2)

        sharphi : float (optional)
            Upper bound on sharpness for object detection (default: 1.0)

        theta : float (optional)
            Position angle of major axis of Gaussian kernel for object
            detection (default: 0.0)

        threshold: float (optional)
            Threshold in sigma for object detection (default: 3.0)

        varorder : int (optional)
            Variation of psf model: 0=constant, 1=linear, 2=cubic
            varorder = -1 (analytic) gives very poor results
            (though it may be more robust in crowded fields, in principle).
            varorder 1 or 2 is possible marginally better than 0,
            though it is not obvious in VPHAS data.

        wsannulus_fwhm : float (optional)
            The width of the sky annulus in units `fwhmpsf` used by `allstar`
            to recompute the sky values. (default: 2.)
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
        iraf.daopars.sannulus = sannulus_fwhm * fwhmpsf
        # => Width of sky fitting annulus in scale units, default: 11.0
        iraf.daopars.wsannulus = wsannulus_fwhm * fwhmpsf
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
        iraf.daopars.fitsky = fitsky
        iraf.daopars.saturated = saturated
        iraf.daopars.maxiter = maxiter
        iraf.daopars.function = function
        # Object detection (daofind) parameters
        iraf.findpars.threshold = threshold
        iraf.findpars.sharplo = sharplo
        iraf.findpars.sharphi = sharphi
        iraf.findpars.roundlo = roundlo
        iraf.findpars.roundhi = roundhi
        # PSF fitting star selection
        iraf.pstselect.maxnpsf = maxnpsf

    def do_psf_photometry(self):
        """Runs the daofind, apphot, pstselect, psf, and allstar tasks.

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
        self.pstselect()
        self.psf()
        self.allstar()
        return self.get_allstar_phot_table()

    def daofind(self, output_fn='output-daofind.txt'):
        """DAOFIND searches the image for local density maxima, with a
        peak amplitude greater than `threshold` * `sky_sigma` above the
        local background.

        Parameters
        ----------
        output_fn : str (optional)
            Where to write the output text file? (defaults to a temporary file)
        """
        self._path_cache['daofind_output'] = os.path.join(self.workdir,
                                                          output_fn)
        daofind_args = dict(image=self._path_cache['image_path'],
                            output=self._path_cache['daofind_output'],
                            verify='no',
                            verbose='no',
                            starmap=os.path.join(self.workdir, 'starmap.'),
                            skymap=os.path.join(self.workdir, 'skymap.'))
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
        self._path_cache['apphot_output'] = os.path.join(self.workdir,
                                                         output_fn)
        phot_args = dict(image=self._path_cache['image_path'],
                         output=self._path_cache['apphot_output'],
                         coords=coords,
                         verify='no',
                         interactive='no',
                         cache='yes',
                         verbose='no')
        iraf.daophot.phot(**phot_args)

    def pstselect(self, prune_outliers=True):
        """Selects suitable stars for PSF model fitting.

        Parameters
        ----------
        prune_outliers : bool (optional)
            If True, remove any selected stars for which the sky estimate
            is an outlier, as determined using sigma-clipping. This is effective
            in ensuring that spurious objects in the wings of bright stars
            are removed. (default: True)
        """
        if 'apphot_output' not in self._path_cache:
            raise DaophotError('You need to run Daophot.apphot '
                               'before Daophot.psf can be used.')
        output_path = os.path.join(self.workdir, 'output-pstselect.txt')
        pstselect_args = dict(mode='h',
                              image=self._path_cache['image_path'],
                              photfile=self._path_cache['apphot_output'],
                              pstfile=output_path,
                              verify='no',
                              interactive='no',
                              verbose='yes',
                              Stdout=str(os.path.join(self.workdir,
                                                      'log-pstselect.txt')))
        iraf.daophot.pstselect(**pstselect_args)
        # Prune the selected stars using sigma-clipping on the sky count
        if prune_outliers:
            output_path_pruned = os.path.join(self.workdir,
                                              'output-pstselect-pruned.txt')
            pstselect_prune(output_path, output_path_pruned)
            self._path_cache['pstselect_output'] = output_path_pruned
        else:
            self._path_cache['pstselect_output'] = output_path

    @timed
    def psf(self, failsafe=True, norm_scatter_limit=0.05):
        """Runs the DAOPHOT PSF model fitting task.

        Parameters
        ----------
        failsafe : bool (optional)
            If True and the PSF fitting fails to converge, then re-try the fit
            automatically with fewer stars and varorder = 0. (default: True)

        norm_scatter_limit : float (optional)
        """
        if 'pstselect_output' not in self._path_cache:
            raise DaophotError('You need to run DaoPhot.pstselect '
                               'before Daophot.psf can be used.')
        self._path_cache['psf_output'] = os.path.join(self.workdir,
                                                      'output-psf')  # daophot will append .fits
        self._path_cache['psg_output'] = os.path.join(self.workdir,
                                                      'output-psg.txt')
        self._path_cache['psf_pst_output'] = os.path.join(self.workdir,
                                                          'output-psf-pst.txt')
        path_psf_log = os.path.join(self.workdir, 'log-psf.txt')
        psf_args = dict(image=self._path_cache['image_path'],
                        photfile=self._path_cache['apphot_output'],
                        pstfile=self._path_cache['pstselect_output'],
                        psfimage=self._path_cache['psf_output'],
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
        success, norm_scatter = psf_success(path_psf_log, norm_scatter_limit)
        if not success:
            if not failsafe:
                log.error('daophot.psf failed to fit the PSF')
                return norm_scatter
            # psf output does not exist if the model failed to converge
            # Fitting the PSF model usually fails because one or more spurious
            # objects have been selected.  Reducing the number of objects,
            # and simplifying the model, often delivers an acceptable model.
            orig_maxnpsf = iraf.pstselect.getParDict()['maxnpsf'].value  # calling `iraf.pstselect.maxnpsf` doesnt work for some strange reason
            tmp_varorder = iraf.daopars.varorder    
            iraf.daopars.varorder = 0
            attempts = 5
            for attempt_no, maxnpsf in enumerate(
                np.linspace(orig_maxnpsf / 2, 3, attempts, dtype=int)):
                # It's important to remove the PSF output file before repeating
                # the fit, otherwise daophot will add a 2nd extension to it.
                try:
                    os.remove(self._path_cache['psf_output'] + '.fits')
                except OSError:
                    pass  
                iraf.pstselect.maxnpsf = maxnpsf
                self.pstselect(prune_outliers=True)
                iraf.daophot.psf(**psf_args)
                # Do not set a limit on the last attempt
                if attempt_no == (attempts - 1):
                    limit = None
                else:
                    limit = norm_scatter_limit
                success, norm_scatter = psf_success(path_psf_log,
                                                    norm_scatter_limit=limit)
                log.warning('daophot.psf: norm_scatter on attempt '
                            '#{0} = {1:.3f} (maxnpsf={2})'.format(attempt_no+2,
                                                                  norm_scatter,
                                                                  maxnpsf))
                if success:
                    break
            # Restore the original config
            iraf.daopars.varorder = tmp_varorder
            iraf.pstselect.maxnpsf = orig_maxnpsf
            if not success:
                raise DaophotError('daophot.psf failed on failsafe attempt')
        
        # Save the resulting PSF into a user-friendly FITS file
        self._path_cache['seepsf_output'] = os.path.join(self.workdir, 'output-seepsf')  # daophot will append .fits
        seepsf_args = dict(psfimage=self._path_cache['psf_output'],
                           image=self._path_cache['seepsf_output'])
        iraf.daophot.seepsf(**seepsf_args)

        return norm_scatter    

    @timed
    def allstar(self):
        """Run the DAOPHOT allstar task, which extracts PSF photometry."""
        if 'psf_output' not in self._path_cache:
            raise DaophotError('You need to run Daophot.psf '
                               'before Daophot.allstar can be used.')
        self._path_cache['allstar_output'] = os.path.join(self.workdir,
                                                          'output-allstar.txt')
        self._path_cache['subimage_output'] = os.path.join(self.workdir,
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
            self.catalogue_daofind = os.path.join(self.workdir,
                                                  'output-daofind.fits')
            self.get_daofind_table().write(self.catalogue_daofind,
                                           format='fits', overwrite=True)

        self.catalogue_phot = os.path.join(self.workdir, 'output-phot.fits')
        self.get_phot_table().write(self.catalogue_phot,
                                    format='fits', overwrite=True)

        self.catalogue_allstar = os.path.join(self.workdir,
                                              'output-allstar.fits')
        self.get_allstar_table().write(self.catalogue_allstar,
                                       format='fits', overwrite=True)

    def pix2world(self, x, y, origin=1):
        """Shorthand to convert pixel(x,y) into equatorial(ra,dec) coordinates.

        Use origin=1 if x/y positions were produced by IRAF/DAOPHOT,
        0 if they were produced by astropy."""
        img = self._path_cache['image_path'].rpartition('[')[0]
        return wcs.WCS(fits.getheader(img)).wcs_pix2world(x, y, origin)

    def get_daofind_table(self):
        tbl = Table.read(self._path_cache['daofind_output'], format='daophot')
        # Convert pixel coordinates into ra/dec
        ra, dec = self.pix2world(tbl['XCENTER'], tbl['YCENTER'], origin=1)
        ra_col = table.Column(name='ra', data=ra)
        dec_col = table.Column(name='dec', data=dec)
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


###################
# HELPER FUNCTIONS
###################

def pstselect_prune(pstselect_output_path, new_path):
    """Removes likely spurious objects from the output of the `pstselect` task.
    
    This routine will deem an object to be spurious if its sky level is an
    outlier compared to the other objects, which is an effective way to remove
    spurious detections in the wings of saturated stars (where sky levels
    are abberant).  We do not need to prune stars with nearby neighbours or
    invalid pixel values, because `pstselect` should have pruned those itself.

    Parameters
    ----------
    pstselect_output_path : str
        Path to the output file produced by the `pstselect` task,
        which is a file containing the stars to be used for PSF fitting.

    new_path : str
        Location to write a new file in the same format as the `pstselect`
        output file, but with likely spurious objects removed.
    """
    tbl = Table.read(pstselect_output_path, format='daophot')
    # We prune objects using sigma-clipping on the sky estimate.
    # We will try increasing values of sigma, until less than half of the stars
    # are rejected.
    from astropy.stats import sigma_clip
    for sigma in [2., 3., 4., 5., 10.]:
        bad_mask = sigma_clip(tbl['MSKY'].data, sig=sigma, iters=None).mask
        if bad_mask.sum() < 0.5*len(tbl):  # stop if <50% rejected
            break
    log.info('Rejected {0} stars for PSF fitting ({1})'.format(
                 bad_mask.sum(), pstselect_output_path))
    # Now write the new file without the pruned objects to disk
    fh = open(new_path, 'w')
    fh.write("#N ID    XCENTER   YCENTER   MAG         MSKY\n"
             "#U ##    pixels    pixels    magnitudes  counts\n"
             "#F %-9d  %-10.3f   %-10.3f   %-12.3f     %-15.7g\n")
    for row in tbl[~bad_mask]:
        fh.write('{0:<9d}{1:<10.3f}{2:<10.3f}{3:<12.3f}{4:<15.7g}\n'.format(
                  row['ID'], row['XCENTER'], row['YCENTER'],
                  row['MAG'], row['MSKY']))
    fh.close()


def psf_success(path_psf_log, norm_scatter_limit=0.1):
    """Returns True if the daophot.psf log indicates a good PSF fit.

    Parameters
    ----------
    path_psf_log : str
        Path to the log file produced by DaoPhot's psf-fitting task.

    norm_scatter_limit : float (optional)
        Maximum tolerated 'norm scatter' fit score of the PSF model.

    Returns
    -------
    (success, norm_scatter) : (bool, float)
        Success is True if the fitting succeeded, norm_scatter contains
        DaoPhot's PSF model fitting score.
    """
    logfile = open(path_psf_log, 'r')
    logtxt = logfile.read()
    logfile.close()
    success = True
    if len(re.findall("failed to converge", logtxt)) > 0:
        log.warning('daophot.psf: failed to converge '
                    '({0})'.format(path_psf_log))
        success = False
        norm_scatter_best = 999.
    else:
        # Check for a very poor fit score
        norm_scatter = re.findall("norm scatter[ :=]+([\d\.]+)", logtxt)
        norm_scatter.sort()
        norm_scatter_best = float(norm_scatter[0])
        log.info('daophot.psf: norm scatter = '
                 '{0} ({1})'.format(norm_scatter_best,
                                    path_psf_log))
        if (norm_scatter_limit is not None
        and norm_scatter_best > norm_scatter_limit):
            log.warning('daophot.psf: norm scatter exceeds limit '
                        '({0:.2f} > {1}) ({2})'.format(norm_scatter_best,
                                                       norm_scatter_limit,
                                                       path_psf_log))
            success = False
    return (success, norm_scatter_best)
