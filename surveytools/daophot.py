"""IRAF/DAOPHOT wrapper to carry out PSF photometry in a user-friendly way."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
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
    # Ignore the ImportWarning we oft get when importing pyraf, which results
    # from the fact that pyraf likes to create working dirs with its name in the local dir.
    warnings.simplefilter("ignore", category=ImportWarning)
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
    """Raised if a requested field has not been observed yet."""
    pass


class Daophot(object):
    """DAOPHOT wrapper class.

    The speed of psf and allstar tasks are mostly affected by the psfrad and psfnmax parameters.
    IRAF is very stateful, and hence the constructor of this class expects
    to receive all the configuration parameters.
    """
    def __init__(self, image_path, workdir='/tmp', **kwargs):
        self._workdir = tempfile.mkdtemp(prefix='daophot-', dir=workdir)
        self._path_cache = {}
        self._path_cache['image_path'] = image_path
        self._setup_iraf(**kwargs)      

    def __del__(self):
        """Destructor; cleans up the temporary directory."""
        pass #shutil.rmtree(self._workdir)

    def _setup_iraf(self, datamin=0, datamax=60000, epadu=1., fitrad_fwhm=1., fitsky='yes', function='moffat25', fwhmpsf=3., itime=10., maxiter=50,
                    maxnpsf=25, mergerad_fwhm=2., nclean=5, psfrad_fwhm=10., ratio=1., theta=0., readnoi=0, sigma=5., threshold=3.,
                    recenter='yes', varorder=1, zmag=20.):
        """Sets the IRAF/DAOPHOT configuration parameters.

        Parameters
        ----------
        fitrad_fwhm : float

        fitsky : str
            Recompute sky during fit? One of 'yes' or 'no'.

        function : str
            PSF model function. One of "auto", "gauss", "moffat15", "moffat25",
            "lorentz", "penny1", or "penny2".

        maxiter : int

        maxnpsf : int
            The maximum number of candidate psf stars to be selected.

        mergerad_fwhm : float (optional)
            Use 0 to disable source merging during PSF fitting.

        nclean: int
            No. of passes used to clean the PSF from bad pixels, neighbours, etc.
            In the DAOPHOT manual, Stetson recommends 5 passes.

        psfrad_fwhm : float
             Radius of psf model. Must be somewhat larger than fitrad. The wings
             beyond fitrad will not determine the fit, but a large value is
             necessary to subtract the wings of bright stars properly if desired.
             A large value comes at a computational cost, however.

        threshold: float
            Daofind object detection threshold.

        recenter : str
            One of 'yes' or 'no'.

        varorder : int
            Variation of psf model: 0=constant, 1=linear, 2=cubic
            varorder = -1 (analytic) gives very poor results (thought it may be more robust in crowded fields, in principle)
            varorder 1 or 2 is possible marginally better than 0, though it is not obvious in VPHAS data.
        """
        # Ensure we start from the iraf defaults
        for module in ['datapars', 'findpars', 'centerpars', 'fitskypars',
                       'photpars', 'daopars', 'daofind', 'phot']:
            iraf.unlearn(module)
        # Avoid the "Out of space in image header" exception
        iraf.set(min_lenuserarea=640000)
        # Set data-dependent IRAF/DAOPHOT configuration parameters
        iraf.datapars.fwhmpsf = fwhmpsf  # [pixels]
        iraf.datapars.sigma = sigma   # sigma(background) [ADU]
        iraf.datapars.datamin = datamin   # min good pix value [ADU]
        iraf.datapars.datamax = datamax   # max good pix value [ADU]
        iraf.datapars.readnoi = readnoi # [electrons]
        iraf.datapars.epadu = epadu        # [electrons per ADU]
        iraf.datapars.itime = itime
        iraf.daofind.ratio = ratio    # 2D Gaussian PSF fit ratio
        iraf.daofind.theta = theta    # 2D Gaussian PSF fit angle
        iraf.photpars.aperture = fwhmpsf # Aperture radius
        iraf.photpars.zmag = zmag    # Magnitude zero point
        # PSF fitting radius; recommended by DAOPHOT manual to be close to the FWHM
        # experimentation suggests that increasing this value does not allow
        # saturated stars to be fit
        iraf.daopars.fitrad = fitrad_fwhm * fwhmpsf
        iraf.daopars.psfrad = psfrad_fwhm * fwhmpsf
        iraf.daopars.mergerad = mergerad_fwhm * fwhmpsf
        # Setting a good sky annulus is important; a large annulus will ignore
        # background changes on small scales, and will cause aperture photom
        # in the wings of bright stars to be overestimated
        iraf.daopars.sannulus = 2 * fwhmpsf             # Inner radius of sky fitting annulus in scale units, default: 0.0
        iraf.daopars.wsannulus = 2 * fwhmpsf              # Width of sky fitting annulus in scale units, default: 11.0
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
        iraf.pstselect.maxnpsf = maxnpsf
        iraf.findpars.threshold = threshold

    def psf_photometry(self):
        """Runs the daofind, phot, psf, and allstar tasks.

        This method will carry out source detection, aperture photometry,
        psf fitting, and psf photometry. The results are returned as a table.

        Returns
        -------
        table : astropy.table.Table object
            Merged table containing the results of the apphot and allstar tasks.
        """
        self.daofind()
        self.apphot()
        self.psf()
        self.allstar()
        return self.get_allstar_phot_table()

    @timed
    def daofind(self, output_filename='output-daofind.txt'):
        """DAOFIND searches the image for local density maxima, with a 
        peak amplitude greater than `threshold` * `sky_sigma` above the
        local background.

        Returns
        -------
        A table of detected objects.
        """
        self._path_cache['daofind_output'] = os.path.join(self._workdir, output_filename)
        daofind_args = dict(image = self._path_cache['image_path'],
                            output = self._path_cache['daofind_output'],
                            verify = 'no',
                            verbose = 'no',
                            starmap= os.path.join(self._workdir, 'starmap.'),
                            skymap = os.path.join(self._workdir, 'skymap.'))
                            #Stdout = os.path.join(self._workdir,
                            #                     'log-daofind.txt'))
        iraf.daophot.daofind(**daofind_args)

    def apphot(self, output_filename='output-apphot.txt',
                            coords=None):
        if coords is None:
            try:
                coords = self._path_cache['daofind_output']
            except KeyError:
                raise DaophotError('You need to run Daophot.daofind '
                                   'before Daophot.apphot can be used.')
        self._path_cache['apphot_output'] = os.path.join(self._workdir, output_filename)
        phot_args = dict(image = self._path_cache['image_path'],
                         output = self._path_cache['apphot_output'],
                         coords = coords,
                         verify = 'no',
                         interactive = 'no',
                         cache = 'yes',
                         verbose = 'no')
        iraf.daophot.phot(**phot_args)

    @timed
    def psf(self, failsafe=True):
        """Fits a PSF model.

        Parameters
        ----------
        failsafe : bool
            If true and the PSF fitting fails to converge, then re-try the fit
            automatically with varorder = 0 and function = 'auto'.
        """
        if not self._path_cache.has_key('apphot_output'):
            raise DaophotError('You need to run Daophot.apphot '
                               'before Daophot.psf can be used.')
        self._path_cache['pstselect_output'] = os.path.join(self._workdir, 'output-pstselect.txt')
        pstselect_args = dict(mode = 'h',
                              image = self._path_cache['image_path'],
                              photfile = self._path_cache['apphot_output'],
                              pstfile = self._path_cache['pstselect_output'],
                              verify = 'no',
                              interactive = 'no',
                              verbose = 'yes',
                              Stdout = str(os.path.join(self._workdir,
                                                    'log-pstselect.txt')))
        iraf.daophot.pstselect(**pstselect_args)
        
        #self._path_cache['psf_output'] = os.path.join(self._workdir, 'output-psf')  # daophot will append .fits
        self._path_cache['psf_output'] = os.path.join(self._workdir, 'output-psf')
        self._path_cache['psg_output'] = os.path.join(self._workdir, 'output-psg.txt')
        self._path_cache['psf_pst_output'] = os.path.join(self._workdir, 'output-psf-pst.txt')
        psf_args = dict(image = self._path_cache['image_path'],
                        photfile = self._path_cache['apphot_output'],
                        psfimage = self._path_cache['psf_output'],
                        pstfile = self._path_cache['pstselect_output'],
                        groupfile = self._path_cache['psg_output'],
                        opstfile = self._path_cache['psf_pst_output'],
                        verify = 'no',
                        interactive = 'no',
                        cache = 'yes',
                        verbose = 'yes',
                        Stdout = str(os.path.join(self._workdir,
                                              'log-psf.txt')))
        iraf.daophot.psf(**psf_args)

        # It is possible for iraf.daophot.psf() to fail to converge.
        # In this case, we re-try with more easy-to-fit settings.
        if failsafe and not os.path.exists(self._path_cache['psf_output'] + '.fits'):
            log.warning('iraf.daophot.psf appears to have failed; '
                        'now trying again in failsafe mode.')
            tmp_varorder, tmp_function = iraf.daopars.varorder, iraf.daopars.function
            iraf.daopars.varorder = 0
            iraf.daopars.function = 'auto'
            iraf.daophot.psf(**psf_args)
            iraf.daopars.varorder, iraf.daopars.function = tmp_varorder, tmp_function

        # Save the resulting PSF into a user-friendly FITS file
        self._path_cache['seepsf_output'] = os.path.join(self._workdir, 'output-seepsf')  # daophot will append .fits
        seepsf_args = dict(psfimage = self._path_cache['psf_output'],
                           image = self._path_cache['seepsf_output'])
        iraf.daophot.seepsf(**seepsf_args)

    @timed
    def allstar(self):
        if not self._path_cache.has_key('psf_output'):
            raise DaophotError('You need to run Daophot.psf '
                               'before Daophot.allstar can be used.')
        self._path_cache['allstar_output'] = os.path.join(self._workdir, 'output-allstar.txt')
        self._path_cache['subimage_output'] = os.path.join(self._workdir, 'output-subimage')  # daophot will append .fits
        stderr = allstar_args = dict(image = self._path_cache['image_path'],
                                    photfile = self._path_cache['apphot_output'],
                                    psfimage = self._path_cache['psf_output'] + '.fits[0]',
                                    allstarfile = self._path_cache['allstar_output'],
                                    rejfile = None,
                                    subimage = self._path_cache['subimage_output'],
                                    verify = 'no',
                                    verbose = 'no',
                                    cache = 'yes',
                                    Stderr = 1)
        iraf.daophot.allstar(**allstar_args)

    def save_fits_catalogue(self):
        if hasattr(self, 'output_daofind'):
            self.catalogue_daofind = os.path.join(self._workdir, 'output-daofind.fits')
            self.get_daofind_table().write(self.catalogue_daofind, format='fits', overwrite=True)

        self.catalogue_phot = os.path.join(self._workdir, 'output-phot.fits')
        self.get_phot_table().write(self.catalogue_phot, format='fits', overwrite=True)

        self.catalogue_allstar = os.path.join(self._workdir, 'output-allstar.fits')
        self.get_allstar_table().write(self.catalogue_allstar, format='fits', overwrite=True)

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
        # Compute a Signal-To-Noise estimate based on the FWHM aperture photometry
        # Our SNR is marginally better than the MERR computed by DAOPHOT, likely because we don't fold in the flatfield error?
        variance_per_pixel = tbl['STDEV']**2 + (iraf.datapars.readnoi / iraf.datapars.epadu)**2 # stdev in ADU, gain in photons/ADU, readnoise in photons
        variance_signal = tbl['FLUX'] / iraf.datapars.epadu
        with np.errstate(divide='ignore', invalid='ignore'):
            tbl['SNR'] = tbl['FLUX'] / np.sqrt(variance_signal + tbl['AREA']*variance_per_pixel)
            # Add the 3-sigma detection limit; cf. e.g. http://www.ast.cam.ac.uk/~xmmssc/xid-imaging/dqc/wfc_tech/flux2mag.html
            tbl['LIM3SIG'] = iraf.photpars.zmag - 2.5*np.log10(3*np.sqrt(tbl['AREA']*variance_per_pixel) / iraf.datapars.itime)
            #Identical to daophot.phot: tbl['APERMAG'] = iraf.photpars.zmag - 2.5*np.log10(tbl['FLUX'] / iraf.datapars.itime)

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
