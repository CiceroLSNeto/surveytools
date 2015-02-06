"""Plots a visualisation of a VST/OmegaCam exposure, showing all 32 CCD frames.

Usage
=====
`vst-pawplot filename.fits`
"""
import os
import numpy as np
from progressbar import ProgressBar

import matplotlib
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
import matplotlib.image as mimg

from astropy import log
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import AsymmetricPercentileInterval, SqrtStretch

from . import VPHAS_DATA_PATH, OMEGACAM_CCD_ARRANGEMENT
from .footprint import VphasOffset


##########
# CLASSES
##########

class LuptonColorStretch():
    """
    Stretch a MxNx3 (RGB) array using the colour-preserving method by
    Lupton et al. (arXiv:astro-ph/0312483).

    Parameters
    ----------
    intensity_stretch : object derived from `BaseStretch`
        The stretch to apply on the integrated intensity, e.g. `LogStretch`.
    """
    def __init__(self, intensity_stretch=None):
        if intensity_stretch is None:
            self.intensity_stretch = SqrtStretch()
        else:
            self.intensity_stretch = intensity_stretch

    def __call__(self, values):
        """Transform the image using this stretch.

        Parameters
        ----------
        values : `~numpy.ndarray` of shape MxNx3 (RGB)
            The input image, which should already be normalized to the [0:1]
            range.

        Returns
        -------
        new_values : `~numpy.ndarray`
            The transformed values.
        """
        intensity = values.sum(axis=2) / 3.
        old_settings = np.seterr(divide='ignore', invalid='ignore')
        for channel in range(3):
            values[:, :, channel] *= self.intensity_stretch(intensity) / intensity
            values[:, :, channel][intensity == 0] = 0
        np.seterr(**old_settings)
        maxrgb = values.max(axis=2)
        maxrgb_gt_1 = maxrgb > 1
        for channel in range(3):
            values[:, :, channel][maxrgb_gt_1] /= maxrgb[maxrgb_gt_1]
        return values


class VphasColourFrame():
    """Create a pretty colour quicklook from a VPHAS frame.

    Parameters
    ----------
    offset : str
        Name of the VPHAS offset, e.g. "0001a".

    ccd : int
        CCD number, referring to the FITS extension number.
    """
    def __init__(self, offset, ccd):
        self.offset = offset
        self.ccd = ccd
        self.filenames = VphasOffset(offset).get_filenames()

    def as_array(self,
                 interval_r=AsymmetricPercentileInterval(2.5, 99.0),
                 interval_g=AsymmetricPercentileInterval(5., 99.2),
                 interval_b=AsymmetricPercentileInterval(10., 99.2)):
        """Returns the colour image as a MxNx3 (RGB) array."""
        # First we determine the shifts
        cx, cy = 2050, 1024  # central coordinates of the image
        maxshiftx, maxshifty = 0, 0
        cimg = {}
        for idx, band in enumerate(self.filenames.keys()):
            fn = self.filenames[band]
            hdu = fits.open(os.path.join(VPHAS_DATA_PATH, fn))[self.ccd]
            wcs = WCS(hdu.header)
            img = hdu.data
            if idx == 0:
                cra, cdec = wcs.wcs_pix2world(cx, cy, 1)
            else:
                refx, refy = wcs.wcs_world2pix(cra, cdec, 1)
                shiftx = int(refx - cx)
                shifty = int(refy - cy)
                maxshiftx = max(abs(shiftx), maxshiftx)
                maxshifty = max(abs(shifty), maxshifty)
                if shiftx > 0:
                    img = np.pad(img, ((0, 0), (0, shiftx)), mode='constant')[:, shiftx:]
                elif shiftx < 0:
                    img = np.pad(img, ((0, 0), (-shiftx, 0)), mode='median')[:, :shiftx]            
                if shifty > 0:
                    img = np.pad(img, ((0, shifty), (0, 0)), mode='median')[shifty:, :]
                elif shifty < 0:
                    img = np.pad(img, ((-shifty, 0), (0, 0)), mode='constant')[:shifty, :]          
            cimg[band] = img

        r = interval_r(cimg['i'] + cimg['ha'])
        g = interval_g(cimg['g'] + cimg['r'] + cimg['r2'])
        b = interval_b(cimg['u'] + 2 * cimg['g'])
        stacked = np.dstack((r[maxshifty:-maxshifty, maxshiftx:-maxshiftx],
                             g[maxshifty:-maxshifty, maxshiftx:-maxshiftx],
                             b[maxshifty:-maxshifty, maxshiftx:-maxshiftx]))
        return LuptonColorStretch()(stacked)

    def imsave(self, output_fn=None):
        if output_fn is None:
            output_fn = 'vphas-{0}-{1}.jpg'.format(self.offset, self.ccd)
        log.info('Writing {0}'.format(output_fn))
        mimg.imsave(output_fn, np.rot90(self.as_array()), origin='lower')


############
# FUNCTIONS
############

def vphas_quicklook(offset, ccd, output_fn=None):
    VphasColourFrame(offset, ccd).imsave(output_fn)

def vst_pawplot(filename, out_fn=None, dpi=100,
                min_percent=1.0, max_percent=99.5,
                cmap='gist_heat', show_hdu=False):
    """Plots the 32-CCD OmegaCam mosaic as a pretty bitmap.

    Parameters
    ----------
    filename : str
        The filename of the FITS file.
    out_fn : str
        The filename of the output bitmap image.  The type of bitmap
        is determined by the filename extension (e.g. '.jpg', '.png').
        The default is a PNG file with the same name as the FITS file.
    dpi : float, optional [dots per inch]
        Resolution of the output 10-by-9 inch output bitmap.
        The default is 100.
    min_percent : float, optional
        The percentile value used to determine the pixel value of
        minimum cut level.  The default is 1.0.
    max_percent : float, optional
        The percentile value used to determine the pixel value of
        maximum cut level.  The default is 99.5.
    cmap : str, optional
        The matplotlib color map name.  The default is 'gist_heat',
        can also be e.g. 'Greys_r'.
    show_hdu : boolean, optional
        Plot the HDU extension number if True (default: False).
    """
    if out_fn is None:
        out_fn = filename + '.png'
    if cmap not in pl.cm.datad.keys():
        raise ValueError('{0} is not a valid matplotlib colormap '
                         'name'.format(cmap))
    log.info('Writing {0}'.format(out_fn))
    # Prepare the plot
    f = fits.open(filename)
    pl.interactive(False)
    fig = pl.figure(figsize=(10, 9))
    # Determine vmin/vmax based on a sample of pixels across the mosaic
    sample = np.concatenate((f[1].data[::20, ::10],
                             f[11].data[::20, ::10],
                             f[22].data[::20, ::10],
                             f[32].data[::20, ::10]))
    vmin, vmax = np.percentile(sample, [min_percent, max_percent])
    del sample  # save memory
    log.debug('vst_pawplot: vmin={0}, vmax={1}'.format(vmin, vmax))
    # Plot the extensions
    pbar = ProgressBar(32).start()
    for idx, hduno in enumerate(OMEGACAM_CCD_ARRANGEMENT):
        log.debug('vst_pawplot: adding HDU #{0}'.format(hduno))
        ax = fig.add_subplot(4, 8, idx+1)
        sampling = int(500 / dpi)
        im = ax.matshow(f[hduno].data[::sampling, ::-sampling],
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap=cmap, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        if show_hdu:
            # Show the HDU extension number on top of the image
            txt = ax.text(0.05, 0.97, hduno, fontsize=14, color='white',
                          ha='left', va='top', transform=ax.transAxes)
            # Add a black border around the text
            txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()])
        pbar.update(idx)
    pbar.finish()

    # Aesthetics and colorbar
    fig.subplots_adjust(left=0.04, right=0.85,
                        top=0.93, bottom=0.05,
                        wspace=0.02, hspace=0.02)
    cbar_ax = fig.add_axes([0.9, 0.06, 0.02, 0.86])
    t = np.logspace(np.log10(vmin), np.log10(vmax), num=10)
    cb = fig.colorbar(im, cax=cbar_ax, ticks=t, format='%.0f')
    cb.set_label('Pixel count [ADU]')
    # Title and footer text
    fig.text(0.05, 0.95, filename, fontsize=16, ha='left')
    try:
        filterfooter = '{0} ({1}/{2})'.format(
                       f[0].header['ESO INS FILT1 NAME'],
                       f[0].header['ESO TPL EXPNO'],
                       f[0].header['ESO TPL NEXP'])
        fig.text(0.04, 0.02, f[0].header['OBJECT'], fontsize=12, ha='left')
        fig.text(0.50, 0.02, filterfooter, fontsize=12, ha='right')
        fig.text(0.85, 0.02, f[0].header['DATE-OBS'][0:19],
                 fontsize=12, ha='right')
    except KeyError as e:
        log.warning('Could not write footer text: {0}'.format(e))
        pass
    # We're done
    fig.savefig(out_fn, dpi=dpi)
    pl.close()
    del f

def vst_pawplot_main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a beautiful plot showing the pixel data in a 32-CCD VST/OmegaCAM FITS file.')
    parser.add_argument('-o', metavar='filename', type=str, default=None,
                    help='Filename for the output image (Default is a '
                    'PNG file with the same name as the FITS file)')
    parser.add_argument('--dpi', type=float, default=100.,
                        help=('The resolution of the output image.'))
    parser.add_argument('--min_percent', type=float, default=1.0,
                        help=('The percentile value used to determine the '
                              'minimum cut level'))
    parser.add_argument('--max_percent', type=float, default=99.5,
                        help=('The percentile value used to determine the '
                              'maximum cut level'))
    parser.add_argument('--cmap', metavar='colormap_name', type=str,
                        default='gist_heat', help='matplotlib color map name')
    parser.add_argument('--show-hdu', action='store_true',
                        help='Display the HDU extension numbers.')
    parser.add_argument('filename', nargs='+',
                        help='Path to one or more FITS files to convert')
    args = parser.parse_args(args)

    for filename in args.filename:
        vst_pawplot(filename,
                    out_fn=args.o,
                    dpi=args.dpi,
                    min_percent=args.min_percent,
                    max_percent=args.max_percent,
                    cmap=args.cmap,
                    show_hdu=args.show_hdu)


if __name__ == '__main__':
    # Example use:
    vst_pawplot(os.path.join(VPHAS_DATA_PATH, 'o20120918_00025.fit'), '/tmp/test.jpg')
    vphas_quicklook("0778a", 24, "/tmp/test2.jpg")
