"""Tools to create quicklook visualisations of VST/OmegaCAM data."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import imageio

import matplotlib
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
import matplotlib.image as mimg

from astropy import log
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ManualInterval, AsymmetricPercentileInterval, SqrtStretch, LogStretch, AsinhStretch
from astropy.utils.console import ProgressBar

from . import VPHAS_DATA_PATH, OMEGACAM_CCD_ARRANGEMENT
from .footprint import VphasOffset, NotObservedException


##########
# CLASSES
##########

class VphasDataException(Exception):
    pass

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
            self.intensity_stretch = AsinhStretch(a=0.03)
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
        for chn in range(3):
            values[:, :, chn] *= self.intensity_stretch(intensity) / intensity
            values[:, :, chn][intensity == 0] = 0
        np.seterr(**old_settings)
        maxrgb = values.max(axis=2)
        maxrgb_gt_1 = maxrgb > 1
        for chn in range(3):
            values[:, :, chn][maxrgb_gt_1] /= maxrgb[maxrgb_gt_1]
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
        if len(self.filenames) < 6:
            raise NotObservedException('Not all six bands have been observed'
                                       ' for offset {0}.'.format(offset))

    def as_array(self,
                 interval_r=AsymmetricPercentileInterval(2.5, 99.0),
                 interval_g=AsymmetricPercentileInterval(5., 99.2),
                 interval_b=AsymmetricPercentileInterval(10., 99.2)):
        """Returns the colour image as a MxNx3 (RGB) array."""
        # First we determine the shifts
        cx, cy = 2050, 1024  # central coordinates of the image
        maxshiftx, maxshifty = 0, 0
        aligned_imgs = {}
        for idx, band in enumerate(self.filenames.keys()):
            fn = self.filenames[band]
            hdu = fits.open(os.path.join(VPHAS_DATA_PATH, fn))[self.ccd]
            wcs = WCS(hdu.header)
            img = hdu.data
            if idx == 0:  # The first image acts as reference
                cra, cdec = wcs.wcs_pix2world(cx, cy, 1)
            else:  # For all subsequent images, compute the shift using the WCS
                refx, refy = wcs.wcs_world2pix(cra, cdec, 1)
                shiftx = int(refx - cx)
                shifty = int(refy - cy)
                # Now apply the required shift to the image
                if shiftx > 0:
                    img = np.pad(img, ((0, 0), (0, shiftx)),
                                 mode=str('constant'))[:, shiftx:]
                elif shiftx < 0:
                    img = np.pad(img, ((0, 0), (-shiftx, 0)),
                                 mode=str('constant'))[:, :shiftx]
                if shifty > 0:
                    img = np.pad(img, ((0, shifty), (0, 0)),
                                 mode=str('constant'))[shifty:, :]
                elif shifty < 0:
                    img = np.pad(img, ((-shifty, 0), (0, 0)),
                                 mode=str('constant'))[:shifty, :]
                # The maximum shift applied will determine the final img shape
                maxshiftx = max(abs(shiftx), maxshiftx)
                maxshifty = max(abs(shifty), maxshifty)
            aligned_imgs[band] = img

        if maxshiftx > cx or maxshifty > cy:
            raise VphasDataException('{0}-{1}: bands do not overlap'.format(self.offset, self.ccd))
        # New stretch, scale, and stack the data into an MxNx3 array
        r = aligned_imgs['i'] + aligned_imgs['ha']
        g = aligned_imgs['g'] + aligned_imgs['r'] + aligned_imgs['r2']
        b = aligned_imgs['u'] + 2 * aligned_imgs['g']
        r, g, b = 1.5*r, 0.8*g, 2.2*b
        vmin_r, vmax_r = np.percentile(r, [1., 99.5])
        vmin_g, vmax_g = np.percentile(g, [5., 99.5])
        vmin_b, vmax_b = np.percentile(b, [10., 99.5])
        #log.info((vmin_r, vmin_g, vmin_b))
        #log.info((vmax_r, vmax_g, vmax_b))
        #if vmin_b < 100:
        #    vmin_b = 100
        minrange = np.max((1250., vmax_g-vmin_g, vmax_g-vmin_g, vmax_b-vmin_b))
        if (vmax_r - vmin_r) < minrange:
            vmax_r = vmin_r + minrange
        if (vmax_g - vmin_g) < minrange:
            vmax_g = vmin_g + minrange
        if (vmax_b - vmin_b) < minrange:
            vmax_b = vmin_b + minrange
        interval_r = ManualInterval(vmin_r, vmax_r)
        interval_g = ManualInterval(vmin_g, vmax_g)
        interval_b = ManualInterval(vmin_b, vmax_b)
        
        r = interval_r(r)
        g = interval_g(g)
        b = interval_b(b)
        stacked = np.dstack((r[maxshifty:-maxshifty, maxshiftx:-maxshiftx],
                             g[maxshifty:-maxshifty, maxshiftx:-maxshiftx],
                             b[maxshifty:-maxshifty, maxshiftx:-maxshiftx]))
        return LuptonColorStretch()(stacked)

    def imsave(self, out_fn=None):
        if out_fn is None:
            out_fn = 'vphas-{0}-{1}.jpg'.format(self.offset, self.ccd)
        log.info('Writing {0}'.format(out_fn))
        #mimg.imsave(out_fn, np.rot90(self.as_array()), origin='lower')
        img = np.rot90(self.as_array())
        imageio.imsave(out_fn, img, quality=90, optimize=True)


############
# FUNCTIONS
############

def vphas_quicklook(offset, ccd=None, out_fn=None):
    VphasColourFrame(offset, ccd).imsave(out_fn)


def vphas_quicklook_main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a beautiful color image from single VPHAS frames.')
    parser.add_argument('-o', '--output', metavar='filename',
                        type=str, default=None,
                        help='Filename for the output image (Default is a '
                             'JPG file named vphas-offset-ccd.jpg)')
    parser.add_argument('-c', '--ccd', nargs='+', type=int, default=None,
                        help='CCD number between 1 and 32.'
                             '(Default is to save all CCDs.)')
    parser.add_argument('--min_percent_r', type=float, default=1.0,
                        help=('The percentile value used to determine the '
                              'minimum cut level for the red channel'))
    parser.add_argument('--max_percent_r', type=float, default=99.5,
                        help=('The percentile value used to determine the '
                              'maximum cut level for the red channel'))
    parser.add_argument('offset', nargs='+',
                        help='Name of the VPHAS offset pointing.')
    args = parser.parse_args(args)

    if args.ccd is None:
        args.ccd = range(1, 33)
    for offset in args.offset:
        try:
            for ccd in args.ccd:
                vphas_quicklook(offset,
                                ccd=ccd,
                                out_fn=args.output)
        except NotObservedException as e:
            log.error(e)


def vst_pawplot(filename, out_fn=None, dpi=100,
                min_cut=None, max_cut=None,
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
        The default is a JPG file with the same name as the FITS file.

    dpi : float, optional [dots per inch]
        Resolution of the output 10-by-9 inch output bitmap.
        The default is 100.

    min_cut : float, optional
        The pixel value of the minimum cut level.  Data values less than
        ``min_cut`` will set to ``min_cut`` before scaling the image.
        The default is the image minimum.  ``min_cut`` overrides
        ``min_percent``.

    max_cut : float, optional
        The pixel value of the maximum cut level.  Data values greater
        than ``min_cut`` will set to ``min_cut`` before scaling the
        image.  The default is the image maximum.  ``max_cut`` overrides
        ``max_percent``.

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
    # Check input
    if out_fn is None:
        out_fn = filename + '.jpg'
    log.info('Writing {0}'.format(out_fn))
    if cmap not in pl.cm.datad.keys():
        raise ValueError('{0} is not a valid matplotlib colormap '
                         'name'.format(cmap))
    # Determine the interval
    f = fits.open(filename)
    if min_cut is not None or max_cut is not None:
        vmin, vmax = min_cut or 0, max_cut or 65536
    else:
        # Determine vmin/vmax based on a sample of pixels across the mosaic
        sample = np.concatenate([f[hdu].data[::200, ::100] for hdu
                                 in [1, 6, 8, 12, 13, 20, 21, 23, 25, 27, 32]])
        vmin, vmax = np.percentile(sample, [min_percent, max_percent])
        del sample  # save memory
    log.debug('vst_pawplot: vmin={0}, vmax={1}'.format(vmin, vmax))
    # Setup the figure and plot the extensions
    pl.interactive(False)
    fig = pl.figure(figsize=(10, 9))
    idx = 0
    for hduno in ProgressBar(OMEGACAM_CCD_ARRANGEMENT):
        idx += 1
        log.debug('vst_pawplot: adding HDU #{0}'.format(hduno))
        ax = fig.add_subplot(4, 8, idx)
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
    # Aesthetics and colorbar
    fig.subplots_adjust(left=0.04, right=0.85,
                        top=0.93, bottom=0.05,
                        wspace=0.02, hspace=0.02)
    cbar_ax = fig.add_axes([0.9, 0.06, 0.02, 0.86])
    t = np.logspace(np.log10(vmin), np.log10(vmax), num=10)
    if (vmax - vmin) > 10:
        fmt = '%.0f'
    elif (vmax - vmin) > 1:
        fmt = '%.1f'
    else:
        fmt = '%.2f'
    cb = fig.colorbar(im, cax=cbar_ax, ticks=t, format=fmt)
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
    # Save output and clean up
    fig.savefig(out_fn, dpi=dpi)
    pl.close()
    del f


def vst_pawplot_main(args=None):
    """Interface inspired by AstroPy's ``fits2bitmap`` script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a beautiful plot showing the data for all 32 CCDs '
                    'in a single image.')
    parser.add_argument('-o', metavar='filename',
                        type=str, default=None,
                        help='Filename for the output image (Default is a '
                        'PNG file with the same name as the FITS file)')
    parser.add_argument('--dpi', type=float, default=100.,
                        help=('The resolution of the output image.'))
    parser.add_argument('--min_cut', type=float, default=None,
                        help='The pixel value of the minimum cut level')
    parser.add_argument('--max_cut', type=float, default=None,
                        help='The pixel value of the maximum cut level')
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
                    min_cut=args.min_cut,
                    max_cut=args.max_cut,
                    min_percent=args.min_percent,
                    max_percent=args.max_percent,
                    cmap=args.cmap,
                    show_hdu=args.show_hdu)


if __name__ == '__main__':
    # Example use:
    vst_pawplot(os.path.join(VPHAS_DATA_PATH, 'o20120918_00025.fit'),
                '/tmp/test.jpg')
    vphas_quicklook("0778a", 24, "/tmp/test2.jpg")
