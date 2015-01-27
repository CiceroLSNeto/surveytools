"""Plots a visualisation of a VST/OmegaCam exposure, showing all 32 CCD frames.

Usage
=====
`vst-pawplot filename.fits`
"""
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
import numpy as np
from progressbar import ProgressBar

from astropy import log
from astropy.io import fits

# Position of the CCDs: left-right = East-West and top-bottom = North-South;
# the numbers refer to the FITS HDU extension number of an OmegaCam image.
OMEGACAM_CCD_ARRANGEMENT = [32, 31, 30, 29, 16, 15, 14, 13,
                            28, 27, 26, 25, 12, 11, 10,  9,
                            24, 23, 22, 21,  8,  7,  6,  5,
                            20, 19, 18, 17,  4,  3,  2,  1]


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
                        cmap=cmap,
                        origin='lower')
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
    vst_pawplot('/local/home/gb/tmp/vphas-201209/single/o20120918_00025.fit', '/tmp/test.jpg')
