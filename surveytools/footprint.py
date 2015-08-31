"""Defines the VPHAS survey footprint.

Example use
-----------
fp = VphasFootprint()
fp.get_field_dict()  # returns {'0001a': SkyCoord(), ...}
fp.get_field_corners(fieldname)
fp.get_plot()

Warning
-------
In the first ~month of the survey, the third H-alpha offset was
observed at the wrong position (at RA -888" / Dec +1010").
The pointings affected are:
['0004b', '0005b', '0030b', '0032b', '0034b',
 '0298b', '0299b', '0392b', '0393b', '0394b',
 '0441b', '0442b', '0443b', '1043b', '1044b',
 '1045b', '1046b', '1321b', '1322b', '1326b',
 '1327b', '1328b', '1413b', '1414b', '1786b',
 '1787b', '1788b', '1789b', '1790b', '1791b']:
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import warnings
import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy import log

from . import SURVEYTOOLS_DATA, VPHAS_DATA_PATH
from .utils import timed, cached_property


###########
# CLASSES
###########


class NotObservedException(Exception):
    """Raised if a requested field has not been observed yet."""
    pass


class VphasExposure():

    def __init__(self, filename, directory=VPHAS_DATA_PATH):
        self.filename = filename
        self.path = os.path.join(directory, filename)

    @cached_property
    def hdulist(self):
        return fits.open(self.path)

    @cached_property
    def header(self):
        """FITS header object."""
        return self.hdulist[0].header

    @cached_property
    def offset_name(self):
        """VPHAS name of the offset, e.g. '0001a'."""
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
    def name(self):
        """VPHAS name of the exposure, e.g. '0001a-ha'."""
        return '{0}-{1}'.format(self.offset_name, self.band)

    def frames(self):
        """Returns metadata about the 32 frames in the exposure.
        
        pseudocode:
        
        for filename in filenames:
            for frame in frames:
                obtain frame_corners, seeing, limmag,
                background level, qcgrade, indr
                add to table
        """
        tbl = Table(names=('frame', 'offset', 'ccd', 'band', 'filename', 'ra1', 'dec1', 'ra2', 'dec2', 'ra3', 'dec3', 'ra4', 'dec4'),
                    dtype=('O', 'O', int, 'O', 'O', float, float, float, float, float, float, float, float))
        for ccd in np.arange(1, 33):
            name = '{0}-{1}-{2}'.format(self.offset_name, ccd, self.band)
            row = [name, self.offset_name, ccd, self.band, self.filename]
            xmax, ymax = self.hdulist[ccd].header['NAXIS1'], self.hdulist[ccd].header['NAXIS2']
            corners = [[0, 0], [xmax, 0], [xmax, ymax], [0, ymax]]
            wcs = WCS(self.hdulist[ccd].header)
            mycorners = wcs.wcs_pix2world(corners, 1)
            import pdb
            pdb.set_trace()
            for value in mycorners.reshape(8):
                row.append(value)
            tbl.add_row(row)
        return tbl


class VphasOffset():
    """Provides filenames and meta data on VPHAS offset pointings.

    Parameters
    ----------
    name : str
        Identifier of the VPHAS offset; must be a 5-character wide string
        composed of a 4-digit zero padded number followed by 'a', 'b', 
        or 'c' to denote the offset, e.g. '0149a' is the first offset
        of field 'vphas_0149'.
    """
    def __init__(self, name):
        if len(name) != 5 or name[-1] not in ['a', 'b', 'c']:
            raise ValueError('Incorrectly formatted VPHAS offset name.')
        self.name = name

    @cached_property
    def image_filenames(self):
        return self.get_filenames()

    def get_filenames(self):
        try:
            red = self.get_red_filenames()
        except NotObservedException:
            red = {}
        try:
            blue = self.get_blue_filenames()
        except NotObservedException:
            blue = {}
        red.update(blue)
        return red

    def get_red_filenames(self):
        """Returns the H-alpha, r- and i-band FITS filenames of the red concat.

        Returns
        -------
        filenames : dict
            Dictionary of the form
            {'ha':'filename', 'r': 'filename', 'i': 'filename'}
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='(.*)did not parse '
                                                      'as fits unit(.*)')
            metadata = Table.read(os.path.join(SURVEYTOOLS_DATA,
                                               'vphas-dr2-red-images.fits'))
        fieldname = 'vphas_' + self.name[:-1]
        # Has the field been observed?
        if (metadata['Field_1'] == fieldname).sum() == 0:
            raise NotObservedException('{0} has not been observed in the red filters'.format(self.name))

        offset2idx = {'a': 0, 'b': -1, 'c': 1}
        idx = offset2idx[self.name[-1:]]
        # Define the colloquial band names used in the catalogue
        filter2band = {'NB_659': 'ha', 'r_SDSS': 'r', 'i_SDSS': 'i'}
        result = {}
        for filtername, bandname in filter2band.items():
            mask = ((metadata['Field_1'] == fieldname)
                    & (metadata['filter'] == filtername))
            filenames = metadata['image file'][mask]
            if filtername == 'NB_659':
                assert len(filenames) == 3  # sanity check
            else:
                assert len(filenames) == 2  # sanity check
            # TODO: it is not generally true that the sequence is chronological, i.e. the sort below does not work in a few cases!
            filenames.sort()
            result[bandname] = filenames[idx]
        return result

    def get_blue_filenames(self):
        """Returns the u-, g- and r-band FITS filenames of the blue concat.

        Returns
        -------
        filenames : dict
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='(.*)did not parse '
                                                      'as fits unit(.*)')
            metadata = Table.read(os.path.join(SURVEYTOOLS_DATA, 'vphas-dr2-blue-images.fits'))
        fieldname = 'vphas_' + self.name[:-1]
        # Has the field been observed?
        if (metadata['Field_1'] == fieldname).sum() == 0:
            raise NotObservedException('{0} has not been observed in the blue filters'.format(self.name))
        offset2idx = {'a': 0, 'b': -1, 'c': 1}
        idx = offset2idx[self.name[-1:]]
        # Define the colloquial band names used in the catalogue
        filter2band = {'u_SDSS': 'u', 'g_SDSS': 'g', 'r_SDSS': 'r2'}
        result = {}
        for filtername, bandname in filter2band.items():
            mask = ((metadata['Field_1'] == fieldname)
                    & (metadata['filter'] == filtername))
            filenames = metadata['image file'][mask]
            if filtername != 'g_SDSS':
                assert len(filenames) == 2  # sanity check
            filenames.sort()
            result[bandname] = filenames[idx]
        return result

    @cached_property
    def properties(self):
        meta = {'name': self.name}
        for band in self.image_filenames:
            fn = self.image_filenames[band]
            f = fits.open(os.path.join(VPHAS_DATA_PATH, fn))
            meta[band+'_filename'] = fn
            # The pointing center is near the SW corner of CCD #25;
            # the inter-CCD gaps being roughly 88px (long side) and 46px (short side)
            ra, dec = WCS(f[25].header).wcs_pix2world([-46/2.], [-88/2.], 1)
            ra, dec = WCS(f[25].header).wcs_pix2world([0], [0], 1)
            meta[band+'_ra'] = ra[0]
            meta[band+'_dec'] = dec[0]
        return meta


class VphasFootprint():

    def __init__(self):
        pass

    def get_field_table(self, compute_corners=True):
        return Table.read(os.path.join(SURVEYTOOLS_DATA,
                                       'vphas-field-coordinates.csv'))

    @cached_property
    def offsets(self):
        """Returns a dictionary detailing the coverage of each VPHAS pointin.

        Both the centres and corners of all the offsets are returned,
        both in equatorial and galactic coordinates.

        The implementation looks ugly because it has been optimized for speed.
        """
        tbl = self.get_field_table()
        # Offset centres in equatorial
        cosdec = np.cos(np.radians(tbl['dec']))
        cntr = {'ra_a': tbl['ra'],
                'dec_a': tbl['dec'],
                'ra_b': tbl['ra'] - (588/3600.) / cosdec,
                'dec_b': tbl['dec'] + (660/3600.),
                'ra_c': tbl['ra'] - (300/3600.) / cosdec,
                'dec_c': tbl['dec'] + (350/3600.),
                }
        # Offsets centres in galactic system
        for pos in ['a', 'b', 'c']:
            cntr['l_'+pos], cntr['b_'+pos] = self._icrs2gal(cntr['ra_'+pos], cntr['dec_'+pos])
        # Compute the corners of the field in ICRS and galactic coordinates
        # this code is ugly because it is optimized for speed
        cornershifts = [[-0.5,-0.5], [-0.5,+0.5], [+0.5, +0.5], [+0.5, -0.5], [-0.5,-0.5]]
        polygons = {}
        polygons_gal = {}
        for pos in ['a', 'b', 'c']:
            corners_ra, corners_dec = [], []
            corners_l, corners_b = [], []
            for idx, shift in enumerate(cornershifts):
                ra = cntr['ra_'+pos] + shift[0] / cosdec
                dec = cntr['dec_'+pos] + shift[1]
                corners_ra.append(ra)
                corners_dec.append(dec)
                l, b = self._icrs2gal(ra, dec)
                l[l > 180] -= 360
                corners_l.append(l)
                corners_b.append(b)
            polygons[pos] = [
                            [[corners_ra[idx][fieldidx], corners_dec[idx][fieldidx]] for idx in range(len(cornershifts))]
                            for fieldidx in range(len(tbl))
                            ]
            polygons_gal[pos] = [
                            [[corners_l[idx][fieldidx], corners_b[idx][fieldidx]] for idx in range(len(cornershifts))]
                            for fieldidx in range(len(tbl))
                            ]
        # Creating the resulting dictionary we will return to the user
        result = {}
        for idx in range(len(tbl)):
            for pos in ['a', 'b', 'c']:
                result[tbl['field'][idx][-4:]+pos] = {'ra': cntr['ra_'+pos][idx],
                                                      'dec': cntr['dec_'+pos][idx],
                                                      'l': cntr['l_'+pos][idx],
                                                      'b': cntr['b_'+pos][idx],
                                                      'polygon': polygons[pos][idx],
                                                      'polygon_gal': polygons_gal[pos][idx]
                                                   }
        return result

    def _icrs2gal(self, ra, dec):
        """Convert (ra,dec) in ICRS to galactic."""
        crd = SkyCoord(ra, dec, unit=('deg', 'deg'))
        gal = crd.galactic
        return [gal.l.deg, gal.b.deg]

    @timed
    def plot(self, offsets=None):
        """
        Parameters
        -----------
        highlight : list of tuples
            each tuple should be composed of (fieldlist, colour, title), e.g.
            [(['0001a', '0002a'], '#e41a1c', 'u,g,r observed')]
        """
        if offsets is None:
            offsets = self.offsets
        fig = pl.figure(figsize=(7.5, 4)) #7,5
        fig.subplots_adjust(0.08, 0.04, 0.96, 0.99, wspace=0.15, hspace=0.15)
        glat1 = -11
        glat2 = 11
        stepsize = 100
        xstart = [-60, -160]
        for i in range(len(xstart)):
            ax = fig.add_subplot(2.0, 1, i+1)
            fig.subplots_adjust(left=0.09, bottom=0.12, 
                                right=0.98, top=0.87, 
                                hspace=0.2, wspace=0.2)

            patches = {}
            for name in offsets:
                try:
                    facecolor = offsets[name]['facecolor']
                except KeyError:
                    facecolor = "#dddddd"
                try:
                    edgecolor = offsets[name]['edgecolor']
                except KeyError:
                    edgecolor = "#999999"
                try:
                    zorder = offsets[name]['zorder']
                except KeyError:
                    zorder = -100
                try:
                    label = offsets[name]['label']
                except KeyError:
                    label = 'planned'
                poly = mpl.patches.Polygon(offsets[name]['polygon_gal'],
                                                  alpha=1,
                                                  facecolor=facecolor, 
                                                  edgecolor=edgecolor,
                                                  zorder=zorder)
                try:
                    patches[label].append(poly)
                except KeyError:
                    patches[label] = [poly]
                ax.add_patch(poly)

            glon1 = xstart[i]
            glon2 = glon1+stepsize
            
            ax.set_xlim(glon1+stepsize, glon1)
            ax.set_ylim([glat1, glat2])

            ax.xaxis.set_major_locator( pl.MultipleLocator(base=10.0) )
            ax.xaxis.set_minor_locator( pl.MultipleLocator(base=1.0) )
            ax.yaxis.set_minor_locator( pl.MultipleLocator(base=1.0) )
            ax.xaxis.set_major_formatter( pl.FuncFormatter(glon_formatter) )
            ax.yaxis.set_major_formatter( pl.FuncFormatter(glat_formatter) )
            if i == 1:
                ax.set_xlabel("Galactic longitude ($l$)", fontsize=10)

        pl.ylabel("Galactic latitude ($b$)", fontsize=10)
        pl.gca().yaxis.set_label_coords(-0.06, 1.)
        return fig, patches


############
# FUNCTIONS
############

def vphas_filenames_main(args=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description='Lists the details of a VPHAS+ survey field.')
    parser.add_argument('field', nargs=1, help='Field identifier')
    args = parser.parse_args(args)
    for field in args.field:
        try:
            vo = VphasOffset(field)
            print(json.dumps(vo.get_filenames(), indent=2))
        except Exception as e:
            log.error(e)

def glon_formatter(l, tickno):
    if l < 0:
        l += 360
    return '%0.f°' % l

def glat_formatter(b, tickno):
    return '%.0f°' % b

def vphas_offset_name_generator():
    """Generator function yielding all the names of the VPHAS offsets."""
    for fieldno in np.arange(1, 2284+1):  # 2284 fields
        for offset in ['a', 'b']:
            yield '{0:04d}{1}'.format(fieldno, offset)

def vphas_offset_generator():
    """Generator function yielding all the VphasOffset objects."""
    for name in vphas_offset_name_generator():
        yield VphasOffset(name)


# Dev testing
if __name__ == '__main__':
    fp = VphasFootprint()
    d = fp.offsets
    p = fp.plot().savefig('test.png')
