"""Defines the VPHAS survey.

Example use
-----------
fp = VphasFootprint()
fp.get_field_dict()  # returns {'0001a': SkyCoord(), ...}
fp.get_field_corners(fieldname)
fp.get_plot()
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
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='(.*)did not parse '
                                                      'as fits unit(.*)')
            metadata = Table.read(os.path.join(SURVEYTOOLS_DATA,
                                               'list-hari-image-files.fits'))
        fieldname = 'vphas_' + self.name[:-1]
        # Has the field been observed?
        if (metadata['Field_1'] == fieldname).sum() == 0:
            raise NotObservedException('{0} has not been observed in the red filters'.format(self.name))
        offset2idx = {'a': 0, 'b': -1, 'c': 1}
        idx = offset2idx[self.name[-1:]]
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
            metadata = Table.read(os.path.join(SURVEYTOOLS_DATA, 'list-ugr-image-files.fits'))
        fieldname = 'vphas_' + self.name[:-1]
        # Has the field been observed?
        if (metadata['Field_1'] == fieldname).sum() == 0:
            raise NotObservedException('{0} has not been observed in the blue filters'.format(self.name))
        offset2idx = {'a': 0, 'b': -1, 'c': 1}
        idx = offset2idx[self.name[-1:]]
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
            result[bandname] = filenames[idx]
        return result

    def metadata(self):
        fn = self.get_filenames()
        meta = {}
        for band in fn:
            f = fits.open(os.path.join(VPHAS_DATA_PATH, fn[band]))
            meta[band+'_filename'] = fn[band]
            meta[band+'_ra'] = f[0].header['RA']
            meta[band+'_dec'] = f[0].header['DEC']
        return meta


class VphasFootprint():

    def __init__(self):
        pass

    @timed
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
        cosdec = np.cos(np.radians(tbl['dec']))
        cntr = {'ra_a': tbl['ra'],
                'dec_a': tbl['dec'],
                'ra_b': tbl['ra'] - (588/3600.) * cosdec,
                'dec_b': tbl['dec'] + (660/3600.),
                'ra_c': tbl['ra'] - (300/3600.) * np.cos(tbl['dec']),
                'dec_c': tbl['dec'] + (350/3600.),
                }
        for pos in ['a', 'b', 'c']:  # Convert to Galactic
            cntr['l_'+pos], cntr['b_'+pos] = self._icrs2gal(cntr['ra_'+pos], cntr['dec_'+pos])
        # Compute the corners of the field in ICRS and galactic coordinates
        # this is very ugly because it is optimized for speed
        cornershifts = [[-0.5,-0.5], [-0.5,+0.5], [+0.5, +0.5], [+0.5, -0.5], [-0.5,-0.5]]
        polygons = {}
        polygons_gal = {}
        for pos in ['a', 'b', 'c']:
            corners_ra, corners_dec = [], []
            corners_l, corners_b = [], []
            for idx, shift in enumerate(cornershifts):
                ra = cntr['ra_a'] + shift[0] / cosdec
                dec = cntr['dec_a'] + shift[1]
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
    def plot(self):
        """
        Parameters
        -----------
        highlight : list of tuples
            each tuple should be composed of (fieldlist, colour, title), e.g.
            [(['0001a', '0002a'], '#e41a1c', 'u,g,r observed')]
        """
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

            for name in self.offsets.keys():
                if name.endswith('a'):
                    poly = mpl.patches.Polygon(self.offsets[name]['polygon_gal'],
                                                      alpha=1,
                                                      facecolor="#dddddd", 
                                                      edgecolor="#999999",
                                                      zorder=-100)
                    ax.add_patch(poly)
            """
            for fieldno, f in enumerate(fields):
                if (f[0][0] < xstart[i]-5 ) or (f[0][0] > xstart[i]+stepsize+5 ):
                    continue
                
                if finished_mask[fieldno]:
                    poly = matplotlib.patches.Polygon(f, alpha=1,
                                                      facecolor="#4daf4a", 
                                                      edgecolor="#222222")
                    poly_finished.append(poly)
                elif hari_mask[fieldno]:
                    poly = matplotlib.patches.Polygon(f, alpha=1,
                                                      facecolor="#e41a1c",
                                                      edgecolor="#222222")
                    poly_hari.append(poly)
                elif ugr_mask[fieldno]:
                    poly = matplotlib.patches.Polygon(f, alpha=1,
                                                      facecolor="#377eb8",
                                                      edgecolor="#222222")
                    poly_ugr.append(poly)
                else:
                    poly = matplotlib.patches.Polygon(f, alpha=1.,
                                                      facecolor="#dddddd", 
                                                      edgecolor="#999999",
                                                      zorder=-100)
                    poly_notstarted.append(poly)
                ax.add_patch(poly) 
            """
            glon1 = xstart[i]
            glon2 = glon1+stepsize
            
            ax.set_xlim(glon1+stepsize, glon1)
            ax.set_ylim([glat1, glat2])

            ax.xaxis.set_major_locator( pl.MultipleLocator(base=10.0) )
            ax.xaxis.set_minor_locator( pl.MultipleLocator(base=1.0) )
            ax.yaxis.set_minor_locator( pl.MultipleLocator(base=1.0) )
            ax.xaxis.set_major_formatter( pl.FuncFormatter(glon_formatter) )
            ax.yaxis.set_major_formatter( pl.FuncFormatter(glat_formatter) )
            """
            if i == 0:
                legend((poly_finished[0], poly_ugr[0], poly_hari[0], poly_notstarted[0]),
                       ('u,g,r and H$\\mathrm{\\alpha}$,r,i observed',
                        'u,g,r observed',
                        'H$\\textrm{\\alpha}$,r,i observed',
                        'Awaiting observation'),
                       fontsize=9,
                       bbox_to_anchor=(0., 1.1, 1., .102),
                       loc=3,
                       ncol=4,
                       borderaxespad=0.,
                       handlelength=0.8,
                       frameon=False )
            """
            if i == 1:
                ax.set_xlabel("Galactic longitude ($l$)", fontsize=11)


        pl.ylabel("Galactic latitude ($b$)", fontsize=11)
        pl.gca().yaxis.set_label_coords(-0.06, 1.)
        return fig


def glon_formatter(l, tickno):
    if l < 0:
        l += 360
    return '$%0.f^\circ$' % l

def glat_formatter(b, tickno):
    return '$%.0f^\circ$' % b


if __name__ == '__main__':
    fp = VphasFootprint()
    #d = fp.get_field_dict()
    d = fp.offsets
    p = fp.plot().savefig('test.png')

"""
# Load the VPHAS field positions and their status
tbl = Table.read('vphas_pos_status.dat.20141030', format='ascii')
ra, dec = tbl['RA'], tbl['Dec']
finished_mask = tbl['finished'] == 'true'
hari_mask = (tbl['Hari'] == 'true') & -finished_mask
ugr_mask = (tbl['ugr'] == 'true') & -finished_mask


def glon_formatter(l, tickno):
    if l < 0:
        l += 360
    return '$%0.f^\circ$' % l

def glat_formatter(b, tickno):
    return '$%.0f^\circ$' % b


fields = []

for (myra, mydec) in zip(ra, dec):
    cd = cos(radians(mydec))
    
    corners_ra = [myra-0.5/cd, myra-0.5/cd, myra+0.5/cd, myra+0.5/cd, myra-0.5/cd]
    corners_dec = [mydec-0.5, mydec+0.5, mydec+0.5, mydec-0.5, mydec-0.5]

    icrs = coordinates.ICRS(ra=corners_ra, dec=corners_dec, unit=('deg','deg'))
    field = [[c.l.to(u.deg).value, c.b.to(u.deg).value] for c in icrs.galactic]
    fields.append(field)

fields = np.array(fields)
fields[fields > 180] -= 360

fig = figure(figsize=(7.5, 4)) #7,5
subplots_adjust(0.08, 0.04, 0.96, 0.99, wspace=0.15, hspace=0.15)


poly_finished, poly_hari, poly_ugr = [], [], []
poly_notstarted = []

glat1 = -11
glat2 = 11
stepsize = 100
xstart = [-60, -160]
for i in range(len(xstart)):
    ax = subplot(2.0, 1, i+1)
    subplots_adjust(left=0.09, bottom=0.12, 
                    right=0.98, top=0.87, 
                    hspace=0.2, wspace=0.2)    
    
    for fieldno, f in enumerate(fields):
        if (f[0][0] < xstart[i]-5 ) or (f[0][0] > xstart[i]+stepsize+5 ):
            continue
        
        if finished_mask[fieldno]:
            poly = matplotlib.patches.Polygon(f, alpha=1,
                                              facecolor="#4daf4a", 
                                              edgecolor="#222222")
            poly_finished.append(poly)
        elif hari_mask[fieldno]:
            poly = matplotlib.patches.Polygon(f, alpha=1,
                                              facecolor="#e41a1c",
                                              edgecolor="#222222")
            poly_hari.append(poly)
        elif ugr_mask[fieldno]:
            poly = matplotlib.patches.Polygon(f, alpha=1,
                                              facecolor="#377eb8",
                                              edgecolor="#222222")
            poly_ugr.append(poly)
        else:
            poly = matplotlib.patches.Polygon(f, alpha=1.,
                                              facecolor="#dddddd", 
                                              edgecolor="#999999",
                                              zorder=-100)
            poly_notstarted.append(poly)
        ax.add_patch(poly) 

    glon1 = xstart[i]
    glon2 = glon1+stepsize
    
    xlim(glon1+stepsize, glon1)
    ylim([glat1,glat2])
    

    ax.xaxis.set_major_locator( MultipleLocator(base=10.0) )
    
    ax.xaxis.set_minor_locator( MultipleLocator(base=1.0) )
    ax.yaxis.set_minor_locator( MultipleLocator(base=1.0) )

    ax.xaxis.set_major_formatter( plt.FuncFormatter(glon_formatter) )
    ax.yaxis.set_major_formatter( plt.FuncFormatter(glat_formatter) )

    if i == 0:
        legend((poly_finished[0], poly_ugr[0], poly_hari[0], poly_notstarted[0]),
               ('u,g,r and H$\\mathrm{\\alpha}$,r,i observed',
                'u,g,r observed',
                'H$\\textrm{\\alpha}$,r,i observed',
                'Awaiting observation'),
               fontsize=9,
               bbox_to_anchor=(0., 1.1, 1., .102),
               loc=3,
               ncol=4,
               borderaxespad=0.,
               handlelength=0.8,
               frameon=False )
    
    if i == 1:
        xlabel("Galactic longitude ($l$)", fontsize=11)


ylabel("Galactic latitude ($b$)", fontsize=11)
gca().yaxis.set_label_coords(-0.06, 1.)

#plt.tight_layout()
#interactive(True)
#show()
savefig("vphas-observed.pdf")
close()
"""