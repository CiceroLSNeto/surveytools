"""Tools to merge single-ccd catalogs into the final catalogue.

Usage
-----
To produce a catalogue tile covering galactic latitudes
[270, 270.5] and longitudes [0, 0.5], use:
```
$ vphas-tile-merge 270 0 -s 0.5
```
"""
import os
import configparser

import numpy as np
from shapely.geometry.polygon import LinearRing, Polygon

from astropy import log
from astropy import units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.utils.console import ProgressBar

from . import VPHAS_BANDS, STILTS
from .catalogue import DEFAULT_CONFIGFILE


# Which columns do we want to keep in the final tiled catalogues?
RELEASE_COLUMNS = ['ra', 'dec', 'u_g', 'g_r', 'r_i', 'r_ha',
                   'photID', 'primaryID', 'is_primary']
for band in VPHAS_BANDS:
    for prefix in ['clean_', '', 'err_', 'chi_', 'error_',
                   'aperMag_', 'aperMagErr_', 'snr_', 'magLim_',
                   'psffwhm_', 'mjd_', 'x_', 'y_', 'detectionID_']:
        RELEASE_COLUMNS.append(prefix+band)


class VphasCatalogSet(object):
    """Representation of a set of catalogs, from which spatial subsets may be
    selected."""

    def __init__(self, table):
        self.table = table

    def subset_galactic(self, l_min, l_max, b_min, b_max):
        poly = Polygon(LinearRing([(l_min, b_min),
                                   (l_min, b_max),
                                   (l_max, b_max),
                                   (l_max, b_min)]))
        mask = np.array(
                    [poly.intersects(self.table['polygon_gal'][i])
                     for i in range(len(self.table))]
                    )
        log.debug('{} catalogs intersect.'.format(mask.sum()))
        return VphasCatalogSet(self.table[mask])

    def subset(self, ra_min, ra_max, dec_min, dec_max):
        poly = Polygon(LinearRing([(ra_min, dec_min),
                                   (ra_min, dec_max),
                                   (ra_max, dec_max),
                                   (ra_max, dec_min)]))
        mask = np.array(
                    [poly.intersects(self.table['polygon'][i])
                     for i in range(len(self.table))]
                    )
        log.debug('{} catalogs intersect.'.format(mask.sum()))
        return VphasCatalogSet(self.table[mask])



class VphasCatalogTile(object):

    def __init__(self, l, b, size=0.5, configfile=None):
        # Init parameters
        if l < 180:
            l += 360
        self.l = l
        self.b = b
        self.size = size
        self.name = 'vphas-{}-{}-{}'.format(l, b, size)
        # Read the configuration
        if configfile is None:
            configfile = DEFAULT_CONFIGFILE
        self.cfg = configparser.ConfigParser()
        self.cfg.read(configfile)
        # Init the set of catalogue and cache
        self.catalogset = self._get_all_catalogs().subset_galactic(l, l+size, b, b+size)
        self.primary_id_cache = {}

    def _get_all_catalogs(self):
        """
        Note: we use galactic longitudes in the range [180, 540] to avoid
        edge issues.
        """
        cats = Table.read(os.path.join(self.cfg['vphas']['cat_dir'], 'vphas-offsetcat-index.fits'))
        # Add the min/max range in galactic latitude and longitude
        galcrd = [SkyCoord(cats[ra_col], cats[dec_col], unit='deg').galactic
                  for ra_col in ['ra_min', 'ra_max']
                  for dec_col in ['dec_min', 'dec_max']]
        longitudes = np.array([c.l for c in galcrd])
        # Ensure longitudes are in the range [180, 540]
        longitudes[longitudes < 180] += 360.
        latitudes = np.array([c.b for c in galcrd])
        cats['l_min'] = np.min(longitudes, axis=0)
        cats['l_max'] = np.max(longitudes, axis=0)
        cats['b_min'] = np.min(latitudes, axis=0)
        cats['b_max'] = np.max(latitudes, axis=0)
        cats['polygon'] = np.array([
                                Polygon(
                                    LinearRing([(cats['ra_min'][i], cats['dec_min'][i]),
                                                (cats['ra_min'][i], cats['dec_max'][i]),
                                                (cats['ra_max'][i], cats['dec_max'][i]),
                                                (cats['ra_max'][i], cats['dec_min'][i])])
                                        )
                                     for i in range(len(cats))], dtype='O')
        cats['polygon_gal'] = np.array([Polygon(LinearRing([(cats['l_min'][i], cats['b_min'][i]),
                                                     (cats['l_min'][i], cats['b_max'][i]),
                                                     (cats['l_max'][i], cats['b_max'][i]),
                                                     (cats['l_max'][i], cats['b_min'][i])]))
                                     for i in range(len(cats))], dtype='O')
        return VphasCatalogSet(cats)


    def create(self):
        self.resolve()
        self.concatenate()

    def resolve(self):
        # Ensure the output dir exists
        destination_dir = self.cfg['vphas']['resolved_cat_dir']
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        # Start resolving, with fields ordered by seeing
        order = np.argsort(self.catalogset.table['psffwhm_i'])
        log.info('Resolving duplicate detections.')
        for idx in ProgressBar(order):
            self._resolve_one(idx)

    def _resolve_one(self, idx, max_distance=0.5):
        refcat_info = self.catalogset.table[idx]
        log.debug('Now resolving {}'.format(refcat_info['filename']))
        overlap_set = self.catalogset.subset(refcat_info['ra_min'], refcat_info['ra_max'],
                                             refcat_info['dec_min'], refcat_info['dec_max'])
        overlap_fns = [c['filename'] for c in overlap_set.table if c['filename'] != refcat_info['filename']]
        log.debug('{} catalogs overlap: {}'.format(len(overlap_fns), overlap_fns))

        # Load the reference catalog
        refcat = self._get_catalog_data(refcat_info['filename'])
        log.debug('{} contains {} rows.'.format(refcat_info['filename'], len(refcat)))

        # Identify the alternative detections in other catalogs,
        # and store the result in the candidates dictionary
        candidates = dict(zip(refcat['photID'],
                           [[photid] for photid in refcat['photID']]))
        # Also store metadata for the candidate ids
        band_counts = dict(zip(refcat['photID'], refcat['band_count']))
        clean_counts = dict(zip(refcat['photID'], refcat['clean_count']))
        # Crossmatch each overlapping cat
        for fn in overlap_fns:
            overlap_cat = self._get_catalog_data(fn)
            idx, sep2d, dist3d = refcat['crd'].match_to_catalog_sky(overlap_cat['crd'])
            mask_match = sep2d < (max_distance * u.arcsec)
            log.debug('=> {} provides {} matches.'.format(fn, mask_match.sum()))
            for j in np.where(mask_match)[0]:
                candidates[refcat['photID'][j]].append(overlap_cat['photID'][idx[j]])

            band_counts.update(zip(overlap_cat['photID'], overlap_cat['band_count']))
            clean_counts.update(zip(overlap_cat['photID'], overlap_cat['clean_count']))

        # Now determine the primaryIDs
        for sourceid, photid_candidates in candidates.items():
            bandcounts = [band_counts[photid] for photid in photid_candidates]
            cleancounts = [clean_counts[photid] for photid in photid_candidates]

            bandcount_mask = bandcounts == np.max(bandcounts)
            if bandcount_mask.sum() > 1:
                winner_arg = np.argmax(cleancounts)
            else:
                winner_arg = np.argmax(bandcount_mask)
            myprimary = photid_candidates[winner_arg]
           
            for photid in photid_candidates:
                if photid not in self.primary_id_cache:
                    self.primary_id_cache[photid] = myprimary

        self._write_resolved_catalog(refcat_info['filename'])

    def _get_catalog_data(self, filename):
        """Returns an offset ccd catalogue, augmented with columns needed for seaming."""
        path = os.path.join(self.cfg['catalogue']['destdir'], filename)
        tbl = Table.read(path)
        tbl = add_photid(tbl, filename)
        tbl['crd'] = SkyCoord(tbl['ra'], tbl['dec'], unit='deg', frame='icrs')
        # Add the "clean_count" column which details the number of bands with clean photometry
        # Add the "band_count" column which details the number of bands with psf photometry
        clean_count = np.zeros(len(tbl), dtype=int)
        band_count = np.zeros(len(tbl), dtype=int)
        for band in ['u', 'g', 'r2', 'r', 'i', 'ha']:
            try:
                clean_count[tbl['clean_' + band]] += 1
                band_count[~np.isnan(tbl[band])] += 1
            except KeyError:
                pass  # band missing
        tbl['clean_count'] = clean_count
        tbl['band_count'] = band_count
        return tbl

    def _write_resolved_catalog(self, fn):
        path = os.path.join(self.cfg['catalogue']['destdir'], fn)
        tbl = Table.read(path)
        tbl = add_photid(tbl, fn)
        tbl = self._add_blue_cols(tbl)
        tbl['primaryID'] = [self.primary_id_cache[photid] for photid in tbl['photID']]
        tbl['is_primary'] = tbl['photID'] == tbl['primaryID']
        destination = os.path.join(self.cfg['vphas']['resolved_cat_dir'], fn.replace('cat', 'resolved'))
        log.debug('Writing {}'.format(destination))
        tbl[RELEASE_COLUMNS].write(destination, overwrite=True)

    def _add_blue_cols(self, tbl):
        from astropy.table import MaskedColumn
        col_nan = np.array([np.nan for i in range(len(tbl))])
        col_false = np.array([False for i in range(len(tbl))])
        col_nullbyte = np.array(['\x00' for i in range(len(tbl))])
        
        for band in ['u', 'g', 'r2']:
            for colname in ['detectionID_', 'x_', 'y_', '', 'err_', 'chi_',
                            'sharpness_', 'sky_', 'error_', 'aperMag_',
                            'aperMagErr_', 'snr_', 'magLim_', 'psffwhm_',
                            'airmass_', 'mjd_', 'pixelShift_', 'clean_',
                            'offsetRa_', 'offsetDec_']:
                if colname == 'clean_':
                    mydtype = 'bool'
                    mydata = col_false
                elif colname in ['detectionID_', 'error_']:
                    mydtype = 'str'
                    mydata = col_nullbyte
                else:
                    mydtype = 'double'
                    mydata = col_nan
                tbl[colname+band] = MaskedColumn(mydata, mask=col_false, dtype=mydtype)
        tbl['u_g'] = col_nan
        tbl['g_r'] = col_nan
        return tbl

    def concatenate(self):
        """Concatenate and trim the catalogues that cover the subset."""
        # Ensure the output dir exists
        destination_dir = self.cfg['vphas']['tiled_cat_dir']
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        instring = ''
        for fn in self.catalogset.table['filename']:
            fnres = os.path.join(self.cfg['vphas']['resolved_cat_dir'], fn.replace('cat', 'resolved'))
            instring += 'in={0} '.format(fnres)
        param = {'l_min': self.l % 360, 'l_max': (self.l + self.size) % 360,
                 'b_min': self.b, 'b_max': self.b + self.size,}
        ocmd = """'addskycoords -inunit deg -outunit deg icrs galactic ra dec l b; \
                  select "l >= {l_min} & l < {l_max} & b >= {b_min} & b <= {b_max}"; \
                  sort primaryID;'""".format(**param)
        destination = os.path.join(self.cfg['vphas']['tiled_cat_dir'],
                                   '{}.fits'.format(self.name))
        log.info('Writing {}'.format(destination))
        cmd = '{} tcat {} countrows=true lazy=true ocmd={} ofmt=fits out={}'.format(STILTS, instring, ocmd, destination)
        #ofmt=colfits-basic
        log.info(cmd)
        status = os.system(cmd)
        log.info('concat: '+str(status))


def add_photid(tbl, filename):
    sourceid_prefix = filename.rsplit('-', maxsplit=1)[0]
    tbl['photID'] = ['{}-{}'.format(sourceid_prefix, detid.rsplit('-', maxsplit=1)[1])
                     for detid in tbl['detectionID_i']]
    return tbl



def vphas_tile_merge_main(args=None):
    """Command-line interface to merge frame catalogues into a tile."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge frame catalogues into a tile.')
    parser.add_argument('-s', '--size', metavar='size',
                        type=float, default=1,
                        help='Width and height of the tile in degrees.')
    parser.add_argument('-c', '--config', metavar='configfile',
                        type=str, default=None,
                        help='Configuration file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on debug output.')
    parser.add_argument('l', type=float, help='Galactic longitude.')
    parser.add_argument('b', type=float, help='Galactic latitude.')
    args = parser.parse_args(args)

    if args.verbose:
        log.setLevel('DEBUG')
    else:
        log.setLevel('INFO')

    tile = VphasCatalogTile(args.l, args.b, args.size, configfile=args.config)
    tile.create()


if __name__ == '__main__':
    tile = VphasCatalogTile(262.1, -1.2, 0.1)
    tile.create()

