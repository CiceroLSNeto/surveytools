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
from astropy.table import Column, Table, vstack
from astropy.coordinates import SkyCoord
from astropy.utils.console import ProgressBar

from . import VPHAS_BANDS, STILTS
from .catalogue import DEFAULT_CONFIGFILE


# Which columns do we want to keep in the final tiled catalogues?
RELEASE_COLUMNS = ['RAJ2000', 'DEJ2000', 'sourceID', 'primaryID', 'primary_source',
                   'nObs', 'clean', 'u_g', 'g_r2', 'r_i', 'r_ha']
for band in VPHAS_BANDS:
    for prefix in ['clean_', '', 'err_', 'chi_', 'warning_',
                   'aperMag_', 'aperMagErr_', 'snr_', 'magLim_',
                   'psffwhm_', 'mjd_', 'detectionID_']:
        RELEASE_COLUMNS.append(prefix + band)
        # Hack: add the AB magnitude columns
        if not band == 'ha':
            if prefix == '':
                RELEASE_COLUMNS.append(band + '_AB')
            elif prefix == 'aperMag_':
                RELEASE_COLUMNS.append(prefix + band + '_AB')
for extra in ['field', 'ext', 'l', 'b', 'nbDist']:
    RELEASE_COLUMNS.append(extra)


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

    def __init__(self, l, b, size=0.5, name=None, configfile=None):
        # Init parameters
        if l < 180:
            l += 360
        self.l = l
        self.b = b
        self.size = size
        # The name of the tile will determine the filename
        if name is None:
            if size > 0.999:
                self.name = 'vphas_l{:.0f}_b{:+.0f}'.format(l % 360, b)
            else:
                self.name = 'vphas_{}_{}_{}'.format(l % 360, b, size)
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
        cats = Table.read(os.path.join(self.cfg['vphas']['cat_dir'],
                                       'vphas-offsetcats.fits'))
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
        if len(self.catalogset.table) < 1:
            log.warning('Tile is empty.')
        else:
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
        # We only resolve if the expected output file does not already exists;
        # this way we avoid repeating work for frames on tile edges.
        expected_path = os.path.join(self.cfg['vphas']['resolved_cat_dir'],
                                     refcat_info['filename'].replace('cat', 'resolved'))
        if os.path.exists(expected_path):
            log.debug('Skipping {}: already found {}'.format(refcat_info['filename'],
                                                             expected_path))
            return
        # The frame hasn't been processed yet:
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
        nobs = {}
        for sourceid, photid_candidates in candidates.items():
            # The code below decides the winner of the primaryID fame
            # First, we collect the data on which the decision is based:
            # number of bands in which the source is detected:
            bandcounts = [band_counts[photid] for photid in photid_candidates]
            # number of bands in which the source is clean:
            cleancounts = [clean_counts[photid] for photid in photid_candidates]
            # If a candidate has more bands than the others, it wins,
            # otherwise the candidates with the highest number of clean bands wins.
            bandcount_mask = bandcounts == np.max(bandcounts)
            if bandcount_mask.sum() > 1:
                winner_arg = np.argmax(cleancounts)
            else:
                winner_arg = np.argmax(bandcount_mask)
            myprimary = photid_candidates[winner_arg]
            # Store the winner in the cache for all the candidates
            for photid in photid_candidates:
                # But avoid conflicting a previous decision!
                if photid not in self.primary_id_cache:
                    self.primary_id_cache[photid] = myprimary
            # Keep a record of the number of alternatives available
            nobs[sourceid] = len(photid_candidates)

        self._write_resolved_catalog(refcat_info['filename'], nobs)

    def _get_catalog_data(self, filename):
        """Returns an offset ccd catalogue, augmented with columns needed for seaming."""
        path = os.path.join(self.cfg['catalogue']['destdir'], filename)
        tbl = Table.read(path)
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

    def _write_resolved_catalog(self, fn, nobs):
        path = os.path.join(self.cfg['catalogue']['destdir'], fn)
        tbl = Table.read(path)
        if 'g' not in tbl.columns:
            tbl = self._add_blue_cols(tbl)
        primaryids = np.array([self.primary_id_cache[photid] for photid in tbl['photID']])
        tbl['primaryID'] = Column(primaryids, dtype='14a')
        tbl['primary_source'] = tbl['photID'] == primaryids
        tbl['nObs'] = Column([nobs[photid] for photid in tbl['photID']], dtype='uint8')
        destination = os.path.join(self.cfg['vphas']['resolved_cat_dir'], fn.replace('cat', 'resolved'))
        log.debug('Writing {}'.format(destination))
        # Hack: use sensible types
        # this should have been done in catalogue.py, but due to time pressure
        # (i.e. to avoid re-generating all catalogues) we fix the data types here
        for col in RELEASE_COLUMNS:
            if (col not in ['RAJ2000', 'DEJ2000', 'sourceID', 'primaryID', 'primary_source', 'nObs', 'field', 'ext', 'l', 'b', 'nbDist']
                and not col.startswith('mjd')
                and not col.startswith('detectionID') 
                and not col.startswith('clean')
                and not col.startswith('warning')
                and not col.endswith('AB')):
                tbl.columns[col] = tbl[col].astype('float32')
        for band in VPHAS_BANDS:
            tbl.columns['detectionID_' + band] = tbl['detectionID_' + band].astype('23a')
            tbl.columns['warning_' + band] = tbl['error_' + band].astype('12a')
            # Hack: add the AB columns
            tbl[band + '_AB'] = tbl[band]
            tbl['aperMag_' + band + '_AB'] = tbl['aperMag_' + band]
        tbl.columns['sourceID'] = tbl['photID'].astype('14a')
        tbl.columns['field'] = tbl['field'].astype('5a')
        tbl.columns['ext'] = tbl['ccd'].astype('uint8')
        tbl.columns['nbDist'] = tbl['nndist'].astype('float32')
        # Hack: last-minute changes to comply with the ESO standard
        tbl['ra'].name = 'RAJ2000'
        tbl['dec'].name = 'DEJ2000'
        # Finally, write to disk
        tbl = Table(tbl, copy=False)  # necessary!
        # file may have been written by another process in meanwhile
        if not os.path.exists(destination):
            try:
                tbl[RELEASE_COLUMNS].write(destination)
            except OSError:
                pass  # dont let the script crash if the file was written by a competing process
        else:
            log.warning('Not overwriting ' + destination)
        
    def _add_blue_cols(self, tbl):
        """Add empty columns for the Ugr data."""
        from astropy.table import MaskedColumn
        # Prepare the column arrays
        col_nan = np.array([np.nan for i in range(len(tbl))])
        col_nullbyte = np.array(['\x00' for i in range(len(tbl))])
        col_false = np.array([False for i in range(len(tbl))])
        col_errormsg = np.array(['No_blue_data' for i in range(len(tbl))])
        # Now add identical empty columns for each band
        for band in ['u', 'g', 'r2']:
            for colname in ['detectionID_', '', 'err_', 'chi_',
                            'sharpness_', 'sky_', 'error_', 'aperMag_',
                            'aperMagErr_', 'snr_', 'magLim_', 'psffwhm_',
                            'airmass_', 'mjd_', 'pixelShift_', 'clean_',
                            'offsetRa_', 'offsetDec_']:
                if colname == 'clean_':
                    mydtype = 'bool'
                    mydata = col_false
                elif colname == 'detectionID_':
                    mydtype = '23a'
                    mydata = col_nullbyte
                elif colname == 'warning_':
                    mydtype = '12a'
                    mydata = col_errormsg
                elif colname == 'mjd_':
                    mydtype = 'float64'
                    mydata = col_nan
                else:
                    mydtype = 'float32'
                    mydata = col_nan
                # Note that mask=false, because we want to impose our own choice of missing values
                # to be written to the FITS files, rather than depending on astropy's behaviour here
                tbl[colname+band] = MaskedColumn(mydata, mask=col_false, dtype=mydtype)
        # Also add the composite colour columns
        tbl['u_g'] = MaskedColumn(col_nan, mask=col_false, dtype='float32')
        tbl['g_r2'] = MaskedColumn(col_nan, mask=col_false, dtype='float32')
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
        ocmd = """'select "l >= {l_min} & l < {l_max} & b >= {b_min} & b < {b_max}"; \
                   addcol -before RAJ2000 \
                          -desc "Source designation (JHHMMSS.ss+DDMMSS.s)." \
                          name \
                          "concat(\\"VPHASDR2 J\\", 
                                            replaceAll(degreesToHms(RAJ2000, 1),
                                                       \\":\\", \\"\\"), 
                                            replaceAll(degreesToDms(DEJ2000, 1),
                                                       \\":\\", \\"\\")
                                            )"; \
                  sort name;'""".format(**param)
        destination = os.path.join(self.cfg['vphas']['tiled_cat_dir'],
                                   '{}.fits'.format(self.name))
        log.info('Writing {}'.format(destination))
        cmd = 'sleep 2; {} tcat {} countrows=true lazy=true ocmd={} ofmt=fits-basic out={}'.format(STILTS, instring, ocmd, destination)
        #ofmt=colfits-basic
        log.info(cmd)
        status = os.system(cmd)
        log.info('concat: '+str(status))


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

