VphasTools
==========
A Python library used to create data products for the VPHAS+ photometric
survey of the Galactic Plane.

Example use
-----------
Creating a multi-band catalogue of PSF photometry for a VPHAS pointing:
```Python
import vphas
pointing = vphas.VphasPointing('0149a')
pointing.create_catalogue().write('mycatalogue.fits')
```

Dependencies
------------
* `astropy` v1.0
* `astropy-photutils` v0.1
* `pyraf` (requires a local installation of IRAF)
