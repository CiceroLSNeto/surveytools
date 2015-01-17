VphasTools
==========
A Python library used to create data products for the VPHAS+ photometric
survey of the Galactic Plane.

Example use
-----------
Creating a multi-band PSF photometry catalogue from single-CCD data:

```
from vphas import VphasFrameCatalogue
cat = VphasFrameCatalogue('vphas_0149a', ccd=8).create_catalogue()
```

Dependencies
------------
* `astropy` v1.0
* `astropy-photutils` v0.1
* `pyraf` (requires a local installation of IRAF)
