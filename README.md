SurveyTools
===========
A Python library to aid data processing and analysis of wide-area surveys,
focused entirely on the VPHAS+ photometric survey of the Galactic Plane
for now.

Example use
-----------
Create a pretty pawprint plot of an exposure obtained with the VST/OmegaCAM
instrument, which visualized all 32 CCDs in a single JPG:
```
vst-pawplot -o plot.jpg omegacam-exposure.fits
```

Creating a multi-band catalogue of PSF photometry for a VPHAS pointing:
```Python
import surveytools
pointing = surveytools.VphasPointing('0149a')
pointing.create_catalogue().write('mycatalogue.fits')
```

Dependencies
------------
* `astropy` v1.0
* `astropy-photutils` v0.1
* `pyraf` (requires a local installation of IRAF; only needed for daophot use)
* `progressbar`
