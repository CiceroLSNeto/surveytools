# SurveyTools

***Aids data analysis for VST/OmegaCAM surveys.***

SurveyTools is a Python library to aid the data processing and analysis
of wide-area surveys carried out using ESO's VST/OmegaCAM instrument.
The library is focused entirely on the VPHAS+ survey of the Galactic Plane,
but may contain useful tools for other VST users as well.


### Installation

If you have a working installation of Python on your system,
you can install SurveyTools using pip:
```
pip install git+https://github.com/barentsen/surveytools
```
Alternatively, you can clone the repository and install from source:
```
$ git clone https://github.com/barentsen/surveytools.git
$ cd surveytools
$ python setup.py install
```
Note that SurveyTools has only been tested in Linux at present.


### Using SurveyTools

Creating a pretty *pawprint plot* which combines the data from all of
OmegaCAM's 32 CCDs into a single quicklook JPG:
```
vst-pawplot -o plot.jpg omegacam-exposure.fits
```

Creating a beautiful colour jpg from CCD #10 in VPHAS offset '1835a':
```
vphas-quicklook 1835a --ccd 10
```

Creating a multi-band catalogue of PSF photometry for VPHAS
offset "0149a":
```Python
from surveytools impot VphasOffsetCatalogue
catalogue = VphasOffsetCatalogue('0149a').create_catalogue()
catalogue.write('mycatalogue.fits')
```
Note that the catalogue-generating software requires a working version of `pyraf` and `iraf` to be installed.


### Contributing

To report bugs and request features, please use the [issue tracker](https://github.com/barentsen/surveytools/issues).


### License

Copyright 2015 Geert Barentsen.

SurveyTools is free software made available under the MIT License. For details see the LICENSE file.
