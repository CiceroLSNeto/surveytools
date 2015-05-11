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

The above command is sufficient for most features.
For the catalogue-generating module, however,
IRAF needs to be installed and the environment variables
`iraf` and `IRAFARCH` need to be set.
On 64-bit Linux, the following sequence of commands
installs and prepares IRAF v216 for use with surveytools:
```
mkdir ~/bin/iraf
cd ~/bin/iraf
wget ftp://iraf.noao.edu/iraf/v216/PCIX/iraf.lnux.x86_64.tar.gz
tar -xf iraf.lnux.x86_64.tar.gz
export iraf='~/bin/iraf'
export IRAFARCH='linux64'
```
To prevent IRAF from making a "caching" directory in whatever
directory you run it from, it is recommended that you initialize an `iraf` directory in your home dir: (optional)
```
mkdir ~/iraf
cd ~/iraf
~/bin/iraf/unix/hlib/mkiraf.sh
```

### Dependencies

```
astropy 1.0
photutils 0.1
```

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



### Contributing

To report bugs and request features, please use the [issue tracker](https://github.com/barentsen/surveytools/issues).


### License

Copyright 2015 Geert Barentsen.

SurveyTools is free software made available under the MIT License. For details see the LICENSE file.
