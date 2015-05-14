#!/usr/bin/env python
from setuptools import setup

entry_points = {'console_scripts': [
    'vst-pawplot = surveytools.quicklook:vst_pawplot_main',
    'vphas-quicklook = surveytools.quicklook:vphas_quicklook_main',
    'vphas-filenames = surveytools.footprint:vphas_filenames_main',
    'vphas-offset-catalogue = surveytools.catalogue:vphas_offset_catalogue_main',
    'vphas-index-offset-catalogues = surveytools.catalogue:vphas_index_offset_catalogues_main'
]}

setup(name='surveytools',
      version='0.1',
      description='Tools for the VPHAS+ astronomy survey.',
      author='Geert Barentsen',
      license='MIT',
      url='http://www.vphas.eu',
      packages=['surveytools'],
      install_requires=['numpy',
                        'matplotlib',
                        'imageio>=1.0',
                        'astropy',
                        'photutils',
                        'pyraf'],
      entry_points=entry_points,
      )
