#!/usr/bin/env python
from setuptools import setup

entry_points = {'console_scripts': [
    'vst-pawplot = surveytools.quicklook:vst_pawplot_main'
]}

setup(name='surveytools',
      version='0.1',
      description='Tools for the VPHAS+ astronomy survey.',
      author='Geert Barentsen',
      license='MIT',
      url='http://www.vphas.eu',
      packages=['surveytools'],
      entry_points=entry_points,
      )
