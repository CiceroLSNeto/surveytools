import os
SURVEYTOOLS_PATH = os.path.abspath(os.path.dirname(__file__))
SURVEYTOOLS_DATA = os.path.join(SURVEYTOOLS_PATH, 'data')

if os.environ.get('DISPLAY') is None:
    import matplotlib
    matplotlib.use('Agg')

from catalogue import VphasFrame, VphasFrameCatalogue, VphasOffsetPointing
