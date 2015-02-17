from astropy import log

from surveytools.footprint import VphasExposure

if __name__ == '__main__':
    e = VphasExposure('o20120214_00107.fit')
    tbl = e.frames()
    log.info(tbl)