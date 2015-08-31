"""Creates a plot showing the VPHAS footprint status."""
import matplotlib
matplotlib.use('Agg')
from surveytools import footprint
from astropy.table import Table
from astropy import log
from astropy.utils.console import ProgressBar
import matplotlib.pyplot as pl


INPUT_FN = "vphas_pos_status.dat"
OUTPUT_FN = 'vphas-footprint-sep2015.pdf'
DPI = 150


if __name__ == '__main__':
    status = Table.read(INPUT_FN, format='ascii')
    mask_finished = (status['Hari'] == 'true') & (status['ugr'] == 'true')
    mask_red = (status['Hari'] == 'true') & (status['ugr'] != 'true')
    mask_blue = (status['Hari'] != 'true') & (status['ugr'] == 'true')

    fp = footprint.VphasFootprint()

    # Fields traded in 2015
    SKIP = ["2050", "2051", "2052", "2053", "2054", "2055", "2056", "2057",
            "2263", "2268", "1408", "1441", "1475", "1510", "1546"]

    myoffsets = {}
    for offset in ProgressBar(fp.offsets):
        if offset[0:4] in SKIP:
            continue
        if offset.endswith('a'):
            myoffsets[offset] = fp.offsets[offset]
            longname = 'vphas_' + offset[0:4]
            if longname in status['Field'][mask_finished]:
                myoffsets[offset]['facecolor'] = "#4daf4a"
                myoffsets[offset]['edgecolor'] = "#222222"
                myoffsets[offset]['zorder'] = 100
                myoffsets[offset]['label'] = 'finished'
            elif longname in status['Field'][mask_red]:
                myoffsets[offset]['facecolor'] = "#e41a1c"
                myoffsets[offset]['edgecolor'] = "#222222"
                myoffsets[offset]['zorder'] = 90
                myoffsets[offset]['label'] = 'red'
            elif longname in status['Field'][mask_blue]:
                myoffsets[offset]['facecolor'] = "#377eb8"
                myoffsets[offset]['edgecolor'] = "#222222"
                myoffsets[offset]['zorder'] = 90
                myoffsets[offset]['label'] = 'blue'
            else:
                myoffsets[offset]['facecolor'] = "#eeeeee"
                myoffsets[offset]['edgecolor'] = "#aaaaaa"
                myoffsets[offset]['zorder'] = -10
                myoffsets[offset]['label'] = 'planned'        

    fig, patches = fp.plot(offsets=myoffsets)

    fig.axes[0].legend((
                        patches['finished'][0],
                        patches['red'][0],
                        patches['blue'][0],
                        patches['planned'][0]
                        ),
                       (
                        u'u,g,r and Hα,r,i observed',
                        u'Hα,r,i observed',
                        'u,g,r observed',
                        'future'
                        ),
                       fontsize=10,
                       bbox_to_anchor=(0., 1.1, 1., .102),
                       loc=3,
                       ncol=5,
                       borderaxespad=0.,
                       handlelength=0.8,
                       frameon=False)

    log.info("Writing {}".format(OUTPUT_FN))
    fig.savefig(OUTPUT_FN, dpi=DPI)
    pl.close(fig)
