TODO
====
General
~~~~~~~
* What's wrong with o20120819_00066.fit ?

Offset catalogues
~~~~~~~~~~~~~~~~~
* Give proper error messages in the catalogue, e.g. "Chi score too high".
* Verify a10 completeness.
* Produce diagnostic ccd/cmd plots.
* Check if the photometry for stars at the edges of masked-out areas reliable?
* msky vs mstar diagnostic plot (no relationship expected)
* mstar vs chi diagnostic plot (no relationship expected)
* Add nearest neighbour distance column?
* Take care of both aperture correction & calibration using APASS.
* Add a "write debug output" config flag to avoid plots in batch mode.
* Update the csv file with coordinates to reflect the changed P95 fields?
* Early fields need to use the "c" pointing for H-alpha.

Catalogue seaming
~~~~~~~~~~~~~~~~~
* Ignore CCDs with gain variations? (Study until when they occured.)
CCD's affected: chip ESO_CCD_82 and ESO_CCD_92

Quicklook
~~~~~~~~~
* Why does the quicklook for 0004b not work? There is some overlap.
* How to get rid of all the green frames?