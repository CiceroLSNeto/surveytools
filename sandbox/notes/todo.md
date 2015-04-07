TODO
====

Catalogue
---------
* What sigma to use for i-band detection step? 2 seems better for sparse; 1 for dense? (review sigma estimate?)
* Ensure the merged output catalogue has the right column order of bands.
* Is the photometry for stars at the edges of masked-out areas reliable?
* Avoid pyraf from doing: "Created directory /home/gb/dev/surveytools/sandbox/catalogue/pyraf for cache"
* Can reduce maxnpsf during source list building?

Offset catalogues
~~~~~~~~~~~~~~~~~
* Give proper error messages in the catalogue, e.g. "Chi score too high".
* Verify a10 completeness.
* Produce diagnostic ccd/cmd plots.
* Check if the photometry for stars at the edges of masked-out areas reliable?
* msky vs mstar diagnostic plot (no relationship expected)
* mstar vs chi diagnostic plot (no relationship expected)
* Take care of both aperture correction & calibration using APASS.
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