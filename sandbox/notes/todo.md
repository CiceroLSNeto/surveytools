TODO
====
General
-------
What's wrong with o20120819_00066.fit ?

Catalogue
---------
* What sigma to use for i-band detection step? 2 seems better for sparse; 1 for dense? (review sigma estimate?)
* What about aperture correction?
* Remove objects with large shifts altogether?
* Ensure the merged output catalogue has the right column order of bands.
* Is the photometry for stars at the edges of masked-out areas reliable?
* Recover notes from my laptop.
* Avoid writing subimage and seepsf during source list iteration.
* Avoid pyraf from doing: "Created directory /home/gb/dev/surveytools/sandbox/catalogue/pyraf for cache"
* Can reduce maxnpsf during source list building?
* Update the csv file with coordinates to reflect the changed P95 fields?

Quicklook
---------
* Why does the quicklook for 0004b not work? There is some overlap.
* How to get rid of all the green frames?