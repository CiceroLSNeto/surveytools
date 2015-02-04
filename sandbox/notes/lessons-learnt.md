Lessons learnt
==============

DAOPHOT PSF-fitting
-------------------
* It is important to set the DATAMAX parameters conservatively.
The wings of saturated stars should not be used for PSF fitting,
because charge bleeding can spoil the wings.
* DaoPhot's PSF fitting routine seems likely to fail if the routine is given
a spurious object (e.g. in the wing of a bright star).
Requiring an object to be detected at the ~same position in
all bands greatly helps reduce this problem.  Imposing a stricter criterion
on roundness may help too.  Even better: pruning stars from pstselect was
found to be the killer trick.
