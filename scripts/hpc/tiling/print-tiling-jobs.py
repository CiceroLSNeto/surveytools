"""
This script prints out the shell commands to schedule the cluster jobs
for source-resolved and tiling the VPHAS catalogue.

Usage
=====
python print-tiling-jobs.py > tiling-jobs.sh
bash tiling-jobs.sh
"""
size = 1  # degrees
for lprime in range(205, 400+1):
    # Determine the galactic latitude steps
    if (lprime > 347) and (lprime < 372):
        bmin, bmax = -11, +11
    else:
        bmin, bmax = -6, +6

    for b in range(bmin, bmax+1):
        l = lprime % 360 
        cmd = 'qsub -v L={},B={},SIZE={} -N tile_{:.0f}_{:.0f} tiling.pbs'.format(l, b, size, l, b)
        print(cmd)
