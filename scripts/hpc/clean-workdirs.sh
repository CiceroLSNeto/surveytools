#!/bin/bash
# Makes sure that /tmp/gb-scratch is deleted on all nodes of the cluster.
# This directory is normally deleted as part of the script, but it may not
# have happened if e.g. the script crashed.
for i in {1..144}
do
    node=$(printf "node%03d" $i)
    echo $node
    ssh $node -t "rm -rf /tmp/gb-scratch"
done
