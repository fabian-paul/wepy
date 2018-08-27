#!/bin/bash

# Make sure we are in the correct directory
cd $WEST_SIM_ROOT
source env.sh
cd $WEST_STRUCT_DATA_REF

# Make a temporary file in which to store output from the python script
DIST=$(mktemp)

$WEST_PYTHON $WEST_SIM_ROOT/westpa_scripts/dump_coords.py seg.npy > $DIST

# Pipe the relevant part of the output file (the distance) to $WEST_PCOORD_RETURN
cat $DIST | tail -n 1 > $WEST_PCOORD_RETURN

# Remove the temporary file to clean up
rm $DIST

