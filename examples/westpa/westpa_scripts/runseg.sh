#!/bin/bash
#
# runseg.sh
#
# WESTPA runs this script for each trajectory segment. WESTPA supplies
# environment variables that are unique to each segment, such as:
#
#   WEST_CURRENT_SEG_DATA_REF: A path to where the current trajectory segment's
#       data will be stored. This will become "WEST_PARENT_DATA_REF" for any
#       child segments that spawn from this segment
#   WEST_PARENT_DATA_REF: A path to a file or directory containing data for the
#       parent segment.
#   WEST_CURRENT_SEG_INITPOINT_TYPE: Specifies whether this segment is starting
#       anew, or if this segment continues from where another segment left off.
#   WEST_RAND16: A random integer
#
# This script has the following three jobs:
#  1. Create a directory for the current trajectory segment, and set up the
#     directory for running pmemd/sander 
#  2. Run the dynamics
#  3. Calculate the progress coordinates and return data to WESTPA


######################## Set up for running the dynamics #######################

# Set up the directory where data for this segment will be stored.
cd $WEST_SIM_ROOT
mkdir -p $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

# This trajectory segment will start off where its parent segment left off.
# The "ln" command makes symbolic links to the parent segment's edr, gro, and 
# and trr files. This is preferable to copying the files, since it doesn't
# require writing all the data again.
ln -s $WEST_PARENT_DATA_REF/seg.npy ./parent.npy

############################## Run the dynamics ################################
# Propagate the segment
$WEST_PYTHON $WEST_SIM_ROOT/toy_config/toy --seed $WEST_RAND16 --n 50 --initial parent.npy seg.npy

########################## Calculate and return data ###########################

$WEST_PYTHON $WEST_SIM_ROOT/westpa_scripts/dump_coords.py --zero seg.npy > $WEST_PCOORD_RETURN

# Output coordinates. We use a custom python script that converts to the dcd 
# file to a multi-frame pdb (named seg.pdb)
if [ ${WEST_COORD_RETURN} ]; then
 $WEST_PYTHON $WEST_SIM_ROOT/westpa_scripts/dump_coords.py seg.npy > $WEST_COORD_RETURN
fi

# Clean up

