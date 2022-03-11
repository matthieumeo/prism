#!/bin/bash
#
# A template for building scripts to run the data extension software

set -x

# location of dei software
progdir="/ccg/dei/ext/src/python"

# set the gas to use
gas="co2"

# create a timestamp string for the results directory name
thismonth=`date +%Y-%m`

# set the starting directory for the results output
basedir="/ccg/dei/ext"

# set the number of bootstrap runs to do
numbs=100

# set the init file
initfile="/ccg/dei/ext/co2/work/init.${gas}.flask.master.txt"

# build the output results directory name
bsdir="$basedir/$gas/results.$thismonth"

# calculate the mbl data, and run the atmospheric bootstrap
python $progdir/dei_driver.py --initfile=$initfile --bsdir=$bsdir --numbs=$numbs --anchor --gas=$gas --quickfilter --unctype=atmospheric

# add the network bootstrap runs to the results
python $progdir/dei_driver.py --initfile=$initfile --bsdir=$bsdir --numbs=$numbs --anchor --gas=$gas --quickfilter --unctype=network --bootstrap_only

# get a summary of atmospheric bootstrap runs
python $progdir/dei_bs_driver.py --initfile=$initfile --bsdir=$bsdir --gas=$gas --numbs=$numbs --unctype=atmospheric

# get a summary of network bootstrap runs
python $progdir/dei_bs_driver.py --initfile=$initfile --bsdir=$bsdir --gas=$gas --numbs=$numbs --unctype=network

# combine the bootstrap methods to get total surface uncertainty file
python $progdir/dei_effective_unc.py --bsdir=$bsdir --gas=$gas
