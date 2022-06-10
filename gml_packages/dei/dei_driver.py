"""
Driver script for running the dei calculations in the dei class module.
"""

from __future__ import print_function


import os
import sys
import optparse

import dei

###########################################################################

parser = optparse.OptionParser(usage='%prog [options]',
        description="Create MBL surface data extension files, and optional bootstrap uncertainty runs. ")

parser.add_option('--initfile', help="Specify initialization file to use.")
parser.add_option('--numbs', default=0, type='int', help="Specify number of bootstrap runs.  Default is 0.")
parser.add_option('--gas', default='co2', help="Specify gas to use.  Default is 'co2'.")
parser.add_option('--anchor', action="store_true", help="Ignore sync2 value in initfile. Use today's date instead, and extrapolate data for anchor sites if necessary.")
parser.add_option('--quickfilter', action="store_true", help="Apply quick filtering of outliers > +/- 3 sigma of residuals.")
parser.add_option('--bsdir', help="Top level directory where to store results. Default is 'dei_results' in current directory.")
parser.add_option('--inputdir', help="Directory containing input data files.  Use these instead of data from database.")
parser.add_option('--unctype', type='choice', choices=['atmospheric', 'bias', 'network', 'analysis'], help="Specify the type of uncertainty to apply to data when doing bootstrap runs. Must be one of 'atmospheric', 'bias', 'analysis', or 'network'.")
parser.add_option('--stopdate', type='float', help="Specify the last date (as decimal date, i.e 2016.25) of input data to use.")
parser.add_option('--useadate', action="store_true", help="Use analysis date for --stopdate option.")
parser.add_option('--autosync', action="store_true", help="Automatically set sync2 date in initfile to now-4 months, last day of month.")
parser.add_option('--bootstrap_only', action="store_true", help="Run bootstrap for unctype only.  Do not perform normal mbl calculation. Use this to add a bootstrap run to an existing mbl result.")
parser.add_option('--config', help="Specify initialization file to use for bias, and analysis bootstrap values. REQUIRED if --unctype=bias or --unctype=analysis")
parser.add_option('--siteinfo', help="Specify file with site information.  Use this instead of getting information from database.")
options, args = parser.parse_args()


if not options.initfile:
	initfile = "/ccg/dei/ext/%s/work/init.%s.flask.master.txt" % (options.gas.lower(), options.gas.lower())
	if not os.path.exists(initfile):
		parser.print_help()
		sys.exit("\nERROR: default initfile %s not found.  Specify an initfile." % initfile)
	options.initfile = initfile

if options.numbs > 0 and options.unctype is None:
	sys.exit("ERROR: Must specify unctype if numbs is > 0")

if options.unctype == "bias" and options.config is None:
	sys.exit("ERROR: Must specify config file if unctype is 'bias'")
if options.unctype == "analysis" and options.config is None:
	sys.exit("ERROR: Must specify config file if unctype is 'analysis'")

if not options.bsdir:
	options.bsdir = "dei_results/"

if not options.bsdir.endswith("/"):
	options.bsdir = options.bsdir + "/"

print("Using init file ", options.initfile)
print("Bootstrap bias results will be written in ", options.bsdir)


if not options.bootstrap_only:
	# do mbl without modifications to data
	ext = dei.dei(options.initfile,
		options.gas,
		resultsdir=options.bsdir,
		create_subdirs=False,
		anchor=options.anchor,
		quickfilter=options.quickfilter,
		stopdate=options.stopdate,
		use_adate=options.useadate,
		autosync=options.autosync,
		siteinfo=options.siteinfo)

	if options.inputdir:
		ext.set_source_type(ext.FILE)
		ext.set_data_source(options.inputdir)  # if different from bsdir
	ext.run(1)

if options.numbs > 0:

	# now do bootstrap runs modifying input data with appropriate uncertainty type

	# get name of directory for bootstrap results
	options.bsdir = options.bsdir + "bs_" + options.unctype + "/"

	# create the dei class
	ext = dei.dei(options.initfile,
		options.gas,
		resultsdir=options.bsdir,
		create_subdirs=True,
		anchor=options.anchor,
		quickfilter=options.quickfilter,
		stopdate=options.stopdate,
		use_adate=options.useadate,
		siteinfo=options.siteinfo)

	# set location of input data. We'll make text files to use for all the bootstrap runs
	if options.inputdir:
		ext.set_data_source(options.inputdir)  # if different from bsdir
	else:
		ext.make_input_files(options.bsdir)
	ext.set_source_type(ext.FILE)

	# specify type of bootstrap to do
	if options.unctype == "atmospheric":
		ext.modify_source(ext.ATMOS_UNC)
	elif options.unctype == "bias":
		ext.modify_source(ext.BIAS)
		ext.set_bias_config(options.config)
	elif options.unctype == "analysis":
		ext.modify_source(ext.ANALYSIS)
		ext.set_analysis_config(options.config)
	elif options.unctype == "network":
		ext.modify_source(ext.NETWORK)

	# run it
	ext.run(options.numbs)
