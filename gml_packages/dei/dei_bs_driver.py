
"""
Script to start up dei_bs
"""

from __future__ import print_function


import os
import sys
import optparse

import dei_bs

###########################################################################

parser = optparse.OptionParser(usage='%prog [options]',
	description="Process MBL bootstrap files. ")

parser.add_option('--initfile', help="Specify initialization file to use.")
parser.add_option('--numbs', default=0, type='int', help="Specify number of bootstrap runs.  Default is 0.")
parser.add_option('--gas', default='co2', help="Specify gas to use.  Default is 'co2'.")
parser.add_option('--bsdir', help="Top level directory of bootstrap results. Default is 'dei_results'.")
parser.add_option('--unctype', default='atmospheric', type='choice', choices=['atmospheric', 'bias', 'network', 'analysis'], help="Specify the type of uncertainty to apply to data when doing bootstrap runs. Must be one of 'atmospheric' or 'bias'.")
parser.add_option('--npoly', default=3, type='int', help="Number of polynomial terms to use in function fit of curve fitting.")
parser.add_option('--outdir', help="Specify output directory (subdirectory of --bsdir) to place files. Default is directory specified by --bsdir")
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

if not options.bsdir:
	options.bsdir = "results/"

if not options.bsdir.endswith("/"):
	options.bsdir = options.bsdir + "/"

print("Using init file ", options.initfile)


options.bsdir = options.bsdir + "bs_" + options.unctype + "/"
print("Using Bootstrap bias results from ", options.bsdir)

ext = dei_bs.dei_bs(options.initfile,
	options.gas,
	resultsdir=options.bsdir,
	unctype=options.unctype,
	outdir=options.outdir,
	siteinfo=options.siteinfo,
	)

ext.set_filter_param(npoly=options.npoly)

ext.run(options.numbs)
