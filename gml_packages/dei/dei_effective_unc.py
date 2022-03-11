
""" create effective uncertianty surface from available bootstrap uncertainty surfaces """

import sys
import optparse
import os
import numpy

###############################################################################
def ccg_savetxt(filename, a, fmt="%.18e", header=None):
	""" write a grid to text file """

	fh = open(filename, "w")

#	if header:
#		fh.write(header)

	num_ts = a.shape[0]
	for i in range(num_ts):
		fh.write(fmt % tuple(a[i]) + '\n')

	fh.close()


###############################################################################
parser = optparse.OptionParser(usage='%prog [options]',
	description="Create MBL effective uncertainty surface files, based on available bootstrap uncertainty surfaces. ")

parser.add_option('--gas', default='co2', help="Specify gas to use.  Default is 'co2'.")
parser.add_option('--bsdir', help="Top level directory where bootstrap results are located.")
parser.add_option('--outdir', help="Specify output directory (subdirectory of --bsdir) where surface.mbl file is located.")
parser.add_option('--verbose', action="store_true", help="Print extra messages")
options, args = parser.parse_args()

if options.bsdir is None:
	parser.error("bsdir option is required")

bootstrap_types = ["bs_network", "bs_atmospheric", "bs_bias", "bs_analysis"]

unc = None

for btype in bootstrap_types:
	if options.verbose:
		print "Checking for bootstrap type", btype

	if os.path.exists(options.bsdir + "/" + btype):

		s = btype.replace("bs_", "")
		mblfile = "surface.mbl." + s + ".unc.%s" % options.gas

		if options.outdir is None:
			uncsurf = "/".join([options.bsdir, btype, mblfile])
		else:
			uncsurf = "/".join([options.bsdir, btype, options.outdir, mblfile])
		if options.verbose:
			print "mbl uncertainty surface file is ", uncsurf

		if os.path.exists(uncsurf):
			surf = numpy.loadtxt(uncsurf)
		else:
			print >> sys.stderr, "mbl uncertainty surface file not found for %s." % btype
			continue

		if unc is None:
			unc = numpy.square(surf)
		else:
			unc = unc + numpy.square(surf)

if unc is None:
	if options.verbose:
		print "No bootstrap uncertainty files found."
	sys.exit()

effective_unc = numpy.sqrt(unc)
# replace first column with dates
effective_unc[:, 0] = surf[:, 0]
num_bins = effective_unc.shape[1]
format_str = "%13.8f" + " %8.3f" * (num_bins-1)


filename = "%s/surface.mbl.effective_unc.%s" % (options.bsdir, options.gas)

if options.verbose:
	print "Creating effective uncertainty file", filename

header = """# sp=%s
# resultsdir=%s
# FORMAT='(F13.8,%d(1X,F8.3))'
""" % (options.gas, options.bsdir, num_bins-1)


ccg_savetxt(filename, effective_unc, fmt=format_str, header=header)
