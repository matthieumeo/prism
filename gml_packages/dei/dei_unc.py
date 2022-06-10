

""" combine bootstrap uncertainties for zones and curve types """

import sys
import optparse
import os
import numpy

ccgvu = ['ftn', 'tr', 'gr', 'fsc', 'ssc', 'residf', 'residsc', 'ann.ave.sc', 'ann.inc.tr', None]

# zone abbreviations
zones = ['arctic', 'hsh', 'lsh', 'lnh', 'hnh', 'sh', 'nh', 'equ', 'gl', 'psh', 'tsh', 'tropics', 'tnh', 'pnh']



###############################################################################
def get_unc_filename(options, bstype, zone, crvtype):
	""" Get the filename that has the zonal bootstrap uncertainties """

	if crvtype is None:
		uncfile = "zone_" + zone + ".mbl.unc." + options.gas.lower()
	else:
		uncfile = "zone_" + zone + ".mbl." + crvtype + ".unc." + options.gas.lower()

	if options.outdir is None:
		uncfile = "/".join([options.bsdir, bstype, uncfile])
	else:
		uncfile = "/".join([options.bsdir, bstype, options.outdir, uncfile])

	if options.verbose:
		print("uncertainty file is ", uncfile)

	return uncfile

###############################################################################
def get_data_filename(options, zone, crvtype):
	""" Get the filename that has the zonal data values """

	if crvtype is None:
		datafile = "zone_" + zone + ".mbl." + options.gas.lower()
	else:
		datafile = "zone_" + zone + ".mbl." + crvtype + "." + options.gas.lower()

	datafile = "/".join([options.bsdir, datafile])
	if options.verbose:
		print("data file is ", uncfile)

	return datafile

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


for zone in zones:

	for crvtype in ccgvu:
		unc = None

		for btype in bootstrap_types:
			if options.verbose:
				print("Checking for bootstrap type", btype)

			if os.path.exists(options.bsdir + "/" + btype):

				s = btype.replace("bs_", "")

				uncfile = get_unc_filename(options, btype, zone, crvtype)
				if options.verbose:
					print("Uncertainty file is", uncfile)

				if os.path.exists(uncfile):
					data = numpy.loadtxt(uncfile)
				else:
					print("!!! Zonal uncertainty file not found: %s." % uncfile, file=sys.stderr)
					continue

				# uncertainty values are in 3rd column. The [2] makes this into a column vector
				uncdata = data[:, [2]]

				if unc is None:
					unc = numpy.square(uncdata)
				else:
					unc = unc + numpy.square(uncdata)


		if unc is None:
			if options.verbose:
				print("No bootstrap uncertainty files found.")
				continue

		unc = numpy.sqrt(unc)

		datafile = get_data_filename(options, zone, crvtype)
		data = numpy.loadtxt(datafile)
		data = numpy.hstack((data, unc))

		if crvtype is None:
			uncfile = "zone_" + zone + ".mbl.unc." + options.gas.lower()
		else:
			uncfile = "zone_" + zone + ".mbl." + crvtype + ".unc." + options.gas.lower()
		uncfile = "/".join([options.bsdir, uncfile])
#		if options.verbose:
		print("Writing results to", uncfile)

		numpy.savetxt(uncfile, data, fmt="%14.8f %10.3f %10.3f")

#		sys.exit()
