#!/usr/bin/env python
"""
Program for applying curve fitting/filtering of time series data and
printing results.

Takes an input file containing two columns of data, a decimal date and value.
Applies curve fitting algorithm to the data, and depending on options,
prints results to stdout (default) or to files if specified.

General python requirements:
	numpy, scipy, dateutil

ccgg python requirements:
	ccg_filter - curve fitting/filtering
	ccg_dates - date conversion routines

Update Sep 2021:
        Add capability of input file being in obspack text format.
"""
from __future__ import print_function

import sys
import argparse

from dateutil.parser import parse
from dateutil.rrule import rrule, DAILY

import ccg_filter
from ccg_dates import calendarDate, decimalDateFromDatetime, datetimeFromDecimalDate

##########################################################################
def check_export(options):
	""" Check if any options are set for filtered curve data. """

	a = options.orig \
               or options.func \
               or options.poly \
               or options.smooth \
               or options.trend \
               or options.detrend \
               or options.smcycle \
               or options.harm \
               or options.res \
               or options.smres \
               or options.trres \
               or options.ressm \
               or options.gr

	return a


##########################################################################
def export_data(options, filt):
	""" print out the curve data """

	if options.sample:
		if options.samplefile:
			try:
				fp = open(options.samplefile, "w")
			except IOError as e:
				sys.exit("Can't open file for writing. %s" % e)

		else:
			fp = sys.stdout

		export_dates(options, fp, filt, filt.xp)

	if options.equal or options.user:
		if options.file:
			try:
				fp = open(options.file, "w")
			except IOError as e:
				sys.exit("Can't open file for writing. %s" % e)

		else:
			fp = sys.stdout

		if options.equal:
			# Create a new list of dates at sample interval to give to export_dates().
			# Not quite the same as filt.xinterp because it takes into account leap years
			dates = rrule(DAILY, interval=int(filt.sampleinterval), dtstart=options.startdate, until=options.lastdate)
			xdates = [decimalDateFromDatetime(dt) for dt in dates]
			if xdates[-1] > filt.xp[-1]:
				xdates[-1] = filt.xp[-1]  # avoid problems with rounding and interpolation in ccgfilt
			if xdates[0] < filt.xp[0]:
				xdates[0] = filt.xp[0]  # avoid problems with rounding and interpolation in ccgfilt

		else:
			f = open(options.user)
			xdates = [float(line.split()[0]) for line in f]
			f.close()


		export_dates(options, fp, filt, xdates)


##########################################################################
def export_dates(options, fp, filt, x):
	""" Export (print) data to file pointer fp at dates given by x.
	The values to print are given as boolean flags in the export class.
	Some values can only be printed at sample dates, i.e. original data and residuals
	"""

	if options.showheader:
		export_header(options, fp)

	frmt = "%13.6e"

	h = filt.getHarmonicValue(x)	# harmonics
	p = filt.getPolyValue(x)	# poly
	s = filt.getSmoothValue(x)	# function + short term smoothing
	t = filt.getTrendValue(x)	# poly + long term smoothing
	g = filt.getGrowthRateValue(x)	# growth rate, derivative of trend
	f = filt.getFunctionValue(x)    # function, poly + harmonics

	for i, xp in enumerate(x):
		if options.cal:
			(yr, mon, dy, hr, mn, sec) = calendarDate(xp)
			if options.hour:
				print("%4d %02d %02d %2d" % (yr, mon, dy, hr), end='', file=fp)
			else:
				print("%4d %02d %02d" % (yr, mon, dy), end='', file=fp)
		else:
			print("%13.8f" % xp, end='', file=fp)

		if options.sample and options.orig:    print(frmt % filt.yp[i], end='', file=fp)
		if options.func:                       print(frmt % f[i], end='', file=fp)
		if options.poly:                       print(frmt % p[i], end='', file=fp)
		if options.smooth:                     print(frmt % s[i], end='', file=fp)
		if options.trend:                      print(frmt % (t[i]), end='', file=fp)
		if options.sample and options.detrend: print(frmt % (filt.yp[i] - t[i]), end='', file=fp)
		if options.smcycle:                    print(frmt % (s[i] - t[i]), end='', file=fp)
		if options.harm:                       print(frmt % (h[i]), end='', file=fp)
		if options.sample and options.res:     print(frmt % (filt.yp[i] - f[i]), end='', file=fp)
		if options.smres:                      print(frmt % (s[i] - f[i]), end='', file=fp)
		if options.trres:                      print(frmt % (t[i] - p[i]), end='', file=fp)
		if options.sample and options.ressm:   print(frmt % (filt.yp[i] - s[i]), end='', file=fp)
		if options.gr:                         print(frmt % (g[i]), end='', file=fp)

		print(file=fp)


##########################################################################
def export_header(options, fp):
	""" Export a line with column header names to file pointer fp.
	"""

	frmt = "%-13s"

	print(frmt % "date", end='', file=fp)

	# make sure these are in same order as in export_dates()
	if options.sample and options.orig:    print(frmt % "value", end='', file=fp)
	if options.func:                       print(frmt % "function", end='', file=fp)
	if options.poly:                       print(frmt % "polynomial", end='', file=fp)
	if options.smooth:                     print(frmt % "smooth", end='', file=fp)
	if options.trend:                      print(frmt % "trend", end='', file=fp)
	if options.sample and options.detrend: print(frmt % "detrended", end='', file=fp)
	if options.smcycle:                    print(frmt % "smooth_cycle", end='', file=fp)
	if options.harm:                       print(frmt % "harmonics", end='', file=fp)
	if options.sample and options.res:     print(frmt % "residuals", end='', file=fp)
	if options.smres:                      print(frmt % "smooth_resid", end='', file=fp)
	if options.trres:                      print(frmt % "trend_resid", end='', file=fp)
	if options.sample and options.ressm:   print(frmt % "resid_smooth", end='', file=fp)
	if options.gr:                         print(frmt % "growth_rate", end='', file=fp)

	print(file=fp)

#########################################################################
def read_data(filename=None):
	"""
	# Read in the input data file.
	# Format is always two columns,
	# the first column a decimal date value, (e.g. 2010.5 is halfway through 2010)
	# the second column is the corrsponding measurement value.
	"""

	if filename is None:
		fp = sys.stdin
		x, y = read_dd_val(fp)
	else:
		try:
			fp = open(filename)
		except IOError as e:
			sys.exit("Cannot open input file. %s" % e)

		line = fp.readline()
		if line.startswith("# header_lines"):
			fp.close()
			x, y = read_obspack(filename)
		else:
			fp.seek(0)
			x, y = read_dd_val(fp)

        return x, y

#########################################################################
def read_dd_val(fp):
	""" read data where each line has the format 'dd value'
	where dd - decimal date
	"""

	x = []
	y = []
	for line in fp:
		a = line.split()
		xv = a[0]
		yv = a[1]
		x.append(float(xv))
		y.append(float(yv))

	fp.close()


	return x, y

#########################################################################
def read_obspack(filename):
	""" Read an obspack formatted text file.

	See also https://gml.noaa.gov/ccgg/obspack
	"""

	# first line of file specifies number of header lines
	f = open(filename)
	line = f.readline()
	(s, nheader) = line.split(":")
	nheader = nheader.strip()
	nheader = int(nheader)

	# skip header lines
	for i in range(nheader-2):
		f.readline()

	# read header line with column names
	header = f.readline()
	header = header.strip()
	colnames = header.split()

	# get columns that contain the decimal date, value and obs_flag
	ddcol = colnames.index('time_decimal')
	valcol = colnames.index('value')
	flagcol = colnames.index('obs_flag')

	x = []
	y = []
	for line in f:
		a = line.split()
		flag = int(a[flagcol])
		if flag == 1:
			x.append(float(a[ddcol]))
			y.append(float(a[valcol])*1e6)

	f.close()

	return x, y


#########################################################################

startdate = None

parser = argparse.ArgumentParser(description="Apply curve fitting/filtering to input data and print results. ")

group = parser.add_argument_group("Filter Options")
group.add_argument('--npoly', default=3, type=int, help="Number of polynomial terms in function.")
group.add_argument('--nharm', default=4, type=int, help="Number of harmonic terms in function.")
group.add_argument('--interv', default=0, type=float, help="Sampling interval of the data in days.")
group.add_argument('--short', default=80, type=int, help="Short-term filter cutoff in days.")
group.add_argument('--long', default=667, type=int, help="Long-term filter cutoff in days.")
group.add_argument('--gap', default=0, type=int, help="Fill gaps larger than GAP days with function value instead of linear interpolation.")
group.add_argument('--gain', action="store_true", default=False, help="Use seasonal amplitude gain factor in function fit.")
group.add_argument('--timez', type=float, help="Specify time zero for coefficients of function. Default is year of first data point.")

group = parser.add_argument_group("Output Options")
group.add_argument('-f', '--file', help="Write equally spaced or user spaced output data to file instead of stdout.")
group.add_argument('-s', '--samplefile', help="Write sample spaced output data to file instead of stdout.")
group.add_argument('--equal', action="store_true", default=False, help="Output data at equal intervals.")
group.add_argument('--sample', action="store_true", default=False, help="Output data at sample data times (default).")
group.add_argument('--cal', action="store_true", default=False, help="Output dates in calendar format.")
group.add_argument('--hour', action="store_true", default=False, help="Include hour in calendar format.")
group.add_argument('--date', help="Output data starting at date.")
group.add_argument('--user', help="Output data based on user supplied dates in file.")
group.add_argument('--showheader', action="store_true", default=False, help="Include header on output to identify columns.")


group = parser.add_argument_group("Output Parameters to include")
group.add_argument('--orig', action="store_true", default=False, help="Output original data points.")
group.add_argument('--func', action="store_true", default=False, help="Output function values.")
group.add_argument('--poly', action="store_true", default=False, help="Output polynomial values.")
group.add_argument('--smooth', action="store_true", default=False, help="Output smoothed data.")
group.add_argument('--trend', action="store_true", default=False, help="Output long term trend values.")
group.add_argument('--detrend', action="store_true", default=False, help="Output detrended values.")
group.add_argument('--smcycle', action="store_true", default=False, help="Output smoothed, detrended annual cycle.")
group.add_argument('--harm', action="store_true", default=False, help="Output values of annual harmonic functions.")
group.add_argument('--res', action="store_true", default=False, help="Output residuals from the function.")
group.add_argument('--smres', action="store_true", default=False, help="Output smoothed residuals from the function.")
group.add_argument('--trres', action="store_true", default=False, help="Output long-term smoothed residuals from the function.")
group.add_argument('--ressm', action="store_true", default=False, help="Output residuals from the smoothed curve.")
group.add_argument('--gr', action="store_true", default=False, help="Output growth rate values.")
group.add_argument('--coef', help="Output coefficients from index num1 to index num2. e.g. 1,4")
group.add_argument('--stats', action="store_true", default=False, help="Output table of summary statistics for curve fit.")
group.add_argument('--amp', action="store_true", default=False, help="Output table of statistics for annual amplitudes.")
group.add_argument('--mm', action="store_true", default=False, help="Output Monthly means computed using smooth curve.")
group.add_argument('--annual', action="store_true", default=False, help="Output Annual means computed using smooth curve.")

parser.add_argument('args', nargs=1)


options = parser.parse_args()



if options.npoly < 0 or options.npoly > 10:
	sys.exit("Error in --npoly argument: value out of range (0-10) %s" % options.npoly)

if options.nharm < 0 or options.nharm > 10:
	sys.exit("Error in --npoly argument: value out of range (0-10) %s" % options.nharm)

if options.interv < 0:
	sys.exit("Error in --interval argument: value out of range (>=0) %s" % options.interv)

if options.short < 0:
	sys.exit("Error in --short argument: value out of range (must be >=0) %s" % options.short)

if options.long < 0:
	sys.exit("Error in --long argument: value out of range ( must be >=0) %s" % options.long)

if options.gap < 0:
	sys.exit("Error in --gap argument: value out of range (must be >=0) %s" % options.gap)

if options.date:
	try:
		startdate = parse(options.date)
	except ValueError as err:
		sys.exit("Can not get valid date from --date argument '%s': %s" % (options.date, err))



if options.coef:
	try:
		(begcoef, endcoef) = options.coef.split(",")
		begcoef = int(begcoef)
		endcoef = int(endcoef)
	except ValueError:
		sys.exit("Cannot get coefficient range.")


args = options.args
if not len(args):
	xp, yp = read_data()
else:
	inputfile = args[0]
	xp, yp = read_data(inputfile)


# if user dates or equal spaced dates aren't specified, use sample dates as default
if not options.user and not options.equal: options.sample = True

# Compute the filtered data
if options.timez is None: options.timez = int(xp[0])
filt = ccg_filter.ccgFilter(xp, yp, options.short, options.long, options.interv, options.npoly, options.nharm, options.timez, options.gap, options.gain)


# If starting date is not specified, set it to the date of the first data point
# Set ending date to date of last data point
if startdate is None:
	options.startdate = datetimeFromDecimalDate(filt.xp[0])
else:
	options.startdate = startdate
options.lastdate = datetimeFromDecimalDate(filt.xp[-1])

if check_export(options):
	export_data(options, filt)

if options.amp:
	months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
	amps = filt.getAmplitudes()

	print(" *****  Seasonal Cycle Statistics.  *****")
	print(" Year      Amplitude     Maximum   Date     Minimum   Date")
	print("-----------------------------------------------------------")

	frmt = "%5.0f %12.2f %12.2f   %3s %2d %9.2f   %3s %2d"
	for (year, amp, maxdate, maxval, mindate, minval) in amps:

		(yr, mnmax, dmax, hr, mn, sec) = calendarDate(maxdate)
		(yr, mnmin, dmin, hr, mn, sec) = calendarDate(mindate)

		print(frmt % (year, amp, maxval, months[mnmax], dmax, minval, months[mnmin], dmin))

if options.stats:
	print(filt.stats())

if options.mm:
	mm = filt.getMonthlyMeans()
	for (year, month, val, std, n) in mm:
		print("%4d %02d %7.2f %5.2f %2d" % (year, month, val, std, n))

if options.annual:
	am = filt.getAnnualMeans()
	for (year, val, std, n) in am:
		print("%4d %7.2f %5.2f %2d" % (year, val, std, n))

if options.coef:
	for i in range(begcoef, min(filt.numpm, endcoef+1)):
		print(" %.6f" % filt.params[i], end='')
#		print("%d %.6f" % (i, filt.params[i]))
	print()
