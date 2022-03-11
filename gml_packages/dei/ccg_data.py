""" routines for accessing data for the dei software,
such as from database or files.
Also determines bias and noise values to apply to data
for certain bootstrap methods.
"""

from __future__ import print_function

import os
import sys
import datetime
import random
import numpy


import ccg_db
import ccg_dates
import ccg_filter

###########################################################################
def ccg_data(site, gas, startdate=None, stopdate=None, project=1, use_adate=False):
	""" Get data from database.
	Input:
		site - site code
		gas - gas species
		stopdate - don't use any data after this date
	Output:
		A numpy array containing 2 columns of data,
			the sample date in decimal year
			the measured mixing ratio value (pair average)
	"""

	strategy_num = 1

	include_lat = False
	gasnum = ccg_db.getGasNum(gas)
	if "_" in site:
		(code, s) = site.split("_")
	else:
		code = site

	if len(code) > 3:
		# get bin width for scode here

		scode = code[0:3]
		(bintype, binmin, binmax, binwidth) = ccg_db.getBinInfo(scode, project)

		hemi = code[3]
		lat = float(code[4:])
		if hemi.lower() == "s":
			lat = -lat
		lat_south = lat - binwidth/2
		lat_north = lat + binwidth/2
		include_lat = True
	else:
		scode = code

	sitenum = ccg_db.getSiteNum(scode)

	# create query to get data from database and
	# average data with same date and time
	sql = "SELECT flask_event.dd, avg(flask_data.value) FROM flask_event,flask_data "
	sql += "WHERE flask_event.num=flask_data.event_num AND flask_event.site_num=%d " % sitenum
	sql += "AND flask_data.value > -999 "
	sql += "AND flask_event.project_num=1 "
	sql += "AND flask_event.me != 'H' "
	sql += "AND flask_data.program_num != 8 "	# skip hats measurements
	sql += "AND flask_data.parameter_num=%d " % gasnum
	sql += "AND flask_data.flag like '..%%' AND strategy_num=%d " % strategy_num
	if include_lat:
		sql += "AND flask_event.lat between %.1f and %.1f " % (lat_south, lat_north)
	if startdate is not None:
		if use_adate:
			sql += "AND flask_data.dd>=%f " % startdate
		else:
			sql += "AND flask_event.dd>=%f " % startdate
	if stopdate is not None:
		if use_adate:
			sql += "AND flask_data.dd<=%f " % stopdate
		else:
			sql += "AND flask_event.dd<=%f " % stopdate
	sql += "GROUP BY flask_event.dd "   # this averages data on same date
	sql += "ORDER BY flask_event.dd "
#	print sql

	result = ccg_db.dbQueryAndFetch(sql)
	if len(result):

		a = numpy.array(result)
		dd = a.T[0]
		mr = numpy.array(a.T[1], dtype='double')

		return dd, mr
	else:
		return None, None


##############################################################################
def ccg_data_atmos_unc(dd, mr, site, initparam):
	""" call ccg_data and add some 'atmospheric' noise to the data """


	n = initparam['site'].index(site)

	# fit smooth curve to data
	short_cutoff = int(initparam['short'][n])
	long_cutoff = int(initparam['long'][n])
	interval = initparam['int'][n]
	npoly = int(initparam['poly'][n])
	nharm = int(initparam['harm'][n])
	filt = ccg_filter.ccgFilter(dd, mr, short_cutoff, long_cutoff, interval, npoly, nharm, timezero=initparam['sync1'])

	# add 'atmospheric' variability to smooth curve
	yv = atmospheric_unc(dd, mr, filt)

	return dd, yv


###########################################################################
def atmospheric_unc(x, y, filt):
	""" Modify the input 'y' values with random values based on
	the atmospheric variability of the month of the year.
	Used in the 'atmospheric' bootstrap method of the data extension software.
	"""

	# Get month numbers for data points from dates array
	months = numpy.empty( len(x), dtype=numpy.int )
	for i, xp in enumerate(x):
		dt = ccg_dates.datetimeFromDecimalDate(xp)
		months[i] = dt.month


	# get standard deviation of residuals from smooth curve
	overall_rsd = filt.rsd2

	smooth = filt.getSmoothValue(x)
	resid = y - smooth

	npoints = len(x)
	random_value = numpy.random.randn(npoints)

	# find rsd for each month
	month_rsd = numpy.zeros((13))
	for i in range(1, 13):
		w = numpy.where(months==i)
		if w[0].size == 0:
			std = overall_rsd
		else:
			r = resid[w]
			std = numpy.std(r, ddof=1)

		month_rsd[i] = std

	yv = numpy.array(y)
	ymod = numpy.empty( (npoints) )
	for i, month in enumerate(months):
		err = month_rsd[month] * random_value[i]
		w = numpy.where(months == month)
		ymod[w] = yv[w] + err


	return ymod

###########################################################################
def ccg_data_file(site, gas, dirname, stopdate=None, verbose=False):
	""" Read decimal date and mixing ratio values from a file. 
	First column is decimal date, second column is mixing ratio.
	Any other columns are ignored.
	"""

	filename = dirname + site + "_dat." + gas
	if os.path.exists(filename):
		if verbose: print("Reading file", filename)
		try:
			a = numpy.loadtxt(filename)
		except IOError as e:
			print("WARNING: Error reading", filename, e, "Skipping.", file=sys.stderr)
			return None, None

		x = a.T[0]
		y = a.T[1]
		if stopdate is not None:
			w = numpy.where(x<=stopdate)
			x = x[w]
			y = y[w]

		return x, y
	else:
		if verbose: print("WARNING: File not found.", filename, file=sys.stderr)
		return None, None

###########################################################################
def get_bias_data(gas, start_date, end_date, values):
	""" Determine random bias values for numerous time periods.
	Each time period is a random length from 3 to 24 months, and a
	bias value is set for each time period.  Multiple time periods and
	bias values are set for the entire analysis time span set by
	the sample start_date and end_date.

	values is a 3 member tuple, with min period, max period, bias value

	The length of time that a bias value is applied is a random number
	between min_period and max_period.

	The actual bias value applied to the data is a random value chosen
	from a normal distribution with a mean of 0 and a standard deviation
	of the given bias value
	"""

	(min_period, max_period, bias_value) = values

	gasnum = ccg_db.getGasNum(gas)

	# get min and max analysis dates for given time range
	sql = "SELECT min(a_date), max(a_date) FROM flask_data_view "
	sql += "WHERE ev_dd>=%f AND ev_dd<=%f AND parameter_num=%d" % (start_date, end_date, gasnum)
	result = ccg_db.dbQueryAndFetch(sql)

	sdate = result[0][0]
	edate = result[0][1]
	dlist = []
	while sdate < edate:
		days = int(random.uniform(min_period, max_period))      # length of this time period in days
		bias = round(random.gauss(0, bias_value), 2)            # the bias value for this time period
		date_span = datetime.timedelta(days=days)               # convert time period to timedelta
		dlist.append((sdate, sdate+date_span, bias))            # set start, end, bias for this time period
		sdate += date_span                                      # set start of next time period


	return dlist

###########################################################################
def get_analysis_dates(site, gas, dd):
	""" Get analysis dates for a given site and gas, and
	sample dates (dd) given as array of decimal values.

	NOTE: This algorithm may not give exactly correct analysis dates.
	Because the input sample dates are for flask pairs, and because
	more than one pair can be taken on the same day, we make assumptions
	about the analysis date.  So for any sample date, we use the first
	analysis date that matches that sample date.
	Usually this is the correct date, but sometimes may be off by a few days.
	"""


	gasnum = ccg_db.getGasNum(gas)
	a = site.split("_")

	if site[0:3].lower() in ["poc", "scs"]:
		sitecode = site[0:3]
	else:
		sitecode = a[0]


	sitenum = ccg_db.getSiteNum(sitecode)

	adates = []

# The slow way
#	db, c = ccg_db.dbConnect()
#	for dt in dd:
#		date = ccg_dates.dateFromDecimalDate(dt)

#		sql = "SELECT flask_data.date FROM flask_event,flask_data "
#		sql += "WHERE flask_event.num=flask_data.event_num AND flask_event.site_num=%d " % sitenum
#		sql += "AND flask_data.parameter_num=%d " % gasnum
#		sql += "AND flask_event.date='%s' " % date
#		sql += "ORDER BY flask_data.date "
#
#		c.execute(sql)
#		row = c.fetchone()
#		if row:
#			adates.append(row[0])
#
#	c.close()
#	db.close()


# The fast way
	# Grab all sample dates, analysis dates for this site and gas
	sql = "SELECT DISTINCT flask_event.date, flask_data.date FROM flask_event, flask_data "
	sql += "WHERE flask_event.num=flask_data.event_num AND flask_event.site_num=%d " % sitenum
	sql += "AND flask_data.parameter_num=%d " % gasnum
	sql += "ORDER BY flask_event.date, flask_data.date;"
	rows = ccg_db.dbQueryAndFetch(sql)

	# make a dict with sample date as key, analysis date as value
	# this makes it easier to find the analysis date given a sample date
	# In cases where there are multiple analysis dates for the same sample date,
	# use the first analysis date
	dates = {}
	for sample_date, adate in rows:
		if sample_date not in dates:
			dates[sample_date] = adate

	# now go through all the sample dates and find the corresponding analysis date
	for x in dd:
		dt = ccg_dates.datetimeFromDecimalDate(x)
		adates.append(dates[dt.date()])



	return numpy.array(adates)



###########################################################################
if __name__ == "__main__":

	site = "brw"
	gas = "co2"

#	t0 = datetime.datetime.now()
#	x, y = ccg_data_file(site, gas, "./", verbose=True)
#	t1 = datetime.datetime.now()
#	print t1-t0


	t2 = datetime.datetime.now()
	x, y = ccg_data(site, gas, stopdate=2010, use_adate=True)
	t0 = datetime.datetime.now()
	print("time to get data from database", t0-t2)
	print(x)
	print(y)
	print(x.size, y.size)
	sys.exit()


	adates = get_analysis_dates(site, gas, x)
	t1 = datetime.datetime.now()
	print("time to get analysis dates", t1-t0)
	sys.exit()

#	f = open("hba.dat", "w")
#	for xp, yp, adate in zip(x, y, adates):
	for xp, yp in zip(x, y):
#		print >> f, xp, yp
		print(xp, yp)
#		print xp, yp, adate


#	f.close()
