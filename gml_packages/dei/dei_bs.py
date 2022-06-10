
"""
data extension class for processing bootstrap files previously created using dei.py

Typical usage is something like

	bsext = dei_bs(initfile, gas, resultsdir)
	bsext.run(numbs)

Creation:
	bsext = dei_bs(initfile, gas, resultsdir)

    Input Parameters
    ----------------
	initfile : string
	    dei initfile that was used for bootstrap calculations
	gas : string
	    gas species to use, e.g. 'co2'
	resultsdir : string
	    location where results from bootstrap runs are located
	    Note that this is the bootstrap directory, not the extension
	    results directory.  For example, it would be something like
	    'co2/results/bs_bias' for a bias bootstrap run.
	outdir : string
	    Use a sub directory of the resultsdir in which to place results.
	    Otherwise results are place in resultsdir

    Methods
    -------

        run(numbs=1)
	    Begin processing bootstrap files.
	    numbs - Number of bootstrap runs to process.

	set_filter_param(self, npoly=3, nharm=4, cutoff1=80, cutoff2=667, interval=7):
	    npoly - Number of polynomial terms to use in curve fit
	    nharm - Number of harmonic terms to use in curve fit
	    cutoff1 - short term cutoff value for filter
	    cutoff2 - long term cutoff value for filter
	    interval - sampling inverval in days for the data.

    surface_unc(numbs)
        Combine multiple surface.mbl.gas files and calculate the standard deviation for each
        time step/lat bin across all surface files for 'numbs' bootstrap runs.
	    This is called by the run() method.


OUTPUT:
	surface.mbl.network.unc.n2o -

	15 files are created, containing mean and std. dev. across all the bootstrap runs
	for each time step and each zone, with names, if gas='n2o'

	zonal.ann.ave.sc.n2o.bs - annual averages from the smooth curve to each zone.
	zonal.ave.n2o.bs - annual averages for the zones
	zonal.fsc.n2o.bs - harmonic function values
	zonal.gr.n2o.bs - growth rate values
	zonal.iphd.n2o.bs - polar interhemispheric differences
	zonal.mm.tr.n2o.bs - monthly means of the trend
	zonal.residsc.n2o.bs - residuals from the smooth curve
	zonal.tr.n2o.bs - trend values
	zonal.ann.inc.tr.n2o.bs - jan 1 to jan 1 increase
	zonal.coef.n2o.bs - coefficients of the function fits
	zonal.ftn.n2o.bs - function values
	zonal.ihd.n2o.bs - interhemispheric differences
	zonal.mm.sc.n2o.bs - monthly means of smooth curve
	zonal.residf.n2o.bs - residuals from the function
	zonal.ssc.n2o.bs - smoothed seasonal cycle

	10 files for each zone containing uncertainties merged with values, e.g.:
		zone_tnh.mbl.ann.ave.sc.unc.co2 - annual averages from the smooth curve
		zone_tnh.mbl.ann.inc.tr.unc.co2 - annual increase of the trend
		zone_tnh.mbl.fsc.unc.co2 - harmonic function values
		zone_tnh.mbl.ftn.unc.co2 - function (poly + harm) values
		zone_tnh.mbl.gr.unc.co2 - growth rate
		zone_tnh.mbl.residf.unc.co2 - residuals from the function
		zone_tnh.mbl.residsc.unc.co2 - residuals from the smooth curve
		zone_tnh.mbl.ssc.unc.co2 - smoothed seasonal cycle
		zone_tnh.mbl.tr.unc.co2 - trend
		zone_tnh.mbl.unc.co2 - mbl values



"""

from __future__ import print_function

import os
import sys
import datetime
import numpy

import ccg_dates
import ccg_filter

from ccg_readinit import ccg_readinit

DEFAULT = -999.999
STEPS_PER_YEAR = 48

class dei_bs:
	""" class to process bootstrap dei files already created using dei.py and dei_driver.py """

	#-----------------------------------------------------------------
	def __init__(self, initfile, gas, resultsdir="dei_results/", unctype="", outdir=None, siteinfo=None, verbose=True):

		self.today = datetime.datetime.now().date()

		if initfile is None:
			sys.exit("initfile required.")

		self.initparam = ccg_readinit(initfile, siteinfo=siteinfo)
		if self.initparam is None:
			sys.exit("Init file '%s'not found." % initfile)

		if len(self.initparam['site']) == 0:
			sys.exit("ERROR: No initialization data found")

		choices = ['atmospheric', 'bias', 'network', 'analysis']
		if unctype not in choices:
			sys.exit("ERROR: Wrong unctype %s. Must be one of %s" % (unctype, choices))
		self.unctype = unctype

		self.outdir = outdir
		self.verbose = verbose
		self.gas = gas.lower()

		if not resultsdir.endswith("/"): resultsdir = resultsdir + "/"
		self.resultsdir = resultsdir

		if not os.path.exists(self.resultsdir):
			sys.exit("Results directory %s does not exist." % self.resultsdir)

		if self.outdir:
			if not self.outdir.endswith("/"): self.outdir = self.outdir + "/"
			self.outputdir = self.resultsdir + self.outdir
		else:
			self.outputdir = self.resultsdir

		# set default filter parameters. Can be chanegd with self.set_filter_param()
		self.npoly = 4#3
		self.nharm = 5
		self.cutoff1 = 667 #80
		self.cutoff2 = 667
		self.interval = 7
		self.tz = self.initparam['sync1']

		# Get the sync steps that were used.
		self.syncsteps = numpy.loadtxt(resultsdir + "syncsteps").tolist()
		self.nsyncsteps = len(self.syncsteps)

		self.nsites = len(self.initparam['siteinfo'])
		self.fyear = int(self.syncsteps[0])
		self.lyear = int(self.syncsteps[-1])
		self.nyears = self.lyear - self.fyear + 1
		self.years = [self.fyear + yr for yr in range(self.nyears)]

		# locations for latitude gradient.  -1 to 1 at 0.05 intervals
		self.nbins = 41
		self.latbins = [n * 0.05 - 1 for n in range(self.nbins)]

		# create a list of months for the time span used
		self.months = []
		for yr in range(self.fyear, self.lyear + 1):
			for month in range(1, 13):
				xres = ccg_dates.decimalDate(yr, month, 1)
				if xres <= self.syncsteps[-1]:
					dd = ccg_dates.decimalDate(yr, month, 15)
					if dd >= self.syncsteps[0]:
						self.months.append(dd)
		self.nmonths = len(self.months)

		# curve fit/filtering output types to include
		self.ccgvu = ['ftn', 'tr', 'gr', 'fsc', 'ssc', 'residf', 'residsc']

		# zone abbreviations
		self.zones = ['arctic', 'hsh', 'lsh', 'lnh', 'hnh', 'sh', 'nh', 'equ', 'gl', 'psh', 'tsh', 'tropics', 'tnh', 'pnh']

	#-----------------------------------------------------------------
	def set_filter_param(self, npoly=3, nharm=4, cutoff1=80, cutoff2=667, interval=7):
		""" set the parameters used by the ccg_filter curve fit """

		self.npoly = npoly
		self.nharm = nharm
		self.cutoff1 = cutoff1
		self.cutoff2 = cutoff2
		self.interval = interval


	#-----------------------------------------------------------------
	def _get_monthly_means(self, x, y):
		""" Calculate monthly means for array y, based on decimal dates in array x
		INPUT:
			x - decimal dates of data
			y - values of data

		RETURNS:
			numpy array with 4 columns of data;
				decimal dates,
				monthly averages,
				monthly standard deviation,
				number of points in month
			One row of data for each month in the x range.
		"""

		means = []
		dates = ccg_dates.dec2date(x)
		minyr = int(numpy.min(dates[:, 0]))
		maxyr = int(numpy.max(dates[:, 0]))
		for yr in range(minyr, maxyr+1):
			for month in range(1, 13):
				w = numpy.where((dates[:, 0] == yr) & (dates[:, 1] == month))
				n = w[0].size
				if n > 0:
					if n == 1:
						avg = numpy.average(y[w])
						stdv = -9.99
					else:
						avg = numpy.average(y[w])
						stdv = numpy.std(y[w], ddof=1)

					xres = ccg_dates.decimalDate(yr, month, 1)
					if xres <= self.syncsteps[-1]:
						dd = ccg_dates.decimalDate(yr, month, 15)
						if dd >= self.syncsteps[0]:
							means.append((xres, avg, stdv, n))

		return numpy.array(means)


	#-----------------------------------------------------------------
	def _get_annual_means(self, x, y):
		""" Calculate averages for each year from given data
		INPUT:
			x - decimal dates of data
			y - values of data

		RETURNS:
			numpy array of shape nyears, containing annual average for each year

		"""

		means = []

		for yr in self.years:
			a = numpy.where((x >= yr) & (x < yr+1))
			n = a[0].size
			if n > 0:
				avg = numpy.average(y[a])
				stdv = numpy.std(y[a], ddof=1)
			else:
				avg = DEFAULT
				stdv = -9.99
			means.append((yr, avg, stdv, n))


		return numpy.array(means)


	#-----------------------------------------------------------------
	def _write_file(self, filename, x, data, numrows, numbs, label, zonename=""):
		""" Write the results to a file.
		INPUT
			x - array of decimal dates for the data
			data - the data, either a dict with data for each zone,
			   or a single array.  Each zone data contains 2 columns,
			   the value and standard deviation.
			numrows - the length of the data and x
			numbs - number of bootstrap nums that were done.
			label - label to identify the type of data
			zonename - if data is a single array, (not a dict), then this is
				   the name to use instead of an actual zone name.
		"""

		f = open(filename, "w")
		if self.verbose:
			print("Creating", filename)

		f.write("Creation date: %s\n" % datetime.datetime.now().strftime("%d %b %Y"))
		f.write(" # of bs runs: %d\n" % numbs)
		f.write(" # of %s: %d\n" % (label, numrows))
		f.write("\n")
		f.write("%12s" % "STEP")
		if isinstance(data, dict):
			for s in self.zones:
				f.write("%21s" % s.upper())
			f.write("\n")
			f.write("            ")
			for i in range(len(self.zones)):
				f.write("       mean     sigma")
		else:
			f.write("%21s" % zonename)
			f.write("\n")
			f.write("            ")
			f.write("       mean     sigma")
		f.write("\n")

		for i in range(numrows):
			f.write("%12.6f" % x[i])
			if isinstance(data, dict):
				for zone in self.zones:
					f.write("%11.3f %9.3f" % (data[zone][0][i], data[zone][1][i]))
			else:
				f.write("%11.3f %9.3f" % (data[0][i], data[1][i]))

			f.write("\n")

		f.close()

	#-----------------------------------------------------------------
	def _savetxt(self, filename, x, y, data, header=None, useMonth=False):
		""" Write annual or monthly means to file. """

		f = open(filename, "w")
		if self.verbose:
			print("Creating", filename)

		if header is not None:
			f.write(header + "\n")

		numrows = len(x)
		for i in range(numrows):
			if useMonth:
				dt = data[0][i]
				(year, month, day, hour, minute, second) = ccg_dates.calendarDate(dt)
				f.write("%5d %4d %7.2f" % (year, month, data[1][i]))
			else:
				f.write("%5d %7.2f" % (int(data[0][i]), data[1][i]))

		f.close()


	#-----------------------------------------------------------------
	def _write_surface_file(self, filename, mbl_surface):
		""" Write a surface data array to file. Need to add dates as first column """

		s = numpy.empty((self.nsyncsteps, 1))
		s[:, 0] = self.syncsteps
		a = numpy.hstack( (s, mbl_surface) )
		format_str = "%13.8f" + " %8.3f" * self.nbins
		if self.verbose:
			print("Saving", filename)
		numpy.savetxt(filename, a, fmt=format_str)


	#-----------------------------------------------------------------
	def surface_unc(self, numbs):
		""" Combine multiple surface.mbl.gas files and calculate
		the standard deviation for each time step/lat bin across all files.
		"""

		results = numpy.empty( (numbs, self.nsyncsteps, self.nbins) )
		for i in range(numbs):

			filename = self.resultsdir + str(i+1) + "/surface.mbl." + self.gas
			if self.verbose: print("Reading", filename)

			a = numpy.loadtxt(filename)
			# remove date column from surface data
			results[i] = a[:, 1:]

		std = numpy.std(results, ddof=1, axis=0)

		filename = self.outputdir + "surface.mbl." + self.unctype + ".unc." + self.gas
		self._write_surface_file(filename, std)


	#-----------------------------------------------------------------
	def zonal_unc(self, numbs):
		""" calculate zonal uncertainties """

		annual_data = {}
		monthly_data_smooth = {}
		monthly_data_trend = {}
		zonal_data = {}
		coef_data = {}
		diff_data = {}

		ncoef = self.npoly + 2*self.nharm

		nccgvu = len(self.ccgvu)

		b = numpy.zeros((nccgvu, numbs, self.nsyncsteps))
#		a = numpy.empty((numbs, self.nyears))
		a = None  # we'll create this later depending on number of complete years
		c = numpy.empty((numbs, self.nmonths))
		d = numpy.empty((numbs, self.nmonths))
		e = numpy.empty((numbs, self.nsyncsteps))
		coef = numpy.empty((numbs, ncoef))
		diff = None

		zzdata = {}
		for i in range(nccgvu):
			zzdata[i] = {}

		for zone in self.zones:

			# read all bootstrap files for this zone
			for i in range(numbs):

				resultsdir = self.resultsdir + str(i+1) + "/"
				zonefile = resultsdir + "zone_" + zone + ".mbl." + self.gas

				if self.verbose:
					print("Reading", zonefile)

				zonedata = numpy.loadtxt(zonefile)
				dd = zonedata.T[0]
				mr = zonedata.T[1]


				# fit smooth curve to data
				filt = ccg_filter.ccgFilter(dd, mr, self.cutoff1, self.cutoff2, self.interval, self.npoly, self.nharm, timezero=self.tz)

				sc = filt.getSmoothValue(dd)
				ftn = filt.getFunctionValue(dd)
				tr = filt.getTrendValue(dd)
				gr = filt.getGrowthRateValue(dd)
				fsc = filt.getHarmonicValue(dd)
				jan1vals = filt.getTrendValue(self.years)
				ssc = sc - tr
				residf = mr - ftn
				residsc = mr - sc

				b[0][i] = ftn
				b[1][i] = tr
				b[2][i] = gr
				b[3][i] = fsc
				b[4][i] = ssc
				b[5][i] = residf
				b[6][i] = residsc

				# save zonal values
				e[i] = mr

				# save zonal annual averages
				means = self._get_annual_means(dd, sc)
				w = numpy.where(means.T[3] == STEPS_PER_YEAR)
				means = means[w]
				if a is None:
					a = numpy.empty((numbs, means.shape[0]))
					annual_years = means.T[0]
				a[i] = means.T[1]

				# save zonal monthly means of smooth curve
				mm = self._get_monthly_means(dd, sc)
				c[i] = mm.T[1]

				# save zonal monthly means of trend curve
				mm = self._get_monthly_means(dd, tr)
				d[i] = mm.T[1]

				# filter coefficients
				coef[i] = filt.params

				# jan1 to jan 1 annual increase
				# remove NaN's first then
				# drop the last value since it's last year - first year
				w = numpy.where(~numpy.isnan(jan1vals))
				jan1vals = jan1vals[w]
				diffs = numpy.roll(jan1vals, -1) - jan1vals
				diffs = diffs[0:-1]
				if diff is None:
					diff = numpy.empty((numbs, diffs.shape[0]))
				diff[i] = diffs


			# now average the bootstrap runs for this zone

			# averages of the zones
			m = numpy.average(e, axis=0)
			stdv = numpy.std(e, axis=0, ddof=1)
			zonal_data[zone] = (m, stdv)

			# annual means of the smooth curve
			m = numpy.average(a, axis=0)
			stdv = numpy.std(a, axis=0, ddof=1)
			annual_data[zone] = (m, stdv)

			# monthly means of smooth curve
			m = numpy.average(c, axis=0)
			stdv = numpy.std(c, axis=0, ddof=1)
			monthly_data_smooth[zone] = (m, stdv)

			# monthly means of trend curve
			m = numpy.average(d, axis=0)
			stdv = numpy.std(d, axis=0, ddof=1)
			monthly_data_trend[zone] = (m, stdv)

			# timestep means for each of the curve fit output
			for n in range(nccgvu):
				m = numpy.average(b[n], axis=0)
				stdv = numpy.std(b[n], axis=0, ddof=1)
				zzdata[n][zone] = (m, stdv)

			# filter coefficients
			m = numpy.average(coef, axis=0)
			stdv = numpy.std(coef, axis=0, ddof=1)
			coef_data[zone] = (m, stdv)

			# annual increase from jan 1 to jan 1
			m = numpy.average(diff, axis=0)
			stdv = numpy.std(diff, axis=0, ddof=1)
			diff_data[zone] = (m, stdv)



	# now write output files


		# hemispheric difference of trend
		nhtrend = numpy.array(zzdata[1]["nh"])
		shtrend = numpy.array(zzdata[1]["sh"])
		ihd = nhtrend - shtrend
		ihd[1] = numpy.sqrt(nhtrend[1]*nhtrend[1] + shtrend[1]*shtrend[1])
		filename = self.outputdir + "zonal.ihd." + self.gas + ".bs"
		self._write_file(filename, self.syncsteps, ihd, self.nsyncsteps, numbs, "steps", zonename="IHD")


		# polar hemispheric difference of trend
		pnhtrend = numpy.array(zzdata[1]["pnh"])
		pshtrend = numpy.array(zzdata[1]["psh"])
		iphd = pnhtrend - pshtrend
		iphd[1] = numpy.sqrt(pnhtrend[1]*pnhtrend[1] + pshtrend[1]*pshtrend[1])
		filename = self.outputdir + "zonal.iphd." + self.gas + ".bs"
		self._write_file(filename, self.syncsteps, iphd, self.nsyncsteps, numbs, "steps", zonename="IPHD")


		# zonal data
		filename = self.outputdir + "zonal.ave." + self.gas + ".bs"
		self._write_file(filename, self.syncsteps, zonal_data, self.nsyncsteps, numbs, "steps")

		# write data for all zones for each ccgvu curve
		for n, name in enumerate(self.ccgvu):
			filename = self.outputdir + "zonal." + name + "." + self.gas + ".bs"
			self._write_file(filename, self.syncsteps, zzdata[n], self.nsyncsteps, numbs, "steps")

		# annual averages and std. dev., skipping last year since it's incomplete
		# not true anymore, annual averages have only complete years now.  Changed Nov 2018 kwt
		filename = self.outputdir + "zonal.ann.ave.sc." + self.gas + ".bs"
		self._write_file(filename, annual_years, annual_data, annual_years.shape[0], numbs, "years")
#		self._write_file(filename, self.years, annual_data, self.nyears, numbs, "years")

		# monthly means from smooth curve
		filename = self.outputdir + "zonal.mm.sc." + self.gas + ".bs"
		self._write_file(filename, self.months, monthly_data_smooth, self.nmonths, numbs, "months")

		# montly means from trend curve
		filename = self.outputdir + "zonal.mm.tr." + self.gas + ".bs"
		self._write_file(filename, self.months, monthly_data_trend, self.nmonths, numbs, "months")

		# filter coefficients
		filename = self.outputdir + "zonal.coef." + self.gas + ".bs"
		self._write_file(filename, [i for i in range(ncoef)], coef_data, ncoef, numbs, "coefficients")

		# year to year increase of trend, skipping last year since it's incomplete
		# not true anymore, annual averages have only complete years now.  Changed Nov 2018 kwt
		filename = self.outputdir + "zonal.ann.inc.tr." + self.gas + ".bs"
		self._write_file(filename, annual_years, diff_data, annual_years.shape[0], numbs, "years")
#		self._write_file(filename, self.years, diff_data, self.nyears, numbs, "years")

		# now files with annual and monthly mean values

	#	header = "# Global Annual Average uncertainty estimates."
	#	header += "# Bootstrap file: %s" % filename
	#	header += "#"
	#	header += "# Today: %s" % self.today
	#	header += "#"
	#	header += "# year unc"
	#        filename = self.outputdir + "bs_gl_ann_ave_unc_" + self.gas + ".txt"
	#        self._savetxt(filename, self.years[0:-1], annual_data['gl'], header)
	#
	#
	#	header = "# Global Annual Increase uncertainty estimates."
	#	header += "# Bootstrap file: %s" % filename
	#	header += "#"
	#	header += "# Today: %s" % self.today
	#	header += "#"
	#	header += "# year unc"
	#	filename = self.outputdir + "bs_gl_ann_inc_unc_" + self.gas + ".txt"
	#	self._savetxt(filename, self.years[0:-1], diff_data['gl'], header)
	#
	#
	#	header = "# Global Monthly Mean uncertainty estimates from S(t)."
	#	header += "# Bootstrap file: %s" % filename
	#	header += "#"
	#	header += "# Today: %s" % self.today
	#	header += "#"
	#	header += "# year month unc"
	#	filename = self.outputdir + "bs_gl_mm_sc_unc_" + self.gas + ".txt"
	#	self._savetxt(filename, self.months, monthly_data_smooth['gl'], header, useMonths=True)
	#
	#	header = "# Global Monthly Mean uncertainty estimates from T(t)."
	#	header += "# Bootstrap file: %s" % filename
	#	header += "#"
	#	header += "# Today: %s" % self.today
	#	header += "#"
	#	header += "# year month unc"
	#	filename = self.outputdir + "bs_gl_mm_tr_unc_" + self.gas + ".txt"
	#	self._savetxt(filename, self.months, monthly_data_trend['gl'], header, useMonths=True)


	#-----------------------------------------------------------------
	def merge_unc(self):
		""" merge uncertainty values with result values.
		Requires that zonal_unc() has already been run.
		"""

		# formats for the columns in numpy.savetxt
		fmt = ['%14.8f', '%10.3f', '%10.3f']

		# or could use the zonal_data array from zonal_unc()

		#  created from zonal_unc().
		filename = self.outputdir + "zonal.ave." + self.gas + ".bs"
		zavg = numpy.loadtxt(filename, skiprows=6)

		for zn, zone in enumerate(self.zones):
			resultdir = self.resultsdir + "../"
			filename = resultdir + "zone_" + zone + ".mbl." + self.gas

			# the file has 2 columns: date and value
			zone_file = numpy.loadtxt(filename)

			# this is the column number where the uncertainty values for this zone are in the 'zonal' files
			zone_unc_column = zn*2 + 2

			# get the uncertainty values from the zonal.ave data
			# add the uncertainty as the third column to to zone data
			filename = self.outputdir + "zone_" + zone + ".mbl.unc." + self.gas
			if self.verbose:
				print("Creating", filename)
			unc = zavg.T[zone_unc_column]
			b = numpy.reshape(unc, (unc.size, 1))
			a = numpy.hstack((zone_file, b))
			numpy.savetxt(filename, a, fmt=fmt)


			#--------------------------
			# fit smooth curve to data
			dd = zone_file.T[0]
			mr = zone_file.T[1]
			filt = ccg_filter.ccgFilter(dd, mr, self.cutoff1, self.cutoff2, self.interval, self.npoly, self.nharm, timezero=self.tz)

			sc = filt.getSmoothValue(dd)
			ftn = filt.getFunctionValue(dd)
			tr = filt.getTrendValue(dd)
			gr = filt.getGrowthRateValue(dd)
			fsc = filt.getHarmonicValue(dd)
			ssc = sc - tr
			residf = mr - ftn
			residsc = mr - sc

			# these need to be in same order as self.ccgvu
			b = {}
			b[0] = ftn
			b[1] = tr
			b[2] = gr
			b[3] = fsc
			b[4] = ssc
			b[5] = residf
			b[6] = residsc

			for nv, name in enumerate(self.ccgvu):

				filename = self.outputdir + "zonal." + name + "." + self.gas + ".bs"
				bs_data = numpy.loadtxt(filename, skiprows=6)

				unc = bs_data.T[zone_unc_column]
				a = numpy.array( (bs_data.T[0], b[nv], unc) ).T

				filename = self.outputdir + "zone_" + zone + ".mbl." + name + ".unc." + self.gas
				if self.verbose:
					print("Creating", filename)
				numpy.savetxt(filename, a, fmt=fmt)


			# ------------
			# Now merge computed annual average from resultsdir with bootstrap uncertainty

			filename = resultdir + "zone_" + zone + ".mbl.ann.ave.sc." + self.gas
			zone_file = numpy.loadtxt(filename)
#			print(zone_file)
#			print(zone_file.shape)

			#; Read annual average bootstrap result file from bs_resultsdir.
			#  created from zonal_unc().
			filename = self.outputdir + "zonal.ann.ave.sc." + self.gas + ".bs"
			bs_data = numpy.loadtxt(filename, skiprows=6)

			unc = bs_data.T[zone_unc_column]
#			print(unc)
#			print(unc.shape)
			b = numpy.reshape(unc, (unc.size, 1))
#			print(b)
#			print(b.shape)
			a = numpy.hstack((zone_file, b))

			filename = self.outputdir + 'zone_' + zone + '.mbl.ann.ave.sc.unc.' + self.gas
			if self.verbose:
				print("Creating", filename)
			numpy.savetxt(filename, a, fmt=fmt)


			# ------------
			# Now merge computed annual increase from resultsdir with bootstrap uncertainty

			filename = resultdir + "zone_" + zone + ".mbl.ann.inc.tr." + self.gas
			zone_file = numpy.loadtxt(filename)

			#; Read annual average bootstrap result file from bs_resultsdir
			#  created from zonal_unc().
			filename = self.outputdir + "zonal.ann.inc.tr." + self.gas + ".bs"
			bs_data = numpy.loadtxt(filename, skiprows=6)

			unc = bs_data.T[zone_unc_column]
			b = numpy.reshape(unc, (unc.size,1))
			a = numpy.hstack((zone_file, b))

			filename = self.outputdir + 'zone_' + zone + '.mbl.ann.inc.tr.unc.' + self.gas
			if self.verbose:
				print("Creating", filename)
			numpy.savetxt(filename, a, fmt=fmt)


	#-----------------------------------------------------------------
	def run(self, numbs=1):
		""" Start the processing of bootstrap files """

		if self.outdir:
			if not os.path.exists(self.outputdir):
				os.makedirs(self.outputdir)

		# do the surface uncertainty first
		self.surface_unc(numbs)

		# other stuff
		self.zonal_unc(numbs)

		# merged files
		self.merge_unc()



####################################################################################
if __name__ == "__main__":

	initfile = "init.co2.flask.master.txt"
	bsdir = "/home/ccg/kirk/bs_dei/co2/flask/results.2017-07/bs_network"
#	bsdir = "/home/ccg/ed/data_extension/new_py/sf6/flask/results.2017-07/bs_network"

	ext = dei_bs(initfile, gas="co2", resultsdir=bsdir, unctype="network", outdir="3and4")
	ext.run(3)
