
# vim: tabstop=4 shiftwidth=4
"""
Data extension class for making marine boundary layer (mbl) surface grids.


Typical usage is something like:

ext = dei.dei(
		options.initfile,
		options.gas,
		resultsdir=options.bsdir,
		create_subdirs=True,
		anchor=options.anchor)

ext.make_input_files(options.bsdir)  # make text input files we can reuse for each bootstrap run
ext.set_source_type(ext.FILE)	     # tell dei to get input data from files
ext.modify_source(ext.ATMOS_UNC)     # modify the input data with atmospheric uncertainty
ext.run(options.numbs)               # run the data extension and save results

Ported from IDL code June 2017 - kwt

Methodology:
------------
	Read an init file with list of sites to use, and the time period.  Fit smooth
	curves to each site.  At each time step, fit a latitude gradient for sites
	designated as 'mbl' sites, and compute the value at 0.05 sin latitude steps.
	This gives a grid of time vs latitude.  Compute differences between the
	smooth curves and the grid, and then recalculate the latitude gradients using
	these values.  This gives final grid of time vs latitude.  Combine various values
	from this grid to get zonal data.  Gaps in the data are handled by substituting
	various values from the curve fits, or from function fits to the smooth-grid data.

Creation:
---------
	ext = dei.dei(initfile, gas, resultsdir, create_subdirs, anchor, quickfilter, stopdate, verbose)

Input Parameters
----------------
	initfile : string
	    path and name of initialization file to use
	gas : string
	    gas species to use, e.g. 'co2'
	resultsdir : string
	    directory name for where to put result files
	create_subdirs : boolean
	    Whether to create a subdirectory under the resultsdir for the results output.
	    Usually False on first initial run, then True for subsequent bootstrap runs
	anchor : boolean
	    Set to True to ignore the sync2 value from the initialization file,
	    and use current date instead. Also creates 'estimated' data for brw, kum, smo, spo
	    beyond the last data point up to current date.
	quickfilter: boolean
	    Perform a +/- 3 sigma outlier rejection from the smooth curve to the input data.
	    This is done only when the input source is from the database (die.DB).
	    Default is false.
	stopdate : float
	    Specify the ending date of input data to use.  Any available data after this date
	    is not used in the calculations.
	verbose : boolean
	    Set to True to get extra output messages while running.


Methods
----------

        set_data_source(inputdir)
	    Set the directory where input data files are located.

	make_input_files(bsdir)
	    Get flask data from database and create text files with the data. These can then
	    be reused for multiple bootstrap runs, instead of getting from database each time.

        set_source_type(ext.FILE)
	    Set where input data is coming from, either database or file. Can be either dei.FILE or dei.DB

        modify_source(ext.ATMOS_UNC)
	    Set modifications to input data before calculations.  Must be one of ATMOS_UNC, BIAS, NETWORK or CUSTOM

	set_data_callback(funcname)
	    Set a function callback that will be called if the source type is CUSTOM.
	    The function will be called with the input data lists,
	    and it should return 2 modified lists,
		e.g. x, y = self.data_callback(x0, y0)

        run(numbs)
	    Start the data extension calculations.  Input value sets the number
	    of bootstrap runs to perform.
	    Default is 1


Differences from IDL code:
---------------------------

- No scale adjustments.
- mbl is based on smooth curve to data.  idl code was based on trend + detrended curves to data.
- No 'exclusion' file for determining which sites to use for mbl surface.
- quickfilter rejects only 'preliminary' data, specified in database.  idl code rejected data at any time.
- A 'bias' bootstrap method.  A random 'bias' or offset is applied to a random time period of the data based on analysis date.


"""

from __future__ import print_function

import sys
import os
import datetime
import calendar
from math import sqrt
import shutil
import copy
import numpy

import ccg_dates
import ccg_filter
import ccg_db

from ccg_quickfilter import ccg_quickfilter
from ccg_pptfit import ccg_pptfit
import ccg_readinit
import ccg_data
import bs_network

STEPS_PER_YEAR = 48
WEEK = 1.0 / STEPS_PER_YEAR
RELAX = 8
DEFAULT = -999.999

class dei:
	""" Data extension class """

	#-----------------------------------------------------------------
	def __init__(self,
				initfile,
				gas,
				resultsdir="dei_results/",
				create_subdirs=False,
				anchor=False,
				quickfilter=False,
				stopdate=None,
				siteinfo=None,
				use_adate=False,
				autosync=False,
				verbose=True
		):

		# source types
		self.DB = 0
		self.FILE = 1

		# data modification types
		self.CUSTOM = 2
		self.ATMOS_UNC = 3
		self.BIAS = 4
		self.NETWORK = 5
		self.ANALYSIS = 6

		self.source_type = self.DB
		self.source_types = [self.DB, self.FILE]
		self.modify_types = [self.ATMOS_UNC, self.BIAS, self.CUSTOM, self.NETWORK, self.ANALYSIS]
		self.data_source = resultsdir
		self.data_source_mod = None
		self.verbose = verbose
		self.data_callback = None
		self.create_bs_subdir = create_subdirs
		self.quickfilter = quickfilter
		self.stopdate = stopdate
		self.anchor = anchor
		self.gas = gas.lower()
		if not resultsdir.endswith("/"): resultsdir = resultsdir + "/"
		self.resultsdir = resultsdir
		self.siteinfo = siteinfo
		self.ccgcrv = ['ftn', 'tr', 'gr', 'fsc', 'ssc', 'residf', 'residsc']
		self.use_adate = use_adate
		self.autosync = autosync

		# will need to include these in __init__ call sometime
		self.interp = True
		self.extrap = False

		# will hold curve fit results for each site
		self.fit_data = {}
		self.rsd_data = {}
		self.total_rsd = {}

		# locations for latitude gradient.  -1 to 1 at 0.05 intervals
		self.nbins = 41
		self.latbins = [n * 0.05 - 1 for n in range(self.nbins)]

		# now read the init file and set variables based on its content
		self._read_init_file(initfile)

		if stopdate is not None and anchor==False:
			if stopdate < self.initparam['sync2']:
				sys.exit("ERROR: stopdate is less than sync2.  Stopping.")


		# delete if necessary and create the results directory
		if os.path.exists(self.resultsdir):
			if self.verbose:
				print("Removing files from", self.resultsdir)
			shutil.rmtree(self.resultsdir)
		os.makedirs(self.resultsdir)

		self.weights = numpy.zeros((self.nsites, self.nyears))
		self.wtvec = numpy.zeros((self.nsites))


	#-----------------------------------------------------------------
	def _read_init_file(self, initfile):
		""" now read the init file and set variables based on its content """

		if initfile is None:
			raise ValueError("initfile required.")
		else:
			self.initfile = initfile

		self.initparam = ccg_readinit.ccg_readinit(initfile, extonly=True, siteinfo=self.siteinfo)
		if self.initparam is None:
			sys.exit("ERROR: Init file '%s' not found." % initfile)

		if len(self.initparam['site']) == 0:
			raise ValueError("ERROR: No initialization data found")

		if self.verbose:
			if self.interp:
				print("Interpolated data WILL be used.")
			if self.extrap:
				print("Extrapolated data WILL be used.")
			if not self.interp and not self.extrap:
				print("Extended records WILL NOT be used.")

		if self.anchor:
			self.initparam['sync2'] = ccg_dates.decimalDateFromDatetime(datetime.datetime.now())
			print("sync2 reset to", self.initparam['sync2'], "because anchor option was set.")

		#!!! tmp fix for ed's gl_unc_est runs
		# set sync2 to stopdate if use_date is True
		if self.stopdate is not None and self.use_adate:
			self.initparam['sync2'] = self.stopdate
			print("sync2 reset to", self.initparam['sync2'], "because stopdate and use_adate options were set")

		if self.autosync and not self.anchor:
			now = datetime.datetime.now()
			then = now.replace(month=now.month-4)
			(weekday, daysinmonth) = calendar.monthrange(then.year, then.month)
			sync2 = then.replace(day=daysinmonth, hour=0, minute=0, second=0, microsecond=0)
			self.initparam['sync2'] = ccg_dates.decimalDateFromDatetime(sync2)
			print("sync2 reset to", self.initparam['sync2'], "(", sync2, ") because autosync was True and anchor is off")

	#!!! NOTE
	# if bootstrap runs are made some time after the mbl run, and the anchor option is set,
	# the number of syncsteps can change.  Need some way to be able to use the same syncsteps
	# for a bootstrap only run that the mbl run used.

		# Create the sync steps, 48 steps per year.
		self.nsyncsteps = int((self.initparam['sync2'] - self.initparam['sync1']) * STEPS_PER_YEAR + 1)
		self.syncsteps = [WEEK*j + self.initparam['sync1'] for j in range(self.nsyncsteps)]

		self.nsites = len(self.initparam['siteinfo'])
		self.fyear = int(self.syncsteps[0])
		self.lyear = int(self.syncsteps[-1])
		self.nyears = self.lyear - self.fyear + 1
		self.years = [self.fyear + yr for yr in range(self.nyears)]


		# sin of latitude for each site
		self.sinlats = [self.initparam['siteinfo'][n]['sinlat'] for n in range(self.nsites)]


		# make up separate lists of only mbl sites
		list1 = []
		list2 = []
		for i, val in enumerate(self.initparam['mbl']):
			if val == 1:
				list1.append(self.initparam['siteinfo'][i]['sinlat'])
				list2.append(self.initparam['site'][i])

		# sort the mbl sites from south to north
		# create an array with the sin latitudes,
		# and a corresponding list with site codes
		idx = numpy.argsort(numpy.array(list1))
		self.mblpos = numpy.array(list1)[idx]
		self.mblsites = numpy.array(list2)[idx].tolist()
		self.nmblsites = len(self.mblsites)


	#-----------------------------------------------------------------
	def set_source_type(self, source_type):
		""" Set where the original data source will come from. """

		if source_type in self.source_types:
			self.source_type = source_type
		else:
			raise ValueError("Bad source type")


	#-----------------------------------------------------------------
	def set_data_callback(self, funcname):
		""" Set a function callback that will be called if the source type is CUSTOM.
		The function will be called with the input data lists,
		and it should return 2 modified lists,
			e.g. x, y = self.data_callback(x0, y0)
		"""

		self.data_callback = funcname

	#-----------------------------------------------------------------
	def set_data_source(self, dirname):
		""" If source_type == FILE, then set directory name
		where files are located. """

		self.data_source = dirname
		if not self.data_source.endswith("/"):
			self.data_source += "/"

	#-----------------------------------------------------------------
	def modify_source(self, modify_type):
		""" Set type of modification to input data, such as atmospheric uncertainty,
		analysis bias """

		if modify_type in self.modify_types:
			self.data_source_mod = modify_type
		else:
			raise ValueError("Bad modify type")

	#-----------------------------------------------------------------
	def set_bias_config(self, filename):
		""" Read in the file with values for the bias bootstrap """

		self.bias_values = ccg_readinit.read_bias_config(filename, self.gas)
	#	print(self.bias_values)

	#-----------------------------------------------------------------
	def set_analysis_config(self, filename):
		""" Read in the file with values for the analysis uncertainty bootstrap """

		self.analysis_values = ccg_readinit.read_analysis_config(filename, self.gas)
	#	print(self.analysis_values)

	#-----------------------------------------------------------------
	def _get_year_rsd(self, site, filt, x, y):
		""" Calculate residual standard deviation of the data about the
		smoothed curve for each year
		INPUT:
			site - site code
			x - decimal dates of original input data
			y - mixing ratios of original input data
			filt - ccg_filter object created with x, y data

		RETURNS:
			numpy array of shape nyears x 3, where columns are
			year, standard deviation, number of points

		"""


		rsd = []

		# calculate rsd only for years of interest. x and y contain
		# data for entire record, so subset the residuals.

		w = numpy.where( (x >= self.fyear) & (x < self.lyear+1) )

		smooth = filt.getSmoothValue(x)
		resid = y - smooth
		totalrsd = numpy.std(resid[w], ddof=1)
		totaln = resid[w].size
		if totaln > 6:
			self.total_rsd[site] = sqrt(totaln)/totalrsd
		else:
			self.total_rsd[site] = 0.0

		for yr in self.years:
			a = numpy.where( (x >= yr) & (x < yr+1) )
			n = a[0].size
			if n > 1:
				stdv = numpy.std(resid[a], ddof=1)
			else:
				stdv = DEFAULT
			rsd.append((yr, stdv, n))


		return numpy.array(rsd)


	#-----------------------------------------------------------------
	def _get_fit_data(self, filt, x, y, interval):
		""" Fit smooth curves to the data, and identify any gaps in the data.
		INPUT:
			filt - ccg_filter instance used to create smooth curves
			x, y - Input data used for making smooth curves
			interval - sampling interval used in making smooth curves

		RETURNS:
			numpy array of shape ntimesteps x 8, where columns are
			x, smooth, trend, function, detrended, number of points,
			0/1 for gaps, 0/1 for mbl gaps
		"""

		# Determine if there are any internal gaps in measurement record that exceed FILLGAP and MBLGAP.
		# Use this a bit later.
		#
		# There are two gap specifications.
		#
		# 1) There is a MBL gap length which applies to MBL sites and used
		# only for construction of the reference MBL matrix.  Note that MBL gap length
		# is identified in all site records when constructing the ".dat" files but is
		# only used for MBL sites when constructing MBL reference.
		#
		# 2) There is a FILL gap length which applies to all sites and identifies periods
		# where measurement interruptions should be filled with interpolated values derived
		# from data extension.

		xr = numpy.roll(x, -1)	# shift xd by one timestep backwards
		fillgaps = numpy.where( xr-x > self.initparam['fillgap'] * WEEK )
		mblgaps = numpy.where( xr-x > self.initparam['mblgap'] * WEEK )

		smooth_arr = filt.getSmoothValue(self.syncsteps)
		trend_arr = filt.getTrendValue(self.syncsteps)
		harm_arr = filt.getHarmonicValue(self.syncsteps)

		# create a numpy array to hold results
		data = numpy.zeros( (self.nsyncsteps, 8) )

		for stepnum, xval in enumerate(self.syncsteps):

			# Don't try to predict values from the smooth curve fit
			# that are prior to or beyond the measurement record.
			# Also ensure that the last predict timestep is less than
			# the last data point minus the interval.
			# Problems occur when smoothing residuals
			# if you try to predict a value that is not bracketed by
			# residuals (as determined by the interval).

			if xval < x[0] or xval > x[-1]-interval/365.0:
				smooth = DEFAULT
				trend = DEFAULT
				harm = DEFAULT
				detrend = DEFAULT
				num = 0
				gap = 1
				mblgap = 1
			else:
				smooth = smooth_arr[stepnum]
				trend = trend_arr[stepnum]
				harm = harm_arr[stepnum]
				detrend = smooth - trend
				gap = 0
				mblgap = 0

				# Determine number of actual data points within +/- WEEK/2 days
				# of synchronized timestep
				k = numpy.where( (x >= xval-WEEK/2.) & (x <= xval+WEEK/2.) )
				num = k[0].size

				# handle fill gaps
				if fillgaps[0].size > 0:
					for fgap in fillgaps[0].tolist():
						if xval > x[fgap] and xval < x[fgap+1]:
							gap = 1

				# handle mbl gaps
				if mblgaps[0].size > 0:
					for mgap in mblgaps[0].tolist():
						if xval > x[mgap] and xval < x[mgap+1]:
							mblgap = 1


			data[stepnum, 0] = xval
			data[stepnum, 1] = smooth
			data[stepnum, 2] = trend
			data[stepnum, 3] = harm
			data[stepnum, 4] = detrend
			data[stepnum, 5] = num
			data[stepnum, 6] = gap
			data[stepnum, 7] = mblgap

		return data


	#-----------------------------------------------------------------
	def _set_weights(self):
		""" Set the correct values for the weights.  This should only be called
		after self._get_year_rsd() has been called for all sites.
		"""

		# set the weights by year
		# Clip 'N', i.e., number of residuals
		# used in 'ccgvu' to calculate RSD of
		# smooth curve.  We clip 'N' at 2*48.
		# This implies that air sampled ~ twice
		# weekly are independent measurements.
		# Air sampled more frequently, i.e.,
		# continuous, are not independent
		# measurements.
		for site_index, site in enumerate(self.initparam['site']):
			if site in self.rsd_data:
				rsd = self.rsd_data[site]
				for i in range(self.nyears):
					(yr, stdv, n) = rsd[i]
					if n > 6:
						if n > 2*STEPS_PER_YEAR: n = 2*STEPS_PER_YEAR
						if stdv > 0:
							wgt = sqrt(n)/stdv
						else:
							wgt = 0.0
						self.weights[site_index, i] = wgt

				self.wtvec[site_index] = self.total_rsd[site]


		# get a list of weights for all years and all sites
		w = numpy.nonzero(self.weights)
		wgts = self.weights[w]
		if wgts.size == 0:
			print("No data found. Stopping.", file=sys.stderr)
			sys.exit()
			

		wgts = numpy.sort(wgts)
		# find the 70th% index and 5% index in the list
		# use the values at these indices as the max and min values
		maxclip = wgts[int(wgts.size*0.70)]
		minclip = wgts[int(wgts.size*0.05)]
		clipspan = maxclip - minclip

		# now reset any weights above or below the max or min,
		# and then scale all weights to a range of 2 to 10

		w = numpy.where( self.weights > maxclip )
		self.weights[w] = maxclip
		w = numpy.where( (self.weights < minclip) & (self.weights > 0) )
		self.weights[w] = minclip

		w = numpy.nonzero(self.weights)
		self.weights[w] = 2 + 8. * ((self.weights[w] - minclip)/clipspan)

		# set zero weights to 1
		w = numpy.where(self.weights == 0)
		self.weights[w] = 1.0

		# do same thing for overall weights by site

		w = numpy.nonzero(self.wtvec)
		wgts = self.wtvec[w]
		wgts = numpy.sort(wgts)
		maxclip = wgts[int(wgts.size*0.70)]
		minclip = wgts[int(wgts.size*0.05)]
		clipspan = maxclip - minclip
		w = numpy.where( self.wtvec > maxclip )
		self.wtvec[w] = maxclip
		w = numpy.where( (self.wtvec < minclip) & (self.wtvec > 0) )
		self.wtvec[w] = minclip
		w = numpy.nonzero(self.wtvec)
		self.wtvec[w] = 2 + 8. * ((self.wtvec[w] - minclip)/clipspan)

		# set zero weights to 1
		w = numpy.where(self.wtvec == 0)
		self.wtvec[w] = 1.0


	#-----------------------------------------------------------------
	def _calc_meridional(self, fitdata, column=-1, column_name="", outputfile=None):
		""" Calculate latitude gradients using pptfit.
		INPUT:
			fitdata -  a dict with data arrays for each site.
				   The shape of the array is ntimesteps x ncolumns
			column - The correct column number to use from the data array.
				 If column is -1, then use the appropriate columns from the
				 self.fit_data array.
			column_name - A name to use for the given column number. For information
				  purposes only.
			outputfile - File name for where to write the values that go into
				 the latitude gradient fit at each time step.

		RETURNS:
		An array with latitude gradient values at each latitude bin, for
		every time step, i.e. ntimesteps x nlatbins

		"""

		if self.verbose:
			print("Calculating meridional gradient for ", column_name)

		if outputfile:
			try:
				fpout = open(outputfile, "w")
			except IOError as e:
				sys.exit("ERROR: Cannot create output file %s: %s" % (outputfile, e))


		d = numpy.zeros( (self.nsyncsteps, self.nbins) )
		for i, step in enumerate(self.syncsteps):

			y = []
			x = []
			w = []
			s = []

			# only use mbl sites for latitude gradient
			for (site, sinlat) in zip(self.mblsites, self.mblpos):
				if site not in fitdata: continue
				data = fitdata[site]

				# get data for this site and timestep
				# use trend + harmonics value if timestep is in an mbl gap
				if column == -1:
					mblgap = data[i, 7]
					if mblgap:
						val = data[i, 2] + data[i, 3] # trend + harmonics
					else:
						val = data[i, 1]              # smooth curve
				else:

					# figure out value to use based on self.interp and self.extrap
					val = self._get_point_val(site, data, i, column)


				if val > -999:
					y.append(val)
					x.append(sinlat)

					# get weight for this site.
					# Use overall weight when column == -1, yearly weight when column != -1
					idx = self.initparam['site'].index(site)
					if column == -1:
						wt = self.wtvec[idx]
					else:
						gap = self.fit_data[site][i, 6]
						if gap:
							wt = 1.0
						else:
							yr = int(step)-self.fyear
							wt = self.weights[idx, yr]
					w.append(wt)

					s.append(site)

			#   ;Before making the fit to the meridional
			#   ;distribution, look for multiple contributions
			#   ;from a single sampling location.  For example,
			#   ;both spo_00 and spo_01 should not get their
			#   ;full weight as SPO would get nearly twice the
			#   ;weighting.  Use the following scheme for multiple
			#   ;records at the same sampling location.
			#   ;
			#   ;If the sum of the weights is <= 10 then do nothing
			#   ;If the sum of the weights is >  10 then do the following
			#   ;
			#   ;Divide each weight by the sum of the weights divided by 10.
			#   ;Because the maximum weight is 10.  The sum of the adjusted
			#   ;weights will sum to 10 and no more.
			#   ;
			#   ;Code added on January 21, 1998 - kam
			# Needed only when doing network bootstrap, since we can have multiple
			# entries at one location  - kwt

			x = numpy.array(x)
			y = numpy.array(y)
			w = numpy.array(w)

			unique_positions = numpy.unique(x)
			if unique_positions.size != x.size:
				for z in range(unique_positions.size):
					j = numpy.where(x == unique_positions[z])
					wsum = numpy.sum(w[j])
					if wsum > 10:
						w[j] = 10.0 * w[j]/wsum


			# compute latitude gradient and
			# get interpolated latitude gradient values at each latbin location
			pptf = ccg_pptfit(x, y, w)
			d[i] = pptf.predict(self.latbins)

			if outputfile:
				fpout.write("%12.6f %5d\n" % (step, len(x)))
				for j in range(len(x)):
					fpout.write("%12.6f %12.4f %12.2f %12s\n" % (x[j], y[j], w[j], s[j]))

		if outputfile:
			fpout.close()

		return d


	#-----------------------------------------------------------------
	def _get_point_val(self, site, data, i, column):
		""" Determine if we use the extended data record for this timestep and site.
		This depends on the values of self.interp (using interpolated data in gaps)
		and self.extrap (use extrapolated data before and after data record).
		"""

		gap = self.fit_data[site][:, 6]
		a = numpy.where(gap == 0)
		first_data_point = a[0][0]
		last_data_point = a[0][-1]

		# use both interpolated and extrapolated data
		if self.interp and self.extrap:
			val = data[i, column]

		# use only interpolated data
		elif self.interp is True and self.extrap is False:
			if i < first_data_point or i > last_data_point:
				val = DEFAULT
			else:
				val = data[i, column]

		# use only extrapolated data
		elif self.interp is False and self.extrap is True:
			if i < first_data_point or i > last_data_point:
				val = data[i, column]
			else:
				if gap[i]:
					val = DEFAULT
				else:
					val = data[i, column]

		# interp=false, extrap=false. Don't use extended data anywhere
		else:
			if gap[i]:
				val = DEFAULT
			else:
				val = data[i, column]

		return val


	#-----------------------------------------------------------------
	def _fill_external_gaps(self, xsc, ysc, d_t, gaps):
		"""
		# -----------------------------
		# FILLING EXTERNAL GAPS IN D(t)
		# -----------------------------
		#
		# Linear interpolate across a RELAX week time period
		#
		# (a)    from D(t0) to d(t0-RELAX) then use d(t) from t0-RELAX+1
		#       to beginning of synch. period.
		# or
		#
		# (b)    from D(tn) to d(tn+RELAX) then use d(t) from tn+RELAX+1
		#       to end of synch. period.
		#
		"""

		result = d_t

		# Identify pointers to first and last smooth values
		j = numpy.where(gaps == 0)
		ptr_min_sc = numpy.min(j)
		ptr_max_sc = numpy.max(j)

		# Pointers to first and last sync dates
		ptr_min_sync = 0
		ptr_max_sync = self.nsyncsteps - 1

		# Does an external gap exist at BEGINNING of record?
		if ptr_min_sync < ptr_min_sc:

			# D(t) for beginning smooth curve value
			x0 = xsc[ptr_min_sc]
			y0 = d_t[ptr_min_sc]

			# Determine number of weeks in beginning external gap
			nweeks = ptr_min_sc - ptr_min_sync
			if nweeks < RELAX:
				# Identify d(t) value at ptr_min_sync time step
				xp = xsc[ptr_min_sync]
				yp = d_t[ptr_min_sync]

				m = (yp - y0) / (xp - x0)
				for i in range(1, nweeks+1):
					result[ptr_min_sc-i] = y0 + m * (xsc[ptr_min_sc-i] - x0)


			else:
				# Identify d(t) value at ptr_min_sc-RELAX time step
				xp = xsc[ptr_min_sc-RELAX]
				yp = d_t[ptr_min_sc-RELAX]

				m = (yp - y0) / (xp - x0)
				for i in range(1, RELAX+1):
					result[ptr_min_sc-i] = y0 + m * (xsc[ptr_min_sc-i] - x0)


		# Does an external gap exist at END of record?
		if ptr_max_sync > ptr_max_sc:

			# D(t) for last smooth curve value
			x0 = xsc[ptr_max_sc]
			y0 = d_t[ptr_max_sc]

			# Determine number of weeks in ending external gap
			nweeks = ptr_max_sync - ptr_max_sc
			if nweeks < RELAX:

				# Identify d(t) value at ptr_max_sync time step
				xp = xsc[ptr_max_sync]
				yp = d_t[ptr_max_sync]

				m = (yp - y0) / (xp - x0)
				for i in range(1, nweeks+1):
					result[ptr_max_sc+i] = y0 + m * (xsc[ptr_max_sc+i] - x0)

			else:
				# Identify d(t) value at ptr_max_sc+RELAX time step
				xp = xsc[ptr_max_sc + RELAX]
				yp = d_t[ptr_max_sc + RELAX]

				m = (yp -y0) / (xp - x0)
				for i in range(1, RELAX+1):
					result[ptr_max_sc+i] = y0 + m * (xsc[ptr_max_sc+i] - x0)

		return result


	#-----------------------------------------------------------------
	def _fill_internal_gaps(self, xsc, ysc, d_t, gaps):
		"""
		# -----------------------------
		# FILLING INTERNAL GAPS IN D(t)
		# -----------------------------
		#
		# IF internal gap is < 2*RELAX weeks THEN use
		# linear interpolate across the gap in D(t)
		#
		# IF internal gap is >= 2*RELAX weeks THEN
		#
		# Linear interpolate across a RELAX week time period
		#
		# (1)    from D(ti) to d(ti+RELAX) and from D(tii) to d(tii-RELAX)
		# (2)    fill gaps between d(ti+RELAX) and d(tii-RELAX) with d(t).
		"""


		result = d_t

		# Identify pointers to first and last smooth values
		j = numpy.where(gaps == 0)
		ptr_min_sc = numpy.min(j)
		ptr_max_sc = numpy.max(j)

		# Do internal gaps exist in record?
		# find all default values after first default value and before last default value
		ptr_intgaps = numpy.where( (gaps == 1) & (xsc > xsc[ptr_min_sc]) & (xsc < xsc[ptr_max_sc]) )[0]
		if ptr_intgaps.size:
			m = ptr_intgaps - numpy.roll(ptr_intgaps, -1)
			w = numpy.where(m != -1)
			ptr2 = ptr_intgaps[w]   # end of gap

			m = ptr_intgaps - numpy.roll(ptr_intgaps, 1)
			w = numpy.where(m != 1)
			ptr1 = ptr_intgaps[w]   # start of gap

			ngaps = ptr2.size

			for i in range(ngaps):
				nweeks = ptr2[i] - ptr1[i] + 1

				# D(t) for smooth curve value before beginning of gap
				x1 = xsc[ptr1[i]-1]
				y1 = d_t[ptr1[i]-1]

				# D(t) for first smooth curve value after ending of gap
				x2 = xsc[ptr2[i]+1]
				y2 = d_t[ptr2[i]+1]

				if nweeks < 2*RELAX:
					m = (y2 - y1) / (x2 - x1)
					for j in range(1, nweeks+1):
						result[ptr1[i]+j-1] = y1 + m * (xsc[ptr1[i]+j-1] - x1)
				else:
					# Identify d(t) value at ptr1+RELAX-1 time step
					xp = xsc[ptr1[i]+RELAX-1]
					yp = d_t[ptr1[i]+RELAX-1]

					# interpolate first relax number of weeks from actual value to function value
					m = (yp -y1) / (xp - x1)
					for j in range(1, RELAX+1):
						result[ptr1[i]+j-1] = y1 + m * (xsc[ptr1[i]+j-1] - x1)

					# Identify d(t) value at ptr2-RELAX+1 time step
					xp = xsc[ptr2[i]-RELAX+1]
					yp = d_t[ptr2[i]-RELAX+1]

					# interpolate last relax number of weeks from function value to actual value
					m = (yp - y2) / (xp - x2)
					for j in range(1, RELAX+1):
						result[ptr2[i]-j+1] = y2 + m * (xsc[ptr2[i]-j+1] - x2)

					# other time steps remain at function value

		return result


	#-----------------------------------------------------------------
	def _calc_mbl_smooth_differences(self, smfits):
		"""
		# Here is where we fill gaps in the difference
		# climatology.  This procedure was modified to
		# minimize discontinuities at external gap transitions
		# and to improve the "average behavior" assumption for
		# internal gaps greater than 2*RELAX weeks length.
		#
		# Here is the idea.
		#
		# Notation.  D(t) is the difference distribution, that is,
		#            smooth curve - latitude gradient curve value at latitude of site.
		#          d(t) is the fit to D(t)
		#          t is in weeks (7.6 days)
		#          t0=time step of first value in D(t)
		#          tn=time step of last value in D(t)
		#          ti=time step of last value in D(t) before start of internal gap
		#          tii=time step of first value in D(t) at end of internal gap
		#
		# INPUT:
		#    smfits - 2d smooth curve values, ntimesteps x nlatbins
		#
		# RETURNS:
		#    array with 3 columns of data: mbl, diff, combined mbl+diff for each site
		"""

		mbl = numpy.zeros( (self.nsyncsteps, self.nsites) )

		# create a time series of mbl value - smooth curve value
		for stepnum in range(self.nsyncsteps):

			# find the value of the latitude gradient at the latitude of the sites
			mbl[stepnum] = numpy.interp(self.sinlats, self.latbins, smfits[stepnum])


		datarr = {}
		for n, site in enumerate(self.initparam['site']):
			if self.verbose: print("Calculating difference for", site)

			if site not in self.fit_data: continue

			xsc = self.fit_data[site][:, 0]       # dates
			ysc = self.fit_data[site][:, 1]       # smooth curve fit
			gaps = self.fit_data[site][:, 6]       # 1 for when time is in gap, 0 otherwise

			w = numpy.where(gaps == 0)
			# need at least 5 points for ccgFilter call to work
			if w[0].size < 5: 
				if self.verbose:
					print("Warning: Not enough data points for difference. Skipping...")
				continue

			mbl_site = mbl[:, n]

			xdiff = xsc[w]
			ydiff = ysc[w] - mbl_site[w]      # smooth curve - latitude gradient value at latitude of site

			# fit function to time series of smooth - mbl differences
			diff_filt = ccg_filter.ccgFilter(xdiff, ydiff, 80, 667, 7, 1, 2, self.initparam['sync1'])
			d_t = diff_filt.getFunctionValue(xsc)


			# Overwrite all d(t) values with D(t) values
			# This leaves gaps with the function value, difference value everywhere else
			a = numpy.searchsorted(xsc, xdiff) 	# find indices in xsc where all values of xdiff are located
			d_t[a] = ydiff				# replace values in d_t with those in ydiff where xdiff=xsc

			d_t = self._fill_external_gaps(xsc, ysc, d_t, gaps)
			d_t = self._fill_internal_gaps(xsc, ysc, d_t, gaps)

			# store mbl, d_t, and combined mbl+d_t
			# reset combined values to default where mbl is default
			datarr[site] = numpy.array( (mbl_site, d_t, mbl_site + d_t) ).T
			w = numpy.where(mbl_site < -900)
			datarr[site][:, 2][w] = DEFAULT


		return datarr


	#-----------------------------------------------------------------
	def _make_zonal_data(self, mbl_surface):
		"""
		# zonal averages
		# Pieter's computation goes as follows:
		# To calculate HSH, for example, which represents bands 0-10,
		# you sum 1-9 and add mean of 0 and 10, and then divide
		# the mess by 10.

		# HSH      ->   -90    to   -30    high southern hemisphere
		# LSH      ->   -30    to     0    low southern hemisphere
		# LNH      ->     0    to   +30    low northern hemisphere
		# HNH      ->   +30    to   +90    high northern hemisphere
		# SH       ->   -90    to     0    southern hemisphere
		# NH       ->     0    to   +90    northern hemisphere
		# GL       ->   -90    to   +90    global
		# EQU      ->   -14.5  to   +14.5  equatorial
		#
		# PSH      ->   -90    to   -53.1  polar southern hemisphere
		# TSH      ->   -53.1  to   -17.5  temperate southern hemisphere
		# TROPICS  ->   -17.5  to   +17.5  tropics
		# TNH      ->   +17.5  to   +53.1  temperate northern hemisphere
		# PNH      ->   +53.1  to   +90    polar northern hemisphere
		#
		# ARCTIC   ->   +58.2  to   +90    arctic
		"""

		# make keys and range for every zone, but sh, nh, global will be handled differently
		self.zones = {
			'hsh': (0, 10),
			'lsh': (10, 20),
			'lnh': (20, 30),
			'hnh': (30, 40),
			'equ': (15, 25),
			'psh': (0, 4),
			'tsh': (4, 14),
			'tropics': (14, 26),
			'tnh': (26, 36),
			'pnh': (36, 40),
			'arctic': (37, 40),
			'sh': (0, 20),
			'nh': (20, 40),
			'gl': (0, 40)
		}
		zone = {}

		# calculate zonal averages at each timestep
		for z in self.zones:
			zone[z] = numpy.empty( (self.nsyncsteps, 2) )
			zone[z][:, 0] = self.syncsteps

			lower_bin, upper_bin = self.zones[z]
			nbins = upper_bin - lower_bin
			sum_middle = numpy.sum(mbl_surface[:, lower_bin+1:upper_bin], axis=1)
			avg_edges = numpy.mean([mbl_surface[:, lower_bin], mbl_surface[:, upper_bin]], axis=0)
			zone[z][:, 1] = (sum_middle + avg_edges) / nbins

		# these zones are handled differently
		# ken's idl code had these lines in for sh, nh, and gl.
		# But I don't see a difference from what's calculated above.
	#	self.zone['sh'][:, 1] = (self.zone['lsh'][:, 1] + self.zone['hsh'][:, 1]) / 2.
	#	self.zone['nh'][:, 1] = (self.zone['lnh'][:, 1] + self.zone['hnh'][:, 1]) / 2.
	#	self.zone['gl'][:, 1] = (self.zone['sh'][:, 1] + self.zone['nh'][:, 1]) / 2.
		zone['equ'][:, 1] = numpy.mean( mbl_surface[:, 15:26], axis=1 )	# why is this handled differently from other zones?

		return zone


	#-----------------------------------------------------------------
	def _make_zone_ann_ave(self, zone):
		""" Make annual averages and annual increase for each zone. """

		# calculate annual averages for each zone

		self.zone_ann_avg = {}
		self.zone_ann_increase = {}

		for z in self.zones:
			tmp = []
			tmp2 = []
			data = zone[z]
			filt = ccg_filter.ccgFilter(data.T[0], data.T[1])
			sm = filt.getSmoothValue(data.T[0])
			tr = filt.getTrendValue(self.years)

			for year in self.years:
				w = numpy.where( (data.T[0] >= year) & (data.T[0] < year+1) & (data.T[1] > -999) )
				avg = numpy.mean(sm[w])
				jan1val = tr[year-self.fyear]  # get value of trend at jan 1 for each year
				tmp.append((year, avg, len(w[0])))
				tmp2.append((year, jan1val))

			# only keep annual means for a complete year
			a = numpy.array(tmp)
			w = numpy.where(a.T[2] == STEPS_PER_YEAR)
			ann_avgs = a[w]

			# remove Nan from jan1 values
			# jan1 val has Nan if outside the domain
			a = numpy.array(tmp2)
			w = numpy.where(~numpy.isnan(a.T[1]))
			a = a[w]
			years = a.T[0]

			diff = numpy.roll(a.T[1], -1) - a.T[1]	# jan1 to jan1 difference
			diff = diff[0:-1]  # drop the last one (which is first year - last year)
			years = years[0:-1]

			# delete the n column from ann avgs before saving
			ann_avgs = numpy.delete(ann_avgs, 2, 1)
			self.zone_ann_avg[z] = ann_avgs
			self.zone_ann_increase[z] = numpy.array( (years, diff) ).T


	#-----------------------------------------------------------------
	def _parse_surface(self, surface, bsdir):
		""" Make a surface file for each of the curve components (gr, fsc, tr...) """

		crvsurface = {}
		# create a smooth curve for each latitude bin,
		# get growth rate values at each time step
		for crvtype in self.ccgcrv:
			crvsurface[crvtype] = numpy.zeros( (surface.shape) )

		short_cutoff = 80
		long_cutoff = 667
		interval = 7
		npoly = 3
		nharm = 4
		tz = self.initparam['sync1']

		dd = self.syncsteps
		for i in range(self.nbins):
			mr = surface[:, i]

			filt = ccg_filter.ccgFilter(dd, mr, short_cutoff, long_cutoff, interval, npoly, nharm, timezero=tz)

			sc = filt.getSmoothValue(dd)
			ftn = filt.getFunctionValue(dd)
			tr = filt.getTrendValue(dd)
			gr = filt.getGrowthRateValue(dd)
			fsc = filt.getHarmonicValue(dd)
			ssc = sc - tr
			residf = mr - ftn
			residsc = mr - sc

			crvsurface["gr"][:, i] = gr
			crvsurface["tr"][:, i] = tr
			crvsurface["fsc"][:, i] = fsc
			crvsurface["ftn"][:, i] = ftn
			crvsurface["ssc"][:, i] = ssc
			crvsurface["residf"][:, i] = residf
			crvsurface["residsc"][:, i] = residsc


		for crvtype in self.ccgcrv:
			outputfile = bsdir + "surface.mbl." + crvtype + "." + self.gas
			self._write_surface_file(outputfile, crvsurface[crvtype])

		return crvsurface


	#-----------------------------------------------------------------
	def _ext_header(self, site, siteinfo):
		""" Generate the strings for the header of the _fits file.
		Also used for other output files, but not all header lines are used.
		"""

		alt = siteinfo['elev'] + float(siteinfo['intake_ht'])
		now = datetime.datetime.now()
		sdate = self.initparam['sync1']
		edate = self.initparam['sync2']
		t1 = ccg_dates.datetimeFromDecimalDate(sdate)
		t2 = ccg_dates.datetimeFromDecimalDate(edate)

		nlines = self.nsyncsteps

		header = []
		header.append("# %s" % site.upper())
		header.append("#")
		header.append("# NOAA Global Monitoring Division, United States")
		header.append("# %s" % siteinfo['strategy_name'])
		header.append("#")
		if 'platform_name' in siteinfo.keys():
			header.append("# %s" % siteinfo['platform_name'])
		header.append("# %s, %s" % (siteinfo['site_name'], siteinfo['site_country']))
		if 'agency_name' in siteinfo.keys():
			header.append("# %s" % siteinfo['agency_name'])
		header.append("#")
		header.append("#                  lat                 lon           alt(masl)     lst2utc")
		header.append("#               %6.2f              %6.2f                %4d         %3d" % (siteinfo['lat'], siteinfo['lon'], alt, siteinfo['lst2utc']))
		header.append("#")
		header.append("# Creation Date:  %s" % now.strftime("%a %b %d %H:%M:%S %Y"))
		header.append("#")
		header.append("# Time Period:  %s thru %s" % (t1.strftime("%Y-%m-%d"), t2.strftime("%Y-%m-%d")))
		header.append("#")
		header.append("# Number of rows after column header:     %d" % nlines)
		header.append("#")

		return header


	#-----------------------------------------------------------------
	def _write_syncsteps(self, filename):
		""" Write the time steps to file """

		numpy.savetxt(filename, self.syncsteps, fmt="%12.6f")


	#-----------------------------------------------------------------
	def _write_lat_info(self, filename):
		""" Write latitude info for each site to a file """

		try:
			f = open(filename, "w")
		except OSError as e:
			print("Can't open file %s for writing. %s" % (filename, e), file=sys.stderr)
			return

		for n in range(self.nsites):
			lat = self.initparam['siteinfo'][n]['lat']
			sinlat = self.initparam['siteinfo'][n]['sinlat']
			site = self.initparam['site'][n]

			f.write("%12s %12.4f %12.8f\n" % (site, lat, sinlat))

		f.close()


	#-----------------------------------------------------------------
	def _write_fit_file(self, outputfile, site, data):
		""" Write the fit data to file, one file per site. """

		if self.verbose:
			print("Saving", outputfile)

		try:
			fpout = open(outputfile, "w")
		except IOError as e:
			sys.exit("ERROR: Cannot create output file %s: %s" % (outputfile, e))

		idx = self.initparam['site'].index(site)
		header = self._ext_header(site, self.initparam['siteinfo'][idx])
		fpout.write("\n".join(header))
		fpout.write("\n")
		fpout.write("#         UTC    S(t) mbl        S(t)        T(t)        H(t)   S(t)-T(t)           N \n")

		for i in range(self.nsyncsteps):
			date = data[i, 0]
			smooth = data[i, 1]
			trend = data[i, 2]
			function = data[i, 3]
			detrend = data[i, 4]
			num = data[i, 5]
			gap = data[i, 6]
			mblgap = data[i, 7]
			if mblgap:
				smooth_mbl = DEFAULT
			else:
				smooth_mbl = smooth
			if gap:
				smooth = DEFAULT

			fpout.write("%13.6f %11.4f %11.4f %11.4f %11.4f %11.4f %11d\n" % (date, smooth_mbl, smooth, trend, function, detrend, num))

		fpout.close()


	#-----------------------------------------------------------------
	def _write_summary_file(self, outputfile, site, rsd, filt):
		""" Write summary file, which is filter statistics and annual rsd values. """

		try:
			fpout = open(outputfile, "w")
		except OSError as e:
			print(e)
			sys.exit("ERROR: Cannot create output file %s" % outputfile)

		idx = self.initparam['site'].index(site)
		header = self._ext_header(site, self.initparam['siteinfo'][idx])
		fpout.write("\n".join(header[0:-4]))
		fpout.write("\n")
		fpout.write(filt.stats())
		fpout.write("\n")
		fpout.write("RSD by year about S(t):\n")

		for (yr, stdv, n) in rsd:
			fpout.write("%12d %11.3f %12d\n" % (yr, stdv, n))

		fpout.close()


	#-----------------------------------------------------------------
	def _write_dat_file(self, outputfile, x, y):
		""" Write the input data to file. """

		if self.verbose: print("Saving", outputfile)

		a = numpy.array((x, y))
		numpy.savetxt(outputfile, a.T, fmt="%.9f %.3f")


	#-----------------------------------------------------------------
	def _write_weights_files(self, resultsdir):
		""" Write weight data for all sites to file """

		for site in self.initparam['site']:
			outputfile = resultsdir + site + "_wts." + self.gas
			try:
				fpout = open(outputfile, "w")
			except IOError as e:
				sys.exit("ERROR: Cannot create output file %s: %s" % (outputfile, e))

			site_index = self.initparam['site'].index(site)
			header = self._ext_header(site, self.initparam['siteinfo'][site_index])
			fpout.write("\n".join(header[0:-2]))
			fpout.write("\n")
			fpout.write("# Number of rows after column header:     %d\n" % (self.nyears + 1))
			fpout.write("# year             weight\n")
			fpout.write("  %d.%d %13.3f\n" % (self.fyear, self.lyear, self.wtvec[site_index]))

			for year in self.years:
				fpout.write("%6d %18.3f\n" % (year, self.weights[site_index, year-self.fyear]))

			fpout.close()


	#-----------------------------------------------------------------
	def _write_ext_files(self, resultsdir, datarr):
		""" Write extended data for all sites to file """

		for site in self.initparam['site']:
			if site not in datarr: continue
			mbl = datarr[site][:, 0]
			d_t = datarr[site][:, 1]
			mbl_dt = datarr[site][:, 2]
			xsc = self.fit_data[site][:, 0]       # dates
			ysc = self.fit_data[site][:, 1]       # smooth curve fit to data
			gap = self.fit_data[site][:, 6]       # 1 where gaps in data
			outputfile = resultsdir + site + "_ext." + self.gas

			if self.verbose: print("Saving", outputfile)

			try:
				fpout = open(outputfile, "w")
			except IOError as e:
				sys.exit("ERROR: Cannot create output file %s: %s" % (outputfile, e))

			idx = self.initparam['site'].index(site)
			header = self._ext_header(site, self.initparam['siteinfo'][idx])
			fpout.write("\n".join(header))
			fpout.write("\n")
			fpout.write("#         UTC        S(t)      REF(t)        diff  REF + diff\n")

			for i in range(self.nsyncsteps):
				if gap[i]:
					yval = DEFAULT
				else:
					yval = ysc[i]
				fpout.write("%13.6f %11.4f %11.4f %11.4f %11.4f\n" % (xsc[i], yval, mbl[i], d_t[i], mbl_dt[i]))

			fpout.close()


	#-----------------------------------------------------------------
	def _write_surface_file(self, filename, mbl_surface):
		""" Write a surface data array to file. Need to add dates as first column """

		s = numpy.empty((self.nsyncsteps, 1))
		s[:, 0] = self.syncsteps
		a = numpy.hstack( (s, mbl_surface) )
		format_str = "%13.6f" + " %8.3f" * self.nbins
		if self.verbose:
			print("Saving", filename)
		numpy.savetxt(filename, a, fmt=format_str)


	#-----------------------------------------------------------------
	def _write_zonal_files(self, resultsdir, zone, name=None):
		""" Write zonal data to text files. """

		for z in self.zones:
			if name is None:
				filename = resultsdir + 'zone_' + z + '.mbl.' + self.gas
			else:
				filename = resultsdir + 'zone_' + z + '.mbl.' + name + "." + self.gas
			if self.verbose:
				print("Saving", filename)
			numpy.savetxt(filename, zone[z], fmt=['%14.7f', '%8.3f'])


	#-----------------------------------------------------------------
	def make_input_files(self, dirname):
		""" Get data from database and write to text file for each site.
		This will allow us to reuse the input data without querying the
		database each time. """

		for site in self.initparam['site']:

			outputfile = dirname + site + "_dat." + self.gas
			if self.verbose:
				print("Creating data file ", outputfile)

			dd, mr = ccg_data.ccg_data(site, self.gas, stopdate=self.stopdate, use_adate=self.use_adate)
			if len(dd) == 0: continue

			if max(dd) - min(dd) < self.initparam['rlm']:
				print("In make_input_files", site, "has less than", self.initparam['rlm'], "years of data. Skipping...", file=sys.stderr)
				continue

			if self.quickfilter:
				dd, mr = self._get_quickfilter(dd, mr, site)

			self._write_dat_file(outputfile, dd, mr)

	#-----------------------------------------------------------------
	def _get_input_data(self, site):
		""" Get the data for the site.  Apply any modifications to the data """

		# get input data
		if self.source_type == self.DB:
			dd, mr = ccg_data.ccg_data(site, self.gas, stopdate=self.stopdate, use_adate=self.use_adate)
			if dd is not None: 
				if self.quickfilter:
					dd, mr = self._get_quickfilter(dd, mr, site)

		elif self.source_type == self.FILE:
			dd, mr = ccg_data.ccg_data_file(site, self.gas, self.data_source, stopdate=self.stopdate, verbose=self.verbose)

		else:
			raise ValueError("Unknown source type %s" % self.source_type)

		# apply any modifications to input data
		if self.data_source_mod == self.ATMOS_UNC:
			if self.verbose:
				print("Adding atmospheric uncertainty to", site)
			dd, mr = ccg_data.ccg_data_atmos_unc(dd, mr, site, self.initparam)

		elif self.data_source_mod == self.BIAS:
			# apply bias data to actual data
			if self.verbose:
				print("Add bias data to ", site)
			adate_filename = self.resultsdir + site + "_adates.npy"
			if os.path.exists(adate_filename):
				adates = numpy.load(adate_filename)
			else:
				adates = ccg_data.get_analysis_dates(site, self.gas, dd)   # really don't want to do this every bootstrap run
				numpy.save(adate_filename, adates)
			for (start_date, end_date, bias_val) in self.bias_data:
				a = numpy.where( (adates >= start_date) & (adates < end_date) )
				if  a[0].size > 0:
					mr[a] += bias_val

		elif self.data_source_mod == self.CUSTOM:
			if self.data_callback:
				dd, mr = self.data_callback(dd, mr)
			else:
				print("WARNING: No callback specified for custom data modification.", file=sys.stderr)

		elif self.data_source_mod == self.ANALYSIS:
	#		t1 = datetime.datetime.now()
			if self.verbose:
				print("Adding analysis uncertainty to", site)
			adate_filename = self.resultsdir + site + "_adates.npy"
			if os.path.exists(adate_filename):
				adates = numpy.load(adate_filename)
			else:
				adates = ccg_data.get_analysis_dates(site, self.gas, dd)   # really don't want to do this every bootstrap run
				numpy.save(adate_filename, adates)

			for (start_date, end_date, unc_val1, unc_val2, distribution, rflg) in self.analysis_values:
				a = numpy.where( (adates >= start_date) & (adates < end_date) )
				if  a[0].size > 0:
					if rflg == 0:
						if distribution == "normal":
							bias_val = numpy.random.normal(unc_val1, unc_val2, a[0].size)
						elif distribution == "uniform":
							bias_val = numpy.random.uniform(unc_val1, unc_val2, a[0].size)
						mr[a] += bias_val
					elif rflg == 1:
						for idx in a[0]:
							if distribution == "normal":
								bias_val = numpy.random.normal(unc_val1, unc_val2, 1)
							elif distribution == "uniform":
								bias_val = numpy.random.uniform(unc_val1, unc_val2, 1)
							mr[idx] += bias_val

	#		if self.verbose:
	#			print "Done Adding analysis uncertainty to", site
	#			t2 = datetime.datetime.now()
	#			print t2-t1


		return dd, mr


	#-----------------------------------------------------------------
	def _get_quickfilter(self, dd, mr, site):
		""" Do a quick filter of outliers on the data.
		Only reject outliers after the preliminary data date.
		"""

		s = ccg_db.getPrelimDate(site, self.gas)
		prelimdate = ccg_dates.decimalDate(s.year, s.month, s.day)
		if s.year > 9000:	# if prelimary date hasn't been set, make all data prelimary for quickfilter.
			prelimdate = 1900.0

		n = self.initparam['site'].index(site)
		short_cutoff = int(self.initparam['short'][n])
		long_cutoff = int(self.initparam['long'][n])
		interval = self.initparam['int'][n]
		npoly = int(self.initparam['poly'][n])
		nharm = int(self.initparam['harm'][n])
		tz = self.initparam['sync1']
		# make sure we have enough data points to do the curve fit
		if numpy.max(dd) - numpy.min(dd) > self.initparam['rlm']:
			dd, mr = ccg_quickfilter(dd, mr, short_cutoff, long_cutoff, interval, npoly, nharm, tz, date=prelimdate)
		else:
			print(site, "has less than", self.initparam['rlm'], "years of data. Skipping quickfilter...", file=sys.stderr)

		return dd, mr


	#-----------------------------------------------------------------
	def _prep_anchor_site(self, dd, mr, site):
		"""
		; Using the last 5 years of data from the anchor site,
		; compute average seasonal cycle and trend, f(t).
		; Use function to extrapolate forward to date.
		"""

		nyears = 5

		site_index = self.initparam['site'].index(site)

		syear = int(numpy.max(dd)) - nyears
		w = numpy.where( dd >= syear )

		x = dd[w]
		y = mr[w]

		# Build x out vector based on x and date
		xmin = numpy.min(x)
		xmax = numpy.max(x)
		n = int((self.initparam['sync2'] - xmin) * STEPS_PER_YEAR + 5)
		xo = numpy.empty(n)
		for i in range(n):
			xo[i] = WEEK * i + xmin

		short_cutoff = int(self.initparam['short'][site_index])
		long_cutoff = int(self.initparam['long'][site_index])
		interval = self.initparam['int'][site_index]
		npoly = int(self.initparam['poly'][site_index])
		nharm = int(self.initparam['harm'][site_index])
		filt = ccg_filter.ccgFilter(x, y, short_cutoff, long_cutoff, interval, npoly, nharm, timezero=xmin)

		# Add some random variability based on RSD from function fit to data

		yf = filt.getFunctionValue(x)
		diff = y - yf
		rsd = numpy.std(diff, ddof=1)

		yo = filt.getFunctionValue(xo)
		yo = yo + numpy.random.rand(yo.size)*rsd*2

		w = numpy.where(xo > xmax)
		xx = numpy.hstack((dd, xo[w]))
		yy = numpy.hstack((mr, yo[w]))

		return xx, yy


	#-----------------------------------------------------------------
	def _step1(self, bsdir):
		""" First step, read in data and perform curve fits for each site. """

		# loop through sites, get data, fit smooth curve to data
		for n, site in enumerate(self.initparam['site']):

			dd, mr = self._get_input_data(site)
			if dd is None: continue
	#		if not dd: continue
			if numpy.max(dd) - numpy.min(dd) < self.initparam['rlm']:
				print(site, "has less than", self.initparam['rlm'], "years of data. Skipping...", file=sys.stderr)
				continue

			# special case for drp.  Can't handle only 2003-2005 data
			if "drp" in site.lower() and numpy.max(dd)<2006:
				print(site, "has less than", self.initparam['rlm'], "years of data. Skipping...", file=sys.stderr)
				continue

			if self.anchor:
				if site[0:3].lower() in ['brw', 'kum', 'smo', 'spo']:
					dd, mr = self._prep_anchor_site(dd, mr, site)

			# save input data to file
			outputfile = bsdir + site + "_dat." + self.gas
			self._write_dat_file(outputfile, dd, mr)

	#--- step 1

			# fit smooth curve to data
			short_cutoff = int(self.initparam['short'][n])
			long_cutoff = int(self.initparam['long'][n])
			interval = self.initparam['int'][n]
			npoly = int(self.initparam['poly'][n])
			nharm = int(self.initparam['harm'][n])
			tz = self.initparam['sync1']
			filt = ccg_filter.ccgFilter(dd, mr, short_cutoff, long_cutoff, interval, npoly, nharm, timezero=tz)

			# columns are x, smooth, trend, function, detrend, num, gap, mblgap
			data = self._get_fit_data(filt, dd, mr, interval)
			self.fit_data[site] = data

			outputfile = bsdir + site + "_fits." + self.gas
			self._write_fit_file(outputfile, site, data)

			rsd = self._get_year_rsd(site, filt, dd, mr)
			self.rsd_data[site] = rsd

			outputfile = bsdir + site + "_sum." + self.gas
			self._write_summary_file(outputfile, site, rsd, filt)


	#-----------------------------------------------------------------
	def run(self, numbs=1):
		""" Run all instances of bootstrap calculations """


		# copy the original init file to the results directory
		if self.verbose:
			print("Copy", self.initfile, "to", self.resultsdir)
		shutil.copy(self.initfile, self.resultsdir)

		filename = self.resultsdir + "syncsteps"
		self._write_syncsteps(filename)

		filename = self.resultsdir + "site_info.txt"
		self._write_lat_info(filename)

		for i in range(numbs):

			if self.verbose:
				if numbs > 1:
					print("Working on bootstrap loop", i+1)
			self._run_one(i)


	#-----------------------------------------------------------------
	def _run_one(self, bs_num=0):
		""" Run one instance of a mbl calculation """


		# handle sub directories here.
		if self.create_bs_subdir:
			bsdir = self.resultsdir + str(bs_num+1) + "/"

			if os.path.exists(bsdir):
				try:
					shutil.rmtree(bsdir)
				except OSError as e:
					print(e)
					sys.exit()

			os.makedirs(bsdir)
		else:
			bsdir = self.resultsdir

		# get new bias data for each bootstrap run
		if self.data_source_mod == self.BIAS:
			self.bias_data = ccg_data.get_bias_data(self.gas, self.initparam['sync1'], self.initparam['sync2'], self.bias_values)


	#--- step 1

		if self.data_source_mod == self.NETWORK:
			# for network bootstrap, the first run will be with the orignal site list, subsequent runs will vary the sites
			if bs_num == 0:
				# do the curve fits to the original site list
				self._step1(bsdir)
				# save the original initparam.  We need this later to create new site lists and init files.
				self.original_initparam = copy.deepcopy(self.initparam)
			else:
				# create a new list of sites based on the original sites
				network_sites = bs_network.get_network_sites(self.original_initparam['site'], self.gas)

				# create an init file with new sites
				filename = bsdir + os.path.basename(self.initfile)
				ccg_readinit.ccg_writeinit(filename, self.original_initparam, network_sites)

				# read in the new init file and set values
				self._read_init_file(filename)
		else:
			self._step1(bsdir)



	#--- step 2

		self._set_weights()
		self._write_weights_files(bsdir)


		# calculate latitude gradients from smooth curves
		# smfits is numpy array ntimesteps x nlatbins
		filename = bsdir + "merid.smdata.ext.log"
		smfits = self._calc_meridional(self.fit_data, column_name="smooth", outputfile=filename)
		outputfile = bsdir+"merid.smfits.ext.log"
		self._write_surface_file(outputfile, smfits)

		# calculate extended data
		datarr = self._calc_mbl_smooth_differences(smfits)
		self._write_ext_files(bsdir, datarr)


	#--- srfc

		# calculate final latitude gradients and mbl surface
		filename = bsdir + "merid.data.srfc.mbl.log"
		mbl_surface = self._calc_meridional(datarr, column=2, column_name="mbl + d(t)", outputfile=filename)
		outputfile = bsdir + "surface.mbl." + self.gas
		self._write_surface_file(outputfile, mbl_surface)


	#--- zonal

		zones = self._make_zonal_data(mbl_surface)
		self._make_zone_ann_ave(zones)
		self._write_zonal_files(bsdir, zones)

		self._write_zonal_files(bsdir, self.zone_ann_avg, "ann.ave.sc")
		self._write_zonal_files(bsdir, self.zone_ann_increase, "ann.inc.tr")

	#--- surface and zones for the various ccgcrv components

		surfaces = self._parse_surface(mbl_surface, bsdir)

		for crvtype in self.ccgcrv:

			zones = self._make_zonal_data(surfaces[crvtype])
			self._write_zonal_files(bsdir, zones, crvtype)


####################################################################################
if __name__ == "__main__":

	initfile = "co2/results/init.co2.flask.master.txt"
#	initfile = "co2/naboundary/init.naboundary.atlantic.txt"
	ext = dei(initfile, gas="co2", anchor=False) #, "spo_01D0")


	keys = list(ext.initparam.keys())
#	for name in keys:
#		print name, ext.initparam[name]
#		print name

#	print ext.initparam['siteinfo']
#	print len(ext.initparam['siteinfo'])
#	print ext.initparam['siteinfo'][0]

#	print ext.syncsteps
#	print ext.nsyncsteps
#	print ext.initparam['sync2']
#	print ext.initparam['sync1']

#	print ext.mblsites
#	print ext.nmblsites


	print(ext.fyear, ext.lyear, ext.nyears, ext.nsites)

	ext.run()
