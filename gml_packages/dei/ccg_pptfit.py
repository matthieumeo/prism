
from __future__ import print_function

"""
Converted to python, May 2017 kwt
Results are very close to original fortran pptfit, but not exact.

ccccc
ccccc  Modified CO2FIT so that it can be called from
ccccc  within IDL - kam 3 April 1995.
ccccc
ccccc  Fitting routine
ccccc
ccccc  by Pieter Tans
ccccc  Dec 1986
ccccc
ccccc  Nov 1991 ported to HP Unix, removed Dissplay graphics, added
ccccc           bootstrap sampling of flask sites.
ccccc
ccccc  Mar 1994 modified to accept files of extended data
ccccc
ccccc ******************************************************************
ccccc  On HP 9000 Series 700 workstations compile as follows
ccccc
ccccc       f77 pptfit.f +e -Wall -O -o pptfit
ccccc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccc
ccccc  INPUT PARAMETERS
ccccc
ccccc  X        - Array containing sample positions
ccccc  Y        - Array containing sample values
ccccc  NW       - Array containing each sample value weight
ccccc  NXY      - Number of elements in X,Y, and WTS
ccccc  XPRED    - Array of positions for constructing predicted values
ccccc
ccccc  This line is required to parse command-line
ccccc  arguments. kam.
ccccc

ccccc     Fits curves to the biweekly meridional gradient defined by the smooth
ccccc  curve fits to the flask station data.  The station data can be in any
ccccc  latitude order.   The meridional gradient fits have the following
ccccc  properties:
ccccc  1. (Optional)  Either zero derivative at the poles (in sine(lat.)
ccccc     coordinates) so that the second derivative in regular polar coord.
ccccc     is zero, or the 'natural' derivative at the poles.  This option is
ccccc     set with NDERIV=0,1 respectively.
ccccc  2. The fits should interpolate in regions where there is no data, like
ccccc     between SMO and AMS.
ccccc  3. The stations should be given weight inversely proportional to the
ccccc     standard deviation of the data.
ccccc  Property 1 is taken care of by extending the data beyond sine(lat)=
ccccc  +-1 thru mirroring with respect to the poles and fitting the extended
ccccc  data set.  Property 2 rules out global fits with sets of orthogonal
ccccc  functions because they will result in spurious wiggles in areas without
ccccc  any data.  Instead, I will use digital filtering of straight line
ccccc  segments connecting the points.  To take care of property 3, each
ccccc  station will be represented by a number (inv. proportional to st. dev.)
ccccc  of points spread out over an interval DX in sine(lat) space, centered
ccccc  on the original point.  The straight line segments connect all these
ccccc  overlapping sets of points in order.  Missing data, indicated by 0.0
ccccc  or 999.99, will be skipped in each curve fit.  Results of the fits are
ccccc  stored as 41-point (including sin=+-1 endpoints) characterizations of
ccccc  the curves in sin(lat) space, at biweekly intervals.
ccccc     The fitting process is done in two steps.  First a rough fit is done
ccccc  that gives the general overall latitude trend of the data.  Then the
ccccc  assigned points over DX are given a slope according to the rough trend,
ccccc  so that in the second, final, fit a 'staircase' effect is avoided.  The
ccccc  second fit is also more flexible (higher freq. cutoff) than the first.


Usage:
	pptf = pptfit.pptfit(x, y, wt)
	ypred = pptf.predict(xpred)

"""

import numpy
from scipy import fftpack



class ccg_pptfit:

	def __init__(self, x, y, wt=None, dx=0.15, zero_deriv=False):
		""" x, y, and wt are lists or numpy arrays """

		if isinstance(x, list):
			xp = numpy.array(x)
		else:
			xp = x

		if isinstance(y, list):
			yp = numpy.array(y)
		else:
			yp = y

		if wt is None:
			wt = [1 for z in x]
		if isinstance(wt, list):
			wtp = numpy.array(wt)
		else:
			wtp = wt


		if xp.size == 0 or yp.size == 0 or wtp.size == 0:
			self.valid = False
			return
		else:
			self.valid = True

		if xp.size != yp.size:
			raise ValueError("pptfit: Length of input arrays don't match.")
		if xp.size != wtp.size:
			raise ValueError("pptfit: Length of input arrays don't match.")


		cutoff = 2.0

		self.zero_deriv = zero_deriv


		# create a new latitude list with offsets if needed
	#	lat = []
	#	lat = numpy.array(xp.shape)
		lat = xp
		w = numpy.where(lat == -1)
		lat[w] += 1e-6
		w = numpy.where(lat == 1)
		lat[w] -= 1e-6
	#	for xz in xp:
			# need to check for duplicate locations here and offset them

			# ccccc  Offset sine of latitude at poles by a small amount
			# ccccc  so that no "real" site is at -1 or +1.
			# ccccc  November 1996 - kam,ppt.

	#		if xz == -1: xz += 1e-6
	#		if xz == 1: xz -= 1e-6
	#		lat.append(xz)

		# ccccc  assign points to the remaining stations according to
		# ccccc  their weight. Initially without a slope.
		# sort the data, add end points, convert to numpy arrays
		xx, yy = self._set_weights(lat, yp, wtp, dx)
		xx, yy = self._sort_data(xx, yy)

		w = numpy.where( xx < -0.9 )
		z = yy[w]
		ystart = numpy.average(z)
		w = numpy.where( xx > 0.9 )
		z = yy[w]
		yend = numpy.average(z)



		nfft = 1024
		nbgn = int(nfft/4)
		nend = 3 * nbgn - 1
		step = 4.0/(nfft-1)

		# create nfft values from -2 to 2
		xd = numpy.linspace(-2, 2, nfft)

		if self.zero_deriv:

			yd = numpy.interp(xd, xx, yy, left=0, right=0)
			w = numpy.where( (xd <= 0) & (xd > -1) )
			yp = numpy.flipud(yd[w])	# reverse the data from 0 to -1 and put in -1 to -2
			w = numpy.where( xd < -1 )
			yd[w] = yp

			w = numpy.where( (xd >= 0) & (xd < 1) )
			yp = numpy.flipud(yd[w])	# reverse the data from 0 to 1 and put in 1 to 2
			w = numpy.where( xd > 1 )
			yd[w] = yp

		else:

			# create interpolated y values at the xd values
			w = numpy.where( xx < -0.9 )
			left = numpy.mean(yy[w])  # should be average of all values where xx<-0.9
			w = numpy.where( xx > 0.9 )
			right = numpy.mean(yy[w]) # should be average of all values where xx>0.9
			yd = numpy.interp(xd, xx, yy, left=left, right=right)

	# good up to here

		# do the first smoothing
		smooth = self._filt_data(xd, yd, cutoff*2)

		# Second round starts, to get a more detailed fit.
		# Construct XX and YY again, this time with a slope in the assigned points.
		xx, yy = self._set_weights(lat, yp, wtp, dx, smooth, step, useSlope=True)
		xx, yy = self._sort_data(xx, yy)

		# create interpolated y values at the xd values
		yd = numpy.interp(xd, xx, yy, left=0, right=0)

		if self.zero_deriv:
			# NOTE, this option doesn't work as well for the northern hemisphere poles - my opinion kwt
			yd = numpy.interp(xd, xx, yy, left=0, right=0)
			w = numpy.where( (xd <= 0) & (xd > -1) )
			yz = numpy.flipud(yd[w])
			w = numpy.where( xd < -1 )
			yd[w] = yz

			w = numpy.where( (xd >= 0) & (xd < 1) )
			yz = numpy.flipud(yd[w])
			w = numpy.where( xd > 1 )
			yd[w] = yz

		else:
			# fill in the edges with the slope values
			slope1 = (smooth[nbgn+20] - smooth[nbgn]) / (xd[nbgn+20] - xd[nbgn])
			slope2 = (smooth[nend-20] - smooth[nend]) / (xd[nend-20] - xd[nend])
			w = numpy.where( xd < xx[0] )
			yd[w] = ystart + (xd[w] - xd[nbgn]) * slope1
			w = numpy.where( xd > xx[-1] )
			yd[w] = yend + (xd[w] - xd[nend]) * slope2

		# final smoothing
		self.smooth = self._filt_data(xd, yd, cutoff*4)
		self.xd = xd



	#------------------------------------------------------------
	def _set_weights(self, x, y, wt, dx, yfft=None, step=0, useSlope=False):
		""" Set the weighting. """

		npoints = len(x)

		xx = []
		yy = []
		for i in range(npoints):

			if useSlope:
				k1 = int( (x[i] - dx/2 + 2) / step)
				k2 = int( (x[i] + dx/2 + 2) / step)
				slope = yfft[k2] - yfft[k1]
			else:
				slope = 0



			nrep = max( int( round(wt[i], 0)), 1)
			if nrep == 1:
				xdist = 0
				ydist = 0
			else:
				xdist = dx/(nrep-1)
				ydist = slope/(nrep-1)

			start = -(nrep+1)/2.0

			for nn in range(nrep):
				d = x[i] + (start + nn + 1) * xdist
				if d > -1 and d < 1:
					xx.append(d)
					yy.append(y[i] + (start+nn+1) * ydist)

		return numpy.array(xx), numpy.array(yy)

	#------------------------------------------------------------
	def _sort_data(self, xx, yy):
		""" sort the two lists, add end points at -1, 1,
		and convert to numpy arrays.
		"""

		# add endpoints at -1, 1
		# pieters original fortran code used an average of the last points
		a = numpy.array((xx, yy)).T
		w = numpy.lexsort((a[:, 1], a[:, 0]))   # sort on latitude
		a = a[w]                              # get sorted array

		a = numpy.insert(a, 0, [-1, a[0][1]], axis=0) # insert point at lat -1
		a = numpy.append(a, [[1, a[-1][1]]], axis=0)  # add point at lat 1

		x = a.T[0]  # array of latitudes
		y = a.T[1]  # array of values

		return x, y

	#------------------------------------------------------------
	def _adjustend(self, x, y):
		""" calculate the slope and intercept from end points of data """

		slope = (y[0] - y[-1]) / (x[0] - x[-1])
		intercept = y[0]

		return intercept, slope

	#------------------------------------------------------------
	def _filt_data(self, xd, yd, cutoff):
		""" Filter the data by converting to frequency domain with fft,
		multiplying with filter value, invere fft back to time domain.
		"""

		# remove trend from data
		ca, cb = self._adjustend(xd, yd)
		ya = yd - (ca + cb*xd)

		# do fft on interpolated data
		fft = fftpack.rfft(ya)

		nfft = xd.size

		# not sure this is correct, but it's required to match pieter's fortran code
		dinterval = 1.0/nfft  # 4./nfft

		# do short term filter and convert back to time domain
		a = self._freq_filter(fft, dinterval, cutoff)
		yfilt = fftpack.irfft(a)

		# add trend back in
		smooth = yfilt + (ca + cb*xd)

		return smooth

	#------------------------------------------------------------
	def _freq_filter(self, fft, dinterv, cutoff):
		""" Apply low-pass filter to fft data.
		Multiply each discrete frequency in fft by a
		low-pass filter function value set by the value 'cutoff'
		input:
			fft - results of fft
			dinterv - sampling interval
			cutoff - cutoff value
		"""

		n2 = len(fft)

		freq = fftpack.rfftfreq(n2, dinterv)    # get array of frequencies

		# get filter value at frequencies
		z = numpy.power((freq/cutoff), 6)
		rw = 1.0 / (1 + z)

		filt = fft*rw                           # apply filter values to fft

		return filt


	#------------------------------------------------------------
	def predict(self, xpred):
		""" Calculate predicted values of smoothed latitude gradient at locations in xpred """

		# probably should do some error checking if xpred is out of range
		if self.valid:
			ypred = numpy.interp(xpred, self.xd, self.smooth, left=-999.99, right=-999.99)

		else:
			ypred = numpy.empty( len(xpred) )
			ypred[:] = -999.99

		return ypred


################################################################################################

if __name__ == "__main__":

	x = []
	y = []
	wt = []
	f = open("pptfit.data")
	for line in f:
		(xp, yp, wtp) = line.split()
		x.append(float(xp))
		y.append(float(yp))
		wt.append(float(wtp))


	dx = 0.15
	nbins = 41
	latbins = [n*0.05 -1 for n in range(nbins)]

	pptf = ccg_pptfit(x, y, wt)
	ypred = pptf.predict(latbins)


	for xp, yp in zip(latbins, ypred):
		print("%8.2f %10.5f" % (xp, yp))
