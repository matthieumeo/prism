"""
Perform an outlier rejection of data using ccg_filter curver fit to data
"""
from __future__ import print_function

import ccg_filter

import numpy


################################################################################################
def ccg_quickfilter(dd, mr, short=80, longt=667, interval=0, numpoly=3, numharm=4, timezero=-1, date=None):
	""" Perform an outlier rejection of data using ccg_filter curve fit to data """

	rejectPrelimOnly = False
	if date is not None:
		prelim_dd = date
		rejectPrelimOnly = True

	npoints = len(mr)
	flag = [1 for i in range(npoints)]

	# loop until number of rejected points = 0
	nr = 1
	loopnum = 0
	while nr > 0 and loopnum < 10:

		loopnum += 1

		x = []
		y = []
		for xp, yp, f in zip(dd, mr, flag):
			if f:
				x.append(xp)
				y.append(yp)


		filt = ccg_filter.ccgFilter(x, y, short, longt, interval, numpoly, numharm, timezero)

		# Find the residual std. dev about the smooth curve
		sigma = filt.rsd2
		sigma3 = 3 * sigma
#		print loopnum, "3 sigma is", sigma3

		y1 = filt.getSmoothValue(dd)

		nr = 0
		for i, (xp, yp, f) in enumerate(zip(dd, mr, flag)):

			# only check on non flagged data. y1 is Nan if outside of domain
			if f:
				ydiff = yp - y1[i]
				if abs(ydiff) > sigma3:
					if rejectPrelimOnly:
						if xp > prelim_dd:
							flag[i] = 0
							nr += 1
					else:
						flag[i] = 0
						nr += 1
#						print(i, xp, yp, ydiff)



	# retain any flagged data that may now be inside of 3 sigma
	for i, (xp, yp, f) in enumerate(zip(dd, mr, flag)):
		if f == 0:
			if not numpy.isnan(y1[i]):
				ydiff = yp - y1[i]
				if abs(ydiff) < sigma3:
					flag[i] = 1
#					print "retain", xp, yp, ydiff





	# create new lists to return
	x = []
	y = []
	for (xp, yp, f) in zip(dd, mr, flag):
		if f:
			x.append(xp)
			y.append(yp)
#		else:
#			print xp, yp, f


	return numpy.array(x), numpy.array(y)



################################################################################################

if __name__ == "__main__":

	x = []
	y = []
	wt = []
	f = open("../hba.dat")
	for line in f:
		(xp, yp) = line.split()
		x.append(float(xp))
		y.append(float(yp))

	print(len(y))
	xnew, ynew = ccg_quickfilter(x, y, interval=7, date=2016.0)
	print(xnew)
	print(xnew.size)

#	for xp, yp in zip(xnew, ynew):
#		print("%14.8f %10.3f" % (xp, yp))
