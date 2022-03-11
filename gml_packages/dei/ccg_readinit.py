""" Read an initialization file for data extension """

from __future__ import print_function

import sys
import os
from dateutil.parser import parse


from ccg_utils import cleanFile

from ccg_siteinfo import ccg_siteinfo
#from ccg_siteinfo_old import ccg_siteinfo


###################################################################################################
def ccg_readinit(filename, site="all", extonly=False, siteinfo=None):
	""" Read an initialization file for data extension """

	if not filename:
		print("Initialization file must be specified, e.g., file='init.co2.flask.1997'.", file=sys.stderr)
		return None

	NUMHEADERS = 3

	if not os.path.exists(filename):
		print(filename, "does not exist.", file=sys.stderr)
		return None

	lines = cleanFile(filename)

	headers = lines[0:NUMHEADERS]
	tmp = headers[1].split()
	ntmp = len(tmp)
	sync1 = float(tmp[0])
	sync2 = float(tmp[1])
	fillgap = 0.0
	mblgap = 0.0
	rlm = 0.0
	if ntmp > 2: rlm = float(tmp[2])
	if ntmp > 3: fillgap = float(tmp[3])
	if ntmp > 4: mblgap = float(tmp[4])


	sitelist = lines[NUMHEADERS:]

#	print sitelist

	fields_in_header = headers[2].split()
	if extonly:
		if "ext" in fields_in_header:
			extfield = fields_in_header.index("ext")
		else:
			extfield = None

	result = {}
	for field in fields_in_header:
		result[field] = []

	for line in sitelist:
		a = line.split()

		if extonly:
			if extfield is not None:
				val = float(a[extfield])
				if val == 0:
					continue


		# skip lines that don't match given site code
		if site.lower() != "all":
			if site.lower() != a[0].lower(): continue

		for i, f in enumerate(fields_in_header):
			try:
				if "." in a[i]:
					result[f].append(float(a[i]))
				else:
					result[f].append(int(a[i]))
			except:
				result[f].append(a[i])


	result['fields'] = fields_in_header
	result['sync1'] = sync1
	result['sync2'] = sync2
	result['fillgap'] = fillgap
	result['mblgap'] = mblgap
	result['rlm'] = rlm
	result['siteinfo'] = ccg_siteinfo(",".join(result['site']), siteinfo)


	return result

###################################################################################################
def ccg_writeinit(filename, initparam, sites=None):
	""" Write back an initfile using data in initparam, possibly modified with a
	separate list of sites.
	Input:
		initparam - data from a previous read of an init file.
		sites - a list of sites to use instead of the original site list.
	"""


	if sites is None:
		sitelist = initparam['site']
	else:
		sitelist = sites

	print("length of sitelist is", len(sitelist))

	try:
		fp = open(filename, "w")
	except:
		raise ValueError("Can't open %s for writing." % filename)


	fp.write("   sync1    sync2      rlm   fillgap   mblgap\n")
	fp.write("%8s %8s %8s %8s %8s\n" % (initparam['sync1'], initparam['sync2'], initparam['rlm'], initparam['fillgap'], initparam['mblgap']))

	for f in initparam['fields']:
		if f == 'site':
			fp.write("%-15s" % f)
		else:
			fp.write("%9s" % f)
	fp.write("\n")

	for site in sitelist:
		idx = initparam['site'].index(site)
		for f in initparam['fields']:
			if f == 'site':
				fp.write("%-15s" % site)
			else:
				fp.write("%9s" % initparam[f][idx])
		fp.write("\n")

	fp.close()


###################################################################################################
def read_bias_config(filename, gas):
	""" Read in a bias bootstrap configuration file. """

	lines = cleanFile(filename)
	for line in lines:
		(sp, min_period, max_period, value) = line.split()
		if sp.lower() == gas.lower():
			return (float(min_period), float(max_period), float(value))

	return (0, 0, 0)

###################################################################################################
def read_analysis_config(filename, gas):
	""" Read in analysis uncertainty bootstrap configuration file. """

	a = []

	lines = cleanFile(filename)
	for linenum, line in enumerate(lines):
		(sp, min_date, max_date, value1, value2, distribution, rndflag) = line.split()

		if sp.lower() == gas.lower():
			if distribution not in ["normal", "uniform"]:
				raise ValueError("Unknown distribution name in config file %s, line number %d" % (filename, linenum+1))

			try:
				d1 = parse(min_date)
				d2 = parse(max_date)
				val1 = float(value1)
				val2 = float(value2)
				rndf = int(rndflag)
			except ValueError as err:
				raise ValueError("Can not parse line from config file %s line number %d" % (filename, linenum+1))


			a.append((d1.date(), d2.date(), val1, val2, distribution, rndf))


	return a


###################################################################################################

if __name__ == "__main__":

#	initparam = ccg_readinit("co2/results/init.co2.flask.master.txt", 'mlo_01D0')
#	initparam = ccg_readinit("/ccg/dei/ext/co2/work/init.co2.flask.master.txt", "all")
#	initparam = ccg_readinit("init.co2.txt", "all")
	initparam = ccg_readinit("/nfs/hats/REGINV/HCFC22/MBL/init.HCFC22.flask", "all")

	keys = initparam.keys()
	for name in keys:
		print(name , initparam[name])

	for s in initparam['siteinfo']:
		print(s['site_code'], s['intake_ht'])
#	print(initparam['siteinfo'][0])


#	print initparam['ext']
#	print len(initparam['ext'])

#	print initparam['fields']

#	print initparam['site']
#	print len(initparam['site'])


#	newsites = []
#	for i in range(len(initparam['site'])):
#		newsites.append('alt_01D0')


#	ccg_writeinit("init_test.txt", initparam)
#	ccg_writeinit(initparam, newsites)
#	ccg_write_init(initparam)
