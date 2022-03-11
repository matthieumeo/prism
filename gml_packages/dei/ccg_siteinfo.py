
""" module for getting site information """

from __future__ import print_function
import sys

import math

import ccg_db
import ccg_utils

############################################################################
def GV2CCGG(gv_strategy, gv_platform):
	""" convert globalview strategy and platform strings
	to ccgg project number and strategy number.
	"""

	gv_strategy = gv_strategy.upper()

	if gv_strategy == 'D' and gv_platform == '0':
		ccgg_project_num = '1'
		ccgg_strategy_num = '1'
	elif  gv_strategy == 'D' and gv_platform == '1':
		ccgg_project_num = '1'
		ccgg_strategy_num = '1'
	elif  gv_strategy == 'P' and gv_platform == '1':
		ccgg_project_num = '1'
		ccgg_strategy_num = '2'
	elif  gv_strategy == 'P' and gv_platform == '0':
		ccgg_project_num = '1'
		ccgg_strategy_num = '2'
	elif  gv_strategy == 'D' and gv_platform == '2':
		ccgg_project_num = '2'
		ccgg_strategy_num = '1'
	elif  gv_strategy == 'P' and gv_platform == '2':
		ccgg_project_num = '2'
		ccgg_strategy_num = '2'
	elif  gv_strategy == 'D' and gv_platform == '3':
		ccgg_project_num = '1'
		ccgg_strategy_num = '1'
	elif  gv_strategy == 'C' and gv_platform == '3':
		ccgg_project_num = '3'
		ccgg_strategy_num = '3'
	elif  gv_strategy == 'C' and gv_platform == '0':
		ccgg_project_num = '4'
		ccgg_strategy_num = '3'
	elif  gv_strategy == 'P' and gv_platform == '3':
		ccgg_project_num = '1'
		ccgg_strategy_num = '2'
	else:
		ccgg_project_num = '-1'
		ccgg_strategy_num = '-1'

	return ccgg_project_num, ccgg_strategy_num


############################################################################
def ccg_siteinfo(site, siteinfo_file=None):
	""" Get site information.
	Input:
	    site - A comma separated string of site codes, in either
	           normal 3 letter code, or in GV format (e.g. alt_01D0)
	    siteinfo_file - A file containing information about the site.

	Output:
	    A python list, with each item a dict containing the information
		for the sites in the 'site' string.  The list items are in the
		same order as the site codes in the 'site' string.
	"""

	if siteinfo_file is not None:
		result = ccg_siteinfo_file(site, siteinfo_file)
	else:
		result = ccg_siteinfo_db(site)

	return result


############################################################################
def ccg_siteinfo_db(site):

	db, c = ccg_db.dbConnect("gmd", readonly=True)

	fieldlist = "num,code,name,country,lat,lon,elev,lst2utc"

#	if not addl_siteinfo_file:
	addl_siteinfo_file = "/ccg/src/db/ccg_siteinfo.txt"

	result = []
	for sitecode in site.split(","):
		sitecode = sitecode.strip()
		info = {}
		if "_" in sitecode:
			(code, ext) = sitecode.split("_")
		else:
			code = sitecode
			ext = None
#			ext = "01D0"	# assume if only station code is present, it's a gmd site


		sql = "select %s from site where code='%s' " % (fieldlist, code)
		c.execute(sql)
		if c.rowcount == 0:
			# try again using only 1st 3 characters of code
			code = sitecode[0:3]
			sql = "select %s from site where code='%s' " % (fieldlist, code)
			c.execute(sql)

		if c.rowcount == 0:
			# try again with default addl_siteinfo_file
			info = ccg_siteinfo_file(sitecode, addl_siteinfo_file)
			info = info[0]

		else:
			# get data from database
			a = c.fetchone()
			info["site_num"] = a[0]
			info["site_code"] = a[1]
			info["site_name"] = a[2]
			info["site_country"] = a[3]
			info["lat"] = float(a[4])
			info["sinlat"] = math.sin(math.radians(float(a[4])))
			info["lon"] = float(a[5])
			info["elev"] = float(a[6])
			info["lst2utc"] = float(a[7])
			h1 = "N"
			if info["lat"] < 0: h1 = "S"
			h2 = "E"
			if info["lon"] < 0: h2 = "W"
			info["position"] = u"[%.2f\N{DEGREE SIGN}%s, %.2f\N{DEGREE SIGN}%s]" % (info["lat"], h1, info["lon"], h2)

		# take a drastic approach and stop the program if site info not found (we must have latitude for a site)
		if len(info.keys()) == 0:
			sys.exit("ERROR: No site information found for site %s" % sitecode)

		if ext is not None:
			labnumber = int(ext[0:2])
			gv_strategy = ext[2]
			gv_platform = ext[3]

			# Get Lab information
			sql = "SELECT num, name, country, abbr, logo FROM obspack.lab WHERE num=%s" % labnumber
			c.execute(sql)
			a = c.fetchone()
			info["lab_num"] = int(a[0])
			info["lab_name"] = a[1]
			info["lab_country"] = a[2]
			info["lab_abbr"] = a[3]
			info["lab_logo"] = a[4]

			# Get Strategy information
			sql = "SELECT abbr, name FROM ccgg.gv_strategy WHERE abbr='%s'" % gv_strategy
			c.execute(sql)
			a = c.fetchone()
			info["strategy_code"] = a[0]
			info["strategy_name"] = a[1]

			# Get Platform information
			sql = "SELECT num, name FROM ccgg.gv_platform WHERE num=%s" % gv_platform
			c.execute(sql)
			a = c.fetchone()
			info["platform_num"] = int(a[0])
			info["platform_name"] = a[1]

			if labnumber == 1 or labnumber == 99:

				# Get cooperating agency
				ccgg_project_num, ccgg_strategy_num = GV2CCGG(gv_strategy, gv_platform)
				sql = "SELECT site_coop.name, site_coop.abbr FROM ccgg.site_coop,gmd.site WHERE gmd.site.code='%s'" % code
				sql += " AND gmd.site.num=site_num AND strategy_num=%s AND project_num=%s" % (ccgg_strategy_num, ccgg_project_num)
				c.execute(sql)
				if c.rowcount > 0:
					a = c.fetchone()
					info["agency_name"] = a[0]
					info["agency_abbr"] = a[1]
				else:
					info["agency_name"] = ""
					info["agency_abbr"] = ""

				sql = "SELECT intake_ht FROM ccgg.site_desc,gmd.site WHERE gmd.site.code='%s'" % code
				sql += " AND gmd.site.num=site_num AND strategy_num=%s AND project_num=%s" % (ccgg_strategy_num, ccgg_project_num)
				c.execute(sql)
				if c.rowcount > 0:
					a = c.fetchone()
					info["intake_ht"] = float(a[0])
				else:
					info["intake_ht"] = 0

			else:
				info["intake_ht"] = 0
				info["agency_name"] = ""
				info["agency_abbr"] = ""


		else:
			info["lab_num"] = 0
			info["lab_name"] = "Unknown"
			info["lab_country"] = "Unknown"
			info["lab_abbr"] = "Unknown"
			info["lab_logo"] = "Unknown"
			info["strategy_code"] = "Unknown"
			info["strategy_name"] = "Unknown"
			info["platform_num"] = 0
			info["platform_name"] = "Unknown"
			info["intake_ht"] = 0


		result.append(info)

	c.close()
	db.close()

	return result


############################################################################
def ccg_siteinfo_file(site, siteinfo_file="siteinfo.txt"):
	""" Get site information.
	Input:
	    site - A comma separated string of site codes, in either
		   normal 3 letter code, or in GV format (e.g. alt_01D0)
	    siteinfo_file - A file containing information about the site.

	Output:
	    A python list, with each item a dict containing the information 
		for the sites in the 'site' string.  The list items are in the
		same order as the site codes in the 'site' string.
	"""

	sitedata = ReadFile(siteinfo_file)

	result = []
	for sitecode in site.split(","):
		sitecode = sitecode.strip()
		info = {}
		if "_" in sitecode:
			(code, ext) = sitecode.split("_")
		else:
			code = sitecode
			ext = None

		if code.upper() in sitedata.keys():
			a = sitedata[code.upper()]

			info["site_code"] = a[0]
			info["site_name"] = a[1]
			info["site_country"] = a[2]
			info["lat"] = float(a[3])
			info["sinlat"] = math.sin(math.radians(float(a[3])))
			info["lon"] = float(a[4])
			info["elev"] = float(a[5])
			info["intake_ht"] = float(a[6])
			info["lst2utc"] = float(a[7])
			h1 = "N"
			if info["lat"] < 0: h1 = "S"
			h2 = "E"
			if info["lon"] < 0: h2 = "W"
			info["position"] = u"[%.2f\N{DEGREE SIGN}%s, %.2f\N{DEGREE SIGN}%s]" % (info["lat"], h1, info["lon"], h2)
			info["strategy_name"] = "discrete"

		result.append(info)


	return result


############################################################################
def ReadFile(filename):

	lines = ccg_utils.cleanFile(filename)

	data = {}
	for line in lines:
		a = line.split("|")
		result = []
		for item in a:
			b = item.strip()
			result.append(b)
		data[a[0].strip().upper()] = result

	return data



############################################################################
if __name__ == "__main__":


	info = ccg_siteinfo("alt", "siteinfo.txt")
	for data in info:
		print(data)



#	info = ccg_siteinfo("alt_01D0,pocs30_01D0")

#	for data in info:
#		print(data)
#		s = []
#		s.append(str(data["site_num"]))
#		s.append(data["site_code"])
#		s.append(data["site_name"])
#		s.append(data["site_country"])
#		s.append(str(data["lat"]))
#		s.append(str(data["lon"]))
#		s.append(str(data["elev"]))
#		s.append(str(data["lst2utc"]))

#		print("|".join(s))
