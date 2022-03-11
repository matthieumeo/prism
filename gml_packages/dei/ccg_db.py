""" Utility routines for connecting to a database, and retreiving common values from database """

import datetime
try:
	# for python 2
	import MySQLdb
	from MySQLdb.converters import conversions
	from MySQLdb.constants import FIELD_TYPE

except ImportError:
	# for python 3
	import pymysql as MySQLdb
	from pymysql.converters import conversions
	from pymysql.constants import FIELD_TYPE

#---------------------------------------------------------
def dbConnect(database="ccgg", readonly=True):
	""" Make a connection to a database """

	# convert the Decimal data type to float
	conversions[FIELD_TYPE.DECIMAL] = float
	conversions[FIELD_TYPE.NEWDECIMAL] = float

	# force connect always readonly
	db = MySQLdb.connect(host="db", user="guest", passwd="", db=database)
		
	c = db.cursor()
	return (db, c)

#---------------------------------------------------------
def dbTableExists(table, database="ccgg"):
	""" Check if a table exists in a database """

	db, c = dbConnect(database, readonly=True)
	sql = " SHOW TABLES LIKE '%s'" % table
	c.execute(sql)
	result = c.fetchall()
	c.close()
	db.close()
	if c.rowcount > 0:
		return True
	else:
		return False


#---------------------------------------------------------
def dbQueryAndFetch(query, database="ccgg"):
	""" Execute a query on the database and return all results """

	db, c = dbConnect(database)
	c.execute(query)
	result = c.fetchall()
	c.close()
	db.close()


	return result

#---------------------------------------------------------
def getSiteNum(code):
	""" Get the site number for a site code """

	query = "SELECT num FROM site WHERE code='%s'" % code
	myDb, c = dbConnect("gmd", readonly=True)
	c.execute(query)
	row = c.fetchone()

	c.close()
	myDb.close()

	if row:
		num = row[0]
	else:
		num = -1

	return int(num)

#---------------------------------------------------------
def getSiteCode(num):
	""" Get a site code from a site number """

	myDb, c = dbConnect("gmd", readonly=True)

	query = "SELECT code FROM site WHERE num=%s" % num
	c.execute(query)
	row = c.fetchone()

	c.close()
	myDb.close()

	if row:
		code = row[0]
	else:
		code = ""

	return code

#---------------------------------------------------------
def getSiteName(code):
	""" Get a site name from a site code """

	mydb, c = dbConnect("gmd", readonly=True)
	query = "SELECT name FROM gmd.site WHERE code='%s'" % code
	c.execute(query)
	row = c.fetchone()

	c.close()
	mydb.close()

	if row:
		name = row[0]
	else:
		name = ""

	return name

#---------------------------------------------------------
def getSiteNameFromNum(num):
	"""Get a site name from a site number """

	mydb, c = dbConnect("gmd", readonly=True)
	query = "SELECT name FROM gmd.site WHERE num=%s" % num
	c.execute(query)
	row = c.fetchone()

	c.close()
	mydb.close()

	if row:
		name = row[0]
	else:
		name = ""

	return name

#---------------------------------------------------------
def getSiteInfo(code, asdict=False):
	""" Get site information from a site code.
	If asdict is True, then return data as a dict,
	with column names as keys,
	otherwise as a list.
	 """

	mydb, c = dbConnect("gmd", readonly=True)
	query = "SELECT num, code, name, country, lat, lon, elev, lst2utc FROM gmd.site WHERE code='%s'" % code
	c.execute(query)
	row = c.fetchone()

	c.close()
	mydb.close()

	if row:
		if asdict:
			keys = ["num", "code", "name", "country", "lat", "lon", "elev", "lst2utc"]
			result = {}
			for val, key in zip(row, keys):
				result[key] = val
			return result
		else:
			return row
	else:
		return ()


#---------------------------------------------------------
def getSiteInfoFromNum(num):
	""" Get site information from a site number """

	mydb, c = dbConnect("gmd", readonly=True)
	query = "SELECT num, code, name, country, lat, lon, elev, lst2utc FROM gmd.site WHERE num=%s" % num
	c.execute(query)
	row = c.fetchone()

	c.close()
	mydb.close()

	if row:
		return row
	else:
		return ()


#---------------------------------------------------------
def getGasNum(gas):
	""" Get parameter number from formula or name """

	if not gas: return 0

	myDb, c = dbConnect("gmd", readonly=True)

	query = "SELECT num from parameter where formula='%s' or name='%s'" % (gas, gas)
	c.execute(query)
	row = c.fetchone()

	c.close()
	myDb.close()

	if row:
		num = row[0]
	else:
		num = -1

	return int(num)

#---------------------------------------------------------
def getGasFormula(gasnum):
	""" Get parameter formula from number """

	myDb, c = dbConnect("gmd", readonly=True)

	query = "SELECT formula from parameter where num=%s" % gasnum
	c.execute(query)
	row = c.fetchone()

	c.close()
	myDb.close()

	if row:
		return row[0]
	else:
		return ""

#---------------------------------------------------------
def getGasNameFromNum(gasnum):
	""" Get parameter name from number """

	myDb, c = dbConnect("gmd", readonly=True)

	query = "SELECT name from parameter where num=%s" % gasnum
	c.execute(query)
	row = c.fetchone()

	c.close()
	myDb.close()

	if row:
		return row[0]
	else:
		return ""

#---------------------------------------------------------
def getGasName(formula):
	""" Get paramter name from formula """

	myDb, c = dbConnect("gmd", readonly=True)

	query = "SELECT name from parameter where formula='%s'" % formula
	c.execute(query)
	row = c.fetchone()

	c.close()
	myDb.close()

	if row:
		return row[0]
	else:
		return ""

#---------------------------------------------------------
def getGasInfo(formula):
	""" Get paramter info from formula """

	myDb, c = dbConnect("gmd", readonly=True)

	query = "SELECT num, formula, name, unit, unit_name, formula_html, unit_html, formula_idl, unit_idl, formula_matplotlib, unit_matplotlib "
	query += "FROM parameter WHERE formula='%s'" % formula
	c.execute(query)
	row = c.fetchone()

	c.close()
	myDb.close()

	if row:
		return row
	else:
		return ()

#---------------------------------------------------------
def getGasInfoFromNum(num):
	""" Get paramter info from number """

	myDb, c = dbConnect("gmd", readonly=True)

	query = "SELECT num, formula, name, unit, unit_name, formula_html, unit_html, formula_idl, unit_idl, formula_matplotlib, unit_matplotlib "
	query += "FROM parameter WHERE num=%s" % num
	c.execute(query)
	row = c.fetchone()

	c.close()
	myDb.close()

	if row:
		return row
	else:
		return ()

#---------------------------------------------------------
def getProgramNum(abbr):
	""" Get program number from a program abbreviation """

	myDb, c = dbConnect("gmd", readonly=True)

	query = "SELECT num from program where abbr='%s'" % abbr
	c.execute(query)
	row = c.fetchone()

	c.close()
	myDb.close()

	if row:
		return row[0]
	else:
		return ()


#---------------------------------------------------------
def getIntakeHeights(stacode, param):
	""" Get all intake heights from an insitu table """

	table = stacode.lower() + "_" + param.lower() + "_insitu"
	query = "select distinct intake_ht from %s " % table

	myDb, c = dbConnect("gmd", readonly=True)
	c.execute(query)
	result = c.fetchall()
	c.close()
	myDb.close()

	a = []
	for row in result:
		a.append(float(row[0]))

	return a

#---------------------------------------------------------
def getBinInfo(stacode, project_num):
	""" Get any binning information for a site code and project """

	sitenum = getSiteNum(stacode)

	query = "select method, min, max, width "
	query += "from data_binning where site_num=%s and project_num=%s" % (sitenum, project_num)

	myDb, c = dbConnect("ccgg", readonly=True)
	c.execute(query)
	result = c.fetchone()
	c.close()
	myDb.close()

	return result

#---------------------------------------------------------
def getProjectName(projectnum):
	""" Get project name from project number """

	query = "SELECT name FROM gmd.project WHERE num=%d" % projectnum
	myDb, c = dbConnect("gmd", readonly=True)
	c.execute(query)
	row = c.fetchone()

	if c.rowcount > 0:
		projectname = row[0]
	else:
		projectname = ""

	c.close()
	myDb.close()

	return projectname

#---------------------------------------------------------
def getProgramAbbrFromProject(proj):
	""" Get program abbreviation from a project number """

	query = "select program.abbr from gmd.program, gmd.project where project.num=%d and program_num=program.num;" % proj
	myDb, c = dbConnect("gmd", readonly=True)
	c.execute(query)
	row = c.fetchone()
	if c.rowcount > 0:
		name = row[0]
	else:
		name = ""

	c.close()
	myDb.close()

	return name


#---------------------------------------------------------
def getPrelimDate(site, gas):
	""" Return date for preliminary data for a site code and gas """

	sitecode = site[0:3].lower()
	sitenum = getSiteNum(sitecode)
	paramnum = getGasNum(gas)


	query  = "SELECT begin FROM ccgg.data_release "
	query += "WHERE site_num=%s " % (sitenum)
	query += "AND parameter_num=%d " % (paramnum)
	query += "AND project_num=%d " % (1)

	myDb, c = dbConnect("ccgg", readonly=True)

	c.execute(query)
	row = c.fetchone()
	if c.rowcount > 0:
		prelimdate = row[0]
	else:
		prelimdate = datetime.date(9999, 12, 31)


	c.close()
	myDb.close()


	return prelimdate
