"""
utility function for removing comments and unneeded characters from a file.
"""

from __future__ import print_function

import sys

#############################################################
def clean_line(line):
	"""
	Remove unwanted characters from line,
	such as leading and trailing white space, new line,
	and comments
	"""

	line = line.strip('\n')                 # get rid on new line
	line = line.split('#')[0]               # discard '#' and everything after it
	line = line.strip()                     # strip white space from both ends
	line = line.replace("\t", " ")          # replace tabs with spaces

	return line

#############################################################
def cleanFile(filename, showError=True):
	""" Read a file, remove lines and parts of lines
	following the comment character '#',
	remove white space from ends,
	and new line characters.
	return the cleaned up lines.
	"""

	data = []


	try:
		f = open(filename)
	except OSError:
		if showError:
			print("cleanFile: Can't open file", filename, file=sys.stderr)
		return data

	for line in f:
		line = clean_line(line)
		if line:
			data.append(line)

	f.close()

	return data
