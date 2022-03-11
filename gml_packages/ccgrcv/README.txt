This directory contains the source code and executables for the ccgcrv program 
for applying curve fitting and filtering of time series data as explained at
https://gml.noaa.gov/ccgg/mbl/crvfit/crvfit.html

The python code consists of 3 files:
   ccgcrv.py - driver program
   ccg_dates.py - module needed by ccgcrv.py
   ccg_filter.py - module that does the curve fitting/filtering.

Run ccgcrv.py with the --help option to see all available options.  The documentation
of the ccg_filter.py module is in the pdf file ccgfilt.pdf

The python code should run under either python 2.6+ or python 3.  It requires that
the numpy, scipy and dateutil python packages are installed.

Files:

	ccgcrv.py - Driver program for python version of ccgcrv
	ccg_dates.py - module needed by ccgcrv.py
	ccgfilt.pdf - documentation of ccgfilt.py
	ccg_filter.py - module for performing curve fitting and filtering

For any question on the code, contact kirk.w.thoning@noaa.gov.
