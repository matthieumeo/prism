
"""
Function to return a new list of site codes, based on an original given list of sites.

Used in the 'network' bootstrap method for the data extension software.
"""

from __future__ import print_function

import random



def get_network_sites(sites, gas):
	""" Return a new list of site codes, based on the original input list 'sites'
	for the given gas.  Some sites are required to be included, depending on the gas.
	"""

	nsites = len(sites)
	print("nsites is ", nsites)


	cond = {}

	# set up required conditions
	if gas == 'ch4c13':
	        #cond[1] = ['cgo_01D0']
	        #cond[2] = ['smo_01D0']
	        #cond[3] = ['kum_01D0']
	        #cond[4] = ['brw_01D0']

		cond[1] = ['cgo_01D0', 'spo_01D0']
		cond[2] = ['smo_01D0', 'kum_01D0']
		cond[3] = ['alt_01D0', 'brw_01D0']

	elif gas == 'co2c13':
		cond[1] = ['cgo_01D0']
		cond[2] = ['smo_01D0']
		cond[3] = ['kum_01D0']
		cond[4] = ['brw_01D0']

	elif gas == 'co2o18':
		cond[1] = ['cgo_01D0']
		cond[2] = ['smo_01D0', 'asc_01D0']
		cond[3] = ['kum_01D0']
		cond[4] = ['brw_01D0']

	elif gas == 'h2':
		cond[1] = ['cgo_01D0']
		cond[2] = ['smo_01D0', 'asc_01D0']
		cond[3] = ['kum_01D0']
		cond[4] = ['brw_01D0']

	elif gas == 'co':
		cond[1] = ['cgo_01D0']
		cond[2] = ['smo_01D0', 'asc_01D0']
		cond[3] = ['kum_01D0']
		cond[4] = ['brw_01D0']

	elif gas == 'co2':
	       # cond[1] = ['spo_01D0','spo_01C0']
	       # cond[2] = ['brw_01D0','brw_01C0']
	       # cond[3] = ['kum_01D0']
	       # cond[4] = ['smo_01D0','smo_01C0']

	       # Revision to site requirements per discussion with Pieter.
	       # See e-mail to Pieter (03/07/12).
	       # March 8, 2012 (kam)

		cond[1] = ['spo_01D0', 'psa_01D0']
		cond[2] = ['smo_01D0']                    # (pacific basin)
		cond[3] = ['asc_01D0']                    # (atlantic basin)
		cond[4] = ['kum_01D0']                    # (pacific basin)
		cond[5] = ['azr_01D0']                    # (atlantic basin) sometimes intermittent
		cond[6] = ['brw_01D0']                    # (north pacific)
		cond[7] = ['stm_01D0']                    # (north atlantic) long record but ends in 2009
		cond[8] = ['ice_01D0', 'zep_01D0']        # (north atlantic) start in 1990s

	elif gas == 'ch4':
		cond[1] = ['spo_01D0', 'psa_01D0']
		cond[2] = ['smo_01D0']
		cond[3] = ['asc_01D0']
		cond[4] = ['kum_01D0']
		cond[5] = ['cba_01D0', 'stm_01D0', 'brw_01D0']

	elif gas == 'n2o':
		cond[1] = ['cgo_01D0']
		cond[2] = ['smo_01D0', 'asc_01D0']
		cond[3] = ['kum_01D0']
		cond[4] = ['brw_01D0']

	elif gas == 'sf6':
		cond[1] = ['cgo_01D0']
		cond[2] = ['smo_01D0', 'asc_01D0']
		cond[3] = ['kum_01D0']
		cond[4] = ['brw_01D0']

	else:
		raise ValueError("Unknown gas type in bs_network.")

	# include the required conditions in the new network
	# but only if they appear in the original site list
	fakenet = []
	for n in cond:
		item = cond[n]
		s = random.choice(item)
		if s in sites:
			fakenet.append(s)


	# fill in the rest of the network with randomly chosen sites
	nstart = len(fakenet)
	for i in range(nstart, nsites):
		rn = random.randint(0, nsites-1)
		fakenet.append(sites[rn])


	return sorted(fakenet)

if __name__ == "__main__":
	import ccg_readinit

	gas = "co2"
	initfile = "/ccg/dei/ext/%s/work/init.%s.flask.trends.txt" % (gas, gas)
#	initfile = "/ccg/dei/ext/work/init.naboundary.atlantic.txt"

	initparam = ccg_readinit.ccg_readinit(initfile)

	print(initparam['site'])

	sites = initparam['site']

	f = get_network_sites(sites, gas)
#	f.sort()
	print(f)

#	ccg_readinit.ccg_writeinit("init.test", initparam, f)
