import numpy as np
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	data = np.loadtxt(args.filename)
	x = data[:, 0]/max(data[:, 0])
	y = data[:, 1]
	ALC = np.trapz(y, x)
	# ALC_rand = ALC
	Amax = 1.0
	print("ALC: ", ALC)
	# global_score = (ALC - ALC_rand)/(Amax-ALC_rand)
	# print("Global score: ", global_score)