#/usr/bin/pthon
import os
import sys
os.chdir("../") # the base directory while running any code
sys.path.append("{0}/res/".format(os.getcwd()))
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import data_processing as dp
import matplotlib.pyplot as plt




def generate_model_graph():

	if args.mnist_dnn:


		pass
	if args.mnist_dnn:


		pass
	if args.mnist_dnn:


		pass

	return 0


def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Generate Graphs for Models")
	parser.add_argument("mnist_dnn", type=bool, default=False, help="mnist_dnn=True to Generate Graph for DNN model")
	parser.add_argument("cap_dnn", type=bool, default=False, help="cap_dnn=True to Generate Graph for Capsule model")
	parser.add_argument("gan_dnn", type=bool, default=False, help="gan_dnn=True to Generate Graph for Gan model")
	global args
	args = parser.parse_args()
	generate_model_graph()
	return 0


if __name__ == '__main__':
	main()
