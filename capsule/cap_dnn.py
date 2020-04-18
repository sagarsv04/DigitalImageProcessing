#/usr/bin/pthon
import os
import sys
os.chdir("../") # the base directory while running any code
sys.path.append("{0}/res/".format(os.getcwd()))
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import matplotlib.pyplot as plt
import data_processing as dp
from scipy.stats import truncnorm


def tarin_test_model(is_train, is_test):

	dnn_model = None
	train_data, train_labels = None, None
	test_data, test_labels = None, None

	try:
		if is_train:
			train_data, train_labels = dp.load_data()
			dnn_model = DeepNN([28*28, 28*28*2, 28*28*2 , 10], 0.1, True)
			dnn_model.initializing()
			dnn_model.train(train_data, train_labels, 8)
			dnn_model.save("mnist_dnn")

		if is_test:
			test_data, test_labels = dp.load_data(False)
			dnn_model = load_model("mnist_dnn")
			if dnn_model != None:
				dnn_model.test(test_data, test_labels)
	except Exception as ex:
		print("Error:: {0}".format(ex))

	return 0



def main():

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("train", type=int,help="1 to Train model, 0 to !Train model")
	parser.add_argument("test", type=int,help="1 to Test model, 0 to !Test model")
	args = parser.parse_args()

	if args.train and args.test:
		tarin_test_model(True, True)
	elif args.train:
		tarin_test_model(True, False)
	elif args.test:
		tarin_test_model(False, True)
	else:
		print("Thanks for wasiting your time!")
	return 0


if __name__ == '__main__':
	main()
