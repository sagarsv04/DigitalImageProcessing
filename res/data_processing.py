#/usr/bin/pthon
import os
import numpy as np
import pickle
from mnist import MNIST
import matplotlib.pyplot as plt



def read_data(train_data=True):
	# train=False
	mnist_data = MNIST("./data/")
	mnist_data.gz = True
	data, labels = None, None
	if train_data:
		print("Loading Training Data ...")
		data, labels = mnist_data.load_training()
	else:
		print("Loading Testing Data ...")
		data, labels = mnist_data.load_testing()

	data = np.array(data)
	labels = np.array(labels)
	# data[0], labels[0]
	# len(data), len(labels)
	return data, labels



def split_data(data, labels, ratio):

	# data = images
	ratio = 0.5

	if 0 < ratio < 1:
		train_size = int(ratio*data.shape[0])
		data_bool = np.zeros(data.shape[0], dtype=bool)
		data_bool[:train_size] = True
		train = data[data_bool]
		train_lable = labels[data_bool]
		# train.shape, train_lable.shape
		test = data[~data_bool]
		test_lable = labels[~data_bool]
		# test.shape, test_lable.shape
	else:
		print("Ratio Value Invalid :: 0 < ratio < 1")
		train, test = None, None
		train_lable, test_lable = None, None

	return train, train_lable, test, test_lable



def save_data(data, labels, train_data=True):

	if train_data:
		print("Saving ... training_data.pkl")
		with open("./data/training_data.pkl", "wb") as file:
			pickle.dump([data, labels], file)
	else:
		print("Saving ... testing_data.pkl")
		with open("./data/testing_data.pkl", "wb") as file:
			pickle.dump([data, labels], file)

	return 0


def load_data(train_data=True):

	data, labels = None, None
	if train_data:
		print("Loading ... training_data.pkl")
		if os.path.exists("./data/training_data.pkl"):
			with open("./data/training_data.pkl", "rb") as file:
				data, labels = pickle.load(file)
		else:
			print("No File ... training_data.pkl")
	else:
		print("Loading ... testing_data.pkl")
		if os.path.exists("./data/training_data.pkl"):
			with open("./data/testing_data.pkl", "rb") as file:
				data, labels = pickle.load(file)
		else:
			print("No File ... testing_data.pkl")

	return data, labels



def main():
	train_data, train_labels = read_data()
	save_data(train_data, train_labels)

	test_data, test_labels = read_data(False)
	save_data(test_data, test_labels, False)

	# train_data, train_labels = load_data()

	return 0



if __name__ == '__main__':
	main()
