#/usr/bin/pthon
from sklearn.datasets import fetch_mldata
import numpy as np
import pandas as pd



def load_data():
	data = fetch_mldata("MNIST original")
	return data


def split_data(data):
	train = fetch_mldata("MNIST original")
	train_lable = fetch_mldata("MNIST original")
	test = fetch_mldata("MNIST original")
	test_lable = fetch_mldata("MNIST original")
	return train, train_lable, test, test_lable








def main():

	return 0



if __name__ == '__main__':
	main()
