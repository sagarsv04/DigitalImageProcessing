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


# Activation Functions

def sigmoid(X):
	return 1/(1+np.exp(-X))

def relu(X):
	return np.maximum(0,X)

def leaky_relu(X):
	return np.maximum(0.1 * X, X)

def softmax(X):
	expo = np.exp(X)
	expo_sum = np.sum(np.exp(X))
	return expo/expo_sum

def tanh(X):
	return np.tanh(X)

# passing reference of activation function
activation_function = sigmoid


def truncated_normal(mean=0, sd=1, low=0, upp=10):
	'''To generate random numbers with normal distribution'''
	# mean,sd,low,upp=0,1,0,10
	return truncnorm((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)


# Calculate the derivative of an neuron output
def transfer_derivative(output_vector, output_errors):
	# output_vector.shape, output_errors.shape
	return output_errors * output_vector * (1.0 - output_vector)


def get_one_hot_vector_array(labels):

	print("Creating One Hot Vector for Labels ...")
	one_hot_vector_size = np.unique(labels).shape[0]
	labels_one_hot_array = np.zeros((labels.shape[0], one_hot_vector_size))

	for idx in range(labels.shape[0]):
		# idx = 1
		labels_one_hot_array[idx][labels[idx]] = 1

	return labels_one_hot_array


def load_model(save_name):
	model = None
	if os.path.exists("./out/{0}.pkl".format(save_name)):
		with open("./out/{0}.pkl".format(save_name), "rb") as file:
			model = pickle.load(file)
	else:
		print("No File ... {0}.pkl".format(save_name))
	return model


class DeepNN:

	def __init__(self, network_structure, learning_rate, bias=None):
		# network_structure = [28*28, 28*28*2, 28*28*2 , 10]
		# learning_rate = 0.01
		# bias = 1
		self.structure = network_structure  # ie. [input_nodes, hidden1_nodes, ... , hidden_n_nodes, output_nodes]
		# structure = network_structure
		self.no_of_layers = len(self.structure)
		# no_of_layers = len(structure)
		self.learning_rate = learning_rate
		self.bias = bias
		self.weights_matrices = []

	def initializing(self):

		self.create_weight_matrices()
		return 0


	def create_weight_matrices(self):
		'''To generate random weights between neural layers'''

		if self.bias:
			bias_node = 1
		else:
			bias_node = 0

		print("Initializing Weight Matrices ...")
		for layer_index in tqdm(range(self.no_of_layers-1)):
			# layer_index = 2
			input_nodes = self.structure[layer_index]
			# input_nodes = structure[layer_index]
			output_nodes = self.structure[layer_index+1]
			# output_nodes = structure[layer_index+1]
			total_node = (input_nodes + bias_node) * output_nodes
			# rad = 1 / np.sqrt(784)
			rad = 1 / np.sqrt(input_nodes)
			X = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)
			# weights_matrix = X.rvs(n).reshape((1568, 784 + 1))
			weights_matrix = X.rvs(total_node).reshape((output_nodes, input_nodes + bias_node))
			# weights_matrix.shape
			self.weights_matrices.append(weights_matrix)

		return 0

	def update_weights(self, layer_index, output_derivative):
		'''To update weights of neural layer
			: output_derivative is ndarray of delta values
		'''
		self.weights_matrices[layer_index-1] += self.learning_rate * output_derivative
		# self.weights_matrices[layer_index-1] += learning_rate * output_derivative
		# weights_matrices[1].shape
		return 0

	def get_output_errors(self, layer_index, output_errors):
		'''To calculate new output error
			: output_errors is target_vector - output_vector
		'''
		output_errors = np.dot(self.weights_matrices[layer_index-1].T, output_errors)
		# output_errors = np.dot(weights_matrices[layer_index-1].T, output_errors)
		return output_errors

	def train_single_data(self, input_vector, target_vector):
		# input_vector, target_vector = data[data_idx], labels_one_hot_array[data_idx]
		'''To train weights between neural layers
			: input_vector is ndarray input data
			: target_vector is ndarray(mnist 1d) data lable
		'''
		input_vector = np.array(input_vector, ndmin=2).T
		# print("input_vector.shape", input_vector.shape)
		# The output computed here becomes input vectors for the next layers:
		res_vectors = [input_vector]
		for layer_index in range(self.no_of_layers-1):
			# layer_index = 2
			input_vector = res_vectors[-1]
			# input_vector.shape
			if self.bias:
				# adding bias node to the end of the 'input_vector'
				input_vector = np.concatenate((input_vector, [[self.bias]]))
				# input_vector = np.concatenate((input_vector, [[bias]]))
				res_vectors[-1] = input_vector
			# Taking dot product of nth layer wieghts and input_vector
			X = np.dot(self.weights_matrices[layer_index], input_vector)
			# X = np.dot(weights_matrices[layer_index], input_vector)
			output_vector = activation_function(X)
			res_vectors.append(output_vector)

		target_vector = np.array(target_vector, ndmin=2).T

		# The input->output vectors of the various layers:
		output_errors = target_vector - res_vectors[-1]
		# output_errors.shape
		for layer_index in reversed(range(1, self.no_of_layers)):
			# layer_index = 1 #
			output_vector = res_vectors[layer_index]
			# output_vector.shape
			input_vector = res_vectors[layer_index-1]
			# input_vector.shape
			if self.bias:
				# layer_index!=(no_of_layers-1)
				if layer_index!=(self.no_of_layers-1):
					output_vector = output_vector[:-1,:].copy()
					output_errors = output_errors[:-1:].copy()

			output_derivative = transfer_derivative(output_vector, output_errors)
			# output_derivative.shape
			output_derivative = np.dot(output_derivative, input_vector.T)
			# output_derivative.shape
			self.update_weights(layer_index, output_derivative)
			output_errors = self.get_output_errors(layer_index, output_errors)

		return 0

	def train(self, data, labels, num_epochs, intermediate_results=False):
		# data, labels = train_data, train_labels
		# num_epochs = 3
		'''To train all the weights of neural layers
			: input_vector is ndarray input data
			: target_vector is ndarray(mnist 1d) data lable
		'''
		labels_one_hot_array = get_one_hot_vector_array(labels)

		intermediate_weights = []
		for epoch in range(num_epochs):
			# epoch = 0
			print("Training Epoch ... {0}".format(epoch+1))
			# epoch = 0 # number of training iterations
			for data_idx in tqdm(range(data.shape[0])):
				# data_idx = 0 # nth training data
				self.train_single_data(data[data_idx], labels_one_hot_array[data_idx])

		return 0

	def predict(self, input_vector):
		input_vector = np.array(input_vector, ndmin=2).T
		for layer_index in range(self.no_of_layers):
			if self.bias:
				input_vector = np.concatenate((input_vector, [[self.bias]]))
			X = np.dot(self.weights_matrices[layer_index], input_vector)
			output_vector = activation_function(X)
			input_vector = output_vector

		res_max = output_vector.argmax()
		return res_max

	def test(self, data, labels):
		# data, labels = test_data, test_labels
		labels_one_hot_array = get_one_hot_vector_array(labels)
		correct_count = 0
		print("Testing Model ...")
		for data_idx in tqdm(range(data.shape[0])):
			# data_idx = 0 # nth training data
			res_max = self.predict(data[data_idx])
			if res_max==labels[data_idx]:
				correct_count += 1
		print("Accuracy ... {0}".format(correct_count/data.shape[0]))
		return 0

	def save(self, save_name):

		print("Saving Model ...")
		with open("./out/{0}.pkl".format(save_name), "wb") as file:
			pickle.dump(self, file)
		return 0


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
