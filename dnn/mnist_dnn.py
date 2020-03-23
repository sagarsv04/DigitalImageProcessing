#/usr/bin/pthon
import os
import numpy as np
import tqdm as tqdm
import pandas as pd
import matplotlib.pyplot as plt
import res.data_processing as dp
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
activation_function = relu


def truncated_normal(mean=0, sd=1, low=0, upp=10):
	'''To generate random numbers with normal distribution'''
	# mean,sd,low,upp=0,1,0,10
	return truncnorm((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)


# Calculate the derivative of an neuron output
def transfer_derivative(output_vector, output_errors):
	return output_errors * output_vector * (1.0 - output_vector)



class DeepNN:

	def __init__(self, network_structure, learning_rate, bias=None):
		# network_structure = [28*28, 28*28*2, 28*28*2 , 10]
		self.structure = network_structure  # ie. [input_nodes, hidden1_nodes, ... , hidden_n_nodes, output_nodes]
		self.no_of_layers = len(self.structure)
		self.learning_rate = learning_rate
		self.bias = bias
		self.weights_matrices = []
		self.create_weight_matrices()

	def create_weight_matrices(self):
		'''To generate random weights between neural layers'''

		if self.bias:
			bias_node = 1
		else:
			bias_node = 0

		for layer_index in range(self.no_of_layers-1):
			input_nodes = self.structure[layer_index]
			output_nodes = self.structure[layer_index+1]
			total_node = (input_nodes + bias_node) * output_nodes
			# rad = 1 / np.sqrt(784)
			rad = 1 / np.sqrt(input_nodes)
			X = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)
			# weights_matrix = X.rvs(n).reshape((1568, 784 + 1))
			weights_matrix = X.rvs(n).reshape((output_nodes, input_nodes + bias_node))
			# weights_matrix.shape
			self.weights_matrices.append(weights_matrix)

		return 0

	def update_weights(self, layer_index, output_derivative):
		'''To update weights of neural layer
			: output_derivative is ndarray of delta values
		'''
		self.weights_matrices[layer_index-1] += self.learning_rate * output_derivative

		return 0

	def get_output_errors(self, layer_index, output_errors):
		'''To calculate new output error
			: output_errors is target_vector - output_vector
		'''
		output_errors = np.dot(self.weights_matrices[layer_index-1].T, output_errors)

		return output_errors

	def train_single_epoch(self, input_vector, target_vector):
		'''To train weights between neural layers
			: input_vector is ndarray input data
			: target_vector is ndarray(mnist 1d) data lable
		'''
		input_vector = np.array(input_vector, ndmin=2).T

		# The output->input vectors of the various layers:
		res_vectors = [input_vector]
		for layer_index in range(self.no_of_layers-1):
			input_vector = res_vectors[-1]
			if self.bias:
				# adding bias node to the end of the 'input_vector'
				input_vector = np.concatenate((input_vector, [[self.bias]]))
				res_vectors[-1] = input_vector
			# Taking dot product of nth layer wieghts and input_vector
			X = np.dot(self.weights_matrices[layer_index], input_vector)
			output_vector = activation_function(X)
			res_vectors.append(output_vector)

		target_vector = np.array(target_vector, ndmin=2).T

		# The input->output vectors of the various layers:
		output_errors = target_vector - res_vectors[-1]
		for layer_index in reversed(range(1, self.no_of_layers)):
			output_vector = res_vectors[layer_index]
			input_vector = res_vectors[layer_index-1]
			if self.bias
				if layer_index!=(self.no_of_layers-1):
					output_vector = output_vector[:-1,:].copy()

			output_derivative = transfer_derivative(output_vector, output_errors)
			output_derivative = np.dot(output_derivative, input_vector.T)
			update_weights(layer_index, output_derivative)
			output_errors = get_output_errors(layer_index, output_errors)

			if self.bias:
				output_errors = output_errors[:-1,:]
		return 0

	def train_dnn(self, data, labels, num_epochs, intermediate_results=False):

		labels_one_hot_array = get_one_hot_vector_array(labels)

		intermediate_weights = []
		for epoch in range(num_epochs):
			print("Training Epoch ... {0}".format(epoch+1))
			# epoch = 0 # number of training iterations
			for data_idx in tqdm(range(len(data.shape[0]))):
				# data_idx = 0 # nth training data
				self.train_single(data[data_idx], labels_one_hot_array[data_idx])

		return 0



def main():

	return 0



if __name__ == '__main__':
	main()
