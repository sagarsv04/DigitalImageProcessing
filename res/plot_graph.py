#/usr/bin/pthon
import os
import sys
if __name__ == '__main__':
	os.chdir("../") # the base directory while running any code
sys.path.append("{0}/res/".format(os.getcwd()))
sys.path.append("{0}/dnn/".format(os.getcwd()))
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import data_processing as dp
import matplotlib.pyplot as plt
from mnist_dnn import MyNN



def plot_loss_graph(graph_name, loss_array, epoch_array, train_acc, test_acc):
	# graph_name, loss_array, epoch_array = "mnist_dnn", loss_array, epoch_size
	fig, ax = plt.subplots()
	ax.plot(epoch_array, loss_array)
	ax.set(xlabel="Epoch (s)", ylabel="Loss (s)", title="Loss graph for model {0}".format(graph_name))
	ax.grid()
	plt.text(0.80, 0.85, "Train: {}%".format(train_acc), fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	plt.text(0.80, 0.75, "Test: {}%".format(test_acc), fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	fig.savefig("./out/{0}_loss.png".format(graph_name))
	plt.show()
	return 0


def generate_model_graph():
	model = None
	if args.mnist_dnn:
		model = dp.load_model("mnist_dnn")
		loss_array = np.array(model.loss)
		epoch = model.epoch
		train_acc = model.train_accuracy
		test_acc = model.test_accuracy
		epoch_size = np.arange(1, epoch+1, 1)
		plot_loss_graph("mnist_dnn", loss_array, epoch_size, train_acc, test_acc)

	if args.mnist_dnn:


		pass
	if args.mnist_dnn:


		pass

	return 0


def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Generate Graphs for Models")
	parser.add_argument("mnist_dnn", type=int, help="mnist_dnn = 1 to Generate Graph for DNN model")
	parser.add_argument("cap_dnn", type=int, help="cap_dnn = 1 to Generate Graph for Capsule model")
	parser.add_argument("gan_dnn", type=int, help="gan_dnn = 1 to Generate Graph for Gan model")
	global args
	args = parser.parse_args()
	generate_model_graph()
	return 0


if __name__ == '__main__':
	main()
