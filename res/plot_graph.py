#/usr/bin/pthon
import os
import sys
if __name__ == '__main__':
	os.chdir("../") # the base directory while running any code
sys.path.append("{0}/res/".format(os.getcwd()))
sys.path.append("{0}/dnn/".format(os.getcwd()))
sys.path.append("{0}/capsule/".format(os.getcwd()))
sys.path.append("{0}/gan/".format(os.getcwd()))
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import data_processing as dp
from mnist_dnn import MyNN
from cap_dnn import CapsNet, ConvLayer, PrimaryCaps, DigitCaps, Decoder
from gan_dnn import GAN



def plot_loss_graph(graph_name, loss_array, epoch_array, train_acc=None, test_acc=None):
	# graph_name, loss_array, epoch_array = "mnist_dnn", loss_array, epoch_size
	fig, ax = plt.subplots()
	ax.plot(epoch_array, loss_array)
	ax.set(xlabel="Epoch (s)", ylabel="Loss (s)", title="Loss graph for model {0}".format(graph_name))
	ax.grid()
	if train_acc != None:
		plt.text(0.80, 0.85, "Train: {}%".format(train_acc), fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	if test_acc != None:
		plt.text(0.80, 0.75, "Test: {}%".format(test_acc), fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	fig.savefig("./out/{0}_loss.png".format(graph_name))
	plt.show()
	return 0


def plot_score_graph(graph_name, score_array, epoch_array, fake=False):
	# graph_name, loss_array, epoch_array = "mnist_dnn", loss_array, epoch_size
	fig, ax = plt.subplots()
	if fake:
		ax.plot(epoch_array, score_array, "r")
		ax.set(xlabel="Epoch (s)", ylabel="Score (s)", title="Fake Score graph for model {0}".format(graph_name))
	else:
		ax.plot(epoch_array, score_array, "b")
		ax.set(xlabel="Epoch (s)", ylabel="Score (s)", title="Real Score graph for model {0}".format(graph_name))
	ax.grid()
	if fake:
		fig.savefig("./out/{0}_fake_score.png".format(graph_name))
	else:
		fig.savefig("./out/{0}_real_score.png".format(graph_name))
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

	if args.cap_dnn:
		model = dp.load_model("mnist_capsule")
		print("Creating loss array from Tensors ...")
		loss_list = dp.get_numpy_array(model.loss_list)
		epoch = 8 # copied from cap_dnn.py EPOCHS
		train_acc = model.train_accuracy
		test_acc = model.test_accuracy
		loss_array = [loss_list[int(idx*(loss_list.shape[0]/epoch))] for idx in range(epoch)]
		loss_array = np.array(loss_array)
		epoch_size = np.arange(1, epoch+1, 1)
		plot_loss_graph("cap_dnn", loss_array, epoch_size, train_acc, test_acc)

	if args.gan_dnn:
		model = dp.load_model("mnist_gan")
		d_losses_array = model.d_losses
		g_losses_array = model.g_losses
		real_scores_array = model.real_scores
		fake_scores_array = model.fake_scores
		epoch = fake_scores_array.shape[0]
		epoch_size = np.arange(1, epoch+1, 1)
		# g_losses_array = np.flip(g_losses_array, axis=None)
		plot_loss_graph("gan_dnn_generator", g_losses_array, epoch_size)
		plot_loss_graph("gan_dnn_discriminator", d_losses_array, epoch_size)
		plot_score_graph("gan_dnn", real_scores_array, epoch_size)
		plot_score_graph("gan_dnn", fake_scores_array, epoch_size, True)

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
