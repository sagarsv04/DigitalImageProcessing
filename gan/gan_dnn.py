#/usr/bin/pthon
import os
import sys
os.chdir("../") # the base directory while running any code
sys.path.append("{0}/gan/".format(os.getcwd()))
import numpy as np
from tqdm import tqdm
import pickle
import random
import argparse
import torch
import torchvision
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
import discriminator as dis
import generator as gen
from PIL import Image
import torch.nn.functional as F
import scipy.stats as stats


def to_variable(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)


def denorm(x):
	out = (x + 1) / 2
	return out.clamp(0, 1)


def gen_dc(n_size, dim):
	codes=[]
	code = np.zeros((n_size, dim))
	random_cate = np.random.randint(0, dim, n_size)
	code[range(n_size), random_cate] = 1.0
	codes.append(code)
	codes = np.concatenate(codes,1)
	return torch.Tensor(codes)



class MarginLoss(nn.Module):
	def __init__(self, size_average=False, loss_lambda=0.5):
		super(MarginLoss, self).__init__()
		self.size_average = size_average
		self.m_plus = 0.9
		self.m_minus = 0.1
		self.loss_lambda = loss_lambda

	def forward(self, inputs, labels):
		L_k = labels * F.relu(self.m_plus - inputs)**2 + self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus)**2
		L_k = L_k.sum(dim=1)

		if self.size_average:
			return L_k.mean()
		else:
			return L_k.sum()


class CapsuleLoss(nn.Module):
	def __init__(self, loss_lambda=0.5, recon_loss_scale=5e-4, size_average=False):
		super(CapsuleLoss, self).__init__()
		self.size_average = size_average
		self.margin_loss = MarginLoss(size_average=size_average, loss_lambda=loss_lambda)


	def forward(self, inputs, labels):
		margin_loss = self.margin_loss(inputs, labels)
		return margin_loss



















def tarin_test_model(is_train, is_test, is_generate):

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
	parser.add_argument("generate", type=int,help="1 to Generate Images, 0 to !Generate Images")
	args = parser.parse_args()

	if args.train and args.test and args.generate:
		tarin_test_model(True, True, True)
	elif args.train and not args.test and not args.generate:
		tarin_test_model(True, False, False)
	elif not args.train and args.test and not args.generate:
		tarin_test_model(False, True, False)
	elif not args.train and not args.test and args.generate:
		tarin_test_model(False, False, True)
	else:
		print("Thanks for wasiting your time!")
	return 0


if __name__ == '__main__':
	main()
