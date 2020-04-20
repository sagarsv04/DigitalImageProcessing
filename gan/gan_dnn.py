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
import torch.nn as nn
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


class MNISTData():
	def __init__(self, batch_size, num_workers):
		dataset_transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307,), (0.3081,))
					])
		train_dataset = datasets.MNIST("./data/", train=True, download=True, transform=dataset_transform)
		test_dataset = datasets.MNIST("./data/", train=False, download=True, transform=dataset_transform)
		self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
		self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


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

	generator = None
	discriminator = None

	mnist_data = MNISTData(args.batch_size, args.num_workers)

	if is_train:
		try:
			# Networks
			generator = Generator(in_caps=6*6*32, num_caps=10, in_dim=8, dim_caps=16, dim_real=args.dim_real)
			discriminator = Discriminator(img_shape=(1,28,28), channels=256, primary_dim=8, num_classes=10, out_dim=16, dim_real=args.dim_real, num_routing=3)
			# Optimizers
			g_optimizer = optim.Adam(generator.parameters(), args.lrG, [args.beta1, args.beta2])
			d_optimizer = optim.Adam(discriminator.parameters(), args.lrD, [args.beta1, args.beta2])

			if torch.cuda.is_available():
				generator.cuda()
				discriminator.cuda()

			# setup loss function
			criterion = nn.BCELoss().cuda()
			margin = CapsuleLoss()






			dnn_model.save("mnist_dnn")
		except Exception as ex:
			print("Training Failed ...")
			print("Error:: {0}".format(ex))

	if is_test:
		try:
			train_data, train_labels = dp.load_data()
			dnn_model = DeepNN([28*28, 28*28*2, 28*28*2 , 10], 0.1, True)
			dnn_model.initializing()
			dnn_model.train(train_data, train_labels, 8)
			dnn_model.save("mnist_dnn")
		except Exception as ex:
			print("Testing Failed ...")
			print("Error:: {0}".format(ex))

	if is_generate:
		try:
			train_data, train_labels = dp.load_data()
			dnn_model = DeepNN([28*28, 28*28*2, 28*28*2 , 10], 0.1, True)
			dnn_model.initializing()
			dnn_model.train(train_data, train_labels, 8)
			dnn_model.save("mnist_dnn")
		except Exception as ex:
			print("Generate Failed ...")
			print("Error:: {0}".format(ex))

	return 0



def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Generative model based on Capsule Network")
	parser.add_argument("train", type=int,help="1 to Train model, 0 to !Train model")
	parser.add_argument("test", type=int,help="1 to Test model, 0 to !Test model")
	parser.add_argument("generate", type=int,help="1 to Generate Images, 0 to !Generate Images")
	# model hyper-parameters
	parser.add_argument("--image_size", type=int, default=28) # 28 for MNIST
	# training hyper-parameters
	parser.add_argument("--num_epochs", type=int, default=8) # 30 or 50 for MNIST
	parser.add_argument("--batch_size", type=int, default=200)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--lrD", type=float, default=0.00002) # Learning Rate for D
	parser.add_argument("--lrG", type=float, default=0.0002) # Learning Rate for G
	parser.add_argument("--beta1", type=float, default=0.5)  # momentum1 in Adam
	parser.add_argument("--beta2", type=float, default=0.999)  # momentum2 in Adam
	# Generator and Discriminator hyperparameters
	parser.add_argument("--dim_real", type=int, default=62)
	# misc
	parser.add_argument("--db", type=str, default="mnist_gan")  # Model Tmp Save
	parser.add_argument("--model_path", type=str, default="./out/")  # Model Tmp Save
	parser.add_argument("--sample_path", type=str, default="./results")  # Results
	parser.add_argument("--sample_size", type=int, default=100)
	parser.add_argument("--log_step", type=int, default=20)
	parser.add_argument("--sample_step", type=int, default=100)
	# seed
	parser.add_argument("--seed", type=int, default=3145) # 28 for MNIST

	global args
	args = parser.parse_args()
	# print(args)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

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


if __name__ == "__main__":
	main()
