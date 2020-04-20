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


print("heheheheheheh")


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


class ARGS():
	def __init__(self):
		self.image_size=28
		self.num_epochs=8
		self.batch_size=100
		self.num_workers=2
		self.lrD=0.00002
		self.lrG=0.0002
		self.beta1=0.5
		self.beta2=0.999
		self.dim_real=62
		self.db='mnist'
		self.model_path='./out'
		self.sample_path='./results'
		self.sample_size=200
		self.log_step=20
		self.sample_step=100
		self.seed=3145


def print_discriminator(dc_out_real, dc_out_fake, gc_caps, sf_real, dc):
	# print c caps of discriminator
	l1, l2, l3, l4 = [], [], [], []
	c_norm_real = torch.norm(dc_out_real, dim=-1)
	c_norm_fake = torch.norm(dc_out_fake, dim=-1)
	cg_norm = torch.norm(gc_caps, dim=-1)
	for k in range(c_norm_fake.size()[1]):
		l1.append(round(c_norm_real[0, k].item(), 5))
		l2.append(round(c_norm_fake[0, k].item(), 5))
		l3.append(round(cg_norm[0, k].item(), 5))
		l4.append(dc[0,k].item())

	print('\n\nc_norm_real {} \n\nc_norm_fake {} \n\ncg_norm {} \n\ndc {}'.format(l1,l2,l3,l4))
	print('*'*100)
	_, freq_cls = torch.max(sf_real, dim=-1)
	freq_cls = np.unique(freq_cls.cpu(), return_counts=True)
	freq_cls = np.stack((freq_cls[0], freq_cls[1]), axis=-1)
	print(freq_cls)
	print('*'*100)
	return 0


def print_generator(dp_out_fake, gp_caps):

	l1, l2 = [], []
	randint = [np.random.randint(0, gp_caps.size()[1]) for i in range(300)]
	p_norm_fake = torch.norm(dp_out_fake, dim=-1)
	gp_norm = torch.norm(gp_caps, dim=-1)
	for k in range(300):
		l1.append(round(p_norm_fake[0, randint[k]].item(), 5))
		l2.append(round(gp_norm[0, randint[k]].item(), 5))
	print('*'*100)
	print('p_norm_fake\n {}'.format(l1))
	print('-'*100)
	print('gp_norm\n {}'.format(l2))
	print('*'*100)
	return 0


def generate_images(generator, epoch, batch_id):

	real_struc = torch.randn(100, 1, args.dim_real - 10)
	tmp = np.zeros((100, 10))
	for k in range(10):
		tmp[k * 10:(k + 1) * 10, k] = 1
	tmp = torch.Tensor(tmp)
	real_struc = torch.cat((real_struc, tmp.unsqueeze(1)), dim=-1)
	real_struc = to_variable(real_struc)
	real_struc = gen.squash(real_struc)
	fake_images, _, _ = generator(real_struc, epoch)
	torchvision.utils.save_image(denorm(fake_images.data),
			os.path.join(args.sample_path, 'generated-%d-%d.png' % (epoch + 1, batch_id + 1)), nrow=10)
	return 0



def tarin_test_model(is_train, is_generate):

	generator = None
	discriminator = None

	mnist_data = MNISTData(args.batch_size, args.num_workers)

	if is_train:
		try:
			# Networks
			generator = gen.Generator(in_caps=6*6*32, num_caps=10, in_dim=8, dim_caps=16, dim_real=args.dim_real)
			discriminator = dis.Discriminator(img_shape=(1,28,28), channels=256, primary_dim=8, num_classes=10, out_dim=16, dim_real=args.dim_real, num_routing=3)
			# Optimizers
			g_optimizer = optim.Adam(generator.parameters(), args.lrG, [args.beta1, args.beta2])
			d_optimizer = optim.Adam(discriminator.parameters(), args.lrD, [args.beta1, args.beta2])

			if torch.cuda.is_available():
				generator.cuda()
				discriminator.cuda()

			# setup loss function
			criterion = nn.BCELoss().cuda()
			margin = CapsuleLoss()
			total_step = len(mnist_data.train_loader)
			for epoch in tqdm(range(args.num_epochs)):
				# epoch = 0
				print("Traning Epoch ... {0}".format(epoch))
				for batch_id, (data, target) in enumerate(mnist_data.train_loader):
					# batch_id = 0
					# print("Train Batch Number ... {0}".format(batch_id))
					# data, target = list(mnist_data.train_loader)[0]
					# ===================== Train Discriminator =====================#
					images = to_variable(data)
					batch_size = images.size(0)
					# reality_structure
					real_struc = torch.randn(batch_size, 1, args.dim_real - 10) # -> (batch_size, 1, dim_real)
					real_struc = to_variable(real_struc)
					# real_struc = Variable(real_struc)
					dc = gen_dc(batch_size, 10)
					dc = to_variable(dc)
					# dc = Variable(dc)
					real_struc = torch.cat((real_struc, dc.unsqueeze(1)), dim=-1)
					real_struc = gen.squash(real_struc)

					fake_images, gp_caps, gc_caps = generator(real_struc, epoch)
					d_out_norm_real, dp_out_real, dc_out_real, dr_out_real, sf_real = discriminator(images)
					d_out_norm_fake, dp_out_fake, dc_out_fake, dr_out_fake, sf_fake = discriminator(fake_images)

					# Mutual Information Loss
					#d_loss_dc = -(torch.mean(torch.sum(dc * sf_fake, 1)) + 1)
					d_loss_dc = margin(sf_fake, dc)*1e-02 + 1
					# classes capsules
					gc_caps= gc_caps.squeeze(1)
					dist_c_caps = torch.sum((dc_out_fake - gc_caps)**2, -1)**0.5
					loss_c_caps = torch.mean(torch.sum(dist_c_caps, -1))
					# primary capsules
					dist_p_caps = torch.sum((gp_caps - dp_out_fake)**2, -1)**0.5
					loss_p_caps = torch.mean(torch.sum(dist_p_caps, -1))
					# Cosine Similarity
					# classes capsules
					cos_c = F.cosine_similarity(gc_caps, dc_out_fake, dim=-1)
					cos_c = torch.mean(torch.sum(cos_c, dim=-1))
					# primary capsules
					cos_p = F.cosine_similarity(gp_caps, dp_out_fake, dim=-1)
					cos_p = torch.mean(torch.sum(cos_p, dim=-1))
					d_loss_a = -torch.mean(torch.log(d_out_norm_real[:,0]) + torch.log(1 - d_out_norm_fake[:,0]))
					d_loss = d_loss_a + 1.0*d_loss_dc

					# Optimization
					discriminator.zero_grad()
					d_loss.backward(retain_graph=True)
					d_optimizer.step()

					# ===================== Train Generator =====================#
					# Fake -> Real
					g_loss_a = -torch.mean(torch.log(d_out_norm_fake[:,0]))
					g_loss = g_loss_a + 1.0*d_loss_dc
					# Optimization
					generator.zero_grad()
					g_loss.backward()
					g_optimizer.step()

					if (batch_id + 1) % (args.log_step + 100) == 0:
						print_discriminator(dc_out_real, dc_out_fake, gc_caps, sf_real, dc)
						pass

					if (batch_id + 1) % (args.log_step + 400) == 0:
						print_generator(dp_out_fake, gp_caps)
						pass

					# print the log info
					if (batch_id + 1) % args.log_step == 0:
						print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, dc_loss: %.4f, loss_p_caps: %.4f, loss_c_caps: %.4f, cos_p: %.4f, cos_c: %.4f'
						% (epoch + 1, args.num_epochs, batch_id + 1, total_step, d_loss, g_loss, d_loss_dc, loss_p_caps, loss_c_caps, cos_p, cos_c ))

					# save the sampled images (10 Category(Discrete), 10 Continuous Code Generation : 10x10 Image Grid)
					if (batch_id + 1) % args.sample_step == 0:
						generate_images(generator, epoch, batch_id)
						pass

				# save the model parameters for each epoch
				g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
				torch.save(generator.state_dict(), g_path)
			torch.save(generator.state_dict(), "./out/mnist_generator.pkl")
		except Exception as ex:
			print("Training Failed ...")
			print("Error:: {0}".format(ex))

	if is_generate:
		try:
			print("The Generate function is not ready yet")
		except Exception as ex:
			print("Generate Failed ...")
			print("Error:: {0}".format(ex))

	return 0



def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Generative model based on Capsule Network")
	parser.add_argument("train", type=int,help="1 to Train model, 0 to !Train model")
	parser.add_argument("generate", type=int,help="1 to Generate Images, 0 to !Generate Images")
	# model hyper-parameters
	parser.add_argument("--image_size", type=int, default=28) # 28 for MNIST
	# training hyper-parameters
	parser.add_argument("--num_epochs", type=int, default=8) # 30 or 50 for MNIST
	parser.add_argument("--batch_size", type=int, default=100)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--lrD", type=float, default=0.00002) # Learning Rate for D
	parser.add_argument("--lrG", type=float, default=0.0002) # Learning Rate for G
	parser.add_argument("--beta1", type=float, default=0.5)  # momentum1 in Adam
	parser.add_argument("--beta2", type=float, default=0.999)  # momentum2 in Adam
	# Generator and Discriminator hyperparameters
	parser.add_argument("--dim_real", type=int, default=62)
	# misc
	parser.add_argument("--db", type=str, default="mnist_gan")  # Model Tmp Save
	parser.add_argument("--model_path", type=str, default="./out/")  # Model Tmp Save
	parser.add_argument("--sample_path", type=str, default="./results/")  # Results
	parser.add_argument("--sample_size", type=int, default=100)
	parser.add_argument("--log_step", type=int, default=20)
	parser.add_argument("--sample_step", type=int, default=100)
	# seed
	parser.add_argument("--seed", type=int, default=3145) # 28 for MNIST

	global args
	args = parser.parse_args()
	# args = ARGS()

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	if not os.path.exists(args.sample_path):
		os.makedirs(args.sample_path)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	if args.train and args.generate:
		tarin_test_model(True, True)
	elif args.train and not args.generate:
		tarin_test_model(True, False)
	elif not args.train and args.generate:
		tarin_test_model(False, True)
	else:
		print("Thanks for wasiting your time!")
	return 0


if __name__ == "__main__":
	main()
