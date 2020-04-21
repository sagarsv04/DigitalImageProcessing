#/usr/bin/pthon
import os
import sys
os.chdir("../") # the base directory while running any code
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
from tqdm import tqdm
import pickle
import argparse


np.random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 200
EPOCHS = 30


class MNISTData():
	def __init__(self, batch_size):
		# batch_size = BATCH_SIZE
		dataset_transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307,), (0.3081,))
					])
		train_dataset = datasets.MNIST("./data/", train=True, download=True, transform=dataset_transform)
		test_dataset = datasets.MNIST("./data/", train=False, download=True, transform=dataset_transform)
		self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Discriminator
def discriminator_model(image_size, hidden_size):
	model = nn.Sequential(
		nn.Linear(image_size, hidden_size),
		nn.LeakyReLU(0.2),
		nn.Linear(hidden_size, hidden_size),
		nn.LeakyReLU(0.2),
		nn.Linear(hidden_size, 1),
		nn.Sigmoid())
	return model

# Generator
def generator_model(image_size, latent_size, hidden_size):
	model = nn.Sequential(
		nn.Linear(latent_size, hidden_size),
		nn.ReLU(),
		nn.Linear(hidden_size, hidden_size),
		nn.ReLU(),
		nn.Linear(hidden_size, image_size),
		nn.Tanh())
	return model


def denorm(x):
	out = (x + 1) / 2
	return out.clamp(0, 1)


class GAN(nn.Module):
	def __init__(self, num_epochs, image_size, latent_size, hidden_size):
		super(GAN, self).__init__()

		self.d_losses = np.zeros(num_epochs)
		self.g_losses = np.zeros(num_epochs)
		self.real_scores = np.zeros(num_epochs)
		self.fake_scores = np.zeros(num_epochs)

		self.generator = generator_model(image_size, latent_size, hidden_size)
		self.discriminator = discriminator_model(image_size, hidden_size)
		self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
		self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)

	def reset_grad(self):
		self.d_optimizer.zero_grad()
		self.g_optimizer.zero_grad()
		return 0

	def save(self, save_name):
		print("Saving Model ...")
		if not os.path.exists("./out/"):
			os.mkdir("./out/")
		with open("./out/{0}.pkl".format(save_name), "wb") as file:
			pickle.dump(self, file)
		return 0


def load_model(save_name):
	model = None
	if os.path.exists("./out/{0}.pkl".format(save_name)):
		with open("./out/{0}.pkl".format(save_name), "rb") as file:
			model = pickle.load(file)
	else:
		print("No File ... {0}.pkl".format(save_name))
	return model


def to_variable(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)



def tarin_test_model(is_train, is_generate):

	gan_model = None
	mnist_data = MNISTData(args.batch_size)

	if is_train:
		try:
			gan_model = GAN(EPOCHS, args.image_size, args.latent_size, args.hidden_size)
			if torch.cuda.is_available():
				gan_model.cuda()

			# Binary cross entropy loss and optimizer
			criterion = nn.BCELoss()
			last_batch = len(mnist_data.train_loader)
			for epoch in tqdm(range(EPOCHS)):
				# epoch = 0
				print("Traning Epoch ... {0}".format(epoch))
				for batch_id, (data, target) in enumerate(mnist_data.train_loader):
					# batch_id = 0
					# print("Train Batch Number ... {0}".format(batch_id))
					# data, target = list(mnist_data.train_loader)[0]
					images = data.view(args.batch_size, -1)
					images = to_variable(images)
					# Create the labels which are later used as input for the BCE loss
					real_labels = torch.ones(args.batch_size, 1)
					real_labels = to_variable(real_labels)
					fake_labels = torch.zeros(args.batch_size, 1)
					fake_labels = to_variable(fake_labels)
					# ===================== Train Discriminator =====================#
					# Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
					# Second term of the loss is always zero since real_labels == 1
					outputs = gan_model.discriminator(images)
					d_loss_real = criterion(outputs, real_labels)
					real_score = outputs

					# Compute BCELoss using fake images
					# First term of the loss is always zero since fake_labels == 0
					z = torch.randn(args.batch_size, args.latent_size)
					z = to_variable(z)
					fake_images = gan_model.generator(z)
					outputs = gan_model.discriminator(fake_images)
					d_loss_fake = criterion(outputs, fake_labels)
					fake_score = outputs

					# Backprop and optimize
					# If D is trained so well, then don't update
					d_loss = d_loss_real + d_loss_fake
					gan_model.reset_grad()
					d_loss.backward()
					gan_model.d_optimizer.step()

					# ===================== Train Generator =====================#
					# Compute loss with fake images
					z = torch.randn(args.batch_size, args.latent_size)
					z = to_variable(z)
					fake_images = gan_model.generator(z)
					outputs = gan_model.discriminator(fake_images)

					# We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
					# For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
					g_loss = criterion(outputs, real_labels)

					# Backprop and optimize
					# if G is trained so well, then don't update
					gan_model.reset_grad()
					g_loss.backward()
					gan_model.g_optimizer.step()
					# ===================== Save Statistics =====================#
					gan_model.d_losses[epoch] = gan_model.d_losses[epoch]*(batch_id/(batch_id+1.)) + d_loss.data*(1./(batch_id+1.))
					gan_model.g_losses[epoch] = gan_model.g_losses[epoch]*(batch_id/(batch_id+1.)) + g_loss.data*(1./(batch_id+1.))
					gan_model.real_scores[epoch] = gan_model.real_scores[epoch]*(batch_id/(batch_id+1.)) + real_score.mean().data*(1./(batch_id+1.))
					gan_model.fake_scores[epoch] = gan_model.fake_scores[epoch]*(batch_id/(batch_id+1.)) + fake_score.mean().data*(1./(batch_id+1.))

					if ((batch_id > 0) and (batch_id % BATCH_SIZE == 0)) or (batch_id == last_batch-1):
						print("Train Batch {}, d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f} ".
						format(batch_id, gan_model.d_losses[epoch], gan_model.g_losses[epoch], gan_model.real_scores[epoch], gan_model.fake_scores[epoch]))

				if epoch == 0:
					# Save real images
					images = images.view(images.size(0), 1, 28, 28)
					save_image(denorm(images.data), "{}/real_images.png".format(args.sample_path))
					# Save sampled images
					fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
					save_image(denorm(fake_images.data), "{}/fake_images-{}.png".format(args.sample_path, epoch+1))
				else:
					# Save sampled images
					fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
					save_image(denorm(fake_images.data), "{}/fake_images-{}.png".format(args.sample_path, epoch+1))
			gan_model.save("mnist_gan")
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

	parser.add_argument("--image_size", type=int, default=28*28) # 28 for MNIST
	parser.add_argument("--batch_size", type=int, default=100)
	parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--latent_size", type=int, default=64)
	parser.add_argument("--model_path", type=str, default="./out/")  # Model Save
	parser.add_argument("--sample_path", type=str, default="./results/")  # Results

	global args
	args = parser.parse_args()
	# args = ARGS()

	# Create a directory if not exists
	if not os.path.exists(args.sample_path):
		os.makedirs(args.sample_path)

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
