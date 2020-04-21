#/usr/bin/pthon
import os
os.chdir("../") # the base directory while running any code
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import pickle
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms


np.random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 200
EPOCHS = 8


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


class ConvLayer(nn.Module):
	def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
		super(ConvLayer, self).__init__()

		self.conv = nn.Conv2d(in_channels=in_channels,
							out_channels=out_channels,
							kernel_size=kernel_size,
							stride=1)

	def forward(self, x):
		return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
	def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
		super(PrimaryCaps, self).__init__()

		self.capsules = nn.ModuleList([
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
						for _ in range(num_capsules)])

	def forward(self, x):
		u = [capsule(x) for capsule in self.capsules]
		u = torch.stack(u, dim=1)
		u = u.view(x.size(0), 32 * 6 * 6, -1)
		return self.squash(u)

	def squash(self, input_tensor):
		squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
		output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
		return output_tensor


class DigitCaps(nn.Module):
	def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
		super(DigitCaps, self).__init__()

		self.in_channels = in_channels
		self.num_routes = num_routes
		self.num_capsules = num_capsules

		self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

	def forward(self, x):
		batch_size = x.size(0)
		x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

		W = torch.cat([self.W] * batch_size, dim=0)
		u_hat = torch.matmul(W, x)

		b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))

		if torch.cuda.is_available():
			b_ij = b_ij.cuda()

		num_iterations = 3
		for iteration in range(num_iterations):
			# print("b_ij", b_ij.shape)
			c_ij = F.softmax(b_ij, dim=1)
			c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

			s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
			v_j = self.squash(s_j)

			if iteration < num_iterations - 1:
				a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
				b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

		return v_j.squeeze(1)

	def squash(self, input_tensor):
		squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
		output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
		return output_tensor


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

		self.reconstraction_layers = nn.Sequential(
			nn.Linear(16 * 10, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, 784),
			nn.Sigmoid()
		)

	def forward(self, x, data):
		classes = torch.sqrt((x ** 2).sum(2))
		# print("classes", classes.shape)
		classes = F.softmax(classes, dim=1)
		_, max_length_indices = classes.max(dim=1)
		masked = Variable(torch.sparse.torch.eye(10))
		if torch.cuda.is_available():
			masked = masked.cuda()
		masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
		reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
		reconstructions = reconstructions.view(-1, 1, 28, 28)
		return reconstructions, masked


class CapsNet(nn.Module):
	def __init__(self):
		super(CapsNet, self).__init__()
		self.conv_layer = ConvLayer()
		self.primary_capsules = PrimaryCaps()
		self.digit_capsules = DigitCaps()
		self.decoder = Decoder()
		self.mse_loss = nn.MSELoss()
		self.loss_list = []
		self.train_accuracy = 0
		self.test_accuracy = 0

	def forward(self, data):
		output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
		reconstructions, masked = self.decoder(output, data)
		return output, reconstructions, masked

	def loss(self, data, x, target, reconstructions):
		return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

	def margin_loss(self, x, labels, size_average=True):
		batch_size = x.size(0)
		v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
		left = F.relu(0.9 - v_c).view(batch_size, -1)
		right = F.relu(v_c - 0.1).view(batch_size, -1)
		loss = labels * left + 0.5 * (1.0 - labels) * right
		loss = loss.sum(dim=1).mean()
		return loss

	def reconstruction_loss(self, data, reconstructions):
		loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
		return loss * 0.0005

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


def tarin_test_model(is_train, is_test):

	capsule_net = None
	mnist_data = MNISTData(BATCH_SIZE)
	if is_train:
		try:
			capsule_net = CapsNet()
			if torch.cuda.is_available():
				capsule_net = capsule_net.cuda()
			optimizer = Adam(capsule_net.parameters())

			for epoch in tqdm(range(EPOCHS)):
				# epoch = 0
				print("Traning Epoch ... {0}".format(epoch))
				capsule_net.train()
				epoch_loss = 0
				batch_loss = 0
				train_accuracy = 0
				last_batch = len(mnist_data.train_loader)
				for batch_id, (data, target) in enumerate(mnist_data.train_loader):
					target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
					data, target = Variable(data), Variable(target)

					if torch.cuda.is_available():
						data, target = data.cuda(), target.cuda()

					optimizer.zero_grad()
					output, reconstructions, masked = capsule_net(data)
					loss = capsule_net.loss(data, output, target, reconstructions)
					loss.backward()
					optimizer.step()
					batch_loss = loss.data
					capsule_net.loss_list.append(batch_loss) # to plot
					if ((batch_id > 0) and (batch_id % BATCH_SIZE == 0)) or (batch_id == last_batch-1):
						currect_sum = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
						accuracy = (currect_sum/float(BATCH_SIZE))*100
						train_accuracy = accuracy
						epoch_loss = batch_loss
						print("Train Batch {0} Accuracy ... {1}".format(batch_id, accuracy))
				capsule_net.loss_list.append(epoch_loss) # to plot
				print("Epoch {0} Training Loss ... {1}".format(epoch, epoch_loss))
			capsule_net.train_accuracy = train_accuracy
			print("Training Accuracy ...{0}".format(capsule_net.train_accuracy))
			capsule_net.save("mnist_capsule")
		except Exception as ex:
			print("Training Failed ...")
			print("Error:: {0}".format(ex))


	if is_test:
		try:
			capsule_net = load_model("mnist_capsule")
			capsule_net.eval()
			print("Testing Model ...")
			test_accuracy = 0
			last_batch = len(mnist_data.test_loader)
			for batch_id, (data, target) in enumerate(mnist_data.test_loader):
				target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
				data, target = Variable(data), Variable(target)

				if torch.cuda.is_available():
					data, target = data.cuda(), target.cuda()

				output, reconstructions, masked = capsule_net(data)
				loss = capsule_net.loss(data, output, target, reconstructions)
				if ((batch_id > 0) and (batch_id % BATCH_SIZE == 0)) or (batch_id == last_batch-1):
					currect_sum = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
					accuracy = (currect_sum/float(BATCH_SIZE))*100
					test_accuracy = accuracy
					print("Test Batch {0} Accuracy ... {1}".format(batch_id, accuracy))
			capsule_net.test_accuracy = test_accuracy
			print("Training Accuracy ...{0}".format(capsule_net.test_accuracy))
		except Exception as ex:
			print("Testing Failed ...")
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
