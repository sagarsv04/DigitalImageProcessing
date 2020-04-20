#/usr/bin/pthon
import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(s, dim=-1):
	squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
	return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)


class GenCapsules(nn.Module):

	def __init__(self, in_caps, num_caps, in_dim, dim_caps, dim_real):
		super(GenCapsules, self).__init__()

		self.dim_real = dim_real
		self.W1 = nn.Parameter(torch.randn(1, 6*6*32, num_caps, in_dim, dim_caps)*(3/(in_dim + dim_caps + 6*6*32))**0.5)
		self.W0 = nn.Parameter(torch.randn(1, num_caps, 1, dim_caps, dim_real)*(3/(dim_caps + num_caps + dim_real))**0.5)

		self.dconv1 = nn.ConvTranspose2d(256, 1, 9, 1, 0)
		self.dconv0 = nn.ConvTranspose2d(256, 256, 10, 2, 0)
		torch.nn.init.xavier_normal_(self.dconv1.weight)
		torch.nn.init.xavier_normal_(self.dconv0.weight)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

		self.batchnorm = nn.BatchNorm2d(256)
		self.batchnorm0 = nn.BatchNorm2d(256)

	def forward(self, x, epoch):

		batch_size = x.size()[0]
		x = x.unsqueeze(1).unsqueeze(-1)
		c_caps = torch.matmul(self.W0, x)
		c_caps = c_caps.transpose(1,2).transpose(-2,-1)
		c_caps = squash(c_caps).transpose(-2,-1)
		p_caps = torch.matmul(self.W1, c_caps)
		c_caps = c_caps.squeeze(1).squeeze(-1)
		p_caps = p_caps.squeeze(-1)
		p_caps = p_caps.sum(dim=2)
		p_caps = squash(p_caps)
		out = p_caps.view(p_caps.size(0), 32, 6, 6, 8)
		out = out.view(p_caps.size(0), 256, 6, 6)

		# apply deconvs
		out = self.dconv0(out)
		out = self.batchnorm(out)
		out = self.relu(out)
		out = self.dconv1(out)
		out = self.tanh(out)

		return out, p_caps, c_caps


class Generator(nn.Module):
	def __init__(self, in_caps, num_caps, in_dim, dim_caps, dim_real):
		super(Generator, self).__init__()
		self.in_caps = in_caps
		self.num_caps = num_caps
		self.in_dim = in_dim
		self.dim_caps = dim_caps
		self.dim_real = dim_real
		self.gen = GenCapsules(in_caps, num_caps, in_dim, dim_caps, dim_real)

	def forward (self, x, epoch):
		out, p_caps, c_caps = self.gen(x, epoch)
		return out, p_caps, c_caps
