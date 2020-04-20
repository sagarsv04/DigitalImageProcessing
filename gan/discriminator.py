#/usr/bin/pthon
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from torch.nn.init import kaiming_normal, calculate_gain


def squash(s, dim=-1):
	squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
	return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)


class PrimaryCapsules(nn.Module):
	def __init__(self, in_channels, out_channels, dim_caps,
	kernel_size=9, stride=2, padding=0):
		super(PrimaryCapsules, self).__init__()
		self.dim_caps = dim_caps
		self._caps_channel = int(out_channels / dim_caps)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
		torch.nn.init.xavier_normal_(self.conv.weight)

	def forward(self, x):
		out = self.conv(x)
		out = out.view(out.size(0), self._caps_channel, out.size(2), out.size(3), self.dim_caps)
		out = out.view(out.size(0), -1, self.dim_caps)
		return squash(out)


class RoutingCapsules(nn.Module):
	def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing):
		super(RoutingCapsules, self).__init__()
		self.in_dim = in_dim
		self.in_caps = in_caps
		self.num_caps = num_caps
		self.dim_caps = dim_caps
		self.num_routing = num_routing
		self.W = nn.Parameter(torch.randn(1, num_caps, in_caps, dim_caps, in_dim )*(3/(in_dim  + dim_caps + num_caps))**0.5)


	def __repr__(self):
		tab = '  '
		line = '\n'
		next = ' -> '
		res = self.__class__.__name__ + '('
		res = res + line + tab + '(' + str(0) + '): ' + 'CapsuleLinear('
		res = res + str(self.in_dim) + ', ' + str(self.dim_caps) + ')'
		res = res + line + tab + '(' + str(1) + '): ' + 'Routing('
		res = res + 'num_routing=' + str(self.num_routing) + ')'
		res = res + line + ')'
		return res

	def forward(self, x):
		batch_size = x.size(0)
		x = x.unsqueeze(1).unsqueeze(4)
		u_hat = torch.matmul(self.W, x)
		u_hat = u_hat.squeeze(-1)
		temp_u_hat = u_hat
		b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).cuda()

		for route_iter in range(self.num_routing-1):
			c = F.softmax(b, dim=1)
			s = (c * temp_u_hat).sum(dim=2)
			v = squash(s)
			uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
			b += uv
			c = F.softmax(b, dim=1)
			s = (c * u_hat).sum(dim=2)

			return s


class RealOrFake(nn.Module):
	def __init__(self, num_caps, dim_caps, dim_real, num_routing):
		super(RealOrFake, self).__init__()
		self.num_caps = num_caps
		self.dim_caps = dim_caps
		self.dim_real = dim_real
		self.num_routing = num_routing

		self.W = nn.Parameter(torch.randn(1, 1, self.num_caps, self.dim_real, self.dim_caps)*(2/(dim_real + dim_caps))**0.5)

	def forward(self, x):

		batch_size = x.size(0)
		x = x.unsqueeze(1).unsqueeze(4)
		u_hat = torch.matmul(self.W, x)
		u_hat = u_hat.squeeze(-1)
		temp_u_hat = u_hat
		b = torch.zeros(batch_size, 1, self.num_caps, 1).cuda()

		for route_iter in range(self.num_routing-1):
			c = F.softmax(b, dim=1)
			s = (c * temp_u_hat).sum(dim=2)
			v = squash(s)
			uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
			b += uv
			c = F.softmax(b, dim=1)
			s = (c * u_hat).sum(dim=2)
			v = squash(s)

			return v


class Discriminator(nn.Module):
	def __init__(self, img_shape, channels, primary_dim, num_classes, out_dim, dim_real, num_routing, kernel_size=9):
		super(Discriminator, self).__init__()
		self.img_shape = img_shape
		self.num_classes = num_classes
		self.conv1 = nn.Conv2d(img_shape[0], channels, kernel_size, stride=1, bias=True)
		torch.nn.init.xavier_normal_(self.conv1.weight)
		self.relu = nn.ReLU(inplace=True)
		self.dim_real = dim_real
		self.primary = PrimaryCapsules(channels, channels, primary_dim, kernel_size)
		self.batchnorm = nn.BatchNorm2d(channels)

		primary_caps = int(channels / primary_dim * ( img_shape[1] - 2*(kernel_size-1) ) * ( img_shape[2] - 2*(kernel_size-1) ) / 4)
		self.digits = Dis.RoutingCapsules(primary_dim, primary_caps, num_classes, out_dim, num_routing)
		self.real = Dis.RealOrFake(num_classes, out_dim, self.dim_real - 10, num_routing)
		self.convR = nn.Conv1d(52, 1, 1)
	def forward(self, x):
		out = self.conv1(x)
		out = self.batchnorm(out)
		out = self.relu(out)
		p_caps = self.primary(out)
		c_caps = self.digits(p_caps)
		c_caps = squash(c_caps)
		norm_c = torch.norm(c_caps, dim=-1)
		c_caps = (norm_c.unsqueeze(-1) + 1)*c_caps
		r_caps = self.real(c_caps) # -> (batch_size, 1, dim_real)

		preds = self.convR(r_caps.transpose(1,2)).squeeze(-1)
		preds = torch.sigmoid(preds)

		return preds, p_caps, c_caps, r_caps, norm_c
