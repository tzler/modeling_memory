import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

δ = 1e-6

def θ(a, b, dimA=2, dimB=2, normBy=2):
	"""Batchwise Cosine distance
	Cosine distance
	Arguments:
	  a {Tensor} -- A 3D Tensor (b * N * W)
	  b {Tensor} -- A 3D Tensor (b * r * W)
	Keyword Arguments:
	  dimA {number} -- exponent value of the norm for `a` (default: {2})
	  dimB {number} -- exponent value of the norm for `b` (default: {1})
	Returns:
	  Tensor -- Batchwise cosine distance (b * r * N)
	"""
	a_norm = T.norm(a, normBy, dimA, keepdim=True).expand_as(a) + δ
	b_norm = T.norm(b, normBy, dimB, keepdim=True).expand_as(b) + δ

	x = T.bmm(a, b.transpose(1, 2)).transpose(1, 2) / (
	  T.bmm(a_norm, b_norm.transpose(1, 2)).transpose(1, 2) + δ)

	return x


def σ(input, axis=1):
	"""Softmax on an axis
	Softmax on an axis
	Arguments:
	  input {Tensor} -- input Tensor
	Keyword Arguments:
	  axis {number} -- axis on which to take softmax on (default: {1})
	Returns:
	  Tensor -- Softmax output Tensor
	"""
	input_size = input.size()

	trans_input = input.transpose(axis, len(input_size) - 1)
	trans_size = trans_input.size()

	input_2d = trans_input.contiguous().view(-1, trans_size[-1])
	soft_max_2d = F.softmax(input_2d)
	soft_max_nd = soft_max_2d.view(*trans_size)
	return soft_max_nd.transpose(axis, len(input_size) - 1)


class Memory(nn.Module):
	def __init__(self, input_size, num_mem=512, mem_len=32, read_heads=4):
		super(Memory, self).__init__()

		self.input_size = input_size
		self.num_mem = num_mem
		self.mem_len = mem_len
		self.read_heads = read_heads

		N = self.num_mem
		W = self.mem_len
		r = self.read_heads

		self.interface_size = (W * r) + (3 * W) + (5 * r) + 3
		self.interface_weights = nn.Linear(self.input_size, self.interface_size)
        self.I = 1 - T.eye(m).unsqueeze(0)  # (1 * n * n)

    def reset(self, batch_size=1, hidden=None, erase=False):
        N = self.num_mem
        W = self.mem_len
        r = self.read_heads
        b = batch_size

	    if hidden is None:
	        return {
	          'memory': T.zeros(b, N, W).fill_(0),
	          'link_matrix': T.zeros(b, 1, N, N),
	          'precedence': T.zeros(b, 1, N),
	          'read_weights': T.zeros(b, r, N).fill_(0),
	          'write_weights': T.zeros(b, 1, N).fill_(0),
	          'usage_vector': T.zeros(b, N)
	          }
	    else:
			hidden['memory'] = hidden['memory'].clone()
			hidden['link_matrix'] = hidden['link_matrix'].clone()
			hidden['precedence'] = hidden['precedence'].clone()
			hidden['read_weights'] = hidden['read_weights'].clone()
			hidden['write_weights'] = hidden['write_weights'].clone()
			hidden['usage_vector'] = hidden['usage_vector'].clone()

			if erase:
				hidden['memory'].data.fill_(0)
				hidden['link_matrix'].data.zero_()
				hidden['precedence'].data.zero_()
				hidden['read_weights'].data.fill_(0)
				hidden['write_weights'].data.fill_(0)
				hidden['usage_vector'].data.zero_()

        return hidden


