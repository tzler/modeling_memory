import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .util import *
from .memory import *

class DNC(nn.Module):

	def __init__(
	  self,
	  input_size,
	  hidden_size,
	  rnn_type='lstm',
	  num_ctrl_layers=1,
	  batch_first=True,
	  bias=True,
	  dropout=0,
	  nonlinearity='tanh',
	  num_mem=5,
	  mem_size=10,
	  read_heads=2
	):
    super(DNC, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.rnn_type = rnn_type
    self.num_ctrl_layers = num_ctrl_layers
    self.bias = bias
    self.batch_first = batch_first
    self.dropout = dropout
    self.num_mem = num_mem
    self.read_heads = read_heads
    self.mem_size = mem_size
    self.nonlinearity = nonlinearity

    self.W = self.mem_size
    self.r = self.read_heads

    self.read_vectors_size = self.r * self.W
    self.interface_size = self.read_vectors_size + (3 * self.W) + (5 * self.r) + 3
    self.output_size = self.hidden_size

    self.nn_input_size = self.input_size + self.read_vectors_size
    self.nn_output_size = self.output_size + self.read_vectors_size

	if self.rnn_type.lower() == 'rnn':
		self.ctrl = nn.RNN(self.nn_input_size, self.output_size,
	                        bias=self.bias, nonlinearity=self.nonlinearity, batch_first=self.batch_first, dropout=self.dropout, num_layers=self.num_ctrl_layers)
	elif self.rnn_type.lower() == 'gru':
		self.ctrl = nn.GRU(self.nn_input_size, self.output_size, 
			                bias=self.bias, batch_first=self.batch_first, dropout=self.dropout, num_layers=self.num_ctrl_layers)
	elif self.rnn_type.lower() == 'lstm':
		self.ctrl = nn.LSTM(self.nn_input_size self.nn_output_size,
	                        bias=self.bias, batch_first=self.batch_first, dropout=self.dropout, num_layers=self.num_ctrl_layers)

    self.memory = Memory(
              input_size=self.output_size,
              mem_size=self.num_mem,
              cell_size=self.W,
              read_heads=self.r)

    # final output layer
    self.output = nn.Linear(self.nn_output_size, self.input_size)



