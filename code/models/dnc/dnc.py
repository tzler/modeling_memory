import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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
                  num_mem=self.num_mem,
                  mem_len=self.W,
                  read_heads=self.r)

        # final output layer
        self.output = nn.Linear(self.nn_output_size, self.input_size)

    # might want to revisit this to determine when we want to reset memory and hidden states
    def _init_hidden(self, hx, batch_size, reset_experience):
        # create empty hidden states if not provided
        if hx is None:
            hx = (None, None, None)
        (chx, mhx, last_read) = hx

        # Last read vectors
        if last_read is None:
          last_read = T.zeros(batch_size, self.W * self.r)

        # memory states
        if mhx is None:
            mhx = self.memory.reset(batch_size, erase=reset_experience)
        else:
            mhx = self.memory.reset(batch_size, mhx, erase=reset_experience)

        return chx, mhx, last_read

    def _layer_forward(self, input, hx=(None, None)):
        (chx, mhx) = hx

        # pass through the controller layer
        input, chx = self.ctrl(input.unsqueeze(1), chx)
        input = input.squeeze(1)

        # the interface vector
        ξ = input
        output = input
        read_vecs, mhx = self.memory(ξ, mhx)
        # the read vectors
        read_vectors = read_vecs.view(-1, self.W * self.r)

        return output, (chx, mhx, read_vectors)

    def forward(self, input, hx=(None, None, None), reset_experience=False):
        max_length = input.size(1) if self.batch_first else input.size(0)
        lengths = [input.size(1)] * max_length if self.batch_first else [input.size(0)] * max_length

        batch_size = input.size(0) if self.batch_first else input.size(1)

        if not self.batch_first:
            input = input.transpose(0, 1)

        controller_hidden, mem_hidden, last_read = self._init_hidden(hx, batch_size, reset_experience)
        # concat input with last read (or padding) vectors
        inputs = [T.cat([input[:, x, :], last_read], 1) for x in range(max_length)]

        outs = [None] * max_length
        read_vectors = None

        # pass through time
        for time in range(max_length):
            # get hidden states
            chx = controller_hidden
            m = mem_hidden
            # pass through controller
            outs[time], (chx, m, read_vectors) = \
              self._layer_forward(inputs[time], (chx, m))

            # get new memory and controller hidden states
            mem_hidden = m

            controller_hidden = chx

            if read_vectors is not None:
                # the controller output + read vectors go into next layer
                outs[time] = T.cat([outs[time], read_vectors], 1)
            else:
                outs[time] = T.cat([outs[time], last_read], 1)

            inputs[time] = outs[time]


        # pass through final output layer
        inputs = [self.output(i) for i in inputs]
        outputs = T.stack(inputs, 1 if self.batch_first else 0)

        return outputs, (controller_hidden, mem_hidden, read_vectors)

