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

    def get_usage_vector(self, usage, free_gates, read_weights, write_weights):
        usage = usage + (1 - usage) * (1 - T.prod(1 - write_weights, 1))
        ψ = T.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
        return usage * ψ


    def allocate(self, usage, write_gate):
        # ensure values are not too small prior to cumprod.
        usage = δ + (1 - δ) * usage
        batch_size = usage.size(0)
        # free list
        sorted_usage, φ = T.topk(usage, self.num_mem, dim=1, largest=False, sorted=True)

        # cumprod with exclusive=True
        # https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
        v = var(sorted_usage.data.new(batch_size, 1).fill_(1))
        cat_sorted_usage = T.cat((v, sorted_usage), 1)
        prod_sorted_usage = T.cumprod(cat_sorted_usage, 1)[:, :-1]

        sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()

        # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
        _, φ_rev = T.topk(φ, k=self.num_mem, dim=1, largest=False, sorted=True)
        allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())

        return allocation_weights.unsqueeze(1), usage

    def write_weighting(self, memory, write_content_weights, allocation_weights, write_gate, allocation_gate):
        ag = allocation_gate.unsqueeze(-1)
        wg = write_gate.unsqueeze(-1)
        return wg * (ag * allocation_weights + (1 - ag) * write_content_weights)

    def get_link_matrix(self, link_matrix, write_weights, precedence):
        precedence = precedence.unsqueeze(2)
        write_weights_i = write_weights.unsqueeze(3)
        write_weights_j = write_weights.unsqueeze(2)

        prev_scale = 1 - write_weights_i - write_weights_j
        new_link_matrix = write_weights_i * precedence

        link_matrix = prev_scale * link_matrix + new_link_matrix
        # remove diagonal elements since I has 0's on the diagonal and 1 on off diagonal
        return self.I.expand_as(link_matrix) * link_matrix

    def forward(self, ξ, hidden):
        N = self.num_mem
        W = self.mem_len
        r = self.read_heads
        b = ξ.size()[0]

        ξ = self.interface_weights(ξ)
        # r read keys (b * W * r)
        read_keys = F.tanh(ξ[:, :r * w].contiguous().view(b, r, W))
        # r read strengths (b * r)
        read_strengths = F.softplus(ξ[:, r * W:r * W + r].contiguous().view(b, r))
        # write key (b * W * 1)
        write_key = F.tanh(ξ[:, r * W + r:r * W + r + W].contiguous().view(b, 1, W))
        # write strength (b * 1)
        write_strength = F.softplus(ξ[:, r * W + r + W].contiguous().view(b, 1))
        # erase vector (b * W)
        erase_vector = F.sigmoid(ξ[:, r * W + r + W + 1: r * W + r + 2 * W + 1].contiguous().view(b, 1, W))
        # write vector (b * W)
        write_vector = F.tanh(ξ[:, r * W + r + 2 * W + 1: r * W + r + 3 * W + 1].contiguous().view(b, 1, W))
        # r free gates (b * r)
        free_gates = F.sigmoid(ξ[:, r * W + r + 3 * W + 1: r * W + 2 * r + 3 * W + 1].contiguous().view(b, r))
        # allocation gate (b * 1)
        allocation_gate = F.sigmoid(ξ[:, r * W + 2 * r + 3 * W + 1].contiguous().unsqueeze(1).view(b, 1))
        # write gate (b * 1)
        write_gate = F.sigmoid(ξ[:, r * W + 2 * r + 3 * W + 2].contiguous()).unsqueeze(1).view(b, 1)
        # read modes (b * 3 * r)
        read_modes = σ(ξ[:, r * W + 2 * r + 3 * W + 2: r * W + 5 * r + 3 * W + 2].contiguous().view(b, r, 3), 1)

        hidden = self.write(write_key, write_vector, erase_vector, free_gates,
                        read_strengths, write_strength, write_gate, allocation_gate, hidden)

        return self.read(read_keys, read_strengths, read_modes, hidden)


