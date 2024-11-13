import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import math
import torch.nn as nn
#from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import weight_norm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.att = None
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal: bool = False):
        src2 = self.self_attn(src, src, src)[0]
        self.att = self.self_attn(src, src, src)[1]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_is_causal: bool = False, memory_is_causal: bool = False):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        In fact, this is a cropping module, cropping the extra rightmost padding (default is padding on both sides)

        tensor.contiguous() will return the same tensor with contiguous memory
        Some tensors do not occupy a whole block of memory, but are composed of different blocks of data
        The tensor's view() operation relies on the memory being a whole block, in which case it is only necessary
        to execute the contiguous() function, which turns the tensor into a continuous distribution in memory
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalCnn(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalCnn, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)
        self.leakyrelu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv, self.chomp, self.leakyrelu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, out_channel, seq_len)
        :return:size of (Batch, out_channel, seq_len)
        """
        out = self.net(x)
        return out


class Tcn_Local(nn.Module):

    def __init__(self, num_outputs, kernel_size=3, dropout=0.2):  # k>=3
        """
        TCN, the current paper gives a TCN structure that supports well the case of one number per moment, i.e., the sequence structure.
        For a one-dimensional structure where each moment is a vector, it is barely possible to split the vector into several input channels at that moment.
        For the case where each moment is a matrix or a higher dimensional image, it is not so good.

        :param num_outputs: int, the number of output channels
        :param input_length: int, the length of the sliding window input sequence
        :param kernel_size: int, the size of the convolution kernel
        :param dropout: float, drop_out ratio
        """
        super(Tcn_Local, self).__init__()
        layers = []
        num_levels = 3
        out_channels = num_outputs
        for i in range(num_levels):
            layers += [TemporalCnn(out_channels, out_channels, kernel_size, stride=1, dilation=1,
                                   padding=(kernel_size - 1),
                                   dropout=dropout)]  # Adding padding to the convolved tensor to achieve causal convolution by slicing the tensor

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        The structure of input x is different from RNN, which generally has size (Batch, seq_len, channels) or (seq_len, Batch, channels).
        Here the seq_len is put after channels, and the data of all time steps are put together and used as the input size of Conv1d to realize the operation of convolution across time steps.
        Very clever design.

        :param x: size of (Batch, out_channel, seq_len)
        :return: size of (Batch, out_channel, seq_len)
        """
        return self.network(x)


class Tcn_Global(nn.Module):

    def __init__(self, num_inputs, num_outputs, kernel_size=3, dropout=0.2):  # k>=d
        """
        TCN, the current paper gives a TCN structure that supports well the case of one number per moment, i.e., the sequence structure.
        For a one-dimensional structure where each moment is a vector, it is barely possible to split the vector into several input channels at that moment.
        For the case where each moment is a matrix or a higher dimensional image, it is not so good.

        :param num_inputs: int, input length
        :param num_outputs: int, the number of output channels
        :param input_length: int, the length of the sliding window input sequence
        :param kernel_size: int, convolutional kernel size
        :param dropout: float, drop_out ratio
        """
        super(Tcn_Global, self).__init__()
        layers = []
        num_levels = math.ceil(math.log2((num_inputs - 1) * (2 - 1) / (kernel_size - 1) + 1))
        out_channels = num_outputs
        for i in range(num_levels):
            dilation_size = 2 ** i  # Expansion coefficient: 1，2，4，8……
            layers += [TemporalCnn(out_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size - 1) * dilation_size,
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        The structure of input x is different from RNN, which generally has size (Batch, seq_len, channels) or (seq_len, Batch, channels).
        Here the seq_len is put after channels, and the data of all time steps are put together and used as the input size of Conv1d to realize the operation of convolution across time steps.
        Very clever design.

        :param x: size of (Batch, out_channel, seq_len)
        :return: size of (Batch, out_channel, seq_len)
        """
        return self.network(x)

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.linalg.cholesky(a)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
