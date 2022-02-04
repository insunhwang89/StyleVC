import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()

        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        
        return self.linear_layer(x)


class Conv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
            dilation=1, padding=None, groups=1, bias=True, w_init_gain='linear'):
        super(Conv1d, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias,
                                    groups=groups)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

        self.weight = self.conv.weight

    def forward(self, signal):

        conv_signal = self.conv(signal)

        return conv_signal

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', padding_mode='zeros'):
        super(Conv2d, self).__init__()

        if padding is None:
            if type(kernel_size) is tuple:
                padding = []
                for k in kernel_size:
                    assert(k % 2 == 1)
                    p = int(dilation * (k - 1) / 2)
                    padding.append(p)
                padding = tuple(padding)
            else:
                assert(kernel_size % 2 == 1)
                padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=kernel_size, 
                                    stride=stride, 
                                    padding=padding,
                                    bias=bias,
                                    padding_mode=padding_mode)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

        self.weight = self.conv.weight
        
    def forward(self, signal):

        conv_signal = self.conv(signal)

        return conv_signal
