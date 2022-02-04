import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Conv1d

class Postnet(nn.Module):
    def __init__(self, hp):
        super(Postnet, self).__init__()
        
        self.n_mel_channels = hp.postnet_in_dim
        self.postnet_embedding_dim = hp.postnet_hidden_dim
        self.postnet_kernel_size = hp.postnet_kernel_size
        self.postnet_n_convolutions = hp.postnet_kernel_size

        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                Conv1d(self.n_mel_channels,
                         self.postnet_embedding_dim,
                         kernel_size=self.postnet_kernel_size,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         w_init_gain='tanh'),
                nn.BatchNorm1d(self.postnet_embedding_dim))
        )

        for i in range(1, self.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    Conv1d(self.postnet_embedding_dim,
                             self.postnet_embedding_dim,
                             kernel_size=self.postnet_kernel_size,
                             padding=int((self.postnet_kernel_size - 1) / 2),
                             w_init_gain='tanh'),
                    nn.BatchNorm1d(self.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                Conv1d(self.postnet_embedding_dim,
                         self.n_mel_channels,
                         kernel_size=self.postnet_kernel_size,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         w_init_gain='linear'),
                nn.BatchNorm1d(self.n_mel_channels))
        )

    def forward(self, x):
        
        # x: [B, T, D]
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)

        return x