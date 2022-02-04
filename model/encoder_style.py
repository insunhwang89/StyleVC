import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.utils import get_mask_from_lengths
from model.layers import Linear, Conv1d


class Mish(nn.Module):
    def forward(self, x):

        return x * torch.tanh(F.softplus(x))

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1DBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            Conv1d(in_channels, out_channels, kernel_size=kernel_size),
            Mish())
        self.dropout = dropout

    def forward(self, x):

        x = x.contiguous().transpose(1, 2)

        x = self.conv_layer(x)
        x = F.dropout(x, self.dropout, self.training)

        return x.contiguous().transpose(1, 2)

class StyleEncoder(nn.Module):
    def __init__(self, hp): 
        super(StyleEncoder, self).__init__()

        n_mel_channels = hp.n_mels
        hidden_dim = hp.style_hidden_dim
        d_melencoder = hp.style_dim 
        n_spectral_layer = 2
        n_temporal_layer = 2
        n_slf_attn_layer = 1
        n_slf_attn_head = 2        
        kernel_size = 5
        dropout = 0.1        

        self.fc_1 = Linear(n_mel_channels, hidden_dim)

        self.spectral_layer = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            Mish(),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim),
            Mish(),
            nn.Dropout(dropout)
        )

        self.temporal_stack = nn.ModuleList([
            nn.Sequential(
                Conv1DBlock(hidden_dim, 2 * hidden_dim, kernel_size, dropout=dropout),
                nn.GLU(),)
            for _ in range(n_temporal_layer)])


        self.slf_attn_stack = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_slf_attn_head, dropout=dropout)
            for _ in range(n_slf_attn_layer)])

        self.fc_2 = Linear(hidden_dim, d_melencoder)

    def forward(self, mel, mel_len, max_len):
        
        if mel_len is not None:
            max_len = mel.shape[1]
            slf_attn_mask = get_mask_from_lengths(mel_len, seq_len=max_len) # [B, T], [F, F, ..., T]
        else:
            slf_attn_mask = None

        x = self.fc_1(mel) # [B, T, D(hidden)]

        # spectral processing
        x = self.spectral_layer(x)

        # temporal processing
        for _, layer in enumerate(self.temporal_stack):
            residual = x
            x = layer(x)
            x = residual + x

        # multi-head self-attention
        for _, layer in enumerate(self.slf_attn_stack):
            residual = x
            x = x.transpose(0,1) # [T, B, D(hidden)] for attention input
            x, _ = layer(
                x, x, x, attn_mask=None, key_padding_mask=slf_attn_mask)
            x = residual + x.transpose(0,1) 

        # final Layer
        x = self.fc_2(x) # [B, T, D(style)]

        # temporal average pooling
        x = torch.mean(x, dim=1)  # [B, 1, D(style)] -> [B, D(style)]

        return x 


