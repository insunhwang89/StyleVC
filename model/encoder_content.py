
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.layers import Linear, Conv1d
from model.FFT_block import PositionalEncoding, ConditionalFFTBlock
from model.utils import create_look_ahead_mask


class Prenet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(Prenet, self).__init__()

        self.layer = nn.Sequential(
            Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, out_dim))

    def forward(self, x):
        
        return self.layer(x) # [B, T, D]        
        

class ContentEncoder(nn.Module):
    def __init__(self, hp):
        super(ContentEncoder, self).__init__()

        in_dim              = hp.wav2vec_dim
        hidden_dim          = hp.hidden_dim
        out_dim             = hp.model_dim 
        dropout             = hp.encoder_dropout
        n_layers            = hp.encoder_attn_n_layers 
        n_heads             = hp.encoder_attn_n_heads
        ffn_dim             = hp.encoder_ffn_dim
        ffn_style_dim       = hp.style_dim

        # prenet        
        self.prenet         = Prenet(in_dim, hidden_dim, out_dim, dropout)
        self.register_buffer('pe', PositionalEncoding(hidden_dim).pe)
        self.alpha          = nn.Parameter(torch.ones(1))
        self.dropout        = nn.Dropout(dropout)

        # encoder
        self.encoder_layer  = nn.ModuleList([
            ConditionalFFTBlock(hidden_dim, n_heads, hidden_dim, ffn_dim, dropout, ffn_style_dim)
            for i in range(n_layers)])

    def forward(self, x, x_mask):       

        # prenet & positional encoding
        prenet_output = self.prenet(x) # [B, T, D(model)]
        pos_embedding = self.alpha * self.pe[:x.size(1)].unsqueeze(0) # [1, T, D]
        x = self.dropout(prenet_output + pos_embedding)

        # attention & FFT
        for enc_layers in self.encoder_layer:
            x, enc_attn = enc_layers(
                x, None, x_mask) # [B, T, D], [B, T, T]

        return x



