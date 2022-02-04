import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.layers import Linear, Conv1d, Conv2d
from model.FFT_block import PositionalEncoding, ConditionalFFTBlock
from model.utils import create_look_ahead_mask
from model.utils import get_mask_from_lengths

class Decoder(nn.Module):
    def __init__(self, hp):
        super(Decoder, self).__init__()

        # attention
        n_layers        = hp.decoder_attn_n_layers
        n_heads         = hp.decoder_attn_n_heads
        hidden_dim      = hp.hidden_dim
        out_dim         = hp.n_mels
        dropout         = hp.decoder_dropout

        # fft
        ffn_dim         = hp.decoder_ffn_dim        
        ffn_style_dim   = hp.style_dim

        self.register_buffer('pe', PositionalEncoding(hp.hidden_dim).pe)
        self.alpha      = nn.Parameter(torch.ones(1))
        self.dropout    = nn.Dropout(dropout)
        
        self.decoder_layer = nn.ModuleList([
            ConditionalFFTBlock(hidden_dim, n_heads, hidden_dim, ffn_dim, dropout, ffn_style_dim) 
            for i in range(n_layers)])
        self.out_layer  = Linear(hidden_dim, out_dim)
    
    def forward(self, x, style_embedding, x_mask):    
        
        pos_embedding = self.alpha * self.pe[:x.size(1)].unsqueeze(0) # [1, T', D]
        x = self.dropout(x + pos_embedding)

        # Attention & FFT
        dec_attns = list()
        for dec_layers in self.decoder_layer:
            x, dec_attn = dec_layers(x, style_embedding, x_mask) #  [B, T', hidden_dim], [B, T', T']
            dec_attns.append(dec_attn)
        
        # mel linear
        x = self.out_layer(x)

        return x, dec_attns


