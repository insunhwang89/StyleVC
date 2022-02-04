
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.layers import Linear, Conv1d
from model.FFT_block import ConditionalFFTBlock
        
class PitchPredictor(nn.Module):
    def __init__(self, hp):
        super(PitchPredictor, self).__init__()
        
        # attention                
        in_dim              = hp.hidden_dim
        hidden_dim          = hp.hidden_dim
        n_layers            = hp.pitch_predictor_attn_n_layers 
        n_heads             = hp.pitch_predictor_attn_n_heads
        dropout             = hp.pitch_predictor_dropout
        ffn_dim             = hp.pitch_predictor_ffn_dim
        ffn_style_dim       = hp.style_dim        

        # encoder
        self.encoder_layer  = nn.ModuleList([
            ConditionalFFTBlock(hidden_dim, n_heads, hidden_dim, ffn_dim, dropout, ffn_style_dim)
            for i in range(n_layers)])
        
    def forward(self, x, style_embedding, x_mask):       

        # mel: [B, T, 80]
        # style embedding: [B, D(speaker)]

        # attention & FFT
        for enc_layers in self.encoder_layer:
            x, enc_attn = enc_layers(
                x, style_embedding, x_mask) # [B, T, D], [B, T, T]

        return x
