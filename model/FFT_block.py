import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Linear, Conv1d, Conv2d

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pe', self._get_pe_matrix(d_model, max_len))

    def forward(self, x):

        return x + self.pe[:x.size(0)].unsqueeze(1)

    def _get_pe_matrix(self, d_model, max_len):

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        return pe

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(FeedForwardNetwork, self).__init__()

        self.layer = nn.Sequential(
            Conv1d(in_dim, hidden_dim, 9, padding=(9 - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Conv1d(hidden_dim, out_dim, 1, padding=0))
            
    def forward(self, x):
         
        return self.layer(x)

class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(StyleAdaptiveLayerNorm, self).__init__()

        self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)

        # AffineLinear
        self.style = nn.Linear(style_dim, in_channel * 2) # AffineLinear(style_dim, in_channel * 2)
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style_code):

        # input: [B, T, D]
        # style code : [B, D]

        # style
        style = self.style(style_code).unsqueeze(1) # [B, 1, D * 2]
        gamma, beta = style.chunk(2, dim=-1) # [B, 1, D], [B, 1, D]
        
        out = self.norm(input) # [B, T, D]
        out = gamma * out + beta

        return out

class ConditionalFFTBlock(nn.Module):
    def __init__(self, in_dim, n_heads, out_dim, ffn_dim, dropout, style_dim):
        super(ConditionalFFTBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(in_dim, n_heads, dropout=dropout)
        
        self.ffn = FeedForwardNetwork(
            in_dim=in_dim, hidden_dim=ffn_dim, out_dim=out_dim, dropout=dropout)

        self.style_norm = True if style_dim != 0 else False
        if self.style_norm == True:
            self.saln_1 = StyleAdaptiveLayerNorm(out_dim, style_dim)
            self.saln_2 = StyleAdaptiveLayerNorm(out_dim, style_dim)
        else:            
            self.norm1 = nn.LayerNorm(normalized_shape=in_dim, eps=1e-12)
            self.norm2 = nn.LayerNorm(normalized_shape=out_dim, eps=1e-12)     

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, style_embedding, mask=None, self_attn_mask=None):
        
        # x: [B, T, D] 
        residual = x         
        x = x.transpose(0,1) # [T, B, D] for attention input
        att, attn_matrix = self.self_attn(x, x, x, attn_mask=self_attn_mask, key_padding_mask=mask) # [T, B, D], [B, T, T]
        x = residual + self.dropout1(att.transpose(0,1))
        x = self.saln_1(x, style_embedding) if self.style_norm else self.norm1(x) 

        residual = x
        x = residual + self.dropout2(self.ffn(x.transpose(1,2)).transpose(1,2)) # [B, T, D(model)]
        x = self.saln_2(x, style_embedding) if self.style_norm else self.norm2(x)

        return x, attn_matrix
