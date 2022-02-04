
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LEAKY_RELU = 0.1

class LinearNorm(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True,  spectral_norm=False):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias)
        
        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        out = self.fc(input)
        return out
        
class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True,  spectral_norm=False):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)
        
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0., spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_model, 0.5), dropout=dropout)

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = nn.utils.spectral_norm(self.w_qs)
            self.w_ks = nn.utils.spectral_norm(self.w_ks)
            self.w_vs = nn.utils.spectral_norm(self.w_vs)
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
                        sz_b, len_x, -1)  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn


def get_mask_from_lengths(lengths, seq_len=None): # , max_len=None):

    # if max_len is None:
    max_len = torch.max(lengths).item() # temp
    if seq_len is not None:
        max_len = seq_len # max_len if max_len > seq_len else seq_len

    ids = lengths.new_tensor(torch.arange(0, max_len)).to(lengths.get_device()) # .cuda()
    mask = (lengths.unsqueeze(1) <= ids).to(lengths.get_device()) # .cuda()

    return mask # [B, seq_len]: [[F, F, F, ..., T, T, T], ... ]

def dot_product_logit(a, b):
    n = a.size(0)
    m = b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = (a*b).sum(dim=2)
    return logits

class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()

        self.style_D = StyleDiscriminator(hp.n_speakers, 80, hp.style_dim, hp.style_dim, 5, 2)        
        self.pitch_P = PitchDiscriminator(80, 128)

    def forward(self, mels, style_embedding, pitch, sids, mel_len):

        # mels: [B, T, D] 
        # # style_embedding: [B, D] or None 
        # # pitch: [B, T]
        # sids: [B]
        # mel_len: [B]

        mask = get_mask_from_lengths(mel_len, 192)  # [F, F, F, ...., T, T], [B, T]
        mels = mels.masked_fill(mask.unsqueeze(-1), 0)
            
        p_val = self.pitch_P(mels, pitch, mask, mel_len)
        s_val, ce_loss = self.style_D(mels, style_embedding, sids, mask)

        return p_val, s_val, ce_loss
    
class StyleDiscriminator(nn.Module):
    def __init__(self, n_speakers, input_dim, hidden_dim, style_dim, kernel_size, n_head):
        super(StyleDiscriminator, self).__init__()

        self.style_prototypes = nn.Embedding(n_speakers, style_dim)

        self.spectral = nn.Sequential(
            LinearNorm(input_dim, hidden_dim, spectral_norm=True),
            nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim, hidden_dim, spectral_norm=True),
            nn.LeakyReLU(LEAKY_RELU),
        )

        self.temporal = nn.ModuleList([nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size, spectral_norm=True),
            nn.LeakyReLU(LEAKY_RELU)) for _ in range(2)])

        self.slf_attn = MultiHeadAttention(n_head, hidden_dim, hidden_dim//n_head, hidden_dim//n_head, spectral_norm=True) 

        self.fc = LinearNorm(hidden_dim, hidden_dim, spectral_norm=True)
        self.V = LinearNorm(style_dim, hidden_dim, spectral_norm=True)
        self.w = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def temporal_avg_pool(self, xs, mask):

        xs = xs.masked_fill(mask.unsqueeze(-1), 0)
        len_ = (~mask).sum(dim=1).unsqueeze(1) # mel len
        xs = torch.sum(xs, dim=1)
        xs = torch.div(xs, len_) 

        return xs

    def forward(self, mels, ws, sids, mask):
        max_len = mels.shape[1]

        # Update style prototypes
        if ws is not None:
            style_prototypes = self.style_prototypes.weight.clone()
            logit = dot_product_logit(ws, style_prototypes) 
            cls_loss = F.cross_entropy(logit, sids)
        else: 
            cls_loss = None

        # Style discriminator
        x = self.spectral(mels)  # [B, T, 384]

        for conv in self.temporal: # 2번
            residual = x
            x = x.transpose(1,2)
            x = conv(x)
            x = x.transpose(1,2)
            x = residual + x

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) # [B, T, T]
        x, _ = self.slf_attn(x, slf_attn_mask)

        x = self.fc(x) # [B, T, 128]
        h = self.temporal_avg_pool(x, mask) # [B, 128]

        ps = self.style_prototypes(sids) # [B, 128]
        s_val = self.w * torch.sum(self.V(ps)*h, dim=1) + self.b # [B]

        return s_val, cls_loss

class PitchDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PitchDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim

        self.mel_prenet = nn.Sequential(
            LinearNorm(input_dim, hidden_dim, spectral_norm=True),
            nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim, hidden_dim, spectral_norm=True),
            nn.LeakyReLU(LEAKY_RELU),
        )

        self.fcs = nn.Sequential(
            LinearNorm(hidden_dim*2, hidden_dim*2, spectral_norm=True),
            nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim*2, hidden_dim*2, spectral_norm=True),
            nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim*2, hidden_dim*2, spectral_norm=True),
            nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim*2, 1, spectral_norm=True)
        )


    def forward(self, mels, pitch, mask, mel_len):

        # mels: [B, T, 80]
        batch_size, max_len = mels.shape[0], mels.shape[1]

        mels = self.mel_prenet(mels) # [B, T, 128]
        
        xs = torch.cat((mels, pitch.unsqueeze(-1).repeat(1,1,mels.size(2))), dim=-1) # [B, T, 128*2]

        xs = self.fcs(xs)
        t_val = xs.squeeze(-1)

        # temporal avg pooling
        t_val = t_val.masked_fill(mask, 0.)
        t_val = torch.div(torch.sum(t_val, dim=1), mel_len)

        return t_val
