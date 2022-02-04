import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def make_masks(input_lengths, output_lengths, src_max_len, trg_max_len):
    """
    Args:
        input_lengths (LongTensor or List): Batch of lengths (B,).
        output_lengths (LongTensor or List): Batch of lengths (B,).

    Examples:
        >>> input_lengths, output_lengths = [5, 2], [8, 5]
        >>> _make_mask(input_lengths, output_lengths)
        tensor([[[1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]],
                [[1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    # in_masks = make_non_pad_mask(input_lengths)  # (B, T_in)
    # out_masks = make_non_pad_mask(olens)  # (B, T_out)
    in_masks = ~get_mask_from_lengths(input_lengths, src_max_len)
    out_masks = ~get_mask_from_lengths(output_lengths, trg_max_len)

    # return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)
    return in_masks.unsqueeze(-1) & out_masks.unsqueeze(-2)  # (B, L, T)

def make_guided_attention_masks(input_lengths, output_lengths, src_max_len, trg_max_len, sigma):

    n_batches = len(input_lengths)
    # max_ilen = max(max(input_lengths), src_max_len)
    # max_olen = max(max(output_lengths), trg_max_len)    
    
    max_ilen = torch.LongTensor([src_max_len]).to(input_lengths.get_device())[0] # todo:....    
    max_olen = torch.LongTensor([trg_max_len]).to(input_lengths.get_device())[0]
    # guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen)).to(input_lengths.get_device()) # .cuda().cuda() # [B, trg_len, src_len]
    guided_attn_masks = torch.zeros((n_batches, max_ilen, max_olen)).to(input_lengths.get_device()) # .cuda().cuda() # [B, L, T]

    for idx, (ilen, olen) in enumerate(zip(input_lengths, output_lengths)):
        ilen = max_ilen if ilen > max_ilen else ilen # todo..
        olen = max_olen if olen > max_olen else olen

        # mask = make_guided_attention_mask(ilen, olen, sigma) # [trg_len, src_len]
        # guided_attn_masks[idx, :olen, :ilen] = mask # [B, trg_len, src_len]
        mask = make_guided_attention_mask(olen, ilen, sigma) # [L, T]
        guided_attn_masks[idx, :ilen, :olen] = mask # [B, L, T]

    return guided_attn_masks

def make_guided_attention_mask(input_lengths, output_lengths, sigma):
    """Make guided attention mask.

    Examples:
        >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
        >>> guided_attn_mask.shape
        torch.Size([5, 5])
        >>> guided_attn_mask
        tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
        >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
        >>> guided_attn_mask.shape
        torch.Size([6, 3])
        >>> guided_attn_mask
        tensor([[0.0000, 0.2934, 0.7506],
                [0.0831, 0.0831, 0.5422],
                [0.2934, 0.0000, 0.2934],
                [0.5422, 0.0831, 0.0831],
                [0.7506, 0.2934, 0.0000],
                [0.8858, 0.5422, 0.0831]])

    """
    grid_x, grid_y = torch.meshgrid(torch.arange(output_lengths), torch.arange(input_lengths))
    grid_x, grid_y = grid_x.float().to(input_lengths.get_device()), grid_y.float().to(input_lengths.get_device()) # .cuda() # to(input_lengths.device)

    return 1.0 - torch.exp(-((grid_y / input_lengths - grid_x / output_lengths) ** 2) / (2 * (sigma ** 2)))

def create_look_ahead_mask(target_seq):
    
    """
    [B, seq_len, dim] -> [1, seq_len, seq_len]
    target sequence인 경우 그 다음 단어를 못보게 가린다(seq2seq하게 생성하기 위해)
    mask = tensor([[[ False,  True,   True,   True,   True,   True,   True],
                    [ False,  False,   True,  True,   True,   True,   True],
                    [ False,  False,  False,  True,   True,   True,   True],
                    [ False,  False,  False,  False,  True,   True,   True],
                    [ False,  False,  False,  False,  False,  True,  True],
                    [ False,  False,  False,  False,  False,  False,  True],
                    [ False,  False,  False,  False,  False,  False,  False]]])
                    
    """
    B, seq_len = target_seq.size(0), target_seq.size(1)

    return ~(1-torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool().cuda()

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        out_list = list()
        max_len = mel_max_length
        for i, batch in enumerate(input_ele):
            if batch.dim() >= 2:
                one_batch_padded = F.pad(
                    batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            else:
                one_batch_padded = F.pad(
                    batch, (0, max_len - batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    else:
        out_list = list()
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

        for i, batch in enumerate(input_ele):
            if batch.dim() >= 2:
                one_batch_padded = F.pad(
                    batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            else:
                one_batch_padded = F.pad(
                    batch, (0, max_len - batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded

def get_mask_from_lengths(lengths, seq_len=None): # , max_len=None):

    # if max_len is None:
    max_len = torch.max(lengths).item() # temp
    if seq_len is not None:
        max_len = seq_len # max_len if max_len > seq_len else seq_len

    ids = lengths.new_tensor(torch.arange(0, max_len)).to(lengths.get_device()) # .cuda()
    mask = (lengths.unsqueeze(1) <= ids).to(lengths.get_device()) # .cuda()

    return mask # [B, seq_len]: [[F, F, F, ..., T, T, T], ... ]


def slice_segments(x, ids_str, segment_size=4):
    
    ret = torch.zeros_like(x[:, :, :segment_size]) # [B, D, 32]
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]

    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):

    b, d, t = x.size()  
    if x_lengths is None:
        x_lengths = t

    ids_str_max = x_lengths - segment_size -1 # + 1 # [ 92,  25, 137,  92, 199, 183] - 32 + 1 = [ 61,  -6, 106,  61, 168, 152]
    if ids_str_max is not None:
        ids_str_max[ids_str_max<0] = 0 # [ 61,   0, 106,  61, 168, 152]

    ids_str = torch.stack([randn * ids_str_max[i] for i, randn in enumerate(list(np.random.rand(b)))]).long()
    ret = slice_segments(x, ids_str, segment_size)
    
    return ret, ids_str


def padding_sequence(x, frames_per_seg):

    padded = list()
    for embedding in x:
        embedding = embedding.transpose(0,1) # [1024, T]
        embedding = torch.nn.functional.pad(
            embedding, (0, frames_per_seg - embedding.size(1)), 'constant').transpose(0,1) # [1024, T] -> [seq_len, 1024]
        padded.append(embedding)

    return torch.stack(padded) # [B, seq_len, 1024]  