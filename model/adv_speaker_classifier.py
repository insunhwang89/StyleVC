import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Linear, Conv1d

class AdversarialSpeakerClassifier(nn.Module):
    def __init__(self, hp):
        super(AdversarialSpeakerClassifier, self).__init__()

        model_in = hp.model_dim
        hidden_dim =  hp.adv_speaker_classifier_dim
        dropout_ratio = hp.adv_speaker_classifier_dropout_ratio
        kernel_size = hp.adv_speaker_classifier_kernel_size
        n_speakers = hp.n_speakers

        self.conv1d_1 = Conv1d(model_in, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv1d_2 = Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        self.layernorm1 = nn.LayerNorm(hidden_dim) 
        self.layernorm2 = nn.LayerNorm(hidden_dim) 

        self.linear_layer = Linear(hidden_dim, n_speakers)
        
    def forward(self, x):
        
        # x: [B, T, D]
        x = F.dropout(F.relu(self.layernorm1(self.conv1d_1(x.transpose(1,2)).transpose(1,2))))
        x = F.dropout(F.relu(self.layernorm2(self.conv1d_2(x.transpose(1,2)).transpose(1,2))))

        x = self.linear_layer(x) # [B, T, n_speakers]

        return x  

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)
