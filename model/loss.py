import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_mask_from_lengths
from utils.utils import display_result_and_save_tensorboard

def get_accuracy(pred, target_id):
    
    _, predicted = pred.max(-1)
    accuracy = (predicted == target_id).float().mean() * 100 

    return accuracy
    
class LSGANLoss(nn.Module):
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.criterion = nn.MSELoss()
        
    def forward(self, r, is_real):

        if is_real: 
            ones = torch.ones(r.size(), requires_grad=False).to(r.device) # [B]: [1, 1, 1, ..., 1]
            loss = self.criterion(r, ones)
        else:
            zeros = torch.zeros(r.size(), requires_grad=False).to(r.device) # [B]: [0, 0, 0, ..., 0]
            loss = self.criterion(r, zeros)

        return loss

class LossFunction(nn.Module):
    def __init__(self, hp):
        super(LossFunction, self).__init__()

        self.l1_distance            = nn.L1Loss(reduction="mean")
        self.cross_entropy          = nn.CrossEntropyLoss(reduction="mean")

        self.recon_weight           = hp.recon_weight           
        self.pitch_weight           = hp.pitch_weight           
        self.adv_spk_weight         = hp.adv_spk_weight         
        self.style_weight           = hp.style_weight      

    def forward(self, outputs, labels, iteration, rank, device, writer):
        
        # parsing(outputs)      
        pitch_embedding_A1          = outputs['pitch_embedding_A1']
        style_embedding_A1          = outputs['style_embedding_A1']
        style_embedding_A2          = outputs['style_embedding_A2']
        mel_output_A1               = outputs['mel_output_A1']
        mel_output_postnet_A1       = outputs['mel_output_postnet_A1']
        adv_content_speaker_logits  = outputs['adv_content_speaker_logits']

        # parsing(labels)
        mel_A1, mel_len_A1, pitch_A1, speaker_id_A1 = labels
        T = mel_A1.size(1)
        
        # recon
        mel_mask_A1                 = ~get_mask_from_lengths(mel_len_A1, 192).unsqueeze(-1)  # [T, T, ...., F], [B, T, 1]
        mel_output_A1               = mel_output_A1.masked_select(mel_mask_A1).float()
        mel_output_postnet_A1       = mel_output_postnet_A1.masked_select(mel_mask_A1).float()
        mel_A1                      = mel_A1.masked_select(mel_mask_A1).float()

        loss_mel                    = self.recon_weight * self.l1_distance(mel_output_A1, mel_A1)
        loss_mel_post               = self.recon_weight * self.l1_distance(mel_output_postnet_A1, mel_A1)
        
        # pitch
        pitch_embedding_A1_mean     = pitch_embedding_A1.mean(-1).masked_select(mel_mask_A1.squeeze(-1)).float() 
        pitch_A1                    = pitch_A1.masked_select(mel_mask_A1.squeeze(-1)).float()
        
        loss_pitch                  = self.pitch_weight * self.l1_distance(pitch_embedding_A1_mean, pitch_A1) 

        # adversarial speaker classifier
        adv_label = speaker_id_A1.unsqueeze(1).repeat(1,T)
        adv_label.masked_fill_(mel_mask_A1.squeeze(-1) == False, -100) 
        loss_adv_content_speaker    = self.adv_spk_weight* F.cross_entropy(
            adv_content_speaker_logits.transpose(1,2), adv_label, ignore_index=-100) # # [B, D, T], [B, T]
        acc_adv_content_speaker     = get_accuracy(adv_content_speaker_logits, adv_label)

        # style generalization        
        loss_style                  = self.style_weight * (1 - F.cosine_similarity(style_embedding_A2, style_embedding_A1.detach(), dim=-1).mean())

        loss_total                  = loss_mel + loss_mel_post + loss_style + loss_adv_content_speaker + loss_pitch

        result_dict = {
            'loss_total': loss_total.item(),
            'loss_mel': loss_mel.item(),
            'loss_mel_post': loss_mel_post.item(),
            'loss_pitch': loss_pitch.item(),
            'loss_style': loss_style.item(),

            'loss_adv_content_speaker': loss_adv_content_speaker.item(),
            'acc_adv_content_speaker': acc_adv_content_speaker.item(),
        }

        if rank == 0:
            print(f'Iter {iteration:<6d} total {loss_total.item():<6.3f} mel {loss_mel.item():<6.3f} mel_post {loss_mel_post.item():<6.3f} pitch {loss_pitch.item():<6.3f} style {loss_style.item():<6.3f} adv_spk {loss_adv_content_speaker.item():<6.3f} adv_spk {acc_adv_content_speaker.item():<6.3f}%')  

        return loss_total, result_dict
