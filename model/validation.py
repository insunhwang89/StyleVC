import os
import torch
import torch.nn as nn
from model.utils import get_mask_from_lengths
from torch.nn import functional as F

def validation(hp, model_g, criterion_g, model_d, criterion_d, val_loader, iteration, device, writer, rank=0):

    model_g.eval()
    model_d.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        samples = 0     
        total_result = dict()
        total_result2 = dict()
        for i, (batch1, batch2) in enumerate(val_loader): # X_(A,i), X_(A,k)
            samples += 1

            # parsing
            text_A1          = batch1["text"]
            mel_A1           = batch1["mel"].to(device, non_blocking=True)              # [B, T, 80]
            mel_len_A1       = batch1["mel_len"].to(device, non_blocking=True)          # [B]
            audio_content_A1 = batch1["audio_content"]                                  # [wav_len] * B        
            speaker_id_A1    = batch1["speaker_id"].to(device, non_blocking=True)       # [B] 
            pitch_A1         = batch1["pitch"].to(device, non_blocking=True)            # [B]  

            mel_A2           = batch2["mel"].to(device, non_blocking=True)              # [B, T, 80]
            mel_len_A2       = batch2["mel_len"].to(device, non_blocking=True)          # [B]

            # Generator
            with torch.cuda.amp.autocast(enabled=True): 
                
                outputs = model_g(
                    audio_content_A1, mel_A1, mel_len_A1, text_A1,
                    mel_A2, mel_len_A2)

                mel_output = outputs['mel_output_postnet_A1']
                pitch_embedding = outputs['pitch_embedding_A1'].mean(-1)

                d_outputs_p, d_outputs_s, _ = model_d(
                    mel_output, None, pitch_embedding, speaker_id_A1, mel_len_A1)

            labels = mel_A1, mel_len_A1, pitch_A1, speaker_id_A1
            loss_total, result_dict = criterion_g(
                outputs, labels, iteration, rank, device, writer)   
            
            loss_g_p = criterion_d(d_outputs_p, is_real=True) 
            loss_g_s = criterion_d(d_outputs_s, is_real=True)

            loss_total = loss_total + loss_g_p + loss_g_s
            result_dict['loss_g_p'] = loss_g_p.item()
            result_dict['loss_g_s'] = loss_g_s.item()

            # Discriminator
            with torch.cuda.amp.autocast(enabled=True): 
                
                mel_output = outputs['mel_output_postnet_A1'].detach()
                style_embedding_A2 = outputs['style_embedding_A2'].detach()
                pitch_embedding = outputs['pitch_embedding_A1'].mean(-1).detach()

                d_real_p, d_real_s, loss_cls = model_d(
                    mel_A1, style_embedding_A2, pitch_A1, speaker_id_A1, mel_len_A1)
                d_fake_p, d_fake_s, _ = model_d(
                    mel_output, None, pitch_embedding, speaker_id_A1, mel_len_A1)
            
            loss_d_s = criterion_d(d_real_s, is_real=True) + criterion_d(d_fake_s, is_real=False)
            loss_d_p = criterion_d(d_real_p, is_real=True) + criterion_d(d_fake_p, is_real=False)

            loss_d_total = loss_d_s + loss_d_p + loss_cls
            result_dict['loss_d_s'] = loss_d_s.item()
            result_dict['loss_d_p'] = loss_d_p.item()
            result_dict['loss_cls'] = loss_cls.item()
            
            # store result
            for key, value in result_dict.items():
                if key in total_result:
                    total_result[key] += value
                else:
                    total_result[key] = value
            
            # display result
            if i == 0:         

                B, T, D = mel_A1.size() 
                mel_mask_A1 = ~get_mask_from_lengths(mel_len_A1, T) # [B, T]: [T, T, T, ..., F, F, F]

                total_result2['mel_source_org'] = mel_A1.transpose(1,2) # [B, D, T]
                total_result2['mel_output_A1']  = (outputs['mel_output_postnet_A1'] * mel_mask_A1.unsqueeze(-1)).transpose(1,2) # [B, D, T]

                for k in range(B):
                    writer.add_specs([
                        total_result2['mel_source_org'][k].detach().cpu().float(), 
                        total_result2['mel_output_A1'][k].detach().cpu().float(),
                        ],
                        iteration, 'val_src_reconA1' + str(k))                         

            if i == 20:                
                for key, value in total_result.items():     
                    writer.add_scalar('val_' + key, value/samples, iteration)
                break

    return True
