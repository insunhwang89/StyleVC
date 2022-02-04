import torch
from utils.utils import display_result_and_save_tensorboard
from torch.nn import functional as F
from model.utils import get_mask_from_lengths
from utils.utils import display_result_and_save_tensorboard

def train(
    hp, model_g, optimizer_g, model_d, optimizer_d, criterion_d,
    criterion_g, writer, iteration, batch1, batch2, device, rank, scaler):

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

    # criterion
    labels = mel_A1, mel_len_A1, pitch_A1, speaker_id_A1
    loss_total, result_dict = criterion_g(
        outputs, labels, iteration, rank, device, writer)   
    
    loss_g_p = criterion_d(d_outputs_p, is_real=True) 
    loss_g_s = criterion_d(d_outputs_s, is_real=True)

    loss_total = loss_total + loss_g_p + loss_g_s

    result_dict['loss_g_p'] = loss_g_p.item()
    result_dict['loss_g_s'] = loss_g_s.item()

    # backward & update
    optimizer_g.zero_grad()
    scaler.scale(loss_total).backward()
    scaler.unscale_(optimizer_g)
    torch.nn.utils.clip_grad_norm_(model_g.parameters(), 1.0)
    scaler.step(optimizer_g)
    scaler.update()       

    # Discriminator
    with torch.cuda.amp.autocast(enabled=True): 
        
        mel_output = outputs['mel_output_postnet_A1'].detach()
        style_embedding_A2 = outputs['style_embedding_A2'].detach()
        pitch_embedding = outputs['pitch_embedding_A1'].mean(-1).detach()
        
        d_real_p, d_real_s, loss_d_cls = model_d(
            mel_A1, style_embedding_A2, pitch_A1, speaker_id_A1, mel_len_A1)
        d_fake_p, d_fake_s, _ = model_d(
            mel_output, None, pitch_embedding, speaker_id_A1, mel_len_A1)
    
    # criterion
    loss_d_s = criterion_d(d_real_s, is_real=True) + criterion_d(d_fake_s, is_real=False)
    loss_d_p = criterion_d(d_real_p, is_real=True) + criterion_d(d_fake_p, is_real=False)

    loss_d_total = loss_d_s + loss_d_cls + loss_d_p

    result_dict['loss_d_s'] = loss_d_s.item()
    result_dict['loss_d_p'] = loss_d_p.item()
    result_dict['loss_d_cls'] = loss_d_cls.item()

    # backward & update
    optimizer_d.zero_grad()
    scaler.scale(loss_d_total).backward()
    scaler.unscale_(optimizer_d)
    torch.nn.utils.clip_grad_norm_(model_d.parameters(), 1.0)
    scaler.step(optimizer_d)
    scaler.update()   

    if rank == 0: 
        display_result_and_save_tensorboard(writer, result_dict, iteration)

    return True


