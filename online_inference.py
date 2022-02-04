import os
import glob
import numpy as np
import librosa

import torch
from torch.nn import functional as F

from vocoder.inference_npy import main as run_vocoder
from utils.inference_utils import make_experiment_conversion_pair, pad_sequences, load_data


def run_online_inference(hp, model, writer, iteration):

    model.eval()
    data_root = os.path.join('data', hp.dataset, hp.dataset_path, 'val')
    experiment_dataset = make_experiment_conversion_pair(
        data_root, hp.dataset, hp.seen_speakers_male, hp.seen_speakers_female)

    # create directory
    save_dir = "generated/{}".format(hp.log_directory)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        outputs = list()

        for i, conversion_pair in enumerate(experiment_dataset):
            # 'data/VCTK/VCTK_22K/val/p298/p298_016.npz|data/VCTK/VCTK_22K/val/p274/p274_015.npz\n'
            src_spk, trg_spk = conversion_pair.strip().split('|') 

            # load data            
            src_spk_name, src_data, audio_A, mel_A, mel_len_A = load_data(hp, src_spk)
            trg_spk_name, trg_data, audio_B, mel_B, mel_len_B = load_data(hp, trg_spk)
            
            outputs_A2B = model.inference(audio_A, mel_A, mel_len_A, audio_B, mel_B, mel_len_B) # [1, 1, wav len]
            outputs_A2A = model.inference(audio_A, mel_A, mel_len_A, audio_A, mel_A, mel_len_A)

            # store result
            file_name = f"sample{i}" # .npy"
            result_dict = {
                "source_mel" : mel_A.squeeze(0).float().detach().cpu().numpy().T, # [1, T, 80] -> [80, T]
                "target_mel" : mel_B.squeeze(0).float().detach().cpu().numpy().T,                     
                "recon_mel" : outputs_A2A.squeeze(0).float().detach().cpu().numpy().T,
                "converted_mel" : outputs_A2B.squeeze(0).float().detach().cpu().numpy().T,
                "source_audio": audio_A.reshape(-1),
                "target_audio": audio_B.reshape(-1), 
                "path": file_name
            }
            outputs.append(result_dict)
            print(file_name + " is done!")

        # Save result to pdf and npy
        for idx, output in enumerate(outputs):

            # parsing
            source_mel = output['source_mel'] 
            target_mel = output['target_mel']
            recon_mel = output['recon_mel']
            converted_mel = output['converted_mel']
            source_audio = output['source_audio']
            target_audio = output['target_audio']
            path = output['path']

            # save npy
            np.save(os.path.join(save_dir, path + '_source'), source_mel)
            np.save(os.path.join(save_dir, path + '_target'), target_mel)
            np.save(os.path.join(save_dir, path + '_source_recon'), recon_mel)
            np.save(os.path.join(save_dir, path + '_source_convert'), converted_mel)

            # add padding
            T1, T2, T3, T4 = source_mel.shape[1], target_mel.shape[1], recon_mel.shape[1], converted_mel.shape[1]
            max_len = max([T1, T2, T3, T4])
                
            source_mel = pad_sequences(source_mel.T, max_len).T
            target_mel = pad_sequences(target_mel.T, max_len).T
            recon_mel = pad_sequences(recon_mel.T, max_len).T 
            converted_mel = pad_sequences(converted_mel.T, max_len).T

            # write melspectrogram on tensorboard
            writer.add_specs([
                source_mel, target_mel, recon_mel, converted_mel],                      
                iteration, 'a_inference_src_tgt_vcRec_vcConvert' + str(idx))

        # run vocoder
        run_vocoder(hp, path=hp.log_directory)

        # write audio on tensorboard
        wav_path = glob.glob('generated/'+ hp.log_directory +'/*.wav')
        for k, path in enumerate(wav_path):
            wav, _ = librosa.load(os.path.join(path), sr=hp.sampling_rate)
            name = path.replace('\\', '/').split('/')[-1].split('.')[0]
            writer.add_audio('generated/{}'.format(name), wav, iteration, hp.sampling_rate)

    return True
    