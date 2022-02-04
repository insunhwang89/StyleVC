import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa
import torch
from model.model import Model
import model.hparams as hp
from utils.inference_utils import make_test_pairs, load_data
from vocoder.inference_npy import main as run_vocoder

def run_inference(): 
    checkpoint_model = 'StyleVC_VCTK'

    # hyper parameter
    experiment_dataset = ['seen_list.txt', 'unseen_list.txt']
    org_save = False # True    
    
    data_path = os.path.join('data', 'VCTK', hp.dataset_path) # data/VCTK/VCTK_16K
    experiment_dataset = {
        'seen': os.path.join('data', 'VCTK', experiment_dataset[0]),
        'unseen': os.path.join('data', 'VCTK', experiment_dataset[1])}
    checkpoint_model_path = os.path.join('outputs', checkpoint_model, 'checkpoint_G_100000')

    print(" load checkpoint path --> ", checkpoint_model_path)
    print(" inference checkpoint --> ", checkpoint_model)

    # load model
    model = Model(hp).cuda()
    model.load_state_dict(torch.load(os.path.join(checkpoint_model_path))['model'])
    model.eval()
    print("load checkpoint: ", checkpoint_model_path)

    with torch.no_grad():
        for experiment_name, test_conversion_pair_path in experiment_dataset.items():
            print("\nstart test! --> ", experiment_name)

            # save directory
            save_dir = "generated/{}/{}".format(experiment_name, checkpoint_model) 
            os.makedirs(save_dir, exist_ok=True)

            # load evaluation test
            all_test_pairs = make_test_pairs(data_path, test_conversion_pair_path)

            for i, (src_spk, trg_spk) in enumerate(tqdm(all_test_pairs)):

                # src_spk:trg_spk --> 'data/VCTK/VCTK_22K/val/p298/p298_016.npz|data/VCTK/VCTK_22K/val/p274/p274_015.npz\n'
                src_spk_name, src_data, audio_A, mel_A, mel_len_A = load_data(hp, src_spk)
                trg_spk_name, trg_data, audio_B, mel_B, mel_len_B = load_data(hp, trg_spk)
                
                mel_outputs_postnet_convert = model.inference(audio_A, mel_A, mel_len_A, audio_B, mel_B, mel_len_B) # [1, 1, wav len]

                # save result
                source_mel = mel_A.squeeze(0).float().detach().cpu().numpy().T # [1, T, 80] -> [80, T] 
                target_mel = mel_B.squeeze(0).float().detach().cpu().numpy().T               
                converted_mel = mel_outputs_postnet_convert.squeeze(0).float().detach().cpu().numpy().T
                mel_path = "{}_to_{}.npy".format(src_data, trg_data) # p294_005_to_p334_005.npy

                # inference result
                np.save(os.path.join(save_dir, mel_path), converted_mel)

                # save original waveform for MOS test
                if org_save == True: 
                    # original ground-truthss
                    wav_path = mel_path.replace('.npy', '.wav')
                    
                    os.makedirs(os.path.join(save_dir, 'GT_source_wav'), exist_ok=True)
                    os.makedirs(os.path.join(save_dir, 'GT_target_wav'), exist_ok=True)
                    sf.write(os.path.join(save_dir, 'GT_source_wav', wav_path), audio_A.cpu().numpy().reshape(-1), hp.sampling_rate, 'PCM_16')
                    sf.write(os.path.join(save_dir, 'GT_target_wav', wav_path), audio_B.cpu().numpy().reshape(-1), hp.sampling_rate, 'PCM_16')

                    # reconstructed ground-truths
                    os.makedirs(os.path.join(save_dir, 'GT_source'), exist_ok=True)
                    os.makedirs(os.path.join(save_dir, 'GT_target'), exist_ok=True)
                    np.save(os.path.join(save_dir, 'GT_source', mel_path), source_mel)
                    np.save(os.path.join(save_dir, 'GT_target', mel_path), target_mel)    
                # break     
            
            if org_save == True:
                run_vocoder(hp, path=os.path.join(experiment_name, 'GT_source')) 
                run_vocoder(hp, path=os.path.join(experiment_name, 'GT_target'))
            run_vocoder(hp, path=os.path.join(experiment_name, checkpoint_model))

    print("complete inference!")


if __name__ == '__main__':
    
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    run_inference()
    