import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa
import torch
from model.model import Model
import model.hparams as hp
from utils.inference_utils import make_test_pairs, load_data # make_experiment_conversion_pair, pad_sequences
from vocoder.inference_npy import main as run_vocoder
import glob
import random


def run_inference():
    
    checkpoint_model = 'ICPR09_DIS_ATT02'

    data_path = os.path.join('data', 'VCTK', hp.dataset_path, 'train') # data/VCTK/VCTK_16K
    checkpoint_model_path = os.path.join('outputs', checkpoint_model, 'checkpoint_G_100000') # checkpoint_G_50000, checkpoint_G_100000
    print(" *** checkpoint 폴더가 아닌 outputs 폴더에서 불러옴 ***")
    print(" load checkpoint path --> ", checkpoint_model_path)
    print(" inference checkpoint --> ", checkpoint_model)

    # load model
    model = Model(hp).cuda()
    model.load_state_dict(torch.load(os.path.join(checkpoint_model_path))['model'])
    model.eval()
    print("load checkpoint: ", checkpoint_model_path)

    with torch.no_grad():

        # load speaker path
        speaker_embeddings = dict()  
        content_embeddings = dict()      
        speaker_paths = glob.glob(os.path.join(data_path, '*')) # ['data/VCTK/VCTK_16K/train/p330', ... ]
        
        for i, speaker_path in tqdm(enumerate(speaker_paths)):
            
            npz_paths = glob.glob(os.path.join(speaker_path, '*npz')) 
            speaker_name = speaker_path.split('/')[-1]
            speaker_embeddings[speaker_name] = list()
            content_embeddings[speaker_name] = list()
            for j, npz_path in enumerate(npz_paths):

                _, _, audio, mel, mel_len = load_data(hp, npz_path)

                speaker_embedding = model.style_encoder(mel, mel_len, mel.size(1))
                speaker_embeddings[speaker_name].append(speaker_embedding.data.cpu().numpy())

                content_embedding_wav2vec = model.get_text_embedding(
                    audio, frames_per_seg=mel.size(1), mel_length=mel_len, text=None, option="Text") # [B, wav_len_A] -> [B, T', 1024]  
                content_embedding = model.content_encoder(content_embedding_wav2vec, None) # [B, T, D(model)] # con->mel mask
                content_embeddings[speaker_name].append(content_embedding.mean(1).data.cpu().numpy())

                # content embedding 추가할 것
                # if j == 3:
                #     break

        # Save result to pdf and npy
        if not os.path.exists('outputs/embeddings'):
            os.makedirs('outputs/embeddings')
        
        for key, item in speaker_embeddings.items():
            speaker_embeddings[key] = np.vstack(speaker_embeddings[key]) # [20, T(192)]
            np.save(f'outputs/embeddings/speaker_embeddings_{key}.npy', speaker_embeddings[key])  # seen unseen 모두 출력

        for key, item in content_embeddings.items():
            content_embeddings[key] = np.vstack(content_embeddings[key]) # [20, T(192)]
            np.save(f'outputs/embeddings/content_embeddings_{key}.npy', content_embeddings[key])  # seen unseen 모두 출력

    print("complete extract speaker embedding!!")


if __name__ == '__main__':
    
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    run_inference()