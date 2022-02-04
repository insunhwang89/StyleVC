import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 상위 폴더내 파일 참조

import glob
import numpy as np
import torch
import random

from preprocessing import fs
from preprocessing import mel_spectrogram

def make_test_pairs(data_root, test_list):

    # make test pairs
    all_test_pairs = []
    with open(test_list, 'rt') as f:
        all_line = f.readlines()
        for line in all_line:
            source, target = line.strip().split('|')
            all_test_pairs.append(
                (os.path.join(data_root, source), os.path.join(data_root, target))
            )

    return all_test_pairs

def generate_path(path, speaker):

    speaker_path_list = glob.glob(os.path.join(path, speaker, '*npz'))
    random_id = random.randint(0, len(speaker_path_list)-1)
    file_path = speaker_path_list[random_id] # 'data/VCTK/VCTK_22K/val/p226/p226_006.npz'

    return file_path 

def make_inference_sample_list(path, source_list, target_list):

    """
        0:'data/VCTK/VCTK_22K/val/p226/p226_003.npz|data/VCTK/VCTK_22K/val/p226/p226_001.npz\n'
        1:'data/VCTK/VCTK_22K/val/p226/p226_003.npz|data/VCTK/VCTK_22K/val/p226/p226_017.npz\n'
        2:'data/VCTK/VCTK_22K/val/p226/p226_003.npz|data/VCTK/VCTK_22K/val/p226/p226_001.npz\n'
        3:'data/VCTK/VCTK_22K/val/p226/p226_003.npz|data/VCTK/VCTK_22K/val/p227/p227_020.npz\n'
        ...
    """
    source_speaker = source_list[random.randint(0, len(source_list)-1)] # ex: 'p226'
    target_speaker = target_list[random.randint(0, len(target_list)-1)]

    source_line = generate_path(path, source_speaker)  
    target_line = generate_path(path, target_speaker)    
    
    conversion_pair = [source_line + "|" + target_line + "\n"]

    return conversion_pair

def make_experiment_conversion_pair(path, dataset, male, female, n_samples=1):

    """
    매번 새로운 conversion pair를 'sample1', 'sample2', ..., 'sampleN' 으로 생성한다.
    ex
    - val/p226/p226_001.npz|val/p225/p225_002.npz --> sample1
    - val/p228/p228_003.npz|val/p227/p227_004.npz --> sample2
    """

    conversion_list = list()
    conversion_list += make_inference_sample_list(path, male, male)
    conversion_list += make_inference_sample_list(path, male, female)
    conversion_list += make_inference_sample_list(path, female, male)
    conversion_list += make_inference_sample_list(path, female, female)

    return conversion_list

def pad_sequences(mel, max_len):

    return np.pad(mel, ((0, max_len - mel.shape[0]), (0, 0)), 'constant')

def load_data(hp, path):

    # path: 'data/VCTK/VCTK_22K/val/p298/p298_016.npz'
    spk_name = path.split('/')[-2] # 'p298'
    spk_data = path.split('/')[-1].split('.')[0] # 'p298_016'

    npz = np.load(path)

    audio = npz['processed_audio'].reshape(1, -1)
    melspec = torch.FloatTensor(npz['melspec']).unsqueeze(0).cuda() # [1, T, 80]
    mel_len = torch.LongTensor([melspec.size(1)]).cuda() # [1]             

    return spk_name, spk_data, audio, melspec, mel_len 