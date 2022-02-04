import os
import sys
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import torch
import librosa
import model.hparams as hp
from preprocessing import call_generate_melspec

# multi processing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

device = "cuda" if torch.cuda.is_available() else "cpu"

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory' + directory)

def preprocessing(a, data):
    
    # preprocessing parameters
    MAX_WAV_VALUE       = hp.MAX_WAV_VALUE
    sampling_rate       = hp.sampling_rate
    n_fft               = hp.n_fft
    n_mels              = hp.n_mels
    hop_size            = hp.hop_size 
    win_size            = hp.win_size 
    fmin                = hp.fmin         
    fmax                = hp.fmax
    eps                 = hp.eps
    clip_val            = hp.clip_val

    for line in tqdm(data):

        # load data
        if a.dataset == "VCTK":
            wav_path, text, speaker_id = line[:-1].split('|')                       # 'wav48/p225/p225_001.wav|Please call Stella.|0\n'
            speaker_name = wav_path.split('/')[1]                                   # 'p225'
            wav_path = os.path.join(a.in_dir, wav_path)                             # 'data/VCTK/original/wav48/p225/p225_001.wav'         
            data_id = int(wav_path.split('/')[-1].split('.')[0].split('_')[-1])     # '1'
        elif a.dataset == "NIKL":
            wav_path, text, speaker_id, speaker_name = line[:-1].split('|')         # ['./wavs/fv01/fv01_t01_s01.wav', '기차도 전기도 없었다.', '0', 'fv01']
            wav_path = wav_path.replace('./', '')                                   # ./wavs/fv01/fv01_t01_s01.wav -> wavs/fv01/fv01_t01_s01.wav
            wav_path = os.path.join(a.in_dir, wav_path)                             # 'data/NIKL/original/wavs/fv01/fv01_t01_s01.wav' 
            data_id1, data_id2 = [int(a.lower().replace('t', '').replace('s','').replace('p','')) for a in wav_path.split('/')[-1].split('.wav')[0].split('_')[1:]]     # ['1', '1']

        if not os.path.exists(wav_path):
            continue
        
        audio, _ = librosa.load(wav_path, sr=sampling_rate)

        # except short or long audio
        if len(audio) < a.mel_min_len: # len(audio) > a.mel_max_len and 
            continue
        
        try:
            # convert wav to mel-spectrogram
            melspec, processed_audio = call_generate_melspec(
                audio, sampling_rate, MAX_WAV_VALUE, n_fft, n_mels, 
                hop_size, win_size, fmin, fmax, eps, clip_val) # [T, 80], [trimed wav length]
            assert melspec.size(1) == n_mels # [T, 80]

            pitch = generate_features(
                processed_audio, melspec.numpy(), 
                sampling_rate=sampling_rate, hopsize=hop_size, winsize=win_size) # [T], [T], [T] , intensity, energy

            # tokenize
            if a.dataset == "VCTK":
                text = english_cleaners(text.rstrip()) # 'Please call Stella.' -> 'PLEASE CALL STELLA.'
                phoneme = g2p(text)  
                token_sequence = phoneme_to_sequence(phoneme, 'english_cleaners') 
            elif a.dataset == "NIKL":
                phoneme_word = g2pk(text) # '이리 도망온 사슴을 못 보았소?' -> '이리 도망온 사스믈 몯 뽀앋쏘?'
                phoneme_sequence = tokenize(phoneme_word) # ['@', 'ᄋ', 'ᅵ', 'ᄅ', 'ᅵ', ' ', 'ᄃ', 'ᅩ', 'ᄆ', 'ᅡ', 'ᆼ', 'ᄋ', 'ᅩ', 'ᆫ', ...]
                token_sequence = phonemes_to_sequence(phoneme_sequence)
            token_len = len(token_sequence)

            # save data
            # split train and validation data(*condition: data_id > 20)
            if a.dataset == "VCTK":
                npz_filename = '{}'.format(wav_path.\
                    replace(a.in_dir_name, a.out_dir_name).
                    replace('wav48', 'train' if data_id > 20 else 'val').
                    replace('wav','npz').
                    replace('\\', '/')) # data/VCTK/VCTK_22K/train/p225/p225_001.npz
            elif a.dataset == "NIKL":
                npz_filename = '{}'.format(wav_path.\
                    replace(a.in_dir_name, a.out_dir_name).
                    replace('wavs', 'val' if data_id1 == 1 and data_id2 < 20 else 'train').
                    replace('wav','npz').
                    replace('\\', '/')) # data/NIKL/NIKL_22K/val/fv01/fv01_t01_s01.npz

            folder = '/'.join(npz_filename.split('/')[:-1]) # data/VCTK/VCTK_22K/train/p225, data/NIKL/NIKL_22K/val/fv01
            createFolder(folder)

            np.savez(
                npz_filename,   
                processed_audio=processed_audio,    # [wav length] # processed audio!(trim + normalize!)
                melspec=melspec,                    # [T, 80]
                pitch=pitch,                        # [T]
                text=text,                          # 'please call stella'
                token=token_sequence,               # [L]
                speaker_name=speaker_name)          # 'p225'
                
        except:
            print("continue!!! error occur!", npz_filename)
            continue
        

    print("end preprocessing")

# def calculate_rhythm(a, g2p, phoneme_to_sequence, tokenize):
    
#     # load npz path
#     spk_path = glob(os.path.join('data', a.dataset, a.out_dir_name, 'train', '*'))
#     npz_path = dict()
#     for spk in spk_path: 
#         spk_name = spk.replace("\\", "//").split('/')[-1] # ['d:/datasets/VCTK/VCTK_22K/train\\p225', ... ,]
#         npz_path[spk_name] = glob(os.path.join(spk, r"*.npz")) 

#     # calcurate average rhythm information
#     outputs = dict()
#     for spk, spk_path in tqdm(npz_path.items()):
#         for path in tqdm(spk_path):
            
#             # load
#             npz = np.load(path, allow_pickle=True)            
#             melspec = npz['melspec'] # [T, 80]
#             text = str(npz['text'])

#             # tokenize
#             if a.dataset == "VCTK":
#                 text = english_cleaners(text.rstrip()) # 'Please call Stella.' -> 'PLEASE CALL STELLA.'
#                 phoneme = g2p(text)  
#                 token_sequence = phoneme_to_sequence(phoneme, 'english_cleaners')
#             elif a.dataset == "NIKL":
#                 phoneme_word = g2p(text) # '이리 도망온 사슴을 못 보았소?' -> '이리 도망온 사스믈 몯 뽀앋쏘?'
#                 phoneme_sequence = tokenize(phoneme_word) # ['@', 'ᄋ', 'ᅵ', 'ᄅ', 'ᅵ', ' ', 'ᄃ', 'ᅩ', 'ᄆ', 'ᅡ', 'ᆼ', 'ᄋ', 'ᅩ', 'ᆫ', ...]
#                 token_sequence = phoneme_to_sequence(phoneme_sequence)   
#             text_len = len(token_sequence)

#             # calculate rhythm
#             rhythm = text_len / melspec.shape[0]

#             # store result
#             if spk not in outputs.keys():
#                 outputs[spk] = rhythm
#             else:
#                 outputs[spk] += rhythm

#     rhythm_path = os.path.join('data', a.dataset, a.out_dir_name, 'rhythm.txt')
#     with open(rhythm_path, encoding='utf-8', errors='ignore', mode="w") as file:
#         for spk, value in outputs.items():
    
#             outputs[spk] /= len(npz_path[spk])
#             file.writelines(spk + ',' + str(outputs[spk]) + '\n') # 'p225,0.20785321248486477\n'
#             print(spk, outputs[spk])

#     print("end calculating rhythm!")


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='data/VCTK/original/') # 'data/VCTK/original/, 'data/NIKL/original/
    parser.add_argument('--in_dir_name', default='original')
    parser.add_argument('--out_dir_name', default='VCTK_16K') # VCTK_16K, NIKL_16K
    parser.add_argument('--dataset', default='VCTK') # VCTK, NIKL
    parser.add_argument('--mel_min_len', default=hp.sampling_rate / 2.0) # over 0.5s    
    parser.add_argument('--num_workers', default=8)  
    a = parser.parse_args()
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--in_dir', default='data/NIKL/original/') # 'data/VCTK/original/, 'data/NIKL/original/
    # parser.add_argument('--in_dir_name', default='original')
    # parser.add_argument('--out_dir_name', default='NIKL_22K') # VCTK_22K, NIKL_22K
    # parser.add_argument('--dataset', default='NIKL') # VCTK, NIKL
    # parser.add_argument('--mel_min_len', default=hp.sampling_rate / 2.0) # over 0.5s    
    # parser.add_argument('--num_workers', default=8)  
    # a = parser.parse_args()
    

    print("start preprocessing")
    num_workers = a.num_workers if a.num_workers is not None else cpu_count()
    executor = ProcessPoolExecutor(max_workers=num_workers)

    # meta data
    if a.dataset == "VCTK":
        path = os.path.join(a.in_dir, 'metadata.csv') # data/VCTK/original/metadata.csv
    elif a.dataset == "NIKL":
        path = os.path.join(a.in_dir, 'metadata118.csv') # data/VCTK/original/metadata.csv

    # tokenizer
    if a.dataset == "VCTK":
        from text.text_English.cleaners import english_cleaners
        from text.text_English import phoneme_to_sequence
        from g2p_en import G2p
        g2p = G2p()
        tokenize = None
    elif a.dataset == "NIKL":
        from g2pk import G2p as g2pk
        from text.text_Korean.korea import tokenize
        from text.text_Korean import phonemes_to_sequence as phoneme_to_sequence
        g2p = g2pk()


    # # for debugging
    # path = os.path.join(a.in_dir, 'metadata118.csv') # data/VCTK/original/metadata.csv
    # with open(path, encoding='utf-8') as f:
    #     print(path)
    #     data = f.readlines()
    # preprocessing(a, data)


    futures = list()
    with open(path, encoding='utf-8') as f:
        print(path)
        data = f.readlines()
        interval = int(len(data)/num_workers)
        for idx in range(0, interval):            
            start = idx*interval
            end = (idx+1)*interval if idx != interval - 1 else -1
            futures.append(executor.submit(partial(preprocessing, a, data[start:end])))
    result_list = [future.result() for future in tqdm(futures)]

    # print("start calculating rhythm!")
    # calculate_rhythm(a, g2p, phoneme_to_sequence, tokenize)

if __name__ == '__main__':
    main()


