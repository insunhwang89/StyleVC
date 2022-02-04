
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import glob
import pickle
import random

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler
from preprocessing import mel_spectrogram, generate_features
import math
import torch.nn.functional as F

from scipy import signal
from preprocessing import fs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, hp):
        
        self.seq_len            = hp.seq_len
        self.speaker_ids        = hp.speaker_ids        
        self.unseen_speakers    = hp.unseen_speakers
        self.cache_data         = dict()

        # dataset        
        self.npz_path = self.get_npz_path(data_dir, self.unseen_speakers)
        random.seed(random.randint(1, 10000))
        random.shuffle(self.npz_path)
        print("load dataset num: {}".format(len(self.npz_path)))

    def get_npz_path(self, path, unseen_speakers):
        
        spk_path = glob.glob(os.path.join(path, '*'))

        npz_path = list()
        for spk in spk_path: # ['d:/datasets/VCTK/VCTK_16K_hifi_org_alignment/train\\p225', ... ,]
            # spk: data/VCTK/VCTK_16K/train/p330

            # unseen speaker는 학습하지 않음.
            if spk.split('/')[-1] in unseen_speakers:
                continue
            
            npz_path += glob.glob(os.path.join(spk, r"*.npz")) 
            
        return npz_path          

    def get_sample(self, npz_path):

        if npz_path in self.cache_data.keys(): 
            return self.cache_data[npz_path]

        else:
            npz = np.load(npz_path, allow_pickle=True)
            
            text = npz['text']
            processed_audio = npz['processed_audio'] # [wav_length]
            melspec = torch.FloatTensor(npz['melspec']).unsqueeze(0).transpose(1,2) # [T, 80] -> [1, 80, T]

            pitch = npz['pitch']
            pitch_min = 74.9358171247276
            pitch_max = 599.9855831449905
            index = np.where(pitch>0)
            pitch[index] = (pitch[index] - pitch_min) / (pitch_max - pitch_min)
            pitch = torch.FloatTensor(pitch).unsqueeze(0)  # [1, T]

            speaker_name = str(npz['speaker_name']) # ex: 'p225'
            speaker_id = self.speaker_ids[speaker_name] # ex: 45
            
            self.cache_data[npz_path] = (processed_audio, melspec, pitch, speaker_id, text)

            return self.cache_data[npz_path]

    def __getitem__(self, index):
        
        # choose X_(A, i)
        data1 = self.get_sample(self.npz_path[index])

        # choose X_(A, k)
        spk_path = '/'.join(self.npz_path[index].split('/')[:-1]) # spk name, data/VCTK/VCTK_16K/train/p283
        choose_list = list()
        for path in self.npz_path: # data/VCTK/VCTK_16K/train/p240/p240_348.npz
            spk_path_ = '/'.join(path.split('/')[:-1]) # data/VCTK/VCTK_16K/train/p240
            if spk_path_ == spk_path:
                choose_list.append(path)        
        npz_path2 = random.choice(choose_list)        
        data2 = self.get_sample(npz_path2)

        return (data1, data2) # X_(A,i), X_(A,k)
    
    def __len__(self):
    
        return len(self.npz_path)

class DatasetCollate():
    def __init__(self, hp):

        self.seq_len = hp.seq_len

        self.segment_size = 49152
        self.sampling_rate = hp.sampling_rate
        self.MAX_WAV_VALUE = hp.MAX_WAV_VALUE
        self.n_fft = hp.n_fft
        self.n_mels = hp.n_mels
        self.hop_size = hp.hop_size
        self.win_size = hp.win_size
        self.fmin = hp.fmin
        self.fmax = hp.fmax
        self.eps = hp.eps
        self.clip_val = hp.clip_val

    def parsing_batch(self, batch):

        audios        = [b[0] for b in batch]
        mels          = [b[1] for b in batch]
        pitchs        = [b[2] for b in batch]
        spk_id        = [b[3] for b in batch] 
        text          = [b[4] for b in batch] 

        audio_seq, audio_len_seq = list(), list()
        mel_seq, mel_len_seq = list(), list()

        pitch_seq = list()
        content_audio_seq = list()

        for i, (audio, mel, pitch) in enumerate(zip(audios, mels, pitchs)):  

            frames_per_seg = math.ceil(self.segment_size / self.hop_size) # 8192 / 256 = 32, 49152 / 256 = 192 
            
            content_audio = fs(audio).reshape(1,-1) # [1, wav_len]

            if mel.size(2) > 192:
                mel_start = random.randint(0, max(mel.size(2) - frames_per_seg - 1, 0)) # 최소값 0 보장(index 0부터 시작)
                mel_end = mel_start + frames_per_seg
                
                mel = mel[:, :, mel_start:mel_end] # [1, 80, 192]
                pitch = pitch[:, mel_start:mel_end] # [1, 192]

                content_audio = content_audio[:,mel_start*self.hop_size : mel_end*self.hop_size] # [1, 8192]

                mel_len = frames_per_seg

            else:
                mel_len = mel.size(2) # [B, 80, T]
                pad_size = frames_per_seg - mel_len

                mel = torch.nn.functional.pad(mel, (0, pad_size), 'constant') # [1, 80, frame_per_seg]
                pitch = torch.nn.functional.pad(pitch, (0, pad_size), 'constant') # [1, frame_per_seg]

                # padding 안함
                # content_audio = torch.nn.functional.pad(content_audio, (0, self.segment_size - content_audio.size(1)), 'constant') # [1, segment_size]
            
            audio_seq.append(audio)
            content_audio_seq.append(content_audio.reshape(-1)) 
            mel_seq.append(mel.squeeze(0).transpose(0,1)) # [T, 80]
            mel_len_seq.append(mel_len)            
            pitch_seq.append(pitch.squeeze(0)) # [T]

        mel_seq = np.stack(mel_seq, axis=0)
        mel_len_seq = np.stack(mel_len_seq, axis=0) # [T(seq_len), 80] * batch -> [B, T(seq_len), 80]
        pitch_seq = np.stack(pitch_seq, axis=0)

        out = {
            "text": text,                                   # [B]
            "audio_content": content_audio_seq,             # [wav_len] * B
            "mel": torch.FloatTensor(mel_seq),              # [B, T(seq_len), 80]
            "mel_len": torch.LongTensor(mel_len_seq),       # [B]
            "speaker_id": torch.LongTensor(spk_id),         # [B]
            "pitch": torch.FloatTensor(pitch_seq),          # [B, T(seq_len)]
        }

        return out

    def __call__(self, batch):
        
        # batch: ((X_(A,i), X_(B,k)), (), ... ) --> batch 기준으로 다시 묶기
        A1_data, A2_data = list(), list()
        for a1, a2 in batch:
            A1_data.append(a1)
            A2_data.append(a2)

        out1 = self.parsing_batch(A1_data)
        out2 = self.parsing_batch(A2_data) 

        return (out1, out2)    

def prepare_dataloaders(hp, rank, num_gpus, num_workers, batch_size):
    
    training_data_path = os.path.join('data', hp.dataset, hp.dataset_path, 'train')
    validation_data_path = os.path.join('data', hp.dataset, hp.dataset_path, 'val')
    
    # Get data, data loaders and collate function ready
    trainset = Dataset(training_data_path, hp)    
    collate_fn = DatasetCollate(hp)
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None

    train_loader = DataLoader(
        trainset, num_workers=num_workers, shuffle=False if num_gpus > 1 else True, pin_memory=True, \
        sampler=train_sampler, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)
    
    val_loader = None
    if rank == 0:
        valset = Dataset(validation_data_path, hp)
        val_loader = DataLoader(
            valset, num_workers=1, shuffle=True, pin_memory=True, sampler=None, 
            batch_size=hp.val_batch_size, drop_last=True, collate_fn=collate_fn)

    return train_loader, val_loader, train_sampler
