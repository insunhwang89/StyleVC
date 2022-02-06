from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from .env import AttrDict
from .meldataset_custom import mel_spectrogram, MAX_WAV_VALUE, load_wav
from .model_mel80 import Generator
import numpy as np
from librosa.util import normalize

h = None
device = None 

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

# def inference(a, option="npy"):
def inference(checkpoint_path, input_wavs_dir, output_dir, option="npy"):

    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(checkpoint_path, device)
    generator.load_state_dict(state_dict_g['generator'])

    if option == "npy":
        filelist = glob.glob(os.path.join(input_wavs_dir, '*.npy'))
    else:
        filelist = glob.glob(os.path.join(input_wavs_dir, '*.npz'))
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):

            # Melspectrogram(from npy)
            if option == "npy":
                x = np.load(filname)
                x = torch.from_numpy(x).cuda()
                if x.dim() == 2:
                    x = x.unsqueeze(0) # [B(1), 80, T]

            if len(x.size()) == 2:
                continue # inference에서 T가 1인 음성을 생성할 수 있음

            # Generate wav 
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            # Save Result
            filname = filname.replace('\\', '/').split('/')[-1].split('.npy')[0] # ex: p226_016-to-p226_016
            output_file = os.path.join(output_dir, filname + '.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)

def main(hparams, path):
    """
    How to use?
    1. Put npy file in folder './inference/npy'
    2. Run inference_npy.py
    3. Get wav file in folder './inference/generated'
    """

    print('Initializing Inference Process..')

    # if hparams.dataset == "VCTK":
    #     checkpoint_path = os.path.join('vocoder/checkpoint/VCTK_22K_V1/g_00500000')
    #     config_path = os.path.join('vocoder/checkpoint/VCTK_22K_V1/config.json')
    # elif a.dataset == "NIKL":
    #     checkpoint_path = os.path.join('vocoder/checkpoint/NIKL_22K_V1/g_00500000')
    #     config_path = os.path.join('vocoder/checkpoint/NIKL_22K_V1/config.json')
    # else:
    #     Exception("Unknown dataset!")
    checkpoint_path = os.path.join('vocoder/checkpoint/VCTK_16K_V1/g_01700000')
    config_path = os.path.join('vocoder/checkpoint/VCTK_16K_V1/config.json')

    
    input_wavs_dir = os.path.join('generated', path)
    output_dir = os.path.join('generated', path)

    print("[*] Load checkpoint path: {}".format(checkpoint_path))
    

    with open(config_path) as f:
        data = f.read()

    global h, device
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # inference(a)
    inference(checkpoint_path, input_wavs_dir, output_dir)


if __name__ == '__main__':
    main()


#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--dataset', default=hparams.dataset)
#     parser.add_argument('--input_wavs_dir', default=os.path.join('generated', path))
#     parser.add_argument('--output_dir', default=os.path.join('generated', path))
#     parser.add_argument('--checkpoint_path', default=checkpoint_path)
#     parser.add_argument('--config_path', default=config_path)
#     a = parser.parse_args()

#     print("[*] Load checkpoint path: {}".format(a.checkpoint_path))
    
#     with open(a.config_path) as f:
#         data = f.read()

#     global h, device
#     json_config = json.loads(data)
#     h = AttrDict(json_config)

#     torch.manual_seed(h.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(h.seed)
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')

#     inference(a)


# if __name__ == '__main__':
#     main()
