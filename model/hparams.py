

import random
import numpy as np

# directory setup
output_directory = 'outputs' 
dataset = "VCTK" 

log_directory = 'StyleVC_VCTK'

print("log directory! ----->", log_directory)

# training parameters
lr = 0.0001
train_batch_size = 24
num_workers = 8
seq_len = 192

dataset_path = 'VCTK_16K' 

unseen_speakers_male = ['p226', 'p227', 'p232', 'p237', 'p241', 'p243', 'p245', 'p246','p247', 'p251']
unseen_speakers_female = ['p225', 'p228', 'p229', 'p230', 'p231', 'p233', 'p234', 'p236', 'p238', 'p239']
unseen_speakers = unseen_speakers_male + unseen_speakers_female

seen_speakers_male = ['p252', 'p254', 'p255', 'p256', 'p258', 'p259', 'p260', 'p263',
                'p270', 'p271', 'p272', 'p273', 'p274', 'p275', 'p278', 'p279', 'p281', 'p284',
                'p285', 'p286', 'p287', 'p292', 'p298', 'p302', 'p304', 'p311', 'p316',
                'p326', 'p334', 'p345', 'p347', 'p360', 'p363', 'p364', 'p374']
seen_speakers_female = ['p240', 'p244', 'p248', 'p249', 'p250', 'p253', 'p257', 'p261', 'p262', 'p264',
                'p265', 'p266', 'p267', 'p268', 'p269', 'p276', 'p277', 'p280', 'p282', 'p283', 'p288',
                'p293', 'p294', 'p295', 'p297', 'p299', 'p300', 'p301', 'p303', 'p305', 'p306',
                'p307', 'p308', 'p310', 'p312', 'p313', 'p314', 'p317', 'p318', 'p323', 'p329',
                'p330', 'p333', 'p335', 'p336', 'p339', 'p340', 'p341', 'p343', 'p351', 'p361',
                'p362']
seen_speakers = seen_speakers_male + seen_speakers_female

print("seen speakers! {} unseen speakers! {}".format(len(seen_speakers), len(unseen_speakers)))
n_speakers = len(seen_speakers)
speaker_ids = spk2idx = dict(zip(seen_speakers, range(len(seen_speakers))))

val_batch_size = 2
iters_per_validation = 1000
iters_per_checkpoint = 50000
iters_per_online_inference = 1000
stop_iteration = 100001
seed = random.randint(1, 10000)

# multi-processing
dist_backend = "nccl"
dist_url = "tcp://localhost:54321"
world_size = 1

# audio parameters
sampling_rate = 16000
MAX_WAV_VALUE = 32768.0
n_fft = 1024
n_mels = 80
hop_size = 256
win_size = 1024
fmin = 0 # Tacotron2, Glow-TTS와 같은 1st stage 모델들의 출력이, 표현 가능한 주파수 영역 전체가 아닌 일부 주파수 영역으로 제한되어 있음
fmax = 8000
fmax_for_loss = None
eps = 1e-9
clip_val = 1e-5

# loss
recon_weight = 10
pitch_weight = 20
adv_spk_weight = 0.5
style_weight = 0.5

# model
mel_dim = 80
model_dim = 192
hidden_dim = 192

# wav2vec
wav2vec_dim = 1024

# style encoder
style_hidden_dim = 192
style_dim = 192

# content encoder
prenet_dim = 192
content_kernel_size = 3
encoder_attn_n_layers = 4
encoder_attn_n_heads = 2
encoder_ffn_dim = 1024
encoder_dropout = 0.1

# pitch predictor
pitch_predictor_attn_n_layers = encoder_attn_n_layers
pitch_predictor_attn_n_heads = encoder_attn_n_heads
pitch_predictor_dropout = 0.5
pitch_predictor_ffn_dim = encoder_ffn_dim

# decoder
decoder_kernel_size = content_kernel_size
decoder_attn_n_layers = encoder_attn_n_layers
decoder_attn_n_heads = encoder_attn_n_heads
decoder_ffn_dim = encoder_ffn_dim
encoder_style_dim = 0 # not use adaptive style normalization
decoder_dropout = encoder_dropout

# postnet
postnet_in_dim = mel_dim
postnet_hidden_dim = 192
postnet_n_layers = 5
postnet_kernel_size = 5

# adversarial speaker classifier
adv_speaker_classifier_dim = 1024
adv_speaker_classifier_dropout_ratio = 0.5
adv_speaker_classifier_kernel_size = 3
