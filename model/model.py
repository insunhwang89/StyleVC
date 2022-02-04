import torch
import torch.nn as nn
import numpy as np

from model.layers import Linear, Conv1d, Conv2d
from model.encoder_style import StyleEncoder
from model.encoder_content import ContentEncoder
from model.decoder import Decoder
from model.postnet import Postnet
from model.utils import get_mask_from_lengths, rand_slice_segments, padding_sequence
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from model.adv_speaker_classifier import AdversarialSpeakerClassifier, grad_reverse
from model.encoder_f0 import PitchPredictor # F0Encoder as 

from jiwer import wer 


class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()

        self.load_wav2vec()

        self.content_encoder        = ContentEncoder(hp)
        self.style_encoder          = StyleEncoder(hp)
        self.pitch_predictor        = PitchPredictor(hp)

        self.decoder                = Decoder(hp)             
        self.postnet                = Postnet(hp)

        self.adv_content_speaker_classifier = AdversarialSpeakerClassifier(hp)  

    def load_wav2vec(self,):

        self.tokenizer              = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
        self.wav2vec2               = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h") 
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()

    def get_wav2vec_embedding(self, audio, frames_per_seg, mel_length, text="None"):

        # finetuning
        # https://www.facebook.com/groups/TensorFlowKR/posts/1441001666240887/
        # https://huggingface.co/facebook/wav2vec2-large-960h
        
        # batch단위로 max길이로 text_encoder가 생성하는 문제가 있음
        # 개별로 구한뒤에 content mask를 생성해야 한다.

        # audio: [B, wav len]
        with torch.no_grad():

            padded_content = list()
            for i, (a, mel_len) in enumerate(zip(audio, mel_length)):
                token_outputs = self.tokenizer(a, return_tensors="pt", padding="longest") # 이거 쓰는게 더 낫네?? ;;
                input_values = token_outputs.input_values.cuda() # [1, wav len]
                
                outputs = self.wav2vec2(input_values, output_hidden_states=True)
                hidden_states = outputs['hidden_states'][12] # [B, T', 1024] (padding안된 T임)                        
                
                hidden_states = torch.nn.functional.interpolate(
                    hidden_states.transpose(1,2), mel_len.item(), mode='nearest').transpose(1,2).squeeze(0) # [T', 1024]
                hidden_len = hidden_states.size(0) # [T']

                if isinstance(frames_per_seg, int):
                    # training
                    hidden_states = hidden_states.transpose(0,1) # [1024, T']
                    hidden_states = torch.nn.functional.pad(
                        hidden_states, (0, frames_per_seg - hidden_len), 'constant').transpose(0,1) # [1024, T] -> [T, 1024]

                # if True:
                #     # temp
                #     predicted_ids = torch.argmax(outputs['logits'], dim=-1)
                #     transcription = self.tokenizer.batch_decode(predicted_ids)[0]
                #     print(f'prediction:{transcription}, target: {text[i]}')
                    
                #     # from jiwer import wer 
                #     # print("WER:", wer(text, transcription))

                padded_content.append(hidden_states)
            text_embedding = torch.stack(padded_content) # [B, seq_len, 1024]  
            
        
        return text_embedding

    def forward(self, audio_content_A1, mel_A1, mel_len_A1, text_A1, mel_A2, mel_len_A2):

        # masking        
        mel_mask_A1 = get_mask_from_lengths(mel_len_A1, mel_A1.size(1)) # [B, T]: [F, F, F, ... , T]
        T1 = mel_mask_A1.size(1)

        # content encoding process
        with torch.no_grad():
            content_embedding_wav2vec_A1 = self.get_wav2vec_embedding(
                audio_content_A1, frames_per_seg=T1, mel_length=mel_len_A1, text=text_A1) # [B, wav_len_A] -> [B, T_A, 1024]  

        content_embedding_A1 = self.content_encoder(content_embedding_wav2vec_A1, mel_mask_A1) # [B, T, D(model)]
        adv_content_speaker_logits = self.adv_content_speaker_classifier(grad_reverse(content_embedding_A1)) # [B, T, n_speakers]
        
        # style # utilize unpaired samples: X_{A,i}, X_{B,k}
        mel_slice_A1, _ = rand_slice_segments(mel_A1.transpose(1,2), mel_len_A1, 96) # [B, 80, 32], [B]
        style_embedding_A1 = self.style_encoder(mel_slice_A1.transpose(1,2), None, None) # [B, D(style)]    
        style_embedding_A2 = self.style_encoder(mel_A2, mel_len_A2, mel_A2.size(1)) 

        # add f0 information
        pitch_embedding_A1 = self.pitch_predictor(content_embedding_A1, style_embedding_A2, mel_mask_A1) # [B, T, D]
        
        # decoding
        decoder_input = content_embedding_A1 + pitch_embedding_A1
        mel_output_A1, _ = self.decoder(decoder_input, style_embedding_A2, mel_mask_A1) # [B, T, D] 
        mel_output_postnet_A1 = self.postnet(mel_output_A1) + mel_output_A1

        outputs = {            
            'pitch_embedding_A1': pitch_embedding_A1,
            'style_embedding_A1': style_embedding_A1, 
            'style_embedding_A2': style_embedding_A2,

            'adv_content_speaker_logits': adv_content_speaker_logits,

            'mel_output_A1': mel_output_A1,
            'mel_output_postnet_A1': mel_output_postnet_A1,
        }

        return outputs

    def inference(self, audio_A, mel_A, mel_len_A, audio_B, mel_B, mel_len_B):
        
        # encoding
        content_embedding_wav2vec_A1 = self.get_wav2vec_embedding(
            audio_A, frames_per_seg=None, mel_length=mel_len_A) # [B, wav_len_A] -> [B, T, 1024]  
        content_embedding_A = self.content_encoder(content_embedding_wav2vec_A1, None) # [B, T, D(model)]

        # style
        style_embedding_B = self.style_encoder(mel_B, mel_len_B, mel_B.size(1))

        # add f0 information
        pitch_embedding_B = self.pitch_predictor(content_embedding_A, style_embedding_B, None) # [B, T, D]

        # decoding
        decoder_input = content_embedding_A + pitch_embedding_B
        mel_output, _ = self.decoder(decoder_input, style_embedding_B, None) # [B, T, D]
        mel_output_postnet = self.postnet(mel_output) + mel_output

        return mel_output_postnet
