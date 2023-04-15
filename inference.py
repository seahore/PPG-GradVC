# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import argparse
import json
from pathlib import Path
import librosa
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS

import sys
from hifigan.env import AttrDict
from hifigan.models import Generator as HiFiGAN

from transformers import Wav2Vec2ForCTC


HIFIGAN_CONFIG = './hifigan/config_v1.json'
HIFIGAN_CHECKPT = './hifigan/g_00875000'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='path to output directory')
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=100, help='number of timesteps of reverse diffusion')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print('Initializing Grad-TTS...')
    generator = GradTTS(params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    

    print("Loading W2V2 for content...")
    cmodel = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").cuda()
    cmodel.eval()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    with torch.no_grad():
        for i, line in enumerate(lines):
            fname, src_wav, tgt_id = line.split("|")
            print(f'Converting {i} wav...', end=' ')
            audio, _ = librosa.load(src_wav, sr=params.sample_rate)
            audio = torch.from_numpy(audio).unsqueeze(0).cuda()
            x = cmodel(audio).logits.transpose(1, 2)
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            spk = torch.LongTensor([int(tgt_id)]).cuda()

            t = dt.datetime.now()
            y_enc, y_dec = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 16000 / (y_dec.shape[-1] * 320)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(f'./{args.out_dir}/{fname}.wav', 16000, audio)

    print('Done. Check out `out` folder for samples.')
