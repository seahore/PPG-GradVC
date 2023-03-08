# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import random
import numpy as np
import torch
import tgt
import librosa

from params import seed as random_seed
from params import n_mels, train_frames, sampling_rate, hop_size


def get_test_speakers():
    test_speakers = ['19', '26', '27', '32', '39', 
                     '40', '60', '78', '83', '87',
                     'KoharuKurama', 'RentaroKurama']
    return test_speakers


# LibriTTS dataset for training speaker-conditional diffusion-based decoder
class VCDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, val_file, exc_file):
        self.wav_dir = os.path.join(data_dir, 'wavs')
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.test_speakers = get_test_speakers()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.test_speakers]
        self.speakers = [spk for spk in self.speakers
                         if len(os.listdir(os.path.join(self.mel_dir, spk))) >= 10]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        with open(val_file) as f:
            valid_ids = f.readlines()
        self.valid_ids = set([v.strip() + '_mel.npy' for v in valid_ids])
        self.exceptions += self.valid_ids

        self.valid_info = [(v.split('_')[0], v.split('-')[0]) for v in self.valid_ids]
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m not in self.exceptions]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        wavs = self.get_wavs(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (wavs, mels, embed)

    def get_wavs(self, audio_id, spk, format='flac'):
        wav_path = os.path.join(self.wav_dir, spk, audio_id + f'.{format}')
        wavs, _ = librosa.load(wav_path, sr=sampling_rate)
        wavs = wavs[:(wavs.shape[0] // hop_size)*hop_size]
        wavs = torch.from_numpy(wavs).float()
        return wavs
    
    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mels = np.load(mel_path)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed

    def __getitem__(self, index):
        wavs, mels, embed = self.get_vc_data(self.train_info[index])
        item = {'wav': wavs, 'mel': mels, 'c': embed}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            wavs, mels, embed = self.get_vc_data(self.valid_info[i])
            pairs.append((wavs, mels, embed))
        return pairs


class VCDecBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels2 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        wavs1 = torch.zeros((B, train_frames * hop_size), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            wav = item['wav']
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            wavs1[i, :mel_length * hop_size] = wav[starts1[i]*hop_size:(starts1[i]+mel_length)*hop_size]
            mels2[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)
        return {'wav': wavs1, 'mel1': mels1, 'mel2': mels2, 'mel_lengths': mel_lengths, 'c': embed}
