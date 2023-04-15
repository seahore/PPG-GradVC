# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch
import torchaudio as ta

from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed
from params import train_frames

import sys
from hifigan.meldataset import mel_spectrogram


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, 
                 n_fft=1280, n_mels=80, sample_rate=16000,
                 hop_length=320, win_length=1280, f_min=0., f_max=8000):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, speaker = line[0], line[1]
        ssl = self.get_ssl(filepath)
        mel = self.get_mel(filepath)
        speaker = self.get_speaker(speaker)
        return (ssl, mel, speaker)

    def get_ssl(self, filepath):
        filepath = filepath.replace(".flac", ".pt")
        filepath = filepath.replace("/wavs/", "/w2v2-ppgs/")
        ssl = torch.load(filepath).squeeze(0)
        return ssl

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        ssl, mel, speaker = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': ssl, 'spk': speaker}
        return item

    def __len__(self):
        return len(self.filelist)


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        # y_max_length = max([item['y'].shape[-1] for item in batch])
        # y_max_length = fix_len_compatibility(y_max_length)
        n_mel_feats = batch[0]['y'].shape[-2]
        n_ssl_feats = batch[0]['x'].shape[-2]

        y = torch.zeros((B, n_mel_feats, train_frames), dtype=torch.float32)
        x = torch.zeros((B, n_ssl_feats, train_frames), dtype=torch.float32)
        y_lengths = []
        spk = []

        max_starts = [max(item['x'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            minl = min(x_.shape[-1], y_.shape[-1]) # some mel_len = ssl_len + 1
            if minl < train_frames:
                y_length = minl
            else:
                y_length = train_frames
            y[i, :, :y_length] = y_[:, starts[i]:starts[i] + y_length]
            x[i, :, :y_length] = x_[:, starts[i]:starts[i] + y_length]
            y_lengths.append(y_length)
            spk.append(spk_)

        # for i, item in enumerate(batch):
        #     y_, x_, spk_ = item['y'], item['x'], item['spk']
        #     minl = min(x_.shape[-1], y_.shape[-1]) # some mel_len = ssl_len + 1
        #     y_lengths.append(minl)
        #     y[i, :, :minl] = y_[:, :minl]
        #     x[i, :, :minl] = x_[:, :minl]
        #     spk.append(spk_)
        
        y_lengths = torch.LongTensor(y_lengths)
        spk = torch.cat(spk, dim=0)
        return {'x': x, 'y': y, 'y_lengths': y_lengths, 'spk': spk}
