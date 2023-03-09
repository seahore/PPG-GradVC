# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch
import librosa
import numpy as np

import params

from model.base import BaseModule
from model.encoder import MelEncoder
from model.postnet import PostNet
from model.diffusion import Diffusion
from model.utils import sequence_mask, fix_len_compatibility, mse_loss

from conformer_ppg_model.build_ppg_model import load_ppg_model


# the whole voice conversion model consisting of the "average voice" encoder 
# and the diffusion-based speaker-conditional decoder
class DiffVC(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, spk_dim, use_ref_t, dec_dim, 
                 beta_min, beta_max):
        super(DiffVC, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.spk_dim = spk_dim
        self.use_ref_t = use_ref_t
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.encoder = None
        self.decoder = Diffusion(n_feats, dec_dim, spk_dim, use_ref_t, 
                                 beta_min, beta_max)

    def load_encoder(self, conf_path, model_path, device='cuda'):
        self.encoder = load_ppg_model(conf_path, model_path, device)

    @torch.no_grad()
    def extract_ppg(self, wavs, sr, device='cuda'):
        ppgs = []
        for wav in wavs:
            if sr != 16000:
                wav = torch.tensor(librosa.resample(wav.cpu().detach().numpy(), orig_sr=sr, target_sr=16000)).to(device)
            ppg = self.encoder(wav.unsqueeze(0), torch.LongTensor([len(wav)]))[0].transpose(0,1)
            ppgs.append(ppg.cpu().detach().numpy())
        ppgs = np.array(ppgs)
        ppgs = np.delete(ppgs, np.shape(ppgs)[2] - 1, 2)
        ppgs = torch.tensor(ppgs).to(device)
        return ppgs

    @torch.no_grad()
    def forward(self, x_wav, x_mel, x_lengths, x_ref, x_ref_lengths, c, n_timesteps, 
                mode='ml'):
        """
        Generates mel-spectrogram from source mel-spectrogram conditioned on
        target speaker embedding. Returns:
            1. 'average voice' encoder outputs
            2. decoder outputs
        
        Args:
            x_wav (torch.Tensor): batch of source wave data.
            x_mel (torch.Tensor): batch of source mel-spectrograms.
            x_lengths (torch.Tensor): numbers of frames in source mel-spectrograms.
            x_ref (torch.Tensor): batch of reference mel-spectrograms.
            x_ref_lengths (torch.Tensor): numbers of frames in reference mel-spectrograms.
            c (torch.Tensor): batch of reference speaker embeddings
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            mode (string, optional): sampling method. Can be one of:
              'pf' - probability flow sampling (Euler scheme for ODE)
              'em' - Euler-Maruyama SDE solver
              'ml' - Maximum Likelihood SDE solver
        """
        x_wav, x_mel, x_lengths = self.relocate_input([x_wav, x_mel, x_lengths])
        x_ref, x_ref_lengths, c = self.relocate_input([x_ref, x_ref_lengths, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x_mel.dtype)
        x_ref_mask = sequence_mask(x_ref_lengths).unsqueeze(1).to(x_ref.dtype)
        ppg = self.extract_ppg(x_wav, params.sampling_rate)

        b = x_mel.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mel_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x_mel.dtype, 
                                device=x_mel.device)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x_mel.dtype)
        ppg_new = torch.zeros((b, 144, max_length_new), dtype=ppg.dtype,
                              device=ppg.device)
        
        for i in range(b):
            x_mel_new[i, :, :x_lengths[i]] = x_mel[i, :, :x_lengths[i]]
            ppg_new[i, :, :x_lengths[i]] = ppg_new[i, :, :x_lengths[i]]

        z = torch.randn_like(x_mel_new, device=x_mel_new.device)

        y = self.decoder(z, x_mask_new, ppg_new, x_ref, x_ref_mask, c, 
                         n_timesteps, mode)
        return ppg, y[:, :, :max_length]

    def compute_loss(self, x_wav, x_mel, x_lengths, x_ref_mel, c):
        """
        Computes diffusion (score matching) loss.
            
        Args:
            x_wav (torch.Tensor): batch of source wave data.
            x_mel (torch.Tensor): batch of source mel-spectrograms.
            x_lengths (torch.Tensor): numbers of frames in source mel-spectrograms.
            x_ref_mel (torch.Tensor): batch of reference mel-spectrograms.
            c (torch.Tensor): batch of reference speaker embeddings
        """
        x_wav, x_mel, x_lengths, x_ref_mel, c = self.relocate_input([x_wav, x_mel,x_lengths, x_ref_mel, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x_mel.dtype)
        ppg = self.extract_ppg(x_wav, params.sampling_rate)
        diff_loss = self.decoder.compute_loss(x_mel, x_mask, ppg, x_ref_mel, c)
        return diff_loss
