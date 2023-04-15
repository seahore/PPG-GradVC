# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch

from model.base import BaseModule
from model.text_encoder import CEncoder
from model.diffusion import Diffusion
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility


class GradTTS(BaseModule):
    def __init__(self, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(GradTTS, self).__init__()
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.spec_min = -12
        self.spec_max = 2

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        #self.encoder = CEncoder(80, 80, 192, 5, 1, 16, gin_channels=spk_emb_dim) 
        self.encoder = CEncoder(392, 80, 192, 5, 1, 16, gin_channels=spk_emb_dim) 
        #self.encoder = CEncoder(40, 80, 192, 5, 1, 16, gin_channels=spk_emb_dim)
        self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x_, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        x_max_length_ = x_.shape[-1]
        x_max_length = fix_len_compatibility(x_max_length_)
        mask = sequence_mask(x_lengths, x_max_length).unsqueeze(1).to(x_)
        x = torch.zeros((x.size(0), x.size(1), x_max_length)).to(x_)
        x[:,:,:x_max_length_] = x_

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu = self.encoder(x, mask, spk.unsqueeze(-1))

        encoder_outputs = mu[:, :, :x_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu + torch.randn_like(mu, device=mu.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, mask, mu, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :x_max_length]
        decoder_outputs = self.denorm_spec(decoder_outputs)

        return encoder_outputs, decoder_outputs

    def compute_loss(self, x, y, y_lengths, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, y, y_lengths = self.relocate_input([x, y, y_lengths])
        y = self.norm_spec(y)

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        y_max_length = y.shape[-1]
        mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu = self.encoder(x, mask, spk.unsqueeze(-1))

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, mask, mu, spk)
        
        return diff_loss

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
