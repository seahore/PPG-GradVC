# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
from data_ppg import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from utils import plot_tensor, save_plot


train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
n_spks = params.n_spks
spk_emb_dim = params.spk_emb_dim

log_dir = "logs/ppg_noth" #params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed
save_every = params.save_every

n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale


if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelSpeakerDataset(train_filelist_path, 
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=2, shuffle=True)
    test_dataset = TextMelSpeakerDataset(valid_filelist_path, 
                                         n_fft, n_feats, sample_rate, hop_length,
                                         win_length, f_min, f_max)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=2, shuffle=True)

    print('Initializing model...')
    model = GradTTS(n_spks, spk_emb_dim, n_enc_channels,
                    filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale)
    model.load_state_dict(torch.load("./logs/ppg_noth/grad_200.pt", map_location=lambda loc, storage: loc))
    _ = model.cuda()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Start training...')
    iteration = 0
    for epoch in range(201, n_epochs + 1):
        model.train()
        diff_losses = []
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                x = batch['x'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                spk = batch['spk'].cuda()
                
                diff_loss = model.compute_loss(x, y, y_lengths,
                                                spk=spk, out_size=out_size)
                loss = sum([diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 
                                                            max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 
                                                            max_norm=1)
                optimizer.step()

                logger.add_scalar('training/diffusion_loss', diff_loss,
                                global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                global_step=iteration)
                
                msg = f'Epoch: {epoch}, iteration: {iteration} | diff_loss: {diff_loss.item()}'
                progress_bar.set_description(msg)
                
                diff_losses.append(diff_loss.item())
                iteration += 1

        msg = 'Epoch %d: diffusion loss = %.5f ' % (epoch, np.mean(diff_losses))
        # msg += '| prior loss = %.5f ' % np.mean(prior_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(msg + '\n')

        if epoch % save_every == 0:
            model.eval()
            print('Synthesis...')
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    x = batch['x'][:1].cuda()
                    y, y_lengths = batch['y'][:1], batch['y_lengths'][:1].cuda()
                    spk = batch['spk'][:1].cuda()
                    break

                y_enc, y_dec = model(x, y_lengths, n_timesteps=50, spk=spk)
                logger.add_image(f'image/ground_truth', plot_tensor(y.squeeze()),
                                global_step=iteration, dataformats='HWC')
                logger.add_image(f'image/generated_enc',
                                plot_tensor(y_enc.squeeze().cpu()),
                                global_step=iteration, dataformats='HWC')
                logger.add_image(f'image/generated_dec',
                                plot_tensor(y_dec.squeeze().cpu()),
                                global_step=iteration, dataformats='HWC')
                save_plot(y.squeeze(), f'{log_dir}/original.png')
                save_plot(y_enc.squeeze().cpu(), 
                        f'{log_dir}/generated_enc.png')
                save_plot(y_dec.squeeze().cpu(), 
                        f'{log_dir}/generated_dec.png')
        
            ckpt = model.state_dict()
            torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
