import pathlib
import numpy as np
import IPython.display as ipd
from tqdm import tqdm
from scipy.io.wavfile import write

import torch
use_gpu = torch.cuda.is_available()

import params
from model import DiffVC

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(sr=params.sampling_rate, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path


def get_mel(wav_path):
    wav, _ = load(wav_path, sr=params.sampling_rate)
    wav = wav[:(wav.shape[0] // params.hop_size)*params.hop_size]
    wav = np.pad(wav, int(params.hop_size * 3), mode='reflect') # pad for hop_size*3 because the window is 3 times longer than hop length.
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=params.hop_size, win_length=640, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

if __name__ == '__main__':
    wav_dir = pathlib.Path('D:\\Workspace\\Datasets\\UnifiedDataset\\wavs')
    mel_dir = pathlib.Path('D:\\Workspace\\Datasets\\UnifiedDataset\\mels')
    embed_dir = pathlib.Path('D:\\Workspace\\Datasets\\UnifiedDataset\\embeds')

    speakers = []
    for sub in wav_dir.iterdir():
        if sub.is_dir():
            speakers.append(sub.name)

    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')
    spk_encoder.load_model(enc_model_fpath, device='cuda' if torch.cuda.is_available() else 'cpu')
    for speaker in speakers:
        for wav_file in wav_dir.joinpath(speaker).glob('**/*.flac'):
            print(f'Analyzing {wav_file.name}')
            mel_speaker_dir = Path(mel_dir.joinpath(speaker))
            embed_speaker_dir = Path(embed_dir.joinpath(speaker))
            mel_speaker_dir.mkdir(parents=True, exist_ok=True)
            embed_speaker_dir.mkdir(parents=True, exist_ok=True)
            mel_path = mel_speaker_dir.joinpath(wav_file.name.split('.')[0]+'_mel.npy')
            np.save(mel_path, get_mel(wav_file))
            embed_path = embed_speaker_dir.joinpath(wav_file.name.split('.')[0]+'_embed.npy')
            np.save(embed_path, get_embed(wav_file))