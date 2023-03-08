import torch
import librosa
import numpy as np
import pathlib
import params
from conformer_ppg_model.build_ppg_model import load_ppg_model

@torch.no_grad()
def extract_ppg(model, wavs, sr, device='cuda'):
    ppgs = []
    for wav in wavs:
        if sr != 16000:
            wav = torch.tensor(librosa.resample(wav.cpu().detach().numpy(), orig_sr=sr, target_sr=16000)).to(device)
        ppg = model(wav.unsqueeze(0), torch.LongTensor([len(wav)]))[0].transpose(0,1)
        ppgs.append(ppg.cpu().detach().numpy())
    ppgs = np.array(ppgs)
    ppgs = np.delete(ppgs, np.shape(ppgs)[2] - 1, 2)
    ppgs = torch.tensor(ppgs).to(device)
    return ppgs



if __name__ == '__main__':
    ppg_model = load_ppg_model(
        './conformer_ppg_model/en_conformer_ctc_att/config.yaml', 
        './conformer_ppg_model/en_conformer_ctc_att/24epoch.pth',
        'cuda'
    )
    wav_path = pathlib.Path('./dataset/wavs/marisa/marisa_11.wav')
    wav, _ = librosa.load(wav_path, sr=params.sampling_rate)
    wav = wav[:(wav.shape[0] // params.hop_size)*params.hop_size]
    ppgs = extract_ppg(ppg_model, [torch.tensor(wav).cuda()], params.sampling_rate)
    ppg = ppgs[0]
    pass

