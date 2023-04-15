import os
import argparse
import torch
import librosa
from glob import glob
from tqdm import tqdm

import utils
from transformers import Wav2Vec2ForCTC


def process(filename):
    basename = os.path.basename(filename)
    speaker = basename[:4]
    save_dir = os.path.join(args.out_dir, speaker)
    os.makedirs(save_dir, exist_ok=True)
    wav, _ = librosa.load(filename, sr=args.sr)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    with torch.no_grad():
        c = cmodel(wav).logits.transpose(1, 2) # size: (1,392,len)
        #print(c.size())
    save_name = os.path.join(save_dir, basename.replace(".flac", ".pt"))
    torch.save(c.cpu(), save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="C:\\GenshinSpeech\\wavs", help="path to input dir")
    parser.add_argument("--out_dir", type=str, default="C:\\GenshinSpeech\\w2v2-ppgs", help="path to output dir")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading W2V2 for content...")
    cmodel = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").cuda()
    cmodel.eval()
    
    filenames = glob(f'{args.in_dir}/*/*.flac', recursive=True)
    
    for filename in tqdm(filenames):
        process(filename)
    