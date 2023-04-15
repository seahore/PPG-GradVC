import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import wavfile
from tqdm import tqdm


def process(wav_name):
    # speaker 's5', 'p280', 'p315' are excluded,
    speaker = wav_name[:4]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '.flac' in wav_path:
        os.makedirs(os.path.join(args.out_dir, speaker), exist_ok=True)
        wav, _ = librosa.load(wav_path, sr=args.sr)
        wav, _ = librosa.effects.trim(wav, top_db=60)
        save_name = wav_name.replace(".flac", ".wav")
        save_path = os.path.join(args.out_dir, speaker, save_name)
        wavfile.write(
            save_path,
            args.sr,
            (wav * np.iinfo(np.int16).max).astype(np.int16)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("-i", "--in_dir", type=str, default="C:\\UnifiedDataset-subset\\wavs", help="path to source dir")
    parser.add_argument("-o", "--out_dir", type=str, default="C:\\UnifiedDataset-subset\\downsampled", help="path to target dir")
    args = parser.parse_args()

    pool = Pool(processes=cpu_count()-2)

    for speaker in os.listdir(args.in_dir):
        spk_dir = os.path.join(args.in_dir, speaker)
        if os.path.isdir(spk_dir):
            for _ in tqdm(pool.imap_unordered(process, os.listdir(spk_dir))):
                pass

