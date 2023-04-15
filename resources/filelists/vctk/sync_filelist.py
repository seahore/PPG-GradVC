import os
from glob import glob


available_wavs = glob(f"../../../DUMMY2/*/*.wav")

wavset = set()
for wav in available_wavs:
    basename = os.path.basename(wav)
    wavset.add(basename)
print(wavset)

with open("vctk_audio_sid_text_test_filelist.txt", "r") as f:
    lines = f.readlines()
with open("test.txt", "w") as f:
    for line in lines:
        path, _, _ = line.strip().split("|")
        basename = os.path.basename(path)
        if basename in wavset:
            f.write(line)

with open("vctk_audio_sid_text_val_filelist.txt", "r") as f:
    lines = f.readlines()
with open("val.txt", "w") as f:
    for line in lines:
        path, _, _ = line.strip().split("|")
        basename = os.path.basename(path)
        if basename in wavset:
            f.write(line)

with open("vctk_audio_sid_text_train_filelist.txt", "r") as f:
    lines = f.readlines()
with open("train.txt", "w") as f:
    for line in lines:
        path, _, _ = line.strip().split("|")
        basename = os.path.basename(path)
        #print(basename)
        if basename in wavset:
            f.write(line)
