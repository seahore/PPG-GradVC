# PPG-Diff-VC

An any-to-many voice conversion model based on the architecture of [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS) and PPG from a [SSL-based phoneme recognizer](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft).

## Training

1. Prepare your multilingual corpora
2. A pretrained HiFi-GAN vocoder is located at `./hifigan/g_00875000`, you can continue from it or train a new one.
3. Extract PPGs

    ```bash
    python preprocess_ppg.py --sr 16000 --in_dir /your/dataset/flacs --out_dir /your/dataset/ppgs
    ```

4. Set parameters defined in `params.py`
5. Run `train_ppg.py`

## Inference

```bash
python inference.py -f infer_for_test.txt -c ./logs/grad_300.pt -t 100
```

## Acknowledgement

- This is NOT my original research. The code was mostly copied from works done by [Li Jingyi](https://github.com/OlaWod) et al. at [NERCMS, Wuhan Univ.](http://multimedia.whu.edu.cn/)