# CleanMel
[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.20040)
[![Demos](https://img.shields.io/badge/🎧-Demos-blue)](https://audio.westlake.edu.cn/Research/CleanMel.html)
[![Try CleanMel](https://img.shields.io/badge/Open%20Demo-Click%20Here-blue)](https://huggingface.co/spaces/SaoYear/CleanMel)
[![GitHub Issues](https://img.shields.io/github/issues/Audio-WestlakeU/CleanMel)](https://github.com/Audio-WestlakeU/CleanMel/issues)
[![Contact](https://img.shields.io/badge/💌-Contact-purple)](https://saoyear.github.io)

PyTorch implementation of "CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR" [accepted by IEEE Trans. ASLPRO (TASLP), 2025].

## For our extension of CleanMel over Hebrew Datasets and SI-SNR metric, scroll to the bottom of the ReadMe file.

## Notice 📢
- The CleanMel model checkpoints are now available on huggingface, the inference can be done using one-line commands.
- All models are available in `pretrained/enhancement/` folder.
- The enhanced results from 4 `offline_CleanMel_S/L_mask/map` models for the CHIME example `noisy_CHIME-real_F05_442C020S_STR_REAL` are given in `src/inference_example/pretrained_example_output` folder.

## Overview 🚀
<p align="center"><img src="./src/imgs/cleanmel_arch.png"alt="jpg name" width="60%"/></p>

**CleanMel** enhances logMel spectrograms for improved speech quality and ASR performance. Outputs compatible with:
- 🎙️ Vocoders for enhanced waveforms
- 🤖 ASR systems for transcription

## Demo Page 🎤
The demo page of CleanMel is published on [Hugging Face Spaces](https://huggingface.co/spaces/SaoYear/CleanMel).

If you downloaded the pretrained models (follwing [instructions](https://github.com/Audio-WestlakeU/CleanMel/tree/main/pretrained)), you can also activate this demo page locally by running the following command:
```bash
python app.py
```
Then, open your browser and visit `http://localhost:7860` to access the demo page.

## Quick Start ⚡

### Environment Setup
```bash
conda create -n CleanMel python=3.10.14
conda activate CleanMel
pip install -r requirements.txt
```

### Inference
Pretrained models can be downloaded manually [here](https://huggingface.co/WestlakeAudioLab/CleanMel), or automatically with the help of `huggingface-hub` package.

```bash 
# Inference with pretrained models from huggingface
## Offline example (offline_CleanMel_S_mask)
cd shell
bash inference.sh 0, offline S mask huggingface

## Online example (online_CleanMel_S_map)
bash inference.sh 0, online S map huggingface

# Inference with local pretrained models
cd shell
bash inference.sh 0, offline S mask

## Online example (online_CleanMel_S_map)
bash inference.sh 0, online S map
```
**Custom Input**: Modify `speech_folder` in `inference.sh`

**Output**: Results saved to `output_folder` (default to `./my_output`)

### Training
```bash
# Offline training example (offline_CleanMel_S_mask)
cd shell
bash train.sh 0,1,2,3 offline S mask
```
Configure datasets in `./config/dataset/train.yaml`

Default 4 GPUs trained with batch size 32

## Pretrained Models 🧠
```bash
pretrained/
├── enhancement/
│   ├── offline_CleanMel_S_map.ckpt
│   ├── offline_CleanMel_S_mask.ckpt
│   ├── online_CleanMel_S_map.ckpt
|   └── ...
└── vocos/
    ├── vocos_offline.pt
    └── vocos_online.pt
```
**Enhancement**: `offline_CleanMel_S/L_mask/map.ckpt` are available.

**Vocos**: `vocos_offline.pt` and `vocos_online.pt` are [here](https://drive.google.com/file/d/13Q0995DmOLMQWP-8MkUUV9bJtUywBzCy/view?usp=drive_link).

## Performance 📊
### Speech Enhancement
<p align="center"><img src="./src/imgs/dnsmos_performance.png" alt="jpg name" width="70%"/></p>
<p align="center"><img src="./src/imgs/pesq_performance.png" alt="jpg name" width="40%"/></p>

### ASR Accuracy
<p align="center"><img src="./src/imgs/asr_performance.png" alt="png name" width="40%"/></p>

💡 ASR implementation details in [`asr_infer` branch](https://github.com/Audio-WestlakeU/CleanMel/tree/asr_infer)

## Citation 📝
```bibtex
@ARTICLE{11097896,
  author={Shao, Nian and Zhou, Rui and Wang, Pengyu and Li, Xian and Fang, Ying and Yang, Yujie and Li, Xiaofei},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR}, 
  year={2025},
  volume={33},
  pages={3202-3214},
  doi={10.1109/TASLPRO.2025.3592333}}
}
```

## Acknowledgement 🙏
- Built using [NBSS](https://github.com/Audio-WestlakeU/NBSS) template
- Vocoder implementation from [Vocos](https://github.com/gemelo-ai/vocos)



## CleanMel Extension - Our addition of Hebrew Fine Tuning and SI-SNR loss
1. The datasets used for Hebrew Fine tuning are - Hebrew (Common Voice Scripted Speech 24.0 - Hebrew)
2. Noising Datasets used are - Noise Dataset (DNS challenge)
3. RIR Dataset (openslr)
   
**To obtain the datasets run the following py files under setup dir:**
1. get_Hebrew_dataset.py
2. get_noise_dataset.py
3. get_rir_dataset.py
4. get_rir_csvs.py
5. preprocess_Hebrew_audio.py
6. reconfigure_datasets.py

To fine tune the model run either:
fine_tune_hebrew_mask.py or fine_tune_hebrew_sisnr.py
under the src dir. 
Note that the yaml configuration file points to the correct location of the datasets.

To test the fine tuned models run either:
test_model_fine_tuned_hebrew_mask.py or test_model_fine_tuned_hebrew_sisnr.py

Ensure first that the arch checkpoints exist under:
"./logs/finetune_hebrew_mask/version_0/checkpoints/last.ckpt"
"./logs/finetune_hebrew_sisnr/version_0/checkpoints/last.ckpt"

Finally, run the show_mel.py file, under src to compare the mel spectogram of the signal before and after fine tuning the model.

