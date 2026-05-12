# ASR Consilium

A repository for Automatic Speech Recognition (ASR) that ensembles multiple open-source models to achieve SOTA quality of recognition. Useful if you need to get the maximum quality of recognition despite the computational time.


## Usage
```
pip install asr-consilium 
```
or copy folder `asr_consilium` into your project. After:

```python
from asr_consilium import inference as asr_inference

jsonl_dataset = 'markdown.jsonl' # Dataset in jsonl format
out_file = 'result.jsonl' # File where results will be stored in JSONL format
asr_inference(
    jsonl_file=jsonl_dataset, # input dataset for processing
    out_file=out_file, # Outputs file
    batch_size=16, # Batch size
    model_list=None, # Models for ensemble - see usage below
    weights=None, # Weights for models
    language='English', # Language
    normalize=True, # Normalize texts before ensemble
    char_level=False, # ensemble on char or word level
    ensemble_type='median_extended', # Type of ensemble ('greedy', 'median', 'median_extended')
    skip_existed=True, # if some model already calculated the results they will not recalculate
)
```

Or with command line:

```bash
python3 inference.py --input_data "samples/markdown.jsonl" --output "result.jsonl" --batch_size 16
```

## Supported models

| Model name                                                                                  | Params (B) | Languages | Average WER (English) |
|---------------------------------------------------------------------------------------------|----------|-----------|-----------------------|
| [Nvidia Parakeet v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)                    | 0.6      | en        | 6.05                  |
| [Nvidia Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)                    | 0.6      | 26        | 6.32                  |
| [Nvidia Parakeet tdt1.1](https://huggingface.co/nvidia/parakeet-tdt-1.1b)                   | 1.1      | en        | 7.02                  |
| [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)                                | 0.6      | 30        | 6.42                  |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)                                | 1.7      | 52        | 5.76                  |
| [Canary Qwen](https://huggingface.co/nvidia/canary-qwen-2.5b)                               | 2.5      | en        | 5.63                  |
| [IBM Granite Speech 4.0-1B](https://huggingface.co/ibm-granite/granite-4.0-1b-speech)       | 1.0      | 6         | 5.52                  |
| [IBM Granite Speech 3.3-8B](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)       | 8.0      | 5         | 5.74                  |
| [Cohere Transcribe (03-2026)](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)* | 2.0      | 14        | 5.42                  |
| [Granite-Speech-4.1-2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)           | 2.0      | 6         | 5.33                  | 
| [Microsoft-Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)               | 5.6      | 8         | 6.02                  | 


## Format of markdown.jsonl

`markdown.jsonl` must contain 2 fields: 
* 'audio' - name of audio file in wav format and 16000 Hz sample rate. It must be in the same folder with markdown.jsonl file. 
* 'duration' - duration of audio in seconds (needed for sorting and for tokens approximation)
* 'text' (optional) - needed for validation. If you have real speech text. Code will automatically calculate WER and CER metrics for your data. 

```json lines
...
{"audio": "483_0.wav", "duration": 2.079875}
{"audio": "461_1.wav", "duration": 9.6599375}
{"audio": "243_2.wav", "duration": 7.3400625}
...
```

You can find sample [here](samples/markdown.jsonl)

## Choose models for ensembling

List of models and their weights are chosen with parameters `model_list` and `weights`. You may leave it `None`. So default list of models and default weights will be used.

### Default models and weights

```python
model_list = [
    'nvidia/parakeet-tdt-0.6b-v2',
    'nvidia/parakeet-tdt-0.6b-v3',
    'Qwen/Qwen3-ASR-1.7B',
    'nvidia/canary-qwen-2.5b',
    'ibm-granite/granite-speech-3.3-8b',
    'ibm-granite/granite-4.0-1b-speech',
    'ibm-granite/granite-speech-4.1-2b',
    'ZFTurbo/Phi-4-multimodal-instruct',
]

weights = [4.5, 4.2, 8.4, 9.8, 8.7, 3.5, 8.9, 9.4]
```

## Results for different datasets

Note: WER/CER metric calculation slightly different from Open ASR Leaderboard

### WER table (English language)
| Model name                                                                            | avg  | [AMI](https://huggingface.co/datasets/edinburghcstr/ami)   | [Earnings22](https://huggingface.co/datasets/distil-whisper/earnings22) | [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) | [LibriSpeech (clean)](https://huggingface.co/datasets/openslr/librispeech_asr) | [LibriSpeech (other)](https://huggingface.co/datasets/openslr/librispeech_asr) | [SPGISpeech](https://huggingface.co/datasets/kensho/spgispeech) | [TED-LIUM v3](https://arxiv.org/abs/1805.04699) | [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) |
|---------------------------------------------------------------------------------------|------|-------|-------|-------|------|--------------------------------------------------------------------------------|------|------|------|
| [Nvidia Parakeet v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)              | 6.09 | 11.27 | 11.28 | 9.78  | 1.70 | 3.19                                                                           | 2.14 | 3.42 | 5.94 |
| [Nvidia Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)              | 6.36 | 11.59 | 11.29 | 9.59  | 1.92 | 3.60                                                                           | 3.99 | 2.82 | 6.10 |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)                          | 5.82 | 11.16 | 10.25 | 8.74  | 1.62 | 3.38                                                                           | 2.84 | 2.30 | 6.33 |
| [Canary Qwen 2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b)                    | 5.57 | 10.17 | 10.34 | 9.24  | 1.62 | 3.12                                                                           | 1.92 | 2.58 | 5.62 |
| [IBM Granite Speech 4.0-1B](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | 5.67 | 8.58  | 8.87  | 10.37 | 1.44 | 2.91                                                                           | 4.15 | 3.14 | 5.97 |
| [IBM Granite Speech 3.3-8B](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) | 5.81 | 8.83  | 10.12 | 10.21 | 1.43 | 2.90                                                                           | 3.86 | 3.40 | 5.73 |
| [Granite-Speech-4.1-2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)     | 5.43 | 8.16  | 8.56  | 9.98  | 1.29 | 2.55                                                                           | 4.10 | 3.04 | 5.79 |
| [Microsoft-Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)         | 6.01 | 11.22 | 10.28 | 9.30  | 1.68 | 3.94                                                                           | 2.87 | 2.88 | 5.98 |
| Ensemble (Default parameters)                                                         | 4.66 | 7.32  | 7.77  | 8.67  | 1.15 | 2.24                                                                           | 2.68 | 2.21 | 5.24 |

### CER Table (English language)
| Model name                                                                            | avg | [AMI](https://huggingface.co/datasets/edinburghcstr/ami)   | [Earnings22](https://huggingface.co/datasets/distil-whisper/earnings22) | [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) | [LibriSpeech (clean)](https://huggingface.co/datasets/openslr/librispeech_asr) | [LibriSpeech (other)](https://huggingface.co/datasets/openslr/librispeech_asr) | [SPGISpeech](https://huggingface.co/datasets/kensho/spgispeech) | [TED-LIUM v3](https://arxiv.org/abs/1805.04699) | [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) |
|---------------------------------------------------------------------------------------|-----|------|------|------|------|------|------|------|------|
| [Nvidia Parakeet v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)              | 3.51 | 7.18 | 7.67 | 5.15 | 0.50 | 1.15 | 0.97 | 2.03 | 3.50 |
| [Nvidia Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)              | 3.63 | 7.48 | 7.47 | 4.94 | 0.59 | 1.33 | 2.06 | 1.54 | 3.63 |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)                          | 3.25 | 6.50 | 6.89 | 4.47 | 0.50 | 1.29 | 1.48 | 1.19 | 3.75 |
| [Canary Qwen 2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b)                    | 3.10 | 5.77 | 6.95 | 4.70 | 0.51 | 1.14 | 0.89 | 1.47 | 3.40 |
| [IBM Granite Speech 4.0-1B](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | 3.17 | 5.12 | 5.55 | 5.56 | 0.45 | 1.15 | 2.19 | 1.76 | 3.60 |
| [IBM Granite Speech 3.3-8B](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) | 3.23 | 5.33 | 6.15 | 5.37 | 0.44 | 1.12 | 2.04 | 1.93 | 3.46 |
| [Granite-Speech-4.1-2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)     | 3.05 | 4.85 | 5.40 | 5.33 | 0.43 | 0.98 | 2.24 | 1.72 | 3.47 |
| [Microsoft-Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)         | 3.38 | 6.89 | 6.69 | 4.78 | 0.54 | 1.65 | 1.43 | 1.45 | 3.66 |
| Ensemble (Default parameters)                                                         | 2.61 | 4.43 | 5.05 | 4.45 | 0.35 | 0.86 | 1.41 | 1.17 | 3.18 |
