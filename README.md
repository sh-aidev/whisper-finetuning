# whisper-finetuning

<div align="center">

# Finetuning Whisper from Higgingface

[![python](https://img.shields.io/badge/-Python_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
![license](https://img.shields.io/badge/License-MIT-green?logo=mit&logoColor=white)
![huggingface](https://img.shields.io/badge/Huggingface-Transformers-ff0066?logo=huggingface&logoColor=yellow)

This repository contains the code to finetune the [Whisper](https://huggingface.co/transformers/model_doc/whisper.html) model from Huggingface.
</div>

## 📌 Feature
- [x] Finetuning a whisper model code in a production ready structured way
- [x] Training a whisper model on a custom dataset from huggingface audio dataset
- [x] finetuned on languages hindi

## 📁  Project Structure
The directory structure of new project looks like this:

```
├── checkpoints
├── configs
│   └── config.toml
├── data
├── LICENSE
├── logs
├── __main__.py
├── README.md
├── requirements.txt
└── src
    ├── app.py
    ├── core
    │   ├── data.py
    │   ├── __init__.py
    │   ├── training.py
    │   └── whisper.py
    ├── __init__.py
    └── utils
        ├── config.py
        ├── __init__.py
        ├── logger.py
        ├── models.py
        ├── textformat.py
        └── utils.py
```
## 🚀 Getting Started
### Step 1: Clone the repository
```bash
git clone https://github.com/sh-aidev/whisper-finetuning.git
cd whisper-finetuning
```

### Step 2: Install the required dependencies or open in devcontainer
```bash
pip install -r requirements.txt

# or

code .
# click on open in devcontainer option in vscode
# this will install all the required dependencies

```

### Step 3: Run the training script
```bash
python __main__.py

# To change the configuration, edit the config.toml file in the configs directory
```

## 📝  Training Results

### Before training these are the results of the model on the test dataset of common voice 9 in original whisper model

![before](./images/whisper_result.png)

<!-- ### After training on the common voice 17 dataset -->



## 📜  References
- [Huggingface](https://huggingface.co/)
- [Whisper](https://huggingface.co/transformers/model_doc/whisper.html)
- [Pytorch](https://pytorch.org/)