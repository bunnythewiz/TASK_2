# Scanned Receipt Text Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.1-red.svg)](https://pytorch.org/)

Deep learning-based text recognition system for scanned receipts using the [Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) framework. This project implements and analyzes the state-of-the-art TPS-ResNet-BiLSTM-Attention architecture for the ICDAR2019 SROIE competition.

## 🎯 Overview

This repository contains our implementation and analysis of **Task 2: Scanned Receipt OCR** from the ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction (SROIE). The goal is to accurately recognize all text in receipt images without requiring localization information.

### Key Features
- ✅ TPS-ResNet-BiLSTM-Attention architecture
- ✅ Virtual Adversarial Training (VAT) for domain adaptation
- ✅ Comprehensive ablation studies
- ✅ Error analysis and visualization tools
- ✅ Pre-trained models available

## 📝 Task Description

**Input:** Scanned receipt image (RGB or grayscale)  
**Output:** List of recognized words (space-tokenized)  
**Evaluation:** Precision, Recall, and F1-score on exact word matches

### Tokenization Rules
- Split on whitespace
- Punctuation attached to adjacent words
- Case-sensitive matching

**Examples:**
"Date: 12/3/56"      → ["Date:", "12/3/56"]
"TOTAL RM25.50"      → ["TOTAL", "RM25.50"]
"Date: 12 / 3 / 56"  → ["Date:", "12", "/", "3", "/", "56"]

## 🏗️ Architecture

Our system uses a four-stage modular architecture:

Input Image → [TPS] → [ResNet+SENet] → [BiLSTM] → [Attention] → Output Text

### Components

1. **Transformation (TPS)**
   - Thin-Plate Spline transformation
   - Normalizes geometric distortions
   - Handles curved and skewed text

2. **Feature Extraction (ResNet + SENet)**
   - ResNet-50 backbone with residual connections
   - Squeeze-and-Excitation blocks for channel attention
   - Extracts hierarchical visual features

3. **Sequence Modeling (BiLSTM)**
   - 2-layer bidirectional LSTM
   - Captures contextual dependencies
   - Processes features in both directions

4. **Prediction (Attention)**
   - Attention-based decoder
   - Learns flexible character-to-feature alignment
   - Generates character sequence output

### Training Strategy
- **Pretraining:** Synthetic datasets (MJSynth + SynthText)
- **Fine-tuning:** SROIE training set with VAT regularization
- **Augmentation:** Rotation, blur, brightness, elastic deformation

## 📊 Dataset

**Characteristics:**
- Scanned receipt images from retail stores
- Variable image quality (faded, blurred, distorted)
- Multiple languages (primarily English)
- Dense text with mixed fonts and sizes

**Download:** [ICDAR2019 SROIE](https://rrc.cvc.uab.es/?ch=13)

## 🚀 Installation

### Prerequisites
Python >= 3.8
CUDA >= 11.1 (for GPU support)

### Setup Environment

## Clone repository
```
git clone https://github.com/yourusername/sroie-task2.git
cd sroie-task2
```

## Create virtual environment
```
conda create -n sroie_ocr python=3.8 -y
conda activate sroie_ocr
```
## Install dependencies
```
pip install -r requirements.txt
```
### Requirements
torch==1.8.1
torchvision==0.9.1
torchtext==0.6.0
numpy==1.19.5
pandas==1.1.5
opencv-python==4.5.3
Pillow==8.3.1
tqdm==4.62.0
tensorboard==2.6.0
allennlp==1.0.0
nltk==3.6.5

## 💻 Usage

### Training

## Train from scratch
```
python train.py --config configs/TPS-ResNet-BiLSTM-Attn.yaml
```
## Fine-tune pretrained model
```
python train.py --config configs/finetune.yaml --pretrained checkpoints/pretrained_model.pth
```
## Train with VAT
```
python train.py --config configs/finetune_vat.yaml --vat_epsilon 1.0
```
### Inference

## Single image
```
python predict.py --image path/to/receipt.jpg --checkpoint checkpoints/best_model.pth
```
## Batch prediction
```
python predict.py --image_dir data/test_images/ --checkpoint checkpoints/best_model.pth --output results.txt
```
### Evaluation
```
python evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/test/ --ground_truth data/test_gt.txt
```

## 📁 Project Structure
```
sroie-task2/
├── README.md
├── requirements.txt
├── configs/
│   ├── TPS-ResNet-BiLSTM-Attn.yaml
│   ├── finetune.yaml
│   └── finetune_vat.yaml
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── transformation.py
│   ├── feature_extraction.py
│   ├── sequence_modeling.py
│   └── prediction.py
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   └── visualization.py
├── train.py
├── predict.py
└── evaluate.py
```
