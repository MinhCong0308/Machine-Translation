# Machine Translation: English to Vietnamese

A from-scratch implementation of a Transformer-based sequence-to-sequence model for translating between Vietnamese and English, built and evaluated in a Kaggle notebook environment.

---

## 📖 Table of Contents

1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Preprocessing](#preprocessing)  
4. [Model Architecture](#model-architecture)  
5. [Training](#training)  
6. [Inference](#inference)  
7. [Evaluation](#evaluation)  
8. [Results](#results)  
9. [Requirements](#requirements)  
10. [Usage](#usage)  
11. [References](#references)  

---

## Introduction

This project demonstrates how to build a Transformer-based neural machine translation (NMT) system from scratch in PyTorch—starting from data loading and tokenization with SentencePiece, through defining multi-head attention and positional encoding modules, to training and evaluating on a Vietnamese–English parallel corpus.

Key goals:

- Understand and implement the core components of the original Transformer architecture.  
- Measure translation quality and analyze performance.

---

## Dataset

We use dataset that be provided in HuggingFace (https://huggingface.co/datasets/ncduy/mt-en-vi), which contains Vietnamese–English sentence pairs divided into:

- **train.csv** (~×,000 sentence pairs)  
- **valid.csv** (~×,000 sentence pairs)  
- **test.csv** (~×,000 sentence pairs)  

Each CSV file has the columns:

| id    | source    | vi_text                  | en_text            |
|-------|-----------|--------------------------|--------------------|
| 00001 | kaggle    | “Xin chào thế giới”      | “Hello world”      |
| …     | …         | …                        | …                  |

---

## Preprocessing

1. **Loading**: Read CSVs with pandas.  
2. **Cleaning**: Remove redundant columns (e.g., `source`).  
3. **Tokenization**:  
   - Train a SentencePiece model (`.model` + `.vocab`) on the combined training set.  
   - Encode both Vietnamese and English sentences to integer IDs.  
4. **Dataset & DataLoader**:  
   - Custom `TranslationDataset` to yield source/target tensors with padding/truncation to fixed length (e.g., 20 tokens).  
   - PyTorch `DataLoader` for batching and shuffling.

---

## Model Architecture

All modules implemented from the ground up using `torch.nn`:

- **EmbeddingForTransformer**  
- **PositionalEncoding**  
- **MultiheadAttention**  
- **PositionWiseFeedForward**  
- **TransformerEncoder** & **TransformerDecoder**  
- **TransformerArchitecture** (wraps encoder + decoder + final linear + softmax)

Configuration highlights:

- `num_layers=6`  
- `embed_dim=512`  
- `num_heads=8`  
- `ffn_dim=2048`  
- Maximum sequence length = 20  
- Dropout = 0.1

---

## Training

- **Loss**: CrossEntropy (ignore `<pad>` tokens)  
- **Optimizer**: Adam with learning-rate scheduling (“Noam” warm-up)  
- **Batch size**: 64  
- **Epochs**: 20 (or until convergence on validation loss)  
- **Logging**: Training/validation loss per epoch and sample translations.

---

## Inference

- Greedy decoding by feeding the `<sos>` token and iteratively sampling the highest-probability next token until `<eos>` or length limit.
- Example usage in the notebook:
  ```python
  def translate_sentence(model, sp_src, sentence, max_len=20):
      # tokenize, encode, run through model.eval(), decode IDs
      return translated_text
