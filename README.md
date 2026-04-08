# Mini-Transformers

A custom implementation of a Decoder-only Transformer architecture designed for language modeling tasks. This project focuses on building a streamlined, efficient, and well-documented transformer from scratch using PyTorch, featuring custom modules for attention, datasets, and evaluation metrics.

## 🚀 Training Flow Overview

The project follows a systematic pipeline from raw text processing to a trained model with comprehensive evaluation.

### 1. Data Processing (`modules/dataset.py`)
- **Dataset**: Uses the `WikiText-103` dataset through the Hugging Face `datasets` library.
- **Tokenization**: Implements a custom regex-based tokenizer that handles punctuation and lowercase mapping for a clean vocabulary.
- **Vocabulary Building**: Dynamically builds a vocabulary from the training set with special tokens (`<EOS>`, `<UNK>`).
- **Tensor Conversion**: Converts text into input and target tensors, shifted by one for next-token prediction, with logic to filter out sequences with high unknown token rates.

### 2. Model Architecture (`modules/transformers.py` & `modules/models.py`)
The model is a stacked Transformer architecture consisting of:
- **Token Embedding**: Maps token IDs to high-dimensional dense vectors.
- **Multi-Head Causal Attention**: Processes tokens with multiple attention heads and implements causal masking to ensure tokens only attend to previous tokens.
- **Feed-Forward Network (FFN)**: Uses a two-layer linear network with GELU activation to capture non-linear relationships.
- **Residual Connections & Layer Normalization**: Ensures stable training and information flow through the stacked blocks.

### 3. Training Loop (`modules/train.py`)
- **Trainer Class**: Orchestrates the entire lifecycle:
  - **Setup**: Initialized with hyperparameters like `d_model`, `num_heads`, and `seq_length`.
  - **Epoch Cycle**: Performs training and evaluation on each epoch.
  - **Optimization**: Uses `nn.CrossEntropyLoss` and the `Adam` optimizer.
  - **Pipeline**: Automates the sequence of training, testing, logging, and checkpointing.

### 4. Evaluation & Metrics (`modules/metrics.py`)
The project implements two primary evaluation methods:
- **Per-token Accuracy**: Measures how often the model correctly predicts the next token in the sequence.
- **MMLU (Massive Multitask Language Understanding)**: A benchmark metric that evaluates the model's ability to pick the correct answer (A, B, C, or D) from multiple-choice questions by comparing the logits of these specific tokens.

### 5. Logging & Visualization
- **Logging (`modules/logging.py`)**: Records training loss and accuracy into JSONL files and saves model checkpoints periodically.
- **Visualization (`modules/visualization.py`)**: Generates plots showing the progress of loss and accuracy over epochs to help monitor training health.

## 🛠️ Usage

To start the training pipeline with default hyperparameters, run:

```bash
python -m modules.train
```

## 📊 Project Structure

- `modules/train.py`: Main entry point for training.
- `modules/transformers.py`: High-level Transformer block assembly.
- `modules/models.py`: Core components (Attention, FFN, Embeddings).
- `modules/dataset.py`: Data loading and processing logic.
- `modules/metrics.py`: Custom evaluation functions.
- `modules/logging.py`: Metadata and model state persistence.
- `modules/visualization.py`: Metric plotting suite.
