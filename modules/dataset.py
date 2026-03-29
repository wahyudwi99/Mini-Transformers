import os
import json
import torch
import re
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader

def tokenize(text):
    # Split punctuation and text to keep vocab clean and compact
    return re.findall(r"\w+|[^\w\s]", text.lower())


def create_and_save_vocab(list_training_data):
    # Use a set for memory efficiency during unique word collection
    vocab = set()
    for text in list_training_data:
        vocab.update(tokenize(text))
    
    # Define special tokens: <EOS> (0) and <UNK> (1) for out-of-vocabulary words
    vocab_json = {"<EOS>": 0, "<UNK>": 1}
    for idx, word in enumerate(sorted(list(vocab)), start=2):
        vocab_json[word] = idx

    os.makedirs("data", exist_ok=True)
    with open("./data/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=4, ensure_ascii=False)

    return vocab_json


def prepare_data(
    list_data,
    vocab_json,
    seq_len,
    label
):  
    # Convert all text to token IDs (Flattened into a single list)
    # This is the "One-Time Write" phase
    all_tokens = []
    for text in list_data:
        tokens = tokenize(text)
        # Extend is more memory-efficient than nested appending
        all_tokens.extend([vocab_json.get(t, 1) for t in tokens])
        all_tokens.append(vocab_json["<EOS>"])

    # Create Chunks using Zero-Copy slicing (PyTorch View)
    all_tokens = torch.tensor(all_tokens)
    
    # Calculate how many full sequences we can create
    # (len - 1) ensures we have a trailing token for the label shift
    num_chunks = (len(all_tokens) - 1) // seq_len
    
    # Input data: [num_chunks, seq_len]
    input_data = all_tokens[:num_chunks * seq_len].view(num_chunks, seq_len)
    
    # Label data: Shifted by 1 position to the right (Target Prediction)
    label_data = all_tokens[1 : num_chunks * seq_len + 1].view(num_chunks, seq_len)

    if label == "train":
        os.makedirs("./data/train_data", exist_ok=True)
        torch.save(input_data, "./data/train_data/input_train_data.pt")
        torch.save(label_data, "./data/train_data/label_train_data.pt")
    else:
        os.makedirs("./data/test_data", exist_ok=True)
        torch.save(input_data, "./data/test_data/input_test_data.pt")
        torch.save(label_data, "./data/test_data/label_test_data.pt")

    return input_data, label_data


def dataset_preparation(batch_size):
    data_path = Path("./data")
    
    if not os.path.exists("./data/train_data") and not os.path.exists("./data/test_data"):
        print("Data not found. Downloading and processing...")
        wikitext = load_dataset("wikitext", "wikitext-103-v1")
        
        # Filter out empty strings or very short noise
        list_train = [data["text"] for data in wikitext["train"] if len(data["text"]) > 10]
        list_test = [data["text"] for data in wikitext["test"] if len(data["text"]) > 10]
        
        vocab_json = create_and_save_vocab(list_train)
        input_train_tensor, label_train_tensor = prepare_data(list_train, vocab_json, 256, "train")
        input_test_tensor, label_test_tensor = prepare_data(list_test, vocab_json, 256, "test")
    else:
        # Load pre-processed tensors directly from disk
        input_train_tensor = torch.load(data_path / "train_data/input_train_data.pt")
        label_train_tensor = torch.load(data_path / "train_data/label_train_data.pt")
        input_test_tensor = torch.load(data_path / "test_data/input_test_data.pt")
        label_test_tensor = torch.load(data_path / "test_data/label_test_data.pt")

    # Wrap in TensorDataset and DataLoader for batching and shuffling
    training_data = DataLoader(
        TensorDataset(input_train_tensor, label_train_tensor),
        batch_size,
        shuffle=True
    )
    testing_data = DataLoader(
        TensorDataset(input_test_tensor, label_test_tensor),
        batch_size,
        shuffle=True
    )

    return training_data, testing_data


if __name__ == "__main__":
    training_data, testing_data = dataset_preparation(batch_size=256)