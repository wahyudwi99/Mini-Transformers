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


def load_vocab_file():
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "data/vocab.json"), "r") as f:
        vocab_json = json.loads(f.read())

    return vocab_json


def decode_vocab(
    prediction,
    ground_truth,
    reversed_vocab
):
    # Get only 5 last predictions
    ground_truth_string = "".join([reversed_vocab[id] for id in ground_truth])
    prediction_string = "".join([reversed_vocab[id] for id in prediction])

    return ground_truth_string, prediction_string


def create_and_save_vocab(list_training_data, num_vocab, wikitext):
    # Use a set for memory efficiency during unique word collection
    vocab = set()
    for text in list_training_data:
        vocab.update(tokenize(text))
        if len(vocab) > num_vocab:
            break
    
    # Define special tokens: <EOS> (0) and <UNK> (1) for out-of-vocabulary words
    vocab_json = {"<EOS>": 0, "<UNK>": 1}
    for idx, word in enumerate(sorted(list(vocab)), start=2):
        vocab_json[word] = idx

    os.makedirs("data", exist_ok=True)
    with open("./data/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=4, ensure_ascii=False)


    # Create generator again
    list_train = (data["text"] for data in wikitext["train"] if len(data["text"]) > 10)

    return vocab_json, list_train


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
        token_ids = [vocab_json.get(token, 1) for token in tokens]
        unknown_token_counts = token_ids.count(1)

        if unknown_token_counts/len(tokens) < 0.3:
            all_tokens.extend(token_ids)
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

    input_data = input_data[:6000]
    label_data = label_data[:6000]

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
    current_dir_path = os.path.dirname(__file__)
    train_dir_path = os.path.join(current_dir_path, "data/train_data")
    test_dir_path = os.path.join(current_dir_path, "data/test_data")
    if not os.path.exists(train_dir_path) and not os.path.exists(test_dir_path):
        print("Data not found. Downloading and processing...")
        wikitext = load_dataset("wikitext", "wikitext-103-v1")
        
        # Filter out empty strings or very short noise
        list_train = (data["text"] for data in wikitext["train"] if len(data["text"]) > 10)
        list_test = (data["text"] for data in wikitext["test"] if len(data["text"]) > 10)
        
        vocab_json, list_train = create_and_save_vocab(list_train, 30000, wikitext)
        input_train_tensor, label_train_tensor = prepare_data(list_train, vocab_json, 256, "train")
        input_test_tensor, label_test_tensor = prepare_data(list_test, vocab_json, 256, "test")
    else:
        vocab_json = load_vocab_file()
        # Load pre-processed tensors directly from disk
        
        input_train_tensor = torch.load(os.path.join(train_dir_path, "input_train_data.pt"))
        label_train_tensor = torch.load(os.path.join(train_dir_path, "label_train_data.pt"))
        input_test_tensor = torch.load(os.path.join(test_dir_path, "input_test_data.pt"))
        label_test_tensor = torch.load(os.path.join(test_dir_path, "label_test_data.pt"))

        input_train_tensor = input_train_tensor[:500]
        label_train_tensor = label_train_tensor[:500]
        input_test_tensor = input_test_tensor[:100]
        label_test_tensor = label_test_tensor[:100]


    print(input_train_tensor.size())
    print(label_train_tensor.size())
    print(input_test_tensor.size())
    print(label_test_tensor.size())
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

    return training_data, testing_data, vocab_json