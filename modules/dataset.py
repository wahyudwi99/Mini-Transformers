import os
import json
import torch
import re
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader

class DatasetProcessor:
    def __init__(self, seq_len=256, batch_size=32, vocab_size=30000):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        
        # Setup directories using Pathlib
        self.base_dir = Path(__file__).resolve().parent.parent / "data"
        self.vocab_path = self.base_dir / "vocab.json"
        self.train_dir = self.base_dir / "train_data"
        self.test_dir = self.base_dir / "test_data"
        
        self.vocab = None
        self.reversed_vocab = None

    def tokenize(self, text):
        """Splits punctuation and text to keep vocab clean and compact."""
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def _create_vocab(self, data_generator):
        """Builds and saves vocabulary from the training generator."""
        print("Building vocabulary...")
        unique_words = set()
        for text in data_generator:
            unique_words.update(self.tokenize(text))
            if len(unique_words) > self.vocab_size:
                break
        
        # Special tokens: <EOS>=0, <UNK>=1
        self.vocab = {"<EOS>": 0, "<UNK>": 1}
        for idx, word in enumerate(sorted(list(unique_words)), start=2):
            self.vocab[word] = idx
            if idx == self.vocab_size - 1:
                break

        self.base_dir.mkdir(exist_ok=True)
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=4, ensure_ascii=False)
        
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        return self.vocab

    def load_vocab(self):
        """Loads vocabulary from disk and creates a reversed mapping."""
        with open(self.vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

        return self.vocab

    def decode(self, token_ids):
        """Decodes a list of token IDs back into a human-readable string."""
        if not self.reversed_vocab:
            self.load_vocab()

        return " ".join([self.reversed_vocab.get(tid, "<UNK>") for tid in token_ids])

    def _process_text_to_tensor(self, data_list):
        """Converts raw text into input and label tensors (Shifted by 1)."""
        all_tokens = []
        for text in data_list:
            tokens = self.tokenize(text)
            if not tokens: continue
            
            token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
            
            # Filter: only keep data where <UNK> rate is less than 30%
            if (token_ids.count(self.vocab["<UNK>"]) / len(tokens)) < 0.3:
                all_tokens.extend(token_ids)
                all_tokens.append(self.vocab["<EOS>"])

        all_tokens = torch.tensor(all_tokens)
        num_chunks = (len(all_tokens) - 1) // self.seq_len
        
        # Zero-copy slicing using .view()
        inputs = all_tokens[:num_chunks * self.seq_len].view(num_chunks, self.seq_len)
        labels = all_tokens[1 : num_chunks * self.seq_len + 1].view(num_chunks, self.seq_len)
        
        return inputs, labels

    def prepare_datasets(self):
        """Main pipeline to load, process, and return DataLoaders."""
        if not self.train_dir.exists() or not self.test_dir.exists():
            print("Processed data not found. Downloading and processing WikiText...")
            dataset = load_dataset("wikitext", "wikitext-103-v1")
            
            # Generators for memory efficiency
            train_gen = (d["text"] for d in dataset["train"] if len(d["text"]) > 10)
            test_gen = (d["text"] for d in dataset["test"] if len(d["text"]) > 10)
            
            self._create_vocab(train_gen)
            
            # Re-initialize generators as the previous ones were consumed
            train_gen = (d["text"] for d in dataset["train"] if len(d["text"]) > 10)
            
            train_in, train_lab = self._process_text_to_tensor(train_gen)
            test_in, test_lab = self._process_text_to_tensor(test_gen)
            
            # Save tensors
            for d, i, l in [(self.train_dir, train_in, train_lab), (self.test_dir, test_in, test_lab)]:
                d.mkdir(parents=True, exist_ok=True)
                torch.save(i, d / "input.pt")
                torch.save(l, d / "label.pt")
        else:
            print("Loading pre-processed tensors from disk...")
            self.load_vocab()
            train_in = torch.load(self.train_dir / "input.pt")
            train_lab = torch.load(self.train_dir / "label.pt")
            test_in = torch.load(self.test_dir / "input.pt")
            test_lab = torch.load(self.test_dir / "label.pt")

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(train_in, train_lab), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_in, test_lab), batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader, self.vocab

