import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_results(log_path):
    """
    Reads a .jsonl log file and plots Training/Test Loss and Accuracy.
    """
    epochs = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    # 1. Read JSONL file
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Error: File {log_path} not found.")
        return

    print(f"Reading logs from {log_path}...")
    with open(log_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data['epoch'])
            train_loss.append(data['train_loss'])
            test_loss.append(data['test_loss'])
            train_acc.append(data['train_accuracy'])
            test_acc.append(data['test_accuracy'])

    # 2. Create Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Plot Loss ---
    ax1.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linestyle='-', marker='o', markersize=4)
    ax1.plot(epochs, test_loss, label='Test Loss', color='#ff7f0e', linestyle='--', marker='s', markersize=4)
    ax1.set_title('Training & Test Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()

    # --- Plot Accuracy ---
    ax2.plot(epochs, train_acc, label='Train Acc', color='#2ca02c', linestyle='-', marker='o', markersize=4)
    ax2.plot(epochs, test_acc, label='Test Acc', color='#d62728', linestyle='--', marker='s', markersize=4)
    ax2.set_title('Training & Test Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    
    # 3. Save and Show
    output_image = log_file.parent / "training_plots.png"
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved successfully at: {output_image}")
    plt.show()

if __name__ == "__main__":
    # Ganti path ini sesuai dengan lokasi file metrics.jsonl kamu
    # Contoh: "logs/log__1/metrics.jsonl"
    LOG_PATH = "logs/log__1/metrics.jsonl" 
    
    plot_training_results(LOG_PATH)