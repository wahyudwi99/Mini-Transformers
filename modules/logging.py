import os
import json
from datetime import datetime



def log_metrics_results(
    epoch,
    train_accuracy,
    train_loss,
    test_accuracy,
    test_loss,
    prediction_string,
    label_string
):
    # Compile eval netrics data
    json_data = {
        "epoch": int(epoch),
        "train_accuracy": train_accuracy,
        "train_loss": train_loss,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "prediction_string": prediction_string,
        "label_string": label_string
    }

    os.makedirs("./logs", exist_ok=True)

    with open("logs.jsonl", "a") as f:
        f.write(json_data + "\n")