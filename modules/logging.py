import json
from pathlib import Path
from safetensors.torch import save_file

class Logger:
    def __init__(self, model_config):
        # 1. Define root directory using Pathlib
        self.root_dir = Path(__file__).resolve().parent.parent / "logs"
        self.root_dir.mkdir(exist_ok=True)
        
        # 2. Create session folder (Executed only once at initialization)
        session_id = len(list(self.root_dir.glob("log__*"))) + 1
        self.session_dir = self.root_dir / f"log__{session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 3. Dedicated directory for model checkpoints
        self.ckpt_dir = self.session_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        
        # 4. Save configuration at the start (Fail-fast principle)
        self._save_config(model_config)

    def _save_config(self, config):
        with open(self.session_dir / "model_config.json", "w") as f:
            json.dump(config, f, indent=4)

    def save_metrics(self, logs_data):
        # Use JSONL format to ensure data persistence if the training crashes
        log_file = self.session_dir / "metrics.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(logs_data) + "\n")

    def save_checkpoint(self, model_state_dict, epoch):
        # Save checkpoints with consistent naming convention
        file_path = self.ckpt_dir / f"model_epoch_{epoch}.safetensors"
        save_file(model_state_dict, str(file_path))