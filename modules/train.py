import torch
import torch.nn as nn
import modules.transformers as transformers
import modules.dataset as dataset
import modules.metrics as metrics
import modules.logging as logging
import modules.visualization as visualization
from tqdm import tqdm
from datetime import datetime



class Trainer():
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_hidden,
        transformers_blocks,
        seq_length,
        batch_size,
        vocab_size
        ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dataset = dataset.DatasetProcessor(
            seq_length,
            batch_size,
            vocab_size
        )
        self.training_data, self.testing_data, self.vocab_json = self._dataset.prepare_datasets()
        self.reversed_vocab = {v: k for k, v in self.vocab_json.items()}
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = transformers.MiniTransformers(
            d_model,
            num_heads,
            ffn_hidden,
            vocab_size,
            transformers_blocks,
            self.device
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self._logger = logging.Logger(
            model_config={
                "d_model": d_model,
                "num_heads": num_heads,
                "ffn_hidden": ffn_hidden,
                "vocab_size": vocab_size,
                "transformers_blocks": transformers_blocks
            }
        )


    def train(self):
        self.model.train()
        train_acc = 0
        train_loss = 0
        for x_batch, y_batch in tqdm(self.training_data):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Make a prediction
            prediction = self.model.forward(x_batch)

            # Calculate loss result
            prediction_val_resiized = prediction.view(-1, prediction.size()[-1])
            y_batch_resized = y_batch.view(-1)
            loss_training = self.loss_fn(prediction_val_resiized, y_batch_resized)
            train_loss += loss_training.item()

            # Back propagation
            self.optimizer.zero_grad()
            loss_training.backward()
            self.optimizer.step()

            # Calculate metrics
            accuracy_train = metrics.calculate_accuracy(
                prediction.detach().cpu(),
                y_batch.detach().cpu()
            )
            train_acc += accuracy_train
        
        # Averaging metrics results
        train_acc = train_acc / len(self.training_data)
        train_loss = train_loss / len(self.training_data)

        return train_acc, train_loss
    

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_acc = 0
            test_loss = 0
            for x_batch, y_batch in self.testing_data:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Make a prediction
                prediction = self.model.forward(x_batch)

                # Loss result
                prediction_val_resiized = prediction.view(-1, prediction.size()[-1])
                y_batch_resized = y_batch.view(-1)
                loss_testing = self.loss_fn(prediction_val_resiized, y_batch_resized)
                test_loss += loss_testing.item()
            
                # Calculate metrics
                accuracy_test = metrics.calculate_accuracy(
                    prediction.detach().cpu(),
                    y_batch.detach().cpu()
                )
                test_acc += accuracy_test

            test_acc = test_acc / len(self.testing_data)
            test_loss = test_loss / len(self.testing_data)

            # Get last sentence prediction and ground truth
            prediction_last = torch.argmax(prediction[-1], dim=1).detach().cpu().numpy().tolist()
            ground_truth_last = y_batch[-1].detach().cpu().numpy().tolist()

        return prediction_last, ground_truth_last, test_acc, test_loss


    def pipeline(self, total_epochs: int):
        for epoch in range(total_epochs):
            # Training
            train_acc, train_loss = self.train()

            # Testing (Evaluation)
            prediction_last, ground_truth_last, test_acc, test_loss = self.test()

            print(f"EPOCH {epoch+1} ==> Loss training = {train_loss} | Accuracy training = {train_acc} | Loss testing = {test_loss} | Accuracy testing = {test_acc}")

            ground_truth_string = self._dataset.decode(ground_truth_last)
            prediction_string = self._dataset.decode(prediction_last)

            logs_data = {
                "epoch": int(epoch),
                "train_accuracy": float(train_acc.item()),
                "train_loss": train_loss,
                "test_accuracy": float(test_acc.item()),
                "test_loss": test_loss,
                "training_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": prediction_string,
                "ground_truth": ground_truth_string
            }
            self._logger.save_metrics(logs_data)
            if (epoch+1) % 5 == 0 and epoch+1 > 0:
                self._logger.save_checkpoint(self.model.state_dict(), epoch+1)
                visualization.plot_training_results(self._logger.session_dir / "metrics.jsonl")


if __name__ == "__main__":
    trainer = Trainer(
        d_model=512,
        num_heads=2,
        ffn_hidden=1024,
        transformers_blocks=2,
        seq_length=256,
        batch_size=64,
        vocab_size=30000
    )

    trainer.pipeline(total_epochs=30)