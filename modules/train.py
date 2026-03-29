import torch
import torch.nn as nn
import modules.transformers as transformers
import modules.dataset as dataset
import modules.metrics as metrics


# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Define core model
model = transformers.MiniTransformers(
    d_model=2048,
    num_heads=8,
    ffn_hidden=1024,
    vocab_size=267723,
    total_blocks=1
)
model = model.to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load training and testing dataset
training_data, testing_data = dataset.dataset_preparation(batch_size=256)


def train(epoch: int):
    # Training loop
    for ep in range(epoch):
        model.train()
        train_acc = 0
        train_loss = 0
        for x_batch, y_batch in training_data:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Make a prediction
            prediction = model.forward(x_batch)

            # Calculate loss result
            prediction_val_resiized = prediction.view(-1, prediction.size()[-1])
            y_batch_resized = y_batch.view(-1)
            loss_training = loss_fn(prediction_val_resiized, y_batch_resized)
            train_loss += loss_training.item()

            # Back propagation
            optimizer.zero_grad()
            loss_training.backward()
            optimizer.step()

            # Calculate metrics
            accuracy_train = metrics.calculate_accuracy(
                prediction.detach().cpu(),
                y_batch.detach().cpu()
            )
            train_acc += accuracy_train
        
        # Averaging metrics results
        train_acc = train_acc / len(training_data)
        train_loss = train_loss / len(training_data)

        # TESTING
        model.eval()
        with torch.no_grad():
            test_acc = 0
            test_loss = 0
            for x_batch, y_batch in testing_data:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Make a prediction
                prediction = model.forward(x_batch)

                # Loss result
                loss_testing = loss_fn(prediction, y_batch)
                test_loss += loss_testing.item()
            
                # Calculate metrics
                accuracy_test = metrics.calculate_accuracy(
                    prediction.detach().cpu(),
                    y_batch.detach().cpu()
                )
                test_acc += accuracy_test

            test_acc = test_acc / len(testing_data)
            test_loss = test_loss / len(testing_data)

        print(f"EPOCH {ep+1} ==> Loss training = {train_loss} | Accuracy training = {train_acc} | Loss testing = {test_loss} | Accuracy testing = {test_acc}")