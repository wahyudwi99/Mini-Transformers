import torch



def calculate_accuracy(
    prediction_result,
    y_batch
):
    # Argmax prediction logits
    result_argmax = torch.argmax(prediction_result, dim=2)

    # Calculate accuracy
    accuracy = (result_argmax == y_batch).sum() / (y_batch.size()[0] * y_batch.size()[1])

    return accuracy