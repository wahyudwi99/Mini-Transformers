import torch


class Metrics:
    """
    A collection of metric evaluation functions for Transformer models.
    """

    @staticmethod
    def calculate_accuracy(prediction_result, y_batch):
        """
        Calculates standard per-token accuracy.
        
        Args:
            prediction_result (torch.Tensor): Model logits [batch_size, seq_len, vocab_size]
            y_batch (torch.Tensor): Ground truth labels [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Scalar tensor containing the accuracy value.
        """
        # Argmax prediction logits to get the most likely token IDs
        result_argmax = torch.argmax(prediction_result, dim=2)

        # Calculate accuracy by comparing predictions with ground truth
        # Using .float().mean() is more concise than .sum() / total
        accuracy = (result_argmax == y_batch).float().mean()

        return accuracy

    @staticmethod
    def calculate_mmlu(prediction_result, targets, choice_indices):
        """
        Calculates MMLU (Massive Multitask Language Understanding) accuracy.
        MMLU evaluation typically involves selecting the choice (A, B, C, or D) with the 
        highest logit at the answer position.
        
        Args:
            prediction_result (torch.Tensor): Model logits. 
                                            Expected shape [batch_size, seq_len, vocab_size] 
                                            or [batch_size, vocab_size] for the last token.
            targets (torch.Tensor): Correct choice indices (0 for A, 1 for B, 2 for C, 3 for D).
                                   Shape: [batch_size]
            choice_indices (list or torch.Tensor): The token IDs in the vocabulary that 
                                                  correspond to ['A', 'B', 'C', 'D'].
            
        Returns:
            torch.Tensor: Scalar tensor containing the MMLU accuracy.
        """
        # If prediction_result is 3D, we assume we want the last token's logits
        if prediction_result.dim() == 3:
            prediction_result = prediction_result[:, -1, :]
            
        # Extract logits only for the possible choice tokens (A, B, C, D)
        # prediction_result shape: [batch_size, vocab_size]
        # choice_logits result: [batch_size, 4]
        choice_logits = prediction_result[:, choice_indices]
        
        # Determine which choice the model thinks is most likely
        predictions = torch.argmax(choice_logits, dim=1)
        
        # Calculate accuracy by comparing with target choice indices
        mmlu_accuracy = (predictions == targets).float().mean()
        
        return mmlu_accuracy