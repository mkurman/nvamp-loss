import torch
import torch.nn as nn
import torch.nn.functional as F


class NVAMPLoss(nn.Module):
    """
    Normalized Variance-Aware Max-Penalized Loss (NVAMP Loss)

    This loss function combines three key components:
    1. Standard cross-entropy loss for token prediction
    2. Variance normalization to distribute learning effort across tokens
    3. Maximum loss penalty to handle extreme outliers

    Parameters:
    - alpha: Weight for the variance-normalized component (default: 1.0)
    - beta: Weight for the cross-entropy loss component (default: 0.02)
    - gamma: Weight for the max loss penalty term (default: 0.001)
    - eps: Small epsilon value for numerical stability (default: 1e-6)
    - ignore_index: Token index to ignore in loss computation (default: -100)

    Benefits:
    - Sensitivity to Outliers: Strongly penalizes rare, large errors.
    - Variance Normalization: Prevents training from overly focusing on average tokens, evenly distributing learning effort.
    - Extreme Error Suppression: Helps regularize training, promoting stability and robustness.
    - Performance Optimized: Uses torch's efficient operations for maximum speed.
    - Configurable Component Weights: Control the influence of each loss component.
    """

    def __init__(self, alpha=1.0, beta=0.02, gamma=0.001, eps=1e-6, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.ignore_index = ignore_index
        # Pre-compile constants for faster execution
        self._minus_100 = -100

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, logits, labels, **loss_kargs):
        """
        Forward pass for NVAMP Loss computation - optimized for speed.

        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            labels: Tensor of shape (batch_size, seq_len)

        Returns:
            Scalar tensor with the computed loss value
        """
        # Ensure float32 precision for stable training
        if logits.dtype != torch.float32:
            logits = logits.float()

        # Handle causal LM shift with optimized operations
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Use view instead of reshape for better performance
        _, _, vocab_size = shift_logits.size()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Fast calculation of token-level loss
        # Apply log_softmax and nll_loss separately for better performance
        token_loss = F.cross_entropy(
            shift_logits, shift_labels, ignore_index=self.ignore_index, reduction="none"
        )

        num_items_in_batch = loss_kargs.get("num_items_in_batch", None)

        # Vectorized variance-aware component
        mean_valid_losses = torch.mean(token_loss)

        # Efficient statistics computation
        ce_loss = (
            torch.sum(token_loss) / num_items_in_batch
            if num_items_in_batch
            else mean_valid_losses
        )

        # Use fused operations where possible
        # Calculate variance in one step using torch.var
        loss_std = torch.sqrt(torch.var(token_loss, unbiased=False) + self.eps)

        # Fast absolute difference and normalization
        normalized_deviations = torch.abs(token_loss - mean_valid_losses) / loss_std
        variance_loss = (
            normalized_deviations.mean() * ce_loss
            if num_items_in_batch
            else torch.mean(normalized_deviations)
        )

        # Efficient max calculation
        max_loss = torch.max(token_loss)

        # Final loss with minimal operations and gamma control for ce_loss
        final_loss = (
            self.alpha * variance_loss + self.beta * ce_loss + self.gamma * max_loss
        )

        return final_loss

    def extra_repr(self) -> str:
        """Return a string representation of the module parameters."""
        return (
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, eps={self.eps}"
        )
