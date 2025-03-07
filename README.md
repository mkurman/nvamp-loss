# NVAMP Loss

Normalized Variance-Aware Max-Penalized Loss (NVAMP Loss) - A robust loss function for language model training.

## Overview

NVAMP Loss is a specialized loss function designed to enhance the training of language models by combining three key components:

1. **Cross-entropy loss** - Standard token prediction loss
2. **Variance normalization** - Ensures learning effort is distributed across all token types
3. **Maximum loss penalty** - Handles extreme outliers to improve stability

This loss function is particularly effective for language models where certain tokens may be underrepresented or difficult to predict.

## Installation

```bash
pip install git+https://github.com/mkurman/nvamp-loss.git
```

## Usage

```python
import torch
from nvamp import NVAMPLoss

# Initialize loss function with default parameters
criterion = NVAMPLoss()

# Or customize the balance of components
criterion = NVAMPLoss(
    alpha=1.0,    # Weight for variance-normalized component
    beta=0.02,    # Weight for cross-entropy component
    gamma=0.001,  # Weight for max loss penalty
    eps=1e-6,     # Numerical stability constant
    ignore_index=-100  # Token index to ignore in calculations
)

# During training
outputs = model(inputs)  # shape: [batch_size, seq_len, vocab_size]
targets = labels          # shape: [batch_size, seq_len]
loss = criterion(outputs, targets)
```

## Key Benefits

- **Sensitivity to Outliers**: Strongly penalizes rare, large errors
- **Variance Normalization**: Prevents training from overly focusing on average tokens
- **Extreme Error Suppression**: Helps regularize training for stability
- **Performance Optimized**: Uses torch's efficient operations for maximum speed
- **Configurable Component Weights**: Adjust the influence of each loss component

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `alpha` | Weight for variance-normalized component | 1.0 |
| `beta` | Weight for cross-entropy loss component | 0.02 |
| `gamma` | Weight for max loss penalty term | 0.001 |
| `eps` | Small epsilon value for numerical stability | 1e-6 |
| `ignore_index` | Token index to ignore in loss computation | -100 |

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.