## Image JEPA

This example demonstrates how to train a Joint Embedding Predictive Architecture (JEPA) on individual images instead of video sequences. The model learns representations from individual frames of the Moving MNIST dataset and is evaluated using linear probing for digit classification.

## Features

- **Image-only training**: Treats individual frames as separate images rather than video sequences
- **Linear probing evaluation**: Evaluates learned representations using a linear classifier for digit classification
- **Same data, different approach**: Uses the same Moving MNIST dataset but processes frames individually
- **Representation learning**: Learns meaningful representations through self-supervised learning

## Architecture

The Image JEPA consists of:
- **Encoder**: ResNet5 backbone that processes individual images
- **Regularizer**: Variance-Covariance (VC) loss to prevent representation collapse
- **Predictor**: Simple reconstruction task for individual images
- **Linear Probe**: Frozen encoder + linear classifier for evaluation

## Usage

### Basic Training

```bash
python main.py
```

### Custom Parameters

```bash
python main.py \
    --batch_size=32 \
    --epochs=50 \
    --lr=1e-3 \
    --henc=64 \
    --dstc=32 \
    --probe_epochs=30 \
    --probe_lr=1e-2
```

### Parameters

- `batch_size`: Batch size for training (default: 64)
- `dobs`: Input channels (default: 1 for grayscale)
- `henc`: Encoder hidden dimension (default: 32)
- `hpre`: Predictor hidden dimension (default: 32)
- `dstc`: Output dimension (default: 16)
- `cov_coeff`: Covariance coefficient for VC loss (default: 100.0)
- `std_coeff`: Standard deviation coefficient for VC loss (default: 10.0)
- `epochs`: Number of training epochs (default: 100)
- `lr`: Learning rate (default: 1e-3)
- `probe_epochs`: Number of epochs for linear probe training (default: 50)
- `probe_lr`: Learning rate for linear probe (default: 1e-3)

## Evaluation

The model is evaluated using linear probing:
1. The encoder is frozen after self-supervised training
2. A linear classifier is trained on top of the frozen representations
3. Performance is measured on a binary classification task (digit present/absent)

## Key Differences from Video JEPA

1. **Input**: Individual images instead of video sequences
2. **Temporal dimension**: Removed - no temporal modeling
3. **Prediction task**: Simple reconstruction instead of future frame prediction
4. **Evaluation**: Linear probing for classification instead of detection metrics

## Expected Results

- The model should learn meaningful representations that enable good linear probe performance
- VC loss should prevent representation collapse
- Linear probe accuracy should improve with better learned representations

## Implementation Details

### ImageOnlyDataset
- Extracts individual frames from the Moving MNIST video dataset
- Each frame becomes a separate training sample
- Preserves digit location information for evaluation

### ImageJEPA
- Simplified JEPA architecture for image-only processing
- Uses reconstruction as the prediction task
- Applies VC loss to prevent representation collapse

### LinearProbe
- Frozen encoder with trainable linear classifier
- Evaluates learned representations on digit classification
- Uses binary classification (digit present/absent)
