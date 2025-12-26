## Self-Supervised Representation Learning from Unlabeled Images

This example demonstrates how to train a Joint Embedding Predictive Architecture (JEPA) on unlabeled images. The model learns representations from individual frames of the CIFAR 10 dataset and is evaluated using linear probing for image classification.

![Image JEPA Architecture](assets/arch_figure.png)

## Features

- **Image-only training**: Training from unlabeled image data
- **Representation learning**: Learns meaningful representations through self-supervised learning, avoids collapse using Variance-Covariance or LeJEPA (SIGReg) Regularization.
- **Linear probing evaluation**: Evaluates learned representations using a linear classifier


## Architecture

The Image JEPA consists of:
- **Encoder**: ResNet5 backbone that processes individual images
- **Regularizer**: Variance-Covariance (VC) loss to prevent representation collapse
- **Predictor**: Simple reconstruction task for individual images
- **Linear Probe**: Frozen encoder + linear classifier for evaluation

## Usage

### Training Configurations

#### 1. ResNet + VICReg Loss

```bash
python main.py \
    --model_type resnet \
    --loss_type vicreg \
    --sim_loss_weight 25.0 \
    --var_loss_weight 25.0 \
    --cov_loss_weight 1.0 \
    --batch_size 256 \
    --epochs 300
```

#### 2. ResNet + LE-JEPA (BCS) Loss

```bash
python main.py \
    --model_type resnet \
    --loss_type bcs \
    --lmbd 10.0 \
    --batch_size 256 \
    --epochs 300
```

#### 3. Vision Transformer + VICReg Loss

```bash
python main.py \
    --model_type vit_s \
    --patch_size 2 \
    --loss_type vicreg \
    --sim_loss_weight 25.0 \
    --var_loss_weight 25.0 \
    --cov_loss_weight 1.0 \
    --batch_size 256 \
    --epochs 300
```

For ViT-Base, use `--model_type vit_b` instead of `vit_s`.

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