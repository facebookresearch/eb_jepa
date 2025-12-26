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
    --var_loss_weight 1.0 \
    --cov_loss_weight 80.0 \
    --batch_size 256 \
    --epochs 300
```

#### 2. ResNet + LE-JEPA (SIGReg) Loss

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

## Results

### Collapse Prevention via VICReg and SIGReg

### Using a Projector

### Using a Transformer

## References
- [JEPA Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [ResNet Architecture](https://arxiv.org/abs/1512.03385)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [Vision Transformer Architecture](https://arxiv.org/abs/2010.11929)
- [VICReg](https://arxiv.org/abs/2105.04906)
- [LeJEPA](https://arxiv.org/abs/2511.08544)