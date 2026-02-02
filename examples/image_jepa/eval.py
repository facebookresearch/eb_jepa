"""
Evaluation utilities for self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class LinearProbe(nn.Module):
    """Linear probe classifier for evaluating representations."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def evaluate_linear_probe(model, linear_probe, val_loader, device, use_amp=True):
    """Evaluate linear probe on validation set."""
    model.eval()
    linear_probe.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                features, _ = model(data)

            outputs = linear_probe(features.float())
            loss = F.cross_entropy(outputs, target)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(val_loader)

    return accuracy, avg_loss
