from typing import Dict

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_count = 0

    for images, mask, targets in loader:
        images = images.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, mask)
            loss = criterion(logits, targets)

        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate_f1_micro(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()

    tp = 0.0
    fp = 0.0
    fn = 0.0

    with torch.no_grad():
        for images, mask, targets in loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                probs = torch.sigmoid(model(images, mask))
            preds = (probs >= threshold).float()

            tp += (preds * targets).sum().item()
            fp += (preds * (1.0 - targets)).sum().item()
            fn += ((1.0 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
    }
