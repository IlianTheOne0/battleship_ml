from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from prompt_toolkit import validation
from torch.utils.data import DataLoader

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,device: torch.device,) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


class ModelCheckpoint:
    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, model: nn.Module, validation_accuracy: float) -> None:
        if validation_accuracy > self.best_acc:
            self.best_acc = validation_accuracy
            torch.save(model.state_dict(), self.filepath)
            self.logger.info("Validation accuracy improved to %.4f. Saving model...", validation_accuracy)

class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, validation_loss: float) -> None:
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info("Early stopping was activated!")