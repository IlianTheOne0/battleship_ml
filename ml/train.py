from __future__ import annotations

import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

import config as config_outer
from logger_setup import setup_logging
from ml.dataset import (GestureDataset, build_eval_transform, build_train_transform, save_meta,)
from ml.model import GestureCNN
from ml.trainer import EarlyStopping, ModelCheckpoint, evaluate, train_epoch
from utils import set_seed

def _split_dataset(dataset: GestureDataset, config: config.Config, seed: int):
    number = len(dataset)
    number_val = int(config.GESTURE_VAL_SPLIT * number)
    number_test = int(config.GESTURE_TEST_SPLIT * number)
    number_train = number - number_val - number_test
    generator = torch.Generator().manual_seed(seed)

    return random_split(dataset, [number_train, number_val, number_test], generator=generator)

def _compute_class_weights(targets: list[int], num_classes: int,) -> np.ndarray:
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)

    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts

    weights = weights / weights.mean()
    return weights

def main() -> int:
    setup_logging()
    logger = logging.getLogger("ml.train")

    config = config_outer.Config()
    set_seed(config.RANDOM_SEED)
    # rng = random.Random(config.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    try:
        full_train = GestureDataset(config, build_train_transform(config))
        full_eval = GestureDataset(config, build_eval_transform(config))
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    train_ds, val_ds, test_ds = _split_dataset(full_train, config, config.RANDOM_SEED)
    _, val_eval_ds, test_eval_ds = _split_dataset(full_eval, config, config.RANDOM_SEED)

    train_targets = [full_train.samples[i][1] for i in train_ds.indices]
    class_weights = _compute_class_weights(train_targets, num_classes=len(config.GESTURE_CLASSES))
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    counts = np.bincount(train_targets, minlength=len(config.GESTURE_CLASSES))
    for cls_name, cnt, w in zip(config.GESTURE_CLASSES, counts, class_weights):
        logger.info("  Class %-10s | samples: %4d | weights: %.4f", cls_name, cnt, w)

    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True,)

    train_loader = DataLoader(train_ds, batch_size=config.GESTURE_BATCH_SIZE, sampler=sampler, num_workers=config.GESTURE_NUM_WORKERS, pin_memory=True,)
    validation_loader = DataLoader(val_eval_ds, batch_size=config.GESTURE_BATCH_SIZE, shuffle=False, num_workers=config.GESTURE_NUM_WORKERS, pin_memory=True,)
    test_loader = DataLoader(test_eval_ds, batch_size=config.GESTURE_BATCH_SIZE, shuffle=False, num_workers=config.GESTURE_NUM_WORKERS, pin_memory=True,)

    logger.info("Split: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds),)

    model = GestureCNN(num_classes=len(config.GESTURE_CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1,)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.GESTURE_LEARNING_RATE, weight_decay=config.GESTURE_WEIGHT_DECAY,)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config.GESTURE_SCHEDULER_FACTOR, patience=config.GESTURE_SCHEDULER_PATIENCE,)
    checkpoint = ModelCheckpoint(config.GESTURE_MODEL_PATH)
    early_stopping = EarlyStopping(config.GESTURE_PATIENCE)

    for epoch in range(config.GESTURE_EPOCHS):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)

        scheduler.step(validation_loss)
        early_stopping(validation_loss)
        checkpoint(model, validation_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "E %d/%d | LR: %.6f | Train Loss: %.4f, Accuracy: %.4f | "
            "Validation Loss: %.4f, Accuracy: %.4f",
            epoch + 1, config.GESTURE_EPOCHS, current_lr,
            train_loss, train_accuracy, validation_loss, validation_accuracy,
        )

        if early_stopping.early_stop:
            logger.info("Early stop was activated (E %d)", epoch + 1)
            break

    state = torch.load(config.GESTURE_MODEL_PATH, weights_only=True)
    model.load_state_dict(state)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    logger.info("Test Loss: %.4f, Test Accuracy: %.4f", test_loss, test_accuracy)

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    logger.info("Classification Report:\n%s", classification_report(y_true, y_pred, target_names=list(config.GESTURE_CLASSES), digits=4, zero_division=0,),)

    save_meta(config, config.GESTURE_META_PATH)
    logger.info("Metadata was saved: %s", config.GESTURE_META_PATH)
    return 0

if __name__ == "__main__": sys.exit(main())